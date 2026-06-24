"""Geometry-augmented MLP — periodic embeddings + optional FiLM gate.

Architecture reference: periodic numeric embeddings follow Gorishniy et al.,
"On Embeddings for Numerical Features in Tabular Deep Learning" (NeurIPS 2022);
FiLM conditioning follows Perez et al., "FiLM: Visual Reasoning with a General
Conditioning Layer" (AAAI 2018).  The latter is the tabular analog of the
metadata→attention-map multiplicative fusion in Lee et al. (MileTS '19,
"Metadata-Augmented Neural Networks for Cross-Location Solar Irradiation").

Motivation (see results/model_comparison/ocean_robustness_comparison.md): under
the k-fold unseen-date holdout, TabM ties the plain MLP on accuracy.  This module
tests whether a *richer representation of geometry* helps the accuracy-leading
MLP backbone.  The base FeaturePipeline already carries `cos_glint_angle`,
`1/cos(sza)`, etc.; the periodic embedding adds the missing sin components, higher
harmonics, and (optionally) location — features the pipeline does not encode.

Backbone is identical to mlp_baseline (Linear64→ReLU→Linear32→ReLU→Linear1), a
*point* predictor: interval metrics are degenerate (q05=q50=q95) by design; only
RMSE / MAE / R² are meaningful.

Geometry fusion modes (--geom_mode):
  none   : ignore geometry → reproduces the plain MLP baseline (sanity check).
  concat : geometry-encoder output concatenated to the base features at the input.
  film   : geometry produces per-unit (gamma, beta) that modulate the first
           hidden layer — h1 = ReLU(gamma * (W1 x) + b1 + beta).  Multiplicative
           gating, the tabular analog of the paper's attention-map fusion.

Periodic features need no fitted scaler (sin/cos are already in [-1, 1]), so they
introduce no train/test leakage under blocked splits.
"""

import argparse
import gc
import json
import logging
import os
import pickle
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .pipeline import FeaturePipeline, _ensure_derived_features, filter_target_outliers
from .splits import split_dataframe
from . import diagnostics as diag
from search.tracking import RunSummary, get_git_commit_hash
from utils import get_storage_dir

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = 'model_geom_mlp_best.pt'
META_FILE = 'geom_mlp_meta.pkl'

# Raw angular / location columns (degrees) used to build periodic features.
# All are treated as real angles: theta_rad = deg * pi/180, then sin/cos harmonics.
GEOM_ANGLE_COLS = ['sza', 'vza', 'glint_angle', 'pol_angle', 'lat', 'lon']


def build_periodic_features(frame: pd.DataFrame, cols, n_harmonics: int):
    """Return (G [N, 2*H*n_cols] float32, feature_names).  Missing cols are skipped."""
    feats, names = [], []
    for c in cols:
        if c not in frame.columns:
            continue
        theta = np.radians(frame[c].to_numpy(dtype=np.float64))
        for k in range(1, n_harmonics + 1):
            feats.append(np.sin(k * theta)); names.append(f"{c}_sin{k}")
            feats.append(np.cos(k * theta)); names.append(f"{c}_cos{k}")
    if not feats:
        raise ValueError(f"No geometry columns found among {cols} in the dataframe.")
    return np.column_stack(feats).astype(np.float32), names


class GeomMLP(nn.Module):
    """MLP point predictor with optional periodic-geometry fusion."""

    def __init__(self, n_features: int, g_dim: int, *, geom_mode: str = 'concat',
                 geom_hidden: int = 32, h1: int = 64, h2: int = 32):
        super().__init__()
        self.geom_mode = geom_mode
        if geom_mode == 'none':
            in_dim = n_features
        elif geom_mode == 'concat':
            self.geom_enc = nn.Sequential(nn.Linear(g_dim, geom_hidden), nn.ReLU())
            in_dim = n_features + geom_hidden
        elif geom_mode == 'film':
            # geometry → (gamma, beta) for the first hidden layer (h1 units each)
            self.geom_enc = nn.Sequential(nn.Linear(g_dim, geom_hidden), nn.ReLU())
            self.film = nn.Linear(geom_hidden, 2 * h1)
            self.fc1 = nn.Linear(n_features, h1)
            in_dim = None
        else:
            raise ValueError(f"geom_mode must be none/concat/film, got {geom_mode!r}")

        if geom_mode in ('none', 'concat'):
            self.net = nn.Sequential(
                nn.Linear(in_dim, h1), nn.ReLU(),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, 1),
            )
        else:  # film
            self.tail = nn.Sequential(
                nn.ReLU(),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, 1),
            )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.geom_mode == 'none':
            return self.net(x).squeeze(-1)
        if self.geom_mode == 'concat':
            return self.net(torch.cat([x, self.geom_enc(g)], dim=1)).squeeze(-1)
        # film
        gb = self.film(self.geom_enc(g))
        gamma, beta = gb.chunk(2, dim=1)
        h = self.fc1(x)
        h = (1.0 + gamma) * h + beta          # init-friendly: gamma≈0 → identity-ish
        return self.tail(h).squeeze(-1)


def _predict_point(model, X_np, G_np, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for s in range(0, len(X_np), batch_size):
            xb = torch.tensor(X_np[s:s + batch_size], dtype=torch.float32).to(device)
            gb = torch.tensor(G_np[s:s + batch_size], dtype=torch.float32).to(device)
            out.append(model(xb, gb).cpu().numpy())
    return np.concatenate(out)


def train_geom_mlp(X_tr, G_tr, y_tr, X_te, G_te, y_te, n_features, g_dim, *,
                   geom_mode='concat', output_dir='.', batch_size=4096, n_epochs=100,
                   patience=50, huber_delta=1.0, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tr = TensorDataset(torch.tensor(X_tr), torch.tensor(G_tr), torch.tensor(y_tr))
    te = TensorDataset(torch.tensor(X_te), torch.tensor(G_te), torch.tensor(y_te))
    pin = device.type in ("cuda", "mps")
    nw = min(8, os.cpu_count() or 1)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, pin_memory=pin,
                    num_workers=nw, persistent_workers=nw > 0)
    vl = DataLoader(te, batch_size=batch_size, shuffle=False, pin_memory=pin,
                    num_workers=nw, persistent_workers=nw > 0)

    model = GeomMLP(n_features, g_dim, geom_mode=geom_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=n_epochs * len(tl),
        pct_start=0.05, div_factor=25, final_div_factor=1000)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    ckpt = os.path.join(output_dir, CHECKPOINT_FILE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write("=" * 60)
    tqdm.write(f"  GeomMLP[{geom_mode}] device={device}  base_feat={n_features} "
               f"geom_dim={g_dim}  params={n_params:,}")
    tqdm.write(f"  Train={len(tr):,}  Val={len(te):,}  batch={batch_size}  epochs={n_epochs}")
    tqdm.write("=" * 60)

    best_val, no_improve = float("inf"), 0
    for epoch in tqdm(range(n_epochs), desc="Training", unit="epoch"):
        model.train()
        for xb, gb, yb in tl:
            xb, gb, yb = xb.to(device), gb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, gb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sched.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, gb, yb in vl:
                xb, gb, yb = xb.to(device), gb.to(device), yb.to(device)
                vloss += loss_fn(model(xb, gb), yb).item()
        vloss /= len(vl)
        if vloss < best_val:
            best_val, no_improve = vloss, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": best_val}, ckpt)
        else:
            no_improve += 1
        if patience is not None and no_improve >= patience:
            tqdm.write(f"  Early stopping at epoch {epoch}."); break
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(f"  epoch {epoch+1:4d}/{n_epochs}  val={vloss:.5f}  best={best_val:.5f}")

    if not os.path.exists(ckpt):
        raise RuntimeError(f"No checkpoint saved to {ckpt}")
    best = torch.load(ckpt, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    tqdm.write(f"  GeomMLP done. best epoch={best['epoch']} val_loss={best['val_loss']:.5f}")
    return model.cpu()


def main():
    p = argparse.ArgumentParser(description="Geometry-augmented MLP (periodic + FiLM).")
    p.add_argument('--sfc_type', type=int, default=0)
    p.add_argument('--val_split', type=str, default='random',
                   choices=['random', 'date', 'date_kfold'])
    p.add_argument('--n_folds', type=int, default=None)
    p.add_argument('--fold', type=int, default=None)
    p.add_argument('--feature_set', type=str, default='full',
                   choices=['full', 'no_xco2', 'no_spec', 'full_fitqual'])
    p.add_argument('--geom_mode', type=str, default='concat', choices=['none', 'concat', 'film'])
    p.add_argument('--n_harmonics', type=int, default=4)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--suffix', type=str, default='')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data', type=str, default=None,
                   help='Override input data file (local multi-date testing).')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    storage_dir = get_storage_dir()
    fdir = storage_dir / 'results/csv_collection'
    data_name = ('combined_2016_2020_dates.parquet' if platform.system() == "Linux"
                 else 'combined_2020-02-01_all_orbits.parquet')
    base_dir = storage_dir / 'results/model_geom_mlp'
    output_dir = base_dir / args.suffix if args.suffix else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.monotonic()
    run_id = args.suffix or datetime.now().strftime('%Y%m%d-%H%M%S')
    commit = get_git_commit_hash(storage_dir)

    _dp = args.data if args.data else os.path.join(fdir, data_name)
    df = pd.read_parquet(_dp) if _dp.endswith('.parquet') else pd.read_csv(_dp)
    df = df[df['sfc_type'] == args.sfc_type]
    df = df[df['snow_flag'] == 0]
    df = _ensure_derived_features(df)
    df = filter_target_outliers(df)

    train_df, held_df = split_dataframe(df, mode=args.val_split, test_size=args.test_size,
                                        random_state=args.seed,
                                        n_folds=args.n_folds, fold=args.fold)
    del df; gc.collect()
    pipeline = FeaturePipeline.fit(train_df, sfc_type=args.sfc_type, feature_set=args.feature_set)
    pipeline.save(output_dir / 'geom_mlp_pipeline.pkl')

    def _prep(frame):
        X = pipeline.transform(frame)
        G, gnames = build_periodic_features(frame, GEOM_ANGLE_COLS, args.n_harmonics)
        y = frame['xco2_bc_anomaly'].to_numpy(dtype=np.float32)
        valid = (np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                 & np.all(np.isfinite(G), axis=1))
        return X[valid], G[valid], y[valid], frame.loc[valid], gnames

    X_tr, G_tr, y_tr, _, gnames = _prep(train_df)
    X_te, G_te, y_te, held_valid, _ = _prep(held_df)
    print(f"[geom_mlp:{args.geom_mode}] X {X_tr.shape}  G {G_tr.shape}  held {X_te.shape}")

    if platform.system() == "Darwin":
        epochs, batch_size = 100, 2048
    else:
        epochs, batch_size = 500, 4096
    model = train_geom_mlp(X_tr, G_tr, y_tr, X_te, G_te, y_te,
                           pipeline.n_features, G_tr.shape[1], geom_mode=args.geom_mode,
                           output_dir=str(output_dir), batch_size=batch_size,
                           n_epochs=epochs, seed=args.seed)
    with open(output_dir / META_FILE, 'wb') as f:
        pickle.dump({'n_features': pipeline.n_features, 'feature_names': pipeline.features,
                     'geom_mode': args.geom_mode, 'n_harmonics': args.n_harmonics,
                     'geom_features': gnames, 'feature_set': args.feature_set,
                     'val_split': args.val_split}, f)

    q50 = _predict_point(model, X_te, G_te)
    preds = np.column_stack([q50, q50, q50])
    g = diag.compute_metrics(y_te, preds)
    strat = diag.stratified_metrics(held_valid, y_te, preds)
    prefix = f"geommlp_{args.geom_mode}_{args.val_split}"
    diag.save_diagnostics(output_dir, prefix, g, strat)
    print(f"[{prefix}] RMSE={g['rmse']:.4f}  MAE={g['mae']:.4f}  R²={g['r2']:.4f}  "
          f"(point predictor — interval metrics degenerate)")

    summary = RunSummary(
        run_id=run_id, script_name=os.path.basename(__file__), model_family='geom_mlp',
        commit=commit, status='success',
        primary_metric_name='geom_mlp_held_rmse', primary_metric_value=g['rmse'],
        secondary_metrics={'geom_mlp_held_mae': g['mae'], 'geom_mlp_held_r2': g['r2']},
        runtime_seconds=float(time.monotonic() - run_start),
        description=f'GeomMLP[{args.geom_mode}] H={args.n_harmonics}, {args.val_split}-split, '
                    f'feature_set={args.feature_set}',
        artifacts={'output_dir': str(output_dir),
                   'metrics_json': str(output_dir / f'{prefix}_metrics.json')},
        config={'sfc_type': args.sfc_type, 'val_split': args.val_split, 'geom_mode': args.geom_mode,
                'n_harmonics': args.n_harmonics, 'feature_set': args.feature_set, 'seed': args.seed},
    )
    with open(output_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, indent=2, sort_keys=True)
    print(f"Saved run summary → {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()

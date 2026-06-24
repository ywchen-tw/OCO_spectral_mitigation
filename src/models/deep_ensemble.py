"""Deep-ensemble MLP with Gaussian-NLL heads + conformal calibration.

Reference: Lakshminarayanan et al. 2017, "Simple and Scalable Predictive Uncertainty
Estimation using Deep Ensembles" (NeurIPS).  Conformal layer: see models/conformal.py.

Motivation (results/model_comparison/ocean_robustness_comparison.md): under k-fold
unseen-date holdout the plain MLP *ties* TabM on point accuracy and is best in the
left tail — but it has no intervals.  This module gives the MLP intervals two ways
and tests whether the accuracy leader can also be well-calibrated:

  M independent MLP members, each a Gaussian head (mu, log_var) trained by NLL.
  Ensemble mixture:  mu* = mean(mu_m)
                     var* = mean(var_m + mu_m^2) - mu*^2   (epistemic + aleatoric)
  Raw 90% interval:  mu* ± 1.645 * sqrt(var*)              (Gaussian approx)

Then a held-out calibration date-block (carved from the train split) recalibrates the
intervals via split and Mondrian conformal (bins = predicted-mu deciles → targets the
low-prediction tail).  Three metric sets are written per run, sharing the same mu (so
RMSE/MAE/R² are identical; only the intervals differ):
  de_raw_<split>       — raw Gaussian-mixture interval
  de_split_<split>     — global split conformal
  de_mondrian_<split>  — regime-conditional (mu-decile) conformal   ← the headline

All intervals are monotone by construction (crossing_rate = 0).
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
from . import conformal as cf
from . import diagnostics as diag
from search.tracking import RunSummary, get_git_commit_hash
from utils import get_storage_dir

logger = logging.getLogger(__name__)

Z90 = 1.6448536269514722  # Gaussian 90% two-sided


class GaussianMLP(nn.Module):
    """n_features → 64 → ReLU → 32 → ReLU → (mu, log_var)."""

    def __init__(self, n_features: int, h1: int = 64, h2: int = 32):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_features, h1), nn.ReLU(),
                                  nn.Linear(h1, h2), nn.ReLU())
        self.head = nn.Linear(h2, 2)

    def forward(self, x):
        h = self.body(x)
        out = self.head(h)
        mu = out[:, 0]
        log_var = torch.clamp(out[:, 1], min=-10.0, max=10.0)  # var in [~4.5e-5, ~2.2e4]
        return mu, log_var


def gaussian_nll(mu, log_var, y):
    return 0.5 * (log_var + (y - mu) ** 2 / torch.exp(log_var)).mean()


def _train_member(X_tr, y_tr, X_val, y_val, n_features, *, seed, device,
                  batch_size, n_epochs, patience, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    va = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    pin = device.type in ("cuda", "mps"); nw = min(8, os.cpu_count() or 1)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, pin_memory=pin,
                    num_workers=nw, persistent_workers=nw > 0)
    vl = DataLoader(va, batch_size=batch_size, shuffle=False, pin_memory=pin,
                    num_workers=nw, persistent_workers=nw > 0)
    model = GaussianMLP(n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                total_steps=n_epochs * len(tl), pct_start=0.05, div_factor=25,
                final_div_factor=1000)
    best, no_imp = float("inf"), 0
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mu, lv = model(xb)
            loss = gaussian_nll(mu, lv, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(device), yb.to(device)
                mu, lv = model(xb)
                vloss += gaussian_nll(mu, lv, yb).item()
        vloss /= len(vl)
        if vloss < best:
            best, no_imp = vloss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1
        if patience is not None and no_imp >= patience:
            break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.cpu()


def _member_predict(model, X, device, batch_size=8192):
    model = model.to(device); model.eval()
    mus, vars = [], []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.tensor(X[s:s + batch_size], dtype=torch.float32).to(device)
            mu, lv = model(xb)
            mus.append(mu.cpu().numpy()); vars.append(torch.exp(lv).cpu().numpy())
    return np.concatenate(mus), np.concatenate(vars)


def ensemble_predict(members, X, device):
    """Gaussian-mixture ensemble → (mu*, sigma*)."""
    mu_stack, var_stack = [], []
    for m in members:
        mu, var = _member_predict(m, X, device)
        mu_stack.append(mu); var_stack.append(var)
    mu_stack = np.stack(mu_stack)            # [M, N]
    var_stack = np.stack(var_stack)
    mu_star = mu_stack.mean(0)
    var_star = (var_stack + mu_stack ** 2).mean(0) - mu_star ** 2
    sigma_star = np.sqrt(np.maximum(var_star, 1e-12))
    return mu_star.astype(np.float32), sigma_star.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Deep-ensemble MLP + conformal calibration.")
    p.add_argument('--sfc_type', type=int, default=0)
    p.add_argument('--val_split', type=str, default='random',
                   choices=['random', 'date', 'date_kfold'])
    p.add_argument('--n_folds', type=int, default=None)
    p.add_argument('--fold', type=int, default=None)
    p.add_argument('--feature_set', type=str, default='full',
                   choices=['full', 'no_xco2', 'no_spec'])
    p.add_argument('--n_members', type=int, default=5)
    p.add_argument('--calib_frac', type=float, default=0.15,
                   help='Fraction of TRAIN dates carved out as the conformal '
                        'calibration block (date split when possible).')
    p.add_argument('--mondrian_bins', type=int, default=10)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--suffix', type=str, default='')
    p.add_argument('--seed', type=int, default=42, help='Base seed; member m uses seed+m.')
    p.add_argument('--data', type=str, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    storage_dir = get_storage_dir()
    fdir = storage_dir / 'results/csv_collection'
    data_name = ('combined_2016_2020_dates.parquet' if platform.system() == "Linux"
                 else 'combined_2020-02-01_all_orbits.parquet')
    base_dir = storage_dir / 'results/model_deep_ensemble'
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

    # Carve a calibration block out of TRAIN (date split if dates allow, else random).
    try:
        if 'date' in train_df.columns and pd.to_datetime(
                train_df['date'].astype(str).str.replace("b'", "").str.replace("'", "")
                if train_df['date'].dtype == object else train_df['date']).nunique() >= 2:
            proper_df, calib_df = split_dataframe(train_df, mode='date', test_size=args.calib_frac)
        else:
            proper_df, calib_df = split_dataframe(train_df, mode='random',
                                                  test_size=args.calib_frac, random_state=args.seed)
    except Exception:
        proper_df, calib_df = split_dataframe(train_df, mode='random',
                                              test_size=args.calib_frac, random_state=args.seed)
    del train_df; gc.collect()

    pipeline = FeaturePipeline.fit(proper_df, sfc_type=args.sfc_type, feature_set=args.feature_set)
    pipeline.save(output_dir / 'deep_ensemble_pipeline.pkl')

    def _prep(frame):
        X = pipeline.transform(frame)
        y = frame['xco2_bc_anomaly'].to_numpy(dtype=np.float32)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        return X[valid], y[valid], frame.loc[valid]

    X_tr, y_tr, _ = _prep(proper_df)
    X_cal, y_cal, _ = _prep(calib_df)
    X_te, y_te, held_valid = _prep(held_df)
    print(f"[deep_ensemble] proper-train {X_tr.shape}  calib {X_cal.shape}  held {X_te.shape}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    epochs, batch_size = (100, 2048) if platform.system() == "Darwin" else (500, 4096)
    # internal val for early stopping = the calibration block (it is held out of training)
    members = []
    for m in range(args.n_members):
        ck = str(output_dir / f'member_{m}.pt')
        print(f"  training member {m+1}/{args.n_members} (seed={args.seed + m})")
        members.append(_train_member(X_tr, y_tr, X_cal, y_cal, pipeline.n_features,
                                      seed=args.seed + m, device=device, batch_size=batch_size,
                                      n_epochs=epochs, patience=50, ckpt=ck))

    mu_cal, sig_cal = ensemble_predict(members, X_cal, device)
    mu_te, sig_te = ensemble_predict(members, X_te, device)

    # ── 1) raw Gaussian-mixture interval ───────────────────────────────────────
    preds_raw = np.column_stack([mu_te - Z90 * sig_te, mu_te, mu_te + Z90 * sig_te])
    # ── 2) global split conformal ──────────────────────────────────────────────
    preds_split, q_split = cf.split_conformal(y_cal, mu_cal, sig_cal, mu_te, sig_te, alpha=0.10)
    # ── 3) Mondrian conformal by predicted-mu deciles ──────────────────────────
    cal_bin, edges = cf.make_quantile_bins(mu_cal, args.mondrian_bins)
    te_bin, _ = cf.make_quantile_bins(mu_te, args.mondrian_bins, edges=edges)
    preds_mond, q_by_bin = cf.mondrian_conformal(y_cal, mu_cal, sig_cal, cal_bin,
                                                 mu_te, sig_te, te_bin, alpha=0.10)

    results = {}
    for tag, preds in [('raw', preds_raw), ('split', preds_split), ('mondrian', preds_mond)]:
        g = diag.compute_metrics(y_te, preds)
        strat = diag.stratified_metrics(held_valid, y_te, preds)
        prefix = f"de_{tag}_{args.val_split}"
        diag.save_diagnostics(output_dir, prefix, g, strat)
        results[tag] = g
        print(f"[{prefix}] RMSE={g['rmse']:.4f} R²={g['r2']:.4f} "
              f"cov90={g['coverage_90']:.4f} width={g['mean_interval_width']:.4f} "
              f"cross={g['crossing_rate']}")

    with open(output_dir / 'deep_ensemble_meta.pkl', 'wb') as f:
        pickle.dump({'n_features': pipeline.n_features, 'n_members': args.n_members,
                     'q_split': q_split, 'q_by_bin': q_by_bin, 'mondrian_edges': edges.tolist(),
                     'feature_set': args.feature_set, 'val_split': args.val_split}, f)

    g = results['mondrian']
    summary = RunSummary(
        run_id=run_id, script_name=os.path.basename(__file__), model_family='deep_ensemble',
        commit=commit, status='success',
        primary_metric_name='de_held_rmse', primary_metric_value=g['rmse'],
        secondary_metrics={'de_held_r2': g['r2'], 'de_mondrian_cov90': g['coverage_90'],
                           'de_split_cov90': results['split']['coverage_90'],
                           'de_raw_cov90': results['raw']['coverage_90']},
        runtime_seconds=float(time.monotonic() - run_start),
        description=f'Deep ensemble M={args.n_members} + conformal, {args.val_split}-split, '
                    f'feature_set={args.feature_set}',
        artifacts={'output_dir': str(output_dir),
                   'metrics_json': str(output_dir / f'de_mondrian_{args.val_split}_metrics.json')},
        config={'sfc_type': args.sfc_type, 'val_split': args.val_split,
                'n_members': args.n_members, 'calib_frac': args.calib_frac,
                'feature_set': args.feature_set, 'seed': args.seed},
    )
    with open(output_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, indent=2, sort_keys=True)
    print(f"Saved run summary → {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()

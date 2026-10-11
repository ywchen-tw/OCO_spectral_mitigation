"""Plain MLP baseline for OCO-2 XCO2 anomaly prediction.

Architecture reference: no external reference (standard feed-forward MLP).

This is a clean re-implementation of the MLP in result_ana.py
(n_features → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)), brought into
the same training scaffold as TabM / FT-Transformer so results are directly
comparable (same raw-dataframe split, same train-only FeaturePipeline, same
diagnostic suite).  result_ana.py itself is NOT modified.

Note on uncertainty: this baseline is a *point* predictor (single output = q50).
It has no calibrated intervals, so for the shared diagnostic suite the three
quantile columns are reported as q05 = q50 = q95 (degenerate, zero-width
interval).  Coverage and interval-width metrics will therefore be near-zero —
that is the honest comparison point against the quantile models; only the point
metrics (RMSE / MAE / R²) are meaningful for this baseline.
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
from tqdm import tqdm

from .pipeline import FeaturePipeline, _ensure_derived_features, filter_target_outliers, resolve_target_col
from .splits import split_dataframe
from . import diagnostics as diag
from . import train_common as tc
from search.tracking import RunSummary, get_git_commit_hash
from utils import get_storage_dir

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = 'model_mlp_baseline_best.pt'
META_FILE = 'mlp_baseline_meta.pkl'


class MLPBaseline(nn.Module):
    """n_features → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1).

    Matches the result_ana.py MLP exactly (point predictor of xco2_bc_anomaly).
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _predict_point(model: MLPBaseline, X_np: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            Xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32).to(device)
            out.append(model(Xb).cpu().numpy())
            del Xb
    return np.concatenate(out)


def train_mlp(X_train, y_train, X_test, y_test, n_features, *,
              output_dir: str = ".", batch_size: int = 4096, n_epochs: int = 100,
              log_every: int = 10, patience: 'int | None' = tc.TrainConfig.patience,
              huber_delta: float = 1.0, seed: int = 42,
              deterministic: bool = False, gpu_resident: 'bool | None' = None) -> MLPBaseline:
    """Train the MLP baseline (Huber loss on the single output) via the shared
    train_common scaffold; writes model_mlp_baseline_best.pt in the legacy
    dict format ({'epoch', 'model_state_dict', 'val_loss'}) that adapters.py
    expects."""
    gen = tc.set_seeds(seed, deterministic)
    device = tc.select_device()
    cfg = tc.TrainConfig(epochs=n_epochs, batch_size=batch_size, patience=patience,
                         seed=seed, deterministic=deterministic,
                         gpu_resident=gpu_resident, log_every=log_every)
    tl = tc.make_batches((X_train.astype(np.float32, copy=False),
                          y_train.astype(np.float32, copy=False)),
                         batch_size, shuffle=True, device=device, generator=gen,
                         gpu_resident=gpu_resident)
    vl = tc.make_batches((X_test.astype(np.float32, copy=False),
                          y_test.astype(np.float32, copy=False)),
                         batch_size, shuffle=False, device=device, generator=gen,
                         gpu_resident=gpu_resident)

    model = MLPBaseline(n_features).to(device)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    def criterion(mdl, batch):
        xb, yb = batch
        return loss_fn(mdl(xb), yb)

    ckpt_path = os.path.join(output_dir, CHECKPOINT_FILE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write("=" * 60)
    tqdm.write(f"  MLP baseline — device={device}  features={n_features}  params={n_params:,}")
    tqdm.write(f"  Train={len(X_train):,}  Val={len(X_test):,}  batch={cfg.batch_size}  epochs={cfg.epochs}")
    tqdm.write("=" * 60)

    model, stats = tc.train_model(model, criterion, tl, vl, cfg, ckpt_path, device,
                                  log_prefix='[mlp]')
    # Re-save in the legacy dict format consumed by adapters.py.
    torch.save({"epoch": stats['best_epoch'], "model_state_dict": model.state_dict(),
                "val_loss": stats['best_val']}, ckpt_path)
    tqdm.write(f"  MLP baseline done. best epoch={stats['best_epoch']} "
               f"val_loss={stats['best_val']:.5f}")
    return model.cpu()


def main():
    parser = argparse.ArgumentParser(description="Plain MLP baseline for XCO2 anomaly prediction")
    parser.add_argument('--sfc_type', type=int, default=0)
    parser.add_argument('--val_split', type=str, default='random',
                        choices=['random', 'date', 'date_kfold'])
    parser.add_argument('--n_folds', type=int, default=None,
                        help='Number of date blocks for --val_split date_kfold.')
    parser.add_argument('--fold', type=int, default=None,
                        help='Which date block (0-based) to hold out for date_kfold.')
    parser.add_argument('--feature_set', type=str, default='full',
                        choices=['full', 'no_xco2', 'no_spec', 'full_fitqual', 'full_contam'])
    parser.add_argument('--target', type=str, default=None,
                        help="Clear-sky reference for the regression target: '10km' "
                             "(default, xco2_bc_anomaly), '15km' (xco2_bc_anomaly_r15), or '5km' (xco2_bc_anomaly_r05).")
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--pipeline', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str, default=None,
                        help='Override the input data file (default: platform data_name in '
                             'results/csv_collection/). Use for local multi-date testing.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    storage_dir = get_storage_dir()
    fdir = storage_dir / 'results/csv_collection'
    data_name = ('combined_2016_2020_dates.parquet' if platform.system() == "Linux"
                 else 'combined_2020-02-01_all_orbits.parquet')
    base_dir = storage_dir / 'results/model_mlp_baseline'
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
    target_col = resolve_target_col(args.target)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not in parquet; regenerate the combined "
            f"parquet (spectral/fitting.py + build_feature_dataset.py) or pass --target 10km.")
    df = filter_target_outliers(df, target_col=target_col)

    train_df, held_df = split_dataframe(df, mode=args.val_split, test_size=args.test_size,
                                        random_state=args.seed,
                                        n_folds=args.n_folds, fold=args.fold)
    del df
    gc.collect()
    if args.pipeline:
        pipeline = FeaturePipeline.load(args.pipeline)
    else:
        pipeline = FeaturePipeline.fit(train_df, sfc_type=args.sfc_type, feature_set=args.feature_set)
        pipeline.save(output_dir / 'mlp_pipeline.pkl')

    def _prep(frame):
        X = pipeline.transform(frame)
        y = frame[target_col].to_numpy(dtype=np.float32)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        return X[valid], y[valid], frame.loc[valid]

    X_train, y_train, _ = _prep(train_df)
    X_held, y_held, held_valid = _prep(held_df)
    print(f"[mlp] X_train {X_train.shape}  X_held {X_held.shape}")

    epochs, batch_size = tc.platform_epochs_batch()
    model = train_mlp(X_train, y_train, X_held, y_held, pipeline.n_features,
                      output_dir=str(output_dir), batch_size=batch_size, n_epochs=epochs,
                      seed=args.seed)
    with open(output_dir / META_FILE, 'wb') as f:
        pickle.dump({'n_features': pipeline.n_features, 'feature_names': pipeline.features,
                     'feature_set': args.feature_set, 'val_split': args.val_split}, f)

    # ── Diagnostics (point predictor → degenerate intervals q05=q50=q95) ───────
    q50 = _predict_point(model, X_held)
    preds = np.column_stack([q50, q50, q50])
    g = diag.compute_metrics(y_held, preds)
    strat = diag.stratified_metrics(held_valid, y_held, preds)
    prefix = f"mlp_{args.val_split}"
    diag.save_diagnostics(output_dir, prefix, g, strat)
    diag.save_correction_and_preds(output_dir, prefix, held_valid, y_held, preds)
    print(f"[{prefix}] RMSE={g['rmse']:.4f}  MAE={g['mae']:.4f}  R²={g['r2']:.4f}  "
          f"(point predictor — interval metrics are degenerate)")

    summary = RunSummary(
        run_id=run_id, script_name=os.path.basename(__file__), model_family='mlp_baseline',
        commit=commit, status='success',
        primary_metric_name='mlp_held_rmse', primary_metric_value=g['rmse'],
        secondary_metrics={'mlp_held_mae': g['mae'], 'mlp_held_r2': g['r2']},
        runtime_seconds=float(time.monotonic() - run_start),
        description=f'MLP baseline, {args.val_split}-split, feature_set={args.feature_set}',
        artifacts={'output_dir': str(output_dir), 'model_best': str(output_dir / CHECKPOINT_FILE),
                   'metrics_json': str(output_dir / f'{prefix}_metrics.json')},
        config={'sfc_type': args.sfc_type, 'val_split': args.val_split,
                'feature_set': args.feature_set, 'test_size': args.test_size, 'seed': args.seed},
    )
    with open(output_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, indent=2, sort_keys=True)
    print(f"Saved run summary → {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()

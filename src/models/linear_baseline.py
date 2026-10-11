"""Ridge linear baseline for OCO-2 XCO2 anomaly prediction.

Architecture reference: no external reference (regularized least squares).

The reviewer-proof baseline table (log/archive/LINREG_XGB_BASELINE_PLAN_2026-07-07.md)
asks: what does the production deep ensemble buy over a *linear* model trained on
the *identical* protocol?  This module answers that for the linear leg.  It shares
every non-model piece with the neural / GBDT baselines so the comparison is
apples-to-apples:

  - same raw-dataframe split (models.splits.split_dataframe),
  - same train-only FeaturePipeline (quantile-transformed continuous features +
    fp one-hots + optional profile-EOF/tropopause block),
  - same date-block calibration carve (used here for alpha selection, mirroring
    the DE calib block) and same training_dates.json leakage manifest,
  - same diagnostic suite (models.diagnostics) → identical metrics.json /
    stratified CSV that models.aggregate_folds consumes.

"Linear regression" here therefore means linear in the *transformed* features —
the same representation every other model consumes (state this in the writeup).

This is a *point* predictor (single mu = corrected anomaly).  It has no calibrated
intervals, so for the shared diagnostic suite the three quantile columns are
reported as q05 = q50 = q95 (degenerate, zero-width interval) — only the point
metrics (RMSE / MAE / R²) are meaningful, identical to mlp_baseline.py.
"""

import argparse
import gc
import json
import logging
import os
import platform
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .pipeline import FeaturePipeline, _ensure_derived_features, filter_target_outliers, resolve_target_col
from .splits import split_dataframe, split_date_kfold_train_calib_test
from . import diagnostics as diag
from search.tracking import RunSummary, get_git_commit_hash
from utils import get_storage_dir

logger = logging.getLogger(__name__)

PIPELINE_FILE = 'linear_pipeline.pkl'
MODEL_FILE = 'model_ridge.joblib'
SEED = 42


def _date_list(frame: pd.DataFrame) -> list:
    """Sorted unique date strings (bytes-decoded), matching deep_ensemble.py."""
    if 'date' not in frame.columns:
        return []
    s = frame['date'].astype(str).str.replace("b'", "", regex=False)
    return sorted(s.str.replace("'", "", regex=False).unique().tolist())


def _carve_calib(train_df: pd.DataFrame, calib_frac: float, seed: int) -> tuple:
    """Carve a calibration block out of TRAIN (date split if dates allow, else
    random) — identical logic to deep_ensemble.py so the linear model selects its
    alpha on the SAME calib footprints the DE calibrates on."""
    try:
        if 'date' in train_df.columns and pd.to_datetime(
                train_df['date'].astype(str).str.replace("b'", "").str.replace("'", "")
                if train_df['date'].dtype == object else train_df['date']).nunique() >= 2:
            return split_dataframe(train_df, mode='date', test_size=calib_frac)
        return split_dataframe(train_df, mode='random', test_size=calib_frac, random_state=seed)
    except Exception:
        return split_dataframe(train_df, mode='random', test_size=calib_frac, random_state=seed)


def main():
    parser = argparse.ArgumentParser(description="Ridge linear baseline for XCO2 anomaly prediction")
    parser.add_argument('--sfc_type', type=int, default=0)
    parser.add_argument('--val_split', type=str, default='random',
                        choices=['random', 'date', 'date_kfold'])
    parser.add_argument('--n_folds', type=int, default=None,
                        help='Number of date blocks for --val_split date_kfold.')
    parser.add_argument('--fold', type=int, default=None,
                        help='Which date block (0-based) to hold out for date_kfold.')
    parser.add_argument('--feature_set', type=str, default='full',
                        choices=['full', 'no_xco2', 'no_spec', 'no_xco2_and_spec',
                                 'no_contam', 'no_contam_and_xco2'])
    parser.add_argument('--exclude_snow', dest='exclude_snow', action='store_true',
                        help='Filter OUT snow/ice footprints (snow_flag==1). Default: KEEP snow.')
    parser.add_argument('--profile-pca', dest='profile_pca', nargs='?', const='auto', default=None,
                        help='Append the profile-EOF + tropopause block (ProfilePCA). '
                             'Bare flag / "auto" loads results/profile_pca/profile_pca_<surface>.pkl; '
                             'or pass an explicit .pkl path (fold-specific for leakage-safe kfold).')
    parser.add_argument('--alphas', type=str, default='0.1,1,10,100',
                        help='Comma-separated Ridge alpha grid; the value with the '
                             'lowest RMSE on the calib block is selected (never TCCON).')
    parser.add_argument('--calib_frac', type=float, default=0.15,
                        help='Fraction of TRAIN carved as the calibration block for alpha selection.')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--pipeline', type=str, default=None)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--data', type=str, default=None,
                        help='Override the input data file (default: platform data_name in '
                             'results/csv_collection/). Use for local multi-date testing.')
    parser.add_argument('--target', type=str, default=None,
                        help="Clear-sky reference for the regression target: '10km' (default, "
                             "xco2_bc_anomaly), '15km' (xco2_bc_anomaly_r15), or '5km' (xco2_bc_anomaly_r05).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    alphas = [float(a) for a in args.alphas.split(',') if a.strip()]

    storage_dir = get_storage_dir()
    fdir = storage_dir / 'results/csv_collection'
    data_name = ('combined_2016_2020_dates.parquet' if platform.system() == "Linux"
                 else 'combined_2020-02-01_all_orbits.parquet')
    base_dir = storage_dir / 'results/model_linear_baseline'
    output_dir = base_dir / args.suffix if args.suffix else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.monotonic()
    run_id = args.suffix or datetime.now().strftime('%Y%m%d-%H%M%S')
    commit = get_git_commit_hash(storage_dir)

    # ── Load + filter ──────────────────────────────────────────────────────────
    _dp = args.data if args.data else os.path.join(fdir, data_name)
    df = pd.read_parquet(_dp) if _dp.endswith('.parquet') else pd.read_csv(_dp)
    df = df[df['sfc_type'] == args.sfc_type]
    # Snow footprints KEPT by default (--exclude_snow to drop them).
    if args.exclude_snow:
        df = df[df['snow_flag'] == 0]
    df = _ensure_derived_features(df)
    target_col = resolve_target_col(args.target)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not in parquet; regenerate the combined "
            f"parquet (spectral/fitting.py + build_feature_dataset.py) or pass --target 10km.")
    df = filter_target_outliers(df, target_col=target_col)

    # ── Split RAW first, then carve calibration without touching held-out dates ─
    if args.val_split == 'date_kfold':
        proper_df, calib_df, held_df = split_date_kfold_train_calib_test(
            df,
            n_folds=args.n_folds,
            fold=args.fold,
            calib_frac=args.calib_frac,
        )
        train_df = None
    else:
        train_df, held_df = split_dataframe(df, mode=args.val_split, test_size=args.test_size,
                                            random_state=args.seed,
                                            n_folds=args.n_folds, fold=args.fold)
        proper_df, calib_df = _carve_calib(train_df, args.calib_frac, args.seed)
    del df; gc.collect()

    # Training-date manifest: the machine-readable leakage guard for the TCCON
    # validation chain (same schema as deep_ensemble.py).
    with open(output_dir / 'training_dates.json', 'w', encoding='utf-8') as f:
        json.dump({'train_dates': _date_list(proper_df),
                   'calib_dates': _date_list(calib_df),
                   'held_dates': _date_list(held_df)}, f, indent=2)
    del train_df; gc.collect()

    # ── Fit pipeline on proper-train ONLY (leakage-safe) ───────────────────────
    if args.pipeline:
        pipeline = FeaturePipeline.load(args.pipeline)
        prof_path = args.pipeline
    else:
        _prof = True if args.profile_pca == 'auto' else args.profile_pca
        pipeline = FeaturePipeline.fit(proper_df, sfc_type=args.sfc_type,
                                       feature_set=args.feature_set, profile_pca=_prof)
        pipeline.save(output_dir / PIPELINE_FILE)
        prof_path = str(args.profile_pca) if args.profile_pca else None

    def _prep(frame):
        X = pipeline.transform(frame)
        y = frame[target_col].to_numpy(dtype=np.float32)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        return X[valid], y[valid], frame.loc[valid]

    X_tr, y_tr, _ = _prep(proper_df)
    X_ca, y_ca, _ = _prep(calib_df)
    X_held, y_held, held_valid = _prep(held_df)
    print(f"[linreg] X_train {X_tr.shape}  X_calib {X_ca.shape}  X_held {X_held.shape}")

    # ── Alpha selection on the calib block (RMSE) ──────────────────────────────
    best_alpha, best_rmse, best_model = None, np.inf, None
    for a in alphas:
        m = Ridge(alpha=a, random_state=args.seed)
        m.fit(X_tr, y_tr)
        rmse = float(np.sqrt(np.mean((y_ca - m.predict(X_ca)) ** 2)))
        logger.info("alpha=%g  calib RMSE=%.4f", a, rmse)
        if rmse < best_rmse:
            best_alpha, best_rmse, best_model = a, rmse, m
    print(f"[linreg] selected alpha={best_alpha:g} (calib RMSE={best_rmse:.4f})")

    joblib.dump(best_model, output_dir / MODEL_FILE)
    with open(output_dir / 'coef.json', 'w', encoding='utf-8') as f:
        json.dump({'alpha': best_alpha, 'intercept': float(best_model.intercept_),
                   'features': list(pipeline.features),
                   'coef': [float(c) for c in np.ravel(best_model.coef_)]}, f, indent=2)

    # ── Diagnostics (point predictor → degenerate intervals q05=q50=q95) ───────
    mu = best_model.predict(X_held)
    preds = np.column_stack([mu, mu, mu])
    g = diag.compute_metrics(y_held, preds)
    strat = diag.stratified_metrics(held_valid, y_held, preds)
    prefix = f"linreg_{args.val_split}"
    diag.save_diagnostics(output_dir, prefix, g, strat)
    diag.save_correction_and_preds(output_dir, prefix, held_valid, y_held, preds)
    print(f"[{prefix}] RMSE={g['rmse']:.4f}  MAE={g['mae']:.4f}  R²={g['r2']:.4f}  "
          f"(point predictor — interval metrics are degenerate)")

    summary = RunSummary(
        run_id=run_id, script_name=os.path.basename(__file__), model_family='linear_baseline',
        commit=commit, status='success',
        primary_metric_name='linreg_held_rmse', primary_metric_value=g['rmse'],
        secondary_metrics={'linreg_held_mae': g['mae'], 'linreg_held_r2': g['r2'],
                           'selected_alpha': best_alpha, 'calib_rmse': best_rmse},
        runtime_seconds=float(time.monotonic() - run_start),
        description=f'Ridge baseline, {args.val_split}-split, feature_set={args.feature_set}, alpha={best_alpha:g}',
        artifacts={'output_dir': str(output_dir), 'model': str(output_dir / MODEL_FILE),
                   'metrics_json': str(output_dir / f'{prefix}_metrics.json')},
        config={'sfc_type': args.sfc_type, 'val_split': args.val_split,
                'feature_set': args.feature_set, 'target': target_col,
                'alphas': alphas, 'selected_alpha': best_alpha,
                'profile_pca': prof_path, 'calib_frac': args.calib_frac,
                'n_folds': args.n_folds, 'fold': args.fold, 'seed': args.seed},
    )
    with open(output_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, indent=2, sort_keys=True)
    print(f"Saved run summary → {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()

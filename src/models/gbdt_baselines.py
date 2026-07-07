"""GBDT quantile baselines (XGBoost / LightGBM) for OCO-2 XCO2 anomaly prediction.

Architecture reference: no single paper — cite the libraries used:
  Chen & Guestrin. "XGBoost: A Scalable Tree Boosting System." KDD 2016.
  Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017.

This module deviates from a plain regressor as follows:
  - One model is trained per quantile (q05, q50, q95) — see TABM_PLAN.md.
  - Quantile crossing (q05 ≥ q95 on some samples) is possible because the three
    models are independent; metrics are reported both as-is and after monotone
    rearrangement so the GBDT is not penalised where the neural monotonic head
    has a structural guarantee.

XGBoost uses ``objective="reg:quantileerror"`` (XGBoost ≥ 1.7).  If unavailable,
falls back to LightGBM, then to sklearn ``GradientBoostingRegressor(loss="quantile")``.
LightGBM uses native ``objective="quantile"``; it is imported lazily and the run
fails with a clear message only if --model lightgbm is requested without it installed.

Determinism: seeds are fixed and single-threaded settings are used where
available (the plan notes GBDTs are not inherently deterministic under parallel
execution — verify two runs match before relying on a single run).
"""

import argparse
import gc
import json
import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .pipeline import FeaturePipeline, _ensure_derived_features, filter_target_outliers, resolve_target_col
from .splits import split_dataframe
from . import diagnostics as diag
from search.tracking import RunSummary, get_git_commit_hash
from utils import get_storage_dir

logger = logging.getLogger(__name__)

QUANTILES = (0.05, 0.5, 0.95)
SEED = 42


def _date_list(frame) -> list:
    """Sorted unique date strings (bytes-decoded), matching deep_ensemble.py."""
    if frame is None or 'date' not in frame.columns:
        return []
    s = frame['date'].astype(str).str.replace("b'", "", regex=False)
    return sorted(s.str.replace("'", "", regex=False).unique().tolist())


def _carve_calib(train_df, calib_frac: float, seed: int) -> tuple:
    """Carve a calib block out of TRAIN (date split if ≥2 dates, else random) —
    identical logic to deep_ensemble.py, used here for --objective mean early
    stopping so the mean model stops on the SAME calib footprints the DE uses."""
    try:
        if 'date' in train_df.columns and pd.to_datetime(
                train_df['date'].astype(str).str.replace("b'", "").str.replace("'", "")
                if train_df['date'].dtype == object else train_df['date']).nunique() >= 2:
            return split_dataframe(train_df, mode='date', test_size=calib_frac)
        return split_dataframe(train_df, mode='random', test_size=calib_frac, random_state=seed)
    except Exception:
        return split_dataframe(train_df, mode='random', test_size=calib_frac, random_state=seed)


# ─── Capability checks ─────────────────────────────────────────────────────────

def _xgboost_has_quantile() -> bool:
    try:
        import xgboost as xgb
        major = int(xgb.__version__.split('.')[0])
        return major >= 2 or xgb.__version__ >= '1.7'
    except Exception:
        return False


def _xgb_version() -> 'str | None':
    """xgboost version string (for the run summary — reproducibility with n_jobs>1)."""
    try:
        import xgboost as xgb
        return str(xgb.__version__)
    except Exception:
        return None


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


# ─── Per-quantile training ─────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, quantiles=QUANTILES, cfg: 'dict | None' = None) -> dict:
    """Train one XGBoost quantile regressor per alpha.  Returns {alpha: model}."""
    import xgboost as xgb
    cfg = cfg or {}
    if not _xgboost_has_quantile():
        raise RuntimeError(
            "Installed XGBoost lacks reg:quantileerror (needs ≥1.7). "
            "Use --model lightgbm or upgrade xgboost."
        )
    models = {}
    for q in quantiles:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=int(cfg.get('n_estimators', 500)),
            max_depth=int(cfg.get('max_depth', 6)),
            learning_rate=float(cfg.get('learning_rate', 0.05)),
            subsample=float(cfg.get('subsample', 1.0)),
            colsample_bytree=float(cfg.get('colsample_bytree', 0.8)),
            reg_lambda=float(cfg.get('reg_lambda', 1.0)),
            n_jobs=int(cfg.get('n_jobs', 1)),       # single-threaded → deterministic
            random_state=SEED,
            tree_method='hist',
        )
        model.fit(X_train, y_train)
        models[q] = model
        logger.info("Trained XGBoost q%.2f", q)
    return models


def train_xgboost_mean(X_train, y_train, X_calib, y_calib, cfg: 'dict | None' = None):
    """Train ONE plain XGBoost mean regressor (reg:squarederror); prediction = mu.

    Early stopping is evaluated on the calibration block only (never TCCON, never
    the held fold), so n_estimators is chosen the same way the DE picks its
    epochs.  Returns the fitted XGBRegressor.
    """
    import xgboost as xgb
    cfg = cfg or {}
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(cfg.get('n_estimators', 2000)),
        max_depth=int(cfg.get('max_depth', 6)),
        learning_rate=float(cfg.get('learning_rate', 0.05)),
        subsample=float(cfg.get('subsample', 1.0)),
        colsample_bytree=float(cfg.get('colsample_bytree', 0.8)),
        reg_lambda=float(cfg.get('reg_lambda', 1.0)),
        n_jobs=int(cfg.get('n_jobs', 1)),
        random_state=SEED,
        tree_method='hist',
        early_stopping_rounds=int(cfg.get('early_stopping_rounds', 50)),
        eval_metric='rmse',
    )
    model.fit(X_train, y_train, eval_set=[(X_calib, y_calib)], verbose=False)
    best_it = getattr(model, 'best_iteration', None)
    logger.info("Trained XGBoost mean regressor (best_iteration=%s)", best_it)
    return model


def train_lightgbm(X_train, y_train, quantiles=QUANTILES, cfg: 'dict | None' = None) -> dict:
    """Train one LightGBM quantile regressor per alpha.  Returns {alpha: model}."""
    import lightgbm as lgb
    cfg = cfg or {}
    models = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=int(cfg.get('n_estimators', 500)),
            max_depth=int(cfg.get('max_depth', -1)),
            num_leaves=int(cfg.get('num_leaves', 63)),
            learning_rate=float(cfg.get('learning_rate', 0.05)),
            subsample=float(cfg.get('subsample', 1.0)),
            colsample_bytree=float(cfg.get('colsample_bytree', 0.8)),
            reg_lambda=float(cfg.get('reg_lambda', 1.0)),
            num_threads=int(cfg.get('num_threads', 1)),   # deterministic
            random_state=SEED,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        models[q] = model
        logger.info("Trained LightGBM q%.2f", q)
    return models


def _sklearn_gbdt_fallback(X_train, y_train, quantiles=QUANTILES, cfg: 'dict | None' = None) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    cfg = cfg or {}
    models = {}
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=int(cfg.get('n_estimators', 500)),
            max_depth=int(cfg.get('max_depth', 3)),
            learning_rate=float(cfg.get('learning_rate', 0.05)),
            subsample=float(cfg.get('subsample', 1.0)),
            random_state=SEED,
        )
        model.fit(X_train, y_train)
        models[q] = model
        logger.info("Trained sklearn GBDT q%.2f (fallback)", q)
    return models


def _predict_quantiles(models: dict, X: np.ndarray, quantiles=QUANTILES) -> np.ndarray:
    """Stack per-quantile predictions into [N, 3] (q05, q50, q95)."""
    return np.column_stack([models[q].predict(X) for q in quantiles]).astype(float)


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_gbdt(models: dict, X_test, y_test, meta: pd.DataFrame,
                  output_dir, output_prefix: str, quantiles=QUANTILES) -> dict:
    """Shared GBDT evaluation: same diagnostic suite as the neural models.

    Reports metrics both as-is (crossing possible) and after monotone
    rearrangement, writing {prefix}_metrics.json / {prefix}_stratified_metrics.csv
    for each variant.  Returns the as-is global metric dict.
    """
    preds = _predict_quantiles(models, X_test, quantiles)

    # As-is (independent per-quantile models → crossing possible)
    g_raw = diag.compute_metrics(y_test, preds, quantiles=quantiles)
    s_raw = diag.stratified_metrics(meta, y_test, preds, quantiles=quantiles)
    c_raw = diag.calibration_report(g_raw, s_raw)
    diag.save_diagnostics(output_dir, output_prefix, g_raw, s_raw, c_raw)

    # Monotone-rearranged (fair interval-based comparison vs the monotonic head)
    preds_re = diag.monotone_rearrange(preds)
    g_re = diag.compute_metrics(y_test, preds_re, quantiles=quantiles)
    s_re = diag.stratified_metrics(meta, y_test, preds_re, quantiles=quantiles)
    c_re = diag.calibration_report(g_re, s_re)
    diag.save_diagnostics(output_dir, f"{output_prefix}_rearranged", g_re, s_re, c_re)

    # Correction-vs-cloud-distance + per-sounding dump (point = q50, same raw/re)
    diag.save_correction_and_preds(output_dir, output_prefix, meta, y_test, preds_re)

    print(f"[{output_prefix}] RMSE={g_raw['rmse']:.4f}  MAE={g_raw['mae']:.4f}  "
          f"R²={g_raw['r2']:.4f}  cov90={g_raw['coverage_90']:.3f}  "
          f"width={g_raw['mean_interval_width']:.3f}  cross={g_raw['crossing_rate']:.3g}  "
          f"(rearranged cross={g_re['crossing_rate']:.3g})")
    return g_raw


# ─── CLI / orchestration ───────────────────────────────────────────────────────

def _resolve_algo(requested: str) -> str:
    """Map a requested algo to one that is actually available, with fallbacks."""
    if requested == 'lightgbm':
        if _lightgbm_available():
            return 'lightgbm'
        raise RuntimeError("lightgbm not installed; install it or use --model xgboost.")
    # xgboost requested (default)
    if _xgboost_has_quantile():
        return 'xgboost'
    if _lightgbm_available():
        logger.warning("XGBoost quantile unavailable — falling back to LightGBM.")
        return 'lightgbm'
    logger.warning("Neither XGBoost-quantile nor LightGBM available — using sklearn GBDT.")
    return 'sklearn'


def main():
    parser = argparse.ArgumentParser(description="GBDT quantile baselines for XCO2 anomaly prediction")
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm'])
    parser.add_argument('--objective', type=str, default='quantile',
                        choices=['quantile', 'mean'],
                        help="'quantile' (default): 3 per-quantile models (q05/q50/q95). "
                             "'mean': one plain reg:squarederror model, prediction = mu "
                             "(early-stopped on the calib block) — the parsimonious "
                             "'plain XGBoost regression' baseline. xgboost only.")
    parser.add_argument('--calib_frac', type=float, default=0.15,
                        help='Fraction of TRAIN carved as the calib block for --objective mean '
                             'early stopping (mirrors deep_ensemble.py).')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='XGBoost/LightGBM threads. Default: SLURM_NTASKS (CPUS_PER_TASK) '
                             'or os.cpu_count(). NOTE: >1 makes the hist builder NOT bit-'
                             'reproducible (seeds are still fixed; xgboost version is recorded). '
                             'Pass 1 for strict determinism at the cost of wall-clock.')
    parser.add_argument('--sfc_type', type=int, default=0)
    parser.add_argument('--val_split', type=str, default='random',
                        choices=['random', 'date', 'date_kfold'])
    parser.add_argument('--n_folds', type=int, default=None,
                        help='Number of date blocks for --val_split date_kfold.')
    parser.add_argument('--fold', type=int, default=None,
                        help='Which date block (0-based) to hold out for date_kfold.')
    parser.add_argument('--feature_set', type=str, default='full',
                        choices=['full', 'no_xco2', 'no_spec', 'no_xco2_and_spec'])
    parser.add_argument('--exclude_snow', dest='exclude_snow', action='store_true',
                        help='Filter OUT snow/ice footprints (snow_flag==1). Default: KEEP snow.')
    parser.add_argument('--profile-pca', dest='profile_pca', nargs='?', const='auto', default=None,
                        help='Append the profile-EOF + tropopause block (ProfilePCA). '
                             'Bare flag / "auto" loads results/profile_pca/profile_pca_<surface>.pkl; '
                             'or pass an explicit .pkl path. Orthogonal to --feature_set: it is '
                             'carried through every set (no_xco2/no_spec keep it). This makes '
                             'full+profile the "new full".')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--pipeline', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the module SEED for this run (seed sweep).')
    parser.add_argument('--data', type=str, default=None,
                        help='Override the input data file (default: platform data_name in '
                             'results/csv_collection/). Use for local multi-date testing.')
    parser.add_argument('--target', type=str, default=None,
                        help="Clear-sky reference for the regression target: '10km' (default, xco2_bc_anomaly), '15km' (xco2_bc_anomaly_r15), or '5km' (xco2_bc_anomaly_r05).")
    args = parser.parse_args()

    algo = _resolve_algo(args.model)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Thread count: explicit --n_jobs, else SLURM allocation, else all local cores.
    n_jobs = args.n_jobs if args.n_jobs is not None else int(
        os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_NTASKS') or os.cpu_count() or 1)
    tree_cfg = {'n_jobs': n_jobs, 'num_threads': n_jobs}
    logger.info("tree threads n_jobs=%d (bit-reproducible only if n_jobs==1)", n_jobs)

    if args.seed is not None:
        global SEED
        SEED = int(args.seed)   # flows into model random_state + random split below

    storage_dir = get_storage_dir()
    fdir = storage_dir / 'results/csv_collection'
    data_name = ('combined_2016_2020_dates.parquet' if platform.system() == "Linux"
                 else 'combined_2020-02-01_all_orbits.parquet')
    base_dir = storage_dir / 'results/model_gbdt'
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
        raise ValueError(f"Target column '{target_col}' not in parquet; regenerate the combined parquet (spectral/fitting.py + build_feature_dataset.py) or pass --target 10km.")
    df = filter_target_outliers(df, target_col=target_col)

    if args.objective == 'mean' and algo != 'xgboost':
        raise SystemExit("--objective mean is xgboost-only (needs reg:squarederror + early stopping). "
                         "Drop --model lightgbm or use the default quantile objective.")

    # ── Split RAW first, fit pipeline on (proper-)train only ───────────────────
    train_df, held_df = split_dataframe(df, mode=args.val_split, test_size=args.test_size,
                                        random_state=SEED,
                                        n_folds=args.n_folds, fold=args.fold)
    del df
    gc.collect()

    # Mean mode carves a calib block out of TRAIN for early stopping (mirrors
    # deep_ensemble.py); the pipeline is fit on proper-train only.  Quantile mode
    # keeps its original behavior (pipeline fit on full TRAIN, no calib block).
    calib_df = None
    if args.objective == 'mean':
        proper_df, calib_df = _carve_calib(train_df, args.calib_frac, SEED)
        fit_df = proper_df
    else:
        fit_df = train_df

    # Training-date manifest: machine-readable leakage guard for the TCCON chain
    # (same schema as deep_ensemble.py; calib_dates empty for quantile mode).
    with open(output_dir / 'training_dates.json', 'w', encoding='utf-8') as f:
        json.dump({'train_dates': _date_list(fit_df),
                   'calib_dates': _date_list(calib_df) if calib_df is not None else [],
                   'held_dates': _date_list(held_df)}, f, indent=2)

    if args.pipeline:
        pipeline = FeaturePipeline.load(args.pipeline)
        prof_path = args.pipeline
    else:
        _prof = True if args.profile_pca == 'auto' else args.profile_pca
        pipeline = FeaturePipeline.fit(fit_df, sfc_type=args.sfc_type,
                                       feature_set=args.feature_set, profile_pca=_prof)
        pipeline.save(output_dir / 'gbdt_pipeline.pkl')
        prof_path = str(args.profile_pca) if args.profile_pca else None
    features = pipeline.features

    def _prep(frame):
        X = pipeline.transform(frame)
        y = frame[target_col].to_numpy(dtype=np.float32)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        return X[valid], y[valid], frame.loc[valid]

    X_train, y_train, _ = _prep(fit_df)
    X_held, y_held, held_valid = _prep(held_df)
    print(f"[{algo}/{args.objective}] X_train {X_train.shape}  X_held {X_held.shape}")

    # ── Mean objective: one plain regressor, prediction = mu ───────────────────
    if args.objective == 'mean':
        X_ca, y_ca, _ = _prep(calib_df)
        print(f"[{algo}/mean] X_calib {X_ca.shape}")
        model = train_xgboost_mean(X_train, y_train, X_ca, y_ca, cfg=tree_cfg)
        joblib.dump(model, output_dir / f'model_mean_{algo}.joblib')
        logger.info("Saved %s mean regressor → %s", algo, output_dir)

        mu = model.predict(X_held)
        preds = np.column_stack([mu, mu, mu])   # degenerate intervals (point predictor)
        prefix = f"{algo}_mean_{args.val_split}"
        g = diag.compute_metrics(y_held, preds)
        strat = diag.stratified_metrics(held_valid, y_held, preds)
        diag.save_diagnostics(output_dir, prefix, g, strat)
        diag.save_correction_and_preds(output_dir, prefix, held_valid, y_held, preds)
        print(f"[{prefix}] RMSE={g['rmse']:.4f}  MAE={g['mae']:.4f}  R²={g['r2']:.4f}  "
              f"(point predictor — interval metrics are degenerate)")
        _bi = getattr(model, 'best_iteration', None)
        secondary = {f'{algo}_held_mae': g['mae'], f'{algo}_held_r2': g['r2'],
                     'best_iteration': int(_bi) if _bi is not None else None}
        description = f'{algo} MEAN baseline, {args.val_split}-split, feature_set={args.feature_set}'
    else:
        # ── Quantile objective: one model per quantile ─────────────────────────
        if algo == 'xgboost':
            models = train_xgboost(X_train, y_train, cfg=tree_cfg)
        elif algo == 'lightgbm':
            models = train_lightgbm(X_train, y_train, cfg=tree_cfg)
        else:
            models = _sklearn_gbdt_fallback(X_train, y_train)
        for q, model in models.items():
            joblib.dump(model, output_dir / f'model_q{int(round(q*100)):02d}_{algo}.joblib')
        logger.info("Saved %d %s quantile models → %s", len(models), algo, output_dir)
        prefix = f"{algo}_{args.val_split}"
        g = evaluate_gbdt(models, X_held, y_held, held_valid, output_dir, prefix)
        secondary = {
            f'{algo}_held_mae': g['mae'], f'{algo}_held_r2': g['r2'],
            'coverage_90': g['coverage_90'], 'mean_interval_width': g['mean_interval_width'],
            'crossing_rate': g['crossing_rate'],
            'pinball_q05': g['pinball_q05'], 'pinball_q50': g['pinball_q50'],
            'pinball_q95': g['pinball_q95'],
        }
        description = f'{algo} quantile baseline, {args.val_split}-split, feature_set={args.feature_set}'

    summary = RunSummary(
        run_id=run_id, script_name=os.path.basename(__file__), model_family=f'gbdt_{algo}',
        commit=commit, status='success',
        primary_metric_name=f'{algo}_held_rmse', primary_metric_value=g['rmse'],
        secondary_metrics=secondary,
        runtime_seconds=float(time.monotonic() - run_start),
        description=description,
        artifacts={'output_dir': str(output_dir), 'metrics_json': str(output_dir / f'{prefix}_metrics.json')},
        config={'model': algo, 'objective': args.objective, 'sfc_type': args.sfc_type,
                'val_split': args.val_split, 'feature_set': args.feature_set,
                'target': target_col, 'profile_pca': prof_path, 'n_jobs': n_jobs,
                'xgboost_version': _xgb_version(),
                'calib_frac': args.calib_frac, 'n_folds': args.n_folds, 'fold': args.fold,
                'test_size': args.test_size},
    )
    with open(output_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, indent=2, sort_keys=True)
    print(f"Saved run summary → {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()

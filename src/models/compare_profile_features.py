"""compare_profile_features.py — A/B test: does adding the profile EOF + tropopause
block to the `full` feature set improve XGBoost prediction of the XCO2 anomaly?

For each surface (0=ocean, 1=land) it trains TWO XGBoost models on an IDENTICAL
train/test split, differing only by the appended profile block:

    A) full          = FeaturePipeline(feature_set='full').transform(df)
    B) full+profile  = A ⊕ ProfilePCA_<surface>.transform(df)
                       (12 EOF scores + tropopause_sigma + tropopause_temp)

Fairness controls
-----------------
* Same rows for both: restricted to rows finite in target ∧ full-features ∧
  profile-features.  ~51 % of 2016-2020 rows lack the reanalysis-profile columns
  (older fitting.py), and their NaN pattern tracks processing batch/date — leaving
  them in (with XGBoost's native NaN handling) would let model B exploit a spurious
  date proxy.  Restricting to profile-present rows isolates the *physical*
  information the EOFs carry.
* Same split (same indices, same seed), same XGBoost hyperparameters + custom
  Huber+L1 objective (imported from xgb.py), same eval/early-stopping set.

Caveats (printed with the results)
* Early stopping and reporting both use the test split (repo convention) — the
  delta between A and B is fair, the absolute R² is mildly optimistic.
* Metrics live on the profile-present subset, which skews toward newer dates.

Run (from src/):
    python -m models.compare_profile_features            # both surfaces
    python -m models.compare_profile_features --sfc_type 1 --max-rows 2000000
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .pipeline import (FeaturePipeline, _resolve_feature_set, _FEATURE_MAP,
                       _DERIVED_FEATURES, resolve_target_col, filter_target_outliers)
from .xgb import _build_huber_l1_pred_obj, _default_run_config, _cuda_available
from .profile_pca import ProfilePCA, _PROFILE_GROUPS, _SCALAR_PASSTHROUGH
from utils import get_storage_dir

logger = logging.getLogger(__name__)

SURF = {0: 'ocean', 1: 'land'}


def _needed_columns(schema_names, target_col):
    """Union of every column FeaturePipeline('full') + ProfilePCA may touch."""
    need = set()
    for sfc in (0, 1):
        need |= set(_resolve_feature_set(_FEATURE_MAP[sfc], 'full', sfc))
    # Include both the derived column NAMES (precomputed in the parquet → read &
    # skipped by _ensure_derived_features) and their base columns (fallback derive).
    need |= set(_DERIVED_FEATURES.keys())
    for _col, (_expr, bases) in _DERIVED_FEATURES.items():
        need |= set(bases)
    need |= {'fp', 'sfc_type', 'snow_flag', target_col}
    need |= {c for c in schema_names
             if any(c.startswith(p) for p in _PROFILE_GROUPS)}
    need |= set(_SCALAR_PASSTHROUGH)
    return [c for c in schema_names if c in need]


def _train_xgb(X_tr, y_tr, X_te, y_te, xgb_cfg, obj_cfg, device):
    """Fit one XGBRegressor with the repo's config; return (model, r2, mae)."""
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators          = int(xgb_cfg['n_estimators']),
        max_depth             = int(xgb_cfg['max_depth']),
        learning_rate         = float(xgb_cfg['learning_rate']),
        subsample             = float(xgb_cfg['subsample']),
        colsample_bytree      = float(xgb_cfg['colsample_bytree']),
        min_child_weight      = float(xgb_cfg['min_child_weight']),
        gamma                 = float(xgb_cfg['gamma']),
        reg_lambda            = float(xgb_cfg['reg_lambda']),
        reg_alpha             = float(xgb_cfg['reg_alpha']),
        objective             = _build_huber_l1_pred_obj(
            delta=float(obj_cfg['huber_delta']), lam=float(obj_cfg['pred_l1_weight'])),
        tree_method           = xgb_cfg['tree_method'],
        device                = device,
        early_stopping_rounds = int(xgb_cfg['early_stopping_rounds']),
        eval_metric           = xgb_cfg['eval_metric'],
        n_jobs                = int(xgb_cfg['n_jobs']),
        verbosity             = 0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    yp = model.predict(X_te)
    ss_res = float(((y_te - yp) ** 2).sum())
    ss_tot = float(((y_te - y_te.mean()) ** 2).sum())
    r2  = 1.0 - ss_res / ss_tot
    mae = float(np.abs(y_te - yp).mean())
    return model, r2, mae


def run_surface(df_sfc, sfc_type, ppca_path, target_col, cfg, max_rows, seed):
    surf = SURF[sfc_type]
    print(f"\n{'='*70}\n  SURFACE {sfc_type} ({surf})\n{'='*70}", flush=True)

    # Drop non-physical target outliers (|anomaly|>100 ppm): a handful of
    # fill-value/QF escapees dominate squared error and collapse land R² to ~0
    # while barely moving MAE (see pipeline.MAX_ABS_ANOMALY_PPM).  Applied to the
    # RAW df so both models see the identical target distribution.
    n0 = len(df_sfc)
    df_sfc = filter_target_outliers(df_sfc, target_col=target_col)
    if len(df_sfc) < n0:
        print(f"  dropped {n0 - len(df_sfc):,} |{target_col}|>100 ppm outliers", flush=True)

    # A) full feature block
    pipe = FeaturePipeline.fit(df_sfc, sfc_type=sfc_type, feature_set='full')
    X_full = pipe.transform(df_sfc)
    full_names = list(pipe.features)

    # B) profile block from the surface's saved ProfilePCA
    ppca = ProfilePCA.load(ppca_path)
    X_prof = ppca.transform(df_sfc)
    prof_names = list(ppca.feature_names)

    y = df_sfc[target_col].to_numpy(np.float32)

    # Identical rows: finite in target ∧ full ∧ profile
    valid = (np.isfinite(y)
             & np.isfinite(X_full).all(axis=1)
             & np.isfinite(X_prof).all(axis=1))
    X_full, X_prof, y = X_full[valid], X_prof[valid], y[valid]
    n = len(y)
    print(f"  profile-present, target+feature-finite rows: {n:,}", flush=True)

    if max_rows and n > max_rows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, max_rows, replace=False)
        X_full, X_prof, y = X_full[keep], X_prof[keep], y[keep]
        print(f"  subsampled to {max_rows:,} rows", flush=True)

    X_aug = np.concatenate([X_full, X_prof], axis=1).astype(np.float32)
    aug_names = full_names + prof_names

    # One split, shared by both models (index-based → identical rows)
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=float(cfg['split']['test_size']),
                                random_state=seed)
    device = 'cuda' if _cuda_available() else 'cpu'
    print(f"  train={len(itr):,}  test={len(ite):,}  device={device}  "
          f"|A|={X_full.shape[1]}  |B|={X_aug.shape[1]}", flush=True)

    print("  training A (full) …", flush=True)
    _, r2_a, mae_a = _train_xgb(X_full[itr], y[itr], X_full[ite], y[ite],
                                cfg['xgb'], cfg['objective'], device)
    print(f"    full:          R²={r2_a:.4f}  MAE={mae_a:.4f} ppm", flush=True)

    print("  training B (full+profile) …", flush=True)
    model_b, r2_b, mae_b = _train_xgb(X_aug[itr], y[itr], X_aug[ite], y[ite],
                                      cfg['xgb'], cfg['objective'], device)
    print(f"    full+profile:  R²={r2_b:.4f}  MAE={mae_b:.4f} ppm", flush=True)
    print(f"    Δ:             ΔR²={r2_b-r2_a:+.4f}  ΔMAE={mae_b-mae_a:+.4f} ppm", flush=True)

    # Where do the profile features rank by gain in model B?
    booster = model_b.get_booster()
    gain = booster.get_score(importance_type='gain')
    gain = {aug_names[int(k[1:])]: v for k, v in gain.items()
            if k[1:].isdigit() and int(k[1:]) < len(aug_names)}
    ranked = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
    rank_of = {f: i + 1 for i, (f, _) in enumerate(ranked)}
    prof_ranks = [(f, rank_of.get(f), gain.get(f, 0.0)) for f in prof_names]
    prof_ranks = [(f, r, g) for f, r, g in prof_ranks if r is not None]
    prof_ranks.sort(key=lambda t: t[1])
    print(f"  profile-feature gain ranks (of {len(ranked)} used):", flush=True)
    for f, r, g in prof_ranks[:8]:
        print(f"    #{r:<3d} {f:<20s} gain={g:.1f}", flush=True)
    top_prof_rank = prof_ranks[0][1] if prof_ranks else None

    del X_full, X_prof, X_aug, y
    gc.collect()
    return {'surface': surf, 'sfc_type': sfc_type, 'n_rows': n,
            'n_features_full': len(full_names), 'n_features_aug': len(aug_names),
            'r2_full': r2_a, 'mae_full': mae_a,
            'r2_aug': r2_b, 'mae_aug': mae_b,
            'delta_r2': r2_b - r2_a, 'delta_mae': mae_b - mae_a,
            'best_profile_gain_rank': top_prof_rank}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=None,
                    help='combined parquet (default: 2016-2020 under storage_dir)')
    ap.add_argument('--ppca-dir', default=None,
                    help='dir with profile_pca_<surface>.pkl (default results/model_mlp_lr)')
    ap.add_argument('--sfc_type', type=int, default=None, choices=[0, 1],
                    help='one surface only (default: both)')
    ap.add_argument('--target', default='10km', help="'10km'(default)/'15km'/col name")
    ap.add_argument('--max-rows', type=int, default=None,
                    help='cap rows per surface for speed (default: use all profile-present)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-estimators', type=int, default=None,
                    help='override XGB n_estimators cap (default: repo config 12000). '
                         'Early stopping still applies; lower bounds runtime for the A/B.')
    ap.add_argument('--out', default=None, help='summary CSV path')
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    storage = get_storage_dir()
    import platform
    _default = ('combined_2016_2020_dates.parquet' if platform.system() == 'Linux'
                else 'combined_2020_dates.parquet')
    data_path = Path(args.data) if args.data else storage / 'results/csv_collection' / _default
    ppca_dir  = Path(args.ppca_dir) if args.ppca_dir else storage / 'results/model_mlp_lr'
    out_csv   = Path(args.out) if args.out else storage / 'results/model_xgb/profile_ab_summary.csv'
    target_col = resolve_target_col(args.target)
    cfg = _default_run_config()
    if args.n_estimators is not None:
        cfg['xgb']['n_estimators'] = int(args.n_estimators)

    import pyarrow.parquet as pq
    schema_names = pq.read_schema(str(data_path)).names
    cols = _needed_columns(schema_names, target_col)
    print(f"Reading {len(cols)}/{len(schema_names)} columns from {data_path.name}", flush=True)
    if target_col not in schema_names:
        raise ValueError(f"target {target_col!r} not in parquet — regenerate or pass --target 10km")

    surfaces = [args.sfc_type] if args.sfc_type is not None else [0, 1]
    rows = []
    for sfc in surfaces:
        # predicate pushdown: read only this surface's snow-free rows
        df = pd.read_parquet(str(data_path), columns=cols,
                             filters=[('sfc_type', '==', sfc), ('snow_flag', '==', 0)])
        ppca_path = ppca_dir / f'profile_pca_{SURF[sfc]}.pkl'
        rows.append(run_surface(df, sfc, ppca_path, target_col, cfg, args.max_rows, args.seed))
        del df
        gc.collect()

    summary = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False, float_format='%.5f')
    print(f"\n{'='*70}\nSUMMARY  (target={target_col})\n{'='*70}", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved → {out_csv}", flush=True)


if __name__ == '__main__':
    main()

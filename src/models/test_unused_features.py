"""test_unused_features.py — with the NEW `full` = full + profile-EOF + tropopause
as the fixed baseline, does adding any of the currently-DISABLED features in
pipeline.py (_FEATURES_SFC0/1 commented-out entries) further help XGBoost predict
the XCO2 anomaly?

Per surface, leave-one-in ablation:
    baseline = FeaturePipeline('full') ⊕ ProfilePCA_<surface>   (frozen)
    for each candidate c not already in baseline:
        train XGBoost on [baseline | c]  vs  baseline
        report ΔR², ΔMAE (same rows, same split, same hyperparameters)

Fairness: rows fixed to finite(target ∧ full ∧ profile); a candidate column is
appended raw (XGBoost is scale-invariant) and its NaNs are handled natively by
XGBoost — so the row set is identical across every candidate.  Each candidate's
NaN fraction is reported so high-missingness "wins" can be discounted.  Target
outliers (|anomaly|>100 ppm) are dropped by default (as in the trainers).

Run (from repo root):
    PYTHONPATH=src python -m models.test_unused_features \
      --data <combined_2020_dates.parquet> --ppca-dir results/profile_pca
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .pipeline import (FeaturePipeline, resolve_target_col, filter_target_outliers)
from .profile_pca import ProfilePCA
from .compare_profile_features import _train_xgb, _needed_columns, SURF
from utils import get_storage_dir

logger = logging.getLogger(__name__)

# Disabled features in pipeline.py _FEATURES_SFC0/1 (commented-out) + the named
# KAPPA_FEATURES group — the "unused features in pipeline.py".  Filtered at runtime
# to those present in the parquet and NOT already in that surface's new `full`
# baseline (some are active on the other surface or enter via the contam set).
CANDIDATE_POOL = [
    # xco2-derived
    'xco2_bc_minus_raw', 'xco2_strong_idp', 'xco2_weak_idp',
    'xco2_strong_idp_minus_apriori', 'xco2_weak_idp_minus_apriori',
    'co2_ratio_bc',
    # spectral intercepts / higher k
    'o2a_intercept', 'wco2_intercept', 'sco2_intercept',
    'exp_wco2_intercept', 'exp_sco2_intercept',
    'wco2_exp_intercept-alb', 'sco2_exp_intercept-alb',
    'o2a_k3', 'sco2_k3',
    'o2a_k2_over_k1', 'wco2_k2_over_k1', 'sco2_k2_over_k1',
    # gamma-shape kappa (re-confirm on new baseline; prior A/B said it hurts)
    'o2a_kappa', 'wco2_kappa', 'sco2_kappa',
    # geometry / viewing
    '1_over_cos_sza', '1_over_cos_vza', 'mu_sza', 'mu_vza',
    'sin_raa', 'cos_raa', 'cos_theta', 'Phi_cos_theta', 'R_rs_factor',
    'cos_glint_angle', 'glint_prox',
    # albedo (raw + normalised + cross-band)
    'alb_o2a', 'alb_wco2', 'alb_sco2',
    'alb_o2a_over_cos_sza', 'alb_wco2_over_cos_sza', 'alb_sco2_over_cos_sza',
    # surface / atmosphere state
    'alt', 'alt_std', 'ws', 'ws_apriori', 'airmass', 'airmass_sq',
    'dp', 'dp_abp', 'dpfrac', 'dp_psfc_ratio', 'fs_rel_0',
    # aerosol
    'aod_total', 'aod_bc', 'aod_ice', 'aod_water', 'dws',
    'dust_height', 'ice_height', 'water_height',
    # signal quality
    'csnr_o2a', 'csnr_wco2', 'csnr_sco2',
    'h_cont_o2a', 'h_cont_wco2', 'h_cont_sco2',
    'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2',
    'snr_o2a', 'snr_wco2', 'snr_sco2',
    # misc
    's31', 's32', 't700',
]


def run_surface(df, sfc, ppca_path, target_col, cfg, max_rows, seed):
    surf = SURF[sfc]
    print(f"\n{'='*72}\n  SURFACE {sfc} ({surf})\n{'='*72}", flush=True)

    n0 = len(df)
    df = filter_target_outliers(df, target_col=target_col)
    if len(df) < n0:
        print(f"  dropped {n0-len(df):,} |{target_col}|>100 ppm outliers", flush=True)

    # Frozen baseline: new `full` = full features ⊕ profile-EOF/tropopause block
    pipe = FeaturePipeline.fit(df, sfc_type=sfc, feature_set='full')
    X_full = pipe.transform(df)
    ppca = ProfilePCA.load(ppca_path)
    X_prof = ppca.transform(df)
    base_names = set(pipe.features) | set(ppca.feature_names)
    y = df[target_col].to_numpy(np.float32)

    valid = np.isfinite(y) & np.isfinite(X_full).all(1) & np.isfinite(X_prof).all(1)
    X_base = np.concatenate([X_full[valid], X_prof[valid]], axis=1).astype(np.float32)
    y = y[valid]
    df_v = df.loc[valid].reset_index(drop=True)
    n = len(y)

    if max_rows and n > max_rows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, max_rows, replace=False)
        X_base, y, df_v = X_base[keep], y[keep], df_v.iloc[keep].reset_index(drop=True)
        print(f"  subsampled to {max_rows:,} of {n:,} rows", flush=True)

    itr, ite = train_test_split(np.arange(len(y)),
                                test_size=float(cfg['split']['test_size']), random_state=seed)
    device = 'cpu'
    from .compare_profile_features import _cuda_available
    if _cuda_available():
        device = 'cuda'

    candidates = [c for c in CANDIDATE_POOL if c in df_v.columns and c not in base_names]
    print(f"  baseline dims={X_base.shape[1]}  |  candidates={len(candidates)}  "
          f"|  train={len(itr):,} test={len(ite):,}  device={device}", flush=True)

    _, r2_base, mae_base = _train_xgb(X_base[itr], y[itr], X_base[ite], y[ite],
                                      cfg['xgb'], cfg['objective'], device)
    print(f"  BASELINE (full+profile):  R²={r2_base:.4f}  MAE={mae_base:.4f} ppm\n", flush=True)

    rows = []
    for i, c in enumerate(candidates, 1):
        col = df_v[c].to_numpy(np.float32)[:, None]
        # inf (e.g. divide-by-zero ratio features) → NaN so XGBoost treats it as
        # missing; XGBoost rejects raw inf ("missing is not set to inf").
        n_inf = int(np.isinf(col).sum())
        col = np.where(np.isfinite(col), col, np.nan).astype(np.float32)
        nan_frac = float(np.isnan(col).mean())
        X = np.concatenate([X_base, col], axis=1)
        _, r2, mae = _train_xgb(X[itr], y[itr], X[ite], y[ite],
                                cfg['xgb'], cfg['objective'], device)
        d_r2, d_mae = r2 - r2_base, mae - mae_base
        rows.append({'surface': surf, 'feature': c, 'nan_frac': nan_frac,
                     'r2': r2, 'mae': mae, 'delta_r2': d_r2, 'delta_mae': d_mae})
        flag = '  <== helps' if (d_r2 > 0.001 and d_mae < -0.0005) else ''
        inf_note = f' inf={n_inf}' if n_inf else ''
        print(f"  [{i:2d}/{len(candidates)}] {c:26s} ΔR²={d_r2:+.4f} ΔMAE={d_mae:+.4f} "
              f"nan={nan_frac:.2f}{inf_note}{flag}", flush=True)

    del X_base, X_full, X_prof, y
    gc.collect()
    res = pd.DataFrame(rows).sort_values('delta_r2', ascending=False)
    res.insert(1, 'r2_base', r2_base)
    res.insert(2, 'mae_base', mae_base)
    print(f"\n  --- {surf}: top improvements by ΔR² ---", flush=True)
    print(res.head(12).to_string(index=False), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=None)
    ap.add_argument('--ppca-dir', default=None)
    ap.add_argument('--sfc_type', type=int, default=None, choices=[0, 1])
    ap.add_argument('--target', default='10km')
    ap.add_argument('--max-rows', type=int, default=300000,
                    help='cap rows per surface for a fast, ranking-stable ablation')
    ap.add_argument('--n-estimators', type=int, default=2500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    storage = get_storage_dir()
    import platform
    _default = ('combined_2016_2020_dates.parquet' if platform.system() == 'Linux'
                else 'combined_2020_dates.parquet')
    data_path = Path(args.data) if args.data else storage / 'results/csv_collection' / _default
    ppca_dir  = Path(args.ppca_dir) if args.ppca_dir else storage / 'results/profile_pca'
    out_csv   = Path(args.out) if args.out else storage / 'results/model_xgb/unused_feature_ablation_2020.csv'
    target_col = resolve_target_col(args.target)

    from .compare_profile_features import _default_run_config  # noqa
    from .xgb import _default_run_config as _drc
    cfg = _drc()
    cfg['xgb']['n_estimators'] = int(args.n_estimators)

    import pyarrow.parquet as pq
    schema = pq.read_schema(str(data_path)).names
    cols = sorted(set(_needed_columns(schema, target_col)) |
                  {c for c in CANDIDATE_POOL if c in schema})
    print(f"Reading {len(cols)}/{len(schema)} columns from {Path(data_path).name}", flush=True)

    surfaces = [args.sfc_type] if args.sfc_type is not None else [0, 1]
    parts = []
    for sfc in surfaces:
        df = pd.read_parquet(str(data_path), columns=cols,
                             filters=[('sfc_type', '==', sfc), ('snow_flag', '==', 0)])
        parts.append(run_surface(df, sfc, ppca_dir / f'profile_pca_{SURF[sfc]}.pkl',
                                 target_col, cfg, args.max_rows, args.seed))
        del df
        gc.collect()

    summary = pd.concat(parts, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False, float_format='%.5f')
    print(f"\nSaved full ranked table → {out_csv}", flush=True)


if __name__ == '__main__':
    main()

"""test_snow_features.py — does the snow_flag feature help the LAND XCO2-anomaly model?

The production pipeline filters snow_flag==0, so snow_flag only *varies* when snow
footprints are kept.  This test therefore runs on SNOW-INCLUSIVE land data and
compares two XGBoost models that differ by exactly one column:

    A) full_contam  = base (incl. contamination features), NO snow_flag
    B) full         = full_contam + snow_flag   (land)

Same rows, same split, same hyperparameters.  Reports R² / RMSE / MAE both overall
and on the snow subset (snow_flag==1, ~4-5 % of land) where the flag can matter.

Run (from repo root):
    PYTHONPATH=src python -m models.test_snow_features \
      --data results/csv_collection/combined_2020_dates.parquet
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .pipeline import FeaturePipeline, filter_target_outliers, resolve_target_col
from .xgb import _default_run_config, _build_huber_l1_pred_obj, _cuda_available
from .compare_profile_features import _needed_columns

logger = logging.getLogger(__name__)


def _fit_predict(X_tr, y_tr, X_te, y_te, xgb_cfg, obj_cfg, device):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=int(xgb_cfg['n_estimators']), max_depth=int(xgb_cfg['max_depth']),
        learning_rate=float(xgb_cfg['learning_rate']), subsample=float(xgb_cfg['subsample']),
        colsample_bytree=float(xgb_cfg['colsample_bytree']),
        min_child_weight=float(xgb_cfg['min_child_weight']), gamma=float(xgb_cfg['gamma']),
        reg_lambda=float(xgb_cfg['reg_lambda']), reg_alpha=float(xgb_cfg['reg_alpha']),
        objective=_build_huber_l1_pred_obj(delta=float(obj_cfg['huber_delta']),
                                           lam=float(obj_cfg['pred_l1_weight'])),
        tree_method=xgb_cfg['tree_method'], device=device,
        early_stopping_rounds=int(xgb_cfg['early_stopping_rounds']),
        eval_metric=xgb_cfg['eval_metric'], n_jobs=int(xgb_cfg['n_jobs']), verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return model.predict(X_te)


def _metrics(y, yp):
    ss_res = float(((y - yp) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
    mae = float(np.mean(np.abs(y - yp)))
    return r2, rmse, mae


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=None)
    ap.add_argument('--target', default='10km')
    ap.add_argument('--n-estimators', type=int, default=2500)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    from utils import get_storage_dir
    storage = get_storage_dir()
    import platform
    _dn = ('combined_2016_2020_dates.parquet' if platform.system() == 'Linux'
           else 'combined_2020_dates.parquet')
    data_path = Path(args.data) if args.data else storage / 'results/csv_collection' / _dn
    target_col = resolve_target_col(args.target)
    cfg = _default_run_config()
    cfg['xgb']['n_estimators'] = int(args.n_estimators)

    import pyarrow.parquet as pq
    schema = pq.read_schema(str(data_path)).names
    cols = _needed_columns(schema, target_col)
    # LAND only, but do NOT filter snow_flag — we need it to vary.
    print(f"Reading {len(cols)} cols (land, snow-inclusive) from {Path(data_path).name}", flush=True)
    df = pd.read_parquet(str(data_path), columns=cols, filters=[('sfc_type', '==', 1)])
    n0 = len(df)
    df = filter_target_outliers(df, target_col=target_col)
    print(f"  land rows: {n0:,} → {len(df):,} after outlier filter", flush=True)
    print(f"  snow footprints (snow_flag==1): {int((df['snow_flag']==1).sum()):,} "
          f"({(df['snow_flag']==1).mean()*100:.1f}%)", flush=True)

    # Two pipelines differing ONLY by snow_flag (land).
    pipe_ns = FeaturePipeline.fit(df, sfc_type=1, feature_set='full_contam')  # no snow_flag
    pipe_sn = FeaturePipeline.fit(df, sfc_type=1, feature_set='full')          # + snow_flag
    assert 'snow_flag' in pipe_sn.features and 'snow_flag' not in pipe_ns.features
    print(f"  A full_contam: {pipe_ns.n_features} feats  |  B full: {pipe_sn.n_features} feats "
          f"(+{[f for f in pipe_sn.features if f not in pipe_ns.features]})", flush=True)

    X_ns = pipe_ns.transform(df)
    X_sn = pipe_sn.transform(df)
    y = df[target_col].to_numpy(np.float32)
    snow = (df['snow_flag'].to_numpy() == 1)

    valid = np.isfinite(y) & np.isfinite(X_ns).all(1) & np.isfinite(X_sn).all(1)
    X_ns, X_sn, y, snow = X_ns[valid], X_sn[valid], y[valid], snow[valid]
    print(f"  usable rows: {len(y):,}  (snow {int(snow.sum()):,})", flush=True)

    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=float(cfg['split']['test_size']),
                                random_state=args.seed)
    device = 'cuda' if _cuda_available() else 'cpu'
    print(f"  train={len(itr):,} test={len(ite):,} device={device}\n", flush=True)

    yp_ns = _fit_predict(X_ns[itr], y[itr], X_ns[ite], y[ite], cfg['xgb'], cfg['objective'], device)
    yp_sn = _fit_predict(X_sn[itr], y[itr], X_sn[ite], y[ite], cfg['xgb'], cfg['objective'], device)

    yte, snow_te = y[ite], snow[ite]
    rows = []
    for label, mask in [('ALL test', np.ones(len(yte), bool)),
                        (f'snow subset (n={int(snow_te.sum())})', snow_te),
                        (f'non-snow (n={int((~snow_te).sum())})', ~snow_te)]:
        if mask.sum() < 10:
            continue
        a = _metrics(yte[mask], yp_ns[mask]); b = _metrics(yte[mask], yp_sn[mask])
        rows.append({'subset': label,
                     'r2_nosnow': a[0], 'r2_snow': b[0], 'dR2': b[0]-a[0],
                     'rmse_nosnow': a[1], 'rmse_snow': b[1], 'dRMSE': b[1]-a[1],
                     'mae_nosnow': a[2], 'mae_snow': b[2], 'dMAE': b[2]-a[2]})
    res = pd.DataFrame(rows)
    print("=== snow_flag A/B (B=full with snow_flag, A=full_contam without) ===", flush=True)
    for _, r in res.iterrows():
        print(f"  {r['subset']:26s}  R²: {r.r2_nosnow:.4f}→{r.r2_snow:.4f} (Δ{r.dR2:+.4f}) | "
              f"RMSE: {r.rmse_nosnow:.4f}→{r.rmse_snow:.4f} (Δ{r.dRMSE:+.4f}) | "
              f"MAE Δ{r.dMAE:+.4f}", flush=True)

    out = storage / 'results/model_xgb/snow_flag_ab_land.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False, float_format='%.5f')
    print(f"\nSaved → {out}", flush=True)
    gc.collect()


if __name__ == '__main__':
    main()

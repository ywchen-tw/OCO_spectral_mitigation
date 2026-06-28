"""eval_snowflag_highlat.py — does adding snow_flag lift high-latitude DE accuracy?

Compares the held-out predictions of the 3-arm snow_flag test (land, 2020, fold 0):
  A nosnow   — full_contam,      snow excluded  (production filter, retrained on 2020)
  B snowdata — full_contam,      snow included  (added-data effect)
  C snowflag — full_contam_snow, snow included  (added-feature effect)

The DE predicts mu = XCO2_bc anomaly; residual = y_true - mu (ppm).  Reports RMSE /
MAE / R² / mean-bias overall and sliced by snow_flag and |lat| band.  The clean
attribution is B vs C on the snow / high-lat slices (identical holdout); A is the
status-quo anchor (its holdout has no snow rows).

Run: PYTHONPATH=src python workspace/eval_snowflag_highlat.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get('CURC_DATA_ROOT')
                 or os.environ.get('OCO2_DATAROOT') or ROOT)
BASE = DATA_ROOT / 'results/model_deep_ensemble'

ARMS = [
    ('A nosnow',   'de_land_fc_nosnow_f0'),
    ('B snowdata', 'de_land_fc_snowdata_f0'),
    ('C snowflag', 'de_land_fc_snowflag_f0'),
]


def _metrics(y, mu):
    """RMSE / MAE / R² / mean-bias of residual (y - mu), all ppm."""
    y = np.asarray(y, float); mu = np.asarray(mu, float)
    m = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[m], mu[m]
    n = y.size
    if n == 0:
        return dict(n=0, rmse=np.nan, mae=np.nan, r2=np.nan, bias=np.nan)
    r = y - mu
    sse = np.sum(r ** 2); sst = np.sum((y - y.mean()) ** 2)
    return dict(n=n,
                rmse=float(np.sqrt(np.mean(r ** 2))),
                mae=float(np.mean(np.abs(r))),
                r2=float(1 - sse / sst) if sst > 0 else np.nan,
                bias=float(np.mean(r)))


def _slices(df):
    lat = df['lat'].to_numpy(float) if 'lat' in df else np.full(len(df), np.nan)
    snow = df['snow_flag'].to_numpy(float) if 'snow_flag' in df else np.full(len(df), np.nan)
    al = np.abs(lat)
    return {
        'overall':      np.ones(len(df), bool),
        '|lat|<=60':    al <= 60,
        '|lat|>60':     al > 60,
        '|lat|>70':     al > 70,
        'snow==0':      snow == 0,
        'snow==1':      snow == 1,
        'snow==1 &':    (snow == 1) & (al > 70),   # snow AND high-lat
    }


def main():
    frames = {}
    for label, d in ARMS:
        p = BASE / d / 'held_out_predictions.parquet'
        if not p.exists():
            print(f"  [skip] {label}: {p} not found"); continue
        frames[label] = pd.read_parquet(p)

    if not frames:
        raise SystemExit("No held-out predictions found — run workspace/run_snowflag_test.sh first.")

    slice_names = ['overall', '|lat|<=60', '|lat|>60', '|lat|>70',
                   'snow==0', 'snow==1', 'snow==1 &']
    print(f"\n{'slice':12s} {'arm':11s} {'n':>7s} {'RMSE':>7s} {'MAE':>7s} {'R2':>7s} {'bias':>7s}")
    print('-' * 62)
    rows = []
    for sl in slice_names:
        for label in frames:
            df = frames[label]
            mask = _slices(df)[sl]
            mt = _metrics(df.loc[mask, 'y_true'], df.loc[mask, 'mu'])
            rows.append(dict(slice=sl, arm=label, **mt))
            print(f"{sl:12s} {label:11s} {mt['n']:>7d} {mt['rmse']:>7.3f} "
                  f"{mt['mae']:>7.3f} {mt['r2']:>7.3f} {mt['bias']:>7.3f}")
        print()

    out = BASE.parent / 'model_comparison' / 'snowflag_highlat_eval.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    # ── headline: B vs C on the slices that matter (identical holdout) ──────────
    print("=" * 62)
    print("FEATURE EFFECT  (C snowflag − B snowdata; negative RMSE = C better):")
    if 'B snowdata' in frames and 'C snowflag' in frames:
        B, C = frames['B snowdata'], frames['C snowflag']
        for sl in ['overall', '|lat|>70', 'snow==1', 'snow==1 &']:
            mb = _metrics(B.loc[_slices(B)[sl], 'y_true'], B.loc[_slices(B)[sl], 'mu'])
            mc = _metrics(C.loc[_slices(C)[sl], 'y_true'], C.loc[_slices(C)[sl], 'mu'])
            d_rmse = mc['rmse'] - mb['rmse']
            d_mae = mc['mae'] - mb['mae']
            print(f"  {sl:12s} n={mc['n']:>6d}  dRMSE={d_rmse:+.3f}  dMAE={d_mae:+.3f}  "
                  f"(B={mb['rmse']:.3f} → C={mc['rmse']:.3f})")
    print(f"\n  [saved] {out}")


if __name__ == '__main__':
    main()

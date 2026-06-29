"""eval_snowflag_highlat_tabm.py — TabM counterpart of eval_snowflag_highlat.py.

Does adding snow training DATA (B) and the snow_flag FEATURE (C) lift TabM accuracy
on the snow / high-latitude land footprints production discards?  Pools held-out
predictions across all date_kfold folds, slices by snow_flag and |lat| band.
B vs C share an identical holdout → clean feature attribution.  (lat/snow_flag for
B & C were recovered post-hoc; arm A has no snow rows.)

Run: PYTHONPATH=src python workspace/eval_snowflag_highlat_tabm.py
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'results/model_tabm'
ARMS = [
    ('A nosnow',   'tabm_land_full_contam_f*'),
    ('B snowdata', 'tabm_land_fc_snowdata_f*'),
    ('C snowflag', 'tabm_land_fc_snowflag_f*'),
]

def _metrics(y, mu):
    y = np.asarray(y, float); mu = np.asarray(mu, float)
    m = np.isfinite(y) & np.isfinite(mu); y, mu = y[m], mu[m]
    if y.size == 0:
        return dict(n=0, rmse=np.nan, mae=np.nan, r2=np.nan, bias=np.nan)
    r = y - mu; sse = np.sum(r**2); sst = np.sum((y - y.mean())**2)
    return dict(n=y.size, rmse=float(np.sqrt(np.mean(r**2))), mae=float(np.mean(np.abs(r))),
                r2=float(1 - sse/sst) if sst > 0 else np.nan, bias=float(np.mean(r)))

def _slices(df):
    lat = df['lat'].to_numpy(float) if 'lat' in df else np.full(len(df), np.nan)
    snow = df['snow_flag'].to_numpy(float) if 'snow_flag' in df else np.full(len(df), np.nan)
    al = np.abs(lat)
    return {'overall': np.ones(len(df), bool), '|lat|<=60': al <= 60, '|lat|>60': al > 60,
            '|lat|>70': al > 70, 'snow==0': snow == 0, 'snow==1': snow == 1,
            'snow==1 & |lat|>70': (snow == 1) & (al > 70)}

def main():
    frames = {}
    for label, pat in ARMS:
        parts = []
        for d in sorted(BASE.glob(pat)):
            p = d / 'held_out_predictions.parquet'
            if p.exists():
                fdf = pd.read_parquet(p); fdf['fold'] = d.name.rsplit('_f', 1)[-1]
                parts.append(fdf)
        if parts:
            frames[label] = pd.concat(parts, ignore_index=True)
            print(f"  {label:11s}: pooled {len(frames[label]):>8d} rows over "
                  f"{frames[label]['fold'].nunique()} fold(s)  "
                  f"(has snow_flag={'snow_flag' in frames[label].columns})")
    if not frames:
        raise SystemExit("No held-out predictions found.")

    slice_names = list(_slices(next(iter(frames.values()))).keys())
    print(f"\n{'slice':20s} {'arm':11s} {'n':>8s} {'RMSE':>7s} {'MAE':>7s} {'R2':>7s} {'bias':>7s}")
    print('-' * 70)
    rows = []
    for sl in slice_names:
        for label, df in frames.items():
            mt = _metrics(df.loc[_slices(df)[sl], 'y_true'], df.loc[_slices(df)[sl], 'mu'])
            rows.append(dict(slice=sl, arm=label, **mt))
            print(f"{sl:20s} {label:11s} {mt['n']:>8d} {mt['rmse']:>7.3f} "
                  f"{mt['mae']:>7.3f} {mt['r2']:>7.3f} {mt['bias']:>7.3f}")
        print()

    out = BASE.parent / 'model_comparison' / 'tabm_snowflag_highlat_eval.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print("=" * 70)
    print("FEATURE EFFECT  (C snowflag − B snowdata; negative dRMSE = C better):")
    if 'B snowdata' in frames and 'C snowflag' in frames:
        B, C = frames['B snowdata'], frames['C snowflag']
        for sl in ['overall', '|lat|>70', 'snow==1', 'snow==1 & |lat|>70']:
            mb = _metrics(B.loc[_slices(B)[sl], 'y_true'], B.loc[_slices(B)[sl], 'mu'])
            mc = _metrics(C.loc[_slices(C)[sl], 'y_true'], C.loc[_slices(C)[sl], 'mu'])
            print(f"  {sl:20s} n={mc['n']:>6d}  dRMSE={mc['rmse']-mb['rmse']:+.3f}  "
                  f"dMAE={mc['mae']-mb['mae']:+.3f}  (B={mb['rmse']:.3f} → C={mc['rmse']:.3f})")
    print(f"\n  [saved] {out}")

if __name__ == '__main__':
    main()

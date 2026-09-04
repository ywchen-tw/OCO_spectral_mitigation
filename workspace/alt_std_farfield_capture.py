#!/usr/bin/env python3
"""Far-cloud clear-sky terrain test, part 2 (2026-08-31).

Question 1 — with no clouds nearby (>= 30 km), does the xco2_bc anomaly
relate to sub-footprint elevation spread (alt_std)?
  * mean and mean-|anomaly| per alt_std decile;
  * Spearman r(alt_std, |anomaly|), overall and within albedo terciles
    (rough terrain is darker, r = -0.30, so albedo must be controlled);
  * anomaly std per alt_std quintile within albedo terciles.

Question 2 — if so, do the path-length statistics catch it?
  * per alt_std quintile: Spearman r(anomaly, feature) for each
    standardized feature departure (zk1 x 3 bands, zexp_o2a);
  * per alt_std quintile: OLS R^2 of anomaly ~ four features
    (+ variant adding alb_wco2). If the terrain-linked extra variance were
    photon-path physics, feature-anomaly coupling should strengthen with
    roughness; flat/declining R^2 means the features do not catch it.

Reuses load() from alt_std_terrain_analysis (land only, QF0 snow-free
applied here, r15 target and references, sigma-unit departures).
Outputs CSVs into the same alt_std_terrain directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alt_std_terrain_analysis import BANDS, FAR_KM, OUT_DIR, TARGET, load  # noqa: E402

ZCOLS = [f'zk1_{b}' for b in BANDS] + ['zexp_o2a']


def ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--qf', type=int, default=0, choices=(0, 1),
                    help='xco2_qf population (snow-free either way)')
    args = ap.parse_args()
    sfx = '' if args.qf == 0 else f'_qf{args.qf}'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    far = df[(df.xco2_qf == args.qf) & (df.snow_flag == 0)
             & (df.cld_dist_km >= FAR_KM)
             & np.isfinite(df.alt_std) & np.isfinite(df[TARGET])
             & np.isfinite(df.alb_wco2)].copy()
    for c in ZCOLS:
        far = far[np.isfinite(far[c])]
    far['absanom'] = far[TARGET].abs()
    print(f'far-field complete-case rows: {len(far):,}')

    # --- Q1a: anomaly mean / |anomaly| per alt_std decile -----------------
    dec = pd.qcut(far.alt_std, 10, duplicates='drop')
    rows = []
    for iv, g in far.groupby(dec, observed=True):
        rows.append({'alt_std_bin_m': f'{iv.left:.3g}..{iv.right:.3g}',
                     'n': len(g),
                     'anom_mean': float(g[TARGET].mean()),
                     'anom_median': float(g[TARGET].median()),
                     'absanom_mean': float(g.absanom.mean()),
                     'anom_std': float(g[TARGET].std()),
                     'alb_wco2_mean': float(g.alb_wco2.mean())})
    d1 = pd.DataFrame(rows)
    d1.to_csv(OUT_DIR / f'farfield_anom_by_altstd_decile{sfx}.csv',
              index=False)
    print(d1.to_string(index=False))

    # --- Q1b: |anomaly| ~ alt_std, overall and albedo-controlled ----------
    r_all, p_all = sps.spearmanr(far.alt_std, far.absanom)
    print(f'far-field spearman alt_std~|anom| (all): r={r_all:+.3f} '
          f'p={p_all:.2g} n={len(far):,}')
    terc = pd.qcut(far.alb_wco2, 3, labels=['dark', 'mid', 'bright'])
    q5 = pd.qcut(far.alt_std, 5, duplicates='drop')
    rows = []
    for t in ['dark', 'mid', 'bright']:
        sub = far[terc == t]
        r, p = sps.spearmanr(sub.alt_std, sub.absanom)
        print(f'  within alb {t}: r={r:+.3f} p={p:.2g} n={len(sub):,}')
        for iv, g in sub.groupby(q5[terc == t], observed=True):
            rows.append({'alb_tercile': t,
                         'alt_std_bin_m': f'{iv.left:.3g}..{iv.right:.3g}',
                         'n': len(g),
                         'anom_mean': float(g[TARGET].mean()),
                         'absanom_mean': float(g.absanom.mean()),
                         'anom_std': float(g[TARGET].std())})
    d2 = pd.DataFrame(rows)
    d2.to_csv(OUT_DIR / f'farfield_anom_by_altstd_x_alb{sfx}.csv',
              index=False)
    print(d2.to_string(index=False))

    # --- Q2: do the features catch the terrain-linked anomaly? ------------
    rows = []
    for iv, g in far.groupby(q5, observed=True):
        row = {'alt_std_bin_m': f'{iv.left:.3g}..{iv.right:.3g}',
               'n': len(g), 'anom_std': float(g[TARGET].std())}
        for c in ZCOLS:
            r, _ = sps.spearmanr(g[c], g[TARGET])
            row[f'r_anom_{c}'] = float(r)
        y = g[TARGET].to_numpy()
        row['r2_features'] = ols_r2(g[ZCOLS].to_numpy(), y)
        row['r2_features_alb'] = ols_r2(
            g[ZCOLS + ['alb_wco2']].to_numpy(), y)
        row['r2_alb_only'] = ols_r2(g[['alb_wco2']].to_numpy(), y)
        rows.append(row)
    d3 = pd.DataFrame(rows)
    d3.to_csv(OUT_DIR / f'farfield_feature_capture{sfx}.csv', index=False)
    print(d3.to_string(index=False))
    print('written to', OUT_DIR)


if __name__ == '__main__':
    main()

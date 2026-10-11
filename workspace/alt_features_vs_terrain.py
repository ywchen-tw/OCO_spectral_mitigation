#!/usr/bin/env python3
"""Far-cloud footprints: path-length statistics vs terrain (2026-08-31).

For land QF0 snow-free footprints >= 30 km from cloud, relate the fitted
photon path-length statistics -- band k1 (mean relative path), band k2
(path variance), and the O2A continuum exp-intercept -- to three terrain
descriptors:
  * alt          surface elevation (m)
  * alt_std      sub-footprint elevation spread (m)
  * rel_alt_std  alt_std / alt (footprints with alt > 0 only)

Each statistic is tested in two forms:
  * raw value -- sensitive to any static terrain response, but also to
    climatology (mountains sit at particular latitudes/geometries);
  * z departure from the same-orbit r15 clear-sky reference -- local
    confounds cancel, but so does any static terrain offset shared with
    the reference. The truth is bracketed between the two readings.

Outputs (CSV) in results/figures/cld_dist_analysis/alt_std_terrain/:
  farfield_features_vs_terrain_corr.csv   Spearman r per stat x terrain
  farfield_features_by_terrain_quintile.csv  stat means per terrain quintile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

_REPO = Path(__file__).resolve().parents[1]

PARQUET = _REPO / 'results/csv_collection/combined_2016_2020_dates.parquet'
OUT_DIR = _REPO / 'results/figures/cld_dist_analysis/alt_std_terrain'

BANDS = ['o2a', 'wco2', 'sco2']
FAR_KM = 30.0
RAW_STATS = ([f'{b}_k1' for b in BANDS] + [f'{b}_k2' for b in BANDS]
             + ['exp_o2a_intercept'])
Z_STATS = ([f'zk1_{b}' for b in BANDS] + [f'zk2_{b}' for b in BANDS]
           + ['zexp_o2a'])
TERRAIN = ['alt', 'alt_std', 'rel_alt_std']


def load(qf: int = 0) -> pd.DataFrame:
    cols = (['sfc_type', 'cld_dist_km', 'xco2_qf', 'snow_flag',
             'alt', 'alt_std', 'alb_wco2',
             'r15_exp_int_o2a_mean', 'r15_exp_int_o2a_std'] + RAW_STATS +
            [f'r15_{b}_k{m}_{s}' for b in BANDS for m in (1, 2)
             for s in ('mean', 'std')])
    df = pd.read_parquet(PARQUET, columns=cols)
    df = df[(df.sfc_type == 1) & (df.xco2_qf == qf) & (df.snow_flag == 0)
            & (df.cld_dist_km >= FAR_KM)].copy()
    for b in BANDS:
        for m in (1, 2):
            df[f'zk{m}_{b}'] = ((df[f'{b}_k{m}'] - df[f'r15_{b}_k{m}_mean'])
                                / df[f'r15_{b}_k{m}_std'].replace(0, np.nan))
    df['zexp_o2a'] = ((df.exp_o2a_intercept - df.r15_exp_int_o2a_mean)
                      / df.r15_exp_int_o2a_std.replace(0, np.nan))
    df['rel_alt_std'] = np.where(df.alt > 0, df.alt_std / df.alt, np.nan)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--qf', type=int, default=0, choices=(0, 1),
                    help='xco2_qf population (snow-free either way)')
    args = ap.parse_args()
    sfx = '' if args.qf == 0 else f'_qf{args.qf}'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load(qf=args.qf)
    stats = RAW_STATS + Z_STATS
    df = df[np.isfinite(df[stats]).all(axis=1)
            & np.isfinite(df.alt) & np.isfinite(df.alt_std)]
    n_neg = int((~np.isfinite(df.rel_alt_std)).sum())
    print(f'far-field complete-case rows: {len(df):,} '
          f'(rel_alt_std undefined for {n_neg:,} rows with alt <= 0)')

    # --- Spearman correlation matrix --------------------------------------
    corr_rows = []
    for t in TERRAIN:
        tv = df[t]
        ok = np.isfinite(tv)
        for c in stats:
            r, p = sps.spearmanr(tv[ok], df[c][ok])
            corr_rows.append({'terrain': t, 'stat': c,
                              'spearman_r': float(r), 'p': float(p),
                              'n': int(ok.sum())})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT_DIR / f'farfield_features_vs_terrain_corr{sfx}.csv',
                index=False)
    wide = corr.pivot(index='stat', columns='terrain',
                      values='spearman_r').loc[stats]
    print(wide.round(3).to_string())

    # --- quintile means ----------------------------------------------------
    q_rows = []
    for t in TERRAIN:
        tv = df[t]
        ok = np.isfinite(tv)
        q = pd.qcut(tv[ok], 5, duplicates='drop')
        g = df[ok].groupby(q, observed=True)
        for iv, sub in g:
            row = {'terrain': t,
                   'bin': f'{iv.left:.3g}..{iv.right:.3g}',
                   'n': len(sub),
                   'alb_wco2_mean': float(sub.alb_wco2.mean())}
            for c in stats:
                row[f'{c}_mean'] = float(sub[c].mean())
            q_rows.append(row)
    qt = pd.DataFrame(q_rows)
    qt.to_csv(OUT_DIR / f'farfield_features_by_terrain_quintile{sfx}.csv',
              index=False)
    show = (['bin', 'n'] + [f'{c}_mean' for c in Z_STATS])
    for t in TERRAIN:
        print(f'\n--- {t} quintiles (z-departure means) ---')
        print(qt[qt.terrain == t][show].round(3).to_string(index=False))
    print('written to', OUT_DIR)


if __name__ == '__main__':
    main()

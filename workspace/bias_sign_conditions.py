#!/usr/bin/env python3
"""Which conditions make the near-cloud XCO2 anomaly positive vs negative?

Follow-up to the 2026-08-01 mean/median decomposition (review item 10.6):
condition-resolved statistics of the screened production anomaly targets in
the near-cloud strong-response zone.  For every candidate driver we bin the
near-cloud population into quantiles (or categories) and report n, mean,
median, and the fraction positive; plus Spearman rank correlations and two
mechanism-testing two-way splits (surface brightness x shadow/brightening
state).

Populations: per surface, cloud distance <= 5 km (the strong-response zone
of both surfaces; the land <= 15 km variant is included in the CSV), target
screen |y| <= 100 ppm (training population).  Reported for all-flag, snow-free
(all flags), QF0 snow-free, QF1 snow-free, and QF1 separately, since flag
state is itself a dominant condition.  The two-way table is written for the
three snow-free populations (column `population`).

Output: results/figures/cld_dist_analysis/bias_sign_conditions/
  bias_sign_conditions.csv   binned stats, all (surface, population, condition)
  bias_sign_corr.csv         Spearman correlations
  bias_sign_twoway.csv       albedo-tercile x shadow/brightening cells
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / 'src'))
from models.pipeline import MAX_ABS_ANOMALY_PPM  # noqa: E402

PARQUET = _REPO / 'results/csv_collection/combined_2016_2020_dates.parquet'
OUT_DIR = _REPO / 'results/figures/cld_dist_analysis/bias_sign_conditions'

SURFACES = {'ocean': dict(sfc=0, target='xco2_bc_anomaly_r05', ref='r05'),
            'land': dict(sfc=1, target='xco2_bc_anomaly_r15', ref='r15')}

# numeric candidate conditions binned into quintiles
NUMERIC = ['alb_o2a', 'alb_wco2', 'alb_sco2', 'sza', 'aod_total', 'ws',
           'tcwv', 'fp_area_km2', 'dexp_o2a', 'dalb_wco2', 'alt_std']


def load():
    cols = (['sfc_type', 'cld_dist_km', 'xco2_qf', 'snow_flag',
             'exp_o2a_intercept', 'alb_o2a', 'alb_wco2', 'alb_sco2',
             'sza', 'aod_total', 'ws', 'tcwv', 'fp_area_km2', 'alt_std'] +
            [s['target'] for s in SURFACES.values()] +
            [f'{r}_exp_int_o2a_mean' for r in ('r05', 'r15')] +
            [f'{r}_exp_int_o2a_std' for r in ('r05', 'r15')] +
            [f'{r}_alb_wco2_mean' for r in ('r05', 'r15')])
    df = pd.read_parquet(PARQUET, columns=sorted(set(cols)))
    for s in SURFACES.values():
        y = df[s['target']]
        df.loc[y.abs() > MAX_ABS_ANOMALY_PPM, s['target']] = np.nan
    return df


def stats_block(y):
    return dict(n=len(y), mean=float(y.mean()), median=float(y.median()),
                frac_pos=float((y > 0).mean()))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    rows, corr_rows, tw_rows = [], [], []
    for name, spec in SURFACES.items():
        d = df[df.sfc_type == spec['sfc']].copy()
        # per-surface reference departures (shadow/brightening + albedo delta)
        d['dexp_o2a'] = d.exp_o2a_intercept - d[f"{spec['ref']}_exp_int_o2a_mean"]
        # standardized departure, the Fig. 5 classification variable:
        # brightened z > +0.5 / neutral |z| <= 0.5 / shadowed z < -0.5
        # (spec_sensitivity.run_shadow_brightening, z_thresh = 0.5)
        d['zexp_o2a'] = d['dexp_o2a'] / d[f"{spec['ref']}_exp_int_o2a_std"].replace(0, np.nan)
        d['dalb_wco2'] = d.alb_wco2 - d[f"{spec['ref']}_alb_wco2_mean"]
        y_all = d[spec['target']]
        for win, dmax in [('near5', 5.0)] + ([('near15', 15.0)] if name == 'land' else []):
            base = np.isfinite(y_all) & d.cld_dist_km.between(0, dmax)
            for pop, pm in [('allqf', base),
                            ('snowfree', base & (d.snow_flag == 0)),
                            ('qf0snowfree', base & (d.xco2_qf == 0) & (d.snow_flag == 0)),
                            ('qf1snowfree', base & (d.xco2_qf == 1) & (d.snow_flag == 0)),
                            ('qf1', base & (d.xco2_qf == 1))]:
                sub, y = d[pm], y_all[pm]
                # categorical conditions
                for cond, cm in [('snow', sub.snow_flag > 0),
                                 ('qf1', sub.xco2_qf == 1),
                                 ('shadow(dexp<0)', sub.dexp_o2a < 0)]:
                    for lab, mm in [('yes', cm), ('no', ~cm)]:
                        if mm.sum() > 200:
                            rows.append({'surface': name, 'window': win,
                                         'population': pop, 'condition': cond,
                                         'bin': lab, **stats_block(y[mm])})
                # three-class Fig.-5 branch (z threshold +-0.5, neutral kept)
                for lab, mm in [('shadowed', sub.zexp_o2a < -0.5),
                                ('neutral', sub.zexp_o2a.abs() <= 0.5),
                                ('brightened', sub.zexp_o2a > 0.5)]:
                    if mm.sum() > 200:
                        rows.append({'surface': name, 'window': win,
                                     'population': pop, 'condition': 'branch3',
                                     'bin': lab, **stats_block(y[mm])})
                # numeric conditions in quintiles
                for cond in NUMERIC:
                    if cond == 'ws' and name == 'land':
                        continue
                    if cond == 'alt_std' and name == 'ocean':
                        continue
                    v = sub[cond]
                    ok = np.isfinite(v)
                    if ok.sum() < 2000:
                        continue
                    q = pd.qcut(v[ok], 5, duplicates='drop')
                    for iv, mm in y[ok].groupby(q, observed=True):
                        rows.append({'surface': name, 'window': win,
                                     'population': pop, 'condition': cond,
                                     'bin': f'{iv.left:.3g}..{iv.right:.3g}',
                                     **stats_block(mm)})
                    r, p = sps.spearmanr(v[ok], y[ok])
                    corr_rows.append({'surface': name, 'window': win,
                                      'population': pop, 'condition': cond,
                                      'spearman_r': float(r), 'p': float(p),
                                      'n': int(ok.sum())})
            # two-way: brightness terciles x three-class branch, for the
            # snow-free populations (all flags, QF0 only, QF1 only); terciles
            # are recomputed within each population
            for pop, pm in [('snowfree', base & (d.snow_flag == 0)),
                            ('qf0snowfree', base & (d.xco2_qf == 0) & (d.snow_flag == 0)),
                            ('qf1snowfree', base & (d.xco2_qf == 1) & (d.snow_flag == 0))]:
                sub, y = d[pm], y_all[pm]
                ok = np.isfinite(sub.alb_wco2) & np.isfinite(sub.zexp_o2a)
                terc = pd.qcut(sub.alb_wco2[ok], 3, labels=['dark', 'mid', 'bright'])
                edges = np.quantile(sub.alb_wco2[ok], [0, 1/3, 2/3, 1])
                for t in ['dark', 'mid', 'bright']:
                    for blab, bm in [('shadowed', sub.zexp_o2a[ok] < -0.5),
                                     ('neutral', sub.zexp_o2a[ok].abs() <= 0.5),
                                     ('brightened', sub.zexp_o2a[ok] > 0.5)]:
                        mm = (terc == t) & bm
                        if mm.sum() > 200:
                            tw_rows.append({'surface': name, 'window': win,
                                            'population': pop,
                                            'alb_wco2': t, 'branch': blab,
                                            'alb_wco2_lo': float(edges[['dark', 'mid', 'bright'].index(t)]),
                                            'alb_wco2_hi': float(edges[['dark', 'mid', 'bright'].index(t) + 1]),
                                            **stats_block(y[ok][mm])})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'bias_sign_conditions.csv', index=False)
    pd.DataFrame(corr_rows).to_csv(OUT_DIR / 'bias_sign_corr.csv', index=False)
    pd.DataFrame(tw_rows).to_csv(OUT_DIR / 'bias_sign_twoway.csv', index=False)
    print('written to', OUT_DIR)


if __name__ == '__main__':
    main()

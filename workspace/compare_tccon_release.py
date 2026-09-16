#!/usr/bin/env python
"""Compare a TCCON-release test arm against the baseline TCCON comparison CSVs.

Joins the baseline ``tccon_comparison<sfx>.csv`` with one or more arm CSVs on
(site, date, surface, qf_group) and reports, per station-day, the change in the
TCCON side of the comparison (n_tccon, tccon_mu, tccon_mu_raw, ak_delta) and in
the resulting OCO-2 metrics (bias_before, bias_after, rmse_before, rmse_after).
Written for the GGG2020 -> GGG2020.1 switch test (log/JQSRT_plan/
ggg2020p1_test_design_2026-09-15.md), where every arm re-runs the same OCO-2
plot data against a different TCCON release / XCO2 scale, so all differences
live on the TCCON side.

HEADLINE ROW FILTER
-------------------
The manuscript headline for the A-Train tree is reproduced from
``tccon_comparison<sfx>.csv`` by the single filter

    surface == 'all'  AND  qf_group == 'all'  AND  n_tccon > 0

which selects 75 of the 494 baseline A-Train rows (one per station-day, pooled
over surface and over quality flags).  On the baseline
(results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_foldpca_o05l15_m5/
atrain/tccon_comparison_r100km.csv) that filter gives, to the manuscript's
precision:

    station-day mean |bias_before| -> |bias_after|   1.2556 -> 0.8158 ppm  (1.26 -> 0.82)
    station-day mean rmse_before   -> rmse_after     2.6676 -> 1.1958 ppm  (2.67 -> 1.20)
    fp-RMSE improved (rmse_after < rmse_before)      71 of 75
    residual improved (|bias_after| < |bias_before|) 46 of 75
    all-footprint RMSE, n_oco-weighted RMS of the
    per-station-day rmse_before/rmse_after          3.2906 -> 1.2162 ppm  (3.29 -> 1.22)

The all-footprint numbers equal the (ref='ak', qf_group='all', surface='all',
cld_group='all') row of ``tccon_metrics_ak<sfx>.csv`` (rmse_before / rmse_after,
n_footprints = 105683), so the two files agree and the filter above is the one
the manuscript uses.  The same filter is applied unchanged to every arm and to
the drift tree.

Usage
-----
    python workspace/compare_tccon_release.py \
        --baseline <TAG>/atrain --arm arm1=<TAG>/ggg2020p1_test/arm1/atrain \
        --arm arm2=... --arm arm3=... \
        --suffix _r100km --label atrain --out-dir <TAG>/ggg2020p1_test/comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['site', 'date', 'surface', 'qf_group']
DELTA_COLS = ['n_tccon', 'tccon_mu', 'tccon_mu_raw', 'ak_delta',
              'bias_before', 'bias_after', 'rmse_before', 'rmse_after']
FLAG_PPM = 0.10          # |delta bias_after| above which a station-day is flagged


def headline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The manuscript's station-day set: pooled surface, pooled QF, TCCON present."""
    return df[(df['surface'] == 'all') & (df['qf_group'] == 'all')
              & (df['n_tccon'] > 0)].copy()


def headline(df: pd.DataFrame) -> dict:
    g = headline_rows(df)
    bb, ba = g['bias_before'].to_numpy(float), g['bias_after'].to_numpy(float)
    rb, ra = g['rmse_before'].to_numpy(float), g['rmse_after'].to_numpy(float)
    w = g['n_oco'].to_numpy(float)
    return dict(
        n_station_days=len(g),
        n_footprints=int(np.nansum(w)),
        sd_mean_abs_bias_before=float(np.nanmean(np.abs(bb))),
        sd_mean_abs_bias_after=float(np.nanmean(np.abs(ba))),
        sd_mean_fp_rmse_before=float(np.nanmean(rb)),
        sd_mean_fp_rmse_after=float(np.nanmean(ra)),
        n_rmse_improved=int(np.nansum(ra < rb)),
        n_resid_improved=int(np.nansum(np.abs(ba) < np.abs(bb))),
        allfp_rmse_before=float(np.sqrt(np.nansum(w * rb ** 2) / np.nansum(w))),
        allfp_rmse_after=float(np.sqrt(np.nansum(w * ra ** 2) / np.nansum(w))),
    )


def deltas(base: pd.DataFrame, arm: pd.DataFrame) -> pd.DataFrame:
    """Per-(site, date, surface, qf_group) arm-minus-baseline deltas."""
    cols = KEYS + [c for c in DELTA_COLS if c in base.columns] + ['n_oco']
    m = base[cols].merge(arm[cols], on=KEYS, how='outer',
                         suffixes=('_base', '_arm'), indicator=True)
    for c in DELTA_COLS:
        if f'{c}_base' in m.columns:
            m[f'd_{c}'] = m[f'{c}_arm'].astype(float) - m[f'{c}_base'].astype(float)
    return m


def _fmt(x, n=4):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{n}f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline', required=True,
                    help='Directory holding the baseline tccon_comparison<sfx>.csv.')
    ap.add_argument('--arm', action='append', default=[], metavar='NAME=DIR',
                    help='Arm to compare; repeatable.  NAME labels the arm in the '
                         'headline table, DIR holds its tccon_comparison<sfx>.csv.')
    ap.add_argument('--suffix', default='_r100km',
                    help="Filename suffix of the report CSVs (default '_r100km').")
    ap.add_argument('--label', default='atrain',
                    help='Tree label used in the output filenames and headline table.')
    ap.add_argument('--flag-ppm', type=float, default=FLAG_PPM,
                    help='Flag station-days whose |delta bias_after| exceeds this (ppm).')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    sfx = args.suffix
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(Path(args.baseline) / f'tccon_comparison{sfx}.csv')

    rows = [dict(arm='baseline', tree=args.label, **headline(base))]
    md = [f'# TCCON release comparison — {args.label}', '',
          f'Baseline: `{args.baseline}`.  Row filter: '
          "`surface == 'all' and qf_group == 'all' and n_tccon > 0`.", '']
    flag_md = []
    shift_rows = []

    for spec in args.arm:
        name, _, d = spec.partition('=')
        if not d:
            raise SystemExit(f'--arm expects NAME=DIR, got {spec!r}')
        arm = pd.read_csv(Path(d) / f'tccon_comparison{sfx}.csv')
        rows.append(dict(arm=name, tree=args.label, **headline(arm)))

        dl = deltas(base, arm)
        dl.to_csv(out / f'release_deltas_{args.label}_{name}{sfx}.csv', index=False)

        hd = dl[(dl['surface'] == 'all') & (dl['qf_group'] == 'all')]
        shift_rows.append(dict(
            arm=name, tree=args.label, n_station_days=len(hd),
            d_tccon_mu_raw_mean=float(np.nanmean(hd['d_tccon_mu_raw'])),
            d_tccon_mu_raw_sd=float(np.nanstd(hd['d_tccon_mu_raw'], ddof=1)),
            d_tccon_mu_mean=float(np.nanmean(hd['d_tccon_mu'])),
            d_tccon_mu_sd=float(np.nanstd(hd['d_tccon_mu'], ddof=1)),
            d_ak_delta_mean=float(np.nanmean(hd['d_ak_delta'])),
            d_ak_delta_sd=float(np.nanstd(hd['d_ak_delta'], ddof=1)),
            d_n_tccon_mean=float(np.nanmean(hd['d_n_tccon'])),
            n_zero_tccon_arm=int((hd['n_tccon_arm'].fillna(0) <= 0).sum()),
        ))

        fl = hd[hd['d_bias_after'].abs() > args.flag_ppm].sort_values(
            'd_bias_after', key=lambda s: s.abs(), ascending=False)
        if len(fl):
            flag_md += ['', f'### {name} — station-days with |Δ bias_after| > '
                            f'{args.flag_ppm:g} ppm ({len(fl)} of {len(hd)})', '',
                        '| site | date | Δ n_tccon | Δ tccon_mu | Δ tccon_mu_raw | '
                        'Δ ak_delta | Δ bias_after | Δ rmse_after |',
                        '|---|---|--:|--:|--:|--:|--:|--:|']
            for _, r in fl.iterrows():
                flag_md.append(
                    f"| {r['site']} | {r['date']} | {int(r['d_n_tccon'])} | "
                    f"{_fmt(r['d_tccon_mu'], 3)} | {_fmt(r['d_tccon_mu_raw'], 3)} | "
                    f"{_fmt(r['d_ak_delta'], 3)} | {_fmt(r['d_bias_after'], 3)} | "
                    f"{_fmt(r['d_rmse_after'], 3)} |")
        else:
            flag_md += ['', f'### {name} — no station-day with |Δ bias_after| > '
                            f'{args.flag_ppm:g} ppm (0 of {len(hd)})']

    head = pd.DataFrame(rows)
    head.to_csv(out / f'release_headline_{args.label}{sfx}.csv', index=False)
    shifts = pd.DataFrame(shift_rows)
    if len(shifts):
        shifts.to_csv(out / f'release_shifts_{args.label}{sfx}.csv', index=False)

    md += ['## Headline', '',
           '| arm | station-days | footprints | mean \\|b\\| before | mean \\|b\\| after | '
           'mean fp-RMSE before | mean fp-RMSE after | fp-RMSE improved | residual improved | '
           'all-fp RMSE before | all-fp RMSE after |',
           '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for _, r in head.iterrows():
        md.append(
            f"| {r['arm']} | {r['n_station_days']} | {r['n_footprints']} | "
            f"{r['sd_mean_abs_bias_before']:.2f} | {r['sd_mean_abs_bias_after']:.2f} | "
            f"{r['sd_mean_fp_rmse_before']:.2f} | {r['sd_mean_fp_rmse_after']:.2f} | "
            f"{r['n_rmse_improved']}/{r['n_station_days']} | "
            f"{r['n_resid_improved']}/{r['n_station_days']} | "
            f"{r['allfp_rmse_before']:.2f} | {r['allfp_rmse_after']:.2f} |")

    if len(shifts):
        md += ['', '## TCCON-side shifts (station-day mean ± sd, arm − baseline)', '',
               '| arm | Δ tccon_mu_raw | Δ tccon_mu (AK) | Δ ak_delta | Δ n_tccon (mean) | '
               'station-days with n_tccon = 0 |', '|---|---|---|---|--:|--:|']
        for _, r in shifts.iterrows():
            md.append(
                f"| {r['arm']} | {r['d_tccon_mu_raw_mean']:+.4f} ± {r['d_tccon_mu_raw_sd']:.4f} | "
                f"{r['d_tccon_mu_mean']:+.4f} ± {r['d_tccon_mu_sd']:.4f} | "
                f"{r['d_ak_delta_mean']:+.4f} ± {r['d_ak_delta_sd']:.4f} | "
                f"{r['d_n_tccon_mean']:+.2f} | {int(r['n_zero_tccon_arm'])} |")

    md += ['', '## Flagged station-days'] + flag_md
    p_md = out / f'release_comparison_{args.label}{sfx}.md'
    p_md.write_text('\n'.join(md) + '\n')
    print(head.to_string(index=False))
    if len(shifts):
        print()
        print(shifts.to_string(index=False))
    print(f'\n[saved] {p_md}')


if __name__ == '__main__':
    main()

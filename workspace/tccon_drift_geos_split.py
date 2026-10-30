"""Post-A-Train (drift) TCCON statistics split by GEOS input to the L2 retrieval.

Reuses the report's own aggregate and significance helpers on the per-case
rows of <tree>/drift/tccon_comparison_r100km.csv (surface == 'all').
Usage: tccon_drift_geos_split.py <per-case csv> <out.md>
"""
import sys

import numpy as np
import pandas as pd

from tccon_comparison_report import _mse_aggregate, _significance

cases = pd.read_csv(sys.argv[1])
cases['date'] = cases['date'].astype(str).str[:10]
cases = cases[cases.surface == 'all']
d = pd.to_datetime(cases.date)
# L2 Full Physics meteorology switched GEOS-FPIT -> GEOS-IT on 2024-04-01
# (OCO-2/OCO-3 L2 User's Guide v4 revB, Sect. 3.2).
cases['l2_met'] = np.where(d >= '2024-04-01', 'GEOS-IT', 'GEOS-FPIT')
cases['period'] = np.select(
    [d < '2024-01-01', d < '2024-04-01'],
    ['2022-01..2023-12 (v11.2r, FPIT L2)', '2024-01..2024-03 (v11.3r, FPIT L2)'],
    '2024-04..2025-12 (v11.3r, GEOS-IT L2)')

SPLITS = [('all drift', None), ('L2 meteorology', 'l2_met'), ('three-way', 'period')]
QF = [('all', 'all QF'), ('qf0', 'QF=0'), ('qf1', 'QF=1')]


def row(g, label):
    a = _mse_aggregate(g)
    s = _significance(g, n_boot=10000) if len(g) >= 6 else {}
    f = lambda k: s.get(k, np.nan)
    return (f"| {label} | {len(g)} | {g.site.nunique()} | "
            f"{g.bias_raw.mean():+.2f} | {g.bias_before.mean():+.2f} | {g.bias_after.mean():+.2f} | "
            f"{a['mean_absbias_raw']:.2f} | {a['mean_absbias_before']:.2f} | **{a['mean_absbias_after']:.2f}** | "
            f"{a['mean_rmse_before']:.2f} | **{a['mean_rmse_after']:.2f}** | "
            f"{(g.rmse_after < g.rmse_before).sum()}/{len(g)} | "
            f"{f('d_mean_absbias_mean'):+.2f} [{f('d_mean_absbias_ci_lo'):+.2f}, {f('d_mean_absbias_ci_hi'):+.2f}] | "
            f"{f('d_mean_absbias_p'):.3g} | {f('wilcoxon_absbias_p'):.3g} | {f('wilcoxon_rmse_p'):.3g} |")


HDR = ['| group | n | sites | signed raw | signed B11 | signed DE | \\|res\\| raw | \\|res\\| B11 | \\|res\\| DE '
       '| fp-RMSE B11 | fp-RMSE DE | RMSE↓ | Δ\\|res\\| [95% CI] | p_boot | Wilcoxon \\|res\\| | Wilcoxon RMSE |',
       '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
lines = ['# Drift TCCON statistics by GEOS input', '',
         f'Source: `{sys.argv[1]}` (surface = all). Residual = OCO-2 − AK-harmonized TCCON (ppm).',
         'Δ|res| = DE − B11, site-clustered bootstrap (10 000).',
         'L2 meteorology: GEOS-FPIT before 2024-04-01, GEOS-IT from 2024-04-01 on.', '']
for qf, qtitle in QF:
    g0 = cases[cases.qf_group == qf]
    lines += [f'## {qtitle}', ''] + HDR
    for title, col in SPLITS:
        if col is None:
            lines.append(row(g0, title))
            continue
        for key, g in g0.groupby(col):
            lines.append(row(g, f'{title}: {key}'))
    lines.append('')
    if qf == 'all':
        lines += ['### Per site, all QF (L2 meteorology)', '',
                  '| site | L2 met | n | \\|res\\| B11 | \\|res\\| DE | fp-RMSE B11 | fp-RMSE DE |',
                  '|---|---|--:|--:|--:|--:|--:|']
        for (s, v), g in g0.groupby(['site', 'l2_met']):
            lines.append(f'| {s} | {v} | {len(g)} | {g.bias_before.abs().mean():.2f} | '
                         f'{g.bias_after.abs().mean():.2f} | {g.rmse_before.mean():.2f} | '
                         f'{g.rmse_after.mean():.2f} |')
        lines.append('')
open(sys.argv[2], 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

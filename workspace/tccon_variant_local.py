"""tccon_variant_local.py — TCCON metrics of ablation variant trees without the CURC report.

Reconstructs the AK-harmonized TCCON comparison of tccon_comparison_report.py for any
de_prof_mix_<variant> tree that holds only plot_data.parquet files.  The production
report CSV fixes the collocation (run_case box + <=100 km of the station + the
|xco2_bc - TCCON| <= 50 ppm sanity filter) and the AK reference per case
(corr_mu - bias_after).  The script first checks that it reproduces production n_oco
and corr_mu for every case, then evaluates each variant on the same footprints.
QF slices are not possible here (plot_data has no QF column).

Usage:  PYTHONPATH=src python workspace/tccon_variant_local.py keep_spec_geom keep_geom ...
Output: results/model_comparison/deep_ensemble/keep_only_tccon_local{.md,_per_case.csv}
"""
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DE = REPO / 'results/model_comparison/deep_ensemble'
PROD = DE / 'de_beta_nll_prof_reg_foldpca_o05l15_m5/atrain'
VARIANTS = sys.argv[1:] or ['full_noprof', 'keep_xco2_geom', 'keep_spec_geom', 'keep_geom']
OUT = DE / 'keep_only_tccon_local'

src = (REPO / 'workspace/tccon_collocate.py').read_text()
m = re.search(r'^SITE_COORDS = (\{.*?^\})', src, re.S | re.M)
SITE = ast.literal_eval(re.sub(r'#.*', '', m.group(1)))

boxes = {}
for line in (REPO / 'curc_shell_blanca_plot_corr_xco2_deepens.sh').read_text().splitlines():
    s = line.split('#')[0].split()
    if not s or s[0] != 'run_case':
        continue
    a = s[1:]
    boxes[(a[0], a[10] if len(a) > 10 else '')] = tuple(float(x) for x in a[2:6])


def hav(lon, lat, lo0, la0):
    p1, p2 = np.radians(lat), np.radians(la0)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo0 - lon) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


csv = pd.read_csv(PROD / 'tccon_comparison_r100km.csv')
csv = csv[(csv.surface == 'all') & (csv.qf_group == 'all')].set_index(['date', 'site'])
COLS = ['lon', 'lat', 'xco2_bc', 'deep_ensemble_corrected_xco2', 'sfc_type', 'cld_dist_km']
MODELS = ['full'] + VARIANTS
fps, rows = [], []
for d in sorted(p for p in PROD.iterdir() if p.name.startswith('combined_')):
    _, date, site = d.name.split('_', 2)
    if (date, site) not in csv.index:
        continue
    r = csv.loc[(date, site)]
    a = pd.read_parquet(d / 'plot_data.parquet', columns=COLS)
    lo1, lo2, la1, la2 = boxes[(date, site)]
    sel = (a.lon >= lo1) & (a.lon <= lo2) & (a.lat >= la1) & (a.lat <= la2)
    sel &= hav(a.lon.values, a.lat.values, *SITE[site]) <= 100.0
    sel &= (a.xco2_bc - r.tccon_mu_raw).abs() <= 50
    ref = r.corr_mu - r.bias_after                      # AK-harmonized TCCON reference
    f = a.loc[sel, ['sfc_type', 'cld_dist_km', 'xco2_bc']].copy()
    f['case'] = f'{date}_{site}'
    f['res_before'] = f.xco2_bc - ref
    f['res_full'] = a.deep_ensemble_corrected_xco2[sel] - ref
    row = dict(date=date, site=site, n_me=int(sel.sum()), n_csv=int(r.n_oco),
               corr_full_me=a.deep_ensemble_corrected_xco2[sel].mean(), corr_full_csv=r.corr_mu,
               land=float((f.sfc_type == 1).mean()), bias_before=r.bias_before, bias_full=r.bias_after)
    for v in VARIANTS:
        b = pd.read_parquet(DE / f'de_prof_mix_{v}' / d.name / 'plot_data.parquet', columns=COLS)
        assert len(a) == len(b) and np.allclose(a.lon, b.lon, equal_nan=True), (v, d.name)
        f[f'res_{v}'] = b.deep_ensemble_corrected_xco2[sel].values - ref
        row[f'bias_{v}'] = r.bias_after + (b.deep_ensemble_corrected_xco2[sel].mean() - row['corr_full_me'])
    for mdl in ['before'] + MODELS:
        row[f'fprmse_{mdl}'] = float(np.sqrt(np.nanmean(f[f'res_{mdl}'] ** 2)))
    fps.append(f)
    rows.append(row)

f, pc = pd.concat(fps), pd.DataFrame(rows)
ok = (pc.n_me == pc.n_csv) & np.isclose(pc.corr_full_me, pc.corr_full_csv, atol=1e-6)
rms = lambda x: float(np.sqrt(np.nanmean(x ** 2)))
oc, la = (f.sfc_type == 0).values, (f.sfc_type == 1).values
cd = f.cld_dist_km.values
slices = [('All surfaces', np.ones(len(f), bool)), ('Ocean', oc), ('Ocean <=5 km', oc & (cd <= 5)),
          ('Ocean >=5 km', oc & (cd >= 5)), ('Land', la), ('Land <=15 km', la & (cd <= 15)),
          ('Land >=15 km', la & (cd >= 15))]

L = [f'# Keep-only ablation arms against AK-harmonized TCCON (local reconstruction)', '',
     f'Production n_oco and corr_mu reproduced {ok.sum()}/{len(pc)} from atrain/tccon_comparison_r100km.csv. '
     'All keep-only arms and full_noprof are trained WITHOUT the profile EOF block; `full` is production. '
     'QF 0+1 only (plot_data has no QF column).', '',
     '## All-footprint RMSE (ppm)', '', '```',
     f"{'slice':14s} {'n':>7s} {'before':>7s} " + ' '.join(f'{m:>15s}' for m in MODELS)]
for name, msk in slices:
    L.append(f'{name:14s} {msk.sum():7d} {rms(f.res_before[msk]):7.2f} '
             + ' '.join(f'{rms(f["res_" + m][msk]):15.2f}' for m in MODELS))
L += ['```', '', '## Station-day metrics (99 station-days)', '', '```',
      f"{'metric':32s} {'before':>7s} " + ' '.join(f'{m:>15s}' for m in MODELS)]
L.append(f"{'mean |residual| (AK)':32s} {pc.bias_before.abs().mean():7.2f} "
         + ' '.join(f'{pc["bias_" + m].abs().mean():15.2f}' for m in MODELS))
L.append(f"{'mean footprint RMSE':32s} {pc.fprmse_before.mean():7.2f} "
         + ' '.join(f'{pc["fprmse_" + m].mean():15.2f}' for m in MODELS))
L.append(f"{'within +-1 ppm (of 99)':32s} {(pc.bias_before.abs() <= 1).sum():7d} "
         + ' '.join(f'{(pc["bias_" + m].abs() <= 1).sum():15d}' for m in MODELS))
L.append(f"{'|residual| smaller than before':32s} {'':7s} "
         + ' '.join(f'{(pc["bias_" + m].abs() < pc.bias_before.abs()).sum():15d}' for m in MODELS))
L.append('```')
OUT.with_suffix('.md').write_text('\n'.join(L) + '\n')
pc.to_csv(OUT.parent / (OUT.name + '_per_case.csv'), index=False)
print('\n'.join(L))

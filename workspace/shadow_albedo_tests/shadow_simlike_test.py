"""Simulation-like subset: land, reference O2A albedo 0.25-0.35 (slab bright surface 0.30),
SZA 52.5-57.5 deg (slab 55 deg). Glint mode (author request) with nadir for comparison.
Same definitions as Fig. 5 (snow-free, mixed-surface windows excluded, land reference r15)."""
import numpy as np, pandas as pd, pyarrow.dataset as ds

ROOT = '/Volumes/YWC_SSD/programming/oco_fp_analysis/results'
P = f'{ROOT}/csv_collection/combined_2016_2020_dates.parquet'
MIX = f'{ROOT}/figures/cld_dist_analysis/spec_sensitivity/prodref/mixed_surface_windows/mixed_window_flags_snowfree_15km.parquet'
OUT = '/private/tmp/claude-501/-Users-wen-programing-oco-fp-analysis/e4b23f15-723f-459c-ba0e-782efd1f5a53/scratchpad/shadow_simlike_test.csv'
BANDS = ['o2a', 'wco2', 'sco2']
cols = ['sfc_type', 'cld_dist_km', 'xco2_qf', 'snow_flag', 'fp_id', 'operation_mode', 'sza', 'vza',
        'exp_o2a_intercept', 'r15_exp_int_o2a_mean', 'r15_exp_int_o2a_std', 'r15_alb_o2a_mean',
        'xco2_bc_anomaly_r15'] + [f'{b}_k1' for b in BANDS] + [f'r15_{b}_k1_mean' for b in BANDS]
names = ds.dataset(P, format='parquet').schema.names
cols = [c for c in cols if c in names]
flt = ((ds.field('snow_flag') == 0) & (ds.field('sfc_type') == 1) & (ds.field('cld_dist_km') >= 0)
       & (ds.field('cld_dist_km') < 15) & (ds.field('sza') >= 52.5) & (ds.field('sza') <= 57.5))
d = ds.dataset(P, format='parquet').to_table(columns=cols, filter=flt).to_pandas()
mix = pd.read_parquet(MIX, columns=['fp_id', 'mixed'])
d = d[~d.fp_id.isin(mix.loc[mix.mixed, 'fp_id'].to_numpy())]
d = d[(d.r15_alb_o2a_mean >= 0.25) & (d.r15_alb_o2a_mean < 0.35)].copy()
d['z'] = (d.exp_o2a_intercept - d.r15_exp_int_o2a_mean) / d.r15_exp_int_o2a_std.replace(0, np.nan)
for b in BANDS:
    d[f'dk1_{b}'] = d[f'{b}_k1'] - d[f'r15_{b}_k1_mean']
y = d.xco2_bc_anomaly_r15
d['anom'] = y.where(y.abs() <= 100)
d['illum'] = np.select([d.z < -2, d.z < -0.5, d.z <= 0.5, d.z > 0.5],
                       ['strong shadow', 'weak shadow', 'neutral', 'brightened'], 'none')
d = d[d.illum != 'none']
d['mode'] = d.operation_mode.map({0: 'nadir', 1: 'glint', 2: 'target'})
if 'vza' in d:
    print('median VZA by mode:', d.groupby('mode').vza.median().round(1).to_dict())

rows = []
for win, wmax in [('0-5 km', 5), ('0-15 km', 15)]:
    w = d[d.cld_dist_km < wmax]
    for qf_lab, qsel in [('QF 0+1', w.xco2_qf.notna()), ('QF 0', w.xco2_qf == 0), ('QF 1', w.xco2_qf == 1)]:
        for (m, il), g in w[qsel].groupby(['mode', 'illum']):
            if m == 'target':
                continue
            row = dict(window=win, qf=qf_lab, mode=m, illum=il, n=len(g), anom_median=g.anom.median())
            for b in BANDS:
                v = g[f'dk1_{b}'].dropna()
                row[f'dk1_{b}'] = v.mean(); row[f'se_{b}'] = v.std(ddof=1) / np.sqrt(max(len(v), 2))
            rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
order = {'strong shadow': 0, 'weak shadow': 1, 'neutral': 2, 'brightened': 3}
res = res.assign(o=res.illum.map(order)).sort_values(['window', 'mode', 'qf', 'o'])
pd.set_option('display.width', 220)
for (win, m), g in res.groupby(['window', 'mode'], sort=False):
    print(f'\n== land {m}, albedo 0.25-0.35, SZA 52.5-57.5, {win}')
    print(g[['qf', 'illum', 'n', 'dk1_o2a', 'se_o2a', 'dk1_sco2', 'se_sco2', 'dk1_wco2', 'anom_median']]
          .round(4).to_string(index=False))

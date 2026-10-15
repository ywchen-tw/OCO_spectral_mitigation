"""Shadow / brightening O2A <l'> response split by viewing mode (nadir, glint) and quality flag.
Same population and definitions as Fig. 5 (snow-free, mixed-surface windows excluded,
land reference r15, ocean r05, z_gamma classes), within 5 km of cloud, binned by the
clear-sky reference O2A albedo."""
import numpy as np, pandas as pd, pyarrow.dataset as ds

ROOT = '/Volumes/YWC_SSD/programming/oco_fp_analysis/results'
P = f'{ROOT}/csv_collection/combined_2016_2020_dates.parquet'
MIX = f'{ROOT}/figures/cld_dist_analysis/spec_sensitivity/prodref/mixed_surface_windows/mixed_window_flags_snowfree_15km.parquet'
OUT = '/private/tmp/claude-501/-Users-wen-programing-oco-fp-analysis/e4b23f15-723f-459c-ba0e-782efd1f5a53/scratchpad/shadow_by_mode_qf_test.csv'

cols = ['sfc_type', 'cld_dist_km', 'xco2_qf', 'snow_flag', 'fp_id', 'operation_mode', 'exp_o2a_intercept', 'o2a_k1',
        'xco2_bc_anomaly_r05', 'xco2_bc_anomaly_r15']
for r in ('r05', 'r15'):
    cols += [f'{r}_exp_int_o2a_mean', f'{r}_exp_int_o2a_std', f'{r}_o2a_k1_mean', f'{r}_alb_o2a_mean']

flt = (ds.field('snow_flag') == 0) & (ds.field('cld_dist_km') >= 0) & (ds.field('cld_dist_km') < 5)
df = ds.dataset(P, format='parquet').to_table(columns=cols, filter=flt).to_pandas()
mix = pd.read_parquet(MIX, columns=['fp_id', 'mixed'])
df = df[~df.fp_id.isin(mix.loc[mix.mixed, 'fp_id'].to_numpy())]
print('surface x operation_mode counts (0-5 km)')
print(pd.crosstab(df.sfc_type, df.operation_mode))

MODE = {0: 'nadir', 1: 'glint', 2: 'target'}
rows = []
for sname, code, r in [('ocean', 0, 'r05'), ('land', 1, 'r15')]:
    d = df[df.sfc_type == code].copy()
    d['z'] = (d.exp_o2a_intercept - d[f'{r}_exp_int_o2a_mean']) / d[f'{r}_exp_int_o2a_std'].replace(0, np.nan)
    d['dk1'] = d.o2a_k1 - d[f'{r}_o2a_k1_mean']
    y = d[f'xco2_bc_anomaly_{r}']
    d['anom'] = y.where(y.abs() <= 100)
    d['illum'] = np.select([d.z < -2, d.z < -0.5, d.z <= 0.5, d.z > 0.5], ['strong shadow', 'weak shadow', 'neutral', 'brightened'], 'none')
    d = d[d.illum != 'none']
    d['alb'] = pd.cut(d[f'{r}_alb_o2a_mean'], [0, 0.2, 0.3, 0.4, 1.5], right=False,
                      labels=['<0.2', '0.2-0.3', '0.3-0.4', '>=0.4'])
    d['mode'] = d.operation_mode.map(MODE)
    d['qf'] = 'QF' + d.xco2_qf.astype('Int64').astype(str)
    for keys, g in d.groupby(['mode', 'qf', 'alb', 'illum'], observed=True):
        v = g.dk1.dropna()
        if len(v) < 200:
            continue
        rows.append(dict(surface=sname, mode=keys[0], qf=keys[1], alb=keys[2], illum=keys[3], n=len(v),
                         dk1=v.mean(), se=v.std(ddof=1) / np.sqrt(len(v)), anom_median=g.anom.median()))
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

order = ['strong shadow', 'weak shadow', 'neutral', 'brightened']
res['cell'] = res.apply(lambda r: f"{r.dk1:+.4f}±{r.se:.4f} ({r.n})", axis=1)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 10)
for (s, m, q), g in res.groupby(['surface', 'mode', 'qf']):
    if m == 'target':
        continue
    t = g.pivot(index='alb', columns='illum', values='cell').reindex(columns=[c for c in order if c in g.illum.values])
    print(f'\n== {s}  {m}  {q}   O2A d<l\'> mean±SE (n)')
    print(t.to_string())

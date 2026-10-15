"""Does a cloud shadow raise the O2A mean path length over bright land, as in the slab
Monte Carlo bright-surface case?  Same population and definitions as Fig. 5:
snow-free, both QF, land reference r15 (ocean r05), z_gamma = standardized O2A continuum
reflectance departure, mixed-surface reference windows excluded."""
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds

ROOT = '/Volumes/YWC_SSD/programming/oco_fp_analysis/results'
P = f'{ROOT}/csv_collection/combined_2016_2020_dates.parquet'
MIX = f'{ROOT}/figures/cld_dist_analysis/spec_sensitivity/prodref/mixed_surface_windows/mixed_window_flags_snowfree_15km.parquet'
BANDS = ['o2a', 'wco2', 'sco2']
cols = ['sfc_type', 'cld_dist_km', 'xco2_qf', 'snow_flag', 'fp_id', 'exp_o2a_intercept', 'alb_o2a', 'alb_wco2', 'r05_alb_o2a_mean', 'r15_alb_o2a_mean',
        'xco2_bc_anomaly_r05', 'xco2_bc_anomaly_r15']
for r in ('r05', 'r15'):
    cols += [f'{r}_exp_int_o2a_mean', f'{r}_exp_int_o2a_std'] + [f'{r}_{b}_k1_mean' for b in BANDS]
cols += [f'{b}_k1' for b in BANDS]

dset = ds.dataset(P, format='parquet')
flt = (ds.field('snow_flag') == 0) & (ds.field('cld_dist_km') >= 0) & (ds.field('cld_dist_km') < 15)
df = dset.to_table(columns=cols, filter=flt).to_pandas()
mix = pd.read_parquet(MIX, columns=['fp_id', 'mixed'])
df = df[~df.fp_id.isin(mix.loc[mix.mixed, 'fp_id'].to_numpy())]
print('rows after filters', len(df))

out = []
for sname, code, r in [('ocean', 0, 'r05'), ('land', 1, 'r15')]:
    d = df[df.sfc_type == code].copy()
    d['z'] = (d.exp_o2a_intercept - d[f'{r}_exp_int_o2a_mean']) / d[f'{r}_exp_int_o2a_std'].replace(0, np.nan)
    for b in BANDS:
        d[f'dk1_{b}'] = d[f'{b}_k1'] - d[f'{r}_{b}_k1_mean']
    y = d[f'xco2_bc_anomaly_{r}']
    d['anom'] = y.where(y.abs() <= 100)
    d['illum'] = np.select([d.z < -2, d.z < -1, d.z < -0.5, d.z <= 0.5], ['shadow z<-2', 'shadow -2..-1', 'shadow -1..-0.5', 'neutral'], 'brightened')
    d['alb'] = pd.cut(d[f'{r}_alb_o2a_mean'], [0, 0.1, 0.2, 0.3, 0.4, 1.5], right=False)
    for win, wmax in [('0-5km', 5), ('0-15km', 15)]:
        w = d[d.cld_dist_km < wmax]
        for (il, ab), g in w.groupby(['illum', 'alb'], observed=True):
            if len(g) < 300:
                continue
            row = dict(surface=sname, window=win, illum=il, alb_o2a=str(ab), n=len(g),
                       anom_median=g.anom.median())
            for b in BANDS:
                v = g[f'dk1_{b}'].dropna()
                row[f'dk1_{b}'] = v.mean(); row[f'se_{b}'] = v.std(ddof=1) / np.sqrt(len(v))
            out.append(row)
res = pd.DataFrame(out)
res.to_csv('/private/tmp/claude-501/-Users-wen-programing-oco-fp-analysis/e4b23f15-723f-459c-ba0e-782efd1f5a53/scratchpad/shadow_by_refalbedo_test.csv', index=False)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
order = ['shadow z<-2', 'shadow -2..-1', 'shadow -1..-0.5', 'neutral', 'brightened']
for (s, w), g in res.groupby(['surface', 'window']):
    g = g.assign(o=g.illum.map({k: i for i, k in enumerate(order)})).sort_values(['alb_o2a', 'o'])
    print(f'\n== {s} {w}')
    print(g[['alb_o2a', 'illum', 'n', 'dk1_o2a', 'se_o2a', 'dk1_sco2', 'dk1_wco2', 'anom_median']].round(4).to_string(index=False))

"""Strongly shadowed stratum (z_gamma < -2) on the Fig. 5 population (2026-11-03).

Usage: PYTHONPATH=src:workspace python workspace/shadow_brightening_strata.py \
    results_doppler/figures/cld_dist_analysis/spec_sensitivity/prodref/\
    shadow_brightening_strata_snowfree_15km_samesfc.csv

Same population and reference as
results_doppler/figures/cld_dist_analysis/spec_sensitivity/prodref/
shadow_brightening_stats_snowfree_15km_samesfc.csv (src/analysis/spec_sensitivity.py
--analyses shadow --fit-schema v2 --reference production --population snowfree
--near-max 15 --exclude-mixed-ref .../mixed_window_flags_snowfree_15km.parquet).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "workspace")
from analysis.spec_sensitivity import _apply_production_reference
from analysis.utils import apply_quality_filter
from analysis.nosg_reference import read_combined

R = Path("results_doppler")
path = R / "csv_collection" / "combined_2016_2020_dates.parquet"
flags_p = (R / "figures/cld_dist_analysis/spec_sensitivity/prodref/"
           "mixed_surface_windows/mixed_window_flags_snowfree_15km.parquet")
cols = ['date', 'lon', 'lat', 'sfc_type', 'cld_dist_km', 'xco2_bc', 'xco2_qf',
        'snow_flag', 'xco2_bc_anomaly', 'fp_id', 'xco2_bc_anomaly_r05',
        'xco2_bc_anomaly_r15']
for band, stub in [('o2a', 'k1'), ('wco2', 'k1'), ('sco2', 'k1'), ('o2a', 'exp_int')]:
    stem = f'{band}_{stub}_am'
    cols += [stem] + [f'{p}_{stem}_{s}' for p in ('r05', 'r15') for s in ('mean', 'std')]
df = read_combined(path, sorted(set(cols)))
print("loaded", len(df), flush=True)
df = apply_quality_filter(df, require_qf0=False)
df = _apply_production_reference(df, 'v2')
d = df['cld_dist_km']
near = df[(d >= 0) & (d < 15)].copy()
flags = pd.read_parquet(flags_p, columns=['fp_id', 'mixed'])
near = near[~near['fp_id'].isin(flags.loc[flags['mixed'], 'fp_id'].to_numpy())]
edges = [0, 1, 2, 3, 5, 7, 10, 12, 15]
labels = [f"{edges[i]}–{edges[i+1]}" for i in range(len(edges) - 1)]
near['_bin'] = pd.cut(near['cld_dist_km'], bins=edges, labels=labels, right=False)
z = near['zexp_o2a']
strata = {'brightened': z > 0.5, 'neutral': z.abs() <= 0.5, 'shadowed': z < -0.5,
          'strongly_shadowed': z < -2, 'strongly_brightened': z > 2}
rows = []
for sname, code in [('ocean', 0), ('land', 1)]:
    s = near[near.sfc_type == code]
    w5 = s.cld_dist_km < 5
    for k, m in strata.items():
        m = m.reindex(s.index, fill_value=False)
        b = s[m]
        for var in ('dk1_o2a', 'dk1_wco2', 'dk1_sco2', 'xco2_bc_anomaly'):
            g = b.groupby('_bin', observed=False)[var].agg(['mean', 'sem', 'count']).reindex(labels)
            for lbl, r in g.iterrows():
                rows.append(dict(surface=sname, stratum=k, variable=var, cld_bin=lbl,
                                 mean=r['mean'], sem=r['sem'], n=int(r['count'])))
            for lbl, bb in (('0–5', b[b.cld_dist_km < 5]), ('all', b)):
                rows.append(dict(surface=sname, stratum=k, variable=var, cld_bin=lbl,
                                 mean=bb[var].mean(), sem=bb[var].sem(), n=int(bb[var].count())))
        print(sname, k, "share within 5 km", f"{(m & w5).sum() / w5.sum():.3f}", flush=True)
out = pd.DataFrame(rows)
out.to_csv(Path(sys.argv[1]), index=False)
v = out[(out.variable == 'dk1_o2a') & out.cld_bin.isin(['0–1', '1–2', '0–5', 'all'])]
print(v.pivot_table(index=['surface', 'stratum'], columns='cld_bin', values='mean').round(4).to_string())

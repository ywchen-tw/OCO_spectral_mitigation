"""Fraction of footprints within 4 and 10 km of the nearest MODIS cloud on the 116 training dates,
by quality flag, for comparison with Massie et al. (2021): ~40 % of QF=0 and 73 % of QF=1 within 4 km."""
import pyarrow.dataset as ds, numpy as np
P='/Volumes/YWC_SSD/programming/oco_fp_analysis/results/csv_collection/combined_2016_2020_dates.parquet'
t=ds.dataset(P,format='parquet').to_table(columns=['cld_dist_km','xco2_qf','sfc_type'],filter=ds.field('cld_dist_km')>=0).to_pandas()
print('n', len(t))
for lab,g in [('all',t),('QF0',t[t.xco2_qf==0]),('QF1',t[t.xco2_qf==1])]:
    print(f"{lab:4s} n={len(g):9d} <=4km {100*(g.cld_dist_km<4).mean():.1f}%  <=10km {100*(g.cld_dist_km<10).mean():.1f}%")

"""diagnose_increment_scatter.py — why does the anchor raise near-cloud fp-RMSE?

Over the land footprints within --radius-km of each evaluation station (all
QF), split by nearest-cloud distance, compares the within-case footprint
scatter of

    raw_corr      = xco2_raw − mu_raw            (ML(raw))
    anchored      = raw_corr − LR(f)             (per-footprint anchor)
    case-mean     = raw_corr − ⟨LR(f)⟩_case      (same anchor, footprint structure removed)

and the pooled fp-RMSE to the case's AK TCCON reference (taken from the
anchor report's per-case CSV).  If the LR increment carries cloud-correlated
footprint structure, σ(anchored) > σ(raw_corr) near clouds while the
case-mean variant keeps the bias gain without the scatter penalty.

Run:  python workspace/anchor_lr/diagnose_increment_scatter.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchor_common as ac                                   # noqa: E402
from tccon_collocate import SITE_COORDS, _haversine_km       # noqa: E402

RADIUS_KM = 100.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', default='', help="'' (b11) or 'nocontam'")
    args = ap.parse_args()
    variant = args.variant
    INC, CORR = ac.inc_col(variant), ac.corr_col(variant)
    COLS = ['lon', 'lat', 'sfc_type', 'cld_dist_km', 'xco2_raw', 'xco2_bc',
            ac.ML_RAW_CORR_COL, INC, CORR]
    rep = pd.read_csv(ac.report_dir(variant) / f'tccon_comparison{ac.report_sfx(variant)}.csv')
    rep = rep[(rep.surface == 'land') & (rep.qf_group == 'all')]
    tref = {(r.date, r.site): float(r.tccon_mu) for r in rep.itertuples()}
    rows = []
    for case_dir in sorted(ac.RAW_TREE.glob('combined_*')):
        pq = case_dir / 'plot_data.parquet'
        date = ac.date_from_name(case_dir.name); site = case_dir.name.split('_')[-1]
        if not pq.exists() or (date, site) not in tref or site not in SITE_COORDS:
            continue
        df = pd.read_parquet(pq, columns=COLS)
        slon, slat = SITE_COORDS[site]
        d = _haversine_km(df.lon.to_numpy(float), df.lat.to_numpy(float), slon, slat)
        m = (df.sfc_type.to_numpy(float) == 1) & (d <= RADIUS_KM) \
            & np.isfinite(df[INC].to_numpy(float))
        if m.sum() < 10:
            continue
        s = df.loc[m].copy()
        s['case'] = f'{date}_{site}'; s['tref'] = tref[(date, site)]
        s['inc_case_mean'] = float(s[INC].mean())
        rows.append(s)
    d = pd.concat(rows, ignore_index=True)
    rc = d[ac.ML_RAW_CORR_COL].to_numpy(float)
    inc = d[INC].to_numpy(float)
    series = {'raw (xco2_raw)': d.xco2_raw.to_numpy(float),
              'B11 (xco2_bc)': d.xco2_bc.to_numpy(float),
              'ML(raw) raw_corr': rc,
              'ML(raw)+anchor (per-fp)': rc - inc,
              'ML(raw)+anchor (case-mean)': rc - d.inc_case_mean.to_numpy(float)}
    cld = d.cld_dist_km.to_numpy(float)
    groups = {'all': np.ones(len(d), bool), '≤10 km': cld <= 10, '>10 km': cld > 10}
    lines = ['| series | group | n_fp | within-case σ (ppm) | fp-RMSE to TCCON_AK (ppm) | mean bias (ppm) | station-equal \\|bias\\| |',
             '|---|---|---|---|---|---|---|']
    case = d.case.to_numpy()
    tr = d.tref.to_numpy(float)
    recs = []
    site = np.array([c.split('_')[-1] for c in case])
    for name, vals in series.items():
        for gname, g in groups.items():
            vv, cc, tt = vals[g], case[g], tr[g]
            dev = vv - pd.Series(vv).groupby(cc).transform('mean').to_numpy()
            # station-equal mean |bias|: footprint-mean residual per site → |·| → mean
            per_site = pd.Series(vv - tt).groupby(site[g]).mean()
            rec = dict(variant=variant or 'b11', series=name, group=gname, n_fp=int(g.sum()),
                       within_case_sd=float(np.sqrt(np.mean(dev ** 2))),
                       fp_rmse=float(np.sqrt(np.mean((vv - tt) ** 2))),
                       bias=float(np.mean(vv - tt)),
                       st_abs_bias=float(per_site.abs().mean()))
            recs.append(rec)
            lines.append(f'| {name} | {gname} | {g.sum():,} | {rec["within_case_sd"]:.3f} | '
                         f'{rec["fp_rmse"]:.3f} | {rec["bias"]:+.3f} | {rec["st_abs_bias"]:.3f} |')
    pd.DataFrame(recs).to_csv(
        ac.OUT_ROOT / f'increment_scatter_diagnostic{ac.variant_tag(variant)}.csv', index=False)
    # increment structure itself
    dev_inc = inc - pd.Series(inc).groupby(case).transform('mean').to_numpy()
    dev_rc = rc - pd.Series(rc).groupby(case).transform('mean').to_numpy()
    lines += ['', '| LR increment | group | within-case σ(LR) | corr(LR dev, raw_corr dev) |',
              '|---|---|---|---|']
    for gname, g in groups.items():
        r = np.corrcoef(dev_inc[g], dev_rc[g])[0, 1]
        lines.append(f'| LR(f) | {gname} | {np.sqrt(np.mean(dev_inc[g] ** 2)):.3f} | {r:+.3f} |')
    out = ac.OUT_ROOT / f'increment_scatter_diagnostic{ac.variant_tag(variant)}.md'
    out.write_text(f'# Anchor increment scatter diagnostic — variant {variant or "b11"} '
                   '(land fp ≤100 km of station, all QF)\n\n'
                   + '\n'.join(lines) + '\n')
    print('\n'.join(lines)); print(f'→ {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""summarize_variants.py — one table across anchor term sets (land, AK reference).

Rows: raw, B11, ML(bc) (same-102-case rerun), ML(raw), then every variant
found under results/anchor_lr (per-footprint anchor and its case-mean twin).
Columns: fp-RMSE on land all / ≤10 km / ≥10 km / QF=1, station-equal mean
|bias| (land rows), and the within-case σ of the LR increment near clouds.

Run:  python workspace/anchor_lr/summarize_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchor_common as ac                       # noqa: E402

BC102 = ac.OUT_ROOT / 'bc_tree_102cases'


def _m(csv, qf, cld):
    d = pd.read_csv(csv)
    r = d[(d.surface == 'land') & (d.qf_group == qf) & (d.cld_group == cld)]
    assert len(r) == 1, (csv, qf, cld, len(r))
    return r.iloc[0]


def _station_equal(csv, col='bias_after'):
    d = pd.read_csv(csv)
    g = d[(d.surface == 'land') & (d.qf_group == 'all')]
    per = []
    for _, s in g.groupby('site'):
        w = s['n_oco'].to_numpy(float); v = s[col].to_numpy(float)
        m = np.isfinite(w) & np.isfinite(v)
        if m.any() and w[m].sum() > 0:
            per.append(abs(np.average(v[m], weights=w[m])))
    return float(np.mean(per))


def main() -> int:
    rows = []

    def add(label, metrics_csv, comp_csv, after=True, sd_near=np.nan, terms=''):
        key = 'rmse_after' if after else None
        def rm(qf, cld, col):
            return float(_m(metrics_csv, qf, cld)[col])
        col = key or label_col
        rows.append(dict(
            series=label, terms=terms,
            rmse_all=rm('all', 'all', col), rmse_near=rm('all', '0–10 km', col),
            rmse_far=rm('all', '≥10 km', col), rmse_qf1=rm('qf1', 'all', col),
            bias_all=float(_m(metrics_csv, 'all', 'all')[col.replace('rmse', 'bias')]),
            st_abs_bias=_station_equal(comp_csv, col.replace('rmse', 'bias')),
            lr_sd_near=sd_near))

    raw_m = ac.RAW_TREE / 'tccon_metrics_ak_r100km.csv'
    raw_c = ac.RAW_TREE / 'tccon_comparison_r100km.csv'
    global label_col
    for label, col in (('raw', 'rmse_raw'), ('B11', 'rmse_before')):
        label_col = col
        add(label, raw_m, raw_c, after=False)
    add('ML(bc)', BC102 / 'tccon_metrics_ak_r100km.csv', BC102 / 'tccon_comparison_r100km.csv')
    add('ML(raw)', raw_m, raw_c)
    for js in sorted(ac.OUT_ROOT.glob('anchor_lr_land*.json')):
        model = ac.load_model(js)
        v = model.get('variant', '')
        name = v or 'b11'
        rd, sfx = ac.report_dir(v), ac.report_sfx(v)
        mcsv, ccsv = rd / f'tccon_metrics_ak{sfx}.csv', rd / f'tccon_comparison{sfx}.csv'
        if not mcsv.exists():
            continue
        diag = ac.OUT_ROOT / f'increment_scatter_diagnostic{ac.variant_tag(v)}.csv'
        sd_near = cm = None
        if diag.exists():
            dd = pd.read_csv(diag)
            sd_near = float(dd[(dd.series == 'ML(raw)+anchor (per-fp)') & (dd.group == '≤10 km')]
                            .within_case_sd.iloc[0])
            cm = dd[dd.series == 'ML(raw)+anchor (case-mean)'].set_index('group')
            rc = dd[dd.series == 'ML(raw) raw_corr'].set_index('group')
            sd_near = float(np.sqrt(max(sd_near ** 2 - rc.loc['≤10 km', 'within_case_sd'] ** 2, 0)))
        terms = ('+'.join(model['terms']['continuous']) or '(fp offsets only)') \
            + (' [per-fp]' if model.get('per_footprint') else '')
        add(f'ML(raw)+anchor[{name}]', mcsv, ccsv, sd_near=sd_near if sd_near is not None else np.nan,
            terms=terms)
        if cm is not None:
            rows.append(dict(series=f'  … case-mean LR [{name}]', terms='',
                             rmse_all=cm.loc['all', 'fp_rmse'], rmse_near=cm.loc['≤10 km', 'fp_rmse'],
                             rmse_far=cm.loc['>10 km', 'fp_rmse'], rmse_qf1=np.nan,
                             bias_all=cm.loc['all', 'bias'],
                             st_abs_bias=(float(cm.loc['all', 'st_abs_bias'])
                                          if 'st_abs_bias' in cm.columns else np.nan),
                             lr_sd_near=np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(ac.OUT_ROOT / 'variant_summary.csv', index=False)
    L = ['# Anchor term-set sweep — land, AK-harmonized TCCON, 102 evaluation cases (ppm)', '',
         'fp-RMSE over pooled land footprints (all QF unless noted); station-equal mean |bias| '
         'over land report rows; "added σ near" = extra within-case footprint scatter the LR '
         'increment injects at ≤10 km (quadrature difference vs ML(raw)). Case-mean rows apply '
         'the same LR as one offset per station-day (diagnostic selection, not the report; their '
         'station |bias| is the diagnostic\'s site-mean residual, comparable in kind, not identical).', '',
         '| series | RMSE all | RMSE ≤10 km | RMSE ≥10 km | RMSE QF=1 | bias all | station \\|bias\\| | added σ near | terms |',
         '|---|---|---|---|---|---|---|---|---|']
    f = lambda x: '—' if not np.isfinite(x) else f'{x:.3f}'          # noqa: E731
    for r in t.itertuples():
        L.append(f'| {r.series} | {f(r.rmse_all)} | {f(r.rmse_near)} | {f(r.rmse_far)} | '
                 f'{f(r.rmse_qf1)} | {"—" if not np.isfinite(r.bias_all) else f"{r.bias_all:+.3f}"} | '
                 f'{f(r.st_abs_bias)} | {f(r.lr_sd_near)} | {r.terms} |')
    out = ac.OUT_ROOT / 'variant_summary.md'
    out.write_text('\n'.join(L) + '\n')
    print('\n'.join(L)); print(f'→ {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

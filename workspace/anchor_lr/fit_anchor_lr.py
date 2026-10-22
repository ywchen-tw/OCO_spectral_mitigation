"""fit_anchor_lr.py — step 3: the single OLS anchor fit on the training-date pool.

Rows: land, QF = 0, not guarded, finite predictors, station-days with
n_tccon > 0 and >= --min-fp such rows.  Target per footprint:

    y = raw_corr − TCCON_ref_AK(case),   raw_corr = xco2_raw − mu_raw (out-of-fold)

TCCON_ref_AK is the AK/prior-harmonized reference computed exactly as in
tccon_comparison_report.py --ak-harmonize (operator from the collocated
footprints' ak_NN/pwf_NN/co2_ap_NN/plev_NN columns; falls back to the direct
window mean if the operator cannot be built, flagged in the per-case table).

Design: see anchor_common.design_matrix (intercept, 8 footprint offsets as
sum-to-zero contrasts, 7 centered continuous B11-style terms).  Weighting
(default 'station-day'): every station-day carries total weight 1 so a
data-rich site (Lamont) does not dominate; 'footprint' = plain OLS.

Outputs (results/anchor_lr/):
    anchor_lr_land.json      coefficients, centers, counts, in-sample fit stats
    anchor_lr_land.md        coefficient table (SE / t descriptive, not a CV)
    anchor_lr_fit_cases.csv  per-station-day residuals before/after + B11 reference

Run:  python workspace/anchor_lr/fit_anchor_lr.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchor_common as ac                                                  # noqa: E402
from ak_harmonize import operator_from_dataframe, ak_adjusted_ref_from_operator  # noqa: E402


def _ak_columns(cols) -> list[str]:
    return sorted(c for c in cols
                  if any(c.startswith(p) and c[len(p):].isdigit()
                         for p in ('ak_', 'pwf_', 'co2_ap_', 'plev_')))


def load_case(r, *, window_min: float, xco2_var: str):
    """Merged plot_data + source rows for one pool case, with the AK reference."""
    src = ac.ROOT / r.source
    pdf = pd.read_parquet(src.parent / 'plot_data.parquet')
    s_all = pd.read_parquet(src)
    want = ac.MERGE_KEYS + ['xco2_qf'] + [c for c in ac.SOURCE_COLS
                                           if c in s_all.columns and c not in pdf.columns]
    want += _ak_columns(s_all.columns)
    s = s_all[want].drop_duplicates(ac.MERGE_KEYS)
    m = pdf.merge(s, on=ac.MERGE_KEYS, how='left')
    # AK/prior-harmonized reference on the collocated footprints (as the report)
    tref_direct = float(r.tccon_ref_direct)
    tref, ak_delta, ref_source = tref_direct, np.nan, 'direct'
    ot = pd.to_datetime(m['time'], unit='s', utc=True, errors='coerce').dropna()
    if len(ot):
        op = operator_from_dataframe(m)
        if op is not None:
            tfile = ac.tccon_file(r.site)
            adj = ak_adjusted_ref_from_operator(
                op, str(tfile), ot.min().tz_localize(None), ot.max().tz_localize(None),
                window_min=window_min, xco2_var=xco2_var)
            if adj is not None:
                tref, ak_delta, ref_source = adj['tccon_ref_ak'], adj['ak_delta'], 'ak'
    return m, tref, tref_direct, ak_delta, ref_source


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weighting', choices=('station-day', 'footprint'), default='station-day')
    ap.add_argument('--qf', type=int, default=0, help='xco2_qf value used for the fit (default 0)')
    ap.add_argument('--min-fp', type=int, default=10)
    ap.add_argument('--window-min', type=float, default=60.0)
    ap.add_argument('--tccon-xco2-var', default='xco2')
    ap.add_argument('--term-set', choices=sorted(ac.TERM_SETS), default='b11',
                    help="continuous terms: 'b11' (full DUG land family) or 'nocontam' "
                         "(drops the contamination-group terms log_dws/aod_fine/aod_ice)")
    ap.add_argument('--per-footprint', action='store_true',
                    help='fit a SEPARATE regression (intercept + terms) for each of the 8 '
                         'footprints instead of shared slopes + footprint offsets')
    ap.add_argument('--out', type=Path, default=None,
                    help='model JSON (default results/anchor_lr/anchor_lr_land[_<term-set>].json)')
    args = ap.parse_args()
    terms = ac.TERM_SETS[args.term_set]
    variant = '' if args.term_set == 'b11' else args.term_set
    if args.per_footprint:
        variant = f'{args.term_set}_perfp'
    tag = ac.variant_tag(variant)
    if args.out is None:
        args.out = ac.coef_json(variant)

    man = pd.read_csv(ac.MANIFEST)
    frames, cases = [], []
    for r in man.itertuples():
        m, tref, tref_direct, ak_delta, ref_source = load_case(
            r, window_min=args.window_min, xco2_var=args.tccon_xco2_var)
        guard = m['clim_guard'].to_numpy(bool) | m['anomaly_guard'].to_numpy(bool)
        X, names, _, ok_x = ac.design_matrix(m, terms=terms)
        sel = ((m['xco2_qf'].to_numpy(float) == args.qf) & ~guard & ok_x
               & np.isfinite(m[ac.ML_RAW_CORR_COL].to_numpy(float)))
        n_sel = int(sel.sum())
        if n_sel < args.min_fp or not np.isfinite(tref):
            print(f'  drop {r.date} {r.site}: {n_sel} usable QF={args.qf} rows (ref {ref_source})')
            continue
        sub = m.loc[sel, ac.MERGE_KEYS + ['fp', 'xco2_bc', 'xco2_raw', ac.ML_RAW_CORR_COL]
                    + [c for c in ac.SOURCE_COLS if c != 'fp' and c in m.columns]].copy()
        sub['case'] = f'{r.date}_{r.site}'
        sub['site'] = r.site
        sub['tref'] = tref
        frames.append(sub)
        cases.append(dict(date=r.date, site=r.site, n_fit=n_sel, n_tccon=int(r.n_tccon),
                          tccon_ref_ak=tref, tccon_ref_direct=tref_direct, ak_delta=ak_delta,
                          ref_source=ref_source, held_fold=int(r.held_fold)))
    d = pd.concat(frames, ignore_index=True)
    cases = pd.DataFrame(cases)
    print(f'fit rows: {len(d):,} footprints, {len(cases)} station-days, '
          f'{cases.site.nunique()} sites ({(cases.ref_source == "ak").sum()} AK-referenced)')

    X, names, centers, ok = ac.design_matrix(d, terms=terms,
                                             footprint_contrasts=not args.per_footprint)
    assert ok.all()
    y = d[ac.ML_RAW_CORR_COL].to_numpy(float) - d['tref'].to_numpy(float)
    y_b11 = d['xco2_bc'].to_numpy(float) - d['tref'].to_numpy(float)
    n_case = d.groupby('case')['case'].transform('size').to_numpy(float)
    w = np.ones(len(d)) if args.weighting == 'footprint' else 1.0 / n_case
    w *= len(d) / w.sum()                      # Σw = n (keeps the SE formula classical)
    sw = np.sqrt(w)

    def _wls(Xs, ys, ws):
        b, *_ = np.linalg.lstsq(Xs * np.sqrt(ws)[:, None], ys * np.sqrt(ws), rcond=None)
        r = ys - Xs @ b
        s2 = float(np.sum(ws * r ** 2) / max(len(ys) - Xs.shape[1], 1))
        # pinv: a constant (zero-variance after centering) or collinear column
        # gets a zero coefficient from lstsq and a NaN/0 SE here instead of a crash
        c = s2 * np.linalg.pinv((Xs * ws[:, None]).T @ Xs)
        se = np.sqrt(np.maximum(np.diag(c), 0.0))
        return b, r, s2, se

    coef_by_fp = {}
    if args.per_footprint:
        fp_all = d['fp'].to_numpy(float)
        res = np.full(len(d), np.nan)
        beta = np.zeros(X.shape[1]); se = np.zeros(X.shape[1])   # footprint-mean, for the table
        se_by_fp = {}
        for k in range(ac.N_FP):
            m = fp_all == k
            bk, rk, _, sek = _wls(X[m], y[m], w[m])
            res[m] = rk
            coef_by_fp[str(k)] = {n: float(b) for n, b in zip(names, bk)}
            se_by_fp[str(k)] = {n: float(s) for n, s in zip(names, sek)}
            beta += bk / ac.N_FP; se += sek / ac.N_FP
        p = X.shape[1] * ac.N_FP
        sigma2 = float(np.sum(w * res ** 2) / (len(d) - p))
    else:
        beta, res, sigma2, se = _wls(X, y, w)
        p = X.shape[1]
    ybar = np.sum(w * y) / w.sum()
    r2 = 1.0 - np.sum(w * res ** 2) / np.sum(w * (y - ybar) ** 2)

    # in-sample diagnostics: footprint RMS and station-day mean residuals
    d['y_before'] = y; d['y_after'] = res; d['y_b11'] = y_b11
    per = d.groupby(['case', 'site']).agg(n=('y_before', 'size'),
                                           bias_before=('y_before', 'mean'),
                                           bias_after=('y_after', 'mean'),
                                           bias_b11=('y_b11', 'mean')).reset_index()
    per = per.merge(cases.assign(case=cases.date + '_' + cases.site)
                    [['case', 'date', 'tccon_ref_ak', 'ak_delta', 'ref_source', 'held_fold']],
                    on='case', how='left')
    per.to_csv(ac.OUT_ROOT / f'anchor_lr_fit_cases{tag}.csv', index=False)
    rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, float) ** 2)))   # noqa: E731
    stats = dict(
        n_rows=int(len(d)), n_station_days=int(len(cases)), n_sites=int(cases.site.nunique()),
        n_dates=int(cases.date.nunique()), weighting=args.weighting, qf=args.qf,
        r2_weighted=float(r2), sigma_resid_ppm=float(np.sqrt(sigma2)),
        fp_rms_before=rms(y), fp_rms_after=rms(res), fp_rms_b11=rms(y_b11),
        fp_mean_before=float(y.mean()), fp_mean_after=float(res.mean()), fp_mean_b11=float(y_b11.mean()),
        stationday_mean_abs_bias_before=float(per.bias_before.abs().mean()),
        stationday_mean_abs_bias_after=float(per.bias_after.abs().mean()),
        stationday_mean_abs_bias_b11=float(per.bias_b11.abs().mean()),
    )
    try:
        git_sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ac.ROOT,
                                 capture_output=True, text=True).stdout.strip()
    except Exception:                                    # noqa: BLE001
        git_sha = ''
    model = dict(
        description='TCCON-anchored ML(raw) land LR (B11-style predictors, additive, scale fixed at 1)',
        created=dt.datetime.now().isoformat(timespec='seconds'), git_sha=git_sha,
        manifest_sha256=hashlib.sha256(ac.MANIFEST.read_bytes()).hexdigest()[:16],
        input_corr_col=ac.ML_RAW_CORR_COL, surface='land', reference='ak',
        variant=variant, term_set=args.term_set, per_footprint=bool(args.per_footprint),
        coef_by_fp=coef_by_fp,
        terms=dict(continuous=list(terms),
                   footprint=('separate regression per footprint' if args.per_footprint
                              else 'sum-to-zero contrasts fp0..fp6 vs fp7'),
                   aod_fine='aod_sulfate + aod_oc + aod_bc (verify vs DUG)',
                   log_dws=f'log(max(dws, {ac.DWS_FLOOR}))',
                   albedo='alb_wco2 and alb_wco2**2 (general quadratic; verify DUG form)'),
        centers=centers, coef={n: float(b) for n, b in zip(names, beta)},
        se={n: float(s) for n, s in zip(names, se)},
        footprint_offsets=(None if args.per_footprint else
                           ac.footprint_offsets({n: float(b) for n, b in zip(names, beta)})),
        fit_stats=stats,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2))

    L = [f'# Anchor LR (land, term set {args.term_set}'
         f'{", separate regression per footprint" if args.per_footprint else ""}) — fit {model["created"]}', '',
         f'Pool: {stats["n_rows"]:,} QF={args.qf} land footprints, {stats["n_station_days"]} '
         f'station-days, {stats["n_sites"]} sites, {stats["n_dates"]} training dates; '
         f'weighting = {args.weighting}; reference = AK-harmonized TCCON '
         f'({int((cases.ref_source == "ak").sum())}/{len(cases)} cases).', '',
         f'Target y = (xco2_raw − mu_raw) − TCCON_AK.  Weighted R² = {r2:.3f}, '
         f'residual σ = {np.sqrt(sigma2):.3f} ppm.', '',
         '| metric | raw_corr − TCCON (before) | after anchor (in-sample) | B11 (xco2_bc − TCCON, same rows) |',
         '|---|---|---|---|',
         f'| footprint RMS (ppm) | {stats["fp_rms_before"]:.3f} | {stats["fp_rms_after"]:.3f} | {stats["fp_rms_b11"]:.3f} |',
         f'| footprint mean (ppm) | {stats["fp_mean_before"]:+.3f} | {stats["fp_mean_after"]:+.3f} | {stats["fp_mean_b11"]:+.3f} |',
         f'| station-day mean abs bias (ppm) | {stats["stationday_mean_abs_bias_before"]:.3f} | '
         f'{stats["stationday_mean_abs_bias_after"]:.3f} | {stats["stationday_mean_abs_bias_b11"]:.3f} |',
         '', '## Coefficients (continuous terms centered at pool medians'
         + (', footprint-mean of the 8 per-footprint fits)' if args.per_footprint else ')'), '',
         '| term | coef | SE | t | center |', '|---|---|---|---|---|']
    for n, b, s in zip(names, beta, se):
        c = centers.get(n, np.nan)
        tval = b / s if s > 0 else np.nan
        L.append(f'| {n} | {b:+.4g} | {s:.3g} | {tval:+.1f} | '
                 f'{"" if not np.isfinite(c) else f"{c:.4g}"} |')
    if args.per_footprint:
        L += ['', '## Per-footprint coefficients', '',
              '| term | ' + ' | '.join(f'fp{k}' for k in range(ac.N_FP)) + ' |',
              '|---|' + '---|' * ac.N_FP]
        for n in names:
            L.append(f'| {n} | ' + ' | '.join(f'{coef_by_fp[str(k)][n]:+.4g}'
                                             for k in range(ac.N_FP)) + ' |')
    else:
        L += ['', 'Footprint offsets (ppm, fp 0..7): '
              + ', '.join(f'{o:+.3f}' for o in model['footprint_offsets'])]
    L += ['', '## Per-station-day (in-sample)', '',
          '| case | n | before | after | B11 | ref |', '|---|---|---|---|---|---|']
    for r in per.sort_values('case').itertuples():
        L.append(f'| {r.case} | {r.n} | {r.bias_before:+.2f} | {r.bias_after:+.2f} | '
                 f'{r.bias_b11:+.2f} | {r.ref_source} |')
    (ac.OUT_ROOT / f'anchor_lr_land{tag}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:16]))
    print(f'→ {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

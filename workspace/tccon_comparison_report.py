"""tccon_comparison_report.py — combined before/after-vs-TCCON summary across cases.

Reads the active `run_case` lines from a deep-ensemble plotting script, and for each
case reproduces the figure's comparison logic:
    drop guarded footprints → lon/lat box → footprints ≤ --radius-km of the TCCON
    station → TCCON within ± --window-min of the OCO-2 pass on that date.
Then computes mean ± std of the ORIGINAL (xco2_bc), CORRECTED, and TCCON XCO2, and
the bias to TCCON before/after correction.

Outputs (to --output-dir):
    tccon_comparison.csv   — one row per case
    tccon_comparison.md    — markdown table + aggregate summary
    tccon_comparison.png   — (a) corrected/original vs TCCON scatter (1:1)
                              (b) bias-to-TCCON before→after dumbbell, per case

Example:
    PYTHONPATH=src python workspace/tccon_comparison_report.py \
        --script curc_shell_blanca_plot_corr_xco2_deepens.sh \
        --output-dir results/model_comparison/deep_ensemble
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_corrected_xco2 import load_tccon, get_storage_dir
from tccon_collocate import collocate, find_plotdata

CORR = 'deep_ensemble_corrected_xco2'


def _ols_fit(x, y):
    """OLS y~x on finite pairs. Returns dict with slope, intercept, slope_se,
    intercept_se, r2, n (or None if <3 valid points)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3:
        return None
    xm, ym = x.mean(), y.mean()
    Sxx = np.sum((x - xm) ** 2)
    if Sxx == 0:
        return None
    slope = np.sum((x - xm) * (y - ym)) / Sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    sse = np.sum(resid ** 2)
    sst = np.sum((y - ym) ** 2)
    dof = n - 2
    s2 = sse / dof if dof > 0 else np.nan
    slope_se = np.sqrt(s2 / Sxx) if dof > 0 else np.nan
    intercept_se = np.sqrt(s2 * (1.0 / n + xm ** 2 / Sxx)) if dof > 0 else np.nan
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return dict(slope=slope, intercept=intercept, slope_se=slope_se,
                intercept_se=intercept_se, r2=r2, n=n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--script', default='curc_shell_blanca_plot_corr_xco2_deepens.sh')
    ap.add_argument('--out-base', default='results/model_comparison/deep_ensemble',
                    help='Base dir holding combined_DATE[_SITE] case dirs.')
    ap.add_argument('--output-dir', default='results/model_comparison/deep_ensemble')
    ap.add_argument('--radius-km', type=float, default=100.0)
    ap.add_argument('--window-min', type=float, default=60.0)
    ap.add_argument('--fname-suffix', default='',
                    help="Appended before the extension of every output filename "
                         "(e.g. '_r100km') so a parameter sweep's reports coexist.")
    args = ap.parse_args()

    out_base = Path(args.out_base)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sfx = args.fname_suffix
    # Output paths (suffix inserted before the extension).
    P_CSV      = out_dir / f'tccon_comparison{sfx}.csv'
    P_MD       = out_dir / f'tccon_comparison{sfx}.md'
    P_PNG      = out_dir / f'tccon_comparison{sfx}.png'
    P_SITE_CSV = out_dir / f'tccon_comparison_by_site{sfx}.csv'
    P_SITE_PNG = out_dir / f'tccon_comparison_by_site{sfx}.png'
    storage = get_storage_dir()

    # ── parse active run_case lines ────────────────────────────────────────────
    cases = []
    for ln in Path(args.script).read_text().splitlines():
        if not re.match(r'^run_case\s', ln):
            continue
        t = ln.split()
        date, tccon = t[1], t[2]
        lonmin, lonmax, latmin, latmax = map(float, t[3:7])
        rest = t[9:]
        site = rest[2] if len(rest) >= 3 else ''
        cases.append(dict(date=date, tccon=tccon, lonmin=lonmin, lonmax=lonmax,
                          latmin=latmin, latmax=latmax, site=site))

    _tccon_cache = {}
    def tccon_df(name):
        if name not in _tccon_cache:
            p = Path('data/TCCON') / name
            p = p if p.exists() else storage / 'data/TCCON' / name
            _tccon_cache[name] = load_tccon(str(p))
        return _tccon_cache[name]

    def source_parquet(date):
        """Locate the pre-apply source parquet (carries xco2_raw, absent from
        plot_data.parquet). Returns a Path or None."""
        name = f"combined_{date}_all_orbits.parquet"
        for base in (Path('results/csv_collection'), storage / 'results/csv_collection'):
            p = base / name
            if p.exists():
                return p
        return None

    # per-value (mu, sd, bias-to-TCCON, per-footprint RMSE-to-TCCON) helper
    def _stat(vals, tmu, n_tc):
        v = np.asarray(vals, float)
        if not np.isfinite(v).any():
            return np.nan, np.nan, np.nan, np.nan
        mu, sd = float(np.nanmean(v)), float(np.nanstd(v))
        bias = (mu - tmu) if n_tc else np.nan
        rmse = float(np.sqrt(np.nanmean((v - tmu) ** 2))) if n_tc else np.nan
        return mu, sd, bias, rmse

    rows = []
    for c in cases:
        site = c['site'] or c['tccon'][:2]
        pq = find_plotdata(out_base, c['date'], site)   # combined_<date>_<site>/plot_data.parquet
        if pq is None:
            continue
        oco = pd.read_parquet(pq)
        # merge raw (pre-bias-correction) XCO2 from the source parquet — plot_data
        # only carries xco2_bc. Match on (time, lon, lat), which pass through unchanged.
        sp = source_parquet(c['date'])
        if sp is not None:
            src = (pd.read_parquet(sp, columns=['time', 'lon', 'lat', 'xco2_raw'])
                     .drop_duplicates(['time', 'lon', 'lat']))
            oco = oco.merge(src, on=['time', 'lon', 'lat'], how='left')
        # SHARED collocation (box → ≤radius → ±50 ppm sanity; guarded KEPT + flagged)
        col = collocate(oco, tccon_df(c['tccon']),
                        box=(c['lonmin'], c['lonmax'], c['latmin'], c['latmax']),
                        radius_km=args.radius_km, window_min=args.window_min)
        near = col['near']; tmu = col['tccon_ref']; tsd = col['tccon_sd']; n_tc = col['n_tccon']
        if not len(near):
            continue
        drop = near[~near['is_guarded']]                # drop-guards subset
        n_g = int(near['is_guarded'].sum())
        _raw = (lambda f: f['xco2_raw'] if 'xco2_raw' in f.columns
                else np.full(len(f), np.nan))
        # KEEP-guards series (headline: the end-to-end result)
        raw_mu, raw_sd, bias_raw, rmse_raw = _stat(_raw(near), tmu, n_tc)
        orig_mu, orig_sd, bias_before, rmse_before = _stat(near['xco2_bc'], tmu, n_tc)
        corr_mu, corr_sd, bias_after, rmse_after = _stat(near[CORR], tmu, n_tc)
        # DROP-guards series (correction quality where the model acted)
        _, _, bias_before_dg, rmse_before_dg = _stat(drop['xco2_bc'], tmu, n_tc)
        _, _, bias_after_dg, rmse_after_dg = _stat(drop[CORR], tmu, n_tc)
        rows.append(dict(
            site=site, date=c['date'], n_oco=len(near), n_guarded=n_g, n_tccon=n_tc,
            raw_mu=raw_mu, raw_sd=raw_sd,
            orig_mu=orig_mu, orig_sd=orig_sd,
            corr_mu=corr_mu, corr_sd=corr_sd,
            tccon_mu=tmu, tccon_sd=tsd,
            bias_raw=bias_raw, bias_before=bias_before, bias_after=bias_after,
            rmse_raw=rmse_raw, rmse_before=rmse_before, rmse_after=rmse_after,
            n_oco_dg=len(drop),
            bias_before_dg=bias_before_dg, bias_after_dg=bias_after_dg,
            rmse_before_dg=rmse_before_dg, rmse_after_dg=rmse_after_dg,
        ))

    if not rows:
        print(f"No cases matched (out-base={out_base}, script={args.script}). "
              "Nothing to report — check that each case's plot_data.parquet exists.",
              file=sys.stderr)
        return
    rep = pd.DataFrame(rows).sort_values(['site', 'date']).reset_index(drop=True)
    rep.to_csv(P_CSV, index=False)

    cmp = rep[rep['n_tccon'] > 0].copy()
    # ── markdown ───────────────────────────────────────────────────────────────
    def f(x, n=2): return '' if pd.isna(x) else f'{x:.{n}f}'
    lines = ['# OCO-2 corrected vs TCCON — combined comparison', '',
             f'{len(rep)} cases ({len(cmp)} with TCCON in ±{args.window_min:g} min, '
             f'≤{args.radius_km:g} km).  XCO2 in ppm (mean ± std).', '',
             '| site | date | n_oco | n_tccon | original | corrected | TCCON | bias before | bias after |',
             '|---|---|--:|--:|---|---|---|--:|--:|']
    for _, r in rep.iterrows():
        lines.append(f"| {r['site']} | {r['date']} | {r['n_oco']} | {r['n_tccon']} | "
                     f"{f(r['orig_mu'])}±{f(r['orig_sd'])} | {f(r['corr_mu'])}±{f(r['corr_sd'])} | "
                     f"{f(r['tccon_mu'])}±{f(r['tccon_sd'])} | {f(r['bias_before'],2)} | {f(r['bias_after'],2)} |")
    site_agg = pd.DataFrame()
    if len(cmp):
        bb, ba = cmp['bias_before'].to_numpy(), cmp['bias_after'].to_numpy()
        rb, ra = cmp['rmse_before'].to_numpy(), cmp['rmse_after'].to_numpy()
        ba_dg, ra_dg = cmp['bias_after_dg'].to_numpy(), cmp['rmse_after_dg'].to_numpy()
        n_g_tot = int(cmp['n_guarded'].sum())
        lines += ['', '## Aggregate (cases with TCCON)', '',
                  '_Headline KEEPS guarded footprints (correction skipped there → corrected = raw '
                  'xco2_bc): the end-to-end result. The drop-guards line below excludes them._', '',
                  f"- mean |bias|:  before **{np.nanmean(np.abs(bb)):.2f}** → after **{np.nanmean(np.abs(ba)):.2f}** ppm",
                  f"- RMS bias:    before **{np.sqrt(np.nanmean(bb**2)):.2f}** → after **{np.sqrt(np.nanmean(ba**2)):.2f}** ppm",
                  f"- mean OCO std: before **{cmp['orig_sd'].mean():.2f}** → after **{cmp['corr_sd'].mean():.2f}** ppm",
                  f"- mean per-footprint RMSE-to-TCCON: before **{np.nanmean(rb):.2f}** → after **{np.nanmean(ra):.2f}** ppm",
                  f"- improved (|bias| down) in **{int((np.abs(ba)<np.abs(bb)).sum())}/{len(cmp)}** cases",
                  f"- improved (per-footprint RMSE down) in **{int((ra<rb).sum())}/{len(cmp)}** cases",
                  f"- **drop-guards** ({n_g_tot} guarded footprints excluded): corrected mean |bias| "
                  f"**{np.nanmean(np.abs(ba_dg)):.2f}** ppm, per-footprint RMSE **{np.nanmean(ra_dg):.2f}** ppm"]

        # ── per-site aggregate ─────────────────────────────────────────────────
        c2 = cmp.copy()
        c2['abs_before'] = c2['bias_before'].abs(); c2['abs_after'] = c2['bias_after'].abs()
        c2['improved'] = c2['abs_after'] < c2['abs_before']
        c2['improved_rmse'] = c2['rmse_after'] < c2['rmse_before']
        site_agg = (c2.groupby('site')
                    .agg(n=('date', 'size'),
                         mean_abs_bias_before=('abs_before', 'mean'),
                         mean_abs_bias_after=('abs_after', 'mean'),
                         mean_rmse_before=('rmse_before', 'mean'),
                         mean_rmse_after=('rmse_after', 'mean'),
                         mean_sd_before=('orig_sd', 'mean'),
                         mean_sd_after=('corr_sd', 'mean'),
                         n_improved=('improved', 'sum'),
                         n_improved_rmse=('improved_rmse', 'sum'))
                    .reset_index().sort_values('mean_rmse_after'))
        site_agg.to_csv(P_SITE_CSV, index=False)
        lines += ['', '## Per-site aggregate (cases with TCCON, sorted by post-correction RMSE)', '',
                  '| site | n | mean \\|bias\\| before | after | mean RMSE before | after | mean σ before | after | \\|bias\\|↓ | RMSE↓ |',
                  '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
        for _, s in site_agg.iterrows():
            lines.append(f"| {s['site']} | {int(s['n'])} | {s['mean_abs_bias_before']:.2f} | "
                         f"**{s['mean_abs_bias_after']:.2f}** | {s['mean_rmse_before']:.2f} | "
                         f"**{s['mean_rmse_after']:.2f}** | {s['mean_sd_before']:.2f} | "
                         f"{s['mean_sd_after']:.2f} | {int(s['n_improved'])}/{int(s['n'])} | "
                         f"{int(s['n_improved_rmse'])}/{int(s['n'])} |")
    P_MD.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # ── figure ─────────────────────────────────────────────────────────────────
    if len(cmp):
        cmp = cmp.sort_values('bias_before').reset_index(drop=True)
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 7))
        # (a) scatter vs TCCON (1:1)
        if cmp['raw_mu'].notna().any():
            axA.errorbar(cmp['tccon_mu'], cmp['raw_mu'], xerr=cmp['tccon_sd'], yerr=cmp['raw_sd'],
                         fmt='o', ms=7, color='darkorange', alpha=0.7, elinewidth=0.8,
                         markeredgecolor='black', markeredgewidth=0.6, label='raw XCO₂')
        axA.errorbar(cmp['tccon_mu'], cmp['orig_mu'], xerr=cmp['tccon_sd'], yerr=cmp['orig_sd'],
                     fmt='o', ms=7, color='steelblue', alpha=0.7, elinewidth=0.8,
                     markeredgecolor='black', markeredgewidth=0.6, label='original XCO₂_bc')
        axA.errorbar(cmp['tccon_mu'], cmp['corr_mu'], xerr=cmp['tccon_sd'], yerr=cmp['corr_sd'],
                     fmt='o', ms=7, color='green', alpha=0.8, elinewidth=0.8,
                     markeredgecolor='black', markeredgewidth=0.6, label='corrected')
        lo = float(np.nanmin([cmp['tccon_mu'].min(), cmp['corr_mu'].min(),
                              cmp['orig_mu'].min(), cmp['raw_mu'].min()])) - 1
        hi = float(np.nanmax([cmp['tccon_mu'].max(), cmp['corr_mu'].max(),
                              cmp['orig_mu'].max(), cmp['raw_mu'].max()])) + 1
        axA.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')
        # ── OLS fits: OCO-2 (y) vs TCCON (x), raw / before / after correction ────
        _xline = np.array([lo, hi])
        _series = [('orig_mu', 'steelblue', 'before'), ('corr_mu', 'green', 'after')]
        if cmp['raw_mu'].notna().any():
            _series = [('raw_mu', 'darkorange', 'raw')] + _series

        def _rmse_to_tccon(col):
            _d = (cmp[col] - cmp['tccon_mu']).to_numpy(float)
            _d = _d[np.isfinite(_d)]
            return float(np.sqrt(np.mean(_d ** 2))) if _d.size else np.nan

        _fits = {}
        for _col, _color, _lbl in _series:
            _f = _ols_fit(cmp['tccon_mu'], cmp[_col])
            if _f is not None:
                _f['rmse'] = _rmse_to_tccon(_col)
            _fits[_lbl] = _f
            if _f is not None:
                axA.plot(_xline, _f['intercept'] + _f['slope'] * _xline,
                         '-', color=_color, lw=1.6, alpha=0.9, zorder=4)
        # annotation box with slope ± SE, R² and RMSE-to-TCCON for each series
        _txt = []
        for _col, _color, _lbl in _series:
            _f = _fits.get(_lbl)
            if _f is not None:
                _txt.append(f"{_lbl}:  slope = {_f['slope']:.3f} ± {_f['slope_se']:.3f}   "
                            f"R² = {_f['r2']:.3f}   RMSE = {_f['rmse']:.2f}")
        if _txt:
            axA.text(0.04, 0.96, '\n'.join(_txt), transform=axA.transAxes,
                     va='top', ha='left', fontsize=10,
                     bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
        axA.set_xlabel('TCCON XCO₂ (ppm)'); axA.set_ylabel('OCO-2 XCO₂ (ppm)')
        axA.set_title('OCO-2 vs TCCON (mean ± std)'); axA.legend(loc='lower right')
        axA.grid(alpha=0.3)
        axA.set_xlim(lo, hi); axA.set_ylim(lo, hi); axA.set_aspect('equal')
        # (b) bias raw→before→after dumbbell, per case (errorbar = OCO-2 σ)
        y = np.arange(len(cmp))
        has_raw = cmp['bias_raw'].notna().any()
        for yi in range(len(cmp)):
            pts = [cmp['bias_before'].iloc[yi], cmp['bias_after'].iloc[yi]]
            if has_raw and np.isfinite(cmp['bias_raw'].iloc[yi]):
                pts.append(cmp['bias_raw'].iloc[yi])
            axB.plot([min(pts), max(pts)], [yi, yi], '-', color='lightgray', lw=1.5, zorder=1)
        if has_raw:
            axB.errorbar(cmp['bias_raw'], y, xerr=cmp['raw_sd'], fmt='o', ms=7,
                         color='darkorange', ecolor='darkorange', elinewidth=0.8,
                         capsize=3, capthick=0.8, markeredgecolor='black',
                         markeredgewidth=0.6, label='raw', zorder=2)
        axB.errorbar(cmp['bias_before'], y, xerr=cmp['orig_sd'], fmt='o', ms=7,
                     color='steelblue', ecolor='steelblue', elinewidth=0.8,
                     capsize=3, capthick=0.8, markeredgecolor='black',
                     markeredgewidth=0.6, label='before', zorder=3)
        axB.errorbar(cmp['bias_after'], y, xerr=cmp['corr_sd'], fmt='o', ms=7,
                     color='green', ecolor='green', elinewidth=0.8,
                     capsize=3, capthick=0.8, markeredgecolor='black',
                     markeredgewidth=0.6, label='after', zorder=4)
        axB.axvline(0, color='k', lw=1)
        # annotation box: per-case bias (mean ± std) AND mean per-footprint
        # RMSE-to-TCCON for each series — RMSE captures scatter the |bias| misses.
        _btxt = []
        for _lbl, _col, _rcol in (('raw', 'bias_raw', 'rmse_raw'),
                                  ('before', 'bias_before', 'rmse_before'),
                                  ('after', 'bias_after', 'rmse_after')):
            if _col == 'bias_raw' and not has_raw:
                continue
            _b = cmp[_col].to_numpy(float); _b = _b[np.isfinite(_b)]
            _r = cmp[_rcol].to_numpy(float); _r = _r[np.isfinite(_r)]
            if _b.size:
                _rtxt = f"   RMSE {np.mean(_r):.2f}" if _r.size else ""
                _btxt.append(f"{_lbl}:  bias {np.mean(_b):+.2f} ± {np.std(_b):.2f}{_rtxt}")
        if _btxt:
            axB.text(0.04, 0.96, '\n'.join(_btxt), transform=axB.transAxes,
                     va='top', ha='left', fontsize=10,
                     bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
        axB.set_yticks(y); axB.set_yticklabels([f"{r.site} {r.date}" for r in cmp.itertuples()], fontsize=6)
        axB.set_xlabel('XCO₂ bias to TCCON (ppm)')
        axB.set_title('Bias to TCCON: raw → before → after' if has_raw
                      else 'Bias before → after correction')
        axB.legend(loc='lower right'); axB.grid(alpha=0.3, axis='x')
        fig.tight_layout()
        fig.savefig(P_PNG, dpi=200, bbox_inches='tight')
        plt.close(fig)

        # ── per-site bar chart: mean |bias| and σ, before vs after ─────────────
        if len(site_agg):
            sa = site_agg.sort_values('mean_abs_bias_before', ascending=False).reset_index(drop=True)
            x = np.arange(len(sa)); w = 0.38
            fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(max(10, 1.0 * len(sa)), 6))
            bx1.bar(x - w/2, sa['mean_abs_bias_before'], w, color='steelblue', label='before')
            bx1.bar(x + w/2, sa['mean_abs_bias_after'], w, color='green', label='after')
            bx1.set_xticks(x); bx1.set_xticklabels(
                [f"{r.site}\n({int(r.n)}, {int(r.n_improved)}↓)" for r in sa.itertuples()], fontsize=8)
            bx1.set_ylabel('mean |bias to TCCON| (ppm)')
            bx1.set_title('Per-site mean |bias| before → after'); bx1.legend(); bx1.grid(alpha=0.3, axis='y')
            bx2.bar(x - w/2, sa['mean_sd_before'], w, color='steelblue', label='before')
            bx2.bar(x + w/2, sa['mean_sd_after'], w, color='green', label='after')
            bx2.set_xticks(x); bx2.set_xticklabels([r.site for r in sa.itertuples()], fontsize=8)
            bx2.set_ylabel('mean OCO-2 σ (ppm)')
            bx2.set_title('Per-site mean scatter (σ) before → after'); bx2.legend(); bx2.grid(alpha=0.3, axis='y')
            fig2.tight_layout()
            fig2.savefig(P_SITE_PNG, dpi=200, bbox_inches='tight')
            plt.close(fig2)
        print(f"\n[saved] {P_CSV}\n[saved] {P_MD}"
              f"\n[saved] {P_PNG}\n[saved] {P_SITE_CSV}"
              f"\n[saved] {P_SITE_PNG}")


if __name__ == '__main__':
    main()

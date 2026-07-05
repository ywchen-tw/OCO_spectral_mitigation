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
from ak_harmonize import (find_lite_file, ak_adjusted_ref,
                          operator_from_dataframe, ak_adjusted_ref_from_operator)

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


def _significance(cmp, n_boot=10000, seed=20260703):
    """Paired significance of the correction across station-days (M3).

    Two complementary tests on the per-case (station-day) metrics:
      * paired Wilcoxon signed-rank on |bias| and on per-footprint RMSE
        (treats station-days as exchangeable pairs), and
      * a SITE-CLUSTERED bootstrap (sites resampled with replacement, each
        bringing all its station-days) for the after−before deltas of
        mean |bias|, RMS bias, and mean RMSE — this respects the strong
        within-site clustering (e.g. Réunion 14 days) that the Wilcoxon
        ignores.  p_boot = 2·min(P(Δ≥0), P(Δ≤0)), floored at 2/(B+1).

    Returns a flat dict of test statistics (NaN where not computable).
    """
    out = {}
    bb = cmp['bias_before'].to_numpy(float); ba = cmp['bias_after'].to_numpy(float)
    rb = cmp['rmse_before'].to_numpy(float); ra = cmp['rmse_after'].to_numpy(float)
    sites = cmp['site'].to_numpy()

    def _wilcoxon(x_after, x_before):
        m = np.isfinite(x_after) & np.isfinite(x_before)
        if m.sum() < 6:
            return np.nan, int(m.sum())
        try:
            from scipy.stats import wilcoxon
            res = wilcoxon(x_after[m], x_before[m])
            return float(res.pvalue), int(m.sum())
        except (ImportError, ValueError):
            return np.nan, int(m.sum())

    out['wilcoxon_absbias_p'], out['wilcoxon_absbias_n'] = _wilcoxon(np.abs(ba), np.abs(bb))
    out['wilcoxon_rmse_p'], out['wilcoxon_rmse_n'] = _wilcoxon(ra, rb)

    # site-clustered bootstrap on after−before deltas
    uniq = np.unique(sites)
    out['n_sites'] = int(uniq.size)
    if uniq.size >= 3:
        rng = np.random.default_rng(seed)
        idx_by_site = {s: np.where(sites == s)[0] for s in uniq}
        deltas = {'d_mean_absbias': [], 'd_rms_bias': [], 'd_mean_rmse': []}
        for _ in range(n_boot):
            pick = rng.choice(uniq, uniq.size, replace=True)
            idx = np.concatenate([idx_by_site[s] for s in pick])
            b0, b1, r0, r1 = bb[idx], ba[idx], rb[idx], ra[idx]
            deltas['d_mean_absbias'].append(np.nanmean(np.abs(b1)) - np.nanmean(np.abs(b0)))
            deltas['d_rms_bias'].append(np.sqrt(np.nanmean(b1 ** 2)) - np.sqrt(np.nanmean(b0 ** 2)))
            deltas['d_mean_rmse'].append(np.nanmean(r1) - np.nanmean(r0))
        p_floor = 2.0 / (n_boot + 1)
        for key, vals in deltas.items():
            v = np.asarray(vals); v = v[np.isfinite(v)]
            out[f'{key}_mean'] = float(v.mean())
            out[f'{key}_ci_lo'] = float(np.percentile(v, 2.5))
            out[f'{key}_ci_hi'] = float(np.percentile(v, 97.5))
            out[f'{key}_p'] = float(max(p_floor, 2 * min((v >= 0).mean(), (v <= 0).mean())))
    return out


def _sig_lines(sig, label):
    """Markdown lines for one _significance() result."""
    def fp(p):
        return '' if not np.isfinite(p) else (f'p = {p:.4f}' if p >= 1e-4 else 'p < 1e-4')
    lines = ['', f'### Significance ({label})', '',
             f"- paired Wilcoxon signed-rank, station-day |bias| after vs before "
             f"(n={sig.get('wilcoxon_absbias_n', 0)}): **{fp(sig.get('wilcoxon_absbias_p', np.nan))}**",
             f"- paired Wilcoxon signed-rank, per-footprint RMSE after vs before "
             f"(n={sig.get('wilcoxon_rmse_n', 0)}): **{fp(sig.get('wilcoxon_rmse_p', np.nan))}**"]
    if 'd_mean_absbias_mean' in sig:
        lines += [f"- site-clustered bootstrap ({sig['n_sites']} sites, 95% CI of after−before):",
                  f"  - Δ mean |bias| = {sig['d_mean_absbias_mean']:+.2f} "
                  f"[{sig['d_mean_absbias_ci_lo']:+.2f}, {sig['d_mean_absbias_ci_hi']:+.2f}] ppm, "
                  f"**{fp(sig['d_mean_absbias_p'])}**",
                  f"  - Δ RMS bias = {sig['d_rms_bias_mean']:+.2f} "
                  f"[{sig['d_rms_bias_ci_lo']:+.2f}, {sig['d_rms_bias_ci_hi']:+.2f}] ppm, "
                  f"**{fp(sig['d_rms_bias_p'])}**",
                  f"  - Δ mean footprint RMSE = {sig['d_mean_rmse_mean']:+.2f} "
                  f"[{sig['d_mean_rmse_ci_lo']:+.2f}, {sig['d_mean_rmse_ci_hi']:+.2f}] ppm, "
                  f"**{fp(sig['d_mean_rmse_p'])}**"]
    return lines


def _draw_pair(axA, axB, cmp, title_prefix=''):
    """Draw the shared 2-panel view for a per-case subset ``cmp``: (axA) OCO-2 mean
    vs TCCON mean (1:1 + OLS fits with slope/R²/RMSE), (axB) per-case bias-to-TCCON
    dumbbell raw→before→after with mean±std + RMSE.  Used by the all-cases, excl-sites
    and by-surface figures so they share one legend/text style."""
    cmp = cmp.sort_values('bias_before').reset_index(drop=True)
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
    # OLS fits: OCO-2 (y) vs TCCON (x), raw / before / after correction
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
    _txt = []
    for _col, _color, _lbl in _series:
        _f = _fits.get(_lbl)
        if _f is not None:
            _txt.append(f"{_lbl}:  slope = {_f['slope']:.3f} ± {_f['slope_se']:.3f}   "
                        f"R² = {_f['r2']:.3f}   RMSE(station) = {_f['rmse']:.2f}")
    if _txt:
        axA.text(0.04, 0.96, '\n'.join(_txt), transform=axA.transAxes,
                 va='top', ha='left', fontsize=10,
                 bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
    axA.set_xlabel('TCCON XCO₂ (ppm)'); axA.set_ylabel('OCO-2 XCO₂ (ppm)')
    axA.set_title(f'{title_prefix}OCO-2 vs TCCON — station-day means')
    axA.legend(loc='lower right', title='station-day mean (± footprint σ)')
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
    _btxt = []
    for _lbl, _col, _rcol in (('raw', 'bias_raw', 'rmse_raw'),
                              ('before', 'bias_before', 'rmse_before'),
                              ('after', 'bias_after', 'rmse_after')):
        if _col == 'bias_raw' and not has_raw:
            continue
        _b = cmp[_col].to_numpy(float); _b = _b[np.isfinite(_b)]
        _r = cmp[_rcol].to_numpy(float); _r = _r[np.isfinite(_r)]
        if _b.size:
            _rtxt = f"   footprint RMSE {np.mean(_r):.2f}" if _r.size else ""
            _btxt.append(f"{_lbl}:  station bias {np.mean(_b):+.2f} ± {np.std(_b):.2f}{_rtxt}")
    if _btxt:
        axB.text(0.04, 0.96, '\n'.join(_btxt), transform=axB.transAxes,
                 va='top', ha='left', fontsize=10,
                 bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
    axB.set_yticks(y); axB.set_yticklabels([f"{r.site} {r.date}" for r in cmp.itertuples()], fontsize=6)
    axB.set_xlabel('XCO₂ bias to TCCON (ppm)')
    axB.set_title((f'{title_prefix}Bias to TCCON: raw → before → after' if has_raw
                   else f'{title_prefix}Bias before → after correction')
                  + '   (marker = station-day mean, error bar = footprint σ)')
    axB.legend(loc='lower right', title='per station-day'); axB.grid(alpha=0.3, axis='x')


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
    ap.add_argument('--exclude-sites', default='',
                    help="Comma-separated TCCON site codes (e.g. 'ny'); also emits an "
                         "_excl_<sites> variant of the scatter/dumbbell and by-surface "
                         "figures with those stations dropped.")
    ap.add_argument('--ak-harmonize', action='store_true',
                    help='AK/prior-harmonize the TCCON reference (Rodgers & Connor 2003 / '
                         'Wunch et al. 2017) using the day OCO-2 Lite file; cases whose '
                         'Lite file is not found fall back to the raw TCCON window mean '
                         '(ak_delta = NaN in the CSV). Shifts absolute biases only — '
                         'before/after improvement metrics are invariant.')
    ap.add_argument('--n-boot', type=int, default=10000,
                    help='Site-clustered bootstrap replicates for the significance block.')
    args = ap.parse_args()

    out_base = Path(args.out_base)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sfx = args.fname_suffix
    excl = [s.strip() for s in args.exclude_sites.split(',') if s.strip()]
    # Output paths (suffix inserted before the extension).
    P_CSV      = out_dir / f'tccon_comparison{sfx}.csv'
    P_MD       = out_dir / f'tccon_comparison{sfx}.md'
    P_PNG      = out_dir / f'tccon_comparison{sfx}.png'
    P_SITE_CSV = out_dir / f'tccon_comparison_by_site{sfx}.csv'
    P_SIG_CSV  = out_dir / f'tccon_significance{sfx}.csv'
    P_SITE_PNG = out_dir / f'tccon_comparison_by_site{sfx}.png'
    P_BYSURF   = out_dir / f'tccon_comparison_by_surface{sfx}.png'
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

    def tccon_path(name):
        p = Path('data/TCCON') / name
        return p if p.exists() else storage / 'data/TCCON' / name

    _tccon_cache = {}
    def tccon_df(name):
        if name not in _tccon_cache:
            _tccon_cache[name] = load_tccon(str(tccon_path(name)))
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

    # KEEP-guards series (headline) + DROP-guards series (correction quality) for one
    # per-case footprint frame (a whole case, or one surface within it).
    def _case_metrics(frame, tmu, n_tc):
        raw = frame['xco2_raw'] if 'xco2_raw' in frame.columns else np.full(len(frame), np.nan)
        drop = frame[~frame['is_guarded']] if 'is_guarded' in frame.columns else frame
        raw_mu, raw_sd, bias_raw, rmse_raw = _stat(raw, tmu, n_tc)
        orig_mu, orig_sd, bias_before, rmse_before = _stat(frame['xco2_bc'], tmu, n_tc)
        corr_mu, corr_sd, bias_after, rmse_after = _stat(frame[CORR], tmu, n_tc)
        _, _, bias_before_dg, rmse_before_dg = _stat(drop['xco2_bc'], tmu, n_tc)
        _, _, bias_after_dg, rmse_after_dg = _stat(drop[CORR], tmu, n_tc)
        return dict(
            n_oco=len(frame),
            n_guarded=int(frame['is_guarded'].sum()) if 'is_guarded' in frame.columns else 0,
            raw_mu=raw_mu, raw_sd=raw_sd, orig_mu=orig_mu, orig_sd=orig_sd,
            corr_mu=corr_mu, corr_sd=corr_sd,
            bias_raw=bias_raw, bias_before=bias_before, bias_after=bias_after,
            rmse_raw=rmse_raw, rmse_before=rmse_before, rmse_after=rmse_after,
            n_oco_dg=len(drop), bias_before_dg=bias_before_dg, bias_after_dg=bias_after_dg,
            rmse_before_dg=rmse_before_dg, rmse_after_dg=rmse_after_dg)

    rows = []
    for c in cases:
        site = c['site'] or c['tccon'][:2]
        pq = find_plotdata(out_base, c['date'], site)   # combined_<date>_<site>/plot_data.parquet
        if pq is None:
            continue
        oco = pd.read_parquet(pq)
        # merge raw (pre-bias-correction) XCO2 from the source parquet — plot_data
        # only carries xco2_bc. Match on (time, lon, lat), which pass through
        # unchanged.  Also pull the flattened Lite column-operator columns
        # (ak_NN/pwf_NN/co2_ap_NN/plev_NN, dual-fit-era parquets) so AK
        # harmonization can run without reopening the Lite files.
        sp = source_parquet(c['date'])
        if sp is not None:
            import pyarrow.parquet as _pq
            avail = set(_pq.ParquetFile(sp).schema_arrow.names)
            ak_cols = sorted(
                col for col in avail
                if any(col.startswith(p) and col[len(p):].isdigit()
                       for p in ('ak_', 'pwf_', 'co2_ap_', 'plev_')))
            # Only pull columns the plot_data doesn't already carry (newer
            # build_deepens_plot_data.py keeps xco2_raw itself); merging a column
            # already present would collide into xco2_raw_x/_y and blank the raw series.
            extra = [c for c in ['xco2_raw'] if c in avail and c not in oco.columns]
            want = ['time', 'lon', 'lat'] + extra + ak_cols
            src = (pd.read_parquet(sp, columns=want)
                     .drop_duplicates(['time', 'lon', 'lat']))
            oco = oco.merge(src, on=['time', 'lon', 'lat'], how='left')
        # SHARED collocation (box → ≤radius → ±50 ppm sanity; guarded KEPT + flagged)
        col = collocate(oco, tccon_df(c['tccon']),
                        box=(c['lonmin'], c['lonmax'], c['latmin'], c['latmax']),
                        radius_km=args.radius_km, window_min=args.window_min)
        near = col['near']; tmu = col['tccon_ref']; tsd = col['tccon_sd']; n_tc = col['n_tccon']
        if not len(near):
            continue
        # AK/prior harmonization (M2): replace the raw TCCON window mean with the
        # Rodgers & Connor-adjusted reference.  Preferred operator source is the
        # collocated footprints' own ak_NN/pwf_NN/co2_ap_NN/plev_NN parquet
        # columns; falls back to the day's Lite file when those are absent.
        tmu_raw, ak_delta, ak_n_lite, ak_source = tmu, np.nan, 0, ''
        if args.ak_harmonize and n_tc and np.isfinite(tmu):
            ot = pd.to_datetime(near['time'], unit='s', utc=True, errors='coerce').dropna()
            adj = None
            if len(ot):
                tpath = str(tccon_path(c['tccon']))
                t0 = ot.min().tz_localize(None); t1 = ot.max().tz_localize(None)
                try:
                    op = operator_from_dataframe(near)
                    if op is not None:
                        adj = ak_adjusted_ref_from_operator(op, tpath, t0, t1,
                                                            window_min=args.window_min)
                        if adj is not None:
                            ak_source = 'parquet'
                    if adj is None:
                        lite = find_lite_file(c['date'], roots=['.', storage])
                        if lite is not None:
                            adj = ak_adjusted_ref(
                                lite, tpath, col['st_lon'], col['st_lat'],
                                args.radius_km, t0, t1, window_min=args.window_min)
                            if adj is not None:
                                ak_source = 'lite'
                except Exception as e:                       # noqa: BLE001 — per-case fallback
                    print(f"[ak-harmonize] {site} {c['date']}: {e}", file=sys.stderr)
                    adj = None
            if adj is not None:
                tmu = adj['tccon_ref_ak']
                ak_delta = adj['ak_delta']
                ak_n_lite = adj['n_lite']
        # one row per surface: all (pooled) + ocean (sfc0) + land (sfc1)
        for sname, frame in (('all', near),
                             ('ocean', near[near['sfc_type'] == 0]),
                             ('land',  near[near['sfc_type'] == 1])):
            if not len(frame):
                continue
            row = dict(site=site, date=c['date'], surface=sname, n_tccon=n_tc,
                       tccon_mu=tmu, tccon_sd=tsd, tccon_mu_raw=tmu_raw,
                       ak_delta=ak_delta, ak_n_lite=ak_n_lite, ak_source=ak_source,
                       **_case_metrics(frame, tmu, n_tc))
            # When AK harmonization is active the PRIMARY metrics above use the AK
            # reference; also compute the direct (raw window-mean) reference metrics
            # so ONE run emits both comparisons (…_direct columns + overlay figure).
            # RMSE-to-TCCON is non-linear in the reference, so it must be recomputed
            # (the bias terms alone would shift by ak_delta, but RMSE would not).
            if args.ak_harmonize and np.isfinite(tmu_raw) and tmu_raw != tmu:
                md = _case_metrics(frame, tmu_raw, n_tc)
                row.update({f'{k}_direct': md[k] for k in (
                    'bias_raw', 'bias_before', 'bias_after',
                    'rmse_raw', 'rmse_before', 'rmse_after',
                    'raw_sd', 'orig_sd', 'corr_sd')})
            rows.append(row)

    if not rows:
        print(f"No cases matched (out-base={out_base}, script={args.script}). "
              "Nothing to report — check that each case's plot_data.parquet exists.",
              file=sys.stderr)
        return
    rep = pd.DataFrame(rows).sort_values(['surface', 'site', 'date']).reset_index(drop=True)
    rep.to_csv(P_CSV, index=False)

    rep_all = rep[rep['surface'] == 'all']            # pooled-surface per-case rows
    cmp = rep_all[rep_all['n_tccon'] > 0].copy()
    # ── markdown ───────────────────────────────────────────────────────────────
    def f(x, n=2): return '' if pd.isna(x) else f'{x:.{n}f}'
    lines = ['# OCO-2 corrected vs TCCON — combined comparison', '',
             f'{len(rep_all)} cases ({len(cmp)} with TCCON in ±{args.window_min:g} min, '
             f'≤{args.radius_km:g} km).  XCO2 in ppm (mean ± std).', '',
             '| site | date | n_oco | n_tccon | original | corrected | TCCON | bias before | bias after |',
             '|---|---|--:|--:|---|---|---|--:|--:|']
    for _, r in rep_all.iterrows():
        lines.append(f"| {r['site']} | {r['date']} | {r['n_oco']} | {r['n_tccon']} | "
                     f"{f(r['orig_mu'])}±{f(r['orig_sd'])} | {f(r['corr_mu'])}±{f(r['corr_sd'])} | "
                     f"{f(r['tccon_mu'])}±{f(r['tccon_sd'])} | {f(r['bias_before'],2)} | {f(r['bias_after'],2)} |")
    site_agg = pd.DataFrame()
    if len(cmp):
        bb, ba = cmp['bias_before'].to_numpy(), cmp['bias_after'].to_numpy()
        rb, ra = cmp['rmse_before'].to_numpy(), cmp['rmse_after'].to_numpy()
        ba_dg, ra_dg = cmp['bias_after_dg'].to_numpy(), cmp['rmse_after_dg'].to_numpy()
        # raw (pre-bias-correction xco2_raw) series — shown when available so the
        # headline reads raw → before(xco2_bc) → after(ML), not just before → after.
        braw = cmp['bias_raw'].to_numpy(float); rraw = cmp['rmse_raw'].to_numpy(float)
        has_raw = np.isfinite(braw).any()
        _r_absbias = f"raw **{np.nanmean(np.abs(braw)):.2f}** → " if has_raw else ''
        _r_rmsbias = f"raw **{np.sqrt(np.nanmean(braw**2)):.2f}** → " if has_raw else ''
        _r_ocostd  = f"raw **{cmp['raw_sd'].mean():.2f}** → " if has_raw else ''
        _r_fprmse  = f"raw **{np.nanmean(rraw):.2f}** → " if has_raw else ''
        n_g_tot = int(cmp['n_guarded'].sum())
        lines += ['', '## Aggregate (cases with TCCON)', '',
                  '_Headline KEEPS guarded footprints (correction skipped there → corrected = raw '
                  'xco2_bc): the end-to-end result. The drop-guards line below excludes them._'
                  + ('' if not has_raw else '  raw = pre-bias-correction xco2_raw; '
                     'before = xco2_bc (operational bias correction); after = ML-corrected.'), '',
                  f"- mean |bias|:  {_r_absbias}before **{np.nanmean(np.abs(bb)):.2f}** → after **{np.nanmean(np.abs(ba)):.2f}** ppm",
                  f"- RMS bias:    {_r_rmsbias}before **{np.sqrt(np.nanmean(bb**2)):.2f}** → after **{np.sqrt(np.nanmean(ba**2)):.2f}** ppm",
                  f"- mean OCO std: {_r_ocostd}before **{cmp['orig_sd'].mean():.2f}** → after **{cmp['corr_sd'].mean():.2f}** ppm",
                  f"- mean per-footprint RMSE-to-TCCON: {_r_fprmse}before **{np.nanmean(rb):.2f}** → after **{np.nanmean(ra):.2f}** ppm",
                  f"- improved (|bias| down) in **{int((np.abs(ba)<np.abs(bb)).sum())}/{len(cmp)}** cases",
                  f"- improved (per-footprint RMSE down) in **{int((ra<rb).sum())}/{len(cmp)}** cases",
                  f"- **drop-guards** ({n_g_tot} guarded footprints excluded): corrected mean |bias| "
                  f"**{np.nanmean(np.abs(ba_dg)):.2f}** ppm, per-footprint RMSE **{np.nanmean(ra_dg):.2f}** ppm"]

        if args.ak_harmonize:
            ak = cmp['ak_delta'].to_numpy(float)
            n_h = int(np.isfinite(ak).sum())
            if n_h:
                _src = cmp['ak_source'].astype(str)
                lines += [f"- **AK/prior harmonization** (Rodgers & Connor 2003 / Wunch et al. 2017): "
                          f"**{n_h}/{len(cmp)}** cases harmonized "
                          f"({int((_src == 'parquet').sum())} from parquet AK columns, "
                          f"{int((_src == 'lite').sum())} from Lite files); TCCON reference shift "
                          f"Δ = {np.nanmean(ak):+.2f} ± {np.nanstd(ak):.2f} ppm "
                          f"(un-harmonized cases keep the raw window mean; the shift moves the "
                          f"reference, so the SIGNED per-case after−before delta is invariant, but "
                          f"the |bias|/RMSE headline below is not — see the direct-vs-AK table)"]
                # Direct (raw window-mean) vs AK-harmonized headline, from the SAME
                # footprints — the two references computed in one run.
                if 'bias_before_direct' in cmp.columns:
                    def _refagg(brw, bb, ba, rrw, rb, ra):
                        """mean |bias| and mean fp-RMSE strings, raw→before→after."""
                        brw, bb, ba = (x.to_numpy(float) for x in (brw, bb, ba))
                        rrw, rb, ra = (x.to_numpy(float) for x in (rrw, rb, ra))
                        _rb = f"{np.nanmean(np.abs(brw)):.2f} → " if has_raw else ''
                        _rr = f"{np.nanmean(rrw):.2f} → " if has_raw else ''
                        return (f"{_rb}{np.nanmean(np.abs(bb)):.2f} → **{np.nanmean(np.abs(ba)):.2f}**",
                                f"{_rr}{np.nanmean(rb):.2f} → **{np.nanmean(ra):.2f}**")
                    _d = _refagg(cmp['bias_raw_direct'], cmp['bias_before_direct'], cmp['bias_after_direct'],
                                 cmp['rmse_raw_direct'], cmp['rmse_before_direct'], cmp['rmse_after_direct'])
                    _a = _refagg(cmp['bias_raw'], cmp['bias_before'], cmp['bias_after'],
                                 cmp['rmse_raw'], cmp['rmse_before'], cmp['rmse_after'])
                    _seq = 'raw→before→after' if has_raw else 'before→after'
                    lines += ['', '### Reference comparison: direct window-mean vs AK-harmonized',
                              '_Same footprints, two TCCON references (both computed this run)._', '',
                              f'| reference | mean \\|bias\\| {_seq} | mean fp-RMSE {_seq} |',
                              '|---|--:|--:|',
                              f"| direct | {_d[0]} | {_d[1]} |",
                              f"| AK-harmonized | {_a[0]} | {_a[1]} |"]
            else:
                lines += ["- **AK/prior harmonization requested but no case could be harmonized** "
                          "(no parquet AK columns and no day Lite files found — rebuild the "
                          "per-date parquets with the dual-fit fitting.py or set $OCO2_LITE_DIR); "
                          "raw TCCON means used"]

        # ── significance: paired Wilcoxon + site-clustered bootstrap (M3) ─────
        sig = _significance(cmp, n_boot=args.n_boot)
        lines += _sig_lines(sig, 'all sites')
        sig_rows = [dict(subset='all', **sig)]
        if excl:
            _sub = cmp[~cmp['site'].isin(excl)]
            if len(_sub) >= 6:
                sig_e = _significance(_sub, n_boot=args.n_boot)
                lines += _sig_lines(sig_e, f'excl {",".join(excl)}')
                sig_rows.append(dict(subset=f'excl_{"_".join(excl)}', **sig_e))
        pd.DataFrame(sig_rows).to_csv(P_SIG_CSV, index=False)

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

    # ── figures (scatter + per-case dumbbell) — ONE shared style for every view ──
    def _fig_pair(sub, png, title_prefix=''):
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 7))
        _draw_pair(axA, axB, sub, title_prefix=title_prefix)
        fig.tight_layout(); fig.savefig(png, dpi=200, bbox_inches='tight'); plt.close(fig)

    def _fig_by_surface(sites_excl, png, ttl_excl=''):
        rows_ok = lambda s: rep[(rep['surface'] == s) & (rep['n_tccon'] > 0)
                                & (~rep['site'].isin(sites_excl))]
        present = [s for s in ('ocean', 'land') if len(rows_ok(s))]
        if not present:
            return None
        fig, axes = plt.subplots(len(present), 2, figsize=(16, 7 * len(present)), squeeze=False)
        for i, s in enumerate(present):
            _draw_pair(axes[i, 0], axes[i, 1], rows_ok(s), title_prefix=f'{s.upper()}{ttl_excl}: ')
        fig.tight_layout(); fig.savefig(png, dpi=200, bbox_inches='tight'); plt.close(fig)
        return png

    def _fig_ak_vs_direct(sub, png):
        """Direct vs AK-harmonized overlay (only when --ak-harmonize produced the
        …_direct columns): (left) per-case corrected-bias direct↔AK dumbbell,
        (right) the AK/prior reference shift Δ = harmonized − raw TCCON per case."""
        s = sub[sub['bias_after_direct'].notna() & sub['ak_delta'].notna()].copy()
        if not len(s):
            return None
        s['lab'] = s['site'] + ' ' + s['date']
        s1 = s.sort_values('bias_after_direct').reset_index(drop=True)
        y = np.arange(len(s1))
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, max(7, 0.30 * len(s1))))
        for i in y:
            axL.plot([s1['bias_after_direct'][i], s1['bias_after'][i]], [i, i],
                     '-', color='lightgray', lw=1.2, zorder=0)
        axL.plot(s1['bias_after_direct'], y, 'o', color='green', ms=6, label='direct')
        axL.plot(s1['bias_after'], y, 's', color='purple', ms=6, label='AK-harmonized')
        axL.axvline(0, color='k', lw=1)
        axL.set_yticks(y); axL.set_yticklabels(s1['lab'], fontsize=6)
        axL.set_xlabel('corrected XCO₂ bias to TCCON (ppm)')
        axL.set_title('Per-case corrected bias: direct vs AK-harmonized reference')
        axL.legend(loc='lower right'); axL.grid(alpha=0.3, axis='x')
        s2 = s.sort_values('ak_delta').reset_index(drop=True); y2 = np.arange(len(s2))
        axR.barh(y2, s2['ak_delta'], color='teal', alpha=0.85)
        axR.axvline(0, color='k', lw=1)
        axR.set_yticks(y2); axR.set_yticklabels(s2['site'] + ' ' + s2['date'], fontsize=6)
        axR.set_xlabel('AK reference shift Δ = harmonized − raw TCCON (ppm)')
        axR.set_title('AK/prior reference shift per case'); axR.grid(alpha=0.3, axis='x')
        fig.tight_layout(); fig.savefig(png, dpi=200, bbox_inches='tight'); plt.close(fig)
        return png

    extra_saved = []
    if len(cmp):
        _fig_pair(cmp, P_PNG)                                        # all cases
        p = _fig_by_surface([], P_BYSURF)
        if p: extra_saved.append(p)
        if args.ak_harmonize and 'bias_after_direct' in cmp.columns:  # direct-vs-AK overlay
            p = _fig_ak_vs_direct(cmp, out_dir / f'tccon_comparison_ak_vs_direct{sfx}.png')
            if p: extra_saved.append(p)
        if excl:                                                     # excl-sites variants
            _sub = cmp[~cmp['site'].isin(excl)]
            if len(_sub):
                _pe = out_dir / f'tccon_comparison_excl_{"_".join(excl)}{sfx}.png'
                _fig_pair(_sub, _pe, title_prefix=f'(excl {",".join(excl)}) ')
                extra_saved.append(_pe)
            _pe = out_dir / f'tccon_comparison_by_surface_excl_{"_".join(excl)}{sfx}.png'
            p = _fig_by_surface(excl, _pe, ttl_excl=f' (excl {",".join(excl)})')
            if p: extra_saved.append(p)

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
        for _p in extra_saved:
            print(f"[saved] {_p}")


if __name__ == '__main__':
    main()

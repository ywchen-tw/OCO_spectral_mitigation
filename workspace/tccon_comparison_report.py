"""tccon_comparison_report.py — combined before/after-vs-TCCON summary across cases.

Reads the active `run_case` lines from a deep-ensemble plotting script, and for each
case reproduces the figure's comparison logic:
    drop guarded footprints → lon/lat box → footprints ≤ --radius-km of the TCCON
    station → TCCON within ± --window-min of the OCO-2 pass on that date.
Then computes mean ± std of the ORIGINAL (xco2_bc), CORRECTED, and TCCON XCO2, and
the bias to TCCON before/after correction.

Outputs (to --output-dir).  Every figure is a SINGLE panel (paper-ready), named
symmetrically for both TCCON references: ref = 'ak' (harmonized) and 'direct' (raw
window mean).  The 'ak' set only appears under --ak-harmonize; 'direct' always does.
    tccon_comparison.csv          — one row per case (headline)
    tccon_comparison.md           — markdown tables + aggregate summary + metrics table
    tccon_metrics_{ref}.csv       — comprehensive per-(surface × cloud-group) metrics
    tccon_{ref}_scatter.png       — corrected/original/raw vs TCCON scatter (1:1 + OLS)
    tccon_{ref}_bias.png          — bias-to-TCCON panel (style = --bias-style)
    tccon_{ref}_by_surface_*.png  — ocean/land stacked, (a)/(b) panels
    tccon_{ref}_by_cld_*.png      — one panel per nearest-cloud-distance bin
    tccon_{ref}_by_surface_by_cld_bias.png,  tccon_by_site_bias.png
    tccon_{ref0}_bias_<style>.png — 4 low-DPI bias-style variants for picking (item 3)
    tccon_ak_vs_direct_bias.png, tccon_ak_shift.png  — reference comparison (AK runs)

Example:
    PYTHONPATH=src python workspace/tccon_comparison_report.py \
        --script curc_shell_blanca_plot_corr_xco2_deepens.sh \
        --output-dir results/model_comparison/deep_ensemble
"""
import argparse
import re
import sys
from pathlib import Path

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paper-figure house style (matches src/models/make_deep_ensemble_figure.py) ──
# Manuscript-friendly serif; every figure below is a SINGLE panel (item 1), with
# no descriptive titles (item 4) — multi-panel stacks get bold "(a)/(b)…" tags via
# _panel_label().  NOTE: iteration uses a low --dpi; render final figures at 300.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
})

# Bias-panel style variants (item 3); --bias-style picks the production one and the
# 4-file low-dpi test set is always emitted for the headline all-cases view.
BIAS_STYLES = ('scatter_clddist', 'dumbbell_clddist', 'dumbbell_nolabel', 'dumbbell_label')

XCO2 = r'$X_{\mathrm{CO}_2}$'   # manuscript XCO₂ rendering used in all figure text (item 5)

sys.path.insert(0, str(Path(__file__).parent))
from plot_corrected_xco2 import load_tccon, get_storage_dir
from tccon_collocate import collocate, find_plotdata
from ak_harmonize import (find_lite_file, ak_adjusted_ref,
                          operator_from_dataframe, ak_adjusted_ref_from_operator)
from tccon_uncertainty_stats import (load_inflation, side_a_case, side_b_case,
                                     compare_case, markdown_block, DELTA_PPM)
try:
    from constants import AQUA_FREE_DRIFT_YEAR
except Exception:                       # noqa: BLE001 — run without PYTHONPATH=src
    AQUA_FREE_DRIFT_YEAR = 2022

CORR = 'deep_ensemble_corrected_xco2'   # overridable with --corr-col (e.g. tabm_corrected_xco2)


def _parse_cld_edges(spec):
    """'0,10,inf' → [(label, lo, hi), …] right-open nearest-cloud-distance bins (km)."""
    vals = []
    for tok in spec.split(','):
        tok = tok.strip()
        if tok:
            vals.append(np.inf if tok.lower() in ('inf', 'np.inf') else float(tok))
    bins = []
    for lo, hi in zip(vals[:-1], vals[1:]):
        lab = f"≥{lo:g} km" if np.isinf(hi) else f"{lo:g}–{hi:g} km"
        bins.append((lab, lo, hi))
    return bins


def _safe_nanmean(a):
    """np.nanmean without the all-NaN RuntimeWarning (returns NaN then)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def _safe_nanstd(a):
    """np.nanstd, NaN when no finite values (no RuntimeWarning)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(a.std()) if a.size else np.nan


def _z_to_p(z):
    """Two-sided normal p-value for a z-score (erfc-based, no scipy)."""
    return np.nan if not np.isfinite(z) else float(math.erfc(abs(z) / math.sqrt(2.0)))


def _panel_label(ax, tag):
    """Bold panel tag (e.g. '(a)' or '(a) ocean') at the top-left, outside the axes.
    Paper convention lifted from src/models/make_deep_ensemble_figure.py; no title.
    No-op when ``tag`` is falsy (single-panel figures need no letter)."""
    if tag:
        ax.text(0.0, 1.02, tag, transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='bottom', ha='left')


def _mse_aggregate(g, series=('raw', 'before', 'after')):
    """Error aggregates over a set of per-case rows ``g`` (needs n_oco and, per
    series tag, bias_<tag>/rmse_<tag>/mae_<tag>).  For each tag in ``series``:
      * pooled_mse    — footprint-weighted mean of (XCO₂−TCCON)², Σ n·RMSE² / Σ n
                        (the 'absolute' overall squared error; big cases dominate),
      * mean_case_mse — mean over station-days of the per-case MSE (= RMSE²),
      * station_mse   — mean over station-days of the squared station-day bias,
      * mean_absbias / mean_mae / mean_rmse — station-day means of |bias|, per-case
        MAE and per-case RMSE.
    A tag whose columns are absent or all-NaN (e.g. raw with no xco2_raw) yields
    NaN and is skipped by the markdown formatter."""
    out = {'n': int(len(g))}
    n = g['n_oco'].to_numpy(float)
    for tag in series:
        if f'rmse_{tag}' not in g.columns:
            continue
        r = g[f'rmse_{tag}'].to_numpy(float)
        b = g[f'bias_{tag}'].to_numpy(float)
        ae = (g[f'mae_{tag}'].to_numpy(float) if f'mae_{tag}' in g.columns
              else np.full(len(g), np.nan))
        sse = r ** 2                                   # per-case MSE = RMSE²
        m = np.isfinite(sse) & np.isfinite(n)
        wsum = float(np.sum(n[m])) if m.any() else 0.0
        out[f'pooled_mse_{tag}'] = (float(np.sum((n * sse)[m]) / wsum)
                                    if wsum > 0 else np.nan)
        out[f'mean_case_mse_{tag}'] = _safe_nanmean(sse)
        out[f'station_mse_{tag}'] = _safe_nanmean(b ** 2)
        out[f'mean_absbias_{tag}'] = _safe_nanmean(np.abs(b))
        out[f'mean_mae_{tag}'] = _safe_nanmean(ae)
        out[f'mean_rmse_{tag}'] = _safe_nanmean(r)
    return out


def _prog(a, keyfmt, series=('raw', 'before', 'after')):
    """'raw → before → after' string for aggregate dict ``a`` (key = keyfmt % tag);
    drops tags whose value is missing/NaN and bolds the 'after' term."""
    parts = []
    for tag in series:
        v = a.get(keyfmt % tag)
        if v is None or not np.isfinite(v):
            continue
        parts.append(f"**{v:.2f}**" if tag == 'after' else f"{v:.2f}")
    return ' → '.join(parts)


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


def _boot_bias_ci(vals, sites, n_boot, seed):
    """Site-clustered bootstrap 95% CI on the mean of ``vals`` (one value per
    station-day), resampling whole sites with replacement so within-site clustering
    is respected.  Returns (lo, hi) or (nan, nan) when too few sites/points."""
    v = np.asarray(vals, float); s = np.asarray(sites)
    m = np.isfinite(v)
    v, s = v[m], s[m]
    uniq = np.unique(s)
    if v.size < 3 or uniq.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx_by = {u: np.where(s == u)[0] for u in uniq}
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, uniq.size, replace=True)
        means[b] = v[np.concatenate([idx_by[u] for u in pick])].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _metrics_agg(g, n_boot=2000, seed=20260707):
    """Aggregate one (ref, surface, cld-group) block of per-case metric rows into the
    comprehensive cross-model metrics dict (item 5).  Station-day level for means,
    biases, robust stats and the OCO-vs-TCCON OLS (reusing _ols_fit); footprint-
    weighted for pooled RMSE/MAE and reduced-χ²."""
    out = dict(n_station_days=int(len(g)), n_footprints=int(g['n_oco'].sum()))
    sites = g['site'].to_numpy() if 'site' in g.columns else np.array([])
    n = g['n_oco'].to_numpy(float)
    out['cld_dist_mu'] = _safe_nanmean(g['cld_dist_mu'])
    out['cld_dist_sd'] = _safe_nanstd(g['cld_dist_mu'])       # spread across station-days
    out['tccon_mu'] = _safe_nanmean(g['tccon_mu'])
    out['tccon_sd'] = _safe_nanmean(g['tccon_sd'])
    out['tccon_err_mean'] = _safe_nanmean(g['tccon_err_mean'])
    for tag, mucol in (('raw', 'raw_mu'), ('before', 'orig_mu'), ('after', 'corr_mu')):
        b = g[f'bias_{tag}'].to_numpy(float)
        r = g[f'rmse_{tag}'].to_numpy(float)
        ae = g[f'mae_{tag}'].to_numpy(float)
        mu = g[mucol].to_numpy(float)
        out[f'{tag}_mu'] = _safe_nanmean(mu); out[f'{tag}_mu_sd'] = _safe_nanstd(mu)
        out[f'bias_{tag}'] = _safe_nanmean(b); out[f'bias_{tag}_sd'] = _safe_nanstd(b)
        mr = np.isfinite(r) & np.isfinite(n); wr = float(n[mr].sum())
        rmse_p = float(np.sqrt(np.sum((n * r ** 2)[mr]) / wr)) if wr > 0 else np.nan
        ma = np.isfinite(ae) & np.isfinite(n); wa = float(n[ma].sum())
        out[f'rmse_{tag}'] = rmse_p
        out[f'mae_{tag}'] = float(np.sum((n * ae)[ma]) / wa) if wa > 0 else np.nan
        mb = out[f'bias_{tag}']
        out[f'crmse_{tag}'] = (float(np.sqrt(max(rmse_p ** 2 - mb ** 2, 0.0)))
                               if np.isfinite(rmse_p) and np.isfinite(mb) else np.nan)
        bb = b[np.isfinite(b)]
        if bb.size:
            med = float(np.median(bb))
            out[f'medbias_{tag}'] = med
            out[f'mad_{tag}'] = float(1.4826 * np.median(np.abs(bb - med)))
        else:
            out[f'medbias_{tag}'], out[f'mad_{tag}'] = np.nan, np.nan
    z2 = g['chi2_z2'].to_numpy(float); nz = g['chi2_n'].to_numpy(float)
    mz = np.isfinite(z2) & np.isfinite(nz) & (nz > 0); wz = float(nz[mz].sum())
    out['chi2_z2'] = float(np.sum((nz * z2)[mz]) / wz) if wz > 0 else np.nan
    out['chi2_frac95'] = _safe_nanmean(g['chi2_frac95'])
    for tag, mucol in (('before', 'orig_mu'), ('after', 'corr_mu')):
        ff = _ols_fit(g['tccon_mu'], g[mucol])
        if ff is None:
            continue
        out[f'slope_{tag}'] = ff['slope']; out[f'slope_se_{tag}'] = ff['slope_se']
        out[f'intercept_{tag}'] = ff['intercept']
        out[f'intercept_se_{tag}'] = ff['intercept_se']; out[f'r2_{tag}'] = ff['r2']
        zs = ((ff['slope'] - 1.0) / ff['slope_se']
              if ff['slope_se'] and ff['slope_se'] > 0 else np.nan)
        zi = (ff['intercept'] / ff['intercept_se']
              if ff['intercept_se'] and ff['intercept_se'] > 0 else np.nan)
        out[f'slope_vs1_z_{tag}'] = zs; out[f'slope_vs1_p_{tag}'] = _z_to_p(zs)
        out[f'intercept_vs0_z_{tag}'] = zi; out[f'intercept_vs0_p_{tag}'] = _z_to_p(zi)
    for tag in ('before', 'after'):
        lo, hi = _boot_bias_ci(g[f'bias_{tag}'], sites, n_boot, seed)
        out[f'bias_{tag}_ci_lo'], out[f'bias_{tag}_ci_hi'] = lo, hi
    # skill score = fractional pooled-RMSE reduction (1 − RMSE_after/RMSE_ref); the
    # ML skill (vs operational xco2_bc) and the total-pipeline skill (vs raw xco2_raw).
    def _skill(ref):
        r0, r1 = out.get(f'rmse_{ref}'), out['rmse_after']
        return (1.0 - r1 / r0) if (np.isfinite(r0) and r0 > 0 and np.isfinite(r1)) else np.nan
    out['skill_rmse'] = _skill('before')          # ML vs operational bias correction
    out['skill_rmse_vs_raw'] = _skill('raw')      # ML vs pre-bias-correction raw
    return out


def _metrics_table(mrep, cld_labels, n_boot):
    """Full (ref × surface × cld-group) aggregated metrics DataFrame."""
    rows = []
    for ref in ('ak', 'direct'):
        for sname in ('all', 'ocean', 'land'):
            for glab in cld_labels:
                g = mrep[(mrep['ref'] == ref) & (mrep['surface'] == sname)
                         & (mrep['cld_group'] == glab) & (mrep['n_tccon'] > 0)]
                if len(g):
                    rows.append(dict(ref=ref, surface=sname, cld_group=glab,
                                     **_metrics_agg(g, n_boot=n_boot)))
    return pd.DataFrame(rows)


def _metrics_md_lines(tbl):
    """Compact markdown for the pooled `all` surface (corrected series); the full
    raw/before/after × ocean/land breakdown lives in tccon_metrics_{ref}.csv."""
    def f(x, n=2):
        return '' if x is None or not np.isfinite(x) else f'{x:.{n}f}'
    out = ['', '## Comprehensive metrics table (per surface × cloud group)', '',
           '_Pooled `all` surface, corrected (after) series shown; full '
           'raw/before/after × ocean/land breakdown in `tccon_metrics_{ref}.csv`. '
           'RMSE/MAE footprint-weighted; cRMSE = bias-removed RMSE; slope/R² from the '
           'station-mean OCO-vs-TCCON OLS; ⟨z²⟩ = reduced-χ² vs `de_sigma` (≈1 ideal); '
           'skill = 1−RMSE_after/RMSE_before (fractional RMSE reduction, >0 = correction '
           'helps); 95% CI is site-clustered bootstrap on the mean bias._']
    for ref in ('ak', 'direct'):
        t = tbl[(tbl['ref'] == ref) & (tbl['surface'] == 'all')]
        if not len(t):
            continue
        out += ['', f'### {ref.upper()} reference', '',
                '| cld group | n_days | n_fp | cloud dist (km) | bias after | 95% CI | '
                'RMSE | cRMSE | MAE | median±MAD | slope±SE | R² | ⟨z²⟩ | skill |',
                '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
        for _, r in t.iterrows():
            ci = (f"[{f(r['bias_after_ci_lo'])}, {f(r['bias_after_ci_hi'])}]"
                  if np.isfinite(r.get('bias_after_ci_lo', np.nan)) else '')
            out.append(
                f"| {r['cld_group']} | {int(r['n_station_days'])} | {int(r['n_footprints'])} "
                f"| {f(r['cld_dist_mu'], 1)}±{f(r['cld_dist_sd'], 1)} | {f(r['bias_after'])} "
                f"| {ci} | {f(r['rmse_after'])} | {f(r['crmse_after'])} | {f(r['mae_after'])} "
                f"| {f(r['medbias_after'])}±{f(r['mad_after'])} "
                f"| {f(r.get('slope_after'), 3)}±{f(r.get('slope_se_after'), 3)} "
                f"| {f(r.get('r2_after'), 3)} | {f(r.get('chi2_z2'), 2)} "
                f"| {f(r.get('skill_rmse'), 3)} |")
    return out


def _tccon_band(ax, frame):
    """Draw each case's OWN ±(reported TCCON xco2_error) zone as a short grey
    segment centered at bias=0, at that row's y-position.  Rows are assumed
    plotted at y = 0..n-1 in ``frame`` order (as the dumbbell / direct-vs-AK
    series do).  Per-case TCCON measurement uncertainty — it varies station-day
    to station-day (sparse or high-error windows are wider).  Reference-
    independent: AK/prior harmonization shifts the reference VALUE, not the
    retrieval error, so the zones are identical on the AK and direct figures.
    Returns the mean σ (ppm) or None when no TCCON error is available."""
    if 'tccon_err_mean' not in getattr(frame, 'columns', ()):
        return None
    e = frame['tccon_err_mean'].to_numpy(float)
    if not np.isfinite(e).any():
        return None
    labelled = False
    for i, err in enumerate(e):
        if not np.isfinite(err):
            continue
        ax.fill_betweenx([i - 0.4, i + 0.4], -err, err, color='gray', alpha=0.20,
                         lw=0, zorder=0,
                         label=(None if labelled else '±TCCON σ (per case)'))
        labelled = True
    return float(np.nanmean(e))


def _draw_scatter(ax, cmp, panel=None):
    """OCO-2-vs-TCCON station-day scatter on a SINGLE axes (1:1 + OLS raw/before/after
    with slope/R²/RMSE).  No title (item 4); ``panel`` is a bold top-left tag."""
    cmp = cmp.sort_values('bias_before').reset_index(drop=True)
    if cmp['raw_mu'].notna().any():
        ax.errorbar(cmp['tccon_mu'], cmp['raw_mu'], xerr=cmp['tccon_sd'], yerr=cmp['raw_sd'],
                    fmt='o', ms=6, color='darkorange', alpha=0.7, elinewidth=0.8,
                    markeredgecolor='black', markeredgewidth=0.5, label=f'raw {XCO2}')
    ax.errorbar(cmp['tccon_mu'], cmp['orig_mu'], xerr=cmp['tccon_sd'], yerr=cmp['orig_sd'],
                fmt='o', ms=6, color='steelblue', alpha=0.7, elinewidth=0.8,
                markeredgecolor='black', markeredgewidth=0.5, label=f'original {XCO2} (bc)')
    ax.errorbar(cmp['tccon_mu'], cmp['corr_mu'], xerr=cmp['tccon_sd'], yerr=cmp['corr_sd'],
                fmt='o', ms=6, color='green', alpha=0.8, elinewidth=0.8,
                markeredgecolor='black', markeredgewidth=0.5, label='corrected')
    lo = float(np.nanmin([cmp['tccon_mu'].min(), cmp['corr_mu'].min(),
                          cmp['orig_mu'].min(), cmp['raw_mu'].min()])) - 1
    hi = float(np.nanmax([cmp['tccon_mu'].max(), cmp['corr_mu'].max(),
                          cmp['orig_mu'].max(), cmp['raw_mu'].max()])) + 1
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')
    _xline = np.array([lo, hi])
    _series = [('orig_mu', 'steelblue', 'before'), ('corr_mu', 'green', 'after')]
    if cmp['raw_mu'].notna().any():
        _series = [('raw_mu', 'darkorange', 'raw')] + _series

    def _rmse_to_tccon(col):
        _d = (cmp[col] - cmp['tccon_mu']).to_numpy(float)
        _d = _d[np.isfinite(_d)]
        return float(np.sqrt(np.mean(_d ** 2))) if _d.size else np.nan

    _txt = []
    for _col, _color, _lbl in _series:
        _f = _ols_fit(cmp['tccon_mu'], cmp[_col])
        if _f is None:
            continue
        ax.plot(_xline, _f['intercept'] + _f['slope'] * _xline,
                '-', color=_color, lw=1.6, alpha=0.9, zorder=4)
        _txt.append(f"{_lbl}:  slope = {_f['slope']:.3f} ± {_f['slope_se']:.3f}   "
                    f"R² = {_f['r2']:.3f}   RMSE(station) = {_rmse_to_tccon(_col):.2f}")
    if _txt:
        ax.text(0.04, 0.96, '\n'.join(_txt), transform=ax.transAxes,
                va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
    ax.set_xlabel(f'TCCON {XCO2} (ppm)'); ax.set_ylabel(f'OCO-2 {XCO2} (ppm)')
    ax.legend(loc='lower right', title='station-day mean (± footprint σ)', fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    _panel_label(ax, panel)


def _bias_stat_box(ax, cmp, has_raw):
    """Shared before/after (and raw) station-bias ± σ + footprint-RMSE annotation box."""
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
        ax.text(0.04, 0.96, '\n'.join(_btxt), transform=ax.transAxes,
                va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))


def _bias_dumbbell(ax, cmp, sort_col, show_labels, has_raw):
    """Per-case bias-to-TCCON dumbbell (raw→before→after, errorbar = footprint σ),
    rows sorted by ``sort_col`` (bias_before or cld_dist_mu).  ``show_labels`` toggles
    the crowded per-case 'site date' y-tick text (item 3)."""
    cmp = cmp.sort_values(sort_col).reset_index(drop=True)
    y = np.arange(len(cmp))
    for yi in range(len(cmp)):
        pts = [cmp['bias_before'].iloc[yi], cmp['bias_after'].iloc[yi]]
        if has_raw and np.isfinite(cmp['bias_raw'].iloc[yi]):
            pts.append(cmp['bias_raw'].iloc[yi])
        ax.plot([min(pts), max(pts)], [yi, yi], '-', color='lightgray', lw=1.2, zorder=1)
    if has_raw:
        ax.errorbar(cmp['bias_raw'], y, xerr=cmp['raw_sd'], fmt='o', ms=6,
                    color='darkorange', ecolor='darkorange', elinewidth=0.8,
                    capsize=2.5, capthick=0.8, markeredgecolor='black',
                    markeredgewidth=0.5, label='raw', zorder=2)
    ax.errorbar(cmp['bias_before'], y, xerr=cmp['orig_sd'], fmt='o', ms=6,
                color='steelblue', ecolor='steelblue', elinewidth=0.8,
                capsize=2.5, capthick=0.8, markeredgecolor='black',
                markeredgewidth=0.5, label='before', zorder=3)
    ax.errorbar(cmp['bias_after'], y, xerr=cmp['corr_sd'], fmt='o', ms=6,
                color='green', ecolor='green', elinewidth=0.8,
                capsize=2.5, capthick=0.8, markeredgecolor='black',
                markeredgewidth=0.5, label='after', zorder=4)
    ax.axvline(0, color='k', lw=1)
    _tccon_band(ax, cmp)
    if show_labels:
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.site} {r.date}" for r in cmp.itertuples()], fontsize=6)
        ax.set_ylabel('station-day (sorted by pre-correction bias)')
    else:
        ax.set_yticks([])
        _order = ('nearest-cloud distance, near→far' if sort_col == 'cld_dist_mu'
                  else 'pre-correction bias')
        ax.set_ylabel(f'station-day (sorted by {_order}; IDs in CSV)')
    ax.set_xlabel(f'{XCO2} bias to TCCON (ppm)')
    ax.legend(loc='lower right', title='per station-day', fontsize=7)
    ax.grid(alpha=0.3, axis='x')


def _bias_scatter_clddist(ax, cmp, has_raw):
    """Bias-to-TCCON vs station-mean nearest-cloud distance (item 3 default): each
    station-day is a point, before→after linked by a thin connector.  Ties the figure
    to the cloud-proximity thesis; no per-case labels."""
    cmp = cmp.sort_values('cld_dist_mu').reset_index(drop=True)
    x = cmp['cld_dist_mu'].to_numpy(float)
    for i in range(len(cmp)):
        ax.plot([x[i], x[i]], [cmp['bias_before'].iloc[i], cmp['bias_after'].iloc[i]],
                '-', color='lightgray', lw=0.8, zorder=1)
    if has_raw:
        ax.scatter(x, cmp['bias_raw'], s=26, color='darkorange', edgecolor='black',
                   linewidths=0.4, alpha=0.7, label='raw', zorder=2)
    ax.scatter(x, cmp['bias_before'], s=26, color='steelblue', edgecolor='black',
               linewidths=0.4, alpha=0.8, label='before', zorder=3)
    ax.scatter(x, cmp['bias_after'], s=30, color='green', edgecolor='black',
               linewidths=0.4, alpha=0.85, label='after', zorder=4)
    ax.axhline(0, color='k', lw=1)
    # Individual per-station-day TCCON σ (item 1): a short grey ±(reported xco2_error)
    # bar at that station's x — NOT the pooled mean band.  Reference-independent.
    # vlines width is in points, so it stays sane even on a single-station panel.
    err = (cmp['tccon_err_mean'].to_numpy(float) if 'tccon_err_mean' in cmp.columns
           else np.full(len(cmp), np.nan))
    m = np.isfinite(x) & np.isfinite(err)
    if m.any():
        ax.vlines(x[m], -err[m], err[m], color='gray', alpha=0.30, lw=4,
                  zorder=0, label='±TCCON σ (per station-day)')
    ax.set_xlabel('station-mean nearest-cloud distance (km)')
    ax.set_ylabel(f'{XCO2} bias to TCCON (ppm)')
    ax.legend(loc='best', fontsize=7); ax.grid(alpha=0.3)


def _draw_bias(ax, cmp, style='scatter_clddist', panel=None):
    """Per-case bias panel on a SINGLE axes, in one of BIAS_STYLES (item 3).  Shared
    stat box + ±TCCON σ shading; no title (item 4)."""
    cmp = cmp.copy()
    has_raw = 'bias_raw' in cmp.columns and cmp['bias_raw'].notna().any()
    if style == 'scatter_clddist':
        _bias_scatter_clddist(ax, cmp, has_raw)
    else:                                              # dumbbell_{clddist,nolabel,label}
        sort_col = 'cld_dist_mu' if style == 'dumbbell_clddist' else 'bias_before'
        _bias_dumbbell(ax, cmp, sort_col, show_labels=(style == 'dumbbell_label'),
                       has_raw=has_raw)
    _bias_stat_box(ax, cmp, has_raw)
    _panel_label(ax, panel)


def main():
    global CORR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--script', default='curc_shell_blanca_plot_corr_xco2_deepens.sh')
    ap.add_argument('--out-base', default='results/model_comparison/deep_ensemble',
                    help='Base dir holding combined_DATE[_SITE] case dirs.')
    ap.add_argument('--output-dir', default='results/model_comparison/deep_ensemble')
    ap.add_argument('--corr-col', default='deep_ensemble_corrected_xco2',
                    help="Corrected-XCO2 column read from each plot_data.parquet "
                         "(default 'deep_ensemble_corrected_xco2'; use "
                         "'tabm_corrected_xco2' for TabM).")
    ap.add_argument('--radius-km', type=float, default=100.0)
    ap.add_argument('--window-min', type=float, default=60.0)
    ap.add_argument('--fname-suffix', default='',
                    help="Appended before the extension of every output filename "
                         "(e.g. '_r100km') so a parameter sweep's reports coexist.")
    ap.add_argument('--exclude-sites', default='',
                    help="Comma-separated TCCON site codes (e.g. 'ny'); also emits an "
                         "_excl_<sites> variant of the scatter/dumbbell and by-surface "
                         "figures with those stations dropped.")
    ap.add_argument('--cld-edges', default='0,10,inf',
                    help="Nearest-cloud-distance bin edges (km, comma-separated, "
                         "right-open; use 'inf' for the open tail) for the cloud-distance-"
                         "grouped TCCON comparison. Default '0,10,inf' → near (≤10 km) vs "
                         "far. Emits tccon_comparison_by_cld{,_by_surface}.")
    ap.add_argument('--dpi', type=int, default=150,
                    help='Figure DPI (default 150 for iteration; use 300 for the final '
                         'manuscript figures). The 4-file bias-style test set always '
                         'renders at a fixed low DPI regardless of this value.')
    ap.add_argument('--bias-style', default='scatter_clddist', choices=BIAS_STYLES,
                    help="Bias-panel style used for the production *_bias figures "
                         "(default 'scatter_clddist' = bias vs nearest-cloud distance). "
                         "All four styles are always emitted as a low-DPI test set for "
                         "the headline all-cases view so the paper style can be chosen.")
    ap.add_argument('--cld-all-years', action='store_true',
                    help=f"Include free-drift-era cases (year ≥ {AQUA_FREE_DRIFT_YEAR}) in "
                         "the cloud-distance-grouped comparison. By default only pre-drift "
                         "cases are used, since the Aqua-MODIS cloud collocation (hence "
                         "cld_dist_km) is reliable only before Aqua entered free drift.")
    ap.add_argument('--ak-harmonize', action='store_true',
                    help='AK/prior-harmonize the TCCON reference (Rodgers & Connor 2003 / '
                         'Wunch et al. 2017) using the day OCO-2 Lite file; cases whose '
                         'Lite file is not found fall back to the raw TCCON window mean '
                         '(ak_delta = NaN in the CSV). Shifts absolute biases only — '
                         'before/after improvement metrics are invariant.')
    ap.add_argument('--n-boot', type=int, default=10000,
                    help='Site-clustered bootstrap replicates for the significance block.')
    ap.add_argument('--uncertainty', action='store_true',
                    help='Phase-4 uncertainty-aware comparison: attach u_oco (Side A, DE '
                         'predictive σ × k(cld_dist)) and u_TC (Side B: meas ⊕ temporal ⊕ '
                         'AK-leakage ⊕ colloc) to each case, then test D = corrected − AK '
                         'TCCON with M1 (z/CI), M3 (TOST equivalence) and M4 (random-effects). '
                         'Implies --ak-harmonize (the absolute comparison lives on the AK '
                         'reference). Needs plot_data.parquet regenerated with de_sigma '
                         '(build_deepens_plot_data.py Phase-0 columns); cases lacking it are '
                         'silently skipped in the uncertainty block.')
    ap.add_argument('--rope-delta', type=float, default=DELTA_PPM,
                    help=f'TOST/ROPE equivalence margin in ppm (default {DELTA_PPM:g}).')
    ap.add_argument('--decorr-km', type=float, default=15.0,
                    help='Spatial decorrelation length (km) for the Side-A N_eff block '
                         'count (default 15).')
    args = ap.parse_args()
    if args.uncertainty:
        args.ak_harmonize = True   # the uncertainty comparison is on the AK reference (§6)
    global CORR
    CORR = args.corr_col

    out_base = Path(args.out_base)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sfx = args.fname_suffix
    excl = [s.strip() for s in args.exclude_sites.split(',') if s.strip()]
    # Output paths (suffix inserted before the extension).
    # Textual/CSV artifacts keep their names; every figure now follows the symmetric
    # tccon_{ak,direct}_* scheme emitted in the figure section (item 2).
    P_CSV      = out_dir / f'tccon_comparison{sfx}.csv'
    P_MD       = out_dir / f'tccon_comparison{sfx}.md'
    P_SITE_CSV = out_dir / f'tccon_comparison_by_site{sfx}.csv'
    P_SIG_CSV  = out_dir / f'tccon_significance{sfx}.csv'
    P_UNC_CSV  = out_dir / f'tccon_uncertainty{sfx}.csv'
    P_UNC_MD   = out_dir / f'tccon_uncertainty{sfx}.md'
    P_CLD_CSV  = out_dir / f'tccon_comparison_by_cld{sfx}.csv'
    P_CLD_AGG  = out_dir / f'tccon_comparison_by_cld_agg{sfx}.csv'
    cld_bins = _parse_cld_edges(args.cld_edges)
    storage = get_storage_dir()

    # Phase-4 per-surface inflation models k(cld_dist) (fit in Phase 2b); missing
    # → k=1 (raw de_sigma).  Loaded once and reused for every case.
    infl_by_sfc = load_inflation(out_base) if args.uncertainty else None

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

    # per-value helper: mu, sd, signed bias-to-TCCON, per-footprint RMSE-to-TCCON,
    # and per-footprint MAE-to-TCCON (mean |XCO2 − TCCON|).
    def _stat(vals, tmu, n_tc):
        v = np.asarray(vals, float)
        if not np.isfinite(v).any():
            return np.nan, np.nan, np.nan, np.nan, np.nan
        mu, sd = float(np.nanmean(v)), float(np.nanstd(v))
        bias = (mu - tmu) if n_tc else np.nan
        rmse = float(np.sqrt(np.nanmean((v - tmu) ** 2))) if n_tc else np.nan
        mae = float(np.nanmean(np.abs(v - tmu))) if n_tc else np.nan
        return mu, sd, bias, rmse, mae

    # KEEP-guards series (headline) + DROP-guards series (correction quality) for one
    # per-case footprint frame (a whole case, or one surface within it).
    def _case_metrics(frame, tmu, n_tc):
        raw = frame['xco2_raw'] if 'xco2_raw' in frame.columns else np.full(len(frame), np.nan)
        drop = frame[~frame['is_guarded']] if 'is_guarded' in frame.columns else frame
        raw_mu, raw_sd, bias_raw, rmse_raw, mae_raw = _stat(raw, tmu, n_tc)
        orig_mu, orig_sd, bias_before, rmse_before, mae_before = _stat(frame['xco2_bc'], tmu, n_tc)
        corr_mu, corr_sd, bias_after, rmse_after, mae_after = _stat(frame[CORR], tmu, n_tc)
        _, _, bias_before_dg, rmse_before_dg, _ = _stat(drop['xco2_bc'], tmu, n_tc)
        _, _, bias_after_dg, rmse_after_dg, _ = _stat(drop[CORR], tmu, n_tc)
        # station-mean nearest-cloud distance (reference-independent; drives the
        # cloud-distance bias figure + the metrics table's cloud-distance column).
        cd = (frame['cld_dist_km'].to_numpy(float) if 'cld_dist_km' in frame.columns
              else np.array([]))
        cd = cd[np.isfinite(cd)]
        cld_dist_mu = float(cd.mean()) if cd.size else np.nan
        cld_dist_sd = float(cd.std()) if cd.size else np.nan
        # reduced-χ² of the CORRECTED residual against the model's per-footprint
        # de_sigma: ⟨z²⟩≈1 ⇔ residual scatter consistent with stated uncertainty.
        chi2_z2, chi2_frac95, chi2_n = np.nan, np.nan, 0
        if n_tc and 'de_sigma' in frame.columns:
            r = np.asarray(frame[CORR], float) - tmu
            s = np.asarray(frame['de_sigma'], float)
            m = np.isfinite(r) & np.isfinite(s) & (s > 0)
            if m.any():
                zz = r[m] / s[m]
                chi2_z2 = float(np.mean(zz ** 2))
                chi2_frac95 = float(np.mean(np.abs(zz) <= 1.96))
                chi2_n = int(m.sum())
        return dict(
            n_oco=len(frame),
            n_guarded=int(frame['is_guarded'].sum()) if 'is_guarded' in frame.columns else 0,
            cld_dist_mu=cld_dist_mu, cld_dist_sd=cld_dist_sd,
            chi2_z2=chi2_z2, chi2_frac95=chi2_frac95, chi2_n=chi2_n,
            raw_mu=raw_mu, raw_sd=raw_sd, orig_mu=orig_mu, orig_sd=orig_sd,
            corr_mu=corr_mu, corr_sd=corr_sd,
            bias_raw=bias_raw, bias_before=bias_before, bias_after=bias_after,
            rmse_raw=rmse_raw, rmse_before=rmse_before, rmse_after=rmse_after,
            mae_raw=mae_raw, mae_before=mae_before, mae_after=mae_after,
            n_oco_dg=len(drop), bias_before_dg=bias_before_dg, bias_after_dg=bias_after_dg,
            rmse_before_dg=rmse_before_dg, rmse_after_dg=rmse_after_dg)

    rows = []
    rows_cld = []          # cloud-distance-grouped per-case rows (pre-drift cases)
    # Unified, reference-tagged per-(case, surface, cld-group) metric rows that drive
    # BOTH the symmetric ak/direct figure set (item 2) and the comprehensive metrics
    # table (item 5).  cld_group 'all' = ungrouped; the named bins are pre-drift only.
    rows_metrics = []
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
        op_case = adj_case = None       # captured for the Phase-4 Side-B budget
        if args.ak_harmonize and n_tc and np.isfinite(tmu):
            ot = pd.to_datetime(near['time'], unit='s', utc=True, errors='coerce').dropna()
            adj = None
            if len(ot):
                tpath = str(tccon_path(c['tccon']))
                t0 = ot.min().tz_localize(None); t1 = ot.max().tz_localize(None)
                try:
                    op = operator_from_dataframe(near)
                    op_case = op
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
                adj_case = adj

        # ── Phase-4 Side-B budget (once per case, on the AK reference) ───────────
        # u_TC = meas ⊕ temporal ⊕ AK-leakage ⊕ colloc; colloc uses the corrected
        # field (CORR) so contamination scatter doesn't inflate the gradient.
        u_tc_case, u_tc_parts = np.nan, {}
        if args.uncertainty and op_case is not None and adj_case is not None:
            try:
                sb = side_b_case(op_case, adj_case, near,
                                 (col['st_lon'], col['st_lat']),
                                 tccon_err_mean=col.get('tccon_err_mean', np.nan),
                                 n_tccon=n_tc, corr_col=CORR)
                if sb is not None:
                    u_tc_case = sb['u_TC']
                    u_tc_parts = {f'uTC_{k}': v for k, v in sb.items() if k != 'u_TC'}
            except Exception as e:                           # noqa: BLE001
                print(f"[uncertainty side-B] {site} {c['date']}: {e}", file=sys.stderr)
        # Pre-drift cases only: the Aqua-MODIS cloud collocation (hence cld_dist_km)
        # is reliable before Aqua entered free drift (--cld-all-years to override).
        case_predrift = args.cld_all_years or int(c['date'][:4]) < AQUA_FREE_DRIFT_YEAR
        # Reference views for the metrics/figure collection: 'direct' is the raw TCCON
        # window mean; 'ak' the harmonized reference (only distinct, and only present,
        # under --ak-harmonize).  When AK is off, the sole 'direct' view IS the mean.
        if args.ak_harmonize and np.isfinite(tmu):
            ref_specs = (('ak', tmu), ('direct', tmu_raw))
        else:
            ref_specs = (('direct', tmu),)
        terr = col.get('tccon_err_mean', np.nan)

        def _mrows(sname, glab, glo, ghi, gframe):
            """One reference-tagged metrics row per ref view for a footprint group."""
            for ref, tref in ref_specs:
                rows_metrics.append(dict(
                    ref=ref, site=site, date=c['date'], surface=sname,
                    cld_group=glab, cld_lo=glo, cld_hi=ghi, predrift=case_predrift,
                    n_tccon=n_tc, tccon_mu=tref, tccon_sd=tsd, tccon_err_mean=terr,
                    **_case_metrics(gframe, tref, n_tc)))

        # one row per surface: all (pooled) + ocean (sfc0) + land (sfc1)
        for sname, frame in (('all', near),
                             ('ocean', near[near['sfc_type'] == 0]),
                             ('land',  near[near['sfc_type'] == 1])):
            if not len(frame):
                continue
            _mrows(sname, 'all', -np.inf, np.inf, frame)   # ungrouped (both refs)
            row = dict(site=site, date=c['date'], surface=sname, n_tccon=n_tc,
                       tccon_mu=tmu, tccon_sd=tsd, tccon_mu_raw=tmu_raw,
                       tccon_err_mean=col.get('tccon_err_mean', np.nan),
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
            # ── Phase-4 Side-A budget + comparison stats (per surface) ──────────
            # D = corrected mean − AK TCCON = bias_after; u_D = √(u_oco²+u_TC²);
            # M1 (z/CI) + M3 (TOST equiv at --rope-delta).  Skipped (NaN) when the
            # plot_data lacks de_sigma or Side B could not be built.
            if args.uncertainty:
                sa = side_a_case(frame, infl_by_sfc, decorr_km=args.decorr_km)
                u_oco = sa['u_oco'] if sa else np.nan
                cmpv = compare_case(row['bias_after'], u_oco, u_tc_case,
                                    delta=args.rope_delta)
                row.update(dict(u_oco=u_oco, u_TC=u_tc_case, **cmpv, **u_tc_parts))
                if sa:
                    row.update(dict(uA_epi=sa['epi_sigma'], uA_avg=sa['avg_sigma'],
                                    uA_Neff=sa['N_eff'], uA_src=sa['epi_src']))
            rows.append(row)
            # ── cloud-distance-grouped rows (same schema + cld_group) ───────────
            # Split this surface's footprints by nearest-cloud distance and emit a
            # per-(case, surface, cld-bin) row for the cloud-distance aggregate + figures.
            if case_predrift and 'cld_dist_km' in frame.columns:
                cd = frame['cld_dist_km'].to_numpy(float)
                for glab, glo, ghi in cld_bins:
                    gframe = frame[np.isfinite(cd) & (cd >= glo) & (cd < ghi)]
                    if not len(gframe):
                        continue
                    rows_cld.append(dict(
                        site=site, date=c['date'], surface=sname, cld_group=glab,
                        cld_lo=glo, cld_hi=ghi, n_tccon=n_tc,
                        tccon_mu=tmu, tccon_sd=tsd, tccon_mu_raw=tmu_raw,
                        tccon_err_mean=col.get('tccon_err_mean', np.nan),
                        **_case_metrics(gframe, tmu, n_tc)))
                    _mrows(sname, glab, glo, ghi, gframe)   # both refs, per cld-bin

    if not rows:
        print(f"No cases matched (out-base={out_base}, script={args.script}). "
              "Nothing to report — check that each case's plot_data.parquet exists.",
              file=sys.stderr)
        return
    rep = pd.DataFrame(rows).sort_values(['surface', 'site', 'date']).reset_index(drop=True)
    rep.to_csv(P_CSV, index=False)

    # Reference-tagged per-(case, surface, cld-group) metrics frame — drives the
    # symmetric ak/direct figures and the comprehensive metrics table below.
    mrep = pd.DataFrame(rows_metrics)

    rep_cld = pd.DataFrame(rows_cld)
    if len(rep_cld):
        rep_cld = (rep_cld.sort_values(['surface', 'cld_group', 'site', 'date'])
                          .reset_index(drop=True))
        rep_cld.to_csv(P_CLD_CSV, index=False)

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

    # ── cloud-distance-grouped aggregate (pre-drift cases) ────────────────────
    rc_all = pd.DataFrame()
    if len(rep_cld):
        rc_all = rep_cld[(rep_cld['surface'] == 'all') & (rep_cld['n_tccon'] > 0)]
    if len(rc_all):
        era = 'all years' if args.cld_all_years else f'pre-drift, year < {AQUA_FREE_DRIFT_YEAR}'
        n_days = rc_all[['site', 'date']].drop_duplicates().shape[0]
        # Aggregate every (surface, cld group) → CSV; markdown shows the pooled 'all' rows.
        agg_rows = []
        for sname in ('all', 'ocean', 'land'):
            for glab, _, _ in cld_bins:
                g = rep_cld[(rep_cld['surface'] == sname) & (rep_cld['n_tccon'] > 0)
                            & (rep_cld['cld_group'] == glab)]
                if len(g):
                    agg_rows.append(dict(surface=sname, cld_group=glab, **_mse_aggregate(g)))
        pd.DataFrame(agg_rows).to_csv(P_CLD_AGG, index=False)
        agg_all = {r['cld_group']: r for r in agg_rows if r['surface'] == 'all'}
        has_raw = any(np.isfinite(a.get('mean_rmse_raw', np.nan)) for a in agg_all.values())
        _seq = 'raw → before → after' if has_raw else 'before → after'
        # (1) error metrics: |bias|, MAE, RMSE — each raw → before(xco2_bc) → after(ML).
        lines += ['', f'## Cloud-distance-grouped aggregate ({era})', '',
                  f'_Each collocation\'s footprints split by nearest-cloud distance '
                  f'(edges {args.cld_edges} km); station-day mean per bin, over '
                  f'{n_days} station-days.  Each cell is {_seq} (ppm).'
                  + ('  raw = pre-bias-correction xco2_raw, before = xco2_bc, '
                     'after = ML-corrected._' if has_raw else '_'), '',
                  f'| cld group | n | mean \\|bias\\| ({_seq}) | MAE ({_seq}) | fp-RMSE ({_seq}) | \\|bias\\|↓ |',
                  '|---|--:|--:|--:|--:|--:|']
        for glab, _, _ in cld_bins:
            g = rc_all[rc_all['cld_group'] == glab]
            a = agg_all.get(glab)
            if not len(g) or a is None:
                continue
            ab, aa = g['bias_before'].abs(), g['bias_after'].abs()
            lines.append(f"| {glab} | {len(g)} | {_prog(a, 'mean_absbias_%s')} | "
                         f"{_prog(a, 'mean_mae_%s')} | {_prog(a, 'mean_rmse_%s')} | "
                         f"{int((aa < ab).sum())}/{len(g)} |")
        # (2) absolute MSE (ppm²): pooled footprint-weighted, mean per-case, station-mean.
        lines += ['', f'### Cloud-distance-grouped absolute MSE (ppm², {_seq})', '',
                  '_pooled = footprint-weighted mean of (XCO₂−TCCON)² (Σ n·RMSE²/Σ n); '
                  'per-case = mean of per-station-day MSE (=RMSE²); station = mean of '
                  'squared station-day bias.  Full surface×bin breakdown in the _agg CSV._', '',
                  f'| cld group | n | pooled fp-MSE ({_seq}) | mean per-case MSE ({_seq}) | station-mean MSE ({_seq}) |',
                  '|---|--:|--:|--:|--:|']
        for glab, _, _ in cld_bins:
            a = agg_all.get(glab)
            if a is None:
                continue
            lines.append(f"| {glab} | {a['n']} | {_prog(a, 'pooled_mse_%s')} | "
                         f"{_prog(a, 'mean_case_mse_%s')} | {_prog(a, 'station_mse_%s')} |")

    # ── Phase-4 uncertainty-aware comparison (M1/M3/M4 + ⟨z²⟩) ────────────────
    if args.uncertainty and len(cmp) and 'u_D' in cmp.columns:
        unc_block = markdown_block(cmp, radius_km=args.radius_km,
                                   window_min=args.window_min, delta=args.rope_delta)
        lines += unc_block
        # standalone artifacts: the block as its own md + the full per-case CSV
        # (all budget components: u_oco/u_TC parts, z, CIs, flags).
        P_UNC_MD.write_text('\n'.join(unc_block).lstrip('\n') + '\n')
        unc_cols = ['site', 'date', 'surface', 'n_oco', 'n_tccon', 'bias_after',
                    'D', 'u_oco', 'u_TC', 'u_D', 'z', 'ci_lo', 'ci_hi',
                    'significant', 'equivalent', 'tost_p',
                    'uA_epi', 'uA_avg', 'uA_Neff', 'uA_src',
                    'uTC_u_meas', 'uTC_u_temporal', 'uTC_u_harm', 'uTC_u_colloc']
        rep_unc = rep[rep['surface'].isin(('all', 'ocean', 'land'))]
        rep_unc = rep_unc[[c for c in unc_cols if c in rep_unc.columns]]
        rep_unc.to_csv(P_UNC_CSV, index=False)

    # ── comprehensive per-(surface × cloud-group) metrics table (item 5) ─────
    # One aggregated CSV per reference (all columns) + a condensed markdown table.
    metrics_saved = []
    if len(mrep):
        metrics_tbl = _metrics_table(mrep, ['all'] + [b[0] for b in cld_bins],
                                     n_boot=min(args.n_boot, 2000))
        if len(metrics_tbl):
            for ref in metrics_tbl['ref'].unique():
                p = out_dir / f'tccon_metrics_{ref}{sfx}.csv'
                metrics_tbl[metrics_tbl['ref'] == ref].to_csv(p, index=False)
                metrics_saved.append(p)
            lines += _metrics_md_lines(metrics_tbl)

    P_MD.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    for _p in metrics_saved:
        print(f"[saved] {_p}")
    if args.uncertainty and len(cmp) and 'u_D' in cmp.columns:
        print(f"\n[saved] {P_UNC_MD}\n[saved] {P_UNC_CSV}")

    # ── figures — one panel per file (item 1); symmetric ak/direct sets (item 2);
    #    no titles, (a)/(b)… letters on multi-panel stacks (item 4) ────────────
    _saved = []

    def _one(draw, png, dpi=None):
        """Single-panel figure: draw(ax) → its own file."""
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        draw(ax)
        fig.tight_layout(); fig.savefig(png, dpi=dpi or args.dpi, bbox_inches='tight')
        plt.close(fig); _saved.append(png); return png

    def _stack(draws, png):
        """Vertical multi-panel stack; each draw(ax, letter) gets an (a)/(b)… tag."""
        draws = [d for d in draws if d is not None]
        if not draws:
            return None
        fig, axes = plt.subplots(len(draws), 1, figsize=(7.6, 5.8 * len(draws)),
                                 squeeze=False)
        for i, draw in enumerate(draws):
            draw(axes[i, 0], chr(ord('a') + i))
        fig.tight_layout(); fig.savefig(png, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig); _saved.append(png); return png

    def _grid(cells, ncols, png):
        """Row-major panel grid (item 2: 2×2 rather than a tall single column).
        ``cells`` may contain None (rendered blank so the surface×bin grid stays
        aligned); (a)/(b)… letters follow the filled cells only."""
        if not any(c is not None for c in cells):
            return None
        nrows = int(np.ceil(len(cells) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.6 * ncols, 5.8 * nrows),
                                 squeeze=False)
        ltr = 0
        for k, draw in enumerate(cells):
            ax = axes[k // ncols, k % ncols]
            if draw is None:
                ax.axis('off'); continue
            draw(ax, chr(ord('a') + ltr)); ltr += 1
        for k in range(len(cells), nrows * ncols):
            axes[k // ncols, k % ncols].axis('off')
        fig.tight_layout(); fig.savefig(png, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig); _saved.append(png); return png

    def _sel(ref, surface, cld='all', excl=()):
        s = mrep[(mrep['ref'] == ref) & (mrep['surface'] == surface)
                 & (mrep['cld_group'] == cld) & (mrep['n_tccon'] > 0)]
        return s[~s['site'].isin(excl)] if len(excl) else s

    _bs = args.bias_style
    cld_labels = [b[0] for b in cld_bins]

    def _emit_figures(ref):
        """Full single-panel figure set for one reference view ('ak' or 'direct')."""
        allc = _sel(ref, 'all', 'all')
        if not len(allc):
            return
        _one(lambda ax: _draw_scatter(ax, allc), out_dir / f'tccon_{ref}_scatter{sfx}.png')
        _one(lambda ax: _draw_bias(ax, allc, _bs), out_dir / f'tccon_{ref}_bias{sfx}.png')
        # by surface (ocean/land) stacked
        _stack([(lambda ax, ltr, g=_sel(ref, s, 'all'), s=s: _draw_scatter(ax, g, panel=f'({ltr}) {s}'))
                for s in ('ocean', 'land') if len(_sel(ref, s, 'all'))],
               out_dir / f'tccon_{ref}_by_surface_scatter{sfx}.png')
        _stack([(lambda ax, ltr, g=_sel(ref, s, 'all'), s=s: _draw_bias(ax, g, _bs, panel=f'({ltr}) {s}'))
                for s in ('ocean', 'land') if len(_sel(ref, s, 'all'))],
               out_dir / f'tccon_{ref}_by_surface_bias{sfx}.png')
        # by cloud group (surface=all) stacked
        _stack([(lambda ax, ltr, g=_sel(ref, 'all', gl), gl=gl: _draw_scatter(ax, g, panel=f'({ltr}) {gl}'))
                for gl in cld_labels if len(_sel(ref, 'all', gl))],
               out_dir / f'tccon_{ref}_by_cld_scatter{sfx}.png')
        _stack([(lambda ax, ltr, g=_sel(ref, 'all', gl), gl=gl: _draw_bias(ax, g, _bs, panel=f'({ltr}) {gl}'))
                for gl in cld_labels if len(_sel(ref, 'all', gl))],
               out_dir / f'tccon_{ref}_by_cld_bias{sfx}.png')
        # surface × cloud group (bias only) — 2×ncld grid: rows = ocean/land,
        # cols = cloud bins (item 2), None-padded so empty (surface, bin) stay aligned.
        cells = [(None if not len(_sel(ref, s, gl)) else
                  (lambda ax, ltr, g=_sel(ref, s, gl), s=s, gl=gl:
                   _draw_bias(ax, g, _bs, panel=f'({ltr}) {s} {gl}')))
                 for s in ('ocean', 'land') for gl in cld_labels]
        _grid(cells, len(cld_labels),
              out_dir / f'tccon_{ref}_by_surface_by_cld_bias{sfx}.png')
        # excl-sites variants
        if excl:
            e = _sel(ref, 'all', 'all', excl=excl)
            if len(e):
                _es = '_'.join(excl)
                _one(lambda ax: _draw_scatter(ax, e),
                     out_dir / f'tccon_{ref}_excl_{_es}_scatter{sfx}.png')
                _one(lambda ax: _draw_bias(ax, e, _bs),
                     out_dir / f'tccon_{ref}_excl_{_es}_bias{sfx}.png')

    if len(mrep):
        for ref in (('ak', 'direct') if args.ak_harmonize else ('direct',)):
            _emit_figures(ref)

        # ── 4-style bias test set (low DPI) for the headline all-cases view (item 3) ──
        ref0 = 'ak' if args.ak_harmonize else 'direct'
        allc0 = _sel(ref0, 'all', 'all')
        if len(allc0):
            for st in BIAS_STYLES:
                _one(lambda ax, st=st: _draw_bias(ax, allc0, st),
                     out_dir / f'tccon_{ref0}_bias_{st}{sfx}.png', dpi=85)

    # ── AK-vs-direct overlay + reference-shift, split into two single-panel files ──
    if args.ak_harmonize and len(cmp) and 'bias_after_direct' in cmp.columns:
        s = cmp[cmp['bias_after_direct'].notna() & cmp['ak_delta'].notna()].copy()
        if len(s):
            s1 = s.sort_values('bias_after_direct').reset_index(drop=True)
            s2 = s.sort_values('ak_delta').reset_index(drop=True)

            def _draw_ak_bias(ax):
                y = np.arange(len(s1))
                xe = s1['corr_sd']            # OCO footprint σ (reference-independent)
                for i in y:
                    ax.plot([s1['bias_after_direct'].iloc[i], s1['bias_after'].iloc[i]],
                            [i, i], '-', color='lightgray', lw=1.2, zorder=0)
                ax.errorbar(s1['bias_after_direct'], y, xerr=xe, fmt='o', color='green',
                            ms=6, ecolor='green', elinewidth=0.8, capsize=2.5, capthick=0.8,
                            markeredgecolor='black', markeredgewidth=0.5, label='direct', zorder=3)
                ax.errorbar(s1['bias_after'], y, xerr=xe, fmt='s', color='purple',
                            ms=6, ecolor='purple', elinewidth=0.8, capsize=2.5, capthick=0.8,
                            markeredgecolor='black', markeredgewidth=0.5,
                            label='AK-harmonized', zorder=4)
                ax.axvline(0, color='k', lw=1); _tccon_band(ax, s1)
                ax.set_yticks([]); ax.set_xlabel(f'corrected {XCO2} bias to TCCON (ppm)')
                ax.set_ylabel('station-day (sorted by direct bias; IDs in CSV)')
                ax.legend(loc='lower right', fontsize=7); ax.grid(alpha=0.3, axis='x')

            def _draw_ak_shift(ax):
                y2 = np.arange(len(s2))
                ax.barh(y2, s2['ak_delta'], color='teal', alpha=0.85)
                ax.axvline(0, color='k', lw=1); ax.set_yticks([])
                ax.set_xlabel('AK reference shift Δ = harmonized − raw TCCON (ppm)')
                ax.set_ylabel('station-day (sorted by Δ; IDs in CSV)')
                ax.grid(alpha=0.3, axis='x')

            _one(_draw_ak_bias, out_dir / f'tccon_ak_vs_direct_bias{sfx}.png')
            _one(_draw_ak_shift, out_dir / f'tccon_ak_shift{sfx}.png')

    # ── per-site 2×2 bar chart, one figure per reference (items 2 + 4) ──────────
    # (a) mean |bias|, (b) mean footprint RMSE, (c) mean OCO-2 σ — before vs after;
    # (d) per-site skill = 1−RMSE_after/RMSE_before.  Built per ref from mrep.
    def _draw_site_fig(ref):
        g = _sel(ref, 'all', 'all')
        if not len(g):
            return
        # Per-site combined TCCON σ = RMS of that station's station-day xco2_error
        # windows (individual per-station, not a global mean).
        def _rms(s):
            v = s.to_numpy(float); v = v[np.isfinite(v)]
            return float(np.sqrt(np.mean(v ** 2))) if v.size else np.nan
        sa = (g.groupby('site').agg(
                  n=('date', 'size'),
                  abias_before=('bias_before', lambda s: np.nanmean(np.abs(s))),
                  abias_after=('bias_after', lambda s: np.nanmean(np.abs(s))),
                  rmse_before=('rmse_before', 'mean'), rmse_after=('rmse_after', 'mean'),
                  sd_before=('orig_sd', 'mean'), sd_after=('corr_sd', 'mean'),
                  terr=('tccon_err_mean', _rms))
                .reset_index())
        sa['skill'] = 1.0 - sa['rmse_after'] / sa['rmse_before']
        sa = sa.sort_values('abias_before', ascending=False).reset_index(drop=True)
        x = np.arange(len(sa)); w = 0.38; labels = list(sa['site'])
        fig, axg = plt.subplots(2, 2, figsize=(max(11, 1.1 * len(sa)), 10), squeeze=False)

        def _bars(ax, before, after, ylabel, letter):
            ax.bar(x - w/2, sa[before], w, color='steelblue', label='before')
            ax.bar(x + w/2, sa[after], w, color='green', label='after')
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(ylabel); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')
            _panel_label(ax, letter)

        _bars(axg[0, 0], 'abias_before', 'abias_after', 'mean |bias to TCCON| (ppm)', '(a)')
        # per-station combined TCCON σ guide segment over each site's bar group
        ts = sa['terr'].to_numpy(float); _lab = False
        for xi, ti in zip(x, ts):
            if not np.isfinite(ti):
                continue
            # bold crimson dashed segment overhanging the bar group so it reads
            # clearly against the before/after bars it sits over.
            axg[0, 0].hlines(ti, xi - 0.48, xi + 0.48, color='crimson', ls='--', lw=2.6,
                             zorder=6, label=(None if _lab else 'combined TCCON σ (per station)'))
            _lab = True
        axg[0, 0].legend(fontsize=7)
        _bars(axg[0, 1], 'rmse_before', 'rmse_after', 'mean footprint RMSE (ppm)', '(b)')
        _bars(axg[1, 0], 'sd_before', 'sd_after', 'mean OCO-2 σ (ppm)', '(c)')
        axg[1, 1].bar(x, sa['skill'], color='teal', alpha=0.85)
        axg[1, 1].axhline(0, color='k', lw=1)
        axg[1, 1].set_xticks(x); axg[1, 1].set_xticklabels(labels, fontsize=8)
        axg[1, 1].set_ylabel(r'skill = 1 − RMSE$_{\mathrm{after}}$/RMSE$_{\mathrm{before}}$')
        axg[1, 1].grid(alpha=0.3, axis='y'); _panel_label(axg[1, 1], '(d)')
        fig.tight_layout()
        p = out_dir / f'tccon_{ref}_by_site_bias{sfx}.png'
        fig.savefig(p, dpi=args.dpi, bbox_inches='tight'); plt.close(fig); _saved.append(p)

    if len(mrep):
        for ref in (('ak', 'direct') if args.ak_harmonize else ('direct',)):
            _draw_site_fig(ref)

    print(f"\n[saved] {P_CSV}\n[saved] {P_MD}\n[saved] {P_SITE_CSV}")
    if len(rep_cld):
        print(f"[saved] {P_CLD_CSV}")
        if P_CLD_AGG.exists():
            print(f"[saved] {P_CLD_AGG}")
    for _p in _saved:
        print(f"[saved] {_p}")


if __name__ == '__main__':
    main()

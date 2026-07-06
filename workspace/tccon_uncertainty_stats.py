"""tccon_uncertainty_stats.py — Phase-4 comparison statistics for the
uncertainty-aware OCO-2 (ML-corrected) vs TCCON (AK-harmonized) comparison
(src/analysis/UNCERTAINTY_AWARE_TCCON_COMPARISON.md §5).

Ties the two per-case budgets together and layers the statistics:

  Side A  u_oco  (sidea_uncertainty.case_uncertainty, per-surface k(cld_dist))
  Side B  u_TC   (tccon_uncertainty.side_b_uncertainty)
  D = x̄_oco − c̄_TC   (= the report's bias_after against the AK reference)
  u_D = √(u_oco² + u_TC²)

  M1  per-case z / 95% CI on D            (significant ⇔ CI excludes 0)
  M3  TOST / ROPE equivalence at δ ppm    (equivalent ⇔ 90% CI ⊂ [−δ, δ])  — HEADLINE
  M4  DerSimonian–Laird random-effects    (global μ ± CI, between-case τ, I²)
  end-to-end case calibration ⟨z²⟩        (≈1 ⇔ the whole budget is calibrated)

The module is import-safe with no side effects; the report calls the small
functions below.  Everything degrades gracefully (returns NaN / None) when a
required column is missing so the report can run on legacy plot_data.parquet.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sidea_uncertainty import (InflationModel, case_uncertainty,
                               calibrated_sigma_by_surface)
from tccon_uncertainty import side_b_uncertainty

# science equivalence margin (ppm): OCO-2 accuracy target used for TOST/ROPE.
DELTA_PPM = 0.5

# ── normal-tail helpers (no scipy dependency) ──────────────────────────────────
_SQRT2 = math.sqrt(2.0)


def _phi(x):
    """Standard-normal CDF via erf (vectorized-safe scalar)."""
    return 0.5 * (1.0 + math.erf(float(x) / _SQRT2))


# ── inflation model loading ────────────────────────────────────────────────────
def load_inflation(base_dir, *, ocean_name='sidea_inflation_ocean.json',
                   land_name='sidea_inflation_land.json'):
    """Load the per-surface k(cld_dist) inflation models fit in Phase 2b.

    Returns {0: InflationModel|None (ocean), 1: InflationModel|None (land)}.
    A surface with no JSON maps to None → its footprints use k=1 (raw de_sigma).
    Searches *base_dir* and its parent (the fit is cached at the DE tag root).
    """
    out = {}
    roots = [Path(base_dir), Path(base_dir).parent]
    for sfc, name in ((0, ocean_name), (1, land_name)):
        model = None
        for r in roots:
            p = r / name
            if p.exists():
                model = InflationModel.load(p)
                break
        out[sfc] = model
    return out


# ── Side A (per case) ──────────────────────────────────────────────────────────
def side_a_case(frame, infl_by_sfc, *, decorr_km=15.0):
    """Per-case Side-A uncertainty u_oco (ppm) for one footprint frame.

    Applies the per-surface inflation k(cld_dist) to each footprint's de_sigma,
    then aggregates (epistemic floor ⊕ averaged predictive term).  Uses the exact
    per-member epistemic (mu_NN cols) ONLY when the frame is a single surface —
    for a mixed (pooled 'all') frame the per-surface members are different models,
    so the fully-correlated fallback is forced.  Returns None if de_sigma absent.
    """
    if 'de_sigma' not in frame.columns or not len(frame):
        return None
    infl_present = {s: m for s, m in (infl_by_sfc or {}).items() if m is not None}
    sig_fp = calibrated_sigma_by_surface(frame, infl_present) if infl_present \
        else frame['de_sigma'].to_numpy(float)
    single_surface = ('sfc_type' not in frame.columns
                      or frame['sfc_type'].nunique() <= 1)
    return case_uncertainty(frame, sigma_fp=sig_fp, use_members=single_surface,
                            decorr_km=decorr_km)


# ── Side B (per case) ──────────────────────────────────────────────────────────
def side_b_case(op, adj, frame, station, *, tccon_err_mean=np.nan,
                n_tccon=0, corr_col=None):
    """Per-case Side-B uncertainty u_TC (ppm).  Thin wrapper over
    side_b_uncertainty that turns the report's scalars into its arguments.
    Returns None when the operator/adjustment is unavailable."""
    if op is None or adj is None:
        return None
    # side_b_uncertainty averages a per-obs error array and divides by sqrt(N);
    # feed the window-mean error replicated N times to reproduce mean/sqrt(N).
    errs = None
    if np.isfinite(tccon_err_mean) and n_tccon > 0:
        errs = np.full(int(n_tccon), float(tccon_err_mean))
    return side_b_uncertainty(op, adj, footprints=frame, station=station,
                              tccon_errors=errs, corr_col=corr_col)


# ── M1 + M3 (per case) ─────────────────────────────────────────────────────────
def compare_case(D, u_oco, u_TC, *, delta=DELTA_PPM):
    """M1 (z / 95% CI) + M3 (TOST equivalence at ±delta) for one case.

    Returns dict(D, u_D, z, ci_lo, ci_hi, significant, tost_p, equivalent).
      significant : 95% CI excludes 0 (the two sides differ).
      equivalent  : TOST — the 90% CI of D lies inside [−delta, delta]
                    (⇔ p_tost < 0.05), i.e. agreement within the science margin.
    """
    u_D = float(np.hypot(u_oco, u_TC)) if np.isfinite(u_oco) and np.isfinite(u_TC) \
        else np.nan
    if not (np.isfinite(D) and np.isfinite(u_D) and u_D > 0):
        return dict(D=float(D) if np.isfinite(D) else np.nan, u_D=u_D, z=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, significant=False,
                    tost_p=np.nan, equivalent=False)
    z = D / u_D
    ci_lo, ci_hi = D - 1.96 * u_D, D + 1.96 * u_D
    significant = (ci_lo > 0) or (ci_hi < 0)
    # TOST: H0a D ≤ −δ (upper-tail), H0b D ≥ +δ (lower-tail); reject both ⇒ equiv.
    p_lower = 1.0 - _phi((D + delta) / u_D)     # evidence D > −δ
    p_upper = _phi((D - delta) / u_D)           # evidence D < +δ
    tost_p = max(p_lower, p_upper)
    equivalent = tost_p < 0.05
    return dict(D=float(D), u_D=u_D, z=float(z), ci_lo=ci_lo, ci_hi=ci_hi,
                significant=bool(significant), tost_p=float(tost_p),
                equivalent=bool(equivalent))


# ── M4 hierarchical random-effects across cases ───────────────────────────────
def hierarchical(D, u_D):
    """DerSimonian–Laird random-effects meta-analysis over cases.

    D, u_D : per-case difference and its combined 1σ (arrays).  Treats each case
    (station-day) as a unit; τ² captures between-case (incl. between-site)
    heterogeneity beyond the within-case u_D.

    Returns dict(mu, mu_se, ci_lo, ci_hi, tau, Q, Q_p, I2, k, fe_mu) — the global
    consistency number μ ± CI and the heterogeneity τ / I².
    """
    D = np.asarray(D, float); u = np.asarray(u_D, float)
    m = np.isfinite(D) & np.isfinite(u) & (u > 0)
    D, u = D[m], u[m]
    k = D.size
    if k == 0:
        return dict(mu=np.nan, mu_se=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    tau=np.nan, Q=np.nan, Q_p=np.nan, I2=np.nan, k=0, fe_mu=np.nan)
    v = u ** 2
    w = 1.0 / v
    fe_mu = float(np.sum(w * D) / np.sum(w))            # fixed-effect mean
    Q = float(np.sum(w * (D - fe_mu) ** 2))
    if k > 1:
        c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
        tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    else:
        tau2 = 0.0
    ws = 1.0 / (v + tau2)                               # random-effect weights
    mu = float(np.sum(ws * D) / np.sum(ws))
    mu_se = float(np.sqrt(1.0 / np.sum(ws)))
    I2 = max(0.0, (Q - (k - 1)) / Q) if (k > 1 and Q > 0) else 0.0
    Q_p = _chi2_sf(Q, k - 1) if k > 1 else np.nan
    return dict(mu=mu, mu_se=mu_se, ci_lo=mu - 1.96 * mu_se, ci_hi=mu + 1.96 * mu_se,
                tau=float(np.sqrt(tau2)), Q=Q, Q_p=Q_p, I2=float(I2), k=int(k),
                fe_mu=fe_mu)


def _chi2_sf(x, df):
    """Upper-tail χ² survival function (scipy if present, else a series/normal
    fallback) — used only for the heterogeneity Q p-value."""
    if not np.isfinite(x) or df <= 0:
        return np.nan
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except Exception:                                   # noqa: BLE001
        # Wilson–Hilferty normal approximation to χ².
        t = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
        return float(1.0 - _phi(t))


# ── end-to-end case calibration ───────────────────────────────────────────────
def case_calibration(D, u_D):
    """Does the standardized case difference z = D / u_D have ⟨z²⟩ ≈ 1?

    This is the whole-budget validator (§10 Phase 4): if both Side-A and Side-B
    budgets are right, z over cases is ~N(0,1).  Returns dict(mean_z2, rms_z,
    frac_within_95, n) — mean_z2 ≈ 1 and frac_within_95 ≈ 0.95 when calibrated.
    """
    D = np.asarray(D, float); u = np.asarray(u_D, float)
    m = np.isfinite(D) & np.isfinite(u) & (u > 0)
    z = D[m] / u[m]
    if z.size == 0:
        return dict(mean_z2=np.nan, rms_z=np.nan, frac_within_95=np.nan, n=0)
    return dict(mean_z2=float(np.mean(z ** 2)), rms_z=float(np.sqrt(np.mean(z ** 2))),
                frac_within_95=float(np.mean(np.abs(z) < 1.96)), n=int(z.size))


# ── markdown rendering ─────────────────────────────────────────────────────────
def markdown_block(cmp, *, radius_km, window_min, delta=DELTA_PPM):
    """Full Phase-4 markdown block from a per-case frame that already carries
    D, u_oco, u_TC, u_D, z, significant, equivalent (one row per case, surface
    'all').  Returns a list of markdown lines (empty if no usable rows)."""
    c = cmp[np.isfinite(cmp.get('u_D', pd.Series(dtype=float)))].copy()
    if not len(c):
        return ['', '## Uncertainty-aware comparison (Phase 4)', '',
                '_No case carried the Side-A uncertainty columns (regenerate '
                'plot_data.parquet with build_deepens_plot_data.py so `de_sigma` '
                'is present)._']
    hi = hierarchical(c['D'], c['u_D'])
    cal = case_calibration(c['D'], c['u_D'])
    n_equiv = int(c['equivalent'].sum())
    n_sig = int(c['significant'].sum())

    def g(x, n=2):
        return '' if not np.isfinite(x) else f'{x:.{n}f}'

    lines = ['', '## Uncertainty-aware comparison (Phase 4)', '',
             f'_Per case: D = mean corrected OCO-2 − AK-harmonized TCCON, with '
             f'u_oco (Side A, DE predictive σ × k(cld_dist)) and u_TC (Side B: '
             f'meas ⊕ temporal ⊕ AK-leakage ⊕ colloc). {len(c)} cases, '
             f'≤{radius_km:g} km / ±{window_min:g} min._', '',
             f'**Headline (M3, TOST at δ = {delta:g} ppm):** '
             f'**{n_equiv}/{len(c)}** cases statistically equivalent to TCCON '
             f'within ±{delta:g} ppm; {n_sig}/{len(c)} differ significantly '
             f'(95% CI excludes 0).', '',
             f'**Global consistency (M4, DerSimonian–Laird random-effects):** '
             f'μ = **{g(hi["mu"])} ± {g(hi["mu_se"])}** ppm '
             f'(95% CI [{g(hi["ci_lo"])}, {g(hi["ci_hi"])}]), '
             f'between-case τ = {g(hi["tau"])} ppm, I² = {g(100*hi["I2"],0)}% '
             + ('(fixed-effect μ = %s)' % g(hi['fe_mu'])) + '.', '',
             f'**Whole-budget calibration:** ⟨z²⟩ = **{g(cal["mean_z2"])}** '
             f'over {cal["n"]} cases (1.0 = calibrated), '
             f'{g(100*cal["frac_within_95"],0)}% within the 95% band '
             f'(target 95%).', '',
             '| site | date | surf | D | u_oco | u_TC | u_D | z | 95%% CI | equiv (δ=%g) |'
             % delta,
             '|---|---|---|--:|--:|--:|--:|--:|:--:|:--:|']
    for _, r in c.sort_values(['site', 'date']).iterrows():
        ci = f"[{g(r['ci_lo'])}, {g(r['ci_hi'])}]" if np.isfinite(r.get('ci_lo', np.nan)) else ''
        flag = '✓' if r.get('equivalent') else ('✗' if r.get('significant') else '·')
        lines.append(
            f"| {r['site']} | {r['date']} | {r.get('surface', 'all')} | "
            f"{g(r['D'])} | {g(r['u_oco'])} | {g(r['u_TC'])} | {g(r['u_D'])} | "
            f"{g(r['z'])} | {ci} | {flag} |")
    lines += ['', '_equiv column: ✓ equivalent within ±%g ppm (TOST p<0.05); '
              '✗ significantly different (95%% CI excludes 0); · neither._' % delta]
    return lines

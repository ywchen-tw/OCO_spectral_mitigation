"""Why does the revised WCO2 fit scatter at low SOD, and what removes it? (orbit 29252a_GL, 2026-10-30)

Model (final_model_test.fit_wco2): ln T = c + a1·x − k1·τ_dry − k1_w·τ_h2o on the Doppler-corrected τ.
Steps
  1. Baseline fit of every sounding, residual matrix R [N, 1016].
  2. Decomposition of the far-clear residual (cld > 20 km, QF0) at SOD < 0.1 and overall into a per-footprint
     fixed channel pattern and a sounding-to-sounding part, also per (footprint, v_inst cluster).
  3. Hypotheses: L1B bad-sample list and spike-EOF flags; solar Doppler (the production TOA solar shifts the
     solar lines by v_sol only, fp_abs_coeff.py `beta_total = v_sol/c`; the spacecraft–surface velocity
     v_inst is missing, as it was for the telluric τ); H2O fraction, ILS shift/width (resid_attrs columns);
     airmass scaling of the leftover fixed pattern (missing weak absorber such as CH4).
  4. Candidate corrections and the metrics: rms at SOD < 0.1 and overall, k1 median and far-clear spread,
     QF0 near-cloud k1 contrast (0–1, 1–3 km; latitude-local reference), corr(k1_WCO2, k1_SCO2) on far-clear
     soundings after removing per-footprint medians (SCO2 with the matching solar treatment).
No retrieved L2 quantity enters a fit. The solar correction uses only L1B geometry (sounding_relative_velocity,
sounding_solar_relative_velocity) and the solar model.

Usage:  python wco2_lowsod_analysis.py          (≈ 2 min; needs results/wco2_lowsod_solar_grid_GL.npz)
Outputs: results/wco2_lowsod_{decomp,candidates,per_fp,solar_scan}_29252a_GL.csv,
         results/wco2_lowsod_patterns_29252a_GL.npz (for the figure), results/wco2_lowsod_summary_29252a_GL.md
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wco2_lowsod_common as wc  # noqa: E402

TAG = "29252a_GL"
OUT = wc.OUT
IB = 1


# --------------------------------------------------------------------------- data


def load():
    orb, tau, tau_h2o, meta, _ = wc.fm.load("GL")
    ex = wc.l1b_extras(wc.fm.DEFAULTS["l1b"], orb.sounding_id, orb.fp_number)
    grid = np.load(OUT / f"wco2_lowsod_solar_grid_GL.npz")["lnS"].astype(float)
    za = np.load(OUT / f"resid_attrs_{TAG}.npz")
    if not np.array_equal(za["j"], np.arange(len(orb.fp_number))):
        raise ValueError("resid_attrs is not in full sounding order")
    attrs = {k: za[f"wco2_{k}"].astype(float) for k in ("dtau_dpix", "dtau_dlnw", "f_h2o", "dtau_dT")}
    airmass = 1 / np.cos(np.deg2rad(meta["sza"].astype(float))) + 1 / np.cos(np.deg2rad(meta["vza"].astype(float)))
    return dict(orb=orb, tau=tau, tau_h2o=tau_h2o, meta=meta, ex=ex, grid=grid, attrs=attrs, airmass=airmass)


def solar_delta(d, ib, alpha=1.0):
    """ln S(v_sol) − ln S(v_sol − α v_inst) [N, 1016]: add to ln T to move the solar lines by α·v_inst."""
    fp, vs, vi = d["orb"].fp_number, d["ex"]["v_sol"], d["ex"]["v_inst"]
    g = d["grid"][ib]
    return np.stack([wc.interp_v(g[fp[j]], vs[j]) - wc.interp_v(g[fp[j]], vs[j] - alpha * vi[j])
                     for j in range(len(fp))])


# --------------------------------------------------------------------------- fitting


def run_wco2(d, dy=None, drop=None, extra=None, keep_resid=False):
    """Fit every sounding. dy: [N,1016] added to ln T; drop: [N,1016] bool excluded channels;
    extra(j, ok) -> list of extra columns."""
    orb = d["orb"]
    y_all, t_all, tw_all, s_all = orb.ln_t[IB], d["tau"][IB], d["tau_h2o"][IB], orb.sigma_ln_t[IB]
    n = y_all.shape[0]
    k1, k1w, a1, rlow, rall = (np.full(n, np.nan) for _ in range(5))
    R = np.full((n, 1016), np.nan, np.float32) if keep_resid else None
    for j in range(n):
        y = y_all[j].astype(float) + (0 if dy is None else dy[j])
        t, tw, s = t_all[j].astype(float), tw_all[j].astype(float), s_all[j].astype(float)
        ok = wc.base_mask(y, t, tw, s)
        if drop is not None:
            ok &= ~drop[j]
        if ok.sum() < 20:
            continue
        cols = [] if extra is None else extra(j, ok)
        c, r = wc.fit_wco2_ext(t[ok] - tw[ok], tw[ok], wc.X_ALL[ok], y[ok], cols)
        k1[j], k1w[j], a1[j] = c[2], c[3], c[1]
        lo = t[ok] < wc.LOW_SOD
        rlow[j] = np.sqrt(np.mean(r[lo] ** 2)) if lo.sum() > 10 else np.nan
        rall[j] = np.sqrt(np.mean(r**2))
        if keep_resid:
            R[j, ok] = r
    return dict(k1=k1, k1_w=k1w, a1=a1, rms_low=rlow, rms_all=rall, R=R)


def run_sco2(d, dy=None):
    orb = d["orb"]
    land = d["meta"]["sfc_type"] == 1
    n = orb.ln_t.shape[1]
    k1 = np.full(n, np.nan)
    for j in range(n):
        y = orb.ln_t[2, j].astype(float) + (0 if dy is None else dy[j])
        t, tw, s = d["tau"][2, j].astype(float), d["tau_h2o"][2, j].astype(float), orb.sigma_ln_t[2, j].astype(float)
        ok = wc.base_mask(y, t, tw, s)
        if ok.sum() < 20:
            continue
        k1[j] = wc.fm.fit_sco2(t[ok] - tw[ok], tw[ok], wc.X_ALL[ok], y[ok], bool(land[j]))["k1"]
    return k1


# --------------------------------------------------------------------------- decomposition


def decompose(R, low, far, fp, sig, groups=None):
    """Fraction of Σr² explained by the per-group mean channel pattern (groups default: footprint).

    Returns rows per footprint and pooled, for SOD < 0.1 and all channels. The expected noise share is
    Σσ²/Σr² over the same entries. The pattern is estimated with cross-fitting (two along-track halves,
    each half's pattern applied to the other half), so its own noise does not inflate the fixed share.
    """
    groups = fp if groups is None else groups
    rows = []
    for sel_lab, chan in (("SOD<0.1", low), ("all", np.ones_like(low))):
        tot = fix = fix_cf = noise = 0.0
        for f in range(8):
            jj = np.flatnonzero(far & (fp == f))
            r = R[jj].astype(float)
            m = np.isfinite(r) & chan[jj]
            rr = np.where(m, r, np.nan)
            pat = np.zeros_like(rr)
            pat_cf = np.zeros_like(rr)
            for g in np.unique(groups[jj]):
                gi = groups[jj] == g
                pat[gi] = np.nanmean(rr[gi], axis=0)
                half = np.arange(gi.sum()) < gi.sum() // 2
                idx = np.flatnonzero(gi)
                for a, b in ((idx[half], idx[~half]), (idx[~half], idx[half])):
                    pat_cf[b] = np.nanmean(rr[a], axis=0)
            res = rr - pat
            res_cf = rr - pat_cf
            t_ = np.nansum(rr**2)
            rows.append(dict(sel=sel_lab, fp=f + 1, n=len(jj), rms=np.sqrt(np.nanmean(rr**2)),
                             fixed_frac=1 - np.nansum(res**2) / t_, fixed_frac_cv=1 - np.nansum(res_cf**2) / t_,
                             noise_frac=np.nansum(np.where(m, sig[jj] ** 2, 0)) / t_,
                             rms_after_pattern=np.sqrt(np.nanmean(res_cf**2))))
            tot += t_
            fix += np.nansum(res**2)
            fix_cf += np.nansum(res_cf**2)
            noise += np.nansum(np.where(m, sig[jj] ** 2, 0))
        rows.append(dict(sel=sel_lab, fp="pooled", n=int(far.sum()), rms=np.nan, fixed_frac=1 - fix / tot,
                         fixed_frac_cv=1 - fix_cf / tot, noise_frac=noise / tot, rms_after_pattern=np.nan))
    return rows


def cross_fit_pattern(R, far, fp, j_split):
    """Per-footprint pattern for each sounding: far-clear soundings get the pattern of the other along-track
    half of the same footprint; all other soundings get the full far-clear pattern."""
    n = R.shape[0]
    P = np.zeros((n, 1016))
    for f in range(8):
        jf = np.flatnonzero(far & (fp == f))
        first = jf[jf < j_split[f]]
        second = jf[jf >= j_split[f]]
        full = np.nan_to_num(np.nanmean(R[jf], axis=0))
        P[fp == f] = full
        P[first] = np.nan_to_num(np.nanmean(R[second], axis=0))
        P[second] = np.nan_to_num(np.nanmean(R[first], axis=0))
    return P


# --------------------------------------------------------------------------- main


def main():
    t0 = time.time()
    d = load()
    orb, meta = d["orb"], d["meta"]
    fp = orb.fp_number.astype(int)
    n = len(fp)
    cld = meta["cld_dist_km"].astype(float)
    far = (cld > 20) & (meta["xco2_qf"] == 0)
    low = d["tau"][IB] < wc.LOW_SOD
    sig = orb.sigma_ln_t[IB].astype(float)
    vi = d["ex"]["v_inst"]
    vcl = (vi > 0).astype(int)                     # far-clear v_inst clusters (−3.6…−3.1, +2.4…+2.6 km/s)
    print(f"loaded in {time.time() - t0:.0f} s; far-clear {far.sum()}, v_inst clusters {np.bincount(vcl[far])}")
    lines = [f"# WCO2 low-SOD spread — {TAG}", "",
             f"Far-clear (cld > 20 km, QF0) n = {far.sum()}; noise median σ(ln T) at SOD < 0.1: "
             f"{np.nanmedian(sig[far][low[far]]):.4f}.", ""]

    # ---------------------------------------------------------------- 1. baseline
    base = run_wco2(d, keep_resid=True)
    R0 = base["R"]
    k1s_base = run_sco2(d)
    m = far[:, None] & np.isfinite(R0)
    lines += ["## 1. Baseline residual (far-clear)", "",
              f"rms at SOD < 0.1: {np.sqrt(np.nanmean(R0[m & low] ** 2)):.4f}; at SOD ≥ 0.1: "
              f"{np.sqrt(np.nanmean(R0[m & ~low] ** 2)):.4f}; channels per sounding at SOD < 0.1: "
              f"{(m & low).sum() / far.sum():.0f} of {m.sum() / far.sum():.0f}.", ""]

    # ---------------------------------------------------------------- 2. bad samples / spikes (H2)
    bad = d["ex"]["bad"][IB][fp]                  # [N, 1016]
    spike = d["ex"]["spike"][IB] != 0
    rows = []
    for lab, sel in (("bad_sample_list = 0", bad == 0), ("= 1 (isolated bad pixel)", bad == 1),
                     ("= 2 (detector edge, channels 0–89)", bad == 2), ("= 3", bad == 3),
                     ("spike-EOF flag ≠ 0", spike), ("spike-EOF flag = 0", ~spike)):
        mm = m & sel
        rows.append(f"| {lab} | {mm.sum() / far.sum():.1f} | {(mm & low).sum() / far.sum():.1f} | "
                    f"{np.sqrt(np.nanmean(R0[mm] ** 2)):.4f} | {np.sqrt(np.nanmean(R0[mm & low] ** 2)):.4f} |")
    lines += ["## 2. L1B flags vs baseline residual (far-clear)", "",
              "| channel class | channels / sounding | of which SOD < 0.1 | rms | rms SOD < 0.1 |", "|---|---|---|---|---|", *rows, ""]

    # ---------------------------------------------------------------- 3. solar Doppler (H3)
    dS = {a: solar_delta(d, IB, a) for a in (0.5, 0.8, 0.9, 1.0, 1.1, 1.2)}
    scan = []
    for a, dy in dS.items():
        r = run_wco2(d, dy=dy)
        scan.append(dict(alpha=a, rms_low=np.nanmedian(r["rms_low"][far]), rms_all=np.nanmedian(r["rms_all"][far]),
                         k1_med=np.nanmedian(r["k1"][far]), k1_spread=wc.spread_pct(r["k1"][far], fp[far])))
    scan.insert(0, dict(alpha=0.0, rms_low=np.nanmedian(base["rms_low"][far]), rms_all=np.nanmedian(base["rms_all"][far]),
                        k1_med=np.nanmedian(base["k1"][far]), k1_spread=wc.spread_pct(base["k1"][far], fp[far])))
    scan = pd.DataFrame(scan)
    scan.to_csv(OUT / f"wco2_lowsod_solar_scan_{TAG}.csv", index=False)
    # per-sounding free solar-shift coefficient (linearised): column = dS(α=1), coefficient β ≈ α
    beta = np.full(n, np.nan)
    D1 = dS[1.0]
    for j in np.flatnonzero(far):
        y = orb.ln_t[IB, j].astype(float)
        t, tw = d["tau"][IB, j].astype(float), d["tau_h2o"][IB, j].astype(float)
        ok = wc.base_mask(y, t, tw, sig[j])
        c, _ = wc.fit_wco2_ext(t[ok] - tw[ok], tw[ok], wc.X_ALL[ok], y[ok], [D1[j, ok]])
        beta[j] = c[4]
    # correlation of the baseline residual with the predicted solar error, per sounding
    rcorr = np.array([np.corrcoef(R0[j, np.isfinite(R0[j])], -D1[j, np.isfinite(R0[j])])[0, 1] for j in np.flatnonzero(far)])
    lines += ["## 3. Solar Doppler", "",
              "Production TOA solar: solar lines at λ_rest·(1 + v_sol/c), |v_sol| ≤ 0.12 km/s. Test: λ_rest·(1 + v_sol/c − α·v_inst/c), "
              "v_inst = L1B sounding_relative_velocity (same sign convention as the telluric correction). "
              "Reproduction of the stored production TOA solar by the local convolution: per-sounding SD of "
              "ln(toa_prod) − ln S(v_sol) ≈ 3.5e-5 (float32 level) in all bands.", "",
              f"- corr(baseline residual, predicted solar error at α = 1) per far-clear sounding: median {np.median(rcorr):+.3f} "
              f"[p10 {np.percentile(rcorr, 10):+.3f}, p90 {np.percentile(rcorr, 90):+.3f}]",
              f"- free per-sounding scale of the solar shift (linearised: α̂ = −coefficient of the α = 1 column added to the "
              f"baseline fit): median {-np.nanmedian(beta[far]):.3f}, p10–p90 {-np.nanpercentile(beta[far], 90):.3f}–"
              f"{-np.nanpercentile(beta[far], 10):.3f}; by v_inst cluster: "
              + ", ".join(f"{'v<0' if c == 0 else 'v>0'} {-np.nanmedian(beta[far & (vcl == c)]):.3f}" for c in (0, 1)), "",
              "| α | rms SOD<0.1 | rms all | k1 median | k1 spread % |", "|---|---|---|---|---|"]
    lines += [f"| {r.alpha:.1f} | {r.rms_low:.4f} | {r.rms_all:.4f} | {r.k1_med:.4f} | {r.k1_spread:.3f} |" for r in scan.itertuples()]
    lines.append("")

    # ---------------------------------------------------------------- solar-corrected residual
    sol = run_wco2(d, dy=D1, keep_resid=True)
    R1 = sol["R"]
    dS_sco2 = solar_delta(d, 2, 1.0)
    k1s_sol = run_sco2(d, dy=dS_sco2)
    m1 = far[:, None] & np.isfinite(R1)
    rows = []
    for lab, sel in (("bad_sample_list = 0", bad == 0), ("= 1 (isolated bad pixel)", bad == 1),
                     ("= 2 (detector edge, channels 0–89)", bad == 2), ("spike-EOF flag ≠ 0", spike)):
        mm = m1 & sel
        rows.append(f"| {lab} | {np.sqrt(np.nanmean(R1[mm] ** 2)):.4f} | {np.sqrt(np.nanmean(R1[mm & low] ** 2)):.4f} | "
                    f"{np.nanmedian(np.abs(R1[mm & low])):.4f} |")
    lines += ["Same flag classes on the solar-corrected residual (far-clear):", "",
              "| channel class | rms | rms SOD < 0.1 | median abs SOD < 0.1 |", "|---|---|---|---|", *rows, ""]

    # ---------------------------------------------------------------- 4. decomposition (H1), before and after solar
    dec = []
    for lab, R in (("baseline", R0), ("solar-corrected", R1)):
        for g_lab, groups in (("fp", None), ("fp × v_inst cluster", fp * 2 + vcl)):
            for r in decompose(R, low, far, fp, sig, groups):
                dec.append(dict(residual=lab, grouping=g_lab, **r))
    dec = pd.DataFrame(dec)
    dec.to_csv(OUT / f"wco2_lowsod_decomp_{TAG}.csv", index=False)
    lines += ["## 4. Fixed-pattern decomposition (far-clear; cross-fitted patterns)", "",
              "fixed = share of Σr² removed by the per-group mean channel pattern estimated on the other along-track "
              "half; noise = Σσ²/Σr² (σ from the L1B noise model).", "",
              "In-sample = pattern from all far-clear soundings of the group (≈ 150 per footprint, so its own noise adds "
              "< 1 % of σ²). Cross-fitted halves are the two along-track halves, which for grouping 'fp' are the two "
              "v_inst clusters: a negative share means the pattern changes sign between them.", "",
              "| residual | grouping | channels | fixed share in-sample | fixed share cross-fitted | noise share | remainder (in-sample) |",
              "|---|---|---|---|---|---|---|"]
    for r in dec[dec.fp == "pooled"].itertuples():
        lines.append(f"| {r.residual} | {r.grouping} | {r.sel} | {r.fixed_frac:.2f} | {r.fixed_frac_cv:.2f} | {r.noise_frac:.2f} | "
                     f"{1 - r.fixed_frac - r.noise_frac:.2f} |")
    lines += ["", "Per footprint, SOD < 0.1 (rms, fixed share by fp, by fp × v cluster, noise share):", "",
              "In-sample shares.", "",
              "| fp | n | baseline rms | fixed fp | fixed fp×v | noise | solar-corr rms | fixed fp | fixed fp×v | noise |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    q = dec[(dec.sel == "SOD<0.1") & (dec.fp != "pooled")]
    for f in range(1, 9):
        a = q[(q.fp == f) & (q.residual == "baseline")].set_index("grouping")
        b = q[(q.fp == f) & (q.residual == "solar-corrected")].set_index("grouping")
        lines.append(f"| {f} | {int(a.n.iloc[0])} | {a.rms.iloc[0]:.4f} | {a.fixed_frac['fp']:.2f} | "
                     f"{a.fixed_frac['fp × v_inst cluster']:.2f} | {a.noise_frac.iloc[0]:.2f} | {b.rms.iloc[0]:.4f} | "
                     f"{b.fixed_frac['fp']:.2f} | {b.fixed_frac['fp × v_inst cluster']:.2f} | {b.noise_frac.iloc[0]:.2f} |")
    lines.append("")

    # patterns and their attributes (for the figure and for H4/H6)
    P0v = np.stack([[np.nanmean(R0[far & (fp == f) & (vcl == c)], axis=0) for c in (0, 1)] for f in range(8)])
    P1 = np.stack([np.nanmean(R1[far & (fp == f)], axis=0) for f in range(8)])
    grid = d["grid"][IB]
    lnS0 = grid[:, np.searchsorted(wc.V_GRID, 0.0)]          # [8, 1016] at v = 0
    env = pd.DataFrame(lnS0.T).rolling(17, center=True, min_periods=1).max().values.T
    depth = 1 - np.exp(lnS0 - env)
    dlnS_dv = (grid[:, np.searchsorted(wc.V_GRID, 250.0)] - grid[:, np.searchsorted(wc.V_GRID, -250.0)]) / 500.0
    sod_med = np.stack([np.nanmedian(d["tau"][IB][far & (fp == f)], axis=0) for f in range(8)])
    fh2o = np.stack([np.nanmedian(d["attrs"]["f_h2o"][far & (fp == f)], axis=0) for f in range(8)])
    dpix = np.stack([np.nanmedian(d["attrs"]["dtau_dpix"][far & (fp == f)], axis=0) for f in range(8)])
    dlnw = np.stack([np.nanmedian(d["attrs"]["dtau_dlnw"][far & (fp == f)], axis=0) for f in range(8)])
    vbar = np.array([np.nanmedian(vi[far & (vcl == c)]) for c in (0, 1)])
    wl = wc.ils_tables(wc.fm.DEFAULTS["l1b"])[IB]["wl"]

    def pc(a, b, mk):
        ok = mk & np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[ok], b[ok])[0, 1]
    att_rows = []
    for f in range(8):
        lo_f = sod_med[f] < wc.LOW_SOD
        diff = P0v[f, 1] - P0v[f, 0]
        att_rows.append(dict(
            fp=f + 1,
            r_clusterpatterns_base=pc(P0v[f, 0], P0v[f, 1], lo_f),
            r_diff_vs_solar_pred=pc(diff, -(vbar[1] - vbar[0]) * dlnS_dv[f], lo_f),
            r_sol_pattern_vs_depth=pc(P1[f], depth[f], lo_f),
            r_sol_pattern_vs_dlnS=pc(P1[f], dlnS_dv[f], lo_f),
            r_sol_pattern_vs_fh2o=pc(P1[f], fh2o[f], lo_f),
            r_sol_pattern_vs_dtau_dpix=pc(P1[f], dpix[f], lo_f),
            r_sol_pattern_vs_dtau_dlnw=pc(P1[f], dlnw[f], lo_f),
            r_sol_pattern_vs_sod=pc(P1[f], sod_med[f], lo_f),
            rms_sol_pattern_low=np.sqrt(np.nanmean(P1[f][lo_f] ** 2))))
    att = pd.DataFrame(att_rows)
    att.to_csv(OUT / f"wco2_lowsod_per_fp_{TAG}.csv", index=False)
    lines += ["## 5. What the fixed patterns correlate with (channels with median SOD < 0.1, per footprint)", "",
              "Columns: r between the two v_inst-cluster patterns of the baseline; r of their difference with the "
              "solar-Doppler prediction −Δv·∂lnS/∂v (sign: residual = −error in ln S); then, for the solar-corrected "
              "fixed pattern, r with solar-line depth, ∂lnS/∂v, H2O fraction of τ, ∂τ/∂pix, ∂τ/∂ln(ILS width), SOD; "
              "rms of the solar-corrected pattern.", "",
              "| fp | r(v−, v+) base | r(Δ, solar pred) | depth | ∂lnS/∂v | f_H2O | ∂τ/∂pix | ∂τ/∂lnw | SOD | pattern rms |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in att.itertuples():
        lines.append(f"| {r.fp} | {r.r_clusterpatterns_base:+.2f} | {r.r_diff_vs_solar_pred:+.2f} | {r.r_sol_pattern_vs_depth:+.2f} | "
                     f"{r.r_sol_pattern_vs_dlnS:+.2f} | {r.r_sol_pattern_vs_fh2o:+.2f} | {r.r_sol_pattern_vs_dtau_dpix:+.2f} | "
                     f"{r.r_sol_pattern_vs_dtau_dlnw:+.2f} | {r.r_sol_pattern_vs_sod:+.2f} | {r.rms_sol_pattern_low:.4f} |")
    lines.append("")

    # frame test: is the solar-corrected pattern tied to the detector pixels (same in both v_inst clusters) or to
    # the solar/telluric line positions (shifted by Δv·λ/(c·Δλ) ≈ 1 px between the clusters)?
    P1v = np.stack([[np.nanmean(R1[far & (fp == f) & (vcl == c)], axis=0) for c in (0, 1)] for f in range(8)])
    shift_px = (vbar[1] - vbar[0]) / 1000 * 0.174
    pix = np.arange(1016.0)
    fr_rows = []
    for f in range(8):
        lo_f = (sod_med[f] < wc.LOW_SOD) & np.isfinite(P1v[f, 0]) & np.isfinite(P1v[f, 1])
        a, b = P1v[f, 0], P1v[f, 1]
        rec = dict(fp=f + 1, r_0=pc(a, b, lo_f))
        for sh in (-1.0, -0.5, 0.5, 1.0):
            bs = np.interp(pix + sh * shift_px, pix[np.isfinite(b)], b[np.isfinite(b)])
            rec[f"r_{sh:+.1f}"] = pc(a, bs, lo_f)
        fr_rows.append(rec)
    fr = pd.DataFrame(fr_rows)
    lines += [f"Frame test: correlation of the solar-corrected far-clear pattern of the v < 0 cluster with that of the v > 0 "
              f"cluster, unshifted and with the second shifted by ±0.5, ±1 × {shift_px:.2f} px (the relative shift of solar and "
              f"telluric lines between the clusters). A detector-fixed pattern peaks at 0; a line-frame pattern at ±1.", "",
              "| fp | shift −1 | −0.5 | 0 | +0.5 | +1 |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['fp']} | {r['r_-1.0']:+.2f} | {r['r_-0.5']:+.2f} | {r['r_0']:+.2f} | {r['r_+0.5']:+.2f} | {r['r_+1.0']:+.2f} |"
              for r in fr.to_dict("records")]
    lines.append("")

    # airmass scaling of the solar-corrected residual (missing weak absorber test): per fp, per channel,
    # regress the far-clear residual on (airmass − mean); share of the pattern variance that scales with airmass
    am = d["airmass"]
    am_rows = []
    for f in range(8):
        jj = np.flatnonzero(far & (fp == f))
        a = am[jj] - am[jj].mean()
        r = np.nan_to_num(R1[jj].astype(float))
        slope = (a @ r) / (a @ a)                               # per channel d r / d airmass
        lo_f = sod_med[f] < wc.LOW_SOD
        am_rows.append(dict(fp=f + 1, airmass_sd=a.std(), slope_rms_low=np.sqrt(np.mean(slope[lo_f] ** 2)),
                            pattern_rms_low=np.sqrt(np.nanmean(P1[f][lo_f] ** 2)),
                            r_slope_pattern=pc(slope, P1[f], lo_f), r_slope_fh2o=pc(slope, fh2o[f], lo_f)))
    amd = pd.DataFrame(am_rows)
    lines += ["Airmass test (missing weak absorber would give a fixed pattern ∝ airmass): per-channel slope of the "
              "solar-corrected far-clear residual on airmass, SOD < 0.1 channels.", "",
              "| fp | airmass SD | rms slope (per unit airmass) | pattern rms | r(slope, pattern) | r(slope, f_H2O) |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.fp} | {r.airmass_sd:.3f} | {r.slope_rms_low:.4f} | {r.pattern_rms_low:.4f} | {r.r_slope_pattern:+.2f} | "
              f"{r.r_slope_fh2o:+.2f} |" for r in amd.itertuples()]
    lines.append("")

    # ---------------------------------------------------------------- 5. candidates
    j_split = np.array([np.median(np.flatnonzero(far & (fp == f))) for f in range(8)])
    Pcf = cross_fit_pattern(R1, far, fp, j_split)
    att_cols = d["attrs"]
    cands = {
        "B0 baseline (agreed model)": dict(),
        "B1 drop bad_sample_list ≠ 0": dict(drop=bad != 0),
        "B2 drop bad_sample_list = 1 only": dict(drop=bad == 1),
        "B3 drop spike-EOF flagged": dict(drop=spike),
        "S solar Doppler (α = 1, no fit parameter)": dict(dy=D1),
        "S + drop bad ≠ 0": dict(dy=D1, drop=bad != 0),
        "S + drop bad = 1 only": dict(dy=D1, drop=bad == 1),
        "S + drop bad = 2 only (edge 0–89)": dict(dy=D1, drop=bad == 2),
        "S + per-fp fixed pattern (cross-fitted)": dict(dy=D1 - Pcf),
        "S + Legendre P2": dict(dy=D1, extra=lambda j, ok: wc.legendre_cols(wc.X_ALL[ok], 2)),
        "S + Legendre P2, P3": dict(dy=D1, extra=lambda j, ok: wc.legendre_cols(wc.X_ALL[ok], 3)),
        "S + ∂τ/∂pix column": dict(dy=D1, extra=lambda j, ok: [att_cols["dtau_dpix"][j, ok]]),
        "S + ∂τ/∂lnw column": dict(dy=D1, extra=lambda j, ok: [att_cols["dtau_dlnw"][j, ok]]),
        "S + ∂τ/∂pix + ∂τ/∂lnw": dict(dy=D1, extra=lambda j, ok: [att_cols["dtau_dpix"][j, ok], att_cols["dtau_dlnw"][j, ok]]),
        "B0 + free solar column (fitted α)": dict(extra=lambda j, ok: [D1[j, ok]]),
        "S + drop bad ≠ 0 + per-fp pattern": dict(dy=D1 - Pcf, drop=bad != 0),
    }
    rows = []
    store = {}
    for lab, kw in cands.items():
        r = run_wco2(d, **kw)
        k1s = k1s_sol if lab.startswith("S") else k1s_base
        rec = dict(candidate=lab, **wc.k1_metrics(r["k1"], k1s, meta, r["rms_low"], r["rms_all"], groups=fp * 2 + vcl))
        rec["corr_sco2_base"] = wc.k1_metrics(r["k1"], k1s_base, meta)["corr_sco2"]
        rec["corr_sco2_solar"] = wc.k1_metrics(r["k1"], k1s_sol, meta)["corr_sco2"]
        rec["k1w_spread"] = wc.spread_pct(r["k1_w"][far], fp[far])
        rows.append(rec)
        store[lab] = r
        print(f"{lab}: {rec['rms_low_far']:.4f} {rec['k1_spread']:.3f} {rec['corr_sco2']:.2f}", flush=True)
    ct = pd.DataFrame(rows)
    ct.to_csv(OUT / f"wco2_lowsod_candidates_{TAG}.csv", index=False)
    sco2_rec = [wc.k1_metrics(k, k1s_base, meta, groups=fp * 2 + vcl) for k in (k1s_base, k1s_sol)]
    lines += ["## 6. Candidate corrections (WCO2)", "",
              "rms = median over far-clear soundings of the per-sounding residual rms; spread = robust SD (%) after removing "
              "per-footprint means, far-clear; contrast = median relative k1 contrast (%) of QF0 soundings vs latitude-local "
              "far references (same fp, ±1°, cld > 10 km, QF0), SNR = contrast / far-clear robust SD of the contrast; "
              "corr = Pearson r of far-clear WCO2 k1 and SCO2 k1 after removing per-footprint medians (SCO2 with the same "
              "solar treatment; both versions in the csv).", "",
              "Extra columns: spread within (fp × v_inst cluster) = robust SD (%) after removing the medians of each footprint "
              "and far-clear v_inst cluster (the two clusters are 2 latitude bands, so this is the sounding-to-sounding part); "
              "cluster step = k1 median of the v > 0 cluster over the v < 0 cluster − 1 (%); corr within = the SCO2 correlation "
              "after removing the same group medians.", "",
              "| candidate | rms SOD<0.1 | rms all | k1 median | k1 spread % | spread within % | cluster step % | Δk1 0–1 km % (SNR) | Δk1 1–3 km % (SNR) | corr SCO2 k1 | corr within | k1_w spread % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ct.to_dict("records"):
        lines.append(f"| {r['candidate']} | {r['rms_low_far']:.4f} | {r['rms_all_far']:.4f} | {r['k1_med']:.4f} | "
                     f"{r['k1_spread']:.3f} | {r['k1_spread_within']:.3f} | {r['cluster_step']:+.3f} | {r['con_0-1']:+.3f} ({r['snr_0-1']:+.2f}) | {r['con_1-3']:+.3f} ({r['snr_1-3']:+.2f}) | "
                     f"{r['corr_sco2']:+.2f} | {r['corr_sco2_within']:+.2f} | {r['k1w_spread']:.2f} |")
    lines += ["", f"QF0 near-cloud counts: 0–1 km {int(ct['n_0-1'].iloc[0])}, 1–3 km {int(ct['n_1-3'].iloc[0])}.",
              f"SCO2 k1 (same model) baseline: median {sco2_rec[0]['k1_med']:.4f}, spread {sco2_rec[0]['k1_spread']:.3f} % (within {sco2_rec[0]['k1_spread_within']:.3f} %, "
              f"cluster step {sco2_rec[0]['cluster_step']:+.3f} %), "
              f"Δ0–1 {sco2_rec[0]['con_0-1']:+.3f} % (SNR {sco2_rec[0]['snr_0-1']:+.2f}); solar-corrected: median "
              f"{sco2_rec[1]['k1_med']:.4f}, spread {sco2_rec[1]['k1_spread']:.3f} % (within {sco2_rec[1]['k1_spread_within']:.3f} %, "
              f"cluster step {sco2_rec[1]['cluster_step']:+.3f} %), Δ0–1 {sco2_rec[1]['con_0-1']:+.3f} % "
              f"(SNR {sco2_rec[1]['snr_0-1']:+.2f}).", ""]

    # O2A: effect of the same solar correction (brief)
    o2a = []
    dS_o2a = solar_delta(d, 0, 1.0)
    land = meta["sfc_type"] == 1
    for lab, dy in (("baseline", None), ("solar-corrected", dS_o2a)):
        k1o, rmso = np.full(n, np.nan), np.full(n, np.nan)
        for j in np.flatnonzero(far):
            y = orb.ln_t[0, j].astype(float) + (0 if dy is None else dy[j])
            t, tw = d["tau"][0, j].astype(float), d["tau_h2o"][0, j].astype(float)
            ok = wc.base_mask(y, t, tw, orb.sigma_ln_t[0, j].astype(float))
            rr = wc.fm.fit_o2a(t[ok], wc.X_ALL[ok], y[ok])
            k1o[j], rmso[j] = rr["k1"], rr["rms"]
        o2a.append(f"{lab}: rms {np.nanmedian(rmso[far]):.4f}, k1 {np.nanmedian(k1o[far]):.4f}, spread {wc.spread_pct(k1o[far], fp[far]):.3f} %")
    sco2_rms = []
    for lab, dy in (("baseline", None), ("solar-corrected", dS_sco2)):
        rr_ = []
        for j in np.flatnonzero(far):
            y = orb.ln_t[2, j].astype(float) + (0 if dy is None else dy[j])
            t, tw = d["tau"][2, j].astype(float), d["tau_h2o"][2, j].astype(float)
            ok = wc.base_mask(y, t, tw, orb.sigma_ln_t[2, j].astype(float))
            rr_.append(wc.fm.fit_sco2(t[ok] - tw[ok], tw[ok], wc.X_ALL[ok], y[ok], bool(land[j]))["rms"])
        sco2_rms.append(f"{lab}: rms {np.median(rr_):.4f}")
    lines += ["## 7. Same solar correction in the other bands (far-clear)", "",
              "- O2A (floor model): " + "; ".join(o2a), "- SCO2: " + "; ".join(sco2_rms), ""]

    (OUT / f"wco2_lowsod_summary_{TAG}.md").write_text("\n".join(lines))
    print("\n".join(lines))
    np.savez_compressed(OUT / f"wco2_lowsod_patterns_{TAG}.npz", P0v=P0v.astype(np.float32), P1=P1.astype(np.float32),
                        depth=depth, dlnS_dv=dlnS_dv, sod_med=sod_med, fh2o=fh2o, wl=wl, vbar=vbar,
                        R0_far_low_rms=np.sqrt(np.nanmean(R0[m & low] ** 2)))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

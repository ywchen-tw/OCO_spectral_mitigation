"""Local test of the agreed fit model before the CURC run (2026-10-30).

Model (Doppler-corrected τ, x = channel index scaled to [−1, 1], τ_h2o = frac_h2o · τ):
    O2A : ln T = c + a1 x + ln[(1 − f) e^{−k1 τ} + f]                       f ∈ [0, 0.2], k2 fixed 0
    WCO2: ln T = c + a1 x − k1_dry τ_dry − k1_w τ_h2o                         k2 fixed 0
    SCO2: ln T = c + a1 x (+ a2 x² over land) − k1_dry τ_dry − k1_w τ_h2o + ½ k2 τ²
Diagnostics stored per sounding and band: the free k2 of the same model with ½k2τ² added (O2A: inside
the path part), residual rms, channel count, convergence flag, timing.

Usage:
    python final_model_test.py GL   # 29252a_GL, all soundings (local corrected τ + integrated-gradient H2O split)
    python final_model_test.py TG   # 48443a_TG, 100-sounding Doppler-variant smoke run (stored frac_h2o)
Outputs: results/final_model_per_sounding_<tag>.parquet, results/final_model_summary_<tag>.md,
         results/final_model_contrasts_29252a_GL.png (GL only)
"""
from __future__ import annotations

import os
import sys
import time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cut_sweep_real import DEFAULTS  # noqa: E402
from h2o_analysis import BIN_LABELS, BINS, bin_stats, lat_local_contrast  # noqa: E402
from orbit_io import BANDS, load_orbit  # noqa: E402
from run_study import robust_sd, spread_pct  # noqa: E402

OUT = HERE / "results"
SMOKE = Path("/private/tmp/claude-370367/-Volumes-YWC-SSD-programming-oco-fp-analysis/"
             "106fa6df-e817-4e62-8f1a-c441f33123ad/scratchpad/smoke/dop")
X_ALL = (np.arange(1016) - 507.5) / 507.5
_G: dict = {}


def _lin(cols, y):
    a = np.column_stack(cols)
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    return c, float(np.sqrt(np.mean((y - a @ c) ** 2)))


def _cov_se(jac, resid):
    """1-sigma parameter errors from the Jacobian and the residual: s² (JᵀJ)⁻¹ with s² = RSS/(n − p).

    The residual is misfit-dominated (reduced χ² ≫ 1 against the L1B noise), so the scale uses the fitted
    residual, not the noise model; the result is the formal error for independent residuals and understates
    scene-correlated error."""
    n, p = jac.shape
    s2 = float(resid @ resid) / max(n - p, 1)
    try:
        cov = s2 * np.linalg.inv(jac.T @ jac)
        return np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)


def _lin_se(cols, y):
    a = np.column_stack(cols)
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    r = y - a @ c
    return c, float(np.sqrt(np.mean(r**2))), _cov_se(a, r)


def fit_o2a(t, x, y):
    one = np.ones_like(t)
    c0, _ = _lin([one, x, -t], y)

    def res(p, free_k2):
        c, a1, k1, f = p[:4]
        k2 = p[4] if free_k2 else 0.0
        return c + a1 * x + np.log((1 - f) * np.exp(-k1 * t + 0.5 * k2 * t**2) + f) - y

    r = least_squares(res, [c0[0], c0[1], max(c0[2], 0.1), 0.01], args=(False,),
                      bounds=([-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, 0.2]))
    se = _cov_se(r.jac, r.fun)
    rec = dict(k1=r.x[2], k1_w=np.nan, k2=0.0, a1=r.x[1], a2=np.nan, f=r.x[3], c=r.x[0],
               rms=float(np.sqrt(np.mean(r.fun**2))), ok=bool(r.success), f_at_bound=bool(r.x[3] >= 0.2 - 1e-9),
               se_k1=se[2], se_k1_w=np.nan, se_k2=np.nan, se_a1=se[1], se_a2=np.nan, se_f=se[3], se_c=se[0])
    rd = least_squares(res, list(r.x) + [0.0], args=(True,),
                       bounds=([-np.inf, -np.inf, 0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf, 0.2, np.inf]))
    rec["k2_diag"] = rd.x[4]
    rec["se_k2_diag"] = _cov_se(rd.jac, rd.fun)[4]
    return rec


def fit_wco2(td, tw, x, y):
    one = np.ones_like(td)
    c, rms, se = _lin_se([one, x, -td, -tw], y)
    cd, _, sed = _lin_se([one, x, -td, -tw, 0.5 * (td + tw) ** 2], y)
    return dict(k1=c[2], k1_w=c[3], k2=0.0, a1=c[1], a2=np.nan, f=np.nan, c=c[0], rms=rms, ok=True,
                f_at_bound=False, k2_diag=cd[4], se_k1=se[2], se_k1_w=se[3], se_k2=np.nan, se_a1=se[1],
                se_a2=np.nan, se_f=np.nan, se_c=se[0], se_k2_diag=sed[4])


def fit_sco2(td, tw, x, y, land):
    one = np.ones_like(td)
    t = td + tw
    cols = [one, x] + ([x**2] if land else []) + [-td, -tw, 0.5 * t**2]
    c, rms, se = _lin_se(cols, y)
    k = 3 if land else 2
    return dict(k1=c[k], k1_w=c[k + 1], k2=c[k + 2], a1=c[1], a2=c[2] if land else np.nan, f=np.nan, c=c[0],
                rms=rms, ok=True, f_at_bound=False, k2_diag=c[k + 2], se_k1=se[k], se_k1_w=se[k + 1],
                se_k2=se[k + 2], se_a1=se[1], se_a2=se[2] if land else np.nan, se_f=np.nan, se_c=se[0],
                se_k2_diag=se[k + 2])


def fit_sounding(j):
    rows = []
    land = bool(_G["land"][j])
    for ib, b in enumerate(BANDS):
        t = _G["tau"][ib, j].astype(float)
        tw = _G["tau_h2o"][ib, j].astype(float)
        y = _G["ln_t"][ib, j].astype(float)
        s = _G["sig"][ib, j].astype(float)
        ok = np.isfinite(y) & np.isfinite(t) & np.isfinite(s) & (s > 0) & np.isfinite(tw)
        if ok.sum() < 20:
            rows.append(dict(j=j, band=b, nchan=int(ok.sum())))
            continue
        t, tw, y, x = t[ok], tw[ok], y[ok], X_ALL[ok]
        t0 = time.perf_counter()
        if b == "o2a":
            rec = fit_o2a(t, x, y)
        elif b == "wco2":
            rec = fit_wco2(t - tw, tw, x, y)
        else:
            rec = fit_sco2(t - tw, tw, x, y, land)
        rec.update(j=j, band=b, nchan=int(ok.sum()), seconds=time.perf_counter() - t0)
        rows.append(rec)
    return rows


def _chunk(js):
    return [r for j in js for r in fit_sounding(j)]


def load(which):
    if which == "GL":
        orb = load_orbit(str(DEFAULTS["l1b"]), str(DEFAULTS["fp_tau"]), str(DEFAULTS["fit_details"]))
        zt = np.load(OUT / "resid_doppler_tau_29252a_GL.npz")
        z = np.load(OUT / "h2o_tau_split_29252a_GL.npz")
        tau = np.transpose(zt["tau"], (1, 0, 2)).astype(np.float32)
        tau_h2o = np.stack([z[f"tau_h2o_{b}"] for b in BANDS]).astype(np.float32)
        det = DEFAULTS["fit_details"]
        tag = "29252a_GL"
    else:
        l1b = HERE.parents[1] / "data/OCO2/2023/222/48443a_TG/oco2_L1bScTG_48443a_230810_B11008r_230912173118.h5"
        det = SMOKE / "fitting_details/fitting_details_2023-08-10_48443a_TG.h5"
        orb = load_orbit(str(l1b), str(SMOKE / "2023-08-10/48443a_TG/fp_tau_combined.h5"), str(det))
        with h5py.File(SMOKE / "2023-08-10/48443a_TG/fp_tau_combined.h5") as f:
            frac = np.stack([f[f"{b}_frac_h2o"][...] for b in BANDS]).astype(np.float32)
        tau, tau_h2o, tag = orb.tau, frac * orb.tau, "48443a_TG_smoke100"
    with h5py.File(det) as f:
        meta = {k: f[k][...] for k in ("cld_dist_km", "xco2_qf", "lat", "fp_number", "sfc_type", "sza", "vza")}
    return orb, tau, tau_h2o, meta, tag


def evaluate(df, meta, tag):
    n = len(meta["lat"])
    cld, qf, lat, fp = meta["cld_dist_km"].astype(float), meta["xco2_qf"], meta["lat"].astype(float), meta["fp_number"].astype(int)
    far = (cld > 20) & (qf == 0)
    no_far = far.sum() < 10
    if no_far:                      # no cloud-distance coverage (target orbit): use every sounding for the medians/spreads
        far = np.ones(n, bool)
    lines = [f"# Agreed fit model on local data — {tag}", "",
             f"Soundings {n}; far-clear (cld > 20 km, QF0) {int(((cld > 20) & (qf == 0)).sum())}"
             + (" — no cloud-distance coverage, statistics below use all soundings" if no_far else "")
             + f"; land (sfc_type == 1) {int((meta['sfc_type'] == 1).sum())}.",
             "Spread = robust SD (%) after removing per-footprint means (far-clear). Contrasts vs latitude-local far "
             "references (same fp, ±1° lat, cld > 10 km, QF0), relative for k1-type, absolute for k2/f/a1; "
             "SNR = bin median / far-clear robust SD.", ""]

    def col(c, b, src=df):
        s = src[src.band == b].set_index("j")[c]
        v = np.full(n, np.nan)
        v[s.index.values] = s.values
        return v

    # baselines on corrected τ (single-τ S2/S3, two-τ T2 from the H2O study) and production (prod τ, order 7,3,7)
    base = {}
    h2o_file = OUT / "h2o_per_sounding_29252a_GL.parquet"
    if tag == "29252a_GL" and h2o_file.exists():
        h2 = pd.read_parquet(h2o_file)
        for c, lab in (("S2_k1", "single-τ order 2"), ("S3_k1", "single-τ order 3 (bounded)"), ("T2_k1_dry", "two-τ order 2 (no continuum)")):
            base[lab] = {b: col(c, b, h2) for b in BANDS}
        with h5py.File(DEFAULTS["fit_details"]) as f:
            base["production (prod τ, order 7,3,7)"] = {b: f[f"{b}_k1_fitting_nosg"][...] for b in BANDS}
    base["agreed model"] = {b: col("k1", b) for b in BANDS}

    # A: far-clear medians and spreads, single band and ratios
    lines += ["## A. Far-clear k1: median and spread (%)", "",
              "| fit | O2A | WCO2 | SCO2 | SCO2/O2A | WCO2/SCO2 |", "|---|---|---|---|---|---|"]
    for lab, d in base.items():
        cells = []
        for b in BANDS:
            v = d[b]
            m = far & np.isfinite(v)
            cells.append(f"{np.median(v[m]):.4f} ({spread_pct(v[m], fp[m]):.2f})" if m.sum() > 10 else "n/a")
        for num, den in (("sco2", "o2a"), ("wco2", "sco2")):
            r = d[num] / d[den]
            m = far & np.isfinite(r)
            cells.append(f"{spread_pct(r[m], fp[m]):.2f}" if m.sum() > 10 else "n/a")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines.append("")

    # B: model terms, far-clear
    lines += ["## B. Model terms, far-clear medians [p05, p95]", "", "| band | a1 | a2 | f | k2 | k2_diag (free) | frac k2_diag<0 | rms | nchan | f at bound | failed |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in BANDS:
        d = df[(df.band == b) & far[df.j.values]]

        def q(c):
            v = d[c].astype(float)
            return f"{v.median():+.4f} [{v.quantile(.05):+.4f}, {v.quantile(.95):+.4f}]" if v.notna().any() else "—"
        lines.append(f"| {b} | {q('a1')} | {q('a2')} | {q('f')} | {q('k2')} | {q('k2_diag')} | {100*(d.k2_diag < 0).mean():.0f} % | "
                     f"{d.rms.median():.4f} | {d.nchan.median():.0f} | {100*d.f_at_bound.mean():.1f} % | {100*(~d.ok.astype(bool)).mean():.1f} % |")
    lines.append("")

    # C: near-cloud contrasts
    rows = []
    if not no_far and np.isfinite(cld).sum() > 100 and (cld < 3).sum() > 50:
        quantities = [("agreed model", "k1", "o2a", True), ("agreed model", "k1", "wco2", True), ("agreed model", "k1", "sco2", True),
                      ("agreed model", "k1_w", "sco2", True), ("agreed model", "k2", "sco2", False), ("agreed model", "f", "o2a", False),
                      ("agreed model", "a1", "o2a", False), ("agreed model", "a1", "wco2", False), ("agreed model", "a1", "sco2", False),
                      ("agreed model", "rms", "o2a", True), ("agreed model", "rms", "sco2", True)]
        for lab, c, b, rel in quantities:
            rec = dict(fit=lab, quantity=c, band=b, **bin_stats(lat_local_contrast(col(c, b), cld, qf, lat, fp, rel), cld, far))
            rows.append(rec)
        for lab in ("single-τ order 2", "two-τ order 2 (no continuum)"):
            if lab in base:
                for b in BANDS:
                    rows.append(dict(fit=lab, quantity="k1", band=b, **bin_stats(lat_local_contrast(base[lab][b], cld, qf, lat, fp, True), cld, far)))
        ct = pd.DataFrame(rows)
        ct.to_csv(OUT / f"final_model_contrasts_{tag}.csv", index=False)
        cols = ["fit", "quantity", "band", "far_spread"] + [f"{k}_{lab}" for lab in BIN_LABELS[:4] for k in ("n", "con", "snr")]
        lines += ["## C. Near-cloud contrasts (relative for k1/rms, absolute for k2/f/a1)", "",
                  "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in ct.iterrows():
            lines.append("| " + " | ".join(f"{r[c]:.3g}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
        lines.append("")
    else:
        lines += ["## C. Near-cloud contrasts: not available (no cloud-distance coverage in this sample)", ""]

    # D: timing
    lines += ["## D. Cost", "", "| band | median s per sounding | total CPU s |", "|---|---|---|"]
    for b in BANDS:
        d = df[df.band == b]
        lines.append(f"| {b} | {d.seconds.median():.4f} | {d.seconds.sum():.0f} |")
    (OUT / f"final_model_summary_{tag}.md").write_text("\n".join(lines))
    print("\n".join(lines))
    if rows:
        figure(pd.DataFrame(rows), tag)
    return rows


def figure(ct, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = [("single-τ order 2", "k1", "o2a"), ("single-τ order 2", "k1", "wco2"), ("single-τ order 2", "k1", "sco2"),
             ("agreed model", "k1", "o2a"), ("agreed model", "k1", "wco2"), ("agreed model", "k1", "sco2"),
             ("agreed model", "k1_w", "sco2"), ("agreed model", "a1", "sco2"), ("agreed model", "f", "o2a")]
    labels = ["S2 k1 O2A", "S2 k1 WCO2", "S2 k1 SCO2", "AM k1 O2A", "AM k1 WCO2", "AM k1 SCO2",
              "AM k1_w SCO2", "AM a1 SCO2 (x100)", "AM f O2A (x100)"]
    vals_c, vals_s = [], []
    for fit, q, b in items:
        r = ct[(ct.fit == fit) & (ct.quantity == q) & (ct.band == b)].iloc[0]
        vals_c.append(100 * r["con_0-1"])
        vals_s.append(r["snr_0-1"])
    colors = ["#2a78d6"] * 3 + ["#eb6834"] * 3 + ["#1baf7a", "0.4", "0.6"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, vals, ylabel in zip(axes, (vals_c, vals_s), ("contrast at 0-1 km (% for k1-type; x100 for a1, f)", "SNR at 0-1 km")):
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
    fig.suptitle(f"{tag}: near-cloud contrasts, single-tau order 2 (S2) vs agreed model (AM)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"final_model_contrasts_{tag}.png", dpi=130)


def main(which, workers=8):
    orb, tau, tau_h2o, meta, tag = load(which)
    _G.update(tau=tau, tau_h2o=tau_h2o, ln_t=orb.ln_t, sig=orb.sigma_ln_t, land=(meta["sfc_type"] == 1))
    n = tau.shape[1]
    chunks = np.array_split(np.arange(n), max(1, n // 40))
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        rows = [r for part in pool.imap(_chunk, chunks) for r in part]
    print(f"{tag}: fitted {n} soundings in {time.time() - t0:.0f} s wall ({workers} workers)")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"final_model_per_sounding_{tag}.parquet", index=False)
    evaluate(df, meta, tag)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "GL")

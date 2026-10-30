"""Solar-Doppler fix and L1B bad-sample exclusion in O2A and SCO2 (agreed-model fits), 2026-10-30.

Extension of wco2_lowsod_analysis.py to the other two bands, orbit 29252a_GL, same far-clear sample, same
channel mask (finite ln T, τ, σ > 0, finite τ_h2o), candidates:
  (1) agreed model as is, (2) solar lines at λ(1 + v_sol/c − v_inst/c) (α = 1, no fit parameter),
  (3) (2) + channels with L1B bad_sample_list ≠ 0 dropped.
Fits: final_model_test.fit_o2a (floor model, k2 = 0, free-k2 diagnostic) and fit_sco2 (two-τ, free k2; a2 over land).
Metrics as the WCO2 table: rms at SOD < 0.1 and all, k1 median, far-clear spread (per-footprint means removed),
within-cluster spread (per footprint × v_inst cluster medians removed), v_inst-cluster k1 step, QF0 near-cloud
contrasts 0–1 and 1–3 km with SNR; O2A floor f and free-k2 diagnostic; SCO2 k2 and k1_w. Also the SOD distribution
of the flagged channels per band.

Land cross-check (38200a_ND, Kozienice): bad-sample exclusion only (|v_inst| ≤ 0.06 km/s, no solar error):
background k1 scatter and median shift, plume Δk1 as kozienice_plume_retest.py (running-median anomaly vs clear
neighbours of the same footprint, ±25 soundings; plume = L2 bc > background + 2 ppm within 30 km, Lite used as
label only).

Usage: python wco2_lowsod_allbands.py [--koz-only]
Outputs: results/wco2_lowsod_allbands_29252a_GL.md, results/wco2_lowsod_allbands_kozienice.md,
         results/wco2_lowsod_allbands_29252a_GL.csv
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wco2_lowsod_common as wc  # noqa: E402
from wco2_lowsod_analysis import load, solar_delta  # noqa: E402

OUT = wc.OUT
fm = wc.fm
_G: dict = {}


# --------------------------------------------------------------------------- per-sounding fits


def fit_one(ib, y, t, tw, s, land, drop=None):
    """Agreed-model fit of one band; returns the record plus rms at SOD < 0.1 (residual rebuilt from the coefficients)."""
    ok = wc.base_mask(y, t, tw, s)
    if drop is not None:
        ok &= ~drop
    if ok.sum() < 20:
        return None
    t, tw, y, x = t[ok], tw[ok], y[ok], wc.X_ALL[ok]
    if ib == 0:
        r = fm.fit_o2a(t, x, y)
        model = r["c"] + r["a1"] * x + np.log((1 - r["f"]) * np.exp(-r["k1"] * t) + r["f"])
    else:
        r = fm.fit_sco2(t - tw, tw, x, y, land)
        model = (r["c"] + r["a1"] * x + (r["a2"] * x**2 if land else 0) - r["k1"] * (t - tw) - r["k1_w"] * tw
                 + 0.5 * r["k2"] * t**2)
    res = y - model
    lo = t < wc.LOW_SOD
    r["rms_low"] = np.sqrt(np.mean(res[lo] ** 2)) if lo.sum() > 10 else np.nan
    r["rms_all"] = np.sqrt(np.mean(res**2))
    r["nchan"] = int(ok.sum())
    return r


def _gl_chunk(js):
    out = []
    for j in js:
        for ib in (0, 2):
            y0 = _G["ln_t"][ib, j].astype(float)
            t, tw, s = _G["tau"][ib, j].astype(float), _G["tau_h2o"][ib, j].astype(float), _G["sig"][ib, j].astype(float)
            for cand, (dy, drop) in (("(1) agreed model", (None, None)),
                                     ("(2) solar fix α = 1", (_G["dS"][ib][j], None)),
                                     ("(3) solar fix + drop bad ≠ 0", (_G["dS"][ib][j], _G["bad"][ib][_G["fp"][j]] != 0))):
                r = fit_one(ib, y0 + (0 if dy is None else dy), t, tw, s, bool(_G["land"][j]), drop)
                if r is not None:
                    out.append(dict(j=j, band=("o2a", "wco2", "sco2")[ib], cand=cand, **r))
    return out


# --------------------------------------------------------------------------- metrics


def band_metrics(df, b, cand, meta, groups):
    n = len(meta["lat"])
    d = df[(df.band == b) & (df.cand == cand)].set_index("j")

    def col(c):
        v = np.full(n, np.nan)
        v[d.index.values] = d[c].astype(float).values
        return v
    k1 = col("k1")
    rec = wc.k1_metrics(k1, k1, meta, col("rms_low"), col("rms_all"), groups=groups)
    far = (meta["cld_dist_km"] > 20) & (meta["xco2_qf"] == 0)
    fp = meta["fp_number"].astype(int)
    if b == "o2a":
        f = col("f")
        rec.update(f_med=np.nanmedian(f[far]), f_rsd=wc.robust_sd(f[far] - pd.Series(f[far]).groupby(fp[far]).transform("median").values),
                   k2d_med=np.nanmedian(col("k2_diag")[far]), k2d_neg=np.nanmean(col("k2_diag")[far] < 0),
                   nchan=np.nanmedian(col("nchan")[far]))
    else:
        rec.update(k2_med=np.nanmedian(col("k2")[far]), k2_rsd=wc.robust_sd(col("k2")[far]),
                   k1w_med=np.nanmedian(col("k1_w")[far]), nchan=np.nanmedian(col("nchan")[far]))
    return rec


def flagged_sod_table(d, far, fp):
    rows = []
    for ib, b in enumerate(wc.BANDS):
        bad = d["ex"]["bad"][ib]                                  # [8, 1016]
        sod = np.stack([np.nanmedian(d["tau"][ib][far & (fp == f)], axis=0) for f in range(8)])
        for v in (1, 2, 3):
            m = bad == v
            if not m.any():
                continue
            s = sod[m]
            rows.append(f"| {b} | {v} | {m.sum() / 8:.1f} | {(s < 0.1).sum() / 8:.1f} | {((s >= 0.1) & (s < 1)).sum() / 8:.1f} | "
                        f"{(s >= 1).sum() / 8:.1f} | {(s >= 3).sum() / 8:.1f} | {np.nanmax(s):.2f} |")
    return rows


def gl_part():
    t0 = time.time()
    d = load()
    orb, meta = d["orb"], d["meta"]
    fp = orb.fp_number.astype(int)
    far = (meta["cld_dist_km"] > 20) & (meta["xco2_qf"] == 0)
    vcl = (d["ex"]["v_inst"] > 0).astype(int)
    groups = fp * 2 + vcl
    dS = {0: solar_delta(d, 0, 1.0), 2: solar_delta(d, 2, 1.0)}
    _G.update(ln_t=orb.ln_t, tau=d["tau"], tau_h2o=d["tau_h2o"], sig=orb.sigma_ln_t, dS=dS, bad=d["ex"]["bad"], fp=fp,
              land=meta["sfc_type"] == 1)
    chunks = np.array_split(np.arange(len(fp)), 400)
    with mp.get_context("fork").Pool(8) as pool:
        df = pd.DataFrame([r for part in pool.imap(_gl_chunk, chunks) for r in part])
    df.to_csv(OUT / "wco2_lowsod_allbands_29252a_GL.csv", index=False)
    print(f"GL fits in {time.time() - t0:.0f} s")
    cands = ("(1) agreed model", "(2) solar fix α = 1", "(3) solar fix + drop bad ≠ 0")
    recs = {(b, c): band_metrics(df, b, c, meta, groups) for b in ("o2a", "sco2") for c in cands}
    lines = ["# Solar-Doppler fix and bad-sample exclusion in O2A and SCO2 — 29252a_GL", "",
             f"Agreed-model fits (O2A floor model, k2 = 0; SCO2 two-τ, free k2), mask as final_model_test. Far-clear n = {far.sum()}; "
             "spread = robust SD (%) after per-footprint means; within = after per-(footprint × v_inst cluster) medians; step = "
             "k1 median of the v > 0 cluster over the v < 0 cluster − 1 (%); Δk1 = median relative contrast of QF0 soundings vs "
             "latitude-local far references, SNR = contrast / far-clear robust SD of the contrast. rms = median per-sounding rms, far-clear.", "",
             "## Main metrics", "",
             "| band | candidate | rms SOD<0.1 | rms all | k1 median | spread % | within % | step % | Δk1 0–1 km % (SNR) | Δk1 1–3 km % (SNR) | channels |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (b, c), r in recs.items():
        lines.append(f"| {b} | {c} | {r['rms_low_far']:.4f} | {r['rms_all_far']:.4f} | {r['k1_med']:.4f} | {r['k1_spread']:.3f} | "
                     f"{r['k1_spread_within']:.3f} | {r['cluster_step']:+.3f} | {r['con_0-1']:+.3f} ({r['snr_0-1']:+.2f}) | "
                     f"{r['con_1-3']:+.3f} ({r['snr_1-3']:+.2f}) | {r['nchan']:.0f} |")
    lines += ["", "## Band-specific terms (far-clear)", "",
              "| band | candidate | O2A f median | f robust SD (fp medians removed) | free-k2 diag median | frac k2_diag < 0 | SCO2 k2 median | k2 robust SD | k1_w median |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (b, c), r in recs.items():
        if b == "o2a":
            lines.append(f"| o2a | {c} | {r['f_med']:.4f} | {r['f_rsd']:.4f} | {r['k2d_med']:+.4f} | {r['k2d_neg']:.2f} | — | — | — |")
        else:
            lines.append(f"| sco2 | {c} | — | — | — | — | {r['k2_med']:+.4f} | {r['k2_rsd']:.4f} | {r['k1w_med']:.4f} |")
    lines += ["", f"QF0 near-cloud counts: 0–1 km {recs[('o2a', cands[0])]['n_0-1']}, 1–3 km {recs[('o2a', cands[0])]['n_1-3']}.", "",
              "## Where the flagged channels sit (median far-clear SOD per channel; channels per footprint, mean over 8)", "",
              "| band | bad_sample_list | channels | SOD < 0.1 | 0.1–1 | ≥ 1 | ≥ 3 | max SOD |", "|---|---|---|---|---|---|---|---|",
              *flagged_sod_table(d, far, fp), ""]
    (OUT / "wco2_lowsod_allbands_29252a_GL.md").write_text("\n".join(lines))
    print("\n".join(lines))


# --------------------------------------------------------------------------- Kozienice


def koz_part():
    from o2a_anomaly_estimator import running_anomaly, rsd
    z = dict(np.load(OUT / "kozienice_38200a_ND_spectra.npz", allow_pickle=True))
    df = pd.read_parquet(OUT / "kozienice_38200a_ND_per_sounding.parquet")
    fp = df.fp.values.astype(int)
    land = df.land.values.astype(bool)
    with h5py.File(wc.ROOT / "data/OCO2/2021/249/38200a_ND/oco2_L1bScND_38200a_210906_B11006r_220325042044.h5") as f:
        badall = f["InstrumentHeader/bad_sample_list"][...]
    bg = ((df.dist_km >= 40) & (df.qf == 0)).values
    order = np.argsort(df.along_km.values, kind="stable")
    clear = (df.cld_dist_km.values > 5.0) & np.isfinite(df.k1_o2a.values)
    qf0 = (df.qf == 0).values
    bgm = np.nanmedian(df.xco2_bc.values[bg])
    plume = (df.xco2_bc.values - bgm > 2) & (df.dist_km.values < 30)
    pl0 = plume & qf0
    lines = ["# Bad-sample exclusion in O2A and SCO2 — Kozienice 38200a_ND (land nadir)", "",
             f"No solar-Doppler error on this orbit (|v_inst| ≤ {np.abs(df.v_inst).max():.0f} m/s). Background = ≥ 40 km from the plant, "
             f"QF0 (n = {bg.sum()}). rSD = robust SD/median (%); spread = after per-footprint means. Plume Δk1 = running-median anomaly "
             f"(same footprint, ±25 soundings, clear neighbours = MODIS cloud distance > 5 km) as kozienice_plume_retest.py, median over "
             f"the plume soundings (L2 bc > background + 2 ppm within 30 km; n = {plume.sum()}, QF0 {pl0.sum()}). WCO2 rows for reference.", "",
             "| band | variant | rms SOD<0.1 | k1 median | median shift % | rSD % | spread % | plume Δk1 % all | plume Δk1 % QF0 | O2A f median | SCO2 k2 median |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ib, b in enumerate(wc.BANDS):
        base_med = None
        for lab, sel in (("as is", None), ("drop bad ≠ 0", (1, 2, 3)), ("drop bad = 1 only", (1,)),
                         ("drop bad = 2 only (edge)", (2,))):
            recs = []
            for j in range(len(fp)):
                drop = np.isin(badall[ib, fp[j]], sel) if sel is not None else None
                y, t, tw, s = (z[k][ib, j].astype(float) for k in ("ln_t", "tau", "tau_h2o", "sig"))
                if ib == 1:
                    ok = wc.base_mask(y, t, tw, s) & (True if drop is None else ~drop)
                    c, r = wc.fit_wco2_ext(t[ok] - tw[ok], tw[ok], wc.X_ALL[ok], y[ok])
                    lo = t[ok] < wc.LOW_SOD
                    rec = dict(k1=c[2], rms_low=np.sqrt(np.mean(r[lo] ** 2)), f=np.nan, k2=np.nan)
                else:
                    rec = fit_one(ib, y, t, tw, s, bool(land[j]), drop)
                recs.append(rec)
            k1 = np.array([r["k1"] for r in recs])
            k1b = k1[bg]
            base_med = np.median(k1b) if base_med is None else base_med
            tmp = df.copy()
            tmp["k1_tmp"] = k1
            an = running_anomaly(tmp, "k1_tmp", clear, order)
            fvals = np.array([r.get("f", np.nan) for r in recs])
            k2vals = np.array([r.get("k2", np.nan) for r in recs])
            lines.append(f"| {b} | {lab} | {np.nanmedian([r['rms_low'] for r in np.array(recs)[bg]]):.4f} | {np.median(k1b):.4f} | "
                         f"{100 * (np.median(k1b) / base_med - 1):+.2f} | {100 * rsd(k1b) / np.median(k1b):.3f} | "
                         f"{wc.spread_pct(k1b, fp[bg]):.3f} | {100 * np.nanmedian(an[plume]):+.2f} | {100 * np.nanmedian(an[pl0]):+.2f} | "
                         + (f"{np.nanmedian(fvals[bg]):.4f} | — |" if ib == 0 else (f"— | {np.nanmedian(k2vals[bg]):+.4f} |" if ib == 2 else "— | — |")))
    lines.append("")
    (OUT / "wco2_lowsod_allbands_kozienice.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    if "--koz-only" not in sys.argv:
        gl_part()
    koz_part()

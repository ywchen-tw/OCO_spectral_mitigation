"""Cross-check of the WCO2 low-SOD findings on the land nadir orbit 38200a_ND (Kozienice, 2021-09-06).

On nadir the spacecraft–surface velocity is tiny (|v_inst| ≤ 0.06 km/s), so the solar-Doppler error found on
the glint orbit is absent here (≤ 0.01 px). The script measures what is left: rms at SOD < 0.1, the fixed
per-footprint residual pattern (background soundings: ≥ 40 km from the plant, QF0), its correlation with the
solar-corrected glint pattern of 29252a_GL (same instrument, 20 months earlier), the residual of L1B-flagged
channels, and the k1 effect of dropping flagged channels or subtracting the 29252a_GL pattern.

Inputs: results/kozienice_38200a_ND_spectra.npz, results/kozienice_38200a_ND_per_sounding.parquet,
        results/wco2_lowsod_patterns_29252a_GL.npz (from wco2_lowsod_analysis.py), the 38200a_ND L1B.
Output: results/wco2_lowsod_kozienice_38200a_ND.md
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wco2_lowsod_common as wc  # noqa: E402

OUT = wc.OUT
L1B = wc.ROOT / "data/OCO2/2021/249/38200a_ND/oco2_L1bScND_38200a_210906_B11006r_220325042044.h5"


def fit_all(z, IB, dy=None, drop=None, fp=None):
    n = z["ln_t"].shape[1]
    k1 = np.full(n, np.nan)
    rlow = np.full(n, np.nan)
    R = np.full((n, 1016), np.nan)
    for j in range(n):
        y = z["ln_t"][IB, j].astype(float) + (0 if dy is None else dy[j])
        t, tw, s = (z[k][IB, j].astype(float) for k in ("tau", "tau_h2o", "sig"))
        ok = wc.base_mask(y, t, tw, s)
        if drop is not None:
            ok &= ~drop[j]
        if ok.sum() < 20:
            continue
        c, r = wc.fit_wco2_ext(t[ok] - tw[ok], tw[ok], wc.X_ALL[ok], y[ok])
        k1[j] = c[2]
        R[j, ok] = r
        lo = t[ok] < wc.LOW_SOD
        rlow[j] = np.sqrt(np.mean(r[lo] ** 2))
    return k1, rlow, R


def main():
    z = dict(np.load(OUT / "kozienice_38200a_ND_spectra.npz", allow_pickle=True))
    df = pd.read_parquet(OUT / "kozienice_38200a_ND_per_sounding.parquet")
    fp = df.fp.values.astype(int)
    bg = ((df.dist_km >= 40) & (df.qf == 0)).values
    IB = 1
    with h5py.File(L1B) as f:
        bad = f["InstrumentHeader/bad_sample_list"][IB][fp]
    gl = np.load(OUT / "wco2_lowsod_patterns_29252a_GL.npz")
    P_gl = gl["P1"]                                           # [8, 1016] solar-corrected far-clear pattern (GL)
    low = z["tau"][IB] < wc.LOW_SOD
    sig = z["sig"][IB].astype(float)
    k1_0, rlow0, R0 = fit_all(z, IB)
    m = bg[:, None] & np.isfinite(R0)
    lines = ["# WCO2 low-SOD cross-check — 38200a_ND (Kozienice, land nadir)", "",
             f"Background (≥ 40 km from the plant, QF0) n = {bg.sum()}; |v_inst| max {np.abs(df.v_inst).max():.0f} m/s "
             f"(solar-line shift ≤ {0.174 * np.abs(df.v_inst).max() / 1000:.3f} px, negligible).",
             f"WCO2 residual rms at SOD < 0.1: {np.sqrt(np.nanmean(R0[m & low] ** 2)):.4f} (noise median "
             f"{np.nanmedian(sig[m & low]):.4f}); all channels {np.sqrt(np.nanmean(R0[m] ** 2)):.4f}.", "",
             "| channel class | rms SOD < 0.1 | median abs |", "|---|---|---|"]
    for lab, sel in (("bad = 0", bad == 0), ("bad = 1", bad == 1), ("bad = 2 (edge)", bad == 2)):
        mm = m & low & sel
        lines.append(f"| {lab} | {np.sqrt(np.nanmean(R0[mm] ** 2)):.4f} | {np.nanmedian(np.abs(R0[mm])):.4f} |")
    # fixed-pattern share and correlation with the GL pattern
    rows = []
    tot = fix = noise = 0.0
    P_k = np.full((8, 1016), np.nan)
    for f in range(8):
        jj = np.flatnonzero(bg & (fp == f))
        rr = np.where(m[jj] & low[jj], R0[jj], np.nan)
        with np.errstate(all="ignore"):
            pat = np.nanmean(rr, axis=0)
        P_k[f] = np.nanmean(np.where(m[jj], R0[jj], np.nan), axis=0)
        t_ = np.nansum(rr**2)
        tot += t_
        fix += np.nansum((rr - pat) ** 2)
        noise += np.nansum(np.where(np.isfinite(rr), sig[jj] ** 2, 0))
        ch = np.isfinite(pat) & np.isfinite(P_gl[f]) & (gl["sod_med"][f] < wc.LOW_SOD) & (bad[jj[0]] == 0)
        rows.append(f"| {f + 1} | {len(jj)} | {1 - np.nansum((rr - pat) ** 2) / t_:.2f} | "
                    f"{np.nansum(np.where(np.isfinite(rr), sig[jj] ** 2, 0)) / t_:.2f} | "
                    f"{np.corrcoef(pat[ch], P_gl[f][ch])[0, 1]:+.2f} | {np.sqrt(np.nanmean(pat[ch] ** 2)):.4f} |")
    lines += ["", "Per-footprint fixed pattern (background, SOD < 0.1, in-sample) and its correlation with the solar-corrected "
              "29252a_GL pattern (unflagged channels with SOD < 0.1 in both):", "",
              "| fp | n | fixed share | noise share | r with 29252a_GL pattern | pattern rms |", "|---|---|---|---|---|---|",
              *rows, f"| pooled | {bg.sum()} | {1 - fix / tot:.2f} | {noise / tot:.2f} | | |", ""]
    # k1 effects
    k1s = df.k1_sco2.values
    core = (df.dist_km < 10).values

    def stats(k1, rl, lab):
        mm = bg & np.isfinite(k1) & np.isfinite(k1s)
        a = pd.Series(k1[mm]).groupby(fp[mm]).transform(lambda s: s - s.median()).values
        b = pd.Series(k1s[mm]).groupby(fp[mm]).transform(lambda s: s - s.median()).values
        return (f"| {lab} | {np.nanmedian(rl[bg]):.4f} | {np.median(k1[mm]):.4f} | {100 * wc.robust_sd(k1[mm]) / np.median(k1[mm]):.3f} | "
                f"{wc.spread_pct(k1[mm], fp[mm]):.3f} | {np.corrcoef(k1[mm], k1s[mm])[0, 1]:+.2f} | {np.corrcoef(a, b)[0, 1]:+.2f} | "
                f"{100 * (np.nanmedian(k1[core]) / np.median(k1[mm]) - 1):+.2f} |")
    P_gl_full = np.nan_to_num(P_gl)[fp]
    k1_b, rl_b, _ = fit_all(z, IB, drop=bad != 0)
    k1_b1, rl_b1, _ = fit_all(z, IB, drop=bad == 1)
    k1_b2, rl_b2, _ = fit_all(z, IB, drop=bad == 2)
    k1_p, rl_p, _ = fit_all(z, IB, dy=-P_gl_full)
    k1_bp, rl_bp, _ = fit_all(z, IB, dy=-P_gl_full, drop=bad != 0)
    lines += ["k1 (background): rSD = robust SD/median (%, as kozienice_38200a_ND.md); spread = after per-fp means; "
              "corr with SCO2 k1 (stored agreed-model fit) raw and after per-fp medians.", "",
              f"Plume = median k1 of all soundings within 10 km of the plant (n = {core.sum()}) over the background median − 1 (%); "
              f"SCO2 (stored): {100 * (np.nanmedian(k1s[core]) / np.nanmedian(k1s[bg]) - 1):+.2f} %.", "",
              "| variant | rms SOD<0.1 | k1 median | rSD % | spread % | corr SCO2 | corr SCO2 (fp-demeaned) | plume Δk1 % |",
              "|---|---|---|---|---|---|---|---|",
              stats(k1_0, rlow0, "baseline"), stats(k1_b, rl_b, "drop bad ≠ 0"),
              stats(k1_b1, rl_b1, "drop bad = 1 only"), stats(k1_b2, rl_b2, "drop bad = 2 only (edge 0–89)"),
              stats(k1_p, rl_p, "subtract 29252a_GL per-fp pattern"), stats(k1_bp, rl_bp, "drop bad ≠ 0 + GL pattern"), ""]
    (OUT / "wco2_lowsod_kozienice_38200a_ND.md").write_text("\n".join(lines))
    np.save(OUT / "wco2_lowsod_kozienice_pattern_38200a_ND.npy", P_k.astype(np.float32))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

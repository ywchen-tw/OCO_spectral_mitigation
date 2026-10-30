"""Does a free k2 make sense in O2A and WCO2 once the solar-Doppler error is removed? (2026-10-30)

29252a_GL, solar-corrected ln T (solar lines at λ(1 + v_sol/c − v_inst/c), α = 1), the revised-fit channel mask,
no channel dropping. All soundings are fitted (near-cloud contrasts need them); statistics on far-clear (n = 1240).
  WCO2 (a) c + a1x − k1 τ_dry − k1_w τ_h2o (k2 = 0, current)   (b) + ½k2 τ², k2 free   (c) same with k2 ≥ 0
  O2A  (a) c + a1x + ln[(1−f)e^{−k1τ} + f], k2 = 0 (current)  (b) floor with free k2: ln[(1−f)e^{−k1τ+½k2τ²} + f]
       (c) no floor: c + a1x − k1τ + ½k2τ²                     (d) no floor: (c) − k3τ³/6
WCO2 aliasing check: correlation of the free k2 with a1 and with k1 over far-clear soundings, before and after
the solar fix. Kozienice (38200a_ND background, no solar error): WCO2 variants only, plus the plume Δk1
(running-median anomaly as kozienice_plume_retest.py).

Usage: python wco2_lowsod_k2_after_fix.py
Output: results/wco2_lowsod_k2_after_fix.md
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares, lsq_linear  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wco2_lowsod_common as wc  # noqa: E402
from wco2_lowsod_analysis import load, solar_delta  # noqa: E402

OUT = wc.OUT
_G: dict = {}


# --------------------------------------------------------------------------- fitters


def wco2_variants(td, tw, x, y):
    t = td + tw
    a = np.column_stack([np.ones_like(td), x, -td, -tw, 0.5 * t**2])
    out = {}
    c = np.linalg.lstsq(a[:, :4], y, rcond=None)[0]
    out["a"] = dict(k1=c[2], k1_w=c[3], a1=c[1], k2=0.0, rms=np.sqrt(np.mean((y - a[:, :4] @ c) ** 2)))
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    out["b"] = dict(k1=c[2], k1_w=c[3], a1=c[1], k2=c[4], rms=np.sqrt(np.mean((y - a @ c) ** 2)))
    lb = np.full(5, -np.inf)
    lb[4] = 0.0
    r = lsq_linear(a, y, bounds=(lb, np.full(5, np.inf)), method="bvls")
    out["c"] = dict(k1=r.x[2], k1_w=r.x[3], a1=r.x[1], k2=r.x[4], rms=np.sqrt(np.mean(r.fun**2)), at_bound=r.x[4] <= 1e-12)
    return out


def o2a_variants(t, x, y):
    out = {}
    one = np.ones_like(t)
    c0 = np.linalg.lstsq(np.column_stack([one, x, -t]), y, rcond=None)[0]

    def res_floor(p, free_k2):
        k2 = p[4] if free_k2 else 0.0
        return p[0] + p[1] * x + np.log((1 - p[3]) * np.exp(-p[2] * t + 0.5 * k2 * t**2) + p[3]) - y
    p0 = [c0[0], c0[1], max(c0[2], 0.1), 0.01]
    r = least_squares(res_floor, p0, args=(False,), bounds=([-np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, 0.2]))
    out["a"] = dict(k1=r.x[2], a1=r.x[1], f=r.x[3], k2=0.0, rms=np.sqrt(np.mean(r.fun**2)), f_at_bound=r.x[3] >= 0.2 - 1e-9)
    r = least_squares(res_floor, list(r.x) + [0.0], args=(True,),
                      bounds=([-np.inf, -np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, 0.2, np.inf]))
    out["b"] = dict(k1=r.x[2], a1=r.x[1], f=r.x[3], k2=r.x[4], rms=np.sqrt(np.mean(r.fun**2)), f_at_bound=r.x[3] >= 0.2 - 1e-9)
    a = np.column_stack([one, x, -t, 0.5 * t**2])
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    out["c"] = dict(k1=c[2], a1=c[1], f=np.nan, k2=c[3], rms=np.sqrt(np.mean((y - a @ c) ** 2)))
    a = np.column_stack([one, x, -t, 0.5 * t**2, -t**3 / 6])
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    out["d"] = dict(k1=c[2], a1=c[1], f=np.nan, k2=c[3], k3=c[4], rms=np.sqrt(np.mean((y - a @ c) ** 2)))
    return out


def _chunk(js):
    rows = []
    for j in js:
        for ib, fitter in ((0, o2a_variants), (1, wco2_variants)):
            t, tw, s = _G["tau"][ib, j].astype(float), _G["tau_h2o"][ib, j].astype(float), _G["sig"][ib, j].astype(float)
            for lab, dy in (("solar", _G["dS"][ib][j]), ("as is", 0.0)):
                if ib == 0 and lab == "as is":
                    continue                                   # O2A: only the solar-corrected version is asked
                if ib == 1 and lab == "as is" and not _G["far"][j]:
                    continue                                   # WCO2 before-fix only for the aliasing check
                y = _G["ln_t"][ib, j].astype(float) + dy
                ok = wc.base_mask(y, t, tw, s)
                if ok.sum() < 20:
                    continue
                x = wc.X_ALL[ok]
                res = fitter(t[ok], x, y[ok]) if ib == 0 else fitter(t[ok] - tw[ok], tw[ok], x, y[ok])
                for v, r in res.items():
                    rows.append(dict(j=j, band=("o2a", "wco2")[ib], ln_t=lab, variant=v, **r))
    return rows


# --------------------------------------------------------------------------- tables


def metrics(df, band, var, meta, groups, lnt="solar"):
    n = len(meta["lat"])
    d = df[(df.band == band) & (df.variant == var) & (df.ln_t == lnt)].set_index("j")

    def col(c):
        v = np.full(n, np.nan)
        if c in d:
            v[d.index.values] = d[c].astype(float).values
        return v
    far = (meta["cld_dist_km"] > 20) & (meta["xco2_qf"] == 0)
    k1 = col("k1")
    rec = wc.k1_metrics(k1, k1, meta, groups=groups)
    k2 = col("k2")
    rec.update(rms=np.nanmedian(col("rms")[far]), k2_med=np.nanmedian(k2[far]), k2_neg=np.nanmean(k2[far] < 0),
               k2_rsd=wc.robust_sd(k2[far]), f_med=np.nanmedian(col("f")[far]), k3_med=np.nanmedian(col("k3")[far]),
               at_bound=np.nanmean(col("at_bound")[far]) if "at_bound" in d else np.nan,
               f_bound=np.nanmean(col("f_at_bound")[far]) if "f_at_bound" in d else np.nan,
               a1_med=np.nanmedian(col("a1")[far]))
    return rec, dict(k1=k1, k2=k2, a1=col("a1"), far=far)


def gl_part(lines):
    t0 = time.time()
    d = load()
    orb, meta = d["orb"], d["meta"]
    fp = orb.fp_number.astype(int)
    far = (meta["cld_dist_km"] > 20) & (meta["xco2_qf"] == 0)
    groups = fp * 2 + (d["ex"]["v_inst"] > 0).astype(int)
    _G.update(ln_t=orb.ln_t, tau=d["tau"], tau_h2o=d["tau_h2o"], sig=orb.sigma_ln_t, far=far,
              dS={0: solar_delta(d, 0, 1.0), 1: solar_delta(d, 1, 1.0)})
    with mp.get_context("fork").Pool(8) as pool:
        df = pd.DataFrame([r for part in pool.imap(_chunk, np.array_split(np.arange(len(fp)), 400)) for r in part])
    print(f"GL fits in {time.time() - t0:.0f} s")
    desc = {("wco2", "a"): "(a) k2 = 0 (current)", ("wco2", "b"): "(b) free k2", ("wco2", "c"): "(c) k2 ≥ 0",
            ("o2a", "a"): "(a) floor, k2 = 0 (current)", ("o2a", "b"): "(b) floor + free k2",
            ("o2a", "c"): "(c) no floor, order 2", ("o2a", "d"): "(d) no floor, order 2 + k3"}
    lines += ["## 29252a_GL, solar-corrected ln T (far-clear n = %d)" % far.sum(), "",
              "spread = robust SD (%) after per-footprint means; within = after per-(footprint × v_inst cluster) medians; "
              "Δk1 = median relative contrast of QF0 soundings vs latitude-local far references (SNR = contrast / far-clear "
              "robust SD of the contrast); rms = median per-sounding rms; k2 frac = fraction of far-clear soundings with k2 < 0 "
              "(for k2 ≥ 0: fraction at the bound).", "",
              "| band | variant | k1 median | spread % | within % | k2 median | k2 robust SD | k2 frac | f median | k3 median | rms | Δk1 0–1 km % (SNR) | Δk1 1–3 km % (SNR) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (b, v), lab in desc.items():
        r, _ = metrics(df, b, v, meta, groups)
        frac = r["at_bound"] if v == "c" and b == "wco2" else r["k2_neg"]
        k2txt = "0 (fixed)" if v == "a" else f"{r['k2_med']:+.4f}"
        k2sd = "—" if v == "a" else f"{r['k2_rsd']:.4f}"
        frtxt = "—" if v == "a" else f"{frac:.2f}"
        ftxt = "—" if np.isnan(r["f_med"]) else f"{r['f_med']:.4f}"
        k3txt = "—" if np.isnan(r["k3_med"]) else f"{r['k3_med']:+.4f}"
        lines.append(f"| {b} | {lab} | {r['k1_med']:.4f} | {r['k1_spread']:.3f} | {r['k1_spread_within']:.3f} | {k2txt} | "
                     f"{k2sd} | {frtxt} | {ftxt} | {k3txt} | {r['rms']:.4f} | {r['con_0-1']:+.3f} ({r['snr_0-1']:+.2f}) | "
                     f"{r['con_1-3']:+.3f} ({r['snr_1-3']:+.2f}) |")
    lines += ["", f"QF0 near-cloud counts: 0–1 km {r['n_0-1']}, 1–3 km {r['n_1-3']}. O2A floor at its 0.2 bound: "
              f"(a) {metrics(df, 'o2a', 'a', meta, groups)[0]['f_bound']:.3f}, (b) {metrics(df, 'o2a', 'b', meta, groups)[0]['f_bound']:.3f}.", ""]
    # aliasing check, WCO2 free k2
    lines += ["WCO2 aliasing check, free k2 (variant b), far-clear: Pearson r of k2 with a1 and with k1, raw and after "
              "per-(footprint × v_inst cluster) medians.", "",
              "| ln T | k2 median | frac k2 < 0 | r(k2, a1) | r(k2, k1) | r(k2, a1) within | r(k2, k1) within | k1 median (b) | k1 spread (b) % |",
              "|---|---|---|---|---|---|---|---|---|"]
    for lnt, lab in (("as is", "revised fit as is"), ("solar", "solar-Doppler corrected")):
        r, c = metrics(df, "wco2", "b", meta, groups, lnt)
        m = c["far"] & np.isfinite(c["k2"])
        g = groups[m]

        def dm(v):
            return pd.Series(v[m]).groupby(g).transform(lambda s: s - s.median()).values
        lines.append(f"| {lab} | {np.median(c['k2'][m]):+.4f} | {np.mean(c['k2'][m] < 0):.2f} | "
                     f"{np.corrcoef(c['k2'][m], c['a1'][m])[0, 1]:+.2f} | {np.corrcoef(c['k2'][m], c['k1'][m])[0, 1]:+.2f} | "
                     f"{np.corrcoef(dm(c['k2']), dm(c['a1']))[0, 1]:+.2f} | {np.corrcoef(dm(c['k2']), dm(c['k1']))[0, 1]:+.2f} | "
                     f"{r['k1_med']:.4f} | {r['k1_spread']:.3f} |")
    lines.append("")


def koz_part(lines):
    from o2a_anomaly_estimator import running_anomaly, rsd
    z = dict(np.load(OUT / "kozienice_38200a_ND_spectra.npz", allow_pickle=True))
    df = pd.read_parquet(OUT / "kozienice_38200a_ND_per_sounding.parquet")
    fp = df.fp.values.astype(int)
    bg = ((df.dist_km >= 40) & (df.qf == 0)).values
    order = np.argsort(df.along_km.values, kind="stable")
    clear = (df.cld_dist_km.values > 5.0) & np.isfinite(df.k1_o2a.values)
    bgm = np.nanmedian(df.xco2_bc.values[bg])
    plume = (df.xco2_bc.values - bgm > 2) & (df.dist_km.values < 30)
    pl0 = plume & (df.qf == 0).values
    res = {v: [] for v in "abc"}
    for j in range(len(fp)):
        y, t, tw, s = (z[k][1, j].astype(float) for k in ("ln_t", "tau", "tau_h2o", "sig"))
        ok = wc.base_mask(y, t, tw, s)
        for v, r in wco2_variants(t[ok] - tw[ok], tw[ok], wc.X_ALL[ok], y[ok]).items():
            res[v].append(r)
    lines += ["## Kozienice 38200a_ND background, WCO2 (no solar error; n = %d)" % bg.sum(), "",
              "rSD = robust SD/median (%); spread = after per-footprint means; plume Δk1 = running-median anomaly as "
              f"kozienice_plume_retest.py, median over plume soundings (n = {plume.sum()}, QF0 {pl0.sum()}).", "",
              "| variant | k1 median | rSD % | spread % | k2 median | k2 frac (< 0 / at bound) | rms | r(k2, a1) | r(k2, k1) | plume Δk1 % all | QF0 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, lab in (("a", "(a) k2 = 0 (current)"), ("b", "(b) free k2"), ("c", "(c) k2 ≥ 0")):
        k1 = np.array([r["k1"] for r in res[v]])
        k2 = np.array([r["k2"] for r in res[v]])
        a1 = np.array([r["a1"] for r in res[v]])
        rms = np.array([r["rms"] for r in res[v]])
        tmp = df.copy()
        tmp["k1_tmp"] = k1
        an = running_anomaly(tmp, "k1_tmp", clear, order)
        frac = np.mean([r["at_bound"] for r in np.array(res[v])[bg]]) if v == "c" else np.mean(k2[bg] < 0)
        if v == "a":
            cr, k2txt, frtxt = "— | —", "0 (fixed)", "—"
        else:
            cr = f"{np.corrcoef(k2[bg], a1[bg])[0, 1]:+.2f} | {np.corrcoef(k2[bg], k1[bg])[0, 1]:+.2f}"
            k2txt, frtxt = f"{np.median(k2[bg]):+.4f}", f"{frac:.2f}"
        lines.append(f"| {lab} | {np.median(k1[bg]):.4f} | {100 * rsd(k1[bg]) / np.median(k1[bg]):.3f} | {wc.spread_pct(k1[bg], fp[bg]):.3f} | "
                     f"{k2txt} | {frtxt} | {np.median(rms[bg]):.4f} | {cr} | {100 * np.nanmedian(an[plume]):+.2f} | "
                     f"{100 * np.nanmedian(an[pl0]):+.2f} |")
    lines.append("")


def main():
    lines = ["# Free k2 in O2A and WCO2 after the solar-Doppler fix (2026-10-30)", ""]
    gl_part(lines)
    koz_part(lines)
    (OUT / "wco2_lowsod_k2_after_fix.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

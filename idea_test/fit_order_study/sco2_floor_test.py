"""Can SCO2 separate an additive floor from the path-curvature term k2? (2026-10-30)

Four SCO2 fit variants per sounding on the same channel mask (wc.base_mask). All share the continuum
c + a1 x (+ a2 x² over land), x = X_ALL, and the two-τ path part with τ = τ_dry + τ_h2o:
    (a) current     ln T = cont − k1 τ_dry − k1_w τ_h2o + ½ k2 τ²                 linear, k2 free
    (b) floor only  ln T = cont + ln[(1 − f) e^{−k1 τ_dry − k1_w τ_h2o} + f]       k1 ≥ 0, f ∈ [0, 0.2], k2 = 0
    (c) floor + k2  ln T = cont + ln[(1 − f) e^{−k1 τ_dry − k1_w τ_h2o + ½k2τ²} + f]  same bounds, k2 free
    (d) linear      ln T = cont − k1 τ_dry − k1_w τ_h2o                             linear, no k2, no floor
(b) starts from the linear fit (d) as final_model_test.fit_o2a does (k1 ≥ 0.1, f = 0.01). (c) is started twice,
from the (b) solution with k2 = 0 and from the (a) solution with f = 1e-3, and keeps the lower cost.
1-σ formal errors from final_model_test._cov_se (design matrix as the Jacobian for the linear variants).

Samples
    29252a_GL: all soundings, solar-Doppler-corrected ln T (wco2_lowsod_analysis.solar_delta, α = 1, band 2);
               statistics on far-clear (cld > 20 km, QF0); near-cloud contrasts via wc.k1_metrics.
    38200a_ND (Kozienice, land nadir): ln T as stored; background (dist ≥ 40 km, QF0) statistics and the plume
               Δk1 from the running-median anomaly (o2a_anomaly_estimator.running_anomaly, as
               wco2_lowsod_k2_after_fix.koz_part).

Usage: python sco2_floor_test.py
Output: results/sco2_floor_test.md, results/sco2_floor_test_29252a_GL.parquet,
        results/sco2_floor_test_38200a_ND.parquet
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wco2_lowsod_common as wc  # noqa: E402
from final_model_test import _cov_se  # noqa: E402

OUT = wc.OUT
IB = 2                      # SCO2
F_MAX = 0.2
VARIANTS = ("a", "b", "c", "d")
LABELS = {"a": "(a) ½k2τ², k2 free (current)", "b": "(b) floor, k2 = 0", "c": "(c) floor + free k2",
          "d": "(d) linear, no k2, no floor"}
PARAMS = ("c", "a1", "a2", "k1", "k1_w", "k2", "f")
_G: dict = {}


# --------------------------------------------------------------------------- fitters


def _record(names, p, se, resid, ok=True):
    rec = {k: np.nan for k in PARAMS}
    rec.update({f"se_{k}": np.nan for k in PARAMS})
    for k, v, e in zip(names, p, se):
        rec[k], rec[f"se_{k}"] = float(v), float(e)
    rec["rms"] = float(np.sqrt(np.mean(resid**2))) if ok else np.nan
    rec["ok"] = ok
    return rec


def _failed(names):
    rec = _record(names, np.full(len(names), np.nan), np.full(len(names), np.nan), np.array([np.nan]), ok=False)
    return rec


def _linear(cols, names, y):
    a = np.column_stack(cols)
    p = np.linalg.lstsq(a, y, rcond=None)[0]
    r = y - a @ p
    return p, _record(names, p, _cov_se(a, r), r)


def _nonlinear(fun, p0, lb, ub, names):
    """Bounded least_squares; returns (result or None, record). Failure = exception, status ≤ 0 or non-finite x."""
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            r = least_squares(fun, p0, bounds=(lb, ub))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None, _failed(names)
    if r.status <= 0 or not np.all(np.isfinite(r.x)) or not np.all(np.isfinite(r.fun)):
        return None, _failed(names)
    return r, _record(names, r.x, _cov_se(r.jac, r.fun), r.fun)


def fit_variants(td, tw, x, y, land):
    """Variants (a)–(d) for one sounding; returns {variant: record}."""
    t = td + tw
    cont = [np.ones_like(td), x] + ([x**2] if land else [])
    cnames = ["c", "a1"] + (["a2"] if land else [])
    nc = len(cont)
    cmat = np.column_stack(cont)
    out = {}

    pd_, out["d"] = _linear(cont + [-td, -tw], cnames + ["k1", "k1_w"], y)
    out["d"]["k2"] = 0.0
    pa, out["a"] = _linear(cont + [-td, -tw, 0.5 * t**2], cnames + ["k1", "k1_w", "k2"], y)

    def res(p, free_k2):
        k1, k1w, f = p[nc], p[nc + 1], p[nc + 2]
        k2 = p[nc + 3] if free_k2 else 0.0
        return cmat @ p[:nc] + np.log((1 - f) * np.exp(-k1 * td - k1w * tw + 0.5 * k2 * t**2) + f) - y

    lb = [-np.inf] * nc + [0.0, -np.inf, 0.0]
    ub = [np.inf] * nc + [np.inf, np.inf, F_MAX]
    names_b = cnames + ["k1", "k1_w", "f"]
    p0 = list(pd_[:nc]) + [max(pd_[nc], 0.1), pd_[nc + 1], 0.01]
    rb, out["b"] = _nonlinear(lambda p: res(p, False), p0, lb, ub, names_b)
    out["b"]["k2"] = 0.0
    tmax = float(t.max())
    if rb is not None:
        # k2 equivalent of the floor at the deepest channel: ½k2τ_max² = ln(1 + f/T_min), T_min = e^{−k1 τ_max}
        with np.errstate(divide="ignore"):
            out["b"]["k2_equiv"] = 2 * np.logaddexp(0.0, np.log(out["b"]["f"]) + out["b"]["k1"] * tmax) / tmax**2
    out["b"]["tau_max"] = tmax

    names_c = names_b + ["k2"]
    starts = []
    if rb is not None:
        starts.append(("b", list(rb.x) + [0.0]))
    if np.all(np.isfinite(pa)):
        starts.append(("a", list(pa[:nc]) + [max(pa[nc], 0.0), pa[nc + 1], 1e-3, pa[nc + 2]]))
    best, best_rec, best_start, costs = None, _failed(names_c), "", {}
    for lab, p0 in starts:
        r, rec = _nonlinear(lambda p: res(p, True), p0, lb + [-np.inf], ub + [np.inf], names_c)
        if r is not None:
            costs[lab] = r.cost
            if best is None or r.cost < best.cost:
                best, best_rec, best_start = r, rec, lab
    best_rec["start"] = best_start
    # relative cost difference between the two starts; > 1e-6 means they reached different minima
    best_rec["cost_diff_starts"] = (costs["b"] - costs["a"]) / min(costs.values()) if len(costs) == 2 else np.nan
    out["c"] = best_rec
    for v in ("b", "c"):
        out[v]["f_at_bound"] = bool(out[v]["f"] >= F_MAX - 1e-9) if out[v]["ok"] else np.nan
    return out


def _chunk(js):
    rows = []
    for j in js:
        y, t, tw, s = (_G[k][j].astype(float) for k in ("y", "tau", "tau_h2o", "sig"))
        ok = wc.base_mask(y, t, tw, s)
        if ok.sum() < 20:
            continue
        res = fit_variants(t[ok] - tw[ok], tw[ok], wc.X_ALL[ok], y[ok], bool(_G["land"][j]))
        for v, r in res.items():
            rows.append(dict(j=int(j), variant=v, land=bool(_G["land"][j]), nchan=int(ok.sum()), **r))
    return rows


def run_fits(y, tau, tau_h2o, sig, land, nchunks):
    _G.clear()
    _G.update(y=y, tau=tau, tau_h2o=tau_h2o, sig=sig, land=land)
    with mp.get_context("fork").Pool(8) as pool:
        rows = [r for part in pool.imap(_chunk, np.array_split(np.arange(len(land)), nchunks)) for r in part]
    df = pd.DataFrame(rows)
    for c in ("ok", "f_at_bound"):
        df[c] = df[c].astype(float)
    return df


# --------------------------------------------------------------------------- helpers


def col(df, v, c, n):
    d = df[df.variant == v]
    out = np.full(n, np.nan)
    if c in d:
        out[d.j.values] = d[c].astype(float).values
    return out


def fmt(v, spec):
    return "—" if v is None or not np.isfinite(v) else format(v, spec)


def frac(v, cond):
    m = np.isfinite(v)
    return np.mean(cond(v[m])) if m.any() else np.nan


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return np.corrcoef(a[m], b[m])[0, 1]


def demean(v, groups):
    return pd.Series(v).groupby(groups).transform(lambda s: s - s.median()).values


def failures(df):
    d = df[df.variant.isin(["b", "c"])]
    return {v: int((d.variant == v).sum() - d[d.variant == v].ok.sum()) for v in ("b", "c")}


# --------------------------------------------------------------------------- 29252a_GL


def gl_part(lines):
    from wco2_lowsod_analysis import load, solar_delta
    t0 = time.time()
    d = load()
    orb, meta = d["orb"], d["meta"]
    fp = orb.fp_number.astype(int)
    n = len(fp)
    far = (meta["cld_dist_km"] > 20) & (meta["xco2_qf"] == 0)
    groups = fp * 2 + (d["ex"]["v_inst"] > 0).astype(int)
    dS = solar_delta(d, IB, 1.0)
    y = orb.ln_t[IB].astype(float) + dS
    t_load = time.time() - t0
    df = run_fits(y, d["tau"][IB], d["tau_h2o"][IB], orb.sigma_ln_t[IB], meta["sfc_type"] == 1, 400)
    t_fit = time.time() - t0 - t_load
    print(f"GL: load {t_load:.0f} s, fits {t_fit:.0f} s ({n} soundings)")
    df.to_parquet(OUT / "sco2_floor_test_29252a_GL.parquet", index=False)

    keys = ("k1", "k1_w", "k2", "f", "a1", "rms", "se_k2", "se_f", "f_at_bound", "k2_equiv", "tau_max", "ok",
            "cost_diff_starts")
    c = {v: {k: col(df, v, k, n) for k in keys} for v in VARIANTS}
    fails = failures(df)
    nfit = int((df.variant == "a").sum())
    split = np.abs(c["c"]["cost_diff_starts"]) > 1e-6
    lines += [f"## 29252a_GL, SCO2, solar-corrected ln T (soundings fitted {nfit}; far-clear n = {int(far.sum())})", "",
              "Statistics over far-clear soundings (cld_dist_km > 20, QF0). spread = robust SD (%) after per-footprint "
              "means; within = after per-(footprint × v_inst sign) medians; Δk1 = median relative contrast of QF0 "
              "soundings vs latitude-local far references, SNR = contrast / far-clear robust SD of the contrast "
              "(wc.k1_metrics). f at bound = f ≥ 0.2 − 1e-9; f at zero = f < 1e-4. se = 1-σ formal error.", ""]
    head = ["variant", "k1 median", "spread %", "within %", "k1_w median", "k2 median", "k2 robust SD",
            "frac k2 < 0", "f median", "f robust SD", "frac f at bound", "frac f at zero", "median se_k2",
            "median se_f", "rms median", "Δk1 0–1 km % (SNR)", "Δk1 1–3 km % (SNR)"]
    lines += ["### Table 1. Far-clear statistics", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    k1m = {}
    for v in VARIANTS:
        cv = c[v]
        r = wc.k1_metrics(cv["k1"], cv["k1"], meta, groups=groups)
        k1m[v] = r
        k2, f = cv["k2"][far], cv["f"][far]
        free_k2, floor = v in ("a", "c"), v in ("b", "c")
        cells = [LABELS[v], fmt(r["k1_med"], ".4f"), fmt(r["k1_spread"], ".3f"), fmt(r["k1_spread_within"], ".3f"),
                 fmt(np.nanmedian(cv["k1_w"][far]), ".4f"),
                 fmt(np.nanmedian(k2), "+.4f") if free_k2 else "0 (fixed)",
                 fmt(wc.robust_sd(k2), ".4f") if free_k2 else "—",
                 fmt(frac(k2, lambda u: u < 0), ".3f") if free_k2 else "—",
                 fmt(np.nanmedian(f), ".4f") if floor else "—",
                 fmt(wc.robust_sd(f), ".4f") if floor else "—",
                 fmt(np.nanmean(cv["f_at_bound"][far]), ".3f") if floor else "—",
                 fmt(frac(f, lambda u: u < 1e-4), ".3f") if floor else "—",
                 fmt(np.nanmedian(cv["se_k2"][far]), ".4f") if free_k2 else "—",
                 fmt(np.nanmedian(cv["se_f"][far]), ".4f") if floor else "—",
                 fmt(np.nanmedian(cv["rms"][far]), ".4f"),
                 f"{r['con_0-1']:+.3f} ({r['snr_0-1']:+.2f})", f"{r['con_1-3']:+.3f} ({r['snr_1-3']:+.2f})"]
        lines.append("| " + " | ".join(cells) + " |")
    r = k1m["a"]
    nfar_ok = {v: int(np.isfinite(c[v]["k1"][far]).sum()) for v in VARIANTS}
    lines += ["", f"QF0 near-cloud counts: 0–1 km {r['n_0-1']}, 1–3 km {r['n_1-3']}. Far-clear soundings with a valid "
              f"fit: " + ", ".join(f"({v}) {nfar_ok[v]}" for v in VARIANTS)
              + ". least_squares failures (all soundings): "
              f"(b) {fails['b']}, (c) {fails['c']}. Variant (c): the two starts reached costs differing by > 1e-6 "
              f"(relative) in {int(split.sum())} of {nfit} soundings ({int((split & far).sum())} far-clear); the "
              f"lower-cost start is kept.", ""]

    # Table 2: degeneracy
    m = far
    g = groups

    def r_pair(a, b):
        ok = m & np.isfinite(a) & np.isfinite(b)
        raw = corr(a[ok], b[ok])
        within = corr(demean(a[ok], g[ok]), demean(b[ok], g[ok]))
        return f"{fmt(raw, '+.3f')} / {fmt(within, '+.3f')}"

    cc = c["c"]
    lines += ["### Table 2. Degeneracy, far-clear", "",
              "Pearson r: raw / after subtracting per-(footprint × v_inst sign) medians. Differences and ratios are "
              "medians over far-clear soundings of the per-sounding values.", "",
              "| quantity | value |", "|---|---|",
              f"| r(f, k2), variant (c) | {r_pair(cc['f'], cc['k2'])} |",
              f"| r(f, k1), variant (c) | {r_pair(cc['f'], cc['k1'])} |",
              f"| r(k2, k1), variant (c) | {r_pair(cc['k2'], cc['k1'])} |",
              f"| r(k2 of (a), f of (b)) | {r_pair(c['a']['k2'], c['b']['f'])} |",
              f"| median k2(a) − k2(c) | {fmt(np.nanmedian((c['a']['k2'] - cc['k2'])[m]), '+.4f')} |",
              f"| median k1(a) − k1(b) | {fmt(np.nanmedian((c['a']['k1'] - c['b']['k1'])[m]), '+.4f')} |",
              f"| median k1(a) − k1(c) | {fmt(np.nanmedian((c['a']['k1'] - cc['k1'])[m]), '+.4f')} |",
              f"| median rms(b)/rms(a) | {fmt(np.nanmedian((c['b']['rms'] / c['a']['rms'])[m]), '.4f')} |",
              f"| median rms(c)/rms(a) | {fmt(np.nanmedian((cc['rms'] / c['a']['rms'])[m]), '.4f')} |",
              f"| median rms(d)/rms(a) | {fmt(np.nanmedian((c['d']['rms'] / c['a']['rms'])[m]), '.4f')} |",
              f"| median 2·ln(1 + f/T_min)/τ_max², variant (b), T_min = exp(−k1 τ_max) | "
              f"{fmt(np.nanmedian(c['b']['k2_equiv'][m]), '+.4f')} |",
              f"| median k2 of (a) (same soundings) | "
              f"{fmt(np.nanmedian(c['a']['k2'][m & np.isfinite(c['b']['k2_equiv'])]), '+.4f')} |",
              f"| median τ_max (masked channels) | {fmt(np.nanmedian(c['b']['tau_max'][m]), '.3f')} |", ""]
    return dict(fails=fails, t_load=t_load, t_fit=t_fit, n=nfit)


# --------------------------------------------------------------------------- Kozienice 38200a_ND


def koz_part(lines):
    from o2a_anomaly_estimator import running_anomaly, rsd
    t0 = time.time()
    z = dict(np.load(OUT / "kozienice_38200a_ND_spectra.npz", allow_pickle=True))
    tab = pd.read_parquet(OUT / "kozienice_38200a_ND_per_sounding.parquet")
    n = len(tab)
    fp = tab.fp.values.astype(int)
    bg = ((tab.dist_km >= 40) & (tab.qf == 0)).values
    order = np.argsort(tab.along_km.values, kind="stable")
    clear = (tab.cld_dist_km.values > 5.0) & np.isfinite(tab.k1_o2a.values)
    bgm = np.nanmedian(tab.xco2_bc.values[bg])
    plume = (tab.xco2_bc.values - bgm > 2) & (tab.dist_km.values < 30)
    pl0 = plume & (tab.qf == 0).values
    df = run_fits(z["ln_t"][IB], z["tau"][IB], z["tau_h2o"][IB], z["sig"][IB], tab.land.values.astype(bool), 64)
    t_fit = time.time() - t0
    print(f"Kozienice: fits {t_fit:.0f} s ({n} soundings)")
    df.to_parquet(OUT / "sco2_floor_test_38200a_ND.parquet", index=False)
    fails = failures(df)
    head = ["variant", "k1 median", "rSD %", "spread %", "k2 median", "frac k2 < 0", "f median", "frac f at bound",
            "frac f at zero", "rms median", "plume Δk1 % all", "plume Δk1 % QF0", "running-median bg rSD %"]
    lines += [f"## Kozienice 38200a_ND, SCO2, ln T as stored (soundings fitted {int((df.variant == 'a').sum())}; "
              f"background n = {int(bg.sum())}; land {int(tab.land.sum())})", "",
              "Background = dist_km ≥ 40 & QF0. rSD = robust SD / median (%); spread = robust SD (%) after "
              "per-footprint means. Plume Δk1 = running-median anomaly (same footprint, ±25 soundings, clear "
              "neighbours cld_dist_km > 5, "
              f"as wco2_lowsod_k2_after_fix.koz_part), median over plume soundings (n = {int(plume.sum())}, "
              f"QF0 {int(pl0.sum())}); running-median bg rSD = rsd of that anomaly over background soundings.", "",
              "### Table 3. Kozienice background", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in VARIANTS:
        k1, k2, f, rms = (col(df, v, k, n) for k in ("k1", "k2", "f", "rms"))
        fb = col(df, v, "f_at_bound", n)
        tmp = tab.copy()
        tmp["k1_tmp"] = k1
        # only differs from koz_part when a fit failed (NaN k1 must not enter the neighbour median)
        an = running_anomaly(tmp, "k1_tmp", clear & np.isfinite(k1), order)
        free_k2, floor = v in ("a", "c"), v in ("b", "c")
        kb = k1[bg][np.isfinite(k1[bg])]
        cells = [LABELS[v], fmt(np.median(kb), ".4f"), fmt(100 * rsd(kb) / np.median(kb), ".3f"),
                 fmt(wc.spread_pct(k1[bg], fp[bg]), ".3f"),
                 fmt(np.nanmedian(k2[bg]), "+.4f") if free_k2 else "0 (fixed)",
                 fmt(frac(k2[bg], lambda u: u < 0), ".3f") if free_k2 else "—",
                 fmt(np.nanmedian(f[bg]), ".4f") if floor else "—",
                 fmt(np.nanmean(fb[bg]), ".3f") if floor else "—",
                 fmt(frac(f[bg], lambda u: u < 1e-4), ".3f") if floor else "—",
                 fmt(np.nanmedian(rms[bg]), ".4f"),
                 f"{100 * np.nanmedian(an[plume]):+.2f}", f"{100 * np.nanmedian(an[pl0]):+.2f}",
                 fmt(100 * rsd(an[bg]), ".3f")]
        lines.append("| " + " | ".join(cells) + " |")
    split = (np.abs(col(df, "c", "cost_diff_starts", n)) > 1e-6).sum()
    lines += ["", f"least_squares failures: (b) {fails['b']}, (c) {fails['c']}. Variant (c): the two starts reached "
              f"costs differing by > 1e-6 (relative) in {int(split)} soundings; the lower-cost start is kept.", ""]
    return dict(fails=fails, t_fit=t_fit)


def main():
    t0 = time.time()
    lines = ["# SCO2: additive floor vs path-curvature k2 (2026-10-30)", "",
             "Variants (same channel mask, continuum c + a1 x (+ a2 x² over land), τ = τ_dry + τ_h2o): " +
             "; ".join(LABELS[v] for v in VARIANTS) + ". Floor fits: least_squares, k1 ≥ 0, f ∈ [0, 0.2]; "
             "(b) starts from the linear fit (d); (c) starts from (b) with k2 = 0 and from (a) with f = 1e-3, "
             "lower cost kept.", ""]
    gl = gl_part(lines)
    kz = koz_part(lines)
    wall = time.time() - t0
    lines += ["## Timing", "", f"29252a_GL: load {gl['t_load']:.0f} s, fits {gl['t_fit']:.0f} s ({gl['n']} soundings, "
              f"8 workers); Kozienice fits {kz['t_fit']:.0f} s; total wall {wall:.0f} s.", ""]
    text = "\n".join(lines)
    (OUT / "sco2_floor_test.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

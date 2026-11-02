"""Round 5 (2026-11-01): per-band "direct-path atom" retrieval, no cross-band path link.

Premise: T(nu) = int p_b(l) exp(-g tau(nu) l) dl leaves g degenerate with an arbitrary per-band
PPDF; the only absolute anchor is the direct surface-reflected beam (l = 1 exactly).  So per band
    p_b(l) = (1 - eps) delta(l - 1) + eps q_b(l),   q_b = gamma(mean theta, shape kappa)
and, per band independently (same channels, x coordinate and T convention as the _am fits):
    ln T = c + a1 x [+ a2 x^2 for SCO2 land] + ln{ (1 - f) [ (1-eps) e^{-t} + eps (1 + t theta/kappa)^-kappa ] + f }
    t = g tau_dry + h tau_h2o   (CO2 bands; tau_dry = tau - tau_h2o as in _am, i.e. incl. Rayleigh)
    t = s tau                   (O2A, total tau as in the O2A _am fit)
H2O choice: the diffuse part uses the same gamma path distribution for dry gas and H2O (one PPDF
per band applied to the band's total scaled gas tau), the direct part is exact for both.
Free per band: g (s), h, eps in [0, 1], theta in [0.05, 5] (shortening or lengthening), kappa
(log, [0.3, 1000]), c, a1 (a2); f in [0, 0.2] only in the "_f" variant (O2A; f can absorb SIF /
zero-level offset, so O2A is reported with and without it).
Bounded TRF least squares (analytic Jacobian), 9 starts (g0 x (eps0, theta0)); multimodality =
distinct g-hat (> 1 ppm-equivalent apart) among starts within delta-chi2 < 1 of the best.
Profile likelihood: g fixed on a grid around g-hat (+-2 %), others refitted; chi2 = RSS / s2_best
(s2_best = RSS/(n-p) at the free optimum, i.e. residual-scaled, as the _am sig_*).  "Well-defined
minimum": the grid minimum is at g-hat (within 0.5 in chi2) and delta-chi2 >= 1 is reached on
both sides inside +-2 %; sigma_g = half-width of the delta-chi2 = 1 interval.

Stages
  slab   regenerate the 2000 test scenes (run_phys.slab_scene; identical spectra), fit all bands
         (noise-free and noisy) -> results_phys/round5_slab.parquet
  real   2018-07-01, the round-3 sample (1000 + 1000 far-cloud QF0): observed ln T from L1B
         radiance / toa_sol (fitting.py convention), tau from fp_tau_combined.h5; bands given by
         --bands -> results_phys/round5_real_<date>.parquet
  report tables -> results_phys/round5_*.csv
xco2_bc only for scoring; Lite fields only as diagnostics; no ref_/r15_/r05_ input.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results_phys"
RD = Path("/scratch/alpine/yuch8913/oco/spec_anal/results_doppler")
L1B_DIR = Path("/scratch/alpine/yuch8913/oco/spec_anal/data/OCO2")
PPM = 0.0025                     # 0.25 % = 1 ppm
G_STARTS = (0.97, 1.0, 1.03)
ET_STARTS = ((0.1, 0.5), (0.1, 2.0), (0.4, 1.0))
PROF = np.array([-0.02, -0.015, -0.01, -0.0075, -0.005, -0.0025, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02])


# ── model ───────────────────────────────────────────────────────────────────────

class AtomModel:
    """Parameter vector: [c, a1, (a2), g, (h), eps, theta, logk, (f)]."""

    def __init__(self, tau_d, tau_w, x, y, x2=False, water=True, use_f=False):
        self.td, self.tw, self.x, self.y = tau_d, tau_w, x, y
        self.x2, self.water, self.use_f = x2, water, use_f
        names = ["c", "a1"] + (["a2"] if x2 else []) + ["g"] + (["h"] if water else []) + \
            ["eps", "theta", "logk"] + (["f"] if use_f else [])
        self.names = names
        self.ix = {n: i for i, n in enumerate(names)}
        lb = {"c": -np.inf, "a1": -np.inf, "a2": -np.inf, "g": 0.9, "h": 0.3, "eps": 0.0,
              "theta": 0.05, "logk": np.log(0.3), "f": 0.0}
        ub = {"c": np.inf, "a1": np.inf, "a2": np.inf, "g": 1.1, "h": 3.0, "eps": 1.0,
              "theta": 5.0, "logk": np.log(1000.0), "f": 0.2}
        self.lb = np.array([lb[n] for n in names])
        self.ub = np.array([ub[n] for n in names])

    def unpack(self, p):
        g = lambda n, d=0.0: p[self.ix[n]] if n in self.ix else d
        return g("c"), g("a1"), g("a2"), g("g"), g("h", 1.0), g("eps"), g("theta"), g("logk"), g("f")

    def resid_jac(self, p, want_jac=True):
        c, a1, a2, g, h, eps, th, lk, f = self.unpack(p)
        k = np.exp(lk)
        t = g * self.td + (h * self.tw if self.water else 0.0)
        D = np.exp(-t)
        u = t * th / k
        lu = np.log1p(u)
        Q = np.exp(-k * lu)
        M = (1 - eps) * D + eps * Q
        Mp = (1 - f) * M + f
        Mp = np.maximum(Mp, 1e-300)
        r = c + a1 * self.x + (a2 * self.x ** 2 if self.x2 else 0.0) + np.log(Mp) - self.y
        if not want_jac:
            return r, None
        J = np.empty((len(r), len(p)))
        dyM = (1 - f) / Mp
        Q1 = np.exp(-(k + 1) * lu)                    # (1+u)^(-k-1)
        dMdt = -(1 - eps) * D - eps * th * Q1
        J[:, self.ix["c"]] = 1.0
        J[:, self.ix["a1"]] = self.x
        if self.x2:
            J[:, self.ix["a2"]] = self.x ** 2
        J[:, self.ix["g"]] = dyM * dMdt * self.td
        if self.water:
            J[:, self.ix["h"]] = dyM * dMdt * self.tw
        J[:, self.ix["eps"]] = dyM * (Q - D)
        J[:, self.ix["theta"]] = dyM * eps * (-t * Q1)
        J[:, self.ix["logk"]] = dyM * eps * Q * k * (-lu + u / (1 + u))
        if self.use_f:
            J[:, self.ix["f"]] = (1 - M) / Mp
        return r, J

    def start(self, g0, eps0, th0, k0=5.0):
        tt = g0 * self.td + (self.tw if self.water else 0.0)
        cols = [np.ones_like(self.x), self.x] + ([self.x ** 2] if self.x2 else [])
        c0 = np.linalg.lstsq(np.column_stack(cols), self.y + tt, rcond=None)[0]
        v = {"c": c0[0], "a1": c0[1], "a2": c0[2] if self.x2 else 0.0, "g": g0, "h": 1.0,
             "eps": eps0, "theta": th0, "logk": np.log(k0), "f": 0.01}
        return np.clip(np.array([v[n] for n in self.names]), self.lb + 1e-9, self.ub - 1e-9)

    def fit(self, p0, fix_g=None):
        free = np.ones(len(p0), bool)
        if fix_g is not None:
            free[self.ix["g"]] = False
            p0 = p0.copy()
            p0[self.ix["g"]] = fix_g

        def fun(q):
            p = p0.copy()
            p[free] = q
            return self.resid_jac(p, False)[0]

        def jac(q):
            p = p0.copy()
            p[free] = q
            return self.resid_jac(p, True)[1][:, free]

        x0 = np.clip(p0[free], self.lb[free] + 1e-12, self.ub[free] - 1e-12)
        with np.errstate(all="ignore"):
            r = least_squares(fun, x0, jac=jac, bounds=(self.lb[free], self.ub[free]), method="trf",
                              x_scale="jac", max_nfev=300, ftol=1e-10, xtol=1e-10)
        p = p0.copy()
        p[free] = r.x
        return p, float(2 * r.cost), bool(r.success or r.status > 0)


def fit_band(model, g_true=None):
    """Multi-start free fit + profile likelihood.  Returns a flat dict."""
    t0 = time.perf_counter()
    n = len(model.y)
    sols = []
    for g0 in G_STARTS:
        for e0, th0 in ET_STARTS:
            try:
                p, rss, ok = model.fit(model.start(g0, e0, th0))
                sols.append((rss, p, ok))
            except Exception:  # noqa: BLE001
                pass
    if not sols:
        return dict(ok=0)
    sols.sort(key=lambda s: s[0])
    rss0, p0, ok0 = sols[0]
    s2 = rss0 / max(n - len(p0), 1)
    gi = model.ix["g"]
    near = [s for s in sols if (s[0] - rss0) / max(s2, 1e-300) < 1.0]
    gs = np.array([s[1][gi] for s in near])
    out = dict(ok=int(ok0), n_starts_ok=len(sols), rms=float(np.sqrt(rss0 / n)),
               n_modes=int(1 + np.sum(np.abs(np.diff(np.sort(gs))) > PPM)) if len(gs) else 0,
               g_spread_ppm=float((gs.max() - gs.min()) / PPM) if len(gs) else np.nan,
               g_start_range_ppm=float((max(s[1][gi] for s in sols) - min(s[1][gi] for s in sols)) / PPM))
    for nm, v in zip(model.names, p0):
        out[nm] = float(v)
    out["kappa"] = float(np.exp(out.pop("logk")))
    out["ms_free"] = (time.perf_counter() - t0) * 1e3
    # profile likelihood around g-hat
    ghat = p0[gi]
    chi = np.full(len(PROF), np.nan)
    pp = p0.copy()
    for side in (np.where(PROF < 0)[0][::-1], np.where(PROF > 0)[0]):
        pp = p0.copy()
        for j in side:
            gv = ghat + PROF[j]
            if not (model.lb[gi] <= gv <= model.ub[gi]):
                continue
            best = None
            for st in (pp, p0):
                try:
                    q, rss, _ = model.fit(st, fix_g=gv)
                except Exception:  # noqa: BLE001
                    continue
                if best is None or rss < best[1]:
                    best = (q, rss)
            if best is not None:
                pp = best[0]
                chi[j] = (best[1] - rss0) / s2
    out["prof_min_dchi2"] = float(np.nanmin(chi)) if np.isfinite(chi).any() else np.nan
    left, right = chi[PROF < 0][::-1], chi[PROF > 0]

    def half(dc, dg):
        k = np.where(dc >= 1.0)[0]
        if not len(k):
            return np.nan
        k = k[0]
        d0, g0_ = (0.0, 0.0) if k == 0 else (dc[k - 1], abs(dg[k - 1]))
        return g0_ + (1.0 - d0) / (dc[k] - d0) * (abs(dg[k]) - g0_)
    hl = half(left, PROF[PROF < 0][::-1])
    hr = half(right, PROF[PROF > 0])
    out["sig_g_prof_ppm"] = float(0.5 * (hl + hr) / PPM) if np.isfinite(hl) and np.isfinite(hr) else np.nan
    out["well_defined"] = int(np.isfinite(hl) and np.isfinite(hr) and out["prof_min_dchi2"] > -0.5)
    for j, dg in enumerate(PROF):
        out[f"prof_{dg:+.4f}"] = float(chi[j])
    out["ms_total"] = (time.perf_counter() - t0) * 1e3
    return out


# ── slab ────────────────────────────────────────────────────────────────────────

def slab_one(args):
    import phys_ppdf as pp
    import run_phys as rp
    import slab_mc as sm
    truth, bands, yc, yn, mu0, muv, land = rp.slab_scene(args)
    row = dict(truth)
    for b in ("o2a", "wco2", "sco2"):
        tau = bands[b]["tau"].astype(float)
        fh = np.nan_to_num(bands[b]["fh"])
        ok = np.isfinite(tau)
        row[f"{b}_taumax"] = float(np.nanmax(tau))
        for tag, ys in (("", yc), ("n_", yn)):
            y = ys[b][ok]
            if b == "o2a":
                variants = (("", False), ("_f", True))
                td, tw, water = tau[ok], np.zeros(ok.sum()), False
            else:
                variants = (("", False),)
                td, tw, water = (tau * (1 - fh))[ok], (tau * fh)[ok], True
            for vt, use_f in variants:
                m = AtomModel(td, tw, sm.X[ok], y, x2=(b == "sco2" and land), water=water, use_f=use_f)
                res = fit_band(m)
                row.update({f"{tag}{b}{vt}_{k}": v for k, v in res.items()})
    return row


def stage_slab(args):
    import run_phys as rp
    import slab_mc as sm
    bank = sm.load_tau_bank()
    jobs = rp.slab_jobs(bank)[rp.N_TRAIN:]
    if args.n:
        jobs = jobs[:args.n]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(slab_one, jobs, chunksize=2))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"round5_slab{args.tag}.parquet", index=False)
    print("done", len(df), flush=True)


# ── real ────────────────────────────────────────────────────────────────────────

def read_real(d, date, bands):
    """Observed ln T [n, 1014] per band (fitting.py convention: T = pi rad / (0.5 toa_sol),
    T > 1 -> NaN) and tau / frac_h2o, matched by sounding id."""
    import h5py
    out = {}
    for orb, g in d.groupby("orbit_id"):
        tf = RD / date / orb / "fp_tau_combined.h5"
        doy = pd.Timestamp(date).dayofyear
        l1 = sorted((L1B_DIR / date[:4] / f"{doy:03d}" / orb).glob("oco2_L1bSc*.h5"))
        if not tf.exists() or not l1:
            print(f"  {orb}: missing tau or L1B", flush=True)
            continue
        want = g.fp_id.astype(np.int64).to_numpy()
        with h5py.File(tf, "r") as f:
            sid = f["sounding_id"][...].astype(np.int64)
            pos = pd.Series(np.arange(len(sid)), index=sid)
            pos = pos[~pos.index.duplicated(keep="first")]
            rows = pos.reindex(want).to_numpy()
            okr = np.isfinite(rows)
            rr = rows[okr].astype(int)
            o = np.argsort(rr)
            tau = {b: f[f"{b}_tau_output"][rr[o], 1:-1] for b in bands}
            fh = {b: f[f"{b}_frac_h2o"][rr[o], 1:-1] for b in bands}
            sol = {b: f[f"{b}_toa_sol_output"][rr[o], 1:-1] * 0.5 for b in bands}
        with h5py.File(l1[0], "r") as f:
            ls = f["SoundingGeometry/sounding_id"][...].astype(np.int64)
            flat = pd.Series(np.arange(ls.size), index=ls.reshape(-1))
            flat = flat[~flat.index.duplicated(keep=False)]
            li = flat.reindex(want[okr][o]).to_numpy().astype(int)
            tr, fp = np.unravel_index(li, ls.shape)
            name = {"o2a": "radiance_o2", "wco2": "radiance_weak_co2", "sco2": "radiance_strong_co2"}
            rad = {}
            for b in bands:
                ds = f[f"SoundingMeasurements/{name[b]}"]
                rad[b] = np.stack([ds[t, p, 1:-1] for t, p in zip(tr, fp)])
        for k, s_id in enumerate(want[okr][o]):
            e = {}
            for b in bands:
                with np.errstate(all="ignore"):
                    T = rad[b][k] / sol[b][k] * np.pi
                    T[T > 1] = np.nan
                    e[b] = dict(lnT=np.where(T > 0, np.log(T), np.nan), tau=tau[b][k], fh=fh[b][k])
            out[int(s_id)] = e
        print(f"  {orb}: {okr.sum()}/{len(g)}", flush=True)
    return out


X_FULL = (np.arange(1, 1015) - 507.5) / 507.5


def real_one(args):
    from spectral.cumulant_fit import _fit_am_o2a, _fit_am_sco2, _fit_am_wco2
    row, dat, bands = args
    out = {"fp_id": row["fp_id"]}
    land = bool(np.isfinite(row["sco2_a2_am"]))
    for b in bands:
        e = dat[b]
        ok = np.isfinite(e["lnT"]) & np.isfinite(e["tau"])
        tau, y, x = e["tau"][ok].astype(float), e["lnT"][ok].astype(float), X_FULL[ok]
        fh = np.nan_to_num(e["fh"][ok].astype(float))
        s = np.argsort(tau)
        tau, y, x, fh = tau[s], y[s], x[s], fh[s]
        out[f"{b}_nchan_obs"] = int(ok.sum())
        # check: the src _am fit on the same ln T reproduces the stored k1_am
        if b == "o2a":
            out["chk_o2a_k1_am"] = _fit_am_o2a(tau, x, y)["k1_am"]
            variants = (("", False), ("_f", True))
            td, tw, water = tau, np.zeros_like(tau), False
        else:
            td, tw = tau * (1 - fh), tau * fh
            fn = _fit_am_wco2(td, tw, x, y) if b == "wco2" else _fit_am_sco2(td, tw, x, y, land)
            out[f"chk_{b}_k1_am"] = fn["k1_am"]
            variants = (("", False),)
            water = True
        for vt, use_f in variants:
            m = AtomModel(td, tw, x, y, x2=(b == "sco2" and land), water=water, use_f=use_f)
            out.update({f"{b}{vt}_{k}": v for k, v in fit_band(m).items()})
    return out


def stage_real(args):
    bands = tuple(args.bands)
    d = pd.read_parquet(OUT / f"real_{args.date}.parquet")
    keep = [c for c in d.columns if not c.startswith(("r_", "Fprior_", "sigchk", "corr_formal"))]
    d = d[keep]
    dat = read_real(d, args.date, bands)
    jobs = [(r, dat[int(r["fp_id"])], bands) for r in d.to_dict("records") if int(r["fp_id"]) in dat]
    print(f"fitting {len(jobs)} soundings, bands {bands}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(real_one, jobs, chunksize=2))
    out = d.merge(pd.DataFrame(res), on="fp_id", how="inner")
    out.to_parquet(OUT / f"round5_real_{args.date}{args.tag}.parquet", index=False)
    print("done", len(out), flush=True)


# ── report ──────────────────────────────────────────────────────────────────────

def _stats(err, d, extra):
    from path_transfer import rsd
    e = err[np.isfinite(err)]
    return dict(n=int(e.size), rsd_ppm=rsd(e), med_bias_ppm=float(np.median(e)) if e.size else np.nan,
                frac_abs_gt_10ppm=float(np.mean(np.abs(e) > 10)) if e.size else np.nan, **extra)


def band_extra(d, pre, m):
    g = lambda c: d[pre + c][m] if pre + c in d else pd.Series(np.nan, index=d.index[m])
    return dict(conv=float(g("ok").mean()), well_defined=float(g("well_defined").mean()),
                multimodal=float((g("n_modes") > 1).mean()),
                start_range_gt1ppm=float((g("g_start_range_ppm") > 1).mean()),
                med_sig_prof_ppm=float(g("sig_g_prof_ppm").median()),
                eps_med=float(g("eps").median()), theta_med=float(g("theta").median()),
                theta_at_bound=float(((g("theta") <= 0.0501) | (g("theta") >= 4.999)).mean()),
                kappa_at_bound=float(((g("kappa") <= 0.3001) | (g("kappa") >= 999.9)).mean()),
                eps_zero=float((g("eps") < 1e-4).mean()),
                ms_med=float(g("ms_total").median()), ms_free_med=float(g("ms_free").median()))


def stage_report(args):
    from path_transfer import SLAB_SUBSETS, corr, rsd
    sp = OUT / "round5_slab.parquet"
    if sp.exists():
        d = pd.read_parquet(sp)
        rows, dep = [], []
        for tag, noise in (("", "noise-free"), ("n_", "noisy")):
            for b in ("o2a", "o2a_f", "wco2", "sco2"):
                pre = f"{tag}{b}_"
                if pre + "g" not in d:
                    continue
                gt = 1.0 if b.startswith("o2a") else d.g_co2
                err = ((d[pre + "g"] - gt) / PPM).to_numpy()
                for sub, fn in SLAB_SUBSETS.items():
                    m = fn(d)
                    rows.append(dict(noise=noise, band=b, subset=sub, **_stats(err[m], d, band_extra(d, pre, m))))
                # sigma_g / error vs eps-hat and vs max channel tau
                eb = pd.cut(d[pre + "eps"], [-1, 0.003, 0.01, 0.03, 0.1, 0.3, 1.01])
                tb = pd.qcut(d["o2a_taumax" if b.startswith("o2a") else f"{b}_taumax"], 4)
                for lab, bins in (("eps_hat", eb), ("taumax", tb)):
                    for k, idx in d.groupby(bins, observed=True).groups.items():
                        m = d.index.isin(idx)
                        dep.append(dict(noise=noise, band=b, by=lab, bin=str(k), n=int(m.sum()),
                                        rsd_err_ppm=rsd(err[m]), med_sig_prof_ppm=float(d[pre + "sig_g_prof_ppm"][m].median()),
                                        well_defined=float(d[pre + "well_defined"][m].mean())))
        t = pd.DataFrame(rows).round(3)
        t.to_csv(OUT / "round5_slab_floor.csv", index=False)
        pd.DataFrame(dep).round(3).to_csv(OUT / "round5_slab_sigma_dependence.csv", index=False)
        sub = ["all", "clear (no cloud, tau_a<0.1)", "cloudy", "land", "ocean", "no cloud, tau_a<0.05",
               "no cloud, 0.1-0.2", "no cloud, 0.2-0.5"]
        print(t[t.subset.isin(sub)].pivot_table(index=["noise", "band"], columns="subset", values="rsd_ppm")[sub].round(2).to_string())
        print(t[t.subset == "all"][["noise", "band", "conv", "well_defined", "multimodal", "start_range_gt1ppm",
                                    "med_sig_prof_ppm", "eps_med", "theta_at_bound", "kappa_at_bound", "eps_zero", "ms_med"]].to_string(index=False))
        print(t[t.subset == "clear (no cloud, tau_a<0.1)"][["noise", "band", "rsd_ppm", "med_bias_ppm", "well_defined",
                                                             "med_sig_prof_ppm", "frac_abs_gt_10ppm"]].to_string(index=False))
        print(pd.DataFrame(dep).round(2).to_string(index=False))
        pcols = [c for c in d.columns if c.startswith("n_sco2_prof_")]
        d.loc[:19, ["i", "g_co2", "land", "cloudy", "tau_a", "n_sco2_g", "n_sco2_eps"] + pcols].to_csv(
            OUT / "round5_slab_profile_sample_sco2_noisy.csv", index=False)
    rp_ = OUT / f"round5_real_{args.date}.parquet"
    if rp_.exists():
        d = pd.read_parquet(rp_)
        rows = []
        for b in [v for v in ("o2a", "o2a_f", "wco2", "sco2") if f"{v}_g" in d]:
            for s_, g in d.groupby("sfc_type"):
                m = (d.sfc_type == s_).to_numpy()
                sh = g[f"{b}_g"]
                r = dict(band=b, surface={0.0: "ocean", 1.0: "land"}[s_], n=len(g),
                         med_s_minus_1_pct=float((sh.median() - 1) * 100), rsd_s_pct=rsd(sh) * 100,
                         rsd_s_ppm_eq=rsd(sh) / PPM, q10_q90_pct=f"{(sh.quantile(.1)-1)*100:.2f} / {(sh.quantile(.9)-1)*100:.2f}",
                         **band_extra(d, f"{b}_", m))
                if b.startswith("o2a"):
                    r["corr_s_psfc_ratio"] = corr(sh, g.psfc / g.psfc_prior)
                    r["rsd_psfc_ratio_pct"] = rsd(g.psfc / g.psfc_prior) * 100
                    r["corr_s_vs_k1am"] = corr(sh, g.o2a_k1_am)
                    r["rsd_k1am_pct"] = rsd(g.o2a_k1_am) * 100
                    r["chk_k1_maxdiff"] = float(np.nanmax(np.abs(g.chk_o2a_k1_am - g.o2a_k1_am)))
                r["spearman_eps_aod"] = float(g[f"{b}_eps"].corr(g.aod_total, method="spearman"))
                r["spearman_s_aod"] = float(sh.corr(g.aod_total, method="spearman"))
                if f"{b}_f" in g:
                    r["f_med"] = float(g[f"{b}_f"].median())
                if not b.startswith("o2a"):
                    pri, bc = g.xco2_apriori.to_numpy(float), g.xco2_bc.to_numpy(float)
                    est = pri * sh.to_numpy()
                    r.update(rsd_vs_bc=rsd(est - bc), bias_vs_bc=float(np.mean(est - bc)),
                             rsd_vs_prior=rsd(est - pri), rsd_bc_vs_prior=rsd(bc - pri),
                             corr_bcdev=corr(est - pri, bc - pri))
                rows.append(r)
        t = pd.DataFrame(rows).round(4)
        t.to_csv(OUT / f"round5_real_{args.date}_table.csv", index=False)
        pd.set_option("display.width", 250)
        print(t.T.to_string())
        pc = [c for c in d.columns if c.startswith("o2a_prof_")]
        pd.concat([d[d.sfc_type == 0].head(10), d[d.sfc_type == 1].head(10)])[
            ["fp_id", "sfc_type", "o2a_g", "o2a_eps", "o2a_theta", "o2a_kappa"] + pc].to_csv(
            OUT / f"round5_real_{args.date}_profile_sample_o2a.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["slab", "real", "report"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--date", default="2018-07-01")
    ap.add_argument("--bands", nargs="+", default=["o2a"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    {"slab": stage_slab, "real": stage_real, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()

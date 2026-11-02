"""Round 4 (2026-11-01): layer-free, non-ML path-transfer retrieval on the _am coefficients.

No tau files and no spectrum synthesis: the method works on the fitted coefficients only.

Observables (per sounding):
  k1_O = s_p <l>_O (O2A k1_am),  k1_W = g <l>_W,  k1_S = g <l>_S (dry k1_am),
  A_b  = band continuum reflectance factor exp(c_am_b) / K (K = mu0^2 for the OCO-2 T of fitting.py,
         N_PHOT/pi in the slab MC).  Only band RATIOS of A matter for g: a common scale kappa is
         absorbed by c_L -> c_L kappa, c_S -> c_S / kappa.  The absolute scale enters only the
         Rayleigh variant (c).
Path model (first-order scattering, layer-free):
  delta_b = <l>_b - 1 = w_b (c_L A_b - c_S / A_b),  w_b = (lambda_b / 0.765)^-alpha,
  c_S = shortening kernel, c_L = lengthening kernel (each includes tau_s(765)), common to all bands.
Per sounding, alpha fixed:
  eq1: delta_O = k1_O - 1                              (s_p = 1)
  eq2: delta_S - r delta_W = r - 1,  r = k1_S / k1_W
  -> 2x2 linear system in (c_L, c_S); g = k1_S / (1 + delta_S)  (= k1_W / (1 + delta_W) by construction).
Variants
  (a) alpha in {0, 0.5, 1, 1.5, 2}
  (b) one-kernel baseline: delta_b = delta_O w_b (albedo ignored), g = k1_S / (1 + delta_O w_S)
      (and the WCO2 analogue g_W = k1_W / (1 + delta_O w_W))
  (c) Rayleigh single-scattering shortening removed first: delta_b -> delta_b - delta_R,b with
      delta_R,b = -0.5 eps_b / (1 + eps_b), eps_b = tau_R,b P_R(Theta) / (4 mu0 muv A_b)
      (Rayleigh scatterers uniform in pressure, so their mean relative O2/CO2 path is 1/2);
      tau_R from the Hansen-Travis fit scaled by psfc_prior / 1013.25.  Real data only (the slab
      has no Rayleigh).
  (d) alpha free with SCO2 k2diag/k1^2 as a third equation.  Extra assumption: each kernel is a
      single path shift of fixed size, Delta_S = Delta_L = Delta (= 1 slant column), so
      var(l)_b = Delta w_b (c_L A_b + c_S / A_b) and k2/k1^2 = var(l)_S / (1 + delta_S)^2.
      alpha is found by a grid search (0..3, step 0.05) on that equation.
Error propagation: linear (numerical derivatives) from sig_k1_am of the three bands -> sigma_g.

Stages:
  slabgen  regenerate the 2000 round-2/3 slab test scenes (run_phys.slab_scene; identical RNG
           stream) and store the src _am fit coefficients incl. sig_* -> results_phys/round4_slab_fits.parquet
  slab     apply the method -> results_phys/round4_slab_*.csv
  real     2018-07-01, the same 1000 + 1000 far-cloud QF0 soundings as round 3 (read from
           results_phys/real_2018-07-01.parquet) -> results_phys/round4_real_*.csv
xco2_bc is used only for scoring.  Lite fields only as diagnostics.  No ref_/r15_/r05_ input.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results_phys"
WL = {"o2a": 0.765, "wco2": 1.61, "sco2": 2.06}
ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0)
COND_FLAG = 1e3
DELTA_VAR = 1.0
N_PHOT = 2000


def rsd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 20 else np.nan


# ── the method (vectorised over soundings) ──────────────────────────────────────

def w(b, alpha):
    return (WL[b] / 0.765) ** (-alpha)


def solve_two_kernel(k1O, k1W, k1S, AO, AW, AS, alpha, dR=None):
    """Two-kernel path transfer.  dR = dict band -> known Rayleigh delta (subtracted) or None.
    Returns dict(g, cL, cS, dO, dW, dS, cond)."""
    dRO, dRW, dRS = (dR["o2a"], dR["wco2"], dR["sco2"]) if dR is not None else (0.0, 0.0, 0.0)
    wO, wW, wS = w("o2a", alpha), w("wco2", alpha), w("sco2", alpha)
    r = k1S / k1W
    # aerosol part: delta_b = dR_b + w_b (cL A_b - cS / A_b)
    # eq1: wO (cL AO - cS/AO) = k1O - 1 - dRO
    # eq2: [dRS + wS(..S)] - r [dRW + wW(..W)] = r - 1
    a11, a12, b1 = wO * AO, -wO / AO, k1O - 1 - dRO
    a21 = wS * AS - r * wW * AW
    a22 = -(wS / AS - r * wW / AW)
    b2 = r - 1 - dRS + r * dRW
    det = a11 * a22 - a12 * a21
    cL = (b1 * a22 - a12 * b2) / det
    cS = (a11 * b2 - a21 * b1) / det
    dO = dRO + wO * (cL * AO - cS / AO)
    dW = dRW + wW * (cL * AW - cS / AW)
    dS = dRS + wS * (cL * AS - cS / AS)
    # 2-norm condition number of [[a11, a12], [a21, a22]]
    fro2 = a11 ** 2 + a12 ** 2 + a21 ** 2 + a22 ** 2
    ad = np.abs(det)
    disc = np.sqrt(np.maximum(fro2 ** 2 - 4 * ad ** 2, 0))
    smax, smin = np.sqrt((fro2 + disc) / 2), np.sqrt(np.maximum((fro2 - disc) / 2, 0))
    cond = np.where(smin > 0, smax / np.where(smin > 0, smin, 1), np.inf)
    return dict(g=k1S / (1 + dS), cL=cL, cS=cS, dO=dO, dW=dW, dS=dS, cond=cond)


def one_kernel(k1O, k1W, k1S, alpha, dR=None):
    dRO, dRW, dRS = (dR["o2a"], dR["wco2"], dR["sco2"]) if dR is not None else (0.0, 0.0, 0.0)
    x = k1O - 1 - dRO
    return dict(g=k1S / (1 + dRS + x * w("sco2", alpha)), gW=k1W / (1 + dRW + x * w("wco2", alpha)))


def sigma_g(fun, k1O, k1W, k1S, sO, sW, sS, eps=1e-5):
    """Linear propagation of sig_k1 (independent bands) through g = fun(k1O, k1W, k1S)."""
    g0 = fun(k1O, k1W, k1S)
    v = np.zeros_like(g0)
    for i, s in enumerate((sO, sW, sS)):
        args = [k1O, k1W, k1S]
        args[i] = args[i] + eps
        v += ((fun(*args) - g0) / eps * s) ** 2
    return np.sqrt(v)


def alpha_free(k1O, k1W, k1S, AO, AW, AS, shapeS, dR=None, grid=np.arange(0, 3.0001, 0.05)):
    """Variant (d): alpha per sounding from SCO2 k2diag/k1^2 = Delta wS (cL AS + cS/AS) / (1+dS)^2."""
    res = np.stack([shape_resid(solve_two_kernel(k1O, k1W, k1S, AO, AW, AS, a, dR), AS, a, shapeS)
                    for a in grid])
    j = np.nanargmin(np.abs(np.where(np.isfinite(res), res, np.inf)), axis=0)
    a_hat = grid[j]
    out = solve_two_kernel(k1O, k1W, k1S, AO, AW, AS, a_hat, dR)
    out["alpha"] = a_hat
    out["shape_resid"] = res[j, np.arange(len(j))]
    return out


def shape_resid(sol, AS, alpha, shapeS):
    var = DELTA_VAR * w("sco2", alpha) * (sol["cL"] * AS + sol["cS"] / AS)
    return var / (1 + sol["dS"]) ** 2 - shapeS


def rayleigh_delta(A, mu0, muv, cos_sca, ps):
    """Known Rayleigh single-scattering delta per band (eps from thin-layer single scattering)."""
    out = {}
    pr = 0.75 * (1 + cos_sca ** 2)
    for b, lam in WL.items():
        tau_r = 0.008569 * lam ** -4 * (1 + 0.0113 * lam ** -2 + 0.00013 * lam ** -4) * ps / 1013.25
        eps = tau_r * pr / (4 * mu0 * muv * A[b])
        out[b] = -0.5 * eps / (1 + eps)
        out[f"eps_{b}"] = eps
    return out


def run_all(df, k1, sig, A, dR=None, shapeS=None):
    """All estimators for a frame.  k1/sig/A: dict band -> array.  Returns dict of columns."""
    O, W, S = k1["o2a"], k1["wco2"], k1["sco2"]
    cols = {}
    t0 = time.perf_counter()
    for tagR, rr in (("", None), ("_ray", dR)):
        if tagR and rr is None:
            continue
        for a in ALPHAS:
            sol = solve_two_kernel(O, W, S, A["o2a"], A["wco2"], A["sco2"], a, rr)
            key = f"TK_a{a:g}{tagR}"
            cols[f"g_{key}"] = sol["g"]
            cols[f"cond_{key}"] = sol["cond"]
            cols[f"cL_{key}"], cols[f"cS_{key}"] = sol["cL"], sol["cS"]
            for b, d in (("o2a", "dO"), ("wco2", "dW"), ("sco2", "dS")):
                cols[f"{d}_{key}"] = sol[d]
            fun = lambda o, w_, s, a=a, rr=rr: solve_two_kernel(o, w_, s, A["o2a"], A["wco2"], A["sco2"], a, rr)["g"]
            cols[f"sig_{key}"] = sigma_g(fun, O, W, S, sig["o2a"], sig["wco2"], sig["sco2"])
            ok = one_kernel(O, W, S, a, rr)
            cols[f"g_OK_a{a:g}{tagR}"], cols[f"gW_OK_a{a:g}{tagR}"] = ok["g"], ok["gW"]
            fun1 = lambda o, w_, s, a=a, rr=rr: one_kernel(o, w_, s, a, rr)["g"]
            cols[f"sig_OK_a{a:g}{tagR}"] = sigma_g(fun1, O, W, S, sig["o2a"], sig["wco2"], sig["sco2"])
        if shapeS is not None:
            sol = alpha_free(O, W, S, A["o2a"], A["wco2"], A["sco2"], shapeS, rr)
            cols[f"g_TKfree{tagR}"], cols[f"alpha_TKfree{tagR}"] = sol["g"], sol["alpha"]
            cols[f"cond_TKfree{tagR}"] = sol["cond"]
            cols[f"shres_TKfree{tagR}"] = sol["shape_resid"]
    cols["_ms_per_sounding_all_variants"] = np.full(len(O), (time.perf_counter() - t0) * 1e3 / len(O))
    t0 = time.perf_counter()
    for _ in range(20):
        solve_two_kernel(O, W, S, A["o2a"], A["wco2"], A["sco2"], 1.0)
    cols["_us_per_sounding_one_solve"] = np.full(len(O), (time.perf_counter() - t0) * 1e6 / 20 / len(O))
    return cols


# ── stage: slabgen (Slurm: Monte Carlo) ─────────────────────────────────────────

def slabgen_one(args):
    import phys_ppdf as pp
    import run_phys as rp
    import slab_mc as sm
    truth, bands, yc, yn, mu0, muv, land = rp.slab_scene(args)
    cos_sca = np.sqrt(1 - mu0 ** 2) * np.sqrt(1 - muv ** 2) - mu0 * muv
    row = dict(truth, mu0=mu0, muv=muv, cos_sca=cos_sca)
    sc0 = pp.make_scene(mu0, muv, cos_sca, sm.N_PHOT / np.pi, land, True,
                        {b: dict(bands[b], c_obs=0.0) for b in pp.BANDS})
    for tag, ys in (("", yc), ("n_", yn)):
        fits = rp.fit_all(sc0, {b: ys[b][sc0["b"][b]["order"]] for b in pp.BANDS})
        for b in pp.BANDS:
            for k, v in fits[b].items():
                row[f"{tag}{b}_{k}"] = v
    return row


def stage_slabgen(args):
    import run_phys as rp
    import slab_mc as sm
    bank = sm.load_tau_bank()
    jobs = rp.slab_jobs(bank)[rp.N_TRAIN:]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(slabgen_one, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    old = pd.read_parquet(OUT / "slab_phys_scenes.parquet", columns=["i", "sco2_k1_am", "n_o2a_k1_am"])
    j = df.merge(old, on="i", suffixes=("", "_r3"))
    print("reproduction vs round 3: max|d sco2_k1|", np.abs(j.sco2_k1_am - j.sco2_k1_am_r3).max(),
          "max|d n_o2a_k1|", np.abs(j.n_o2a_k1_am - j.n_o2a_k1_am_r3).max(), flush=True)
    df.to_parquet(OUT / "round4_slab_fits.parquet", index=False)


# ── stage: slab ─────────────────────────────────────────────────────────────────

SLAB_SUBSETS = {
    "all": lambda d: np.ones(len(d), bool),
    "clear (no cloud, tau_a<0.1)": lambda d: ((d.cloudy == 0) & (d.tau_a < 0.1)).to_numpy(),
    "cloudy": lambda d: (d.cloudy == 1).to_numpy(),
    "land": lambda d: (d.land == 1).to_numpy(),
    "ocean": lambda d: (d.land == 0).to_numpy(),
    "clear land": lambda d: ((d.cloudy == 0) & (d.tau_a < 0.1) & (d.land == 1)).to_numpy(),
    "clear ocean": lambda d: ((d.cloudy == 0) & (d.tau_a < 0.1) & (d.land == 0)).to_numpy(),
    "no cloud, tau_a<0.05": lambda d: ((d.cloudy == 0) & (d.tau_a < 0.05)).to_numpy(),
    "no cloud, 0.05-0.1": lambda d: ((d.cloudy == 0) & (d.tau_a >= 0.05) & (d.tau_a < 0.1)).to_numpy(),
    "no cloud, 0.1-0.2": lambda d: ((d.cloudy == 0) & (d.tau_a >= 0.1) & (d.tau_a < 0.2)).to_numpy(),
    "no cloud, 0.2-0.5": lambda d: ((d.cloudy == 0) & (d.tau_a >= 0.2) & (d.tau_a < 0.5)).to_numpy(),
    "no cloud, >0.5": lambda d: ((d.cloudy == 0) & (d.tau_a >= 0.5)).to_numpy(),
}


def frame_inputs(d, tag, K):
    k1 = {b: d[f"{tag}{b}_k1_am"].to_numpy(float) for b in WL}
    sig = {b: d[f"n_{b}_sig_k1_am" if f"n_{b}_sig_k1_am" in d else f"{b}_sig_k1_am"].to_numpy(float) for b in WL}
    A = {b: np.exp(d[f"{tag}{b}_c_am"].to_numpy(float)) / K for b in WL}
    shapeS = (d[f"{tag}sco2_k2diag_am"] / d[f"{tag}sco2_k1_am"] ** 2).to_numpy(float)
    return k1, sig, A, shapeS


def stage_slab(args):
    d = pd.read_parquet(OUT / "round4_slab_fits.parquet")
    K = N_PHOT / np.pi
    rows, adequacy = [], []
    allcols = {}
    for tag, noise in (("", "noise-free"), ("n_", "noisy")):
        k1, sig, A, shapeS = frame_inputs(d, tag, K)
        cols = run_all(d, k1, sig, A, None, shapeS)
        allcols[noise] = cols
        for key in [c[2:] for c in cols if c.startswith("g_")]:
            err = (cols[f"g_{key}"] - d.g_co2.to_numpy()) * 400
            sg = cols.get(f"sig_{key}")
            cond = cols.get(f"cond_{key}")
            for sub, fn in SLAB_SUBSETS.items():
                m = fn(d) & np.isfinite(err)
                if m.sum() < 10:
                    continue
                e = err[m]
                r = dict(noise=noise, estimator=key, subset=sub, n=int(m.sum()), rsd_ppm=rsd(e),
                         med_bias_ppm=float(np.median(e)), frac_abs_gt_10ppm=float(np.mean(np.abs(e) > 10)))
                if sg is not None:
                    r["med_sig_g_ppm"] = float(np.nanmedian(sg[m]) * 400)
                    r["pull_rsd"] = rsd(e / (sg[m] * 400))
                if cond is not None:
                    r["frac_illcond"] = float(np.mean(cond[m] > COND_FLAG))
                    r["med_cond"] = float(np.nanmedian(cond[m]))
                    okc = m & (cond <= COND_FLAG)
                    r["rsd_ppm_wellcond"] = rsd(err[okc]) if okc.sum() >= 10 else np.nan
                rows.append(r)
        # adequacy of the two-kernel form against the TRUE path factors <l>_b = k1_b / g_true
        gt = d.g_co2.to_numpy()
        tl = {"o2a": k1["o2a"], "wco2": k1["wco2"] / gt, "sco2": k1["sco2"] / gt}
        for a in ALPHAS:
            pre = f"TK_a{a:g}"
            # (i) predicted 1 + delta_b from the retrieved kernels vs true <l>_b
            for b, dk in (("wco2", "dW"), ("sco2", "dS")):
                e = (1 + cols[f"{dk}_{pre}"] - tl[b]) * 400   # ppm-equivalent path error
                for sub, fn in SLAB_SUBSETS.items():
                    m = fn(d) & np.isfinite(e)
                    if m.sum() >= 10:
                        adequacy.append(dict(noise=noise, alpha=a, test="retrieved kernels: 1+delta - true <l>",
                                             band=b, subset=sub, rsd_ppm=rsd(e[m]), med_ppm=float(np.median(e[m]))))
            # (ii) oracle: kernels from TRUE delta_O and delta_W, then predict true delta_S
            wO, wW, wS = w("o2a", a), w("wco2", a), w("sco2", a)
            a11, a12, a21, a22 = wO * A["o2a"], -wO / A["o2a"], wW * A["wco2"], -wW / A["wco2"]
            b1, b2 = tl["o2a"] - 1, tl["wco2"] - 1
            det = a11 * a22 - a12 * a21
            cL, cS = (b1 * a22 - a12 * b2) / det, (a11 * b2 - a21 * b1) / det
            e = (1 + wS * (cL * A["sco2"] - cS / A["sco2"]) - tl["sco2"]) * 400
            # (iii) oracle one-kernel: true delta_O scaled by w_S
            e1 = (1 + (tl["o2a"] - 1) * wS - tl["sco2"]) * 400
            for test, ee in (("oracle 2-kernel (true dO, dW) -> dS", e), ("oracle 1-kernel (true dO) -> dS", e1)):
                for sub, fn in SLAB_SUBSETS.items():
                    m = fn(d) & np.isfinite(ee)
                    if m.sum() >= 10:
                        adequacy.append(dict(noise=noise, alpha=a, test=test, band="sco2", subset=sub,
                                             rsd_ppm=rsd(ee[m]), med_ppm=float(np.median(ee[m]))))
        # best alpha per scene (oracle) vs true Angstrom: least-squares 2-kernel fit to the 3 true deltas
        if tag == "":
            grid = np.arange(0, 3.0001, 0.1)
            best = []
            for a in grid:
                X = np.stack([np.stack([w(b, a) * A[b], -w(b, a) / A[b]], -1) for b in WL], 1)  # n,3,2
                y = np.stack([tl[b] - 1 for b in WL], 1)
                coef = np.linalg.lstsq(X[0], y[0], rcond=None)[0]  # warm-up
                res = []
                for i in range(len(d)):
                    c = np.linalg.lstsq(X[i], y[i], rcond=None)[0]
                    res.append(np.sum((X[i] @ c - y[i]) ** 2))
                best.append(res)
            best = np.array(best)
            d["oracle_alpha"] = grid[np.argmin(best, 0)]
            d["oracle_resid_ppm"] = np.sqrt(np.min(best, 0) / 3) * 400
    tab = pd.DataFrame(rows).round(3)
    tab.to_csv(OUT / "round4_slab_floor.csv", index=False)
    ad = pd.DataFrame(adequacy).round(3)
    ad.to_csv(OUT / "round4_slab_adequacy.csv", index=False)
    nc = d.cloudy == 0
    extra = dict(
        oracle_alpha_median_nocloud=float(d.oracle_alpha[nc].median()),
        corr_oracle_alpha_true_ang_nocloud=corr(d.oracle_alpha[nc], d.ang[nc]),
        oracle_3band_resid_ppm_median_nocloud=float(d.oracle_resid_ppm[nc].median()),
        oracle_3band_resid_ppm_median_cloudy=float(d.oracle_resid_ppm[~nc].median()),
        corr_cS_tau_a_nocloud_a1=corr(allcols["noise-free"]["cS_TK_a1"][nc], d.tau_a[nc]),
        corr_cL_tau_a_nocloud_a1=corr(allcols["noise-free"]["cL_TK_a1"][nc], d.tau_a[nc]),
        corr_free_alpha_true_ang_nocloud=corr(allcols["noise-free"]["alpha_TKfree"][nc], d.ang[nc]),
        ms_all_variants=float(allcols["noise-free"]["_ms_per_sounding_all_variants"][0]),
        us_one_solve=float(allcols["noise-free"]["_us_per_sounding_one_solve"][0]),
    )
    pd.Series(extra).to_csv(OUT / "round4_slab_extra.csv")
    print(pd.Series(extra).round(3).to_string())
    key = ["TK_a0", "TK_a0.5", "TK_a1", "TK_a1.5", "TK_a2", "OK_a0", "OK_a1", "OK_a2", "TKfree"]
    sub = ["all", "clear (no cloud, tau_a<0.1)", "cloudy", "land", "ocean"]
    print(tab[tab.estimator.isin(key) & tab.subset.isin(sub)].pivot_table(
        index=["noise", "estimator"], columns="subset", values="rsd_ppm").round(2).to_string())


# ── stage: real ─────────────────────────────────────────────────────────────────

def score(est, bc, pri):
    ok = np.isfinite(est) & np.isfinite(bc)
    e, b, p = est[ok], bc[ok], pri[ok]
    r_bc, r_same = rsd(e - b), rsd(b - p)
    bias = float(np.mean(e - b))
    return dict(n=int(ok.sum()), rsd_vs_bc=r_bc, bias_vs_bc=bias, med_bias_vs_bc=float(np.median(e - b)),
                total=float(np.hypot(r_bc, bias)), rsd_vs_prior=rsd(e - p), rsd_bc_vs_prior=r_same,
                R2=1 - (r_bc / r_same) ** 2, corr_bcdev=corr(e - p, b - p))


def stage_real(args):
    d = pd.read_parquet(OUT / f"real_{args.date}.parquet")      # round-3 sample (same soundings)
    mu0, muv = np.cos(np.radians(d.sza)), np.cos(np.radians(d.vza))
    K = (mu0 ** 2).to_numpy()
    k1 = {b: d[f"{b}_k1_am"].to_numpy(float) for b in WL}
    sig = {b: d[f"{b}_sig_k1_am"].to_numpy(float) for b in WL}
    A = {b: np.exp(d[f"{b}_c_am"].to_numpy(float)) / K for b in WL}
    shapeS = (d.sco2_k2diag_am / d.sco2_k1_am ** 2).to_numpy(float)
    cos_sca = (-mu0 * muv - np.sin(np.radians(d.sza)) * np.sin(np.radians(d.vza))
               * np.cos(np.radians(d.saa - d.vaa))).to_numpy()
    dR = rayleigh_delta(A, mu0.to_numpy(), muv.to_numpy(), cos_sca, d.psfc_prior.to_numpy(float))
    cols = run_all(d, k1, sig, A, dR, shapeS)
    for b in WL:
        cols[f"dRay_{b}"] = dR[b]
    res = pd.concat([d, pd.DataFrame(cols, index=d.index)], axis=1)
    pri, bc = res.xco2_apriori.to_numpy(float), res.xco2_bc.to_numpy(float)
    rows, diag = [], []
    keys = [c[2:] for c in cols if c.startswith("g_")]
    for s, g in res.groupby("sfc_type"):
        name = {0.0: "ocean", 1.0: "land"}[s]
        m = (res.sfc_type == s).to_numpy()
        rows.append(dict(surface=name, estimator="prior only", **score(pri[m], bc[m], pri[m])))
        rows.append(dict(surface=name, estimator="round 3 PHYS (raw)",
                         **score(pri[m] * res.r_g.to_numpy()[m], bc[m], pri[m])))
        for key in keys:
            est = pri[m] * res[f"g_{key}"].to_numpy()[m]
            sc_ = score(est, bc[m], pri[m])
            off = float(np.nanmean(bc[m] - est))
            r = dict(surface=name, estimator=key, **sc_, same_date_offset_diag=off)
            if f"sig_{key}" in res:
                sg = res[f"sig_{key}"].to_numpy()[m] * pri[m]
                r["med_sig_ppm"] = float(np.nanmedian(sg))
                r["pull_rsd_vs_bc"] = rsd((est - bc[m]) / sg)
            if f"cond_{key}" in res:
                cnd = res[f"cond_{key}"].to_numpy()[m]
                r["frac_illcond"] = float(np.mean(cnd > COND_FLAG))
                r["med_cond"] = float(np.nanmedian(cnd))
                okc = cnd <= COND_FLAG
                r["rsd_vs_bc_wellcond"] = rsd((est - bc[m])[okc]) if okc.sum() >= 20 else np.nan
            if key.startswith("TK_a") and f"cS_{key}" in res:
                r["spearman_cS_aod"] = float(pd.Series(res[f"cS_{key}"][m].to_numpy()).corr(
                    pd.Series(g.aod_total.to_numpy()), method="spearman"))
                r["spearman_cL_aod"] = float(pd.Series(res[f"cL_{key}"][m].to_numpy()).corr(
                    pd.Series(g.aod_total.to_numpy()), method="spearman"))
                r["spearman_cL_alb_sco2"] = float(pd.Series(res[f"cL_{key}"][m].to_numpy()).corr(
                    pd.Series(g.alb_sco2.to_numpy()), method="spearman"))
            rows.append(r)
        # model-free consistency: water vs dry path ratio
        rw = (g.sco2_k1w_am / g.wco2_k1w_am).to_numpy()
        rd = (g.sco2_k1_am / g.wco2_k1_am).to_numpy()
        diag.append(dict(surface=name, med_ratio_dry_S_over_W=float(np.median(rd)), rsd_ratio_dry=rsd(rd),
                         med_ratio_h2o_S_over_W=float(np.median(rw)), rsd_ratio_h2o=rsd(rw),
                         corr_ratio_h2o_vs_dry=corr(rw, rd),
                         spearman_ratio_h2o_vs_dry=float(pd.Series(rw).corr(pd.Series(rd), method="spearman")),
                         med_r_minus_1=float(np.median(rd - 1)),
                         med_kO_minus_1=float(np.median(g.o2a_k1_am - 1)),
                         med_A_ratio_W_over_O=float(np.median(A["wco2"][m] / A["o2a"][m])),
                         med_A_ratio_S_over_O=float(np.median(A["sco2"][m] / A["o2a"][m])),
                         med_eps_ray_o2a=float(np.median(dR["eps_o2a"][m])),
                         med_dRay_o2a=float(np.median(dR["o2a"][m])),
                         med_dRay_sco2=float(np.median(dR["sco2"][m])),
                         med_alpha_free=float(np.nanmedian(res.alpha_TKfree.to_numpy()[m])),
                         ms_all_variants=float(cols["_ms_per_sounding_all_variants"][0]),
                         us_one_solve=float(cols["_us_per_sounding_one_solve"][0])))
    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(OUT / f"round4_real_{args.date}_table.csv", index=False)
    pd.DataFrame(diag).round(4).to_csv(OUT / f"round4_real_{args.date}_diag.csv", index=False)
    res[[c for c in res.columns if not c.startswith(("r_", "Fprior_"))]].to_parquet(
        OUT / f"round4_real_{args.date}.parquet", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["surface", "estimator", "rsd_vs_bc", "bias_vs_bc", "rsd_vs_prior", "R2", "corr_bcdev",
            "med_sig_ppm", "pull_rsd_vs_bc", "frac_illcond", "rsd_vs_bc_wellcond"]
    print(tab[show].to_string(index=False))
    print(pd.DataFrame(diag).T.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["slabgen", "slab", "real"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--date", default="2018-07-01")
    args = ap.parse_args()
    {"slabgen": stage_slabgen, "slab": stage_slab, "real": stage_real}[args.stage](args)


if __name__ == "__main__":
    main()

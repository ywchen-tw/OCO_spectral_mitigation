"""Slab closure test: which fitter's (k1, k2) recovers the true PPDF moments?

Runs T0 (production reproduction), T1 (closure on the Monte Carlo spectra),
T2 (theorem spectra: fitter error vs theorem error), T3 (dense synthetic
spectra on real OCO-2 τ grids with noise and structured misfit), T4 (PPDF
shape vs gamma), T5 (vertical-shape regressor) and T6 (784-nm anchors
dropped).  Inputs are read-only (results/rt_slab_sim, one real fp_tau file
for the τ grids).  Outputs: results/slab_*.csv, results/slab_*.png,
results/slab_closure_summary.md.

Run:  python idea_test/fit_order_study/slab_closure.py [--workers 8] [--t3-reps 12]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slab_truth as st  # noqa: E402
from slab_fitters import FITTERS, SHAPE_FITTERS, fit_all, fit_one  # noqa: E402

OUT = HERE / "results"
SIM = st.SIM_DIR
FP_TAU = st.REPO / "results" / "2020-01-01" / "29252a_GL" / "fp_tau_combined.h5"
FP_TAU_SOUNDING = 5000
CASES = ("dark/3d", "bright/3d", "dark/ipa", "bright/ipa")
TAU_REFS = (0.1, 0.3, 1.0)
TAU_GRID = np.concatenate([[0.0], np.logspace(-3, 1.0, 1200)])
# categorical slots 1..6 of the reference palette, fixed order = FITTERS order
COLORS = dict(zip(FITTERS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")))
REGION_COLORS = {"cloud": "#eb6834", "shadow": "#1baf7a", "clear": "#2a78d6"}

T3_LEVELS = (  # (name, white σ at the continuum, misfit rms)
    ("noiseless", 0.0, 0.0),
    ("white", 0.005, 0.0),
    ("misfit0.05", 0.005, 0.05),
    ("misfit0.1", 0.005, 0.10),
    ("misfit0.2", 0.005, 0.20),
)
AR1_RHO = 0.5


# --------------------------------------------------------------------------
# Truth construction
# --------------------------------------------------------------------------
_CASES: dict = {}


def _truth_task(args):
    key, run = args
    tr, recs = st.column_truths(_CASES[key], run)
    return key, run, tr, recs


def build_truths(cases, workers):
    """Pooled and per-run truths for every case (parallel over case × run)."""
    _CASES.update(cases)
    nrun = next(iter(cases.values())).hist.shape[1]
    tasks = [(k, r) for k in CASES for r in [None, *range(nrun)]]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_truth_task, tasks)
    truths = {(k, r): (tr, recs) for k, r, tr, recs in res}
    return truths, nrun


def truth_targets(case, tr, recs) -> dict[str, np.ndarray]:
    """All per-column reference values used as closure targets."""
    t = {}
    for n in ("geo784", "geo_rw", "geo_band", "abs"):
        t[f"m_{n}"] = tr[n]["mean"]
        t[f"v_{n}"] = tr[n]["var"]
        t[f"skew_{n}"] = tr[n]["skew"]
        t[f"exkurt_{n}"] = tr[n]["exkurt"]
    t["vb_abs"] = tr["abs"]["var_between"]
    k1 = np.array([st.tilted_cumulants(r["w"], r["a"], np.array(TAU_REFS), r["v"])[0] for r in recs])
    k2 = np.array([st.tilted_cumulants(r["w"], r["a"], np.array(TAU_REFS), r["v"])[1] for r in recs])
    for j, tr_ in enumerate(TAU_REFS):
        t[f"K1_abs@{tr_}"] = k1[:, j]
        t[f"K2_abs@{tr_}"] = k2[:, j]
    t["rebuild_rms"] = np.array([r["rebuild_rms"] for r in recs])
    t["missing_mass"] = 1.0 - case.hist[~case.absorbing].mean(axis=(0, 1)).sum(axis=-1)
    return t


def regions(x_km):
    in_cloud = (x_km >= st.CLOUD_X_KM[0]) & (x_km < st.CLOUD_X_KM[1])
    shadow = (x_km >= st.CLOUD_X_KM[1]) & (x_km <= 25.0)
    return np.where(in_cloud, "cloud", np.where(shadow, "shadow", "clear"))


# --------------------------------------------------------------------------
# Fitting helpers
# --------------------------------------------------------------------------
def fit_columns(tau, ln_t, sig, eta, tags=FITTERS, channels=None):
    """ln_t, sig: (nw, nx) -> {tag: (nx, 3) array of k1, k2, c}."""
    sel = slice(None) if channels is None else channels
    nx = ln_t.shape[1]
    out = {t: np.full((nx, 3), np.nan) for t in tags}
    for ix in range(nx):
        res = fit_all(tau[sel], ln_t[sel, ix], sig[sel, ix], eta[sel], tags=tags)
        for t in tags:
            out[t][ix] = res[t]
    return out


def _stats(est, tru, mask, log=False):
    ok = mask & np.isfinite(est) & np.isfinite(tru)
    if log:
        ok &= (est > 0) & (tru > 0)
        est, tru = np.log10(np.where(ok, est, 1.0)), np.log10(np.where(ok, tru, 1.0))
    e, t = est[ok], tru[ok]
    n = int(ok.sum())
    if n < 3:
        return {"n": n}
    d = e - t
    r = np.corrcoef(e, t)[0, 1] if np.std(e) > 0 and np.std(t) > 0 else np.nan
    slope, icpt = np.polyfit(t, e, 1) if np.std(t) > 0 else (np.nan, np.nan)
    return {"n": n, "r": r, "slope": slope, "intercept": icpt, "med_bias": np.median(d),
            "med_rel_bias": np.nan if log else np.median(d / t), "rms": np.sqrt(np.mean(d**2))}


def _boot_ci(est, tru, mask, rng, nboot=500, log=False):
    idx = np.flatnonzero(mask & np.isfinite(est) & np.isfinite(tru))
    if len(idx) < 5:
        return (np.nan,) * 4
    rs, ss = [], []
    for _ in range(nboot):
        b = rng.choice(idx, len(idx))
        m = np.zeros_like(mask)
        m[b] = True
        e, t = est[b], tru[b]
        if log:
            good = (e > 0) & (t > 0)
            e, t = np.log10(e[good]), np.log10(t[good])
        if np.std(e) > 0 and np.std(t) > 0:
            rs.append(np.corrcoef(e, t)[0, 1])
            ss.append(np.polyfit(t, e, 1)[0])
    if not rs:
        return (np.nan,) * 4
    return (*np.percentile(rs, [2.5, 97.5]), *np.percentile(ss, [2.5, 97.5]))


K1_TARGETS = ("m_geo784", "m_geo_rw", "m_abs", *[f"K1_abs@{t}" for t in TAU_REFS])
K2_TARGETS = ("v_geo784", "v_geo_rw", "v_abs", *[f"K2_abs@{t}" for t in TAU_REFS])


def closure_table(fits, targets, masks, label, fits_runs=None, targets_runs=None, rng=None, boot=True):
    rows = []
    for tag, arr in fits.items():
        k1, k2 = arr[:, 0], arr[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            kap = np.where(k2 > 0, k1**2 / k2, np.nan)
        pairs = [("k1", k1, t, False) for t in K1_TARGETS] + [("k2", k2, t, False) for t in K2_TARGETS]
        pairs.append(("kappa", kap, "kappa_abs", True))
        pairs.append(("kappa", kap, "kappa_geo784", True))
        for par, est, tname, log in pairs:
            tru = _target(targets, tname)
            for sub, mask in masks.items():
                s = _stats(est, tru, mask, log=log)
                row = {"set": label, "fitter": tag, "param": par, "target": tname, "subset": sub, **s}
                if par == "k2":
                    row["frac_k2_le0"] = float(np.mean(est[mask] <= 1e-6))
                if par == "kappa":
                    row["frac_kappa_undef"] = float(np.mean(~np.isfinite(est[mask])))
                if boot and rng is not None and sub != "far":
                    row["r_lo"], row["r_hi"], row["slope_lo"], row["slope_hi"] = _boot_ci(est, tru, mask, rng, log=log)
                if fits_runs is not None:
                    mb = []
                    for fr, tr_ in zip(fits_runs, targets_runs):
                        e = fr[tag][:, 0 if par == "k1" else 1]
                        if par == "kappa":
                            with np.errstate(divide="ignore", invalid="ignore"):
                                e = np.where(fr[tag][:, 1] > 0, fr[tag][:, 0] ** 2 / fr[tag][:, 1], np.nan)
                        mb.append(_stats(e, _target(tr_, tname), mask, log=log).get("med_bias", np.nan))
                    row["med_bias_sig_run"] = float(np.nanstd(mb, ddof=1) / np.sqrt(len(mb)))
                rows.append(row)
    return rows


def _target(targets, name):
    if name == "kappa_abs":
        return targets["m_abs"] ** 2 / targets["v_abs"]
    if name == "kappa_geo784":
        return targets["m_geo784"] ** 2 / targets["v_geo784"]
    return targets[name]


def tau_star(rec, k, which):
    """Smallest τ on TAU_GRID with K_which(τ) = k (first crossing), NaN if none."""
    k1, k2 = st.tilted_cumulants(rec["w"], rec["a"], TAU_GRID, rec["v"])
    curve = k1 if which == 1 else k2
    d = curve - k
    s = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
    if not np.isfinite(k) or len(s) == 0:
        return np.nan
    i = s[0]
    return float(TAU_GRID[i] + (TAU_GRID[i + 1] - TAU_GRID[i]) * d[i] / (d[i] - d[i + 1]))


# --------------------------------------------------------------------------
# T3 helpers
# --------------------------------------------------------------------------
def real_tau_grids():
    grids = {}
    try:
        with h5py.File(FP_TAU, "r") as f:
            for band in ("o2a", "sco2"):
                t = f[f"{band}_tau_output"][FP_TAU_SOUNDING].astype(float)
                grids[band] = t[np.isfinite(t) & (t > 0)]          # wavelength order kept
    except OSError:
        rng = np.random.default_rng(1)
        grids = {"o2a": np.exp(rng.uniform(np.log(0.05), np.log(9.0), 1000)),
                 "sco2": np.exp(rng.uniform(np.log(0.02), np.log(4.0), 1000))}
    return grids


def ar1(n, rho, rng):
    z = np.empty(n)
    z[0] = rng.standard_normal()
    e = rng.standard_normal(n) * np.sqrt(1 - rho**2)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + e[i]
    return z


_T3: dict = {}


def _t3_task(job):
    """One (truth kind, case, column, grid) for all levels and realizations."""
    kind, key, ix, band, reps, seed = job
    tau = _T3["grids"][band]
    rec = _T3["recs"][key][ix]
    if kind == "abs":
        ln_t0 = st.laplace_log(rec["w"], rec["a"], tau, rec["v"])
    else:
        h = _T3["geo"][key][ix]
        ln_t0 = st.laplace_log(h, _T3["l_mid"], tau)
    rng = np.random.default_rng(seed)
    eta = np.zeros_like(tau)
    tau_rms = np.sqrt(np.mean(tau**2))
    rows = []
    for lev, s0, mis in T3_LEVELS:
        sig = np.maximum(s0, 1e-4) * np.exp(-0.5 * ln_t0)        # photon-noise scaling, continuum = s0
        sig = np.minimum(sig, 0.2)
        nrep = 1 if lev == "noiseless" else reps
        for irep in range(nrep):
            y = ln_t0.copy()
            if s0 > 0:
                y += sig * rng.standard_normal(len(tau))
            if mis > 0:
                y += mis * (tau / tau_rms) * ar1(len(tau), AR1_RHO, rng)
            res = fit_all(tau, y, sig, eta, tags=FITTERS)
            for tag, (k1, k2, _) in res.items():
                rows.append((kind, key, ix, band, lev, irep, tag, k1, k2))
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--t3-reps", type=int, default=12)
    ap.add_argument("--skip-t3", action="store_true")
    ap.add_argument("--reuse-t3", action="store_true", help="re-summarize the stored T3 raw table")
    args = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261029)

    cases, meta = st.load_cases()
    x_km = cases["dark/3d"].x_km
    masks = st.column_masks(x_km)
    reg = regions(x_km)
    print(f"loaded; l_direct {meta['l_direct_km']:.2f} km, L_peak {meta['l_peak_km']:.1f} km, "
          f"shape-mode share {cases['dark/3d'].meta['shape_frac']:.3f}")

    truths, nrun = build_truths(cases, args.workers)
    print(f"truths built ({time.time() - t_start:.0f} s)")
    targets = {k: truth_targets(cases[k], *truths[(k, None)]) for k in CASES}
    targets_runs = {k: [truth_targets(cases[k], *truths[(k, r)]) for r in range(nrun)] for k in CASES}
    recs = {k: truths[(k, None)][1] for k in CASES}

    # ---------------- per-column truth table ----------------
    rows = []
    for k in CASES:
        for ix in range(st.NX):
            row = {"case": k, "ix": ix, "x_km": x_km[ix], "region": reg[ix]}
            row.update({n: v[ix] for n, v in targets[k].items()})
            for n in ("m_abs", "v_abs", "m_geo784", "v_geo784"):
                row[f"{n}_sig_run"] = np.std([tr[n][ix] for tr in targets_runs[k]], ddof=1) / np.sqrt(nrun)
            rows.append(row)
    truth_df = pd.DataFrame(rows)
    truth_df.to_csv(OUT / "slab_closure_truth.csv", index=False)

    # ---------------- T0 reproduction ----------------
    repro = []
    with h5py.File(SIM / "slab_fit.h5", "r") as f, open(SIM / "closure_stats.json") as fj:
        cs = json.load(fj)
        for k in CASES:
            c = cases[k]
            fit7 = fit_columns(c.tau, c.ln_t, c.sig_ln_t, c.eta, tags=("ols_7",))["ols_7"]
            row = {"case": k, "max_abs_dk1": np.max(np.abs(fit7[:, 0] - f[k]["k1"][...])),
                   "max_abs_dk2": np.max(np.abs(fit7[:, 1] - f[k]["k2"][...]))}
            if k.endswith("3d"):
                s = k.split("/")[0]
                t = targets[k]
                clr = masks["clear"]
                row.update({
                    "r_k1_meanL": np.corrcoef(fit7[:, 0], t["m_geo784"])[0, 1], "json_r_k1_meanL": cs[f"r_k1_meanL_{s}"],
                    "r_k2_varL": np.corrcoef(fit7[:, 1], t["v_geo784"])[0, 1], "json_r_k2_varL": cs[f"r_k2_varL_{s}"],
                    "r_k1_meanL_clear": np.corrcoef(fit7[clr, 0], t["m_geo784"][clr])[0, 1],
                    "json_r_k1_meanL_clear": cs[f"r_k1_meanL_clear_{s}"],
                    "r_k2_varL_clear": np.corrcoef(fit7[clr, 1], t["v_geo784"][clr])[0, 1],
                    "json_r_k2_varL_clear": cs[f"r_k2_varL_clear_{s}"]})
            repro.append(row)
    pd.DataFrame(repro).to_csv(OUT / "slab_closure_repro.csv", index=False)
    print(pd.DataFrame(repro).to_string())

    # ---------------- T1 / T5 / T6: fits on the MC spectra ----------------
    all_tags = FITTERS + SHAPE_FITTERS
    mc_fits, mc_fits_runs, noanchor_fits = {}, {}, {}
    for k in CASES:
        c = cases[k]
        mc_fits[k] = fit_columns(c.tau, c.ln_t, c.sig_ln_t, c.eta, tags=all_tags)
        mc_fits_runs[k] = [fit_columns(c.tau, c.ln_t_runs[..., r], c.sig_ln_t * np.sqrt(nrun), c.eta, tags=FITTERS)
                           for r in range(nrun)]
        noanchor_fits[k] = fit_columns(c.tau, c.ln_t, c.sig_ln_t, c.eta, tags=FITTERS,
                                       channels=np.flatnonzero(c.absorbing))
    print(f"MC fits done ({time.time() - t_start:.0f} s)")

    rows = []
    for k in CASES:
        base = {t: mc_fits[k][t] for t in FITTERS}
        for r in closure_table(base, targets[k], masks, "T1_mc", mc_fits_runs[k], targets_runs[k], rng):
            rows.append({"case": k, **r})
        shp = {t: mc_fits[k][t] for t in SHAPE_FITTERS}
        for r in closure_table(shp, targets[k], masks, "T5_mc_eta", rng=rng):
            rows.append({"case": k, **r})
        for r in closure_table(noanchor_fits[k], targets[k], masks, "T6_mc_noanchor", rng=rng):
            rows.append({"case": k, **r})

    # ---------------- T2: theorem spectra ----------------
    resid_rows, teq_fits = [], {}
    for k in CASES:
        c = cases[k]
        ln_eq_abs = np.full_like(c.ln_t, np.nan)
        ln_eq_geo = np.full_like(c.ln_t, np.nan)
        for ix in range(st.NX):
            rec = recs[k][ix]
            ln_eq_abs[:, ix] = st.laplace_log(rec["w"], rec["a"], c.tau, rec["v"]) + rec["ln_t0"]
            h = st.geometric_hist(c, ix, "geo784")
            ln_eq_geo[:, ix] = st.laplace_log(h, c.l_mid, c.tau) + c.ln_t[~c.absorbing, ix].mean()
            for name, ln_eq in (("abs", ln_eq_abs), ("geo784", ln_eq_geo)):
                rr = c.ln_t[:, ix] - ln_eq[:, ix]
                ab = c.absorbing
                eta_corr = np.corrcoef(rr[ab] / np.maximum(c.tau[ab], 1e-3), c.eta[ab])[0, 1]
                resid_rows.append({"case": k, "ix": ix, "x_km": x_km[ix], "region": reg[ix], "ref": name,
                                   "rms": np.sqrt(np.mean(rr**2)), "rms_absorbing": np.sqrt(np.mean(rr[ab] ** 2)),
                                   "max_abs": np.max(np.abs(rr)), "corr_resid_over_tau_eta": eta_corr,
                                   "anchor_step": float(np.mean(rr[~ab]))})
        teq = {}
        for name, ln_eq, sig in (("abs", ln_eq_abs, c.sig_ln_t), ("geo784", ln_eq_geo, c.sig_ln_t)):
            teq[name] = fit_columns(c.tau, ln_eq, sig, c.eta, tags=FITTERS)
        teq_fits[k] = (teq, ln_eq_abs, ln_eq_geo)
        for r in closure_table(teq["abs"], targets[k], masks, "T2_teq_abs", rng=rng, boot=False):
            rows.append({"case": k, **r})
        for r in closure_table(teq["geo784"], targets[k], masks, "T2_teq_geo784", rng=rng, boot=False):
            rows.append({"case": k, **r})
    closure_df = pd.DataFrame(rows)
    closure_df.to_csv(OUT / "slab_closure_stats.csv", index=False)
    resid_df = pd.DataFrame(resid_rows)
    resid_df.to_csv(OUT / "slab_closure_theorem_resid.csv", index=False)

    # per-column fit table
    rows = []
    for k in CASES:
        for setname, fits in (("mc", mc_fits[k]), ("teq_abs", teq_fits[k][0]["abs"]),
                              ("teq_geo784", teq_fits[k][0]["geo784"]), ("mc_noanchor", noanchor_fits[k])):
            for tag, arr in fits.items():
                for ix in range(st.NX):
                    rows.append({"case": k, "set": setname, "fitter": tag, "ix": ix, "x_km": x_km[ix],
                                 "k1": arr[ix, 0], "k2": arr[ix, 1], "intercept": arr[ix, 2]})
    pd.DataFrame(rows).to_csv(OUT / "slab_closure_fits.csv", index=False)

    # ---------------- τ* : where on the local-cumulant curve does each fitter sit? ----------------
    rows = []
    for k in ("dark/3d", "bright/3d"):
        for setname, fits in (("mc", mc_fits[k]), ("teq_abs", teq_fits[k][0]["abs"])):
            for tag in (*FITTERS, *(SHAPE_FITTERS if setname == "mc" else ())):
                arr = fits[tag]
                for ix in range(st.NX):
                    rows.append({"case": k, "set": setname, "fitter": tag, "ix": ix, "region": reg[ix],
                                 "tau_star_k1": tau_star(recs[k][ix], arr[ix, 0], 1),
                                 "tau_star_k2": tau_star(recs[k][ix], arr[ix, 1], 2)})
    tstar_df = pd.DataFrame(rows)
    tstar_df.to_csv(OUT / "slab_closure_tau_star.csv", index=False)

    # ---------------- T4: shape adequacy ----------------
    rows = []
    for k in CASES:
        t = targets[k]
        for ix in range(st.NX):
            for n in ("abs", "geo784"):
                m, v = t[f"m_{n}"][ix], t[f"v_{n}"][ix]
                kap = m**2 / v
                rows.append({"case": k, "ix": ix, "x_km": x_km[ix], "region": reg[ix], "ppdf": n,
                             "mean": m, "var": v, "kappa": kap, "skew": t[f"skew_{n}"][ix],
                             "skew_gamma": 2 / np.sqrt(kap), "exkurt": t[f"exkurt_{n}"][ix],
                             "exkurt_gamma": 6 / kap})
            # how well does the best gamma represent the true abs spectrum (on the slab τ)?
            rec = recs[k][ix]
            ln_eq = st.laplace_log(rec["w"], rec["a"], cases[k].tau, rec["v"])
            k1g, k2g, cg = teq_fits[k][0]["abs"]["gamma_u"][ix]
            kapg = k1g**2 / k2g
            lng = cg - kapg * np.log1p(k1g * cases[k].tau / kapg)
            m, v = t["m_abs"][ix], t["v_abs"][ix]
            lnm = -(m**2 / v) * np.log1p(cases[k].tau * v / m)
            rows[-2].update({"gamma_fit_max_dlnT": np.max(np.abs(lng - ln_eq - rec["ln_t0"])),
                             "gamma_moment_max_dlnT": np.max(np.abs(lnm - ln_eq))})
    shape_df = pd.DataFrame(rows)
    shape_df.to_csv(OUT / "slab_closure_shape.csv", index=False)

    # ---------------- T3: dense synthetic ----------------
    t3_df = None
    if args.reuse_t3 and (OUT / "slab_closure_t3_raw.csv.gz").exists():
        t3_df = summarize_t3(pd.read_csv(OUT / "slab_closure_t3_raw.csv.gz"), targets, recs, reg)
        t3_df.to_csv(OUT / "slab_closure_t3.csv", index=False)
    elif not args.skip_t3:
        grids = real_tau_grids()
        _T3.update({"grids": grids, "recs": recs, "l_mid": cases["dark/3d"].l_mid,
                    "geo": {k: [st.geometric_hist(cases[k], ix, "geo784") for ix in range(st.NX)] for k in CASES}})
        jobs, seed = [], 1000
        for kind in ("abs", "geo784"):
            for k in ("dark/3d", "bright/3d"):
                for ix in range(st.NX):
                    for band in grids:
                        jobs.append((kind, k, ix, band, args.t3_reps, seed))
                        seed += 1
        with mp.get_context("fork").Pool(args.workers) as pool:
            res = pool.map(_t3_task, jobs, chunksize=4)
        t3_raw = pd.DataFrame([r for rr in res for r in rr],
                              columns=["kind", "case", "ix", "band", "level", "rep", "fitter", "k1", "k2"])
        t3_raw.to_csv(OUT / "slab_closure_t3_raw.csv.gz", index=False)
        t3_df = summarize_t3(t3_raw, targets, recs, reg)
        t3_df.to_csv(OUT / "slab_closure_t3.csv", index=False)
        print(f"T3 done ({time.time() - t_start:.0f} s); grids: " +
              ", ".join(f"{b} n={len(g)} τ∈[{g.min():.3f},{g.max():.2f}]" for b, g in grids.items()))

    # ---------------- figures + summary ----------------
    make_figures(cases, targets, mc_fits, teq_fits, recs, reg, x_km, t3_df, shape_df)
    write_summary(closure_df, resid_df, tstar_df, shape_df, t3_df, truth_df, meta, cases)
    print(f"done ({time.time() - t_start:.0f} s)")


T3_TARGETS = {"abs": (("moments", "m_abs", "v_abs"), ("tau0.3", "K1_abs@0.3", "K2_abs@0.3"),
                       ("tau1.0", "K1_abs@1.0", "K2_abs@1.0")),
              "geo784": (("moments", "m_geo784", "v_geo784"),)}


def summarize_t3(raw, targets, recs, reg):
    """Per (truth kind, case, band, level, fitter, target): bias/scatter/r of the
    realization-mean k1, k2 against the true moments (and, for the
    absorption-weighted PPDF, against the local cumulants K1, K2 at τ = 0.3, 1)."""
    rows = []
    for (kind, key, band, lev, tag), g in raw.groupby(["kind", "case", "band", "level", "fitter"]):
        t = targets[key]
        per = g.groupby("ix").agg(k1=("k1", "mean"), k2=("k2", "mean"), k1_sd=("k1", "std"), k2_sd=("k2", "std"),
                                  k2_bound=("k2", lambda s: np.mean(s <= 1e-6)))
        ix = per.index.values
        with np.errstate(divide="ignore", invalid="ignore"):
            kap_e = np.where(per.k2 > 0, per.k1**2 / per.k2, np.nan)
            kap_each = np.where(g.k2 > 0, g.k1**2 / g.k2, np.nan)
        g_kap = pd.Series(np.log10(kap_each), index=g.index).groupby(g.ix.values).std()
        for tname, n1, n2 in T3_TARGETS[kind]:
            m, v = t[n1][ix], t[n2][ix]
            for sub, sel in (("all", np.ones(len(ix), bool)), ("clear", reg[ix] == "clear")):
                def rel(e, tt):
                    return np.median((e[sel] - tt[sel]) / tt[sel])
                rows.append({
                    "kind": kind, "target": tname, "case": key, "band": band, "level": lev, "fitter": tag,
                    "subset": sub,
                    "k1_med_rel_bias": rel(per.k1.values, m),
                    "k1_rel_scatter": np.median(per.k1_sd.values[sel] / m[sel]),
                    "k1_r": np.corrcoef(per.k1.values[sel], m[sel])[0, 1],
                    "k1_slope": np.polyfit(m[sel], per.k1.values[sel], 1)[0],
                    "k2_med_rel_bias": rel(per.k2.values, v),
                    "k2_rel_scatter": np.median(per.k2_sd.values[sel] / v[sel]),
                    "k2_r": np.corrcoef(per.k2.values[sel], v[sel])[0, 1],
                    "k2_slope": np.polyfit(v[sel], per.k2.values[sel], 1)[0],
                    "k2_frac_bound": float(np.mean(per.k2_bound.values[sel])),
                    "kappa_med_log10_ratio": np.nanmedian(np.log10(kap_e[sel] / (m[sel] ** 2 / v[sel]))),
                    "kappa_log10_scatter": np.nanmedian(g_kap.values[sel]),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _scatter_grid(ax_grid, fits_by_case, targets, tname, par, reg, cases_used, label):
    markers = {"dark": "o", "bright": "s"}
    for j, tag in enumerate(FITTERS):
        ax = ax_grid[j]
        lo, hi = np.inf, -np.inf
        for k in cases_used:
            est = fits_by_case[k][tag][:, 0 if par == "k1" else 1]
            tru = targets[k][tname]
            for rname, col in REGION_COLORS.items():
                m = reg == rname
                ax.scatter(tru[m], est[m], s=14, marker=markers[k.split("/")[0]], facecolor=col,
                           edgecolor="white", linewidth=0.4, label=f"{k.split('/')[0]} {rname}")
            lo = min(lo, np.nanmin(tru), np.nanmin(est))
            hi = max(hi, np.nanmax(tru), np.nanmax(est))
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8, ls="--")
        ax.set_title(tag, fontsize=9)
        ax.grid(color="0.9", lw=0.5)
        ax.tick_params(labelsize=7)
    ax_grid[0].set_ylabel(f"fitted {par} ({label})", fontsize=8)


def make_figures(cases, targets, mc_fits, teq_fits, recs, reg, x_km, t3_df, shape_df):
    cases_used = ("dark/3d", "bright/3d")
    # closure scatter: rows = (k1 vs <A>, k1 vs <l'>_geo, k2 vs var A, k2 vs K2@0.3, k2 vs var l'_geo)
    specs = [("k1", "m_abs", "⟨A⟩ (absorption-weighted)"), ("k1", "m_geo784", "⟨l'⟩ geometric (784 nm)"),
             ("k2", "v_abs", "var A"), ("k2", "K2_abs@0.3", "K2(τ=0.3) of A"),
             ("k2", "v_geo784", "var l' geometric")]
    fig, axes = plt.subplots(len(specs), len(FITTERS), figsize=(15, 2.6 * len(specs)), squeeze=False)
    for i, (par, tname, tl) in enumerate(specs):
        _scatter_grid(axes[i], mc_fits, targets, tname, par, reg, cases_used, "MC")
        for ax in axes[i]:
            ax.set_xlabel(f"true {tl}", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[:6], l[:6], loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("T1 closure on the MC spectra (3-D; o dark, □ bright; dashed 1:1)", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(OUT / "slab_closure_scatter_mc.png", dpi=130)
    plt.close(fig)

    # same on the exact theorem spectrum (pure fitter error)
    fig, axes = plt.subplots(3, len(FITTERS), figsize=(15, 7.8), squeeze=False)
    teq = {k: teq_fits[k][0]["abs"] for k in cases_used}
    for i, (par, tname, tl) in enumerate([specs[0], specs[2], specs[3]]):
        _scatter_grid(axes[i], teq, targets, tname, par, reg, cases_used, "T_eq")
        for ax in axes[i]:
            ax.set_xlabel(f"true {tl}", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[:6], l[:6], loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("T2 fitters on the exact theorem spectrum ln T_eq,A (33 slab τ, no noise)", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT / "slab_closure_scatter_teq.png", dpi=130)
    plt.close(fig)

    # theorem residual vs τ: rows dark-geo, dark-abs, bright-geo, bright-abs
    ex = [(2, "clear far"), (18, "illuminated side, 0.25 km from edge"), (24, "in cloud"), (36, "shadow")]
    fig, axes = plt.subplots(4, 4, figsize=(15, 11), sharex=True, squeeze=False)
    for i, k in enumerate(cases_used):
        c = cases[k]
        _, ln_abs, ln_geo = teq_fits[k]
        for j, (ix, name) in enumerate(ex):
            for row, (ln_eq, lab) in enumerate(((ln_geo, "MC − T_eq(geo784)"), (ln_abs, "MC − T_eq,A (mean profile)"))):
                ax = axes[2 * i + row, j]
                sc = ax.scatter(c.tau, c.ln_t[:, ix] - ln_eq[:, ix], c=c.eta, cmap="viridis", s=22,
                                edgecolor="0.3", linewidth=0.3, vmin=-0.08, vmax=0.06)
                ax.axhline(0, color="0.5", lw=0.6)
                ax.set_xscale("symlog", linthresh=0.03)
                ax.set_title(f"{k} x={x_km[ix]:.2f} km ({name})\n{lab}", fontsize=7.5)
                ax.grid(color="0.9", lw=0.5)
                ax.tick_params(labelsize=7)
                if j == 0:
                    ax.set_ylabel("ln T residual", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("slant τ", fontsize=8)
    cb = fig.colorbar(sc, ax=axes, shrink=0.5, pad=0.01)
    cb.set_label("channel vertical-shape index η (+ = absorption aloft)", fontsize=8)
    fig.suptitle("T2 theorem residual: MC ln T minus the equivalence-theorem spectrum of the true PPDF", fontsize=10)
    fig.savefig(OUT / "slab_theorem_resid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # local cumulants vs τ with fitted values
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), squeeze=False)
    for i, k in enumerate(cases_used):
        for j, (ix, name) in enumerate(ex):
            ax = axes[i, j]
            rec = recs[k][ix]
            k1c, k2c = st.tilted_cumulants(rec["w"], rec["a"], TAU_GRID, rec["v"])
            tg = np.maximum(TAU_GRID, 1e-3)
            ax.plot(tg, k1c, color="0.15", lw=2.0, label="K1(τ) of A")
            ax.plot(tg, k2c, color="0.15", lw=2.0, ls="--", label="K2(τ) of A")
            for tag in FITTERS:
                k1, k2 = mc_fits[k][tag][ix, :2]
                ax.axhline(k1, color=COLORS[tag], lw=1.2, label=tag)
                if k2 > 1e-6:
                    ax.axhline(k2, color=COLORS[tag], lw=1.2, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(3e-3, 3)
            ax.set_title(f"{k} x={x_km[ix]:.2f} km ({name})", fontsize=8)
            ax.grid(color="0.9", lw=0.5)
            ax.tick_params(labelsize=7)
            if i == 1:
                ax.set_xlabel("τ (log; τ = 0 plotted at 1e-3)", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="lower left", ncol=2)
    fig.suptitle("Local cumulants K1(τ), K2(τ) of the true absorption-weighted PPDF (black) vs fitted k1 (solid) / "
                 "k2 (dashed) on the MC spectra; k2 ≤ 0 not drawn", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "slab_local_cumulants.png", dpi=130)
    plt.close(fig)

    # example PPDFs with gamma (binned densities, ΔA = 0.02)
    from scipy.stats import gamma as gdist
    edges = np.linspace(0.0, 3.0, 151)
    mid = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), squeeze=False)
    for i, k in enumerate(cases_used):
        for j, (ix, name) in enumerate(ex):
            ax = axes[i, j]
            rec = recs[k][ix]
            mass = np.zeros(len(mid))
            for w, a, v in zip(rec["w"], rec["a"], rec["v"]):
                if v > 1e-10 and a > 0:
                    cdf = gdist.cdf(edges, a**2 / v, scale=v / a)
                else:
                    cdf = (edges >= a).astype(float)
                mass += w * np.diff(cdf)
            ax.step(mid, mass / np.diff(edges), where="mid", color="0.15", lw=1.4, label="true p(A)")

            def gpdf(k1, k2):
                return gdist.pdf(mid, k1**2 / k2, scale=k2 / k1)
            m, v = targets[k]["m_abs"][ix], targets[k]["v_abs"][ix]
            ax.plot(mid, gpdf(m, v), color=COLORS["ols_3"], lw=1.4, label="gamma with true ⟨A⟩, var A")
            ax.plot(mid, gpdf(*mc_fits[k]["gamma_u"][ix, :2]), color=COLORS["gamma_u"], lw=1.4, label="gamma_u fit (MC)")
            ax.plot(mid, gpdf(*mc_fits[k]["gamma_w"][ix, :2]), color=COLORS["gamma_w"], lw=1.4, label="gamma_w fit (MC)")
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 60)
            ax.set_title(f"{k} x={x_km[ix]:.2f} km ({name})", fontsize=8)
            ax.grid(color="0.9", lw=0.5)
            ax.tick_params(labelsize=7)
            if i == 1:
                ax.set_xlabel("absorption-weighted relative path A", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("T4 true absorption-weighted PPDF (ΔA = 0.02 bins) vs gamma", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "slab_ppdf_gamma.png", dpi=130)
    plt.close(fig)

    if t3_df is not None:
        levels = [lv[0] for lv in T3_LEVELS]
        metrics = [("k1_med_rel_bias", "k1 median rel. bias"), ("k1_rel_scatter", "k1 rel. scatter"),
                   ("k2_med_rel_bias", "k2 median rel. bias"), ("k2_rel_scatter", "k2 rel. scatter"),
                   ("kappa_med_log10_ratio", "log10 κ_fit/κ_true")]
        for kind in ("abs", "geo784"):
            fig, axes = plt.subplots(2, len(metrics), figsize=(16, 6.5), squeeze=False)
            for i, band in enumerate(("o2a", "sco2")):
                d = t3_df[(t3_df.kind == kind) & (t3_df.band == band) & (t3_df.subset == "all") & (t3_df.target == "moments")]
                for j, (mcol, ml) in enumerate(metrics):
                    ax = axes[i, j]
                    for tag in FITTERS:
                        g = d[d.fitter == tag].groupby("level")[mcol].median().reindex(levels)
                        ax.plot(range(len(levels)), g.values, marker="o", ms=4, lw=1.6, color=COLORS[tag], label=tag)
                    ax.set_xticks(range(len(levels)))
                    ax.set_xticklabels(levels, rotation=30, fontsize=7)
                    ax.axhline(0, color="0.5", lw=0.6)
                    if "scatter" in mcol:
                        ax.set_yscale("log")
                        ax.set_ylim(1e-3, 20)
                    elif "k2" in mcol:
                        ax.set_yscale("symlog", linthresh=0.1)
                    ax.set_title(f"{band}: {ml}", fontsize=8)
                    ax.grid(color="0.9", lw=0.5)
                    ax.tick_params(labelsize=7)
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(f"T3 dense synthetic spectra from the {kind} PPDF (median over dark+bright 3-D columns)",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(OUT / f"slab_t3_{kind}.png", dpi=130)
            plt.close(fig)


# --------------------------------------------------------------------------
# Summary markdown (tables for PLAN.md)
# --------------------------------------------------------------------------
def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{nd}f}"


def _md(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> Markdown table (no tabulate dependency)."""
    df = df.reset_index()
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(v, 3) if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def write_summary(closure_df, resid_df, tstar_df, shape_df, t3_df, truth_df, meta, cases):
    lines = ["# Slab closure summary (auto-generated by slab_closure.py)", ""]
    lines.append(f"l_direct = {meta['l_direct_km']:.2f} km, L_peak = {meta['l_peak_km']:.1f} km, "
                 f"shape-mode share = {cases['dark/3d'].meta['shape_frac']:.3f}")
    lines.append("")
    # truth overview
    lines.append("## Truth per region (3-D, medians)")
    lines.append("")
    lines.append("| case | region | ⟨l'⟩ geo784 | var l' geo784 | ⟨A⟩ | var A | var A between-bin | K1(0.3) | K2(0.3) | K2(1.0) | missing mass | rebuild rms |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (k, r), g in truth_df.groupby(["case", "region"]):
        lines.append(f"| {k} | {r} | {_fmt(g.m_geo784.median())} | {_fmt(g.v_geo784.median(), 4)} | "
                     f"{_fmt(g.m_abs.median())} | {_fmt(g.v_abs.median())} | {_fmt(g.vb_abs.median())} | "
                     f"{_fmt(g['K1_abs@0.3'].median())} | {_fmt(g['K2_abs@0.3'].median())} | "
                     f"{_fmt(g['K2_abs@1.0'].median())} | {_fmt(g.missing_mass.median(), 4)} | {_fmt(g.rebuild_rms.median(), 4)} |")
    lines.append("")
    for setname in ("T1_mc", "T2_teq_abs", "T2_teq_geo784", "T5_mc_eta", "T6_mc_noanchor"):
        d = closure_df[closure_df.set == setname]
        for par, tnames in (("k1", ("m_abs", "K1_abs@0.3", "m_geo784")), ("k2", ("v_abs", "K2_abs@0.3", "v_geo784")),
                            ("kappa", ("kappa_abs",))):
            lines.append(f"## {setname}: {par}")
            lines.append("")
            lines.append("| case | fitter | target | subset | r [95% CI] | slope | intercept | med bias (±run) | med rel bias | rms | k2≤0 |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            sub = d[(d.param == par) & d.target.isin(tnames) & d.subset.isin(["all", "clear", "far"])]
            for _, r in sub.iterrows():
                ci = f" [{_fmt(r.get('r_lo'), 2)}, {_fmt(r.get('r_hi'), 2)}]" if np.isfinite(r.get("r_lo", np.nan)) else ""
                sr = f" ±{_fmt(r.get('med_bias_sig_run'), 4)}" if np.isfinite(r.get("med_bias_sig_run", np.nan)) else ""
                lines.append(f"| {r.case} | {r.fitter} | {r.target} | {r.subset} | {_fmt(r.get('r'))}{ci} | "
                             f"{_fmt(r.get('slope'))} | {_fmt(r.get('intercept'))} | {_fmt(r.get('med_bias'), 4)}{sr} | "
                             f"{_fmt(r.get('med_rel_bias'))} | {_fmt(r.get('rms'), 4)} | {_fmt(r.get('frac_k2_le0'), 2)} |")
            lines.append("")
    lines.append("## T2 theorem residual rms (ln T), median by region")
    lines.append("")
    lines.append(_md(resid_df.groupby(["case", "ref", "region"])[["rms", "rms_absorbing", "max_abs", "corr_resid_over_tau_eta", "anchor_step"]]
                 .median()))
    lines.append("")
    lines.append("## τ* where the local cumulant of A equals the fitted value (median [IQR])")
    lines.append("")
    agg = tstar_df.groupby(["case", "set", "fitter", "region"]).agg(
        k1_med=("tau_star_k1", "median"), k1_q1=("tau_star_k1", lambda s: s.quantile(0.25)),
        k1_q3=("tau_star_k1", lambda s: s.quantile(0.75)), k1_nan=("tau_star_k1", lambda s: s.isna().mean()),
        k2_med=("tau_star_k2", "median"), k2_q1=("tau_star_k2", lambda s: s.quantile(0.25)),
        k2_q3=("tau_star_k2", lambda s: s.quantile(0.75)), k2_nan=("tau_star_k2", lambda s: s.isna().mean()))
    lines.append(_md(agg))
    lines.append("")
    lines.append("## T4 shape (medians by region)")
    lines.append("")
    lines.append(_md(shape_df.groupby(["case", "ppdf", "region"])[["kappa", "skew", "skew_gamma", "exkurt", "exkurt_gamma",
                                                                "gamma_fit_max_dlnT", "gamma_moment_max_dlnT"]]
                 .median()))
    lines.append("")
    if t3_df is not None:
        lines.append("## T3 dense synthetic (median over dark+bright 3-D, subset all)")
        lines.append("")
        d = t3_df[t3_df.subset == "all"].groupby(["kind", "target", "band", "level", "fitter"])[
            ["k1_med_rel_bias", "k1_rel_scatter", "k1_r", "k1_slope", "k2_med_rel_bias", "k2_slope", "k2_rel_scatter", "k2_r", "k2_frac_bound",
             "kappa_med_log10_ratio", "kappa_log10_scatter"]].median()
        lines.append(_md(d))
        lines.append("")
    (OUT / "slab_closure_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

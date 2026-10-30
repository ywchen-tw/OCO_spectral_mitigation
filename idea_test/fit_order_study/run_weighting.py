"""Weighting test: pooled variance-function weights (+ Huber IRLS) vs unit/noise weights.

Stage 1  first-pass unit-weight fits (order 2, gamma_u) and production-order refs.
Stage 2  pooled far-clear variance functions σ²(τ) per model × band (× fp) and
         a per-channel pooled variance (``vfc``).
Stage 3  second-pass fits for every weighting variant on all soundings.
Stage 4  stability, near-cloud signal, Huber diagnostics, figures, tables.

Usage:
    python idea_test/fit_order_study/run_weighting.py --l1b ... --fp-tau ... --fit-details ...
Outputs (results/): weight_per_sounding_<orbit>.parquet, weight_summary_<orbit>.csv,
weight_signal_<orbit>.csv, weight_varfunc_<orbit>.csv, weight_summary_<orbit>.md, weight_*.png.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orbit_io import BANDS, load_orbit  # noqa: E402
from run_study import (FAR_KM, NEAR_KM, PROD_TAG, aggregate, md_table,  # noqa: E402
                       robust_sd)
from variants import FIELDS, fit_cumulant  # noqa: E402
from weighting import (EXTRA_FIELDS, P_GRID, POOL_BINS, VarFunc, binned_variance,  # noqa: E402
                       c2_model, fit_varfunc, fit_weighted, gamma_model, gamma_x0, tab_varfunc,
                       k1_kernel, nan_result, sounding_varfunc)

MID_KM = 10.0
LOCAL_REF_KM = 10.0     # local reference footprints: cld_dist > 10 km and QF0
LOCAL_DLAT = 1.0        # ± latitude window for the local reference
N_BOOT = 400
ALL_COLS = (*FIELDS, *EXTRA_FIELDS)

# (tag, model, weight spec, Huber cutoff)
VARIANTS = [
    ("ols_2", "c2", "unit", None),
    ("wls_2", "c2", "noise", None),
    ("vf_2", "c2", "vf_fp", None),
    ("vfb_2", "c2", "vf_band", None),
    ("vfp2_2", "c2", "vf_fp_p2", None),
    ("vfn_2", "c2", "vf_np", None),
    ("vfm_2", "c2", "vf_msq", None),
    ("vfh25_2", "c2", "vf_fp", 2.5),
    ("vfh_2", "c2", "vf_fp", 3.0),
    ("vfh4_2", "c2", "vf_fp", 4.0),
    ("vfs_2", "c2", "vfs", None),
    ("vfc_2", "c2", "vfc", None),
    ("vfch_2", "c2", "vfc", 3.0),
    ("sm_2", "c2", "solar_mask", None),
    ("gamma_u", "gamma", "unit", None),
    ("gamma", "gamma", "noise", None),
    ("vf_g", "gamma", "vf_fp", None),
    ("vfb_g", "gamma", "vf_band", None),
    ("vfp2_g", "gamma", "vf_fp_p2", None),
    ("vfn_g", "gamma", "vf_np", None),
    ("vfm_g", "gamma", "vf_msq", None),
    ("vfh25_g", "gamma", "vf_fp", 2.5),
    ("vfh_g", "gamma", "vf_fp", 3.0),
    ("vfh4_g", "gamma", "vf_fp", 4.0),
    ("vfs_g", "gamma", "vfs", None),
    ("vfc_g", "gamma", "vfc", None),
    ("sm_g", "gamma", "solar_mask", None),
]
SOLAR_DEPTH_MASK = 0.05     # fixed channel mask: mean solar-line depth > 5 %
DOWN_TAGS = ("vfh_2", "vfh_g", "vfch_2")
CLASSES = ("far", "near", "mid", "all")

_G: dict = {}


# --------------------------------------------------------------------------- stage 1


def _unit(t, ch, s):
    return np.ones_like(t)


def _noise(t, ch, s):
    return s**2


def _stage1(j_list):
    tau, ln_t, sig = _G["tau"], _G["ln_t"], _G["sig"]
    c2 = np.full((len(j_list), 3, 3), np.nan)
    gm = np.full((len(j_list), 3, 3), np.nan)
    prod = np.full((len(j_list), 3, 2), np.nan)       # ols_3, ols_7 k1
    for ii, j in enumerate(j_list):
        for ib in range(3):
            args = (tau[ib, j], ln_t[ib, j], sig[ib, j])
            r = fit_weighted("c2", *args, _unit, _unit)
            c2[ii, ib] = (r["intercept"], r["k1"], r["k2"])
            if np.isfinite(r["k1"]):
                g = fit_weighted("gamma", *args, _unit, _unit, x0=gamma_x0(r))
                if np.isfinite(g["k1"]):
                    gm[ii, ib] = (g["intercept"], g["k1"], g["kappa"])
            prod[ii, ib, 0] = fit_cumulant(*args, 3, weighted=False)["k1"]
            prod[ii, ib, 1] = fit_cumulant(*args, 7, weighted=False)["k1"]
    return j_list, c2, gm, prod


# --------------------------------------------------------------------------- stage 2


def first_pass_resid(model, params, tau, ln_t):
    """Residual [N, C] of the first-pass fit for one band."""
    f = c2_model if model == "c2" else gamma_model
    p = [params[:, k][:, None] for k in range(3)]
    return ln_t - f(p, tau)


def build_varfuncs(orb_tau, orb_lnt, orb_sig, params, fp, far):
    """Pooled far-clear variance functions and per-channel variances.

    Returns vf dict keyed (model, spec, ib, fp), per-channel variance arrays
    chanvar[model] [3, 8, C], a parameter table and binned data for plots.
    """
    vf, chanvar, rows, binned = {}, {}, [], {}
    for model in ("c2", "gamma"):
        cv = np.full((3, 8, orb_tau.shape[2]), np.nan)
        for ib, band in enumerate(BANDS):
            tau, lnt, sig = orb_tau[ib], orb_lnt[ib], orb_sig[ib]
            r = first_pass_resid(model, params[model][:, ib], tau, lnt)
            nv = sig.astype(np.float64) ** 2
            ok_far = far & np.all(np.isfinite(params[model][:, ib]), axis=1)
            # band-only function (all fp pooled)
            sel = ok_far
            b_all = binned_variance(tau[sel].ravel(), r[sel].ravel(), nv[sel].ravel(), POOL_BINS)
            f_all, rms_all, costs_all = fit_varfunc(b_all)
            f_all_p2, rms_p2, _ = fit_varfunc(b_all, p=2.0)
            binned[(model, ib, -1)] = (b_all, f_all, f_all_p2, costs_all)
            rows.append(dict(model=model, band=band, fp="all", n_sound=int(sel.sum()),
                             a=f_all.a, b=f_all.b, p=f_all.p, logrms=rms_all,
                             a_p2=f_all_p2.a, b_p2=f_all_p2.b, logrms_p2=rms_p2,
                             a_over_b=f_all.a / f_all.b if f_all.b > 0 else np.inf))
            for ifp in range(8):
                sel = ok_far & (fp == ifp)
                b_fp = binned_variance(tau[sel].ravel(), r[sel].ravel(), nv[sel].ravel(), POOL_BINS)
                f_fp, rms_fp, costs_fp = fit_varfunc(b_fp)
                f_fp_p2, rms_fp_p2, _ = fit_varfunc(b_fp, p=2.0)
                binned[(model, ib, ifp)] = (b_fp, f_fp, f_fp_p2, costs_fp)
                vf[(model, "vf_fp", ib, ifp)] = f_fp
                vf[(model, "vf_fp_p2", ib, ifp)] = f_fp_p2
                vf[(model, "vf_band", ib, ifp)] = f_all
                vf[(model, "vf_np", ib, ifp)] = tab_varfunc(b_fp, "var")
                vf[(model, "vf_msq", ib, ifp)] = tab_varfunc(b_fp, "msq")
                rows.append(dict(model=model, band=band, fp=str(ifp + 1), n_sound=int(sel.sum()),
                                 a=f_fp.a, b=f_fp.b, p=f_fp.p, logrms=rms_fp,
                                 a_p2=f_fp_p2.a, b_p2=f_fp_p2.b, logrms_p2=rms_fp_p2,
                                 a_over_b=f_fp.a / f_fp.b if f_fp.b > 0 else np.inf))
                # per-channel mean-square residual (systematic + random), noise included
                rr = r[sel]
                nfin = np.isfinite(rr).sum(axis=0)
                msq = np.nanmean(rr**2, axis=0)
                cv[ib, ifp] = np.where(nfin >= 20, msq, np.nan)
        chanvar[model] = cv
    return vf, chanvar, pd.DataFrame(rows), binned


# --------------------------------------------------------------------------- stage 3


def _var_fn(spec, model, ib, j):
    fpj = _G["fp"][j]
    if spec == "unit":
        return _unit
    if spec == "noise":
        return _noise
    if spec in ("vf_fp", "vf_band", "vf_fp_p2", "vf_np", "vf_msq"):
        f = _G["vf"][(model, spec, ib, fpj)]
        return lambda t, ch, s: f.total(t, s)
    if spec == "vfc":
        cv = _G["chanvar"][model][ib, fpj]
        fb = _G["vf"][(model, "vf_msq", ib, fpj)]
        return lambda t, ch, s: np.where(np.isfinite(cv[ch]), cv[ch], fb.total(t, s))
    if spec == "solar_mask":
        # unit weights with solar-line channels effectively removed (fixed per orbit × band × fp)
        masked = _G["solar_mask"][ib, fpj]
        return lambda t, ch, s: np.where(masked[ch], 1e8, 1.0)
    if spec == "vfs":
        params = _G["params"][model][j, ib]
        tau, lnt, sig = _G["tau"][ib, j], _G["ln_t"][ib, j], _G["sig"][ib, j]
        f = None
        if np.all(np.isfinite(params)):
            r = lnt - (c2_model if model == "c2" else gamma_model)(params, tau)
            f = sounding_varfunc(tau.astype(float), r.astype(float),
                                 sig.astype(float) ** 2, _G["vf"][(model, "vf_band", ib, fpj)].p)
        if f is None:
            return None
        return lambda t, ch, s: f.total(t, s)
    raise ValueError(spec)


def _stage3(j_list):
    tau, ln_t, sig, cls = _G["tau"], _G["ln_t"], _G["sig"], _G["cls"]
    nchan = tau.shape[2]
    down = {tag: np.zeros((len(CLASSES), 3, 8, nchan)) for tag in DOWN_TAGS}
    down_n = {tag: np.zeros((len(CLASSES), 3, 8)) for tag in DOWN_TAGS}
    rows = []
    for j in j_list:
        fpj = _G["fp"][j]
        for ib, band in enumerate(BANDS):
            args = (tau[ib, j], ln_t[ib, j], sig[ib, j])
            # common calibration of the actual error variance: binned mean square, per fp
            cal = {m: _var_fn("vf_msq", m, ib, j) for m in ("c2", "gamma")}
            gx0 = _G["params"]["gamma"][j, ib]
            gx0 = gx0 if np.all(np.isfinite(gx0)) else gamma_x0(None)
            for tag, model, spec, hub in VARIANTS:
                vfn = _var_fn(spec, model, ib, j)
                want_down = tag in DOWN_TAGS
                if vfn is None:
                    res = nan_result(want_down)
                else:
                    res = fit_weighted(model, *args, vfn, cal[model], huber_c=hub,
                                       x0=gx0 if model == "gamma" else None,
                                       return_down=want_down)
                if want_down and res.get("_down") is not None:
                    for ic, c in enumerate(CLASSES):
                        if cls[c][j]:
                            down[tag][ic, ib, fpj] += res["_down"]
                            down_n[tag][ic, ib, fpj] += 1
                rows.append((j, band, tag, *[res.get(f, np.nan) for f in ALL_COLS]))
    return rows, down, down_n


# --------------------------------------------------------------------------- stage 4


def lat_local_ref(rel, lat, ref_mask):
    """Median rel of reference footprints within ±LOCAL_DLAT of each footprint (NaN if < 15)."""
    ok = ref_mask & np.isfinite(rel)
    lr, vr = lat[ok], rel[ok]
    order = np.argsort(lr)
    lr, vr = lr[order], vr[order]
    lo = np.searchsorted(lr, lat - LOCAL_DLAT)
    hi = np.searchsorted(lr, lat + LOCAL_DLAT)
    out = np.full(len(rel), np.nan)
    for i in range(len(rel)):
        if hi[i] - lo[i] >= 15:
            out[i] = np.median(vr[lo[i]:hi[i]])
    return out


def signal_metrics(long, meta, fp, rng):
    cld, qf, lat = meta["cld_dist_km"], meta["xco2_qf"], meta["lat"]
    anom = meta["xco2_bc_anomaly"]
    far = (cld > FAR_KM) & (qf == 0)
    near = cld < NEAR_KM
    mid = (cld >= NEAR_KM) & (cld < MID_KM)
    ref_local = (cld > LOCAL_REF_KM) & (qf == 0)
    wide = long.pivot_table(index="j", columns=["band", "variant"], values="k1")
    j = wide.index.values
    fpj = fp[j]
    recs, boot_store = [], {}
    tags = [v[0] for v in VARIANTS]
    for band in BANDS:
        for tag in tags:
            k1 = wide[(band, tag)].values
            k1s = pd.Series(k1)
            fpm = k1s[far[j]].groupby(fpj[far[j]]).mean()
            rel = k1 / fpm.reindex(fpj).values - 1.0
            loc = lat_local_ref(rel, lat[j], ref_local[j])
            rel_loc = rel - loc
            f_far, f_near, f_mid = far[j], near[j], mid[j]
            sd_far = robust_sd(rel[f_far])
            sd_far_loc = robust_sd(rel_loc[f_far])
            rec = dict(band=band, variant=tag, sd_far_pct=100 * sd_far)
            for name, m in (("near", f_near), ("mid", f_mid)):
                sh = np.nanmedian(rel[m])
                rec[f"shift_{name}_pct"] = 100 * sh
                rec[f"snr_{name}"] = sh / sd_far
                mq = m & (qf[j] == 0)
                rec[f"snr_{name}_qf0"] = np.nanmedian(rel[mq]) / sd_far
                ml = m & np.isfinite(rel_loc)
                rec[f"shift_{name}_loc_pct"] = 100 * np.nanmedian(rel_loc[ml])
                rec[f"snr_{name}_loc"] = np.nanmedian(rel_loc[ml]) / sd_far_loc
                rec[f"n_{name}_loc"] = int(ml.sum())
                ma = m & np.isfinite(anom[j]) & np.isfinite(k1)
                rec[f"corr_anom_{name}"] = (float(np.corrcoef(k1[ma], anom[j][ma])[0, 1])
                                            if ma.sum() > 30 else np.nan)
            recs.append(rec)
            boot_store[(band, tag)] = rel
    sig = pd.DataFrame(recs)

    # paired bootstrap of the near/far SNR difference against the unit-weight baseline
    i_far = np.flatnonzero(far[j])
    i_near = np.flatnonzero(near[j])
    i_mid = np.flatnonzero(mid[j])
    draws = [(rng.choice(i_far, len(i_far)), rng.choice(i_near, len(i_near)),
              rng.choice(i_mid, len(i_mid))) for _ in range(N_BOOT)]

    def snr_draws(rel):
        out = np.empty((N_BOOT, 2))
        for b, (a, n, m) in enumerate(draws):
            sd = robust_sd(rel[a])
            out[b] = (np.nanmedian(rel[n]) / sd, np.nanmedian(rel[m]) / sd)
        return out

    base_snr = {}
    for band in BANDS:
        for base in ("ols_2", "gamma_u"):
            base_snr[(band, base)] = snr_draws(boot_store[(band, base)])
    lo_n, hi_n, lo_m, hi_m = [], [], [], []
    for _, r in sig.iterrows():
        base = "ols_2" if r.variant.endswith("_2") else "gamma_u"
        d = snr_draws(boot_store[(r.band, r.variant)]) - base_snr[(r.band, base)]
        lo_n.append(np.percentile(d[:, 0], 2.5)); hi_n.append(np.percentile(d[:, 0], 97.5))
        lo_m.append(np.percentile(d[:, 1], 2.5)); hi_m.append(np.percentile(d[:, 1], 97.5))
    sig["dsnr_near_lo"], sig["dsnr_near_hi"] = lo_n, hi_n
    sig["dsnr_mid_lo"], sig["dsnr_mid_hi"] = lo_m, hi_m
    return sig


def extra_metrics(long, meta, fp):
    cld, qf = meta["cld_dist_km"], meta["xco2_qf"]
    far = (cld > FAR_KM) & (qf == 0)
    near = cld < NEAR_KM
    recs = []
    for (band, tag), d in long.groupby(["band", "variant"], sort=False):
        jj = d.j.values
        f, n = far[jj], near[jj]
        k1 = d.k1.values
        rec = dict(band=band, variant=tag,
                   chi2_cal_far=np.nanmedian(d.chi2_cal.values[f]),
                   chi2_own_far=np.nanmedian(d.chi2_own.values[f]),
                   chi2_cal_near=np.nanmedian(d.chi2_cal.values[n]),
                   sig_k1_cal_pct=100 * np.nanmedian(d.sig_k1_cal.values[f] / k1[f]),
                   sig_k1_infl_pct=100 * np.nanmedian(d.sig_k1_infl.values[f] / k1[f]),
                   sig_k2_infl=np.nanmedian(d.sig_k2_infl.values[f]),
                   lag1_std_far=np.nanmedian(d.lag1_std.values[f]),
                   frac_down_far=np.nanmedian(d.frac_down.values[f]),
                   frac_down_near=np.nanmedian(d.frac_down.values[n]),
                   frac_down_mean_far=np.nanmean(d.frac_down.values[f]),
                   frac_down_mean_near=np.nanmean(d.frac_down.values[n]),
                   k2_le0_frac=float(np.nanmean(d.k2.values[f] <= 1e-6)),
                   fit_fail_frac=float(np.mean(~np.isfinite(k1))))
        recs.append(rec)
    return pd.DataFrame(recs)


def wavelengths(l1b_file):
    with h5py.File(l1b_file, "r") as f:
        coef = f["InstrumentHeader/dispersion_coef_samp"][...]       # [3, 8, 6]
    pix = np.arange(1, 1017, dtype=float)
    return np.stack([[np.polyval(coef[ib, ifp][::-1], pix) for ifp in range(8)] for ib in range(3)])


def solar_line_depth(fp_tau_file, idx, fp):
    """Median over the given soundings of 1 − TOA/upper envelope, per band and fp [3, 8, 1016]."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    out = np.full((3, 8, 1016), np.nan)
    with h5py.File(fp_tau_file, "r") as f:
        for ib, b in enumerate(BANDS):
            toa = f[f"{b}_toa_sol_output"][...]
            for ifp in range(8):
                sel = idx[fp[idx] == ifp]
                med = np.median(toa[sel], axis=0)
                env = uniform_filter1d(maximum_filter1d(med, 25), 25)
                out[ib, ifp] = 1.0 - med / env
    return out


# --------------------------------------------------------------------------- figures


def fig_varfunc(binned, out_dir, orbit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for col, model in enumerate(("c2", "gamma")):
        for ib, band in enumerate(BANDS):
            ax = axes[ib, col]
            b, f, f2, _ = binned[(model, ib, -1)]
            ax.loglog(b["tau"], b["var"], "ko", ms=4, label="robust var (all fp)")
            ax.loglog(b["tau"], b["msq"], "x", color="0.5", ms=4, label="mean square")
            ax.loglog(b["tau"], b["noise"], ":", color="C7", label="noise²")
            tt = np.geomspace(b["tau"].min(), b["tau"].max(), 100)
            nn = np.interp(tt, b["tau"], b["noise"])
            ax.loglog(tt, nn + f.misfit_var(tt), "C3-", label=f"a+bτ^p, p={f.p:.2f}")
            ax.loglog(tt, nn + f2.misfit_var(tt), "C0--", label="a+bτ²")
            for ifp in range(8):
                bf = binned[(model, ib, ifp)][0]
                ax.loglog(bf["tau"], bf["var"], "-", color="C2", lw=0.6, alpha=0.6)
            ax.set_title(f"{band}, first pass {'ols_2' if model == 'c2' else 'gamma_u'}")
            ax.set_xlabel("τ"); ax.set_ylabel("residual variance (ln T)²")
            ax.grid(alpha=0.3, which="both")
            if ib == 0 and col == 0:
                ax.legend(fontsize=7)
        # bin median (shape misfit) in the third column for the c2 model only
    for ib, band in enumerate(BANDS):
        ax = axes[ib, 2]
        for model, c in (("c2", "C3"), ("gamma", "k")):
            b = binned[(model, ib, -1)][0]
            ax.semilogx(b["tau"], b["med"], "o-", color=c, ms=3,
                        label=f"{'ols_2' if model == 'c2' else 'gamma_u'} bin median")
            ax.semilogx(b["tau"], np.sqrt(b["var"]), "--", color=c, lw=0.8, label="robust SD")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(f"{band}: systematic shape misfit vs scatter"); ax.set_xlabel("τ")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / f"weight_varfunc_{orbit}.png", dpi=120); plt.close(fig)


def fig_downweight(down_freq, chanvar_ratio, depth, wl, out_dir, orbit, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    for ib, band in enumerate(BANDS):
        ax = axes[ib]
        x = wl[ib, 0, 1:-1] * 1e3
        ax.plot(x, down_freq["far"][ib], "-", color="C0", lw=0.7, label="far-clear")
        ax.plot(x, down_freq["near"][ib], "-", color="C3", lw=0.7, alpha=0.7, label="cld_dist < 3 km")
        ax.set_ylabel("down-weight frequency")
        ax.set_xlabel("wavelength (fp 1) [nm]")
        ax2 = ax.twinx()
        ax2.plot(x, depth[ib, 0, 1:-1], color="C1", lw=0.5, alpha=0.6, label="solar line depth")
        ax2.plot(x, chanvar_ratio[ib], color="C2", lw=0.5, alpha=0.6,
                 label="pooled per-channel MSE / σ²_vf(τ) / 10")
        ax2.set_ylim(0, 1)
        ax.set_title(f"{band}: {tag} Huber down-weight frequency (mean over fp)")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"weight_downweight_{tag}_{orbit}.png", dpi=120)
    plt.close(fig)


def fig_signal(summary, sig, out_dir, orbit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = summary.merge(sig, on=["band", "variant"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ib, band in enumerate(BANDS):
        for row, (ycol, lab) in enumerate((("snr_near", "near (<3 km) shift / far spread"),
                                           ("snr_near_loc", "local near shift / far spread"))):
            ax = axes[row, ib]
            d = m[m.band == band]
            for _, r in d.iterrows():
                c = "C0" if r.variant.endswith("_2") else "C3"
                ax.plot(r.k1_spread_pct, r[ycol], "o", color=c)
                ax.annotate(r.variant, (r.k1_spread_pct, r[ycol]), fontsize=7)
            ax.set_xscale("log"); ax.set_xlabel("far-clear k1 spread [%]")
            ax.set_ylabel(lab); ax.set_title(band); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"weight_snr_vs_spread_{orbit}.png", dpi=120)
    plt.close(fig)


def fig_kernel(orb_tau, orb_lnt, orb_sig, params, vf, chanvar, fp, far, out_dir, orbit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jx = np.flatnonzero(far)[len(np.flatnonzero(far)) // 2]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    rows = []
    for row, model in enumerate(("c2", "gamma")):
        for ib, band in enumerate(BANDS):
            t, y, s = orb_tau[ib, jx].astype(float), orb_lnt[ib, jx].astype(float), orb_sig[ib, jx].astype(float)
            ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(s)
            ch = np.flatnonzero(ok)
            t, s = t[ok], s[ok]
            f = vf[(model, "vf_fp", ib, fp[jx])]
            fm = vf[(model, "vf_msq", ib, fp[jx])]
            cv = chanvar[model][ib, fp[jx]][ch]
            ax = axes[row, ib]
            o = np.argsort(t)
            p = params[model][jx, ib]
            for lab, var, c in (("unit", np.ones_like(t), "C0"), ("noise", s**2, "C1"),
                                ("vf", f.total(t, s), "C3"),
                                ("vfm", fm.total(t, s), "C4"),
                                ("vfc", np.where(np.isfinite(cv), cv, fm.total(t, s)), "C2")):
                h = k1_kernel(model, p, t, 1.0 / np.sqrt(var))
                ah = np.abs(h)
                ax.plot(t[o], np.cumsum(ah[o]) / ah.sum(), "-", color=c, label=lab)
                terc = np.quantile(t, [1 / 3, 2 / 3])
                rows.append(dict(model=model, band=band, weights=lab,
                                 low=ah[t <= terc[0]].sum() / ah.sum(),
                                 mid=ah[(t > terc[0]) & (t <= terc[1])].sum() / ah.sum(),
                                 high=ah[t > terc[1]].sum() / ah.sum()))
            ax.set_xscale("log"); ax.set_xlabel("τ")
            ax.set_ylabel("cumulative |k1 kernel| fraction")
            ax.set_title(f"{band} ({'order 2' if model == 'c2' else 'gamma'})")
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / f"weight_k1_kernel_{orbit}.png", dpi=120); plt.close(fig)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1b", required=True)
    ap.add_argument("--fp-tau", required=True)
    ap.add_argument("--fit-details", required=True)
    ap.add_argument("--orbit", default=None)
    ap.add_argument("--max-soundings", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    orbit = args.orbit or Path(args.fp_tau).parent.name
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    orb = load_orbit(args.l1b, args.fp_tau, args.fit_details, args.max_soundings)
    n = len(orb.sounding_id)
    meta, fp = orb.meta, orb.fp_number
    cld, qf = meta["cld_dist_km"], meta["xco2_qf"]
    far = (cld > FAR_KM) & (qf == 0)
    cls = {"far": far, "near": cld < NEAR_KM, "mid": (cld >= NEAR_KM) & (cld < MID_KM),
           "all": np.ones(n, bool)}
    print(f"loaded {n} soundings ({far.sum()} far-clear) in {time.time() - t0:.0f} s", flush=True)
    tau, lnt, sig = orb.tau[:, :, 1:-1], orb.ln_t[:, :, 1:-1], orb.sigma_ln_t[:, :, 1:-1]
    _G.update(tau=tau, ln_t=lnt, sig=sig, fp=fp, cls=cls)
    chunks = np.array_split(np.arange(n), max(1, min(n // 50, 6 * args.workers)))

    def run(fn):
        if args.workers > 1:
            with mp.get_context("fork").Pool(args.workers) as pool:
                return pool.map(fn, chunks)
        return [fn(c) for c in chunks]

    # ---- stage 1
    t0 = time.time()
    params = {"c2": np.full((n, 3, 3), np.nan), "gamma": np.full((n, 3, 3), np.nan)}
    prod = np.full((n, 3, 2), np.nan)
    for jl, c2, gm, pr in run(_stage1):
        params["c2"][jl], params["gamma"][jl], prod[jl] = c2, gm, pr
    print(f"stage 1 in {time.time() - t0:.0f} s; gamma_u failures "
          f"{np.mean(~np.isfinite(params['gamma'][:, :, 1]), axis=0)}", flush=True)
    repro = {}
    for ib, band in enumerate(BANDS):
        k = prod[:, ib, 0 if PROD_TAG[band] == "ols_3" else 1]
        ref = meta[f"{band}_k1"]
        ok = np.isfinite(k) & np.isfinite(ref)
        repro[band] = float(np.median(np.abs(k[ok] - ref[ok]) / ref[ok]))
    print("reproduction of production nosg k1:", repro, flush=True)

    # ---- stage 2
    vf, chanvar, vf_tab, binned = build_varfuncs(tau, lnt, sig, params, fp, far)
    vf_tab.insert(0, "orbit", orbit)
    vf_tab.to_csv(out_dir / f"weight_varfunc_{orbit}.csv", index=False)
    print(vf_tab[vf_tab.fp == "all"].to_string(), flush=True)
    depth_far = solar_line_depth(args.fp_tau, np.flatnonzero(far), fp)[:, :, 1:-1]
    _G.update(vf=vf, chanvar=chanvar, params=params, solar_mask=depth_far > SOLAR_DEPTH_MASK)
    print("solar-mask channel fraction per band:", (depth_far > SOLAR_DEPTH_MASK).mean(axis=(1, 2)), flush=True)

    # ---- stage 3
    t0 = time.time()
    rows, down, down_n = [], None, None
    for r, d, dn in run(_stage3):
        rows.extend(r)
        if down is None:
            down, down_n = d, dn
        else:
            for tag in DOWN_TAGS:
                down[tag] += d[tag]; down_n[tag] += dn[tag]
    long = pd.DataFrame(rows, columns=["j", "band", "variant", *ALL_COLS])
    for ib, band in enumerate(BANDS):
        for ip, tag in enumerate(("ols_3", "ols_7")):
            d = pd.DataFrame({"j": np.arange(n), "band": band, "variant": tag, "k1": prod[:, ib, ip]})
            long = pd.concat([long, d], ignore_index=True)
    long["bound"] = long["bound"].astype(float)
    print(f"stage 3: {len(long)} rows in {time.time() - t0:.0f} s", flush=True)
    long.to_parquet(out_dir / f"weight_per_sounding_{orbit}.parquet", index=False)

    # ---- stage 4
    rng = np.random.default_rng(1)
    base = aggregate(long, meta, fp)
    extra = extra_metrics(long, meta, fp)
    summary = base.merge(extra, on=["band", "variant"], how="left")
    summary.insert(0, "orbit", orbit)
    sig_tab = signal_metrics(long[long.variant.isin([v[0] for v in VARIANTS])], meta, fp, rng)
    sig_tab.insert(0, "orbit", orbit)
    summary.to_csv(out_dir / f"weight_summary_{orbit}.csv", index=False)
    sig_tab.to_csv(out_dir / f"weight_signal_{orbit}.csv", index=False)

    # Huber down-weight diagnostics
    wl = wavelengths(args.l1b)
    depth = solar_line_depth(args.fp_tau, np.flatnonzero(far), fp)
    down_rows, freq_store = [], {}
    for tag in DOWN_TAGS:
        freq = {c: down[tag][ic] / np.maximum(down_n[tag][ic][..., None], 1)
                for ic, c in enumerate(CLASSES)}                       # [3, 8, C]
        model = "c2" if tag.endswith("_2") else "gamma"
        fmean = {c: np.nanmean(freq[c], axis=1) for c in CLASSES}       # [3, C]
        ratio = np.full((3, tau.shape[2]), np.nan)
        for ib, band in enumerate(BANDS):
            # per-channel pooled MSE relative to the pooled vf at the channel's median τ
            tmed = np.nanmedian(tau[ib][far], axis=0)
            snm = np.nanmedian(sig[ib][far], axis=0)
            vfm = np.nanmean([vf[(model, "vf_fp", ib, k)].total(tmed, snm) for k in range(8)], axis=0)
            cvm = np.nanmean(chanvar[model][ib], axis=0)
            ratio[ib] = cvm / vfm / 10.0
            fa = fmean["far"][ib]
            ind = fa > 0.5
            # concentration: share of far-clear down-weight events in channels down-weighted > 50 % of the time
            conc = float(fa[ind].sum() / fa.sum()) if fa.sum() > 0 else np.nan
            # clustering: lag-1 autocorrelation of the frequency along channel index
            fz = fa - fa.mean()
            ac = float(fz[:-1] @ fz[1:] / (fz @ fz)) if fz @ fz > 0 else np.nan
            dd = depth[ib, :, 1:-1].mean(axis=0)
            okc = np.isfinite(dd) & np.isfinite(fa)
            fin = np.isfinite(ratio[ib]) & np.isfinite(fa)
            # overlap between the per-sounding Huber choice and the fixed per-channel MSE weights
            top = fa >= np.quantile(fa, 0.9)
            topm = ratio[ib] >= np.nanquantile(ratio[ib], 0.9)
            down_rows.append(dict(
                variant=tag, band=band,
                mean_frac_far=float(fa.mean()), mean_frac_near=float(fmean["near"][ib].mean()),
                mean_frac_mid=float(fmean["mid"][ib].mean()),
                n_chan_gt50pct=int(ind.sum()), share_events_in_gt50pct=conc,
                lag1_along_channel=ac,
                corr_solar_depth=float(np.corrcoef(fa[okc], dd[okc])[0, 1]),
                frac_chan_solar05=float(np.mean(dd > 0.05)),
                share_events_solar05=float(fa[dd > 0.05].sum() / fa.sum()) if fa.sum() > 0 else np.nan,
                share_events_solar05_near=float(fmean["near"][ib][dd > 0.05].sum() / fmean["near"][ib].sum())
                if fmean["near"][ib].sum() > 0 else np.nan,
                corr_chan_msq_ratio=float(np.corrcoef(fa[fin], ratio[ib][fin])[0, 1]),
                top10_overlap_msq=float(np.mean(topm[top])),
                corr_near_far_freq=float(np.corrcoef(fa, fmean["near"][ib])[0, 1]),
                tau_med_down=float(np.average(np.nanmedian(tau[ib][far], axis=0), weights=fa + 1e-12)),
                tau_med_all=float(np.nanmedian(tau[ib][far])),
                top_wavelengths_nm=", ".join(f"{x:.3f}" for x in
                                             np.sort(wl[ib, 0, 1:-1][np.argsort(fa)[-8:]] * 1e3))))
        fig_downweight(fmean, ratio, depth, wl, out_dir, orbit, tag)
        freq_store[tag] = np.stack([fmean[c] for c in CLASSES])
    down_tab = pd.DataFrame(down_rows)
    np.savez(out_dir / f"weight_downweight_freq_{orbit}.npz", classes=np.array(CLASSES),
             wavelength_um=wl[:, :, 1:-1], solar_depth=depth[:, :, 1:-1], **freq_store)
    down_tab.to_csv(out_dir / f"weight_downweight_{orbit}.csv", index=False)

    kern = fig_kernel(tau, lnt, sig, params, vf, chanvar, fp, far, out_dir, orbit)
    kern.to_csv(out_dir / f"weight_k1_kernel_{orbit}.csv", index=False)
    fig_varfunc(binned, out_dir, orbit)
    fig_signal(summary, sig_tab, out_dir, orbit)

    # ---- markdown
    stab_cols = ["variant", "k1_median", "k1_bias_vs_prod_pct", "k1_spread_pct", "k1_fp_mean_spread_pct",
                 "ratio_to_o2a_spread_pct", "chi2_own_far", "chi2_cal_far", "chi2_cal_near", "sig_k1_infl_pct",
                 "lag1_std_far", "frac_down_far", "frac_down_near", "k2_median", "k2_le0_frac"]
    sig_cols = ["variant", "sd_far_pct", "shift_near_pct", "snr_near", "dsnr_near_lo", "dsnr_near_hi",
                "shift_mid_pct", "snr_mid", "snr_near_loc", "snr_mid_loc", "snr_near_qf0",
                "corr_anom_near", "corr_anom_mid"]
    tags = [v[0] for v in VARIANTS] + ["ols_3", "ols_7"]
    parts = [f"# Weighting test — orbit {orbit}\n",
             "Reproduction of production `_nosg` k1 (median |Δk1|/k1): "
             + ", ".join(f"{b} {v:.2e}" for b, v in repro.items()) + "\n",
             "\n## Variance functions (far-clear, pooled over fp)\n",
             md_table(vf_tab[vf_tab.fp == "all"], ["model", "band", "n_sound", "a", "b", "p", "logrms",
                                                   "a_p2", "b_p2", "logrms_p2", "a_over_b"])]
    for band in BANDS:
        s = summary[summary.band == band].set_index("variant").reindex(tags).reset_index()
        parts += [f"\n## {band}: stability\n", md_table(s, stab_cols)]
        g = sig_tab[sig_tab.band == band]
        parts += [f"\n## {band}: near-cloud signal\n", md_table(g, sig_cols)]
    parts += ["\n## Huber down-weighting\n", md_table(down_tab, list(down_tab.columns))]
    parts += ["\n## k1 kernel share by τ tercile (one far-clear sounding)\n",
              md_table(kern, list(kern.columns))]
    (out_dir / f"weight_summary_{orbit}.md").write_text("\n".join(parts) + "\n")
    print((out_dir / f"weight_summary_{orbit}.md").read_text())


if __name__ == "__main__":
    main()

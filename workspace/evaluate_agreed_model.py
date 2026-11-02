#!/usr/bin/env python
"""Evaluate the agreed spectral-fit model against production (CURC handoff items 4 and 6).

Reads, per orbit, the variant ("dop") fitting_details file (results_doppler tree) and the production
fitting_details file (results tree; same soundings, same order, checked on fp_id), builds per-sounding
tables only (never spectra), and writes:

  agreed_eval_per_orbit.csv       one headline row per orbit
  agreed_eval_pooled.md           all tables (pooled over orbits/dates; per orbit when few orbits)
  agreed_eval_lat_breakdown.csv   item 4.3
  agreed_eval_anomaly_corr.csv    item 6a (Pearson/Spearman per coefficient, held-out ridge R^2 per set)
  agreed_eval_pseudo_xco2.csv     item 6b
  agreed_eval_flag.csv            item 6c (+ Kozienice along-track rows when 2021-09-06/38200a is input)
  agreed_eval_{farclear,contrasts,diagnostics,bars,target_check}.csv   items 4.1, 4.2, 4.4, 4.5, checks
  fig_*.png                       summary figures
  per_sounding/<date>_<orbit>.parquet   derived per-sounding columns used by the pooled stage

Definitions follow idea_test/fit_order_study (see agreed_model_eval_lib.py):
  far-clear = cld_dist > 20 km and xco2_qf == 0; robust SD = 1.4826 MAD; spread (%) = 100 rSD of
  v/mean_fp(v) - 1 over far-clear.  Latitude-local contrast vs the median of far references (cld > 10 km,
  QF0, same footprint, |dlat| <= 1 deg, >= 5); relative for k1-type quantities, absolute for k2, f, a1.
  SNR = bin median contrast / rSD of far-clear contrasts.  Bin statistics on QF0 unless stated.
  Coefficient local anomaly = coefficient - its mean over the pipeline reference window of the target
  (compute_xco2_anomaly window: +-0.25 deg, cld > r km, all footprints, std gate 1 ppm).
  Pseudo-XCO2 uses fixed kernels (SCO2 1.60, WCO2 1.01, O2A 1.80); nothing is regressed on xco2 to set a
  level or slope.  fs_rel, h2o_scale, albedo, dp are never predictors.

Usage:
  python workspace/evaluate_agreed_model.py --dates 2020-05-21 \
      --prod-dir /scratch/alpine/yuch8913/oco/spec_anal/results \
      --dop-dir /scratch/alpine/yuch8913/oco/spec_anal/results_doppler --out <dir> \
      [--orbits 31309a_GL] [--fixture <fitting_details file>] [--ref-orbit 29252a_GL]
"""
import numpy as np  # noqa: I001  (numpy before pyarrow: libstdc++ load order on CURC)
import argparse
import os
import re
import sys
import time
from pathlib import Path

import h5py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from agreed_model_eval_lib import (BANDS, BAND_UP, BINS, BIN_LABELS, KERNEL, FLAG_PCT, FAR_KM, KOZIENICE,  # noqa: E402
                                   robust_sd, nanmedian, nanpct, fp_relative, fp_demean_median, pearson,
                                   spearman, lat_local_contrast, bin_stats, window_stats, ridge_heldout,
                                   ols_r2, partial_r2, column_mean_prior, along_track_km)
from constants import anomaly_args  # noqa: E402

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
FORBIDDEN_OUT = ("/scratch/alpine/yuch8913/oco/spec_anal/results",)

# --------------------------------------------------------------------------- dataset registry
META_KEYS = ("cld_dist_km", "xco2_qf", "lat", "lon", "fp_number", "sfc_type", "land", "xco2_bc", "xco2",
             "xco2_raw", "xco2_apriori", "xco2_bc_anomaly", "xco2_raw_anomaly", "xco2_bc_anomaly_r05",
             "xco2_bc_anomaly_r15", "xco2_raw_anomaly_r05", "xco2_raw_anomaly_r15", "h2o_scale", "tcwv", "time")
PROF_KEYS = ("co2prior_profile", "p_profile", "q_profile")
# pool of the stored xco2_bc_anomaly (fitting.py ref_extra_vars): smoothed k1-k3, Lite albedo, exp(intercept)
POOL_KEYS = tuple([f"{b}_k{i}_fitting" for b in BANDS for i in (1, 2, 3)] + [f"alb_{b}" for b in BANDS]
                  + [f"exp_intercept_{b}" for b in BANDS])
PRODORDER = ("k1", "k2", "intercept")
EXTRA_NAMES = ("k1_o2", "k2_o2", "k1_dry_t2", "k1w_t2", "k2_t2", "rms_t2", "k1_am", "k1w_am", "k2_am", "a1_am",
               "a2_am", "f_am", "c_am", "rms_am", "k2diag_am", "ok_am")
AM_MASKED = ("k1_am", "k1w_am", "k2_am", "a1_am", "a2_am", "f_am", "c_am", "rms_am", "k2diag_am")

SOURCES = [("prod", "production (prod tree, prod tau, order-7 k1)"),
           ("dopprod", "variant production-order k1 (dop tree)"),
           ("o2", "k1_o2 (single-tau order 2, S2)"),
           ("t2", "k1_dry_t2 (two-tau T2)"),
           ("am", "k1_am (agreed model)")]
SRC_FIELD = {"o2": "k1_o2", "t2": "k1_dry_t2", "am": "k1_am"}
RATIOS = (("sco2_o2a", "sco2", "o2a"), ("wco2_sco2", "wco2", "sco2"))
K1_KEYS = list(BANDS) + [r[0] for r in RATIOS]

# contrast quantities: (name, relative)
CON_QTY = [("k1_am", True), ("k1w_am", True), ("a1_am", False), ("f_am", False), ("k2_am", False),
           ("k1_prod", True), ("k1_o2", True), ("k2_o2", False), ("k1_dry_t2", True), ("k1w_t2", True),
           ("k2_t2", False)]
CON_REQUESTED = [("k1_am", BANDS), ("k1w_am", ("sco2", "wco2")), ("a1_am", BANDS), ("f_am", ("o2a",)),
                 ("k2_am", ("sco2",)), ("k1_prod", BANDS), ("k1_o2", BANDS)]
SUBSETS = [("QF0 cld<1", lambda q, c: (q == 0) & (c < 1)), ("QF0 cld<3", lambda q, c: (q == 0) & (c < 3)),
           ("QF0 cld<10", lambda q, c: (q == 0) & (c < 10)), ("QF0+1 cld<3", lambda q, c: np.isfinite(q) & (c < 3)),
           ("QF0 all", lambda q, c: q == 0)]
AM_SET_FIELDS = ("k1_am", "k1w_am", "k2_am", "a1_am", "a2_am", "f_am", "rms_am", "k2diag_am")
LATBREAK_QTY = [("k1_am", "o2a"), ("k1_am", "sco2"), ("a1_am", "sco2"), ("f_am", "o2a")]


def fd_dir(d):
    p = Path(d)
    return p / "fitting_details" if (p / "fitting_details").is_dir() else p


def parse_name(path):
    m = re.match(r"fitting_details_(\d{4}-\d{2}-\d{2})_(.+)\.h5$", Path(path).name)
    return (m.group(1), m.group(2)) if m else (None, None)


# --------------------------------------------------------------------------- loading

def _read(f, key):
    if key not in f:
        return None
    a = f[key][...]
    if a.dtype.kind in "SU" or a.dtype.kind in "iu":
        return a
    a = a.astype(np.float64)
    a[a < -900] = np.nan
    return a


def load_orbit(dop_path, prod_path):
    """Per-sounding arrays only. Returns dict(meta, prof, pool, dop{(band,name)}, prod{(band,name)}, notes)."""
    notes, missing = [], []
    with h5py.File(dop_path, "r") as f:
        fp_id = f["fp_id"][...].astype(np.int64)
        meta = {k: _read(f, k) for k in META_KEYS}
        prof = {k: _read(f, k) for k in PROF_KEYS}
        pool = {k: _read(f, k) for k in POOL_KEYS}
        dop = {}
        for b in BANDS:
            for nm in PRODORDER:
                dop[(b, nm)] = _read(f, f"{b}_{nm}_fitting_nosg")
            for nm in EXTRA_NAMES:
                # fit_schema doppler_am_t2_v1 (2026-10-31) drops the _nosg suffix
                a = _read(f, f"{b}_{nm}")
                dop[(b, nm)] = a if a is not None else _read(f, f"{b}_{nm}_nosg")
        tau_attr = f.attrs.get("tau_convolution", "")
    alternates = {"land": "sfc_type", "xco2": "xco2_bc"}   # only needed when the primary is absent
    missing += [k for k, v in meta.items() if v is None and not (k in alternates and meta[alternates[k]] is not None)]
    missing += [k for k, v in prof.items() if v is None]
    missing += [k for k, v in pool.items() if v is None]
    missing += [f"{b}_{nm}[_nosg]" for (b, nm), v in dop.items() if v is None and nm in EXTRA_NAMES]
    prod = {}
    if prod_path is not None and Path(prod_path).exists():
        with h5py.File(prod_path, "r") as g:
            pid = g["fp_id"][...].astype(np.int64)
            for b in BANDS:
                for nm in PRODORDER:
                    a = _read(g, f"{b}_{nm}_fitting_nosg")
                    if a is None:
                        a = _read(g, f"{b}_{nm}_fitting")
                        if a is not None:
                            notes.append(f"production {b}_{nm}_fitting_nosg absent; used {b}_{nm}_fitting (SG-smoothed)")
                    prod[(b, nm)] = a
        if np.array_equal(pid, fp_id):
            notes.append("fp_id: production and variant files identical (same soundings, same order)")
        else:
            pos = pd.Series(np.arange(len(pid)), index=pid)
            idx = pos.reindex(fp_id).values
            ok = np.isfinite(idx)
            notes.append(f"fp_id MISMATCH: {ok.sum()} of {len(fp_id)} variant soundings found in production; "
                         "production aligned by fp_id, unmatched set to NaN")
            for k, a in prod.items():
                if a is not None:
                    out = np.full(len(fp_id), np.nan)
                    out[ok] = a[idx[ok].astype(int)]
                    prod[k] = out
    else:
        notes.append(f"production file not found ({prod_path}); production columns skipped")
        prod = {(b, nm): None for b in BANDS for nm in PRODORDER}
    if meta["xco2_bc"] is None and meta["xco2"] is not None:
        meta["xco2_bc"] = meta["xco2"]
        notes.append("xco2_bc absent: used `xco2`")
    return dict(fp_id=fp_id, meta=meta, prof=prof, pool=pool, dop=dop, prod=prod, notes=notes,
                missing=missing, tau_attr=str(tau_attr))


# --------------------------------------------------------------------------- per-orbit frame

def build_frame(d, args):
    """Derived per-sounding columns (float) + per-orbit context.  Returns (F, ctx)."""
    m, dop, prod = d["meta"], d["dop"], d["prod"]
    N = len(d["fp_id"])
    notes = list(d["notes"])
    cld = m["cld_dist_km"] if m["cld_dist_km"] is not None else np.full(N, np.nan)
    qf = m["xco2_qf"] if m["xco2_qf"] is not None else np.full(N, np.nan)
    lat = m["lat"]
    fp = m["fp_number"].astype(int)
    has_cld = bool(np.isfinite(cld).any())
    if not has_cld:
        notes.append("no cloud distances in this file: far-clear, contrasts and anomaly metrics skipped")
    if m["sfc_type"] is not None:
        land = (m["sfc_type"] == 1).astype(float)
        land[~np.isfinite(m["sfc_type"])] = np.nan
        land_src = "sfc_type (Lite; 1 = land)"
    elif m["land"] is not None:
        land = (m["land"] > 0.5).astype(float)
        land_src = "land (> 0.5)"
    else:
        land = np.zeros(N)
        land_src = "none (all treated as water)"
        notes.append("no sfc_type/land: all soundings treated as water")
    far = (cld > FAR_KM) & (qf == 0)
    F = {"fp": fp, "cld": cld, "qf": qf, "land": land, "lat": lat, "far": far.astype(float)}   # -> DataFrame at the end

    # AM fields: values with ok_am == 0 set to NaN
    for b in BANDS:
        ok = dop[(b, "ok_am")]
        if ok is not None:
            if dop[(b, "k1_am")] is not None:
                F[f"rawnan_k1_am_{b}"] = np.isnan(dop[(b, "k1_am")]).astype(float)
            for nm in AM_MASKED:
                if dop[(b, nm)] is not None:
                    dop[(b, nm)] = np.where(ok == 0, np.nan, dop[(b, nm)])

    def coef(name, b):
        if name == "k1_prod":
            return prod[(b, "k1")]
        return dop.get((b, name))

    # ---- item 4.1: k1 per source (+ ratios), fp-relative over far-clear
    k1 = {}
    for src, _ in SOURCES:
        for b in BANDS:
            k1[(src, b)] = prod[(b, "k1")] if src == "prod" else dop[(b, "k1")] if src == "dopprod" else dop[(b, SRC_FIELD[src])]
        for rk, num, den in RATIOS:
            k1[(src, rk)] = (k1[(src, num)] / k1[(src, den)]
                             if k1[(src, num)] is not None and k1[(src, den)] is not None else None)
    for (src, bk), v in k1.items():
        if v is None or not np.isfinite(v).any():
            continue
        F[f"k1_{src}_{bk}"] = v
        rel = np.full(N, np.nan)
        if far.any():
            rel[far] = fp_relative(v[far], fp[far])
        F[f"k1rel_{src}_{bk}"] = rel

    # ---- contrasts (latitude-local, per orbit)
    if has_cld:
        for q, rel in CON_QTY:
            for b in BANDS:
                v = coef(q, b)
                if v is None or np.isfinite(v).sum() < 50 or np.nanstd(v) == 0:
                    continue   # absent, too sparse, or pinned (e.g. k2_am = 0 for O2A/WCO2)
                F[f"con_{q}_{b}"] = lat_local_contrast(v, cld, qf, lat, fp, rel)
        if m["h2o_scale"] is not None:
            F["con_h2o_scale"] = lat_local_contrast(m["h2o_scale"], cld, qf, lat, fp, False)
        if m["tcwv"] is not None:
            F["con_tcwv"] = lat_local_contrast(m["tcwv"], cld, qf, lat, fp, True)

    # ---- diagnostics columns
    for b in BANDS:
        for nm in ("ok_am", "f_am", "k2diag_am", "a1_am", "k1_am", "k2_am"):
            v = dop[(b, nm)]
            if v is not None and np.isfinite(v).any():
                F[f"{nm}_{b}"] = v

    # ---- prior and pseudo-XCO2
    pr = d["prof"]
    if pr["co2prior_profile"] is not None and pr["p_profile"] is not None:
        prior = column_mean_prior(pr["co2prior_profile"], pr["p_profile"], pr["q_profile"])
        prior_src = "column mean of the 72-level CO2Prior profile (co2prior_profile, dry-air pressure-weighted)"
        if m["xco2_apriori"] is not None:
            dd = prior - m["xco2_apriori"]
            prior_src += f"; minus Lite xco2_apriori: median {nanmedian(dd):+.2f} ppm, rSD {robust_sd(dd):.2f}"
    elif m["xco2_apriori"] is not None:
        prior = m["xco2_apriori"]
        prior_src = "Lite xco2_apriori (per-sounding prior profile not stored)"
    else:
        prior, prior_src = None, "none"
        notes.append("no CO2 prior: pseudo-XCO2 and anomaly product skipped")
    xbc, xraw = m["xco2_bc"], m["xco2_raw"]
    for c, v in (("xbc", xbc), ("xraw", xraw)):
        if v is not None:
            F[c] = v
    if prior is not None:
        F["prior"] = prior
        ka = {b: dop[(b, "k1_am")] for b in BANDS}
        if ka["sco2"] is not None:
            F["E_S"] = prior * ka["sco2"]
        if ka["wco2"] is not None:
            F["E_W"] = prior * ka["wco2"]
        if ka["sco2"] is not None and ka["o2a"] is not None:
            F["E1"] = prior * ka["sco2"] / ka["o2a"]

    # ---- pipeline reference windows (6a and 6c)
    ctx = dict(N=N, notes=notes, land_src=land_src, prior_src=prior_src, has_cld=has_cld,
               has_t2=dop[("sco2", "k1_dry_t2")] is not None, has_am=dop[("sco2", "k1_am")] is not None,
               missing=d["missing"], tau_attr=d["tau_attr"])
    ctx["coefs"] = {}
    for b in BANDS:
        for nm in PRODORDER:
            if prod[(b, nm)] is not None:
                ctx["coefs"][f"prod {b} {nm}"] = prod[(b, nm)]
    for b in BANDS:
        for nm in AM_SET_FIELDS:
            v = dop[(b, nm)]
            if v is not None and np.isfinite(v).any() and np.nanstd(v) > 0:
                ctx["coefs"][f"am {b} {nm}"] = v
    if dop[("sco2", "k1_am")] is not None and dop[("o2a", "k1_am")] is not None:
        ctx["coefs"]["am sco2/o2a k1_am"] = dop[("sco2", "k1_am")] / dop[("o2a", "k1_am")]
    ctx["windows"], ctx["target_check"] = {}, []
    if has_cld:
        pool_bc = [v for v in d["pool"].values() if v is not None]
        if len(pool_bc) < len(POOL_KEYS):
            notes.append("some fit-time pool variables absent: bc reference pool rebuilt from those present")
        for kind, x, pv in (("bc", xbc, pool_bc), ("raw", xraw, [])):
            if x is None:
                continue
            for r in (5.0, 10.0, 15.0):
                a = anomaly_args(r)
                anom, means, nref, _ = window_stats(lat, cld, x, r, pv, ctx["coefs"],
                                                    lat_thres=a["lat_thres"], std_thres=a["std_thres"])
                ctx["windows"][(kind, r)] = (anom, means)
                key = {10.0: f"xco2_{kind}_anomaly"}.get(r, f"xco2_{kind}_anomaly_r{int(r):02d}")
                st = m.get(key)
                if st is not None:
                    both = np.isfinite(st) & np.isfinite(anom)
                    ctx["target_check"].append(dict(
                        target=key, n_stored=int(np.isfinite(st).sum()), n_recomputed=int(np.isfinite(anom).sum()),
                        n_both=int(both.sum()), max_abs_diff_ppm=float(np.max(np.abs(st[both] - anom[both]))) if both.any() else np.nan,
                        n_finite_mismatch=int((np.isfinite(st) != np.isfinite(anom)).sum())))
    # ---- 6c anomaly product
    if has_cld and prior is not None and ("bc", 10.0) in ctx["windows"] and dop[("o2a", "k1_am")] is not None:
        if args.anom_radius == "train":
            means = {k: np.where(land == 1, ctx["windows"][("bc", 15.0)][1][k], ctx["windows"][("bc", 5.0)][1][k])
                     for k in ctx["coefs"]}
            xa_bc = _train(m, "bc", land)
            xa_raw = _train(m, "raw", land)
        else:
            r = float(args.anom_radius)
            means = ctx["windows"][("bc", r)][1]
            sfx = "" if r == 10.0 else f"_r{int(r):02d}"
            xa_bc, xa_raw = m.get(f"xco2_bc_anomaly{sfx}"), m.get(f"xco2_raw_anomaly{sfx}")
        dk = {b: dop[(b, "k1_am")] - means[f"am {b} k1_am"] for b in BANDS if f"am {b} k1_am" in means}
        if "sco2" in dk:
            F["dXS"] = prior * dk["sco2"] / KERNEL["sco2"]
        if "wco2" in dk:
            F["dXW"] = prior * dk["wco2"] / KERNEL["wco2"]
        if "o2a" in dk:
            o2rel = dk["o2a"] / means["am o2a k1_am"]
            F["o2a_rel"] = o2rel
            F["flag"] = np.where(np.isfinite(o2rel), (np.abs(o2rel) > FLAG_PCT / 100).astype(float), np.nan)
        if xa_bc is not None:
            F["xa_bc"] = xa_bc
        if xa_raw is not None:
            F["xa_raw"] = xa_raw
        ctx["koz"] = None
        if m["lon"] is not None:
            at, dist = along_track_km(lat, m["lon"], *KOZIENICE)
            F["along_km"], F["dist_koz_km"] = at, dist
    return pd.DataFrame(F), ctx


def _train(m, kind, land):
    a, b = m.get(f"xco2_{kind}_anomaly_r05"), m.get(f"xco2_{kind}_anomaly_r15")
    if a is None or b is None:
        return None
    return np.where(land == 1, b, a)


# --------------------------------------------------------------------------- tables on a frame (per orbit or pooled)

def _col(F, c):
    return F[c].values.astype(float) if c in F else None


def farclear_table(F):
    far = F["far"].values == 1
    rows = []
    for src, lab in SOURCES:
        for bk in K1_KEYS:
            v, r = _col(F, f"k1_{src}_{bk}"), _col(F, f"k1rel_{src}_{bk}")
            if v is None:
                rows.append(dict(source=src, band=bk, n_far=0, median=np.nan, spread_pct=np.nan, note="missing"))
                continue
            rows.append(dict(source=src, band=bk, n_far=int(np.isfinite(v[far]).sum()), median=nanmedian(v[far]),
                             spread_pct=100 * robust_sd(r[far]), note=""))
    return rows


def contrast_table(F, qfsel="QF0"):
    if "con_k1_o2_sco2" not in F and "con_k1_am_sco2" not in F and not any(c.startswith("con_") for c in F):
        return []
    cld, qf = F["cld"].values, F["qf"].values
    far = F["far"].values == 1
    sel = (qf == 0) if qfsel == "QF0" else np.ones(len(F), bool)
    rows = []
    for q, rel in CON_QTY + [("h2o_scale", False), ("tcwv", True)]:
        for b in (BANDS if q not in ("h2o_scale", "tcwv") else ("met",)):
            c = _col(F, f"con_{q}_{b}" if b != "met" else f"con_{q}")
            if c is None:
                continue
            rows.append(dict(quantity=q, band=b, kind="rel" if rel else "abs", qf_sel=qfsel,
                             **bin_stats(c, cld, far, sel)))
    return rows


def latbreak_table(F, min_n):
    cld, qf, lat = F["cld"].values, F["qf"].values, F["lat"].values
    far = F["far"].values == 1
    latb = np.floor(lat / 5.0) * 5.0
    cons = {(q, b): _col(F, f"con_{q}_{b}") for q, b in LATBREAK_QTY}
    fsd = {k: robust_sd(c[far]) if c is not None else np.nan for k, c in cons.items()}
    rows = []
    for (lo, hi), lab in zip(BINS[:2], BIN_LABELS[:2]):
        inbin = (cld >= lo) & (cld < hi)
        for lb in np.unique(latb[inbin & np.isfinite(latb)]):
            mb = inbin & (latb == lb)
            nall = int(mb.sum())
            rec = dict(bin=lab, lat_band=f"{lb:+.0f}..{lb + 5:+.0f}", n_all=nall,
                       qf1_frac=float((qf[mb] == 1).sum() / nall) if nall else np.nan)
            for (q, b), c in cons.items():
                key = f"{b}_{q}"
                if c is None:
                    rec[f"n_{key}"], rec[f"con_{key}"], rec[f"snr_{key}"] = 0, np.nan, np.nan
                    continue
                s = mb & (qf == 0) & np.isfinite(c)
                med = float(np.median(c[s])) if s.sum() >= min_n else np.nan
                rec[f"n_{key}"], rec[f"con_{key}"] = int(s.sum()), med
                rec[f"snr_{key}"] = med / fsd[(q, b)] if np.isfinite(fsd[(q, b)]) and fsd[(q, b)] > 0 else np.nan
            po, ps = rec.get("con_o2a_k1_am", np.nan), rec.get("con_sco2_k1_am", np.nan)
            so, ss = rec.get("snr_o2a_k1_am", np.nan), rec.get("snr_sco2_k1_am", np.nan)
            rec["path_type_signature"] = bool(po > 0 and ps > 0 and so >= 2 and ss >= 2)
            rows.append(rec)
    return rows


def diag_table(F):
    rows = []
    land = F["land"].values
    far = F["far"].values == 1
    for b in BANDS:
        ok = _col(F, f"ok_am_{b}")
        if ok is None:
            continue
        for surf, ms in (("water", land == 0), ("land", land == 1), ("all", np.ones(len(F), bool))):
            for sub, mm in (("all soundings", ms), ("far-clear QF0", ms & far)):
                n = int(mm.sum())
                if n == 0:
                    continue
                rec = dict(band=b, surface=surf, subset=sub, n=n,
                           frac_ok0=float((ok[mm] == 0).mean()), frac_ok_nan=float(np.isnan(ok[mm]).mean()))
                rn = _col(F, f"rawnan_k1_am_{b}")
                rec["frac_k1_am_nan"] = float(rn[mm].mean()) if rn is not None else float(np.isnan(_col(F, f"k1_am_{b}")[mm]).mean()) if f"k1_am_{b}" in F else np.nan
                kd = _col(F, f"k2diag_am_{b}")
                rec["frac_k2diag_neg"] = float((kd[mm] < 0).sum() / max(np.isfinite(kd[mm]).sum(), 1)) if kd is not None else np.nan
                f = _col(F, f"f_am_{b}")
                if f is not None:
                    fm = f[mm]
                    rec.update(frac_f_at_bound=float((fm >= 0.2 - 1e-9).sum() / max(np.isfinite(fm).sum(), 1)),
                               f_median=nanmedian(fm), f_p05=nanpct(fm, 5), f_p95=nanpct(fm, 95), f_rsd=robust_sd(fm))
                k2 = _col(F, f"k2_am_{b}")
                if k2 is not None:
                    rec.update(k2_median=nanmedian(k2[mm]), frac_k2_pos=float((k2[mm] > 0).sum() / max(np.isfinite(k2[mm]).sum(), 1)))
                a1 = _col(F, f"a1_am_{b}")
                if a1 is not None:
                    rec.update(a1_median=nanmedian(a1[mm]), a1_rsd=robust_sd(a1[mm]))
                rows.append(rec)
    return rows


def _decide(R, snr_path, w_con, w_snr):
    if not np.isfinite(R):
        return "undefined"
    if R >= 0.5 and snr_path >= 2:
        return "path dominant"
    if R < 0.25 and np.isfinite(w_con) and w_con > 0 and np.isfinite(w_snr) and w_snr >= 2:
        return "H2O dominant"
    return "mixed"


def bars_table(F, version, qfsel):
    """Pre-registered bars A-D (PLAN.md 'H2O vs cloud-path test'). version 'T2' or 'AM'."""
    path, wq, k2q = ("k1_dry_t2", "k1w_t2", "k2_t2") if version == "T2" else ("k1_am", "k1w_am", "k2_am")
    cld, qf = F["cld"].values, F["qf"].values
    far = F["far"].values == 1
    sel = (qf == 0) if qfsel == "QF0" else np.ones(len(F), bool)
    rows = []

    def st(q, b):
        c = _col(F, f"con_{q}_{b}")
        if c is None:
            return None
        return bin_stats(c, cld, far, sel)

    base = dict(version=version, qf_sel=qfsel)
    for bar, s2q, twoq in (("A", "k1_o2", path), ("B", "k2_o2", k2q)):
        for b in ("sco2", "wco2"):
            s_s2, s_t, s_w = st(s2q, b), st(twoq, b), st(wq, b)
            if s_s2 is None or s_t is None:
                rows.append(dict(base, bar=bar, band=b, decision="skipped (fields missing or coefficient pinned/constant)"))
                continue
            den = s_s2["con_0-1"]
            R = s_t["con_0-1"] / den if np.isfinite(den) and den != 0 else np.nan
            wc, ws = (s_w["con_0-1"], s_w["snr_0-1"]) if s_w else (np.nan, np.nan)
            rows.append(dict(base, bar=bar, band=b, n01=s_t["n_0-1"], s2_con01=s_s2["con_0-1"], s2_snr01=s_s2["snr_0-1"],
                             path_con01=s_t["con_0-1"], path_snr01=s_t["snr_0-1"], R=R, w_con01=wc, w_snr01=ws,
                             decision=_decide(R, s_t["snr_0-1"], wc, ws) + ("" if b == "sco2" else " (reported, not decision)")))
    s_s2, s_t, s_w = st("k1_o2", "o2a"), st(path, "o2a"), st(wq, "o2a")
    if s_s2 is None or s_t is None:
        rows.append(dict(base, bar="C", band="o2a", decision="skipped (fields missing)"))
    else:
        diff = abs(s_t["con_0-1"] - s_s2["con_0-1"])
        thr = 0.5 * s_s2["far_rsd"]
        if s_w is None:
            w_ok, wnote = True, "k1_w not fitted for O2A (criterion 2 n/a)"
            wc = ws = w_ill = np.nan
        else:
            w_ill = s_w["far_rsd"] > 10 * s_t["far_rsd"]
            wc, ws = s_w["con_0-1"], s_w["snr_0-1"]
            w_ok = bool(w_ill or (np.isfinite(ws) and abs(ws) < 2))
            wnote = f"k1_w ill-determined={bool(w_ill)}"
        passed = bool(np.isfinite(diff) and diff <= thr and w_ok)
        rows.append(dict(base, bar="C", band="o2a", n01=s_t["n_0-1"], s2_con01=s_s2["con_0-1"], path_con01=s_t["con_0-1"],
                         abs_diff=diff, threshold=thr, w_con01=wc, w_snr01=ws,
                         decision=("PASS" if passed else "FAIL") + f" ({wnote})"))
    h1, h2 = _col(F, "con_h2o_scale"), _col(F, "con_tcwv")
    if h1 is None or h2 is None:
        rows.append(dict(base, bar="D", band="-", decision="skipped (h2o_scale/tcwv absent)"))
        return rows
    lncld = np.log(np.clip(cld, 0.1, None))
    for q, b in (("k1_o2", "sco2"), ("k1_o2", "wco2"), ("k2_o2", "sco2"), ("k2_o2", "wco2"), (path, "sco2"), (path, "wco2")):
        y = _col(F, f"con_{q}_{b}")
        if y is None:
            continue
        mm = sel & (cld < 3) & np.isfinite(y) & np.isfinite(h1) & np.isfinite(h2)
        if mm.sum() < 30:
            rows.append(dict(base, bar="D", band=b, target=q, n=int(mm.sum()), decision="too few soundings"))
            continue
        yy, hh, cc = y[mm], [h1[mm], h2[mm]], [lncld[mm]]
        r2 = ols_r2(yy, hh)[0]
        dec = ("H2O implicated" if r2 >= 0.5 else "H2O not implicated") + ("" if (q, b) == ("k1_o2", "sco2") else " (reported)")
        rows.append(dict(base, bar="D", band=b, target=q, n=int(mm.sum()), R2_h2o=r2,
                         pR2_h2o_given_lncld=partial_r2(yy, cc, hh), pR2_lncld_given_h2o=partial_r2(yy, hh, cc), decision=dec))
    return rows


def pseudo_table(F):
    rows = []
    if "prior" not in F or "xbc" not in F:
        return rows
    far = F["far"].values == 1
    land, fp = F["land"].values, F["fp"].values
    pri, bc = F["prior"].values, F["xbc"].values
    raw = _col(F, "xraw")
    ests = [(k, _col(F, k)) for k in ("E_S", "E_W", "E1") if k in F] + [("prior", pri)]
    if raw is not None:
        ests.append(("L2 raw", raw))
    for surf, ms in (("water", land == 0), ("land", land == 1), ("all", np.ones(len(F), bool))):
        mm = far & ms
        for name, e in ests:
            ok = mm & np.isfinite(e) & np.isfinite(bc)
            n = int(ok.sum())
            if n < 10:
                rows.append(dict(estimator=name, surface=surf, n_far=n))
                continue
            d_bc = e - bc
            d_pr = e - pri
            rec = dict(estimator=name, surface=surf, n_far=n, med_E_minus_bc=nanmedian(d_bc[ok]),
                       med_E_minus_raw=nanmedian((e - raw)[ok]) if raw is not None else np.nan,
                       rsd_E_minus_bc=robust_sd(d_bc[ok]), rsd_E_minus_raw=robust_sd((e - raw)[ok]) if raw is not None else np.nan,
                       med_E_minus_prior=nanmedian(d_pr[ok]))
            fpm = pd.Series(d_pr[ok]).groupby(fp[ok]).median()
            for k, v in fpm.items():
                rec[f"fp{int(k)}_med_E_minus_prior"] = float(v)
            rec["fp_spread_E_minus_prior"] = float(fpm.max() - fpm.min())
            rec["rsd_fpdm_E_minus_prior"] = robust_sd(fp_demean_median(d_pr[ok], fp[ok]))
            rec["rsd_fpdm_E_minus_bc"] = robust_sd(fp_demean_median(d_bc[ok], fp[ok]))
            rows.append(rec)
    return rows


def flag_table(F):
    rows = []
    if "dXS" not in F and "flag" not in F:
        return rows
    far = F["far"].values == 1
    land, cld, qf = F["land"].values, F["cld"].values, F["qf"].values
    dxs, dxw, flag = _col(F, "dXS"), _col(F, "dXW"), _col(F, "flag")
    xa, xr = _col(F, "xa_bc"), _col(F, "xa_raw")
    o2r = _col(F, "o2a_rel")
    nan = np.full(len(F), np.nan)
    o2r = o2r if o2r is not None else nan
    dxs = dxs if dxs is not None else nan
    dxw = dxw if dxw is not None else nan
    flag = flag if flag is not None else nan
    xa = xa if xa is not None else nan
    xr = xr if xr is not None else nan
    unf = flag == 0
    retrieved = np.isfinite(F["xbc"].values) if "xbc" in F else np.ones(len(F), bool)
    for surf, ms in (("all", np.ones(len(F), bool)), ("water", land == 0), ("land", land == 1)):
        q0 = ms & (qf == 0)
        rec = dict(section="far-clear", surface=surf, n=int((ms & far).sum()),
                   rsd_dXS=robust_sd(dxs[ms & far]), rsd_dXW=robust_sd(dxw[ms & far]),
                   r_dXS_xa_far=pearson(dxs[ms & far], xa[ms & far])[0], r_dXW_xa_far=pearson(dxw[ms & far], xa[ms & far])[0],
                   r_dXS_xa_qf0=pearson(dxs[q0], xa[q0])[0], r_dXW_xa_qf0=pearson(dxw[q0], xa[q0])[0],
                   r_dXS_xraw_qf0=pearson(dxs[q0], xr[q0])[0], r_dXW_xraw_qf0=pearson(dxw[q0], xr[q0])[0],
                   rsd_o2a_rel_pct=100 * robust_sd(o2r[ms & far]), p95_abs_o2a_rel_pct=100 * nanpct(np.abs(o2r[ms & far]), 95),
                   p99_abs_o2a_rel_pct=100 * nanpct(np.abs(o2r[ms & far]), 99))
        rows.append(rec)
        for (lo, hi), lab in list(zip(BINS[:3], BIN_LABELS[:3])) + [((FAR_KM, np.inf), ">20")]:
            mb = q0 & (cld >= lo) & (cld < hi)
            rows.append(dict(section="bin QF0", surface=surf, bin=lab, n=int(mb.sum()),
                             med_dXS=nanmedian(dxs[mb], 5), med_dXW=nanmedian(dxw[mb], 5),
                             med_xa_bc=nanmedian(xa[mb], 5), med_xa_raw=nanmedian(xr[mb], 5),
                             med_dXS_unflagged=nanmedian(dxs[mb & unf], 5), med_dXW_unflagged=nanmedian(dxw[mb & unf], 5),
                             flag_rate=float(np.nanmean(flag[mb])) if np.isfinite(flag[mb]).any() else np.nan))
        groups = [("QF0 far-clear", ms & far), ("QF0 all", q0), ("QF1 all", ms & (qf == 1))]
        if (~retrieved).any():
            groups.append(("non-retrieved (no xco2)", ms & ~retrieved))
        for g, mg in groups:
            fin = mg & np.isfinite(flag)
            rows.append(dict(section="flag rate", surface=surf, group=g, n=int(fin.sum()),
                             flag_rate=float(flag[fin].mean()) if fin.any() else np.nan))
    return rows


def kozienice_table(F, date, orbit):
    if not (date == "2021-09-06" and orbit.startswith("38200a")) or "along_km" not in F:
        return []
    at, dist = F["along_km"].values, F["dist_koz_km"].values
    unf = F["flag"].values == 0 if "flag" in F else np.ones(len(F), bool)
    rows = []
    for lo in np.arange(-60, 60, 5):
        mb = (dist <= 60) & (at >= lo) & (at < lo + 5)
        rows.append(dict(section="Kozienice along-track", bin=f"{lo:+.0f}..{lo + 5:+.0f}", n=int(mb.sum()),
                         n_unflagged=int((mb & unf).sum()),
                         med_dXS_unflagged=nanmedian(_col(F, "dXS")[mb & unf]) if "dXS" in F else np.nan,
                         med_dXW_unflagged=nanmedian(_col(F, "dXW")[mb & unf]) if "dXW" in F else np.nan,
                         med_xa_bc=nanmedian(_col(F, "xa_bc")[mb]) if "xa_bc" in F else np.nan,
                         med_xa_raw=nanmedian(_col(F, "xa_raw")[mb]) if "xa_raw" in F else np.nan))
    return rows


# --------------------------------------------------------------------------- item 6a (per orbit, raw arrays)

def anomaly_corr_rows(d, F, ctx, args):
    if not ctx["has_cld"] or not ctx["windows"]:
        return []
    m = d["meta"]
    land = F["land"].values
    cld, qf = F["cld"].values, F["qf"].values
    coefs = ctx["coefs"]
    targets = {}
    for kind in ("bc", "raw"):
        for r, sfx in ((10.0, ""), (5.0, "_r05"), (15.0, "_r15")):
            v = m.get(f"xco2_{kind}_anomaly{sfx}")
            if v is not None and (kind, r) in ctx["windows"]:
                targets[f"{kind}_r{int(r):02d}"] = (v, ctx["windows"][(kind, r)][1])
        tv = _train(m, kind, land)
        if tv is not None and (kind, 5.0) in ctx["windows"]:
            mt = {k: np.where(land == 1, ctx["windows"][(kind, 15.0)][1][k], ctx["windows"][(kind, 5.0)][1][k]) for k in coefs}
            targets[f"{kind}_train(r05 ocean/r15 land)"] = (tv, mt)
    corr_t = set(targets) if args.all_corr_targets else {t for t in targets if t.endswith("r10") or "train" in t}
    prod_cols = [c for c in coefs if c.startswith("prod ")]
    am_cols = [c for c in coefs if c.startswith("am ")]
    sets = {"production set": prod_cols, "agreed set": am_cols,
            "production k1 x3": [c for c in prod_cols if c.endswith(" k1")],
            "agreed k1 x3": [c for c in am_cols if c.endswith(" k1_am") and "/" not in c],
            "both sets": prod_cols + am_cols}
    rows = []
    for tname, (y_all, means) in targets.items():
        for sname, fsub in SUBSETS:
            msub = fsub(qf, cld) & np.isfinite(y_all)
            for form in ("stored", "local anomaly"):
                src = coefs if form == "stored" else {c: coefs[c] - means[c] for c in coefs}
                if tname in corr_t:
                    for c, v in src.items():
                        r, n = pearson(v[msub], y_all[msub])
                        rows.append(dict(kind="corr", target=tname, subset=sname, form=form, coef=c, n=n,
                                         pearson=r, spearman=spearman(v[msub], y_all[msub]) if n >= 30 else np.nan))
                for setname, cols in sets.items():
                    if not cols:
                        rows.append(dict(kind="ridge", target=tname, subset=sname, form=form, coef=setname, n=0, r2=np.nan,
                                         note="set empty (fields missing)"))
                        continue
                    X = np.column_stack([src[c] for c in cols])[msub]
                    yy = y_all[msub]
                    keep = [i for i in range(X.shape[1]) if np.isfinite(X[:, i]).mean() >= 0.9 and np.nanstd(X[:, i]) > 0]
                    dropped = [cols[i] for i in range(len(cols)) if i not in keep]
                    X = X[:, keep]
                    cc = np.all(np.isfinite(X), axis=1) & np.isfinite(yy)
                    n = int(cc.sum())
                    r2 = ridge_heldout(X[cc], yy[cc]) if n > 40 and X.shape[1] else np.nan
                    rows.append(dict(kind="ridge", target=tname, subset=sname, form=form, coef=setname, n=n, r2=r2,
                                     n_pred=len(keep), note=("dropped: " + ", ".join(dropped)) if dropped else ""))
    return rows


# --------------------------------------------------------------------------- markdown helpers

def fmtv(v, f="{:.3g}"):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return "—" if not np.isfinite(v) else f.format(v)
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def md_table(df, cols, fmts=None):
    fmts = fmts or {}
    cols = [c for c in cols if c in df.columns]
    if df.empty or not cols:
        return "_(no rows)_\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmtv(r[c], fmts.get(c, "{:.3g}")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def farclear_md(df):
    if df.empty:
        return "_(no rows)_\n"
    lines = ["| band | " + " | ".join(f"{s} median (spread %)" for s, _ in SOURCES) + " |", "|---|" + "---|" * len(SOURCES)]
    for bk in K1_KEYS:
        cells = []
        for s, _ in SOURCES:
            r = df[(df.source == s) & (df.band == bk)]
            if r.empty or not np.isfinite(r.iloc[0]["median"]):
                cells.append("—")
            else:
                r = r.iloc[0]
                cells.append(f"{r['median']:.4f} ({r['spread_pct']:.2f}, n {int(r['n_far'])})")
        lines.append(f"| {bk.replace('_', '/')} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def contrast_md(df, requested=True):
    if df.empty:
        return "_(no rows)_\n"
    lines = ["| quantity | band | kind | far rSD | 0–1 km con (SNR, n) | 1–3 km | 3–10 km | 10–20 km |", "|---|---|---|---|---|---|---|---|"]
    want = [(q, b) for q, bs in CON_REQUESTED for b in bs] if requested else list(zip(df.quantity, df.band))
    for q, b in want:
        r = df[(df.quantity == q) & (df.band == b)]
        if r.empty:
            lines.append(f"| {q} | {b} | — | missing | | | | |")
            continue
        r = r.iloc[0]
        f = "{:+.3%}" if r["kind"] == "rel" else "{:+.4f}"
        fs = "{:.3%}" if r["kind"] == "rel" else "{:.4f}"
        cells = [f"{fmtv(r[f'con_{lab}'], f)} ({fmtv(r[f'snr_{lab}'], '{:+.2f}')}, {int(r[f'n_{lab}'])})" for lab in BIN_LABELS[:4]]
        lines.append(f"| {q} | {b} | {r['kind']} | {fmtv(r['far_rsd'], fs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"]


def figures(fc, ct, fl, F_pool_cols, out):
    made = []
    # 1 far-clear spread per source
    if not fc.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(len(K1_KEYS))
        w = 0.16
        for i, (s, lab) in enumerate(SOURCES):
            sub = fc[fc.source == s].set_index("band").reindex(K1_KEYS)
            axes[0].bar(x + (i - 2) * w, sub["spread_pct"].values, w, color=COLORS[i], label=lab)
            axes[1].plot(x, sub["median"].values, "o", ms=8, color=COLORS[i], label=lab)
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels([k.replace("_", "/").upper() for k in K1_KEYS])
            ax.grid(axis="y", lw=0.4, alpha=0.5)
        axes[0].set_ylabel("far-clear spread (%)")
        axes[1].set_ylabel("far-clear median k1")
        axes[0].legend(fontsize=7)
        axes[0].set_title("far-clear k1 spread (per-footprint means removed)", fontsize=9)
        axes[1].set_title("far-clear k1 median", fontsize=9)
        fig.tight_layout()
        p = out / "fig_farclear_k1.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    # 2 near-cloud contrast profiles
    if not ct.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, b in zip(axes, BANDS):
            for i, (q, lab) in enumerate((("k1_am", "k1_am"), ("k1_prod", "production k1"), ("k1_o2", "k1_o2"), ("k1_dry_t2", "k1_dry_t2"))):
                r = ct[(ct.quantity == q) & (ct.band == b) & (ct.qf_sel == "QF0")]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = [100 * r[f"con_{lab_}"] for lab_ in BIN_LABELS[:4]]
                ax.plot(range(4), y, "o-", lw=2, ms=8, color=COLORS[i], label=lab)
            ax.axhline(0, color="0.4", lw=0.6)
            ax.set_xticks(range(4))
            ax.set_xticklabels([l + " km" for l in BIN_LABELS[:4]])
            ax.set_title(f"{BAND_UP[b]} k1 contrast (QF0)", fontsize=9)
            ax.grid(axis="y", lw=0.4, alpha=0.5)
        axes[0].set_ylabel("median latitude-local contrast (%)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        p = out / "fig_nearcloud_contrast.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    # 3 pseudo-XCO2 and anomaly product
    if F_pool_cols is not None and "E_S" in F_pool_cols:
        Fp = F_pool_cols
        far = Fp["far"].values == 1
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        allv = np.concatenate([(Fp[k].values - Fp["xbc"].values)[far] for k in ("E_S", "E_W", "E1") if k in Fp])
        allv = allv[np.isfinite(allv)]
        lo_, hi_ = (np.floor(np.percentile(allv, 0.5)) - 1, np.ceil(np.percentile(allv, 99.5)) + 1) if allv.size else (-15, 15)
        for i, (k, lab) in enumerate((("E_S", "E_S = prior·k1_am SCO2"), ("E_W", "E_W = prior·k1_am WCO2"), ("E1", "E1 = prior·SCO2/O2A"))):
            if k in Fp:
                d = (Fp[k].values - Fp["xbc"].values)[far & (Fp["land"].values == 0)]
                d = d[np.isfinite(d)]
                if d.size:
                    axes[0].hist(d, bins=np.linspace(lo_, hi_, 121), histtype="step", lw=2, color=COLORS[i],
                                 label=f"{lab}: med {np.median(d):+.2f}, rSD {robust_sd(d):.2f}")
        axes[0].set_xlabel("E − xco2_bc (ppm), far-clear QF0 water")
        axes[0].set_ylabel("soundings")
        axes[0].legend(fontsize=7)
        if not fl.empty:
            b = fl[(fl.section == "bin QF0") & (fl.surface == "all")]
            for i, (c, lab) in enumerate((("med_dXS", "ΔX_S (k1_am SCO2 / 1.60)"), ("med_dXW", "ΔX_W (k1_am WCO2 / 1.01)"),
                                          ("med_xa_bc", "xco2_bc anomaly"), ("med_xa_raw", "xco2_raw anomaly"))):
                if c in b:
                    axes[1].plot(range(len(b)), b[c].values, "o-", lw=2, ms=8, color=COLORS[i], label=lab)
            axes[1].set_xticks(range(len(b)))
            axes[1].set_xticklabels([str(v) + " km" for v in b["bin"].values])
            axes[1].axhline(0, color="0.4", lw=0.6)
            axes[1].set_ylabel("median anomaly (ppm), QF0")
            axes[1].legend(fontsize=7)
        for ax in axes:
            ax.grid(lw=0.4, alpha=0.5)
        fig.tight_layout()
        p = out / "fig_pseudo_xco2.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    return made


# --------------------------------------------------------------------------- main

class CSVSink:
    """Per-orbit parts (columns may differ between orbits, e.g. with/without AM fields) under
    <out>/parts/<name>/; read() concatenates them (union of columns) and writes agreed_eval_<name>.csv."""

    def __init__(self, out, name):
        self.dir = out / "parts" / name
        self.path = out / f"agreed_eval_{name}.csv"
        self.dir.mkdir(parents=True, exist_ok=True)
        for p in self.dir.glob("*.csv"):
            p.unlink()
        if self.path.exists():
            self.path.unlink()
        self._df = None

    def write(self, rows, **keys):
        if not rows:
            return
        df = pd.DataFrame(rows)
        for i, (k, v) in enumerate(keys.items()):
            df.insert(i, k, v)
        df.to_csv(self.dir / f"{keys.get('date', 'x')}_{keys.get('orbit', 'x')}.csv", index=False)
        self._df = None

    def read(self):
        if self._df is None:
            parts = [pd.read_csv(p) for p in sorted(self.dir.glob("*.csv"))]
            self._df = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
            if len(self._df):
                self._df.to_csv(self.path, index=False)
        return self._df


def find_orbits(args):
    dop_fd, prod_fd = fd_dir(args.dop_dir), fd_dir(args.prod_dir)
    fx = {}
    if args.fixture:
        dt, ob = parse_name(args.fixture)
        fx[(dt, ob)] = Path(args.fixture)
    jobs = []
    for date in args.dates:
        found = {parse_name(p)[1]: p for p in sorted(dop_fd.glob(f"fitting_details_{date}_*.h5"))}
        for (dt, ob), p in fx.items():
            if dt == date:
                found[ob] = p
        for ob in sorted(found):
            if args.orbits and ob not in args.orbits:
                continue
            jobs.append((date, ob, found[ob], prod_fd / f"fitting_details_{date}_{ob}.h5"))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--prod-dir", required=True)
    ap.add_argument("--dop-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-orbit", default="29252a_GL")
    ap.add_argument("--fixture", default=None, help="fitting_details file used as the dop file for its date/orbit")
    ap.add_argument("--orbits", nargs="*", default=None, help="restrict to these orbit ids (e.g. 31309a_GL)")
    ap.add_argument("--anom-radius", default="10", choices=["5", "10", "15", "train"],
                    help="reference window of the 6c anomaly product (train = r05 ocean / r15 land)")
    ap.add_argument("--all-corr-targets", action="store_true", help="per-coefficient correlations for r05/r15 too")
    ap.add_argument("--lat-min-n", type=int, default=10, help="min QF0 soundings for a 5-deg bin median (4.3)")
    ap.add_argument("--md-per-orbit-max", type=int, default=4, help="per-orbit md tables only up to this many orbits")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if any(str(out).startswith(f) for f in FORBIDDEN_OUT):
        sys.exit(f"refusing to write under {out}: outputs must not go to the results trees")
    (out / "per_sounding").mkdir(parents=True, exist_ok=True)
    jobs = find_orbits(args)
    if not jobs:
        sys.exit("no orbits found")
    sinks = {k: CSVSink(out, k) for k in
             ("per_orbit", "farclear", "contrasts", "lat_breakdown", "diagnostics", "bars", "anomaly_corr",
              "pseudo_xco2", "flag", "target_check")}
    orbit_info = []
    t_all = time.time()
    for date, orbit, dop_path, prod_path in jobs:
        t0 = time.time()
        d = load_orbit(dop_path, prod_path)
        F, ctx = build_frame(d, args)
        key = dict(date=date, orbit=orbit)
        fc = farclear_table(F)
        ct = contrast_table(F, "QF0") + contrast_table(F, "all QF")
        lb = latbreak_table(F, args.lat_min_n) if ctx["has_cld"] else []
        dg = diag_table(F)
        bs = []
        if ctx["has_cld"]:
            for ver in ("T2", "AM"):
                for qs in ("QF0", "all QF"):
                    bs += bars_table(F, ver, qs)
        ac = anomaly_corr_rows(d, F, ctx, args)
        ps = pseudo_table(F)
        fl = flag_table(F) + kozienice_table(F, date, orbit)
        for name, rows in (("farclear", fc), ("contrasts", ct), ("lat_breakdown", lb), ("diagnostics", dg), ("bars", bs),
                           ("anomaly_corr", ac), ("pseudo_xco2", ps), ("flag", fl), ("target_check", ctx["target_check"])):
            sinks[name].write(rows, **key)
        F = pd.concat([pd.DataFrame({"date": date, "orbit": orbit}, index=F.index), F], axis=1)
        F.to_parquet(out / "per_sounding" / f"{date}_{orbit}.parquet", index=False)
        dt = time.time() - t0
        orbit_info.append(dict(date=date, orbit=orbit, dop=str(dop_path), prod=str(prod_path), N=ctx["N"], notes=ctx["notes"],
                               land_src=ctx["land_src"], prior_src=ctx["prior_src"], missing=ctx["missing"],
                               has_t2=ctx["has_t2"], has_am=ctx["has_am"], tau_attr=ctx["tau_attr"], wall_s=dt))
        sinks["per_orbit"].write([headline(F, ctx, fc, ct, dg, bs, ac, ps, fl, dt)], **key)
        print(f"[{date} {orbit}] N={ctx['N']} t2={ctx['has_t2']} am={ctx['has_am']} wall {dt:.1f} s", flush=True)
        del d, F
    pooled(args, out, sinks, orbit_info)
    for sk in sinks.values():
        sk.read()                      # materialise every agreed_eval_<name>.csv
    print(f"total wall {time.time() - t_all:.1f} s; outputs in {out}")


def headline(F, ctx, fc, ct, dg, bs, ac, ps, fl, dt):
    h = dict(N=ctx["N"], n_far=int((F["far"] == 1).sum()), land_frac=float(np.nanmean(F["land"])), has_t2=ctx["has_t2"],
             has_am=ctx["has_am"], wall_s=round(dt, 1))
    fcd = pd.DataFrame(fc)
    for s in ("prod", "am"):
        for bk in K1_KEYS:
            r = fcd[(fcd.source == s) & (fcd.band == bk)]
            if len(r):
                h[f"far_med_k1_{s}_{bk}"] = r.iloc[0]["median"]
                h[f"far_spread_k1_{s}_{bk}"] = r.iloc[0]["spread_pct"]
    ctd = pd.DataFrame(ct)
    for q, b in (("k1_am", "sco2"), ("k1_am", "o2a"), ("k1_prod", "sco2")):
        r = ctd[(ctd.get("quantity") == q) & (ctd.get("band") == b) & (ctd.get("qf_sel") == "QF0")] if len(ctd) else ctd
        if len(r):
            h[f"con01_{q}_{b}"], h[f"snr01_{q}_{b}"] = r.iloc[0]["con_0-1"], r.iloc[0]["snr_0-1"]
    dgd = pd.DataFrame(dg)
    if len(dgd):
        r = dgd[(dgd.band == "o2a") & (dgd.surface == "water") & (dgd.subset == "all soundings")]
        if len(r):
            h["f_am_o2a_water_median"], h["f_am_o2a_water_rsd"] = r.iloc[0].get("f_median"), r.iloc[0].get("f_rsd")
        for b in BANDS:
            r = dgd[(dgd.band == b) & (dgd.surface == "all") & (dgd.subset == "all soundings")]
            if len(r):
                h[f"frac_ok0_{b}"] = r.iloc[0]["frac_ok0"]
    for r in bs:
        if r.get("qf_sel") == "QF0" and r.get("band") in ("sco2", "o2a") and r["bar"] in "ABC":
            h[f"bar{r['bar']}_{r['version']}"] = r["decision"]
    psd = pd.DataFrame(ps)
    if len(psd) and "med_E_minus_bc" in psd:
        for e in ("E_S", "E_W", "E1"):
            r = psd[(psd.estimator == e) & (psd.surface == "water")]
            if len(r):
                h[f"{e}_minus_bc_water_med"], h[f"{e}_minus_bc_water_rsd"] = r.iloc[0]["med_E_minus_bc"], r.iloc[0]["rsd_E_minus_bc"]
    for r in fl:
        if r.get("section") == "flag rate" and r.get("surface") == "all" and r.get("group") == "QF0 far-clear":
            h["flag_rate_far_qf0"] = r["flag_rate"]
    for r in ac:
        if r["kind"] == "ridge" and r["target"].startswith("bc_train") and r["subset"] == "QF0 cld<3" and r["coef"] in ("production set", "agreed set"):
            h[f"R2_bc_train_qf0_cld3_{'anom' if r['form'] == 'local anomaly' else 'stored'}_{r['coef'].split()[0]}"] = r["r2"]
    for c in ctx["target_check"]:
        h[f"chk_{c['target']}_maxdiff"] = c["max_abs_diff_ppm"]
    return h


def pooled(args, out, sinks, orbit_info):
    files = sorted((out / "per_sounding").glob("*.parquet"))
    keyset = {(o["date"], o["orbit"]) for o in orbit_info}
    files = [p for p in files if tuple(p.stem.split("_", 1)) in keyset]
    parts = []
    for p in files:
        x = pd.read_parquet(p)
        parts.append(x.astype({c: np.float32 for c in x.columns if x[c].dtype == np.float64}))
    Fp = pd.concat(parts, ignore_index=True)
    del parts
    n_orb = len(orbit_info)
    dates = sorted({o["date"] for o in orbit_info})
    per_orbit_md = n_orb <= args.md_per_orbit_max
    L = [f"# Agreed-model evaluation (items 4 and 6) — {n_orb} orbit(s), {len(dates)} date(s)", "",
         f"Dates: {', '.join(dates)}. Variant tree: `{args.dop_dir}`; production tree: `{args.prod_dir}`"
         + (f"; fixture: `{args.fixture}`" if args.fixture else "") + ".", "",
         "Definitions: far-clear = cld_dist > 20 km & QF0; rSD = 1.4826·MAD; spread = 100·rSD(v/mean_fp(v) − 1) over far-clear. "
         "Contrast = value vs median of far references (cld > 10 km, QF0, same footprint, |Δlat| ≤ 1°, ≥ 5); relative for "
         "k1-type, absolute for k2, f, a1. SNR = bin median / far-clear (> 20 km, QF0) contrast rSD; bin medians need n ≥ 21; "
         "bins on QF0 unless marked 'all QF'. AM fields with ok_am == 0 are set to NaN before use. Pooled statistics "
         "concatenate soundings over orbits (contrasts and fp-relative values are formed within each orbit); each "
         "source is pooled over the orbits that carry it, so compare sources like-for-like only when every orbit has the "
         "T2/AM fields (see the inputs table).", ""]
    L += ["## Inputs and data notes", "", "| date | orbit | N | T2 fields | AM fields | land split | prior | wall (s) |",
          "|---|---|---|---|---|---|---|---|"]
    for o in orbit_info:
        L.append(f"| {o['date']} | {o['orbit']} | {o['N']} | {'yes' if o['has_t2'] else 'no'} | {'yes' if o['has_am'] else 'no'} | "
                 f"{o['land_src']} | {o['prior_src']} | {o['wall_s']:.1f} |")
    L.append("")
    for o in orbit_info:
        miss = o["missing"]
        L.append(f"- **{o['date']} {o['orbit']}** (tau_convolution attr: `{o['tau_attr'] or 'n/a'}`): " + "; ".join(o["notes"])
                 + (f". Missing datasets ({len(miss)}): " + ", ".join(f"`{k}`" for k in miss[:40]) + (" …" if len(miss) > 40 else "") if miss else ". No requested dataset missing."))
    tc = sinks["target_check"].read()
    if len(tc):
        L += ["", "Reference-window check (anomaly recomputed with the vectorised pipeline window vs the stored target):", "",
              md_table(tc, ["date", "orbit", "target", "n_stored", "n_recomputed", "n_both", "max_abs_diff_ppm", "n_finite_mismatch"])]
    # ---------------- item 4
    fc_p = pd.DataFrame(farclear_table(Fp))
    ct_p = pd.DataFrame(contrast_table(Fp, "QF0") + contrast_table(Fp, "all QF"))
    L += ["## Item 4.1 — far-clear k1 medians and spreads", "",
          "Sources: " + "; ".join(f"`{s}` = {lab}" for s, lab in SOURCES) + ". Cells: median (spread %, n).", "",
          "### Pooled", "", farclear_md(fc_p)]
    fcs = sinks["farclear"].read()
    if per_orbit_md and n_orb > 1:
        for (dt, ob), g in fcs.groupby(["date", "orbit"]):
            L += [f"### {dt} {ob}", "", farclear_md(g)]
    L += ["## Item 4.2 — near-cloud contrasts (QF0)", "", "Cells: median contrast (SNR, n). rel = relative, abs = absolute.", "",
          "### Pooled", "", contrast_md(ct_p[ct_p.qf_sel == "QF0"]) if len(ct_p) else "_(no cloud distances)_\n"]
    cts = sinks["contrasts"].read()
    if per_orbit_md and n_orb > 1 and len(cts):
        for (dt, ob), g in cts[cts.qf_sel == "QF0"].groupby(["date", "orbit"]):
            L += [f"### {dt} {ob}", "", contrast_md(g)]
    if len(ct_p):
        L += ["### Pooled, all contrast quantities (QF0; bars use these)", "", contrast_md(ct_p[ct_p.qf_sel == "QF0"], requested=False)]
    # 4.3
    lbs = sinks["lat_breakdown"].read()
    L += ["## Item 4.3 — 5° latitude breakdown of the 0–1 and 1–3 km bins", "",
          f"n = QF0 soundings with a contrast; QF1 frac = QF1 / all soundings in the bin; median needs n ≥ {args.lat_min_n}; "
          "SNR uses the orbit's far-clear contrast rSD; 'path' = both O2A and SCO2 k1_am contrasts > 0 with SNR ≥ 2.", ""]
    lb_cols = ["bin", "lat_band", "n_all", "qf1_frac"]
    for q, b in LATBREAK_QTY:
        lb_cols += [f"n_{b}_{q}", f"con_{b}_{q}", f"snr_{b}_{q}"]
    lb_cols.append("path_type_signature")
    lb_fmt = {c: ("{:+.3%}" if ("k1_am" in c and c.startswith("con")) else "{:+.4f}" if c.startswith("con") else "{:.2f}") for c in lb_cols}
    lb_fmt["qf1_frac"] = "{:.2f}"
    if len(lbs):
        if per_orbit_md:
            for (dt, ob), g in lbs.groupby(["date", "orbit"]):
                L += [f"### {dt} {ob}", "", md_table(g, lb_cols, lb_fmt)]
        flagged = lbs[lbs.path_type_signature.astype(bool)]
        L += [f"Path-type signature rows over all orbits: {len(flagged)} of {len(lbs)}.", ""]
        if len(flagged) and not per_orbit_md:
            L += [md_table(flagged, ["date", "orbit"] + lb_cols, lb_fmt)]
        # pooled per 5-degree band
        if n_orb > 1:
            Fq = Fp.copy()
            L += ["### Pooled over orbits per 5° band", "", md_table(pd.DataFrame(latbreak_table(Fq, args.lat_min_n)), lb_cols, lb_fmt)]
    else:
        L += ["_(no rows)_", ""]
    # 4.4
    dg_p = pd.DataFrame(diag_table(Fp))
    dg_cols = ["band", "surface", "subset", "n", "frac_ok0", "frac_ok_nan", "frac_k1_am_nan", "frac_f_at_bound", "frac_k2diag_neg",
               "k2_median", "frac_k2_pos", "f_median", "f_p05", "f_p95", "f_rsd", "a1_median", "a1_rsd"]
    dg_fmt = {c: "{:.3f}" for c in dg_cols if c.startswith("frac")}
    dg_fmt.update({c: "{:.4f}" for c in ("k2_median", "f_median", "f_p05", "f_p95", "f_rsd", "a1_median", "a1_rsd")})
    L += ["## Item 4.4 — agreed-model diagnostics (land vs water)", "",
          "frac_ok0 = ok_am == 0; frac_k1_am_nan = k1_am NaN as stored; f at bound = f_am ≥ 0.2 − 1e-9 (O2A only); "
          "frac_k2diag_neg among finite k2diag_am; f and a1 statistics after the ok_am mask; a1_rsd absolute.", "",
          "### Pooled", "", md_table(dg_p, dg_cols, dg_fmt) if len(dg_p) else "_(AM fields absent)_\n"]
    dgs = sinks["diagnostics"].read()
    if per_orbit_md and n_orb > 1 and len(dgs):
        for (dt, ob), g in dgs.groupby(["date", "orbit"]):
            L += [f"### {dt} {ob}", "", md_table(g, dg_cols, dg_fmt)]
    # 4.5
    bar_cols = ["version", "qf_sel", "bar", "band", "target", "n01", "n", "s2_con01", "s2_snr01", "path_con01", "path_snr01", "R",
                "w_con01", "w_snr01", "abs_diff", "threshold", "R2_h2o", "pR2_h2o_given_lncld", "pR2_lncld_given_h2o", "decision"]
    L += ["## Item 4.5 — bars A–D (PLAN.md 'Pre-registered definitions and bars')", "",
          "T2: path = k1_dry_t2, k2 = k2_t2, k1_w = k1w_t2. AM: path = k1_am, k2 = k2_am, k1_w = k1w_am. S2 = k1_o2/k2_o2. "
          "Con/abs_diff in relative units for k1 (fraction), absolute for k2. The PLAN fixed 'all QF'; both QF0 and all-QF bins are "
          "shown (far-clear reference always QF0). Bar D: OLS of the contrast on [h2o_scale (abs), tcwv (rel)] contrasts among "
          "soundings < 3 km; the decision row is S2 SCO2 k1.", ""]
    if Fp["cld"].notna().any():
        bp = []
        for ver in ("T2", "AM"):
            for qs in ("QF0", "all QF"):
                bp += bars_table(Fp, ver, qs)
        L += ["### Pooled", "", md_table(pd.DataFrame(bp), bar_cols, {"R": "{:.2f}", "decision": "{}"})]
        bss = sinks["bars"].read()
        if per_orbit_md and n_orb > 1 and len(bss):
            for (dt, ob), g in bss.groupby(["date", "orbit"]):
                L += [f"### {dt} {ob}", "", md_table(g, bar_cols, {"R": "{:.2f}"})]
    # ---------------- item 6
    L += ["## Item 6a — coefficients vs L2 anomaly", "",
          "Targets as stored in the variant file (`bc_r10` = xco2_bc_anomaly, `raw_r10` = xco2_raw_anomaly, r05/r15 variants, "
          "`train` = r05 over water, r15 over land by sfc_type). Local anomaly = coefficient − its mean over the target's own "
          "reference window (bc pool = fit-time pool of the stored target, raw pool = finite xco2_raw). Ridge: standardised "
          "predictors, λ = 0.01, 10 contiguous along-track blocks, leave-one-block-out, R² of the concatenated held-out "
          "predictions; columns with < 90 % finite values in the subset (e.g. SCO2 a2_am, land-only) or zero variance "
          "(e.g. k2_am pinned at 0) are dropped, then complete cases.", ""]
    acs = sinks["anomaly_corr"].read()
    if len(acs):
        rid = acs[acs.kind == "ridge"]
        show_t = [t for t in rid.target.unique() if t.endswith("r10") or "train" in t]
        set_order = ["production set", "agreed set", "production k1 x3", "agreed k1 x3", "both sets"]
        g = rid.groupby(["target", "subset", "form", "coef"])["r2"]
        agg = pd.DataFrame({"median": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75), "n_orbits": g.count()}).reset_index()
        lines = ["| target | subset | form | " + " | ".join(set_order) + " |", "|---|---|---|" + "---|" * len(set_order)]
        for (t, s, f), gg in agg[agg.target.isin(show_t)].groupby(["target", "subset", "form"], sort=False):
            cells = []
            for so in set_order:
                r = gg[gg.coef == so]
                if r.empty or not np.isfinite(r.iloc[0]["median"]):
                    cells.append("—")
                elif n_orb == 1:
                    cells.append(f"{r.iloc[0]['median']:.2f}")
                else:
                    cells.append(f"{r.iloc[0]['median']:.2f} [{r.iloc[0]['p25']:.2f}, {r.iloc[0]['p75']:.2f}]")
            lines.append(f"| {t} | {s} | {f} | " + " | ".join(cells) + " |")
        nrow = rid[rid.target.isin(show_t)].groupby(["target", "subset"])["n"].median()
        L += ["### Held-out ridge R² (" + ("single orbit" if n_orb == 1 else "median [p25, p75] over orbits") + ")", "",
              "\n".join(lines), "", "Median n of complete cases per subset: "
              + "; ".join(f"{t} {s}: {int(v)}" for (t, s), v in nrow.items() if t == show_t[0]) + ".", ""]
        drops = rid[rid.note.fillna("").str.startswith("dropped")]["note"].value_counts().head(5)
        if len(drops):
            L += ["Most frequent column drops: " + "; ".join(f"{k} ({v} fits)" for k, v in drops.items()), ""]
        # per-date R2
        main_t = next((t for t in rid.target.unique() if t.startswith("bc_train")), "bc_r10")
        pdate = rid[(rid.target == main_t) & (rid.subset == "QF0 cld<10")].groupby(["date", "form", "coef"])["r2"].median().unstack("coef")
        if len(pdate):
            pdate = pdate.reset_index()
            L += [f"### Per-date median R² ({main_t}, QF0 cld<10)", "", md_table(pdate, ["date", "form"] + set_order, {c: "{:.2f}" for c in set_order})]
        # correlations + sign consistency
        cor = acs[acs.kind == "corr"].copy()
        cor["orbit_key"] = cor["date"] + "_" + cor["orbit"]
        ref_present = (cor.orbit == args.ref_orbit).any()
        rows = []
        for (t, s, f, c), gg in cor.groupby(["target", "subset", "form", "coef"]):
            ds = gg.groupby("date")["pearson"].median()
            ds = ds[np.isfinite(ds)]
            if ref_present:
                rr = gg[gg.orbit == args.ref_orbit]["pearson"]
                ref = float(rr.iloc[0]) if len(rr) else np.nan
            else:
                ref = float(gg["pearson"].median())
            frac = float((np.sign(ds) == np.sign(ref)).mean()) if len(ds) and np.isfinite(ref) else np.nan
            rows.append(dict(target=t, subset=s, form=f, coef=c, n_orbits=int(gg["pearson"].notna().sum()), ref_r=ref,
                             median_pearson=float(gg["pearson"].median()), median_spearman=float(gg["spearman"].median()),
                             n_dates=len(ds), frac_dates_same_sign=frac))
        sc = pd.DataFrame(rows)
        sc.to_csv(out / "agreed_eval_anomaly_sign.csv", index=False)
        L += ["### Per-coefficient correlation" + (" and sign consistency" if n_orb > 1 else ""), "",
              ("Reference sign: orbit " + args.ref_orbit if ref_present else
               f"Reference orbit {args.ref_orbit} not among the inputs: reference sign = sign of the median r over all orbits") +
              ". Date sign = sign of the median r over the date's orbits. Full table: agreed_eval_anomaly_sign.csv.", ""]
        for t in [x for x in show_t if x.startswith("bc_")]:
            sub = sc[(sc.target == t) & sc.subset.isin(["QF0 cld<3", "QF0 all"])]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="coef", columns=["subset", "form"], values="median_pearson")
            piv_s = sub.pivot_table(index="coef", columns=["subset", "form"], values="median_spearman")
            fr = sub.pivot_table(index="coef", columns=["subset", "form"], values="frac_dates_same_sign")
            cols = list(piv.columns)
            lines = [f"**{t}** — Pearson / Spearman" + (" / frac dates same sign" if n_orb > 1 else ""), "",
                     "| coef | " + " | ".join(f"{s}, {f}" for s, f in cols) + " |", "|---|" + "---|" * len(cols)]
            for c in piv.index:
                cells = []
                for col in cols:
                    p, sp = piv.loc[c, col], piv_s.loc[c, col]
                    cell = f"{fmtv(p, '{:+.2f}')} / {fmtv(sp, '{:+.2f}')}"
                    if n_orb > 1:
                        cell += f" / {fmtv(fr.loc[c, col], '{:.2f}')}"
                    cells.append(cell)
                lines.append(f"| {c} | " + " | ".join(cells) + " |")
            L += ["\n".join(lines), ""]
    else:
        L += ["_(no anomaly metrics: no cloud distances or no targets)_", ""]
    # 6b
    ps_p = pd.DataFrame(pseudo_table(Fp))
    ps_cols = ["estimator", "surface", "n_far", "med_E_minus_bc", "med_E_minus_raw", "rsd_E_minus_bc", "rsd_E_minus_raw",
               "med_E_minus_prior", "fp_spread_E_minus_prior", "rsd_fpdm_E_minus_prior", "rsd_fpdm_E_minus_bc"] + [f"fp{i}_med_E_minus_prior" for i in range(8)]
    priors = sorted({o["prior_src"].split(";")[0] for o in orbit_info})
    L += ["## Item 6b — pseudo-XCO2 level (unscaled, far-clear)", "",
          "E_S = prior·k1_am(SCO2); E_W = prior·k1_am(WCO2); E1 = prior·k1_am(SCO2)/k1_am(O2A). Prior: " + "; ".join(priors)
          + ". ppm. fp_spread = max − min of the per-footprint medians of E − prior; rsd_fpdm = rSD after removing per-footprint "
          "medians. `prior` and `L2 raw` rows are references.", "", "### Pooled", "",
          md_table(ps_p, ps_cols, {c: "{:+.2f}" if c.startswith("med") or c.startswith("fp") and "med" in c else "{:.2f}" for c in ps_cols}) if len(ps_p) else "_(no rows)_\n"]
    pss = sinks["pseudo_xco2"].read()
    if per_orbit_md and n_orb > 1 and len(pss):
        for (dt, ob), g in pss.groupby(["date", "orbit"]):
            L += [f"### {dt} {ob}", "", md_table(g, ps_cols[:11])]
    # 6c
    fl_p = pd.DataFrame(flag_table(Fp))
    L += ["## Item 6c — anomaly product and O2A flag", "",
          f"Window: pipeline reference window r = {args.anom_radius} km" + (" (r05 water / r15 land)" if args.anom_radius == "train" else "")
          + ", bc pool. ΔX_S = prior·Δk1_am(SCO2)/1.60, ΔX_W = prior·Δk1_am(WCO2)/1.01 (Δ = k1 − window mean); flag = "
          f"|Δk1_am(O2A)/window mean| > {FLAG_PCT:.0f} %. xa = stored xco2 anomaly of the same window. ppm.", ""]
    if len(fl_p):
        L += ["### Pooled — far-clear scatter and correlation", "",
              md_table(fl_p[fl_p.section == "far-clear"], ["surface", "n", "rsd_dXS", "rsd_dXW", "r_dXS_xa_far", "r_dXW_xa_far",
                                                           "r_dXS_xa_qf0", "r_dXW_xa_qf0", "r_dXS_xraw_qf0", "r_dXW_xraw_qf0",
                                                           "rsd_o2a_rel_pct", "p95_abs_o2a_rel_pct", "p99_abs_o2a_rel_pct"],
                       {c: "{:+.2f}" for c in ("r_dXS_xa_far", "r_dXW_xa_far", "r_dXS_xa_qf0", "r_dXW_xa_qf0", "r_dXS_xraw_qf0", "r_dXW_xraw_qf0")}),
              "### Pooled — near-cloud bins (QF0)", "",
              md_table(fl_p[fl_p.section == "bin QF0"], ["surface", "bin", "n", "med_dXS", "med_dXW", "med_xa_bc", "med_xa_raw",
                                                         "med_dXS_unflagged", "med_dXW_unflagged", "flag_rate"],
                       {c: "{:+.2f}" for c in ("med_dXS", "med_dXW", "med_xa_bc", "med_xa_raw", "med_dXS_unflagged", "med_dXW_unflagged")} | {"flag_rate": "{:.3f}"}),
              "### Pooled — flag rates", "",
              md_table(fl_p[fl_p.section == "flag rate"], ["surface", "group", "n", "flag_rate"], {"flag_rate": "{:.3f}"})]
        fls = sinks["flag"].read()
        kz = fls[fls.section == "Kozienice along-track"] if len(fls) else fls
        if len(kz):
            L += ["### Kozienice (2021-09-06, 38200a): 5-km along-track medians within 60 km of the plant", "",
                  md_table(kz, ["bin", "n", "n_unflagged", "med_dXS_unflagged", "med_dXW_unflagged", "med_xa_bc", "med_xa_raw"])]
        else:
            L += ["Kozienice section: 2021-09-06 / 38200a_ND not among the inputs (skipped).", ""]
        if per_orbit_md and n_orb > 1 and len(fls):
            for (dt, ob), g in fls[fls.section == "flag rate"].groupby(["date", "orbit"]):
                L += [f"### {dt} {ob} flag rates", "", md_table(g, ["surface", "group", "n", "flag_rate"], {"flag_rate": "{:.3f}"})]
    else:
        L += ["_(anomaly product not computed: AM fields, prior or cloud distances absent)_", ""]
    figs = figures(fc_p, ct_p, fl_p, Fp, out)
    L += ["## Figures", ""] + [f"- `{p.name}`" for p in figs] + [""]
    (out / "agreed_eval_pooled.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

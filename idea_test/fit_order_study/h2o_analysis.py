"""H2O vs cloud-path test on Doppler-corrected τ (orbit 29252a_GL); bars in PLAN.md.

Per sounding and band: single-τ fits (S2 unconstrained order 2, S3 bounded order 3) and
two-τ fits on the species split of ``h2o_split.py``:

    T2 : ln T = c − k1_dry τ_dry − k1_w τ_h2o + ½ k2 τ_tot²      (unconstrained, reported)
    T2d: same with ½ k2 τ_dry²                                   (variant)
    T3 : T2 − ⅓ k3 τ_tot³, k1_dry, k2 ≥ 0                         (link to resid_fitters)

τ_dry includes Rayleigh. Contrasts use the latitude-local far reference of
``near_cloud_doppler.py``. Outputs results/h2o_per_sounding_29252a_GL.parquet,
h2o_contrasts_29252a_GL.csv, h2o_correlations_29252a_GL.csv, h2o_summary.md, h2o_*.png.

Usage:
    python idea_test/fit_order_study/h2o_analysis.py [--workers 8]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                 # one BLAS thread per forked worker

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cut_sweep_real import DEFAULTS, md_table  # noqa: E402
from h2o_split import SPLIT_FILE, TAU_CORR  # noqa: E402
from orbit_io import BANDS, load_orbit  # noqa: E402
from run_study import robust_sd  # noqa: E402
from variants import design_matrix, fit_cumulant  # noqa: E402

OUT = HERE / "results"
TAG = "29252a_GL"
BINS = [(0, 1), (1, 3), (3, 10), (10, 20), (20, 1e9)]
BIN_LABELS = ["0-1", "1-3", "3-10", "10-20", ">20"]
G_ACC = 9.80665
FITS = ("S2", "S3", "T2", "T2d", "T3")
COLORS = {"S2": "#2a78d6", "S3": "#6250d6", "T2": "#eb6834", "T3": "#e87ba4", "w": "#1baf7a"}
_G: dict = {}


# --------------------------------------------------------------------------- fits


def _lstsq(a, y):
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    r = y - a @ coef
    rss = float(r @ r)
    cov = np.linalg.pinv(a.T @ a) * rss / max(len(y) - a.shape[1], 1)
    return coef, float(np.sqrt(rss / len(y))), np.sqrt(np.diag(cov))


def _bounded(a, y, pos):
    """Least squares with coef[pos] ≥ 0 (BVLS fallback, as resid_fitters.fit_one)."""
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    if np.any(coef[pos] < 0):
        lb = np.full(a.shape[1], -np.inf)
        lb[pos] = 0.0
        coef = lsq_linear(a, y, bounds=(lb, np.full(a.shape[1], np.inf)), method="bvls").x
    r = y - a @ coef
    return coef, float(np.sqrt(r @ r / len(y)))


def fit_sounding(j: int) -> list[dict]:
    rows = []
    k = _G["pos"][j]
    for ib, band in enumerate(BANDS):
        t = _G["tau"][j, ib].astype(float)
        td = (_G["dry"][band][k] + _G["ray"][band][k]).astype(float)
        tw = _G["h2o"][band][k].astype(float)
        y, s = _G["ln_t"][ib, j].astype(float), _G["sig"][ib, j].astype(float)
        ok = np.isfinite(y) & np.isfinite(t) & np.isfinite(s) & (s > 0) & np.isfinite(td) & np.isfinite(tw)
        rec = dict(j=j, band=band, nchan=int(ok.sum()))
        if ok.sum() < 10:
            rows.append(rec)
            continue
        t, td, tw, y = t[ok], td[ok], tw[ok], y[ok]
        c2, rms2, _ = _lstsq(design_matrix(t, 2), y)
        r3 = fit_cumulant(_G["tau"][j, ib], _G["ln_t"][ib, j], _G["sig"][ib, j], 3, weighted=False)
        rec.update(S2_k1=c2[1], S2_k2=c2[2], S2_rms=rms2, S3_k1=r3["k1"], S3_k2=r3["k2"],
                   S3_rms=r3["resid_rms"], f_h2o=float(tw.sum() / t.sum()))
        a2 = np.column_stack([np.ones_like(t), -td, -tw, 0.5 * t**2])
        c, rms, se = _lstsq(a2, y)
        rec.update(T2_k1_dry=c[1], T2_k1_w=c[2], T2_k2=c[3], T2_rms=rms, T2_se_k1_dry=se[1], T2_se_k1_w=se[2],
                   T2_cond=float(np.linalg.cond(a2 / np.linalg.norm(a2, axis=0))))
        c, rms, _ = _lstsq(np.column_stack([np.ones_like(t), -td, -tw, 0.5 * td**2]), y)
        rec.update(T2d_k1_dry=c[1], T2d_k1_w=c[2], T2d_k2=c[3], T2d_rms=rms)
        c, rms = _bounded(np.column_stack([np.ones_like(t), -td, -tw, 0.5 * t**2, -t**3 / 3]), y, [1, 3])
        rec.update(T3_k1_dry=c[1], T3_k1_w=c[2], T3_k2=c[3], T3_rms=rms)
        for f in ("T2", "T2d", "T3"):
            rec[f"{f}_ratio_w_dry"] = rec[f"{f}_k1_w"] / rec[f"{f}_k1_dry"]
        rows.append(rec)
    return rows


def _chunk(js):
    return [r for j in js for r in fit_sounding(j)]


# --------------------------------------------------------------------------- contrasts


def lat_local_contrast(v, cld, qf, lat, fp, relative):
    """value − median of (cld > 10 km, QF0, |Δlat| ≤ 1°, same fp) references (near_cloud_doppler.py)."""
    n = len(v)
    ref_ok = (cld > 10) & (qf == 0) & np.isfinite(v)
    ref = np.full(n, np.nan)
    for f in range(8):
        m_f = fp == f
        rl, rv = lat[ref_ok & m_f], v[ref_ok & m_f]
        for j in np.flatnonzero(m_f):
            sel = np.abs(rl - lat[j]) <= 1.0
            if sel.sum() >= 5:
                ref[j] = np.nanmedian(rv[sel])
    con = v - ref
    return con / ref if relative else con


def bin_stats(con, cld, far):
    far_sd = robust_sd(con[far])
    rec = {"far_spread": far_sd}
    for (lo, hi), lab in zip(BINS, BIN_LABELS):
        m = (cld >= lo) & (cld < hi) & np.isfinite(con)
        rec[f"n_{lab}"] = int(m.sum())
        med = np.nanmedian(con[m]) if m.sum() > 20 else np.nan
        rec[f"con_{lab}"] = med
        rec[f"snr_{lab}"] = med / far_sd
        rec[f"se_{lab}"] = 1.2533 * robust_sd(con[m]) / np.sqrt(m.sum()) if m.sum() > 20 else np.nan
    return rec


# --------------------------------------------------------------------------- regression helpers


def _ols_r2(y, xs):
    a = np.column_stack([np.ones(len(y)), *xs]) if xs else np.ones((len(y), 1))
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    r = y - a @ coef
    return 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())), float(r @ r), coef


def partial_r2(y, x_keep, x_test):
    """(RSS_keep − RSS_full) / RSS_keep."""
    _, rss_k, _ = _ols_r2(y, x_keep)
    _, rss_f, _ = _ols_r2(y, x_keep + x_test)
    return (rss_k - rss_f) / rss_k


def partial_r(y, x, z):
    """Pearson r of y and x after regressing both on z."""
    def res(v):
        a = np.column_stack([np.ones(len(v)), *z])
        return v - a @ np.linalg.lstsq(a, v, rcond=None)[0]
    return float(np.corrcoef(res(y), res(x))[0, 1])


# --------------------------------------------------------------------------- main


def load_extra(n):
    keys = ("h2o_scale", "tcwv", "h2o_ratio_bc", "aod_total", "aod_water", "aod_ice")
    with h5py.File(DEFAULTS["fit_details"], "r") as f:
        ex = {k: f[k][...].astype(float) for k in keys}
        q = f["q_profile"][...].astype(float)
        p = f["p_profile"][...].astype(float)
    ex["tcwv_met"] = np.trapz(q, p, axis=1) / G_ACC
    for k, v in ex.items():
        v[(v < -900) | ~np.isfinite(v)] = np.nan
        if len(v) != n:
            raise ValueError(k)
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for name in (f"h2o_per_sounding_{TAG}.parquet", f"h2o_contrasts_{TAG}.csv", f"h2o_correlations_{TAG}.csv",
                 "h2o_summary.md"):
        if (OUT / name).exists():
            raise FileExistsError(OUT / name)
    orb = load_orbit(str(DEFAULTS["l1b"]), str(DEFAULTS["fp_tau"]), str(DEFAULTS["fit_details"]))
    n = len(orb.sounding_id)
    tau = np.load(TAU_CORR)["tau"]
    zs = np.load(SPLIT_FILE)
    sel = zs["j"]
    pos = np.full(n, -1)
    pos[sel] = np.arange(len(sel))
    _G.update(tau=tau, ln_t=orb.ln_t, sig=orb.sigma_ln_t, pos=pos,
              dry={b: zs[f"tau_dry_{b}"] for b in BANDS}, h2o={b: zs[f"tau_h2o_{b}"] for b in BANDS},
              ray={b: zs[f"tau_ray_{b}"] for b in BANDS})
    with mp.get_context("fork").Pool(args.workers) as pool:
        parts = pool.map(_chunk, np.array_split(sel, 64))
    long = pd.DataFrame([r for p in parts for r in p])
    long.to_parquet(OUT / f"h2o_per_sounding_{TAG}.parquet", index=False)

    meta, fp = orb.meta, orb.fp_number
    cld, qf, lat = meta["cld_dist_km"].astype(float), meta["xco2_qf"], meta["lat"].astype(float)
    far = (cld > 20) & (qf == 0)
    ex = load_extra(n)
    ex["xco2_bc_anomaly"] = meta["xco2_bc_anomaly"].astype(float)
    ex["cld_dist_km"] = cld

    # ---------------- contrasts
    quantities = [("S2", "k1", True), ("S2", "k2", False), ("S2", "rms", True), ("S3", "k1", True),
                  ("S3", "k2", False)]
    for f in ("T2", "T2d", "T3"):
        quantities += [(f, "k1_dry", True), (f, "k1_w", True), (f, "k2", False), (f, "ratio_w_dry", True)]
    quantities += [("T2", "rms", True)]
    con = {}
    recs = []
    for band in BANDS:
        w = long[long.band == band].set_index("j").reindex(np.arange(n))
        for fit, q, rel in quantities:
            v = w[f"{fit}_{q}"].values.astype(float)
            c = lat_local_contrast(v, cld, qf, lat, fp, rel)
            con[(band, fit, q)] = c
            recs.append({"band": band, "fit": fit, "quantity": q, "kind": "rel" if rel else "abs",
                         "far_median": np.nanmedian(v[far]), **bin_stats(c, cld, far)})
    for k in ("h2o_scale", "tcwv", "tcwv_met", "h2o_ratio_bc", "aod_total", "aod_water", "aod_ice"):
        for rel in (False, True) if k in ("h2o_scale", "tcwv", "tcwv_met") else (False,):
            v = ex[k]
            c = lat_local_contrast(v, cld, qf, lat, fp, rel)
            con[("met", k, "rel" if rel else "abs")] = c
            recs.append({"band": "l2/met", "fit": "-", "quantity": k, "kind": "rel" if rel else "abs",
                         "far_median": np.nanmedian(v[far]), **bin_stats(c, cld, far)})
    ct = pd.DataFrame(recs)
    ct.to_csv(OUT / f"h2o_contrasts_{TAG}.csv", index=False)

    # ---------------- correlations
    near = cld < 3
    crecs = []
    preds = ("h2o_scale", "tcwv", "h2o_ratio_bc", "cld_dist_km", "aod_total", "aod_water", "xco2_bc_anomaly")
    targets = [(b, "S2", "k1") for b in ("sco2", "wco2")] + [(b, "S2", "k2") for b in ("sco2", "wco2")] + \
              [(b, "T2", "ratio_w_dry") for b in ("sco2", "wco2")] + [(b, "T2", "k1_dry") for b in ("sco2", "wco2")]
    h2o_con = [con[("met", "h2o_scale", "abs")], con[("met", "tcwv", "rel")]]
    lncld = np.log(np.clip(cld, 0.1, None))
    for sub_name, sub in (("<3km", near), ("far_clear", far)):
        for tgt in targets:
            y = con[tgt]
            for p in preds:
                for form in ("value", "contrast"):
                    if form == "contrast" and p in ("cld_dist_km", "xco2_bc_anomaly"):
                        continue                     # no latitude-local contrast for these
                    x = ex[p] if form == "value" else con[("met", p, "rel" if p == "tcwv" else "abs")]
                    m = sub & np.isfinite(y) & np.isfinite(x)
                    if m.sum() < 30:
                        continue
                    crecs.append({"subset": sub_name, "target": "_".join(tgt) + "_con", "predictor": p, "form": form,
                                  "stat": "pearson_r", "n": int(m.sum()), "value": float(np.corrcoef(y[m], x[m])[0, 1])})
                    crecs.append({"subset": sub_name, "target": "_".join(tgt) + "_con", "predictor": p, "form": form,
                                  "stat": "spearman_r", "n": int(m.sum()), "value": float(spearmanr(y[m], x[m])[0])})
            m = sub & np.isfinite(y) & np.isfinite(h2o_con[0]) & np.isfinite(h2o_con[1])
            yy, hh, cc = y[m], [h[m] for h in h2o_con], [lncld[m]]
            base = {"subset": sub_name, "target": "_".join(tgt) + "_con", "form": "contrast", "n": int(m.sum())}
            crecs.append({**base, "predictor": "h2o_scale+tcwv", "stat": "R2", "value": _ols_r2(yy, hh)[0]})
            if sub_name == "<3km":
                crecs += [{**base, "predictor": "ln_cld", "stat": "R2", "value": _ols_r2(yy, cc)[0]},
                          {**base, "predictor": "h2o_scale+tcwv | ln_cld", "stat": "partial_R2",
                           "value": partial_r2(yy, cc, hh)},
                          {**base, "predictor": "ln_cld | h2o_scale+tcwv", "stat": "partial_R2",
                           "value": partial_r2(yy, hh, cc)},
                          {**base, "predictor": "ln_cld | h2o_scale+tcwv", "stat": "partial_r",
                           "value": partial_r(yy, cc[0], hh)},
                          {**base, "predictor": "h2o_scale | ln_cld", "stat": "partial_r",
                           "value": partial_r(yy, hh[0], cc)}]
    # far-clear-calibrated H2O share of the near-cloud median contrast (extra)
    hs = con[("met", "h2o_scale", "abs")]
    for tgt in targets[:4] + targets[6:]:
        y = con[tgt]
        mf = far & np.isfinite(y) & np.isfinite(hs)
        _, _, coef = _ols_r2(y[mf], [hs[mf]])
        for lab, (lo, hi) in (("0-1", (0, 1)), ("1-3", (1, 3))):
            mb = (cld >= lo) & (cld < hi) & np.isfinite(y) & np.isfinite(hs)
            pred = np.median(coef[0] + coef[1] * hs[mb])
            crecs.append({"subset": lab, "target": "_".join(tgt) + "_con", "predictor": "h2o_scale (far slope)",
                          "form": "contrast", "stat": "pred_share", "n": int(mb.sum()),
                          "value": float(pred / np.median(y[mb])), "slope": float(coef[1])})
    # raw two-τ ratio vs h2o_scale (values)
    for b in ("sco2", "wco2", "o2a"):
        y = long[long.band == b].set_index("j").reindex(np.arange(n))["T2_ratio_w_dry"].values
        for sub_name, sub in (("<3km", near), ("far_clear", far), ("all", np.ones(n, bool))):
            m = sub & np.isfinite(y) & np.isfinite(ex["h2o_scale"])
            crecs.append({"subset": sub_name, "target": f"{b}_T2_ratio_w_dry", "predictor": "h2o_scale",
                          "form": "value", "stat": "pearson_r", "n": int(m.sum()),
                          "value": float(np.corrcoef(y[m], ex["h2o_scale"][m])[0, 1])})
    cr = pd.DataFrame(crecs)
    cr.to_csv(OUT / f"h2o_correlations_{TAG}.csv", index=False)

    summary(ct, cr, long, zs, far, n)
    figures(ct, con, long, ex, cld, far, n)
    print("done")


def summary(ct, cr, long, zs, far, n):
    cols = ["band", "fit", "quantity", "kind", "far_median", "far_spread", "n_0-1", "con_0-1", "snr_0-1",
            "con_1-3", "snr_1-3", "con_3-10", "snr_3-10", "con_10-20", "snr_10-20"]
    lines = [f"# H2O vs cloud-path test — {TAG}", "",
             "Species split validation (Σ dry + H2O + Rayleigh vs stored corrected τ):", "",
             "```", str(zs["validation"]), "```", "",
             f"Split soundings: {len(zs['j'])} of {n}. Contrasts as `near_cloud_doppler.py` "
             "(relative for k1-type/ratio/rms, absolute for k2). SNR = bin median / far-clear robust SD.", ""]
    for band in BANDS:
        lines += [f"## {band}", "", md_table(ct[ct.band == band], cols), ""]
    lines += ["## L2 / met water-vapour and aerosol contrasts", "", md_table(ct[ct.band == "l2/met"], cols), ""]
    w = long.merge(pd.DataFrame({"j": np.arange(n), "far": far}), on="j")
    fs = w[w.far].groupby("band")[["S2_k1", "S2_k2", "T2_k1_dry", "T2_k1_w", "T2_k2", "T2_ratio_w_dry",
                                    "S2_rms", "T2_rms", "T2_se_k1_w", "T2_cond"]].median().reset_index()
    sd = w[w.far].groupby("band")[["T2_k1_dry", "T2_k1_w"]].agg(robust_sd).add_suffix("_robust_sd").reset_index()
    fs = fs.merge(sd, on="band")
    lines += ["## Far-clear medians", "", md_table(fs, list(fs.columns)), ""]
    lines += ["## Correlations, R², partial R² (selected)", "",
              md_table(cr[cr.stat.isin(["R2", "partial_R2", "partial_r", "pred_share"])].fillna(np.nan),
                       ["subset", "target", "predictor", "stat", "n", "value"]), ""]
    sel = cr[(cr.stat == "pearson_r")]
    lines += ["## Pearson r (all)", "", md_table(sel, ["subset", "target", "predictor", "form", "n", "value"]), ""]
    (OUT / "h2o_summary.md").write_text("\n".join(lines))


def figures(ct, con, long, ex, cld, far, n):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(BIN_LABELS))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    for ib, band in enumerate(BANDS):
        ax = axes[0, ib]
        for fit, q, lab, col, off in (("S2", "k1", "single-τ k1 (o2)", COLORS["S2"], -0.15),
                                      ("T2", "k1_dry", "two-τ k1_dry (o2)", COLORS["T2"], -0.05),
                                      ("S3", "k1", "single-τ k1 (o3)", COLORS["S3"], 0.05),
                                      ("T3", "k1_dry", "two-τ k1_dry (o3)", COLORS["T3"], 0.15)):
            r = ct[(ct.band == band) & (ct.fit == fit) & (ct.quantity == q)].iloc[0]
            y = np.array([r[f"con_{b}"] for b in BIN_LABELS]) * 100
            e = np.array([r[f"se_{b}"] for b in BIN_LABELS]) * 100
            ax.errorbar(x + off, y, e, marker="o", ms=4, lw=1.5, capsize=2, color=col, label=lab)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(x, BIN_LABELS)
        ax.set_title(f"{band.upper()}: k1 contrast")
        ax.set_ylabel("contrast vs local far (%)")
        ax = axes[1, ib]
        for fit, q, lab, col, off in (("S2", "k2", "single-τ k2 (o2)", COLORS["S2"], -0.1),
                                      ("T2", "k2", "two-τ k2 (o2)", COLORS["T2"], 0.1)):
            r = ct[(ct.band == band) & (ct.fit == fit) & (ct.quantity == q)].iloc[0]
            y = np.array([r[f"con_{b}"] for b in BIN_LABELS])
            e = np.array([r[f"se_{b}"] for b in BIN_LABELS])
            ax.errorbar(x + off, y, e, marker="o", ms=4, lw=1.5, capsize=2, color=col, label=lab)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(x, BIN_LABELS)
        ax.set_xlabel("cloud distance (km)")
        ax.set_title(f"{band.upper()}: k2 contrast (absolute)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / f"h2o_k1_contrast_bins_{TAG}.png", dpi=150)
    plt.close(fig)

    hs = con[("met", "h2o_scale", "abs")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (b, f, q, lab) in zip(axes, (("sco2", "S2", "k1", "SCO2 single-τ k1 contrast (rel.)"),
                                         ("sco2", "S2", "k2", "SCO2 single-τ k2 contrast (abs.)"),
                                         ("wco2", "S2", "k1", "WCO2 single-τ k1 contrast (rel.)"))):
        y = con[(b, f, q)]
        mf = far & np.isfinite(y) & np.isfinite(hs)
        ax.scatter(hs[mf], y[mf], s=4, color="#b4b3ad", label="far-clear (> 20 km, QF0)", rasterized=True)
        mn = (cld < 10) & np.isfinite(y) & np.isfinite(hs)
        o = np.argsort(-cld[mn])
        sc = ax.scatter(hs[mn][o], y[mn][o], s=6, c=cld[mn][o], cmap="viridis", vmin=0, vmax=10, rasterized=True)
        _, _, coef = _ols_r2(y[mf], [hs[mf]])
        xx = np.linspace(np.nanpercentile(hs[mn], 1), np.nanpercentile(hs[mn], 99), 10)
        ax.plot(xx, coef[0] + coef[1] * xx, color="#0b0b0b", lw=1.2, label="far-clear fit")
        ax.set_xlabel("h2o_scale contrast (absolute)")
        ax.set_ylabel(lab)
        lo, hi = np.nanpercentile(y[mn], [0.5, 99.5])
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.colorbar(sc, ax=axes, label="cloud distance (km), < 10 km")
    fig.savefig(OUT / f"h2o_scatter_scale_{TAG}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, b in zip(axes, ("sco2", "wco2")):
        r = long[long.band == b].set_index("j").reindex(np.arange(n))["T2_ratio_w_dry"].values
        m = np.isfinite(r) & np.isfinite(ex["h2o_scale"])
        mf, mn = m & far, m & (cld < 3)
        ax.scatter(ex["h2o_scale"][mf], r[mf], s=4, color="#b4b3ad", label="far-clear", rasterized=True)
        ax.scatter(ex["h2o_scale"][mn], r[mn], s=6, color="#eb6834", label="< 3 km", rasterized=True)
        lim = np.nanpercentile(np.r_[ex["h2o_scale"][m], r[m]], [0.5, 99.5])
        ax.plot(lim, lim, color="#0b0b0b", lw=0.8, label="1:1")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("L2 h2o_scale")
        ax.set_ylabel("two-τ k1_w / k1_dry")
        ax.set_title(b.upper())
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / f"h2o_ratio_vs_scale_{TAG}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

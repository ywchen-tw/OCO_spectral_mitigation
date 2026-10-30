"""Pseudo-XCO2 estimate from the agreed fit model on 29252a_GL.

Question (2026-10-30): how would the agreed fit model perform as a direct XCO2 estimator?
Estimators are NOT scaled to xco2_bc or to the prior (Steven, 2026-10-30): the absolute level is part of
the test. tau_S is computed with the prior CO2 profile, so k1_S = 1 means "prior CO2 on a geometric path".
  E0   f = k1_S                      (SCO2 alone, path assumed geometric)
  E1   f = k1_S / k1_O2A             (O2A as the path reference, l_S = l_O2A assumed)
  E1w  f = k1_W / k1_O2A             (WCO2 as the CO2 band)
  E2   f = k1_S / k1_S_pred, ridge of k1_S on gas-free fit terms fitted on far-clear soundings with f = 1
       assumed there and predicted held-out by along-track block: calibrated to the prior BY CONSTRUCTION,
       kept only as the stand-in for a path forward model
  IDP  xco2_strong_idp as stored (line-by-line benchmark, no offset)
xco2_est = xco2_apriori * f.  Same estimators on the production fit (order-7 k1 on production tau)
and on the two-tau T2 k1_dry (corrected tau, no continuum terms) for comparison.
Outputs results/pseudo_xco2_29252a_GL.{csv,md,png}.
"""
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_model_test import DEFAULTS  # noqa: E402
from h2o_analysis import lat_local_contrast  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
BANDS = ("o2a", "wco2", "sco2")
TAG = "29252a_GL"


def robust_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan


def lag1(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    x = x[m] - np.nanmean(x[m])
    return np.corrcoef(x[:-1], x[1:])[0, 1] if x.size > 3 else np.nan


with h5py.File(DEFAULTS["fit_details"]) as f:
    meta = {k: f[k][...].astype(float) for k in (
        "cld_dist_km", "xco2_qf", "lat", "fp_number", "sfc_type", "xco2_bc", "xco2_apriori", "xco2_raw",
        "xco2_strong_idp", "xco2_weak_idp", "xco2_bc_anomaly", "airmass", "mu_sza", "mu_vza",
        "aod_total", "h2o_scale", "alb_sco2", "ws", "psurf_met", "s31", "xco2_raw")}
    prod = {b: f[f"{b}_k1_fitting_nosg"][...].astype(float) for b in BANDS}
n = meta["lat"].size
cld, qf = meta["cld_dist_km"], meta["xco2_qf"]
far = (cld > 20) & (qf == 0)
bc, pri = meta["xco2_bc"], meta["xco2_apriori"]

# ---- wide tables of k1 per fit
am = pd.read_parquet(OUT / f"final_model_per_sounding_{TAG}.parquet")
h2 = pd.read_parquet(OUT / f"h2o_per_sounding_{TAG}.parquet")


def wide(df, col, name):
    w = df.pivot(index="j", columns="band", values=col).reindex(range(n))
    return {b: w[b].values.astype(float) for b in BANDS}


fits = {
    "agreed": wide(am, "k1", "k1"),
    "two-tau T2": wide(h2, "T2_k1_dry", "k1"),
    "production": prod,
}
terms = {k: wide(am, k, k) for k in ("k1_w", "k2", "a1", "f", "rms")}


def ridge_heldout(target, X, lam=1e-3, nblock=10):
    """Ridge of target on X, calibrated on far-clear, predicted held-out by along-track block."""
    pred = np.full(n, np.nan)
    blocks = np.floor(np.arange(n) / n * nblock).astype(int)
    good = np.isfinite(target) & np.all(np.isfinite(X), axis=1)
    for k in range(nblock):
        tr = far & good & (blocks != k)
        te = good & (blocks == k)
        if tr.sum() < 30:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Z = (X[tr] - mu) / sd
        A = np.c_[np.ones(tr.sum()), Z]
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ target[tr])
        pred[te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sd] @ beta
    return pred


est = {}
for lab, k1 in fits.items():
    est[(lab, "E0 SCO2")] = pri * k1["sco2"]
    est[(lab, "E1 SCO2/O2A")] = pri * k1["sco2"] / k1["o2a"]
    est[(lab, "E1w WCO2/O2A")] = pri * k1["wco2"] / k1["o2a"]
# E2 only for the agreed model: gas-free terms are its own outputs
X = np.c_[fits["agreed"]["o2a"], terms["f"]["o2a"], terms["a1"]["o2a"], terms["a1"]["sco2"], terms["k2"]["sco2"],
          meta["airmass"], 1 / meta["mu_sza"], 1 / meta["mu_vza"]]
pred = ridge_heldout(fits["agreed"]["sco2"], X)
est[("agreed", "E2 ridge(held-out)")] = pri * fits["agreed"]["sco2"] / pred
est[("prior only", "xco2_apriori")] = pri.copy()
est[("L2 IDP", "xco2_strong_idp")] = meta["xco2_strong_idp"].copy()

# ---- metrics
inc_bc = bc - pri                       # retrieved increment vs the prior
fp = meta["fp_number"].astype(int)
rows = []
for (lab, name), x in est.items():
    d = x - bc
    inc = x - pri
    m = far & np.isfinite(d)
    # within-footprint, within-cluster view: remove per-footprint far-clear means of d
    dd = d.copy()
    for k in np.unique(fp):
        s = m & (fp == k)
        dd[s] -= np.nanmean(d[s])
    r_inc = np.corrcoef(inc[m], inc_bc[m])[0, 1]
    slope = np.polyfit(inc_bc[m], inc[m], 1)[0]
    # within-cluster: remove per-footprint-per-cluster means of both increments so the latitude step drops out
    cl = np.digitize(meta["lat"], [-40, 0])
    iw, bw = inc.copy(), inc_bc.copy()
    for key in set(zip(fp[m], cl[m])):
        s_ = m & (fp == key[0]) & (cl == key[1])
        iw[s_] -= np.nanmean(inc[s_]); bw[s_] -= np.nanmean(inc_bc[s_])
    r_within = np.corrcoef(iw[m], bw[m])[0, 1] if np.nanstd(iw[m]) > 0 else np.nan
    slope_within = np.polyfit(bw[m], iw[m], 1)[0] if np.nanstd(iw[m]) > 0 else np.nan
    # white-noise floor from consecutive far-clear differences
    dm = d[m]
    noise = np.nanstd(np.diff(dm)) / np.sqrt(2)
    # estimator's own near-cloud anomaly (same local reference as xco2_bc_anomaly: same fp, ±1°, cld > 10 km, QF0)
    anom = lat_local_contrast(x, cld, qf, meta["lat"], fp, False)
    rec = dict(fit=lab, estimator=name, n_far=int(m.sum()),
               far_bias_ppm=np.nanmedian(d[m]), far_bias_vs_prior_ppm=np.nanmedian(inc[m]),
               far_sd_ppm=np.nanstd(d[m]), far_rsd_ppm=robust_sd(d[m]), far_rsd_fpdemeaned_ppm=robust_sd(dd[m]),
               far_within2ppm=np.mean(np.abs(d[m]) < 2), far_corr_increment=r_inc, far_slope_increment=slope,
               far_lag1=lag1(d[m]), far_corr_within=r_within, far_slope_within=slope_within, noise_floor_ppm=noise)
    for lo, hi in ((0, 1), (1, 3), (3, 10)):
        s = (cld >= lo) & (cld < hi) & np.isfinite(anom) & np.isfinite(meta["xco2_bc_anomaly"])
        rec[f"near{lo}_{hi}_n"] = int(s.sum())
        rec[f"near{lo}_{hi}_est_anom_ppm"] = np.nanmedian(anom[s]) if s.sum() > 10 else np.nan
        rec[f"near{lo}_{hi}_bc_anom_ppm"] = np.nanmedian(meta["xco2_bc_anomaly"][s]) if s.sum() > 10 else np.nan
        rec[f"near{lo}_{hi}_est_anom_rsd"] = robust_sd(anom[s]) if s.sum() > 10 else np.nan
        rec[f"near{lo}_{hi}_corr_anom"] = np.corrcoef(anom[s], meta["xco2_bc_anomaly"][s])[0, 1] if s.sum() > 10 else np.nan
    rows.append(rec)
res = pd.DataFrame(rows)
res.to_csv(OUT / f"pseudo_xco2_{TAG}.csv", index=False)

# real variation of the target itself, for scale
ref = dict(bc_far_sd=np.nanstd(bc[far]), bc_far_rsd=robust_sd(bc[far]), inc_far_sd=np.nanstd(inc_bc[far]),
           inc_far_median=np.nanmedian(inc_bc[far]), pri_far_sd=np.nanstd(pri[far]),
           xco2_unc_median=np.nan)
with h5py.File(DEFAULTS["fit_details"]) as f:
    ref["xco2_unc_median"] = float(np.nanmedian(f["xco2_uncertainty"][...][far]))

# per-footprint and per-cluster means of the increment: does the estimator see the latitude step?
lat = meta["lat"]
clusters = pd.cut(lat[far], bins=[-90, -40, 0, 90], labels=["S of 40S", "40S-0", "N of 0"])
clus_rows = []
for (lab, name), x in est.items():
    inc = (x - pri)[far]
    for c in clusters.categories:
        s = np.asarray(clusters == c) & np.isfinite(inc)
        clus_rows.append(dict(fit=lab, estimator=name, cluster=c, n=int(s.sum()),
                              est_inc_ppm=np.nanmedian(inc[s]), bc_inc_ppm=np.nanmedian(inc_bc[far][s]),
                              bc_ppm=np.nanmedian(bc[far][s])))
clus = pd.DataFrame(clus_rows)
clus.to_csv(OUT / f"pseudo_xco2_clusters_{TAG}.csv", index=False)

fmt = lambda v: "—" if not np.isfinite(v) else f"{v:.2f}"
lines = [f"# Pseudo-XCO2 estimate on {TAG} (far-clear n = {int(far.sum())})", "",
         f"Target xco2_bc far-clear SD {ref['bc_far_sd']:.2f} ppm (robust {ref['bc_far_rsd']:.2f}); retrieved increment "
         f"bc − prior: median {ref['inc_far_median']:.2f} ppm, SD {ref['inc_far_sd']:.2f}; prior SD {ref['pri_far_sd']:.2f}; "
         f"L2 xco2_uncertainty median {ref['xco2_unc_median']:.2f} ppm.", "",
         "No estimator is scaled to xco2_bc or to the prior (except E2, calibrated to the prior by construction); "
         "bias = far-clear median of est − bc, bias/prior = median of est − prior.", "",
         "Far-clear columns: SD and robust SD of est − bc (ppm); fp-demeaned = robust SD after removing per-footprint "
         "means; corr/slope(inc) = correlation/slope of (est − prior) against (bc − prior); within = same after removing "
         "per-footprint-per-cluster means (latitude step removed); noise = SD of consecutive differences / √2.", "",
         "| fit | estimator | bias | bias/prior | far SD | robust SD | fp-demeaned | |d|<2 ppm | corr(inc) | slope(inc) | corr within | slope within | lag-1 | noise |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in res.itertuples():
    lines.append(f"| {r.fit} | {r.estimator} | {r.far_bias_ppm:+.2f} | {r.far_bias_vs_prior_ppm:+.2f} | {fmt(r.far_sd_ppm)} | {fmt(r.far_rsd_ppm)} | {fmt(r.far_rsd_fpdemeaned_ppm)} | "
                 f"{100 * r.far_within2ppm:.0f} % | {fmt(r.far_corr_increment)} | {fmt(r.far_slope_increment)} | "
                 f"{fmt(r.far_corr_within)} | {fmt(r.far_slope_within)} | {fmt(r.far_lag1)} | {fmt(r.noise_floor_ppm)} |")
lines += ["", "## Near-cloud anomaly (same local reference as xco2_bc_anomaly), medians in ppm", "",
          "| fit | estimator | 0–1 km est / bc anom (rSD, corr, n) | 1–3 km | 3–10 km |", "|---|---|---|---|---|"]
for r in res.itertuples():
    cells = [f"{fmt(getattr(r, f'near{lo}_{hi}_est_anom_ppm'))} / {fmt(getattr(r, f'near{lo}_{hi}_bc_anom_ppm'))} "
             f"({fmt(getattr(r, f'near{lo}_{hi}_est_anom_rsd'))}, {fmt(getattr(r, f'near{lo}_{hi}_corr_anom'))}, "
             f"n={getattr(r, f'near{lo}_{hi}_n')})" for lo, hi in ((0, 1), (1, 3), (3, 10))]
    lines.append(f"| {r.fit} | {r.estimator} | " + " | ".join(cells) + " |")
lines += ["", "## Latitude clusters, far-clear medians of (est − prior) vs (bc − prior), ppm", "",
          "| fit | estimator | " + " | ".join(clusters.categories) + " |", "|---|---|" + "---|" * len(clusters.categories)]
for (lab, name), g in clus.groupby(["fit", "estimator"], sort=False):
    cells = [f"{r.est_inc_ppm:+.2f} (bc {r.bc_inc_ppm:+.2f}, n={r.n})" for r in g.itertuples()]
    lines.append(f"| {lab} | {name} | " + " | ".join(cells) + " |")
# common-mode check: within-cluster correlation of the E1 increment and of the L2 increment with scene variables
cl = np.digitize(lat, [-40, 0])


def within(v):
    v = v.copy()
    for key in set(zip(fp[far], cl[far])):
        s_ = far & (fp == key[0]) & (cl == key[1])
        v[s_] -= np.nanmean(v[s_])
    return v


inc_e1 = within(est[("agreed", "E1 SCO2/O2A")] - pri)
targets = {"bc − prior": within(bc - pri), "raw − prior": within(meta["xco2_raw"] - pri),
           "strong_idp − prior": within(meta["xco2_strong_idp"] - pri)}
lines += ["", "## Common-mode check (agreed E1, far-clear, within footprint and latitude cluster)", "",
          "| variable | corr with E1 − prior | corr with bc − prior |", "|---|---|---|"]
for k in ("airmass", "aod_total", "h2o_scale", "alb_sco2", "ws", "psurf_met", "s31"):
    vw = within(meta[k])
    ok = far & np.isfinite(vw) & np.isfinite(inc_e1)
    lines.append(f"| {k} | {np.corrcoef(inc_e1[ok], vw[ok])[0, 1]:+.2f} | {np.corrcoef(targets['bc − prior'][ok], vw[ok])[0, 1]:+.2f} |")
lines += ["", "| target | corr(E1 − prior, target) | slope | SD target (ppm) |", "|---|---|---|---|"]
for k, tw in targets.items():
    ok = far & np.isfinite(tw) & np.isfinite(inc_e1)
    lines.append(f"| {k} | {np.corrcoef(inc_e1[ok], tw[ok])[0, 1]:+.2f} | {np.polyfit(tw[ok], inc_e1[ok], 1)[0]:+.2f} | {np.nanstd(tw[ok]):.2f} |")
lines += ["", "Hemispheric step (N of 0 vs S of 40S, far-clear medians): "
          + ", ".join(f"{b} k1 {100 * (np.nanmedian(fits['agreed'][b][far & (cl == 2)]) / np.nanmedian(fits['agreed'][b][far & (cl == 0)]) - 1):+.2f} %" for b in BANDS)
          + f"; xco2_bc {100 * (np.nanmedian(bc[far & (cl == 2)]) / np.nanmedian(bc[far & (cl == 0)]) - 1):+.2f} %, "
          + f"prior {100 * (np.nanmedian(pri[far & (cl == 2)]) / np.nanmedian(pri[far & (cl == 0)]) - 1):+.2f} %.",
          "1 ppm = 0.245 % of the k1 ratio at 408 ppm."]
(OUT / f"pseudo_xco2_{TAG}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- figure: along-track far-clear xco2_bc vs estimates (per-footprint-mean removed? no: raw, block medians)
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
idx = np.where(far)[0]
blk = np.floor(np.arange(idx.size) / 24).astype(int)
def blkmed(v):
    return pd.Series(v[idx]).groupby(blk).median().values
bl_lat = blkmed(lat)
axes[0].plot(bl_lat, blkmed(bc), "k.-", label="xco2_bc")
axes[0].plot(bl_lat, blkmed(pri), "k--", lw=0.8, label="prior")
for (lab, name), ls in ((("agreed", "E1 SCO2/O2A"), "C0.-"), (("agreed", "E2 ridge(held-out)"), "C1.-"),
                        (("two-tau T2", "E1 SCO2/O2A"), "C2.-"), (("L2 IDP", "xco2_strong_idp"), "C3.-")):
    axes[0].plot(bl_lat, blkmed(est[(lab, name)]), ls, ms=3, lw=0.8, label=f"{lab}: {name}")
axes[0].set_ylabel("XCO2 (ppm), block-24 medians"); axes[0].legend(fontsize=7, ncol=2)
for (lab, name), ls in ((("agreed", "E1 SCO2/O2A"), "C0"), (("agreed", "E2 ridge(held-out)"), "C1"),
                        (("production", "E1 SCO2/O2A"), "C4")):
    d = est[(lab, name)] - bc
    axes[1].plot(lat[idx], d[idx], ".", color=ls, ms=2, alpha=0.5, label=f"{lab}: {name}")
axes[1].set_ylim(-15, 15); axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_xlabel("latitude (deg)"); axes[1].set_ylabel("est − xco2_bc (ppm), per sounding"); axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / f"pseudo_xco2_{TAG}.png", dpi=130)

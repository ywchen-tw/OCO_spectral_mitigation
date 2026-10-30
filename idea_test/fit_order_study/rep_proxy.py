"""Validate the cloud-distance and anomaly proxies against the MODIS truth on 29252a_GL.

Label-level checks only (no fits): rank correlation, far-clear / near-cloud
confusion against the true ``cld_dist_km``, and the anomaly proxy against the
stored production anomaly.  ``choose_proxy`` applies the rule written in
PLAN.md before the fits (bar P0a).  Fit-level transfer (P0b, P0c) is done in
``rep_run.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rep_orbit_io import (_prod_anomaly_args, cloud_proxy_km, compute_xco2_anomaly,  # noqa: E402
                          fit_details_meta, l1b_geolocation)

FAR_TRUE_KM = 20.0
NEAR_KM = 3.0
T_CANDIDATES = (20.0, 30.0, 40.0, 50.0)
PRECISION_BAR = 0.80
NEAR_PRECISION_BAR = 0.70
MIN_FAR_N = 300


def _prf(pred, truth):
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    jac = tp / max(tp + fp + fn, 1)
    return dict(n_pred=tp + fp, n_true=tp + fn, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, jaccard=jac)


def proxy_distances(l1b_file: str, fp_tau_file: str, fit_details_file: str) -> tuple[dict, dict]:
    with h5py.File(fp_tau_file, "r") as f:
        sid = f["sounding_id"][...].astype(np.int64)
    geo = l1b_geolocation(l1b_file)
    meta = fit_details_meta(fit_details_file, geo, sid)
    prox = {"A": cloud_proxy_km(geo, sid, sid),
            "B": cloud_proxy_km(geo, sid, sid, meta["lite_qf1_sid"])}
    return meta, prox


def label_validation(meta: dict, prox: dict) -> pd.DataFrame:
    true = meta["cld_dist_true_km"]
    qf0 = meta["xco2_qf"] == 0
    rows = []
    for name, d in prox.items():
        rho = spearmanr(d, true).correlation
        r_log = np.corrcoef(np.log(d + 0.1), np.log(true + 0.1))[0, 1]
        base = dict(proxy=name, spearman=rho, pearson_log=r_log)
        for t in T_CANDIDATES:
            rec = _prf((d > t) & qf0, (true > FAR_TRUE_KM) & qf0)
            rec.update(base, kind="far_qf0", proxy_threshold_km=t,
                       frac_true_gt10=float(np.mean(true[(d > t) & qf0] > 10.0)))
            rows.append(rec)
        rec = _prf(d < NEAR_KM, true < NEAR_KM)
        rec.update(base, kind="near_all", proxy_threshold_km=NEAR_KM,
                   frac_true_gt10=float(np.mean(true[d < NEAR_KM] > 10.0)))
        rows.append(rec)
        rec = _prf((d > 10.0) & qf0, (true > 10.0) & qf0)       # local-reference set
        rec.update(base, kind="localref_qf0", proxy_threshold_km=10.0,
                   frac_true_gt10=float(np.mean(true[(d > 10.0) & qf0] > 10.0)))
        rows.append(rec)
    return pd.DataFrame(rows)


def choose_proxy(lab: pd.DataFrame) -> dict:
    """Pre-registered rule (PLAN.md, P0a)."""
    far = lab[lab.kind == "far_qf0"].set_index(["proxy", "proxy_threshold_km"])
    pa, pb = far.loc[("A", 20.0), "precision"], far.loc[("B", 20.0), "precision"]
    proxy = "B" if pb > pa else "A"
    choice = dict(proxy=proxy, far_km=None, p0a_pass_at_20=bool(far.loc[(proxy, 20.0), "precision"] >= PRECISION_BAR))
    for t in T_CANDIDATES:
        r = far.loc[(proxy, t)]
        if r.precision >= PRECISION_BAR and r.n_pred >= MIN_FAR_N:
            choice["far_km"] = t
            break
    choice["far_label"] = "far-clear" if choice["far_km"] is not None else "proxy-far (contaminated)"
    if choice["far_km"] is None:
        choice["far_km"] = 20.0
    near = lab[(lab.kind == "near_all") & (lab.proxy == proxy)].iloc[0]
    choice["near_precision"] = float(near.precision)
    choice["p0b_label_pass"] = bool(near.precision >= NEAR_PRECISION_BAR)
    return choice


def anomaly_validation(meta: dict, prox_d: np.ndarray) -> pd.DataFrame:
    """Anomaly code path with the true distance (reproduction) and with the proxy."""
    args = _prod_anomaly_args()
    stored = meta["xco2_bc_anomaly_true"]
    re_true = compute_xco2_anomaly(meta["lat"], meta["cld_dist_true_km"], meta["xco2_bc"], **args)
    re_prox = compute_xco2_anomaly(meta["lat"], prox_d, meta["xco2_bc"], **args)
    rows = []
    for name, a in (("recomputed_true_dist", re_true), ("proxy_dist", re_prox)):
        both = np.isfinite(a) & np.isfinite(stored)
        rows.append(dict(anomaly=name, frac_finite=float(np.isfinite(a).mean()),
                         frac_finite_stored=float(np.isfinite(stored).mean()),
                         n_both=int(both.sum()),
                         corr_with_stored=float(np.corrcoef(a[both], stored[both])[0, 1]),
                         median_abs_diff_ppm=float(np.median(np.abs(a[both] - stored[both]))),
                         max_abs_diff_ppm=float(np.max(np.abs(a[both] - stored[both])))))
    return pd.DataFrame(rows), re_prox


def scatter_figure(meta, prox_d, anom_prox, path: Path, far_km: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true = meta["cld_dist_true_km"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.scatter(true, prox_d, s=2, alpha=0.3, color="C0", rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(FAR_TRUE_KM, color="k", lw=0.8, ls="--")
    ax.axhline(far_km, color="C3", lw=0.8, ls="--", label=f"proxy far > {far_km:g} km")
    ax.axvline(NEAR_KM, color="0.5", lw=0.8, ls=":")
    ax.axhline(NEAR_KM, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("true cld_dist (MODIS) [km]"); ax.set_ylabel("proxy: distance to non-Lite sounding [km]")
    ax.set_title("29252a_GL cloud-distance proxy"); ax.legend(fontsize=8)
    ax = axes[1]
    a = meta["xco2_bc_anomaly_true"]
    ok = np.isfinite(a) & np.isfinite(anom_prox)
    ax.scatter(a[ok], anom_prox[ok], s=2, alpha=0.3, color="C1", rasterized=True)
    lim = np.nanpercentile(np.abs(np.r_[a[ok], anom_prox[ok]]), 99.5)
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("production xco2_bc_anomaly [ppm]"); ax.set_ylabel("proxy anomaly [ppm]")
    ax.set_title(f"r = {np.corrcoef(a[ok], anom_prox[ok])[0, 1]:.2f}, n = {ok.sum()}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

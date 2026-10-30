"""Conditional fitters for the residual decomposition (decision table in PLAN.md).

Unit-weight cumulant fit (k1, k2 ≥ 0 as production, BVLS fallback) with extra,
unbounded per-sounding regressors built from ``resid_attrs``:

- ``eta``  : − β·τ·η            (vertical weighting; decision-table action for VERT)
- ``h2o``  : − Δ·τ_H2O          (two-τ fit: k1 = k1_dry, k1_w = k1 + Δ; action for H2O)
- ``disp`` : + a·∂τ/∂pix        (wavelength-registration shift; a = −k1·δ)
- ``ilsw`` : + b·∂τ/∂ln w       (ILS width)
- ``dT``   : + c·∂τ/∂T          (temperature / line-strength)

Each is run at order 3 and 2 on every sounding; metrics as ``results/summary.md``
plus the latitude-local near-cloud SNR of ``weight_extra.py``.

Usage:
    python idea_test/fit_order_study/resid_fitters.py [--workers 8]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orbit_io import BANDS  # noqa: E402
from resid_attrs import ORBIT  # noqa: E402
from resid_decompose import ATTR_FILE, load_inputs  # noqa: E402
from run_study import FAR_KM, NEAR_KM, md_table, robust_sd, spread_pct  # noqa: E402
from run_weighting import LOCAL_REF_KM, lat_local_ref  # noqa: E402
from variants import design_matrix  # noqa: E402

OUT = HERE / "results"
EXTRA_SETS = {"": (), "eta": ("eta",), "h2o": ("h2o",), "disp": ("disp",), "ilsw": ("ilsw",),
              "dT": ("dT",), "disp_ilsw": ("disp", "ilsw"), "disp_dT": ("disp", "dT"),
              "inst": ("disp", "ilsw", "dT"), "all": ("eta", "h2o", "disp", "ilsw", "dT")}
ORDERS = (3, 2)
COLS = ("k1", "k2", "intercept", "resid_rms", "lag1", "bound", "nchan",
        "x_eta", "x_h2o", "x_disp", "x_ilsw", "x_dT")
_G: dict = {}


def tag_of(order, key):
    return f"ols_{order}" + (f"_{key}" if key else "")


def fit_one(t, y, order, extra: dict):
    """Bounded unit-weight fit with unbounded extra columns. Returns dict of COLS."""
    a = design_matrix(t, order)
    names = list(extra)
    if names:
        a = np.column_stack([a, *[extra[k] for k in names]])
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    n_pos = min(2, order)
    bound = bool(np.any(coef[1:1 + n_pos] < 0))
    if bound:
        lb = np.full(a.shape[1], -np.inf)
        lb[1:1 + n_pos] = 0.0
        coef = lsq_linear(a, y, bounds=(lb, np.full(a.shape[1], np.inf)), method="bvls").x
    r = y - a @ coef
    den = np.sqrt((r[:-1] @ r[:-1]) * (r[1:] @ r[1:]))
    out = dict(k1=coef[1], k2=coef[2] if order >= 2 else np.nan, intercept=coef[0],
               resid_rms=float(np.sqrt(r @ r / len(r))), lag1=float(r[:-1] @ r[1:] / den) if den > 0 else np.nan,
               bound=bound, nchan=len(r))
    for k in ("eta", "h2o", "disp", "ilsw", "dT"):
        out[f"x_{k}"] = coef[order + 1 + names.index(k)] if k in names else np.nan
    return out


def _rows(j_list):
    tau, lnt, attrs = _G["tau"], _G["lnt"], _G["attrs"]
    rows = []
    for j in j_list:
        for ib, band in enumerate(BANDS):
            a = attrs[band]
            t, y = tau[ib][j], lnt[ib][j]
            m = np.isfinite(t) & np.isfinite(y)
            for k in ("eta", "f_h2o", "dtau_dpix", "dtau_dlnw", "dtau_dT"):
                m &= np.isfinite(a[k][j])
            if m.sum() < 20:
                continue
            t, y = t[m].astype(float), y[m].astype(float)
            cols = {"eta": -t * a["eta"][j][m], "h2o": -t * a["f_h2o"][j][m],
                    "disp": a["dtau_dpix"][j][m].astype(float), "ilsw": a["dtau_dlnw"][j][m].astype(float),
                    "dT": a["dtau_dT"][j][m].astype(float)}
            for order in ORDERS:
                for key, ext in EXTRA_SETS.items():
                    res = fit_one(t, y, order, {k: cols[k] for k in ext})
                    rows.append((j, band, tag_of(order, key), *[res[c] for c in COLS]))
    return rows


def local_snr(x, meta, fp):
    """Latitude-local near-cloud shift / far-clear local SD (weight_extra.py definition)."""
    cld, qf, lat = meta["cld_dist_km"], meta["xco2_qf"], meta["lat"]
    far = (cld > FAR_KM) & (qf == 0)
    near = cld < NEAR_KM
    ref = (cld > LOCAL_REF_KM) & (qf == 0)
    fpm = pd.Series(x[far]).groupby(fp[far]).mean()
    rel = (x - fpm.reindex(fp).values) / np.nanmedian(x[far])
    loc = rel - lat_local_ref(rel, lat, ref)
    sd = robust_sd(loc[far])
    return float(np.nanmedian(loc[near]) / sd), 100 * sd


def aggregate(long, meta, fp):
    cld, qf = meta["cld_dist_km"], meta["xco2_qf"]
    far = (cld > FAR_KM) & (qf == 0)
    near = cld < NEAR_KM
    anom = meta["xco2_bc_anomaly"]
    n = len(cld)
    recs = []
    k1_o2a = {}
    for tag in long.variant.unique():
        d = long[(long.band == "o2a") & (long.variant == tag)].set_index("j").reindex(np.arange(n))
        k1_o2a[tag] = d.k1.values
    for band in BANDS:
        for tag in long.variant.unique():
            d = long[(long.band == band) & (long.variant == tag)].set_index("j").reindex(np.arange(n))
            k1, k2 = d.k1.values, d.k2.values
            m = far & np.isfinite(k1)
            rec = dict(band=band, variant=tag, n_far=int(m.sum()),
                       k1_median=np.nanmedian(k1[m]), k1_spread_pct=spread_pct(k1[m], fp[m]),
                       k2_median=np.nanmedian(k2[m]),
                       k2_zero_frac=float(np.nanmean(k2[m] <= 1e-6)) if np.isfinite(k2[m]).any() else np.nan,
                       k2_spread_pct=spread_pct(k2[m], fp[m]) if np.nanmean(k2[m] <= 1e-6) < 0.5 else np.nan,
                       resid_rms_median=np.nanmedian(d.resid_rms.values[m]),
                       lag1_median=np.nanmedian(d.lag1.values[m]),
                       ratio_to_o2a_spread_pct=np.nan, k1_near_corr_anom=np.nan)
            for k in ("eta", "h2o", "disp", "ilsw", "dT"):
                rec[f"x_{k}_median"] = np.nanmedian(d[f"x_{k}"].values[m])
            if band != "o2a":
                r = k1 / k1_o2a[tag]
                mr = m & np.isfinite(r)
                rec["ratio_to_o2a_spread_pct"] = spread_pct(r[mr], fp[mr])
            mn = near & np.isfinite(k1) & np.isfinite(anom)
            if mn.sum() > 30:
                rec["k1_near_corr_anom"] = float(np.corrcoef(k1[mn], anom[mn])[0, 1])
            rec["snr_k1_local"], rec["sd_k1_local_pct"] = local_snr(k1, {**dict(cld_dist_km=cld, xco2_qf=qf, lat=meta["lat"])}, fp)
            if np.nanmean(k2[m] <= 1e-6) < 0.5:
                rec["snr_k2_local"], _ = local_snr(k2, dict(cld_dist_km=cld, xco2_qf=qf, lat=meta["lat"]), fp)
            else:
                rec["snr_k2_local"] = np.nan
            recs.append(rec)
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--attrs", default=str(ATTR_FILE))
    ap.add_argument("--tau", default=None, help="npz (j, tau [n, 3, 1016]) overriding τ (Doppler-shifted)")
    ap.add_argument("--tag", default="", help="suffix for output names and the summary section")
    args = ap.parse_args()
    orb, attrs, v_inst, extra, _, jj = load_inputs(args.attrs)
    if args.tau:
        zt = np.load(args.tau)
        orb.tau[:, zt["j"], :] = np.moveaxis(zt["tau"], 1, 0)
        jj = np.intersect1d(jj, zt["j"])
    sfx = f"_{args.tag}" if args.tag else ""
    _G.update(tau=orb.tau[:, :, 1:-1], lnt=orb.ln_t[:, :, 1:-1], attrs=attrs)
    chunks = np.array_split(jj, max(1, len(jj) // 100))
    with mp.get_context("fork").Pool(args.workers) as pool:
        parts = pool.map(_rows, chunks)
    long = pd.DataFrame([r for p in parts for r in p], columns=["j", "band", "variant", *COLS])
    long.to_parquet(OUT / f"resid_fitters{sfx}_per_sounding_{ORBIT}.parquet", index=False)
    s = aggregate(long, orb.meta, orb.fp_number)
    s.to_csv(OUT / f"resid_fitters{sfx}_{ORBIT}.csv", index=False)

    cols = ["variant", "k1_median", "k1_spread_pct", "ratio_to_o2a_spread_pct", "k2_median", "k2_zero_frac",
            "k2_spread_pct", "resid_rms_median", "lag1_median", "k1_near_corr_anom", "snr_k1_local",
            "sd_k1_local_pct", "snr_k2_local"]
    head = f"\n## Conditional fitters{(' — ' + args.tag) if args.tag else ''} (resid_fitters.py)\n"
    parts = [head,
             "Unit weights, k1, k2 ≥ 0 (BVLS), extra columns unbounded; far-clear n as `summary.md`. "
             "Spreads: robust SD (%) after per-fp mean removal. snr_*_local: latitude-local near-cloud "
             "(< 3 km) shift / far-clear local SD (`weight_extra.py`).\n"]
    for band in BANDS:
        parts.append(f"\n### {band}\n")
        parts.append(md_table(s[s.band == band], cols))
    xcols = ["band", "variant", "x_eta_median", "x_h2o_median", "x_disp_median", "x_ilsw_median", "x_dT_median"]
    parts.append("\n### Extra-coefficient medians (far-clear)\n")
    parts.append(md_table(s[s.variant.str.count("_") >= 2], xcols))
    summ = OUT / "resid_summary.md"
    text = summ.read_text() if summ.exists() else ""
    if head in text:                                    # replace this section only
        pre, post = text.split(head, 1)
        nxt = post.find("\n## ")
        text = pre + (post[nxt:] if nxt >= 0 else "")
    summ.write_text(text.rstrip("\n") + "\n" + "\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()

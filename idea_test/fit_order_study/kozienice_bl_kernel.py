"""Vertical sensitivity of the agreed-model k1 to a boundary-layer CO2 enhancement (Kozienice orbit).

Question: the agreed model sees the Kozienice plume at 1.7× the L2 amplitude.  Does k1 over-respond to
CO2 added in the lowest layers (pressure-broadened wings) compared with the same column amount spread
over the whole column?  For a sounding with high-resolution CO2 extinction e_gas[nu, layer]:

  d_col(channel) = ILS/transmission-weighted τ contribution of all CO2 layers
  d_bl(channel)  = the same for layers below H_BL km, scaled so that the added column amount equals
                   that of a uniform relative increase ε of the whole column

The fit is linear in τ, so Δln T = −k1·ε·d for either perturbation; regressing Δln T on the agreed
SCO2/WCO2 design ([1, x, (x²), −τ_dry, −τ_h2o, ½τ²]) gives Δk1.  The ratio Δk1_bl / Δk1_col is the
boundary-layer "averaging kernel" of the k1 estimator (1 = column-mean response).  Uses the same channel
mask as the fit.  Outputs results/kozienice_bl_kernel.{csv,md}.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import resid_attrs as ra  # noqa: E402
import kozienice_pseudo_xco2 as kz  # noqa: E402
from final_model_test import X_ALL  # noqa: E402
from orbit_io import BANDS  # noqa: E402

OUT = HERE / "results"
H_BL = 1.0
EPS = 0.01
C_LIGHT = 299792458.0


def layer_contributions(i: int):
    """Per channel [band, 1016]: τ_dry, τ_h2o, τ total, d_col, d_bl (CO2 bands only), BL column fraction."""
    atm, bands = ra._G["atm"], ra._G["bands"]
    fp = int(ra._G["fp"][i])
    airmass = 1 / np.cos(np.deg2rad(ra._G["sza"][i])) + 1 / np.cos(np.deg2rad(ra._G["vza"][i]))
    beta = ra._G["sign"] * atm["v_inst"][i] / C_LIGHT
    u, wq = np.polynomial.legendre.leggauss(ra.N_QUAD)
    u = 0.5 * (u + 1)
    s_nodes, s_w = u**2, wq * u
    dz, h = atm["dz"][i], atm["h_mid"][i]
    bl = (h - h[0]) < H_BL
    co2_col = (atm["co2"][i] * dz)
    frac_bl = co2_col[bl].sum() / co2_col.sum()
    out = {k: np.full((3, 1016), np.nan) for k in ("tau_dry", "tau_h2o", "d_col", "d_bl")}
    for ib, b in enumerate(bands):
        if ib == 0:
            continue
        e_gas, e_h2o = ra._ext_layers(b, atm, i, atm["co2"][i])
        ray = (b["ray_xsec"][:, None] * 1e-28 * atm["air"][i] * dz * 1e5).sum(axis=1)
        comp = np.stack([(e_gas * dz).sum(axis=1), (e_h2o * dz).sum(axis=1), ray,
                         (e_gas[:, bl] * dz[bl]).sum(axis=1) / frac_bl]) * airmass    # [4, nu]
        tau_hr = comp[:3].sum(axis=0)
        wl, xx, yy = b["wl"][fp], b["xx"][fp], b["yy"][fp]
        wv = b["wvldat"] * (1.0 + beta)
        start, end = ra.conv_indices(wl, xx, wv)
        xw = xx + wl[:, None]
        for c in range(len(wl)):
            s, e = start[c], end[c]
            w = np.interp(wv[s:e], xw[c], yy[c])
            th = tau_hr[s:e]
            ex = np.exp(-np.outer(s_nodes, th - th.min())) * w
            p_bar = s_w @ (ex / ex.sum(axis=1, keepdims=True))
            v = comp[:, s:e] @ p_bar
            out["tau_dry"][ib, c] = v[0] + v[2]
            out["tau_h2o"][ib, c] = v[1]
            out["d_col"][ib, c] = v[0]
            out["d_bl"][ib, c] = v[3]
    return out, frac_bl


def kernel_for(i, ln_t, sig, land, k1):
    lc, frac_bl = layer_contributions(i)
    rows = []
    for ib, b in enumerate(BANDS):
        if ib == 0:
            continue
        td, tw = lc["tau_dry"][ib], lc["tau_h2o"][ib]
        y, s = ln_t[ib, i].astype(float), sig[ib, i].astype(float)
        ok = np.isfinite(y) & np.isfinite(td) & np.isfinite(s) & (s > 0) & np.isfinite(tw)
        t = td + tw
        x = X_ALL
        cols = [np.ones(ok.sum()), x[ok]] + ([x[ok]**2] if (land and b == "sco2") else []) + [-td[ok], -tw[ok]]
        if b == "sco2":
            cols.append(0.5 * t[ok]**2)
        a = np.column_stack(cols)
        kcol = 3 if (land and b == "sco2") else 2
        dk = {}
        for name, d in (("col", lc["d_col"][ib]), ("bl", lc["d_bl"][ib])):
            dy = -k1[b] * EPS * d[ok]
            dk[name] = np.linalg.lstsq(a, dy, rcond=None)[0][kcol] / (k1[b] * EPS)
        rows.append(dict(j=i, band=b, frac_bl=frac_bl, dk1_col_per_eps=dk["col"], dk1_bl_per_eps=dk["bl"],
                         kernel_bl=dk["bl"] / dk["col"]))
    return rows


def _work(args):
    return kernel_for(*args)


def main():
    df = pd.read_parquet(OUT / f"{kz.TAG}_per_sounding.parquet")
    z = np.load(OUT / f"{kz.TAG}_spectra.npz", allow_pickle=True)
    ln_t, sig = z["ln_t"], z["sig"]
    sel = kz.select_soundings()
    for k in ("l1b", "met", "cpr"):
        setattr(ra, k.upper(), kz.FILES[k])
    bands, h2o_meta = ra.load_tables()
    atm = ra.build_atmosphere(sel["sid"])
    ra._G.update(bands=bands, h2o_meta=h2o_meta, atm=atm, fp=sel["fp"], sza=sel["sza"], vza=sel["vza"],
                 sign=kz.DOPPLER_SIGN)
    bg = np.flatnonzero((df.dist_km >= 40) & (df.qf == 0))
    rng = np.random.default_rng(0)
    bg_sel = rng.choice(bg, 24, replace=False)
    bgm = df.iloc[bg].xco2_bc.median()
    plume = np.flatnonzero((df.xco2_bc - bgm > 2) & (df.dist_km < 30))
    jobs = [(int(i), ln_t, sig, bool(df.land[i]), {b: float(df[f"k1_{b}"][i]) for b in BANDS})
            for i in np.r_[bg_sel, plume]]
    with mp.get_context("fork").Pool(8) as pool:
        parts = pool.map(_work, jobs)
    res = pd.DataFrame([r for p in parts for r in p])
    res["group"] = np.where(res.j.isin(plume), "plume", "background")
    res.to_csv(OUT / "kozienice_bl_kernel.csv", index=False)
    lines = [f"# k1 response to a {H_BL:.0f}-km boundary-layer CO2 enhancement vs a uniform column increase",
             "", f"Same added column amount; Δk1 per unit relative column increase (1 = column-mean response); "
             f"BL holds {100 * res.frac_bl.median():.1f} % of the CO2 column (median).", "",
             "| group | band | n | Δk1/ε column | Δk1/ε boundary layer | BL kernel (median [p10, p90]) |", "|---|---|---|---|---|---|"]
    for (g, b), d in res.groupby(["group", "band"]):
        q = np.percentile(d.kernel_bl, [10, 90])
        lines.append(f"| {g} | {b} | {len(d)} | {d.dk1_col_per_eps.median():.3f} | {d.dk1_bl_per_eps.median():.3f} | "
                     f"{d.kernel_bl.median():.2f} [{q[0]:.2f}, {q[1]:.2f}] |")
    (OUT / "kozienice_bl_kernel.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Extra test: recompute τ with the telluric Doppler shift applied, then refit.

The residual decomposition attributes 82–94 % of the structured misfit to a
wavelength-registration signature whose per-sounding amplitude tracks
``sounding_relative_velocity`` (r ≥ 0.995). Production computes τ on the
unshifted ABSCO grid (``wvldat_obs = wvldat  # * (1.0 + v_inst / C_LIGHT)`` in
``src/abs_util/fp_abs_coeff.py``). Here the ABSCO wavelength grid is shifted,
λ_obs = λ·(1 + sign·v_inst/c), and τ is recomputed with the same transmission
convolution (``resid_attrs`` copy of the production step). A linear regressor
captures only the first-order shift; this test also covers the second order,
which can bend ln T against τ (the open k2-pinning question).

Usage:
    python idea_test/fit_order_study/resid_doppler.py --sign-test   # pick the sign on 24 soundings
    python idea_test/fit_order_study/resid_doppler.py --sign -1     # all soundings, refit
    python idea_test/fit_order_study/resid_doppler.py --sign -1 --save-far  # τ for resid_decompose --tau
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import resid_attrs as ra  # noqa: E402
from orbit_io import BANDS, load_orbit  # noqa: E402
from resid_fitters import aggregate, fit_one  # noqa: E402
from run_study import FAR_KM, md_table  # noqa: E402

OUT = HERE / "results"
C_LIGHT = 299792458.0
_G = ra._G   # _ext_layers reads resid_attrs._G


def tau_shifted(i: int) -> np.ndarray:
    """τ [3, 1016] for sounding position i with the ABSCO grid Doppler-shifted."""
    atm, bands = _G["atm"], _G["bands"]
    fp = int(_G["fp"][i])
    airmass = 1 / np.cos(np.deg2rad(_G["sza"][i])) + 1 / np.cos(np.deg2rad(_G["vza"][i]))
    beta = _G["sign"] * atm["v_inst"][i] / C_LIGHT
    dz = atm["dz"][i]
    out = np.empty((3, 1016))
    for ib, b in enumerate(bands):
        gas_den = atm["o2"][i] if ib == 0 else atm["co2"][i]
        e_gas, e_h2o = ra._ext_layers(b, atm, i, gas_den)
        ray = (b["ray_xsec"][:, None] * 1e-28 * atm["air"][i] * dz * 1e5).sum(axis=1)
        tau_hr = ((e_gas + e_h2o) * dz).sum(axis=1) * airmass + ray * airmass
        out[ib] = ra._tau_conv(tau_hr, b["wvldat"] * (1.0 + beta), b["wl"][fp], b["xx"][fp], b["yy"][fp])
    return out


def _work(idx):
    return [(i, tau_shifted(i)) for i in idx]


def compute(sel, sign, workers):
    with h5py.File(ra.FP_TAU, "r") as f:
        sid = np.round(f["sounding_id"][...]).astype(np.int64)
        fp = f["fp_number"][...].astype(int)
        sza, vza = f["sza"][...], f["vza"][...]
    bands, h2o_meta = ra.load_tables()
    atm = ra.build_atmosphere(sid[sel])
    _G.update(bands=bands, h2o_meta=h2o_meta, atm=atm, fp=fp[sel], sza=sza[sel], vza=vza[sel], sign=sign)
    chunks = np.array_split(np.arange(len(sel)), max(1, len(sel) // 4))
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        res = [r for part in pool.imap(_work, chunks) for r in part]
    print(f"τ for {len(sel)} soundings in {time.time() - t0:.0f} s", flush=True)
    res.sort(key=lambda x: x[0])
    return np.stack([r[1] for r in res], axis=1)            # [3, n, 1016]


def refit(orb, tau_new, sel):
    rows = []
    for k, j in enumerate(sel):
        for ib, band in enumerate(BANDS):
            for name, tt in (("prod_tau", orb.tau[ib, j]), ("doppler_tau", tau_new[ib, k])):
                t, y = tt[1:-1].astype(float), orb.ln_t[ib, j, 1:-1].astype(float)
                m = np.isfinite(t) & np.isfinite(y)
                for order in (3, 2):
                    r = fit_one(t[m], y[m], order, {})
                    rows.append((j, band, f"ols_{order}_{name}", r["k1"], r["k2"], r["intercept"],
                                 r["resid_rms"], r["lag1"], r["bound"], r["nchan"]))
    return pd.DataFrame(rows, columns=["j", "band", "variant", "k1", "k2", "intercept", "resid_rms",
                                       "lag1", "bound", "nchan"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign-test", action="store_true")
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-far", action="store_true", help="save far-clear Doppler τ only")
    args = ap.parse_args()
    orb = load_orbit(str(ra.L1B), str(ra.FP_TAU), str(ra.FIT_DETAILS))
    n = len(orb.sounding_id)
    far = np.flatnonzero((orb.meta["cld_dist_km"] > FAR_KM) & (orb.meta["xco2_qf"] == 0))
    if args.save_far:
        tau_far = compute(far, args.sign, args.workers)
        np.savez_compressed(OUT / f"resid_doppler_tau_far_{ra.ORBIT}.npz", j=far,
                            tau=np.moveaxis(tau_far, 0, 1).astype(np.float32), sign=args.sign)
        return
    if args.sign_test:
        sel = far[np.linspace(0, len(far) - 1, 24).astype(int)]
        for sign in (1.0, -1.0):
            long = refit(orb, compute(sel, sign, args.workers), sel)
            print(f"sign {sign:+.0f}:")
            print(long.groupby(["band", "variant"]).resid_rms.median().unstack().round(4))
        return
    sel = np.arange(n)
    tau_new = compute(sel, args.sign, args.workers)
    np.savez_compressed(OUT / f"resid_doppler_tau_{ra.ORBIT}.npz", j=sel,
                        tau=np.moveaxis(tau_new, 0, 1).astype(np.float32), sign=args.sign)
    long = refit(orb, tau_new, sel)
    for c in ("x_eta", "x_h2o", "x_disp", "x_ilsw", "x_dT"):
        long[c] = np.nan
    long.to_parquet(OUT / f"resid_doppler_per_sounding_{ra.ORBIT}.parquet", index=False)
    s = aggregate(long, orb.meta, orb.fp_number)
    s.to_csv(OUT / f"resid_doppler_{ra.ORBIT}.csv", index=False)
    cols = ["variant", "k1_median", "k1_spread_pct", "ratio_to_o2a_spread_pct", "k2_median", "k2_zero_frac",
            "k2_spread_pct", "resid_rms_median", "lag1_median", "k1_near_corr_anom", "snr_k1_local",
            "sd_k1_local_pct", "snr_k2_local"]
    parts = [f"\n## Doppler-shifted τ (resid_doppler.py, sign {args.sign:+.0f})\n",
             "ABSCO grid shifted by λ(1 + sign·v_inst/c) before the production transmission convolution; "
             "same fits and metrics as the conditional fitters.\n"]
    for band in BANDS:
        parts.append(f"\n### {band}\n")
        parts.append(md_table(s[s.band == band], cols))
    summ = OUT / "resid_summary.md"
    base = summ.read_text().split("\n## Doppler-shifted τ")[0]
    summ.write_text(base.rstrip("\n") + "\n" + "\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()

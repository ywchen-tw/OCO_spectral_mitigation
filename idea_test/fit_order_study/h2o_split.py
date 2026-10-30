"""Species split of the Doppler-corrected τ: dry gas / H2O / Rayleigh per channel (orbit 29252a_GL).

Same integrated-gradient attribution as ``resid_attrs.sounding_attrs``, but on the
Doppler-shifted ABSCO grid λ(1 + sign·v_inst/c) used by ``resid_doppler.tau_shifted``,
so the components sum to the stored corrected τ (``resid_doppler_tau_29252a_GL.npz``):

    τ_c = −ln⟨e^{−τ_hr}⟩_ILS,   C_x = ∫_0^1 ⟨A_x⟩_{p_s} ds,   p_s ∝ w e^{−s τ_hr},

with A_x the slant high-resolution optical depth of species x (dry gas = O2 for O2A,
CO2 for WCO2/SCO2; H2O; Rayleigh). The resid_attrs split was computed on the unshifted
grid; this one is consistent with the corrected τ.

Usage:
    python idea_test/fit_order_study/h2o_split.py --timing 100      # time 100 soundings
    python idea_test/fit_order_study/h2o_split.py --subset all      # all soundings
    python idea_test/fit_order_study/h2o_split.py --subset reduced  # near (<3 km) + far-clear + 2000 random
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# One BLAS thread per worker: with 8 forked workers, threaded BLAS oversubscribes the cores (~40× slower).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import resid_attrs as ra  # noqa: E402
from orbit_io import BANDS  # noqa: E402

OUT = HERE / "results"
TAU_CORR = OUT / f"resid_doppler_tau_{ra.ORBIT}.npz"
SPLIT_FILE = OUT / f"h2o_tau_split_{ra.ORBIT}.npz"
C_LIGHT = 299792458.0
N_QUAD = ra.N_QUAD
_G = ra._G   # ra._ext_layers reads resid_attrs._G


def split_one(i: int) -> np.ndarray:
    """[3 bands, 3 components (dry, h2o, ray), 1016] for sounding position i."""
    atm, bands = _G["atm"], _G["bands"]
    fp = int(_G["fp"][i])
    airmass = 1 / np.cos(np.deg2rad(_G["sza"][i])) + 1 / np.cos(np.deg2rad(_G["vza"][i]))
    beta = _G["sign"] * atm["v_inst"][i] / C_LIGHT
    u, wq = np.polynomial.legendre.leggauss(N_QUAD)
    u = 0.5 * (u + 1)
    s_nodes, s_w = u**2, wq * u                               # ∫ ds = ∫ 2u du, u ∈ [0, 1]
    dz = atm["dz"][i]
    out = np.full((3, 3, 1016), np.nan)
    for ib, b in enumerate(bands):
        gas_den = atm["o2"][i] if ib == 0 else atm["co2"][i]
        e_gas, e_h2o = ra._ext_layers(b, atm, i, gas_den)
        ray = (b["ray_xsec"][:, None] * 1e-28 * atm["air"][i] * dz * 1e5).sum(axis=1)
        comp = np.stack([(e_gas * dz).sum(axis=1), (e_h2o * dz).sum(axis=1), ray]) * airmass  # [3, nu]
        tau_hr = comp.sum(axis=0)
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
            out[ib, :, c] = comp[:, s:e] @ p_bar
    return out


def _work(idx):
    return [(i, split_one(i)) for i in idx]


def select(subset: str, n: int, seed: int = 20261029) -> np.ndarray:
    if subset == "all":
        return np.arange(n)
    with h5py.File(ra.FIT_DETAILS, "r") as f:
        cld, qf = f["cld_dist_km"][...], f["xco2_qf"][...]
    if subset == "reduced":
        must = (cld < 3) | ((cld > 20) & (qf == 0))
        rest = np.flatnonzero(~must)
        rng = np.random.default_rng(seed)
        extra = rng.choice(rest, size=min(2000, len(rest)), replace=False)
        return np.union1d(np.flatnonzero(must), extra)
    return np.unique(np.linspace(0, n - 1, int(subset)).astype(int))


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
    res = []
    with mp.get_context("fork").Pool(workers) as pool:
        for k, part in enumerate(pool.imap(_work, chunks)):
            res.extend(part)
            if k % 250 == 0:
                print(f"  {len(res)}/{len(sel)} soundings, {time.time() - t0:.0f} s", flush=True)
    dt = time.time() - t0
    print(f"split for {len(sel)} soundings in {dt:.0f} s ({dt / len(sel):.2f} s each, {workers} workers)",
          flush=True)
    res.sort(key=lambda x: x[0])
    return np.stack([r[1] for r in res]), dt                 # [n, 3 bands, 3 comp, 1016]


def validate(split, tau_ref) -> str:
    """Relative closure Σ components vs stored corrected τ, per band."""
    lines = []
    for ib, b in enumerate(BANDS):
        rel = np.abs(split[:, ib].sum(axis=1) / tau_ref[:, ib] - 1)
        lines.append(f"{b}: |Σ components/τ_corr − 1| median {np.nanmedian(rel):.1e}, "
                     f"p99 {np.nanpercentile(rel, 99):.1e}, max {np.nanmax(rel):.1e}; "
                     f"f_h2o median {np.nanmedian(split[:, ib, 1] / tau_ref[:, ib]):.2e}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="all", help="'all', 'reduced' or a number of evenly spaced soundings")
    ap.add_argument("--timing", type=int, default=0, help="only time this many soundings, no output file")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    zt = np.load(TAU_CORR)
    if not np.array_equal(zt["j"], np.arange(len(zt["j"]))):
        raise ValueError("corrected-τ file is not in full sounding order")
    sign = float(zt["sign"])
    n = len(zt["j"])
    sel = select(str(args.timing) if args.timing else args.subset, n)
    split, dt = compute(sel, sign, args.workers)
    tau_ref = zt["tau"][sel].astype(np.float64)
    val = validate(split, tau_ref)
    print(val)
    if args.timing:
        return
    if SPLIT_FILE.exists():
        raise FileExistsError(f"{SPLIT_FILE} exists; not overwriting")
    arrays = {}
    for ib, b in enumerate(BANDS):
        for ic, c in enumerate(("dry", "h2o", "ray")):
            arrays[f"tau_{c}_{b}"] = split[:, ib, ic].astype(np.float32)
    np.savez_compressed(SPLIT_FILE, j=sel, sign=sign, validation=val, seconds=dt, **arrays)
    print("saved", SPLIT_FILE)


if __name__ == "__main__":
    main()

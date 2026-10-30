"""Load one orbit's log-transmittance, optical depth, noise and metadata.

Reads the L1B file, the per-footprint τ file and the production fitting-details
file directly, so this side study does not import the ``spectral`` package
(whose ``__init__`` pulls in ``pyproj``).  The τ file and the fitting-details
file share the sounding order produced by ``spectral.orbit_data.load_orbit_data``.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

BANDS = ("o2a", "wco2", "sco2")
_L1B_RAD = {"o2a": "radiance_o2", "wco2": "radiance_weak_co2",
            "sco2": "radiance_strong_co2"}
# Stokes-I coefficient applied to the TOA solar irradiance in production.
_M1 = 0.5


@dataclass
class OrbitSpectra:
    sounding_id: np.ndarray        # [N] int64
    fp_number: np.ndarray          # [N] int, 0..7
    tau: np.ndarray                # [3, N, 1016] float32
    ln_t: np.ndarray               # [3, N, 1016] float32, NaN where T<=0 or T>1
    sigma_ln_t: np.ndarray         # [3, N, 1016] float32, noise/rad
    meta: dict                     # per-sounding arrays from fitting details


def l1b_noise(rad: np.ndarray, snr_coef: np.ndarray, max_ms: float) -> np.ndarray:
    """OCO-2 L1B noise model: sigma = (MaxMS/100)·sqrt(c_photon²·x + c_bg²), x = 100·rad/MaxMS.

    Verified against the per-sounding continuum SNR stored in the L1B file
    (``snr_*_l1b``): this form reproduces it to 10–20 %, the alternatives are
    off by an order of magnitude.
    """
    x = np.clip(100.0 * rad / max_ms, 0.0, None)
    c_photon = snr_coef[..., 0]
    c_bg = snr_coef[..., 1]
    return (max_ms / 100.0) * np.sqrt(c_photon**2 * x + c_bg**2)


def load_orbit(l1b_file: str, fp_tau_file: str, fit_details_file: str,
               max_soundings: int | None = None) -> OrbitSpectra:
    with h5py.File(fp_tau_file, "r") as f:
        sid = f["sounding_id"][...].astype(np.int64)
        fp = f["fp_number"][...].astype(int)
        tau = np.stack([f[f"{b}_tau_output"][...] for b in BANDS]).astype(np.float32)
        toa = np.stack([f[f"{b}_toa_sol_output"][...] for b in BANDS]).astype(np.float32) * _M1

    meta_keys = ("cld_dist_km", "sfc_type", "xco2_qf", "xco2_bc", "xco2_bc_anomaly",
                 "lat", "lon", "sza", "vza", "glint_angle", "fp_number")
    with h5py.File(fit_details_file, "r") as f:
        meta = {k: f[k][...] for k in meta_keys if k in f}
        ref = {f"{b}_{k}": f[f"{b}_{k}_fitting_nosg"][...]
               for b in BANDS for k in ("k1", "k2", "intercept")}
    n = len(sid)
    if any(len(v) != n for v in meta.values()):
        raise ValueError("fitting-details and fp_tau sounding counts differ")
    if not np.array_equal(meta["fp_number"].astype(int), fp):
        raise ValueError("fitting-details and fp_tau footprint order differ")
    meta.update(ref)

    if max_soundings is not None and n > max_soundings:
        keep = np.linspace(0, n - 1, max_soundings).astype(int)
        sid, fp, tau, toa = sid[keep], fp[keep], tau[:, keep], toa[:, keep]
        meta = {k: v[keep] for k, v in meta.items()}
        n = len(sid)

    rad = np.full((3, n, 1016), np.nan, dtype=np.float32)
    with h5py.File(l1b_file, "r") as f:
        l1b_sid = f["SoundingGeometry/sounding_id"][...].astype(np.int64)   # [T, 8]
        snr_coef = f["InstrumentHeader/snr_coef"][...]                      # [3, 8, 1016, 2]
        max_ms = f["InstrumentHeader/measureable_signal_max_observed"][...]  # [3]
        track = np.full(n, -1)
        for ifp in range(8):
            col = l1b_sid[:, ifp]
            order = np.argsort(col, kind="stable")
            sel = fp == ifp
            pos = np.clip(np.searchsorted(col[order], sid[sel]), 0, len(col) - 1)
            hit = col[order][pos] == sid[sel]
            track[sel] = np.where(hit, order[pos], -1)
        ok = track >= 0
        for ib, b in enumerate(BANDS):
            full = f[f"SoundingMeasurements/{_L1B_RAD[b]}"][...]            # [T, 8, 1016]
            rad[ib, ok] = full[track[ok], fp[ok]]
            del full

    noise = np.full_like(rad, np.nan)
    for ib in range(3):
        noise[ib] = l1b_noise(rad[ib], snr_coef[ib][fp], float(max_ms[ib]))

    with np.errstate(divide="ignore", invalid="ignore"):
        t = rad / toa * np.pi
        t[(t <= 0) | (t > 1)] = np.nan
        ln_t = np.log(t).astype(np.float32)
        sigma_ln_t = (noise / rad).astype(np.float32)
    sigma_ln_t[~np.isfinite(ln_t)] = np.nan
    return OrbitSpectra(sid, fp, tau, ln_t, sigma_ln_t, meta)

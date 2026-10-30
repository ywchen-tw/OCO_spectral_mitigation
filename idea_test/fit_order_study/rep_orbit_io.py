"""Orbit loader for the replication test when no fitting-details file exists.

Produces the same ``OrbitSpectra`` as ``orbit_io.load_orbit`` but builds the
metadata from the Lite file (or, for proxy validation, from a fitting-details
file) and replaces the MODIS cloud distance by a proxy:

* ``cld_dist_km``: great-circle distance to the nearest L1B sounding (any
  footprint, whole orbit, valid geolocation) that did not reach the Lite file
  (proxy "A"; it failed the ABP/IDP prescreen or another L2 gate).  Proxy "B"
  additionally counts Lite soundings with QF1 as cloudy.  The local L2CPr file
  is the CO2-prior product and carries no cloud flag, so this is the only
  cloud information available locally.
* ``xco2_bc_anomaly``: the production ``compute_xco2_anomaly`` (imported by
  file path from ``src/spectral/anomaly.py``, production arguments) applied to
  the Lite bias-corrected ``xco2`` with the proxy cloud distance.

Production reference coefficients (``{band}_k1`` etc.) are absent and filled
with NaN; downstream code uses the production-order fit (``PROD_TAG``) as the
reference instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import netCDF4
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from orbit_io import _L1B_RAD, _M1, BANDS, OrbitSpectra, l1b_noise  # noqa: E402

EARTH_RADIUS_KM = 6371.0


def _load_anomaly_module():
    """Import src/spectral/anomaly.py by path (the package __init__ needs pyproj)."""
    spec = importlib.util.spec_from_file_location(
        "rep_prod_anomaly", REPO / "src" / "spectral" / "anomaly.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ANOM = _load_anomaly_module()
compute_xco2_anomaly = _ANOM.compute_xco2_anomaly


# --------------------------------------------------------------------------- geometry


def _unit_xyz(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def nearest_distance_km(lat, lon, ref_lat, ref_lon):
    """Great-circle distance from each (lat, lon) to the nearest reference point."""
    tree = cKDTree(_unit_xyz(ref_lat, ref_lon))
    chord, _ = tree.query(_unit_xyz(lat, lon))
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def l1b_geolocation(l1b_file: str) -> dict:
    with h5py.File(l1b_file, "r") as f:
        g = f["SoundingGeometry"]
        out = {k: g[f"sounding_{k}"][...].ravel()
               for k in ("id", "latitude", "longitude", "land_fraction", "qual_flag")}
    out["id"] = out["id"].astype(np.int64)
    out["valid"] = (out["id"] > 0) & (out["latitude"] > -900) & (out["longitude"] > -900)
    return out


def cloud_proxy_km(l1b_geo: dict, member_sid: np.ndarray, query_sid: np.ndarray,
                   extra_cloudy_sid: np.ndarray | None = None) -> np.ndarray:
    """Distance (km) from each query sounding to the nearest 'cloudy' L1B sounding.

    Cloudy = valid L1B sounding not in ``member_sid`` (the Lite set), plus any
    sounding in ``extra_cloudy_sid``.
    """
    ids, valid = l1b_geo["id"], l1b_geo["valid"]
    cloudy = valid & ~np.isin(ids, member_sid)
    if extra_cloudy_sid is not None and len(extra_cloudy_sid):
        cloudy |= valid & np.isin(ids, extra_cloudy_sid)
    pos = {int(s): i for i, s in enumerate(ids)}
    idx = np.array([pos.get(int(s), -1) for s in query_sid])
    if np.any(idx < 0):
        raise ValueError(f"{np.sum(idx < 0)} query soundings are not in the L1B file")
    lat, lon = l1b_geo["latitude"][idx], l1b_geo["longitude"][idx]
    return nearest_distance_km(lat, lon, l1b_geo["latitude"][cloudy], l1b_geo["longitude"][cloudy])


# --------------------------------------------------------------------------- metadata sources


def lite_meta(lite_file: str, sid: np.ndarray) -> dict:
    """Per-sounding Lite metadata in the order of ``sid`` (all must be present)."""
    with netCDF4.Dataset(lite_file, "r") as nc:
        lt_id = np.asarray(nc.variables["sounding_id"][:], dtype=np.int64)
        order = np.argsort(lt_id)
        pos = np.searchsorted(lt_id[order], sid)
        pos = np.clip(pos, 0, len(lt_id) - 1)
        if not np.array_equal(lt_id[order][pos], sid):
            raise ValueError("fp_tau soundings missing from the Lite file")
        row = order[pos]

        def root(v):
            return np.asarray(nc.variables[v][:], dtype=float)[row]

        def grp(g, v):
            return np.asarray(nc.groups[g].variables[v][:], dtype=float)[row]

        meta = {
            "xco2_bc": root("xco2"),
            "xco2_raw": grp("Retrieval", "xco2_raw"),
            "xco2_qf": root("xco2_quality_flag").astype(int),
            "lat": root("latitude"),
            "lon": root("longitude"),
            "sza": root("solar_zenith_angle"),
            "vza": root("sensor_zenith_angle"),
            "glint_angle": grp("Sounding", "glint_angle"),
            "land_fraction": grp("Sounding", "land_fraction"),
            "sfc_type": grp("Retrieval", "surface_type"),
            "fp_number": grp("Sounding", "footprint").astype(int) - 1,
            "lite_qf1_sid": lt_id[np.asarray(nc.variables["xco2_quality_flag"][:]) != 0],
        }
    for k in ("xco2_bc", "xco2_raw"):
        meta[k][meta[k] <= 0] = np.nan       # as production fitting.py
    return meta


def fit_details_meta(fit_details_file: str, l1b_geo: dict, sid: np.ndarray) -> dict:
    """Validation path: Lite-equivalent fields from a fitting-details file.

    The true MODIS distance and production anomaly are kept as
    ``cld_dist_true_km`` / ``xco2_bc_anomaly_true``; land fraction comes from
    the L1B (the fitting-details file has none).
    """
    keys = ("xco2_bc", "xco2_raw", "xco2_qf", "lat", "lon", "sza", "vza", "glint_angle",
            "sfc_type", "fp_number", "cld_dist_km", "xco2_bc_anomaly")
    with h5py.File(fit_details_file, "r") as f:
        raw = {k: f[k][...] for k in keys}
        ref = {f"{b}_{k}": f[f"{b}_{k}_fitting_nosg"][...]
               for b in BANDS for k in ("k1", "k2", "intercept")}
    pos = {int(s): i for i, s in enumerate(l1b_geo["id"])}
    idx = np.array([pos[int(s)] for s in sid])
    meta = {k: raw[k] for k in keys if k not in ("cld_dist_km", "xco2_bc_anomaly")}
    meta["xco2_qf"] = meta["xco2_qf"].astype(int)
    meta["fp_number"] = meta["fp_number"].astype(int)
    meta["land_fraction"] = l1b_geo["land_fraction"][idx].astype(float)
    meta["cld_dist_true_km"] = raw["cld_dist_km"]
    meta["xco2_bc_anomaly_true"] = raw["xco2_bc_anomaly"]
    meta["lite_qf1_sid"] = sid[meta["xco2_qf"] != 0]
    meta.update(ref)
    return meta


# --------------------------------------------------------------------------- spectra


def l1b_spectra(l1b_file: str, sid: np.ndarray, fp: np.ndarray, toa: np.ndarray):
    """ln T and σ(ln T) exactly as ``orbit_io.load_orbit`` (same matching and noise model)."""
    n = len(sid)
    rad = np.full((3, n, 1016), np.nan, dtype=np.float32)
    with h5py.File(l1b_file, "r") as f:
        l1b_sid = f["SoundingGeometry/sounding_id"][...].astype(np.int64)
        snr_coef = f["InstrumentHeader/snr_coef"][...]
        max_ms = f["InstrumentHeader/measureable_signal_max_observed"][...]
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
            full = f[f"SoundingMeasurements/{_L1B_RAD[b]}"][...]
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
    return ln_t, sigma_ln_t, int(ok.sum())


# --------------------------------------------------------------------------- loader


def load_orbit_proxy(l1b_file: str, fp_tau_file: str, lite_file: str | None = None,
                     fit_details_file: str | None = None, cloud_proxy: str = "A",
                     max_soundings: int | None = None) -> OrbitSpectra:
    """OrbitSpectra with proxy ``cld_dist_km`` and ``xco2_bc_anomaly``.

    Exactly one of ``lite_file`` / ``fit_details_file`` supplies the Lite-type
    metadata.  The proxy is computed on the full sounding set before any
    ``max_soundings`` subsampling.
    """
    if (lite_file is None) == (fit_details_file is None):
        raise ValueError("pass exactly one of lite_file, fit_details_file")
    with h5py.File(fp_tau_file, "r") as f:
        sid = f["sounding_id"][...].astype(np.int64)
        fp = f["fp_number"][...].astype(int)
        tau = np.stack([f[f"{b}_tau_output"][...] for b in BANDS]).astype(np.float32)
        toa = np.stack([f[f"{b}_toa_sol_output"][...] for b in BANDS]).astype(np.float32) * _M1

    geo = l1b_geolocation(l1b_file)
    if lite_file is not None:
        meta = lite_meta(lite_file, sid)
        meta.update({f"{b}_{k}": np.full(len(sid), np.nan)
                     for b in BANDS for k in ("k1", "k2", "intercept")})
    else:
        meta = fit_details_meta(fit_details_file, geo, sid)
    if not np.array_equal(meta["fp_number"], fp):
        raise ValueError("metadata and fp_tau footprint order differ")

    if cloud_proxy not in ("A", "B"):
        raise ValueError(f"unknown cloud proxy {cloud_proxy!r}")
    extra = meta.pop("lite_qf1_sid") if cloud_proxy == "B" else None
    meta.pop("lite_qf1_sid", None)
    meta["cld_dist_km"] = cloud_proxy_km(geo, sid, sid, extra)
    meta["xco2_bc_anomaly"] = compute_xco2_anomaly(meta["lat"], meta["cld_dist_km"],
                                                   meta["xco2_bc"], **_prod_anomaly_args())
    meta["is_land"] = meta["land_fraction"] >= 50.0

    n = len(sid)
    if max_soundings is not None and n > max_soundings:
        keep = np.linspace(0, n - 1, max_soundings).astype(int)
        sid, fp, tau, toa = sid[keep], fp[keep], tau[:, keep], toa[:, keep]
        meta = {k: v[keep] for k, v in meta.items()}

    ln_t, sigma_ln_t, n_matched = l1b_spectra(l1b_file, sid, fp, toa)
    if n_matched < len(sid):
        print(f"warning: {len(sid) - n_matched} soundings not matched in the L1B")
    return OrbitSpectra(sid, fp, tau, ln_t, sigma_ln_t, meta)


def _prod_anomaly_args() -> dict:
    return {"lat_thres": _ANOM.ANOMALY_LAT_THRES_DEG, "std_thres": _ANOM.ANOMALY_STD_THRES_PPM,
            "min_cld_dist": _ANOM.ANOMALY_MIN_CLD_DIST_KM}

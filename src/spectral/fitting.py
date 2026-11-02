"""OCO-2 footprint spectral analysis — orchestration and CLI.

The implementation is split along its seams (review §7.4):
    spectral/cumulant_fit.py — fit core (models, exact linear solver, chunk worker)
    spectral/orbit_data.py   — per-date / per-orbit input loading + validation
    spectral/anomaly.py      — clear-sky-reference target definition
    spectral/fit_plots.py    — example-fit visualisations
This module keeps process_orbit / preprocess / run_simulation / the CLI and
re-exports every public name from the submodules, so existing imports like
`from spectral.fitting import compute_xco2_anomaly` are unaffected.
"""

import os
import platform
import sys
from datetime import datetime
from pathlib import Path

# When run directly (python src/spectral/fitting.py), Python adds src/spectral/
# to sys.path but not src/ itself.  Add src/ so that sibling packages
# (utils, abs_util, config, …) are importable without PYTHONPATH.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Must be set before any HDF5 library call; Lustre does not support POSIX
# advisory locks and HDF5 >= 1.10 raises NC_EHDF (-101) without this.
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import argparse
import glob
import logging

import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pyproj import Transformer

from constants import FIT_ORDER, anomaly_args as production_anomaly_args
from config import Config
from abs_util.fp_atm import oco_fp_atm_abs
from abs_util.ils_tau import tau_convolution_version
from abs_util.oco_util import timing
from spectral import orbit_lock

# Re-exports (backward compat: this module was a single file until 2026-07).
from spectral.cumulant_fit import (  # noqa: F401
    LOG_TRANSMITTANCE_MODELS, MAX_KAPPAS, SAVGOL_ORDER, SAVGOL_WINDOW,
    EXTRA_FIT_FIELDS, FIT_FIELDS, FIT_SCHEMA, _doppler_fit_chunk,
    _extra_fit_chunk, _fit_chunk, _solve_cumulant, compute_transmittance,
    fit_spectral_model,
    get_design_matrix, log_transmittance_model_1, log_transmittance_model_2,
    log_transmittance_model_3, log_transmittance_model_4,
    log_transmittance_model_5, log_transmittance_model_7,
    log_transmittance_model_9, transmittance_model)
from spectral.orbit_data import (  # noqa: F401
    _decode_hdf5_attr, _discover_orbit_files, _validate_readable_hdf5, cloud_distance_file_path,
    LAND_FRACTION_THRESHOLD, fp_tau_file_is_current, load_orbit_data, load_profile_data,
    load_shared_data, search_oco2_orbit, select_lite_file,
    validate_cloud_distance_file)
from spectral.anomaly import compute_xco2_anomaly, nosg_reference_stats  # noqa: F401
from spectral.fit_plots import (  # noqa: F401
    plot_doppler_fit_example, plot_fitting_example, plot_orbit_fitting_examples)


logger = logging.getLogger(__name__)

# Fit coefficients with clear-sky reference statistics ({ref,r15,r05}_<band>_<field>_{mean,std}).
# REF_FIT_VARS enter the shared pool that also defines the XCO2 anomaly targets
# (NaN only where the fit failed); REF_FIT_VARS_SOLO have extra NaN modes and
# get their own calls.  See the anomaly section of process_orbit.
REF_FIT_VARS = {
    "o2a":  ("k1_am", "a1_am", "f_am", "exp_int_am", "k1_dry_t2", "k1w_t2", "k2_t2"),
    "wco2": ("k1_am", "k1w_am", "a1_am", "exp_int_am", "k2diag_am",
             "k1_dry_t2", "k1w_t2", "k2_t2"),
    "sco2": ("k1_am", "k1w_am", "k2_am", "a1_am", "exp_int_am",
             "k1_dry_t2", "k1w_t2", "k2_t2"),
}
REF_FIT_VARS_SOLO = (("o2a", "k2diag_am"), ("sco2", "a2_am"))


# ─── Orbit orchestration ───────────────────────────────────────────────────────

# Output subdirectory of <result_dir> for fitting_details_<date>_<orbit>.h5.  The v2
# refit was written to fitting_details_v2 next to the v1 files; once it was complete the
# v1 files were deleted and fitting_details_v2 renamed to fitting_details (2026-11-01).
# For a future schema change, write to a new subdir (--fit-subdir) the same way so the
# current fits are never overwritten in place.
FIT_SUBDIR = "fitting_details"


def process_orbit(sat, orbit_id, shared_data, overwrite=False, fit_workers=1,
                  fit_subdir=FIT_SUBDIR):
    """Fit one orbit's soundings on the telluric-Doppler tau and raw ln T.

    Workflow
    --------
    1. Load per-orbit L1B radiances and the v4 (telluric + solar Doppler)
       optical depths with their per-channel H2O fraction (load_orbit_data).
    2. Compute transmittances T = π·L/TOA_sol for all soundings and bands.
    3. Fit cumulant_fit.FIT_FIELDS in contiguous chunks (_doppler_fit_chunk),
       optionally across worker processes: unconstrained orders 2 and 3, the
       two-tau (_t2) and the agreed-model (_am) fits, all on the raw ln T (no
       smoothing), each coefficient with its 1-sigma sig_* error.
    4. Extract Lite retrieval variables via vectorised index lookup.
    5. XCO2 anomalies and clear-sky reference statistics (ref/r15/r05) of the
       _am and _t2 coefficients (see the anomaly section).
    6. Write fitting_details_<date>_<orbit>.h5 with attrs fit_schema and
       tau_convolution.

    A cached file is reused only when both attrs match (FIT_SCHEMA and the tau
    file's tau_convolution); anything else is refitted.  The tau file must be the
    telluric-Doppler version with frac_h2o — there is no fallback to other tau.

    Parameters
    ----------
    sat         : dict from preprocess()
    orbit_id    : str
    shared_data : dict from load_shared_data()
    overwrite   : bool  refit even when a matching cache exists
    fit_workers : int  processes for the fit stage; 1 = in-process
                  (results are identical regardless of worker count)
    """
    date        = sat['date'].strftime("%Y-%m-%d")
    h5_output_dir = f"{sat['result_dir']}/{fit_subdir}"
    output_file = f"{h5_output_dir}/fitting_details_{date}_{orbit_id}.h5"
    os.makedirs(h5_output_dir, exist_ok=True)

    # ── 1. Load orbit data ─────────────────────────────────────────────────
    logger.info(f"[{orbit_id}] Loading orbit data...")
    od = load_orbit_data(sat, orbit_id)
    N  = len(od["sounding_id"])
    tags = ["o2a", "wco2", "sco2"]

    expected_tau = tau_convolution_version(True)
    if od["tau_convolution"] != expected_tau or od["frac_h2o"] is None:
        raise RuntimeError(
            f"[{orbit_id}] tau file is not the telluric-Doppler version "
            f"(tau_convolution={od['tau_convolution']!r}, expected {expected_tau!r}; "
            f"frac_h2o {'missing' if od['frac_h2o'] is None else 'present'}); "
            "recompute it with workspace/compute_tau_only.py --telluric-doppler")

    if os.path.isfile(output_file) and not overwrite:
        cached_schema = cached_tau_conv = None
        try:
            with h5py.File(output_file, "r") as f:
                _s, _t = f.attrs.get("fit_schema"), f.attrs.get("tau_convolution")
                cached_schema = _decode_hdf5_attr(_s) if _s is not None else None
                cached_tau_conv = _decode_hdf5_attr(_t) if _t is not None else None
        except OSError:
            pass
        if cached_schema == FIT_SCHEMA and cached_tau_conv == od["tau_convolution"]:
            logger.info(f"[{orbit_id}] {output_file} is current (fit_schema {FIT_SCHEMA}); skipping")
            return
        logger.info(f"[{orbit_id}] refitting: cached fit_schema {cached_schema!r}, "
                    f"tau_convolution {cached_tau_conv!r} (need {FIT_SCHEMA!r}, "
                    f"{od['tau_convolution']!r})")

    # ── 2. Transmittance for all soundings and bands at once ───────────────
    logger.info(f"[{orbit_id}] Computing transmittances for {N} soundings...")
    T_all    = compute_transmittance(od["radiances"], od["toa_sol"])  # [3, N, 1016]
    ln_T_all = np.where(T_all > 0, np.log(T_all), np.nan)            # [3, N, 1016]

    # ── 3. Chunked fitting (optionally multi-process) ──────────────────────
    valid_idx = np.where(od["valid_l1b"])[0]
    chunk_size = 256
    n_chunks = max(1, int(np.ceil(len(valid_idx) / chunk_size)))
    chunks = np.array_split(valid_idx, n_chunks)
    logger.info(f"[{orbit_id}] Fitting orders 2/3, two-tau and agreed model for "
                f"{len(valid_idx)} soundings in {n_chunks} chunks (fit_workers={fit_workers})...")

    def _chunk_args(chunk):
        # Edge channels excluded here once, matching the historical [1:-1].
        return (chunk, od["tau"][:, chunk, 1:-1], ln_T_all[:, chunk, 1:-1],
                od["frac_h2o"][:, chunk, 1:-1], od["land"][chunk])

    fit = np.full((3, N, len(FIT_FIELDS)), np.nan)
    if fit_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=fit_workers) as pool:
            futures = [pool.submit(_doppler_fit_chunk, *_chunk_args(c)) for c in chunks]
            results = [f.result() for f in futures]
    else:
        results = [_doppler_fit_chunk(*_chunk_args(c)) for c in chunks]
    for j_idx, fc in results:
        fit[:, j_idx, :] = fc

    def _fv(band, name):
        return fit[tags.index(band), :, FIT_FIELDS.index(name)]

    # ── 3a. One example plot per band per orbit ────────────────────────────
    # First sounding (orbit order) with a converged agreed-model fit; drawn from
    # the stored values, written next to the tau file.  A plotting error only
    # logs a warning.
    plot_dir = f"{sat['result_dir']}/{date}/{orbit_id}"
    os.makedirs(plot_dir, exist_ok=True)
    for i_band, tag in enumerate(tags):
        good = np.flatnonzero((_fv(tag, "ok_am") == 1) & np.isfinite(_fv(tag, "k1_am")))
        if not len(good):
            continue
        j = int(good[0])
        try:
            tau_j = od["tau"][i_band, j][1:-1]
            ln_j = ln_T_all[i_band, j][1:-1]
            mask = ~np.isnan(ln_j) & ~np.isnan(tau_j)
            frac_j = np.nan_to_num(od["frac_h2o"][i_band, j][1:-1][mask], nan=0.0)
            x_j = (np.flatnonzero(mask) + 1 - 507.5) / 507.5
            plot_doppler_fit_example(
                tag, i_band, int(od["fp_number"][j]), int(od["sounding_id"][j]),
                od["wvl"][i_band, od["fp_number"][j]], od["radiances"][i_band, j],
                T_all[i_band, j], tau_j[mask], ln_j[mask], frac_j, x_j,
                {name: fit[i_band, j, k] for k, name in enumerate(FIT_FIELDS)}, plot_dir)
        except Exception as exc:   # noqa: BLE001 - a plot must never fail the fit
            logger.warning(f"[{orbit_id}] example plot for {tag} failed: {exc}")


    # ── 4. Lite variable extraction (vectorised) ───────────────────────────
    def _lite(key):
        """Extract one Lite variable for all soundings; NaN where not matched."""
        out = np.full(N, np.nan)
        if valid_lt.any():
            out[valid_lt] = lite[key][row_inds[valid_lt]]
        return out

    logger.info(f"[{orbit_id}] Extracting Lite variables...")
    lite_idx = shared_data["lite_index"]
    lite     = shared_data["lite"]
    row_inds = np.array([lite_idx.get(int(sid), -1) for sid in od["sounding_id"]])
    valid_lt = row_inds >= 0

    lt_xco2_bc  = _lite("xco2_corr")
    lt_xco2_raw = _lite("xco2_raw")
    lt_xco2_raw[lt_xco2_raw <= 0] = np.nan  # Mask unphysical raw XCO2 values (zero or negative)
    lt_xco2_bc[lt_xco2_bc <= 0] = np.nan    # Mask unphysical corrected XCO2 values (zero or negative)
    lt_alb_o2a  = _lite("albedo_o2a")
    lt_alb_wco2 = _lite("albedo_wco2")
    lt_alb_sco2 = _lite("albedo_sco2")

    # ── 5. Cloud distance per sounding (O(1) dict lookup) ─────────────────
    cld_idx     = shared_data["cld_dist_index"]
    fp_cld_dist = np.array([cld_idx.get(int(sid), np.nan) for sid in od["sounding_id"]])
    weighted_cld_idx = shared_data.get("weighted_cld_dist_index", {})
    fp_weighted_cld_dist = np.array([weighted_cld_idx.get(int(sid), np.nan) for sid in od["sounding_id"]])

    # ── 5b. Atmospheric profiles (L2 Met + CO2Prior) on the sigma grid ────────
    logger.info(f"[{orbit_id}] Extracting atmospheric profiles (sigma grid)...")
    profiles = load_profile_data(
        sat[orbit_id]["oco_met"], sat[orbit_id]["oco_co2prior"], od["sounding_id"],
    )

    # ── 6. XCO2 anomaly and clear-sky reference statistics ─────────────────
    # compute_xco2_anomaly builds ONE clear-sky pool shared by the anomaly and
    # every extra variable: a reference sounding must have all of them finite.
    # The main call therefore carries only variables that are NaN just where the
    # fit itself failed (albedo + the core _am/_t2 coefficients); its pool defines
    # the anomaly targets.  O2A k2diag_am (own refit can fail) and SCO2 a2_am
    # (land only) get their own calls (pool = core set + that variable) so they
    # cannot shrink the target pool.  Fields NaN or constant by design (O2A k1w_am,
    # k2_am of O2A/WCO2, f_am outside O2A) have no reference statistics.
    logger.info(f"[{orbit_id}] Computing XCO2 anomalies and reference statistics...")
    core_vars = {"alb_o2a": lt_alb_o2a, "alb_wco2": lt_alb_wco2, "alb_sco2": lt_alb_sco2}
    for band, names in REF_FIT_VARS.items():
        for name in names:
            core_vars[f"{band}_{name}"] = _fv(band, name)
    solo_vars = {f"{band}_{name}": _fv(band, name) for band, name in REF_FIT_VARS_SOLO}

    ref_out = {}
    anomaly_out = {}
    for prefix, kwargs in (("ref", production_anomaly_args()),
                           ("r15", production_anomaly_args(min_cld_dist=15.0)),
                           ("r05", production_anomaly_args(min_cld_dist=5.0))):
        suffix = "" if prefix == "ref" else f"_{prefix}"
        anomaly_out[f"xco2_raw_anomaly{suffix}"] = compute_xco2_anomaly(
            od["lat"], fp_cld_dist, lt_xco2_raw, **kwargs)
        bc_anom, means, stds, _, _, n_ref = compute_xco2_anomaly(
            od["lat"], fp_cld_dist, lt_xco2_bc, extra_vars=core_vars,
            return_ref_stats=True, **kwargs)
        anomaly_out[f"xco2_bc_anomaly{suffix}"] = bc_anom
        ref_out[f"{prefix}_n"] = n_ref
        for key in core_vars:
            ref_out[f"{prefix}_{key}_mean"] = means[key]
            ref_out[f"{prefix}_{key}_std"] = stds[key]
        for key, arr in solo_vars.items():
            _, m_solo, s_solo = compute_xco2_anomaly(
                od["lat"], fp_cld_dist, lt_xco2_bc,
                extra_vars={**core_vars, key: arr}, **kwargs)
            ref_out[f"{prefix}_{key}_mean"] = m_solo[key]
            ref_out[f"{prefix}_{key}_std"] = s_solo[key]


    # -- 6c. Calculate fp areas from lite vertex longitudes and latitudes --
    fp_area_km2 = np.full(N, np.nan)
    if valid_lt.any():
        vlon = lite["vertex_longitude"][row_inds[valid_lt]]  # [n_valid, 4]
        vlat = lite["vertex_latitude"][row_inds[valid_lt]]   # [n_valid, 4]
        # Project to WGS 84 / EASE-Grid 2.0 Global (equal-area, EPSG:6933) for accurate area
        _t = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
        x, y = _t.transform(vlon, vlat)                      # array-in, array-out
        # Shoelace formula per footprint quad — identical to Polygon(...).area
        # for these simple (non-self-intersecting) polygons.
        x2 = np.roll(x, -1, axis=1)
        y2 = np.roll(y, -1, axis=1)
        areas = 0.5 * np.abs(np.sum(x * y2 - x2 * y, axis=1))
        fp_area_km2[valid_lt] = areas * 1e-6  # m² → km²

    # ── 6d. 20-level column operator per sounding (AK harmonization, M2) ───
    # Averaging kernel, pressure weights, prior CO2 profile and pressure grid
    # from the Lite file, so downstream TCCON comparisons never reopen it.
    _ak_vars = ("xco2_averaging_kernel", "pressure_weight",
                "co2_profile_apriori", "pressure_levels")
    ak_arrays = {}
    for _v in _ak_vars:
        if _v in lite:
            _n_lev = lite[_v].shape[1]
            _arr = np.full((N, _n_lev), np.nan)
            if valid_lt.any():
                _arr[valid_lt] = lite[_v][row_inds[valid_lt]]
            ak_arrays[_v] = _arr

    # ── 7. Write output HDF5 ───────────────────────────────────────────────
    logger.info(f"[{orbit_id}] Writing {output_file}...")
    output_dict = {f"{tag}_{name}": fit[ib, :, ik]
                   for ib, tag in enumerate(tags) for ik, name in enumerate(FIT_FIELDS)}
    output_dict.update({
        "date":      np.array([date]*N, dtype='S'),
        "orbit_id":  np.array([orbit_id]*N, dtype='S'),
        "time":      _lite("time"),
        "lon":       od["lon"],
        "lat":       od["lat"],
        "sza":       od["sza"],
        "vza":       od["vza"],
        "mu_sza":     np.cos(np.radians(od["sza"])),
        "mu_vza":     np.cos(np.radians(od["vza"])),
        "fp_number": od["fp_number"],
        "fp_id":     od["sounding_id"],
        "fp_area_km2": fp_area_km2,
        # Cloud proximity
        "cld_dist_km":      fp_cld_dist,
        "weighted_cloud_dist_km": fp_weighted_cld_dist,
        # XCO2
        "xco2_apriori":     _lite("xco2_apriori"),
        "xco2_bc":          lt_xco2_bc,
        "xco2_raw":         lt_xco2_raw,
        # Lite retrieval variables
        "psfc":        _lite("psurf"),
        "airmass":     _lite("airmass"),
        "delT":        _lite("deltaT"),
        "dp":          _lite("dp"),
        "dp_o2a":      _lite("dp_o2a"),
        "dp_sco2":     _lite("dp_sco2"),
        "co2_grad_del": _lite("co2_grad_del"),
        "alb_o2a":     lt_alb_o2a,
        "alb_wco2":    lt_alb_wco2,
        "alb_sco2":    lt_alb_sco2,
        "aod_total":   _lite("aod_total"),
        "fs_rel":      _lite("fs_rel"),
        "alt":         _lite("altitude"),
        "alt_std":     _lite("altitude_std"),
        "xco2_qf":     _lite("qf"),
        "sfc_type":    _lite("sfc_type"),
        "ws":          _lite("windspeed"),
        "ws_apriori":  _lite("windspeed_apriori"),
        # Preprocessor variables
        "co2_ratio_bc":  _lite("co2_ratio_bc"),
        "h2o_ratio_bc":  _lite("h2o_ratio_bc"),
        "csnr_o2a":      _lite("color_slice_noise_ratio_o2a"),
        "csnr_wco2":     _lite("color_slice_noise_ratio_wco2"),
        "csnr_sco2":     _lite("color_slice_noise_ratio_sco2"),
        "dp_abp":        _lite("dp_abp"),
        "h_cont_o2a":    _lite("h_continuum_o2a"),
        "h_cont_wco2":   _lite("h_continuum_wco2"),
        "h_cont_sco2":   _lite("h_continuum_sco2"),
        "max_declock_o2a":  _lite("max_declocking_o2a"),
        "max_declock_wco2": _lite("max_declocking_wco2"),
        "max_declock_sco2": _lite("max_declocking_sco2"),
        "xco2_strong_idp": _lite("xco2_strong_idp"),
        "xco2_weak_idp":   _lite("xco2_weak_idp"),
        # Additional retrieval variables
        "h2o_scale":    _lite("h2o_scale"),
        "dpfrac":       _lite("dpfrac"),
        "aod_bc":       _lite("aod_bc"),
        "aod_dust":     _lite("aod_dust"),
        "aod_ice":      _lite("aod_ice"),
        "aod_water":    _lite("aod_water"),
        "aod_oc":       _lite("aod_oc"),
        "aod_seasalt":  _lite("aod_seasalt"),
        "aod_strataer": _lite("aod_strataer"),
        "aod_sulfate":  _lite("aod_sulfate"),
        "dust_height":  _lite("dust_height"),
        "ice_height":   _lite("ice_height"),
        "dws":          _lite("dws"),
        # Additional sounding variables
        "snr_o2a":      _lite("snr_o2a"),
        "snr_wco2":     _lite("snr_wco2"),
        "snr_sco2":     _lite("snr_sco2"),
        "glint_angle":  _lite("glint_angle"),
        "pol_angle":    _lite("polarization_angle"),
        "saa":          _lite("saa"),
        "vaa":          _lite("vaa"),
        "s31":          _lite("s31"),
        "s32":          _lite("s32"),
        "snow_flag":   _lite("snow_flag"),
        "t700":        _lite("t700"),
        "tcwv":        _lite("tcwv"),
        "operation_mode": _lite("operation_mode"),
        "water_height": _lite("water_height"),
        # Tropopause (L2 Met) — pressure (Pa) and temperature (K).  Normalised
        # (sigma) tropopause is derived from these + psurf in build_feature_dataset.py.
        "tropopause_pressure": profiles["tropopause_pressure"],
        "tropopause_temp":     profiles["tropopause_temp"],
        # Raw native-grid atmospheric profiles (72 GEOS levels) + pressure grid
        # (Pa) and Met surface pressure (Pa).  Resampled to a sigma grid + PCA
        # compressed downstream in build_feature_dataset.py.
        "t_profile":        profiles["t_profile"],
        "q_profile":        profiles["q_profile"],
        "co2prior_profile": profiles["co2prior_profile"],
        "p_profile":        profiles["p_profile"],
        "psurf_met":        profiles["psurf_met"],
        # Spectral-fit-quality diagnostics (Tier-A candidate features for the
        # cloud-contamination tail; see log/archive/MODEL_PLANS_HISTORICAL.md §1.4).
        "chi2_o2a":     _lite("chi2_o2a"),
        "chi2_wco2":    _lite("chi2_wco2"),
        "chi2_sco2":    _lite("chi2_sco2"),
        "rms_rel_o2a":  _lite("rms_rel_o2a"),
        "rms_rel_wco2": _lite("rms_rel_wco2"),
        "rms_rel_sco2": _lite("rms_rel_sco2"),
        "eof3_1_rel":      _lite("eof3_1_rel"),
        "diverging_steps": _lite("diverging_steps"),
        "xco2_uncertainty": _lite("xco2_uncertainty"),
    })
    output_dict.update(anomaly_out)
    output_dict.update(ref_out)       # {ref,r15,r05}_{alb_<band>,<band>_<field>}_{mean,std}, {prefix}_n
    output_dict.update(ak_arrays)   # [N, 20] column-operator datasets

    # float64 → float32 on write (except keys needing full precision) plus
    # shuffle+gzip: shrinks fitting_details.h5 severalfold and speeds the
    # downstream build_feature_dataset reads.  Integer / string datasets are
    # stored as-is.
    keep_float64 = {"time"}   # seconds since epoch: float32 would lose ~minutes
    # Write to a temporary name and rename into place, so another job's cache
    # check never opens a half-written file.
    tmp_output = f"{output_file}.{os.getpid()}.tmp"
    try:
        with h5py.File(tmp_output, "w") as f_out:
            f_out.attrs["tau_convolution"] = od["tau_convolution"]
            f_out.attrs["fit_schema"] = FIT_SCHEMA
            f_out.attrs["land_rule"] = f"sounding_land_fraction >= {LAND_FRACTION_THRESHOLD:g}"
            f_out.attrs["sigma_note"] = (
                "sig_* = formal 1-sigma, s^2 (J^T J)^-1 with s^2 = RSS/(n-p) of the fit's own "
                "residual (raw ln T, no smoothing); understates scene-correlated error")
            for key, value in output_dict.items():
                arr = np.asarray(value)
                if arr.dtype == np.float64 and key not in keep_float64:
                    arr = arr.astype(np.float32)
                if arr.ndim > 0 and arr.size > 0:
                    f_out.create_dataset(key, data=arr, compression="gzip",
                                         compression_opts=4, shuffle=True)
                else:
                    f_out.create_dataset(key, data=arr)
        os.replace(tmp_output, output_file)
    finally:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)

    logger.info(f"[{orbit_id}] Done.")


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def _select_orbits(orbit_list, orbits=None, limit_granules=-1):
    """Restrict the discovered orbit list to ``orbits`` (if given), then to
    the first ``limit_granules`` entries.  Unknown orbit ids raise."""
    if orbits:
        missing = [o for o in orbits if o not in orbit_list]
        if missing:
            raise FileNotFoundError(
                f"Requested orbit(s) not found for this date: {missing}; "
                f"available: {orbit_list}"
            )
        orbit_list = [o for o in orbit_list if o in set(orbits)]
    if limit_granules > 0:
        orbit_list = orbit_list[:limit_granules]
    return orbit_list


def _fp_tau_path(result_dir, date, orbit_id):
    date_str = date.strftime("%Y-%m-%d")
    return os.path.abspath(f"{result_dir}/{date_str}/{orbit_id}/fp_tau_combined.h5")


def _compute_fp_tau(sat0, orbit_id, fp_tau_file):
    print(f"Computing footprint optical depths for orbit {orbit_id}...")
    oco_fp_atm_abs(
        sat=sat0,
        o2mix=0.20935,
        output=fp_tau_file,
        oco_files_dict=sat0[orbit_id],
        oco_nc_file=sat0["oco_lite"],
        overwrite=True,
        telluric_doppler=bool(sat0.get("telluric_doppler", False)),
    )


def _ensure_fp_tau(sat0, orbit_id, lock_stale_seconds, wait_seconds):
    """Make sure the orbit's optical-depth file is current.

    Computes it under the orbit lock when the lock is free; otherwise waits
    for the job that holds the lock.  Returns True when the file is current
    afterwards.
    """
    fp_tau_file = sat0[orbit_id]["fp_tau_file"]

    def ready():
        return fp_tau_file_is_current(fp_tau_file, telluric_doppler=bool(sat0.get("telluric_doppler", False)))

    if ready():
        return True
    for _attempt in range(2):
        with orbit_lock.held(fp_tau_file, lock_stale_seconds) as got:
            if got:
                if not ready():
                    _compute_fp_tau(sat0, orbit_id, fp_tau_file)
                return ready()
        logger.info(
            f"[{orbit_id}] Optical depths are being computed by another job; "
            f"waiting up to {wait_seconds / 60:.0f} min"
        )
        if orbit_lock.wait_for(fp_tau_file, ready, wait_seconds,
                               stale_seconds=lock_stale_seconds):
            return True
        # The lock is gone but the file is not current (the other job died or
        # failed): loop once more and try to claim the orbit ourselves.
    return ready()


@timing
def preprocess(target_date, data_dir="data", result_dir="results", limit_granules=-1,
               lock_stale_seconds=orbit_lock.DEFAULT_STALE_SECONDS, orbits=None,
               telluric_doppler=False, cloud_dist_dir=None):
    """Discover orbit files and compute per-footprint optical depths.

    Returns sat0 dict consumed by run_simulation and load_shared_data.

    telluric_doppler : compute the test-variant tau (telluric Doppler shift of
        the ABSCO grid plus per-channel attribution datasets); the tau files
        carry their own version stamp, so result_dir should be a separate tree.
    cloud_dist_dir   : directory holding results_<date>.h5 (cloud distances)
        when it differs from result_dir (a test tree reusing production inputs).
    """
    date = target_date
    year = date.year
    doy  = date.timetuple().tm_yday

    oco2_orbit_list = _select_orbits(search_oco2_orbit(date, data_dir=data_dir),
                                     orbits, limit_granules)

    if not oco2_orbit_list:
        raise FileNotFoundError(
            f"No OCO-2 orbit data found for {date} in "
            f"{data_dir}/OCO2/{year}/{doy:03d}"
        )
    logger.info(f"Orbits found: {oco2_orbit_list}")

    OCO2_data_dir = f"{data_dir}/OCO2/{year}/{doy:03d}"
    nc4_matches = glob.glob(f"{OCO2_data_dir}/*.nc4")
    if not nc4_matches:
        raise FileNotFoundError(
            f"No L2 Lite .nc4 file found in {OCO2_data_dir}. "
            f"Re-run oco_modis_cloud_distance.py for this date to download it."
        )
    lite_nc_file = select_lite_file(nc4_matches, date)
    _validate_readable_hdf5(lite_nc_file, "L2 Lite", date)

    sat0 = {
        "date":       date,
        "data_dir":   OCO2_data_dir,
        "result_dir": result_dir,
        "orbit_list": oco2_orbit_list,
        "oco_lite":   lite_nc_file,
        "telluric_doppler": bool(telluric_doppler),
        "cloud_dist_dir": str(cloud_dist_dir) if cloud_dist_dir else None,
    }

    # Shuffle so that several jobs started on the same date work on different
    # orbits.  The shuffled order is stored back into sat0, so the fitting loop
    # in run_simulation follows it too (it used to iterate the sorted list).
    oco2_orbit_list = [str(o) for o in np.random.RandomState().permutation(oco2_orbit_list)]
    sat0["orbit_list"] = oco2_orbit_list
    for orbit_id in oco2_orbit_list:
        orbit_dir = f"{OCO2_data_dir}/{orbit_id}"
        sat0[orbit_id] = _discover_orbit_files(orbit_dir)
        for product_key, product_path in sat0[orbit_id].items():
            _validate_readable_hdf5(product_path, product_key, date)

        fp_tau_file = _fp_tau_path(result_dir, date, orbit_id)
        os.makedirs(os.path.dirname(fp_tau_file), exist_ok=True)
        sat0[orbit_id]["fp_tau_file"] = fp_tau_file
        if fp_tau_file_is_current(fp_tau_file, telluric_doppler=telluric_doppler):
            continue
        # Claim the orbit.  A lock held by another job means that job is
        # already computing it; move on, run_simulation waits for it later.
        with orbit_lock.held(fp_tau_file, lock_stale_seconds) as got:
            if got:
                if not fp_tau_file_is_current(fp_tau_file, telluric_doppler=telluric_doppler):
                    _compute_fp_tau(sat0, orbit_id, fp_tau_file)
            else:
                print(f"Optical depths for orbit {orbit_id} are being computed "
                      f"by another job; skipping for now")

    return sat0


@timing
def run_simulation(target_date, data_dir, result_dir,
                   limit_granules=-1,
                   viz_dir=None,
                   visualize=True,
                   delete_ocofiles=False,
                   fit_workers=1,
                   lock_stale_hours=4.0,
                   tau_wait_minutes=180.0,
                   orbits=None,
                   overwrite=False,
                   cloud_dist_dir=None,
                   fit_subdir=FIT_SUBDIR):
    """Top-level pipeline: preprocess → fit → analyse.

    Always on the telluric-Doppler tau (preprocess(telluric_doppler=True));
    cloud_dist_dir: directory of results_<date>.h5 when it is not result_dir.
    fit_subdir: subdirectory of result_dir for the fitting_details files.
    viz_dir / visualize are accepted for launcher compatibility; the fit no
    longer draws example plots.

    Several jobs may run this for the same date: each orbit's optical-depth
    precompute and fit are claimed through ``orbit_lock`` files, so the jobs
    split the orbits instead of repeating each other's work.
    """
    validate_cloud_distance_file(
        cloud_distance_file_path(cloud_dist_dir or result_dir, target_date),
        target_date,
    )
    stale_seconds = float(lock_stale_hours) * 3600.0
    wait_seconds  = float(tau_wait_minutes) * 60.0
    sat0 = preprocess(target_date, data_dir, result_dir, limit_granules,
                      lock_stale_seconds=stale_seconds, orbits=orbits,
                      telluric_doppler=True, cloud_dist_dir=cloud_dist_dir)

    # Load Lite file and cloud distances once, shared across all orbits
    shared_data = load_shared_data(sat0)

    date          = sat0['date'].strftime("%Y-%m-%d")
    h5_output_dir = f"{sat0['result_dir']}/{fit_subdir}"

    def _output_file(orbit_id):
        return f"{h5_output_dir}/fitting_details_{date}_{orbit_id}.h5"

    def _fit_one(orbit_id):
        """Fit one orbit under its lock; False when another job holds it."""
        if not _ensure_fp_tau(sat0, orbit_id, stale_seconds, wait_seconds):
            logger.warning(f"[{orbit_id}] Optical-depth file still not current; skipping orbit")
            return True   # nothing more this job can do for it
        with orbit_lock.held(_output_file(orbit_id), stale_seconds) as got:
            if not got:
                logger.info(f"[{orbit_id}] Fit in progress in another job; deferring")
                return False
            process_orbit(sat0, orbit_id, shared_data,
                          overwrite=overwrite, fit_workers=fit_workers,
                          fit_subdir=fit_subdir)

        if delete_ocofiles:
            for key in ("oco_l1b", "oco_met", "oco_co2prior"):
                fpath = sat0[orbit_id].get(key)
                if fpath and os.path.isfile(fpath):
                    os.remove(fpath)
                    logger.info(f"Deleted input file: {fpath}")
        return True

    deferred = [orbit_id for orbit_id in sat0["orbit_list"] if not _fit_one(orbit_id)]

    # Second pass over orbits another job was fitting during the first pass.
    # If that job finished, process_orbit reuses its output; if it died, the
    # lock is gone (or stale) and the fit is done here.
    for orbit_id in deferred:
        target = _output_file(orbit_id)
        orbit_lock.wait_for(
            target, lambda: not orbit_lock.is_locked(target, stale_seconds),
            wait_seconds, stale_seconds=stale_seconds,
        )
        if not _fit_one(orbit_id):
            logger.warning(f"[{orbit_id}] Still locked by another job after waiting; left to that job")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def validate_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def get_storage_dir():
    if platform.system() == "Darwin":
        logger.info("Detected macOS - using local data directory")
        return Path(Config.get_data_path('local'))
    elif platform.system() == "Linux":
        logger.info("Detected Linux - using CURC storage directory")
        return Path(Config.get_data_path('curc'))
    else:
        logger.warning(f"Unknown platform: {platform.system()}. Using default.")
        return Path(Config.get_data_path('default'))


def main():
    parser = argparse.ArgumentParser(
        description="OCO-2 footprint spectral analysis (cumulant expansion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python oco_fp_spec_anal.py --date 2018-10-18",
    )
    parser.add_argument('--date',           type=str, required=True,
                        help='Target date (YYYY-MM-DD)')
    parser.add_argument('--data-dir',       type=str, default='./data')
    parser.add_argument('--output-dir',     type=str, default=None)
    parser.add_argument('--visualize',      action='store_true')
    parser.add_argument('--viz-dir',        type=str, default='./visualizations_combined')
    parser.add_argument('--limit-granules', type=int, default=-1)
    parser.add_argument('--delete-ocofiles', action='store_true',
                        help='Delete L1b/Met/CO2Prior files for each orbit after processing')
    parser.add_argument('--single-fit', action='store_true',
                        help='Deprecated, ignored (2026-10-31: no smoothed fit any more)')
    parser.add_argument('--fit-workers', type=int, default=0,
                        help='Processes for the per-sounding fit stage; '
                             '0 = auto (cores - 1), 1 = serial. Results are '
                             'identical regardless of worker count.')
    parser.add_argument('--orbit', action='append', default=None, metavar='ORBIT_ID',
                        help='Process only this orbit directory (e.g. 33124a_TG); '
                             'repeat the flag for several orbits')
    parser.add_argument('--overwrite', action='store_true',
                        help='Refit even when a complete fitting_details file exists')
    parser.add_argument('--lock-stale-hours', type=float, default=4.0,
                        help='Per-orbit lock files older than this are treated as '
                             'left behind by a dead job and taken over')
    parser.add_argument('--tau-wait-minutes', type=float, default=180.0,
                        help='How long to wait for another job that is computing '
                             "an orbit's optical depths or fit before giving up on it")
    parser.add_argument('--telluric-doppler', action='store_true',
                        help='Accepted for launcher compatibility; always on since '
                             '2026-10-31 (telluric + solar Doppler tau, results default '
                             '<storage>/results_doppler)')
    parser.add_argument('--cloud-dist-dir', type=str, default=None,
                        help='Directory with results_<date>.h5 cloud distances when it '
                             'differs from the output dir (default: <storage>/results '
                             'whenever the output dir is not that tree)')
    parser.add_argument('--fit-subdir', type=str, default=FIT_SUBDIR,
                        help=f'Subdirectory of the output dir for fitting_details files '
                             f'(default {FIT_SUBDIR}, the current fit_schema)')
    parser.add_argument('--extra-fits', action='store_true',
                        help='Deprecated, ignored (the order-2/3, _t2 and _am fits are always stored)')
    args = parser.parse_args()

    try:
        target_date = validate_date(args.date)
    except ValueError as e:
        logger.error(str(e))
        return 1

    storage_dir = get_storage_dir()
    data_dir    = storage_dir / "data" if args.data_dir == "./data" else Path(args.data_dir)
    production_results = storage_dir / "results"
    output_dir = Path(args.output_dir) if args.output_dir else storage_dir / "results_doppler"
    cloud_dist_dir = None
    if args.cloud_dist_dir:
        cloud_dist_dir = Path(args.cloud_dist_dir)
    elif output_dir != production_results:
        cloud_dist_dir = production_results
    if args.single_fit or args.extra_fits:
        logger.warning("--single-fit / --extra-fits are deprecated and ignored")
    logger.info(f"Telluric-Doppler fits ({FIT_SCHEMA}): results -> {output_dir}, "
                f"cloud distances <- {cloud_dist_dir or output_dir}")
    viz_dir     = Path(args.viz_dir) if args.visualize else storage_dir / "visualizations_combined"

    fit_workers = args.fit_workers
    if fit_workers <= 0:
        fit_workers = max(1, (os.cpu_count() or 2) - 1)

    run_simulation(
        target_date, data_dir, output_dir,
        limit_granules=args.limit_granules,
        viz_dir=viz_dir,
        visualize=args.visualize,
        delete_ocofiles=args.delete_ocofiles,
        fit_workers=fit_workers,
        lock_stale_hours=args.lock_stale_hours,
        tau_wait_minutes=args.tau_wait_minutes,
        orbits=args.orbit,
        overwrite=args.overwrite,
        cloud_dist_dir=cloud_dist_dir,
        fit_subdir=args.fit_subdir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


""" test codes


python src/spectral/fitting.py \
        --date 2018-10-18 --limit-granules 2 --visualize \
        --data-dir /Users/yuch8913/programming/oco_fp_analysis/data \
        --output-dir /Users/yuch8913/programming/oco_fp_analysis/results \
        --viz-dir /Users/yuch8913/programming/oco_fp_analysis/visualizations_combined
        

"""

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
from abs_util.oco_util import timing

# Re-exports (backward compat: this module was a single file until 2026-07).
from spectral.cumulant_fit import (  # noqa: F401
    LOG_TRANSMITTANCE_MODELS, MAX_KAPPAS, SAVGOL_ORDER, SAVGOL_WINDOW,
    _fit_chunk, _solve_cumulant, compute_transmittance, fit_spectral_model,
    get_design_matrix, log_transmittance_model_1, log_transmittance_model_2,
    log_transmittance_model_3, log_transmittance_model_4,
    log_transmittance_model_5, log_transmittance_model_7,
    log_transmittance_model_9, transmittance_model)
from spectral.orbit_data import (  # noqa: F401
    _discover_orbit_files, _validate_readable_hdf5, cloud_distance_file_path,
    fp_tau_file_is_current, load_orbit_data, load_profile_data,
    load_shared_data, search_oco2_orbit, select_lite_file,
    validate_cloud_distance_file)
from spectral.anomaly import compute_xco2_anomaly  # noqa: F401
from spectral.fit_plots import (  # noqa: F401
    plot_fitting_example, plot_orbit_fitting_examples)


logger = logging.getLogger(__name__)


# ─── Orbit orchestration ───────────────────────────────────────────────────────

def process_orbit(sat, orbit_id, shared_data, fit_order=FIT_ORDER, overwrite=True,
                  dual_fit=True, fit_workers=1):
    """Fit the spectral cumulant model for all soundings in one orbit.

    Workflow
    --------
    1. Load per-orbit L1B radiances and optical depths (load_orbit_data).
    2. Compute transmittances for all soundings at once (compute_transmittance).
    3. Fit the cumulant model in contiguous chunks (_fit_chunk), optionally
       across worker processes; example plots are rendered afterwards.
    4. Extract Lite retrieval variables via vectorised index lookup.
    5. Compute XCO2 anomalies with vectorised lat-window approach.
    6. Write results to fitting_details.h5.

    Parameters
    ----------
    sat         : dict from preprocess()
    orbit_id    : str
    shared_data : dict from load_shared_data()
    fit_order   : (o2a_order, wco2_order, sco2_order)
    overwrite   : bool
    dual_fit    : bool  additionally fit WITHOUT Savitzky-Golay smoothing and
                  store the parallel parameter set as *_fitting_nosg datasets,
                  so the smoothed-vs-raw comparison (M9a) needs no second run.
    fit_workers : int  number of processes for the fit stage; 1 = in-process
                  (results are identical regardless of worker count).
    """
    date        = sat['date'].strftime("%Y-%m-%d")
    output_dir  = f"{sat['result_dir']}/{date}/{orbit_id}"
    h5_output_dir = f"{sat['result_dir']}/fitting_details"
    output_file = f"{h5_output_dir}/fitting_details_{date}_{orbit_id}.h5"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(h5_output_dir, exist_ok=True)
    
    # ── 1. Load orbit data ─────────────────────────────────────────────────
    logger.info(f"[{orbit_id}] Loading orbit data...")
    od = load_orbit_data(sat, orbit_id)
    N  = len(od["sounding_id"])
    
    tags       = ["o2a", "wco2", "sco2"]
    fit_orders = list(fit_order)   # (o2a_order, wco2_order, sco2_order)

    kappa_fitting     = np.full((3, N, MAX_KAPPAS), np.nan)
    intercept_fitting = np.full((3, N), np.nan)
    # Parallel no-Savitzky-Golay fit (raw ln_T), for the smoothing-bias A/B.
    kappa_fitting_nosg     = np.full((3, N, MAX_KAPPAS), np.nan)
    intercept_fitting_nosg = np.full((3, N), np.nan)

    # A cached file from before κ was added (or from before the positivity
    # bounds) lacks the gamma-shape datasets; refit rather than reuse it.
    # Likewise, when dual_fit is requested, a cache without the _nosg
    # datasets must be refitted.
    cached_has_kappa = False
    cached_has_nosg = False
    if os.path.isfile(output_file):
        try:
            with h5py.File(output_file, "r") as f:
                cached_has_kappa = all(
                    key in f for key in ("o2a_kappa", "wco2_kappa", "sco2_kappa")
                )
                cached_has_nosg = all(
                    key in f for key in ("o2a_k1_fitting_nosg",
                                         "wco2_k1_fitting_nosg",
                                         "sco2_k1_fitting_nosg")
                )
        except OSError:
            cached_has_kappa = False
            cached_has_nosg = False

    cache_ok = cached_has_kappa and (cached_has_nosg or not dual_fit)
    if not os.path.isfile(output_file) or overwrite or not cache_ok:
        if os.path.isfile(output_file) and not overwrite and not cache_ok:
            missing = ("κ" if not cached_has_kappa else "_nosg")
            logger.info(
                f"[{orbit_id}] Cached {output_file} has no {missing} datasets; refitting."
            )
        # ── 2. Transmittance for all soundings and bands at once ───────────────
        logger.info(f"[{orbit_id}] Computing transmittances for {N} soundings...")
        T_all    = compute_transmittance(od["radiances"], od["toa_sol"])  # [3, N, 1016]
        ln_T_all = np.where(T_all > 0, np.log(T_all), np.nan)            # [3, N, 1016]

        # ── 3. Chunked spectral fitting (optionally multi-process) ─────────────
        valid_idx = np.where(od["valid_l1b"])[0]
        chunk_size = 256
        n_chunks = max(1, int(np.ceil(len(valid_idx) / chunk_size)))
        chunks = np.array_split(valid_idx, n_chunks)
        logger.info(f"[{orbit_id}] Fitting spectral models for {len(valid_idx)} "
                    f"soundings in {n_chunks} chunks (fit_workers={fit_workers})...")

        def _chunk_args(chunk):
            # Edge channels excluded here once, matching the historical [1:-1].
            return (chunk,
                    od["tau"][:, chunk, 1:-1],
                    ln_T_all[:, chunk, 1:-1],
                    tuple(fit_orders), dual_fit)

        plot_cands = {}
        if fit_workers > 1 and len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=fit_workers) as pool:
                futures = [pool.submit(_fit_chunk, *_chunk_args(c)) for c in chunks]
                results = [f.result() for f in futures]
        else:
            results = [_fit_chunk(*_chunk_args(c)) for c in chunks]

        for n_done, (j_idx, k_c, ic_c, kns_c, icns_c, cands) in enumerate(results, 1):
            kappa_fitting[:, j_idx, :]      = k_c
            intercept_fitting[:, j_idx]     = ic_c
            kappa_fitting_nosg[:, j_idx, :] = kns_c
            intercept_fitting_nosg[:, j_idx] = icns_c
            plot_cands.update(cands)
            if n_done % 10 == 0 or n_done == len(results):
                logger.info(f"  [{orbit_id}] chunk {n_done}/{len(results)} merged")

        # Example plots, outside the (possibly multi-process) numeric loop.
        # Replays the sequential selection rule: first successful fit per band,
        # plus every 1000th sounding — plot_cands is a superset of these.
        plot_done = {tag: False for tag in tags}
        for i_band, j in sorted(plot_cands, key=lambda t: (t[1], t[0])):
            tag, band_order = tags[i_band], fit_orders[i_band]
            if not (not plot_done[tag] or j % 1000 == 0):
                continue
            tau_j  = od["tau"][i_band, j][1:-1]
            ln_T_j = ln_T_all[i_band, j][1:-1]
            mask = ~np.isnan(ln_T_j) & ~np.isnan(tau_j)
            plot_fitting_example(
                tag, int(od["fp_number"][j]), int(od["sounding_id"][j]),
                od["wvl"][i_band, od["fp_number"][j]],
                od["radiances"][i_band, j], T_all[i_band, j],
                tau_j[mask], ln_T_j[mask], plot_cands[(i_band, j)],
                band_order, output_dir,
            )
            plot_done[tag] = True

    else:
        logger.info(f"[{orbit_id}] Output file {output_file} already exists and overwrite=False")
        logger.info(f"Loading existing data from {output_file}...")
        with h5py.File(output_file, "r") as f:
            kappa_fitting[0, :, 0] = f["o2a_k1_fitting"][...]
            kappa_fitting[0, :, 1] = f["o2a_k2_fitting"][...]
            kappa_fitting[0, :, 2] = f["o2a_k3_fitting"][...]
            kappa_fitting[0, :, 3] = f["o2a_k4_fitting"][...]
            kappa_fitting[0, :, 4] = f["o2a_k5_fitting"][...]
            intercept_fitting[0]   = f["o2a_intercept_fitting"][...]
            kappa_fitting[1, :, 0] = f["wco2_k1_fitting"][...]
            kappa_fitting[1, :, 1] = f["wco2_k2_fitting"][...]
            kappa_fitting[1, :, 2] = f["wco2_k3_fitting"][...]
            kappa_fitting[1, :, 3] = f["wco2_k4_fitting"][...]
            kappa_fitting[1, :, 4] = f["wco2_k5_fitting"][...]
            intercept_fitting[1]   = f["wco2_intercept_fitting"][...]
            kappa_fitting[2, :, 0] = f["sco2_k1_fitting"][...]
            kappa_fitting[2, :, 1] = f["sco2_k2_fitting"][...]
            kappa_fitting[2, :, 2] = f["sco2_k3_fitting"][...]
            kappa_fitting[2, :, 3] = f["sco2_k4_fitting"][...]
            kappa_fitting[2, :, 4] = f["sco2_k5_fitting"][...]
            intercept_fitting[2]   = f["sco2_intercept_fitting"][...]
            if cached_has_nosg:
                for _ib, _tag in enumerate(tags):
                    for _ik in range(MAX_KAPPAS):
                        _key = f"{_tag}_k{_ik + 1}_fitting_nosg"
                        if _key in f:
                            kappa_fitting_nosg[_ib, :, _ik] = f[_key][...]
                    _ikey = f"{_tag}_intercept_fitting_nosg"
                    if _ikey in f:
                        intercept_fitting_nosg[_ib] = f[_ikey][...]
            logger.info(f"Loaded existing fitting results from {output_file}. Skipping fitting step.")
        logger.info(f"[{orbit_id}] Writing example fitting plots from cached fitting input.")
        plot_orbit_fitting_examples(od, fit_orders, output_dir)
        
    # ── 3b. Exponential of fitting intercepts ─────────────────────────────
    exp_intercept_o2a  = np.exp(intercept_fitting[0])
    exp_intercept_wco2 = np.exp(intercept_fitting[1])
    exp_intercept_sco2 = np.exp(intercept_fitting[2])

    # ── 3c. Gamma shape parameter κ = ⟨l'⟩²/var(l') = k1²/k2 ───────────────
    # Derived from the first two cumulants; undefined where k2 ≤ 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_kappa = kappa_fitting[:, :, 0]**2 / kappa_fitting[:, :, 1]  # [3, N]
    gamma_kappa[kappa_fitting[:, :, 1] <= 0] = np.nan

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

    # ── 6. XCO2 anomaly (vectorised lat-window) ────────────────────────────
    logger.info(f"[{orbit_id}] Computing XCO2 anomalies...")
    anomaly_args = production_anomaly_args()
    xco2_raw_anomaly = compute_xco2_anomaly(od["lat"], fp_cld_dist, lt_xco2_raw, **anomaly_args)
    ref_extra_vars = {
        "o2a_k1":      kappa_fitting[0, :, 0],
        "o2a_k2":      kappa_fitting[0, :, 1],
        "o2a_k3":      kappa_fitting[0, :, 2],
        "wco2_k1":     kappa_fitting[1, :, 0],
        "wco2_k2":     kappa_fitting[1, :, 1],
        "wco2_k3":     kappa_fitting[1, :, 2],
        "sco2_k1":     kappa_fitting[2, :, 0],
        "sco2_k2":     kappa_fitting[2, :, 1],
        "sco2_k3":     kappa_fitting[2, :, 2],
        "alb_o2a":     lt_alb_o2a,
        "alb_wco2":    lt_alb_wco2,
        "alb_sco2":    lt_alb_sco2,
        "exp_int_o2a":  exp_intercept_o2a,
        "exp_int_wco2": exp_intercept_wco2,
        "exp_int_sco2": exp_intercept_sco2,
    }
    xco2_bc_anomaly, ref_means, ref_stds = compute_xco2_anomaly(
        od["lat"], fp_cld_dist, lt_xco2_bc, extra_vars=ref_extra_vars, **anomaly_args)

    # ── 6b. Second reference set with stricter min_cld_dist=15 km ─────────
    anomaly_args_15 = production_anomaly_args(min_cld_dist=15.0)
    xco2_raw_anomaly_15 = compute_xco2_anomaly(od["lat"], fp_cld_dist, lt_xco2_raw, **anomaly_args_15)
    xco2_bc_anomaly_15, ref_means_15, ref_stds_15 = compute_xco2_anomaly(
        od["lat"], fp_cld_dist, lt_xco2_bc, extra_vars=ref_extra_vars, **anomaly_args_15)

    # ── 6c. Third reference set with looser min_cld_dist=5 km ─────────────
    anomaly_args_05 = production_anomaly_args(min_cld_dist=5.0)
    xco2_raw_anomaly_05 = compute_xco2_anomaly(od["lat"], fp_cld_dist, lt_xco2_raw, **anomaly_args_05)
    xco2_bc_anomaly_05, ref_means_05, ref_stds_05 = compute_xco2_anomaly(
        od["lat"], fp_cld_dist, lt_xco2_bc, extra_vars=ref_extra_vars, **anomaly_args_05)

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
    output_dict = {
        # Cumulant coefficients per band
        "o2a_k1_fitting":         kappa_fitting[0, :, 0],
        "o2a_k2_fitting":         kappa_fitting[0, :, 1],
        "o2a_k3_fitting":         kappa_fitting[0, :, 2],
        "o2a_k4_fitting":         kappa_fitting[0, :, 3],
        "o2a_k5_fitting":         kappa_fitting[0, :, 4],
        "o2a_intercept_fitting":  intercept_fitting[0],
        "wco2_k1_fitting":        kappa_fitting[1, :, 0],
        "wco2_k2_fitting":        kappa_fitting[1, :, 1],
        "wco2_k3_fitting":        kappa_fitting[1, :, 2],
        "wco2_k4_fitting":        kappa_fitting[1, :, 3],
        "wco2_k5_fitting":        kappa_fitting[1, :, 4],
        "wco2_intercept_fitting": intercept_fitting[1],
        "sco2_k1_fitting":        kappa_fitting[2, :, 0],
        "sco2_k2_fitting":        kappa_fitting[2, :, 1],
        "sco2_k3_fitting":        kappa_fitting[2, :, 2],
        "sco2_k4_fitting":        kappa_fitting[2, :, 3],
        "sco2_k5_fitting":        kappa_fitting[2, :, 4],
        "sco2_intercept_fitting": intercept_fitting[2],
        # No-Savitzky-Golay parallel fit (raw ln_T), for the smoothing-bias A/B.
        # Populated only when dual_fit=True (removed below otherwise, so a
        # cache without them is refitted when dual_fit is later enabled).
        "o2a_k1_fitting_nosg":         kappa_fitting_nosg[0, :, 0],
        "o2a_k2_fitting_nosg":         kappa_fitting_nosg[0, :, 1],
        "o2a_k3_fitting_nosg":         kappa_fitting_nosg[0, :, 2],
        "o2a_k4_fitting_nosg":         kappa_fitting_nosg[0, :, 3],
        "o2a_k5_fitting_nosg":         kappa_fitting_nosg[0, :, 4],
        "o2a_intercept_fitting_nosg":  intercept_fitting_nosg[0],
        "wco2_k1_fitting_nosg":        kappa_fitting_nosg[1, :, 0],
        "wco2_k2_fitting_nosg":        kappa_fitting_nosg[1, :, 1],
        "wco2_k3_fitting_nosg":        kappa_fitting_nosg[1, :, 2],
        "wco2_k4_fitting_nosg":        kappa_fitting_nosg[1, :, 3],
        "wco2_k5_fitting_nosg":        kappa_fitting_nosg[1, :, 4],
        "wco2_intercept_fitting_nosg": intercept_fitting_nosg[1],
        "sco2_k1_fitting_nosg":        kappa_fitting_nosg[2, :, 0],
        "sco2_k2_fitting_nosg":        kappa_fitting_nosg[2, :, 1],
        "sco2_k3_fitting_nosg":        kappa_fitting_nosg[2, :, 2],
        "sco2_k4_fitting_nosg":        kappa_fitting_nosg[2, :, 3],
        "sco2_k5_fitting_nosg":        kappa_fitting_nosg[2, :, 4],
        "sco2_intercept_fitting_nosg": intercept_fitting_nosg[2],
        # Gamma shape parameter κ = k1²/k2 per band (NaN where k2 ≤ 0)
        "o2a_kappa":   gamma_kappa[0],
        "wco2_kappa":  gamma_kappa[1],
        "sco2_kappa":  gamma_kappa[2],
        # Geometry
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
        "xco2_raw_anomaly": xco2_raw_anomaly,
        "xco2_bc_anomaly":  xco2_bc_anomaly,
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
        # Fitting intercept (exponential)
        "exp_intercept_o2a":  exp_intercept_o2a,
        "exp_intercept_wco2": exp_intercept_wco2,
        "exp_intercept_sco2": exp_intercept_sco2,
        # Reference clear-sky statistics (mean and std over lat-window, same mask as xco2_bc_anomaly)
        "ref_o2a_k1_mean":       ref_means["o2a_k1"],
        "ref_o2a_k1_std":        ref_stds["o2a_k1"],
        "ref_o2a_k2_mean":       ref_means["o2a_k2"],
        "ref_o2a_k2_std":        ref_stds["o2a_k2"],
        "ref_o2a_k3_mean":       ref_means["o2a_k3"],
        "ref_o2a_k3_std":        ref_stds["o2a_k3"],
        "ref_wco2_k1_mean":      ref_means["wco2_k1"],
        "ref_wco2_k1_std":       ref_stds["wco2_k1"],
        "ref_wco2_k2_mean":      ref_means["wco2_k2"],
        "ref_wco2_k2_std":       ref_stds["wco2_k2"],
        "ref_wco2_k3_mean":      ref_means["wco2_k3"],
        "ref_wco2_k3_std":       ref_stds["wco2_k3"],
        "ref_sco2_k1_mean":      ref_means["sco2_k1"],
        "ref_sco2_k1_std":       ref_stds["sco2_k1"],
        "ref_sco2_k2_mean":      ref_means["sco2_k2"],
        "ref_sco2_k2_std":       ref_stds["sco2_k2"],
        "ref_sco2_k3_mean":      ref_means["sco2_k3"],
        "ref_sco2_k3_std":       ref_stds["sco2_k3"],
        "ref_alb_o2a_mean":      ref_means["alb_o2a"],
        "ref_alb_o2a_std":       ref_stds["alb_o2a"],
        "ref_alb_wco2_mean":     ref_means["alb_wco2"],
        "ref_alb_wco2_std":      ref_stds["alb_wco2"],
        "ref_alb_sco2_mean":     ref_means["alb_sco2"],
        "ref_alb_sco2_std":      ref_stds["alb_sco2"],
        "ref_exp_int_o2a_mean":  ref_means["exp_int_o2a"],
        "ref_exp_int_o2a_std":   ref_stds["exp_int_o2a"],
        "ref_exp_int_wco2_mean": ref_means["exp_int_wco2"],
        "ref_exp_int_wco2_std":  ref_stds["exp_int_wco2"],
        "ref_exp_int_sco2_mean": ref_means["exp_int_sco2"],
        "ref_exp_int_sco2_std":  ref_stds["exp_int_sco2"],
        # ── Reference set with stricter min_cld_dist=15 km (r15 prefix) ──
        "xco2_raw_anomaly_r15":      xco2_raw_anomaly_15,
        "xco2_bc_anomaly_r15":       xco2_bc_anomaly_15,
        "r15_o2a_k1_mean":           ref_means_15["o2a_k1"],
        "r15_o2a_k1_std":            ref_stds_15["o2a_k1"],
        "r15_o2a_k2_mean":           ref_means_15["o2a_k2"],
        "r15_o2a_k2_std":            ref_stds_15["o2a_k2"],
        "r15_o2a_k3_mean":           ref_means_15["o2a_k3"],
        "r15_o2a_k3_std":            ref_stds_15["o2a_k3"],
        "r15_wco2_k1_mean":          ref_means_15["wco2_k1"],
        "r15_wco2_k1_std":           ref_stds_15["wco2_k1"],
        "r15_wco2_k2_mean":          ref_means_15["wco2_k2"],
        "r15_wco2_k2_std":           ref_stds_15["wco2_k2"],
        "r15_wco2_k3_mean":          ref_means_15["wco2_k3"],
        "r15_wco2_k3_std":           ref_stds_15["wco2_k3"],
        "r15_sco2_k1_mean":          ref_means_15["sco2_k1"],
        "r15_sco2_k1_std":           ref_stds_15["sco2_k1"],
        "r15_sco2_k2_mean":          ref_means_15["sco2_k2"],
        "r15_sco2_k2_std":           ref_stds_15["sco2_k2"],
        "r15_sco2_k3_mean":          ref_means_15["sco2_k3"],
        "r15_sco2_k3_std":           ref_stds_15["sco2_k3"],
        "r15_alb_o2a_mean":          ref_means_15["alb_o2a"],
        "r15_alb_o2a_std":           ref_stds_15["alb_o2a"],
        "r15_alb_wco2_mean":         ref_means_15["alb_wco2"],
        "r15_alb_wco2_std":          ref_stds_15["alb_wco2"],
        "r15_alb_sco2_mean":         ref_means_15["alb_sco2"],
        "r15_alb_sco2_std":          ref_stds_15["alb_sco2"],
        "r15_exp_int_o2a_mean":      ref_means_15["exp_int_o2a"],
        "r15_exp_int_o2a_std":       ref_stds_15["exp_int_o2a"],
        "r15_exp_int_wco2_mean":     ref_means_15["exp_int_wco2"],
        "r15_exp_int_wco2_std":      ref_stds_15["exp_int_wco2"],
        "r15_exp_int_sco2_mean":     ref_means_15["exp_int_sco2"],
        "r15_exp_int_sco2_std":      ref_stds_15["exp_int_sco2"],
        # ── Reference set with looser min_cld_dist=5 km (r05 prefix) ──
        "xco2_raw_anomaly_r05":      xco2_raw_anomaly_05,
        "xco2_bc_anomaly_r05":       xco2_bc_anomaly_05,
        "r05_o2a_k1_mean":           ref_means_05["o2a_k1"],
        "r05_o2a_k1_std":            ref_stds_05["o2a_k1"],
        "r05_o2a_k2_mean":           ref_means_05["o2a_k2"],
        "r05_o2a_k2_std":            ref_stds_05["o2a_k2"],
        "r05_o2a_k3_mean":           ref_means_05["o2a_k3"],
        "r05_o2a_k3_std":            ref_stds_05["o2a_k3"],
        "r05_wco2_k1_mean":          ref_means_05["wco2_k1"],
        "r05_wco2_k1_std":           ref_stds_05["wco2_k1"],
        "r05_wco2_k2_mean":          ref_means_05["wco2_k2"],
        "r05_wco2_k2_std":           ref_stds_05["wco2_k2"],
        "r05_wco2_k3_mean":          ref_means_05["wco2_k3"],
        "r05_wco2_k3_std":           ref_stds_05["wco2_k3"],
        "r05_sco2_k1_mean":          ref_means_05["sco2_k1"],
        "r05_sco2_k1_std":           ref_stds_05["sco2_k1"],
        "r05_sco2_k2_mean":          ref_means_05["sco2_k2"],
        "r05_sco2_k2_std":           ref_stds_05["sco2_k2"],
        "r05_sco2_k3_mean":          ref_means_05["sco2_k3"],
        "r05_sco2_k3_std":           ref_stds_05["sco2_k3"],
        "r05_alb_o2a_mean":          ref_means_05["alb_o2a"],
        "r05_alb_o2a_std":           ref_stds_05["alb_o2a"],
        "r05_alb_wco2_mean":         ref_means_05["alb_wco2"],
        "r05_alb_wco2_std":          ref_stds_05["alb_wco2"],
        "r05_alb_sco2_mean":         ref_means_05["alb_sco2"],
        "r05_alb_sco2_std":          ref_stds_05["alb_sco2"],
        "r05_exp_int_o2a_mean":      ref_means_05["exp_int_o2a"],
        "r05_exp_int_o2a_std":       ref_stds_05["exp_int_o2a"],
        "r05_exp_int_wco2_mean":     ref_means_05["exp_int_wco2"],
        "r05_exp_int_wco2_std":      ref_stds_05["exp_int_wco2"],
        "r05_exp_int_sco2_mean":     ref_means_05["exp_int_sco2"],
        "r05_exp_int_sco2_std":      ref_stds_05["exp_int_sco2"],
    }
    
    
    output_dict.update(ak_arrays)   # [N, 20] column-operator datasets

    if not dual_fit:
        # Drop the (all-NaN) _nosg datasets so their absence marks the cache
        # as needing a refit if dual_fit is enabled later.
        output_dict = {k: v for k, v in output_dict.items()
                       if not k.endswith('_fitting_nosg')}

    # float64 → float32 on write (except keys needing full precision) plus
    # shuffle+gzip: shrinks fitting_details.h5 severalfold and speeds the
    # downstream build_feature_dataset reads.  Integer / string datasets are
    # stored as-is.
    keep_float64 = {"time"}   # seconds since epoch: float32 would lose ~minutes
    with h5py.File(output_file, "w") as f_out:
        for key, value in output_dict.items():
            arr = np.asarray(value)
            if arr.dtype == np.float64 and key not in keep_float64:
                arr = arr.astype(np.float32)
            if arr.ndim > 0 and arr.size > 0:
                f_out.create_dataset(key, data=arr, compression="gzip",
                                     compression_opts=4, shuffle=True)
            else:
                f_out.create_dataset(key, data=arr)

    logger.info(f"[{orbit_id}] Done.")


# ─── Pipeline ──────────────────────────────────────────────────────────────────

@timing
def preprocess(target_date, data_dir="data", result_dir="results", limit_granules=-1):
    """Discover orbit files and compute per-footprint optical depths.

    Returns sat0 dict consumed by run_simulation and load_shared_data.
    """
    date = target_date
    year = date.year
    doy  = date.timetuple().tm_yday

    oco2_orbit_list = search_oco2_orbit(date, data_dir=data_dir)
    if limit_granules > 0:
        oco2_orbit_list = oco2_orbit_list[:limit_granules]

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
    }

    # randomize oco2_orbit_list for multiple runs to avoid always processing orbits in the same order (which may bias results if some orbits are more likely to fail or have issues)
    oco2_orbit_list = np.random.RandomState().permutation(oco2_orbit_list)
    for orbit_id in oco2_orbit_list:
        orbit_dir = f"{OCO2_data_dir}/{orbit_id}"
        sat0[orbit_id] = _discover_orbit_files(orbit_dir)
        for product_key, product_path in sat0[orbit_id].items():
            _validate_readable_hdf5(product_path, product_key, date)
        
        date_str = date.strftime("%Y-%m-%d")
        fp_tau_file = os.path.abspath(f"{result_dir}/{date_str}/{orbit_id}/fp_tau_combined.h5")
        os.makedirs(os.path.abspath(f"{result_dir}/{date_str}/{orbit_id}"), exist_ok=True)
        if not fp_tau_file_is_current(fp_tau_file):
            print(f"Computing footprint optical depths for orbit {orbit_id}...")
            oco_fp_atm_abs(
                sat=sat0,
                o2mix=0.20935,
                output=fp_tau_file,
                oco_files_dict=sat0[orbit_id],
                oco_nc_file=lite_nc_file,
                overwrite=True,
            )
        sat0[orbit_id]["fp_tau_file"] = fp_tau_file

    return sat0


@timing
def run_simulation(target_date, data_dir, result_dir,
                   limit_granules=-1,
                   viz_dir=None,
                   visualize=True,
                   delete_ocofiles=False,
                   dual_fit=True,
                   fit_workers=1):
    """Top-level pipeline: preprocess → fit → analyse."""
    validate_cloud_distance_file(
        cloud_distance_file_path(result_dir, target_date),
        target_date,
    )
    sat0 = preprocess(target_date, data_dir, result_dir, limit_granules)

    fit_order = FIT_ORDER  # (o2a_order, wco2_order, sco2_order); see constants.py

    # Load Lite file and cloud distances once, shared across all orbits
    shared_data = load_shared_data(sat0)

    for orbit_id in sat0["orbit_list"]:
        date       = sat0['date'].strftime("%Y-%m-%d")
        h5_output_dir = f"{sat0['result_dir']}/fitting_details"
        output_file = f"{h5_output_dir}/fitting_details_{date}_{orbit_id}.h5"
        
        # if os.path.isfile(output_file):
        #     logger.info(f"[{orbit_id}] Output already exists. Skipping orbit.")
        #     continue
        
        process_orbit(sat0, orbit_id, shared_data, fit_order=fit_order, overwrite=False,
                      dual_fit=dual_fit, fit_workers=fit_workers)

        if delete_ocofiles:
            for key in ("oco_l1b", "oco_met", "oco_co2prior"):
                fpath = sat0[orbit_id].get(key)
                if fpath and os.path.isfile(fpath):
                    os.remove(fpath)
                    logger.info(f"Deleted input file: {fpath}")


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
                        help='Skip the parallel no-Savitzky-Golay fit (halves fit time; '
                             'the *_fitting_nosg comparison datasets are then not produced)')
    parser.add_argument('--fit-workers', type=int, default=0,
                        help='Processes for the per-sounding fit stage; '
                             '0 = auto (cores - 1), 1 = serial. Results are '
                             'identical regardless of worker count.')
    args = parser.parse_args()

    try:
        target_date = validate_date(args.date)
    except ValueError as e:
        logger.error(str(e))
        return 1

    storage_dir = get_storage_dir()
    data_dir    = storage_dir / "data" if args.data_dir == "./data" else Path(args.data_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else storage_dir / "results"
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
        dual_fit=not args.single_fit,
        fit_workers=fit_workers,
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

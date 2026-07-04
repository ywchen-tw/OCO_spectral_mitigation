"""Per-date and per-orbit input loading for the spectral-fit pipeline.

Date-level shared data (cloud distances + Lite variables), per-orbit L1B
radiances / optical depths, raw Met + CO2Prior profiles, and the orbit-file
discovery / validation helpers (Lite version selection, HDF5 readability,
fp_tau cache currency).

Split out of fitting.py (2026-07, review §7.4); fitting.py re-exports the
public names.
"""

# When run directly / imported with only src/spectral on sys.path, add src/
# so sibling top-level modules (utils, abs_util, constants, config) resolve.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

# Must be set before any HDF5 library call; Lustre does not support POSIX
# advisory locks and HDF5 >= 1.10 raises NC_EHDF (-101) without this.
_os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import glob
import logging
import os
from datetime import datetime

import h5py
import numpy as np
from netCDF4 import Dataset as dataset

from utils import oco2_rad_nadir
from abs_util.ils_tau import TAU_CONVOLUTION_VERSION

logger = logging.getLogger(__name__)


# ─── Shared data loading (once per date) ──────────────────────────────────────

def load_shared_data(sat):
    """Load date-level inputs that are shared across all orbits.

    Reads the cloud-distance HDF5 produced by demo_combined.py and the
    OCO-2 Lite NetCDF file *once*, then builds O(1) index dicts so that
    per-sounding lookups inside the orbit loop are cheap.

    Parameters
    ----------
    sat : dict
        Pipeline state built by preprocess().  Must contain keys:
        'date', 'result_dir', 'oco_lite'.

    Returns
    -------
    dict with keys
        cld_dist_index : {sounding_id (int) -> nearest_cloud_distance_km (float)}
        lite_index     : {sounding_id (int) -> row index in Lite arrays (int)}
        lite           : dict of 1-D numpy arrays keyed by variable name
    """
    date = sat['date'].strftime("%Y-%m-%d")

    # --- Cloud distances (output of demo_combined.py) ---
    cld_dist_file = cloud_distance_file_path(sat["result_dir"], date)
    validate_cloud_distance_file(cld_dist_file, date)
    logger.info(f"Loading cloud distances from {cld_dist_file}")
    with h5py.File(cld_dist_file, "r") as f:
        cld_snd_id  = f["sounding_id"][...].astype(np.int64)
        cld_dist_km = f["nearest_cloud_distance_km"][...].astype(np.float64)
        if "weighted_cloud_distance_km" in f:
            weighted_cld_dist_km = f["weighted_cloud_distance_km"][...].astype(np.float64)
        else:
            weighted_cld_dist_km = np.full_like(cld_dist_km, np.nan, dtype=np.float64)
    cld_dist_index = dict(zip(cld_snd_id.tolist(), cld_dist_km.tolist()))
    weighted_cld_dist_index = dict(zip(cld_snd_id.tolist(), weighted_cld_dist_km.tolist()))

    # --- OCO-2 Lite file ---
    logger.info(f"Loading Lite file {sat['oco_lite']}")
    with dataset(sat["oco_lite"], "r") as nc:
        lt_id = np.array(nc.variables["sounding_id"][:], dtype=np.int64)
        def _load(var):
            return np.array(nc.variables[var][:])
        def _load_grp(grp, var):
            return np.array(nc.groups[grp].variables[var][:])

        lite = {
            "vertex_latitude": _load("vertex_latitude"),
            "vertex_longitude": _load("vertex_longitude"),
            "xco2_apriori":      _load("xco2_apriori"),
            "xco2_corr":         _load("xco2"),
            "time":             _load("time"),
            "co2_ratio_bc":      _load_grp("Preprocessors", "co2_ratio_bc"),
            "h2o_ratio_bc":      _load_grp("Preprocessors", "h2o_ratio_bc"),
            "color_slice_noise_ratio_o2a": _load_grp("Preprocessors", "color_slice_noise_ratio_o2a"),
            "color_slice_noise_ratio_wco2": _load_grp("Preprocessors", "color_slice_noise_ratio_wco2"),
            "color_slice_noise_ratio_sco2": _load_grp("Preprocessors", "color_slice_noise_ratio_sco2"),
            "dp_abp":                _load_grp("Preprocessors", "dp_abp"),
            "h_continuum_o2a":     _load_grp("Preprocessors", "h_continuum_o2a"),
            "h_continuum_wco2":    _load_grp("Preprocessors", "h_continuum_wco2"),
            "h_continuum_sco2":    _load_grp("Preprocessors", "h_continuum_sco2"),
            "max_declocking_o2a": _load_grp("Preprocessors", "max_declocking_o2a"),
            "max_declocking_wco2": _load_grp("Preprocessors", "max_declocking_wco2"),
            "max_declocking_sco2": _load_grp("Preprocessors", "max_declocking_sco2"),
            "xco2_strong_idp": _load_grp("Preprocessors", "xco2_strong_idp"),
            "xco2_weak_idp": _load_grp("Preprocessors", "xco2_weak_idp"),
            "xco2_raw":          _load_grp("Retrieval", "xco2_raw"),
            "airmass":           _load_grp("Sounding",  "airmass"),
            "h2o_scale":         _load_grp("Retrieval", "h2o_scale"),
            "deltaT":            _load_grp("Retrieval", "deltaT"),
            "psurf":             _load_grp("Retrieval", "psurf"),
            "dp":                _load_grp("Retrieval", "dp"),
            "dp_o2a":            _load_grp("Retrieval", "dp_o2a"),
            "dp_sco2":           _load_grp("Retrieval", "dp_sco2"),
            "dpfrac":             _load_grp("Retrieval", "dpfrac"),
            "co2_grad_del":      _load_grp("Retrieval", "co2_grad_del"),
            "albedo_o2a":        _load_grp("Retrieval", "albedo_o2a"),
            "albedo_wco2":       _load_grp("Retrieval", "albedo_wco2"),
            "albedo_sco2":       _load_grp("Retrieval", "albedo_sco2"),
            "aod_total":         _load_grp("Retrieval", "aod_total"),
            "aod_bc":            _load_grp("Retrieval", "aod_bc"),
            "aod_dust":           _load_grp("Retrieval", "aod_dust"),
            "aod_ice":            _load_grp("Retrieval", "aod_ice"),
            "aod_water":          _load_grp("Retrieval", "aod_water"),
            "aod_oc":          _load_grp("Retrieval", "aod_oc"),
            "aod_seasalt":         _load_grp("Retrieval", "aod_seasalt"),
            "aod_strataer":       _load_grp("Retrieval", "aod_strataer"),
            "aod_sulfate":        _load_grp("Retrieval", "aod_sulfate"),
            "dust_height":       _load_grp("Retrieval", "dust_height"),
            "ice_height":       _load_grp("Retrieval", "ice_height"),
            "water_height":       _load_grp("Retrieval", "water_height"),
            "dws":                _load_grp("Retrieval", "dws"),
            "fs_rel":            _load_grp("Retrieval", "fs_rel"),
            "altitude":          _load_grp("Sounding",  "altitude"),
            "altitude_std":      _load_grp("Sounding",  "altitude_stddev"),
            "snr_o2a":             _load_grp("Sounding",  "snr_o2a"),
            "snr_wco2":            _load_grp("Sounding",  "snr_wco2"),
            "snr_sco2":            _load_grp("Sounding",  "snr_sco2"),
            "qf":                _load("xco2_quality_flag"),
            "sfc_type":          _load_grp("Retrieval", "surface_type"),
            "operation_mode":  _load_grp("Sounding",  "operation_mode"), # 0=Nadir, 1=Glint, 2=Target, 3=Transition"
            "windspeed":         _load_grp("Retrieval", "windspeed"),
            "windspeed_apriori": _load_grp("Retrieval", "windspeed_apriori"),
            "glint_angle":        _load_grp("Sounding",  "glint_angle"),
            "polarization_angle": _load_grp("Sounding",  "polarization_angle"),
            "saa": _load_grp("Sounding",  "solar_azimuth_angle"),
            "vaa": _load_grp("Sounding",  "sensor_azimuth_angle"),
            "s31": _load_grp("Retrieval",  "s31"),
            "s32": _load_grp("Retrieval",  "s32"),
            "snow_flag": _load_grp("Retrieval",  "snow_flag"),
            "t700": _load_grp("Retrieval",  "t700"),
            "tcwv": _load_grp("Retrieval",  "tcwv"),
            # Spectral-fit-quality diagnostics (candidate cloud-contamination
            # fingerprints; admissible — derived from the sounding's own spectrum/fit).
            "chi2_o2a":  _load_grp("Retrieval", "chi2_o2a"),
            "chi2_wco2": _load_grp("Retrieval", "chi2_wco2"),
            "chi2_sco2": _load_grp("Retrieval", "chi2_sco2"),
            "rms_rel_o2a":  _load_grp("Retrieval", "rms_rel_o2a"),
            "rms_rel_wco2": _load_grp("Retrieval", "rms_rel_wco2"),
            "rms_rel_sco2": _load_grp("Retrieval", "rms_rel_sco2"),
            "eof3_1_rel":      _load_grp("Retrieval", "eof3_1_rel"),
            "diverging_steps": _load_grp("Retrieval", "diverging_steps"),
            "xco2_uncertainty": _load("xco2_uncertainty"),   # root-level var
            # 20-level column operator (root-level, [n, 20]) — carried through to
            # fitting_details/parquet so TCCON AK/prior harmonization does not
            # need to reopen the Lite files (workspace/ak_harmonize.py).
            "xco2_averaging_kernel": _load("xco2_averaging_kernel"),
            "pressure_weight":       _load("pressure_weight"),
            "co2_profile_apriori":   _load("co2_profile_apriori"),
            "pressure_levels":       _load("pressure_levels"),
        }

    lite_index = {int(sid): i for i, sid in enumerate(lt_id)}
    logger.info(
        f"Shared data loaded: {len(cld_dist_index)} cloud-dist entries, "
        f"{len(lite_index)} Lite soundings."
    )
    return {
        "cld_dist_index": cld_dist_index,
        "weighted_cld_dist_index": weighted_cld_dist_index,
        "lite_index": lite_index,
        "lite": lite,
    }


def cloud_distance_file_path(result_dir, date):
    """Return the date-level cloud-distance HDF5 path produced by demo_combined.py."""
    if isinstance(date, datetime):
        date = date.strftime("%Y-%m-%d")
    return os.path.join(str(result_dir), f"results_{date}.h5")


def validate_cloud_distance_file(cld_dist_file, date):
    """Fail before expensive tau work if demo_combined.py did not finish."""
    if isinstance(date, datetime):
        date = date.strftime("%Y-%m-%d")

    if not os.path.isfile(cld_dist_file):
        raise FileNotFoundError(
            "Cloud-distance HDF5 is missing; spectral fitting cannot continue:\n"
            f"  expected: {cld_dist_file}\n"
            "This file is produced by workspace/demo_combined.py Step 5. "
            f"Run workspace/demo_combined.py --date {date} and ensure it completes, "
            "or pass --output-dir to fitting.py if demo_combined.py wrote results elsewhere."
        )

    try:
        with h5py.File(cld_dist_file, "r") as h5f:
            required = ("sounding_id", "nearest_cloud_distance_km")
            missing = [name for name in required if name not in h5f]
            if missing:
                raise KeyError(f"missing dataset(s): {', '.join(missing)}")
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise OSError(
            "Cloud-distance HDF5 is corrupted or incomplete:\n"
            f"  {cld_dist_file}\n"
            f"Re-run workspace/demo_combined.py --date {date} --force-recompute "
            "to regenerate it."
        ) from exc


# ─── Per-orbit data loading ────────────────────────────────────────────────────

def load_orbit_data(sat, orbit_id):
    """Load L1B radiances and per-footprint optical depths for one orbit.

    Builds an O(1) lookup dict from (sounding_id, fp_number) to along-track
    index so that radiance/position extraction is fully vectorised.

    Parameters
    ----------
    sat      : dict from preprocess()
    orbit_id : str, e.g. "22845a"

    Returns
    -------
    dict with keys
        sounding_id : int64 [N]
        fp_number   : int   [N]
        lon         : float [N]   (NaN where L1B match not found)
        lat         : float [N]
        radiances   : float [3, N, 1016]  bands: o2a, wco2, sco2
        tau         : float [3, N, 1016]  sco2 reuses wco2 tau (same as original)
        toa_sol     : float [3, N, 1016]  sco2 reuses wco2 toa_sol
        valid_l1b   : bool  [N]   True where L1B sounding was found
    """
    # --- Per-footprint optical depths (from oco_fp_atm_abs) ---
    # tau / toa_sol are stored float64 but carried as float32 (halves the five
    # [3, N, 1016] orbit arrays); the per-sounding fits promote back to float64.
    m1 = 0.5 # coefficient for Stokes Vector I signal in the sensor
    fp_tau_file = sat[orbit_id]["fp_tau_file"]
    with h5py.File(fp_tau_file, "r") as f:
        fp_sounding_id = f["sounding_id"][...].astype(np.int64)
        fp_number      = f["fp_number"][...].astype(int)
        fp_sza         = f["sza"][...]
        fp_vza         = f["vza"][...]
        o2a_tau        = f["o2a_tau_output"][...].astype(np.float32)
        o2a_toa_sol    = (f["o2a_toa_sol_output"][...] * m1).astype(np.float32)
        wco2_tau       = f["wco2_tau_output"][...].astype(np.float32)
        wco2_toa_sol   = (f["wco2_toa_sol_output"][...] * m1).astype(np.float32)
        sco2_tau       = f["sco2_tau_output"][...].astype(np.float32)
        sco2_toa_sol   = (f["sco2_toa_sol_output"][...] * m1).astype(np.float32)
    N = len(fp_sounding_id)

    # --- L1B radiances ---
    l1b = oco2_rad_nadir(l1b_file=sat[orbit_id]["oco_l1b"], lt_file=sat["oco_lite"])

    # Vectorised (sounding_id, fp_number) -> along-track index match, one
    # searchsorted per footprint column (replaces the 8 × n_track dict build).
    track_inds = np.full(N, -1, dtype=np.int64)
    for fp in range(8):
        sel = fp_number == fp
        if not sel.any():
            continue
        col = l1b.snd_id[:, fp].astype(np.int64)
        order = np.argsort(col, kind='stable')
        col_sorted = col[order]
        pos = np.searchsorted(col_sorted, fp_sounding_id[sel])
        pos_c = np.clip(pos, 0, len(col_sorted) - 1)
        hit = col_sorted[pos_c] == fp_sounding_id[sel]
        track_inds[sel] = np.where(hit, order[pos_c], -1)
    valid = track_inds >= 0

    fp_lon    = np.full(N, np.nan)
    fp_lat    = np.full(N, np.nan)
    fp_wvls   = np.full((3, 8, 1016), np.nan)  # placeholder; wavelengths are not used in fitting but for plotting
    radiances = np.full((3, N, 1016), np.nan, dtype=np.float32)

    # print("np.array([l1b.get_wvl_o2_a(i) for i in range(8)]) shape:", np.array([l1b.get_wvl_o2_a(i) for i in range(8)]).shape)
    fp_wvls[0, :, :] = np.array([l1b.get_wvl_o2_a(i) for i in range(8)])
    fp_wvls[1, :, :] = np.array([l1b.get_wvl_co2_weak(i) for i in range(8)])
    fp_wvls[2, :, :] = np.array([l1b.get_wvl_co2_strong(i) for i in range(8)])
    

    if valid.any():
        v_track = track_inds[valid]
        v_fp    = fp_number[valid]
        fp_lon[valid]      = l1b.lon_l1b[v_track, v_fp]
        fp_lat[valid]      = l1b.lat_l1b[v_track, v_fp]
        radiances[0][valid] = l1b.rad_o2_a[v_track, v_fp, :]
        radiances[1][valid] = l1b.rad_co2_weak[v_track, v_fp, :]
        radiances[2][valid] = l1b.rad_co2_strong[v_track, v_fp, :]

    tau     = np.stack([o2a_tau,     wco2_tau,     sco2_tau],     axis=0)  # [3, N, 1016]
    toa_sol = np.stack([o2a_toa_sol, wco2_toa_sol, sco2_toa_sol], axis=0)  # [3, N, 1016]

    return {
        "sounding_id": fp_sounding_id,
        "fp_number":   fp_number,
        "lon":         fp_lon,
        "lat":         fp_lat,
        "sza":         fp_sza,
        "vza":         fp_vza,
        "wvl":         fp_wvls,
        "radiances":   radiances,
        "tau":         tau,
        "toa_sol":     toa_sol,
        "valid_l1b":   valid,
    }



# ─── Atmospheric-profile extraction (L2 Met + CO2Prior) ─────────────────────────
# Records the RAW native-grid profiles (72 GEOS levels) plus the per-sounding
# pressure grid and surface pressure.  The resampling onto a common sigma = P/Psurf
# grid (and PCA compression) is done downstream in analysis/build_feature_dataset.py,
# so the choice of vertical grid can change without re-running the expensive
# spectral fit.  All arrays are aligned to the orbit's sounding order.

def _read_profile_source(h5file, sid_key, profile_specs, scalar_specs):
    """Read sounding_id + requested profile/scalar datasets from an L2 granule.

    Returns (sid_index {sid -> flat row}, {name: [M, 72] array}, {name: [M] array}).
    All (frame, footprint, ...) arrays are flattened to (M, ...) with M = frame*8.
    """
    with h5py.File(h5file, "r") as f:
        sid = f[sid_key][...].astype(np.int64).reshape(-1)
        profiles = {name: f[path][...].reshape(sid.size, -1) for name, path in profile_specs.items()}
        scalars  = {name: f[path][...].reshape(-1) for name, path in scalar_specs.items()}
    sid_index = {int(s): i for i, s in enumerate(sid)}
    return sid_index, profiles, scalars


def load_profile_data(met_file, cpr_file, sounding_ids):
    """Extract raw T / specific-humidity / CO2-prior profiles + tropopause info.

    Profiles are kept on their native 72-level GEOS grid; the pressure grid and
    surface pressure are returned alongside so build_feature_dataset.py can resample
    onto a sigma grid.  All arrays are aligned to ``sounding_ids`` (orbit order),
    NaN where a sounding is not matched in the granule.

    Met and CO2Prior share the same pressure grid for a given sounding (identical
    ``sounding_id`` and GEOS source), so a single ``p_profile`` (from Met) is
    returned and used for all three profiles downstream.

    Returns a dict of per-sounding arrays (N = len(sounding_ids)):
        t_profile, q_profile, co2prior_profile, p_profile : [N, 72]
            temperature (K), specific humidity (kg/kg), CO2 prior (mol/mol),
            pressure grid (Pa)
        psurf_met           : [N]  (Pa)  — surface pressure (for sigma = P/Psurf)
        tropopause_pressure : [N]  (Pa)  — blended tropopause pressure
        tropopause_temp     : [N]  (K)
    """
    sids = np.asarray(sounding_ids, dtype=np.int64)
    N = len(sids)

    # L2 Met: temperature + specific-humidity profiles, pressure grid, surface
    # pressure, and tropopause scalars.
    met_idx, met_prof, met_scal = _read_profile_source(
        met_file, "SoundingGeometry/sounding_id",
        profile_specs={
            "T": "Meteorology/temperature_profile_met",
            "Q": "Meteorology/specific_humidity_profile_met",
            "P": "Meteorology/vector_pressure_levels_met",
        },
        scalar_specs={
            "psurf":  "Meteorology/surface_pressure_met",
            "trop_p": "Meteorology/blended_tropopause_pressure_met",
            "trop_t": "Meteorology/tropopause_temperature_met",
        },
    )
    # L2 CO2Prior: CO2 prior profile (on the matching pressure grid).
    cpr_idx, cpr_prof, _ = _read_profile_source(
        cpr_file, "SoundingGeometry/sounding_id",
        profile_specs={"C": "CO2Prior/co2_prior_profile_cpr"},
        scalar_specs={},
    )

    n_lev = met_prof["P"].shape[1]
    met_rows = np.fromiter((met_idx.get(int(s), -1) for s in sids), dtype=np.int64, count=N)
    cpr_rows = np.fromiter((cpr_idx.get(int(s), -1) for s in sids), dtype=np.int64, count=N)
    m_ok, c_ok = met_rows >= 0, cpr_rows >= 0

    out = {
        "t_profile":           np.full((N, n_lev), np.nan, dtype=np.float32),
        "q_profile":           np.full((N, n_lev), np.nan, dtype=np.float32),
        "co2prior_profile":    np.full((N, n_lev), np.nan, dtype=np.float32),
        "p_profile":           np.full((N, n_lev), np.nan, dtype=np.float32),
        "psurf_met":           np.full(N, np.nan, dtype=np.float32),
        "tropopause_pressure": np.full(N, np.nan, dtype=np.float32),
        "tropopause_temp":     np.full(N, np.nan, dtype=np.float32),
    }
    if m_ok.any():
        mr = met_rows[m_ok]
        out["t_profile"][m_ok] = met_prof["T"][mr]
        out["q_profile"][m_ok] = met_prof["Q"][mr]
        out["p_profile"][m_ok] = met_prof["P"][mr]
        out["psurf_met"][m_ok]           = met_scal["psurf"][mr]
        out["tropopause_pressure"][m_ok] = met_scal["trop_p"][mr]
        out["tropopause_temp"][m_ok]     = met_scal["trop_t"][mr]
    if c_ok.any():
        out["co2prior_profile"][c_ok] = cpr_prof["C"][cpr_rows[c_ok]]

    logger.info("load_profile_data: matched %d/%d Met, %d/%d CO2Prior soundings (%d levels)",
                int(m_ok.sum()), N, int(c_ok.sum()), N, n_lev)
    return out



# ─── Pipeline ──────────────────────────────────────────────────────────────────

def search_oco2_orbit(date, data_dir="data"):
    """Return sorted list of orbit subdirectory names for the given date."""
    year = date.year
    doy  = date.timetuple().tm_yday
    return sorted(
        os.path.basename(p)
        for p in glob.glob(f"{data_dir}/OCO2/{year}/{doy:03d}/*")
        if os.path.isdir(p)
    )


def _discover_orbit_files(orbit_dir):
    """Return required OCO-2 product files for one orbit directory."""
    product_patterns = {
        "oco_l1b": "L1b",
        "oco_met": "Met",
        "oco_co2prior": "CPr",
    }
    entries = sorted(glob.glob(f"{orbit_dir}/*"))
    files = [path for path in entries if os.path.isfile(path)]
    orbit_files = {}

    for filepath in files:
        basename = os.path.basename(filepath)
        for key, pattern in product_patterns.items():
            if pattern in basename:
                orbit_files[key] = filepath

    missing = [key for key in product_patterns if key not in orbit_files]
    if missing:
        present = "\n".join(f"  {os.path.basename(path)}" for path in entries) or "  <empty>"
        raise FileNotFoundError(
            "Orbit directory is missing required OCO-2 product files:\n"
            f"  orbit_dir: {orbit_dir}\n"
            f"  missing: {', '.join(missing)}\n"
            "  present files:\n"
            f"{present}\n"
            "Re-run ingestion/download for this date, or remove incomplete orbit "
            "directories before fitting."
        )

    return orbit_files


def _validate_readable_hdf5(filepath, label, date):
    """Fail early for truncated or otherwise unreadable HDF5/NetCDF4 inputs."""
    try:
        if not h5py.is_hdf5(filepath):
            raise OSError("not an HDF5 file")
        with h5py.File(filepath, "r") as h5f:
            list(h5f.keys())
    except (OSError, RuntimeError, ValueError) as exc:
        raise OSError(
            f"{label} file is corrupted or not readable:\n"
            f"  {filepath}\n"
            f"Delete this file and re-run "
            f"workspace/demo_combined.py --date {date.strftime('%Y-%m-%d')} "
            f"--force-download to re-download it."
        ) from exc


def _decode_hdf5_attr(value):
    """Return a readable string for scalar or one-element HDF5 attributes."""
    arr = np.asarray(value)
    if arr.shape:
        value = arr.flat[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _lite_version_label(filepath):
    """Infer the GES DISC Lite collection version from metadata, then filename."""
    text_parts = [os.path.basename(filepath)]

    try:
        with h5py.File(filepath, "r") as h5f:
            for attr_name in (
                "gesdisc_collection",
                "BuildId",
                "CollectionLabel",
                "lite_definition_module",
                "bc_function",
            ):
                if attr_name in h5f.attrs:
                    text_parts.append(_decode_hdf5_attr(h5f.attrs[attr_name]))
    except (OSError, RuntimeError, ValueError):
        return "unreadable"

    text = " ".join(text_parts).lower()
    version_markers = (
        ("11.3r", ("11.3r", "b11.3", "b113", "lite_b113", "b11_3")),
        ("11.2r", ("11.2r", "b11.2", "b112", "lite_b112", "b11_2")),
        ("11.1r", ("11.1r", "b11.1", "b111", "lite_b111", "b11_1")),
        ("11r", ("11r", "b11.0", "b110", "lite_b110")),
        ("10r", ("10r", "b10")),
    )
    for version, markers in version_markers:
        if any(marker in text for marker in markers):
            return version
    return "unknown"


def select_lite_file(nc4_matches, date):
    """Select one Lite file deterministically when multiple versions are present."""
    preferred_versions = (
        ["11.3r", "11.2r", "11.1r", "11r", "10r"]
        if date.year >= 2024
        else ["11.2r", "11.1r", "11r", "10r", "11.3r"]
    )
    version_priority = {version: rank for rank, version in enumerate(preferred_versions)}
    version_priority["unknown"] = 99
    candidates = []

    for filepath in sorted(nc4_matches):
        version = _lite_version_label(filepath)
        if version == "unreadable":
            logger.warning("Ignoring unreadable Lite file candidate: %s", filepath)
            continue
        candidates.append((version_priority.get(version, 99), filepath, version))

    if not candidates:
        raise OSError(
            "No readable L2 Lite .nc4 file found among candidates:\n"
            + "\n".join(f"  {path}" for path in sorted(nc4_matches))
        )

    candidates.sort(key=lambda item: (item[0], os.path.basename(item[1])))
    _, selected, selected_version = candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Multiple L2 Lite files found for %s; selected %s (%s). Candidates: %s",
            date.strftime("%Y-%m-%d"),
            selected,
            selected_version,
            ", ".join(f"{path} ({version})" for _, path, version in candidates),
        )
    return selected


def fp_tau_file_is_current(fp_tau_file):
    """Return True when cached tau uses the current ILS convolution physics."""
    if not os.path.isfile(fp_tau_file):
        return False
    try:
        with h5py.File(fp_tau_file, "r") as f:
            return (
                f.attrs.get("tau_convolution") == TAU_CONVOLUTION_VERSION
                and not bool(f.attrs.get("use_ring", False))
            )
    except OSError:
        return False


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

import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils import oco2_rad_nadir
from netCDF4 import Dataset as dataset
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from analysis.results import k1k2_analysis
from abs_util.fp_atm import oco_fp_atm_abs
from abs_util.ils_tau import TAU_CONVOLUTION_VERSION
from abs_util.oco_util import timing
import argparse
import logging
import glob
from config import Config
from shapely.geometry import Polygon
from pyproj import Transformer


logger = logging.getLogger(__name__)


# ─── Cumulant expansion models ────────────────────────────────────────────────
# ln(T) = -k1*τ + ½k2*τ² - ⅓k3*τ³ + ... + intercept
# Each function corresponds to a truncation order.

def log_transmittance_model_1(tau, k1, intercept):
    return -k1 * tau + intercept

def log_transmittance_model_2(tau, k1, k2, intercept):
    return -k1 * tau + 0.5 * k2 * tau**2 + intercept

def log_transmittance_model_3(tau, k1, k2, k3, intercept):
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + intercept)

def log_transmittance_model_4(tau, k1, k2, k3, k4, intercept):
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            + intercept)

def log_transmittance_model_5(tau, k1, k2, k3, k4, k5, intercept):
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            - (1/5) * k5 * tau**5
            + intercept)

def log_transmittance_model_7(tau, k1, k2, k3, k4, k5, k6, k7, intercept):
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            - (1/5) * k5 * tau**5
            + (1/6) * k6 * tau**6
            - (1/7) * k7 * tau**7
            + intercept)
    
def log_transmittance_model_9(tau, k1, k2, k3, k4, k5, k6, k7, k8, k9, intercept):
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            - (1/5) * k5 * tau**5
            + (1/6) * k6 * tau**6
            - (1/7) * k7 * tau**7
            + (1/8) * k8 * tau**8
            - (1/9) * k9 * tau**9
            + intercept)

def transmittance_model(tau, l_mean, kappa, intercept):
    """Gamma-distribution transmittance model: γ·(1 + SOD·⟨l'⟩/κ)^(−κ).

    l_mean   = ⟨l'⟩  (mean path-length enhancement, first cumulant k1)
    kappa    = κ      (gamma shape parameter = ⟨l'⟩²/var(l') = k1²/k2)
    intercept = γ     (surface reflectance)
    """
    return (1 + tau * l_mean / kappa) ** (-kappa) * intercept

def universal_quantile_loss(params, x, y, model_func, q=0.05, outlier_threshold=None):
    """
    params: Array of coefficients to optimize.
    x, y: Your data arrays.
    model_func: The function used for fitting (e.g., your fitting_fxn).
    q: The target quantile (0.05 for the bottom 5%).
    outlier_threshold: If set (e.g., 0.5), it caps the 'pull' of extreme low outliers.
    """
    y_pred = model_func(x, *params)
    residual = y - y_pred
    
    if outlier_threshold is not None:
        # Clip extreme negative residuals (bottom outliers)
        residual = np.where(residual < -outlier_threshold, -outlier_threshold, residual)

    # Standard Pinball Loss calculation
    loss = np.where(residual >= 0, q * residual, (q - 1) * residual)
    return np.sum(loss)


# Maps fit_order integer → model function.  Add new orders here as needed.
LOG_TRANSMITTANCE_MODELS = {
    1: log_transmittance_model_1,
    2: log_transmittance_model_2,
    3: log_transmittance_model_3,
    4: log_transmittance_model_4,
    5: log_transmittance_model_5,
    7: log_transmittance_model_7,
    9: log_transmittance_model_9,
}


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
    m1 = 0.5 # coefficient for Stokes Vector I signal in the sensor
    fp_tau_file = sat[orbit_id]["fp_tau_file"]
    with h5py.File(fp_tau_file, "r") as f:
        fp_sounding_id = f["sounding_id"][...].astype(np.int64)
        fp_number      = f["fp_number"][...].astype(int)
        fp_sza         = f["sza"][...]
        fp_vza         = f["vza"][...]
        o2a_tau        = f["o2a_tau_output"][...]
        o2a_toa_sol    = f["o2a_toa_sol_output"][...] * m1
        wco2_tau       = f["wco2_tau_output"][...]
        wco2_toa_sol   = f["wco2_toa_sol_output"][...] * m1
        sco2_tau       = f["sco2_tau_output"][...]
        sco2_toa_sol   = f["sco2_toa_sol_output"][...] * m1
    N = len(fp_sounding_id)

    # --- L1B radiances ---
    l1b = oco2_rad_nadir(l1b_file=sat[orbit_id]["oco_l1b"], lt_file=sat["oco_lite"])

    # Build O(1) lookup: (sounding_id, fp_number) -> along-track index
    l1b_index = {}
    for fp in range(8):
        for track in range(l1b.snd_id.shape[0]):
            l1b_index[(int(l1b.snd_id[track, fp]), fp)] = track

    # Vectorised extraction of track indices for each tau-file sounding
    track_inds = np.array(
        [l1b_index.get((int(sid), int(fp)), -1)
         for sid, fp in zip(fp_sounding_id, fp_number)]
    )
    valid = track_inds >= 0

    fp_lon    = np.full(N, np.nan)
    fp_lat    = np.full(N, np.nan)
    fp_wvls   = np.full((3, 8, 1016), np.nan)  # placeholder; wavelengths are not used in fitting but for plotting
    radiances = np.full((3, N, 1016), np.nan)

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


# ─── Pure computation helpers ─────────────────────────────────────────────────

def compute_transmittance(radiances, toa_sol):
    """T = radiance / TOA_solar; mask unphysical values T > 1.

    Parameters
    ----------
    radiances : array [3, N, 1016]
    toa_sol   : array [3, N, 1016]

    Returns
    -------
    T : array [3, N, 1016], NaN where T > 1 or division is undefined
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        T = radiances / toa_sol * np.pi  # Scale by π to convert from radiance to irradiance ratio
    T[T > 1] = np.nan
    return T


def get_design_matrix(tau, order):
    """
    Creates a design matrix for the equation:
    Intercept + (-1/1)*k1*tau + (1/2)*k2*tau^2 - (1/3)*k3*tau^3 ...
    """
    # 1. Create a column of ones for the intercept
    X = [np.ones(tau.shape)]
    
    # 2. Generate each k_i column dynamically
    for i in range(1, order + 1):
        sign = (-1)**i
        multiplier = 1.0 / i
        column = sign * multiplier * (tau**i)
        X.append(column)
        
    # Stack them horizontally into a single matrix
    return np.column_stack(X)

def fit_bottom_spline(x, y, n_bins=100, q=0.05, s=0.5):
    """
    Fits the lower envelope of a dataset using binning and a smoothing spline.
    
    x, y: Your original numpy arrays.
    n_bins: How many slices to divide the x-axis into.
    q: Target quantile (e.g., 0.05 for the bottom 5%).
    s: Smoothing factor (higher = smoother/flatter, lower = follows data more closely).
    """
    # 1. Sort data by x (required for binning and splines)
    idx = np.argsort(x)
    x_s, y_s = x[idx], y[idx]
    
    # 2. Binning logic
    # bin_edges = np.linspace(x_s.min(), x_s.max(), n_bins + 1)
    if x_s.max() < 5:
        bin_edges = np.linspace(x_s.min(), x_s.max(), n_bins + 1)
    else:
        first_half_bins = np.min([70, n_bins*2//3])
        bin_edges = np.unique(np.concatenate([
            np.linspace(x.min(), 10, first_half_bins+1),       # more bins between min and 10
            np.linspace(10, x.max(), n_bins+1-first_half_bins) # Standard bins for the rest
        ]))
    
    
    bin_centers = []
    bin_quantiles = []
    
    
    for i in range(n_bins):
        mask = (x_s >= bin_edges[i]) & (x_s < bin_edges[i+1])
        if np.any(mask):
            # Calculate the bin center and the q-th percentile of y in that bin
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_quantiles.append(np.quantile(y_s[mask], q))
    
            # print(f"Bin {i}: x in [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}), ")
            # print("Bin center:", bin_centers[-1])
            # print("Bin y min/max:", y_s[mask].min(), y_s[mask].max())
            # print("Bin y last 5 values:", y_s[mask][-5:])
            # print("Bin quantiles:", np.quantile(y_s[mask], [0.05, 0.5, 0.95]))
    
    bin_centers.append([bin_centers[-1]*10, bin_centers[-1]*20])  # Add an extra point at the end to anchor the spline
    bin_quantiles.append([bin_quantiles[-1], bin_quantiles[-1]])  # Use the last quantile value for the extra point
    
    # 3. Fit a Univariate Spline to the binned "bottom" points
    # This is much faster than fitting 50,000 points directly
    
    # Weights: give more importance to the first 10% of points
    weights = np.ones(len(bin_centers))
    weights[:10] = 5.0
    
    spline = UnivariateSpline(bin_centers, bin_quantiles, w=weights, s=s)
    
    return spline, np.array(bin_centers), np.array(bin_quantiles)

def fit_spectral_model(tau, ln_T, fit_order):
    """Fit a cumulant expansion to smoothed log-transmittance vs optical depth.

    Applies a Savitzky-Golay smooth to ln_T (sorted by tau) before fitting,
    which suppresses high-frequency noise without biasing the spectral shape.

    Parameters
    ----------
    tau      : 1-D array, optical depth per spectral channel
    ln_T     : 1-D array, log(transmittance) corresponding to tau
    fit_order: int  cumulant truncation order; must be a key of LOG_TRANSMITTANCE_MODELS

    Returns
    -------
    popt : 1-D array [k1, k2, ..., k_order, intercept]
    """
    model_func  = LOG_TRANSMITTANCE_MODELS[fit_order]
    sort_idx    = np.argsort(tau)
    tau_sorted  = tau[sort_idx]
    ln_T_sorted = ln_T[sort_idx]
    ln_T_smooth = savgol_filter(ln_T[sort_idx], window_length=51, polyorder=3) 
    
    # if fit_order >= 5:
    #     tau_sorted = np.concatenate([tau_sorted, [tau_sorted[-1]*5, tau_sorted[-1]*10]])  # Add extra points to anchor the fit
    #     ln_T_smooth = np.concatenate([ln_T_smooth, [ln_T_smooth[-1], ln_T_smooth[-1]]])  # Use the last value for the extra points
    
    n_params    = fit_order + 1  # k1..kN + intercept
    n_pos       = min(2, fit_order)  # k1 (and k2 if order>=2) must be >0
    lb          = [0.0] * n_pos + [-np.inf] * (n_params - n_pos)
    ub          = [np.inf] * n_params
    p0         = [1.0, 0.5][:n_pos] + [0.01] * (n_params - n_pos)  # Initial guess: small positive k's, zero intercept
    popt, _     = curve_fit(model_func, tau_sorted, ln_T_smooth, bounds=(lb, ub), p0=p0)
    
    # discard_fraction = 0.004  # Discard the top 0.4% of tau values to focus on the lower envelope
    # tau_fit = tau_sorted[: int(len(tau_sorted)*(1-discard_fraction))]
    # ln_T_fit = ln_T_sorted[: int(len(ln_T_sorted)*(1-discard_fraction))]
    
    
    
    # # 1. Run the fit
    # spline_model, centers, quantiles = fit_bottom_spline(tau_fit, ln_T_fit, n_bins=100, q=0.01, s=0.5)

    # # 2. Generate smooth points for plotting
    # x_smooth = np.linspace(tau_fit.min(), tau_fit.max(), 100)
    # y_smooth = spline_model(x_smooth)
    
    
    # plt.figure(figsize=(10, 6))
    # plt.scatter(tau, ln_T, s=2, alpha=0.2, color='gray', label='Original Data')
    # plt.scatter(centers, quantiles, color='red', s=15, label='Bin 5th Percentiles')
    # plt.plot(x_smooth, y_smooth, color='blue', linewidth=2, label='Spline Floor')
    # plt.legend()
    # plt.show()
    # sys.exit(0)
    
    
    # popt, _     = curve_fit(model_func, x_smooth, y_smooth)
    
    
    return popt


def compute_xco2_anomaly(fp_lat, cld_dist_km, xco2,
                         lat_thres=0.5, std_thres=2.0, min_cld_dist=10.0,
                         chunk_size=128, extra_vars=None):
    """XCO2 anomaly relative to nearby clear-sky soundings.

    For each footprint i, the reference set is all footprints within ±lat_thres°
    latitude that are more than min_cld_dist km from a cloud.  The anomaly is
    defined only when the reference std is below std_thres (stable background).

    Parameters
    ----------
    fp_lat      : [N] footprint latitudes (may contain NaN)
    cld_dist_km : [N] nearest-cloud distance in km (may contain NaN)
    xco2        : [N] XCO2 values (may contain NaN)
    lat_thres   : float, half-width of latitude search window (degrees)
    std_thres   : float, maximum allowed std of reference XCO2 (ppm)
    min_cld_dist: float, minimum cloud distance to be considered clear-sky (km)
    chunk_size  : int, rows processed per iteration (controls peak memory).
                  Peak memory per chunk ≈ chunk_size × N × 24 bytes (three
                  [chunk, N] arrays).  Default 32 → ~65 MB for N=84 000.
    extra_vars  : dict[str, np.ndarray] | None
                  Optional additional [N] arrays whose mean and std over the
                  same clear-sky reference window are returned alongside the
                  anomaly.  When provided, returns (anomaly, means, stds).

    Returns
    -------
    anomaly : [N] float array, NaN where reference is unavailable or noisy.
    If extra_vars is not None, also returns:
    extra_means : dict[str, [N] float array]  per-sounding reference mean
    extra_stds  : dict[str, [N] float array]  per-sounding reference std
    """
    N          = len(fp_lat)
    chunk_size = int(chunk_size)   # guard against float passed via **kwargs
    anomaly  = np.full(N, np.nan)
    ref_mean = np.full(N, np.nan)
    ref_std  = np.full(N, np.nan)

    valid_lat  = ~np.isnan(fp_lat)
    clear_mask = valid_lat & (cld_dist_km > min_cld_dist) & ~np.isnan(xco2)  # [N] bool

    # Extend clear_mask to require valid values for ALL extra_vars so that
    # every reference sounding contributes to every variable (shared reference pool).
    if extra_vars is not None:
        for v in extra_vars.values():
            clear_mask = clear_mask & ~np.isnan(np.asarray(v))

    # Pre-extract clear-sky reference latitudes and xco2 values to avoid
    # broadcasting the full array on every chunk.
    ref_lat  = np.where(clear_mask, fp_lat, np.nan)          # [N]
    ref_xco2 = np.where(clear_mask, xco2,   np.nan)          # [N]

    # Pre-extract extra reference arrays (if any)
    extra_ref      = {}
    extra_mean_out = {}
    extra_std_out  = {}
    if extra_vars is not None:
        for k, v in extra_vars.items():
            extra_ref[k]      = np.where(clear_mask, np.asarray(v), np.nan)
            extra_mean_out[k] = np.full(N, np.nan)
            extra_std_out[k]  = np.full(N, np.nan)

    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        q_lat = fp_lat[start:end]                             # [chunk]

        # [chunk, N] pairwise latitude difference for this row block only
        lat_diff  = np.abs(q_lat[:, None] - ref_lat[None, :])   # [chunk, N]
        in_window = lat_diff <= lat_thres                        # [chunk, N] bool

        # xco2 of clear-sky refs within window; NaN elsewhere
        xco2_win  = np.where(in_window, ref_xco2[None, :], np.nan)  # [chunk, N]

        n_refs   = np.sum(~np.isnan(xco2_win), axis=1)          # [chunk] clear-sky refs in window
        has_refs = n_refs >= 5                                    # [chunk] require ≥5 for meaningful std

        chunk_mean = np.full(end - start, np.nan)
        chunk_std  = np.full(end - start, np.nan)
        if has_refs.any():
            chunk_mean[has_refs] = np.nanmean(xco2_win[has_refs], axis=1)
            chunk_std[has_refs]  = np.nanstd( xco2_win[has_refs], axis=1)

        ref_mean[start:end] = chunk_mean
        ref_std[start:end]  = chunk_std

        # Only populate extra vars for soundings whose XCO2 background is stable
        # (chunk_std <= std_thres).  Noisy-ref soundings keep NaN in-loop rather
        # than relying solely on the post-hoc not_valid mask below.
        has_stable = has_refs & (chunk_std <= std_thres)

        if extra_vars is not None and has_stable.any():
            for k, ref_v in extra_ref.items():
                ev_win = np.where(in_window, ref_v[None, :], np.nan)
                chunk_emean = np.full(end - start, np.nan)
                chunk_estd  = np.full(end - start, np.nan)
                chunk_emean[has_stable] = np.nanmean(ev_win[has_stable], axis=1)
                chunk_estd[has_stable]  = np.nanstd( ev_win[has_stable], axis=1)
                extra_mean_out[k][start:end] = chunk_emean
                extra_std_out[k][start:end]  = chunk_estd

        del lat_diff, in_window, xco2_win

    valid = valid_lat & ~np.isnan(ref_mean) & (ref_std <= std_thres)
    anomaly[valid] = xco2[valid] - ref_mean[valid]
    if extra_vars is not None:
        not_valid = ~valid
        for k in extra_mean_out:
            extra_mean_out[k][not_valid] = np.nan
            extra_std_out[k][not_valid]  = np.nan
        return anomaly, extra_mean_out, extra_std_out
    return anomaly


# ─── Optional visualisation ────────────────────────────────────────────────────

def plot_fitting_example(tag, fp, sounding_ind, wvl, rad, transmittance, tau, ln_T, popt_log, fit_order, output_dir):
    """Save example scatter plots of the cumulant and gamma-model fits.

    Produces two PNG files per call:
      - {tag}_log_T_fit_fp{fp}_snd{snd}.png   : ln(T) vs tau with polynomial fit
      - {tag}_T_fit_fp{fp}_snd{snd}.png        : T vs tau with gamma-dist fit
    """
    model_func  = LOG_TRANSMITTANCE_MODELS[fit_order]
    sort_idx    = np.argsort(tau)
    tau_sorted  = tau[sort_idx]
    ln_T_smooth = savgol_filter(ln_T[sort_idx], window_length=31, polyorder=3)
    tau_fit     = np.linspace(tau.min(), tau.max(), 100)

    kappa_1 = popt_log[0]
    kappa_2 = popt_log[1] if fit_order >= 2 else 0.0

    _FONT        = "Arial"
    _LABEL_FS    = 20
    _TICK_FS     = 17
    _LEGEND_FS   = 17
    _TITLE_FS    = 20
    _SUPTITLE_FS = 24

    _rc = {
        "font.family":     _FONT,
        "axes.labelsize":  _LABEL_FS,
        "axes.titlesize":  _TITLE_FS,
        "legend.fontsize": _LEGEND_FS,
        "xtick.labelsize": _TICK_FS,
        "ytick.labelsize": _TICK_FS,
    }
    with plt.rc_context(_rc):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        ax1r = ax1.twinx()
        l1 = ax1.plot(wvl, rad, label="Radiance", color="green", linewidth=2.5)
        l2 = ax1r.plot(wvl, transmittance, label="Transmittance", color="blue")
        ax1.set(xlabel="Wavelength (nm)", ylabel="Radiance")
        ax1r.set(ylabel="Transmittance")
        ax1.legend(l1 + l2, [l.get_label() for l in l1 + l2],
                   loc="lower left", bbox_to_anchor=(0.18, 1.02),
                   ncol=2, borderaxespad=0)
        ax2.scatter(tau, ln_T, label="Observed", color="blue", s=10)
        ax2.plot(tau_fit, model_func(tau_fit, *popt_log), label="Fitted", color="red")
        ax2.plot(tau_sorted, ln_T_smooth, label="Smoothed Observed", color="orange", alpha=0.7)
        ax2.set(
            xlabel=f"Total {tag.upper()} Optical Depth",
            ylabel="ln(Transmittance)",
        )
        ax2.set_title(f"κ1: {kappa_1:.3e}  κ2: {kappa_2:.3e}", pad=14)
        ax2.legend()
        fig.suptitle(f"FP {fp}  Sounding {sounding_ind}",
                     fontsize=_SUPTITLE_FS, y=1.05)
        fig.savefig(
            f"{output_dir}/{tag}_log_T_fit_fp{fp}_snd{sounding_ind}.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # # Plot 2: T vs tau using the gamma-distribution model
    # try:
    #     popt2, _ = curve_fit(transmittance_model, tau, np.exp(ln_T))
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.scatter(tau, np.exp(ln_T), label="Observed", color="blue", s=10)
    #     ax.plot(tau_fit, transmittance_model(tau_fit, *popt2), label="Fitted (gamma)", color="red")
    #     ax.set(
    #         xlabel=f"Total {tag.upper()} Optical Depth",
    #         ylabel="Transmittance",
    #         title=(f"FP {fp}  Sounding {sounding_ind}\n"
    #                f"κ₁: {popt2[0]:.3e}  κ₂: {popt2[1]:.3e}  intercept: {popt2[2]:.3e}"),
    #     )
    #     ax.legend()
    #     fig.savefig(
    #         f"{output_dir}/{tag}_T_fit_fp{fp}_snd{sounding_ind}.png",
    #         dpi=150, bbox_inches="tight",
    #     )
    #     plt.close(fig)
    # except (RuntimeError, ValueError):
    #     pass  # Gamma model may not converge for every sounding; skip quietly


def plot_orbit_fitting_examples(od, fit_orders, output_dir):
    """Save one representative fitting example per band for an orbit."""
    os.makedirs(output_dir, exist_ok=True)
    tags = ["o2a", "wco2", "sco2"]
    T_all = compute_transmittance(od["radiances"], od["toa_sol"])
    ln_T_all = np.where(T_all > 0, np.log(T_all), np.nan)
    plot_done = {tag: False for tag in tags}

    for j in range(len(od["sounding_id"])):
        if all(plot_done.values()):
            return
        if not od["valid_l1b"][j]:
            continue

        for i_band, (tag, band_order) in enumerate(zip(tags, fit_orders)):
            if plot_done[tag]:
                continue

            tau_j = od["tau"][i_band, j][1:-1]
            ln_T_j = ln_T_all[i_band, j][1:-1]
            mask = ~np.isnan(ln_T_j) & ~np.isnan(tau_j)
            if mask.sum() < band_order + 2:
                continue

            try:
                popt = fit_spectral_model(tau_j[mask], ln_T_j[mask], band_order)
            except (RuntimeError, ValueError):
                continue

            plot_fitting_example(
                tag, int(od["fp_number"][j]), int(od["sounding_id"][j]),
                od["wvl"][i_band, od["fp_number"][j]],
                od["radiances"][i_band, j], T_all[i_band, j],
                tau_j[mask], ln_T_j[mask], popt, band_order, output_dir,
            )
            plot_done[tag] = True


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


# ─── Orbit orchestration ───────────────────────────────────────────────────────

def process_orbit(sat, orbit_id, shared_data, fit_order=(7, 2, 7), overwrite=True):
    """Fit the spectral cumulant model for all soundings in one orbit.

    Workflow
    --------
    1. Load per-orbit L1B radiances and optical depths (load_orbit_data).
    2. Compute transmittances for all soundings at once (compute_transmittance).
    3. Loop over soundings to fit the cumulant model (fit_spectral_model).
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
    MAX_KAPPAS = 5                 # store k1..k5; higher kappas are not saved

    kappa_fitting     = np.full((3, N, MAX_KAPPAS), np.nan)
    intercept_fitting = np.full((3, N), np.nan)

    # A cached file from before κ was added (or from before the positivity
    # bounds) lacks the gamma-shape datasets; refit rather than reuse it.
    cached_has_kappa = False
    if os.path.isfile(output_file):
        try:
            with h5py.File(output_file, "r") as f:
                cached_has_kappa = all(
                    key in f for key in ("o2a_kappa", "wco2_kappa", "sco2_kappa")
                )
        except OSError:
            cached_has_kappa = False

    if not os.path.isfile(output_file) or overwrite or not cached_has_kappa:
        if os.path.isfile(output_file) and not cached_has_kappa and not overwrite:
            logger.info(
                f"[{orbit_id}] Cached {output_file} has no κ datasets; refitting."
            )
        # ── 2. Transmittance for all soundings and bands at once ───────────────
        logger.info(f"[{orbit_id}] Computing transmittances for {N} soundings...")
        T_all    = compute_transmittance(od["radiances"], od["toa_sol"])  # [3, N, 1016]
        ln_T_all = np.where(T_all > 0, np.log(T_all), np.nan)            # [3, N, 1016]

        # ── 3. Per-sounding spectral fitting ───────────────────────────────────
        logger.info(f"[{orbit_id}] Fitting spectral models...")
        plot_done = {tag: False for tag in tags}   # save one example plot per band

        for j in range(N):
            if j % 500 == 0:
                logger.info(f"  [{orbit_id}] sounding {j}/{N}")

            if not od["valid_l1b"][j]:
                continue

            for i_band, (tag, band_order) in enumerate(zip(tags, fit_orders)):
                # tau_j  = od["tau"][i_band, j]      # [1016]
                # ln_T_j = ln_T_all[i_band, j]        # [1016]
                tau_j  = od["tau"][i_band, j][1:-1]   # Exclude edge channels which often have NaNs or artifacts
                ln_T_j = ln_T_all[i_band, j][1:-1] # Exclude edge channels which often have NaNs or artifacts

                mask = ~np.isnan(ln_T_j) & ~np.isnan(tau_j)
                if mask.sum() < band_order + 2:     # need more points than free params
                    continue

                try:
                    popt = fit_spectral_model(tau_j[mask], ln_T_j[mask], band_order)
                except (RuntimeError, ValueError):
                    continue

                intercept_fitting[i_band, j]            = popt[-1]
                n_kappas = min(band_order, MAX_KAPPAS)
                kappa_fitting[i_band, j, :n_kappas]     = popt[:n_kappas]

                if not plot_done[tag] or j % 1000 == 0:  # Save an example plot for the first successful fit of each band
                    plot_fitting_example(
                        tag, int(od["fp_number"][j]), int(od["sounding_id"][j]),
                        od["wvl"][i_band, od["fp_number"][j]],
                        od["radiances"][i_band, j], T_all[i_band, j],
                        tau_j[mask], ln_T_j[mask], popt, band_order, output_dir,
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
    anomaly_args = {'lat_thres': 0.25, 'std_thres': 1.0, 'min_cld_dist': 10.0}
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
    anomaly_args_15 = {'lat_thres': 0.25, 'std_thres': 1.0, 'min_cld_dist': 15.0}
    xco2_raw_anomaly_15 = compute_xco2_anomaly(od["lat"], fp_cld_dist, lt_xco2_raw, **anomaly_args_15)
    xco2_bc_anomaly_15, ref_means_15, ref_stds_15 = compute_xco2_anomaly(
        od["lat"], fp_cld_dist, lt_xco2_bc, extra_vars=ref_extra_vars, **anomaly_args_15)

    # ── 6c. Third reference set with looser min_cld_dist=5 km ─────────────
    anomaly_args_05 = {'lat_thres': 0.25, 'std_thres': 1.0, 'min_cld_dist': 5.0}
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
        areas = np.empty(len(vlon))
        for i in range(len(vlon)):
            x, y = _t.transform(vlon[i], vlat[i])
            areas[i] = Polygon(zip(x, y)).area
        fp_area_km2[valid_lt] = areas * 1e-6  # m² → km²



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
        # cloud-contamination tail; see TABM_PLAN "New-feature investigation").
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
    
    
    with h5py.File(output_file, "w") as f_out:
        for key, value in output_dict.items():
            f_out.create_dataset(key, data=np.asarray(value))
            
            
    # ── 8. Visualization  ───────────────────────────────────────────────
    if 0:
        plot_lat_interval = 2.5
        for lat_min in np.arange(-90, 90, plot_lat_interval):
            lat_max = lat_min + plot_lat_interval
            mask = (od["lat"] >= lat_min) & (od["lat"] < lat_max) & od["valid_l1b"] & ~np.isnan(xco2_bc_anomaly)
            if mask.sum() == 0:
                continue

            fig, ((ax10, ax11, ax12, ax13, ax14),
                (ax20, ax21, ax22, ax23, ax24)
                ) = plt.subplots(2, 5, figsize=(26, 10), sharey=True)
            # ax10 plots lon, lat colored by cloud distance
            # ax11 plots lon, lat colored by XCO2 bc
            # ax12 plots lon, lat colored by XCO2 bc anomaly
            # ax13 plots lon, lat colored by k1_fitting for o2a
            # ax14 plots lon, lat colored by k2_fitting for o2a
            # ax20 plots lon, lat colored by XCO2 raw
            # ax21 plots lon, lat colored by k1_fitting for wco2
            # ax22 plots lon, lat colored by k2_fitting for wco2
            # ax23 plots lon, lat colored by k1_fitting for sco2
            # ax24 plots lon, lat colored by k2_fitting for sco2
            
            lon_plot, lat_plot = output_dict["lon"][mask], output_dict["lat"][mask]
            if lon_plot.min() < 0 and lon_plot.max() > 0:
                # Handle dateline crossing by plotting in two segments
                lon_plot = np.where(lon_plot < 0, lon_plot + 360, lon_plot)
            
            sc10 = ax10.scatter(lon_plot, lat_plot, c=output_dict["cld_dist_km"][mask], cmap="jet", s=10, vmin=0, vmax=20)
            sc11 = ax11.scatter(lon_plot, lat_plot, c=output_dict["xco2_bc"][mask], cmap="coolwarm", s=10)
            sc12 = ax12.scatter(lon_plot, lat_plot, c=output_dict["xco2_bc_anomaly"][mask], cmap="coolwarm", s=10)
            sc13 = ax13.scatter(lon_plot, lat_plot, c=output_dict["o2a_k1_fitting"][mask], cmap="plasma", s=10)
            sc14 = ax14.scatter(lon_plot, lat_plot, c=output_dict["o2a_k2_fitting"][mask], cmap="plasma", s=10)
            sc20 = ax20.scatter(lon_plot, lat_plot, c=output_dict["xco2_raw"][mask], cmap="coolwarm", s=10)
            sc21 = ax21.scatter(lon_plot, lat_plot, c=output_dict["wco2_k1_fitting"][mask], cmap="plasma", s=10)
            sc22 = ax22.scatter(lon_plot, lat_plot, c=output_dict["wco2_k2_fitting"][mask], cmap="plasma", s=10)
            sc23 = ax23.scatter(lon_plot, lat_plot, c=output_dict["sco2_k1_fitting"][mask], cmap="plasma", s=10)
            sc24 = ax24.scatter(lon_plot, lat_plot, c=output_dict["sco2_k2_fitting"][mask], cmap="plasma", s=10)
            ax10.set_title(f"Cloud Distance (km)")
            ax11.set_title(f"XCO2 bc (ppm)")
            ax12.set_title(f"XCO2 bc Anomaly (ppm)")
            ax13.set_title(f"O2A k1 Fitting")
            ax14.set_title(f"O2A k2 Fitting")
            ax20.set_title(f"XCO2 raw (ppm)")
            ax21.set_title(f"WCO2 k1 Fitting")
            ax22.set_title(f"WCO2 k2 Fitting")
            ax23.set_title(f"SCO2 k1 Fitting")
            ax24.set_title(f"SCO2 k2 Fitting")
            fig.suptitle(f"Orbit {orbit_id}  Latitude {lat_min}° to {lat_max}°", fontsize=16)
            fig.colorbar(sc10, ax=ax10, label="Cloud Distance (km)")
            fig.colorbar(sc11, ax=ax11, label="XCO2 bc (ppm)")
            fig.colorbar(sc12, ax=ax12, label="XCO2 Anomaly (ppm)")
            fig.colorbar(sc13, ax=ax13, label="O2A k1 Fitting")
            fig.colorbar(sc14, ax=ax14, label="O2A k2 Fitting")
            fig.colorbar(sc20, ax=ax20, label="XCO2 raw (ppm)")
            fig.colorbar(sc21, ax=ax21, label="WCO2 k1 Fitting")
            fig.colorbar(sc22, ax=ax22, label="WCO2 k2 Fitting")
            fig.colorbar(sc23, ax=ax23, label="SCO2 k1 Fitting")
            fig.colorbar(sc24, ax=ax24, label="SCO2 k2 Fitting")
            xmin, xmax = lon_plot.min(), lon_plot.max()
            ymin, ymax = lat_plot.min(), lat_plot.max()
            for ax in [ax10, ax11, ax12, ax13, ax14, ax20, ax21, ax22, ax23, ax24]:
                ax.set_xlim(xmin - 0.5, xmax + 0.5)
                ax.set_ylim(ymin - 0.5, ymax + 0.5)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
            fig.savefig(
                f"{output_dir}/final_visualization_lat{lat_min}_{lat_max}_all.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
            
        for fp in range(8):    
            for lat_min in np.arange(-90, 90, plot_lat_interval):
                lat_max = lat_min + plot_lat_interval
                mask = (od["lat"] >= lat_min) & (od["lat"] < lat_max) & od["valid_l1b"] & ~np.isnan(xco2_bc_anomaly) & (od["fp_number"] == fp)
                if mask.sum() == 0:
                    continue
                fig, ((ax10, ax11, ax12, ax13, ax14),
                    (ax20, ax21, ax22, ax23, ax24)
                    ) = plt.subplots(2, 5, figsize=(26, 10), sharey=True)
                # ax10 plots lon, lat colored by cloud distance
                # ax11 plots lon, lat colored by XCO2 bc
                # ax12 plots lon, lat colored by XCO2 bc anomaly
                # ax13 plots lon, lat colored by k1_fitting for o2a
                # ax14 plots lon, lat colored by k2_fitting for o2a
                # ax20 plots lon, lat colored by XCO2 raw
                # ax21 plots lon, lat colored by k1_fitting for wco2
                # ax22 plots lon, lat colored by k2_fitting for wco2
                # ax23 plots lon, lat colored by k1_fitting for sco2
                # ax24 plots lon, lat colored by k2_fitting for sco2
                
                lon_plot, lat_plot = output_dict["lon"][mask], output_dict["lat"][mask]
                if lon_plot.min() < 0 and lon_plot.max() > 0:
                    # Handle dateline crossing by plotting in two segments
                    lon_plot = np.where(lon_plot < 0, lon_plot + 360, lon_plot)
                
                sc10 = ax10.scatter(lon_plot, lat_plot, c=output_dict["cld_dist_km"][mask], cmap="jet", s=10, vmin=0, vmax=20)
                sc11 = ax11.scatter(lon_plot, lat_plot, c=output_dict["xco2_bc"][mask], cmap="coolwarm", s=10)
                sc12 = ax12.scatter(lon_plot, lat_plot, c=output_dict["xco2_bc_anomaly"][mask], cmap="coolwarm", s=10)
                sc13 = ax13.scatter(lon_plot, lat_plot, c=output_dict["o2a_k1_fitting"][mask], cmap="plasma", s=10)
                sc14 = ax14.scatter(lon_plot, lat_plot, c=output_dict["o2a_k2_fitting"][mask], cmap="plasma", s=10)
                sc20 = ax20.scatter(lon_plot, lat_plot, c=output_dict["xco2_raw"][mask], cmap="coolwarm", s=10)
                sc21 = ax21.scatter(lon_plot, lat_plot, c=output_dict["wco2_k1_fitting"][mask], cmap="plasma", s=10)
                sc22 = ax22.scatter(lon_plot, lat_plot, c=output_dict["wco2_k2_fitting"][mask], cmap="plasma", s=10)
                sc23 = ax23.scatter(lon_plot, lat_plot, c=output_dict["sco2_k1_fitting"][mask], cmap="plasma", s=10)
                sc24 = ax24.scatter(lon_plot, lat_plot, c=output_dict["sco2_k2_fitting"][mask], cmap="plasma", s=10)
                ax10.set_title(f"Cloud Distance (km)")
                ax11.set_title(f"XCO2 bc (ppm)")
                ax12.set_title(f"XCO2 bc Anomaly (ppm)")
                ax13.set_title(f"O2A k1 Fitting")
                ax14.set_title(f"O2A k2 Fitting")
                ax20.set_title(f"XCO2 raw (ppm)")
                ax21.set_title(f"WCO2 k1 Fitting")
                ax22.set_title(f"WCO2 k2 Fitting")
                ax23.set_title(f"SCO2 k1 Fitting")
                ax24.set_title(f"SCO2 k2 Fitting")
                fig.suptitle(f"Orbit {orbit_id}  Latitude {lat_min}° to {lat_max}°", fontsize=16)
                fig.colorbar(sc10, ax=ax10, label="Cloud Distance (km)")
                fig.colorbar(sc11, ax=ax11, label="XCO2 bc (ppm)")
                fig.colorbar(sc12, ax=ax12, label="XCO2 Anomaly (ppm)")
                fig.colorbar(sc13, ax=ax13, label="O2A k1 Fitting")
                fig.colorbar(sc14, ax=ax14, label="O2A k2 Fitting")
                fig.colorbar(sc20, ax=ax20, label="XCO2 raw (ppm)")
                fig.colorbar(sc21, ax=ax21, label="WCO2 k1 Fitting")
                fig.colorbar(sc22, ax=ax22, label="WCO2 k2 Fitting")
                fig.colorbar(sc23, ax=ax23, label="SCO2 k1 Fitting")
                fig.colorbar(sc24, ax=ax24, label="SCO2 k2 Fitting")
                xmin, xmax = lon_plot.min(), lon_plot.max()
                ymin, ymax = lat_plot.min(), lat_plot.max()
                for ax in [ax10, ax11, ax12, ax13, ax14, ax20, ax21, ax22, ax23, ax24]:
                    ax.set_xlim(xmin - 0.5, xmax + 0.5)
                    ax.set_ylim(ymin - 0.5, ymax + 0.5)
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                fig.savefig(
                    f"{output_dir}/final_visualization_lat{lat_min}_{lat_max}_fp_{fp}.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)


    logger.info(f"[{orbit_id}] Done.")


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
            f"Re-run demo_combined.py for this date to download it."
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
                   delete_ocofiles=False):
    """Top-level pipeline: preprocess → fit → analyse."""
    validate_cloud_distance_file(
        cloud_distance_file_path(result_dir, target_date),
        target_date,
    )
    sat0 = preprocess(target_date, data_dir, result_dir, limit_granules)

    fit_order = (7, 3, 7)  # (o2a_order, wco2_order, sco2_order)

    # Load Lite file and cloud distances once, shared across all orbits
    shared_data = load_shared_data(sat0)

    for orbit_id in sat0["orbit_list"]:
        date       = sat0['date'].strftime("%Y-%m-%d")
        h5_output_dir = f"{sat0['result_dir']}/fitting_details"
        output_file = f"{h5_output_dir}/fitting_details_{date}_{orbit_id}.h5"
        
        # if os.path.isfile(output_file):
        #     logger.info(f"[{orbit_id}] Output already exists. Skipping orbit.")
        #     continue
        
        process_orbit(sat0, orbit_id, shared_data, fit_order=fit_order, overwrite=False)

        if delete_ocofiles:
            for key in ("oco_l1b", "oco_met", "oco_co2prior"):
                fpath = sat0[orbit_id].get(key)
                if fpath and os.path.isfile(fpath):
                    os.remove(fpath)
                    logger.info(f"Deleted input file: {fpath}")

        # k1k2_analysis(sat0, orbit_id)
    #     # k1k2_analysis(sat0, orbit_id, reference_csv='/Users/yuch8913/programming/oco_fp_analysis/results/2018-10-18/22846a/combined_k1_k2_individual_fp.csv')
    # print("sat0:", sat0)
    # k1k2_analysis(sat0)
    # k1k2_analysis(sat0, '22849a', reference_csv='/Users/yuch8913/programming/oco_fp_analysis/results/2018-10-18/combined_k1_k2_individual_fp_3granules.csv')
    


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

    run_simulation(
        target_date, data_dir, output_dir,
        limit_granules=args.limit_granules,
        viz_dir=viz_dir,
        visualize=args.visualize,
        delete_ocofiles=args.delete_ocofiles,
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

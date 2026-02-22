import os
import platform
import sys
from datetime import datetime
from pathlib import Path

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

from result_ana import k1k2_analysis
from abs_util.fp_atm import oco_fp_atm_abs
from abs_util.oco_util import timing
import argparse
import logging
import glob
from config import Config


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

def transmittance_model(tau, k1, k2, intercept):
    """Gamma-distribution transmittance model: (1 + τ·k1/k2)^(−k2) · intercept."""
    return (1 + tau * k1 / k2) ** (-k2) * intercept

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
    cld_dist_file = f"{sat['result_dir']}/results_{date}.h5"
    logger.info(f"Loading cloud distances from {cld_dist_file}")
    with h5py.File(cld_dist_file, "r") as f:
        cld_snd_id  = f["sounding_id"][...].astype(np.int64)
        cld_dist_km = f["nearest_cloud_distance_km"][...].astype(np.float64)
    cld_dist_index = dict(zip(cld_snd_id.tolist(), cld_dist_km.tolist()))

    # --- OCO-2 Lite file ---
    logger.info(f"Loading Lite file {sat['oco_lite']}")
    with dataset(sat["oco_lite"], "r") as nc:
        lt_id = np.array(nc.variables["sounding_id"][:], dtype=np.int64)
        def _load(var):
            return np.array(nc.variables[var][:])
        def _load_grp(grp, var):
            return np.array(nc.groups[grp].variables[var][:])

        lite = {
            "xco2_corr":         _load("xco2"),
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
        }

    lite_index = {int(sid): i for i, sid in enumerate(lt_id)}
    logger.info(
        f"Shared data loaded: {len(cld_dist_index)} cloud-dist entries, "
        f"{len(lite_index)} Lite soundings."
    )
    return {"cld_dist_index": cld_dist_index, "lite_index": lite_index, "lite": lite}


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
        T = radiances / toa_sol
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
    p0         = [1.0, 0.5] + [0.01] * (n_params - n_pos)  # Initial guess: small positive k's, zero intercept
    popt, _     = curve_fit(model_func, tau_sorted, ln_T_smooth,)# bounds=(lb, ub), p0=p0)
    
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
                         lat_thres=0.5, std_thres=2.0, min_cld_dist=10.0):
    """Vectorised XCO2 anomaly relative to nearby clear-sky soundings.

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

    Returns
    -------
    anomaly : [N] float array, NaN where reference is unavailable or noisy

    Notes
    -----
    Builds an [N, N] pairwise latitude-difference matrix.  For N > ~5 000 this
    can use significant memory (~N²×8 bytes).  Consider chunking if needed.
    """
    N = len(fp_lat)
    anomaly = np.full(N, np.nan)

    valid_lat  = ~np.isnan(fp_lat)
    clear_mask = valid_lat & (cld_dist_km > min_cld_dist)  # [N]

    # Pairwise lat-difference matrix: ref_mask[i, j] = True if j is a valid
    # clear-sky reference for footprint i
    lat_mat  = np.abs(fp_lat[:, None] - fp_lat[None, :])          # [N, N]
    ref_mask = (lat_mat <= lat_thres) & clear_mask[None, :]        # [N, N]

    # Reference statistics per footprint
    xco2_refs = np.where(ref_mask, xco2[None, :], np.nan)          # [N, N]
    ref_mean  = np.nanmean(xco2_refs, axis=1)                       # [N]
    ref_std   = np.nanstd(xco2_refs,  axis=1)                       # [N]

    has_refs = ref_mask.any(axis=1)
    valid    = valid_lat & has_refs & (ref_std <= std_thres)
    anomaly[valid] = xco2[valid] - ref_mean[valid]
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

    # Plot 1: ln(T) vs tau
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    # ax1 plots radiances vs wavelength and transmittance vs wavelength for visual context, but the main focus is ax2 which shows the spectral fit.
    ax1r = ax1.twinx()
    l1 = ax1.plot(wvl, rad, label="Radiance", color="green", linewidth=2.5)
    l2 = ax1r.plot(wvl, transmittance, label="Transmittance", color="blue")
    ax1.set(xlabel="Wavelength (nm)", ylabel="Radiance")
    ax1r.set(ylabel="Transmittance")
    ax1.legend(l1 + l2, [l.get_label() for l in l1 + l2])
    
    # ax2 plots the original ln(T) vs tau scatter, the fitted curve from the cumulant model, and the smoothed ln(T) for visual clarity. The title includes the fitted kappa_1 and kappa_2 values for reference. The plot is saved to a PNG file named according to the tag, footprint number, and sounding index.
    ax2.scatter(tau, ln_T, label="Observed", color="blue", s=10)
    ax2.plot(tau_fit, model_func(tau_fit, *popt_log), label="Fitted", color="red")
    ax2.plot(tau_sorted, ln_T_smooth, label="Smoothed Observed", color="orange", alpha=0.7)
    ax2.set(
        xlabel=f"Total {tag.upper()} Optical Depth",
        ylabel="ln(Transmittance)",
        title=f"κ1: {kappa_1:.3e}  κ2: {kappa_2:.3e}",
    )
    ax2.legend()
    fig.suptitle(f"FP {fp}  Sounding {sounding_ind}", fontsize=16)
    fig.savefig(
        f"{output_dir}/{tag}_log_T_fit_fp{fp}_snd{sounding_ind}.png",
        dpi=150, bbox_inches="tight"
    )

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

    if os.path.isfile(output_file) and not overwrite:
        logger.info(f"[{orbit_id}] Skipping — output exists and overwrite=False.")
        return

    # ── 1. Load orbit data ─────────────────────────────────────────────────
    logger.info(f"[{orbit_id}] Loading orbit data...")
    od = load_orbit_data(sat, orbit_id)
    N  = len(od["sounding_id"])

    tags       = ["o2a", "wco2", "sco2"]
    fit_orders = list(fit_order)   # (o2a_order, wco2_order, sco2_order)
    MAX_KAPPAS = 5                 # store k1..k5; higher kappas are not saved

    kappa_fitting     = np.full((3, N, MAX_KAPPAS), np.nan)
    intercept_fitting = np.full((3, N), np.nan)

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

    # ── 4. Lite variable extraction (vectorised) ───────────────────────────
    logger.info(f"[{orbit_id}] Extracting Lite variables...")
    lite_idx = shared_data["lite_index"]
    lite     = shared_data["lite"]
    row_inds = np.array([lite_idx.get(int(sid), -1) for sid in od["sounding_id"]])
    valid_lt = row_inds >= 0

    def _lite(key):
        """Extract one Lite variable for all soundings; NaN where not matched."""
        out = np.full(N, np.nan)
        if valid_lt.any():
            out[valid_lt] = lite[key][row_inds[valid_lt]]
        return out

    lt_xco2_bc  = _lite("xco2_corr")
    lt_xco2_raw = _lite("xco2_raw")

    # ── 5. Cloud distance per sounding (O(1) dict lookup) ─────────────────
    cld_idx     = shared_data["cld_dist_index"]
    fp_cld_dist = np.array([cld_idx.get(int(sid), np.nan) for sid in od["sounding_id"]])

    # ── 6. XCO2 anomaly (vectorised lat-window) ────────────────────────────
    logger.info(f"[{orbit_id}] Computing XCO2 anomalies...")
    anomaly_args = {'lat_thres': 0.25, 'std_thres': 1.0, 'min_cld_dist': 10.0}
    xco2_raw_anomaly = compute_xco2_anomaly(od["lat"], fp_cld_dist, lt_xco2_raw, **anomaly_args)
    xco2_bc_anomaly  = compute_xco2_anomaly(od["lat"], fp_cld_dist, lt_xco2_bc, **anomaly_args)

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
        # Geometry
        "lon":       od["lon"],
        "lat":       od["lat"],
        "sza":       od["sza"],
        "vza":       od["vza"],
        "mu_sza":     np.cos(np.radians(od["sza"])),
        "mu_vza":     np.cos(np.radians(od["vza"])),
        "fp_number": od["fp_number"],
        "fp_id":     od["sounding_id"],
        # Cloud proximity
        "cld_dist_km":      fp_cld_dist,
        # XCO2
        "xco2_bc":          lt_xco2_bc,
        "xco2_raw":         lt_xco2_raw,
        "xco2_raw_anomaly": xco2_raw_anomaly,
        "xco2_bc_anomaly":  xco2_bc_anomaly,
        # Lite retrieval variables
        "psfc_lt":        _lite("psurf"),
        "airmass_lt":     _lite("airmass"),
        "delT_lt":        _lite("deltaT"),
        "dp_lt":          _lite("dp"),
        "dp_o2a_lt":      _lite("dp_o2a"),
        "dp_sco2_lt":     _lite("dp_sco2"),
        "co2_grad_del_lt": _lite("co2_grad_del"),
        "alb_o2a_lt":     _lite("albedo_o2a"),
        "alb_wco2_lt":    _lite("albedo_wco2"),
        "alb_sco2_lt":    _lite("albedo_sco2"),
        "aod_total_lt":   _lite("aod_total"),
        "fs_rel_lt":      _lite("fs_rel"),
        "alt_lt":         _lite("altitude"),
        "alt_std_lt":     _lite("altitude_std"),
        "xco2_qf_lt":     _lite("qf"),
        "sfc_type_lt":    _lite("sfc_type"),
        "ws_lt":          _lite("windspeed"),
        "ws_apriori_lt":  _lite("windspeed_apriori"),
        # Preprocessor variables
        "co2_ratio_bc_lt":  _lite("co2_ratio_bc"),
        "h2o_ratio_bc_lt":  _lite("h2o_ratio_bc"),
        "csnr_o2a_lt":      _lite("color_slice_noise_ratio_o2a"),
        "csnr_wco2_lt":     _lite("color_slice_noise_ratio_wco2"),
        "csnr_sco2_lt":     _lite("color_slice_noise_ratio_sco2"),
        "dp_abp_lt":        _lite("dp_abp"),
        "h_cont_o2a_lt":    _lite("h_continuum_o2a"),
        "h_cont_wco2_lt":   _lite("h_continuum_wco2"),
        "h_cont_sco2_lt":   _lite("h_continuum_sco2"),
        "max_declock_o2a_lt":  _lite("max_declocking_o2a"),
        "max_declock_wco2_lt": _lite("max_declocking_wco2"),
        "max_declock_sco2_lt": _lite("max_declocking_sco2"),
        "xco2_strong_idp_lt": _lite("xco2_strong_idp"),
        "xco2_weak_idp_lt":   _lite("xco2_weak_idp"),
        # Additional retrieval variables
        "h2o_scale_lt":    _lite("h2o_scale"),
        "dpfrac_lt":       _lite("dpfrac"),
        "aod_bc_lt":       _lite("aod_bc"),
        "aod_dust_lt":     _lite("aod_dust"),
        "aod_ice_lt":      _lite("aod_ice"),
        "aod_water_lt":    _lite("aod_water"),
        "aod_oc_lt":       _lite("aod_oc"),
        "aod_seasalt_lt":  _lite("aod_seasalt"),
        "aod_strataer_lt": _lite("aod_strataer"),
        "aod_sulfate_lt":  _lite("aod_sulfate"),
        "dust_height_lt":  _lite("dust_height"),
        "ice_height_lt":   _lite("ice_height"),
        "dws_lt":          _lite("dws"),
        # Additional sounding variables
        "snr_o2a_lt":      _lite("snr_o2a"),
        "snr_wco2_lt":     _lite("snr_wco2"),
        "snr_sco2_lt":     _lite("snr_sco2"),
        "glint_angle_lt":  _lite("glint_angle"),
        "pol_angle_lt":    _lite("polarization_angle"),
        "saa_lt":          _lite("saa"),
        "vaa_lt":          _lite("vaa"),
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
    lite_nc_file = nc4_matches[0]
    if not h5py.is_hdf5(lite_nc_file):
        raise OSError(
            f"L2 Lite file is corrupted or not a valid HDF5/NetCDF4 file:\n"
            f"  {lite_nc_file}\n"
            f"Delete this file and re-run demo_combined.py --date {date.date()} to re-download."
        )

    sat0 = {
        "date":       date,
        "data_dir":   OCO2_data_dir,
        "result_dir": result_dir,
        "orbit_list": oco2_orbit_list,
        "oco_lite":   lite_nc_file,
    }

    for orbit_id in oco2_orbit_list:
        sat0[orbit_id] = {}
        for file in glob.glob(f"{OCO2_data_dir}/{orbit_id}/*"):
            if "L1b" in file:
                if "TG" not in file:
                    print(f"Found L1b file for orbit {orbit_id}: {file}")
                    sat0[orbit_id]["oco_l1b"] = file
                elif "TG" in file:
                    print(f"Found L1b TG file for orbit {orbit_id}: {file}")
                    sat0[orbit_id]["oco_l1b_tg"] = file
            if "Met" in file:
                sat0[orbit_id]["oco_met"] = file
            if "CPr" in file:
                sat0[orbit_id]["oco_co2prior"] = file
                
        fp_tau_file = os.path.abspath(f"{result_dir}/{orbit_id}/fp_tau_combined.h5")
        os.makedirs(os.path.dirname(fp_tau_file), exist_ok=True)
        if not os.path.isfile(fp_tau_file):
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
    sat0 = preprocess(target_date, data_dir, result_dir, limit_granules)

    fit_order = (7, 3, 7)  # (o2a_order, wco2_order, sco2_order)

    # Load Lite file and cloud distances once, shared across all orbits
    shared_data = load_shared_data(sat0)

    for orbit_id in sat0["orbit_list"]:
        date       = sat0['date'].strftime("%Y-%m-%d")
        output_dir  = f"{sat0['result_dir']}/{date}/{orbit_id}"
        output_file = f"{output_dir}/fitting_details.h5"
        
        if os.path.isfile(output_file):
            logger.info(f"[{orbit_id}] Output already exists. Skipping orbit.")
            continue
        
        process_orbit(sat0, orbit_id, shared_data, fit_order=fit_order, overwrite=True)

        if delete_ocofiles:
            for key in ("oco_l1b", "oco_met", "oco_co2prior"):
                fpath = sat0[orbit_id].get(key)
                if fpath and os.path.isfile(fpath):
                    os.remove(fpath)
                    logger.info(f"Deleted input file: {fpath}")

        # k1k2_analysis(sat0, orbit_id)
    #     # k1k2_analysis(sat0, orbit_id, reference_csv='/Users/yuch8913/programming/oco_fp_analysis/results/2018-10-18/22846a/combined_k1_k2_individual_fp.csv')
    print("sat0:", sat0)
    k1k2_analysis(sat0)
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

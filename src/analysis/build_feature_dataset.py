import h5py
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime
import copy
import glob
import pathlib
import gc
import platform
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
import pickle

logger = logging.getLogger(__name__)

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


# ─── Vertical-profile resampling (sigma grid) ───────────────────────────────────
# fitting.py stores the RAW native-grid (72-level GEOS) T / specific-humidity /
# CO2-prior profiles plus the per-sounding pressure grid and surface pressure.
# Here they are resampled onto a fixed sigma = P/Psurf grid so profiles are
# comparable across soundings (and PCA-compressible downstream).  Sigma is used
# rather than absolute pressure because surface pressure spans ~706–1022 hPa over
# land glint: sigma pins the surface at sigma=1 for every sounding, keeping the
# near-surface / boundary-layer structure (where the cloud-proximity XCO2 bias is
# expected) on the same grid index everywhere.  Denser near the surface.
# Column names t_sigma_NN / q_sigma_NN / co2prior_sigma_NN index into this array.
SIGMA_LEVELS = np.array([
    0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00,
], dtype=np.float64)


def _interp_profile_to_sigma(prof, p_native, psurf, sigma_levels=SIGMA_LEVELS):
    """Resample one native profile onto the fixed sigma = P/Psurf grid.

    Interpolates in log-sigma (≡ log-pressure shifted by log Psurf), the natural
    coordinate for atmospheric profiles.  ``np.interp`` clamps to the endpoint
    value outside the native support, so no unphysical extrapolation occurs:
    sigma is bounded by construction (top ≈ 1.5 Pa / Psurf; bottom ≈ 1.0 at the
    surface), so target levels never fall "underground".  Returns all-NaN when
    Psurf or the profile is unusable.
    """
    L = len(sigma_levels)
    if not np.isfinite(psurf) or psurf <= 0:
        return np.full(L, np.nan)
    sig_native = np.asarray(p_native, dtype=np.float64) / psurf
    prof = np.asarray(prof, dtype=np.float64)
    good = np.isfinite(sig_native) & np.isfinite(prof) & (sig_native > 0)
    if good.sum() < 2:
        return np.full(L, np.nan)
    x = np.log(sig_native[good])
    order = np.argsort(x)                       # np.interp requires increasing x
    return np.interp(np.log(sigma_levels), x[order], prof[good][order])


def compute_sigma_profile_columns(combined, sigma_levels=SIGMA_LEVELS):
    """Build flat sigma-grid profile columns from the raw profiles in `combined`.

    Reads the [N, n_lev] arrays t_profile / q_profile / co2prior_profile, the
    pressure grid p_profile, and psurf_met (all written by fitting.py).  Returns a
    dict of flat per-level columns t_sigma_NN / q_sigma_NN / co2prior_sigma_NN
    (NN = index into ``sigma_levels``) plus tropopause_sigma (= P_trop / Psurf).
    Returns {} when the raw profile keys are absent (older fitting_details.h5).
    """
    required = ('t_profile', 'q_profile', 'co2prior_profile', 'p_profile', 'psurf_met')
    if not all(k in combined for k in required):
        return {}
    P     = np.asarray(combined['p_profile'], dtype=np.float64)   # [N, n_lev]
    psurf = np.asarray(combined['psurf_met'], dtype=np.float64)   # [N]
    N, L  = P.shape[0], len(sigma_levels)

    cols = {}
    for src, prefix in (('t_profile', 't_sigma'),
                        ('q_profile', 'q_sigma'),
                        ('co2prior_profile', 'co2prior_sigma')):
        prof = np.asarray(combined[src], dtype=np.float64)        # [N, n_lev]
        sig  = np.full((N, L), np.nan, dtype=np.float32)
        for k in range(N):
            sig[k] = _interp_profile_to_sigma(prof[k], P[k], psurf[k], sigma_levels)
        for lev in range(L):
            cols[f'{prefix}_{lev:02d}'] = sig[:, lev]

    if 'tropopause_pressure' in combined:
        tp = np.asarray(combined['tropopause_pressure'], dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            cols['tropopause_sigma'] = np.where(
                np.isfinite(psurf) & (psurf > 0), tp / psurf, np.nan
            ).astype(np.float32)
    logger.info("compute_sigma_profile_columns: %d sigma levels × 3 profiles + tropopause_sigma (N=%d)", L, N)
    return cols


# ─── HDF5 loader ──────────────────────────────────────────────────────────────

def load_output_dict(filepath):
    """Load one orbit's fitting_details.h5 written by process_orbit().

    Returns a flat dict with all per-band kappa arrays plus shared variables.
    Keys that are absent from the file are silently skipped.
    """
    keys = [
        # basic identifiers
        'date', 'time', 'orbit_id',
        # Cumulant coefficients (per band)
        'o2a_k1_fitting', 'o2a_k2_fitting', 'o2a_k3_fitting',
        'o2a_k4_fitting', 'o2a_k5_fitting', 'o2a_intercept_fitting',
        'wco2_k1_fitting', 'wco2_k2_fitting', 'wco2_k3_fitting',
        'wco2_k4_fitting', 'wco2_k5_fitting', 'wco2_intercept_fitting',
        'sco2_k1_fitting', 'sco2_k2_fitting', 'sco2_k3_fitting',
        'sco2_k4_fitting', 'sco2_k5_fitting', 'sco2_intercept_fitting',
        'o2a_kappa', 'wco2_kappa', 'sco2_kappa',   # gamma shape κ = k1²/k2
        'exp_intercept_o2a', 'exp_intercept_wco2', 'exp_intercept_sco2',
        # Reference clear-sky statistics
        'ref_o2a_k1_mean', 'ref_o2a_k1_std', 'ref_o2a_k2_mean', 'ref_o2a_k2_std',
        'ref_o2a_k3_mean', 'ref_o2a_k3_std',
        'ref_wco2_k1_mean', 'ref_wco2_k1_std', 'ref_wco2_k2_mean', 'ref_wco2_k2_std',
        'ref_wco2_k3_mean', 'ref_wco2_k3_std',
        'ref_sco2_k1_mean', 'ref_sco2_k1_std', 'ref_sco2_k2_mean', 'ref_sco2_k2_std',
        'ref_sco2_k3_mean', 'ref_sco2_k3_std',
        'ref_alb_o2a_mean', 'ref_alb_o2a_std', 'ref_alb_wco2_mean', 'ref_alb_wco2_std',
        'ref_alb_sco2_mean', 'ref_alb_sco2_std',
        'ref_exp_int_o2a_mean', 'ref_exp_int_o2a_std',
        'ref_exp_int_wco2_mean', 'ref_exp_int_wco2_std',
        'ref_exp_int_sco2_mean', 'ref_exp_int_sco2_std',
        # Geometry
        'lon', 'lat', 'sza', 'vza', 'mu_sza', 'mu_vza', 'fp_number', 'fp_id', 'fp_area_km2',
        # Cloud proximity
        'cld_dist_km',
        'weighted_cloud_dist_km',
        # XCO2
        'xco2_apriori',
        'xco2_bc', 'xco2_raw', 'xco2_bc_anomaly', 'xco2_raw_anomaly',
        # Lite retrieval variables
        'psfc', 'airmass', 'delT', 'dp', 'dp_o2a', 'dp_sco2', 'co2_grad_del',
        'alb_o2a', 'alb_wco2', 'alb_sco2', 'aod_total', 'fs_rel',
        'alt', 'alt_std', 'xco2_qf', 'sfc_type', 'ws', 'ws_apriori',
        # Preprocessor variables
        'co2_ratio_bc', 'h2o_ratio_bc',
        'csnr_o2a', 'csnr_wco2', 'csnr_sco2',
        'dp_abp',
        'h_cont_o2a', 'h_cont_wco2', 'h_cont_sco2',
        'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2',
        'xco2_strong_idp', 'xco2_weak_idp',
        # Additional retrieval variables
        'h2o_scale', 'dpfrac',
        'aod_bc', 'aod_dust', 'aod_ice', 'aod_water',
        'aod_oc', 'aod_seasalt', 'aod_strataer', 'aod_sulfate',
        'dust_height', 'ice_height', 'dws',
        # Additional sounding variables
        'snr_o2a', 'snr_wco2', 'snr_sco2',
        'glint_angle', 'pol_angle', 'saa', 'vaa',
        # additional Lite variables that may be missing from older output files but are needed for consistency in the combined DataFrame
        "s31", "s32", "snow_flag", "t700", "tcwv", "operation_mode", "water_height",
        # spectral-fit-quality diagnostics (Tier-A cloud-contamination candidates;
        # absent in older fitting_details.h5 → skipped via `if key in f`)
        "chi2_o2a", "chi2_wco2", "chi2_sco2",
        "rms_rel_o2a", "rms_rel_wco2", "rms_rel_sco2",
        "eof3_1_rel", "diverging_steps", "xco2_uncertainty",
        # Raw native-grid atmospheric profiles + pressure grid + surface pressure
        # (2-D [N, n_lev]); resampled to the sigma grid by compute_sigma_profile_columns.
        # Absent in older files → skipped, and the sigma columns are simply not built.
        "t_profile", "q_profile", "co2prior_profile", "p_profile", "psurf_met",
        "tropopause_pressure", "tropopause_temp",
        # r15 reference set (min_cld_dist=15 km)
        'xco2_raw_anomaly_r15', 'xco2_bc_anomaly_r15',
        'r15_o2a_k1_mean', 'r15_o2a_k1_std', 'r15_o2a_k2_mean', 'r15_o2a_k2_std',
        'r15_o2a_k3_mean', 'r15_o2a_k3_std',
        'r15_wco2_k1_mean', 'r15_wco2_k1_std', 'r15_wco2_k2_mean', 'r15_wco2_k2_std',
        'r15_wco2_k3_mean', 'r15_wco2_k3_std',
        'r15_sco2_k1_mean', 'r15_sco2_k1_std', 'r15_sco2_k2_mean', 'r15_sco2_k2_std',
        'r15_sco2_k3_mean', 'r15_sco2_k3_std',
        'r15_alb_o2a_mean', 'r15_alb_o2a_std', 'r15_alb_wco2_mean', 'r15_alb_wco2_std',
        'r15_alb_sco2_mean', 'r15_alb_sco2_std',
        'r15_exp_int_o2a_mean', 'r15_exp_int_o2a_std',
        'r15_exp_int_wco2_mean', 'r15_exp_int_wco2_std',
        'r15_exp_int_sco2_mean', 'r15_exp_int_sco2_std',
    ]
    with h5py.File(filepath, 'r') as f:
        out = {key: f[key][()] for key in keys if key in f}
        if 'sounding_id' in f and 'weighted_cloud_dist_km' not in out:
            n = len(f['sounding_id'])
            out['weighted_cloud_dist_km'] = np.full(n, np.nan)
        return out



def main():
    fdir = '.'
    
def raw_processing_single_date(result_dir, date, orbit_id=None):
    """Load fitting results for all orbits, build combined DataFrame, run analysis.

    Parameters
    ----------
    sat : dict
        Pipeline state from preprocess() with keys: 'date', 'result_dir', 'orbit_list'.
    reference_csv : str or None
        Optional path to a CSV from a reference scene for cross-scene mitigation.
    """
    
    output_dir = f"{result_dir}/csv_collection"
    os.makedirs(output_dir, exist_ok=True)

    # ── Collect and concatenate data across all orbits ─────────────────────────
    all_data = []
    if date is not None and orbit_id is None:
        h5_output_dir = f"{result_dir}/fitting_details"
        h5_files = glob.glob(os.path.join(h5_output_dir, f"fitting_details_{date}_*.h5"))
        for filepath in h5_files:
            if os.path.exists(filepath):
                all_data.append(load_output_dict(filepath))
            else:
                print(f"File not found: {filepath}")
    elif date is not None and orbit_id is not None:
        h5_output_dir = f"{result_dir}/fitting_details"
        filepath = f"{h5_output_dir}/fitting_details_{date}_{orbit_id}.h5"
        if os.path.exists(filepath):
            all_data.append(load_output_dict(filepath))
        else:
            print(f"File not found: {filepath}")
    else:
        print("Invalid input: Please provide either both date and orbit_id, or neither.")
        raise ValueError("Invalid input: Please provide either both date and orbit_id, or neither.")

    if not all_data:
        print("No orbit data found — k1k2_analysis cannot proceed.")
        return

    combined = {key: np.concatenate([d[key] for d in all_data])
                for key in all_data[0]}

    # ── Convenience references ─────────────────────────────────────────────────
    xco2_bc          = combined['xco2_bc']
    xco2_raw         = combined['xco2_raw']
    xco2_bc_anomaly  = combined['xco2_bc_anomaly']
    xco2_raw_anomaly = combined['xco2_raw_anomaly']
    cld_dist_km      = combined['cld_dist_km']
    weighted_cloud_dist_km = combined.get('weighted_cloud_dist_km')

    # ── Build combined DataFrame ───────────────────────────────────────────────
    final_dict = {
        # basic identifiers
        'date': combined['date'],
        'time': combined['time'],
        'orbit_id': combined['orbit_id'],
        'lon': combined['lon'],
        'lat': combined['lat'],
        'sza': combined['sza'],
        'vza': combined['vza'],
        # Per-band kappa coefficients (short names for downstream use)
        'o2a_intercept': combined['o2a_intercept_fitting'],
        'wco2_intercept': combined['wco2_intercept_fitting'],
        'sco2_intercept': combined['sco2_intercept_fitting'],
        'o2a_k1': combined['o2a_k1_fitting'],
        'o2a_k2': combined['o2a_k2_fitting'],
        'o2a_k3': combined['o2a_k3_fitting'],
        'o2a_k4': combined['o2a_k4_fitting'],
        'o2a_k5': combined['o2a_k5_fitting'],
        'wco2_k1': combined['wco2_k1_fitting'],
        'wco2_k2': combined['wco2_k2_fitting'],
        'wco2_k3': combined['wco2_k3_fitting'],
        'wco2_k4': combined['wco2_k4_fitting'],
        'wco2_k5': combined['wco2_k5_fitting'],
        'sco2_k1': combined['sco2_k1_fitting'],
        'sco2_k2': combined['sco2_k2_fitting'],
        'sco2_k3': combined['sco2_k3_fitting'],
        'sco2_k4': combined['sco2_k4_fitting'],
        'sco2_k5': combined['sco2_k5_fitting'],
        # Gamma shape parameter κ = k1²/k2 per band
        'o2a_kappa':  combined['o2a_kappa'],
        'wco2_kappa': combined['wco2_kappa'],
        'sco2_kappa': combined['sco2_kappa'],
        # XCO2
        'xco2_apriori':     combined.get('xco2_apriori'),
        'xco2_bc':          xco2_bc,
        'xco2_raw':         xco2_raw,
        'xco2_bc_anomaly':  xco2_bc_anomaly,   # per-footprint, pre-computed
        'xco2_raw_anomaly': xco2_raw_anomaly,  # per-footprint, pre-computed
        'xco2_strong_idp': combined.get('xco2_strong_idp'),
        'xco2_weak_idp': combined.get('xco2_weak_idp'),
        # Geometry
        'mu_sza': combined['mu_sza'],
        'mu_vza': combined['mu_vza'],
        # Cloud distance
        'cld_dist_km': cld_dist_km,
        'weighted_cloud_dist_km': weighted_cloud_dist_km,
        # Lite retrieval variables
        'psfc':    combined.get('psfc'),
        'airmass': combined.get('airmass'),
        'delT':    combined.get('delT'),
        'dp':      combined.get('dp'),
        'dp_o2a':  combined.get('dp_o2a'),
        'dp_sco2': combined.get('dp_sco2'),
        'co2_grad_del': combined.get('co2_grad_del'),
        'alb_o2a': combined.get('alb_o2a'),
        'alb_wco2': combined.get('alb_wco2'),
        'alb_sco2': combined.get('alb_sco2'),
        'aod_total': combined.get('aod_total'),
        'fs_rel': combined.get('fs_rel'),
        'alt':     combined.get('alt'),
        'alt_std': combined.get('alt_std'),
        'xco2_qf': combined.get('xco2_qf'),
        'sfc_type': combined.get('sfc_type'),
        'ws': combined.get('ws'),
        'ws_apriori': combined.get('ws_apriori'),
        # Preprocessor variables
        'co2_ratio_bc': combined.get('co2_ratio_bc'),
        'h2o_ratio_bc': combined.get('h2o_ratio_bc'),
        'csnr_o2a': combined.get('csnr_o2a'),
        'csnr_wco2': combined.get('csnr_wco2'),
        'csnr_sco2': combined.get('csnr_sco2'),
        'dp_abp': combined.get('dp_abp'),
        'h_cont_o2a': combined.get('h_cont_o2a'),
        'h_cont_wco2': combined.get('h_cont_wco2'),
        'h_cont_sco2': combined.get('h_cont_sco2'),
        'max_declock_o2a': combined.get('max_declock_o2a'),
        'max_declock_wco2': combined.get('max_declock_wco2'),
        'max_declock_sco2': combined.get('max_declock_sco2'),
        # Additional retrieval variables
        'h2o_scale': combined.get('h2o_scale'),
        'dpfrac': combined.get('dpfrac'),
        'aod_bc': combined.get('aod_bc'),
        'aod_dust': combined.get('aod_dust'),
        'aod_ice': combined.get('aod_ice'),
        'aod_water': combined.get('aod_water'),
        'aod_oc': combined.get('aod_oc'),
        'aod_seasalt': combined.get('aod_seasalt'),
        'aod_strataer': combined.get('aod_strataer'),
        'aod_sulfate': combined.get('aod_sulfate'),
        'dust_height': combined.get('dust_height'),
        'ice_height': combined.get('ice_height'),
        'dws': combined.get('dws'),
        # Additional sounding variables
        'snr_o2a': combined.get('snr_o2a'),
        'snr_wco2': combined.get('snr_wco2'),
        'snr_sco2': combined.get('snr_sco2'),
        'glint_angle': combined.get('glint_angle'),
        'pol_angle': combined.get('pol_angle'),
        'saa': combined.get('saa'),
        'vaa': combined.get('vaa'),
        'fp':         combined['fp_number'],
        'fp_id':      combined['fp_id'],
        'fp_area_km2': combined['fp_area_km2'],
        # additional Lite variables that may be missing from older output files but are needed for consistency in the combined DataFrame
        "s31": combined.get("s31"),
        "s32": combined.get("s32"),
        "snow_flag": combined.get("snow_flag"),
        "t700": combined.get("t700"),
        "tcwv": combined.get("tcwv"),       
        "operation_mode": combined.get("operation_mode"),
        "water_height": combined.get("water_height"),
        # spectral-fit-quality diagnostics (Tier-A cloud-contamination candidates)
        "chi2_o2a": combined.get("chi2_o2a"),
        "chi2_wco2": combined.get("chi2_wco2"),
        "chi2_sco2": combined.get("chi2_sco2"),
        "rms_rel_o2a": combined.get("rms_rel_o2a"),
        "rms_rel_wco2": combined.get("rms_rel_wco2"),
        "rms_rel_sco2": combined.get("rms_rel_sco2"),
        "eof3_1_rel": combined.get("eof3_1_rel"),
        "diverging_steps": combined.get("diverging_steps"),
        "xco2_uncertainty": combined.get("xco2_uncertainty"),
    }
    
    raa = 180 - np.abs((combined.get('saa') - combined.get('vaa')) % 360 - 180)
    cos_sza = np.cos(np.radians(combined.get('sza')))
    cos_vza = np.cos(np.radians(combined.get('vza')))
    sin_sza = np.sin(np.radians(combined.get('sza')))
    sin_vza = np.sin(np.radians(combined.get('vza')))
    cos_theta = -cos_sza * cos_vza + sin_sza * sin_vza * np.cos(np.radians(raa))
    Phi_cos_theta = 3/4 * (1 + cos_theta**2)
    R_rs_factor = Phi_cos_theta/(4 * cos_sza * cos_vza)
    
    final_dict['xco2_bc_minus_apriori'] = combined.get('xco2_bc') - combined.get('xco2_apriori')
    final_dict['xco2_raw_minus_apriori'] = combined.get('xco2_raw') - combined.get('xco2_apriori')
    final_dict['xco2_bc_minus_raw'] = combined.get('xco2_bc') - combined.get('xco2_raw')
    final_dict['xco2_raw_minus-xco2_strong_idp_minus'] = combined.get('xco2_raw') - combined.get('xco2_strong_idp')
    final_dict['xco2_strong_idp_minus_apriori'] = combined.get('xco2_strong_idp') - combined.get('xco2_apriori')
    final_dict['xco2_weak_idp_minus_apriori'] = combined.get('xco2_weak_idp') - combined.get('xco2_apriori')
    final_dict['xco2_strong_idp_minus_raw'] = combined.get('xco2_strong_idp') - combined.get('xco2_raw')
    final_dict['xco2_strong_idp_minus_xco2_weak_idp'] = combined.get('xco2_strong_idp') - combined.get('xco2_weak_idp')
    final_dict['xco2_strong_idp_over_xco2_weak_idp'] = combined.get('xco2_strong_idp') / combined.get('xco2_weak_idp')
    final_dict['airmass_sq'] = combined.get('airmass') ** 2
    final_dict['alb_o2a_over_cos_sza'] = combined.get('alb_o2a') / cos_sza
    final_dict['alb_wco2_over_cos_sza'] = combined.get('alb_wco2') / cos_sza
    final_dict['alb_sco2_over_cos_sza'] = combined.get('alb_sco2') / cos_sza
    final_dict['sin_raa'] = np.sin(np.radians(raa))
    final_dict['cos_raa'] = np.cos(np.radians(raa))
    final_dict['cos_theta'] = cos_theta
    final_dict['Phi_cos_theta'] = Phi_cos_theta
    final_dict['R_rs_factor'] = R_rs_factor
    
    final_dict['log_P'] = np.log10(combined.get('psfc'))  # Logarithm of surface pressure
    final_dict['dp_psfc_ratio'] = combined.get('dp') / combined.get('psfc')  # Ratio of dp to surface pressure
    final_dict['psfc_prior'] = combined.get('psfc') - combined.get('dp')  # Prior surface pressure estimate
    final_dict['dp_psfc_prior_ratio'] = combined.get('dp') / (combined.get('psfc') - combined.get('dp'))  # Ratio of dp to prior surface pressure
    fs_rel_0 = combined.get('fs_rel')
    fs_rel_0[fs_rel_0 < 0] = 0  # Set any negative relative humidity values to 0
    fs_rel_0[np.isnan(fs_rel_0)] = 0  # Set any NaN relative humidity values to 0
    final_dict['fs_rel_0'] = fs_rel_0  # Relative humidity at surface (assuming fs_rel is at surface)
    final_dict['pol_ang_rad'] = np.radians(combined.get('pol_angle'))  # Convert polarization angle to radians
    
    
    final_dict['o2a_k2_over_k1'] = combined.get('o2a_k2_fitting') / combined.get('o2a_k1_fitting')
    final_dict['wco2_k2_over_k1'] = combined.get('wco2_k2_fitting') / combined.get('wco2_k1_fitting')
    final_dict['sco2_k2_over_k1'] = combined.get('sco2_k2_fitting') / combined.get('sco2_k1_fitting')    
    
    final_dict['exp_o2a_intercept'] = np.exp(combined.get('o2a_intercept_fitting'))
    final_dict['exp_wco2_intercept'] = np.exp(combined.get('wco2_intercept_fitting'))
    final_dict['exp_sco2_intercept'] = np.exp(combined.get('sco2_intercept_fitting'))
    
    final_dict['o2a_exp_intercept-alb'] = np.exp(combined.get('o2a_intercept_fitting')) - combined.get('alb_o2a')
    final_dict['wco2_exp_intercept-alb'] = np.exp(combined.get('wco2_intercept_fitting')) - combined.get('alb_wco2')
    final_dict['sco2_exp_intercept-alb'] = np.exp(combined.get('sco2_intercept_fitting')) - combined.get('alb_sco2')
    
    final_dict['o2a_exp_intercept_over_alb'] = np.exp(combined.get('o2a_intercept_fitting')) / combined.get('alb_o2a')
    final_dict['wco2_exp_intercept_over_alb'] = np.exp(combined.get('wco2_intercept_fitting')) / combined.get('alb_wco2')
    final_dict['sco2_exp_intercept_over_alb'] = np.exp(combined.get('sco2_intercept_fitting')) / combined.get('alb_sco2')

    # ── cross-band ratios (wco2/o2a, sco2/o2a, sco2/wco2) for three quantities:
    #    albedo, exp_intercept, and (exp_intercept − alb). Isolate band-to-band
    #    shape changes that a single-band profile cannot reveal. ───────────────
    _alb = {b: combined.get(f'alb_{b}') for b in ('o2a', 'wco2', 'sco2')}
    _exp = {b: np.exp(combined.get(f'{b}_intercept_fitting')) for b in ('o2a', 'wco2', 'sco2')}
    _ema = {b: _exp[b] - _alb[b] for b in ('o2a', 'wco2', 'sco2')}
    for _num, _den in (('wco2', 'o2a'), ('sco2', 'o2a'), ('sco2', 'wco2')):
        final_dict[f'alb_{_num}_over_{_den}'] = _alb[_num] / _alb[_den]
        final_dict[f'exp_int_{_num}_over_{_den}'] = _exp[_num] / _exp[_den]
        final_dict[f'exp_int_minus_alb_{_num}_over_{_den}'] = _ema[_num] / _ema[_den]


    
    final_dict['1_over_cos_sza'] = 1 / cos_sza
    final_dict['1_over_cos_vza'] = 1 / cos_vza
    
    
    final_dict['cos_glint_angle'] = np.cos(np.radians(combined.get('glint_angle')))
    final_dict['glint_prox'] = np.exp(-1 * combined.get('glint_angle') / 10.0) # Decay constant of 10 degrees

    # add reference clear-sky statistics for downstream use in mitigation or stratification
    final_dict['ref_o2a_k1_mean'] = combined.get('ref_o2a_k1_mean')
    final_dict['ref_o2a_k1_std'] = combined.get('ref_o2a_k1_std')
    final_dict['ref_o2a_k2_mean'] = combined.get('ref_o2a_k2_mean')
    final_dict['ref_o2a_k2_std'] = combined.get('ref_o2a_k2_std')
    final_dict['ref_o2a_k3_mean'] = combined.get('ref_o2a_k3_mean')
    final_dict['ref_o2a_k3_std'] = combined.get('ref_o2a_k3_std')
    final_dict['ref_wco2_k1_mean'] = combined.get('ref_wco2_k1_mean')
    final_dict['ref_wco2_k1_std'] = combined.get('ref_wco2_k1_std')
    final_dict['ref_wco2_k2_mean'] = combined.get('ref_wco2_k2_mean')
    final_dict['ref_wco2_k2_std'] = combined.get('ref_wco2_k2_std')
    final_dict['ref_wco2_k3_mean'] = combined.get('ref_wco2_k3_mean')
    final_dict['ref_wco2_k3_std'] = combined.get('ref_wco2_k3_std')
    final_dict['ref_sco2_k1_mean'] = combined.get('ref_sco2_k1_mean')
    final_dict['ref_sco2_k1_std'] = combined.get('ref_sco2_k1_std')
    final_dict['ref_sco2_k2_mean'] = combined.get('ref_sco2_k2_mean')
    final_dict['ref_sco2_k2_std'] = combined.get('ref_sco2_k2_std')
    final_dict['ref_sco2_k3_mean'] = combined.get('ref_sco2_k3_mean')
    final_dict['ref_sco2_k3_std'] = combined.get('ref_sco2_k3_std')
    final_dict['ref_alb_o2a_mean'] = combined.get('ref_alb_o2a_mean')
    final_dict['ref_alb_o2a_std'] = combined.get('ref_alb_o2a_std')
    final_dict['ref_alb_wco2_mean'] = combined.get('ref_alb_wco2_mean')
    final_dict['ref_alb_wco2_std'] = combined.get('ref_alb_wco2_std')
    final_dict['ref_alb_sco2_mean'] = combined.get('ref_alb_sco2_mean')
    final_dict['ref_alb_sco2_std'] = combined.get('ref_alb_sco2_std')
    final_dict['ref_exp_int_o2a_mean'] = combined.get('ref_exp_int_o2a_mean')
    final_dict['ref_exp_int_o2a_std'] = combined.get('ref_exp_int_o2a_std')
    final_dict['ref_exp_int_wco2_mean'] = combined.get('ref_exp_int_wco2_mean')
    final_dict['ref_exp_int_wco2_std'] = combined.get('ref_exp_int_wco2_std')
    final_dict['ref_exp_int_sco2_mean'] = combined.get('ref_exp_int_sco2_mean')
    final_dict['ref_exp_int_sco2_std'] = combined.get('ref_exp_int_sco2_std')

    # r15 reference set (min_cld_dist=15 km)
    final_dict['xco2_raw_anomaly_r15']      = combined.get('xco2_raw_anomaly_r15')
    final_dict['xco2_bc_anomaly_r15']       = combined.get('xco2_bc_anomaly_r15')
    final_dict['r15_o2a_k1_mean']           = combined.get('r15_o2a_k1_mean')
    final_dict['r15_o2a_k1_std']            = combined.get('r15_o2a_k1_std')
    final_dict['r15_o2a_k2_mean']           = combined.get('r15_o2a_k2_mean')
    final_dict['r15_o2a_k2_std']            = combined.get('r15_o2a_k2_std')
    final_dict['r15_o2a_k3_mean']           = combined.get('r15_o2a_k3_mean')
    final_dict['r15_o2a_k3_std']            = combined.get('r15_o2a_k3_std')
    final_dict['r15_wco2_k1_mean']          = combined.get('r15_wco2_k1_mean')
    final_dict['r15_wco2_k1_std']           = combined.get('r15_wco2_k1_std')
    final_dict['r15_wco2_k2_mean']          = combined.get('r15_wco2_k2_mean')
    final_dict['r15_wco2_k2_std']           = combined.get('r15_wco2_k2_std')
    final_dict['r15_wco2_k3_mean']          = combined.get('r15_wco2_k3_mean')
    final_dict['r15_wco2_k3_std']           = combined.get('r15_wco2_k3_std')
    final_dict['r15_sco2_k1_mean']          = combined.get('r15_sco2_k1_mean')
    final_dict['r15_sco2_k1_std']           = combined.get('r15_sco2_k1_std')
    final_dict['r15_sco2_k2_mean']          = combined.get('r15_sco2_k2_mean')
    final_dict['r15_sco2_k2_std']           = combined.get('r15_sco2_k2_std')
    final_dict['r15_sco2_k3_mean']          = combined.get('r15_sco2_k3_mean')
    final_dict['r15_sco2_k3_std']           = combined.get('r15_sco2_k3_std')
    final_dict['r15_alb_o2a_mean']          = combined.get('r15_alb_o2a_mean')
    final_dict['r15_alb_o2a_std']           = combined.get('r15_alb_o2a_std')
    final_dict['r15_alb_wco2_mean']         = combined.get('r15_alb_wco2_mean')
    final_dict['r15_alb_wco2_std']          = combined.get('r15_alb_wco2_std')
    final_dict['r15_alb_sco2_mean']         = combined.get('r15_alb_sco2_mean')
    final_dict['r15_alb_sco2_std']          = combined.get('r15_alb_sco2_std')
    final_dict['r15_exp_int_o2a_mean']      = combined.get('r15_exp_int_o2a_mean')
    final_dict['r15_exp_int_o2a_std']       = combined.get('r15_exp_int_o2a_std')
    final_dict['r15_exp_int_wco2_mean']     = combined.get('r15_exp_int_wco2_mean')
    final_dict['r15_exp_int_wco2_std']      = combined.get('r15_exp_int_wco2_std')
    final_dict['r15_exp_int_sco2_mean']     = combined.get('r15_exp_int_sco2_mean')
    final_dict['r15_exp_int_sco2_std']      = combined.get('r15_exp_int_sco2_std')



    # Tropopause scalars (pass-through) + sigma-grid atmospheric profiles.
    # Both are no-ops for older fitting_details.h5 that lack the raw profile /
    # tropopause datasets (guarded so no all-None columns are created).
    if 'tropopause_pressure' in combined:
        final_dict['tropopause_pressure'] = combined['tropopause_pressure']
        final_dict['tropopause_temp']     = combined.get('tropopause_temp')
    final_dict.update(compute_sigma_profile_columns(combined))

    df = pd.DataFrame(final_dict)
    df = df[df.xco2_bc > 0]  # Filter out invalid XCO2 value

    if orbit_id is not None:
        stem = f'combined_{date}_orbit_{orbit_id}'
    else:
        stem = f'combined_{date}_all_orbits'
    df.to_parquet(os.path.join(output_dir, f'{stem}.parquet'), index=False, compression='zstd')


def raw_processing_multipe_dates(fdir, date_list, output_fname, n_workers=8):
    """
    Collect single dates' data in the date_list and concatenate into one DataFrame for analysis.

    Prefers Parquet per-date files over CSV (much faster I/O, smaller on disk).
    Uses parallel reads to overlap I/O across files.
    Output format is determined by the extension of output_fname (.parquet or .csv).
    """
    date_set = set(date_list)

    def _get_date(path):
        return os.path.basename(path).split('_')[1]

    # Prefer parquet over csv for per-date input files
    parquet_files = glob.glob(os.path.join(fdir, 'combined_*_all_orbits.parquet'))
    if parquet_files:
        all_files = parquet_files
        reader = pd.read_parquet
    else:
        all_files = glob.glob(os.path.join(fdir, 'combined_*_all_orbits.csv'))
        reader = pd.read_csv

    selected_files = [f for f in all_files if _get_date(f) in date_set]
    if not selected_files:
        print("No files found for the specified dates.")
        return None

    print(f"Reading {len(selected_files)} files with {n_workers} workers...")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        dfs = list(pool.map(reader, selected_files))

    combined_df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    out_path = os.path.join(fdir, output_fname)
    if out_path.endswith('.parquet'):
        combined_df.to_parquet(out_path, index=False, compression='zstd')
    else:
        combined_df.to_csv(out_path, index=False)

    print(f"Written {len(combined_df):,} rows → {out_path}")
    return out_path

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build the per-date feature parquet from spectral-fit outputs.")
    parser.add_argument('--date', default=None,
                        help="Process a single date (YYYY-MM-DD or YYYYMMDD). "
                             "If omitted, the built-in date_list below is used. "
                             "Used by the per-date SLURM launcher "
                             "(curc_shell_blanca_build_feature_dataset_perdate.sh).")
    args = parser.parse_args()

    storage_dir = get_storage_dir()
    fdir      = storage_dir / 'results'
    # # List of dates to process
    # date_list = [
    #             '20200101', '20200201', '20200301', '20200401',
    #              '20200501', '20200601', '20200701', '20200801',
    #              '20200903', '20201001', '20201101', '20201201']  
    
    date_list = [
                #  '20160101', '20160201', '20160301', 
                #                                      '20160405',
                #  '20160501', '20160601', '20160701', '20160801',
                #  '20160901', '20161001', '20161101', '20161201',
                #  '20170101', '20170201', '20170301', '20170401',
                #  '20170501', '20170601', '20170701', 
                #              '20171001', 
                #                          '20171105', 
                #                                      '20171201',
                #  '20180101', '20180201', 
                 '20180301', 
                #  '20180401',
                #  '20180501', '20180601', '20180701', '20180801',
                #  '20180901', '20181001', '20181101', '20181201',
                #  '20190101', '20190201', '20190301', '20190401',
                #  '20190501', '20190601', '20190701', '20190801',
                #  '20190901', '20191001', '20191101', '20191201',
                #  '20200101', '20200201', '20200301', '20200401',
                #  '20200501', '20200601', '20200701', '20200801',
                #  '20200903', 
                #              '20201001', '20201101', '20201201',
                 
                #  '20160115', '20160215', '20160315', '20160415',
                #  '20160515', '20160615', '20160715', 
                #                                      '20160821',
                #  '20160915', '20161015', '20161115', '20161215',
                #  '20170115', '20170215', '20170315', '20170415',
                #  '20170515', '20170615', '20170715', 
                #              '20171015', '20171115', '20171215',
                #  '20180115', 
                #              '20180212', 
                #                          '20180315', '20180415',
                #  '20180515', '20180615', '20180715', '20180815',
                #  '20180915', '20181015', 
                #                          '20181117', 
                #                                      '20181215',
                #  '20190115', '20190215', '20190315', '20190415',
                #  '20190515', '20190615', '20190715', '20190815',
                #  '20190915', '20191015', '20191115', '20191215',
                #  '20200115', '20200215', '20200315', '20200415',
                #  '20200515', '20200615', '20200715', '20200815',
                #  '20200915', '20201015', '20201115', '20201215'
                 ]
    
    # date_list = [
                #  '20141203', '20141217', 
                #  '20150213', '20150218', '20150317', '20150323',
                #  '20150518', '20150520', '20150605', '20150629',  
                #  '20150706', '20150713', '20150714', 
                #  '20150723', '20160910', '20150925',
                 
                #  '20160303', '20160506', '20160529', '20160607',
                #  '20160709', '20160911', '20161013', 
                #  '20161105', '20161107',
                 
                #  '20170117', '20170322', '20170423', '20170516',
                #  '20170525', '20170617', '20170626', '20171203', 
                 
                #  '20180221', '20180313', '20180410', '20180512', 
                #  '20180603', '20180710', '20180807', '20180817',
                #  '20180902', '20180908', '20180917', '20181010', 
                #  '20181024', '20181129', '20181130',
                 
                #  '20190313', '20190429', '20190710', '20190730', 
                #  '20190914',
                 
                #  '20200211', '20200314', '20200330', '20200331',
                #  '20200405', '20200415', '20200517', 
                #  '20200711', '20200726', '20200906', '20200916', 
                #  '20201005', '20201223', '20201224', 
                 
                #  '20210210', '20210318', '20210329', '20210424', 
                #  '20210526', '20210609', '20210621', 
                #  '20210703', '20210727', '20210825', 
                #  '20210906', '20210908', '20210926', 
                #  '20211016', '20211229',
                  
                # ]  
    
        
    # --date overrides the built-in list (one date per SLURM job).  Accepts
    # YYYY-MM-DD or YYYYMMDD; normalise to the YYYYMMDD the loop expects.
    if args.date:
        date_list = [args.date.replace('-', '')]

    for date in date_list:
        date_dt = datetime.strptime(date, '%Y%m%d')
        print(f"Processing date: {date_dt.strftime('%Y-%m-%d')}")
        raw_processing_single_date(result_dir=fdir, date=date_dt.strftime('%Y-%m-%d'), orbit_id=None)

    # Per-date mode (SLURM launcher): build only this date's parquet and stop.
    # The multi-date combine below must run once, after ALL per-date jobs finish
    # — never inside a per-date job — so skip it whenever --date is given.
    if args.date:
        return

    # date_list_hyphen = [datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d') for date in date_list]
    # csv_output_dir = os.path.join(fdir, 'csv_collection')
    # output_fname = 'combined_2016_2020_dates.parquet'
    # # output_fname = 'combined_2020_dates.parquet'
    # raw_processing_multipe_dates(fdir=csv_output_dir, date_list=date_list_hyphen, output_fname=output_fname)

if __name__ == "__main__":
    main()

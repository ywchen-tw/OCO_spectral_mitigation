import h5py
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
import copy
import glob
import pathlib
import gc


# ─── HDF5 loader ──────────────────────────────────────────────────────────────

def load_output_dict(filepath):
    """Load one orbit's fitting_details.h5 written by process_orbit().

    Returns a flat dict with all per-band kappa arrays plus shared variables.
    Keys that are absent from the file are silently skipped.
    """
    keys = [
        # Cumulant coefficients (per band)
        'o2a_k1_fitting', 'o2a_k2_fitting', 'o2a_k3_fitting',
        'o2a_k4_fitting', 'o2a_k5_fitting', 'o2a_intercept_fitting',
        'wco2_k1_fitting', 'wco2_k2_fitting', 'wco2_k3_fitting',
        'wco2_k4_fitting', 'wco2_k5_fitting', 'wco2_intercept_fitting',
        'sco2_k1_fitting', 'sco2_k2_fitting', 'sco2_k3_fitting',
        'sco2_k4_fitting', 'sco2_k5_fitting', 'sco2_intercept_fitting',
        # Geometry
        'lon', 'lat', 'sza', 'vza', 'mu_sza', 'mu_vza', 'fp_number', 'fp_id',
        # Cloud proximity
        'cld_dist_km',
        # XCO2
        'xco2_bc', 'xco2_raw', 'xco2_bc_anomaly', 'xco2_raw_anomaly',
        # Lite retrieval variables
        'psfc_lt', 'airmass_lt', 'delT_lt', 'dp_lt', 'dp_o2a_lt', 'dp_sco2_lt', 'co2_grad_del_lt',
        'alb_o2a_lt', 'alb_wco2_lt', 'alb_sco2_lt', 'aod_total_lt', 'fs_rel_lt',
        'alt_lt', 'alt_std_lt', 'xco2_qf_lt', 'sfc_type_lt', 'ws_lt', 'ws_apriori_lt',
        # Preprocessor variables
        'co2_ratio_bc_lt', 'h2o_ratio_bc_lt',
        'csnr_o2a_lt', 'csnr_wco2_lt', 'csnr_sco2_lt',
        'dp_abp_lt',
        'h_cont_o2a_lt', 'h_cont_wco2_lt', 'h_cont_sco2_lt',
        'max_declock_o2a_lt', 'max_declock_wco2_lt', 'max_declock_sco2_lt',
        'xco2_strong_idp_lt', 'xco2_weak_idp_lt',
        # Additional retrieval variables
        'h2o_scale_lt', 'dpfrac_lt',
        'aod_bc_lt', 'aod_dust_lt', 'aod_ice_lt', 'aod_water_lt',
        'aod_oc_lt', 'aod_seasalt_lt', 'aod_strataer_lt', 'aod_sulfate_lt',
        'dust_height_lt', 'ice_height_lt', 'dws_lt',
        # Additional sounding variables
        'snr_o2a_lt', 'snr_wco2_lt', 'snr_sco2_lt',
        'glint_angle_lt', 'pol_angle_lt', 'saa_lt', 'vaa_lt',
        # additional Lite variables that may be missing from older output files but are needed for consistency in the combined DataFrame
        "s31", "s32", "snow_flag", "t700", "tcwv", "operation_mode", "water_height_lt"
    ]
    with h5py.File(filepath, 'r') as f:
        return {key: f[key][()] for key in keys if key in f}



def main():
    fdir = '.'
    
def raw_processing_single_date(sat, orbit_id=None, reference_csv=None):
    """Load fitting results for all orbits, build combined DataFrame, run analysis.

    Parameters
    ----------
    sat : dict
        Pipeline state from preprocess() with keys: 'date', 'result_dir', 'orbit_list'.
    reference_csv : str or None
        Optional path to a CSV from a reference scene for cross-scene mitigation.
    """
    date       = sat['date'].strftime("%Y-%m-%d")
    result_dir = sat['result_dir']
    orbit_list = sat['orbit_list']
    output_dir = f"{result_dir}/csv_collection"
    if orbit_id is not None:
        output_dir = f"{output_dir}/{orbit_id}"
    os.makedirs(output_dir, exist_ok=True)

    # ── Collect and concatenate data across all orbits ─────────────────────────
    all_data = []
    if orbit_id is None:
        for orbit_id in orbit_list:
            h5_output_dir = f"{sat['result_dir']}/fitting_details"
            filepath = f"{h5_output_dir}/fitting_details_{date}_{orbit_id}.h5"
            if os.path.exists(filepath):
                all_data.append(load_output_dict(filepath))
            else:
                print(f"File not found: {filepath}")
    else:
        h5_output_dir = f"{sat['result_dir']}/fitting_details"
        filepath = f"{h5_output_dir}/fitting_details_{date}_{orbit_id}.h5"
        if os.path.exists(filepath):
            all_data.append(load_output_dict(filepath))
        else:
            print(f"File not found: {filepath}")

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

    # ── Build combined DataFrame ───────────────────────────────────────────────
    final_dict = {
        'lon': combined['lon'],
        'lat': combined['lat'],
        # Per-band kappa coefficients (short names for downstream use)
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
        # XCO2
        'xco2_bc':          xco2_bc,
        'xco2_raw':         xco2_raw,
        'xco2_bc_anomaly':  xco2_bc_anomaly,   # per-footprint, pre-computed
        'xco2_raw_anomaly': xco2_raw_anomaly,  # per-footprint, pre-computed
        # Geometry
        'lon': combined['lon'],
        'lat': combined['lat'],
        'mu_sza': combined['mu_sza'],
        'mu_vza': combined['mu_vza'],
        # Cloud distance
        'cld_dist_km': cld_dist_km,
        # Lite retrieval variables
        'psfc':    combined.get('psfc_lt'),
        'airmass': combined.get('airmass_lt'),
        'delT':    combined.get('delT_lt'),
        'dp':      combined.get('dp_lt'),
        'dp_o2a':  combined.get('dp_o2a_lt'),
        'dp_sco2': combined.get('dp_sco2_lt'),
        'co2_grad_del': combined.get('co2_grad_del_lt'),
        'alb_o2a': combined.get('alb_o2a_lt'),
        'alb_wco2': combined.get('alb_wco2_lt'),
        'alb_sco2': combined.get('alb_sco2_lt'),
        'aod_total': combined.get('aod_total_lt'),
        'fs_rel': combined.get('fs_rel_lt'),
        'alt':     combined.get('alt_lt'),
        'alt_std': combined.get('alt_std_lt'),
        'xco2_qf': combined.get('xco2_qf_lt'),
        'sfc_type': combined.get('sfc_type_lt'),
        'ws': combined.get('ws_lt'),
        'ws_apriori': combined.get('ws_apriori_lt'),
        # Preprocessor variables
        'co2_ratio_bc': combined.get('co2_ratio_bc_lt'),
        'h2o_ratio_bc': combined.get('h2o_ratio_bc_lt'),
        'csnr_o2a': combined.get('csnr_o2a_lt'),
        'csnr_wco2': combined.get('csnr_wco2_lt'),
        'csnr_sco2': combined.get('csnr_sco2_lt'),
        'dp_abp': combined.get('dp_abp_lt'),
        'h_cont_o2a': combined.get('h_cont_o2a_lt'),
        'h_cont_wco2': combined.get('h_cont_wco2_lt'),
        'h_cont_sco2': combined.get('h_cont_sco2_lt'),
        'max_declock_o2a': combined.get('max_declock_o2a_lt'),
        'max_declock_wco2': combined.get('max_declock_wco2_lt'),
        'max_declock_sco2': combined.get('max_declock_sco2_lt'),
        'xco2_strong_idp': combined.get('xco2_strong_idp_lt'),
        'xco2_weak_idp': combined.get('xco2_weak_idp_lt'),
        # Additional retrieval variables
        'h2o_scale': combined.get('h2o_scale_lt'),
        'dpfrac': combined.get('dpfrac_lt'),
        'aod_bc': combined.get('aod_bc_lt'),
        'aod_dust': combined.get('aod_dust_lt'),
        'aod_ice': combined.get('aod_ice_lt'),
        'aod_water': combined.get('aod_water_lt'),
        'aod_oc': combined.get('aod_oc_lt'),
        'aod_seasalt': combined.get('aod_seasalt_lt'),
        'aod_strataer': combined.get('aod_strataer_lt'),
        'aod_sulfate': combined.get('aod_sulfate_lt'),
        'dust_height': combined.get('dust_height_lt'),
        'ice_height': combined.get('ice_height_lt'),
        'dws': combined.get('dws_lt'),
        # Additional sounding variables
        'snr_o2a': combined.get('snr_o2a_lt'),
        'snr_wco2': combined.get('snr_wco2_lt'),
        'snr_sco2': combined.get('snr_sco2_lt'),
        'glint_angle': combined.get('glint_angle_lt'),
        'pol_angle': combined.get('pol_angle_lt'),
        'saa': combined.get('saa_lt'),
        'vaa': combined.get('vaa_lt'),
        'fp':         combined['fp_number'],
        'fp_id':      combined['fp_id'],
        # additional Lite variables that may be missing from older output files but are needed for consistency in the combined DataFrame
        "s31": combined.get("s31"),
        "s32": combined.get("s32"),
        "snow_flag": combined.get("snow_flag"),
        "t700": combined.get("t700"),
        "tcwv": combined.get("tcwv"),       
        "operation_mode": combined.get("operation_mode"),
        "water_height": combined.get("water_height_lt"),
    }
    
    raa = 180 - np.abs((combined.get('saa_lt') - combined.get('vaa_lt')) % 360 - 180)
    cos_sza = np.cos(np.radians(combined.get('sza')))
    cos_vza = np.cos(np.radians(combined.get('vza')))
    sin_sza = np.sin(np.radians(combined.get('sza')))
    sin_vza = np.sin(np.radians(combined.get('vza')))
    cos_theta = -cos_sza * cos_vza + sin_sza * sin_vza * np.cos(np.radians(raa))
    Phi_cos_theta = 3/4 * (1 + cos_theta**2)
    R_rs_factor = Phi_cos_theta/(4 * cos_sza * cos_vza)
    
    final_dict['sin_raa'] = np.sin(np.radians(raa))
    final_dict['cos_raa'] = np.cos(np.radians(raa))
    final_dict['cos_theta'] = cos_theta
    final_dict['Phi_cos_theta'] = Phi_cos_theta
    final_dict['R_rs_factor'] = R_rs_factor
    
    final_dict['log_P'] = np.log10(combined.get('psfc_lt'))  # Logarithm of surface pressure
    final_dict['dp_psfc_ratio'] = combined.get('dp_lt') / combined.get('psfc_lt')  # Ratio of dp to surface pressure
    fs_rel_0 = combined.get('fs_rel_lt')
    fs_rel_0[fs_rel_0 < 0] = 0  # Set any negative relative humidity values to 0
    fs_rel_0[np.isnan(fs_rel_0)] = 0  # Set any NaN relative humidity values to 0
    final_dict['fs_rel_0'] = fs_rel_0  # Relative humidity at surface (assuming fs_rel_lt is at surface)
    final_dict['pol_ang_rad'] = np.radians(combined.get('pol_angle_lt'))  # Convert polarization angle to radians
    
    
    final_dict['cos_glint_angle'] = np.cos(np.radians(combined.get('glint_angle_lt')))
    final_dict['glint_prox'] = np.exp(-1 * combined.get('glint_angle_lt') / 10.0) # Decay constant of 10 degrees

    df = pd.DataFrame(final_dict)
    df.to_csv(os.path.join(output_dir, f'combined_orbits_{date}.csv'), index=False)


def raw_processing_multipe_dates(fdir, date_list, output_fname):
    """
    Collect single dates' data in the date_list and concatenate into one DataFrame for analysis.
    """
    files_list = glob.glob(os.path.join(fdir, 'combined_orbits_*.csv'))
    dates = [os.path.basename(f).split('_')[2].split('.')[0] for f in files_list]
    selected_files = [f for f, d in zip(files_list, dates) if d in date_list]
    if not selected_files:
        print("No files found for the specified dates.")
        return None
    
    combined_df = pd.concat([pd.read_csv(f) for f in selected_files], ignore_index=True)
    combined_df.to_csv(os.path.join(fdir, output_fname), index=False)
    
    return os.path.join(fdir, output_fname)
    

if __name__ == "__main__":
    main()

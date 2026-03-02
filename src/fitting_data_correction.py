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
        # Geometry
        'lon', 'lat', 'sza', 'vza', 'mu_sza', 'mu_vza', 'fp_number', 'fp_id',
        # Cloud proximity
        'cld_dist_km',
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
        "s31", "s32", "snow_flag", "t700", "tcwv", "operation_mode", "water_height"
    ]
    with h5py.File(filepath, 'r') as f:
        return {key: f[key][()] for key in keys if key in f}



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

    # ── Build combined DataFrame ───────────────────────────────────────────────
    final_dict = {
        # basic identifiers
        'date': combined['date'],
        'time': combined['time'],
        'orbit_id': combined['orbit_id'],
        'lon': combined['lon'],
        'lat': combined['lat'],
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
        # XCO2
        'xco2_apriori':     combined.get('xco2_apriori'),
        'xco2_bc':          xco2_bc,
        'xco2_raw':         xco2_raw,
        'xco2_bc_anomaly':  xco2_bc_anomaly,   # per-footprint, pre-computed
        'xco2_raw_anomaly': xco2_raw_anomaly,  # per-footprint, pre-computed
        # Geometry
        'mu_sza': combined['mu_sza'],
        'mu_vza': combined['mu_vza'],
        # Cloud distance
        'cld_dist_km': cld_dist_km,
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
        'xco2_strong_idp': combined.get('xco2_strong_idp'),
        'xco2_weak_idp': combined.get('xco2_weak_idp'),
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
        # additional Lite variables that may be missing from older output files but are needed for consistency in the combined DataFrame
        "s31": combined.get("s31"),
        "s32": combined.get("s32"),
        "snow_flag": combined.get("snow_flag"),
        "t700": combined.get("t700"),
        "tcwv": combined.get("tcwv"),       
        "operation_mode": combined.get("operation_mode"),
        "water_height": combined.get("water_height"),
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
    fs_rel_0 = combined.get('fs_rel')
    fs_rel_0[fs_rel_0 < 0] = 0  # Set any negative relative humidity values to 0
    fs_rel_0[np.isnan(fs_rel_0)] = 0  # Set any NaN relative humidity values to 0
    final_dict['fs_rel_0'] = fs_rel_0  # Relative humidity at surface (assuming fs_rel is at surface)
    final_dict['pol_ang_rad'] = np.radians(combined.get('pol_angle'))  # Convert polarization angle to radians
    
    
    final_dict['cos_glint_angle'] = np.cos(np.radians(combined.get('glint_angle')))
    final_dict['glint_prox'] = np.exp(-1 * combined.get('glint_angle') / 10.0) # Decay constant of 10 degrees

    df = pd.DataFrame(final_dict)
    df = df[df.xco2_bc > 0]  # Filter out invalid XCO2 value
    
    if orbit_id is not None:
        df.to_csv(os.path.join(output_dir, f'combined_{date}_orbit_{orbit_id}.csv'), index=False)
    else:
        df.to_csv(os.path.join(output_dir, f'combined_{date}_all_orbits.csv'), index=False)


def raw_processing_multipe_dates(fdir, date_list, output_fname):
    """
    Collect single dates' data in the date_list and concatenate into one DataFrame for analysis.
    """
    files_list = glob.glob(os.path.join(fdir, 'combined_*_all_orbits.csv'))
    dates = [os.path.basename(f).split('_')[1] for f in files_list]
    selected_files = [f for f, d in zip(files_list, dates) if d in date_list]
    if not selected_files:
        print("No files found for the specified dates.")
        return None
    
    combined_df = pd.concat([pd.read_csv(f) for f in selected_files], ignore_index=True)
    combined_df.to_csv(os.path.join(fdir, output_fname), index=False)
    
    return os.path.join(fdir, output_fname)

def main():
    storage_dir = get_storage_dir()
    fdir      = storage_dir / 'results'
    # # List of dates to process
    # date_list = ['20190101', '20190201', '20190301', '20190401',
    #             '20190501', '20190601', '20190701', '20190801',
    #             '20190901', '20191001', '20191101', '20191201',
    #             '20200101', '20200201', '20200301', '20200401',
    #              '20200501', '20200601', '20200701', '20200801',
    #              '20200903', '20201001', '20201101', '20201201']  
    
    date_list = [
                #  '20160101', '20160201', '20160301', '20160405',
                #  '20160501', '20160601', '20160701', '20160801',
                #  '20160901', '20161001', '20161101', '20161201',
                 '20170101', '20170201', '20170301', '20170401',
                 '20170501', '20170601', '20170701', 
                             '20171001', '20171105', '20171201',
                #  '20180101', '20180201', '20180301', '20180401',
                #  '20180501', '20180601', '20180701', '20180801',
                #  '20180901', '20181001', '20181101', '20181201',
                #  '20190101', '20190201', '20190301', '20190401',
                #  '20190501', '20190601', '20190701', '20190801',
                #  '20190901', '20191001', '20191101', '20191201',
                #  '20200101', '20200201', '20200301', '20200401',
                #  '20200501', '20200601', '20200701', '20200801',
                #  '20200903', '20201001', '20201101', '20201201'
                 ]  
    
    # date_list = ['20180221', '20180313', '20180710', '20180902',
    #              '20181024', '20181129', '20181130',
    #              '20200115', '20200211', '20200330', '20200415', 
    #              '20200517', '20200906',
    #              '20201005', '20201224', 
    #              '20210210', '20210424', '20211229']  
    
    # date_list = ['20200101', '20190101']  
    
        
    for date in date_list:
        date_dt = datetime.strptime(date, '%Y%m%d')
        print(f"Processing date: {date_dt.strftime('%Y-%m-%d')}")
        raw_processing_single_date(result_dir=fdir, date=date_dt.strftime('%Y-%m-%d'), orbit_id=None)

    # date_list_hyphen = [datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d') for date in date_list]
    # csv_output_dir = os.path.join(fdir, 'csv_collection')
    # output_fname = 'combined_2017_2020_dates.csv'
    # raw_processing_multipe_dates(fdir=csv_output_dir, date_list=date_list_hyphen, output_fname=output_fname)

if __name__ == "__main__":
    main()

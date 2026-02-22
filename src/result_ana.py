import h5py
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import torch
import torch.nn as nn
import copy
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
    ]
    with h5py.File(filepath, 'r') as f:
        return {key: f[key][()] for key in keys if key in f}


# ─── Main analysis entry point ─────────────────────────────────────────────────

def main():
    fdir = '.'
    k1k2_analysis(fdir=fdir)


def k1k2_analysis(sat, orbit_id=None, reference_csv=None):
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
    output_dir = f"{result_dir}/{date}"
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

    # Global clear-sky means (footprints > 10 km from cloud) — used for colour centering
    xco2_bc_near_cld_mean  = np.nanmean(xco2_bc[cld_dist_km > 10])
    xco2_raw_near_cld_mean = np.nanmean(xco2_raw[cld_dist_km > 10])

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
        'psfc_lt':    combined.get('psfc_lt'),
        'airmass_lt': combined.get('airmass_lt'),
        'delT_lt':    combined.get('delT_lt'),
        'dp_lt':      combined.get('dp_lt'),
        'dp_o2a_lt':  combined.get('dp_o2a_lt'),
        'dp_sco2_lt': combined.get('dp_sco2_lt'),
        'co2_grad_del_lt': combined.get('co2_grad_del_lt'),
        'alb_o2a_lt': combined.get('alb_o2a_lt'),
        'alb_wco2_lt': combined.get('alb_wco2_lt'),
        'alb_sco2_lt': combined.get('alb_sco2_lt'),
        'aod_total_lt': combined.get('aod_total_lt'),
        'fs_rel_lt': combined.get('fs_rel_lt'),
        'alt_lt':     combined.get('alt_lt'),
        'alt_lt_std': combined.get('alt_std_lt'),
        'xco2_qf_lt': combined.get('xco2_qf_lt'),
        'sfc_type_lt': combined.get('sfc_type_lt'),
        'ws_lt': combined.get('ws_lt'),
        'ws_apriori_lt': combined.get('ws_apriori_lt'),
        # Preprocessor variables
        'co2_ratio_bc_lt': combined.get('co2_ratio_bc_lt'),
        'h2o_ratio_bc_lt': combined.get('h2o_ratio_bc_lt'),
        'csnr_o2a_lt': combined.get('csnr_o2a_lt'),
        'csnr_wco2_lt': combined.get('csnr_wco2_lt'),
        'csnr_sco2_lt': combined.get('csnr_sco2_lt'),
        'dp_abp_lt': combined.get('dp_abp_lt'),
        'h_cont_o2a_lt': combined.get('h_cont_o2a_lt'),
        'h_cont_wco2_lt': combined.get('h_cont_wco2_lt'),
        'h_cont_sco2_lt': combined.get('h_cont_sco2_lt'),
        'max_declock_o2a_lt': combined.get('max_declock_o2a_lt'),
        'max_declock_wco2_lt': combined.get('max_declock_wco2_lt'),
        'max_declock_sco2_lt': combined.get('max_declock_sco2_lt'),
        'xco2_strong_idp_lt': combined.get('xco2_strong_idp_lt'),
        'xco2_weak_idp_lt': combined.get('xco2_weak_idp_lt'),
        # Additional retrieval variables
        'h2o_scale_lt': combined.get('h2o_scale_lt'),
        'dpfrac_lt': combined.get('dpfrac_lt'),
        'aod_bc_lt': combined.get('aod_bc_lt'),
        'aod_dust_lt': combined.get('aod_dust_lt'),
        'aod_ice_lt': combined.get('aod_ice_lt'),
        'aod_water_lt': combined.get('aod_water_lt'),
        'aod_oc_lt': combined.get('aod_oc_lt'),
        'aod_seasalt_lt': combined.get('aod_seasalt_lt'),
        'aod_strataer_lt': combined.get('aod_strataer_lt'),
        'aod_sulfate_lt': combined.get('aod_sulfate_lt'),
        'dust_height_lt': combined.get('dust_height_lt'),
        'ice_height_lt': combined.get('ice_height_lt'),
        'dws_lt': combined.get('dws_lt'),
        # Additional sounding variables
        'snr_o2a_lt': combined.get('snr_o2a_lt'),
        'snr_wco2_lt': combined.get('snr_wco2_lt'),
        'snr_sco2_lt': combined.get('snr_sco2_lt'),
        'glint_angle_lt': combined.get('glint_angle_lt'),
        'pol_angle_lt': combined.get('pol_angle_lt'),
        'saa_lt': combined.get('saa_lt'),
        'vaa_lt': combined.get('vaa_lt'),
        'fp':         combined['fp_number'],
        'fp_id':      combined['fp_id'],
    }
    
    raa = 180 - np.abs((combined.get('saa_lt') - combined.get('vaa_lt')) % 360 - 180)
    cos_sza = np.cos(np.radians(combined.get('sza')))
    cos_vza = np.cos(np.radians(combined.get('vza')))
    sin_sza = np.sin(np.radians(combined.get('sza')))
    sin_vza = np.sin(np.radians(combined.get('vza')))
    cos_theta = -cos_sza * cos_vza + sin_sza * sin_vza * np.cos(np.radians(raa))
    Phi_cos_theta = 3/4 * (1 + cos_theta**2)
    R_rs_factor = Phi_cos_theta/(4 * cos_sza * cos_vza)
    
    final_dict['cos_raa'] = np.cos(np.radians(raa))
    final_dict['cos_theta'] = cos_theta
    final_dict['Phi_cos_theta'] = Phi_cos_theta
    final_dict['R_rs_factor'] = R_rs_factor

    df = pd.DataFrame(final_dict)
    df.to_csv(os.path.join(output_dir, 'combined_k1_k2_individual_fp.csv'), index=False)

    # ── Per-footprint plots ────────────────────────────────────────────────────
    fp_output_dir = os.path.join(output_dir, 'fp_{}_plots')
    # for fp in ['all'] + list(np.arange(8)):
    #     plot_comparison(df, fp=fp, output_dir=fp_output_dir.format(fp),
    #                     xco2_bc_near_cld_mean=xco2_bc_near_cld_mean)

    mitigation_test(sat, df, output_dir=output_dir, reference_csv=reference_csv)


# ─── Per-footprint scatter / regression plots ──────────────────────────────────

def plot_comparison(df, fp='all', output_dir='.', xco2_bc_near_cld_mean=np.nan):
    """Scatter / regression plots for one footprint (or all combined).

    Parameters
    ----------
    df : DataFrame from k1k2_analysis
    fp : int 0-7, or 'all'
    output_dir : str
    xco2_bc_near_cld_mean : float  used to centre the anomaly colour scale
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = df[df['sfc_type_lt'] == 0]  # Ocean only for now
    df = df[np.isfinite(df['xco2_bc_anomaly']) & np.isfinite(df['xco2_raw_anomaly'])]  # Only valid XCO2 values

    if fp in np.arange(8):
        df = df[df['fp'] == fp]
    elif fp == 'all':
        pass
    else:
        raise ValueError(f"Invalid fp value: {fp}. Must be 'all' or an integer in [0, 7].")

    tags = ['o2a', 'wco2', 'sco2']

    lon             = df['lon']
    lat             = df['lat']
    xco2_bc         = df['xco2_bc']
    xco2_bc_anomaly = df['xco2_bc_anomaly']
    xco2_raw        = df['xco2_raw']
    xco2_raw_anomaly= df['xco2_raw_anomaly']
    psfc_lt         = df['psfc_lt']
    dp_lt           = df['dp_lt']
    dp_o2a_lt       = df['dp_o2a_lt']
    dp_sco2_lt      = df['dp_sco2_lt']
    
    alt_lt          = df['alt_lt']
    cld_dist_km     = df['cld_dist_km']

    # ── k1 vs k2..k5 scatter coloured by xco2_raw ──────────────────────────
    fig, ((ax11, ax12, ax13),
          (ax21, ax22, ax23),
          (ax31, ax32, ax33),
          ) = plt.subplots(3, 3, figsize=(18, 15))
    for tag, ax1, ax2, ax3 in zip(
            tags,
            [ax11, ax12, ax13], 
            [ax21, ax22, ax23],
            [ax31, ax32, ax33], ):
        k1 = df[f'{tag}_k1']
        for ax, i in zip([ax1, ax2, ax3], [2, 3, 4]):
            sc = ax.scatter(k1, df[f'{tag}_k{i}'], alpha=0.7, c=xco2_raw, cmap='jet')
            mask = np.isfinite(k1) & np.isfinite(df[f'{tag}_k{i}'])
            if np.sum(mask) < 10:
                ax.set_xlabel('k1')
                continue
            slope, intercept = np.polyfit(k1[mask], df[f'{tag}_k{i}'][mask], 1)
            ax.plot(k1, slope * k1 + intercept, '--', color='red',
                    label=f'Fit: y={slope:.2f}x+{intercept:.2f}', alpha=0.7)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            corr = np.corrcoef(k1[mask], df[f'{tag}_k{i}'][mask])[0, 1]
            ax.text(0.05 * (xmax - xmin) + xmin, 0.95 * (ymax - ymin) + ymin,
                    f"Corr: {corr:.3f}", fontsize=10, verticalalignment='top')
            ax.text(0.05 * (xmax - xmin) + xmin, 0.90 * (ymax - ymin) + ymin,
                    f"Slope: {slope:.3e}", fontsize=10, verticalalignment='top')
            ax.text(0.05 * (xmax - xmin) + xmin, 0.85 * (ymax - ymin) + ymin,
                    f"Intercept: {intercept:.3e}", fontsize=10, verticalalignment='top')
            ax.set_xlabel('k1')
            ax.set_ylabel(f'k{i}')
            ax.set_title(f'{tag} k1 vs k{i} coloured by xco2_raw')
            fig.colorbar(sc, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'k1_ki_cbar_xco2_raw_{fp}.png'), dpi=150)
    plt.close(fig)

    # ── k1, k2 vs cloud distance ────────────────────────────────────────────
    fig, ((ax11, ax12, ax13),
          (ax21, ax22, ax23)) = plt.subplots(2, 3, figsize=(18, 10))
    scatter_arg = {'cmap': 'jet', 'alpha': 0.7, 'c': xco2_raw}
    cb_label = 'xco2_raw'
    sc11 = ax11.scatter(df['o2a_k1'],  cld_dist_km, **scatter_arg)
    ax11.set_title('O2A k1 vs cld_dist_km');  fig.colorbar(sc11, ax=ax11, label=cb_label)
    sc12 = ax12.scatter(df['o2a_k2'],  cld_dist_km, **scatter_arg)
    ax12.set_title('O2A k2 vs cld_dist_km');  fig.colorbar(sc12, ax=ax12, label=cb_label)
    sc13 = ax13.scatter(df['wco2_k1'], cld_dist_km, **scatter_arg)
    ax13.set_title('WCO2 k1 vs cld_dist_km'); fig.colorbar(sc13, ax=ax13, label=cb_label)
    sc21 = ax21.scatter(df['wco2_k2'], cld_dist_km, **scatter_arg)
    ax21.set_title('WCO2 k2 vs cld_dist_km'); fig.colorbar(sc21, ax=ax21, label=cb_label)
    sc22 = ax22.scatter(df['sco2_k1'], cld_dist_km, **scatter_arg)
    ax22.set_title('SCO2 k1 vs cld_dist_km'); fig.colorbar(sc22, ax=ax22, label=cb_label)
    sc23 = ax23.scatter(df['sco2_k2'], cld_dist_km, **scatter_arg)
    ax23.set_title('SCO2 k2 vs cld_dist_km'); fig.colorbar(sc23, ax=ax23, label=cb_label)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'k1_k2_cld_dist_km_xco2_comparison_{fp}.png'), dpi=150)
    plt.close(fig)

    # ── Linear regression: feature sets → target variables ─────────────────

    X = np.vstack([df.o2a_k1, df.wco2_k1, df.sco2_k1,
                   df.o2a_k2, df.sco2_k2,
                   df.delT_lt, df.dp_lt,
                   df.psfc_lt,
                   df.dp_lt/df.psfc_lt,
                   df.co2_grad_del_lt,
                   df.alb_o2a_lt,
                   df.aod_total_lt,
                   df.mu_sza, df.mu_vza,
                   df.ws_lt,
                   # Preprocessor variables
                   df.co2_ratio_bc_lt, df.h2o_ratio_bc_lt,
                   df.csnr_o2a_lt, df.csnr_wco2_lt, df.csnr_sco2_lt,
                   df.dp_abp_lt,
                   df.h_cont_o2a_lt, df.h_cont_wco2_lt, df.h_cont_sco2_lt,
                   df.max_declock_o2a_lt, df.max_declock_wco2_lt, df.max_declock_sco2_lt,
                   df.xco2_strong_idp_lt, df.xco2_weak_idp_lt,
                   # Additional retrieval variables
                   df.h2o_scale_lt, df.dpfrac_lt,
                   df.aod_bc_lt, df.aod_dust_lt, df.aod_ice_lt, df.aod_water_lt,
                   df.aod_oc_lt, df.aod_seasalt_lt, df.aod_strataer_lt, df.aod_sulfate_lt,
                   df.dust_height_lt, df.ice_height_lt, df.dws_lt,
                   # Additional sounding variables
                   df.snr_o2a_lt, df.snr_wco2_lt, df.snr_sco2_lt,
                   df.glint_angle_lt, df.pol_angle_lt, df.saa_lt, df.vaa_lt,
                   ]).T

    X_var_names = ['o2a_k1', 'wco2_k1', 'sco2_k1',
                   'o2a_k2', 'sco2_k2',
                   'delta T', 'dp',
                   'psfc', 'dp/psfc', 'co2_grad_del',
                   'alb_o2a', 'aod_total',
                   'mu_sza', 'mu_vza', 'ws',
                   # Preprocessor
                   'co2_ratio_bc', 'h2o_ratio_bc',
                   'csnr_o2a', 'csnr_wco2', 'csnr_sco2',
                   'dp_abp',
                   'h_cont_o2a', 'h_cont_wco2', 'h_cont_sco2',
                   'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2',
                   'xco2_strong_idp', 'xco2_weak_idp',
                   # Retrieval
                   'h2o_scale', 'dpfrac',
                   'aod_bc', 'aod_dust', 'aod_ice', 'aod_water',
                   'aod_oc', 'aod_seasalt', 'aod_strataer', 'aod_sulfate',
                   'dust_height', 'ice_height', 'dws',
                   # Sounding
                   'snr_o2a', 'snr_wco2', 'snr_sco2',
                   'glint_angle', 'pol_angle', 'saa', 'vaa',
                   ]
    
    # plot the correlation matrix of the features
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = np.corrcoef(X, rowvar=False)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(X_var_names)))
    ax.set_yticks(np.arange(len(X_var_names)))
    ax.set_xticklabels(X_var_names, rotation=90)
    ax.set_yticklabels(X_var_names)
    fig.colorbar(im, ax=ax, label='Correlation coefficient')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'feature_correlation_matrix_{fp}.png'), dpi=150)
    plt.close(fig)

    y_xco2_bc_anomaly  = np.array(xco2_bc_anomaly)
    y_xco2_raw_anomaly = np.array(xco2_raw_anomaly)

    y_set = [y_xco2_bc_anomaly, y_xco2_raw_anomaly,]
    y_set_description = [
        'xco2_bc_anomaly',
        'xco2_raw_anomaly',
    ]

    R2_scores      = np.zeros(len(y_set))
    slopes         = np.zeros(len(y_set))
    intercepts     = np.zeros(len(y_set))
    model_weights  = np.zeros((len(y_set), len(X_var_names)))
    model_weights[...] = np.nan  # Initialize with NaN to distinguish from valid zero weights
    model_intercepts = np.zeros(len(y_set))

    # MLP (PyTorch) result arrays — parallel to the linear baseline
    R2_scores_mlp    = np.full(len(y_set), np.nan)
    slopes_mlp       = np.full(len(y_set), np.nan)
    intercepts_mlp   = np.full(len(y_set), np.nan)

    X_norm = X / np.nanmax(X, axis=0)
    
    # X_norm[~np.isfinite(X_norm)] = np.nan  # Ensure any non-finite values are NaN for masking
    
    for j, y in enumerate(y_set):
        xymask = np.isfinite(y) & np.all(np.isfinite(X_norm), axis=1)
        print("fp:", fp, "y:", y_set_description[j], "valid samples:", np.sum(xymask))
        print("y:", y[xymask][:5])
        print("X_norm:", X_norm[xymask][:5])

        if np.sum(xymask) < 10:
            print(f"Skipping regression for {y_set_description[j]} using "
                    f"({np.sum(xymask)} valid samples)")
            R2_scores[j]     = np.nan
            slopes[j]        = np.nan
            intercepts[j]    = np.nan
            model_weights[j, :X.shape[1]] = np.nan
            model_intercepts[j] = np.nan
            continue

        X_norm_mask = X_norm[xymask]
        y_mask      = y[xymask]

        # model   = LinearRegression()
        model = Lasso(alpha=0.0001)
        model.fit(X_norm_mask, y_mask)
        y_pred  = model.predict(X_norm_mask)

        R2_scores[j]  = model.score(X_norm_mask, y_mask)
        slope, intercept = np.polyfit(y_mask, y_pred, 1)
        slopes[j]     = slope
        intercepts[j] = intercept
        model_weights[j, :X_norm_mask.shape[1]] = model.coef_
        model_intercepts[j] = model.intercept_

        # ── PyTorch MLP ────────────────────────────────────────────────────
        n_features = X_norm_mask.shape[1]

        class _MLP(nn.Module):
            def __init__(self, n_in):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_in, 64), nn.ReLU(),
                    nn.Linear(64, 32),   nn.ReLU(),
                    nn.Linear(32, 1),
                )
            def forward(self, x):
                return self.net(x).squeeze(-1)

        _X_mean = X_norm_mask.mean(axis=0)
        _X_std  = X_norm_mask.std(axis=0) + 1e-8
        X_t = torch.tensor((X_norm_mask - _X_mean) / _X_std, dtype=torch.float32)
        y_t = torch.tensor(y_mask,                            dtype=torch.float32)

        mlp       = _MLP(n_features)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        def _loss(pred, target, alpha=0.5, eps=0.5):
            mse = torch.mean((pred - target) ** 2)
            rel = torch.mean(((pred - target) / (target.abs() + eps)) ** 2)
            return mse + alpha * rel

        mlp.train()
        for _ in range(1000):
            optimizer.zero_grad()
            _loss(mlp(X_t), y_t).backward()
            optimizer.step()

        mlp.eval()
        with torch.no_grad():
            y_pred_mlp = mlp(X_t).numpy()

        ss_res = ((y_mask - y_pred_mlp) ** 2).sum()
        ss_tot = ((y_mask - y_mask.mean()) ** 2).sum()
        R2_scores_mlp[j]  = 1.0 - ss_res / ss_tot
        _slope, _intercept = np.polyfit(y_mask, y_pred_mlp, 1)
        slopes_mlp[j]      = _slope
        intercepts_mlp[j]  = _intercept
        # ── end MLP ────────────────────────────────────────────────────────

        # Scatter: actual vs predicted, coloured by xco2_bc
        xco2_bc_mask = np.array(xco2_bc)[xymask]
        qf0 = np.array(df['xco2_qf_lt'])[xymask] == 0
        qf1 = np.array(df['xco2_qf_lt'])[xymask] == 1

        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 6))
        if np.isfinite(xco2_bc_near_cld_mean):
            norm = colors.CenteredNorm(vcenter=xco2_bc_near_cld_mean)
            scatter_arg = {'cmap': 'RdBu_r', 'alpha': 0.75, 'norm': norm}
        else:
            vmin, vmax = xco2_bc_mask.min(), xco2_bc_mask.max()
            scatter_arg = {'cmap': 'jet', 'alpha': 0.75, 'vmin': vmin, 'vmax': vmax}
        sc = ax.scatter(y_mask, y_pred, c=xco2_bc_mask, edgecolor=None, s=5, **scatter_arg)
        ax.scatter(y_mask[qf0], y_pred[qf0], c=xco2_bc_mask[qf0],
                    edgecolor='k',    label='QF=0', s=35, **scatter_arg)
        ax.scatter(y_mask[qf1], y_pred[qf1], c=xco2_bc_mask[qf1],
                    edgecolor='grey', label='QF=1', s=35, **scatter_arg)
        ax.set_xlabel(f'Actual {y_set_description[j]}')
        ax.set_ylabel(f'Predicted {y_set_description[j]}')
        ax.set_title(f'Predicting {y_set_description[j]}')
        ax.legend()
        ax.text(0.05, 0.95, f'R²={R2_scores[j]:.3f}',
                transform=ax.transAxes, verticalalignment='top')
        fig.colorbar(sc, ax=ax, label='xco2_bc')
        ax.plot([y_mask.min(), y_mask.max()], [y_mask.min(), y_mask.max()], 'r--')
        fig.savefig(
            os.path.join(output_dir,
                            f'{y_set_description[j]}_X{i+1}_regression_{fp}.png'),
            dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Scatter for MLP
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(y_mask, y_pred_mlp, c=xco2_bc_mask, edgecolor=None, s=5, **scatter_arg)
        ax.scatter(y_mask[qf0], y_pred_mlp[qf0], c=xco2_bc_mask[qf0],
                    edgecolor='k',    label='QF=0', s=35, **scatter_arg)
        ax.scatter(y_mask[qf1], y_pred_mlp[qf1], c=xco2_bc_mask[qf1],
                    edgecolor='grey', label='QF=1', s=35, **scatter_arg)
        ax.set_xlabel(f'Actual {y_set_description[j]}')
        ax.set_ylabel(f'Predicted {y_set_description[j]} (MLP)')
        ax.set_title(f'Predicting {y_set_description[j]} — MLP')
        ax.legend()
        ax.text(0.05, 0.95, f'R²={R2_scores_mlp[j]:.3f}',
                transform=ax.transAxes, verticalalignment='top')
        fig.colorbar(sc, ax=ax, label='xco2_bc')
        ax.plot([y_mask.min(), y_mask.max()], [y_mask.min(), y_mask.max()], 'r--')
        fig.savefig(
            os.path.join(output_dir,
                            f'{y_set_description[j]}_X{i+1}_mlp_{fp}.png'),
            dpi=150, bbox_inches='tight')
        plt.close(fig)


    # Save regression results to CSV
    regression_results = []
    for j in range(len(y_set)):
        row = {
            'y_description':   y_set_description[j],
            # linear baseline
            'R2_score':        R2_scores[j],
            'slope':           slopes[j],
            'intercept':       intercepts[j],
            'model_intercept': model_intercepts[j],
            # MLP
            'R2_score_mlp':    R2_scores_mlp[j],
            'slope_mlp':       slopes_mlp[j],
            'intercept_mlp':   intercepts_mlp[j],
        }
        for k in range(len(X_var_names)):
            row[f'model_coef_{k}'] = model_weights[j, k]
        regression_results.append(row)
    pd.DataFrame(regression_results).to_csv(
        os.path.join(output_dir, f'regression_results_{fp}.csv'), index=False)


# ─── LR mitigation test ────────────────────────────────────────────────────────

def mitigation_test(sat, df, output_dir, reference_csv=None):
    """Train a per-footprint linear regression to predict XCO2 bias from kappas.

    Parameters
    ----------
    sat          : dict from preprocess()
    df           : DataFrame from k1k2_analysis
    output_dir   : str
    reference_csv: str or None
    """
    fp_lon    = df['lon']
    fp_lat    = df['lat']
    fp_number = df['fp']
    


    df = df[df['sfc_type_lt'] == 0]  # Ocean only for now
    
    # onehot encode the fp_number into 8 binary columns (fp_0, fp_1, ..., fp_7)
    fp_onehot = np.zeros((len(df), 8))
    for i in range(8):
        fp_onehot[:, i] = (df['fp'] == i).astype(int)
        df[f'fp_{i}'] = fp_onehot[:, i]
    
    df_orig = copy.deepcopy(df)  

    
    

    # Determine training data and clear-sky reference level
    if reference_csv is None:
        train_df = df  
    else:
        train_df = pd.read_csv(reference_csv)
        train_df = train_df[train_df['sfc_type_lt'] == 0]  # Ocean only for now
        # onehot encode the fp_number in the training DataFrame as well
        for i in range(8):
            train_df[f'fp_{i}'] = (train_df['fp'] == i).astype(int)
            

    train_df = train_df[np.isfinite(train_df['xco2_bc_anomaly']) & np.isfinite(train_df['xco2_raw_anomaly'])]  # Only valid XCO2 values


    # y must be aligned with train_df (same footprint, same rows as X)
    y_bc  = np.array(train_df['xco2_bc_anomaly'])
    y_raw = np.array(train_df['xco2_raw_anomaly'])
    
    X_xco2_bc   = np.array(train_df['xco2_bc'])
    X_xco2_raw  = np.array(train_df['xco2_raw'])

    def _build_feature_matrix(d):
        """Build the feature matrix X from a DataFrame d."""
        return np.vstack([
            d.o2a_k1, d.wco2_k1, d.sco2_k1,
            d.o2a_k2, d.wco2_k2, d.sco2_k2,
            d.delT_lt, d.dp_lt,
            d.psfc_lt,
            d.dp_lt / d.psfc_lt,
            d.co2_grad_del_lt,
            d.alb_o2a_lt,
            d.aod_total_lt,
            d.mu_sza, d.mu_vza,
            d.ws_lt,
            # Preprocessor variables
            d.co2_ratio_bc_lt, d.h2o_ratio_bc_lt,
            d.csnr_o2a_lt, d.csnr_wco2_lt, d.csnr_sco2_lt,
            d.dp_abp_lt,
            d.h_cont_o2a_lt, d.h_cont_wco2_lt, d.h_cont_sco2_lt,
            d.max_declock_o2a_lt, d.max_declock_wco2_lt, d.max_declock_sco2_lt,
            d.xco2_strong_idp_lt, d.xco2_weak_idp_lt,
            # Additional retrieval variables
            d.h2o_scale_lt, d.dpfrac_lt,
            d.aod_bc_lt, d.aod_dust_lt, d.aod_ice_lt, d.aod_water_lt,
            d.aod_oc_lt, d.aod_seasalt_lt, d.aod_strataer_lt, d.aod_sulfate_lt,
            d.dust_height_lt, d.ice_height_lt, d.dws_lt,
            # Additional sounding variables
            d.snr_o2a_lt, d.snr_wco2_lt, d.snr_sco2_lt,
            d.glint_angle_lt, d.pol_angle_lt, d.saa_lt, d.vaa_lt,
            # Footprint one-hot
            d.fp_0, d.fp_1, d.fp_2, d.fp_3,
            d.fp_4, d.fp_5, d.fp_6, d.fp_7,
        ]).T

    feature_names = [
        'o2a_k1', 'wco2_k1', 'sco2_k1',
        'o2a_k2', 'wco2_k2', 'sco2_k2',
        'delT', 'dp', 'psfc', 'dp/psfc', 'co2_grad_del',
        'alb_o2a', 'aod_total', 'mu_sza', 'mu_vza', 'ws',
        # Preprocessor
        'co2_ratio_bc', 'h2o_ratio_bc',
        'csnr_o2a', 'csnr_wco2', 'csnr_sco2',
        'dp_abp',
        'h_cont_o2a', 'h_cont_wco2', 'h_cont_sco2',
        'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2',
        'xco2_strong_idp', 'xco2_weak_idp',
        # Retrieval
        'h2o_scale', 'dpfrac',
        'aod_bc', 'aod_dust', 'aod_ice', 'aod_water',
        'aod_oc', 'aod_seasalt', 'aod_strataer', 'aod_sulfate',
        'dust_height', 'ice_height', 'dws',
        # Sounding
        'snr_o2a', 'snr_wco2', 'snr_sco2',
        'glint_angle', 'pol_angle', 'saa', 'vaa',
        # Footprint one-hot
        'fp_0', 'fp_1', 'fp_2', 'fp_3',
        'fp_4', 'fp_5', 'fp_6', 'fp_7',
    ]

    X = _build_feature_matrix(train_df)

    if reference_csv is None:
        df_orig = train_df
        df_orig_all = df
    else:
        valid_xco2_anomaly_mask = np.isfinite(df['xco2_bc_anomaly']) & np.isfinite(df['xco2_raw_anomaly'])
        df_orig = df[valid_xco2_anomaly_mask]  # Only valid XCO2 values
        df_orig_all = df
    X_pred = _build_feature_matrix(df_orig)
    X_pred_all = _build_feature_matrix(df_orig_all)
    
    # LR correction arrays (including those not in the training set)
    xco2_bc_pred_anomaly = np.full(X_pred.shape[0], np.nan)
    xco2_raw_pred_anomaly = np.full(X_pred.shape[0], np.nan)
    xco2_bc_corrected       = np.full(X_pred.shape[0], np.nan)
    xco2_raw_corrected      = np.full(X_pred.shape[0], np.nan)

    # MLP parallel correction arrays
    xco2_bc_pred_anomaly_mlp  = np.full(X_pred.shape[0], np.nan)
    xco2_raw_pred_anomaly_mlp = np.full(X_pred.shape[0], np.nan)
    xco2_bc_corrected_mlp     = np.full(X_pred.shape[0], np.nan)
    xco2_raw_corrected_mlp    = np.full(X_pred.shape[0], np.nan)
    
    # LR correction arrays for all data (including those without valid XCO2 anomaly for training)
    xco2_bc_predict_all_anomaly = np.full(X_pred_all.shape[0], np.nan)
    xco2_bc_corrected_all      = np.full(X_pred_all.shape[0], np.nan)
    xco2_raw_predict_all_anomaly = np.full(X_pred_all.shape[0], np.nan)
    xco2_raw_corrected_all      = np.full(X_pred_all.shape[0], np.nan)
    
    # MLP correction arrays for all data (including those without valid XCO2 anomaly for training)
    xco2_bc_predict_all_anomaly_mlp = np.full(X_pred_all.shape[0], np.nan)
    xco2_bc_corrected_all_mlp      = np.full(X_pred_all.shape[0], np.nan)
    xco2_raw_predict_all_anomaly_mlp = np.full(X_pred_all.shape[0], np.nan)
    xco2_raw_corrected_all_mlp      = np.full(X_pred_all.shape[0], np.nan)

    y_labels = ['xco2_bc_anomaly', 'xco2_raw_anomaly']

    for y_idx, (y, xco2,
            xco2_all,
            xco2_predict_anomaly,     xco2_corrected,
            xco2_predict_anomaly_mlp, xco2_corrected_mlp,
            xco2_predict_all_anomaly, xco2_corrected_all,
            xco2_predict_all_anomaly_mlp, xco2_corrected_all_mlp,
            ) in enumerate(zip(
            [y_bc,                     y_raw],
            [df_orig['xco2_bc'],     df_orig['xco2_raw']],
            [df_orig_all['xco2_bc'],     df_orig_all['xco2_raw']],
            [xco2_bc_pred_anomaly,        xco2_raw_pred_anomaly],
            [xco2_bc_corrected,           xco2_raw_corrected],
            [xco2_bc_pred_anomaly_mlp,    xco2_raw_pred_anomaly_mlp],
            [xco2_bc_corrected_mlp,       xco2_raw_corrected_mlp],
            [xco2_bc_predict_all_anomaly, xco2_raw_predict_all_anomaly],
            [xco2_bc_corrected_all,   xco2_raw_corrected_all],
            [xco2_bc_predict_all_anomaly_mlp, xco2_raw_predict_all_anomaly_mlp],
            [xco2_bc_corrected_all_mlp, xco2_raw_corrected_all_mlp],
            )):

        print("y shape:", y.shape, "X shape:", X.shape)
        xymask    = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X_mask         = X[xymask]
        y_mask         = y[xymask]
        norm_factor = np.nanmax(X_mask, axis=0)
        nonzero_mask = norm_factor > 0
        norm_factor[~nonzero_mask] = 1.0  # Avoid division by zero; these features will be all-zero after normalization
        X_mask         = X_mask / norm_factor
        X_pred    = X_pred / norm_factor
        X_pred_all = X_pred_all / norm_factor

        # ── Train/test split: equal-count per y-bin (Option 4) ────────────
        # Ensures rare large anomalies appear in both train and test sets.
        rng      = np.random.default_rng(17)
        n_bins   = 10
        bin_edges = np.percentile(y_mask, np.linspace(0, 100, n_bins + 1))
        bin_ids   = np.digitize(y_mask, bin_edges[1:-1])  # 0-indexed bins

        MIN_BIN_TEST = 5   # bins with fewer points than this send all to train
        train_idx, test_idx = [], []
        for b in np.unique(bin_ids):
            idx_b = np.where(bin_ids == b)[0]
            idx_b = rng.permutation(idx_b)
            if len(idx_b) < MIN_BIN_TEST:
                # too sparse (typical for tail bins) — keep all in train so
                # the model still sees these extreme y values
                train_idx.extend(idx_b)
            else:
                split = max(1, int(len(idx_b) * 0.2))
                test_idx.extend(idx_b[:split])
                train_idx.extend(idx_b[split:])

        train_idx = np.array(train_idx)
        test_idx  = np.array(test_idx)
        X_train, y_train = X_mask[train_idx], y_mask[train_idx]
        X_test,  y_test  = X_mask[test_idx],  y_mask[test_idx]
        
        print(f"Train/test size: {len(train_idx)}/{len(test_idx)}")

        # ── Backup: StratifiedShuffleSplit with quantile bins (Option 3) ──
        # from sklearn.model_selection import StratifiedShuffleSplit
        # y_bins = pd.qcut(y_mask, q=10, labels=False, duplicates='drop')
        # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=17)
        # train_idx, test_idx = next(sss.split(X_mask, y_bins))
        # X_train, y_train = X_mask[train_idx], y_mask[train_idx]
        # X_test,  y_test  = X_mask[test_idx],  y_mask[test_idx]

        # ── Linear baseline ────────────────────────────────────────────
        model = LinearRegression()
        model.fit(X_train, y_train)
        print(f"All FP R² (linear) for predicting xco2_bc anomaly: {model.score(X_mask, y_mask):.3f}")

        X_pred_mask     = np.all(np.isfinite(X_pred), axis=1)
        y_pred_df   = model.predict(X_pred[X_pred_mask])
        xco2_predict_anomaly[X_pred_mask] = y_pred_df
        xco2_corrected[X_pred_mask] = xco2[X_pred_mask] - y_pred_df
        
        X_pred_all_mask = np.all(np.isfinite(X_pred_all), axis=1)
        y_pred_all_df = model.predict(X_pred_all[X_pred_all_mask])
        xco2_predict_all_anomaly[X_pred_all_mask] = y_pred_all_df
        xco2_corrected_all[X_pred_all_mask]       = xco2_all[X_pred_all_mask] - y_pred_all_df
        
        # X_pred_all_mask = np.all(np.isfinite(X_pred_all), axis=1)
        # y_pred_all_df = model.predict(X_pred_all[X_pred_all_mask])
        # idx_all = np.where(sub_all_pred_fp_mask)[0][X_pred_all_mask]
        # xco2_predict_anomaly[idx_all] = y_pred_all_df
        # xco2_corrected[idx_all]       = xco2_bc[idx_all] - y_pred_all_df

        # ── PyTorch MLP ────────────────────────────────────────────────
        n_in = X_train.shape[1]

        class _MLP(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 32), nn.ReLU(),
                    nn.Linear(32, 16), nn.ReLU(),
                    nn.Linear(16, 1),
                )
            def forward(self, x):
                return self.net(x).squeeze(-1)

        _X_mean = X_train.mean(axis=0)
        _X_std  = X_train.std(axis=0) + 1e-8

        Xt_train = torch.tensor((X_train - _X_mean) / _X_std, dtype=torch.float32)
        yt_train = torch.tensor(y_train,                       dtype=torch.float32)

        mlp       = _MLP(n_in)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        def _loss(pred, target, alpha=0.5, eps=0.5):
            mse = torch.mean((pred - target) ** 2)
            rel = torch.mean(((pred - target) / (target.abs() + eps)) ** 2)
            return mse + alpha * rel

        mlp.train()
        for _ in range(1000):
            optimizer.zero_grad()
            _loss(mlp(Xt_train), yt_train).backward()
            optimizer.step()

        mlp.eval()
        # Must use X_mask (masked + norm_factor-scaled), not raw X —
        # _X_mean/_X_std were computed from X_train which is a slice of X_mask.
        Xt_all = torch.tensor((X_mask - _X_mean) / _X_std, dtype=torch.float32)
        with torch.no_grad():
            y_all_mlp = mlp(Xt_all).numpy()
        ss_res = ((y_mask - y_all_mlp) ** 2).sum()
        ss_tot = ((y_mask - y_mask.mean()) ** 2).sum()
        print(f"All FP R² (MLP)    for predicting xco2_bc anomaly: {1 - ss_res/ss_tot:.3f}")

        X_pred_valid = X_pred[X_pred_mask]
        Xt_pred = torch.tensor((X_pred_valid - _X_mean) / _X_std, dtype=torch.float32)
        with torch.no_grad():
            y_pred_mlp = mlp(Xt_pred).numpy()

        xco2_predict_anomaly_mlp[X_pred_mask] = y_pred_mlp
        xco2_corrected_mlp[X_pred_mask]       = xco2[X_pred_mask] - y_pred_mlp
        
        X_pred_all_valid = X_pred_all[X_pred_all_mask]
        Xt_all_pred = torch.tensor((X_pred_all_valid - _X_mean) / _X_std, dtype=torch.float32)
        with torch.no_grad():
            y_all_pred_mlp = mlp(Xt_all_pred).numpy()
            
        xco2_predict_all_anomaly_mlp[X_pred_all_mask] = y_all_pred_mlp
        
        xco2_corrected_all_mlp[X_pred_all_mask]       = xco2_all[X_pred_all_mask] - y_all_pred_mlp
        # ── end MLP ────────────────────────────────────────────────────

        # ── Feature importance (LR + MLP permutation) ─────────────────
        y_label = y_labels[y_idx]

        # LR: standardised absolute coefficients  |coef_i * std(X_train_i)|
        lr_std_importance = np.abs(model.coef_) * X_train.std(axis=0)

        # Permutation importance on test set (shared helper for LR & MLP)
        def _permutation_importance(predict_fn, X_eval, y_eval, n_repeats=10):
            """Return mean R² drop per feature when that feature is shuffled."""
            ss_tot = ((y_eval - y_eval.mean()) ** 2).sum()
            baseline_r2 = 1.0 - ((y_eval - predict_fn(X_eval)) ** 2).sum() / ss_tot
            importances = np.zeros(X_eval.shape[1])
            rng_pi = np.random.default_rng(42)
            for col in range(X_eval.shape[1]):
                drops = np.zeros(n_repeats)
                for r in range(n_repeats):
                    X_shuf = X_eval.copy()
                    X_shuf[:, col] = rng_pi.permutation(X_shuf[:, col])
                    r2_shuf = 1.0 - ((y_eval - predict_fn(X_shuf)) ** 2).sum() / ss_tot
                    drops[r] = baseline_r2 - r2_shuf
                importances[col] = drops.mean()
            return importances

        perm_imp_lr = _permutation_importance(model.predict, X_test, y_test)

        def _mlp_predict_np(X_np):
            Xt = torch.tensor((X_np - _X_mean) / _X_std, dtype=torch.float32)
            with torch.no_grad():
                return mlp(Xt).numpy()
        perm_imp_mlp = _permutation_importance(_mlp_predict_np, X_test, y_test)

        # Save importance to CSV
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'lr_std_coef': lr_std_importance,
            'lr_perm_importance': perm_imp_lr,
            'mlp_perm_importance': perm_imp_mlp,
        })
        imp_df = imp_df.sort_values('mlp_perm_importance', ascending=False)
        imp_df.to_csv(os.path.join(output_dir, f'feature_importance_{y_label}.csv'), index=False)

        # Bar plot: top 25 features by permutation importance
        top_n = min(25, len(feature_names))
        top_df = imp_df.head(top_n).iloc[::-1]  # reverse for horizontal bar (top at top)

        plt.close('all')
        fig, (ax_lr, ax_mlp) = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.32)))

        ax_lr.barh(top_df['feature'], top_df['lr_perm_importance'], color='steelblue', label='LR perm.')
        ax_lr.set_xlabel('Permutation importance (R² drop)')
        ax_lr.set_title(f'LR — {y_label}')

        ax_mlp.barh(top_df['feature'], top_df['mlp_perm_importance'], color='forestgreen', label='MLP perm.')
        ax_mlp.set_xlabel('Permutation importance (R² drop)')
        ax_mlp.set_title(f'MLP — {y_label}')

        fig.suptitle(f'Feature importance (top {top_n})', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'feature_importance_{y_label}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        # ── end feature importance ────────────────────────────────────

        del y  # Free memory before plotting
        
    del X  # Free memory before next footprint
    gc.collect()

    # ── Recompute XCO2 anomaly on corrected fields ─────────────────────────
    # Lazy import avoids circular import (oco_fp_spec_anal ↔ result_ana)
    from oco_fp_spec_anal import compute_xco2_anomaly
    # Same parameters as used in oco_fp_spec_anal.py
    _anomaly_args = {'lat_thres': 0.25, 'std_thres': 1.0, 'min_cld_dist': 10.0}
    _lat      = np.array(df_orig['lat'])
    _cld_dist = np.array(df_orig['cld_dist_km'])

    anomaly_orig    = np.array(df_orig['xco2_bc_anomaly'])
    anomaly_lr      = compute_xco2_anomaly(_lat, _cld_dist, xco2_bc_corrected,     **_anomaly_args)
    anomaly_mlp     = compute_xco2_anomaly(_lat, _cld_dist, xco2_bc_corrected_mlp, **_anomaly_args)

    # Scatter + histogram comparison of anomalies
    plt.close('all')
    fig, (ax_sc1, ax_sc2, ax_hist) = plt.subplots(1, 3, figsize=(18, 6))

    valid = np.isfinite(anomaly_orig) & np.isfinite(anomaly_lr)
    ax_sc1.scatter(anomaly_orig[valid], anomaly_lr[valid],
                   c='orange', edgecolor=None, s=5, alpha=0.6)
    _lim = np.nanpercentile(np.abs(anomaly_orig[valid]), 99)
    ax_sc1.set_xlim(-_lim, _lim); ax_sc1.set_ylim(-_lim, _lim)
    ax_sc1.set_aspect('equal', adjustable='box')
    ax_sc1.axline((0, 0), slope=1, color='r', linestyle='--')
    ax_sc1.set_xlabel('Original XCO2_bc anomaly (ppm)')
    ax_sc1.set_ylabel('LR-corrected XCO2_bc anomaly (ppm)')
    ax_sc1.set_title('Original vs LR-corrected anomaly')
    r2_lr = 1 - np.nansum((anomaly_orig[valid] - anomaly_lr[valid])**2) / \
                np.nansum((anomaly_orig[valid] - np.nanmean(anomaly_orig[valid]))**2)
    ax_sc1.text(0.05, 0.95, f'R²={r2_lr:.3f}', transform=ax_sc1.transAxes, va='top')

    valid2 = np.isfinite(anomaly_orig) & np.isfinite(anomaly_mlp)
    ax_sc2.scatter(anomaly_orig[valid2], anomaly_mlp[valid2],
                   c='green', edgecolor=None, s=5, alpha=0.6)
    ax_sc2.set_xlim(-_lim, _lim); ax_sc2.set_ylim(-_lim, _lim)
    ax_sc2.set_aspect('equal', adjustable='box')
    ax_sc2.axline((0, 0), slope=1, color='r', linestyle='--')
    ax_sc2.set_xlabel('Original XCO2_bc anomaly (ppm)')
    ax_sc2.set_ylabel('MLP-corrected XCO2_bc anomaly (ppm)')
    ax_sc2.set_title('Original vs MLP-corrected anomaly')
    r2_mlp = 1 - np.nansum((anomaly_orig[valid2] - anomaly_mlp[valid2])**2) / \
                 np.nansum((anomaly_orig[valid2] - np.nanmean(anomaly_orig[valid2]))**2)
    ax_sc2.text(0.05, 0.95, f'R²={r2_mlp:.3f}', transform=ax_sc2.transAxes, va='top')

    _bins = np.linspace(-3, 3, 211)
    for _anom, _color, _label in [
            (anomaly_orig, 'blue',   'Original'),
            (anomaly_lr,   'orange', 'LR-corrected'),
            (anomaly_mlp,  'green',  'MLP-corrected'),
    ]:
        _v = _anom[np.isfinite(_anom)]
        _mu, _sigma = np.nanmean(_v), np.nanstd(_v)
        ax_hist.hist(_v, bins=_bins, color=_color, alpha=0.6, density=True,
                     label=f'{_label}\nμ={_mu:.3f}, σ={_sigma:.3f}')
        ax_hist.axvline(_mu, color=_color, linestyle='-',  linewidth=1.2)
        ax_hist.axvline(_mu - _sigma, color=_color, linestyle=':', linewidth=0.9)
        ax_hist.axvline(_mu + _sigma, color=_color, linestyle=':', linewidth=0.9)

    ax_hist.set_xlabel('XCO2_bc anomaly (ppm)')
    ax_hist.set_title('Anomaly distribution comparison')
    ax_hist.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax_hist.legend(fontsize=10)

    fig.tight_layout()
    fname = (f"anomaly_comparison_reference_{os.path.basename(reference_csv).split('.')[0]}.png"
             if reference_csv else "anomaly_comparison.png")
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 3×3 XCO2_bc comparison by cloud-distance regime ──────────────────
    # Row masks
    _cd_orig       = np.array(df_orig['cld_dist_km'])
    mask_r1        = _cd_orig > 10
    mask_r2        = _cd_orig < 10
    _cd_all_arr    = np.array(df_orig_all['cld_dist_km'])
    _anom_all_orig = np.array(df_orig_all['xco2_bc_anomaly'])
    mask_r3        = (_cd_all_arr < 10) & np.isnan(_anom_all_orig)

    # Data sources per row: (xco2_orig, xco2_lr, xco2_mlp, mask)
    row_configs = [
        (np.array(df_orig['xco2_bc']),     xco2_bc_corrected,     xco2_bc_corrected_mlp,
         mask_r1, 'Clear-sky FPs (cld_dist > 10 km)'),
        (np.array(df_orig['xco2_bc']),     xco2_bc_corrected,     xco2_bc_corrected_mlp,
         mask_r2, 'Cloud-affected FPs from df_orig (cld_dist < 10 km)'),
        (np.array(df_orig_all['xco2_bc']), xco2_bc_corrected_all, xco2_bc_corrected_all_mlp,
         mask_r3, 'Cloud FPs with NaN anomaly from df_orig_all (cld_dist < 10 km)'),
    ]

    # Shared histogram bins across all rows
    _all_xco2 = np.concatenate([
        np.array(df_orig['xco2_bc']),
        np.array(df_orig_all['xco2_bc']),
        xco2_bc_corrected, xco2_bc_corrected_mlp,
        xco2_bc_corrected_all, xco2_bc_corrected_all_mlp,
    ])
    _xco2_lo = np.nanpercentile(_all_xco2, 1)
    _xco2_hi = np.nanpercentile(_all_xco2, 99)
    _bins_3x3 = np.linspace(_xco2_lo, _xco2_hi, 100)

    plt.close('all')
    fig, axes = plt.subplots(3, 3, figsize=(18, 17))

    for row_i, (xco2_orig, xco2_lr, xco2_mlp, mask, row_label) in enumerate(row_configs):
        ax_sc1, ax_sc2, ax_h = axes[row_i]

        x_orig = xco2_orig[mask]
        x_lr   = xco2_lr[mask]
        x_mlp  = xco2_mlp[mask]

        v_lr  = np.isfinite(x_orig) & np.isfinite(x_lr)
        v_mlp = np.isfinite(x_orig) & np.isfinite(x_mlp)

        _lo = np.nanpercentile(x_orig[np.isfinite(x_orig)], 1)  if np.isfinite(x_orig).any() else _xco2_lo
        _hi = np.nanpercentile(x_orig[np.isfinite(x_orig)], 99) if np.isfinite(x_orig).any() else _xco2_hi

        for ax, x_corr, _color, method in [
                (ax_sc1, x_lr,  'orange', 'LR'),
                (ax_sc2, x_mlp, 'green',  'MLP'),
        ]:
            v = np.isfinite(x_orig) & np.isfinite(x_corr)
            ax.scatter(x_orig[v], x_corr[v], c=_color, edgecolor=None, s=5, alpha=0.6)
            ax.set_xlim(_lo, _hi); ax.set_ylim(_lo, _hi)
            ax.set_aspect('equal', adjustable='box')
            ax.axline((_lo, _lo), slope=1, color='r', linestyle='--')
            ax.set_xlabel('Original XCO2_bc (ppm)')
            ax.set_ylabel(f'{method}-corrected XCO2_bc (ppm)')
            ax.set_title(f'{row_label}\n[{method} scatter]')
            if v.sum() > 1:
                r2 = 1 - np.nansum((x_orig[v] - x_corr[v])**2) / \
                         np.nansum((x_orig[v] - np.nanmean(x_orig[v]))**2)
                ax.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax.transAxes, va='top')

        for _xco2, _color, _label in [
                (x_orig, 'blue',   'Original'),
                (x_lr,   'orange', 'LR-corrected'),
                (x_mlp,  'green',  'MLP-corrected'),
        ]:
            _v = _xco2[np.isfinite(_xco2)]
            if len(_v) == 0:
                continue
            _mu, _sigma = _v.mean(), _v.std()
            ax_h.hist(_v, bins=_bins_3x3, color=_color, alpha=0.6, density=True,
                      label=f'{_label}\nμ={_mu:.3f}, σ={_sigma:.3f}')
            ax_h.axvline(_mu,          color=_color, linestyle='-',  linewidth=1.2)
            ax_h.axvline(_mu - _sigma, color=_color, linestyle=':',  linewidth=0.9)
            ax_h.axvline(_mu + _sigma, color=_color, linestyle=':',  linewidth=0.9)

        ax_h.set_title(f'{row_label}\n[Distribution]')
        ax_h.set_xlabel('XCO2_bc (ppm)')
        ax_h.legend(fontsize=10)

    fig.suptitle('XCO2_bc by cloud-distance regime', fontsize=13, y=1.01)
    fig.tight_layout()
    fname = (f"xco2bc_comparison_3x3_reference_{os.path.basename(reference_csv).split('.')[0]}.png"
             if reference_csv else "xco2bc_comparison_3x3.png")
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Scatter: original vs corrected (linear baseline + MLP) ───────────
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax, ax_mlp, ax2 = axes

    _vmin = min(np.nanmin(df_orig['xco2_bc']), np.nanmin(xco2_bc_corrected), np.nanmin(xco2_bc_corrected_mlp))
    _vmax = max(np.nanmax(df_orig['xco2_bc']), np.nanmax(xco2_bc_corrected), np.nanmax(xco2_bc_corrected_mlp))

    ax.scatter(df_orig['xco2_bc'], xco2_bc_corrected, c='blue', edgecolor=None, s=5, alpha=0.7)
    ax.set_xlabel('Original XCO2 (ppm)')
    ax.set_ylabel('LR Corrected XCO2 (ppm)')
    ax.set_title('LR Correction of OCO-2 L2 XCO2')
    ax.plot([_vmin, _vmax], [_vmin, _vmax], 'r--')

    ax_mlp.scatter(df_orig['xco2_bc'], xco2_bc_corrected_mlp, c='green', edgecolor=None, s=5, alpha=0.7)
    ax_mlp.set_xlabel('Original XCO2 (ppm)')
    ax_mlp.set_ylabel('MLP Corrected XCO2 (ppm)')
    ax_mlp.set_title('MLP Correction of OCO-2 L2 XCO2')
    ax_mlp.plot([_vmin, _vmax], [_vmin, _vmax], 'r--')

    bins = np.arange(390, 420.1, 0.5)
    ax2.hist(df_orig['xco2_bc'],   bins=bins, color='blue',   alpha=0.6, density=True, label='Original Lite')
    ax2.hist(xco2_bc_corrected,       bins=bins, color='orange', alpha=0.6, density=True, label='LR Corrected')
    ax2.hist(xco2_bc_corrected_mlp,   bins=bins, color='green',  alpha=0.6, density=True, label='MLP Corrected')
    ax2.set_xlabel('XCO2 (ppm)')
    ax2.set_title('Distribution: Original / LR / MLP Corrected XCO2')
    ax2.legend()

    fname = (f"LR_MLP_correction_lt_xco2_scatter_reference_{os.path.basename(reference_csv).split('.')[0]}.png"
             if reference_csv else "LR_MLP_bc_correction_lt_xco2_scatter.png")
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Map: spatial distribution of k1, XCO2, and LR correction ─────────
    xco2_min = min(np.nanmin(df_orig_all['xco2_bc']), np.nanmin(df_orig_all['xco2_raw']))
    xco2_max = max(np.nanmax(df_orig_all['xco2_bc']), np.nanmax(df_orig_all['xco2_raw']))

    plt.close('all')
    fig, ((ax1, ax2, ax6, ax7), (ax4, ax3, ax5, ax8)) = plt.subplots(2, 4, figsize=(20, 10))
    
    plot_lon = np.array(df_orig_all.lon)
    if plot_lon.min() < -90 and plot_lon.max() > 90:
        # Assume longitude is in [0, 360] and convert to [-180, 180] for better map visualization
        plot_lon[plot_lon < 0] += 360
    
    sc1 = ax1.scatter(plot_lon, df_orig_all.lat, c=df_orig_all.o2a_k1, cmap='jet', s=20, alpha=0.7)
    ax1.set_title('Retrieved O2A k1');  fig.colorbar(sc1, ax=ax1, label='k1')

    sc2 = ax2.scatter(plot_lon, df_orig_all.lat, c=df_orig_all.o2a_k2, cmap='jet', s=20, alpha=0.7)
    ax2.set_title('Retrieved O2A k2');  fig.colorbar(sc2, ax=ax2, label='k2')

    sc3 = ax3.scatter(plot_lon, df_orig_all.lat, c=df_orig_all['xco2_bc'], cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max)
    ax3.set_title('OCO-2 L2 XCO2 (bias-corrected)');  fig.colorbar(sc3, ax=ax3, label='XCO2 (ppm)')

    sc4 = ax4.scatter(plot_lon, df_orig_all.lat, c=df_orig_all['xco2_raw'], cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max)
    ax4.set_title('OCO-2 L2 XCO2 raw');  fig.colorbar(sc4, ax=ax4, label='XCO2 raw (ppm)')

    sc5 = ax5.scatter(plot_lon, df_orig_all.lat, c=xco2_bc_corrected_all, cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max)
    ax5.set_title('LR Corrected XCO2');  fig.colorbar(sc5, ax=ax5, label='LR corrected XCO2 (ppm)')

    sc6 = ax6.scatter(plot_lon, df_orig_all.lat, c=xco2_bc_predict_all_anomaly, cmap='jet', s=20, alpha=0.7)
    ax6.set_title('LR predicted XCO2 anomaly (ppm)')
    fig.colorbar(sc6, ax=ax6, label='LR predicted XCO2 anomaly (ppm)')

    sc7 = ax7.scatter(plot_lon, df_orig_all.lat, c=xco2_bc_predict_all_anomaly_mlp, cmap='jet', s=20, alpha=0.7)
    ax7.set_title('MLP predicted XCO2 anomaly (ppm)')
    fig.colorbar(sc7, ax=ax7, label='MLP predicted XCO2 anomaly (ppm)')

    sc8 = ax8.scatter(plot_lon, df_orig_all.lat, c=xco2_bc_corrected_all_mlp, cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max)
    ax8.set_title('MLP Corrected XCO2');  fig.colorbar(sc8, ax=ax8, label='MLP corrected XCO2 (ppm)')

    fig.suptitle(f"Cloud distance threshold for XCO2 mean: 10 km")
    fname = (f"LR_MLP_correction_lt_xco2_map_reference_{os.path.basename(reference_csv).split('.')[0]}.png"
             if reference_csv else "LR_MLP_correction_lt_xco2_map_all.png")
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)


    

if __name__ == "__main__":
    main()

import h5py
import sys
import os
import platform
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import bisect as bs
from abs_util.fp_abs_coeff import oco_fp_abs_all_bands
from netCDF4 import Dataset


def oco_fp_atm_abs(sat=None, o2mix=0.20935, output='fp_tau_{}.h5', 
                   oco_files_dict=None,
                    oco_nc_file=None,
                   overwrite=False):
    """

    """
    
    abs_skip = os.path.isfile(output.format('all')) and not overwrite
    
    
    # --------- Constants ------------
    Rd = 287.052874
    EPSILON = 0.622
    kb = 1.380649e-23
    g = 9.81
    AU_M = 149597870700.0
    # ---------------------------------
    if sat == None:
        sys.exit("[Error] sat information must be provided!")
    elif sat != None:
        print("sat0: ", sat)
        # Get reanalysis from met and CO2 prior sounding data
        with h5py.File(oco_files_dict['oco_l1b'], 'r') as oco_l1b:
            lon_l1b = oco_l1b['SoundingGeometry/sounding_longitude'][...]
            lat_l1b = oco_l1b['SoundingGeometry/sounding_latitude'][...]
            print("lat_l1b shape before logic: ", lat_l1b.shape)
            logic = np.isfinite(lon_l1b) & np.isfinite(lat_l1b)
            snd_id_l1b = oco_l1b['SoundingGeometry/sounding_id'][...][logic]
            lat_l1b_select = lat_l1b[logic]
            print("lat_l1b_select shape after logic: ", lat_l1b_select.shape)
            # Solar Doppler Stretch (Fraunhofer vs Telluric)
            # Includes Earth rotation
            v_solar = oco_l1b['SoundingGeometry/sounding_solar_relative_velocity'][...][logic]
            
            # Absolute Wavelength Shift (Spacecraft vs Ground)
            v_inst = oco_l1b['SoundingGeometry/sounding_relative_velocity'][...][logic]
            
            # Earth-Sun distance (converted to AU for 1/r^2 scaling)
            dist_m = oco_l1b['SoundingGeometry/sounding_solar_distance'][...][logic]
            dist_au = dist_m / AU_M
            
            # Wavelength calibration (Dispersion)
            wvl_coef = oco_l1b['InstrumentHeader/dispersion_coef_samp'][...]
            del_lambda_all   = oco_l1b['InstrumentHeader/ils_delta_lambda'][...]
            rel_lresponse_all = oco_l1b['InstrumentHeader/ils_relative_response'][...]

        with h5py.File(oco_files_dict['oco_met'], 'r') as oco_met:
            lon_oco_met = oco_met['SoundingGeometry/sounding_longitude'][...]
            lat_oco_met = oco_met['SoundingGeometry/sounding_latitude'][...]
            print("lat_oco_met shape before logic: ", lat_oco_met.shape)
            logic = np.isfinite(lon_oco_met) & np.isfinite(lat_oco_met)
            hprf_l = oco_met['Meteorology/height_profile_met'][...][logic][:, ::-1]
            qprf_l = oco_met['Meteorology/specific_humidity_profile_met'][...][logic][:, ::-1]      # specific humidity mid grid
            sfc_p = oco_met['Meteorology/surface_pressure_met'][...][logic]
            tprf_l = oco_met['Meteorology/temperature_profile_met'][...][logic][:, ::-1]          # temperature mid grid in K
            pprf_l = oco_met['Meteorology/vector_pressure_levels_met'][...][logic][:, ::-1]      # pressure mid grid in Pa
            o3mrprf_l = oco_met['Meteorology/ozone_profile_met'][...][logic][:, ::-1] # kg kg-1
            uprf_l = oco_met['Meteorology/wind_u_profile_met'][...][logic][:, ::-1]
            vprf_l = oco_met['Meteorology/wind_v_profile_met'][...][logic][:, ::-1]
            sfc_gph = oco_met['Meteorology/gph_met'][...][logic]
            print("lat_oco_met shape after logic: ", lat_oco_met.shape)


        with h5py.File(oco_files_dict['oco_co2prior'], 'r') as oco_co2_aprior:
            lon_oco_co2p = oco_co2_aprior['SoundingGeometry/sounding_longitude'][...]
            lat_oco_co2p = oco_co2_aprior['SoundingGeometry/sounding_latitude'][...]
            logic = np.isfinite(lon_oco_co2p) & np.isfinite(lat_oco_co2p)
            co2_prf_l = oco_co2_aprior['CO2Prior/co2_prior_profile_cpr'][...][logic][:, ::-1] # reorder as from surface to top
            sounding_id = oco_co2_aprior['SoundingGeometry/sounding_id'][...][logic]
            sza_id = oco_co2_aprior['SoundingGeometry/sounding_solar_zenith'][...][logic]
            vza_id = oco_co2_aprior['SoundingGeometry/sounding_zenith'][...][logic]
            fp_number_list = np.arange(8).repeat(oco_co2_aprior['SoundingGeometry/sounding_longitude'][...].shape[0]).reshape(8, -1).T # 8 footprints for each sounding
            fp_number   = fp_number_list[logic]
            
        with Dataset(oco_nc_file, "r") as oco_lt:
            oco_lt_id = oco_lt.variables["sounding_id"][:]

        # convert invalid value -999999 to NaN
        for var in [hprf_l, qprf_l, sfc_p, tprf_l, pprf_l, sfc_gph, co2_prf_l]:
            var[var==-999999] = np.nan

        # Ap, Bp from http://wiki.seas.harvard.edu/geos-chem/index.php/GEOS-Chem_vertical_grids#Hybrid_grid
        Ap =  np.array([0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01, 1.961311e+01, 2.609201e+01,
                        3.257081e+01, 3.898201e+01, 4.533901e+01, 5.169611e+01, 5.805321e+01, 6.436264e+01,
                        7.062198e+01, 7.883422e+01, 8.909992e+01, 9.936521e+01, 1.091817e+02, 1.189586e+02,
                        1.286959e+02, 1.429100e+02, 1.562600e+02, 1.696090e+02, 1.816190e+02, 1.930970e+02,
                        2.032590e+02, 2.121500e+02, 2.187760e+02, 2.238980e+02, 2.243630e+02, 2.168650e+02,
                        2.011920e+02, 1.769300e+02, 1.503930e+02, 1.278370e+02, 1.086630e+02, 9.236572e+01,
                        7.851231e+01, 6.660341e+01, 5.638791e+01, 4.764391e+01, 4.017541e+01, 3.381001e+01,
                        2.836781e+01, 2.373041e+01, 1.979160e+01, 1.645710e+01, 1.364340e+01, 1.127690e+01,
                        9.292942e+00, 7.619842e+00, 6.216801e+00, 5.046801e+00, 4.076571e+00, 3.276431e+00,
                        2.620211e+00, 2.084970e+00, 1.650790e+00, 1.300510e+00, 1.019440e+00, 7.951341e-01,
                        6.167791e-01, 4.758061e-01, 3.650411e-01, 2.785261e-01, 2.113490e-01, 1.594950e-01,
                        1.197030e-01, 8.934502e-02, 6.600001e-02, 4.758501e-02, 3.270000e-02, 2.000000e-02,
                        1.000000e-02])

        Bp =  np.array([1.000000e+00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
                        8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
                        7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
                        5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
                        2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
                        6.960025e-03, 8.175413e-09, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00])

        P_edge = Ap * 100 + Bp * sfc_p[:, np.newaxis]
        dP = P_edge[:, 1:] - P_edge[:, :-1]
        log_P_ratio = np.log(P_edge[:, :-1] / P_edge[:, 1:])
        P_mid = (P_edge[:, 1:] + P_edge[:, :-1])/2

        r = qprf_l/(1-qprf_l)# mass mixing ratio
        eprf_l = pprf_l*r/(EPSILON+r)
        #Tv = tprf_l/(1-eprf_l/pprf_l*(1-EPSILON))
        Tv = tprf_l/(1-(r/(r+EPSILON))*(1-EPSILON))
        dz_hydrostatic = (Rd*Tv)/g*log_P_ratio      # in meter

        h_edge = np.empty((sfc_p.shape[0], 73))
        h_edge[:, 0] = sfc_gph
        h_edge[:, 1:] = np.cumsum(dz_hydrostatic[:, :], axis=1) + np.repeat(sfc_gph.reshape(len(sfc_gph), 1), repeats=72, axis=1)

        air_layer = pprf_l/(kb*tprf_l)/1e6  # air number density in molec/cm3
        dry_air_layer = (pprf_l-eprf_l)/(kb*tprf_l)/1e6  # air number density in molec/cm3
        o2_layer = dry_air_layer*o2mix          # O2 number density in molec/cm3
        h2o_layer = eprf_l/(kb*tprf_l)/1e6  # H2O number density in molec/cm3
        co2_layer = dry_air_layer*co2_prf_l     # CO2 number density in molec/cm3
        air_ml = 28.0134*(1-o2mix) + 31.999*o2mix
        o3_layer = dry_air_layer*air_ml*o3mrprf_l/47.9982     # O3 number density in molec/cm3
        h2o_vmr = h2o_layer/dry_air_layer       # H2O volume mixing ratio


        P_edge /= 100 # convert to hPa
        P_mid /= 100 # convert to hPa
        pprf_l /= 100 # convert to hPa
        
        dz_hydrostatic /= 1e3 # convert to km
        h_edge /= 1e3 # convert to km

        z_last_layer_ind = -1

        sfc_p_mask = np.isfinite(sfc_p)
        

        # oco2_LtCO2_170718_B10206Ar_200730084737s.nc4
        date = oco_nc_file.split("_")[-3]
        doy = pd.to_datetime('20'+date, format="%Y%m%d").dayofyear
        
        
        
        print("lat_l1b_select size: ", lat_l1b_select.shape)
        print("sfc_p_mask size: ", sfc_p_mask.shape)
        if lat_l1b_select.shape[0] != sfc_p_mask.shape[0]:
            print("Mismatch in number of soundings between L1B and selection mask!")
            print("Checking L1B TG file...")

            if 'oco_l1b_tg' in oco_files_dict.keys():
                with h5py.File(oco_files_dict['oco_l1b_tg'], 'r') as oco_l1b_tg:
                    lon_l1b_tg = oco_l1b_tg['SoundingGeometry/sounding_longitude'][...]
                    lat_l1b_tg = oco_l1b_tg['SoundingGeometry/sounding_latitude'][...]
                    tg_logic = np.isfinite(lon_l1b_tg) & np.isfinite(lat_l1b_tg)
                    lat_l1b_tg_select = lat_l1b_tg[tg_logic]
                    snd_id_l1b_tg = oco_l1b_tg['SoundingGeometry/sounding_id'][...][tg_logic]
                    v_solar_tg = oco_l1b_tg['SoundingGeometry/sounding_solar_relative_velocity'][...][tg_logic]
                    v_inst_tg = oco_l1b_tg['SoundingGeometry/sounding_relative_velocity'][...][tg_logic]
                    dist_au_tg = oco_l1b_tg['SoundingGeometry/sounding_solar_distance'][...][tg_logic] / AU_M
                    
                    print("lat_l1b_tg shape: ", lat_l1b_tg.shape)
                    print("lat_l1b_tg flat + lat_l1b_select flat length sum: ", lat_l1b_tg.flatten().shape[0] + lat_l1b_select.flatten().shape[0])
            else:
                print("[Error] No L1B TG file provided for debugging!")
                
            
            sys.exit(1)
            
        # mask fp_id not in oco_l1b_snd_id
        # snd_id_mask = np.isin(sounding_id, oco_l1b_snd_id)
        # do not process sounding not oco_lt_id
        oco_lt_id_mask = np.isin(sounding_id, oco_lt_id)
        
        print("sounding_id: ", sounding_id)
        print("oco_lt_id: ", oco_lt_id)
        
        
        
        print("np.sum(sfc_p_mask): ", np.sum(sfc_p_mask))
        print("np.sum(oco_lt_id_mask): ", np.sum(oco_lt_id_mask))
        
        id_select_all = sfc_p_mask & oco_lt_id_mask
        
        final_length = np.sum(id_select_all)
        print(f'Total number of soundings to process: {final_length}')
        
        if platform.system() == "Darwin":
            processing_length = 1000
        elif platform.system() == "Linux":
            processing_length = 3000
        for i in range(0, final_length, processing_length):
            print(f'Processing sounding {i} to {min(i+processing_length, final_length)} out of {final_length}...')
            output_tmp = output.replace('.h5', '_tmp_{}.h5'.format(i))
            id_select = np.where(id_select_all)[0][i:i+processing_length]
            atm_dict = {'p_edge': P_edge[id_select, :], 'lat': lat_l1b_select[id_select],
                        'p_lay': pprf_l[id_select, :z_last_layer_ind], 't_lay': tprf_l[id_select, :z_last_layer_ind], 'h_lay': (h_edge[id_select, :z_last_layer_ind] + h_edge[id_select, 1:])/2,
                        'd_air_lay': air_layer[id_select, :z_last_layer_ind],
                        'd_o2_lay': o2_layer[id_select, :z_last_layer_ind], 'd_co2_lay': co2_layer[id_select, :z_last_layer_ind], 'd_h2o_lay': h2o_layer[id_select, :z_last_layer_ind],
                        'h2o_vmr': h2o_vmr[id_select, :z_last_layer_ind], 'dz': dz_hydrostatic[id_select, :z_last_layer_ind], 
                        'v_solar': v_solar[id_select], 'v_inst': v_inst[id_select], 'dist_au': dist_au[id_select], 
                        'wvl_coef': wvl_coef, 'del_lambda_all': del_lambda_all, 'rel_lresponse_all': rel_lresponse_all,
                        'sza': sza_id[id_select], 'vza': vza_id[id_select], 'fp_number': fp_number[id_select], 'doy': doy}


            need = (not os.path.isfile(output_tmp) or overwrite) and not abs_skip

            if need:
                (tau_o2a,  me_o2a,  sol_o2a), \
                (tau_wco2, me_wco2, sol_wco2), \
                (tau_sco2, me_sco2, sol_sco2) = oco_fp_abs_all_bands(atm_dict)

                print('Saving to file ' + output_tmp.format("combined"))
                with h5py.File(output_tmp, 'w') as h5f:
                    h5f.create_dataset('sza',                   data=sza_id[id_select])
                    h5f.create_dataset('vza',                   data=vza_id[id_select])
                    h5f.create_dataset('sounding_id',           data=sounding_id[id_select])
                    h5f.create_dataset('fp_number',             data=fp_number[id_select])
                    h5f.create_dataset('o2a_tau_output',        data=tau_o2a)
                    h5f.create_dataset('o2a_mean_ext_output',   data=me_o2a)
                    h5f.create_dataset('o2a_toa_sol_output',    data=sol_o2a)
                    h5f.create_dataset('wco2_tau_output',       data=tau_wco2)
                    h5f.create_dataset('wco2_mean_ext_output',  data=me_wco2)
                    h5f.create_dataset('wco2_toa_sol_output',   data=sol_wco2)
                    h5f.create_dataset('sco2_tau_output',       data=tau_sco2)
                    h5f.create_dataset('sco2_mean_ext_output',  data=me_sco2)
                    h5f.create_dataset('sco2_toa_sol_output',   data=sol_sco2)
            else:
                print(f'[Warning] Output file {output_tmp.format("combined")} exists - skipping!')
        
        
        sza_select_all = np.empty(final_length)
        vza_select_all = np.empty(final_length)
        snd_id_select_all = np.empty(final_length)
        fp_number_select_all = np.empty(final_length, dtype=int)
        o2a_tau_output_all = np.empty((final_length, 1016))
        o2a_mean_ext_output_all = np.empty((final_length, 1016))
        o2a_toa_sol_output_all = np.empty((final_length, 1016))
        wco2_tau_output_all = np.empty((final_length, 1016))
        wco2_mean_ext_output_all = np.empty((final_length, 1016))
        wco2_toa_sol_output_all = np.empty((final_length, 1016))
        sco2_tau_output_all = np.empty((final_length, 1016))
        sco2_mean_ext_output_all = np.empty((final_length, 1016))
        sco2_toa_sol_output_all = np.empty((final_length, 1016))
        
        
        
        tmp_files = [output.replace('.h5', '_tmp_{}.h5'.format(i))
                     for i in range(0, final_length, processing_length)]
        all_tmp_exist = all(os.path.isfile(f) for f in tmp_files)

        for i, output_tmp in zip(range(0, final_length, processing_length), tmp_files):
            id_select = np.where(id_select_all)[0][i:i+processing_length]
            sza_select_all[i:i+processing_length] = sza_id[id_select]
            vza_select_all[i:i+processing_length] = vza_id[id_select]
            snd_id_select_all[i:i+processing_length] = sounding_id[id_select]
            fp_number_select_all[i:i+processing_length] = fp_number[id_select]
            if os.path.isfile(output_tmp) and not abs_skip:
                with h5py.File(output_tmp, 'r') as h5_input:
                    o2a_tau_output_all[i:i+processing_length]       = np.asarray(h5_input['o2a_tau_output'])
                    o2a_mean_ext_output_all[i:i+processing_length]  = np.asarray(h5_input['o2a_mean_ext_output'])
                    o2a_toa_sol_output_all[i:i+processing_length]   = np.asarray(h5_input['o2a_toa_sol_output'])
                    wco2_tau_output_all[i:i+processing_length]      = np.asarray(h5_input['wco2_tau_output'])
                    wco2_mean_ext_output_all[i:i+processing_length] = np.asarray(h5_input['wco2_mean_ext_output'])
                    wco2_toa_sol_output_all[i:i+processing_length]  = np.asarray(h5_input['wco2_toa_sol_output'])
                    sco2_tau_output_all[i:i+processing_length]      = np.asarray(h5_input['sco2_tau_output'])
                    sco2_mean_ext_output_all[i:i+processing_length] = np.asarray(h5_input['sco2_mean_ext_output'])
                    sco2_toa_sol_output_all[i:i+processing_length]  = np.asarray(h5_input['sco2_toa_sol_output'])
            else:
                print(f'[Warning] Output file {output_tmp} does not exist - skipping!')

        if not abs_skip and all_tmp_exist:
            print('Saving combined output to file ' + output)
            with h5py.File(output, 'w') as h5_output:
                h5_output.create_dataset('sza',             data=sza_select_all)
                h5_output.create_dataset('vza',             data=vza_select_all)
                h5_output.create_dataset('sounding_id',     data=snd_id_select_all)
                h5_output.create_dataset('fp_number',       data=fp_number_select_all)
                h5_output.create_dataset('o2a_tau_output',       data=o2a_tau_output_all)
                h5_output.create_dataset('o2a_mean_ext_output',  data=o2a_mean_ext_output_all)
                h5_output.create_dataset('o2a_toa_sol_output',   data=o2a_toa_sol_output_all)
                h5_output.create_dataset('wco2_tau_output',      data=wco2_tau_output_all)
                h5_output.create_dataset('wco2_mean_ext_output', data=wco2_mean_ext_output_all)
                h5_output.create_dataset('wco2_toa_sol_output',  data=wco2_toa_sol_output_all)
                h5_output.create_dataset('sco2_tau_output',      data=sco2_tau_output_all)
                h5_output.create_dataset('sco2_mean_ext_output', data=sco2_mean_ext_output_all)
                h5_output.create_dataset('sco2_toa_sol_output',  data=sco2_toa_sol_output_all)

            print('Deleting temporary files...')
            for output_tmp in tmp_files:
                os.remove(output_tmp)
        
    return None

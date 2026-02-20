import os
import sys
import multiprocessing
import h5py
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory


# +
# NAME:
#   fp_abs_coeff.py
#
# PURPOSE:
#   Calculate the absorption coefficients for O2A, WCO2, and SCO2 bands
#   for each track and wavelength based on the atmospheric profiles and the ABSCO coefficients,
#   and then calculate the mean extinction coefficient and optical depth for each track and wavelength.
#
# EXAMPLE:
#   > abs(iband=0, nx=5, reextract=True, plot=True)
#
# MODIFICATION HISTORY:
#  - written : Sebastian Schmidt, January 25, 2014
#  -      v3 : Partition in Transmittance rather than optical thickness
#         v4 : Use actual OCO wavelengths & ILS
#         v5 : Minor bug fix# port to cluster
#         v6 : Calculate *all* OCO-2 channels (setting "all_r")
#         v7 : Include water vapor absorption (only implemented in O2A so far)
#         v8 : Converted to Python-based, change the ILS function
#
#  Done:
#  - generalize to SCO2 & WCO2 for water vapor/methane inclusion & consolidate sub-routines
#  - get accurate wavelengths from recent cal & accurate OCO line shape (but just from csv file for now)
#  - Remove getdatah5, getdatah5a by just use h5py module
#  - implement H2O broadening
#  - bi/trilinear interpolation in {T,p,(h)} - can re-use Odele's code
# ---------------------------------------------------------------------------
from abs_util.abs.calc_ext_absco import calc_ext_absco  # calculates extinction profiles & layer transmittance from oco_util.absorption coefficients & density profiles (CO2, O2)
from abs_util.abs.find_bound import find_boundary  # get wavenumber indices.initialize()import abs/get_index  # find levels in absco files that are closest to the atmosphere
from abs_util.abs.get_index import get_PT_index, get_T_index, get_P_index # find levels in absco files that are closest to the atmosphere.initialize()import abs/rdabscoo2.pro   # read absorption coefficients O2
from abs_util.abs.oco_convolve_absco import oco_conv_absco # reads OCO line shape ("slit function")
from abs_util.abs.oco_wl_absco import oco_wv_absco  # reads OCO wavelengths
from abs_util.abs.rdabs_gas_absco import rdabs_species_absco
from abs_util.abs.rdabsco_gas_absco import rdabsco_species_absco  # read absorption coefficients O2 or CO2 for a given pressure, temperature, and H2O mixing ratio
from abs_util.oco_util import timing
from scipy.interpolate import interp1d

# Edit from EaR3T code (See Chen et al., 2023, AMT, https://doi.org/10.5194/amt-16-1971-2023)
def cal_sol_fac(doy):

    """
    Calculate solar factor that accounts for Sun-Earth distance
    Input:
        dtime: datetime.datetime object
    Output:
        solfac: solar factor
    """

    eps = 0.0167086
    perh= 4.0
    rsun = (1.0 - eps*np.cos(0.017202124161707175*(doy-perh)))
    solfac = 1.0/(rsun**2)

    return solfac

def mol_ext_wvl(wv0):
    """
    Calculate the rayleigh scattering cross-section for given wavelength.

    according to Eq. 29 of Bodhaine et al, `On Rayleigh optical depth calculations', J. Atm. Ocean Technol., 16, 1854-1861, 1999.

    Input:
        wv0: wavelength (in microns)
    """

    num = 1.0455996 - 341.29061*wv0**(-2.0) - 0.90230850*wv0**2.0
    den = 1.0 + 0.0027059889*wv0**(-2.0) - 85.968563*wv0**2.0
    crs = num/den

    return crs   # in 10^-28 cm^2/molecule

def g0_calc(lat):
    """
    Calculate the surface gravity acceleration.

    according to Eq. 11 of Bodhaine et al, `On Rayleigh optical depth calculations', J. Atm. Ocean Technol., 16, 1854-1861, 1999.
    """
    lat_rad = lat * np.pi / 180
    return 9.806160 * (1 - 0.0026373 * np.cos(2*lat_rad) + 0.0000059 * np.cos(2*lat_rad)**2) # in m/s^2


def g_alt_calc(g0, lat, z):
    """
    Calculate the gravity acceleration at z.

    according to Eq. 10 of Bodhaine et al, `On Rayleigh optical depth calculations', J. Atm. Ocean Technol., 16, 1854-1861, 1999.

    Input:
        g0: gravity acceleration at the surface (m/s^2)
        lat: latitude (degrees)
        z: height (m)
    """
    lat_rad = lat * np.pi / 180
    g = g0*100 - (3.085462e-4 + 2.27e-7 * np.cos(2 * lat_rad)) * z \
           + (7.254e-11 + 1.0e-13 * np.cos(2 * lat_rad)) * z**2 \
           - (1.517e-17 + 6.0e-20 * np.cos(2 * lat_rad)) * z**3
    return g/100

def cal_mol_ext(mol_ext_wvl_array, d_air_lay, dz, lat=0, h_lay=0, dp_lev=0):

    """
    Input:
        wv0: wavelength (in microns) --- can be an array
        high_p: numpy array, Pressure of lower layer (hPa)
        low_p: numpy array, Pressure of upper layer (hPa; high_p > low_p)
    Output:
        tauray: extinction
    Example: calculate Rayleigh optical depth between 37 km (~4 hPa) and sea level (1000 hPa) at 0.5 microns:
    in Python program:
        result=bodhaine(0.5,1000,4)
    Note: If you input an array of wavelengths, the result will also be an
          array corresponding to the Rayleigh optical depth at these wavelengths.
    """

    """
    Input:
        wv0: wavelength (in microns) --- can be an array
        pz1: numpy array, Pressure of lower layer (hPa)
        pz2: numpy array, Pressure of upper layer (hPa; pz1 > pz2)
        atm0: er3t atmosphere object
        method: string, 'sfc' or 'lay'
    Output:
        tauray: extinction
    Example: calculate Rayleigh optical depth between 37 km (~4 hPa) and sea level (1000 hPa) at 0.5 microns:
    in Python program:
        result=bodhaine(0.5,1000,4)
    Note: If you input an array of wavelengths, the result will also be an
          array corresponding to the Rayleigh optical depth at these wavelengths.
    """
    # avogadro's number
    # A_ = 6.02214179e23


    # g0 = g0_calc(lat) # m/s^2
    # g0 = g0_calc(0) # m/s^2
    # g = g_alt_calc(g0, lat, h_lay*1000) * 100 # convert to cm/s^2

    # g0 = g0 * 100 # convert to cm/s^2
    # ma = 28.9595 # mean molecular weight of air in g/mol
    # dp_lev = dp_lev * 1000 # convert to dyne/cm^2
    # crs = mol_ext_wvl_array

    # const_lay = dp_lev * A_ / (g * ma) * 1e-28
    # tauray = const_lay*(crs)

    # original calculation
    # tauray = 0.00210966*(crs)*(p_lev[:-1]-p_lev[1:])/1013.25

    # mol_ext_wvl_array: crs of 1016 wavelengths in O2A, WCO2, SCO2 bands, in 10^-28 cm^2/molecule
    # d_air_lay: air number density in molecule/cm^3 of 71 layers
    # dz: layer thickness in km of 71 layers
    tauray = np.repeat(mol_ext_wvl_array[:, np.newaxis], d_air_lay.shape[0], axis=1)* 1e-28 * d_air_lay * dz * 1000 * 100

    # ext_array = np.repeat(mol_ext_wvl_array[:, np.newaxis], d_air_lay.shape[0], axis=1)* 1e-28 * d_air_lay * 1000 * 100

    return tauray


# Constants
C_LIGHT = 299792458.0  # Speed of light in m/s

def derive_oco2_solar_radiance(solar_h5_path,
                               v_solar, v_inst, dist_au, target_wl_microns, band_num):
    """
    Final implementation using exact OCO-2 L1B metadata and solar H5 structure.
    band_num: 1 (O2), 2 (WCO2), or 3 (SCO2)
    """

    # 2. Extract Solar Model Tables
    with h5py.File(solar_h5_path, 'r') as f_sol:
        # Using the specific paths from your solar h5 metadata
        abs_path = f'Solar/Absorption/Absorption_{band_num}'
        cont_path = f'Solar/Continuum/Continuum_{band_num}'

        sol_abs_nu = f_sol[abs_path]['wavenumber'][:]
        sol_abs_val = f_sol[abs_path]['spectrum'][:]

        sol_cont_nu = f_sol[cont_path]['wavenumber'][:]
        sol_cont_val = f_sol[cont_path]['spectrum'][:]

    # 3. Create Observed Wavenumber Grid (Instrument frame)
    target_nu_obs = 10000.0 / target_wl_microns  # Convert microns to cm^-1

    # 4. Doppler Correction Logic
    # The tables are in the SOLAR REST FRAME.
    # We shift our observed wavenumbers to match the table's frame.
    # Formula: nu_rest = nu_obs * (1 + (v_solar + v_inst)/c)
    beta_total = (v_solar + v_inst) / C_LIGHT
    nu_shifted = target_nu_obs * (1.0 + beta_total)

    # 5. Interpolate and Scale
    # Interpolate solar absorption (dimensionless) and continuum (ph/s/m^2/um)
    interp_abs = interp1d(sol_abs_nu, sol_abs_val, bounds_error=False, fill_value=1.0)
    interp_cont = interp1d(sol_cont_nu, sol_cont_val, bounds_error=False, fill_value="extrapolate")

    # Final TOA solar radiance = Transmittance * Continuum * (1/d^2)
    dist_scaling = (1.0 / dist_au)**2

    toa_solar_radiance = interp_abs(nu_shifted) * interp_cont(nu_shifted) * dist_scaling

    return target_wl_microns, toa_solar_radiance


# ---------------------------------------------------------------------------
# Parallel worker support — module-level so pickling works under spawn
# ---------------------------------------------------------------------------
_shared_state = {}
_shm_handles = []   # SharedMemory objects kept alive inside each worker


def _init_worker(state):
    """Receive lightweight state; reconstruct Gas_Absorption arrays from shared memory."""
    global _shared_state, _shm_handles
    
    print(f"Worker {os.getpid()} starting initialization...", flush=True)
    
    try:

        # Attach H2O Gas_Absorption from shared memory
        shm_h2o = SharedMemory(name=state['h2o_gas_abs_shm']['name'])
        _shm_handles.append(shm_h2o)
        h2o_gas_abs = np.ndarray(
            state['h2o_gas_abs_shm']['shape'],
            dtype=state['h2o_gas_abs_shm']['dtype'],
            buffer=shm_h2o.buf,
        )
        absco_data_h2o = dict(state['absco_data_h2o'])
        absco_data_h2o['Gas_Absorption'] = h2o_gas_abs

        # Attach per-band Gas_Absorption from shared memory.
        # CO2 bands (1 and 2) share the same shm block — open it only once.
        _opened: dict = {}
        bands = {}
        for iband, bs in state['bands'].items():
            shm_meta = bs['gas_abs_shm']
            shm_name = shm_meta['name']
            if shm_name not in _opened:
                shm = SharedMemory(name=shm_name)
                _shm_handles.append(shm)
                _opened[shm_name] = shm
            gas_abs = np.ndarray(
                shm_meta['shape'], dtype=shm_meta['dtype'],
                buffer=_opened[shm_name].buf,
            )
            _shm_keys = {'gas_abs_shm', 'xx_fps_shm', 'yy_fps_shm', 'wloco_fps_shm'}
            new_bs = {k: v for k, v in bs.items() if k not in _shm_keys}
            new_bs['absco_data_gas'] = dict(new_bs['absco_data_gas'])
            new_bs['absco_data_gas']['Gas_Absorption'] = gas_abs
            for key in ('xx_fps', 'yy_fps', 'wloco_fps'):
                meta = bs[f'{key}_shm']
                shm = SharedMemory(name=meta['name'])
                _shm_handles.append(shm)
                new_bs[key] = np.ndarray(meta['shape'], dtype=meta['dtype'], buffer=shm.buf)
            bands[iband] = new_bs

        _shared_state = {
            'bands': bands,
            'absco_data_h2o': absco_data_h2o,
            'ph2o': state['ph2o'],
            'tkh2o': state['tkh2o'],
            'broadh2o': state['broadh2o'],
            'hpah2o': state['hpah2o'],
            'trilinear_matrix': state['trilinear_matrix'],
            'pdmax': state['pdmax'],
            'tdmax': state['tdmax'],
            'solar_h5_path': state['solar_h5_path'],
        }
    
        print(f"Worker {os.getpid()} successfully attached to SharedMemory.", flush=True)
    except Exception as e:
        print(f"Worker {os.getpid()} FAILED: {e}", flush=True)

import psutil

def monitor_memory_and_init(func):
    """Decorator to log worker health and memory usage to SLURM output."""
    def wrapper(*args, **kwargs):
        pid = os.getpid()
        process = psutil.Process(pid)
        # Check memory before
        mem_gb = process.memory_info().rss / 1e9
        
        # Run the physics task
        result = func(*args, **kwargs)
        
        # Periodic log for HPC (only log every 10th track to avoid clutter)
        # You can adjust this logic or remove for local Mac use.
        if np.random.rand() < 0.1: 
            print(f"  [Worker {pid}] Mem: {mem_gb:.2f}GB | Task complete.", flush=True)
        return result
    return wrapper

# Apply it to your worker
@monitor_memory_and_init
def _process_track_all_bands(track_data):
    """Process a single track for all 3 bands in one pool call."""
    s = _shared_state
    pprf, tprf, d_air_lay, o2den, co2den, h2oden, h2o_vmr, dzf, solzen, obszen, fp, v_solar, v_inst, dist_au = track_data

    # Data shared across all bands
    absco_data_h2o   = s['absco_data_h2o']
    solar_h5_path    = s['solar_h5_path']
    ph2o, tkh2o      = s['ph2o'], s['tkh2o']
    broadh2o, hpah2o = s['broadh2o'], s['hpah2o']
    trilinear_matrix = s['trilinear_matrix']
    pdmax, tdmax     = s['pdmax'], s['tdmax']

    convr    = np.pi / 180.0
    musolzen = 1.0 / np.cos(solzen * convr)
    muobszen = 1.0 / np.cos(obszen * convr)
    nlay     = len(dzf)

    band_results = []
    for iband in range(3):
        bs = s['bands'][iband]
        absco_data_gas     = bs['absco_data_gas']
        p_gas, tk_gas      = bs['p_gas'], bs['tk_gas']
        broad_gas, hpa_gas = bs['broad_gas'], bs['hpa_gas']
        nudat, wvldat    = bs['nudat'], bs['wvldat']
        nwav, nwavh2o      = bs['nwav'], bs['nwavh2o']
        inu1, inu2       = bs['inu1'], bs['inu2']
        inu1h2o, inu2h2o = bs['inu1h2o'], bs['inu2h2o']
        iw_h2o_start       = bs['iw_um_h2o_in_gas_start']
        iw_h2o_end         = bs['iw_um_h2o_in_gas_end']
        wloco              = bs['wloco_fps'][fp]
        xx                 = bs['xx_fps'][fp]
        yy                 = bs['yy_fps'][fp]
        rdabs_gas          = bs['rdabs_gas']
        # Rayleigh cross-section at atmosphere-frame wavelengths (wl_atm = wl_inst*(1+v_inst/c))
        wloco_atm    = wloco# * (1.0 + v_inst / C_LIGHT) # already accounted for Doppler and thermo-mechanical shifts observed
        molec_ext_fp = mol_ext_wvl(wloco_atm)
        sol_abs_nu         = bs['sol_nu']
        sol_abs_val        = bs['sol_abs_val']
        sol_cont_val       = bs['sol_cont_val']

        rdabs_gas_den = o2den if iband == 0 else co2den
        tau_molec_ext_lays = np.sum(cal_mol_ext(molec_ext_fp, d_air_lay, dzf), axis=1)

        h2o_needs_padding = (nwav != nwavh2o)
        ext1 = np.zeros(nwav)
        ext = np.empty((nwav, nlay))
        P_inds = get_P_index(pprf, hpa_gas, trilinear=True)
        P_inds_h2o = get_P_index(pprf, hpah2o, trilinear=True)
        for iz in range(nlay)[::-1]:
            tkobs, pobs = tprf[iz], pprf[iz]

            # T_ind,     P_ind     = get_PT_index(tkobs, pobs, tk_gas,  hpa_gas,  trilinear=True)
            # T_ind_h2o, P_ind_h2o = get_PT_index(tkobs, pobs, tkh2o,   hpah2o,   trilinear=True)

            P_ind = P_inds[iz]
            P_ind_h2o = P_inds_h2o[iz]
            
            T_ind = get_T_index(tkobs, tk_gas, P_ind, trilinear=True)
            T_ind_h2o = get_T_index(tkobs, tkh2o, P_ind_h2o, trilinear=True)

            absco = rdabsco_species_absco(
                absco_data_gas, p_gas, tk_gas, broad_gas, hpa_gas,
                tkobs, pobs, T_ind, P_ind, h2o_vmr[iz],
                inu1, inu2, species=rdabs_gas,
                trilinear_matrix=trilinear_matrix, mode="trilinear",
            )
            abscoh2o = rdabsco_species_absco(
                absco_data_h2o, ph2o, tkh2o, broadh2o, hpah2o,
                tkobs, pobs, T_ind_h2o, P_ind_h2o, h2o_vmr[iz],
                inu1h2o, inu2h2o, species="h2o",
                trilinear_matrix=trilinear_matrix, mode="trilinear",
            )

            ext0  = calc_ext_absco(rdabs_gas_den[iz], absco)
            ext1_ = calc_ext_absco(h2oden[iz], abscoh2o)
            if h2o_needs_padding:
                ext1[:] = 0
                ext1[iw_h2o_start:iw_h2o_end + 1] = ext1_
            else:
                ext1 = ext1_
            ext[:, iz] = ext0 + ext1
        
        # Doppler-shift ABSCO wavenumbers from atmosphere rest frame to instrument frame.
        # nu_inst = nu_atm / (1 + v_inst/c) — consistent with solar shift sign convention.
        wvldat_obs = wvldat# * (1.0 + v_inst / C_LIGHT)  # already accounted for Doppler and thermo-mechanical shifts observed 
        indlr   = oco_conv_absco(wloco, xx, yy, wvldat_obs, nwav)
        nx      = len(wloco)
        wvlabsco = wvldat_obs
        
        
        # Doppler-shift the solar wavenumber grid from  solar rest frame to instrument frame.
        # wvlsol_obs is in um (instrument frame); wvlsol_obs = wvlsol_rest * (1 - (v_solar+v_inst)/c)
        fsol = (sol_abs_val * sol_cont_val * (1.0 / dist_au) ** 2)
        # beta_total   = (v_solar + v_inst) / C_LIGHT
        beta_total   = (v_solar) / C_LIGHT # already accounted for Doppler and thermo-mechanical shifts observed 
        wvlsol_ori = 1e4/sol_abs_nu
        wvlsol_obs = wvlsol_ori * (1.0 + beta_total)

        # Adjust flux density for the 'stretching' of the wavelength bins
        fsol_obs = fsol / (1.0 + beta_total)
        indlr_sol  = oco_conv_absco(wloco, xx, yy, wvlsol_obs, len(wvlsol_obs))
    
        xx_wloco    = xx + wloco[:, np.newaxis]   # (1016, 200) — precomputed once per band
        ext_profile = np.empty((nx, nlay))
        toa_sol     = np.empty(nx)

        for ind in range(nx):
            start, end = indlr[ind, 1], indlr[ind, 0] + 1
            ilg0     = np.interp(wvlabsco[start:end], xx_wloco[ind], yy[ind])
            ilg0_sum = np.sum(ilg0)
            ext_profile[ind, :] = ext[start:end, :].T @ ilg0 / ilg0_sum
            start_sol, end_sol = indlr_sol[ind, 1], indlr_sol[ind, 0] + 1
            ilg0_sol     = np.interp(wvlsol_obs[start_sol:end_sol], xx_wloco[ind], yy[ind])
            ilg0_sol_sum = np.sum(ilg0_sol)
            toa_sol[ind] = np.dot(fsol_obs[start_sol:end_sol], ilg0_sol) / ilg0_sol_sum * musolzen

            # if (ind == nx-1 or ind == nx-2) and iband == 1:
            #     import matplotlib.pyplot as plt
            #     fig, ax = plt.subplots(figsize=(10, 6))
            #     l1 = ax.plot(xx[ind, :] + wloco[ind], yy[ind, :], label='Line Shape (Instrument Frame)')
            #     l2 = ax.plot(wvlsol_obs[start_sol:end_sol], ilg0_sol, label='Solar Spectrum (Doppler-Shifted)')
            #     ax1 = ax.twinx()
            #     l3 = ax1.plot(wvlsol_obs[start_sol:end_sol], fsol_obs[start_sol:end_sol], label='Solar Flux Density (Doppler-Shifted)', color='orange')
            #     ax.set_xlabel('Wavelength (um)')
            #     ax.set_title(f'Band {iband+1} - FP {fp} - Wavelength Bin {ind}')
            #     ax.legend([l1[0], l2[0], l3[0]], ['Line Shape (Instrument Frame)', 'Solar Spectrum (Doppler-Shifted)', 'Solar Flux Density (Doppler-Shifted)'])
            #     # ax.legend([l2[0], l3[0]], ['Solar Spectrum (Doppler-Shifted)', 'Solar Flux Density (Doppler-Shifted)'])
            #     plt.show()
                

        tau      = (np.sum(ext_profile * dzf, axis=1) + tau_molec_ext_lays) * (musolzen + muobszen)
        mean_ext = tau / (np.sum(dzf) * (musolzen + muobszen))
        band_results.append((tau, mean_ext, toa_sol))
        
    return band_results[0], band_results[1], band_results[2]


# ---------------------------------------------------------------------------

@timing
def oco_fp_abs_all_bands(atm_dict, n_workers=None):
    """
    Compute gas absorption tau, mean extinction, and TOA solar irradiance for
    all three OCO bands (O2A, WCO2, SCO2) in a single parallel pool pass.

    Compared to calling oco_fp_abs three times this avoids:
      - 3x ProcessPoolExecutor spawn/teardown
      - 3x ABSCO and L1b HDF5 reads
      - 3x pickle+send of large shared state to workers
    """
    import platform
    if platform.system() == "Darwin":
        pathinp = "./src/abs_util/abs/"
    elif platform.system() == "Linux":
        pathinp = "/pl/active/vikas-arcsix/yuch8913/oco/data/absco/v5.2_final/"
    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")

    pdmax = 100
    tdmax = 15

    xr_dict        = {0: [0.755, 0.784], 1: [1.587, 1.629], 2: [2.020, 2.101]}
    lb_dict        = {0: "o2a",          1: "wco2",          2: "sco2"}
    rdabs_gas_dict = {0: "o2",           1: "co2",           2: "co2"}
    gas_id_dict    = {0: "07",           1: "02",            2: "02"}
    absco_files    = [pathinp + f for f in ("o2_v52.hdf", "co2_v52.hdf", "co2_v52.hdf")]

    print("***********************************************************")
    print("Calculate gas absorption for all bands: O2A, WCO2, SCO2")
    print("***********************************************************")

    solar_h5_path = pathinp + "l2_solar_model.h5"

    # --- Read H2O ABSCO once (identical for all bands) ---
    with h5py.File(pathinp + "h2o_v52.hdf", 'r') as f:
        absco_data_h2o = {
            "Pressure":         f['Pressure'][...],
            "Temperature":      f['Temperature'][...],
            "Wavenumber":       f['Wavenumber'][...],
            "Broadener_01_VMR": f['Broadener_01_VMR'][...],
            "Gas_Absorption":   f['Gas_01_Absorption'][...],
        }
    nuh2o, ph2o, tkh2o, broadh2o, hpah2o, _ = rdabs_species_absco(
        absco_data=absco_data_h2o, species="h2o"
    )

    # --- Get L1b ILS / wavelength data once (passed in atm_dict) ---
    wvl_coef          = atm_dict["wvl_coef"]
    del_lambda_all    = atm_dict["del_lambda_all"]
    rel_lresponse_all = atm_dict["rel_lresponse_all"]

    # --- Build per-band state (cache CO2 ABSCO so it is read only once) ---
    _absco_cache = {}   # path -> absco_data_gas dict
    band_states  = {}
    for iband in range(3):
        wavel1, wavel2 = xr_dict[iband]
        nu2, nu1     = 1.0e4 / wavel1, 1.0e4 / wavel2

        absco_path = absco_files[iband]
        if absco_path not in _absco_cache:
            with h5py.File(absco_path, 'r') as f:
                _absco_cache[absco_path] = {
                    "Pressure":         f['Pressure'][...],
                    "Temperature":      f['Temperature'][...],
                    "Wavenumber":       f['Wavenumber'][...],
                    "Broadener_01_VMR": f['Broadener_01_VMR'][...],
                    "Gas_Absorption":   f[f'Gas_{gas_id_dict[iband]}_Absorption'][...],
                }
        absco_data_gas = _absco_cache[absco_path]

        rdabs_gas = rdabs_gas_dict[iband]
        nu_gas, p_gas, tk_gas, broad_gas, hpa_gas, _ = rdabs_species_absco(
            absco_data=absco_data_gas, species=rdabs_gas
        )
        inu1, inu2, nwav, nudat, wvldat = find_boundary(nu1, nu2, nu_gas)

        band_num = iband + 1
        with h5py.File(solar_h5_path, 'r') as f_sol:
            sol_abs_nu   = f_sol[f'Solar/Absorption/Absorption_{band_num}/wavenumber'][:]
            sol_abs_val  = f_sol[f'Solar/Absorption/Absorption_{band_num}/spectrum'][:]
            sol_cont_nu  = f_sol[f'Solar/Continuum/Continuum_{band_num}/wavenumber'][:]
            sol_cont_val = f_sol[f'Solar/Continuum/Continuum_{band_num}/spectrum'][:]
            
        sol_cont_val = np.interp(sol_abs_nu, sol_cont_nu, sol_cont_val) 

        inu1h2o, inu2h2o, nwavh2o, nudath2o, _ = find_boundary(nu1, nu2, nuh2o)
        if nwav != nwavh2o:
            print(f"[Warning] Wavenumber gridding of {rdabs_gas.upper()} & H2O ABSCO files do not match for band {lb_dict[iband]}.")
        iw_h2o_start, iw_h2o_end = np.searchsorted(nudat, nudath2o[[0, -1]], side='left')

        wloco_fps = np.empty((8, 1016), dtype=np.float32)
        xx_fps    = np.empty((8, 1016, 200), dtype=np.float32)
        yy_fps    = np.empty((8, 1016, 200), dtype=np.float32)
        for fp in range(8):
            wloco_fps[fp] = oco_wv_absco(wvl_coef, iband, footprint=fp)  # already accounted for Doppler and thermo-mechanical shifts observed 
            xx_fps[fp]    = del_lambda_all[iband, fp, :, :]
            rel_lr        = rel_lresponse_all[iband, fp, :, :]
            norm          = np.repeat(rel_lr.max(axis=1), 200).reshape(1016, 200)
            yy_fps[fp]    = rel_lr / norm

        band_states[iband] = {
            'absco_data_gas': absco_data_gas,
            'p_gas': p_gas, 'tk_gas': tk_gas, 'broad_gas': broad_gas, 'hpa_gas': hpa_gas,
            'nudat': nudat, 'wvldat': wvldat, 'nwav': nwav, 'nwavh2o': nwavh2o,
            'inu1': inu1, 'inu2': inu2, 'inu1h2o': inu1h2o, 'inu2h2o': inu2h2o,
            'iw_um_h2o_in_gas_start': iw_h2o_start,
            'iw_um_h2o_in_gas_end':   iw_h2o_end,
            'wloco_fps': wloco_fps, 'xx_fps': xx_fps, 'yy_fps': yy_fps,
            'rdabs_gas': rdabs_gas,
            'sol_nu': sol_abs_nu, 'sol_abs_val': sol_abs_val, 'sol_cont_val': sol_cont_val,
        }

    # --- Move Gas_Absorption arrays into OS shared memory to avoid pickle overhead ---
    # Each worker attaches zero-copy; only small metadata (name/shape/dtype) is pickled.
    _shm_blocks = []
    _gas_shm_cache = {}   # absco_path -> shm metadata  (CO2 is reused for bands 1 and 2)

    ga_h2o = absco_data_h2o['Gas_Absorption']
    _shm_h2o = SharedMemory(create=True, size=ga_h2o.nbytes)
    _shm_blocks.append(_shm_h2o)
    np.copyto(np.ndarray(ga_h2o.shape, dtype=ga_h2o.dtype, buffer=_shm_h2o.buf), ga_h2o)
    h2o_gas_abs_shm = {'name': _shm_h2o.name, 'shape': ga_h2o.shape, 'dtype': ga_h2o.dtype}
    absco_data_h2o_small = {k: v for k, v in absco_data_h2o.items() if k != 'Gas_Absorption'}

    for iband in range(3):
        absco_path = absco_files[iband]
        if absco_path not in _gas_shm_cache:
            ga = band_states[iband]['absco_data_gas']['Gas_Absorption']
            _shm_gas = SharedMemory(create=True, size=ga.nbytes)
            _shm_blocks.append(_shm_gas)
            np.copyto(np.ndarray(ga.shape, dtype=ga.dtype, buffer=_shm_gas.buf), ga)
            _gas_shm_cache[absco_path] = {'name': _shm_gas.name, 'shape': ga.shape, 'dtype': ga.dtype}
        band_states[iband]['gas_abs_shm'] = _gas_shm_cache[absco_path]
        band_states[iband]['absco_data_gas'] = {
            k: v for k, v in band_states[iband]['absco_data_gas'].items() if k != 'Gas_Absorption'
        }

    # --- Move xx_fps / yy_fps / wloco_fps into shared memory ---
    for iband in range(3):
        for key in ('xx_fps', 'yy_fps', 'wloco_fps'):
            arr = band_states[iband][key]
            shm = SharedMemory(create=True, size=arr.nbytes)
            _shm_blocks.append(shm)
            np.copyto(np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf), arr)
            band_states[iband][f'{key}_shm'] = {'name': shm.name, 'shape': arr.shape, 'dtype': arr.dtype}
            del band_states[iband][key]

    shared_state = {
        'bands': band_states,
        'absco_data_h2o': absco_data_h2o_small,
        'h2o_gas_abs_shm': h2o_gas_abs_shm,
        'ph2o': ph2o, 'tkh2o': tkh2o, 'broadh2o': broadh2o, 'hpah2o': hpah2o,
        'trilinear_matrix': np.array([[ 1,  0,  0,  0,  0,  0,  0,  0],
                                      [-1,  0,  0,  0,  1,  0,  0,  0],
                                      [-1,  0,  1,  0,  0,  0,  0,  0],
                                      [-1,  1,  0,  0,  0,  0,  0,  0],
                                      [ 1,  0, -1,  0, -1,  0,  1,  0],
                                      [ 1, -1, -1,  1,  0,  0,  0,  0],
                                      [ 1, -1,  0,  0, -1,  1,  0,  0],
                                      [-1,  1,  1, -1,  1, -1, -1,  1]], dtype=np.float64),
        'pdmax': pdmax, 'tdmax': tdmax,
        'solar_h5_path': solar_h5_path,
    }

    n_tracks = len(atm_dict["fp_number"])
    track_args = [
        (
            atm_dict["p_lay"][i, :],
            atm_dict["t_lay"][i, :],
            atm_dict["d_air_lay"][i, :],
            atm_dict["d_o2_lay"][i, :],
            atm_dict["d_co2_lay"][i, :],
            atm_dict["d_h2o_lay"][i, :],
            atm_dict["h2o_vmr"][i, :],
            atm_dict["dz"][i, :],
            float(atm_dict["sza"][i]),
            float(atm_dict["vza"][i]),
            int(atm_dict["fp_number"][i]),
            float(atm_dict["v_solar"][i]),
            float(atm_dict["v_inst"][i]),
            float(atm_dict["dist_au"][i]),
        )
        for i in range(n_tracks)
    ]

    if n_workers is not None:
        n_workers_actual = n_workers
    else:
        slurm_ntasks = int(os.environ.get('SLURM_NTASKS', 0))
        cpu_count = slurm_ntasks if slurm_ntasks > 0 else (os.cpu_count() or 1)
        n_workers_actual = (cpu_count - 1) if platform.system() in ("Darwin", "Windows") else cpu_count
        n_workers_actual = max(1, min(n_workers_actual, n_tracks))

    # fork is safe on Linux *only when* MKL/OpenBLAS are forced to single-threaded mode
    # (OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1) before Python starts, so the parent never
    # builds a thread pool.  The shell script exports those vars before invoking Python.
    # forkserver is avoided because it re-imports every module per worker from scratch;
    # on Lustre (/pl/active/) that cold-import penalty causes 10-30 min of apparent hang
    # at "Tracks: 0/N" before any computation begins.
    if platform.system() == "Darwin":
        mp_ctx = multiprocessing.get_context("spawn")
    else:
        # mp_ctx = multiprocessing.get_context("fork")
        mp_ctx = multiprocessing.get_context("spawn")
        

    print(f"Dispatching {n_tracks} tracks across {n_workers_actual} workers (all 3 bands) ...")
    # try:
    #     with ProcessPoolExecutor(
    #         max_workers=n_workers_actual,
    #         mp_context=mp_ctx,
    #         initializer=_init_worker,
    #         initargs=(shared_state,),
    #     ) as pool:
    #         results = list(tqdm(pool.map(_process_track_all_bands, track_args), total=n_tracks, desc="Tracks"))
    # finally:
    #     for shm in _shm_blocks:
    #         shm.close()
    #         shm.unlink()
    results = []
    try:
        # Use mp_ctx.Pool instead of ProcessPoolExecutor
        # This allows us to use .imap() which is much better for large HPC jobs
        with mp_ctx.Pool(
            processes=n_workers_actual,
            initializer=_init_worker,
            initargs=(shared_state,),
        ) as pool:

            # imap with chunksize=1 is the "Secret Sauce" for HPC:
            # 1. It doesn't pickle the entire 'track_args' list at once (prevents memory spike).
            # 2. It lets tqdm update the second a worker finishes (prevents the 'stuck' look).
            pbar = tqdm(total=n_tracks, desc="Tracks")
            for res in pool.imap(_process_track_all_bands, track_args, chunksize=1):
                results.append(res)
                pbar.update(1)
            pbar.close()

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}", flush=True)
        raise
    finally:
        # Cleanup SharedMemory blocks
        for shm in _shm_blocks:
            try:
                shm.close()
                shm.unlink()
            except:
                pass

    o2a_tau  = np.zeros((n_tracks, 1016));  o2a_me  = np.zeros((n_tracks, 1016));  o2a_sol  = np.zeros((n_tracks, 1016))
    wco2_tau = np.zeros((n_tracks, 1016));  wco2_me = np.zeros((n_tracks, 1016));  wco2_sol = np.zeros((n_tracks, 1016))
    sco2_tau = np.zeros((n_tracks, 1016));  sco2_me = np.zeros((n_tracks, 1016));  sco2_sol = np.zeros((n_tracks, 1016))

    for i, ((t0, me0, s0), (t1, me1, s1), (t2, me2, s2)) in enumerate(results):
        o2a_tau[i],  o2a_me[i],  o2a_sol[i]  = t0, me0, s0
        wco2_tau[i], wco2_me[i], wco2_sol[i] = t1, me1, s1
        sco2_tau[i], sco2_me[i], sco2_sol[i] = t2, me2, s2

    return (o2a_tau,  o2a_me,  o2a_sol), \
           (wco2_tau, wco2_me, wco2_sol), \
           (sco2_tau, sco2_me, sco2_sol)


if __name__ == "__main__":
    None

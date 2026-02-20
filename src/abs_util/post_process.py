import numpy as np
import sys
import h5py
import glob
from er3t.util.modis import modis_l1b
from er3t.util import grid_by_dxdy


def convert_photon_unit(data_photon, wavelength, scale_factor=2.0):

    c = 299792458.0
    h = 6.62607015e-34
    wavelength = wavelength * 1e-9
    data = data_photon/1000.0*c*h/wavelength*scale_factor

    return data

class oco2_rad_nadir:

    def __init__(self, sat):
        self.fname_l1b = sat.fnames['oco_l1b'][0]
        self.fname_std = sat.fnames['oco_std'][0]
        self.extent = sat.extent

        # =================================================================================
        self.cal_wvl()
        # after this, the following three functions will be created
        # Input: index, range from 0 to 7, e.g., 0, 1, 2, ..., 7
        # self.get_wvl_o2_a(index)
        # self.get_wvl_co2_weak(index)
        # self.get_wvl_co2_strong(index)
        # =================================================================================

        # =================================================================================
        self.get_index(self.extent)
        # after this, the following attributes will be created
        # self.index_s: starting index
        # self.index_e: ending index
        # =================================================================================

        # =================================================================================
        self.overlap(index_s=self.index_s, index_e=self.index_e)
        # after this, the following attributes will be created
        # self.logic_l1b
        # self.lon_l1b
        # self.lat_l1b
        # =================================================================================

        # =================================================================================
        self.get_data(index_s=self.index_s, index_e=self.index_e)
        # after this, the following attributes will be created
        # self.rad_o2_a
        # self.rad_co2_weak
        # self.rad_co2_strong
        # =================================================================================

    def cal_wvl(self, Nchan=1016):
        """
        Calculate wavelength for th following bands
        Oxygen A band: centered at 765 nm
        Weak CO2 band: centered at 1610 nm
        Strong CO2 band: centered at 2060 nm
        """

        with h5py.File(self.fname_l1b, 'r') as f:
            wvl_coef = f['InstrumentHeader/dispersion_coef_samp'][...]

        _, Nfoot, Ncoef = wvl_coef.shape
        wvl_o2_a       = np.zeros((Nfoot, Nchan), dtype=np.float64)
        wvl_co2_weak   = np.zeros((Nfoot, Nchan), dtype=np.float64)
        wvl_co2_strong = np.zeros((Nfoot, Nchan), dtype=np.float64)

        chan = np.arange(1, Nchan+1)
        for i in range(Nfoot):
            for j in range(Ncoef):
                wvl_o2_a[i, :]       += wvl_coef[0, i, j]*chan**j
                wvl_co2_weak[i, :]   += wvl_coef[1, i, j]*chan**j
                wvl_co2_strong[i, :] += wvl_coef[2, i, j]*chan**j

        wvl_o2_a       *= 1000.0
        wvl_co2_weak   *= 1000.0
        wvl_co2_strong *= 1000.0

        self.get_wvl_o2_a       = lambda index: wvl_o2_a[index, :]
        self.get_wvl_co2_weak   = lambda index: wvl_co2_weak[index, :]
        self.get_wvl_co2_strong = lambda index: wvl_co2_strong[index, :]

    def get_index(self, extent):
        if extent is None:
            sys.exit('Error   [oco_rad_nadir]: extent is not specified.')
        else:
            with h5py.File(self.fname_l1b, 'r') as f:
                lon_l1b     = f['SoundingGeometry/sounding_longitude'][...]
                lat_l1b     = f['SoundingGeometry/sounding_latitude'][...]

            logic = (lon_l1b>=extent[0]) & (lon_l1b<=extent[1]) &\
                    (lat_l1b>=extent[2]) & (lat_l1b<=extent[3])
            indices = np.where(np.sum(logic, axis=1)>0)[0]
            self.index_s = indices[0]
            self.index_e = indices[-1]

    def overlap(self, index_s=0, index_e=None, lat0=0.0, lon0=0.0):
        with h5py.File(self.fname_l1b, 'r') as f:
            self.lon_l1b     = f['SoundingGeometry/sounding_longitude'][...][index_s:index_e, ...]
            self.lat_l1b     = f['SoundingGeometry/sounding_latitude'][...][index_s:index_e, ...]
            self.lon_l1b_o2a = f['FootprintGeometry/footprint_longitude'][...][index_s:index_e, ..., 0]
            self.lat_l1b_o2a = f['FootprintGeometry/footprint_latitude'][...][index_s:index_e, ..., 0]
            self.lon_l1b_wco2 = f['FootprintGeometry/footprint_longitude'][...][index_s:index_e, ..., 1]
            self.lat_l1b_wco2 = f['FootprintGeometry/footprint_latitude'][...][index_s:index_e, ..., 1]
            self.lon_l1b_sco2 = f['FootprintGeometry/footprint_longitude'][...][index_s:index_e, ..., 2]
            self.lat_l1b_sco2 = f['FootprintGeometry/footprint_latitude'][...][index_s:index_e, ..., 2]
            self.snd_id_l1b  = f['SoundingGeometry/sounding_id'][...][index_s:index_e, ...]

        
        shape    = self.lon_l1b.shape

        with h5py.File(self.fname_std, 'r') as f:
            self.lon_std      = f['RetrievalGeometry/retrieval_longitude'][...]
            self.lat_std      = f['RetrievalGeometry/retrieval_latitude'][...]
            self.xco2_std     = f['RetrievalResults/xco2'][...]
            self.snd_id_std   = f['RetrievalHeader/sounding_id'][...]
            self.sfc_pres_std = f['RetrievalResults/surface_pressure_fph'][...]

        self.logic_l1b = np.in1d(self.snd_id_l1b, self.snd_id_l1b).reshape(shape)
        self.snd_id    = self.snd_id_l1b
        
        xco2      = np.zeros_like(self.lon_l1b); xco2[...] = np.nan
        sfc_pres  = np.zeros_like(self.lon_l1b); sfc_pres[...] = np.nan

        for i in range(xco2.shape[0]):
            for j in range(xco2.shape[1]):
                logic = (self.snd_id_std==self.snd_id_l1b[i, j])
                if logic.sum() == 1:
                    xco2[i, j] = self.xco2_std[logic]
                    sfc_pres[i, j] = self.sfc_pres_std[logic]
                elif logic.sum() > 1:
                    sys.exit('Error   [oco_rad_nadir]: More than one point is found.')

        self.xco2      = xco2
        self.sfc_pres  = sfc_pres

    def get_data(self, index_s, index_e):

        with h5py.File(self.fname_l1b, 'r') as f:
            self.rad_o2_a       = f['SoundingMeasurements/radiance_o2'][...][index_s:index_e, ...]
            self.rad_co2_weak   = f['SoundingMeasurements/radiance_weak_co2'][...][index_s:index_e, ...]
            self.rad_co2_strong = f['SoundingMeasurements/radiance_strong_co2'][...][index_s:index_e, ...]
            self.sza            = f['SoundingGeometry/sounding_solar_zenith'][...][index_s:index_e, ...]
            self.saa            = f['SoundingGeometry/sounding_solar_azimuth'][...][index_s:index_e, ...]

            for i in range(8):
                self.rad_o2_a[:, i, :]       = convert_photon_unit(self.rad_o2_a[:, i, :]      , self.get_wvl_o2_a(i))
                self.rad_co2_weak[:, i, :]   = convert_photon_unit(self.rad_co2_weak[:, i, :]  , self.get_wvl_co2_weak(i))
                self.rad_co2_strong[:, i, :] = convert_photon_unit(self.rad_co2_strong[:, i, :], self.get_wvl_co2_strong(i))


def cdata_all(date, tag, fdir_mca, fname_abs, sat, fdir_out=None, 
              sfc_alb=None, sza=None, aod_550=None,
              cld_manual=False, cot=None, cer=None, cth=None,):
    """
    Post-processing - combine the all the calculations into one dataset
    """

    print(date)
    print(tag)

    # ==================================================================================================
    oco = oco2_rad_nadir(sat)

    wvl_o2a  = np.zeros_like(oco.rad_o2_a      , dtype=np.float64)
    wvl_wco2 = np.zeros_like(oco.rad_co2_weak  , dtype=np.float64)
    wvl_sco2 = np.zeros_like(oco.rad_co2_strong, dtype=np.float64)
    for i in range(oco.rad_o2_a.shape[0]):
        for j in range(oco.rad_o2_a.shape[1]):
            wvl_o2a[i, j, :]  = oco.get_wvl_o2_a(j)
            wvl_wco2[i, j, :] = oco.get_wvl_co2_weak(j)
            wvl_sco2[i, j, :] = oco.get_wvl_co2_strong(j)
    # ==================================================================================================

    # ==================================================================================================
    with h5py.File(fname_abs, 'r') as f:
        wvls = f['lamx'][...]*1000.0
        wvls  = np.sort(wvls)
        trans = f['tx'][...][np.argsort(f['lamx'][...])]
        
    wvl_center = np.mean(wvls)

    modl1b = modis_l1b(fnames=sat.fnames['mod_02'], extent=sat.extent)
    lon_2d, lat_2d, rad_2d_mod = grid_by_dxdy(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][0, ...], extent=sat.extent, dx=250, dy=250, method='nearest')

    rad_mca_ipa0 = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1], wvls.size), dtype=np.float64)
    rad_mca_3d   = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1], wvls.size), dtype=np.float64)

    rad_mca_ipa0_std = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1], wvls.size), dtype=np.float64)
    rad_mca_3d_std   = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1], wvls.size), dtype=np.float64)

    # modified ==============================================
    # fname = glob.glob('%s/mca-out-rad-oco2-ipa0_%s*nm.h5' % (fdir_mca, ('%.4f' % wvls[0])[:-1]))[0]
    fname = glob.glob('%s/mca-out-rad-oco2-ipa0_%s*nm.h5' % (fdir_mca, ('%.4f' % wvl_center)[:-1]))[0]
    with h5py.File(fname, 'r') as f:
        rad_ipa0     = f['mean/rad'][...]

    rad_mca_ipa0_domain = np.zeros((rad_ipa0.shape[0], rad_ipa0.shape[1], wvls.size), dtype=np.float64)
    rad_mca_3d_domain   = np.zeros((rad_ipa0.shape[0], rad_ipa0.shape[1], wvls.size), dtype=np.float64)

    rad_mca_ipa0_domain_std = np.zeros((rad_ipa0.shape[0], rad_ipa0.shape[1], wvls.size), dtype=np.float64)
    rad_mca_3d_domain_std   = np.zeros((rad_ipa0.shape[0], rad_ipa0.shape[1], wvls.size), dtype=np.float64)
    # =======================================================

    toa = np.zeros(wvls.size, dtype=np.float64)
    Np = np.zeros(wvls.size, dtype=np.float64)

    for k in range(wvls.size):
        print(wvls[k])

        # fname = glob.glob('%s/mca-out-rad-oco2-ipa0_%s*nm.h5' % (fdir_mca, ('%.4f' % wvls[k])[:-1]))[0]
        fname = glob.glob('%s/mca-out-rad-oco2-ipa0_%s*nm.h5' % (fdir_mca, ('%.4f' % wvl_center)[:-1]))[0]
        with h5py.File(fname, 'r') as f:
            rad_ipa0     = f['mean/rad'][...]
            rad_ipa0_std = f['mean/rad_std'][...]

        # fname = glob.glob('%s/mca-out-rad-oco2-3d_%s*nm.h5' % (fdir_mca, ('%.4f' % wvls[k])[:-1]))[0]
        fname = glob.glob('%s/mca-out-rad-oco2-3d_%s*nm.h5' % (fdir_mca, ('%.4f' % wvl_center)[:-1]))[0]
        with h5py.File(fname, 'r') as f:
            rad_3d     = f['mean/rad'][...]
            rad_3d_std = f['mean/rad_std'][...]
            toa0       = f['mean/toa'][...]
            photons    = f['mean/N_photon'][...]
        toa[k] = toa0
        Np[k] = photons.sum()

        # ===================================
        rad_mca_ipa0_domain[:, :, k] = rad_ipa0.copy()
        rad_mca_3d_domain[:, :, k]   = rad_3d.copy()

        rad_mca_ipa0_domain_std[:, :, k] = rad_ipa0_std.copy()
        rad_mca_3d_domain_std[:, :, k]   = rad_3d_std.copy()
        # ===================================

        for i in range(wvl_o2a.shape[0]):
            for j in range(wvl_o2a.shape[1]):
                lon0 = oco.lon_l1b[i, j]
                lat0 = oco.lat_l1b[i, j]
                if tag == 'o2a':
                    lon0 = oco.lon_l1b_o2a[i, j]
                    lon0 = oco.lon_l1b_o2a[i, j]
                elif tag == 'wco2':
                    lon0 = oco.lon_l1b_wco2[i, j]
                    lon0 = oco.lon_l1b_wco2[i, j]
                elif tag == 'sco2':
                    lon0 = oco.lon_l1b_sco2[i, j]
                    lon0 = oco.lon_l1b_sco2[i, j]
                index_lon = np.argmin(np.abs(lon_2d[:, 0]-lon0))
                index_lat = np.argmin(np.abs(lat_2d[0, :]-lat0))

                rad_mca_ipa0[i, j, k] = rad_ipa0[index_lon, index_lat]
                rad_mca_3d[i, j, k]   = rad_3d[index_lon, index_lat]

                rad_mca_ipa0_std[i, j, k] = rad_ipa0_std[index_lon, index_lat]
                rad_mca_3d_std[i, j, k]   = rad_3d_std[index_lon, index_lat]
    # ==================================================================================================
    if aod_550 is None and cth is None:
        output_file = 'data_all_%s_%s_%4.4d_%4.4d_sfc_alb_%.3f_sza_%.1f.h5' % (date.strftime('%Y%m%d'), tag, oco.index_s, oco.index_e, sfc_alb, sza)
    elif aod_550 is not None and cth is None:
        output_file = 'data_all_%s_%s_%4.4d_%4.4d_sfc_alb_%.3f_sza_%.1f_aod550_%.3f.h5' % (date.strftime('%Y%m%d'), tag, oco.index_s, oco.index_e, sfc_alb, sza, aod_550)
    elif aod_550 is None and cth is not None:
        output_file = 'data_all_%s_%s_%4.4d_%4.4d_sfc_alb_%.3f_sza_%.1f_cot_%.1f_cer_%.0f_cth_%.0f.h5' % (date.strftime('%Y%m%d'), tag, oco.index_s, oco.index_e, sfc_alb, sza, cot, cer, cth)
    else:
        output_file = 'data_all_%s_%s_%4.4d_%4.4d_sfc_alb_%.3f_sza_%.1f_aod550_%.3f_cot_%.1f_cer_%.0f_cth_%.0f.h5' % (date.strftime('%Y%m%d'), tag, oco.index_s, oco.index_e, sfc_alb, sza, aod_550, cot, cer, cth)
    if fdir_out is not None:
        output_file = '/'.join([fdir_out, output_file])
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('lon',     data=oco.lon_l1b)
        f.create_dataset('lat',     data=oco.lat_l1b)
        f.create_dataset('logic',   data=oco.logic_l1b)
        f.create_dataset('toa',     data=toa)
        f.create_dataset('Np',      data=Np)
        f.create_dataset('sfc_alb', data=sfc_alb)
        f.create_dataset('sza_avg', data=sza)


        if tag == 'o2a':
            f.create_dataset('lon_fp',    data=oco.lon_l1b_o2a)
            f.create_dataset('lat_fp',    data=oco.lat_l1b_o2a)
            f.create_dataset('rad_oco',   data=oco.rad_o2_a)
            f.create_dataset('wvl_oco',   data=wvl_o2a)
        elif tag == 'wco2':
            f.create_dataset('lon_fp',    data=oco.lon_l1b_wco2)
            f.create_dataset('lat_fp',    data=oco.lat_l1b_wco2)
            f.create_dataset('rad_oco',   data=oco.rad_co2_weak)
            f.create_dataset('wvl_oco',   data=wvl_wco2)
        elif tag == 'sco2':
            f.create_dataset('lon_fp',    data=oco.lon_l1b_sco2)
            f.create_dataset('lat_fp',    data=oco.lat_l1b_sco2)
            f.create_dataset('rad_oco',   data=oco.rad_co2_strong)
            f.create_dataset('wvl_oco',   data=wvl_sco2)

        f.create_dataset('snd_id',   data=oco.snd_id)
        f.create_dataset('xco2',     data=oco.xco2)
        f.create_dataset('sfc_pres', data=oco.sfc_pres)
        f.create_dataset('sza',      data=oco.sza)
        f.create_dataset('saa',      data=oco.saa)

        logic = (oco.lon_l1b>=sat.extent[0]) & (oco.lon_l1b<=sat.extent[1]) & (oco.lat_l1b>=sat.extent[2]) & (oco.lat_l1b<=sat.extent[3])
        sza_mca = np.zeros_like(oco.sza)
        saa_mca = np.zeros_like(oco.saa)
        sza_mca[...] = oco.sza[logic].mean()
        saa_mca[...] = oco.saa[logic].mean()
        f.create_dataset('sza_mca', data=sza_mca)
        f.create_dataset('saa_mca', data=saa_mca)

        f.create_dataset('rad_mca_3d',   data=rad_mca_3d)
        f.create_dataset('rad_mca_ipa0', data=rad_mca_ipa0)
        f.create_dataset('rad_mca_3d_std',   data=rad_mca_3d_std)
        f.create_dataset('rad_mca_ipa0_std', data=rad_mca_ipa0_std)
        # ==============
        f.create_dataset('lon2d',                data=lon_2d)
        f.create_dataset('lat2d',                data=lat_2d)
        f.create_dataset('rad_mca_3d_domain',    data=rad_mca_3d_domain)
        f.create_dataset('rad_mca_ipa0_domain',  data=rad_mca_ipa0_domain)
        f.create_dataset('rad_mca_3d_domain_std',    data=rad_mca_3d_domain_std)
        f.create_dataset('rad_mca_ipa0_domain_std',  data=rad_mca_ipa0_domain_std)
        f.create_dataset('wvl_mca',             data=wvls)
        f.create_dataset('tra_mca',             data=trans)
        f.create_dataset('extent_domain',       data=sat.extent)
        f.create_dataset('extent_analysis',     data=sat.extent_analysis)

    return output_file
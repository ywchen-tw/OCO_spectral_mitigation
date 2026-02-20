import numpy as np
import sys
import h5py


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

import h5py
import sys
import os
import numpy as np

def read_oco_zpt(zpt_file='zpt.h5'):
    """
    Read the zpt.h5 file to get vertical structure information.
    """
    if not os.path.isfile(zpt_file):
        sys.exit("[Error] zpt output file does not exit!")

    with h5py.File(zpt_file, 'r') as h5_zpt:
        h_edge = h5_zpt['h_edge'][...]
        p_edge = h5_zpt['p_edge'][...]
        h_lay = h5_zpt['h_lay'][...]
        p_lay = h5_zpt['p_lay'][...]
        t_lay = h5_zpt['t_lay'][...]
        d_o2_lay = h5_zpt['d_o2_lay'][...]
        d_co2_lay = h5_zpt['d_co2_lay'][...]
        d_h2o_lay = h5_zpt['d_h2o_lay'][...]
        h2o_vmr = h5_zpt['h2o_vmr'][...]
        dz = h5_zpt['dz'][...]
        albedo_o2a = h5_zpt['albedo_o2a'][...]
        albedo_wco2 = h5_zpt['albedo_wco2'][...]
        albedo_sco2 = h5_zpt['albedo_sco2'][...]
        sza = h5_zpt['sza'][...]
        vza = h5_zpt['vza'][...]

        return h_edge, p_edge, p_lay, t_lay, d_o2_lay, d_co2_lay, d_h2o_lay,\
               h2o_vmr, dz, h_lay, albedo_o2a, albedo_wco2, albedo_sco2, sza, vza

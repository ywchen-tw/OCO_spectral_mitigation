import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

def oco_ils(iband, sat, footprint=1):

    # Get real ILS from l1b sounding data
    with h5py.File(sat.fnames['oco_l1b'][0], 'r') as f:
        del_lambda_mean = f['InstrumentHeader/ils_delta_lambda'][...][iband, footprint, :, :]
        rel_lresponse_mean = f['InstrumentHeader/ils_relative_response'][...][iband, footprint, :, :]


    xx = del_lambda_mean#*1000 # micron to nm
    norm = np.repeat((rel_lresponse_mean).max(axis=1), 200).reshape(1016, 200)
    yy = rel_lresponse_mean/norm

    return np.array(xx), np.array(yy)

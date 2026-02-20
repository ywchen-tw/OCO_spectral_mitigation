import sys
from numba import njit
import numpy as np
import bisect as bs

# def warn_h2o_out_of_range():
#     print('H2O mixing ratio out ouf range, using extrapolation!', file=sys.stderr)

@njit
def trilinear_interpolation(matrix, absco_array, Q_vec):

    coeff = np.dot(matrix, absco_array)
    absco = np.dot(Q_vec.T, coeff).flatten()

    return absco

def rdabsco_species_absco(absco_data, p_species, tk_species, broad_species,
                    hpa_species, tkobs,pobs,
                    T_ind, P_ind, h2o_vmr, iwcm1, iwcm2,
                    species,
                    trilinear_matrix,
                    mode="trilinear"):
    """
    mode: single    -> use the closest indices for P, T, H2O mixing ratio
          linear    -> use the closest indices for P, T and linear interpolation for H2O mixing ratio
          trilinear -> trilinear interpolation for P, T, H2O mixing ratio

    T_ind:  temperature index
    P_ind:  pressure index

    """

    absco_abs = absco_data["Gas_Absorption"]

    """
    Will read in every absco coefficient for the temperature index T_ind
    and pressure index P_ind for a range of wavenumber
    Gas_XX_Absorption shape: 64 x  17   x 3   x very large
                                p     temp   broad  wcm
    """

    if mode == 'single':
        jbroad = 0 if species == 'h2o' else 1
        absco = absco_abs[P_ind, T_ind, jbroad, iwcm1:iwcm2+1]

    elif mode == 'linear':
        # H2O mixing ratio linear interpoation
        hh = np.searchsorted(broad_species, h2o_vmr, side='left')
        if hh == 2:
            warn_h2o_out_of_range()
            hh = 1
        absco_h0 = absco_abs[P_ind,    T_ind,     hh,      iwcm1:iwcm2+1]
        absco_h1 = absco_abs[P_ind,    T_ind,     hh+1,    iwcm1:iwcm2+1]
        dH2O_vmr = (h2o_vmr-broad_species[hh])/(broad_species[hh+1]-broad_species[hh])
        absco = absco_h0+dH2O_vmr*absco_h1

    elif mode == 'trilinear':
        # P, T, H2O mixing ratio trilinear interpolation
        # print(f'p={pobs:.2f} hPa, [{p_species[P_ind]/100:.2f} hPa, {p_species[P_ind+1]/100:.2f} hPa]', file=sys.stderr)
        # print(f'T={tkobs:2f} K, [{tk_species[P_ind, T_ind]:2f} K, {tk_species[P_ind, T_ind+1]:2f} K]', file=sys.stderr)

        hh = bs.bisect_left(broad_species, h2o_vmr)-1
        # hh = np.searchsorted(broad_species, h2o_vmr, side='left')
        if hh == 2:
            warn_h2o_out_of_range()
            hh = 1
        # print(f'H2O vmr={h2o_vmr:.2e}, [{broad_species[hh]:.2e}, {broad_species[hh+1]:.2f}]', file=sys.stderr)

        dp = (pobs-hpa_species[P_ind])/(hpa_species[P_ind+1]-hpa_species[P_ind])
        dT = (tkobs-tk_species[P_ind, T_ind])/(tk_species[P_ind, T_ind+1]-tk_species[P_ind, T_ind])
        dH2O_vmr = (h2o_vmr-broad_species[hh])/(broad_species[hh+1]-broad_species[hh])
        # print(f'dp: {dp:.2f} hPa', file=sys.stderr)
        # print(f'dT: {dT:.2f} K', file=sys.stderr)
        # print(f'dH2O_vmr: {dH2O_vmr:.2e}', file=sys.stderr)
        # trilinear_matrix =  np.array([[ 1,  0,  0,  0,  0,  0,  0,  0],
        #                     [-1,  0,  0,  0,  1,  0,  0,  0],
        #                     [-1,  0,  1,  0,  0,  0,  0,  0],
        #                     [-1,  1,  0,  0,  0,  0,  0,  0],
        #                     [ 1,  0, -1,  0, -1,  0,  1,  0],
        #                     [ 1, -1, -1,  1,  0,  0,  0,  0],
        #                     [ 1, -1,  0,  0, -1,  1,  0,  0],
        #                     [-1,  1,  1, -1,  1, -1, -1,  1]], dtype=np.float64)

        # N = absco_abs[P_ind,    T_ind,     hh,     iwcm1:iwcm2+1].shape[0]
        # absco_array = np.empty((8, N), dtype=np.float64)
        # absco_array[0, :] = absco_abs[P_ind,    T_ind,     hh,     iwcm1:iwcm2+1] # absco_000
        # absco_array[1, :] = absco_abs[P_ind,    T_ind,     hh+1,   iwcm1:iwcm2+1] # absco_001
        # absco_array[2, :] = absco_abs[P_ind,    T_ind+1,   hh,     iwcm1:iwcm2+1] # absco_010
        # absco_array[3, :] = absco_abs[P_ind,    T_ind+1,   hh+1,   iwcm1:iwcm2+1] # absco_011
        # absco_array[4, :] = absco_abs[P_ind+1,  T_ind,     hh,     iwcm1:iwcm2+1] # absco_100
        # absco_array[5, :] = absco_abs[P_ind+1,  T_ind,     hh+1,   iwcm1:iwcm2+1] # absco_101
        # absco_array[6, :] = absco_abs[P_ind+1,  T_ind+1,   hh,     iwcm1:iwcm2+1] # absco_110
        # absco_array[7, :] = absco_abs[P_ind+1,  T_ind+1,   hh+1,   iwcm1:iwcm2+1] # absco_111

        absco_block = absco_abs[P_ind:P_ind+2, T_ind:T_ind+2, hh:hh+2, iwcm1:iwcm2+1]
        absco_array = absco_block.reshape(8, -1).astype(np.float64)

        Q_vec = np.array([1.0, dp, dT, dH2O_vmr, dp*dT, dT*dH2O_vmr, dH2O_vmr*dp, dp*dT*dH2O_vmr], dtype=np.float64).reshape(8, 1)
        absco = trilinear_interpolation(trilinear_matrix, absco_array, Q_vec)


    return absco

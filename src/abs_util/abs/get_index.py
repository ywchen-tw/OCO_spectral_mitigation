import sys
import numpy as np
import bisect as bs
from numba import njit

@njit(cache=True)
def get_P_index(pobs_arr, hpa, trilinear=True):
    """
    tkobs: temperatrue of the layer in K
    pobs: pressure of the layer in hPa

    # Output
    T_ind:  temperature index
    P_ind:  pressure index

    """

    # *********
    # hpa values range from small to large
    P_inds = np.searchsorted(hpa, pobs_arr, side='left')
    

    if trilinear:
        P_inds -= 1

    P_inds = np.clip(P_inds, 0, len(hpa)-2)
    
    

    return P_inds

@njit(cache=True)
def get_T_index(tkobs, tk, P_ind, trilinear=True):
    """
    tkobs: temperatrue of the layer in K
    P_ind:  pressure index

    # Output
    T_ind:  temperature index
    """


    # *********
    # temperature values range from small to large
    # note that tk is stored tk(temp index, pressure index)
    T_ind = np.searchsorted(tk[P_ind, :], tkobs, side='left')

    if trilinear:
        # get the left index for trilinear interpolation
        T_ind -= 1

    T_ind = max(0, min(T_ind, tk.shape[1]-2))

    if tkobs >= tk[P_ind, T_ind] and tkobs < tk[P_ind, T_ind+1]:
        pass
    else:
        print('[Warning!!!]', tkobs, tk[P_ind, T_ind], tk[P_ind, T_ind+1])

    return T_ind

def get_PT_index(tkobs, pobs, tk, hpa, trilinear=True):
    """
    tkobs: temperatrue of the layer in K
    pobs: pressure of the layer in hPa

    # Output
    T_ind:  temperature index
    P_ind:  pressure index

    """

    # *********
    # hpa values range from small to large
    P_ind = bs.bisect_left(hpa, pobs)

    if trilinear:
        P_ind -= 1

    # *********
    # temperature values range from small to large
    # note that tk is stored tk(temp index, pressure index)
    T_ind = bs.bisect_left(tk[P_ind, :], tkobs)


    if trilinear:
        # get the left index for trilinear interpolation
        T_ind -= 1

    if pobs >= hpa[P_ind] and pobs < hpa[P_ind+1]:
        None
    else:
        print('[Warning!!!]', pobs, [hpa[P_ind], hpa[P_ind+1]], file=sys.stderr)

    if tkobs >= tk[P_ind, T_ind] and tkobs < tk[P_ind, T_ind+1]:
        None
    else:
        print('[Warning!!!]', tkobs, [tk[P_ind, T_ind], tk[P_ind, T_ind+1], tk[P_ind, T_ind+2]], file=sys.stderr)
        print('tk', tk[P_ind, :], file=sys.stderr)

    return T_ind, P_ind

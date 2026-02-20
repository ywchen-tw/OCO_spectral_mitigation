
import sys
import numpy as np
# from bisect import bisect_right as bs

def find_boundary(nu1, nu2, nu):
    """
    nu1, nu2: wavenumber boundary for the band in cm-1
    nu: wavenumber series from the absco file
    """

    # inu1 = bs(nu, nu1) #-1
    # inu2 = bs(nu, nu2) #bs(nu, nu2)-1

    inu1, inu2 = np.searchsorted(nu, [nu1, nu2], side='left')

    if inu2 == len(nu):
        inu2 = inu2-1
    if nu[inu2]-nu[inu2-1] > 10:
        inu2 = inu2-1
    if inu1 == 0:
        print(f'[Warning] nu2={nu1} cm-1 is out of range (too low)', file=sys.stderr)
    nwav = inu2-inu1+1
    nudat = nu[inu1:inu2+1]
    wvldat = 1e4/nudat

    return inu1, inu2, nwav, nudat, wvldat

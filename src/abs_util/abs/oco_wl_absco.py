import sys
import numpy as np

def oco_wv_absco(wvl_coef, iband, footprint=1):
    """
    Sub-program calculates OCO wavelengths by L1b dispersion coefficients

    # output
    wl: wavelength in mircon
    """
    if iband < 0 or iband > 2:
        sys.exit('oco_wl: wrong band #')


    Nspec, Nfoot, Ncoef = wvl_coef.shape
    lam = np.zeros([8, 1016, 3])
    wli = np.arange(1, 1017, dtype=float)

    for i in range(Nfoot):
        for j in range(Nspec):
            for k in range(Ncoef):
                lam[i, :, j] = lam[i,:,j] + wvl_coef[j,i,k]*wli**k

    return lam[footprint, :, iband]

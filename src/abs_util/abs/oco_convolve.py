import sys
import numpy as np
from abs_util.abs.oco_wl import oco_wv      # reads OCO wavelengths
from abs_util.abs.oco_ils import oco_ils    # reads OCO line shape ("slit function")


def oco_conv(iband, sat, ils0, wavedat, nwav, trns, fp=1):

    # *********
    # Get OCO wavelengths & slit function
    # ---------------------------------------------
    #       sampling interval (nm)      FWHM (nm)
    # O2A:      0.015                       0.04
    # WCO2:     0.031                       0.08
    # SCO2:     0.04                        0.10
    # ---------------------------------------------
    # (1) read wavelengths
    wloco = oco_wv(iband, sat, footprint=fp) # (micron)
    nlo = len(wloco)
    # (2) read instrument line shape
    xx, yy = oco_ils(iband, sat, footprint=fp) # xx: relative wl shift (micron)# yy: normalized ILS

    # (3) convolute tau & trns across entire wavelength range -- & how about kval

    trnsc = np.empty(nlo)
    trnsc0 = np.empty(nlo)
    indlr = np.empty((nlo, 3), dtype=int) # left & right index for cropped ILS (in ABSCO gridding) + total #

    for l in range(nlo):
        ils = yy[l, :]/np.max(yy[l, :]) >= ils0
        nils0 = ils.nonzero()[0]
        # get wl range within absco that falls within the ILS (total range) & within pre-set threshold
        # --- left and right full ranges ---
        abswlL, abswlR = wloco[l] + np.min(xx[l, :]), wloco[l] + np.max(xx[l, :])
        # --- left and right "valid" ranges (ILS above threshold ils0) ---
        abswlL0, abswlR0 = wloco[l] + np.min(xx[l, ils]), wloco[l] + np.max(xx[l, ils])
        il, ir = np.argmin(np.abs(wavedat-abswlL)),  np.argmin(np.abs(wavedat-abswlR))
        il0, ir0 = np.argmin(np.abs(wavedat-abswlL0)), np.argmin(np.abs(wavedat-abswlR0))
        indlr[l,0] = il0        # left index
        indlr[l,1] = ir0        # right index
        indlr[l,2] = il0-ir0+1  # how many
        if ir0 == 0:
            print('[Warning] Range exceeded (R)')
        if il0 == nwav-1:
            print('[Warning] Range exceeded (L)')
        if ir  >= il :
            print('[Warning] Something wrong with range/ILS')
        if ir0 >= il0:
            print('[Warning] Something wrong with range/ILS0')
        # actual convolution ---
        ilg = np.interp(wavedat[ir:il], xx[l, :]+wloco[l], yy[l, :])                # full slit function in absco gridding
        ilg0 = np.interp(wavedat[ir0:il0], xx[l, ils]+wloco[l], yy[l, ils])   # partial slit function within valid range
        trnsc[l] = np.sum(trns[ir:il]*ilg)/np.sum(ilg)                  # ir:il because it is descending in wl
        trnsc0[l] = np.sum(trns[ir0:il0]*ilg0)/np.sum(ilg0)

    return trnsc, trnsc0, indlr, ils

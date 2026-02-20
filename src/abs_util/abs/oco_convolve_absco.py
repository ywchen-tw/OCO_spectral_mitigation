import sys
import numpy as np


def oco_conv_absco(wvloco, xx, yy, wvldat, nwav):

    # *********
    # Get OCO wavelengths & slit function
    # ---------------------------------------------
    #       sampling interval (nm)      FWHM (nm)
    # O2A:      0.015                       0.04
    # WCO2:     0.031                       0.08
    # SCO2:     0.04                        0.10
    # ---------------------------------------------

    nlo = len(wvloco)
    indlr = np.empty((nlo, 3), dtype=int) # left & right index (in ABSCO gridding) + total #

    abswlL_all = wvloco + np.min(xx, axis=1)  # left edge of ILS in absco gridding
    abswlR_all = wvloco + np.max(xx, axis=1)  # right edge of ILS in absco gridding
    # for l in range(nlo):
    #     # get wl range within absco that falls within the ILS (total range) & within pre-set threshold
    #     # --- left and right full ranges ---
    #     abswlL = abswlL_all[l]
    #     abswlR = abswlR_all[l]
    #     il, ir = np.argmin(np.abs(wvldat-abswlL)),  np.argmin(np.abs(wvldat-abswlR))
    #     indlr[l,0] = il       # left index
    #     indlr[l,1] = ir        # right index
    #     indlr[l,2] = il-ir+1  # how many
    #     if ir == 0:
    #         print('[Warning] Range exceeded (R)')
    #     if il == nwav-1:
    #         print('[Warning] Range exceeded (L)')
    #     if ir  >= il :
    #         print('[Warning] Something wrong with range/ILS')

    wvldat_reversed = wvldat[::-1]
    il_all = np.searchsorted(wvldat_reversed, abswlL_all, side='left')-1
    ir_all = np.searchsorted(wvldat_reversed, abswlR_all, side='right') 
    
    il_all = np.clip(il_all, 0, nwav-1)
    ir_all = np.clip(ir_all, 0, nwav-1)
    
    # reverse back to original order
    il_all = nwav - 1 - il_all
    ir_all = nwav - 1 - ir_all

    indlr[:, 0] = il_all
    indlr[:, 1] = ir_all
    indlr[:, 2] = il_all - ir_all + 1

    # # Optional: batch warnings
    # if np.any(ir_all == 0):
    #     print('[Warning] Range exceeded (R)')
    # if np.any(il_all == nwav-1):
    #     print('[Warning] Range exceeded (L)')
    # if np.any(ir_all >= il_all):
    #     print('[Warning] Something wrong with range/ILS')


    return indlr

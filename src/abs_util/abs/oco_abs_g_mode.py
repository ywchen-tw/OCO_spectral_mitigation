import sys
import numpy as np
import h5py
from abs_util.abs.oco_wl import oco_wv      # reads OCO wavelengths
from abs_util.abs.oco_ils import oco_ils    # reads OCO line shape ("slit function")    


def oco_wv_select(trnsx, Trn_min, refl, nlay, nx, all_r, 
                  wlc, wlf, wls, wl, indlr, xx, yy, ext, fsol, iband,
                  g_mode, g=16):
    # ***** do spectral sub-sample using equi-distant transmittance values *****
    # maximum transmittance (T) within spectral sub-range
    mx = np.max(trnsx)            
    # minimum transmittance with spectral sub-range
    mn = np.max([Trn_min*np.max(trnsx), np.min(trnsx)])    
    m0 = (mx-mn)/np.float64(nx)  # T increments
    ods = np.empty(nx+1) # T sorted (nx sub-samples)
    wli = np.empty(nx+1, dtype=int) # wl index

    abs_g_final = np.zeros((nlay, nx+1, g))
    prob_g_final = np.zeros((nlay, nx+1, g))
    weight_g_final = np.zeros((nlay, nx+1, g))
    sol_g_final = np.zeros((nx+1, g))

    for i in range(nx+1):
        ods[i] = (m0*np.float64(i)+mn)*refl
        wli0 = np.argmin(np.abs(ods[i]-trnsx*refl))

        if g_mode:
            g_test = False
            print(ods[i], wli0)
            while not g_test:
                if all_r > 0 :
                    l0 = np.argmin(np.abs(wlc-wlf[np.int(wli0)]))
                else:
                    l0 = np.argmin(np.abs(wlc-wls[np.int(wli0)]))

                absgx_tmp = wl[indlr[l0,1]:indlr[l0,0]+1] # ILS - xx (lamda)
                ilg0 = np.interp(wl[indlr[l0,1]:indlr[l0,0]+1], xx[l0, :]+wlc[l0], yy[l0, :]) # partial slit function within valid range
                absgy_tmp = ilg0 # ILS - yy ("weight")
                absgn_tmp = indlr[l0,2]
                absgl_tmp = np.zeros((nlay, absgn_tmp))
                abs_g = np.zeros((nlay, g))
                prob_g = np.zeros((nlay, g))
                weight_g = np.zeros((nlay, g))
                for z in range(0, nlay):
                    absgl_tmp[z, :] = ext[indlr[l0,1]:indlr[l0,0]+1,z]
                    sort_ind = np.argsort(absgl_tmp[z, :])
                    x_sort = np.arange(len(sort_ind))+1
                    prob = x_sort/np.max(x_sort)

                    prob_select = np.linspace(prob[0], prob[-1], g+1)

                    y = (absgl_tmp[z, :])*absgy_tmp[:]
                    sort_w_weight_ind = np.argsort(y)
                    sorted_y = y[sort_w_weight_ind]
                    for j in range(g):
                        start_ind = np.argmin(np.abs(prob-prob_select[j]))
                        end_ind = np.argmin(np.abs(prob-prob_select[j+1]))
                        prob_g[z, j] = np.mean([prob[start_ind:end_ind+1]])
                        abs_g[z, j] = np.mean([sorted_y[start_ind:end_ind+1]])
                        weight_g[z, j] = prob_select[j+1] - prob_select[j]
                ori_abs = np.nansum(absgl_tmp[:, :]*absgy_tmp[:], axis=1)
                g_abs = (np.sum(abs_g*weight_g, axis=1)/np.sum(weight_g, axis=1))*absgn_tmp

                ratio_mean, ratio_std = np.mean(g_abs/ori_abs), np.std(g_abs/ori_abs)
                threshold = 0.01 if iband != 1 else 0.06
                print(f'{iband} g mode result:')
                if np.abs(1-ratio_mean) <= threshold:
                    g_test = True
                    
                    print(f'index {i} with wli0 {wli0} g_abs/ori_abs: {(ratio_mean):.4f}')
                    print('Succes!')
                else:
                    if iband == 1:
                        print(f'index {i} with wli0 {wli0} g_abs/ori_abs: {(ratio_mean):.4f}')
                        print('Test next...')
                    if i != nx:
                        wli0 += 1
                    else:
                        wli0 -= 1
            
            abs_g_final[:, i, :] = abs_g
            prob_g_final[:, i, :] = prob_g
            weight_g_final[:, i, :] = weight_g

            

            solx_tmp = fsol[indlr[l0,1]:indlr[l0,0]+1]
            sort_sol_ind = np.argsort(solx_tmp)
            sol_indx_sort = np.arange(len(sort_sol_ind))+1
            prob_sol = sol_indx_sort/np.max(sol_indx_sort)
            prob_sol_select = np.linspace(prob_sol[0], prob_sol[-1], g+1)
            y_sol = (solx_tmp)*absgy_tmp
            sort_sol_weight_ind = np.argsort(y_sol)
            sorted_y_sol = y_sol[sort_sol_weight_ind]
            for j in range(g):
                start_ind = np.argmin(np.abs(prob_sol-prob_sol_select[i]))
                end_ind = np.argmin(np.abs(prob_sol-prob_sol_select[i+1]))
                sol_g_final[i, j] = np.mean([sorted_y_sol[start_ind:end_ind+1]])

        
        print(i, wli0, ods[i], trnsx[wli0]*refl)
        wli[i] = wli0
        
    weight_g_final = np.mean(weight_g_final, axis=0)
    return wli, abs_g_final, prob_g_final, weight_g_final, sol_g_final
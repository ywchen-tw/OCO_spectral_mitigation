import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def solar(file, val=None):
    """
    # to solve: kurudz.dat and val?
    """
    h = 6.62607004e-34
    c = 299792458.
    hc = h*c
    if val != None > 0:
        v=1
    else:
        v=0

    with open(file, 'r') as fp:
        for count, line in enumerate(fp):
            if v == 1 and count+1 < 7:
                print(line)
            else:
                pass

    # the setting is specific to the assigned solar.txt filw
    data = pd.read_csv(file, skiprows=6, header=None, sep='     ', engine='python')
    data['Wavenumber'] = data[0].astype(float)
    data['Irradiance'] = data[1].astype(float)
    wn = np.array(data['Wavenumber'])
    ss = np.array(data['Irradiance']) # photons/sec/m2/micron
    sx = 1e-1*hc*wn*ss # W/m2/nm
    wl = 1.0e4/wn # convert wavenumer to wavelength in micron


    """
    if v == 1:
        kur='/Users/schmidt/data/arise/cal/kurudz.dat'
        n=file_lines(kur)-6
        openr,us,kur,/get_lun
        wk=dblarr(n)
        sk=dblarr(n)
        for i in range(n):
            readf,us,w0,s0
            wk[i]=w0*0.001
            sk[i]=s0*0.001
            endfor
            plot,wl,sx
            oplot,wk,sk,color=120
        sys.exit()
    """
    #print(sx.min(), sx.max(), file=sys.stderr)
    

    return wl[::-1], sx[::-1]

import os
import time
import numpy as np
from functools import wraps
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import matplotlib.font_manager as _fm
if any(f.name == "Arial" for f in _fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "Arial"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"

def path_dir(path_dir):
    """
    Description:
        Create a directory if it does not exist.
    Return:
        path_dir: path of the directory
    """
    abs_path = os.path.abspath(path_dir)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    return abs_path

class sat_tmp:

    def __init__(self, data):

        self.data = data

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r took: %.4f min (%.4f h)' % \
          (f.__name__, (te-ts)/60, (te-ts)/3600))
        return result
    return wrap

def plot_mca_simulation(sat, modl1b, out0, oco_std0, oco_l1b0,
                         solver, fdir, cth, scale_factor, wavelength):
    if out0.data['rad']['data'].ndim==2:
        plot_rad = out0.data['rad']['data'][:,:] 
    else:
        print("out0.data['rad']['data'] shape:", out0.data['rad']['data'].shape)
        plot_rad = out0.data['rad']['data'][:,:,0]
    
    mod_img = mpl_img.imread(sat.fnames['mod_rgb'][0])
    mod_img_wesn = sat.extent
    fig = plt.figure(figsize=(18, 5.5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(mod_img, extent=mod_img_wesn)
    ax2.imshow(mod_img, extent=mod_img_wesn)
    ax3.imshow(mod_img, extent=mod_img_wesn)
    c1 = ax1.pcolormesh(modl1b.data['lon_2d']['data'], modl1b.data['lat_2d']['data'], 
                   modl1b.data['rad_2d']['data'], 
                   cmap='Greys_r', vmin=0.0, vmax=0.3, zorder=0)
    scatter_arg = {'s':20, 'c':oco_std0.data['xco2']['data'], 'cmap':'jet',
                   'alpha':0.4, 'zorder':1,}
    ax1.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], 
                **scatter_arg)
    ax1.set_title('MODIS Chanel 1')
    ax2.pcolormesh(modl1b.data['lon_2d']['data'], modl1b.data['lat_2d']['data'], 
                   plot_rad,
                   cmap='Greys_r', zorder=0, 
                #    vmin=0.0, vmax=0.3
                   )
    c2 = ax2.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], **scatter_arg)
    ax2.set_title('MCARaTS %s at %.4f nm' % (solver, wavelength))
    
    l1b_wvls = oco_l1b0.get_wvl_o2_a(0)
    wvl_idx = (abs(l1b_wvls - wavelength)).argmin()
    oco2_l1b_rad = oco_l1b0.rad_o2_a[:, :, wvl_idx].flatten()
    lon_l1b = oco_l1b0.lon_l1b.flatten()
    lat_l1b = oco_l1b0.lat_l1b.flatten()
    
    oco2_l1b_rad_mask = (oco2_l1b_rad > 0.0) & np.isfinite(oco2_l1b_rad)
    lon_l1b = lon_l1b[oco2_l1b_rad_mask]
    lat_l1b = lat_l1b[oco2_l1b_rad_mask]
    oco2_l1b_rad = oco2_l1b_rad[oco2_l1b_rad_mask]
    
    
    print("oco2_l1b_rad shape:", oco2_l1b_rad.shape)
    print("lon_l1b shape:", lon_l1b.shape)
    print("lat_l1b shape:", lat_l1b.shape)
    
    scatter_arg_rad = {'s':20, 'c':oco2_l1b_rad, 'cmap':'jet',
                   'alpha':0.4, 'zorder':1,
                   'vmin':0.0, 'vmax':np.nanmax(oco2_l1b_rad)*1.1}
    
    c3 = ax3.pcolormesh(modl1b.data['lon_2d']['data'], modl1b.data['lat_2d']['data'], 
                   plot_rad, 
                   cmap='jet', zorder=0, 
                   vmin=scatter_arg_rad['vmin'],
                   vmax=scatter_arg_rad['vmax'],
                   )
    ax3.scatter(lon_l1b, lat_l1b, **scatter_arg_rad)
    ax3.set_title('MCARaTS %s at %.4f nm' % (solver, wavelength))
    
    cb1 = fig.colorbar(c1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    cb1.set_label('MODIS Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')
    cb2 = fig.colorbar(c2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
    cb2.set_label('XCO2 (ppm)')
    cb3 = fig.colorbar(c3, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
    cb3.set_label('MCARaTS %s Radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)' % solver)
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Longitude ($^\circ$)')
        ax.set_ylabel('Latitude ($^\circ$)')
        ax.set_xlim(sat.extent[:2])
        ax.set_ylim(sat.extent[2:])
    plt.subplots_adjust(hspace=0.5)
    if cth is not None:
        plt.savefig('%s/mca-out-rad-modis-%s_cth-%.2fkm_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), cth, scale_factor, wavelength), bbox_inches='tight')
    else:
        plt.savefig('%s/mca-out-rad-modis-%s_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), scale_factor, wavelength), bbox_inches='tight')
    plt.close(fig)
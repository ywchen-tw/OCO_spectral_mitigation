import os
import sys
import h5py
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams
from er3t.util.modis import modis_35_l2
from er3t.util.oco2 import oco2_rad_nadir


import matplotlib.image as mpl_img
from mpl_toolkits.axes_grid1 import make_axes_locatable


def cdata_sat_raw(sat0, overwrite=False, plot=True):
    """
    Purpose: Collect satellite data for OCO-2 retrieval
    oco_band: 'o2a', 'wco2', 'sco2'
    """

    # Check if preprocessed data exists and return if overwrite is False
    if os.path.isfile(f'{sat0.fdir_pre_data}/pre-data.h5') and not overwrite:
        print(f'Message [pre_data]: {sat0.fdir_pre_data}/pre-data.h5 exsit.')
        return None
    else:
        # Open the HDF file and create MODIS data groups
        f0 = h5py.File(f'{sat0.fdir_pre_data}/pre-data.h5', 'w')
        f0['extent'] = sat0.extent

        # MODIS data groups in the HDF file
        #/--------------------------------------------------------------\#
        g  = f0.create_group('mod')
        g2 = g.create_group('cld')
        #\--------------------------------------------------------------/#

        # Process MODIS RGB imagery
        mod_rgb = mpl_img.imread(sat0.fnames['mod_rgb'][0])
        g['rgb'] = mod_rgb
        print('Message [cdata_sat_raw]: the processing of MODIS RGB imagery is complete.')


        # cloud
        #/--------------------------------------------------------------\#
        # cloud mask
        #/--------------------------------------------------------------\#
        mod35 = modis_35_l2(fnames=sat0.fnames['mod_35'], extent=sat0.extent)
        lon0, lat0, fov_qa_cat = [mod35.data[var]['data'] for var in ['lon', 'lat', 'fov_qa_cat']]
        fov_qa_cloudy_mask = fov_qa_cat == 0
        fov_qa_uncertain_clear_mask = fov_qa_cat == 1
        fov_qa_probably_clear_mask = fov_qa_cat == 2
        fov_qa_confident_clear_mask = fov_qa_cat == 3
        final_cloudy_mask = np.logical_or(fov_qa_cloudy_mask, fov_qa_uncertain_clear_mask)
        


        g2.update({'lon_1km': lon0, 'lat_1km': lat0, 
                   'fov_qa_cat': fov_qa_cat,
                   'final_cloudy_mask': final_cloudy_mask,
                   })

        print('Message [cdata_sat_raw]: the processing of MODIS cloud properties is complete.')
        #\--------------------------------------------------------------/#



        # OCO-2 data groups in the HDF file
        #/--------------------------------------------------------------\#
        gg = f0.create_group('oco')
        gg11 = gg.create_group('o2a')
        gg12 = gg.create_group('wco2')
        gg13 = gg.create_group('sco2')
        gg2 = gg.create_group('geo')
        #\--------------------------------------------------------------/#

        # Read OCO-2 radiance and wavelength data
        #/--------------------------------------------------------------\#
        oco = oco2_rad_nadir(sat0)

        wvl_o2a  = np.zeros_like(oco.rad_o2_a, dtype=np.float64)
        wvl_wco2  = np.zeros_like(oco.rad_o2_a, dtype=np.float64)
        wvl_sco2  = np.zeros_like(oco.rad_o2_a, dtype=np.float64)
        for i in range(oco.rad_o2_a.shape[0]):
            for j in range(oco.rad_o2_a.shape[1]):
                wvl_o2a[i, j, :]  = oco.get_wvl_o2_a(j)
                wvl_wco2[i, j, :] = oco.get_wvl_co2_weak(j)
                wvl_sco2[i, j, :] = oco.get_wvl_co2_strong(j)
        #\--------------------------------------------------------------/#

        # OCO L1B
        #/--------------------------------------------------------------\#
        gg.update({'lon': oco.lon_l1b, 'lat': oco.lat_l1b, 'logic': oco.logic_l1b, 'snd_id': oco.snd_id})
        gg11.update({'rad': oco.rad_o2_a, 'wvl': wvl_o2a})
        gg12.update({'rad': oco.rad_co2_weak, 'wvl': wvl_wco2})
        gg13.update({'rad': oco.rad_co2_strong, 'wvl': wvl_sco2})
        gg2.update({'sza': oco.sza, 'saa': oco.saa, 'vza': oco.vza, 'vaa': oco.vaa})
        print('Message [cdata_sat_raw]: the processing of OCO-2 radiance is complete.')
        #\--------------------------------------------------------------/#



        # print('Message [cdata_sat_raw]: the processing of OCO-2 surface reflectance is complete.')
        #\--------------------------------------------------------------/#
        f0.close()
    #/----------------------------------------------------------------------------\#

    if plot:
        with h5py.File(f'{sat0.fdir_pre_data}/pre-data.h5', 'r') as f0:
            extent = f0['extent'][...]
            rgb = f0['mod/rgb'][...]
            lon_1km = f0['mod/cld/lon_1km'][...]
            lat_1km = f0['mod/cld/lat_1km'][...]
            cloudy_mask = f0['mod/cld/final_cloudy_mask'][...]
            fov_qa_cat = f0['mod/cld/fov_qa_cat'][...]


        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        rcParams['font.size'] = 12
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        fig.suptitle('MODIS Products Preview')

        titles = ['RGB Imagery', 'MODIS35 Cloudy Mask', 'MODIS35 FOV QA Category']

        for idx, (ax, title, ) in enumerate(zip(np.ravel(axes), titles, )):
            cs = ax.imshow(rgb, zorder=0, extent=extent)
            ax.set_title(title)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_xlabel('Longitude [°]')
            ax.set_ylabel('Latitude [°]')
            
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        
        axes[1].scatter(lon_1km[cloudy_mask], lat_1km[cloudy_mask], c='r', s=5, zorder=1)
        divider2 = make_axes_locatable(axes[1])
        cax2 = divider2.append_axes('right', '5%', pad='3%')
        cax2.axis('off')
        
        axes[2].scatter(lon_1km, lat_1km, c=fov_qa_cat, cmap='jet', s=5, zorder=1)
        divider3 = make_axes_locatable(axes[2])
        cax3 = divider3.append_axes('right', '5%', pad='3%')
        cbar3 = fig.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=3)), cax=cax3, ticks=[0, 1, 2, 3])
        cbar3.set_ticklabels(['Cloudy', 'Uncertain Clear', 'Probably Clear', 'Confident Clear'])
        
        # save figure
        #/--------------------------------------------------------------\#
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s/<%s>.png' % (sat0.fdir_pre_data, _metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#



from genericpath import isfile
import os
import sys
import h5py
import numpy as np
import datetime
from scipy import interpolate
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import platform
import matplotlib.pyplot as plt
from matplotlib import rcParams

from er3t.util import cal_geodesic_lonlat
from er3t.util import move_correlate
from er3t.util import find_nearest
from er3t.rtm.mca_mwvl import func_ref_vs_cot
# from er3t.rtm.mca import func_ref_vs_cot

from abs_util.oco_atm_atmmod import atm_atmmod
from abs_util.oco_util import path_dir

from mpl_toolkits.axes_grid1 import make_axes_locatable


def para_corr(lon0, lat0, vza, vaa, cld_h, sfc_h, R_earth=6378000.0, verbose=True):
    """
    Parallax correction for the cloud positions

    lon0: input longitude
    lat0: input latitude
    vza : sensor zenith angle [degree]
    vaa : sensor azimuth angle [degree]
    cld_h: cloud height [meter]
    sfc_h: surface height [meter]
    R_earth: earth radius [meter]
    """

    if verbose:
        print('Message [para_corr]: Please make sure the units of \'cld_h\' and \'sfc_h\' are in \'meter\'.')

    dist = (cld_h-sfc_h)*np.tan(np.deg2rad(vza))

    # delta_lon = dist*np.sin(np.deg2rad(vaa)) / (np.pi*R_earth) * 180.0
    # delta_lat = dist*np.cos(np.deg2rad(vaa)) / (np.pi*R_earth) * 180.0

    # lon = lon0 + delta_lon
    # lat = lat0 + delta_lat

    lon, lat = cal_geodesic_lonlat(lon0, lat0, dist, vaa)

    return lon, lat



def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def cloud_mask_rgb(rgb, extent, lon_2d, lat_2d,
                   ref_470_2d, ref_threshold,
                   frac=0.5, a_r=1.06, a_g=1.06, a_b=1.06,
                   logic_good=None):

    # Find cloudy pixels based on MODIS RGB imagery and upscale/downscale to 250m resolution
    #/----------------------------------------------------------------------------\#
    lon_rgb0 = np.linspace(extent[0], extent[1], rgb.shape[1]+1)
    lat_rgb0 = np.linspace(extent[2], extent[3], rgb.shape[0]+1)
    lon_rgb = (lon_rgb0[1:]+lon_rgb0[:-1])/2.0
    lat_rgb = (lat_rgb0[1:]+lat_rgb0[:-1])/2.0

    _r = rgb[:, :, 0]
    _g = rgb[:, :, 1]
    _b = rgb[:, :, 2]

    logic_rgb_nan0 = (_r<=(np.quantile(_r, frac)*a_r)) |\
                     (_g<=(np.quantile(_g, frac)*a_g)) |\
                     (_b<=(np.quantile(_b, frac)*a_b))
    logic_rgb_nan = np.flipud(logic_rgb_nan0).T

    if logic_good is not None:
        logic_rgb_nan[logic_good] = False

    x0_rgb = lon_rgb[0]
    y0_rgb = lat_rgb[0]
    dx_rgb = lon_rgb[1] - x0_rgb
    dy_rgb = lat_rgb[1] - y0_rgb

    indices_x = np.int16(np.round((lon_2d-x0_rgb)/dx_rgb, decimals=0))
    indices_y = np.int16(np.round((lat_2d-y0_rgb)/dy_rgb, decimals=0))

    # print("indices_x min, max:", indices_x.min(), indices_x.max())
    # print("indices_y min, max:", indices_y.min(), indices_y.max())
    # print("lon_rgb min/max:", lon_rgb.min(), lon_rgb.max())
    # print("lat_rgb min/max:", lat_rgb.min(), lat_rgb.max())
    # print("lon_2d min/max:", lon_2d.min(), lon_2d.max())
    # print("lat_2d min/max:", lat_2d.min(), lat_2d.max())
    # print("indices_x shape:", indices_x.shape)
    # print("lon_2d shape:", lon_2d.shape)

    logic_ref470_nan0 = (ref_470_2d>=(np.quantile(ref_470_2d, frac)/a_b))
    logic_ref470_nan = logic_ref470_nan0

    print("ref_470_2d shape:", ref_470_2d.shape)
    print("logic_ref470_nan shape:", logic_ref470_nan.shape)

    counts, bins = np.histogram(ref_470_2d.flatten(), bins=51, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    # Perform the fit

    popt, _ = curve_fit(gaussian, centers, counts)
    a0, x0, sigma0 = popt
    cld_sigma0 = 7
    plt.close('all')
    plt.hist(ref_470_2d, bins=51, density=True, alpha=0.6)
    plt.plot(centers, gaussian(centers, *popt), 'r-', label='Fit')
    plt.vlines(x0+cld_sigma0*sigma0, 0, 20, 'b')
    plt.show()


    logic_ref470_nan0 = (ref_470_2d>=x0+cld_sigma0*sigma0)
    logic_ref470_nan = logic_ref470_nan0



    indices_x, indices_y = np.where(logic_ref470_nan)

    #\----------------------------------------------------------------------------/#

    return indices_x, indices_y





def cdata_cld_ipa(sat0, cfg_info, plot=True):

    # read in data
    #/----------------------------------------------------------------------------\#
    with h5py.File(f'{sat0.fdir_pre_data}/pre-data.h5', 'r') as f0:
        extent, ref_2d, ref_470_2d, rgb, cot_l2, cer_l2, lon_2d, lat_2d, cth, sfh, sza, saa, vza, vaa, u_10m, v_10m, delta_t = \
            [np.array(f0[k][...]) for k in ['extent', 'mod/rad/ref_650', 'mod/rad/ref_470', 'mod/rgb', 'mod/cld/cot_l2', 'mod/cld/cer_l2',
                                            'lon', 'lat', 'mod/cld/cth_l2', 'mod/geo/sfh', 'mod/geo/sza', 'mod/geo/saa', 'mod/geo/vza',
                                            'mod/geo/vaa',
                                            'oco/met/u_10m', 'oco/met/v_10m', 'oco/met/delta_t']]
    ref_threshold = float(cfg_info['ref_threshold'])
    photons=float(cfg_info['cot_Nphotons'])
    #\----------------------------------------------------------------------------/#


    # cloud mask method based on rgb image and l2 data
    #/----------------------------------------------------------------------------\#
    # primary selection (over-selection of cloudy pixels is expected)
    #/--------------------------------------------------------------\#

    # plt.close('all')
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # ax.hist(ref_470_2d.flatten(), bins=101)
    # ax.set_xlabel('Ref 470')
    # plt.show()
    # sys.exit()

    # cld_frac0 = (np.logical_not(np.isnan(cot_l2)) & (cot_l2>0.0)).sum() / cot_l2.size
    # frac0     = 1.0 - cld_frac0
    # scale_factor = 1.08
    # indices_x0, indices_y0 = cloud_mask_rgb(rgb, extent, lon_2d, lat_2d, ref_2d,
    #                                         ref_threshold, frac=frac0, a_r=scale_factor,
    #                                         a_g=scale_factor, a_b=scale_factor)
    # print("indices_x0 min, max:", indices_x0.min(), indices_x0.max())
    # print("indices_y0 min, max:", indices_y0.min(), indices_y0.max())
    
    logic_cth_nan = (cth>0)
    indices_x, indices_y = np.where(logic_cth_nan)
    

    lon_cld = lon_2d[indices_x, indices_y]
    lat_cld = lat_2d[indices_x, indices_y]

    indices = np.column_stack((indices_x, indices_y))

    






    # write cot_ipa into file
    #/----------------------------------------------------------------------------\#
    with h5py.File(f'{sat0.fdir_pre_data}/pre-data.h5', 'r+') as f0:
        # Update or create groups with try-except blocks
        group_data = {'mod/cld/cot0': cot_l2,
                      'mod/cld/cer0': cer_l2,
                      'mod/cld/cth0': cth,
                      'mod/cld/logic_cld0': indices>0,
                      'mod/cld/lon0': lon_cld,
                      'mod/cld/lat0':lat_cld,
                      }

        for group_name, group_data in group_data.items():
            if f0.get(group_name):
                del f0[group_name]
            f0[group_name] = group_data


        if f0.get('cld_msk'):
            del f0['cld_msk']
        g0 = f0.create_group('cld_msk')
        g0.update({
            f'indices_{suffix}': indices for suffix, indices in [
                ('x0', indices_x),
                ('y0', indices_y),
            ]
        })

        if f0.get('cld'):
            del f0['cld']
        g0 = f0.create_group('cld')
        g0.update({
            'lon': lon_cld,
            'lat': lat_cld,
        })
    #\----------------------------------------------------------------------------/#

    if plot:
        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        rcParams['font.size'] = 12
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('MODIS Cloud Re-Processing')

        # RGB
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(431)
        cs = ax1.imshow(rgb, zorder=0, extent=extent)
        ax1.set_title('RGB Imagery')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\--------------------------------------------------------------/#

        # L1B reflectance
        #/----------------------------------------------------------------------------\#
        ax2 = fig.add_subplot(432)
        cs = ax2.imshow(ref_2d.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=1.0)
        ax2.set_title('L1B Reflectance (650 nm)')

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cloud mask (primary)
        #/----------------------------------------------------------------------------\#
        ax4 = fig.add_subplot(434)
        cs = ax4.imshow(rgb, zorder=0, extent=extent)
        ax4.scatter(lon_2d[indices_x, indices_y], lat_2d[indices_x, indices_y], s=0.1, c='r', alpha=0.1)
        ax4.set_title('Primary Cloud Mask')

        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\----------------------------------------------------------------------------/#


        # 470 reflectance (MODIS)
        #/----------------------------------------------------------------------------\#
        ax5 = fig.add_subplot(4, 3, 5)
        cs = ax5.imshow(ref_470_2d.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.4)
        ax5.set_title('MODIS 470nm (filled and scaled)')

        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#


        # cot l2
        #/----------------------------------------------------------------------------\#
        ax7 = fig.add_subplot(437)
        cs = ax7.imshow(cot_l2.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax7.set_title('L2 COT')

        divider = make_axes_locatable(ax7)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer l2
        #/----------------------------------------------------------------------------\#
        ax8 = fig.add_subplot(438)
        cs = ax8.imshow(cer_l2.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax8.set_title('L2 CER [$\mu m$]')

        divider = make_axes_locatable(ax8)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth l2
        #/----------------------------------------------------------------------------\#
        ax9 = fig.add_subplot(439)
        cs = ax9.imshow(cth.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax9.set_title('L2 CTH [km]')

        divider = make_axes_locatable(ax9)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#


        ax_list = [f'ax{num}' for num in range(1, 10)]
        ax_list.remove('ax3')
        ax_list.remove('ax6')
        for num in range(len(ax_list)):
            ax = vars()[ax_list[num]]
            ax.set_xlim((extent[:2]))
            ax.set_ylim((extent[2:]))
            ax.set_xlabel('Longitude [$^\circ$]')
            ax.set_ylabel('Latitude [$^\circ$]')

        # save figure
        #/--------------------------------------------------------------\#
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s/<%s>.png' % (sat0.fdir_pre_data, _metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#


def cdata_cld_modis_only(sat0, fdir_cot, zpt_file, cfg_info, plot=True):

    # read in data
    #/----------------------------------------------------------------------------\#
    with h5py.File(f'{sat0.fdir_pre_data}/pre-data.h5', 'r') as f0:
        extent, ref_2d, ref_470_2d, rgb, cot_l2, cer_l2, lon_2d, lat_2d, cth, sfh, sza, saa, vza, vaa, alb_650, alb_470 = \
            [np.array(f0[k][...]) for k in ['extent', 'mod/rad/ref_650', 'mod/rad/ref_470', 'mod/rgb', 'mod/cld/cot_l2', 'mod/cld/cer_l2',
                                            'lon', 'lat', 'mod/cld/cth_l2', 'mod/geo/sfh', 'mod/geo/sza', 'mod/geo/saa', 'mod/geo/vza',
                                            'mod/geo/vaa', f"mod/sfc/alb_43_650", 'mod/sfc/alb_43_470',
                                            ]]
    ref_threshold = float(cfg_info['ref_threshold'])
    photons=float(cfg_info['cot_Nphotons'])
    #\----------------------------------------------------------------------------/#


    # cloud mask method based on rgb image and l2 data
    #/----------------------------------------------------------------------------\#
    # primary selection (over-selection of cloudy pixels is expected)
    #/--------------------------------------------------------------\#
    cld_frac0 = (np.logical_not(np.isnan(cot_l2)) & (cot_l2>0.0)).sum() / cot_l2.size
    frac0     = 1.0 - cld_frac0
    scale_factor = 1.08
    indices_x0, indices_y0 = cloud_mask_rgb(rgb, extent, lon_2d, lat_2d, ref_470_2d, alb_470,
                                            ref_threshold, frac=frac0, a_r=scale_factor,
                                            a_g=scale_factor, a_b=scale_factor)

    lon_cld0 = lon_2d[indices_x0, indices_y0]
    lat_cld0 = lat_2d[indices_x0, indices_y0]
    #\--------------------------------------------------------------/#

    """
    # secondary filter (remove incorrect cloudy pixels)
    #/--------------------------------------------------------------\#
    ref_cld0    = ref_2d[indices_x0, indices_y0]

    logic_nan_cth = np.isnan(cth[indices_x0, indices_y0])
    logic_nan_cot = np.isnan(cot_l2[indices_x0, indices_y0])
    logic_nan_cer = np.isnan(cer_l2[indices_x0, indices_y0])

    logic_bad = (ref_cld0<np.median(ref_cld0)) & \
                (logic_nan_cth & \
                 logic_nan_cot & \
                 logic_nan_cer)

    logic = np.logical_not(logic_bad)
    lon_cld = lon_cld0[logic]
    lat_cld = lat_cld0[logic]


    Nx, Ny = ref_2d.shape
    indices_x = indices_x0[logic]
    indices_y = indices_y0[logic]
    #"""
    lon_cld = lon_cld0
    lat_cld = lat_cld0


    Nx, Ny = ref_2d.shape
    indices_x = indices_x0
    indices_y = indices_y0

    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # ipa retrievals
    #/----------------------------------------------------------------------------\#
    # cth_ipa0
    # get cth for new cloud field obtained from radiance thresholding
    # [indices_x[logic], indices_y[logic]] from cth from MODIS L2 cloud product
    # this is counter-intuitive but we need to account for the parallax
    # correction (approximately) that has been applied to the MODIS L2 cloud
    # product before assigning CTH to cloudy pixels we selected from reflectance
    # field, where the clouds have not yet been parallax corrected
    #/--------------------------------------------------------------\#
    data0 = np.zeros_like(ref_2d)
    data0[indices_x, indices_y] = 1

    data = np.zeros_like(cth)
    data[cth>0.0] = 1
    dx, dy = 250, 250 # in meter
    offset_nx, offset_ny = move_correlate(data0, data)
    if offset_nx > 0:
        dist_x = dx * offset_nx
        lon_2d_, _ = cal_geodesic_lonlat(lon_2d, lat_2d, dist_x, 90.0)
        lon_2d_ = lon_2d_.reshape(lon_2d.shape)
    else:
        lon_2d_ = lon_2d.copy()

    if offset_ny > 0:
        dist_y = dy * offset_ny
        _, lat_2d_ = cal_geodesic_lonlat(lon_2d, lat_2d, dist_y, 0.0)
        lat_2d_ = lat_2d_.reshape(lat_2d.shape)
    else:
        lat_2d_ = lat_2d.copy()
    extent_ = [np.min(lon_2d_), np.max(lon_2d_), np.min(lat_2d_), np.max(lat_2d_)]

    # cth_ = cth.copy()
    cth_ = cth[indices_x, indices_y]
    cth_[cth_==0.0] = np.nan

    cth_ipa0 = np.zeros_like(ref_2d)

    print(f"cth_ipa0.shape: {cth_ipa0.shape}")
    print(f"lon_cld.shape: {lon_cld.shape}")
    print(f"lon_2d_.shape: {lon_2d_.shape}")
    print(f"cth_.shape: {cth_.shape}")
    print(f"indices_x.shape: {indices_x.shape}")
    print(f"cth_ipa0[indices_x, indices_y].shape: {cth_ipa0[indices_x, indices_y].shape}")

    # cth_ipa0[indices_x, indices_y] = find_nearest(lon_cld, lat_cld, cth_, lon_2d_, lat_2d_)

    # cth_ipa0 = find_nearest(lon_cld, lat_cld, cth_, lon_2d_, lat_2d_)

    # cth_ipa0[np.isnan(cth_ipa0)] = 0.0
    #\--------------------------------------------------------------/#

    # cer_ipa0
    #/--------------------------------------------------------------\#
    cer_ipa0 = np.zeros_like(ref_2d)
    # # cer_ipa0[indices_x, indices_y] = find_nearest(lon_cld, lat_cld, cer_l2, lon_2d_, lat_2d_)
    # cer_ipa0 = find_nearest(lon_cld, lat_cld, cer_l2[indices_x, indices_y], lon_2d_, lat_2d_)

    #\--------------------------------------------------------------/#

    # cot_ipa0
    # two relationships: one for geometrically thick clouds, one for geometrically thin clouds
    # ipa relationship of reflectance vs cloud optical thickness
    #/--------------------------------------------------------------\#
    dx = np.pi*6378.1*(lon_2d[1, 0]-lon_2d[0, 0])/180.0
    dy = np.pi*6378.1*(lat_2d[0, 1]-lat_2d[0, 0])/180.0

    fdir  = path_dir('%s/ipa-%06.1fnm_thick' % (fdir_cot, 650))

    cot_ipa = np.concatenate((np.arange(0.0, 2.0, 0.5),     \
                              np.arange(2.0, 30.0, 2.0),    \
                              np.arange(30.0, 60.0, 5.0),   \
                              np.arange(60.0, 100.0, 10.0), \
                              np.arange(100.0, 201.0, 50.0) \
                            ))
    print('cot_ipa shape:', cot_ipa.shape)

    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(zpt_file=zpt_file, fname=fname_atm, overwrite=False)

    # cpu number used
    if platform.system() in ['Windows', 'Darwin']:
        Ncpu=os.cpu_count()-1
    else:
        Ncpu=32

    f_mca_thick = func_ref_vs_cot(
                    cot_ipa,
                    cer0=25.0,
                    fdir=fdir,
                    date=sat0.date,
                    wavelength=650,
                    surface_albedo=alb_650.mean(),
                    solar_zenith_angle=sza.mean(),
                    solar_azimuth_angle=saa.mean(),
                    sensor_zenith_angle=vza.mean(),
                    sensor_azimuth_angle=vaa.mean(),
                    cloud_top_height=float(cfg_info['cth_thick']),
                    cloud_geometrical_thickness=float(cfg_info['cgt_thick']),
                    Nphoton=photons,
                    solver='3d',
                    overwrite=False,
                    atmospheric_profile=atm0,
                    Ncpu=Ncpu,
                    )

    fdir  = path_dir('%s/ipa-%06.1fnm_thin' % (fdir_cot, 650))
    f_mca_thin= func_ref_vs_cot(
                    cot_ipa,
                    cer0=10.0,
                    fdir=fdir,
                    date=sat0.date,
                    wavelength=650,
                    surface_albedo=alb_650.mean(),
                    solar_zenith_angle=sza.mean(),
                    solar_azimuth_angle=saa.mean(),
                    sensor_zenith_angle=vza.mean(),
                    sensor_azimuth_angle=vaa.mean(),
                    cloud_top_height=float(cfg_info['cth_thin']),
                    cloud_geometrical_thickness=float(cfg_info['cgt_thin']),
                    Nphoton=photons,
                    solver='3d',
                    overwrite=False,
                    atmospheric_profile=atm0,
                    Ncpu=Ncpu,
                    )

    ref_cld_norm = ref_2d[indices_x, indices_y]/np.cos(np.deg2rad(sza.mean()))

    logic_thick = (cth_ipa0[indices_x, indices_y] > 4.0)
    logic_thin  = (cth_ipa0[indices_x, indices_y] <= 4.0)

    cot_ipa0 = np.zeros_like(ref_2d)

    popt_thick, pcov_thick = curve_fit(cot_fitting_func, cot_ipa, f_mca_thick.ref)
    popt_thin, pcov_thin = curve_fit(cot_fitting_func, cot_ipa, f_mca_thin.ref)

    f_mca_thick.ref = cot_fitting_func(cot_ipa, *popt_thick)
    f_mca_thin.ref = cot_fitting_func(cot_ipa, *popt_thin)

    cot_ipa0[indices_x[logic_thick], indices_y[logic_thick]] = f_mca_thick.get_cot_from_ref(ref_cld_norm[logic_thick])
    cot_ipa0[indices_x[logic_thin] , indices_y[logic_thin]]  = f_mca_thin.get_cot_from_ref(ref_cld_norm[logic_thin])

    logic_out = (cot_ipa0<cot_ipa[0]) | (cot_ipa0>cot_ipa[-1])
    logic_low = (logic_out) & (ref_2d<np.median(ref_2d[indices_x, indices_y]))
    logic_high = logic_out & np.logical_not(logic_low)
    cot_ipa0[logic_low]  = cot_ipa[0]
    cot_ipa0[logic_high] = cot_ipa[-1]

    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

    #/----------------------------------------------------------------------------\#
    cot_ipa0_650 = cot_ipa0.copy()
    cth_ipa0_650 = cth_ipa0.copy()
    cer_ipa0_650 = cer_ipa0.copy()
    #/----------------------------------------------------------------------------\#


    # for 3D calculation
    # MODIS 3D: parallax correction only
    # OCO 3D: parallax correction and wind correction
    #/----------------------------------------------------------------------------\#
    # parallax correction
    # calculate new lon_corr, lat_corr based on cloud, surface and sensor geometries
    #/--------------------------------------------------------------\#
    vza_cld = vza[indices_x, indices_y]
    vaa_cld = vaa[indices_x, indices_y]
    sfh_cld = sfh[indices_x, indices_y] * 1000.0  # convert to meter from km
    cth_cld = cth_ipa0[indices_x, indices_y] * 1000.0 # convert to meter from km
    lon_corr_p, lat_corr_p = para_corr(lon_cld, lat_cld, vza_cld, vaa_cld, cth_cld, sfh_cld)
    #\--------------------------------------------------------------/#

    # ===== MODIS 3D =====
    Nx, Ny = ref_2d.shape
    cot_3d_650 = np.zeros_like(ref_2d)
    cer_3d_650 = np.zeros_like(ref_2d)
    cth_3d_650 = np.zeros_like(ref_2d)
    cld_msk_3d_650  = np.zeros(ref_2d.shape, dtype=np.int32)
    for i in range(indices_x.size):
        ix = indices_x[i]
        iy = indices_y[i]

        lon_corr0_p = lon_corr_p[i]
        lat_corr0_p = lat_corr_p[i]
        ix_corr = int((lon_corr0_p-lon_2d[0, 0])//(lon_2d[1, 0]-lon_2d[0, 0]))
        iy_corr = int((lat_corr0_p-lat_2d[0, 0])//(lat_2d[0, 1]-lat_2d[0, 0]))
        if (ix_corr>=0) and (ix_corr<Nx) and (iy_corr>=0) and (iy_corr<Ny):
            cot_3d_650[ix_corr, iy_corr] = cot_ipa0[ix, iy]
            cer_3d_650[ix_corr, iy_corr] = cer_ipa0[ix, iy]
            cth_3d_650[ix_corr, iy_corr] = cth_ipa0[ix, iy]
            cld_msk_3d_650[ix_corr, iy_corr] = 1
    # fill-in the empty cracks
    crack_adjustment(indices_x, indices_y, Nx, Ny, cot_3d_650, cer_3d_650, cth_3d_650,
                     cld_msk_3d_650, cot_ipa0_650,
                     Npixel=2, percent_a=0.7, percent_b=0.7)




    # write cot_ipa into file
    #/----------------------------------------------------------------------------\#
    with h5py.File(f'{sat0.fdir_pre_data}/pre-data.h5', 'r+') as f0:
        # Update or create groups with try-except blocks
        group_data = {'mod/cld/cot_ipa_650': cot_ipa0_650,
                      'mod/cld/cer_ipa_650': cer_ipa0_650,
                      'mod/cld/cth_ipa_650': cth_ipa0_650,
                      'mod/cld/cot_3d_650': cot_3d_650,
                      'mod/cld/cer_3d_650': cer_3d_650,
                      'mod/cld/cth_3d_650': cth_3d_650,
                      'mod/cld/logic_cld_650': (cot_ipa0_650>0),
                      'mod/cld/logic_cld_3d_650': (cot_3d_650>0),
                      'mod/cld/cld_msk_3d_650': (cld_msk_3d_650>0),
                      }

        for group_name, group_data in group_data.items():
            if f0.get(group_name):
                del f0[group_name]
            f0[group_name] = group_data

        # Delete and create cld_msk group
        if f0.get('cld_msk'):
            del f0['cld_msk']

        g0 = f0.create_group('cld_msk')
        g0.update({
            f'indices_{suffix}': indices for suffix, indices in [
                ('x0', indices_x0),
                ('y0', indices_y0),
                ('x', indices_x),
                ('y', indices_y)
            ]
        })


        # Delete and create mca_ipa_thick group
        if f0.get('mca_ipa_thick'):
            del f0['mca_ipa_thick']
        g0 = f0.create_group('mca_ipa_thick')
        g0.update({
            'cot': f_mca_thick.cot,
            'ref': f_mca_thick.ref,
            'ref_std': f_mca_thick.ref_std
        })
        # Delete and create mca_ipa_thin group
        if f0.get('mca_ipa_thin'):
            del f0['mca_ipa_thin']
        g0 = f0.create_group('mca_ipa_thin')
        g0.update({
            'cot': f_mca_thin.cot,
            'ref': f_mca_thin.ref,
            'ref_std': f_mca_thin.ref_std
        })

        # Delete and create cld_corr group
        if f0.get('cld_corr'):
            del f0['cld_corr']
        g0 = f0.create_group('cld_corr')
        g0.update({
            'lon_ori': lon_cld,
            'lat_ori': lat_cld,
            'lon_corr_p': lon_corr_p,
            'lat_corr_p': lat_corr_p,
        })
    #\----------------------------------------------------------------------------/#
    if plot:
        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        rcParams['font.size'] = 12
        fig = plt.figure(figsize=(16, 16))
        fig.suptitle('MODIS Cloud Re-Processing')

        # RGB
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(441)
        cs = ax1.imshow(rgb, zorder=0, extent=extent)
        ax1.set_title('RGB Imagery')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\--------------------------------------------------------------/#

        # L1B reflectance
        #/----------------------------------------------------------------------------\#
        ax2 = fig.add_subplot(442)
        cs = ax2.imshow(ref_2d.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=1.0)
        ax2.set_title('L1B Reflectance (650 nm)')

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cloud mask (primary)
        #/----------------------------------------------------------------------------\#
        ax3 = fig.add_subplot(443)
        cs = ax3.imshow(rgb, zorder=0, extent=extent)
        ax3.scatter(lon_2d[indices_x0, indices_y0], lat_2d[indices_x0, indices_y0], s=0.1, c='r', alpha=0.1)
        ax3.set_title('Primary Cloud Mask')

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\----------------------------------------------------------------------------/#

        # cloud mask (final)
        #/----------------------------------------------------------------------------\#
        ax4 = fig.add_subplot(444)
        cs = ax4.imshow(rgb, zorder=0, extent=extent)
        ax4.scatter(lon_2d[indices_x, indices_y], lat_2d[indices_x, indices_y], s=0.1, c='r', alpha=0.1)
        ax4.set_title('Secondary Cloud Mask (Final)')

        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\----------------------------------------------------------------------------/#

        # 470 reflectance (MODIS)
        #/----------------------------------------------------------------------------\#
        ax17 = fig.add_subplot(4, 4, 5)
        cs = ax17.imshow(ref_470_2d.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.4)
        ax17.set_title('MODIS 470nm (filled and scaled)')

        divider = make_axes_locatable(ax17)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # surface albedo (MYD43A3, white sky albedo)
        #/----------------------------------------------------------------------------\#
        ax16 = fig.add_subplot(4, 4, 9)
        cs = ax16.imshow(alb_470.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.4)
        ax16.set_title('43A3 WSA  470nm(filled and scaled)')

        divider = make_axes_locatable(ax16)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cot l2
        #/----------------------------------------------------------------------------\#
        ax5 = fig.add_subplot(446)
        cs = ax5.imshow(cot_l2.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax5.set_title('L2 COT')

        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer l2
        #/----------------------------------------------------------------------------\#
        ax6 = fig.add_subplot(447)
        cs = ax6.imshow(cer_l2.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax6.set_title('L2 CER [$\mu m$]')

        divider = make_axes_locatable(ax6)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth l2
        #/----------------------------------------------------------------------------\#
        ax7 = fig.add_subplot(448)
        cs = ax7.imshow(cth.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax7.set_title('L2 CTH [km]')

        divider = make_axes_locatable(ax7)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cot ipa0
        #/----------------------------------------------------------------------------\#
        ax9 = fig.add_subplot(4,4,10)
        cs = ax9.imshow(cot_ipa0.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax9.set_title('New IPA COT')

        divider = make_axes_locatable(ax9)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer ipa0
        #/----------------------------------------------------------------------------\#
        ax10 = fig.add_subplot(4, 4, 11)
        cs = ax10.imshow(cer_ipa0.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax10.set_title('New L2 CER [$\mu m$]')

        divider = make_axes_locatable(ax10)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth ipa0
        #/----------------------------------------------------------------------------\#
        ax11 = fig.add_subplot(4, 4, 12)
        cs = ax11.imshow(cth_ipa0.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax11.set_title('New L2 CTH [km]')

        divider = make_axes_locatable(ax11)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

       # surface albedo (MYD43A3, white sky albedo)
        #/----------------------------------------------------------------------------\#
        ax12 = fig.add_subplot(4, 4, 13)
        cs = ax16.imshow(alb_650.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.4)
        ax16.set_title('43A3 WSA  650nm(filled and scaled)')

        divider = make_axes_locatable(ax16)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#


        ax_list = [f'ax{num}' for num in range(1, 13  )]
        ax_list.remove('ax8')
        # ax_list.remove('ax12')
        for num in range(len(ax_list)):
            ax = vars()[ax_list[num]]
            ax.set_xlim((extent[:2]))
            ax.set_ylim((extent[2:]))
            ax.set_xlabel('Longitude [$^\circ$]')
            ax.set_ylabel('Latitude [$^\circ$]')

        # save figure
        #/--------------------------------------------------------------\#
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s/<%s>.png' % (sat0.fdir_pre_data, _metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#



if __name__ == '__main__':
    None

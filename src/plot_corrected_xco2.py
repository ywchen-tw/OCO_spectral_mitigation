"""plot_corrected_xco2.py — Compare OCO-2 model-corrected XCO2 with a TCCON ground station.

Reads plot_data.csv produced by apply_models.py and a TCCON NetCDF4 file.

Figure layout (3 rows × n_models columns, n_models = number of active models ≥ 3):
  Row 1 : lon/lat scatter maps — one panel per active model correction
          (Ridge | MLP | FT | XGBoost | Hybrid — only present models shown)
  Row 2 : lon/lat scatter maps — original XCO₂_bc | ideal-corrected | TCCON time series
  Row 3 : histogram — all corrected XCO₂ distributions vs TCCON

Shared colorbar covers all lon/lat map panels (rows 1–2, cols 0–2).

Usage:
    python src/plot_corrected_xco2.py \\
        --plot-data  results/combined_2020_dates/plot_data.csv \\
        --tccon      /path/to/ra20150301_20200718.public.qc.nc \\
        --output-dir results/combined_2020_dates/plots/

    Relative paths for --plot-data, --tccon, and --output-dir are resolved
    against get_storage_dir() (platform-dependent root), matching apply_models.py.

Optional filters:
    --lon-range  30 80          (spatial filter on OCO-2 lon)
    --lat-range -40 0           (spatial filter on OCO-2 lat)
    --date-range 2018-01-01 2020-12-31  (TCCON date filter)
    --vmin 395 --vmax 415       (force colorbar / histogram range)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc4

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_storage_dir

# ── Styling constants ──────────────────────────────────────────────────────────
_CMAP = 'jet'

_MODEL_CFGS = [
    ('Ridge',   'ridge_corrected_xco2',  'orange'),
    ('MLP',     'mlp_corrected_xco2',    'limegreen'),
    ('FT',      'ft_corrected_xco2',     'purple'),
    ('XGBoost', 'xgb_corrected_xco2',    'crimson'),
    ('Hybrid',  'hybrid_corrected_xco2', 'deepskyblue'),
]


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_tccon(nc_path: str) -> pd.DataFrame:
    """Return DataFrame (time [UTC-aware], lat, lon, xco2, xco2_error)."""
    with nc4.Dataset(nc_path, 'r') as ds:
        # TCCON files use 'long' (not 'lon') and time in seconds since 1970-01-01 UTC
        time_sec  = np.ma.filled(ds.variables['time'][:],      np.nan).astype(np.float64)
        lat_arr   = np.ma.filled(ds.variables['lat'][:],       np.nan).astype(np.float64)
        lon_arr   = np.ma.filled(ds.variables['long'][:],      np.nan).astype(np.float64)
        xco2_arr  = np.ma.filled(ds.variables['xco2'][:],      np.nan).astype(np.float64)
        xco2e_arr = np.ma.filled(ds.variables['xco2_error'][:], np.nan).astype(np.float64)

    # TCCON time: "seconds since 1970-01-01 00:00:00" (gregorian UTC).
    # pd.to_datetime(float_array, unit='s') fails in pandas 2.x when the array
    # contains NaN (tries int64 cast first).  Route through pd.to_timedelta
    # which converts NaN → NaT without error.
    _epoch = pd.Timestamp('1970-01-01 00:00:00', tz='UTC')
    times  = _epoch + pd.to_timedelta(time_sec, unit='s', errors='coerce')
    df = pd.DataFrame({
        'time':       times,
        'lat':        lat_arr,
        'lon':        lon_arr,
        'xco2':       xco2_arr,
        'xco2_error': xco2e_arr,
    })
    valid = (df['xco2'] > 300) & (df['xco2'] < 550) & df['xco2'].notna()
    return df[valid].reset_index(drop=True)


def load_plot_data(path: str) -> pd.DataFrame:
    """Load plot_data.parquet (or .csv); compute ideal-corrected column if anomaly is present."""
    df = pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)
    if 'xco2_bc' in df.columns and 'xco2_bc_anomaly' in df.columns:
        df['ideal_corrected_xco2'] = df['xco2_bc'] - df['xco2_bc_anomaly']
    return df


# ── MODIS RGB background ────────────────────────────────────────────────────────

def download_modis_rgb(
        date,
        extent,
        which='terra',
        wmts_cgi='https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi',
        fdir='.',
        proj=None,
        coastline=False,
        run=True,
        ):
    """Download a MODIS true-colour RGB tile from NASA GIBS and save as PNG.

    Parameters
    ----------
    date    : datetime.date or pandas.Timestamp
    extent  : [lon_min, lon_max, lat_min, lat_max]  (degrees, PlateCarree)
    which   : 'terra' or 'aqua'
    fdir    : directory to save the PNG
    run     : if False, skip the download and just return the expected filename
    Returns the PNG file path.

    Notes
    -----
    Uses the GIBS EPSG:4326 (geographic) WMTS endpoint rather than EPSG:3857
    (Mercator).  The geographic endpoint:
      - pairs naturally with PlateCarree axes (no CRS reprojection),
      - avoids the OWSLib TileMatrixLimits-duplication warnings that corrupt
        tile coordinates in the Mercator endpoint,
      - has reliable tile coverage at all zoom levels (no 404s at TILEMATRIX=9).
    """
    which  = which.lower()
    date_s = date.strftime('%Y-%m-%d')
    fname  = '%s/%s_rgb_%s_%s.png' % (
        fdir, which, date_s,
        '-'.join(['%.2f' % e for e in extent]),
    )

    if run:
        try:
            from owslib.wmts import WebMapTileService
        except ImportError:
            raise ImportError(
                "download_modis_rgb requires 'owslib'. Install with: pip install owslib"
            )
        try:
            import cartopy.crs as ccrs
        except ImportError:
            raise ImportError(
                "download_modis_rgb requires 'cartopy'. Install with: pip install cartopy"
            )

        if which == 'terra':
            layer_name = 'MODIS_Terra_CorrectedReflectance_TrueColor'
        elif which == 'aqua':
            layer_name = 'MODIS_Aqua_CorrectedReflectance_TrueColor'
        else:
            raise ValueError(f"which must be 'terra' or 'aqua', got {which!r}")

        if proj is None:
            proj = ccrs.PlateCarree()

        # EPSG:4326 endpoint — no Mercator CRS registration required
        wmts = WebMapTileService(wmts_cgi)

        # Print the GetTile URL pattern so the user can verify / curl-test it.
        # Cartopy chooses {zoom}/{row}/{col} at render time; the rest is fixed.
        _tms_id = next(iter(wmts.contents[layer_name].tilematrixsetlinks), '?')
        _fmt    = (wmts.contents[layer_name].formats or ['image/jpeg'])[0]
        print(
            f"  GetTile URL pattern:\n"
            f"    {wmts_cgi}?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
            f"&LAYER={layer_name}&STYLE=default"
            f"&TILEMATRIXSET={_tms_id}&TILEMATRIX={{zoom}}"
            f"&TILEROW={{row}}&TILECOL={{col}}"
            f"&FORMAT={_fmt}&time={date_s}",
            flush=True,
        )

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111, projection=proj)
        ax1.add_wmts(wmts, layer_name, wmts_kwargs={'time': date_s})
        if coastline:
            ax1.coastlines(resolution='10m', color='black', linewidth=0.5, alpha=0.8)
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        ax1.patch.set_visible(False)
        ax1.axis('off')
        plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
        print(f"  MODIS RGB saved → {fname}", flush=True)

    return fname


# ── Granule date from Phase-4 HDF5 ────────────────────────────────────────────

def _dominant_granule_date(h5_path: Path):
    """Return ``(date_ts, granule_ts)`` for the most-represented MODIS granule.

    Reads the ``granule_id`` dataset (strings like
    ``MYD35_L2.A2020122.0530.061.*.hdf``), parses the embedded
    ``A{YYYY}{DDD}.{HHMM}`` tokens, finds the granule ID that appears most
    often, and returns:

    * ``date_ts``    – :class:`pd.Timestamp` at midnight (for GIBS date query)
    * ``granule_ts`` – :class:`pd.Timestamp` with the granule start time (HHMM)

    Both are timezone-naive UTC.
    Raises ``ValueError`` if no parseable granule IDs are found.
    """
    import re
    import h5py
    from collections import Counter

    pattern = re.compile(r'\.A(\d{4})(\d{3})\.(\d{4})\.')

    with h5py.File(str(h5_path), 'r') as f:
        raw = f['granule_id'][:]

    gids = [g.decode() if isinstance(g, bytes) else str(g) for g in raw]

    timestamps = {}   # granule_id_string → pd.Timestamp (with HHMM)
    gid_per_row = []
    for gid in gids:
        m = pattern.search(gid)
        if m:
            year = int(m.group(1))
            doy  = int(m.group(2))
            hhmm = m.group(3)
            if gid not in timestamps:
                timestamps[gid] = (pd.Timestamp(f'{year}-01-01')
                                   + pd.Timedelta(days=doy - 1,
                                                  hours=int(hhmm[:2]),
                                                  minutes=int(hhmm[2:])))
            gid_per_row.append(gid)

    if not gid_per_row:
        raise ValueError(f"No parseable granule_id timestamps found in {h5_path}")

    dominant_gid = Counter(gid_per_row).most_common(1)[0][0]
    granule_ts   = timestamps[dominant_gid].tz_localize('UTC')
    date_ts      = pd.Timestamp(granule_ts.date())
    return date_ts, granule_ts


# ── Per-granule MODIS RGB ──────────────────────────────────────────────────────

def render_modis_granule_rgb(myd021km_path, myd03_path, output_png,
                              extent=None, gamma=2.0, resolution=0.01):
    """Render a per-granule MODIS true-colour RGB from MYD021KM + MYD03.

    GIBS WMTS only provides daily composites.  This function reads the actual
    per-granule Level-1B calibrated radiances (HDF4) and the paired geolocation
    file, resamples the swath onto a regular lat/lon grid, and saves a PNG.

    Bands used (standard MODIS true-colour):
        Red   — EV_250_Aggr1km_RefSB[0]  (Band 1, 620–670 nm)
        Green — EV_500_Aggr1km_RefSB[1]  (Band 4, 545–565 nm)
        Blue  — EV_500_Aggr1km_RefSB[0]  (Band 3, 459–479 nm)

    Parameters
    ----------
    myd021km_path : str | Path   MYD021KM HDF4 file
    myd03_path    : str | Path   MYD03    HDF4 geolocation file
    output_png    : str | Path   Destination PNG
    extent        : [lon_min, lon_max, lat_min, lat_max] or None
                    Clip swath to this bounding box before resampling.
    gamma         : float        Gamma correction (default 2.0 brightens image).
    resolution    : float        Output grid spacing in degrees (default 0.01°≈1 km).

    Returns
    -------
    (output_png_str, actual_extent)

    Requirements
    ------------
    pyhdf   : conda install -c conda-forge pyhdf
    scipy   : already a pipeline dependency
    """
    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        raise ImportError(
            "render_modis_granule_rgb requires pyhdf.\n"
            "  conda install -c conda-forge pyhdf"
        )
    from scipy.spatial import cKDTree

    # ── Read calibrated reflectances from MYD021KM ────────────────────────────
    hdf = SD(str(myd021km_path), SDC.READ)

    def _refl(sds_name, band_idx):
        sds    = hdf.select(sds_name)
        raw    = sds[band_idx].astype(np.float32)
        attrs  = sds.attributes()
        scale  = attrs['reflectance_scales'][band_idx]
        offset = attrs['reflectance_offsets'][band_idx]
        fill   = attrs.get('_FillValue', 65535)
        ref    = np.where(raw == fill, np.nan, (raw - offset) * scale)
        return np.clip(ref, 0.0, 1.0)

    r = _refl('EV_250_Aggr1km_RefSB', 0)   # Band 1 — Red
    g = _refl('EV_500_Aggr1km_RefSB', 1)   # Band 4 — Green
    b = _refl('EV_500_Aggr1km_RefSB', 0)   # Band 3 — Blue
    hdf.end()

    # ── Read geolocation from MYD03 ───────────────────────────────────────────
    geo    = SD(str(myd03_path), SDC.READ)
    lat_2d = geo.select('Latitude')[:]    # float32 [lines, samples]
    lon_2d = geo.select('Longitude')[:]   # float32 [lines, samples]
    geo.end()

    # ── Clip to extent ────────────────────────────────────────────────────────
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        pad  = 0.1
        mask = ((lon_2d >= lon_min - pad) & (lon_2d <= lon_max + pad) &
                (lat_2d >= lat_min - pad) & (lat_2d <= lat_max + pad))
        if mask.any():
            rows   = np.where(mask.any(axis=1))[0]
            cols   = np.where(mask.any(axis=0))[0]
            sl_r   = slice(rows[0], rows[-1] + 1)
            sl_c   = slice(cols[0], cols[-1] + 1)
            lat_2d = lat_2d[sl_r, sl_c]
            lon_2d = lon_2d[sl_r, sl_c]
            r, g, b = r[sl_r, sl_c], g[sl_r, sl_c], b[sl_r, sl_c]

    actual_extent = [
        float(np.nanmin(lon_2d)), float(np.nanmax(lon_2d)),
        float(np.nanmin(lat_2d)), float(np.nanmax(lat_2d)),
    ]

    # ── Resample swath → regular lat/lon grid (nearest-neighbour) ────────────
    lon_out = np.arange(actual_extent[0], actual_extent[1] + resolution, resolution)
    # lat from top (north) to bottom (south) so imshow renders correctly
    lat_out = np.arange(actual_extent[3], actual_extent[2] - resolution, -resolution)
    lon_mg, lat_mg = np.meshgrid(lon_out, lat_out)

    valid  = (np.isfinite(lat_2d) & np.isfinite(lon_2d) &
              np.isfinite(r) & np.isfinite(g) & np.isfinite(b))
    pts    = np.column_stack([lon_2d[valid], lat_2d[valid]])
    tree   = cKDTree(pts)
    _, idx = tree.query(np.column_stack([lon_mg.ravel(), lat_mg.ravel()]),
                        workers=-1)

    def _ch(arr):
        return arr[valid][idx].reshape(lon_mg.shape)

    rgb = np.stack([_ch(r), _ch(g), _ch(b)], axis=-1)

    # ── Gamma correction ──────────────────────────────────────────────────────
    rgb = np.clip(rgb ** (1.0 / gamma), 0.0, 1.0)

    # ── Save ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb,
              extent=[actual_extent[0], actual_extent[1],
                      actual_extent[2], actual_extent[3]],
              aspect='auto', origin='upper')
    ax.axis('off')
    plt.savefig(str(output_png), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"  MODIS granule RGB saved → {output_png}", flush=True)

    return str(output_png), actual_extent


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _scatter_map(ax, lon, lat, values, title, norm, cmap,
                 tccon_lon=None, tccon_lat=None,
                 bg_img=None, bg_extent=None,
                 view_extent=None):
    """lon/lat scatter map coloured by *values*; optionally mark TCCON station.

    Parameters
    ----------
    view_extent : [lon_min, lon_max, lat_min, lat_max] or None
        If provided, the axes are locked to this range after all plotting,
        preventing the scatter from auto-expanding the view.  Derived in
        main() from bg_extent (GIBS/L1B) or --lon-range / --lat-range args.
    """
    if bg_img is not None and bg_extent is not None:
        # extent=[lon_min, lon_max, lat_min, lat_max] → imshow wants [left, right, bottom, top]
        ax.imshow(bg_img,
                  extent=[bg_extent[0], bg_extent[1], bg_extent[2], bg_extent[3]],
                  aspect='auto', origin='upper', zorder=0)
    valid = np.isfinite(values)
    ax.scatter(lon[valid], lat[valid], c=values[valid],
               norm=norm, cmap=cmap, s=7.5, alpha=0.5, rasterized=True)
    if tccon_lon is not None and tccon_lat is not None:
        ax.scatter([tccon_lon], [tccon_lat], c='red', s=80, marker='*',
                   zorder=5, label='TCCON station')
        ax.legend(fontsize=7, loc='lower right', markerscale=1.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Lon (°E)', fontsize=8)
    ax.set_ylabel('Lat (°N)', fontsize=8)
    ax.tick_params(labelsize=7)
    # Lock axis limits — must come after all plot calls so scatter doesn't override
    if view_extent is not None:
        ax.set_xlim(view_extent[0], view_extent[1])
        ax.set_ylim(view_extent[2], view_extent[3])


def _tccon_panel(ax, tccon_df: pd.DataFrame, vmin: float, vmax: float, title: str,
                 fp_times=None, oco_vals=None):
    """Time series of TCCON XCO₂ with per-measurement error and monthly mean.

    Parameters
    ----------
    fp_times : pd.Series of timezone-aware Timestamps, optional
        OCO-2 footprint observation times (after spatial filtering).
        Consecutive times are grouped into orbit passes (gap threshold: 30 min)
        and each pass is shaded as a vertical span.
    oco_vals : array-like, optional
        OCO-2 XCO₂ values used to set the y-axis limits (1st–99th percentile
        with a 0.5 ppm margin).  Falls back to vmin/vmax when None.
    """
    ax.set_title(title, fontsize=9)
    ax.set_ylabel('XCO₂ (ppm)', fontsize=8)
    ax.tick_params(labelsize=7)

    if len(tccon_df) == 0:
        ax.text(0.5, 0.5, 'No TCCON data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
        return

    t  = tccon_df['time']          # timezone-aware pandas Series
    y  = tccon_df['xco2'].values
    ye = tccon_df['xco2_error'].values

    # ── OCO-2 footprint time ranges (shaded, drawn first so they sit behind data) ──
    if fp_times is not None and len(fp_times) > 0:
        fp_sorted = fp_times.sort_values().reset_index(drop=True)
        gap_mask  = fp_sorted.diff() > pd.Timedelta(minutes=30)
        # Build list of (start, end) for each continuous orbit pass
        starts = [fp_sorted.iloc[0]]
        ends   = []
        for idx in fp_sorted[gap_mask].index:
            ends.append(fp_sorted.iloc[idx - 1])
            starts.append(fp_sorted.iloc[idx])
        ends.append(fp_sorted.iloc[-1])

        for i, (t0, t1) in enumerate(zip(starts, ends)):
            ax.axvspan(t0, t1, alpha=0.18, color='royalblue', zorder=0,
                       label='OCO-2 pass' if i == 0 else '_nolegend_')

    # Raw measurements + ±1σ shading
    ax.scatter(t, y, s=15, c='salmon', alpha=0.7, zorder=2, label='Measurements')
    ax.fill_between(t, y - ye, y + ye, color='red', alpha=0.12, zorder=1)

    # Monthly mean (group by year+month to avoid resample API version issues)
    tccon_df2 = tccon_df.copy()
    tccon_df2['_ym'] = tccon_df2['time'].dt.year * 100 + tccon_df2['time'].dt.month
    mon = (tccon_df2.groupby('_ym')
                    .agg(xco2=('xco2', 'mean'), time=('time', 'mean'))
                    .reset_index())
    ax.plot(mon['time'], mon['xco2'], '-', color='darkred',
            linewidth=1.2, zorder=3, label='Monthly mean')

    if oco_vals is not None:
        _v = np.asarray(oco_vals, dtype=float)
        _v = _v[np.isfinite(_v)]
        if len(_v) > 0:
            _margin = 0.5
            vmin = float(np.nanpercentile(_v, 1))  - _margin
            vmax = float(np.nanpercentile(_v, 99)) + _margin
    ax.set_ylim(vmin, vmax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=6)
    ax.legend(fontsize=7)

    mu, sigma = float(y.mean()), float(y.std())
    ax.text(0.02, 0.97,
            f'n={len(y):,}  μ={mu:.2f}  σ={sigma:.2f} ppm',
            transform=ax.transAxes, va='top', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _histogram_panel(ax, oco_df: pd.DataFrame, tccon_df: pd.DataFrame,
                     vmin: float, vmax: float):
    """Overlapping density histograms: original, each model correction, TCCON.

    Ideal-corrected XCO₂ is drawn on a twinx right-hand axis so its density
    scale is independent of the model-correction curves.
    """
    bins = np.linspace(vmin, vmax, 120)

    def _draw(a, vals, label: str, color: str, lw: float = 1.2):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return
        mu, sigma = v.mean(), v.std()
        a.hist(v, bins=bins, density=True, color=color, alpha=0.35,
               label=f'{label}  μ={mu:.2f} σ={sigma:.2f} ppm')
        a.axvline(mu,          color=color, linewidth=lw)
        a.axvline(mu - sigma,  color=color, linewidth=lw * 0.7, linestyle=':')
        a.axvline(mu + sigma,  color=color, linewidth=lw * 0.7, linestyle=':')

    if 'xco2_bc' in oco_df.columns:
        _draw(ax, oco_df['xco2_bc'], 'OCO-2 original', 'steelblue')
    for name, col, color in _MODEL_CFGS:
        if col in oco_df.columns:
            _draw(ax, oco_df[col], f'{name} corrected', color)
    ax.set_xlabel('XCO₂ (ppm)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('XCO₂ distribution — OCO-2 corrected vs TCCON ground station',
                 fontsize=10)

    # ── Ideal-corrected + TCCON on twin right-hand axis ───────────────────────
    ax_twin = ax.twinx()
    if 'ideal_corrected_xco2' in oco_df.columns:
        _draw(ax_twin, oco_df['ideal_corrected_xco2'], 'Ideal corrected', 'gray')
    if len(tccon_df) > 0:
        _draw(ax_twin, tccon_df['xco2'], 'TCCON (ground)', 'red', lw=1.5)
    ax_twin.set_ylabel('Density (ideal / TCCON)', fontsize=9, color='dimgray')
    ax_twin.tick_params(axis='y', labelcolor='dimgray', labelsize=7)

    # Merge legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=12, ncol=2, loc='upper left')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot OCO-2 model-corrected XCO2 vs TCCON ground station.'
    )
    parser.add_argument('--plot-data',  required=True,
                        help='plot_data.parquet (or .csv) produced by apply_models.py')
    parser.add_argument('--tccon',      required=True,
                        help='TCCON NetCDF4 file (*.public.qc.nc)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same directory as --plot-data)')
    parser.add_argument('--lon-range',  nargs=2, type=float, default=None,
                        metavar=('LON_MIN', 'LON_MAX'),
                        help='Spatial filter on OCO-2 longitude')
    parser.add_argument('--lat-range',  nargs=2, type=float, default=None,
                        metavar=('LAT_MIN', 'LAT_MAX'),
                        help='Spatial filter on OCO-2 latitude')
    parser.add_argument('--date-plot', type=str, default=None,
                        metavar='YYYY-MM-DD',
                        help='TCCON date range filter, e.g. 2018-01-01')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Force colorbar/histogram lower bound (ppm)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Force colorbar/histogram upper bound (ppm)')
    parser.add_argument('--modis-auto', action='store_true',
                        help='Download a MODIS true-colour RGB background from GIBS. '
                             'Date is derived from --results-h5 granule_id (preferred) '
                             'or from the time column of plot_data.csv. '
                             'No local MODIS files are required.')
    parser.add_argument('--results-h5', default=None,
                        help='Phase-4 results HDF5 (results_{date}.h5). '
                             'When supplied with --modis-auto, the dominant MODIS granule '
                             'date is parsed from granule_id rather than the time column.')
    parser.add_argument('--modis-rgb', action='store_true',
                        help='Download and overlay a MODIS true-colour RGB background on scatter maps '
                             '(GIBS daily composite; use --modis-l1b for per-granule accuracy)')
    parser.add_argument('--modis-which', default='aqua', choices=['terra', 'aqua'],
                        help='MODIS instrument for RGB download (default: aqua)')
    parser.add_argument('--modis-date', default=None,
                        help='Date for MODIS RGB (YYYY-MM-DD). '
                             'If omitted, derived from the median time column of plot_data.csv')
    parser.add_argument('--modis-l1b', default=None,
                        help='MYD021KM HDF4 file for per-granule RGB rendering '
                             '(overrides --modis-rgb; requires pyhdf)')
    parser.add_argument('--modis-geo', default=None,
                        help='MYD03 HDF4 geolocation file paired with --modis-l1b '
                             '(default: auto-derived from --modis-l1b filename in the same directory)')
    args = parser.parse_args()

    storage_dir = get_storage_dir()

    def _abs(p):
        """Resolve a relative path against storage_dir; leave absolute paths unchanged."""
        if p is None:
            return None
        pp = Path(p)
        return str(storage_dir / pp) if not pp.is_absolute() else p

    plot_data_path = _abs(args.plot_data)
    tccon_path     = _abs(args.tccon)
    output_dir     = _abs(args.output_dir)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading plot data: {plot_data_path}", flush=True)
    oco = load_plot_data(plot_data_path)
    print(f"  OCO-2: {len(oco):,} rows", flush=True)

    print(f"Loading TCCON:     {tccon_path}", flush=True)
    tccon = load_tccon(tccon_path)
    print(f"  TCCON: {len(tccon):,} rows", flush=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    if args.lon_range and 'lon' in oco.columns:
        oco = oco[(oco['lon'] >= args.lon_range[0]) & (oco['lon'] <= args.lon_range[1])]
    if args.lat_range and 'lat' in oco.columns:
        oco = oco[(oco['lat'] >= args.lat_range[0]) & (oco['lat'] <= args.lat_range[1])]
    oco_start_time = pd.to_datetime(oco['time'], unit='s', utc=True).min() if 'time' in oco.columns else 'N/A'
    oco_end_time   = pd.to_datetime(oco['time'], unit='s', utc=True).max() if 'time' in oco.columns else 'N/A'
    print(f" OCO-2 after spatial filter", flush=True)
    print(f" start time: {oco_start_time}", flush=True)
    print(f" end time:   {oco_end_time}", flush=True)
    if args.date_plot:
        # Step 1: filter to the target date (single calendar day, UTC)
        t_date = pd.Timestamp(args.date_plot, tz='UTC').normalize()
        tccon = tccon[
            (tccon['time'] >= t_date) & (tccon['time'] < t_date + pd.Timedelta(days=1))
        ]
        print(f" TCCON after date filter", flush=True)
        print(f" start time: {tccon['time'].min() if len(tccon) > 0 else 'N/A'}", flush=True)
        print(f" end time:   {tccon['time'].max() if len(tccon) > 0 else 'N/A'}", flush=True)
        
        # # Switch to an interactive backend so plt.show() opens a window
        # import matplotlib
        # _prev_backend = matplotlib.get_backend()
        # for _be in ('MacOSX', 'TkAgg', 'Qt5Agg', 'WxAgg'):
        #     try:
        #         plt.switch_backend(_be)
        #         break
        #     except Exception:
        #         continue

        plt.close('all')
        plt.figure(figsize=(8, 4))
        oco_dt = pd.to_datetime(oco['time'], unit='s', utc=True, errors='coerce')
        plt.plot(oco_dt, oco['xco2_bc'], '.', label='OCO-2 original', markersize=1)
        plt.scatter(tccon['time'], tccon['xco2'], c='red', s=10, label='TCCON', alpha=0.6)
        plt.xlabel('Time (UTC)')
        plt.ylabel('XCO₂ (ppm)')
        # set x_axis as HH:MM and rotate labels
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.title('OCO-2 vs TCCON on ' + t_date.strftime('%Y-%m-%d'))
        plt.legend()
        plt.tight_layout()
        _debug_dir = Path(output_dir) if output_dir else Path(plot_data_path).parent
        _debug_dir.mkdir(parents=True, exist_ok=True)
        _debug_out = _debug_dir / f'debug_oco_tccon_{t_date.strftime("%Y-%m-%d")}.png'
        plt.savefig(_debug_out, dpi=120, bbox_inches='tight')
        print(f"  Debug figure saved → {_debug_out}", flush=True)
        # plt.show()
        plt.close('all')
        # plt.switch_backend(_prev_backend)
        
        # Step 2: further narrow to the OCO-2 footprint time window if valid
        if isinstance(oco_start_time, pd.Timestamp) and isinstance(oco_end_time, pd.Timestamp):
            # add a 30-minute buffer on either side to account for potential time mismatches
            buffer = pd.Timedelta(minutes=30)
            oco_start_time_buffered = oco_start_time - buffer
            oco_end_time_buffered   = oco_end_time   + buffer
            
            tccon = tccon[
                (tccon['time'] >= oco_start_time_buffered) & (tccon['time'] <= oco_end_time_buffered)
            ]
        
        print(f" TCCON after oco time filter", flush=True)
        print(f" start time: {tccon['time'].min() if len(tccon) > 0 else 'N/A'}", flush=True)
        print(f" end time:   {tccon['time'].max() if len(tccon) > 0 else 'N/A'}", flush=True)
    
    print(f"  OCO-2 after spatial filter: {len(oco):,}", flush=True)
    print(f"  TCCON after date filter:    {len(tccon):,}", flush=True)

    # ── Station location (fixed for TCCON) ────────────────────────────────────
    tccon_lon = float(tccon['lon'].median()) if len(tccon) > 0 else None
    tccon_lat = float(tccon['lat'].median()) if len(tccon) > 0 else None
    if tccon_lon is not None:
        print(f"  TCCON station: lon={tccon_lon:.3f}  lat={tccon_lat:.3f}", flush=True)

    # ── Shared colour range ───────────────────────────────────────────────────
    _pool = []
    for col in ('xco2_bc', 'ridge_corrected_xco2', 'mlp_corrected_xco2',
                'ft_corrected_xco2', 'xgb_corrected_xco2', 'hybrid_corrected_xco2',
                'ideal_corrected_xco2'):
        if col in oco.columns:
            _pool.append(oco[col].dropna().values)
    if len(tccon) > 0:
        _pool.append(tccon['xco2'].values)

    if _pool:
        _cat = np.concatenate([v[np.isfinite(v)] for v in _pool])
        vmin = args.vmin if args.vmin is not None else float(np.nanpercentile(_cat, 1))
        vmax = args.vmax if args.vmax is not None else float(np.nanpercentile(_cat, 99))
    else:
        vmin, vmax = 395.0, 415.0
    print(f"  XCO₂ colour range: [{vmin:.2f}, {vmax:.2f}] ppm", flush=True)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ── Active models (only those with data in plot_data.csv) ─────────────────
    active = [(nm, col, clr) for nm, col, clr in _MODEL_CFGS if col in oco.columns]

    lon_arr = oco['lon'].values if 'lon' in oco.columns else np.full(len(oco), np.nan)
    lat_arr = oco['lat'].values if 'lat' in oco.columns else np.full(len(oco), np.nan)

    # ── MODIS RGB background ───────────────────────────────────────────────────
    bg_img    = None
    bg_extent = None

    fin_lon = lon_arr[np.isfinite(lon_arr)]
    fin_lat = lat_arr[np.isfinite(lat_arr)]
    _have_coords = len(fin_lon) > 0 and len(fin_lat) > 0
    pad = 2.0
    rgb_extent = (
        [float(fin_lon.min()) - pad, float(fin_lon.max()) + pad,
         float(fin_lat.min()) - pad, float(fin_lat.max()) + pad]
        if _have_coords else None
    )
    out_dir_pre = (Path(str(output_dir)) if output_dir else Path(plot_data_path).parent)
    out_dir_pre.mkdir(parents=True, exist_ok=True)

    if args.modis_auto:
        # ── Auto GIBS: derive date from results HDF5 or time column ───────────
        if not _have_coords:
            print("  Warning: no valid lon/lat for MODIS RGB extent — skipping.",
                  flush=True)
        else:
            modis_date  = None
            date_source = None

            granule_ts  = None   # full timestamp (date + HHMM) for extent filtering

            # 1. Prefer granule_id from Phase-4 results HDF5
            if args.results_h5:
                h5_path = Path(_abs(args.results_h5))
                if h5_path.exists():
                    try:
                        modis_date, granule_ts = _dominant_granule_date(h5_path)
                        date_source = f'granule_id in {h5_path.name}'
                    except Exception as exc:
                        print(f"  Warning: could not parse granule_id from "
                              f"{h5_path} ({exc}) — falling back to time column.",
                              flush=True)
                else:
                    print(f"  Warning: --results-h5 path not found ({h5_path}) "
                          "— falling back to time column.", flush=True)

            # 2. Fallback: median of time column in plot_data.csv
            if modis_date is None and 'time' in oco.columns:
                _t = pd.to_datetime(oco['time'], unit='s', utc=True, errors='coerce').dropna()
                if len(_t) > 0:
                    granule_ts  = _t.median()
                    modis_date  = pd.Timestamp(granule_ts.date())
                    date_source = 'time column (median)'

            if modis_date is None:
                print("  Warning: --modis-auto could not determine a date "
                      "(supply --results-h5 or ensure a time column exists) "
                      "— skipping MODIS RGB.", flush=True)
            else:
                # Compute the GIBS download extent.
                # Explicit --lon-range / --lat-range always win; otherwise derive
                # from the granule's soundings (±20 min), falling back to all data.
                if args.lon_range and args.lat_range:
                    gibs_extent = [args.lon_range[0], args.lon_range[1],
                                   args.lat_range[0], args.lat_range[1]]
                elif args.lon_range:
                    gibs_extent = [args.lon_range[0], args.lon_range[1],
                                   float(fin_lat.min()) - pad, float(fin_lat.max()) + pad]
                elif args.lat_range:
                    gibs_extent = [float(fin_lon.min()) - pad, float(fin_lon.max()) + pad,
                                   args.lat_range[0], args.lat_range[1]]
                else:
                    gibs_extent = rgb_extent
                _explicit_range = bool(args.lon_range or args.lat_range)
                if not _explicit_range and granule_ts is not None and 'time' in oco.columns:
                    _ot = pd.to_datetime(oco['time'], unit='s', utc=True, errors='coerce')
                    _win = pd.Timedelta(minutes=20)
                    _gmask = (_ot >= granule_ts - _win) & (_ot <= granule_ts + _win)
                    _sub = oco[_gmask]
                    if len(_sub) > 0:
                        _gl = _sub['lon'].to_numpy(dtype=np.float64) if 'lon' in _sub.columns else np.array([])
                        _gt = _sub['lat'].to_numpy(dtype=np.float64) if 'lat' in _sub.columns else np.array([])
                        _gl = _gl[np.isfinite(_gl)]
                        _gt = _gt[np.isfinite(_gt)]
                        if len(_gl) > 0 and len(_gt) > 0:
                            gibs_extent = [
                                float(np.min(_gl)) - pad, float(np.max(_gl)) + pad,
                                float(np.min(_gt)) - pad, float(np.max(_gt)) + pad,
                            ]
                            print(f"  Granule extent: lon [{gibs_extent[0]:.1f}, {gibs_extent[1]:.1f}]"
                                  f"  lat [{gibs_extent[2]:.1f}, {gibs_extent[3]:.1f}]"
                                  f"  ({_gmask.sum():,} soundings)", flush=True)

                        # Re-filter TCCON to the time range of the selected OCO soundings
                        if 'time' in _sub.columns:
                            _ot_sub = pd.to_datetime(_sub['time'], unit='s', utc=True, errors='coerce').dropna()
                            if len(_ot_sub) > 0:
                                _tc_t0 = _ot_sub.min()   # UTC-aware (unit='s', utc=True)
                                _tc_t1 = _ot_sub.max()
                                _n_prev = len(tccon)
                                tccon = tccon[
                                    (tccon['time'] >= _tc_t0) & (tccon['time'] <= _tc_t1)
                                ]
                                print(f"  TCCON filtered to OCO footprint time "
                                      f"[{_tc_t0.strftime('%Y-%m-%d %H:%M')}, "
                                      f"{_tc_t1.strftime('%H:%M')}] UTC: "
                                      f"{_n_prev:,} → {len(tccon):,}", flush=True)

                print(f"  --modis-auto: GIBS {args.modis_which.upper()} composite "
                      f"for {modis_date.date()} (from {date_source}) …", flush=True)
                try:
                    rgb_path  = download_modis_rgb(
                        modis_date, gibs_extent,
                        which=args.modis_which,
                        fdir=str(out_dir_pre),
                        coastline=True,
                    )
                    bg_img    = plt.imread(rgb_path)
                    bg_extent = gibs_extent
                except Exception as exc:
                    print(f"  Warning: GIBS download failed ({exc}) — "
                          "scatter maps will have no background.", flush=True)

    elif args.modis_l1b:
        # ── Per-granule rendering from MYD021KM + MYD03 (no GIBS composite) ──
        l1b_path = Path(_abs(args.modis_l1b))
        if args.modis_geo:
            geo_path = Path(_abs(args.modis_geo))
        else:
            # Auto-derive MYD03 path: same dir, same A{YYYY}{DDD}.{HHMM} stem
            stem_prefix = l1b_path.name.split('.061.')[0].replace('MYD021KM', 'MYD03')
            candidates  = list(l1b_path.parent.glob(f'{stem_prefix}.061.*.hdf'))
            geo_path    = candidates[0] if candidates else None

        if geo_path is None or not geo_path.exists():
            print(f"  Warning: MYD03 geolocation file not found for {l1b_path.name} "
                  "— skipping per-granule RGB.", flush=True)
        else:
            print(f"  Rendering per-granule MODIS RGB from {l1b_path.name} …", flush=True)
            try:
                rgb_path, bg_extent = render_modis_granule_rgb(
                    l1b_path, geo_path,
                    out_dir_pre / f'modis_granule_{l1b_path.stem}.png',
                    extent=rgb_extent,
                )
                bg_img = plt.imread(rgb_path)
            except Exception as exc:
                print(f"  Warning: per-granule RGB failed ({exc}) — "
                      "scatter maps will have no background.", flush=True)

    elif args.modis_rgb:
        # ── GIBS daily composite (fallback; coarser temporal accuracy) ────────
        if not _have_coords:
            print("  Warning: no valid lon/lat for MODIS RGB extent — skipping download.",
                  flush=True)
        else:
            if args.modis_date:
                modis_date = pd.Timestamp(args.modis_date)
            elif 'time' in oco.columns:
                modis_date = pd.Timestamp(pd.to_datetime(oco['time'], unit='s', utc=True).median())
            elif 'date' in oco.columns:
                modis_date = pd.Timestamp(pd.to_datetime(oco['date']).median())
            else:
                print("  Warning: --modis-date not provided and no time/date column found "
                      "in plot_data.csv — skipping MODIS RGB.", flush=True)
                modis_date = None

            if modis_date is not None:
                print(f"  Downloading MODIS {args.modis_which.upper()} RGB for "
                      f"{modis_date.date()} extent={rgb_extent} …", flush=True)
                try:
                    rgb_path = download_modis_rgb(
                        modis_date, rgb_extent,
                        which=args.modis_which,
                        fdir=str(out_dir_pre),
                        coastline=True,
                    )
                    bg_img    = plt.imread(rgb_path)
                    bg_extent = rgb_extent
                except Exception as exc:
                    print(f"  Warning: MODIS RGB download failed ({exc}) — "
                          "scatter maps will have no background.", flush=True)

    # ── Map view extent ───────────────────────────────────────────────────────
    # Priority: --lon-range / --lat-range (explicit) > bg_extent (auto) > None
    if args.lon_range and args.lat_range:
        map_extent = [args.lon_range[0], args.lon_range[1],
                      args.lat_range[0], args.lat_range[1]]
    elif args.lon_range:
        map_extent = [args.lon_range[0], args.lon_range[1],
                      float(fin_lat.min()) - pad, float(fin_lat.max()) + pad]
    elif args.lat_range:
        map_extent = [float(fin_lon.min()) - pad, float(fin_lon.max()) + pad,
                      args.lat_range[0], args.lat_range[1]]
    elif bg_extent is not None:
        map_extent = bg_extent
    else:
        map_extent = None

    # ── Figure ────────────────────────────────────────────────────────────────
    # n_map_cols: enough columns for all active models; row 2 always needs ≥3
    n_map_cols = max(len(active), 3)
    fig = plt.figure(figsize=(6 * n_map_cols + 2, 16))
    gs  = fig.add_gridspec(3, n_map_cols + 1, hspace=0.50, wspace=0.30,
                            height_ratios=[1.0, 1.0, 0.75],
                            width_ratios=[1.0] * n_map_cols + [0.07])
    cbar_ax = fig.add_subplot(gs[0:2, n_map_cols])   # dedicated colorbar column

    map_axes = []   # collect lon/lat map axes for the shared colorbar

    # Row 1: model-corrected maps
    for j, (name, col, _) in enumerate(active[:n_map_cols]):
        ax = fig.add_subplot(gs[0, j])
        vals = oco[col].values
        _scatter_map(ax, lon_arr, lat_arr, vals,
                     f'{name}-corrected XCO₂ (ppm)',
                     norm, _CMAP, tccon_lon, tccon_lat,
                     bg_img=bg_img, bg_extent=bg_extent,
                     view_extent=map_extent)
        map_axes.append(ax)
    for j in range(len(active), n_map_cols):   # hide unused slots
        fig.add_subplot(gs[0, j]).set_visible(False)

    # Row 2 col 0: original XCO₂_bc
    ax20 = fig.add_subplot(gs[1, 0])
    if 'xco2_bc' in oco.columns:
        _scatter_map(ax20, lon_arr, lat_arr, oco['xco2_bc'].values,
                     'Original XCO₂_bc (ppm)',
                     norm, _CMAP, tccon_lon, tccon_lat,
                     bg_img=bg_img, bg_extent=bg_extent,
                     view_extent=map_extent)
        map_axes.append(ax20)
    else:
        ax20.text(0.5, 0.5, 'xco2_bc not in plot_data.csv',
                  ha='center', va='center', transform=ax20.transAxes)
        ax20.set_title('Original XCO₂_bc', fontsize=9)

    # Row 2 col 1: ideal-corrected (xco2_bc − anomaly)
    ax21 = fig.add_subplot(gs[1, 1])
    if 'ideal_corrected_xco2' in oco.columns:
        _scatter_map(ax21, lon_arr, lat_arr, oco['ideal_corrected_xco2'].values,
                     'Ideal-corrected XCO₂ (ppm)\n(xco2_bc − anomaly)',
                     norm, _CMAP, tccon_lon, tccon_lat,
                     bg_img=bg_img, bg_extent=bg_extent,
                     view_extent=map_extent)
        map_axes.append(ax21)
    else:
        ax21.text(0.5, 0.5, 'xco2_bc_anomaly not available\n(ideal-corrected cannot be computed)',
                  ha='center', va='center', transform=ax21.transAxes, fontsize=9)
        ax21.set_title('Ideal-corrected XCO₂', fontsize=9)

    # Row 2 cols 3+: hide any extra slots not used by fixed panels
    for j in range(3, n_map_cols):
        fig.add_subplot(gs[1, j]).set_visible(False)

    # Row 2 col 2: nearest-cloud distance scatter
    ax22 = fig.add_subplot(gs[1, 2])
    _cld_col = next((c for c in ('cld_dist_km', 'nearest_cloud_dist_km', 'cloud_dist_km', 'min_cloud_dist_km')
                     if c in oco.columns), None)
    if _cld_col:
        _cld_vals = oco[_cld_col].values.astype(float)
        _cld_finite = _cld_vals[np.isfinite(_cld_vals)]
        _cld_vmax = float(np.nanpercentile(_cld_finite, 95)) if len(_cld_finite) > 0 else 100.0
        _cld_norm = mcolors.Normalize(vmin=0.0, vmax=_cld_vmax)
        _scatter_map(ax22, lon_arr, lat_arr, _cld_vals,
                     'Nearest-cloud distance (km)',
                     _cld_norm, 'viridis', tccon_lon, tccon_lat,
                     bg_img=bg_img, bg_extent=bg_extent,
                     view_extent=map_extent)
        _sm_cld = mcm.ScalarMappable(norm=_cld_norm, cmap='viridis')
        _sm_cld.set_array([])
        plt.colorbar(_sm_cld, ax=ax22, fraction=0.046, pad=0.04).set_label('km', fontsize=7)
    else:
        ax22.text(0.5, 0.5, 'Cloud distance not available\n(no nearest_cloud_dist_km column)',
                  ha='center', va='center', transform=ax22.transAxes, fontsize=9)
        ax22.set_title('Nearest-cloud distance', fontsize=9)

    # Row 3: histogram (left portion) | TCCON time series (rightmost map column)
    ax3 = fig.add_subplot(gs[2, :n_map_cols - 1])
    _histogram_panel(ax3, oco, tccon, vmin, vmax)

    ax3_tccon = fig.add_subplot(gs[2, n_map_cols - 1])
    fp_times = None
    if 'time' in oco.columns:
        fp_times = pd.to_datetime(
            oco['time'], unit='s', utc=True, errors='coerce'
        ).dropna()
    _tccon_panel(ax3_tccon, tccon, vmin, vmax,
                 'TCCON XCO₂ (ppm)\n(ground station)',
                 fp_times=fp_times,
                 oco_vals=oco['xco2_bc'].values if 'xco2_bc' in oco.columns else None)

    # ── Shared colorbar for all lon/lat map panels ─────────────────────────────
    if map_axes:
        sm = mcm.ScalarMappable(norm=norm, cmap=_CMAP)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('XCO₂ (ppm)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    else:
        cbar_ax.set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = (Path(output_dir) if output_dir
               else Path(plot_data_path).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'corrected_xco2_vs_tccon.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved → {out_path}", flush=True)


if __name__ == '__main__':
    main()

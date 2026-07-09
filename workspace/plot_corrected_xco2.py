"""plot_corrected_xco2.py — Compare OCO-2 model-corrected XCO2 with a TCCON ground station.

Reads plot_data.csv produced by apply_models.py and a TCCON NetCDF4 file.

Two manuscript figures per run (Arial, plasma XCO2 maps, panel letters):
  corrected_xco2_vs_tccon.png       2×3 compact —
      (a) Lite XCO2 | (b) ML-corrected XCO2 | (c) predicted sigma
      (d) predicted correction mu | (e) histogram | (f) TCCON time series
  corrected_xco2_vs_tccon_full.png  3×3 — adds ideal-corrected and
      nearest-cloud-distance maps in a middle row.
(a)+(b) share one horizontal XCO2 colorbar; map view extent defaults to the
TCCON station ±100 km (--extent-radius-km, 0 = legacy lon/lat-range extent).

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

Poster figure:
    Add --poster-model ridge|mlp|ft|xgboost|hybrid to save a compact
    poster-oriented figure with original XCO₂_bc, the chosen model correction,
    a shared colorbar, and a three-way histogram against TCCON.
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils import get_storage_dir

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (apply_manuscript_style, panel_label, CMAPS,
                        XCO2_LABEL, station_extent)

# ── Styling constants ──────────────────────────────────────────────────────────
# Sequential XCO2 colormap (perceptually uniform, CVD-safe; AMT-friendly).
# Overridable with --cmap (e.g. --cmap jet to reproduce legacy figures).
_CMAP = CMAPS['xco2']

_MODEL_CFGS = [
    ('Ridge',   'ridge_cond_corrected_xco2',   'ridge_corrected_xco2',   'orange'),
    ('MLP',     'mlp_cond_corrected_xco2',     'mlp_corrected_xco2',     'limegreen'),
    ('FT',      'ft_cond_corrected_xco2',      'ft_corrected_xco2',      'purple'),
    ('XGBoost', 'xgb_cond_corrected_xco2',     'xgb_corrected_xco2',     'crimson'),
    ('Hybrid',  'hybrid_cond_corrected_xco2',  'hybrid_corrected_xco2',  'deepskyblue'),
    ('DeepEns',
     'deep_ensemble_cond_corrected_xco2', 'deep_ensemble_corrected_xco2', 'limegreen'),
    ('StructRes',
     'structured_residual_cond_corrected_xco2',
     'structured_residual_corrected_xco2',
     'crimson'),
]

def _resolve_model_cfgs(df):
    """Return (name, col, color) list, preferring cond_corrected_xco2 if present."""
    out = []
    for name, col_cond, col_reg, color in _MODEL_CFGS:
        col = col_cond if col_cond in df.columns else col_reg
        out.append((name, col, color))
    return out


def _model_key(name: str) -> str:
    """Normalize model names/aliases for command-line selection."""
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')


def _select_model_cfg(df: pd.DataFrame, requested: str):
    """Return (name, column, color) for a requested model name or column."""
    if requested in df.columns:
        return requested, requested, 'crimson'

    requested_key = _model_key(requested)
    aliases = {
        'ridge': 'ridge',
        'mlp': 'mlp',
        'ft': 'ft',
        'transformer': 'ft',
        'xgb': 'xgboost',
        'xgboost': 'xgboost',
        'hybrid': 'hybrid',
    }
    requested_key = aliases.get(requested_key, requested_key)

    for name, col, color in _resolve_model_cfgs(df):
        if requested_key in {_model_key(name), _model_key(col)}:
            if col not in df.columns:
                available = ', '.join(
                    f'{nm}: {c}' for nm, c, _ in _resolve_model_cfgs(df)
                    if c in df.columns
                )
                raise ValueError(
                    f"Requested model {requested!r} maps to {col!r}, but that "
                    f"column is not in plot_data. Available models: {available}"
                )
            return name, col, color

    available = ', '.join(
        f'{nm}: {c}' for nm, c, _ in _resolve_model_cfgs(df) if c in df.columns
    )
    raise ValueError(
        f"Unknown poster model {requested!r}. Use one of ridge, mlp, ft, "
        f"xgboost, hybrid, or an explicit column name. Available models: {available}"
    )


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


def _haversine_km(lon, lat, lon0, lat0):
    """Great-circle distance (km) from each (lon, lat) to a single (lon0, lat0)."""
    R = 6371.0088
    lon = np.radians(np.asarray(lon, dtype=float)); lat = np.radians(np.asarray(lat, dtype=float))
    lon0 = np.radians(lon0); lat0 = np.radians(lat0)
    dlon = lon - lon0; dlat = lat - lat0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


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
                 view_extent=None, show_station_legend=True,
                 circle_km=None):
    """lon/lat scatter map coloured by *values*; optionally mark TCCON station.

    Parameters
    ----------
    view_extent : [lon_min, lon_max, lat_min, lat_max] or None
        If provided, the axes are locked to this range after all plotting,
        preventing the scatter from auto-expanding the view.  Derived in
        main() from bg_extent (GIBS/L1B) or --lon-range / --lat-range args.
    show_station_legend : bool
        Draw the 'TCCON station' legend box (the star marker is always drawn).
        Multi-panel grids pass False on all but one panel to avoid repetition.
    circle_km : float, optional
        Draw a dashed circle of this great-circle radius around the TCCON
        station (the collocation area used in the histogram/report analysis).
    """
    if bg_img is not None and bg_extent is not None:
        # extent=[lon_min, lon_max, lat_min, lat_max] → imshow wants [left, right, bottom, top]
        ax.imshow(bg_img,
                  extent=[bg_extent[0], bg_extent[1], bg_extent[2], bg_extent[3]],
                  aspect='auto', origin='upper', zorder=0)
    valid = np.isfinite(values)
    ax.scatter(lon[valid], lat[valid], c=values[valid],
               norm=norm, cmap=cmap, s=9, alpha=0.6, rasterized=True)
    if tccon_lon is not None and tccon_lat is not None:
        ax.scatter([tccon_lon], [tccon_lat], c='red', s=90, marker='*',
                   edgecolors='white', linewidths=0.6,
                   zorder=5, label='TCCON station')
        if circle_km is not None and circle_km > 0:
            import matplotlib.patheffects as _pe
            _th = np.linspace(0.0, 2.0 * np.pi, 361)
            _clat = tccon_lat + circle_km / 111.195 * np.sin(_th)
            _clon = (tccon_lon + circle_km
                     / (111.195 * max(np.cos(np.radians(tccon_lat)), 0.05))
                     * np.cos(_th))
            ax.plot(_clon, _clat, ls='--', lw=1.2, color='white', zorder=4,
                    path_effects=[_pe.Stroke(linewidth=2.4, foreground='black'),
                                  _pe.Normal()],
                    label=f'{circle_km:g} km radius')
        if show_station_legend:
            ax.legend(loc='lower left', markerscale=1.3)
    ax.set_title(title)
    ax.set_xlabel('Lon (°E)')
    ax.set_ylabel('Lat (°N)')
    # Lock axis limits — must come after all plot calls so scatter doesn't override
    if view_extent is not None:
        ax.set_xlim(view_extent[0], view_extent[1])
        ax.set_ylim(view_extent[2], view_extent[3])
        # geographic aspect (equal km in x and y) — keeps the collocation
        # circle round regardless of the grid-cell shape
        _mid_lat = 0.5 * (view_extent[2] + view_extent[3])
        ax.set_aspect(1.0 / max(np.cos(np.radians(_mid_lat)), 0.05))


def _tccon_panel(ax, tccon_df: pd.DataFrame, vmin: float, vmax: float, title: str,
                 fp_times=None):
    """Time series of TCCON XCO₂ with per-measurement error shading.

    The y-limits follow the TCCON ±1σ shading range (1st–99th percentile of
    xco2 ∓/± xco2_error, with a 0.3 ppm margin); vmin/vmax are only the
    fallback when no TCCON data exist.

    Parameters
    ----------
    fp_times : pd.Series of timezone-aware Timestamps, optional
        OCO-2 footprint observation times (after spatial filtering).
        Consecutive times are grouped into orbit passes (gap threshold: 30 min)
        and each pass is shaded as a vertical span.
    """
    ax.set_title(title)
    ax.set_ylabel(f'{XCO2_LABEL} (ppm)')

    if len(tccon_df) == 0:
        ax.text(0.5, 0.5, 'No TCCON data', ha='center', va='center',
                transform=ax.transAxes)
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
    ax.fill_between(t, y - ye, y + ye, color='red', alpha=0.12, zorder=1,
                    label=r'$\pm 1\sigma$')

    # y-limits from the TCCON ±1σ shading range
    _lo = y - np.where(np.isfinite(ye), ye, 0.0)
    _hi = y + np.where(np.isfinite(ye), ye, 0.0)
    _margin = 0.3
    vmin = float(np.nanpercentile(_lo, 1)) - _margin
    vmax = float(np.nanpercentile(_hi, 99)) + _margin
    ax.set_ylim(vmin, vmax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right')
    ax.legend(loc='lower right')

    mu, sigma = float(y.mean()), float(y.std())
    ax.text(0.02, 0.97,
            f'n={len(y):,}  μ={mu:.2f}  σ={sigma:.2f} ppm',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _histogram_panel(ax, oco_df: pd.DataFrame, tccon_df: pd.DataFrame,
                     vmin: float, vmax: float):
    """Overlapping density histograms: original, each model correction, TCCON.

    TCCON is drawn on a twinx right-hand axis so its density scale is
    independent of the model-correction curves.
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
    for name, col, color in _resolve_model_cfgs(oco_df):
        if col in oco_df.columns:
            _draw(ax, oco_df[col], f'{name} corrected', color)
    ax.set_xlabel(f'{XCO2_LABEL} (ppm)')
    ax.set_ylabel('Density')
    ax.set_title(f'{XCO2_LABEL} distribution vs TCCON')

    # ── TCCON on twin right-hand axis ─────────────────────────────────────────
    ax_twin = ax.twinx()
    if len(tccon_df) > 0:
        _draw(ax_twin, tccon_df['xco2'], 'TCCON (ground)', 'red', lw=1.5)
    ax_twin.set_ylabel('Density (TCCON)', color='dimgray')
    ax_twin.tick_params(axis='y', labelcolor='dimgray')

    # Legend placement is the caller's job (compact puts it in the grid cell
    # right below this panel; full uses a figure-bottom legend) — return the
    # merged handles/labels from both axes.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_twin.get_legend_handles_labels()
    return h1 + h2, l1 + l2


def _save_poster_comparison_figure(
        oco_df: pd.DataFrame,
        tccon_df: pd.DataFrame,
        model_name: str,
        model_col: str,
        model_color: str,
        lon_arr,
        lat_arr,
        norm,
        vmin: float,
        vmax: float,
        out_path: Path,
        tccon_lon=None,
        tccon_lat=None,
        bg_img=None,
        bg_extent=None,
        view_extent=None,
        dpi: int = 300,
        hist_df=None,
        ):
    """Save a compact poster figure for one corrected XCO₂ model.

    hist_df : DataFrame for the histogram panel (e.g. footprints near TCCON).
        Defaults to oco_df.  Maps always use oco_df (full lon/lat box).
    """
    if hist_df is None:
        hist_df = oco_df
    if 'xco2_bc' not in oco_df.columns:
        raise ValueError("Poster figure requires an 'xco2_bc' column.")
    if model_col not in oco_df.columns:
        raise ValueError(f"Poster figure requires model column {model_col!r}.")

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
    })

    if view_extent is None:
        finite_lon = np.asarray(lon_arr, dtype=float)
        finite_lat = np.asarray(lat_arr, dtype=float)
        finite_lon = finite_lon[np.isfinite(finite_lon)]
        finite_lat = finite_lat[np.isfinite(finite_lat)]
        if len(finite_lon) > 0 and len(finite_lat) > 0:
            pad_lon = max((finite_lon.max() - finite_lon.min()) * 0.05, 0.05)
            pad_lat = max((finite_lat.max() - finite_lat.min()) * 0.05, 0.05)
            view_extent = [
                float(finite_lon.min() - pad_lon),
                float(finite_lon.max() + pad_lon),
                float(finite_lat.min() - pad_lat),
                float(finite_lat.max() + pad_lat),
            ]

    def _draw_map(ax, values, title, show_ylabel=True):
        if bg_img is not None and bg_extent is not None:
            ax.imshow(
                bg_img,
                extent=[bg_extent[0], bg_extent[1], bg_extent[2], bg_extent[3]],
                aspect='auto',
                origin='upper',
                zorder=0,
            )

        values = np.asarray(values, dtype=float)
        valid = np.isfinite(lon_arr) & np.isfinite(lat_arr) & np.isfinite(values)
        ax.scatter(
            np.asarray(lon_arr)[valid],
            np.asarray(lat_arr)[valid],
            c=values[valid],
            cmap=_CMAP,
            norm=norm,
            s=14,
            alpha=0.68,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        if tccon_lon is not None and tccon_lat is not None:
            ax.scatter(
                [tccon_lon], [tccon_lat],
                c='white',
                edgecolors='black',
                linewidths=1.4,
                s=330,
                marker='*',
                zorder=5,
                label='TCCON',
            )
            ax.legend(
                loc='upper right',
                fontsize=18,
                frameon=True,
            )

        ax.set_title(title, fontsize=23, weight='bold', pad=14)
        ax.set_xlabel('Longitude ($^o$E)', fontsize=18)
        ax.set_ylabel('Latitude ($^o$N)' if show_ylabel else '', fontsize=18)
        ax.tick_params(labelsize=16)
        if view_extent is not None:
            ax.set_xlim(view_extent[0], view_extent[1])
            ax.set_ylim(view_extent[2], view_extent[3])
        ax.grid(alpha=0.18, linewidth=0.6)

    fig = plt.figure(figsize=(23.5, 6.2))
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1.0, 1.0, 0.045, 0.12, 1.08],
        wspace=0.30,
    )

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[0, 4])

    _draw_map(ax_orig, oco_df['xco2_bc'].values, 'OCO-2 Lite $X_{\mathrm{CO2}}$')
    _draw_map(
        ax_corr,
        oco_df[model_col].values,
        'ML-corrected $X_{\mathrm{CO2}}$',
        show_ylabel=False,
    )

    sm = mcm.ScalarMappable(norm=norm, cmap=_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('$X_{\mathrm{CO2}}$ (ppm)', fontsize=20, labelpad=16)
    cbar.ax.tick_params(labelsize=16)

    bins = np.linspace(vmin, vmax, 80)

    def _draw_hist(values, label, color, linestyle='-'):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return
        mu = float(values.mean())
        sigma = float(values.std())
        ax_hist.hist(
            values,
            bins=bins,
            density=True,
            histtype='step',
            linewidth=2.6,
            color=color,
            alpha=0.6,
            linestyle=linestyle,
            label=f'{label}: {mu:.2f} $\\pm$ {sigma:.2f} ppm',
        )
        ax_hist.axvline(mu, color=color, linewidth=3.0, alpha=0.95)

    _draw_hist(hist_df['xco2_bc'].values, 'OCO-2 Lite $X_{\mathrm{CO2}}$', 'black')
    _draw_hist(hist_df[model_col].values, 'Corrected $X_{\mathrm{CO2}}$', 'green')
    if len(tccon_df) > 0:
        _draw_hist(tccon_df['xco2'].values, 'TCCON $X_{\mathrm{CO2}}$', 'coral', linestyle='--')

    ax_hist.set_title('$X_{\mathrm{CO2}}$ distributions', fontsize=23, weight='bold', pad=14)
    ax_hist.set_xlabel('$X_{\mathrm{CO2}}$ (ppm)', fontsize=18)
    ax_hist.set_ylabel('Density', fontsize=18)
    ax_hist.tick_params(labelsize=16)
    ax_hist.grid(alpha=0.22, linewidth=0.7)
    ax_hist.legend(
        fontsize=17,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
    )
    ax_hist.set_xlim(vmin, vmax)

    if bg_img is None:
        fig.text(
            0.5, 0.02,
            'MODIS background was not available; maps show OCO-2 soundings only.',
            ha='center',
            fontsize=10,
            color='dimgray',
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _compose_case_figure(*, full, oco, oco_hist, tccon, active,
                         lon_arr, lat_arr, norm, vmin, vmax,
                         tccon_lon, tccon_lat, bg_img, bg_extent,
                         map_extent, fp_times, radius_km, out_path, dpi):
    """Manuscript case figure.

    Compact (full=False):
        (a) OCO-2 Lite XCO2 | (b) ML-corrected XCO2 | (c) predicted sigma
        (d) predicted correction mu | (e) histogram | (f) TCCON time series
    Full (full=True) adds a middle map row:
        (d) mu | (e) ideal-corrected | (f) nearest-cloud distance
        (g) histogram (2 cols) | (h) TCCON time series

    Every map panel has its own horizontal colorbar in a dedicated thin grid
    row ((a)/(b) share the same normalisation but separate bars) so all
    overlaid maps render at identical size.  A dashed circle marks the
    ±radius_km TCCON collocation area the histogram/report analysis uses.
    """
    primary = next(((nm, col) for nm, col, _ in active if 'deep_ensemble' in col),
                   (active[0][0], active[0][1]) if active else None)

    if full:
        fig = plt.figure(figsize=(13.0, 14.2), layout='constrained')
        gs = fig.add_gridspec(5, 3, height_ratios=[1, 0.07, 1, 0.07, 0.85])
    else:
        fig = plt.figure(figsize=(13.0, 9.6), layout='constrained')
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 0.07, 1, 0.07])

    import string as _string
    _tags = iter(f'({c})' for c in _string.ascii_lowercase)
    _cb_snap = []   # (map_ax, cbar_ax) pairs — widths matched after layout

    def _map_panel(cell, cax_cell, values, title, norm_, cmap_, cbar_label,
                   legend=False):
        ax = fig.add_subplot(cell)
        _scatter_map(ax, lon_arr, lat_arr, np.asarray(values, dtype=float),
                     title, norm_, cmap_, tccon_lon, tccon_lat,
                     bg_img=bg_img, bg_extent=bg_extent,
                     view_extent=map_extent, show_station_legend=legend,
                     circle_km=radius_km)
        panel_label(ax, next(_tags))
        sm = mcm.ScalarMappable(norm=norm_, cmap=cmap_)
        sm.set_array([])
        cax = fig.add_subplot(cax_cell)
        fig.colorbar(sm, cax=cax, orientation='horizontal').set_label(cbar_label)
        _cb_snap.append((ax, cax))
        return ax

    def _text_panel(cell, msg, title):
        ax = fig.add_subplot(cell)
        ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax

    xco2_ppm = f'{XCO2_LABEL} (ppm)'

    # ── Row 0: Lite | ML-corrected | predicted σ (all bars horizontal) ───────
    if 'xco2_bc' in oco.columns:
        _map_panel(gs[0, 0], gs[1, 0], oco['xco2_bc'].values,
                   f'OCO-2 Lite {XCO2_LABEL}', norm, _CMAP, xco2_ppm,
                   legend=True)
    else:
        _text_panel(gs[0, 0], 'xco2_bc not in plot_data',
                    f'OCO-2 Lite {XCO2_LABEL}')
    if primary is not None:
        _map_panel(gs[0, 1], gs[1, 1], oco[primary[1]].values,
                   f'ML-corrected {XCO2_LABEL}', norm, _CMAP, xco2_ppm)
    else:
        _text_panel(gs[0, 1], 'no corrected-XCO2 column',
                    f'ML-corrected {XCO2_LABEL}')

    sigma_col = next((c for c in ('sigma', 'corrected_xco2_sigma', 'pred_sigma')
                      if c in oco.columns), None)
    if sigma_col is not None:
        _sig = oco[sigma_col].to_numpy(dtype=float)
        _fin = _sig[np.isfinite(_sig)]
        _smax = float(np.nanpercentile(_fin, 98)) if len(_fin) else 1.0
        _map_panel(gs[0, 2], gs[1, 2], _sig, 'Predicted uncertainty',
                   mcolors.Normalize(vmin=0.0, vmax=max(_smax, 1e-3)),
                   CMAPS['sigma'], 'σ (ppm)')
    else:
        _text_panel(gs[0, 2], 'no σ column', 'Predicted uncertainty')

    # ── μ map (predicted correction, diverging) ──────────────────────────────
    if primary is not None and 'xco2_bc' in oco.columns:
        _mu = (oco['xco2_bc'].to_numpy(dtype=float)
               - oco[primary[1]].to_numpy(dtype=float))
        _fin = np.abs(_mu[np.isfinite(_mu)])
        _mmax = max(float(np.nanpercentile(_fin, 98)) if len(_fin) else 1.0, 1e-3)
        _map_panel(gs[2, 0], gs[3, 0], _mu, 'Predicted correction',
                   mcolors.Normalize(vmin=-_mmax, vmax=_mmax),
                   CMAPS['mu'], 'μ (ppm)')
    else:
        _text_panel(gs[2, 0], 'correction not available', 'Predicted correction')

    if full:
        # ── Full variant: ideal-corrected + cloud distance ────────────────────
        if 'ideal_corrected_xco2' in oco.columns:
            _map_panel(gs[2, 1], gs[3, 1], oco['ideal_corrected_xco2'].values,
                       f'Ideal-corrected {XCO2_LABEL}', norm, _CMAP, xco2_ppm)
        else:
            _text_panel(gs[2, 1], 'xco2_bc_anomaly not available',
                        f'Ideal-corrected {XCO2_LABEL}')

        _cld_col = next((c for c in ('cld_dist_km', 'nearest_cloud_dist_km',
                                     'cloud_dist_km', 'min_cloud_dist_km')
                         if c in oco.columns), None)
        if _cld_col is not None:
            _cld = oco[_cld_col].to_numpy(dtype=float)
            _fin = _cld[np.isfinite(_cld)]
            _cmax = float(np.nanpercentile(_fin, 95)) if len(_fin) else 100.0
            _map_panel(gs[2, 2], gs[3, 2], _cld, 'Nearest-cloud distance',
                       mcolors.Normalize(vmin=0.0, vmax=_cmax),
                       CMAPS['cld_dist'], 'km')
        else:
            _text_panel(gs[2, 2], 'no cloud-distance column',
                        'Nearest-cloud distance')

        ax_h = fig.add_subplot(gs[4, :2])
        ax_t = fig.add_subplot(gs[4, 2])
    else:
        ax_h = fig.add_subplot(gs[2, 1])
        ax_t = fig.add_subplot(gs[2, 2])

    _hh, _ll = _histogram_panel(ax_h, oco_hist, tccon, vmin, vmax)
    panel_label(ax_h, next(_tags))
    _tccon_panel(ax_t, tccon, vmin, vmax, f'TCCON {XCO2_LABEL}',
                 fp_times=fp_times)
    panel_label(ax_t, next(_tags))

    # histogram legend: compact → the empty grid cell directly below the
    # histogram; full → figure bottom (directly below the spanning histogram)
    if full:
        fig.legend(_hh, _ll, ncol=1, loc='outside lower center', frameon=False)
    else:
        lax = fig.add_subplot(gs[3, 1])
        lax.axis('off')
        lax.legend(_hh, _ll, ncol=1, loc='upper center',
                   bbox_to_anchor=(0.5, 1.0), frameon=False)

    # Solve the constrained layout, then freeze it and snap every horizontal
    # colorbar to the width of its (aspect-locked) map panel.
    fig.canvas.draw()
    fig.set_layout_engine('none')
    for _ax, _cax in _cb_snap:
        _p = _ax.get_position()
        _c = _cax.get_position()
        _cax.set_position([_p.x0, _c.y0, _p.width, _c.height])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved → {out_path}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global _CMAP
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
    parser.add_argument('--poster-model', default=None,
                        help='Save an additional poster-oriented figure for one model. '
                             'Accepts ridge, mlp, ft, xgb/xgboost, hybrid, or an explicit '
                             'corrected-XCO2 column name.')
    parser.add_argument('--poster-output', default=None,
                        help='Output path for --poster-model figure '
                             '(default: poster_xco2_<model>.png in --output-dir).')
    parser.add_argument('--poster-dpi', type=int, default=300,
                        help='DPI for --poster-model figure (default: 300).')
    parser.add_argument('--hist-radius-km', type=float, default=None,
                        help='Restrict the histogram comparison to footprints within this '
                             'great-circle distance (km) of the TCCON station. Maps and the '
                             'time series still use the full lon/lat box. Default: no limit.')
    parser.add_argument('--tccon-window-min', type=float, default=60.0,
                        help='Half-width (minutes) of the TCCON time window around the OCO-2 '
                             'footprint pass: TCCON is kept within [first footprint − W, '
                             'last footprint + W]. Default: 60.')
    parser.add_argument('--extent-radius-km', type=float, default=100.0,
                        help='Map view (and GIBS download) extent = TCCON station '
                             '± this radius, matching the collocation radius '
                             '(default 100). 0 disables → legacy lon/lat-range extent.')
    parser.add_argument('--cmap', default=_CMAP,
                        help=f"XCO2 colormap for the map panels (default {_CMAP}; "
                             "'jet' reproduces the legacy figures).")
    parser.add_argument('--dpi', type=int, default=300,
                        help='Main-figure DPI (default 300, manuscript-ready).')
    args = parser.parse_args()
    _CMAP = args.cmap
    apply_manuscript_style()   # Arial (AMT), Arial mathtext, thin axes, 300 dpi

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
    poster_output  = _abs(args.poster_output)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading plot data: {plot_data_path}", flush=True)
    oco = load_plot_data(plot_data_path)
    print(f"  OCO-2: {len(oco):,} rows", flush=True)

    # Drop guarded footprints (non-physical retrievals / model blow-ups the
    # correction was withheld from) — keeping them pollutes the maps and inflates
    # the histogram σ (e.g. fill-value xco2_bc up to thousands of ppm).
    _guard_cols = [c for c in ('clim_guard', 'anomaly_guard') if c in oco.columns]
    if _guard_cols:
        _g = np.zeros(len(oco), dtype=bool)
        for c in _guard_cols:
            _g |= oco[c].to_numpy(dtype=bool)
        if _g.any():
            print(f"  dropping {int(_g.sum()):,} guarded footprint(s) "
                  f"({'+'.join(_guard_cols)}) from the plot", flush=True)
            oco = oco[~_g].reset_index(drop=True)

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
        plt.ylabel('$X_{\mathrm{CO2}}$ (ppm)')
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
            # buffer on either side of the footprint pass to account for time mismatches
            buffer = pd.Timedelta(minutes=args.tccon_window_min)
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

    # Station-centred view extent (± collocation radius) — wins over lon/lat
    # ranges for the MAP VIEW only (data filters above are unchanged).
    station_ext = None
    if args.extent_radius_km and args.extent_radius_km > 0 and tccon_lon is not None:
        station_ext = station_extent(tccon_lon, tccon_lat, args.extent_radius_km)
        print(f"  View extent: station ±{args.extent_radius_km:g} km → "
              f"lon [{station_ext[0]:.2f}, {station_ext[1]:.2f}] "
              f"lat [{station_ext[2]:.2f}, {station_ext[3]:.2f}]", flush=True)

    # ── Shared colour range ───────────────────────────────────────────────────
    _pool = []
    for col in ('xco2_bc', 'ridge_corrected_xco2', 'mlp_corrected_xco2',
                'ft_corrected_xco2', 'xgb_corrected_xco2', 'hybrid_corrected_xco2',
                'deep_ensemble_corrected_xco2', 'ideal_corrected_xco2'):
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
    active = [(nm, col, clr) for nm, col, clr in _resolve_model_cfgs(oco) if col in oco.columns]

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
                # Station ±radius wins; then explicit --lon-range / --lat-range;
                # otherwise derive from the granule's soundings (±20 min),
                # falling back to all data.
                if station_ext is not None:
                    gibs_extent = list(station_ext)
                elif args.lon_range and args.lat_range:
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
                _explicit_range = bool(station_ext is not None
                                       or args.lon_range or args.lat_range)
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
    # Priority: station ±radius > --lon-range/--lat-range > bg_extent > None
    if station_ext is not None:
        map_extent = list(station_ext)
    elif args.lon_range and args.lat_range:
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

    # ── Histogram subset: footprints within --hist-radius-km of TCCON station ──
    oco_hist = oco
    if args.hist_radius_km is not None and tccon_lon is not None and 'lon' in oco.columns:
        _d = _haversine_km(oco['lon'].values, oco['lat'].values, tccon_lon, tccon_lat)
        oco_hist = oco[_d <= args.hist_radius_km]
        print(f"  Histogram restricted to ≤{args.hist_radius_km:g} km of TCCON: "
              f"{len(oco):,} → {len(oco_hist):,} footprints", flush=True)

    fp_times = None
    if 'time' in oco.columns:
        fp_times = pd.to_datetime(
            oco['time'], unit='s', utc=True, errors='coerce'
        ).dropna()

    # ── Figures: compact manuscript version + full (ideal + cloud distance) ──
    out_dir = (Path(output_dir) if output_dir
               else Path(plot_data_path).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    _fig_kw = dict(oco=oco, oco_hist=oco_hist, tccon=tccon, active=active,
                   lon_arr=lon_arr, lat_arr=lat_arr, norm=norm,
                   vmin=vmin, vmax=vmax,
                   tccon_lon=tccon_lon, tccon_lat=tccon_lat,
                   bg_img=bg_img, bg_extent=bg_extent, map_extent=map_extent,
                   fp_times=fp_times, radius_km=args.hist_radius_km,
                   dpi=args.dpi)
    _compose_case_figure(full=False, out_path=out_dir / 'corrected_xco2_vs_tccon.png',
                         **_fig_kw)
    _compose_case_figure(full=True,
                         out_path=out_dir / 'corrected_xco2_vs_tccon_full.png',
                         **_fig_kw)

    if args.poster_model:
        model_name, model_col, model_color = _select_model_cfg(oco, args.poster_model)
        if poster_output:
            poster_path = Path(poster_output)
        else:
            model_slug = _model_key(model_name)
            poster_path = out_dir / f'poster_xco2_{model_slug}.png'

        print(f"Saving poster figure for {model_name} ({model_col}) → {poster_path}",
              flush=True)
        _save_poster_comparison_figure(
            oco,
            tccon,
            model_name,
            model_col,
            model_color,
            lon_arr,
            lat_arr,
            norm,
            vmin,
            vmax,
            poster_path,
            tccon_lon=tccon_lon,
            tccon_lat=tccon_lat,
            bg_img=bg_img,
            bg_extent=bg_extent,
            view_extent=map_extent,
            dpi=args.poster_dpi,
            hist_df=oco_hist,
        )
        print(f"Poster figure saved → {poster_path}", flush=True)


if __name__ == '__main__':
    main()

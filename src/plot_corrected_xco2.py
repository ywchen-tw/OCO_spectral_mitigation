"""plot_corrected_xco2.py — Compare OCO-2 model-corrected XCO2 with a TCCON ground station.

Reads plot_data.csv produced by apply_models.py and a TCCON NetCDF4 file.

Figure layout (3 rows × 3 columns):
  Row 1 : lon/lat scatter maps — Ridge-corrected | MLP-corrected | FT-corrected XCO₂
  Row 2 : lon/lat scatter maps — original XCO₂_bc | ideal-corrected | TCCON time series
  Row 3 : histogram — all corrected XCO₂ distributions vs TCCON

Shared colorbar covers all lon/lat map panels (rows 1–2, cols 0–1).

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
_CMAP = 'plasma'

_MODEL_CFGS = [
    ('Ridge', 'ridge_corrected_xco2', 'orange'),
    ('MLP',   'mlp_corrected_xco2',   'limegreen'),
    ('FT',    'ft_corrected_xco2',    'purple'),
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

    times = pd.to_datetime(time_sec, unit='s', utc=True)
    df = pd.DataFrame({
        'time':       times,
        'lat':        lat_arr,
        'lon':        lon_arr,
        'xco2':       xco2_arr,
        'xco2_error': xco2e_arr,
    })
    valid = (df['xco2'] > 350) & (df['xco2'] < 450) & df['xco2'].notna()
    return df[valid].reset_index(drop=True)


def load_plot_data(csv_path: str) -> pd.DataFrame:
    """Load plot_data.csv; compute ideal-corrected column if anomaly is present."""
    df = pd.read_csv(csv_path)
    if 'xco2_bc' in df.columns and 'xco2_bc_anomaly' in df.columns:
        df['ideal_corrected_xco2'] = df['xco2_bc'] - df['xco2_bc_anomaly']
    return df


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _scatter_map(ax, lon, lat, values, title, norm, cmap,
                 tccon_lon=None, tccon_lat=None):
    """lon/lat scatter map coloured by *values*; optionally mark TCCON station."""
    valid = np.isfinite(values)
    ax.scatter(lon[valid], lat[valid], c=values[valid],
               norm=norm, cmap=cmap, s=1, alpha=0.5, rasterized=True)
    if tccon_lon is not None and tccon_lat is not None:
        ax.scatter([tccon_lon], [tccon_lat], c='red', s=80, marker='*',
                   zorder=5, label='TCCON station')
        ax.legend(fontsize=7, loc='lower right', markerscale=1.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Lon (°E)', fontsize=8)
    ax.set_ylabel('Lat (°N)', fontsize=8)
    ax.tick_params(labelsize=7)


def _tccon_panel(ax, tccon_df: pd.DataFrame, vmin: float, vmax: float, title: str):
    """Time series of TCCON XCO₂ with per-measurement error and monthly mean."""
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

    # Raw measurements + ±1σ shading
    ax.scatter(t, y, s=1, c='salmon', alpha=0.4, zorder=2, label='Measurements')
    ax.fill_between(t, y - ye, y + ye, color='red', alpha=0.12, zorder=1)

    # Monthly mean (group by year+month to avoid resample API version issues)
    tccon_df2 = tccon_df.copy()
    tccon_df2['_ym'] = tccon_df2['time'].dt.year * 100 + tccon_df2['time'].dt.month
    mon = (tccon_df2.groupby('_ym')
                    .agg(xco2=('xco2', 'mean'), time=('time', 'mean'))
                    .reset_index())
    ax.plot(mon['time'], mon['xco2'], '-', color='darkred',
            linewidth=1.2, zorder=3, label='Monthly mean')

    ax.set_ylim(vmin, vmax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
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
    """Overlapping density histograms: original, ideal, each model, TCCON."""
    bins = np.linspace(vmin, vmax, 120)

    def _draw(vals, label: str, color: str, lw: float = 1.2):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return
        mu, sigma = v.mean(), v.std()
        ax.hist(v, bins=bins, density=True, color=color, alpha=0.35,
                label=f'{label}  μ={mu:.2f} σ={sigma:.2f} ppm')
        ax.axvline(mu,          color=color, linewidth=lw)
        ax.axvline(mu - sigma,  color=color, linewidth=lw * 0.7, linestyle=':')
        ax.axvline(mu + sigma,  color=color, linewidth=lw * 0.7, linestyle=':')

    if 'xco2_bc' in oco_df.columns:
        _draw(oco_df['xco2_bc'],             'OCO-2 original',   'steelblue')
    if 'ideal_corrected_xco2' in oco_df.columns:
        _draw(oco_df['ideal_corrected_xco2'], 'Ideal corrected',  'gray')
    for name, col, color in _MODEL_CFGS:
        if col in oco_df.columns:
            _draw(oco_df[col], f'{name} corrected', color)
    if len(tccon_df) > 0:
        _draw(tccon_df['xco2'], 'TCCON (ground)', 'red', lw=1.5)

    ax.set_xlabel('XCO₂ (ppm)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('XCO₂ distribution — OCO-2 corrected vs TCCON ground station',
                 fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc='upper left')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot OCO-2 model-corrected XCO2 vs TCCON ground station.'
    )
    parser.add_argument('--plot-data',  required=True,
                        help='plot_data.csv produced by apply_models.py')
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
    parser.add_argument('--date-range', nargs=2, default=None,
                        metavar=('START', 'END'),
                        help='TCCON date range filter, e.g. 2018-01-01 2020-12-31')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Force colorbar/histogram lower bound (ppm)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Force colorbar/histogram upper bound (ppm)')
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
    if args.date_range:
        t0 = pd.Timestamp(args.date_range[0], tz='UTC')
        t1 = pd.Timestamp(args.date_range[1], tz='UTC')
        tccon = tccon[(tccon['time'] >= t0) & (tccon['time'] <= t1)]

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
                'ft_corrected_xco2', 'ideal_corrected_xco2'):
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

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 16))
    gs  = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.30,
                            height_ratios=[1.0, 1.0, 0.75])

    map_axes = []   # collect lon/lat map axes for the shared colorbar

    # Row 1: model-corrected maps
    for j, (name, col, _) in enumerate(active[:3]):
        ax = fig.add_subplot(gs[0, j])
        vals = oco[col].values
        _scatter_map(ax, lon_arr, lat_arr, vals,
                     f'{name}-corrected XCO₂ (ppm)',
                     norm, _CMAP, tccon_lon, tccon_lat)
        map_axes.append(ax)
    for j in range(len(active), 3):          # hide unused slots
        fig.add_subplot(gs[0, j]).set_visible(False)

    # Row 2 col 0: original XCO₂_bc
    ax20 = fig.add_subplot(gs[1, 0])
    if 'xco2_bc' in oco.columns:
        _scatter_map(ax20, lon_arr, lat_arr, oco['xco2_bc'].values,
                     'Original XCO₂_bc (ppm)',
                     norm, _CMAP, tccon_lon, tccon_lat)
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
                     norm, _CMAP, tccon_lon, tccon_lat)
        map_axes.append(ax21)
    else:
        ax21.text(0.5, 0.5, 'xco2_bc_anomaly not available\n(ideal-corrected cannot be computed)',
                  ha='center', va='center', transform=ax21.transAxes, fontsize=9)
        ax21.set_title('Ideal-corrected XCO₂', fontsize=9)

    # Row 2 col 2: TCCON time series (ground station, fixed location)
    ax22 = fig.add_subplot(gs[1, 2])
    _tccon_panel(ax22, tccon, vmin, vmax,
                 'TCCON XCO₂ (ppm)\n(ground station)')

    # Row 3: histogram spanning all 3 columns
    ax3 = fig.add_subplot(gs[2, :])
    _histogram_panel(ax3, oco, tccon, vmin, vmax)

    # ── Shared colorbar for all lon/lat map panels ─────────────────────────────
    if map_axes:
        sm = mcm.ScalarMappable(norm=norm, cmap=_CMAP)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=map_axes,
                            orientation='vertical', shrink=0.85,
                            aspect=30, pad=0.02)
        cbar.set_label('XCO₂ (ppm)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

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

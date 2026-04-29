"""analyze_area.py — Spatial area analysis of OCO-2 footprint spectral coefficients.

Given a lat/lon range and date, downloads a MODIS true-colour background and produces
scatter / hexbin plots for all footprints combined and each of fp_0..fp_7:

  (4) k1/k2/k3 vs albedo         — scatter colored by nearest_cld_distance
  (5) k1/k2/k3 vs exp_intercept  — scatter colored by nearest_cld_distance
  (6) exp_intercept vs albedo     — hexbin density
  (6b) exp_intercept vs albedo    — scatter colored by nearest cloud distance
  map  — MODIS background + footprint positions colored by nearest cloud distance
  map_<band>_<k>.png             — 7 maps: o2a/wco2/sco2 k1&k2 + xco2_bc on MODIS bg

All three spectral bands (O2A, WCO2, SCO2) are shown in each figure.

Usage
-----
    python workspace/analyze_area.py \\
        --lon-range 100 130 \\
        --lat-range -10 20  \\
        --date 2020-01-01   \\
        [--parquet-fname combined_2020-01-01_all_orbits.parquet] \\
        [--output-dir /path/to/output] \\
        [--modis-which aqua|terra] \\
        [--no-modis]

Output layout (under --output-dir/<date>/)
    modis_<which>_<date>.png        — MODIS true-colour background
    {subset}/map.png                — footprint positions on MODIS background
    {subset}/k_vs_alb.png          — (4) kN vs albedo, 3×3 grid
    {subset}/k_vs_exp.png          — (5) kN vs exp_intercept, 3×3 grid
    {subset}/exp_vs_alb.png        — (6) exp_intercept vs albedo, 1×3 hexbin
    {subset}/exp_vs_alb_cld.png    — (6b) exp_intercept vs albedo, colored by cld dist
    {subset}/map_<band>_k1.png     — k1 for each band overlaid on map
    {subset}/map_<band>_k2.png     — k2 for each band overlaid on map
    {subset}/map_xco2_bc.png       — xco2_bc overlaid on map

where {subset} = all | fp_0 | fp_1 | ... | fp_7
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats as _stats

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils import get_storage_dir

# ── Band / k-term definitions ──────────────────────────────────────────────────
BANDS = [
    ('o2a',  r'$O_2A$',  'C0'),
    ('wco2', r'$WCO_2$', 'C1'),
    ('sco2', r'$SCO_2$', 'C2'),
]
K_TERMS = [('k1', r"$\langle l'\rangle$"), ('k2', r"$\mathrm{var}(l')$"), ('k3', r'$k_3$')]

_MAX_SCATTER_PTS = 8_000   # subsample cap for colored scatter plots


# ── MODIS background ───────────────────────────────────────────────────────────

def _download_modis_bg(date_str: str, extent: list, out_dir: Path, which: str = 'aqua') -> str | None:
    """Download MODIS true-colour tile from NASA GIBS; return PNG path or None."""
    try:
        from owslib.wmts import WebMapTileService
        import cartopy.crs as ccrs
    except ImportError as exc:
        print(f"  MODIS download skipped ({exc})", flush=True)
        return None

    layer = ('MODIS_Aqua_CorrectedReflectance_TrueColor' if which == 'aqua'
             else 'MODIS_Terra_CorrectedReflectance_TrueColor')
    date_s = pd.Timestamp(date_str).strftime('%Y-%m-%d')
    fname  = str(out_dir / f'{which}_rgb_{date_s}.png')

    dlon = extent[1] - extent[0]
    dlat = extent[3] - extent[2]
    base_w = 12.0
    wmts = WebMapTileService('https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi')
    fig = plt.figure(figsize=(base_w, max(2.0, base_w * dlat / dlon)))
    ax  = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_wmts(wmts, layer, wmts_kwargs={'time': date_s})
    ax.coastlines(resolution='10m', color='black', linewidth=0.5, alpha=0.8)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.patch.set_visible(False)
    ax.axis('off')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"  MODIS RGB saved → {fname}", flush=True)
    return fname


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_and_filter(parquet_path: Path, lon_range, lat_range) -> pd.DataFrame:
    print(f"Loading {parquet_path}", flush=True)
    df = pd.read_parquet(parquet_path)
    if lon_range:
        df = df[(df['lon'] >= lon_range[0]) & (df['lon'] <= lon_range[1])]
    if lat_range:
        df = df[(df['lat'] >= lat_range[0]) & (df['lat'] <= lat_range[1])]
    print(f"  {len(df):,} soundings after spatial filter", flush=True)
    return df


# ── Footprint subsetting ───────────────────────────────────────────────────────

def _subset_for_fp(df: pd.DataFrame, fp_idx: int) -> pd.DataFrame | None:
    """Return subset for one footprint (0-based).  Returns None if no FP column found."""
    fp_col = f'fp_{fp_idx}'
    if fp_col in df.columns:
        return df[df[fp_col] == 1]

    for num_col in ('fp_number', 'fp', 'footprint', 'footprint_id'):
        if num_col not in df.columns:
            continue
        vals = df[num_col].dropna().astype(int)
        if vals.empty:
            return df.iloc[0:0]
        offset = 1 if int(vals.min()) == 1 else 0
        return df[df[num_col].astype(int) == (fp_idx + offset)]

    return None


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _fig_for_extent(bg_img, extent: list, base_w: float = 6.0):
    """Return (fig, ax) with figsize matching the bg_img ratio (or extent lon/lat ratio)."""
    if bg_img is not None:
        img_h, img_w = bg_img.shape[:2]
        fig_h = base_w * img_h / img_w
    else:
        dlon = max(extent[1] - extent[0], 1e-3)
        dlat = max(extent[3] - extent[2], 1e-3)
        fig_h = max(2.0, base_w * dlat / dlon)
    return plt.subplots(figsize=(base_w, fig_h))


def _plot_map(df: pd.DataFrame, bg_img, extent: list, outpath: Path, title: str,
             pt_size: float = 12):
    """Footprint scatter map on MODIS background, colored by nearest cloud distance."""
    fig, ax = _fig_for_extent(bg_img, extent)
    if bg_img is not None:
        ax.imshow(bg_img,
                  extent=[extent[0], extent[1], extent[2], extent[3]],
                  aspect='equal', origin='upper', zorder=0)

    cld = (df['cld_dist_km'].values.astype(float)
           if 'cld_dist_km' in df.columns else np.zeros(len(df)))
    fin = cld[np.isfinite(cld)]
    vmax = float(np.nanpercentile(fin, 95)) if len(fin) > 0 else 50.0
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    sc = ax.scatter(df['lon'].values, df['lat'].values,
                    c=cld, cmap='viridis_r', norm=norm,
                    s=pt_size, alpha=0.65, rasterized=True, zorder=2)
    cax = make_axes_locatable(ax).append_axes('right', size='6%', pad=0.2)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label('Nearest cloud distance (km)', fontsize=20)
    cb.ax.tick_params(labelsize=18)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('Lon (°E)', fontsize=20)
    ax.set_ylabel('Lat (°N)', fontsize=20)
    ax.tick_params(labelsize=18)
    ax.set_title(title, fontsize=22, pad=26)
    ax.text(0.96, 0.98, f'n = {len(df):,}', transform=ax.transAxes,
            ha='right', va='top', fontsize=18,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    map → {outpath.name}", flush=True)


def _scatter_cld_colored(ax, x, y, cld, xlabel: str, ylabel: str, title: str):
    """Scatter of y vs x; each point colored by cld (nearest cloud distance)."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(cld)
    xm, ym, cm = x[mask], y[mask], cld[mask]
    if len(xm) == 0:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return

    # subsample to avoid overplotting
    if len(xm) > _MAX_SCATTER_PTS:
        idx = np.random.default_rng(42).choice(len(xm), _MAX_SCATTER_PTS, replace=False)
        xm, ym, cm = xm[idx], ym[idx], cm[idx]

    fin = cm[np.isfinite(cm)]
    vmax = float(np.nanpercentile(fin, 95)) if len(fin) > 0 else 50.0
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    sc = ax.scatter(xm, ym, c=cm, cmap='jet_r', norm=norm,
                    s=25, alpha=0.6, rasterized=True, linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label('cld (km)', fontsize=7)

    full_mask = np.isfinite(x) & np.isfinite(y)
    r_str = ''
    if full_mask.sum() > 2:
        r, _ = _stats.pearsonr(x[full_mask], y[full_mask])
        r_str = f'  r={r:.3f}'
    ax.set_title(f'{title}{r_str}', fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=8)


def _plot_k_vs_x(df: pd.DataFrame, x_prefix: str, x_label_fn,
                 outpath: Path, fig_title: str):
    """Generic 3×3 scatter grid: rows=k1/k2/k3, cols=O2A/WCO2/SCO2.

    x_prefix : 'alb' or 'exp_{bp}_intercept' pattern key
    x_label_fn : callable(bp, bname) → x column name and x-axis label
    """
    cld = (df['cld_dist_km'].values.astype(float)
           if 'cld_dist_km' in df.columns else np.zeros(len(df)))

    avail_k = [(kt, kl) for kt, kl in K_TERMS
               if any(f'{bp}_{kt}' in df.columns for bp, _, _ in BANDS)]
    if not avail_k:
        print(f"    skipping {outpath.name} — no k columns", flush=True)
        return

    nrows, ncols = len(avail_k), len(BANDS)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for ri, (kt, kl) in enumerate(avail_k):
        for ci, (bp, bname, _) in enumerate(BANDS):
            ax = axes[ri, ci]
            k_col = f'{bp}_{kt}'
            x_col, x_label = x_label_fn(bp, bname)
            if k_col not in df.columns or x_col not in df.columns:
                ax.set_visible(False)
                continue
            _scatter_cld_colored(
                ax,
                df[x_col].values.astype(float),
                df[k_col].values.astype(float),
                cld,
                xlabel=x_label,
                ylabel=f'{bname} {kl}',
                title=f'{bname} {kl}',
            )

    fig.suptitle(fig_title, fontsize=12)
    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    {outpath.name}", flush=True)


def _plot_k_vs_alb(df: pd.DataFrame, outpath: Path, title_prefix: str = ''):
    """(4) kN vs albedo, colored by nearest cloud distance."""
    def _x(bp, bname):
        return f'alb_{bp}', f'{bname} albedo'

    _plot_k_vs_x(df, 'alb', _x, outpath,
                 f'{title_prefix}k vs albedo  (color = nearest cloud distance km)')


def _plot_k_vs_exp(df: pd.DataFrame, outpath: Path, title_prefix: str = ''):
    """(5) kN vs exp_intercept, colored by nearest cloud distance."""
    def _x(bp, bname):
        return f'exp_{bp}_intercept', f'{bname} exp_intercept'

    _plot_k_vs_x(df, 'exp', _x, outpath,
                 f'{title_prefix}k vs exp_intercept  (color = nearest cloud distance km)')


def _plot_exp_vs_alb(df: pd.DataFrame, outpath: Path, title_prefix: str = ''):
    """(6) exp_intercept vs albedo, hexbin density, 1×3 grid (3 bands)."""
    avail = [(bp, bname)
             for bp, bname, _ in BANDS
             if f'alb_{bp}' in df.columns and f'exp_{bp}_intercept' in df.columns]
    if not avail:
        print(f"    skipping {outpath.name} — no alb/exp columns", flush=True)
        return

    fig, axes = plt.subplots(1, len(avail),
                             figsize=(6 * len(avail), 5),
                             squeeze=False)

    for ci, (bp, bname) in enumerate(avail):
        ax = axes[0, ci]
        x = df[f'alb_{bp}'].values.astype(float)
        y = df[f'exp_{bp}_intercept'].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        xm, ym = x[mask], y[mask]
        if len(xm) < 5:
            ax.set_visible(False)
            continue
        hb = ax.hexbin(xm, ym, gridsize=60, cmap='YlOrRd',
                       mincnt=1, norm=mcolors.LogNorm())
        plt.colorbar(hb, ax=ax, label='count')
        r, _ = _stats.pearsonr(xm, ym)
        ax.set_title(f'{bname}: exp_intercept vs albedo  r={r:.3f}', fontsize=9)
        ax.set_xlabel(f'{bname} albedo', fontsize=8)
        ax.set_ylabel(f'{bname} exp_intercept', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(f'{title_prefix}exp_intercept vs albedo', fontsize=12)
    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    {outpath.name}", flush=True)


def _plot_exp_vs_alb_cld(df: pd.DataFrame, outpath: Path, title_prefix: str = ''):
    """(6b) exp_intercept vs albedo scatter colored by nearest cloud distance, 1×3 grid."""
    avail = [(bp, bname)
             for bp, bname, _ in BANDS
             if f'alb_{bp}' in df.columns and f'exp_{bp}_intercept' in df.columns]
    if not avail:
        print(f"    skipping {outpath.name} — no alb/exp columns", flush=True)
        return

    cld = (df['cld_dist_km'].values.astype(float)
           if 'cld_dist_km' in df.columns else np.zeros(len(df)))

    fig, axes = plt.subplots(1, len(avail),
                             figsize=(6 * len(avail), 5),
                             squeeze=False)
    for ci, (bp, bname) in enumerate(avail):
        ax = axes[0, ci]
        x = df[f'alb_{bp}'].values.astype(float)
        y = df[f'exp_{bp}_intercept'].values.astype(float)
        _scatter_cld_colored(ax, x, y, cld,
                             xlabel=f'{bname} albedo',
                             ylabel=f'{bname} exp_intercept',
                             title=f'{bname}: exp_intercept vs albedo')

    fig.suptitle(f'{title_prefix}exp_intercept vs albedo  (color = nearest cloud distance km)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    {outpath.name}", flush=True)


def _plot_spectral_maps(df: pd.DataFrame, bg_img, extent: list,
                        out: Path, title_prefix: str = '', pt_size: float = 12):
    """7 map images: o2a/wco2/sco2 k1&k2 + xco2_bc, each colored by that quantity."""
    targets = []
    for bp, bname, _ in BANDS:
        for kt, kl in (('k1', r"$\langle l'\rangle$"), ('k2', r"$\mathrm{var}(l')$")):
            col = f'{bp}_{kt}'
            if col in df.columns:
                targets.append((col, f'{bname} {kl}', f'map_{bp}_{kt}.png'))
    if 'xco2_bc' in df.columns:
        targets.append(('xco2_bc', 'XCO₂_bc (ppm)', 'map_xco2_bc.png'))

    if not targets:
        print(f"    skipping spectral maps — no k/xco2_bc columns", flush=True)
        return

    for col, label, fname in targets:
        v = df[col].values.astype(float)
        fin = v[np.isfinite(v)]
        if len(fin) == 0:
            continue
        vmin = float(np.nanpercentile(fin, 2))
        vmax = float(np.nanpercentile(fin, 98))

        fig, ax = _fig_for_extent(bg_img, extent)
        if bg_img is not None:
            ax.imshow(bg_img,
                      extent=(extent[0], extent[1], extent[2], extent[3]),
                      aspect='equal', origin='upper', zorder=0)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sc = ax.scatter(np.asarray(df['lon'], dtype=float), np.asarray(df['lat'], dtype=float),
                        c=v, cmap='RdYlBu_r', norm=norm,
                        s=pt_size, alpha=0.75, rasterized=True, zorder=2)
        cax = make_axes_locatable(ax).append_axes('right', size='6%', pad=0.2)
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(label, fontsize=20)
        cb.ax.tick_params(labelsize=18)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel('Lon (°E)', fontsize=20)
        ax.set_ylabel('Lat (°N)', fontsize=20)
        ax.tick_params(labelsize=18)
        ax.set_title(f'{title_prefix}{label}', fontsize=22, pad=26)
        ax.text(0.96, 0.98, f'n = {len(df):,}', transform=ax.transAxes,
                ha='right', va='top', fontsize=18,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        fig.tight_layout()
        outpath = out / fname
        fig.savefig(str(outpath), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    {outpath.name}", flush=True)


# ── Per-subset orchestration ───────────────────────────────────────────────────

def _analyze_subset(df: pd.DataFrame, name: str, out: Path,
                    bg_img, extent: list, is_single_fp: bool = False):
    """Run all plot types for one footprint subset."""
    if len(df) == 0:
        print(f"  [{name}] empty — skipping", flush=True)
        return

    out.mkdir(parents=True, exist_ok=True)
    print(f"\n  [{name}]  n={len(df):,}", flush=True)

    # single-footprint subsets have ~1/8 the point density → use larger markers
    pt_size = 40 if is_single_fp else 12

    prefix = f'{name}: '
    _plot_map(df, bg_img, extent, out / 'map.png',
              title=f'{name} — footprints colored by nearest cloud distance',
              pt_size=pt_size)
    _plot_k_vs_alb(df, out / 'k_vs_alb.png',          title_prefix=prefix)
    _plot_k_vs_exp(df, out / 'k_vs_exp.png',           title_prefix=prefix)
    _plot_exp_vs_alb(df, out / 'exp_vs_alb.png',       title_prefix=prefix)
    _plot_exp_vs_alb_cld(df, out / 'exp_vs_alb_cld.png', title_prefix=prefix)
    _plot_spectral_maps(df, bg_img, extent, out,       title_prefix=prefix,
                        pt_size=pt_size)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Area analysis: k/exp/alb scatter plots per footprint with MODIS background.'
    )
    parser.add_argument('--lon-range', nargs=2, type=float, required=True,
                        metavar=('LON_MIN', 'LON_MAX'),
                        help='Longitude range (e.g. 100 130)')
    parser.add_argument('--lat-range', nargs=2, type=float, required=True,
                        metavar=('LAT_MIN', 'LAT_MAX'),
                        help='Latitude range (e.g. -10 20)')
    parser.add_argument('--date', required=True, metavar='YYYY-MM-DD',
                        help='Target date')
    parser.add_argument('--parquet-fname', default=None,
                        help='Parquet filename inside results/csv_collection/ '
                             '(default: combined_<date>_all_orbits.parquet)')
    parser.add_argument('--parquet', default=None,
                        help='Full path to parquet file (overrides --parquet-fname)')
    parser.add_argument('--output-dir', default=None,
                        help='Root output directory (default: <storage>/results/figures/area_analysis)')
    parser.add_argument('--modis-which', default='aqua', choices=['terra', 'aqua'],
                        help='MODIS instrument for RGB background (default: aqua)')
    parser.add_argument('--no-modis', action='store_true',
                        help='Skip MODIS background download')
    args = parser.parse_args()

    storage_dir = get_storage_dir()

    # ── Resolve parquet path ───────────────────────────────────────────────────
    if args.parquet:
        parquet_path = Path(args.parquet)
        if not parquet_path.is_absolute():
            parquet_path = storage_dir / parquet_path
    else:
        fname = args.parquet_fname or f'combined_{args.date}_all_orbits.parquet'
        parquet_path = storage_dir / 'results' / 'csv_collection' / fname

    if not parquet_path.exists():
        print(f"ERROR: parquet not found: {parquet_path}", flush=True)
        sys.exit(1)

    # ── Resolve output directory ───────────────────────────────────────────────
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = storage_dir / 'results' / 'figures' / 'area_analysis'
    lon0, lon1 = args.lon_range
    lat0, lat1 = args.lat_range
    folder_name = (f"{args.date}"
                   f"_lon{lon0:g}-{lon1:g}"
                   f"_lat{lat0:g}-{lat1:g}")
    out_root = out_root / folder_name
    out_root.mkdir(parents=True, exist_ok=True)

    extent = [lon0, lon1, lat0, lat1]

    # ── Load and filter ────────────────────────────────────────────────────────
    df = _load_and_filter(parquet_path, args.lon_range, args.lat_range)
    if len(df) == 0:
        print("No soundings in specified area — exiting.", flush=True)
        sys.exit(0)

    print(f"\nColumns: {list(df.columns)}", flush=True)

    # ── MODIS background ───────────────────────────────────────────────────────
    bg_img = None
    if not args.no_modis:
        try:
            bg_path = _download_modis_bg(args.date, extent, out_root, which=args.modis_which)
            if bg_path:
                bg_img = plt.imread(bg_path)
        except Exception as exc:
            print(f"  MODIS download failed ({exc}) — maps will have no background.", flush=True)

    # ── Build subset list: all + fp_0..fp_7 ───────────────────────────────────
    subsets = [('all', df)]
    for fp_idx in range(8):
        sub = _subset_for_fp(df, fp_idx)
        if sub is not None and len(sub) > 0:
            subsets.append((f'fp_{fp_idx}', sub))
        elif sub is None:
            print(f"  Warning: no footprint index column found — skipping fp_{fp_idx}", flush=True)
            break

    print(f"\nRunning analysis for {len(subsets)} subset(s) …", flush=True)

    for name, sdf in subsets:
        _analyze_subset(sdf, name, out_root / name, bg_img, extent,
                        is_single_fp=(name != 'all'))

    print(f"\nDone. All outputs in {out_root}", flush=True)


if __name__ == '__main__':
    main()



"""
test code

python workspace/analyze_area.py \
    --lon-range -79.9 -78.6 --lat-range 17.5 19.5 \
    --date 2020-01-01 \
    --output-dir results/figures/area_anal
    
python workspace/analyze_area.py \
    --lon-range -79.45 -78.75  --lat-range 17.6 18.6 \
    --date 2020-01-01 \
    --output-dir results/figures/area_anal

python workspace/analyze_area.py \
    --lon-range -79.35 -78.65 --lat-range 17.0 18.0 \
    --date 2020-01-01 \
    --output-dir results/figures/area_anal
    
python workspace/analyze_area.py \
    --lon-range -79.0 -78.1 --lat-range 15.4 16.6 \
    --date 2020-01-01 \
    --output-dir results/figures/area_anal


python workspace/analyze_area.py \
    --lon-range 128.75 129.45 --lat-range -27.35 -26.35 \
    --date 2020-01-01 \
    --output-dir results/figures/area_anal


python workspace/analyze_area.py \
    --lon-range 119.6 120.3 --lat-range 12.8 13.8 \
    --date 2020-01-01 \
    --output-dir results/figures/area_anal
    
python workspace/analyze_area.py \
    --lon-range 118.8 119.5 --lat-range 15.7 16.7 \
    --date 2020-01-01 \
    --output-dir results/figures/area_anal
    
python workspace/analyze_area.py \
    --lon-range 47.05 47.75 --lat-range 39.6 40.6 \
    --date 2019-10-01 \
    --output-dir results/figures/area_anal
    
python workspace/analyze_area.py \
    --lon-range 119.3 120.0 --lat-range -26.4 -25.4 \
    --date 2019-06-01 \
    --output-dir results/figures/area_anal
    
python workspace/analyze_area.py \
    --lon-range 118.7 119.6 --lat-range -24.2 -23.2 \
    --date 2019-06-01 \
    --output-dir results/figures/area_anal
    
python workspace/analyze_area.py \
    --lon-range 116.6 117.3 --lat-range -13.3 -12.3 \
    --date 2019-06-01 \
    --output-dir results/figures/area_anal
    
python workspace/analyze_area.py \
    --lon-range 120.85 121.55 --lat-range 24.6 25.4 \
    --date 2016-01-01 \
    --output-dir results/figures/area_anal
    
python workspace/analyze_area.py \
    --lon-range 119.15 119.95 --lat-range 23.3 24.3 \
    --date 2020-05-01 \
    --output-dir results/figures/area_anal
    
python workspace/analyze_area.py \
    --lon-range -39.6 -39.2 --lat-range -0.8 -0.2 \
    --date 2020-11-15 \
    --output-dir results/figures/area_anal

"""
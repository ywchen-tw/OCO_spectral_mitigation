"""plot_spectral_params.py — map the per-band spectral-fit parameters for a case.

Companion to plot_corrected_xco2.py.  Reads the combined_DATE_all_orbits parquet
(raw fit output) and draws a 3×3 grid of lon/lat scatter maps over the same MODIS
background / box:

    rows  = bands     : O2-A | weak-CO2 | strong-CO2
    cols  = quantity  : k1 | k2 | exp_intercept-alb   (ext intercept − albedo)

Each panel auto-scales to its own 2–98 percentile (these parameters have very
different ranges).  Output: spectral_params_<date>.png in --output-dir.

Example:
    PYTHONPATH=src python workspace/plot_spectral_params.py \
        --input results/csv_collection/combined_2019-07-10_all_orbits.parquet \
        --tccon data/TCCON/df20130720_20260121.public.qc.nc \
        --results-h5 results/results_2019-07-10.h5 \
        --output-dir results/model_comparison/deep_ensemble/combined_2019-07-10_df \
        --modis-auto --lon-range -118.34 -117.39 --lat-range 34.46 35.46 \
        --date-plot 2019-07-10
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
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_corrected_xco2 import (download_modis_rgb, _dominant_granule_date,
                                 _scatter_map, load_tccon, get_storage_dir)

BANDS = [('o2a', 'O$_2$-A'), ('wco2', 'weak CO$_2$'), ('sco2', 'strong CO$_2$')]
QUANTS = [('k1', '{b}_k1'), ('k2', '{b}_k2'),
          ('exp_intercept−alb', '{b}_exp_intercept-alb')]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', nargs='+', required=True,
                    help='combined_DATE_all_orbits parquet(s) with the spectral columns.')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--lon-range', nargs=2, type=float, default=None)
    ap.add_argument('--lat-range', nargs=2, type=float, default=None)
    ap.add_argument('--date-plot', default=None, help='YYYY-MM-DD (MODIS date).')
    ap.add_argument('--sfc_type', type=int, default=None, help='Optional surface filter.')
    ap.add_argument('--tccon', default=None, help='TCCON file (for the station marker).')
    ap.add_argument('--results-h5', default=None)
    ap.add_argument('--modis-auto', action='store_true')
    ap.add_argument('--modis-which', default='aqua', choices=['terra', 'aqua'])
    ap.add_argument('--cmap', default='viridis')
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    storage = get_storage_dir()
    def _abs(p): return str(storage / p) if p and not Path(p).is_absolute() else p

    need = ['lon', 'lat', 'sfc_type'] + [q[1].format(b=b) for b, _ in BANDS for q in QUANTS]
    frames = [pd.read_parquet(_abs(f)) for f in args.input]
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"input missing spectral columns: {missing[:12]}")
    if args.sfc_type is not None:
        df = df[df['sfc_type'] == args.sfc_type]
    if args.lon_range:
        df = df[(df['lon'] >= args.lon_range[0]) & (df['lon'] <= args.lon_range[1])]
    if args.lat_range:
        df = df[(df['lat'] >= args.lat_range[0]) & (df['lat'] <= args.lat_range[1])]
    print(f"  {len(df):,} footprints in box", flush=True)
    lon = df['lon'].to_numpy(float); lat = df['lat'].to_numpy(float)

    # ── station marker ──
    tccon_lon = tccon_lat = None
    if args.tccon:
        try:
            tc = load_tccon(_abs(args.tccon))
            if len(tc):
                tccon_lon = float(tc['lon'].median()); tccon_lat = float(tc['lat'].median())
        except Exception as exc:
            print(f"  (TCCON marker skipped: {exc})", flush=True)

    # ── view extent + MODIS background ──
    out_dir = Path(_abs(args.output_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    view = None
    if args.lon_range and args.lat_range:
        view = [args.lon_range[0], args.lon_range[1], args.lat_range[0], args.lat_range[1]]
    elif len(lon):
        view = [lon.min(), lon.max(), lat.min(), lat.max()]
    bg_img = None
    if args.modis_auto and view is not None:
        mdate = None
        if args.results_h5 and Path(_abs(args.results_h5)).exists():
            try: mdate, _ = _dominant_granule_date(Path(_abs(args.results_h5)))
            except Exception: mdate = None
        if mdate is None and args.date_plot:
            mdate = pd.Timestamp(args.date_plot)
        if mdate is not None:
            try:
                rgb = download_modis_rgb(mdate, view, which=args.modis_which,
                                         fdir=str(out_dir), coastline=True)
                bg_img = plt.imread(rgb)
            except Exception as exc:
                print(f"  MODIS bg failed ({exc})", flush=True)

    # ── 3×3 grid ──
    fig, axes = plt.subplots(len(BANDS), len(QUANTS), figsize=(6 * len(QUANTS), 5 * len(BANDS)))
    for i, (b, blabel) in enumerate(BANDS):
        for j, (qlabel, qpat) in enumerate(QUANTS):
            ax = axes[i, j]
            col = qpat.format(b=b)
            vals = df[col].to_numpy(float)
            fin = vals[np.isfinite(vals)]
            if len(fin):
                vmin, vmax = np.nanpercentile(fin, 2), np.nanpercentile(fin, 98)
                if vmax <= vmin: vmax = vmin + 1e-6
            else:
                vmin, vmax = 0.0, 1.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            _scatter_map(ax, lon, lat, vals, f'{blabel}  {qlabel}', norm, args.cmap,
                         tccon_lon, tccon_lat, bg_img=bg_img, bg_extent=view,
                         view_extent=view)
            sm = mcm.ScalarMappable(norm=norm, cmap=args.cmap); sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    title_date = args.date_plot or ''
    fig.suptitle(f'OCO-2 spectral-fit parameters {title_date}', fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = out_dir / f'spectral_params_{title_date or "case"}.png'
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight'); plt.close(fig)
    print(f"  [saved] {out}", flush=True)


if __name__ == '__main__':
    main()

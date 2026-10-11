"""Fine-resolution cloud-distance boxplot (ocean vs land) for any variable.

Generalizes the original `results/figures/cld_dist_analysis/<col>_ocean_land_boxplot.png`
(coarse [0,2,5,10,15,20,30,50] bins) to configurable 1-km bins, used to read the
bias-vs-distance decay and pick per-surface correction thresholds.

Default reproduces the analysis behind Phase 2f (final gate thresholds: ocean ~5km,
land ~15km): 1-km bins 0..15 + a 15..ref-km far-reference bin, on the full 66-date set.

Only the columns needed by apply_quality_filter + the plot are loaded, so it is light
on the 10M-row file.

Examples:
    PYTHONPATH=src python workspace/plot_cld_dist_boxplot_fine.py
    PYTHONPATH=src python workspace/plot_cld_dist_boxplot_fine.py \
        --parquet-fname combined_2020_dates.parquet --max-km 20 --col xco2_raw_anomaly
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.utils import apply_quality_filter, cld_dist_bins, get_storage_dir
from analysis.k_coeff import plot_xco2_anomaly_ocean_land


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parquet-fname', default='combined_2016_2020_dates.parquet',
                    help="Parquet under results/csv_collection/ (default: full 66-date set).")
    ap.add_argument('--col', default='xco2_bc_anomaly', help="Variable to box-plot.")
    ap.add_argument('--label', default=None, help="Pretty axis/title label (default: col).")
    ap.add_argument('--max-km', type=float, default=15.0,
                    help="1-km bins span 0..max-km (default 15).")
    ap.add_argument('--bin-width', type=float, default=1.0, help="Bin width in km (default 1).")
    ap.add_argument('--ref-km', type=float, default=50.0,
                    help="Upper edge of the far-reference bin (default 50).")
    ap.add_argument('--outname', default=None,
                    help="Output PNG name (default: <col>_ocean_land_boxplot_<w>km_0to<max>.png).")
    args = ap.parse_args()

    storage = get_storage_dir()
    path = storage / 'results/csv_collection' / args.parquet_fname

    want = ['xco2_bc', 'xco2_qf', 'snow_flag', 'cld_dist_km', 'sfc_type', args.col]
    have = set(pq.ParquetFile(path).schema.names)
    cols = list(dict.fromkeys(c for c in want if c in have))
    missing = [c for c in want if c not in have]
    if missing:
        print(f"[warn] columns absent from {args.parquet_fname}: {missing}")
    if args.col not in have or 'cld_dist_km' not in have:
        raise SystemExit(f"required column missing ({args.col!r} / cld_dist_km).")

    print(f"Loading {len(cols)} cols from {args.parquet_fname} ...")
    df = pd.read_parquet(path, columns=cols)
    print(f"  raw rows: {len(df):,}")
    df = apply_quality_filter(df)
    print(f"  after QF: {len(df):,}  ocean={(df.sfc_type == 0).sum():,}  "
          f"land={(df.sfc_type == 1).sum():,}")

    edges = list(np.arange(0, args.max_km + args.bin_width / 2, args.bin_width)) + [args.ref_km]
    edges = [float(e) for e in edges]
    # integer-looking labels when bin_width divides evenly
    bins, labels = cld_dist_bins(tuple(int(e) if float(e).is_integer() else e for e in edges))

    outdir = str(storage / 'results/figures/cld_dist_analysis')
    label = args.label or (args.col.replace('_', ' '))
    plot_xco2_anomaly_ocean_land(df, bins, labels, outdir, col=args.col, label=label)

    src = os.path.join(outdir, f"{args.col.replace(' ', '_')}_ocean_land_boxplot.png")
    wtag = (f"{int(args.bin_width)}" if float(args.bin_width).is_integer()
            else f"{args.bin_width}")
    outname = args.outname or (f"{args.col}_ocean_land_boxplot_"
                               f"{wtag}km_0to{int(args.max_km)}.png")
    dst = os.path.join(outdir, outname)
    os.replace(src, dst)
    print(f"saved {dst}")

    # Numeric bias table — the basis for picking the threshold.
    b = pd.cut(df['cld_dist_km'], bins=edges, labels=labels, right=False)
    print(f"\nmedian {args.col} by bin:")
    print(f"{'bin':10} {'ocean_med':>10} {'ocean_n':>11} {'land_med':>10} {'land_n':>11}")
    for lbl in labels:
        o = df.loc[(df.sfc_type == 0) & (b == lbl), args.col].dropna()
        l = df.loc[(df.sfc_type == 1) & (b == lbl), args.col].dropna()
        om = np.median(o) if len(o) else float('nan')
        lm = np.median(l) if len(l) else float('nan')
        print(f"{str(lbl):10} {om:>10.4f} {len(o):>11,} {lm:>10.4f} {len(l):>11,}")


if __name__ == '__main__':
    raise SystemExit(main())

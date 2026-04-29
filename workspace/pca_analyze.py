"""
pca_analyze.py  —  ENTRY POINT
===============================
Run PCA analysis on all non-XCO2 spectral / atmospheric feature variables.

Reads the same combined parquet as combined_analyze.py, applies the standard
quality filter, then runs PCA for ocean and land separately (plus combined).

Input
-----
- combined_2020-01-01_all_orbits.parquet  (macOS, single-date test)
- combined_2016_2020_dates.parquet        (Linux / full dataset)

Output
------
results/figures/cld_dist_analysis/pca/
    pca_scree.png
    pca_loadings_heatmap.png
    pca_scores_vs_cld_dist.png
    pca_biplot.png
    pca_correlation_with_cld_dist.png
results/figures/cld_dist_analysis/ocean/pca/   (same 5 figures, tag=ocean)
results/figures/cld_dist_analysis/land/pca/    (same 5 figures, tag=land)
"""

import gc
import sys
import logging
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.utils import (
    get_storage_dir, load_data, apply_quality_filter,
    cld_dist_bins,
)
from analysis.pca import run_pca_analysis

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    storage_dir = get_storage_dir()
    result_dir  = storage_dir / 'results'
    csv_dir     = result_dir / 'csv_collection'

    # ── load ──────────────────────────────────────────────────────────────────
    if platform.system() == 'Darwin':
        df = load_data(csv_dir, parquet_fname='combined_2020-01-01_all_orbits.parquet')
    elif platform.system() == 'Linux':
        df = load_data(csv_dir, parquet_fname='combined_2016_2020_dates.parquet')
    else:
        df = load_data(csv_dir)

    # ── quality filter ────────────────────────────────────────────────────────
    df = apply_quality_filter(df)

    # ── cloud-distance bins ───────────────────────────────────────────────────
    edges  = [0, 2, 5, 10, 15, 20, 30, 50]
    bins, labels = cld_dist_bins(edges)

    base_outdir = result_dir / 'figures' / 'cld_dist_analysis'

    # ── combined (all surfaces) ───────────────────────────────────────────────
    logger.info("Running PCA on combined (all surfaces) …")
    run_pca_analysis(df, bins, labels,
                     outdir=str(base_outdir / 'pca'),
                     tag='')
    gc.collect()

    # ── per surface type ──────────────────────────────────────────────────────
    sfc_codes = {'ocean': 0, 'land': 1} if 'sfc_type' in df.columns else {}

    for sfc_name, sfc_code in sfc_codes.items():
        sdf = df[df['sfc_type'] == sfc_code]
        logger.info(f"\n{'='*55}\nPCA for surface type: {sfc_name.upper()}  "
                    f"({len(sdf):,} soundings)\n{'='*55}")
        outdir = str(base_outdir / sfc_name / 'pca')
        run_pca_analysis(sdf, bins, labels, outdir=outdir, tag=sfc_name)
        del sdf
        gc.collect()


if __name__ == '__main__':
    main()

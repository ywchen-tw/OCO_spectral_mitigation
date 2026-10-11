"""footprint_analyze.py — Run footprint-only analysis plots.

This entrypoint avoids running the full combined_analyze.py workflow.
It loads the combined parquet, applies the same quality filter and cloud-distance
bins, then runs the analysis suite only for footprint subsets.
"""

import argparse
import logging
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.utils import get_storage_dir, load_data, apply_quality_filter, cld_dist_bins
from analysis.footprint import subset_for_fp
from analysis.run_all import _run_subset_analysis


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _select_distance_column(df, distance_col: str):
    """Route selected distance column into cld_dist_km for downstream modules."""
    if distance_col not in df.columns:
        raise ValueError(
            f"Distance column '{distance_col}' not found in dataframe. "
            f"Available columns include: {', '.join(sorted(df.columns[:20]))} ..."
        )

    if distance_col == 'cld_dist_km':
        return df

    out = df.copy()
    out['cld_dist_km'] = out[distance_col]
    return out


def _analysis_dirname(distance_col: str) -> str:
    if distance_col == 'weighted_cloud_dist_km':
        return 'weighted_cloud_dist_analysis'
    return 'cld_dist_analysis'


def _default_parquet_name() -> str:
    if platform.system() == 'Darwin':
        return 'combined_2020-01-01_all_orbits.parquet'
    return 'combined_2016_2020_dates.parquet'


def _run_footprint_surface_analysis(df, bins, labels, result_dir, fp_idx, logger, analysis_dir):
    """Run fp_idx analysis separately for ocean and land subsets."""
    fp_name = f'fp_{fp_idx}'
    fp_df = subset_for_fp(df, fp_idx, logger=logger)
    if fp_df is None:
        logger.error('Could not determine footprint columns. Aborting.')
        return 1
    if fp_df.empty:
        logger.warning(f'No rows for {fp_name}. Nothing to plot.')
        return 0

    for sfc_name, sfc_code in (('ocean', 0), ('land', 1)):
        if 'sfc_type' not in fp_df.columns:
            logger.warning('No sfc_type column found. Skipping surface split.')
            fp_outdir = str(result_dir / 'figures' / analysis_dir / 'footprints' / fp_name)
            _run_subset_analysis(fp_df, bins, labels, fp_name, fp_outdir)
            return 0

        sdf = fp_df[fp_df['sfc_type'] == sfc_code]
        if sdf.empty:
            logger.warning(f'No rows for {fp_name}/{sfc_name}. Skipping.')
            continue

        fp_outdir = str(result_dir / 'figures' / analysis_dir / 'footprints' / fp_name / sfc_name)
        logger.info('=' * 70)
        logger.info(f'OUTPUT DIR [{fp_name}/{sfc_name}]: {fp_outdir}')
        logger.info('=' * 70)
        logger.info(f'Running footprint analysis for {fp_name}/{sfc_name}.')
        _run_subset_analysis(sdf, bins, labels, f'{fp_name}_{sfc_name}', fp_outdir)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Run ocean/land and footprint plots from combined parquet data.'
    )
    parser.add_argument(
        '--parquet-fname',
        type=str,
        default=None,
        help='Parquet filename inside results/csv_collection (default: platform-specific).',
    )
    parser.add_argument(
        '--fp-index',
        type=int,
        default=None,
        help='Optional single footprint index to run (0..7). If omitted, runs all footprints.',
    )
    parser.add_argument(
        '--distance-col',
        type=str,
        default='cld_dist_km',
        choices=['cld_dist_km', 'weighted_cloud_dist_km'],
        help='Distance variable to use for all cloud-distance plots.',
    )
    args = parser.parse_args()

    if args.fp_index is not None and not (0 <= args.fp_index <= 7):
        raise ValueError('--fp-index must be between 0 and 7')

    storage_dir = get_storage_dir()
    result_dir = storage_dir / 'results'
    csv_dir = result_dir / 'csv_collection'

    parquet_name = args.parquet_fname or _default_parquet_name()
    logger.info(f'Loading parquet: {parquet_name}')

    df = load_data(csv_dir, parquet_fname=parquet_name)
    df = apply_quality_filter(df)
    logger.info(f'Using distance column: {args.distance_col}')
    df = _select_distance_column(df, args.distance_col)
    analysis_dir = _analysis_dirname(args.distance_col)
    logger.info(f'Writing figures under: results/figures/{analysis_dir}')

    edges = [0, 2, 5, 10, 15, 20, 30, 50]
    bins, labels = cld_dist_bins(edges)

    if args.fp_index is None:
        logger.info('Running footprint analysis for all footprints (fp_0..fp_7), split by ocean/land.')
        for fp_idx in range(8):
            _run_footprint_surface_analysis(df, bins, labels, result_dir, fp_idx, logger, analysis_dir)
    else:
        logger.info(f'Running footprint analysis for fp_{args.fp_index}, split by ocean/land.')
        return _run_footprint_surface_analysis(df, bins, labels, result_dir, args.fp_index, logger, analysis_dir)

    logger.info('Footprint surface-split analysis complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

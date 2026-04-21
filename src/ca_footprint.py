"""ca_footprint.py — Footprint-specific analysis orchestration.

Extracts footprint slicing and per-footprint analysis loop from
combined_analyze.py so the entrypoint stays focused on top-level flow.
"""

import gc
import logging
from pathlib import Path


def subset_for_fp(df, fp_idx: int, logger: logging.Logger | None = None):
    """Return one-footprint subset for fp_idx in [0..7].

    Supports either one-hot footprint columns (fp_0..fp_7) or a numeric
    footprint index column (fp_number/fp/footprint/footprint_id) that may
    be 0-based or 1-based.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    fp_col = f'fp_{fp_idx}'
    if fp_col in df.columns:
        return df[df[fp_col] == 1]

    fp_num_col = next(
        (c for c in ('fp_number', 'fp', 'footprint', 'footprint_id') if c in df.columns),
        None,
    )

    if fp_num_col is not None:
        fp_vals = df[fp_num_col].dropna().astype(int)
        if fp_vals.empty:
            return df.iloc[0:0]

        unique_vals = set(fp_vals.unique().tolist())
        if set(range(8)).issubset(unique_vals) or (unique_vals and min(unique_vals) == 0):
            return df[df[fp_num_col].astype(int) == fp_idx]

        if set(range(1, 9)).issubset(unique_vals) or (unique_vals and min(unique_vals) == 1):
            return df[df[fp_num_col].astype(int) == (fp_idx + 1)]

        return df[df[fp_num_col].astype(int) == fp_idx]

    logger.warning("No footprint index column found (expected fp_0..fp_7 or fp_number/fp/footprint/footprint_id)")
    return None


def run_footprint_analysis(
    df,
    bins,
    labels,
    result_dir: Path,
    run_subset_analysis,
    logger: logging.Logger | None = None,
    analysis_subdir: str = 'cld_dist_analysis',
):
    """Run the same analysis suite for fp_0 .. fp_7 subsets.

    Parameters
    ----------
    df : pandas.DataFrame
        Full filtered dataframe.
    bins, labels : sequence
        Cloud-distance bin edges and labels from cld_dist_bins().
    result_dir : pathlib.Path
        Base results directory.
    run_subset_analysis : callable
        Callback with signature (sdf, bins, labels, subset_name, subset_outdir).
    logger : logging.Logger | None
        Logger used for progress messages.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    for fp_idx in range(8):
        fp_name = f'fp_{fp_idx}'
        fp_df = subset_for_fp(df, fp_idx, logger=logger)
        if fp_df is None:
            break
        if fp_df.empty:
            logger.warning(f"No rows for {fp_name} — skipping")
            continue

        fp_outdir = str(result_dir / 'figures' / analysis_subdir / 'footprints' / fp_name)
        run_subset_analysis(fp_df, bins, labels, fp_name, fp_outdir)
        del fp_df
        gc.collect()

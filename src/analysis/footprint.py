"""ca_footprint.py — Footprint-specific analysis orchestration.

Extracts footprint slicing and per-footprint analysis loop from
combined_analyze.py so the entrypoint stays focused on top-level flow.

`run_footprint_overlay` is the default footprint analysis: one overlay figure
set with the footprint index as the group (via the land_class engine),
replacing the legacy fp_0..7 loop that re-ran the entire subset suite eight
times (~18k figures). The legacy loop (`run_footprint_analysis`) remains
available behind run_all's --legacy-full.
"""

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd


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


def _fp_labels(df, logger: logging.Logger):
    """Return a per-row 'fp_{i}' label Series, or None if no footprint info.

    Sources, in order: fp_0..fp_7 one-hots, then a small-integer index column
    (fp / fp_number / footprint / footprint_id — values must lie in 0..8;
    NOTE fp_id is the 16-digit sounding ID, not a footprint index).
    """
    onehots = [f'fp_{i}' for i in range(8) if f'fp_{i}' in df.columns]
    if onehots:
        arr = df[onehots].to_numpy()
        idx = arr.argmax(axis=1)
        has = arr.max(axis=1) == 1
        lab = pd.Series(pd.NA, index=df.index, dtype='object')
        lab[has] = np.array(onehots, dtype=object)[idx[has]]
        return lab

    for col in ('fp', 'fp_number', 'footprint', 'footprint_id'):
        if col not in df.columns:
            continue
        vals = df[col]
        good = vals.notna()
        if not good.any() or vals[good].max() > 8 or vals[good].min() < 0:
            continue
        lab = pd.Series(pd.NA, index=df.index, dtype='object')
        lab[good] = 'fp_' + vals[good].astype(int).astype(str)
        return lab

    logger.warning("No footprint column found (fp_0..fp_7 one-hots or a "
                   "0..8-valued fp/fp_number/footprint column) — "
                   "skipping footprint overlay")
    return None


def run_footprint_overlay(df, bins, labels, outdir: str,
                          logger: logging.Logger | None = None,
                          n_min: int = 500, min_class_n: int = 5000) -> None:
    """Footprint-as-group overlay suite (replaces the legacy fp_0..7 loop).

    Produces one land_class-style figure set — count matrix, per-variable
    overlay profiles (raw + z), k overview, effect-size CSV/heatmap — with
    the 8 footprints as lines on shared axes.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    from .land_class import run_land_class_analysis

    lab = _fp_labels(df, logger)
    if lab is None:
        return
    import matplotlib.pyplot as plt
    order = sorted(lab.dropna().unique())
    cmap = plt.colormaps['viridis']
    colors = {g: cmap(0.05 + 0.9 * i / max(len(order) - 1, 1))
              for i, g in enumerate(order)}
    df = df.assign(_fp=pd.Categorical(lab, categories=order))
    run_land_class_analysis(
        df, bins, labels, outdir,
        n_min=n_min, min_class_n=min_class_n,
        group_col='_fp', group_order=order, colors=colors,
        prefix='footprint', title_tag='footprint')


def run_footprint_analysis(
    df,
    bins,
    labels,
    result_dir: Path,
    run_subset_analysis,
    logger: logging.Logger | None = None,
    analysis_subdir: str = 'cld_dist_analysis',
    **subset_kwargs,
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
        Callback with signature (sdf, bins, labels, subset_name, subset_outdir, ...).
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
        run_subset_analysis(fp_df, bins, labels, fp_name, fp_outdir, **subset_kwargs)
        del fp_df
        gc.collect()

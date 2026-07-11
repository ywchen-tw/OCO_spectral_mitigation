"""Train / held-out splits on the *raw* DataFrame (before FeaturePipeline.fit).

Reference: no external reference (standard ML validation methodology).

Centralises the split logic shared by tabm.py, gbdt_baselines.py, and
mlp_baseline.py so every model evaluates under the same regime.

Hard rule (see log/archive/MODEL_PLANS_HISTORICAL.md "pipeline fitted on train split only"): always
split the raw DataFrame *first*, then fit FeaturePipeline on the train split
only.  Fitting the pipeline on the full dataset leaks scaler means/variances
and quantile boundaries into the held-out set — for random splits this is
already an issue; for blocked (date) splits it makes results uninterpretable.

    train_df, heldout_df = split_dataframe(df, mode="date", test_size=0.1)
    pipeline = FeaturePipeline.fit(train_df, sfc_type=sfc_type, feature_set=...)
    X_train  = pipeline.transform(train_df)
    X_held   = pipeline.transform(heldout_df)

Modes
-----
random : sklearn-style random row split (default; matches FT-Transformer for
         comparability).  Optionally stratified by a pre-computed column.
date   : hold out the last ``ceil(test_size * n_unique_dates)`` chronological
         dates (temporal leakage probe).  Requires a 'date' column.
date_kfold : block-rotation k-fold over dates.  Partition the sorted unique
         dates into ``n_folds`` contiguous blocks; fold ``k`` holds out block
         ``k`` and trains on the rest.  One fold per call (pass ``fold`` and
         ``n_folds``).  Every date serves as test exactly once across the K
         folds → a lower-variance estimate of *general* unseen-date robustness
         than the single trailing-block ``date`` mode.  Requires a 'date'
         column.  Like ``date``, it removes same-day leakage; unlike
         rolling-origin it allows training on later dates to predict earlier
         ones (acceptable here — the corrector is applied across the mission
         record, not as a forecaster).
orbit  : (follow-on) every Mth orbit held out — not yet implemented.
region : (follow-on) hold out a lat/lon tile — not yet implemented.
"""

import logging
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

VALID_MODES = ("random", "date", "date_kfold", "orbit", "region")


def _decode_dates(df: pd.DataFrame) -> pd.Series:
    """Parse the 'date' column to datetime, decoding bytes if needed."""
    if 'date' not in df.columns:
        raise ValueError(
            "date-based split requires a 'date' column in the DataFrame; "
            "none found.  Use --val_split random or add the column."
        )
    col = df['date']
    if col.dtype == object and len(col) > 0 and isinstance(col.iloc[0], bytes):
        col = col.str.decode('utf-8')
    return pd.to_datetime(col)


def _split_random(df: pd.DataFrame, test_size: float, random_state: int,
                  stratify: 'np.ndarray | None') -> tuple:
    train_idx, held_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_idx.sort()
    held_idx.sort()
    return df.iloc[train_idx], df.iloc[held_idx]


def _split_date(df: pd.DataFrame, test_size: float) -> tuple:
    """Hold out the last ceil(test_size * n_unique_dates) chronological dates."""
    dates = _decode_dates(df)
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)
    if n_dates < 2:
        raise ValueError(
            f"date-block split needs ≥2 unique dates, found {n_dates}. "
            "Use --val_split random for single-date data."
        )
    n_held = max(1, math.ceil(test_size * n_dates))
    n_held = min(n_held, n_dates - 1)            # always keep ≥1 train date
    held_dates = set(unique_dates[-n_held:])
    held_mask = dates.isin(held_dates).to_numpy()

    train_df = df.loc[~held_mask]
    held_df = df.loc[held_mask]
    logger.info(
        "date-block split: %d unique dates → %d train / %d held "
        "(held dates: %s … %s)",
        n_dates, n_dates - n_held, n_held,
        str(np.min(unique_dates[-n_held:]))[:10],
        str(np.max(unique_dates[-n_held:]))[:10],
    )
    return train_df, held_df


def _split_date_kfold(df: pd.DataFrame, n_folds: int, fold: int) -> tuple:
    """Block-rotation k-fold over dates: hold out fold ``fold`` of ``n_folds``.

    The sorted unique dates are partitioned into ``n_folds`` contiguous,
    near-equal blocks (``np.array_split``).  Fold ``fold`` holds out its block
    and trains on every other date.  Contiguous (not interleaved) blocks keep
    each test set a temporally coherent set of dates and avoid putting adjacent
    dates in both train and test.
    """
    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError(f"date_kfold needs n_folds >= 2, got {n_folds!r}")
    if not isinstance(fold, int) or not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold!r}")

    dates = _decode_dates(df)
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)
    if n_dates < n_folds:
        raise ValueError(
            f"date_kfold needs >= n_folds unique dates; found {n_dates} dates "
            f"for n_folds={n_folds}.  Reduce --n_folds or use --val_split random."
        )
    blocks = np.array_split(unique_dates, n_folds)
    held_dates = set(blocks[fold])
    held_mask = dates.isin(held_dates).to_numpy()

    train_df = df.loc[~held_mask]
    held_df = df.loc[held_mask]
    logger.info(
        "date_kfold split: fold %d/%d — %d unique dates → %d train / %d held "
        "(held block: %s … %s)",
        fold, n_folds, n_dates, len(train_df.index), len(held_df.index),
        str(np.min(blocks[fold]))[:10], str(np.max(blocks[fold]))[:10],
    )
    return train_df, held_df


def split_date_kfold_train_calib_test(
        df: pd.DataFrame,
        *,
        n_folds: int,
        fold: int,
        calib_frac: float,
        calib_side: str = "after_then_wrap_before") -> tuple:
    """Return proper-train, calibration, and held-out frames for date_kfold.

    The held-out dates are exactly the same contiguous block used by
    ``split_dataframe(..., mode="date_kfold")``.  The calibration set is a
    fold-local contiguous date block, with the same date-count formula
    previously used when carving calibration from the non-test dates:
    ``ceil(calib_frac * n_non_test_dates)``.  By default the calibration block
    is placed immediately after the held-out block, wrapping to the beginning
    of the record for terminal folds, and falling back to immediately before
    only when wrapping is impossible.  At least one proper-training date is
    always retained.
    """
    if not 0.0 < calib_frac < 1.0:
        raise ValueError(f"calib_frac must be in (0, 1), got {calib_frac}")
    valid_calib_sides = {
        "after_then_wrap_before",
        "after_then_before",
        "before_then_after",
        "after",
        "before",
        "wrap",
    }
    if calib_side not in valid_calib_sides:
        raise ValueError(
            f"calib_side must be one of {sorted(valid_calib_sides)}, "
            f"got {calib_side!r}"
        )
    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError(f"date_kfold needs n_folds >= 2, got {n_folds!r}")
    if not isinstance(fold, int) or not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold!r}")

    dates = _decode_dates(df)
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)
    if n_dates < n_folds:
        raise ValueError(
            f"date_kfold needs >= n_folds unique dates; found {n_dates} dates "
            f"for n_folds={n_folds}.  Reduce --n_folds or use --val_split random."
        )

    blocks = np.array_split(unique_dates, n_folds)
    held_block = blocks[fold]
    n_test_dates = len(held_block)
    n_non_test_dates = n_dates - n_test_dates
    if n_non_test_dates < 2:
        raise ValueError(
            "date_kfold train/calib/test split needs at least two non-test "
            f"dates; found {n_non_test_dates}."
        )

    n_calib = max(1, math.ceil(calib_frac * n_non_test_dates))
    n_calib = min(n_calib, n_non_test_dates - 1)
    test_start = sum(len(block) for block in blocks[:fold])
    test_stop = test_start + n_test_dates

    def after_block() -> np.ndarray | None:
        if test_stop + n_calib <= n_dates:
            return unique_dates[test_stop:test_stop + n_calib]
        return None

    def before_block() -> np.ndarray | None:
        if test_start - n_calib >= 0:
            return unique_dates[test_start - n_calib:test_start]
        return None

    def wrap_block() -> np.ndarray | None:
        if n_calib <= test_start:
            return unique_dates[:n_calib]
        return None

    selectors = {
        "after_then_wrap_before": (after_block, wrap_block, before_block),
        "after_then_before": (after_block, before_block),
        "before_then_after": (before_block, after_block),
        "after": (after_block,),
        "before": (before_block,),
        "wrap": (wrap_block,),
    }
    calib_block = None
    for selector in selectors[calib_side]:
        calib_block = selector()
        if calib_block is not None:
            break
    if calib_block is None:
        raise ValueError(
            "Could not place a contiguous fold-local calibration block adjacent "
            "to or wrapped around fold "
            f"{fold}; n_dates={n_dates}, n_test_dates={n_test_dates}, "
            f"n_calib={n_calib}, calib_side={calib_side!r}."
        )

    held_dates = set(held_block)
    calib_dates = set(calib_block)
    held_mask = dates.isin(held_dates).to_numpy()
    calib_mask = dates.isin(calib_dates).to_numpy()
    proper_mask = ~(held_mask | calib_mask)

    proper_df = df.loc[proper_mask]
    calib_df = df.loc[calib_mask]
    held_df = df.loc[held_mask]
    logger.info(
        "date_kfold train/calib/test split: fold %d/%d — %d unique dates → "
        "%d proper / %d calib / %d held rows (%d/%d/%d dates; held: %s … %s; "
        "calib: %s … %s)",
        fold, n_folds, n_dates,
        len(proper_df.index), len(calib_df.index), len(held_df.index),
        n_dates - n_test_dates - n_calib, n_calib, n_test_dates,
        str(np.min(held_block))[:10], str(np.max(held_block))[:10],
        str(np.min(calib_block))[:10], str(np.max(calib_block))[:10],
    )
    return proper_df, calib_df, held_df


def split_dataframe(df: pd.DataFrame,
                    mode: str = "random",
                    *,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    stratify: 'np.ndarray | None' = None,
                    n_folds: 'int | None' = None,
                    fold: 'int | None' = None) -> tuple:
    """Return (train_df, heldout_df) using the requested split mode.

    Parameters
    ----------
    df : raw feature DataFrame (already filtered by sfc_type / snow_flag).
    mode : 'random' (default), 'date', or 'date_kfold'.  'orbit' / 'region' are
        reserved follow-on modes and raise NotImplementedError until implemented.
    test_size : fraction held out.  For 'date' this is a fraction of *unique
        dates*, not rows (the plan recommends 0.1 for the date block).  Ignored
        for 'date_kfold' (block size is determined by ``n_folds``).
    random_state : seed for the random split only.
    stratify : optional per-row label array for the random split (e.g. PC1
        quintile).  Ignored for blocked modes.
    n_folds, fold : required for 'date_kfold' — total number of date blocks and
        which block (0-based) to hold out on this call.  One fold per call.

    The returned frames preserve the original DataFrame index so callers can
    pull metadata columns (cld_dist_km, aod_*, fp, …) for stratified metrics.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    if mode == "random":
        return _split_random(df, test_size, random_state, stratify)
    if mode == "date":
        return _split_date(df, test_size)
    if mode == "date_kfold":
        if n_folds is None or fold is None:
            raise ValueError(
                "date_kfold split requires both n_folds and fold "
                "(pass --n_folds and --fold)."
            )
        return _split_date_kfold(df, int(n_folds), int(fold))
    raise NotImplementedError(
        f"split mode {mode!r} is a reserved follow-on (see log/archive/MODEL_PLANS_HISTORICAL.md "
        "Validation strategy); only 'random' and 'date' are implemented in v1."
    )

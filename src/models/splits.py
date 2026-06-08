"""Train / held-out splits on the *raw* DataFrame (before FeaturePipeline.fit).

Reference: no external reference (standard ML validation methodology).

Centralises the split logic shared by tabm.py, gbdt_baselines.py, and
mlp_baseline.py so every model evaluates under the same regime.

Hard rule (see TABM_PLAN.md "pipeline fitted on train split only"): always
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
orbit  : (follow-on) every Mth orbit held out — not yet implemented.
region : (follow-on) hold out a lat/lon tile — not yet implemented.
"""

import logging
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

VALID_MODES = ("random", "date", "orbit", "region")


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
    if 'date' not in df.columns:
        raise ValueError(
            "date-block split requires a 'date' column in the DataFrame; "
            "none found.  Use --val_split random or add the column."
        )
    # Sort unique dates chronologically.  Values may be strings ('2020-02-01'),
    # datetimes, or bytes (parquet written on Python 2 / certain HDF5 paths);
    # decode bytes first so pd.to_datetime can parse them.
    col = df['date']
    if col.dtype == object and len(col) > 0 and isinstance(col.iloc[0], bytes):
        col = col.str.decode('utf-8')
    dates = pd.to_datetime(col)
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


def split_dataframe(df: pd.DataFrame,
                    mode: str = "random",
                    *,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    stratify: 'np.ndarray | None' = None) -> tuple:
    """Return (train_df, heldout_df) using the requested split mode.

    Parameters
    ----------
    df : raw feature DataFrame (already filtered by sfc_type / snow_flag).
    mode : 'random' (default) or 'date'.  'orbit' / 'region' are reserved
        follow-on modes and raise NotImplementedError until implemented.
    test_size : fraction held out.  For 'date' this is a fraction of *unique
        dates*, not rows (the plan recommends 0.1 for the date block).
    random_state : seed for the random split only.
    stratify : optional per-row label array for the random split (e.g. PC1
        quintile).  Ignored for blocked modes.

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
    raise NotImplementedError(
        f"split mode {mode!r} is a reserved follow-on (see TABM_PLAN.md "
        "Validation strategy); only 'random' and 'date' are implemented in v1."
    )

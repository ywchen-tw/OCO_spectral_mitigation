"""
ca_utils.py
===========
Shared utility functions extracted from combined_analyze.py.

Contents
--------
- get_storage_dir       Platform-aware data root (macOS / Linux / default)
- load_data             Load combined parquet or concatenate per-date parquets
- apply_quality_filter  xco2_bc > 0, xco2_qf == 0, snow_flag == 0; float32 downcast
- split_by_surface      sfc_type 0=ocean / 1=land
- cld_dist_bins         Fixed cloud-distance bin edges + labels
- bin_by_cld_dist       pd.cut wrapper
- _save                 Save figure to disk and close
- rolling_median_iqr    O(n) binned rolling median + IQR (replaces O(n²) window)
- print_summary_stats   Print key statistics to stdout
"""

import gc
import os
import glob
import platform
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import Config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_storage_dir() -> Path:
    if platform.system() == "Darwin":
        return Path(Config.get_data_path('local'))
    elif platform.system() == "Linux":
        return Path(Config.get_data_path('curc'))
    return Path(Config.get_data_path('default'))


def load_data(csv_dir: Path, parquet_fname: str | None = None) -> pd.DataFrame:
    """Load combined parquet.  Falls back to all per-date parquets."""
    if parquet_fname:
        path = csv_dir / parquet_fname
        if path.exists():
            logger.info(f"Loading {path}")
            return pd.read_parquet(path)

    files = sorted(glob.glob(str(csv_dir / 'combined_*_all_orbits.parquet')))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {csv_dir}")
    logger.info(f"Loading {len(files)} per-date files …")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def apply_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep good-quality soundings: valid xco2_bc, qf==0, no snow."""
    mask = df['xco2_bc'] > 0
    if 'xco2_qf' in df.columns:
        mask &= df['xco2_qf'] == 0
    if 'snow_flag' in df.columns:
        # snow_flag is stored as uint8/int; treat any non-zero value as snowy
        mask &= df['snow_flag'] == 0
    df = df[mask].copy()
    # downcast float64 → float32 to halve memory for all numeric columns
    float_cols = df.select_dtypes('float64').columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype('float32')
        logger.info(f"Downcast {len(float_cols)} float64 columns to float32")
    logger.info(f"After QF+snow filter: {len(df):,} soundings")
    return df


def split_by_surface(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {'ocean': ..., 'land': ...} subsets based on sfc_type."""
    subsets = {}
    if 'sfc_type' not in df.columns:
        logger.warning("sfc_type column missing — treating all as 'all'")
        subsets['all'] = df
        return subsets
    ocean = df[df['sfc_type'] == 0]
    land  = df[df['sfc_type'] == 1]
    logger.info(f"Ocean soundings: {len(ocean):,}  |  Land soundings: {len(land):,}")
    subsets['ocean'] = ocean
    subsets['land']  = land
    return subsets


def cld_dist_bins(edges=(0, 5, 10, 20, 30, 40, 50)):
    """Return (edges, labels) for cloud-distance bin edges in km."""
    labels = [f"{edges[i]}–{edges[i+1]}" for i in range(len(edges) - 1)]
    return list(edges), labels


def bin_by_cld_dist(df: pd.DataFrame, edges, labels) -> pd.Series:
    return pd.cut(df['cld_dist_km'], bins=edges, labels=labels, right=False)


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  saved → {p}")


def rolling_median_iqr(x, y, n_pts=80):
    """Return (bin_centers, median, q25, q75) using fixed-width x bins.

    Replaces the O(n²) sliding-window with an O(n) binned approach:
    divide x into n_pts equal bins, compute percentiles per bin.
    Empty bins are dropped so the curve stays clean.
    """
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    edges = np.linspace(x_min, x_max, n_pts + 1)
    centers, meds, q25s, q75s = [], [], [], []
    bin_idx = np.digitize(x, edges) - 1          # 0-based bin index
    bin_idx = np.clip(bin_idx, 0, n_pts - 1)
    for b in range(n_pts):
        w = y[bin_idx == b]
        if len(w) < 5:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        meds.append(np.median(w))
        q25s.append(np.percentile(w, 25))
        q75s.append(np.percentile(w, 75))
    return (np.array(centers), np.array(meds),
            np.array(q25s), np.array(q75s))


def print_summary_stats(df, bins, labels):
    """Print key statistics to stdout."""
    print("=" * 65)
    print("SUMMARY STATISTICS")
    print("=" * 65)
    print(f"Total soundings: {len(df):,}")
    print()

    _bin = bin_by_cld_dist(df, bins, labels)

    k_cols = ['o2a_k1', 'o2a_k2', 'wco2_k1', 'wco2_k2', 'sco2_k1', 'sco2_k2']
    anom_cols = ['xco2_bc_anomaly', 'xco2_raw_anomaly']
    avail_k = [c for c in k_cols if c in df.columns]
    avail_a = [c for c in anom_cols if c in df.columns]

    print("--- k1 / k2 global stats ---")
    print(df[avail_k].describe().round(4).to_string())
    print()
    print("--- XCO2 anomaly global stats ---")
    print(df[avail_a].describe().round(4).to_string())
    print()

    print("--- k1 mean by cloud-distance bin ---")
    for col in avail_k:
        grp = df.groupby(_bin, observed=True)[col].mean().round(4)
        print(f"  {col}: {dict(grp)}")
    print()

    print("--- XCO2 anomaly mean by cloud-distance bin ---")
    for col in avail_a:
        grp = df.groupby(_bin, observed=True)[col].mean().round(4)
        print(f"  {col}: {dict(grp)}")
    print()

    print("--- Pearson r: k1/k2 vs xco2_bc_anomaly ---")
    if 'xco2_bc_anomaly' in df.columns:
        for col in avail_k:
            mask = df[col].notna() & df['xco2_bc_anomaly'].notna()
            if mask.sum() > 2:
                r, p = stats.pearsonr(df.loc[mask, col], df.loc[mask, 'xco2_bc_anomaly'])
                print(f"  {col}: r={r:.4f}  p={p:.3e}")
    print()

    print("--- Pearson r: k1/k2 vs cld_dist_km ---")
    if 'cld_dist_km' in df.columns:
        for col in avail_k:
            mask = df[col].notna() & df['cld_dist_km'].notna()
            if mask.sum() > 2:
                r, p = stats.pearsonr(df.loc[mask, 'cld_dist_km'], df.loc[mask, col])
                print(f"  {col}: r={r:.4f}  p={p:.3e}")
    print("=" * 65)

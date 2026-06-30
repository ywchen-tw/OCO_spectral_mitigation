"""
cld_dist_cdf.py
===============
Plot the cumulative density function (CDF) of footprint cloud-proximity
distance (cld_dist_km) for three quality-flag (QF) subsets, overlaid on a
single figure:

  (1) QF = 0  (best quality)
  (2) QF = 1  (good quality)
  (3) QF = 0 or 1  (all footprints with valid data)

Input
-----
results/csv_collection/combined_2016_2020_dates.parquet
  (column ``xco2_qf`` for the quality flag, ``cld_dist_km`` for distance)

Output
------
results/figures/cld_dist_analysis/cld_dist_cdf_by_qf.png

Usage
-----
    python src/analysis/cld_dist_cdf.py
    python src/analysis/cld_dist_cdf.py --distance-col weighted_cloud_dist_km
    python src/analysis/cld_dist_cdf.py --parquet-fname combined_2016_2020_dates.parquet
"""

import sys
import logging
import argparse
from pathlib import Path

# ── path setup (mirror run_all.py so package imports work under SLURM) ─────────
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.utils import get_storage_dir, _save

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# QF subset definitions: (legend label, color, mask-builder on the qf series)
_QF_SUBSETS = [
    ('QF = 0',      'C0', lambda qf: qf == 0),
    ('QF = 1',      'C1', lambda qf: qf == 1),
    ('QF = 0 or 1', 'k',  lambda qf: qf.isin([0, 1])),
]

# Surface subsets for the two-panel plot: (panel title, sfc_type code)
_SURFACES = [('Ocean', 0), ('Land', 1)]

# Cloud distances (km) at which to report the CDF value
_CDF_MARKERS = [4, 5, 10, 12, 15]


def _ecdf(values: np.ndarray):
    """Return (sorted_values, cumulative_fraction) for an empirical CDF."""
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _date_title(df: pd.DataFrame) -> str:
    """Return a title fragment like '115 dates from 2016 to 2020'."""
    if 'date' not in df.columns:
        return 'cloud-distance CDF by quality flag'
    s = df['date']
    if s.dtype == object and len(s) and isinstance(s.iloc[0], (bytes, bytearray)):
        s = s.str.decode('utf-8')
    s = s.dropna().astype(str)
    n_dates = s.nunique()
    y0, y1 = s.min()[:4], s.max()[:4]
    return f"{n_dates} dates from {y0} to {y1}"


def plot_cld_dist_cdf(df: pd.DataFrame, distance_col: str, outdir: str,
                      date_label: str | None = None):
    """Overlay the cloud-distance CDF for the three QF subsets on one figure."""
    if 'xco2_qf' not in df.columns:
        raise ValueError("Column 'xco2_qf' not found — cannot split by quality flag.")
    if distance_col not in df.columns:
        raise ValueError(f"Distance column '{distance_col}' not found in dataframe.")

    qf = df['xco2_qf']
    dist = df[distance_col]

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, color, mask_fn in _QF_SUBSETS:
        mask = mask_fn(qf) & dist.notna()
        vals = dist[mask].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            logger.warning(f"No valid soundings for subset '{label}' — skipping")
            continue

        x, y = _ecdf(vals)
        median = float(np.median(vals))
        ax.plot(x, y, color=color, lw=1.8,
                label=f"{label}  (n={vals.size:,}, med={median:.1f} km)")
        logger.info(f"{label}: n={vals.size:,}  median={median:.2f} km  "
                    f"mean={vals.mean():.2f} km")

    title = f"Cloud-distance CDF — {date_label}" if date_label \
        else 'Cloud-distance CDF by quality flag'
    ax.set_xlabel('Footprint cloud distance (km)')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    _save(fig, outdir, 'cld_dist_cdf_by_qf.png')


def _plot_cdf_on_ax(ax, qf, dist, title):
    """Draw the three QF-subset CDFs on a single axis and annotate CDF markers."""
    text_lines = ['CDF @ ' + ' / '.join(f'{m} km' for m in _CDF_MARKERS)]
    for label, color, mask_fn in _QF_SUBSETS:
        mask = mask_fn(qf) & dist.notna()
        vals = dist[mask].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            logger.warning(f"[{title}] no valid soundings for '{label}' — skipping")
            continue

        x, y = _ecdf(vals)
        median = float(np.median(vals))
        ax.plot(x, y, color=color, lw=1.8,
                label=f"{label}  (n={vals.size:,}, med={median:.1f} km)")

        cdf_at = {m: float((vals <= m).mean()) for m in _CDF_MARKERS}
        logger.info(f"[{title}] {label}: n={vals.size:,}  median={median:.2f} km  "
                    f"mean={vals.mean():.2f} km  "
                    + "  ".join(f"CDF({m}km)={cdf_at[m]:.4f}" for m in _CDF_MARKERS))
        text_lines.append(
            f"{label}: " + ' / '.join(f"{cdf_at[m]:.2f}" for m in _CDF_MARKERS))

    for m in _CDF_MARKERS:
        ax.axvline(m, color='grey', ls=':', lw=1, alpha=0.7)

    ax.text(0.97, 0.05, '\n'.join(text_lines), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8.5, family='monospace',
            bbox=dict(boxstyle='round', fc='white', ec='grey', alpha=0.85))

    ax.set_xlabel('Footprint cloud distance (km)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', fontsize=9)


def plot_cld_dist_cdf_by_surface(df: pd.DataFrame, distance_col: str, outdir: str,
                                 date_label: str | None = None):
    """Two-panel cloud-distance CDF (ocean | land), each split by QF subset."""
    if 'xco2_qf' not in df.columns:
        raise ValueError("Column 'xco2_qf' not found — cannot split by quality flag.")
    if 'sfc_type' not in df.columns:
        raise ValueError("Column 'sfc_type' not found — cannot split by surface.")
    if distance_col not in df.columns:
        raise ValueError(f"Distance column '{distance_col}' not found in dataframe.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (sfc_name, sfc_code) in zip(axes, _SURFACES):
        sdf = df[df['sfc_type'] == sfc_code]
        _plot_cdf_on_ax(ax, sdf['xco2_qf'], sdf[distance_col], sfc_name)

    suptitle = f"Cloud-distance CDF by quality flag — {date_label}" if date_label \
        else 'Cloud-distance CDF by quality flag'
    axes[0].set_ylabel('Cumulative fraction')
    fig.suptitle(suptitle, y=1.00)
    fig.tight_layout()

    _save(fig, outdir, 'cld_dist_cdf_by_qf_ocean_land.png')


def main():
    parser = argparse.ArgumentParser(
        description='Plot footprint cloud-distance CDF split by quality flag.'
    )
    parser.add_argument(
        '--parquet-fname',
        type=str,
        default='combined_2016_2020_dates.parquet',
        help='Parquet filename inside results/csv_collection.',
    )
    parser.add_argument(
        '--distance-col',
        type=str,
        default='cld_dist_km',
        choices=['cld_dist_km', 'weighted_cloud_dist_km'],
        help='Distance variable to use for the CDF.',
    )
    args = parser.parse_args()

    storage_dir = get_storage_dir()
    csv_dir = storage_dir / 'results' / 'csv_collection'
    parquet_path = csv_dir / args.parquet_fname

    cols = ['xco2_qf', 'sfc_type', 'date', args.distance_col]
    logger.info(f"Loading {parquet_path} (columns: {', '.join(cols)}) …")
    df = pd.read_parquet(parquet_path, columns=cols)
    logger.info(f"Loaded {len(df):,} soundings")

    date_label = _date_title(df)
    logger.info(f"Effective dates in data: {date_label}")

    outdir = str(storage_dir / 'results' / 'figures' / 'cld_dist_analysis')
    plot_cld_dist_cdf(df, args.distance_col, outdir, date_label=date_label)
    plot_cld_dist_cdf_by_surface(df, args.distance_col, outdir, date_label=date_label)


if __name__ == '__main__':
    main()

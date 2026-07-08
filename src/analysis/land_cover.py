"""
land_cover.py
=============
MCD12C1 (v061) IGBP land-cover lookup for OCO-2 soundings.

Assigns each sounding an IGBP class (``igbp_class``) and a collapsed
analysis group (``igbp_group``) by direct indexing into the yearly
0.05-degree MCD12C1 CMG grid — no collocation machinery needed.

Data layout expected (one file per year, standard LP DAAC names):

    data/MODIS/MCD12C1/MCD12C1.A{year}001.061.*.hdf

Usage
-----
    from analysis.land_cover import assign_land_cover
    df = assign_land_cover(df, mcd12c1_dir)      # adds igbp_class, igbp_group

Sounding years outside the downloaded range are clamped to the nearest
available year (land-cover change at 0.05 deg is negligible on ±2 yr).

CLI smoke test:
    python -m analysis.land_cover --parquet results/csv_collection/<file>.parquet
"""

import glob
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── IGBP legend (MCD12C1 Majority_Land_Cover_Type_1) ─────────────────────────
IGBP_CLASS_NAMES = {
    0:  'water',
    1:  'evergreen needleleaf forest',
    2:  'evergreen broadleaf forest',
    3:  'deciduous needleleaf forest',
    4:  'deciduous broadleaf forest',
    5:  'mixed forests',
    6:  'closed shrubland',
    7:  'open shrublands',
    8:  'woody savannas',
    9:  'savannas',
    10: 'grasslands',
    11: 'permanent wetlands',
    12: 'croplands',
    13: 'urban and built-up',
    14: 'cropland/natural vegetation mosaic',
    15: 'snow and ice',
    16: 'barren or sparsely vegetated',
    255: 'fill',
}

# Collapse 17 IGBP classes into analysis groups.  Forest classes are pooled
# (individual near-cloud bins would be too thin); cropland mosaic joins
# cropland; urban/wetland kept as report-only bins.
IGBP_TO_GROUP = {
    0: 'water',
    1: 'forest', 2: 'forest', 3: 'forest', 4: 'forest', 5: 'forest',
    6: 'shrubland', 7: 'shrubland',
    8: 'savanna', 9: 'savanna',
    10: 'grassland',
    11: 'wetland',
    12: 'cropland', 14: 'cropland',
    13: 'urban',
    15: 'snow_ice',
    16: 'barren',
    255: 'fill',
}

# Plot/report order (dark → bright surfaces, then minor groups).
GROUP_ORDER = ['forest', 'savanna', 'shrubland', 'grassland', 'cropland',
               'barren', 'wetland', 'urban', 'snow_ice', 'water']

# Fixed colors so figures stay comparable across runs.
GROUP_COLORS = {
    'forest':    '#1b7837',
    'savanna':   '#a6611a',
    'shrubland': '#dfc27d',
    'grassland': '#7fbc41',
    'cropland':  '#e08214',
    'barren':    '#c51b7d',
    'wetland':   '#35978f',
    'urban':     '#762a83',
    'snow_ice':  '#92c5de',
    'water':     '#2166ac',
}

_CMG_RES = 0.05          # degrees per cell
_CMG_NROW = 3600         # row 0 at +90 lat
_CMG_NCOL = 7200         # col 0 at -180 lon
_SDS_NAME = 'Majority_Land_Cover_Type_1'


def find_mcd12c1_files(mcd12c1_dir) -> dict[int, Path]:
    """Return {year: path} for MCD12C1 HDF files found in *mcd12c1_dir*."""
    files = {}
    for f in sorted(glob.glob(str(Path(mcd12c1_dir) / 'MCD12C1.A*.hdf'))):
        m = re.search(r'MCD12C1\.A(\d{4})001\.', Path(f).name)
        if m:
            files[int(m.group(1))] = Path(f)
    return files


class LandCoverLookup:
    """Lazy per-year MCD12C1 IGBP class lookup on the 0.05-degree CMG grid."""

    def __init__(self, mcd12c1_dir):
        self.files = find_mcd12c1_files(mcd12c1_dir)
        if not self.files:
            raise FileNotFoundError(
                f"No MCD12C1.A*.hdf files found in {mcd12c1_dir}")
        self._grids: dict[int, np.ndarray] = {}
        self.years = np.array(sorted(self.files))
        logger.info(f"MCD12C1 years available: {list(self.years)}")

    def _grid(self, year: int) -> np.ndarray:
        if year not in self._grids:
            from pyhdf.SD import SD, SDC
            path = self.files[year]
            logger.info(f"Loading MCD12C1 {year}: {path.name}")
            sd = SD(str(path), SDC.READ)
            self._grids[year] = sd.select(_SDS_NAME).get()   # uint8 (3600, 7200)
            sd.end()
        return self._grids[year]

    def clamp_year(self, years: np.ndarray) -> np.ndarray:
        """Map each requested year to the nearest available MCD12C1 year."""
        idx = np.searchsorted(self.years, years).clip(0, len(self.years) - 1)
        left = (self.years[(idx - 1).clip(0)])
        take_left = (idx > 0) & (np.abs(years - left)
                                 < np.abs(self.years[idx] - years))
        return np.where(take_left, left, self.years[idx])

    def lookup(self, lat: np.ndarray, lon: np.ndarray,
               year: np.ndarray) -> np.ndarray:
        """Vectorized IGBP class lookup. NaN coords → 255 (fill)."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        year = self.clamp_year(np.asarray(year, dtype=np.int64))

        valid = np.isfinite(lat) & np.isfinite(lon)
        row = np.floor((90.0 - lat) / _CMG_RES)
        col = np.floor((lon + 180.0) / _CMG_RES)
        row = np.clip(np.nan_to_num(row), 0, _CMG_NROW - 1).astype(np.int32)
        col = np.clip(np.nan_to_num(col), 0, _CMG_NCOL - 1).astype(np.int32)

        out = np.full(lat.shape, 255, dtype=np.uint8)
        for yr in np.unique(year):
            m = (year == yr) & valid
            if m.any():
                out[m] = self._grid(int(yr))[row[m], col[m]]
        return out


def _years_from_dates(dates: pd.Series) -> np.ndarray:
    """Extract the year from a date column (bytes/str 'YYYY-MM-DD' or datetime)."""
    codes, uniques = pd.factorize(dates)
    uy = np.empty(len(uniques), dtype=np.int64)
    for i, u in enumerate(uniques):
        if isinstance(u, bytes):
            u = u.decode()
        uy[i] = int(str(u)[:4])
    return uy[codes]


def assign_land_cover(df: pd.DataFrame, mcd12c1_dir,
                      year: int | None = None) -> pd.DataFrame:
    """Add ``igbp_class`` (uint8) and ``igbp_group`` (category) columns.

    Year per sounding comes from the ``date`` column when present, else from
    the *year* argument. Returns the same DataFrame (columns added in place).
    """
    lut = LandCoverLookup(mcd12c1_dir)

    if 'date' in df.columns:
        years = _years_from_dates(df['date'])
    elif year is not None:
        years = np.full(len(df), int(year), dtype=np.int64)
    else:
        raise ValueError("DataFrame has no 'date' column — pass year= explicitly")

    df['igbp_class'] = lut.lookup(df['lat'].values, df['lon'].values, years)

    group_lut = np.full(256, 'fill', dtype=object)
    for cls, grp in IGBP_TO_GROUP.items():
        group_lut[cls] = grp
    df['igbp_group'] = pd.Categorical(
        group_lut[df['igbp_class'].values],
        categories=GROUP_ORDER + ['fill'])

    counts = df['igbp_group'].value_counts()
    logger.info("Land-cover groups assigned: "
                + ", ".join(f"{g}={n:,}" for g, n in counts.items() if n))
    return df


# ── CLI smoke test ────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    ap = argparse.ArgumentParser(description='Assign MCD12C1 IGBP classes to a parquet.')
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--mcd12c1-dir', default='data/MODIS/MCD12C1')
    ap.add_argument('--year', type=int, default=None,
                    help="Fallback year if the parquet has no 'date' column.")
    ap.add_argument('--out', default=None,
                    help='Optional output parquet (lon/lat/sfc_type/igbp columns).')
    args = ap.parse_args()

    cols = ['lon', 'lat', 'sfc_type', 'date']
    import pyarrow.parquet as pq
    avail = pq.read_schema(args.parquet).names
    df = pd.read_parquet(args.parquet, columns=[c for c in cols if c in avail])
    df = assign_land_cover(df, args.mcd12c1_dir, year=args.year)

    land = df[df['sfc_type'] == 1] if 'sfc_type' in df.columns else df
    print("\nLand soundings by IGBP group:")
    print(land['igbp_group'].value_counts().to_string())
    ocean = df[df['sfc_type'] == 0] if 'sfc_type' in df.columns else None
    if ocean is not None and len(ocean):
        agree = (ocean['igbp_group'] == 'water').mean()
        print(f"\nQC: sfc_type==0 mapped to 'water' for {agree:.1%} of ocean soundings")

    if args.out:
        df.to_parquet(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()

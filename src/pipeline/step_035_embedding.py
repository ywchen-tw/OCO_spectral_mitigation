"""
Phase 3.5: GEE Satellite Embedding Extraction
==============================================

Extracts 64-dimensional Google Satellite Embedding V1 (Annual) features for each
OCO-2 footprint polygon using Google Earth Engine.  Runs once per date after
Phase 3 and before Phase 4.

Setup (one-time):
    pip install earthengine-api pyarrow pandas
    earthengine authenticate        # opens browser OAuth
    # then pass your GCP project ID to run_phase_035()

Output: data/processing/{year}/{doy:03d}/embedding_stats_{date}.parquet
Schema: sounding_id (int64) | A00_mean…A63_mean (float32) | A00_stdDev…A63_stdDev (float32)
"""

import logging
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from .step_03_processing import OCO2Footprint

EMBEDDING_COLLECTION = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
N_DIMS = 64
BAND_NAMES = [f'A{i:02d}' for i in range(N_DIMS)]


# ---------------------------------------------------------------------------
# Helper: load footprints from Phase 3 cache and attach vertex data
# ---------------------------------------------------------------------------

def _load_vertex_data(processing_dir: Path, year: int, doy: int) -> dict:
    """Load lite_vertex_data.pkl for the given date (day-level cache)."""
    cache_path = processing_dir / str(year) / f"{doy:03d}" / "lite_vertex_data.pkl"
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"  Loaded vertex data: {len(data)} entries from {cache_path}")
        return data
    except Exception as e:
        logger.warning(f"  Failed to load vertex cache {cache_path}: {e}")
        return {}


def load_footprints(processing_dir: Path, year: int, doy: int) -> Dict[int, OCO2Footprint]:
    """
    Load OCO2Footprint objects from all orbit footprints.pkl files for the date,
    then attach vertex data from the day-level lite_vertex_data.pkl cache.
    """
    day_dir = processing_dir / str(year) / f"{doy:03d}"
    all_footprints: Dict[int, OCO2Footprint] = {}

    if not day_dir.exists():
        logger.warning(f"  Processing day directory not found: {day_dir}")
        return all_footprints

    for orbit_dir in sorted(day_dir.iterdir()):
        if not orbit_dir.is_dir():
            continue
        fp_file = orbit_dir / "footprints.pkl"
        if not fp_file.exists():
            continue
        try:
            with open(fp_file, 'rb') as f:
                footprints = pickle.load(f)
            all_footprints.update(footprints)
            logger.info(f"  Loaded {len(footprints)} footprints from {orbit_dir.name}/footprints.pkl")
        except Exception as e:
            logger.warning(f"  Failed to load {fp_file}: {e}")

    # Attach vertex data (handles caches written before Phase A)
    vertex_data = _load_vertex_data(processing_dir, year, doy)
    if vertex_data:
        attached = 0
        for sid, fp in all_footprints.items():
            if sid in vertex_data and fp.vertex_lon is None:
                fp.vertex_lon, fp.vertex_lat = vertex_data[sid]
                attached += 1
        if attached:
            logger.info(f"  Attached vertex data to {attached} footprints from day-level cache")

    return all_footprints


# ---------------------------------------------------------------------------
# B1: footprint dict → GEE FeatureCollection
# ---------------------------------------------------------------------------

def build_feature_collection(footprints: Dict[int, OCO2Footprint],
                             obs_year: int) -> 'ee.FeatureCollection':
    """Convert OCO2Footprint objects (with vertex data) into a GEE FeatureCollection."""
    if not EE_AVAILABLE:
        raise ImportError("earthengine-api not installed: pip install earthengine-api")

    features = []
    skipped = 0
    for sid, fp in footprints.items():
        if fp.vertex_lon is None or fp.vertex_lat is None:
            skipped += 1
            continue
        coords = list(zip(fp.vertex_lon.tolist(), fp.vertex_lat.tolist()))
        # geodesic=False (3rd positional arg): planar geometry avoids GEE
        # interpreting the small polygon as its global complement when vertex
        # winding order is ambiguous in geodesic mode.
        poly = ee.Geometry.Polygon([coords], None, False)
        features.append(ee.Feature(poly, {'sounding_id': int(sid), 'year': obs_year}))

    if skipped:
        logger.warning(f"  Skipped {skipped} footprints without vertex data")
    logger.info(f"  FeatureCollection: {len(features)} footprint polygons")
    return ee.FeatureCollection(features)


# ---------------------------------------------------------------------------
# B2: load annual embedding image
# ---------------------------------------------------------------------------

def load_embedding_image(obs_year: int) -> 'ee.Image':
    """Load the annual embedding image; clamp to 2017 (earliest available year)."""
    if not EE_AVAILABLE:
        raise ImportError("earthengine-api not installed: pip install earthengine-api")

    year = max(obs_year, 2017)  # 2016 observations → use 2017 embedding
    # Filter by date range — the collection uses system:time_start (standard GEE
    # temporal property), not a custom 'year' metadata field.
    col = (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
    )
    n_images = col.limit(1).size().getInfo()
    if n_images == 0:
        total = ee.ImageCollection(EMBEDDING_COLLECTION).limit(1).size().getInfo()
        if total == 0:
            raise ValueError(
                f"{EMBEDDING_COLLECTION} returned no images at all. "
                f"Verify your GEE account has access to this collection."
            )
        raise ValueError(
            f"No images found for year={year} in {EMBEDDING_COLLECTION} "
            f"(filterDate {year}-01-01 – {year + 1}-01-01). "
            f"Collection coverage is 2017–2024; check that {year} is in range."
        )
    embedding = col.first()
    logger.info(f"  Embedding image: {EMBEDDING_COLLECTION} year={year}")
    return embedding


# ---------------------------------------------------------------------------
# B3: reduceRegion — mean + stdDev per footprint
# ---------------------------------------------------------------------------

def add_embedding_stats(fc: 'ee.FeatureCollection',
                        embedding: 'ee.Image') -> 'ee.FeatureCollection':
    """Map mean+stdDev reduceRegion over the FeatureCollection at 10 m scale."""
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)

    def reduce_feature(feat):
        stats = embedding.reduceRegion(
            reducer=reducer,
            geometry=feat.geometry(),
            scale=10,
            maxPixels=50000,  # ~29 000 pixels per ~1.3×2.25 km footprint at 10 m
        )
        return feat.set(stats)

    return fc.map(reduce_feature)


# ---------------------------------------------------------------------------
# B4a: synchronous export (dev / single-orbit)
# ---------------------------------------------------------------------------

def _fc_info_to_dataframe(fc_info: dict) -> 'pd.DataFrame':
    """Parse FeatureCollection.getInfo() result into a tidy DataFrame."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not installed: pip install pandas")

    records = []
    for feat in fc_info.get('features', []):
        props = feat.get('properties', {})
        record: dict = {'sounding_id': props.get('sounding_id')}
        for band in BAND_NAMES:
            record[f'{band}_mean'] = props.get(f'{band}_mean')
            record[f'{band}_stdDev'] = props.get(f'{band}_stdDev')
        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        df['sounding_id'] = df['sounding_id'].astype('int64')
        float_cols = [c for c in df.columns if c != 'sounding_id']
        df[float_cols] = df[float_cols].astype('float32')
    return df


def extract_embeddings_sync(footprints: Dict[int, OCO2Footprint],
                            obs_year: int,
                            output_path: Path) -> 'pd.DataFrame':
    """
    Synchronous extraction via getInfo() — suitable for single orbits / dev runs.
    Blocks until GEE returns results; avoid for >~2 000 footprints.
    """
    fc = build_feature_collection(footprints, obs_year)
    embedding = load_embedding_image(obs_year)
    result_fc = add_embedding_stats(fc, embedding)

    logger.info("  Fetching results via getInfo() — this may take a minute...")
    fc_info = result_fc.getInfo()
    df = _fc_info_to_dataframe(fc_info)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"  Saved {len(df)} records → {output_path}")
    return df


# ---------------------------------------------------------------------------
# B4b: batch export to Google Drive (full dataset)
# ---------------------------------------------------------------------------

def export_embeddings_to_drive(footprints: Dict[int, OCO2Footprint],
                               obs_year: int,
                               description: str,
                               folder: str = 'oco_embeddings') -> str:
    """
    Submit a GEE batch export task to Google Drive.
    Suitable for full-dataset runs (~84K soundings across many dates).

    Returns the GEE task ID; poll status with ee.data.getTaskStatus(task_id).
    After download, convert with concat_drive_csvs().
    """
    fc = build_feature_collection(footprints, obs_year)
    embedding = load_embedding_image(obs_year)
    result_fc = add_embedding_stats(fc, embedding)

    task = ee.batch.Export.table.toDrive(
        collection=result_fc,
        description=description,
        folder=folder,
        fileFormat='CSV',
    )
    task.start()
    logger.info(f"  GEE batch task started: {description!r} (ID: {task.id})")
    return task.id


def get_task_status(task_id: str) -> dict:
    """Poll the status of a GEE export task."""
    if not EE_AVAILABLE:
        raise ImportError("earthengine-api not installed")
    statuses = ee.data.getTaskStatus(task_id)
    return statuses[0] if statuses else {}


def concat_drive_csvs(csv_paths: list, output_path: Path) -> 'pd.DataFrame':
    """
    Concatenate per-orbit CSV files downloaded from Google Drive into a single
    parquet file.  Call this after all batch tasks have completed.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not installed: pip install pandas")

    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Failed to read {p}: {e}")

    if not dfs:
        raise ValueError("No CSV files could be loaded")

    combined = pd.concat(dfs, ignore_index=True)
    combined['sounding_id'] = combined['sounding_id'].astype('int64')
    float_cols = [c for c in combined.columns if c != 'sounding_id']
    combined[float_cols] = combined[float_cols].astype('float32')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    logger.info(f"  Merged {len(dfs)} CSVs → {len(combined)} rows → {output_path}")
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase_035(target_date: datetime,
                  data_dir: Path,
                  gcp_project: str,
                  use_batch: bool = False,
                  drive_folder: str = 'oco_embeddings',
                  limit_orbits: int = None) -> Optional['pd.DataFrame']:
    """
    Phase 3.5 entry point: extract GEE satellite embeddings for all OCO-2
    footprints on target_date.

    Args:
        target_date:  Observation date.
        data_dir:     Project data root (same convention as other phases).
        gcp_project:  GCP project ID passed to ee.Initialize().
        use_batch:    Submit to GEE batch export (True) or use synchronous
                      getInfo (False, default).  Batch is needed for full runs.
        drive_folder: Google Drive folder name for batch CSV outputs.
        limit_orbits: If set, restrict extraction to the first N granules.
                      Useful for a quick test without processing all orbits.
                      Writes to a separate *_test.parquet and skips cache.

    Returns:
        DataFrame with embedding stats in sync mode; None in batch mode.
    """
    if not EE_AVAILABLE:
        logger.error("earthengine-api not installed.  Run: pip install earthengine-api")
        return None
    if not PANDAS_AVAILABLE:
        logger.error("pandas not installed.  Run: pip install pandas pyarrow")
        return None

    ee.Initialize(project=gcp_project)

    year = target_date.year
    doy = target_date.timetuple().tm_yday
    date_str = target_date.strftime('%Y-%m-%d')

    processing_dir = Path(data_dir) / 'processing'
    is_test = limit_orbits is not None
    suffix = '_test' if is_test else ''
    output_path = (processing_dir / str(year) / f"{doy:03d}"
                   / f"embedding_stats_{date_str}{suffix}.parquet")

    if not is_test and output_path.exists():
        logger.info(f"  Embedding stats already cached: {output_path}")
        return pd.read_parquet(output_path)

    logger.info(f"\n[Phase 3.5] GEE Embedding Extraction — {date_str}"
                + (f" (test: {limit_orbits} granule(s))" if is_test else ""))

    footprints = load_footprints(processing_dir, year, doy)
    if not footprints:
        logger.error("  No footprints found — run Phase 3 first")
        return None

    with_vertices = {sid: fp for sid, fp in footprints.items() if fp.vertex_lon is not None}
    logger.info(f"  Footprints: {len(footprints)} total, {len(with_vertices)} with vertex data")

    if not with_vertices:
        logger.error("  No footprints have vertex data — ensure Phase A (A1–A3) is complete")
        return None

    # Restrict to the first N granules for testing
    if is_test:
        seen_granules: list = []
        limited: Dict[int, OCO2Footprint] = {}
        for sid, fp in with_vertices.items():
            if fp.granule_id not in seen_granules:
                if len(seen_granules) >= limit_orbits:
                    break
                seen_granules.append(fp.granule_id)
            limited[sid] = fp
        logger.info(f"  Limited to {len(seen_granules)} granule(s): {seen_granules}")
        with_vertices = limited

    if use_batch:
        task_id = export_embeddings_to_drive(
            with_vertices, year,
            description=f'oco_embedding_{date_str}{suffix}',
            folder=drive_folder,
        )
        logger.info(f"  Batch task submitted: {task_id}")
        logger.info(f"  Poll status: get_task_status('{task_id}')")
        logger.info(f"  After download, merge with: concat_drive_csvs(csv_paths, output_path)")
        return None

    return extract_embeddings_sync(with_vertices, year, output_path)

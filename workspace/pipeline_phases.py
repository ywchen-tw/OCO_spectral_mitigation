"""Per-phase runners for the OCO-2/MODIS collocation pipeline
(run_phase_1 … run_phase_5: metadata → ingestion → processing →
geometry → synthesis).

Split out of demo_combined.py (2026-07, review §7.4).  demo_combined.py
is the orchestrator/CLI and imports these; each phase stays independently
callable with the same signatures as before.
"""

import sys
import os
import logging
import argparse
import re
import gc
from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np
import pickle
import platform
import shutil
import traceback
import h5py
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.phase_01_metadata import OCO2MetadataRetriever
from pipeline.phase_02_ingestion import DataIngestionManager, DownloadedFile
from pipeline.phase_03_processing import SpatialProcessor
from pipeline.phase_04_geometry import GeometryProcessor, CollocationResult
from config import Config
from constants import (AQUA_FREE_DRIFT_YEAR, modis_match_buffer_minutes,
                       CLOUD_DIST_BAND_WIDTH_DEG, CLOUD_DIST_BAND_OVERLAP_DEG)
from utils import setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



from demo_utils import print_banner, print_step_header

logger = logging.getLogger(__name__)


# ============================================================================
# Step Execution Functions
# ============================================================================

def run_phase_1(target_date: datetime, orbit: Optional[str] = None, 
                mode: str = None) -> Tuple[Dict, bool]:
    """
    Step 1: Metadata Acquisition and Temporal Filtering
    
    Args:
        target_date: Target date for metadata retrieval
        orbit: Optional specific orbit number
        mode: Optional viewing mode filter ('GL' or 'ND' or 'TG')
    
    Returns:
        Tuple of (metadata_dict, success_flag)
    """
    print_step_header(1, "Metadata Acquisition and Temporal Filtering")
    
    try:
        logger.info(f"Target Date: {target_date.date()}")
        if orbit:
            logger.info(f"Orbit Filter: {orbit}")
        if mode:
            logger.info(f"Mode Filter: {mode}")
        
        retriever = OCO2MetadataRetriever()
        
        logger.info("\n[Step 1] Fetching OCO-2 L1B Science XML from GES DISC...")
        xml_content = retriever.fetch_oco2_xml(target_date, orbit, mode)
        logger.info(f"✓ Retrieved XML metadata ({len(xml_content):,} bytes)")
        
        logger.info("[Step 2] Parsing orbit information...")
        granules = retriever.parse_orbit_info(xml_content)
        logger.info(f"✓ Found {len(granules)} granule(s)")
        
        if not granules:
            logger.error(f"✗ No OCO-2 data found for the specified date ({target_date.date()})")
            logger.error("  This date may not have any OCO-2 observations available.")
            return {}, False
        
        logger.info("[Step 3] Extracting temporal window...")
        start_time_arr, end_time_arr = retriever.extract_temporal_window(granules, orbit, mode)
        
        metadata = {
            'target_date': target_date,
            'granules': granules,
            'start_time': start_time_arr,
            'end_time': end_time_arr,
            'num_granules': len(granules),
        }
        
        logger.info(f"\n✓ Step 1 Complete: {len(granules)} granule(s) identified")
        return metadata, True
        
    except Exception as e:
        logger.error(f"✗ Step 1 Failed: {e}")
        return {}, False


def run_phase_2(target_date: datetime, data_dir: Path,
                dry_run: bool = False,
                force_download: bool = False, limit_granules: Optional[int] = None,
                orbit: Optional[str] = None, granule_suffix: Optional[str] = None,
                mode: Optional[str] = None) -> Tuple[Dict, bool]:
    """
    Step 2: Targeted Data Ingestion

    Args:
        target_date: Target date for data download
        data_dir: Data storage directory
        dry_run: Only check file existence without downloading
        force_download: Force re-download even if status file exists
        limit_granules: Limit to first N granules (for testing)
        orbit: Optional orbit str filter (passed through from --orbit)
        mode: Optional viewing mode filter (passed through from --mode)

    Returns:
        Tuple of (file_info_dict, success_flag)
    """
    print_step_header(2, "Targeted Data Ingestion")

    try:
        ingestion_manager = DataIngestionManager(
            output_dir=str(data_dir),
            dry_run=dry_run
        )

        logger.info("\n[Step 1] Downloading OCO-2 and MODIS products...")
        logger.info("  OCO-2: L1B Science, L2 Lite, L2 Met, L2 CO2Prior")
        logger.info("  MODIS: MYD35_L2 (cloud mask), MYD03 (geolocation)")
        if orbit:
            logger.info(f"  Orbit filter: {orbit}")
        if mode:
            logger.info(f"  Mode filter: {mode}")
        result = ingestion_manager.download_all_for_date(
            target_date=target_date,
            orbit_filter=orbit,
            mode_filter=mode,
            include_modis=True,
            limit_granules=limit_granules,
            skip_existing=(not force_download)
        )
        
        oco2_files = result.get('oco2_files', [])
        modis_files = result.get('modis_files', [])
        granules = result.get('granules', [])
        
        logger.info(f"\n✓ Downloaded/loaded {len(oco2_files)} OCO-2 file(s)")
        logger.info(f"✓ Downloaded/loaded {len(modis_files)} MODIS file(s)")

        if not granules:
            logger.error(
                "✗ Step 2 found zero OCO-2 granules for %s. "
                "This is an ingestion/metadata failure or a true no-data date; "
                "Phase 3 will not be started.",
                target_date.date()
            )
            return {}, False

        if not oco2_files:
            logger.error(
                "✗ Step 2 found %d granule(s) but no OCO-2 files were ready. "
                "Check GES DISC authentication, CMR fallback results, and failed download logs.",
                len(granules)
            )
            return {}, False

        lite_files = [
            file
            for file in oco2_files
            if file.product_type in {"OCO2_L2_Lite", "L2_Lite"}
            and Path(file.filepath).suffix == ".nc4"
        ]
        if not lite_files:
            year = target_date.year
            doy = target_date.timetuple().tm_yday
            expected_dir = data_dir / "OCO2" / str(year) / f"{doy:03d}"
            failed_lite = [
                failure
                for failure in result.get("stats", {}).get("failed_downloads", [])
                if failure.get("product_type") == "L2_Lite"
            ]
            logger.error(
                "✗ Step 2 did not produce the required OCO-2 L2 Lite .nc4 file "
                "for %s. Spectral fitting requires a day-level Lite file in:\n  %s",
                target_date.date(),
                expected_dir,
            )
            if failed_lite:
                logger.error("  L2 Lite failed download/query records:")
                for failure in failed_lite[:5]:
                    logger.error(
                        "    granule_id=%s url=%s",
                        failure.get("granule_id"),
                        failure.get("url"),
                    )
            return {}, False
        
        file_info = {
            'oco2_files': oco2_files,
            'modis_files': modis_files,
            'total_files': len(oco2_files) + len(modis_files),
        }
        
        logger.info(f"\n✓ Step 2 Complete: {file_info['total_files']} file(s) ready")
        return file_info, True
        
    except Exception as e:
        logger.error(f"✗ Step 2 Failed: {e}")
        return {}, False


def run_phase_3(target_date: datetime, data_dir: Path,
                force_recompute: bool = False) -> Tuple[Dict, bool]:
    """
    Step 3: Spatial and Bitmask Processing

    Processes OCO-2 footprints and MODIS cloud data. Automatically processes any
    granules that don't have cached results.

    Args:
        target_date: Target date for processing
        data_dir: Data storage directory
        force_recompute: If True, re-evaluate placeholder-only granules (post-2022
            granules that got a -999 cloud-distance result because no MODIS was
            collocated) so newly-available MODIS files are picked up.

    Returns:
        Tuple of (processing_info_dict, success_flag)
    """
    print_step_header(3, "Spatial and Bitmask Processing")
    
    try:
        data_dir = Path(data_dir)
        year = target_date.year
        doy = target_date.timetuple().tm_yday
        
        # Check how many OCO-2 L1B files exist for this date
        logger.info("\n[Step 1] Checking downloaded OCO-2 data...")
        oco2_l1b_dir = data_dir / "OCO2" / str(year) / f"{doy:03d}"
        
        oco2_granules = set()
        if oco2_l1b_dir.exists():
            for subdir in oco2_l1b_dir.glob("*"):
                if subdir.is_dir() and any(
                    p.is_file() and "L1BSC" in p.name.upper()
                    for p in subdir.glob("*.h5")
                ):
                    oco2_granules.add(subdir.name)
        
        if oco2_granules:
            logger.info(f"✓ Found {len(oco2_granules)} OCO-2 granule(s) in data directory")
        else:
            logger.warning("⚠ No OCO-2 granules found in data directory")
            return {}, False
        
        # Check if Step 3 cache already exists
        logger.info("\n[Step 2] Checking for Step 3 cache...")
        processing_day_dir = data_dir / "processing" / str(year) / f"{doy:03d}"

        # Find the newest mtime of any MODIS HDF file for this date.
        # A cache built before that mtime is stale (new/re-downloaded MODIS files
        # are available) and must be rebuilt so the matching reflects current data.
        latest_modis_mtime = 0.0
        for modis_dir in [
            data_dir / "MODIS" / "MYD35_L2" / str(year) / f"{doy:03d}",
            data_dir / "MODIS" / "MYD03"    / str(year) / f"{doy:03d}",
        ]:
            if modis_dir.exists():
                for hdf in modis_dir.glob("*.hdf"):
                    latest_modis_mtime = max(latest_modis_mtime, hdf.stat().st_mtime)

        cached_granules = set()
        if processing_day_dir.exists():
            cache_dirs = [d for d in processing_day_dir.glob("*") if d.is_dir()]
            for cache_dir in cache_dirs:
                combined_files = list(cache_dir.glob("granule_combined_*.pkl"))
                if combined_files:
                    cache_mtime = min(f.stat().st_mtime for f in combined_files)
                    if latest_modis_mtime > cache_mtime:
                        logger.info(f"  ↻ {cache_dir.name}: newer MODIS files on disk — cache invalidated, will reprocess")
                    else:
                        cached_granules.add(cache_dir.name)
                        logger.info(f"  ✓ {cache_dir.name}: {len(combined_files)} combined cache file(s) (up-to-date)")
                elif (cache_dir / "phase4_results.pkl").exists():
                    # Placeholder-only granule: Phase 3 wrote -999 cloud-distance
                    # results directly because no MODIS was collocated (post-2022).
                    # Under --force-recompute, treat it as missing so the MODIS
                    # collocation is re-checked (new MODIS may now be on disk).
                    if force_recompute:
                        logger.info(f"  ↻ {cache_dir.name}: -999 placeholder — force-recompute, re-checking MODIS collocation")
                    else:
                        cached_granules.add(cache_dir.name)
                        logger.info(f"  ✓ {cache_dir.name}: -999 placeholder result (no MODIS overlap)")

        # Check if all granules are cached
        missing_granules = oco2_granules - cached_granules

        if not missing_granules and cached_granules:
            logger.info(f"\n✓ All {len(cached_granules)} granule(s) already cached")
            processing_info = {
                'cached': True,
                'granules': len(cached_granules),
                'cache_location': str(processing_day_dir),
            }
            logger.info(f"✓ Step 3 Complete: Using existing cache for {len(cached_granules)} granule(s)")
            return processing_info, True
        
        # Process missing granules
        if missing_granules:
            logger.info(f"\n[Step 3] Processing {len(missing_granules)} missing granule(s)...")
            if cached_granules:
                logger.info(f"  ({len(cached_granules)} granule(s) already cached)")
            
            spatial_processor = SpatialProcessor(data_dir=str(data_dir))
            
            # Load all OCO-2 and MODIS files for this date
            try:
                # Get OCO-2 files for missing granules
                oco2_files = []
                # Track Lite directories already added to avoid duplicates when
                # multiple cross-date granules share the same previous date.
                _lite_dirs_added: set = {oco2_l1b_dir}
                for granule_id in sorted(missing_granules):
                    granule_dir = oco2_l1b_dir / granule_id
                    if granule_dir.exists():
                        for file_path in granule_dir.glob("*"):
                            if file_path.is_file():
                                product_type = ""
                                if "L1bSc" in file_path.name:
                                    product_type = "L1B_Science"
                                    # Cross-date granule detection: OCO-2 Lite files are
                                    # organised by orbit date, so if the L1B filename date
                                    # differs from target_date the sounding IDs live in the
                                    # *previous* day's Lite file, not the target day's.
                                    # Load that Lite file now so quality filtering works.
                                    _m = re.search(r'_(\d{6})_B', file_path.name)
                                    if _m and _m.group(1) != target_date.strftime("%y%m%d"):
                                        _l1b_ds = _m.group(1)
                                        _l1b_dt = datetime(2000 + int(_l1b_ds[:2]),
                                                           int(_l1b_ds[2:4]),
                                                           int(_l1b_ds[4:6]))
                                        _prev_dir = (data_dir / "OCO2"
                                                     / str(_l1b_dt.year)
                                                     / f"{_l1b_dt.timetuple().tm_yday:03d}")
                                        if _prev_dir.exists() and _prev_dir not in _lite_dirs_added:
                                            for _lp in _prev_dir.glob("*.nc4"):
                                                if _lp.is_file() and "Lt" in _lp.name:
                                                    oco2_files.append(DownloadedFile(
                                                        filepath=_lp,
                                                        product_type="L2_Lite",
                                                        target_year=year,
                                                        target_doy=doy,
                                                        granule_id=_lp.name,
                                                        file_size_mb=_lp.stat().st_size / (1024 * 1024),
                                                        download_time_seconds=0.0
                                                    ))
                                                    logger.info(
                                                        f"    Cross-date granule detected: {file_path.name[:45]}"
                                                        f"\n      Loading Lite from {_l1b_dt.date()}: {_lp.name}"
                                                    )
                                            _lite_dirs_added.add(_prev_dir)
                                        elif _prev_dir not in _lite_dirs_added:
                                            logger.warning(
                                                f"    Cross-date granule {file_path.name[:45]}: "
                                                f"previous-day Lite dir not found ({_prev_dir}). "
                                                f"Sounding IDs will not be quality-filtered."
                                            )

                                if product_type:
                                    file_size = file_path.stat().st_size / (1024 * 1024)
                                    oco2_files.append(DownloadedFile(
                                        filepath=file_path,
                                        target_year=year,
                                        target_doy=doy,
                                        product_type=product_type,
                                        granule_id=file_path.name,
                                        file_size_mb=file_size,
                                        download_time_seconds=0.0
                                    ))
                    # get lite file at the parent directory level (e.g. oco2_L2Lite_34145a_201201_20120101T000000Z.nc)
                    for file_path in oco2_l1b_dir.glob("*.nc4"):
                        if file_path.is_file() and "Lt" in file_path.name:
                            product_type = "L2_Lite"
                            file_size = file_path.stat().st_size / (1024 * 1024)
                            oco2_files.append(DownloadedFile(
                                filepath=file_path,
                                product_type=product_type,
                                target_year=year,
                                target_doy=doy,
                                granule_id=file_path.name,
                                file_size_mb=file_size,
                                download_time_seconds=0.0
                            ))
                
                # Load all MODIS files
                modis_files = []
                myd35_dir = data_dir / "MODIS" / "MYD35_L2" / str(year) / f"{doy:03d}"
                myd03_dir = data_dir / "MODIS" / "MYD03" / str(year) / f"{doy:03d}"
                
                if myd35_dir.exists():
                    for modis_file in myd35_dir.glob("*.hdf"):
                        file_size = modis_file.stat().st_size / (1024 * 1024)
                        modis_files.append(DownloadedFile(
                            filepath=modis_file,
                            product_type="MYD35_L2",
                            target_year=year,
                            target_doy=doy,
                            granule_id=modis_file.name,
                            file_size_mb=file_size,
                            download_time_seconds=0.0
                        ))
                
                if myd03_dir.exists():
                    for modis_file in myd03_dir.glob("*.hdf"):
                        file_size = modis_file.stat().st_size / (1024 * 1024)
                        modis_files.append(DownloadedFile(
                            filepath=modis_file,
                            product_type="MYD03",
                            target_year=year,
                            target_doy=doy,
                            granule_id=modis_file.name,
                            file_size_mb=file_size,
                            download_time_seconds=0.0
                        ))
                
                logger.info(f"  Loaded {len(oco2_files)} OCO-2 file(s) and {len(modis_files)} MODIS file(s)")
                
                # Extract footprints for all viewing modes so GL, ND, and TG granules
                # are all available when the per-granule loop looks them up.
                logger.info("  Extracting OCO-2 footprints (all viewing modes)...")
                oco2_footprints = spatial_processor.extract_oco2_footprints(oco2_files, 
                                                                            target_year=year,
                                                                            target_doy=doy, viewing_mode=None)
                footprints_by_granule = spatial_processor.group_footprints_by_granule(oco2_footprints)
                
                # Calculate time windows for each OCO-2 granule
                logger.info("  Calculating temporal windows for OCO-2 granules...")
                granule_time_ranges = {}
                for granule_id_full, granule_footprints in footprints_by_granule.items():
                    if not granule_footprints:
                        continue
                    times = [fp.sounding_time for fp in granule_footprints]
                    start_time = min(times)
                    end_time = max(times)
                    granule_time_ranges[granule_id_full] = (start_time, end_time)
                    
                    print(f"    {granule_id_full}: {start_time.isoformat()} to {end_time.isoformat()} ({len(granule_footprints)} footprints)")
                
                # Match MODIS files to OCO-2 granules based on temporal proximity.
                # Buffer is adaptive (mirrors the logic in match_temporal_windows):
                # ±10 min before AQUA_FREE_DRIFT_YEAR, ±20 min from then on.
                # Night passes are excluded here so that off-scene MODIS data
                # (which can fall inside the time window near orbit boundaries)
                # does not contaminate the cloud pixels.
                logger.info("  Matching MODIS files to OCO-2 granules...")
                modis_to_oco2_mapping = {}

                buffer_minutes = modis_match_buffer_minutes(year)
                buffer_seconds = buffer_minutes * 60
                logger.info(f"  Using ±{buffer_minutes} min matching buffer (year={year})")

                # Extract MODIS cloud masks - process per granule with orbit_id
                logger.info("  Extracting MODIS cloud masks per granule...")

                for modis_file in modis_files:
                    if modis_file.product_type != 'MYD35_L2':
                        continue

                    # Extract MODIS observation time from filename
                    match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', modis_file.filepath.name)
                    if not match:
                        continue

                    year_m = int(match.group(1))
                    doy_m  = int(match.group(2))
                    hhmm   = match.group(3)
                    hour   = int(hhmm[:2])
                    minute = int(hhmm[2:4])
                    modis_date = datetime(year_m, 1, 1) + timedelta(days=doy_m - 1)
                    modis_time = modis_date.replace(hour=hour, minute=minute)

                    # Skip night passes (bit 3 of byte 0 of the MODIS cloud mask).
                    # Night MODIS granules observe a completely different scene and
                    # must not be collocated with daytime OCO-2 soundings.
                    try:
                        from pyhdf.SD import SD as _SD
                        _hdf = _SD(str(modis_file.filepath))
                        _cm  = _hdf.select('Cloud_Mask')
                        _b0  = _cm.get()
                        _cm.endaccess()
                        _hdf.end()
                        # Determine byte-0 regardless of storage order
                        if _b0.ndim == 3 and _b0.shape[0] == 6:
                            _byte0 = _b0[0, :, :]
                        elif _b0.ndim == 3 and _b0.shape[2] == 6:
                            _byte0 = _b0[:, :, 0]
                        else:
                            _byte0 = _b0
                        _day_flag = (_byte0 >> 3) & 0b1
                        _is_day = int(np.sum(_day_flag == 1)) > int(np.sum(_day_flag == 0))
                        if not _is_day:
                            logger.debug(f"    Skipping night pass: {modis_file.filepath.name}")
                            continue
                    except Exception as _e:
                        logger.debug(f"    Could not check day/night for {modis_file.filepath.name}: {_e}")
                        # Include the file if the check cannot be performed

                    # Assign this MODIS file to ALL OCO-2 granules whose window it
                    # falls in.  A single 5-minute swath can overlap the time windows
                    # of two adjacent OCO-2 orbits; picking only the "closest" would
                    # silently drop cloud data for the other granule.
                    for oco2_granule_id, (start_time, end_time) in granule_time_ranges.items():
                        window_start = start_time - timedelta(seconds=buffer_seconds)
                        window_end   = end_time   + timedelta(seconds=buffer_seconds)
                        if window_start <= modis_time <= window_end:
                            modis_to_oco2_mapping.setdefault(oco2_granule_id, []).append(modis_file)

                # --- Continuity check: warn if any ~5-min slot in an OCO-2 granule
                # has no corresponding MODIS granule. ---
                logger.info("  Checking MODIS granule continuity per OCO-2 granule...")
                for oco2_gid, matched in modis_to_oco2_mapping.items():
                    g_start, g_end = granule_time_ranges[oco2_gid]
                    duration_min = (g_end - g_start).total_seconds() / 60.0
                    expected_n = max(1, int(duration_min / 5) + 1)

                    mtimes = []
                    for mf in matched:
                        mm = re.search(r'A(\d{4})(\d{3})\.(\d{4})', mf.filepath.name)
                        if mm:
                            mt = (datetime(int(mm.group(1)), 1, 1)
                                  + timedelta(days=int(mm.group(2)) - 1,
                                              hours=int(mm.group(3)[:2]),
                                              minutes=int(mm.group(3)[2:])))
                            mtimes.append(mt)
                    mtimes.sort()

                    logger.info(
                        f"    {oco2_gid[:30]}: {len(mtimes)} MODIS granule(s) matched "
                        f"(expected ~{expected_n} for {duration_min:.0f}-min orbit)"
                    )
                    for i in range(len(mtimes) - 1):
                        gap = (mtimes[i + 1] - mtimes[i]).total_seconds() / 60.0
                        if gap > 6.0:  # >6 min gap → at least one 5-min slot missing
                            logger.warning(
                                f"    ⚠ MODIS coverage gap: "
                                f"{mtimes[i].strftime('%H:%M')} → "
                                f"{mtimes[i+1].strftime('%H:%M')} "
                                f"({gap:.0f} min) for {oco2_gid}"
                            )
                    # Warn about unmatched granules (no MODIS at all)
                for oco2_gid, _ in granule_time_ranges.items():
                    if oco2_gid not in modis_to_oco2_mapping:
                        logger.warning(f"    ⚠ No MODIS granules matched for {oco2_gid}")
                
                # Process each missing granule
                for granule_id in sorted(missing_granules):
                    logger.info(f"\n  Processing granule: {granule_id}")
                    
                    # Find the full granule ID whose _extract_short_orbit_id matches
                    # the folder name.  The folder name is now "{orbit_id}_{mode}"
                    # (e.g. "34145a_GL"), which is NOT a simple substring of the full
                    # L1B filename ("oco2_L1bScGL_34145a_201201_..."), so a plain
                    # `granule_id in fg_id` check would silently fail for every granule.
                    full_granule_id = None
                    for fg_id in footprints_by_granule.keys():
                        if spatial_processor._extract_short_orbit_id(fg_id) == granule_id:
                            full_granule_id = fg_id
                            break
                    
                    if not full_granule_id:
                        logger.warning(f"    ⚠ No footprints found for {granule_id}")
                        # remove this granule from missing_granules so we don't try to process it again next time
                        oco2_granules.discard(granule_id)
                        continue
                    
                    # Get footprints for this granule
                    granule_footprints = {sid: fp for sid, fp in oco2_footprints.items() 
                                         if fp.granule_id == full_granule_id}
                    
                    # Get matched MODIS files for this granule
                    matched_modis_files = modis_to_oco2_mapping.get(full_granule_id, [])
                    
                    logger.info(f"    Matched {len(matched_modis_files)} MODIS file(s)")
                    logger.info(f"    Found {len(granule_footprints)} footprints")
                    
                    # Create cache directory for this granule
                    cache_dir = processing_day_dir / granule_id
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    
                        # Extract MODIS cloud masks for this granule with proper orbit_id
                    # This will save myd35_*.pkl files in the correct orbit folder
                    if matched_modis_files:
                        # Find matching MYD03 files
                        matched_myd03 = []
                        for myd35_file in matched_modis_files:
                            match = re.search(r'A(\d{4}\d{3}\.\d{4})', myd35_file.filepath.name)
                            if match:
                                time_id = 'A' + match.group(1)
                                for myd03_file in modis_files:
                                    if myd03_file.product_type == 'MYD03' and time_id in myd03_file.filepath.name:
                                        matched_myd03.append(myd03_file)
                                        break
                        
                        granule_cloud_masks = spatial_processor.extract_modis_cloud_mask(
                            target_year=year,
                            target_doy=doy,
                            modis_files=matched_modis_files,
                            myd03_files=matched_myd03 if matched_myd03 else None,
                            oco2_orbit_id=granule_id
                        )
                        
                        # Free memory: delete matched file references
                        del matched_myd03
                    else:
                        granule_cloud_masks = {}
                        if target_date >= datetime(AQUA_FREE_DRIFT_YEAR, 1, 1):
                            # Aqua free-drift era: a MODIS gap over an OCO-2
                            # orbit is expected and legitimate. Rather than silently
                            # dropping these soundings, emit placeholder cloud-distance
                            # results flagged with the -999 sentinel so they still
                            # appear in the Phase 4/5 output.
                            placeholder_results = [
                                CollocationResult(
                                    sounding_id=fp.sounding_id,
                                    granule_id=fp.granule_id,
                                    footprint_lat=fp.latitude,
                                    footprint_lon=fp.longitude,
                                    viewing_mode=fp.viewing_mode,
                                    nearest_cloud_dist_km=-999.0,
                                    nearest_cloud_lat=-999.0,
                                    nearest_cloud_lon=-999.0,
                                    cloud_classification='NoMODIS',
                                    weighted_cloud_dist_km=-999.0,
                                )
                                for fp in granule_footprints.values()
                            ]
                            spatial_processor._save_cached_result(
                                cache_dir / "phase4_results.pkl", placeholder_results
                            )
                            logger.warning(
                                f"    ⚠ No matched MODIS files for {granule_id} (post-2022): "
                                f"wrote {len(placeholder_results)} placeholder -999 result(s)"
                            )
                            cached_granules.add(granule_id)
                            del placeholder_results, granule_footprints
                            gc.collect()
                            continue
                        print(f"No matched MODIS files for granule {granule_id}, not using this granule for processing.")
                        continue  # Skip to next granule since we have no cloud data to combine
                    
                    # Combine and cache (this function handles saving myd35_*.pkl files internally)
                    result = spatial_processor.combine_OCO_fp_cloud_masks_by_granule(
                        granule_cloud_masks,
                        granule_footprints,
                        full_granule_id,
                        cache_dir
                    )
                    
                    if result:
                        pixel_count = len(result.get('lon', [])) if result.get('lon') is not None else 0
                        logger.info(f"    ✓ Successfully processed: {pixel_count} cloud pixels")
                        cached_granules.add(granule_id)
                    else:
                        logger.warning(f"    ⚠ Processing returned no result")
                    
                    # Free memory: delete granule-specific data
                    del granule_cloud_masks, granule_footprints, result
                    gc.collect()  # Force garbage collection after each granule
                        
            except Exception as e:
                logger.error(f"    ✗ Error during Step 3 processing: {e}")
                traceback.print_exc()
            
            logger.info(f"\n✓ Step 3 Processing Complete")
            logger.info(f"  Total granules cached: {len(cached_granules)}/{len(oco2_granules)}")
        
        processing_info = {
            'cached': True,
            'granules': len(cached_granules),
            'cache_location': str(processing_day_dir),
        }
        
        if len(cached_granules) == len(oco2_granules):
            logger.info(f"✓ Step 3 Complete: All {len(cached_granules)} granule(s) cached")
            return processing_info, True
        else:
            logger.warning(f"⚠ Step 3 Incomplete: {len(cached_granules)}/{len(oco2_granules)} granules cached")
            return processing_info, False
        
    except Exception as e:
        logger.error(f"✗ Step 3 Failed: {e}")
        traceback.print_exc()
        return {}, False


def run_phase_4(target_date: datetime, data_dir: Path, max_distance: float = 50.0,
                band_width: float = CLOUD_DIST_BAND_WIDTH_DEG,
                band_overlap: float = CLOUD_DIST_BAND_OVERLAP_DEG,
                visualize: bool = False, viz_dir: Optional[Path] = None,
                force_recompute: bool = False) -> Tuple[List, bool]:
    """
    Step 4: High-Performance Computational Geometry
    
    Args:
        target_date: Target date for processing
        data_dir: Data storage directory
        max_distance: Maximum cloud distance in km
        band_width: Latitude band width in degrees
        band_overlap: Latitude band overlap in degrees
        visualize: Create visualizations
        viz_dir: Visualization output directory
        force_recompute: Force recalculation of distances (ignore Step 4 cache)
    
    Returns:
        Tuple of (results_list, success_flag)
    """
    print_step_header(4, "High-Performance Computational Geometry")
    
    try:
        data_dir = Path(data_dir)
        spatial_processor = SpatialProcessor(data_dir=str(data_dir))
        geometry_processor = GeometryProcessor(data_dir=str(data_dir))
        
        # Load cached data from Step 3
        logger.info("\n[Step 1] Loading Step 3 cache...")
        processing_day_dir = data_dir / "processing" / str(target_date.year) / f"{target_date.timetuple().tm_yday:03d}"
        
        if not processing_day_dir.exists():
            logger.error("Step 3 cache directory not found")
            return [], False
        
        granule_dirs = [d for d in processing_day_dir.glob("*") if d.is_dir()]
        logger.info(f"✓ Found {len(granule_dirs)} granule cache director(ies)")
        
        # Process each granule
        logger.info("\n[Step 2] Per-Granule Processing and Distance Calculation...")
        results = []
        cache_hits = 0
        cache_misses = 0
        
        # Track cloud statistics for visualization (counts only, not pixels)
        total_cloudy_pixels = 0
        total_uncertain_pixels = 0
        
        if visualize and viz_dir:
            viz_dir = Path(viz_dir)
            viz_dir.mkdir(parents=True, exist_ok=True)
        
        for granule_dir in sorted(granule_dirs):
            granule_id = granule_dir.name
            logger.info(f"\n  Processing granule: {granule_id}")
            
            # Load combined cache
            combined_cache_files = list(granule_dir.glob("granule_combined_*.pkl"))

            if not combined_cache_files:
                # No collocated-MODIS cloud cache. Post-2022 granules with no MODIS
                # overlap get a -999 placeholder written straight to phase4_results.pkl
                # by Phase 3; surface those so the soundings still reach Phase 5.
                phase4_cache_path = granule_dir / "phase4_results.pkl"
                if phase4_cache_path.exists():
                    placeholder_results = spatial_processor._load_cached_result(phase4_cache_path)
                    if placeholder_results:
                        results.extend(placeholder_results)
                        cache_hits += 1
                        logger.info(f"    📂 Loaded {len(placeholder_results):,} placeholder -999 result(s) (no MODIS overlap)")
                        continue
                logger.warning(f"    No combined cache found for {granule_id}")
                continue
            
            try:
                combined_data = spatial_processor._load_cached_result(combined_cache_files[0])
                if combined_data is None:
                    logger.warning(f"    Failed to load combined cache")
                    continue
            except Exception as e:
                logger.warning(f"    Error loading cache: {e}")
                continue
            
            # Extract data
            cloud_lon = combined_data.get('lon')
            cloud_lat = combined_data.get('lat')
            cloud_flag = combined_data.get('cloud_flag')
            
            fp_lon = combined_data.get('oco2_fp_lons')
            fp_lat = combined_data.get('oco2_fp_lats')
            fp_ids = combined_data.get('oco2_fp_sounding_ids')
            fp_modes = combined_data.get('oco2_fp_viewing_modes')

            if cloud_lon is None or cloud_lat is None:
                logger.warning(f"    No cloud data in cache")
                continue
            
            logger.info(f"    Footprints: {len(fp_ids):,} | Clouds: {len(cloud_lon):,}")
            
            # Skip if no clouds (can't calculate distances)
            if len(cloud_lon) == 0:
                logger.warning(f"    ⊘ Skipping - no cloud data for distance calculation")
                continue
            
            # Count cloudy and uncertain pixels for statistics
            cloudy_count = np.sum(cloud_flag == 1)
            uncertain_count = np.sum(cloud_flag == 0)
            total_cloudy_pixels += cloudy_count
            total_uncertain_pixels += uncertain_count
            
            # Check for Step 4 cache (unless force_recompute is True)
            phase4_cache_path = granule_dir / "phase4_results.pkl"
            cached_results = None
            if phase4_cache_path.exists() and not force_recompute:
                try:
                    cached_results = spatial_processor._load_cached_result(phase4_cache_path)
                except Exception:
                    cached_results = None
            
            if cached_results is not None:
                results.extend(cached_results)
                cache_hits += 1
                logger.info(f"    📂 Loaded from cache: {len(cached_results):,} results")
                continue
            
            cache_misses += 1
            
            # Calculate distances
            logger.info(f"    Calculating distances (banded geometry)...")
            granule_results = geometry_processor.calculate_nearest_cloud_distances_banded(
                footprints_by_granule=(fp_lon, fp_lat, fp_ids, fp_modes),
                cloud_lons=cloud_lon,
                cloud_lats=cloud_lat,
                cloud_flags=cloud_flag,
                band_width_deg=band_width,
                band_overlap_deg=band_overlap,
                max_distance_km=max_distance,
                oco2_granule_id=granule_id,
            )
            
            # Cache results
            spatial_processor._save_cached_result(phase4_cache_path, granule_results)
            
            results.extend(granule_results)
            logger.info(f"    ✓ Processed {len(granule_results):,} soundings")

            
            # Create per-granule latitude-band visualizations
            if visualize and viz_dir:
                logger.info(f"    Creating per-granule latitude-band visualizations...")
                vis_dir_date_dir = viz_dir / f"{target_date.date()}"
                vis_dir_date_dir.mkdir(parents=True, exist_ok=True)
                granule_vis_dir = vis_dir_date_dir / f"granule_{granule_id}"
                granule_vis_dir.mkdir(parents=True, exist_ok=True)
                
                # try:
                if 1:
                    # Use granule-specific cloud and result data
                    granule_vis_paths = geometry_processor.visualize_latband_distance(
                        results=granule_results,
                        cloud_lons=cloud_lon,
                        cloud_lats=cloud_lat,
                        cloud_flags=cloud_flag,
                        output_dir=granule_vis_dir,
                        max_distance=max_distance,
                        lat_band_size=5.0,
                        max_clouds_per_band=50000,
                        dpi=100
                    )
                    
                    if granule_vis_paths:
                        logger.info(f"    ✓ Created {len(granule_vis_paths)} latitude-band plot(s) for this granule")
                # except Exception as e:
                #     logger.warning(f"    ⚠ Visualization failed for {granule_id}: {e}")
                #     logger.warning(f"    → Continuing with next granule...")
                    
            # Free memory: delete large arrays for this granule
            del cloud_lon, cloud_lat, cloud_flag
            del fp_lon, fp_lat, fp_ids, fp_modes
            del combined_data, granule_results
            gc.collect()  # Force garbage collection
        
        logger.info(f"\n✓ Total results: {len(results):,} soundings")
        logger.info(f"  Cache: {cache_hits} hit(s), {cache_misses} miss(es)")
        
        if not results:
            logger.error("No collocation results generated")
            return [], False
        
        logger.info(f"  Cloud pixel statistics:")
        logger.info(f"    Cloudy: {total_cloudy_pixels:,}")
        logger.info(f"    Uncertain: {total_uncertain_pixels:,}")
        
        # Always create KD-Tree spatial range visualization
        kdtree_output_dir = Path(viz_dir) if viz_dir else Path("./visualizations_combined")
        kdtree_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n[Step 3] Creating KD-Tree spatial range visualization...")
        try:
            geometry_processor.visualize_kdtree_spatial_range(
                results=results,
                num_cloudy=total_cloudy_pixels,
                num_uncertain=total_uncertain_pixels,
                output_path=kdtree_output_dir / f"kdtree_range_{target_date.date()}.png",
                max_distance=max_distance,
                dpi=200
            )
            logger.info("  ✓ KD-Tree spatial range visualization saved")
        except Exception as e:
            logger.warning(f"  ⚠ KD-Tree visualization failed: {e}")
        
        logger.info(f"\n✓ Step 4 Complete: {len(results):,} soundings with distances")
        return results, True
        
    except Exception as e:
        logger.error(f"✗ Step 4 Failed: {e}")
        traceback.print_exc()
        return [], False


def run_phase_5(results: List, target_date: datetime, output_dir: Path,
                data_dir: Path, max_distance: float = 50.0) -> bool:
    """
    Step 5: Synthesis and Data Export

    Args:
        results: Results from Step 4
        target_date: Target date
        output_dir: Output directory for results
        data_dir: Data storage directory
        max_distance: Maximum cloud distance in km

    Returns:
        Success flag
    """
    print_step_header(5, "Synthesis and Data Export")

    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        geometry_processor = GeometryProcessor(data_dir=str(data_dir))
        
        # Determine output paths
        results_file = output_dir / f"results_{target_date.date()}.h5"
        stats_file = output_dir / f"results_{target_date.date()}.json"
        
        logger.info(f"\n[Step 1] Exporting results to HDF5...")
        
        # Count unique MODIS granules
        unique_modis_granules = set()
        for result in results:
            if hasattr(result, 'modis_granule_id') and result.modis_granule_id != "N/A":
                unique_modis_granules.add(result.modis_granule_id)
        
        metadata = {
            'date': str(target_date.date()),
            'max_distance_km': max_distance,
            'num_soundings': len(results),
            'num_modis_granules': len(unique_modis_granules),
        }
        
        geometry_processor.export_results_hdf5(results, results_file, metadata)
        logger.info(f"✓ HDF5: {results_file}")
        
        logger.info(f"\n[Step 2] Computing and exporting statistics...")
        stats = geometry_processor.get_statistics(results)
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✓ JSON: {stats_file}")
        
        # Log statistics
        logger.info(f"\n📊 Summary Statistics:")
        logger.info(f"   Total soundings: {stats['total_soundings']}")
        logger.info(f"   Collocated:      {stats['collocated_soundings']}")
        logger.info(f"   No collocation:  {stats['no_collocation_soundings']} (no daytime MODIS overlap, flagged -999)")
        logger.info(f"   Distance (km) [collocated only]:")
        logger.info(f"     Min:    {stats['distance_km']['min']:.2f}")
        logger.info(f"     Max:    {stats['distance_km']['max']:.2f}")
        logger.info(f"     Mean:   {stats['distance_km']['mean']:.2f}")
        logger.info(f"     Median: {stats['distance_km']['median']:.2f}")
        logger.info(f"     Std:    {stats['distance_km']['std']:.2f}")
        logger.info(f"   Distance distribution [collocated only]:")
        logger.info(f"     0-2 km:    {stats['distance_distribution']['0-2_km']}")
        logger.info(f"     2-5 km:    {stats['distance_distribution']['2-5_km']}")
        logger.info(f"     5-10 km:   {stats['distance_distribution']['5-10_km']}")
        logger.info(f"     10-20 km:  {stats['distance_distribution']['10-20_km']}")
        logger.info(f"     20+ km:    {stats['distance_distribution']['20+_km']}")
        logger.info(f"     No collocation: {stats['distance_distribution']['no_collocation']}")
        
        logger.info(f"\n✓ Step 5 Complete: Results exported to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Step 5 Failed: {e}")
        traceback.print_exc()
        return False



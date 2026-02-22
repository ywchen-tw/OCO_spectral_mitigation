#!/usr/bin/env python3
"""
Combined Demo: Complete OCO-2/MODIS Collocation Pipeline
=========================================================

This unified script orchestrates all 5 steps of the OCO-2/MODIS footprint
analysis workflow to produce final results in a single command.

Step Workflow:
    Step 1: Metadata Acquisition (temporal/orbital boundaries)
    Step 2: Targeted Data Ingestion (download OCO-2 & MODIS)
    Step 3: Spatial and Bitmask Processing (extract footprints & clouds)
    Step 4: High-Performance Computational Geometry (calculate distances)
    Step 5: Synthesis and Data Export (consolidate results)

Cache Structure:
  data/processing/{year}/{doy:03d}/{orbit_id}/
    ├── footprints.pkl              # Step 3: OCO-2 footprints
    ├── clouds.pkl                  # Step 3: MODIS cloud pixels (myd35*.pkl data)
    ├── granule_combined_*.pkl      # Step 3: Combined OCO-2 + MODIS for this orbit
    └── phase4_results.pkl          # Step 4: Distance calculation results
  
  All cache files for a specific orbit are stored in its own subdirectory
  (e.g., 22845a/, 22846a/) for clean separation and independent processing.

Usage:
    python workspace/demo_combined.py --date 2018-10-18
    python workspace/demo_combined.py --date 2018-10-18 --visualize
    python workspace/demo_combined.py --date 2018-10-18 --skip-phase 2

"""

import sys
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
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase_01_metadata import OCO2MetadataRetriever
from phase_02_ingestion import DataIngestionManager, DownloadedFile
from phase_03_processing import SpatialProcessor
from phase_04_geometry import GeometryProcessor
from config import Config
from utils import setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def print_banner(text: str):
    """Print a formatted banner."""
    width = 70
    logger.info("=" * width)
    logger.info(text.center(width))
    logger.info("=" * width)


def print_step_header(step_num: int, title: str):
    """Print step header."""
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP {step_num}: {title}".ljust(70))
    logger.info(f"{'='*70}")


def validate_date(date_str: str) -> datetime:
    """Validate and parse date string."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def parse_orbit_arg(orbit_str: str) -> Tuple[int, Optional[str]]:
    """Parse --orbit argument into (orbit_number, granule_suffix).

    Accepts plain orbit numbers ('31017') or orbit+suffix ('31017b').
    Returns the integer orbit number and the optional lowercase suffix letter.
    """
    m = re.match(r'^(\d+)([a-z]?)$', orbit_str)
    if not m:
        raise ValueError(
            f"Invalid --orbit value '{orbit_str}'. "
            "Use a plain orbit number (e.g. 31017) or add a suffix letter (e.g. 31017b)."
        )
    return int(m.group(1)), m.group(2) or None


def get_storage_dir() -> Path:
    """Determine storage directory based on platform."""
    if platform.system() == "Darwin":
        logger.info("Detected macOS - using local data directory")
        return Path(Config.get_data_path('local'))
    elif platform.system() == "Linux":
        logger.info("Detected Linux - using CURC storage directory")
        return Path(Config.get_data_path('curc'))
    else:
        logger.warning(f"Unknown platform: {platform.system()}. Using default.")
        return Path(Config.get_data_path('default'))


def cleanup_modis_data(target_date: datetime, data_dir: Path) -> bool:
    """Delete MODIS data files for a specific date to save disk space.
    
    Args:
        target_date: Target date for cleanup
        data_dir: Data storage directory
    
    Returns:
        Success flag
    """
    logger.info("\n" + "="*70)
    logger.info("CLEANUP: Deleting MODIS Data")
    logger.info("="*70)
    
    try:
        data_dir = Path(data_dir)
        doy = target_date.timetuple().tm_yday
        year = target_date.year
        
        # MODIS directories to delete
        myd35_dir = data_dir / "MODIS" / "MYD35_L2" / str(year) / f"{doy:03d}"
        myd03_dir = data_dir / "MODIS" / "MYD03" / str(year) / f"{doy:03d}"
        
        deleted_size = 0
        deleted_files = 0
        
        for modis_dir in [myd35_dir, myd03_dir]:
            if modis_dir.exists():
                # Calculate size before deletion
                for file in modis_dir.glob("*.hdf"):
                    deleted_size += file.stat().st_size
                    deleted_files += 1
                
                # Delete directory
                shutil.rmtree(modis_dir)
                logger.info(f"✓ Deleted: {modis_dir}")
            else:
                logger.info(f"⊘ Not found: {modis_dir}")
        
        if deleted_files > 0:
            deleted_size_mb = deleted_size / (1024 * 1024)
            logger.info(f"\n✓ Cleanup Complete: Deleted {deleted_files} file(s), freed {deleted_size_mb:.1f} MB")
        else:
            logger.info("\n⊘ No MODIS files found to delete")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Cleanup Failed: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Step Execution Functions
# ============================================================================

def run_phase_1(target_date: datetime, orbit: Optional[int] = None, 
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
        start_time, end_time = retriever.extract_temporal_window(granules, orbit, mode)
        logger.info(f"✓ Temporal window: {start_time} to {end_time}")
        logger.info(f"  Duration: {(end_time - start_time).total_seconds() / 60:.1f} minutes")
        
        metadata = {
            'target_date': target_date,
            'granules': granules,
            'start_time': start_time,
            'end_time': end_time,
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
                orbit: Optional[int] = None, granule_suffix: Optional[str] = None,
                mode: Optional[str] = None) -> Tuple[Dict, bool]:
    """
    Step 2: Targeted Data Ingestion

    Args:
        target_date: Target date for data download
        data_dir: Data storage directory
        dry_run: Only check file existence without downloading
        force_download: Force re-download even if status file exists
        limit_granules: Limit to first N granules (for testing)
        orbit: Optional orbit number filter (passed through from --orbit)
        granule_suffix: Optional granule suffix letter to narrow orbit filter (e.g. 'b')
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
            suffix_str = granule_suffix or ''
            logger.info(f"  Orbit filter: {orbit}{suffix_str}")
        if mode:
            logger.info(f"  Mode filter: {mode}")

        result = ingestion_manager.download_all_for_date(
            target_date=target_date,
            orbit_filter=orbit,
            granule_suffix=granule_suffix,
            mode_filter=mode,
            include_modis=True,
            limit_granules=limit_granules,
            skip_existing=(not force_download)
        )
        
        oco2_files = result.get('oco2_files', [])
        modis_files = result.get('modis_files', [])
        
        logger.info(f"\n✓ Downloaded/loaded {len(oco2_files)} OCO-2 file(s)")
        logger.info(f"✓ Downloaded/loaded {len(modis_files)} MODIS file(s)")
        
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


def run_phase_3(target_date: datetime, data_dir: Path) -> Tuple[Dict, bool]:
    """
    Step 3: Spatial and Bitmask Processing

    Processes OCO-2 footprints and MODIS cloud data. Automatically processes any
    granules that don't have cached results.

    Args:
        target_date: Target date for processing
        data_dir: Data storage directory

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
                if subdir.is_dir():
                    oco2_granules.add(subdir.name)
        
        if oco2_granules:
            logger.info(f"✓ Found {len(oco2_granules)} OCO-2 granule(s) in data directory")
        else:
            logger.warning("⚠ No OCO-2 granules found in data directory")
            return {}, False
        
        # Check if Step 3 cache already exists
        logger.info("\n[Step 2] Checking for Step 3 cache...")
        processing_day_dir = data_dir / "processing" / str(year) / f"{doy:03d}"
        
        cached_granules = set()
        if processing_day_dir.exists():
            cache_dirs = [d for d in processing_day_dir.glob("*") if d.is_dir()]
            for cache_dir in cache_dirs:
                combined_files = list(cache_dir.glob("granule_combined_*.pkl"))
                if combined_files:
                    cached_granules.add(cache_dir.name)
                    logger.info(f"  ✓ {cache_dir.name}: {len(combined_files)} combined cache file(s)")
        
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
                for granule_id in sorted(missing_granules):
                    granule_dir = oco2_l1b_dir / granule_id
                    if granule_dir.exists():
                        for file_path in granule_dir.glob("*"):
                            if file_path.is_file():
                                product_type = ""
                                if "L1bSc" in file_path.name:
                                    product_type = "L1B_Science"
                                elif "L2Lite" in file_path.name:
                                    product_type = "L2_Lite"
                                
                                if product_type:
                                    file_size = file_path.stat().st_size / (1024 * 1024)
                                    oco2_files.append(DownloadedFile(
                                        filepath=file_path,
                                        product_type=product_type,
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
                            granule_id=modis_file.name,
                            file_size_mb=file_size,
                            download_time_seconds=0.0
                        ))
                
                logger.info(f"  Loaded {len(oco2_files)} OCO-2 file(s) and {len(modis_files)} MODIS file(s)")
                
                # Extract footprints for all viewing modes so GL, ND, and TG granules
                # are all available when the per-granule loop looks them up.
                logger.info("  Extracting OCO-2 footprints (all viewing modes)...")
                oco2_footprints = spatial_processor.extract_oco2_footprints(oco2_files, viewing_mode=None)
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
                
                # Match MODIS files to OCO-2 granules based on temporal proximity
                logger.info("  Matching MODIS files to OCO-2 granules...")
                modis_to_oco2_mapping = {}
                
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
                    doy_m = int(match.group(2))
                    hhmm = match.group(3)
                    hour = int(hhmm[:2])
                    minute = int(hhmm[2:4])
                    modis_date = datetime(year_m, 1, 1) + timedelta(days=doy_m - 1)
                    modis_time = modis_date.replace(hour=hour, minute=minute)
                    
                    # Assign this MODIS file to ALL OCO-2 granules whose ±20-minute
                    # window it falls in.  A single 5-minute MODIS swath can overlap
                    # the time windows of two adjacent (or mode-separated) OCO-2 orbits;
                    # picking only the "closest" would silently drop cloud data for the other.
                    buffer_seconds = 20 * 60  # ±20 minutes (matches Phase 3 spec)
                    for oco2_granule_id, (start_time, end_time) in granule_time_ranges.items():
                        window_start = start_time - timedelta(seconds=buffer_seconds)
                        window_end = end_time + timedelta(seconds=buffer_seconds)
                        if window_start <= modis_time <= window_end:
                            if oco2_granule_id not in modis_to_oco2_mapping:
                                modis_to_oco2_mapping[oco2_granule_id] = []
                            modis_to_oco2_mapping[oco2_granule_id].append(modis_file)
                
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
                            matched_modis_files,
                            myd03_files=matched_myd03 if matched_myd03 else None,
                            oco2_orbit_id=granule_id  # Pass orbit_id so files save to correct folder
                        )
                        
                        # Free memory: delete matched file references
                        del matched_myd03
                    else:
                        granule_cloud_masks = {}
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
                band_width: float = 10.0, band_overlap: float = 1.0,
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
                granule_vis_dir = viz_dir / f"granule_{granule_id}"
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
        logger.info(f"   Distance (km):")
        logger.info(f"     Min:    {stats['distance_km']['min']:.2f}")
        logger.info(f"     Max:    {stats['distance_km']['max']:.2f}")
        logger.info(f"     Mean:   {stats['distance_km']['mean']:.2f}")
        logger.info(f"     Median: {stats['distance_km']['median']:.2f}")
        logger.info(f"     Std:    {stats['distance_km']['std']:.2f}")
        logger.info(f"   Distance distribution:")
        logger.info(f"     0-2 km:    {stats['distance_distribution']['0-2_km']}")
        logger.info(f"     2-5 km:    {stats['distance_distribution']['2-5_km']}")
        logger.info(f"     5-10 km:   {stats['distance_distribution']['5-10_km']}")
        logger.info(f"     10-20 km:  {stats['distance_distribution']['10-20_km']}")
        logger.info(f"     20+ km:    {stats['distance_distribution']['20+_km']}")
        
        logger.info(f"\n✓ Step 5 Complete: Results exported to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Step 5 Failed: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Combined Demo: Complete OCO-2/MODIS Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all steps for a specific date
  python demo_combined.py --date 2018-10-18
  
  # Run with visualizations
  python demo_combined.py --date 2018-10-18 --visualize
  
  # Force download all files (ignore existing download status)
  python demo_combined.py --date 2018-10-18 --force-download
  
  # Force recompute Phase 4 distances
  python demo_combined.py --date 2018-10-18 --force-recompute
  
  # Download only first 2 granules for testing
  python demo_combined.py --date 2018-10-18 --limit-granules 2
  
  # Delete MODIS data after completion (save disk space)
  python demo_combined.py --date 2018-10-18 --delete-modis
  
    # Skip Step 2 (use existing data)
  python demo_combined.py --date 2018-10-18 --skip-phase 2
  
  # Custom output directory
  python demo_combined.py --date 2018-10-18 --output-dir ./my_results
        """
    )
    
    # Required arguments
    parser.add_argument('--date', type=str, required=True,
                       help='Target date (YYYY-MM-DD, e.g., 2018-10-18)')
    
    # Optional algorithm parameters
    parser.add_argument('--max-distance', type=float, default=25.0,
                       help='Maximum cloud distance in km (default: 25.0)')
    parser.add_argument('--band-width', type=float, default=2.5,
                       help='Latitude band width in degrees (default: 2.5)')
    parser.add_argument('--band-overlap', type=float, default=0.5,
                       help='Latitude band overlap in degrees (default: 1.0)')
    
    # Optional data directory
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data storage directory (default: ./data)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Results output directory (default: data/results/)')
    
    # Optional features
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--viz-dir', type=str, default=None,
                       help='Visualization output directory')
    
    # Workflow control
    parser.add_argument('--skip-phase', type=int, action='append',
                       help='Skip specific step(s) (can use multiple times)')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation (ignore Phase 4 cache)')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download all files (ignore Phase 2 download status)')
    parser.add_argument('--delete-modis', action='store_true',
                       help='Delete MODIS data after successful completion (to save disk space)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Check file existence without downloading')
    
    # Filtering
    parser.add_argument('--orbit', type=str,
                       help='Specific orbit filter: orbit number (e.g. 31017) or orbit+suffix (e.g. 31017b)')
    parser.add_argument('--mode', type=str, choices=['GL', 'ND', 'TG'],
                       help='Viewing mode filter (GL=Glint, ND=Nadir, TG=Target)')
    parser.add_argument('--limit-granules', type=int,
                       help='Limit to first N granules (for testing, Phase 2)')
    
    args = parser.parse_args()
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    skip_phases = set(args.skip_phase) if args.skip_phase else set()

    try:
        target_date = validate_date(args.date)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Parse --orbit into integer orbit number + optional granule suffix (e.g. 'b')
    orbit_num: Optional[int] = None
    granule_suffix: Optional[str] = None
    if args.orbit:
        try:
            orbit_num, granule_suffix = parse_orbit_arg(args.orbit)
        except ValueError as e:
            logger.error(str(e))
            return 1
    
    storage_dir = get_storage_dir()
    data_dir = storage_dir / "data" if args.data_dir == "./data" else Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = storage_dir / "results"

    if args.viz_dir:
        viz_dir = Path(args.viz_dir) 
    else:
        viz_dir = storage_dir / "visualizations_combined"
    
    print_banner("OCO-2/MODIS Footprint Analysis - Combined Pipeline")
    logger.info(f"Date: {target_date.date()}")
    logger.info(f"Data Directory: {data_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Visualizations: {'Enabled' if args.visualize else 'Disabled'}")
    if args.dry_run:
        logger.info("Mode: DRY RUN (file existence check only)")
    if args.force_download:
        logger.info("Mode: FORCE DOWNLOAD (ignore existing download status)")
    if args.force_recompute:
        logger.info("Mode: FORCE RECOMPUTE (recalculate Phase 4 distances)")
    logger.info("")
    
    # ========================================================================
    # Phase 1: Metadata
    # ========================================================================
    if 1 not in skip_phases:
        metadata, success = run_phase_1(target_date, orbit_num, args.mode)
        if not success:
            if not metadata.get('granules'):
                logger.error("\nPipeline cannot proceed: No OCO-2 data available for this date.")
                logger.error(f"Date requested: {target_date.date()}")
                logger.error("Please select a different date with available OCO-2 data.")
            else:
                logger.error("Pipeline aborted at Step 1")
            return 1
    else:
        logger.info("[STEP 1] SKIPPED")
        metadata = {}
    
    # ========================================================================
    # Phase 2: Ingestion
    # ========================================================================
    if 2 not in skip_phases:
        file_info, success = run_phase_2(
            target_date, data_dir,
            dry_run=args.dry_run,
            force_download=args.force_download,
            limit_granules=args.limit_granules,
            orbit=orbit_num,
            granule_suffix=granule_suffix,
            mode=args.mode
        )
        if not success:
            logger.error("Pipeline aborted at Step 2")
            return 1
    else:
        logger.info("[STEP 2] SKIPPED - Using existing data")
        file_info = {}
    
    # ========================================================================
    # Phase 3: Spatial Processing
    # ========================================================================
    if 3 not in skip_phases:
        processing_info, success = run_phase_3(target_date, data_dir)
        if not success:
            logger.error("Pipeline aborted at Step 3")
            return 1
    else:
        logger.info("[STEP 3] SKIPPED - Using cached data")
        processing_info = {}
    
    # ========================================================================
    # Phase 4: Geometry
    # ========================================================================
    if 4 not in skip_phases:
        results, success = run_phase_4(
            target_date, data_dir,
            max_distance=args.max_distance,
            band_width=args.band_width,
            band_overlap=args.band_overlap,
            visualize=args.visualize,
            viz_dir=viz_dir,
            force_recompute=args.force_recompute
        )
        if not success:
            logger.error("Pipeline aborted at Step 4")
            return 1
    else:
        logger.info("[STEP 4] SKIPPED")
        results = []
        return 1
    
    # ========================================================================
    # Phase 5: Synthesis
    # ========================================================================
    if 5 not in skip_phases and results:
        success = run_phase_5(
            results, target_date, output_dir,
            data_dir=data_dir,
            max_distance=args.max_distance
        )
        if not success:
            logger.error("Pipeline aborted at Step 5")
            return 1
    elif not results:
        logger.warning("[STEP 5] SKIPPED - No results from Step 4")
    else:
        logger.info("[STEP 5] SKIPPED")
    
    # ========================================================================
    # Cleanup: Delete MODIS data if requested
    # ========================================================================
    if args.delete_modis:
        cleanup_success = cleanup_modis_data(target_date, data_dir)
        if not cleanup_success:
            logger.warning("⚠ MODIS cleanup failed, but pipeline completed successfully")
    
    # ========================================================================
    # Complete
    # ========================================================================
    print_banner("Pipeline Complete!")
    logger.info(f"✅ All phases executed successfully")
    logger.info(f"Results saved to: {output_dir}")
    if args.delete_modis:
        logger.info("🗑️  MODIS data deleted (use --delete-modis flag)")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

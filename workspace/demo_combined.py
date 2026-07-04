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



# Split along the orchestration/per-phase seam (2026-07, review §7.4):
# helpers live in demo_utils, the five phase runners in pipeline_phases.
# All names are re-imported here so external usage is unchanged.
from demo_utils import (  # noqa: F401
    LITE_VERSION_RANK, cleanup_modis_data, delete_lite_files_before, get_storage_dir,
    infer_lite_version, invalidate_lite_downstream_cache,
    lite_version_is_before, parse_orbit_arg, print_banner,
    print_step_header, select_local_lite_file, validate_date)
from pipeline_phases import (  # noqa: F401
    run_phase_1, run_phase_2, run_phase_3, run_phase_4, run_phase_5)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    parser.add_argument('--max-distance', type=float, default=50.0,
                       help='Maximum cloud distance in km (default: 50.0)')
    parser.add_argument('--band-width', type=float, default=CLOUD_DIST_BAND_WIDTH_DEG,
                       help=f'Latitude band width in degrees (default: {CLOUD_DIST_BAND_WIDTH_DEG})')
    parser.add_argument('--band-overlap', type=float, default=CLOUD_DIST_BAND_OVERLAP_DEG,
                       help=f'Latitude band overlap in degrees (default: {CLOUD_DIST_BAND_OVERLAP_DEG})')
    
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
    parser.add_argument('--force-recompute-if-lite-before',
                       choices=sorted(LITE_VERSION_RANK, key=LITE_VERSION_RANK.get),
                       default=None,
                       metavar='VERSION',
                       help='If the selected local L2 Lite file is older than VERSION, '
                            'invalidate Lite-derived processing caches and force Phase 4 '
                            'recompute. Example: --force-recompute-if-lite-before 11.2r')
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

    # Phase 3.5: GEE embedding
    parser.add_argument('--gcp-project', type=str, nargs='?', const='', default=None,
                       help='Enable GEE embedding (Phase 3.5). '
                            'Provide a GCP project ID (--gcp-project my-id) or supply '
                            'the flag alone to read from the GEE_PROJECT env var. '
                            'Omitting the flag entirely skips Phase 3.5.')
    parser.add_argument('--embedding-batch', action='store_true',
                       help='Use GEE batch export to Drive instead of synchronous getInfo '
                            '(needed for large runs; requires --gcp-project)')
    parser.add_argument('--embedding-limit-orbits', type=int, default=None, metavar='N',
                       help='Test mode: restrict Phase 3.5 to the first N granules. '
                            'Output goes to embedding_stats_{date}_test.parquet.')

    args = parser.parse_args()

    # Phase 3.5: resolve GCP project ID
    # --gcp-project not given  → None  → Phase 3.5 skipped
    # --gcp-project            → ''    → read GEE_PROJECT env var
    # --gcp-project my-id      → str   → use that ID directly
    if args.gcp_project is None:
        gcp_project = None
    elif args.gcp_project == '':
        gcp_project = os.environ.get('GEE_PROJECT')
        if not gcp_project:
            logger.error("--gcp-project flag given but GEE_PROJECT env var is not set")
            return 1
    else:
        gcp_project = args.gcp_project

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
            orbit_str = f"{orbit_num}{granule_suffix if granule_suffix else ''}"
        except ValueError as e:
            logger.error(str(e))
            return 1
    else:
        orbit_str = None
    
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
    if args.force_recompute_if_lite_before:
        logger.info(
            "Mode: FORCE RECOMPUTE IF LITE BEFORE %s",
            args.force_recompute_if_lite_before,
        )
    logger.info("")

    lite_before_path: Optional[Path] = None
    lite_before_version: Optional[str] = None
    if args.force_recompute_if_lite_before:
        lite_before_path, lite_before_version = select_local_lite_file(data_dir, target_date)
        if lite_before_path:
            logger.info(
                "Existing selected Lite file before ingestion: %s (%s)",
                lite_before_path,
                lite_before_version,
            )
        else:
            logger.info("No existing local Lite file found before ingestion")
    
    # ========================================================================
    # Phase 1: Metadata
    # ========================================================================
    # if 1 not in skip_phases:
    #     metadata, success = run_phase_1(target_date, orbit_str, args.mode)
    #     if not success:
    #         if not metadata.get('granules'):
    #             logger.error("\nPipeline cannot proceed: No OCO-2 data available for this date.")
    #             logger.error(f"Date requested: {target_date.date()}")
    #             logger.error("Please select a different date with available OCO-2 data.")
    #         else:
    #             logger.error("Pipeline aborted at Step 1")
    #         return 1
    # else:
    #     logger.info("[STEP 1] SKIPPED")
    #     metadata = {}
    
    # ========================================================================
    # Phase 2: Ingestion
    # ========================================================================
    if 2 not in skip_phases:
        file_info, success = run_phase_2(
            target_date, data_dir,
            dry_run=args.dry_run,
            force_download=args.force_download,
            limit_granules=args.limit_granules,
            orbit=orbit_str,
            granule_suffix=granule_suffix,
            mode=args.mode
        )
        if not success:
            logger.error("Pipeline aborted at Step 2")
            return 1
    else:
        logger.info("[STEP 2] SKIPPED - Using existing data")
        file_info = {}

    if args.force_recompute_if_lite_before:
        lite_after_path, lite_after_version = select_local_lite_file(data_dir, target_date)
        if lite_after_path:
            logger.info(
                "Selected Lite file after ingestion: %s (%s)",
                lite_after_path,
                lite_after_version,
            )
        else:
            logger.warning(
                "No local Lite file found after ingestion; downstream phases may fail"
            )

        deleted_lite = delete_lite_files_before(
            data_dir,
            target_date,
            args.force_recompute_if_lite_before,
        )
        if deleted_lite:
            logger.info(
                "Deleted %d local Lite file(s) older than %s",
                deleted_lite,
                args.force_recompute_if_lite_before,
            )

        stale_before = (
            lite_before_path is not None
            and lite_version_is_before(lite_before_version, args.force_recompute_if_lite_before)
        )
        stale_after = (
            lite_after_path is not None
            and lite_version_is_before(lite_after_version, args.force_recompute_if_lite_before)
        )
        lite_changed = (
            lite_before_path is not None
            and lite_after_path is not None
            and lite_before_path != lite_after_path
        )

        if stale_before or stale_after or lite_changed:
            reasons = []
            if stale_before:
                reasons.append(f"previous Lite version was {lite_before_version}")
            if stale_after:
                reasons.append(f"selected Lite version is {lite_after_version}")
            if lite_changed:
                reasons.append("selected Lite file changed during ingestion")
            logger.warning(
                "Lite-derived caches are stale relative to minimum %s (%s)",
                args.force_recompute_if_lite_before,
                "; ".join(reasons),
            )
            removed = invalidate_lite_downstream_cache(data_dir, target_date)
            logger.info("Removed %d Lite-derived cache file(s)", removed)
            args.force_recompute = True
            if 3 in skip_phases:
                logger.warning(
                    "--skip-phase 3 was requested, but Lite-derived Phase 3 caches "
                    "were invalidated. Remove --skip-phase 3 to rebuild them."
                )
    
    # ========================================================================
    # Phase 3: Spatial Processing
    # ========================================================================
    if 3 not in skip_phases:
        processing_info, success = run_phase_3(target_date, data_dir,
                                               force_recompute=args.force_recompute)
        if not success:
            logger.error("Pipeline aborted at Step 3")
            return 1
    else:
        logger.info("[STEP 3] SKIPPED - Using cached data")
        processing_info = {}
    
    # ========================================================================
    # Phase 3.5: GEE Satellite Embedding Extraction
    # ========================================================================
    if gcp_project and 3 not in skip_phases:
        logger.info("\n[STEP 3.5] GEE Satellite Embedding Extraction")
        try:
            from pipeline.phase_035_embedding import run_phase_035 as _run_phase_035
            df_emb = _run_phase_035(
                target_date=target_date,
                data_dir=data_dir,
                gcp_project=gcp_project,
                use_batch=args.embedding_batch,
                limit_orbits=args.embedding_limit_orbits,
            )
            if df_emb is not None:
                logger.info(f"✓ Step 3.5 Complete: {len(df_emb)} footprints embedded")
            else:
                logger.info("✓ Step 3.5 Complete: batch task submitted to GEE")
        except Exception as e:
            logger.warning(f"⚠ Step 3.5 failed (non-fatal): {e}")
            logger.warning("  Pipeline continues without embedding features")
    else:
        logger.info("[STEP 3.5] SKIPPED - pass --gcp-project <id> to enable GEE embedding")

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

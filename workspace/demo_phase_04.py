#!/usr/bin/env python3
"""
Demo: Phase 4 - High-Performance Computational Geometry
========================================================

This script demonstrates the complete Phase 4 workflow:
1. Load OCO-2 footprints and MODIS cloud pixels from Phase 3
2. Integrate MYD03 geolocation data
3. Build KD-Tree spatial index
4. Calculate nearest cloud distances (by granule)
5. Export results to HDF5 and CSV

Usage:
    python workspace/demo_phase_04.py --date 2018-10-18 [--output results.h5]
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import platform

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase_01_metadata import OCO2MetadataRetriever
from phase_02_ingestion import DataIngestionManager
from phase_03_processing import SpatialProcessor
from phase_04_geometry import GeometryProcessor
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Phase 4: Computational Geometry Demo')
    parser.add_argument('--date', type=str, required=True,
                       help='Target date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output HDF5 file path (default: results_DATE.h5)')
    parser.add_argument('--max-distance', type=float, default=50.0,
                       help='Maximum cloud distance in km (default: 50)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    parser.add_argument('--band-width', type=float, default=10.0,
                       help='Latitude band width in degrees (default: 10.0)')
    parser.add_argument('--band-overlap', type=float, default=1.0,
                       help='Latitude band overlap in degrees (default: 1.0)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations with MODIS Aqua RGB imagery')
    parser.add_argument('--vis-dir', type=str, default='./visualizations_phase4',
                       help='Visualization output directory (default: ./visualizations_phase4)')
    
    args = parser.parse_args()
    
    # Parse date
    try:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
        return 1
    
    # Set output path
    if platform.system() == "Darwin":
        logger.info("Running on macOS - using local data directory")
        storage_dir = Config.get_data_path('local')
        
    elif platform.system() == "Linux":
        logger.info("Running on Linux - using 'external' storage directory")
        # Assume running CURC
        storage_dir = Config.get_data_path('curc')
    result_dir = Path(f"{storage_dir}/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = result_dir / f"results_{args.date}.h5"
    
    stats_path = output_path.with_suffix('.json')
    
    logger.info("=" * 70)
    logger.info("Phase 4: High-Performance Computational Geometry")
    logger.info("=" * 70)
    logger.info(f"Target date: {args.date}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Max distance: {args.max_distance} km")
    logger.info(f"  Band width: {args.band_width}° | Overlap: {args.band_overlap}°")
    logger.info(f"Output files:")
    logger.info(f"  - HDF5: {output_path}")
    logger.info(f"  - Stats: {stats_path}")
    logger.info("=" * 70)
    
    # Initialize processors
    data_dir = Path(args.data_dir)
    ingestion_manager = DataIngestionManager(output_dir=str(data_dir))
    spatial_processor = SpatialProcessor(data_dir=str(data_dir))
    geometry_processor = GeometryProcessor(data_dir=str(data_dir))
    
    # Import DownloadedFile before using it
    from phase_02_ingestion import DownloadedFile
    
    # ========================================================================
    # Step 1: Get downloaded files from previous phases
    # ========================================================================
    logger.info("\n[Step 1] Loading downloaded files from Phase 2")
    
    # Get OCO2 files
    oco2_date_dir = data_dir / "OCO2" / str(target_date.year) / f"{target_date.timetuple().tm_yday:03d}"
    if not oco2_date_dir.exists():
        logger.error(f"OCO2 data directory not found: {oco2_date_dir}")
        logger.error("Please run demo_phase_02.py first to download data")
        return 1
    
    oco2_files = []
    # Check both .h5 and .nc4 extensions (including per-orbit subfolders)
    for filepath in list(oco2_date_dir.rglob("*.h5")) + list(oco2_date_dir.rglob("*.nc4")):
        # Determine product type from filename
        if "L1bSc" in filepath.name:
            product_type = "L1B_Science"
        elif "Lite" in filepath.name or "LtCO2" in filepath.name:
            # L2 Lite files are NOT loaded - only downloaded in Phase 2 for future use
            logger.debug(f"    Skipping L2 Lite file: {filepath.name}")
            continue
        elif "L2Met" in filepath.name:
            product_type = "L2_Met"
        elif "L2CPr" in filepath.name:
            product_type = "L2_CO2Prior"
        else:
            continue
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        oco2_files.append(DownloadedFile(
            filepath=filepath,
            product_type=product_type,
            granule_id=filepath.stem,
            file_size_mb=file_size_mb,
            download_time_seconds=0.0  # Already downloaded
        ))
    
    logger.info(f"  Found {len(oco2_files)} OCO-2 file(s)")
    
    # Get MODIS files
    modis_myd35_dir = data_dir / "MODIS" / "MYD35_L2" / str(target_date.year) / f"{target_date.timetuple().tm_yday:03d}"
    modis_myd03_dir = data_dir / "MODIS" / "MYD03" / str(target_date.year) / f"{target_date.timetuple().tm_yday:03d}"
    
    myd35_files = []
    myd03_files = []
    
    if modis_myd35_dir.exists():
        for filepath in modis_myd35_dir.glob("*.hdf"):
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            myd35_files.append(DownloadedFile(
                filepath=filepath,
                product_type="MYD35_L2",
                granule_id=filepath.stem,
                file_size_mb=file_size_mb,
                download_time_seconds=0.0
            ))
    
    if modis_myd03_dir.exists():
        for filepath in modis_myd03_dir.glob("*.hdf"):
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            myd03_files.append(DownloadedFile(
                filepath=filepath,
                product_type="MYD03",
                granule_id=filepath.stem,
                file_size_mb=file_size_mb,
                download_time_seconds=0.0
            ))
    
    logger.info(f"  Found {len(myd35_files)} MYD35_L2 file(s)")
    logger.info(f"  Found {len(myd03_files)} MYD03 file(s)")
    
    if not oco2_files:
        logger.error("Missing required OCO-2 files. Please run demo_phase_02.py first")
        return 1
    
    # ========================================================================
    # Step 2: Extract OCO-2 footprints (from Phase 3)
    # ========================================================================
    logger.info("\n[Step 2] Extracting OCO-2 footprints")
    logger.info("  NOTE: L2 Lite files are not loaded (reserved for Phase 5+ use)")
    
    footprints = spatial_processor.extract_oco2_footprints(oco2_files, viewing_mode='GL')
    
    if not footprints:
        logger.error("No OCO-2 footprints extracted")
        return 1
    
    # Group by granule
    footprints_by_granule = spatial_processor.group_footprints_by_granule(footprints)
    
    # ========================================================================
    # Step 3: Identify cache directories (cache loading deferred to per-granule)
    # ========================================================================
    logger.info("\n[Step 3] Checking for combined cache files")
    
    processing_day_dir = data_dir / "processing" / str(target_date.year) / f"{target_date.timetuple().tm_yday:03d}"
    
    if not processing_day_dir.exists():
        logger.error("No combined cache directory found under data/processing. Run Phase 3 first.")
        return 1
    
    # Build mapping of granule_id -> cache_dir
    granule_cache_map = {}
    for granule_dir in sorted(processing_day_dir.glob("*")):
        if not granule_dir.is_dir():
            continue
        if list(granule_dir.glob("granule_combined_*.pkl")):
            granule_cache_map[granule_dir.name] = granule_dir
    
    logger.info(f"✓ Found {len(granule_cache_map)} granule cache director(ies)")
    
    # ========================================================================
    # Step 4: Per-Granule Processing with Caching (load cache per-granule)
    # ========================================================================
    logger.info("\n[Step 4] Per-Granule Processing: Matching OCO-2 with temporally close MODIS")
    logger.info("   Loading cache data per-granule for independent KD-Tree building")
    
    results = []
    cache_hits = 0
    cache_misses = 0
    
    # Track all clouds for visualization purposes
    all_cloud_lons = np.array([], dtype=np.float32)
    all_cloud_lats = np.array([], dtype=np.float32)
    all_cloud_flags = np.array([], dtype=np.int8)
    
    # Create visualization directory if needed
    if args.visualize:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    for oco2_granule_id, footprints in footprints_by_granule.items():
        logger.info(f"\n  Processing OCO-2 granule: {oco2_granule_id}")
        logger.info(f"    Footprints: {len(footprints):,}")
        
        # Get footprint time for matching (use first footprint's time)
        if not footprints:
            continue
        
        oco2_time = footprints[0].sounding_time
        
        # Get short orbit ID for cache directory lookup
        short_orbit_id = spatial_processor._extract_short_orbit_id(oco2_granule_id)
        
        # Get cache directory for this granule from mapping
        cache_dir = granule_cache_map.get(short_orbit_id)
        if not cache_dir:
            logger.warning(f"    No cache directory found for orbit {short_orbit_id}")
            continue
        
        
        # Load granule's combined cache file (MODIS + OCO-2)
        combined_cache_file = list(cache_dir.glob("granule_combined_*.pkl")) if cache_dir.exists() else []
        
        if not combined_cache_file:
            logger.warning(f"    No combined cache found for {oco2_granule_id}")
            continue
        
        try:
            combined_data = spatial_processor._load_cached_result(combined_cache_file[0])
            if combined_data is None:
                logger.warning(f"    Failed to load combined cache data")
                continue
        except Exception as e:
            logger.warning(f"    Error loading combined cache: {e}")
            continue
        
        # Extract cloud data for this granule
        cloud_lon_granule = combined_data.get('lon')
        cloud_lat_granule = combined_data.get('lat')
        cloud_flag_granule = combined_data.get('cloud_flag')
        
        fp_lon_granule = combined_data.get('oco2_fp_lons')
        fp_lat_granule = combined_data.get('oco2_fp_lats')
        fp_ids_granule = combined_data.get('oco2_fp_sounding_ids')
        fp_viewing_modes_granule = combined_data.get('oco2_fp_viewing_modes')
        fp_lonlat_granule = (fp_lon_granule, fp_lat_granule, fp_ids_granule, fp_viewing_modes_granule)
        
        if cloud_lon_granule is None or cloud_lat_granule is None:
            logger.warning(f"    No cloud data in combined cache")
            continue
        
        logger.info(f"    Clouds for this granule: {len(cloud_lon_granule):,}")
                
        all_cloud_lons = np.concatenate((all_cloud_lons, cloud_lon_granule))
        all_cloud_lats = np.concatenate((all_cloud_lats, cloud_lat_granule))
        all_cloud_flags = np.concatenate((all_cloud_flags, cloud_flag_granule))
        
        # Try to load per-granule distance results cache
        cache_path = cache_dir / "phase4_distance_results.pkl"
        cached_results = None
        if cache_path.exists():
            cached_results = spatial_processor._load_cached_result(cache_path)
        
        if cached_results is not None:
            results.extend(cached_results)
            cache_hits += 1
            logger.info(f"    📂 Loaded from cache: {len(cached_results):,} results")
            continue
        
        cache_misses += 1
        
        
        
        # Calculate distances for this granule
        logger.info(f"    Calculating distances...")
        
        # Banding mode: skip global KD-Tree, build per-band instead
        granule_results = geometry_processor.calculate_nearest_cloud_distances_banded(
            footprints_by_granule=fp_lonlat_granule,
            cloud_lons=cloud_lon_granule,
            cloud_lats=cloud_lat_granule,
            cloud_flags=cloud_flag_granule,
            band_width_deg=args.band_width,
            band_overlap_deg=args.band_overlap,
            max_distance_km=args.max_distance,
            oco2_granule_id=oco2_granule_id,
        )
        
        # Save to cache (in same directory as combined cache file)
        spatial_processor._save_cached_result(cache_path, granule_results)
        
        results.extend(granule_results)
        logger.info(f"    ✓ Processed {len(granule_results):,} soundings")
        
        # Create per-granule latitude-band visualizations
        if args.visualize:
            logger.info(f"    Creating per-granule latitude-band visualizations...")
            granule_vis_dir = vis_dir / f"granule_{short_orbit_id}"
            granule_vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Use granule-specific cloud and result data
            granule_vis_paths = geometry_processor.visualize_latband_distance(
                results=granule_results,
                cloud_lons=cloud_lon_granule,
                cloud_lats=cloud_lat_granule,
                cloud_flags=cloud_flag_granule,
                output_dir=granule_vis_dir,
                max_distance=args.max_distance,
                lat_band_size=5.0,
                max_clouds_per_band=50000,
                dpi=100
            )
            
            if granule_vis_paths:
                logger.info(f"    ✓ Created {len(granule_vis_paths)} latitude-band plot(s) for this granule")
    
    logger.info(f"\n✓ Total results: {len(results):,} soundings from {len(footprints_by_granule)} granule(s)")
    logger.info(f"   Cache: {cache_hits} hit(s), {cache_misses} miss(es)")
    
    if not results:
        logger.error("No collocation results generated")
        return 1
    
    # ========================================================================
    # Quick Visualization: KD-Tree Spatial Range
    # ========================================================================
    if args.visualize:
        dpi = 100
        logger.info("\n[Quick Viz] Creating KD-Tree spatial filtering visualization")
        
        # Create output directory
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Use geometry_processor to create visualization
        quick_viz_path = vis_dir / f'kdtree_spatial_range_{args.date}.png'
        geometry_processor.visualize_kdtree_spatial_range(
            results=results,
            cloud_lons=all_cloud_lons,
            cloud_lats=all_cloud_lats,
            cloud_flags=all_cloud_flags,
            output_path=quick_viz_path,
            max_distance=args.max_distance,
            dpi=200
        )
        logger.info(f"   ✓ Saved: {quick_viz_path}")

        # ====================================================================
        # Latitude-band visualization (footprints + cloud locations from all granules)
        # ====================================================================
        logger.info("\n[Lat-Band Viz] Creating footprint distance plots by latitude band")
        
        distance_paths = geometry_processor.visualize_latband_distance(
            results=results,
            cloud_lons=all_cloud_lons,
            cloud_lats=all_cloud_lats,
            cloud_flags=all_cloud_flags,
            output_dir=vis_dir,
            max_distance=args.max_distance,
            lat_band_size=10.0,
            max_clouds_per_band=50000,
            dpi=dpi
        )
        
        for path in distance_paths:
            logger.info(f"   ✓ Saved: {path}")

    # ========================================================================
    
    stats = geometry_processor.get_statistics(results)

    logger.info("\n📊 Summary Statistics:")
    logger.info(f"   Total soundings: {stats['total_soundings']}")
    logger.info(f"   Distance (km):")
    logger.info(f"     Min:    {stats['distance_km']['min']:.2f}")
    logger.info(f"     Max:    {stats['distance_km']['max']:.2f}")
    logger.info(f"     Mean:   {stats['distance_km']['mean']:.2f}")
    logger.info(f"     Median: {stats['distance_km']['median']:.2f}")
    logger.info(f"     Std:    {stats['distance_km']['std']:.2f}")
    logger.info(f"   Cloud classification:")
    logger.info(f"     Cloudy:    {stats['cloud_classification']['cloudy']}")
    logger.info(f"     Uncertain: {stats['cloud_classification']['uncertain']}")
    logger.info(f"   Distance distribution:")
    logger.info(f"     0-2 km:    {stats['distance_distribution']['0-2_km']}")
    logger.info(f"     2-5 km:    {stats['distance_distribution']['2-5_km']}")
    logger.info(f"     5-10 km:   {stats['distance_distribution']['5-10_km']}")
    logger.info(f"     10-20 km:  {stats['distance_distribution']['10-20_km']}")
    logger.info(f"     20+ km:    {stats['distance_distribution']['20+_km']}")
    
    # ========================================================================
    # Step 6: Export results
    # ========================================================================
    logger.info("\n[Step 6] Exporting results")
    
    # Export HDF5
    # Count unique MODIS granules from results
    unique_modis_granules = set()
    for result in results:
        if hasattr(result, 'modis_granule_id') and result.modis_granule_id != "N/A":
            unique_modis_granules.add(result.modis_granule_id)
    
    metadata = {
        'date': args.date,
        'max_distance_km': args.max_distance,
        'num_oco2_granules': len(footprints_by_granule),
        'num_modis_granules': len(unique_modis_granules),
    }
    geometry_processor.export_results_hdf5(results, output_path, metadata)
    
    # Export statistics
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Statistics saved: {stats_path}")
    
    
    # ========================================================================
    # Complete
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("✅ Phase 4 Complete!")
    logger.info("=" * 70)
    logger.info(f"Results saved to:")
    logger.info(f"  - {output_path}")
    logger.info(f"  - {stats_path}")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Phase 3 Demo: Spatial and Bitmask Processing
==============================================

Demonstrates OCO-2 footprint extraction, MODIS cloud mask unpacking,
and temporal matching for collocation.

Usage:
    python workspace/demo_phase_03.py --date 2018-10-18 [--mode GL]
"""

import argparse
import logging
import re
import numpy as np
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# Handle imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase_02_ingestion import DataIngestionManager
from phase_03_processing import SpatialProcessor


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Spatial and Bitmask Processing"
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2018-10-18",
        help="Target date (YYYY-MM-DD, default: 2018-10-18)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["GL", "ND", "All"],
        default="GL",
        help="OCO-2 viewing mode to extract (GL=Glint, ND=Nadir)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory (default: ./data)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization plots for granule collocations"
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="./visualizations",
        help="Output directory for visualization plots (default: ./visualizations)"
    )
    
    args = parser.parse_args()
    
    # Parse date
    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD")
        return 1
    
    print("\n" + "=" * 70)
    print("Phase 3: Spatial and Bitmask Processing")
    print("=" * 70)
    print(f"Date: {target_date.date()}")
    print(f"Viewing mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    
    try:
        # Initialize managers
        ingestion = DataIngestionManager(output_dir=args.data_dir)
        processor = SpatialProcessor(data_dir=args.data_dir)
        
        print("\n[Step 1] Downloading Phase 2 data (if needed)...")
        # This would normally use Phase 2 results, but for demo we simulate
        result = ingestion.download_all_for_date(
            target_date=target_date,
            mode_filter=args.mode if args.mode != "All" else None,
            limit_granules=2,
            include_modis=True
        )
        
        oco2_files = result.get('oco2_files', [])
        modis_files = result.get('modis_files', [])
        
        print(f"  ✓ OCO-2 files: {len(oco2_files)}")
        print(f"  ✓ MODIS files: {len(modis_files)}")
        
        # Separate MYD35_L2 and MYD03 files
        myd35_files = [f for f in modis_files if f.product_type == 'MYD35_L2']
        myd03_files = [f for f in modis_files if f.product_type == 'MYD03']
        print(f"    - MYD35_L2 (cloud mask): {len(myd35_files)}")
        print(f"    - MYD03 (geolocation): {len(myd03_files)}")
        
        print("\n[Step 2] Extracting OCO-2 footprints...")
        oco2_footprints = processor.extract_oco2_footprints(
            oco2_files,
            viewing_mode=args.mode if args.mode != "All" else "GL"
        )
        print(f"  ✓ Extracted {len(oco2_footprints)} footprint(s)")
        
        # Filter to target date (in case L2 Lite file contains multiple dates)
        if oco2_footprints:
            oco2_footprints = processor.filter_footprints_by_date(
                oco2_footprints,
                target_date,
                buffer_hours=12
            )
            print(f"  ✓ Filtered to {len(oco2_footprints)} footprint(s) for target date")
        
        # Show sample footprints
        if oco2_footprints:
            print("\n  Sample footprints:")
            for i, (sid, fp) in enumerate(list(oco2_footprints.items())[:3]):
                print(f"    - {fp}")
        
        # Group footprints by granule to determine which MODIS files to extract per orbit
        footprints_by_granule = processor.group_footprints_by_granule(oco2_footprints)
        
        # Debug: Show granule IDs being grouped
        print(f"\n  Grouped footprints by granule (keys):")
        for granule_id in footprints_by_granule.keys():
            count = len(footprints_by_granule[granule_id])
            print(f"    - {granule_id} ({count} soundings)")
        
        print("\n[Step 3] Matching MODIS files to OCO-2 orbits (temporal proximity)...")
        # Match each MODIS file to its closest OCO-2 granule using per-granule time windows
        modis_to_oco2_mapping = {}
        granule_time_ranges = {}
        for granule_id, granule_footprints in footprints_by_granule.items():
            if not granule_footprints:
                continue
            times = [fp.sounding_time for fp in granule_footprints]
            start_time = min(times)
            end_time = max(times)
            mid_time = start_time + (end_time - start_time) / 2
            granule_time_ranges[granule_id] = (start_time, end_time, mid_time)
        
        for modis_file in myd35_files:
            # Extract MODIS time from filename: MYD35_L2.A{YEAR}{DOY}.{HHMM}...
            match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', modis_file.filepath.name)
            if match:
                year = int(match.group(1))
                doy = int(match.group(2))
                hhmm = match.group(3)
                hour = int(hhmm[:2])
                minute = int(hhmm[2:4])
                modis_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                modis_time = modis_date.replace(hour=hour, minute=minute)
                
                # Assign this MODIS granule to ALL OCO-2 granules whose time window it
                # overlaps.  A single 5-minute MODIS swath can legitimately fall inside
                # the ±buffer window of two different OCO-2 granules (e.g. a GL and an ND
                # orbit that are temporally adjacent, or two orbits from the same day with
                # overlapping edges).  Picking only the "closest" would silently deprive
                # the other orbit of cloud pixels in that region.
                buffer_minutes = 20  # matches Phase 3 spec (±20 min for Aqua drift)
                matched_any = False

                for oco2_granule_id, (start_time, end_time, mid_time) in granule_time_ranges.items():
                    window_start = start_time - timedelta(minutes=buffer_minutes)
                    window_end = end_time + timedelta(minutes=buffer_minutes)
                    if window_start <= modis_time <= window_end:
                        time_diff = abs((modis_time - mid_time).total_seconds())
                        if oco2_granule_id not in modis_to_oco2_mapping:
                            modis_to_oco2_mapping[oco2_granule_id] = []
                        modis_to_oco2_mapping[oco2_granule_id].append(modis_file)
                        orbit_id = processor._extract_short_orbit_id(oco2_granule_id)
                        print(f"    {modis_file.filepath.name} → orbit {orbit_id} (Δt={time_diff/60:.1f} min)")
                        matched_any = True

                if not matched_any:
                    print(f"    {modis_file.filepath.name} → no OCO-2 granule within ±{buffer_minutes} min (skipped)")
        print(f"\n  ✓ Matched {len(myd35_files)} MODIS files to {len(modis_to_oco2_mapping)} OCO-2 orbit(s)")
        
        print("\n[Step 4] Extracting MODIS cloud masks (per OCO-2 orbit)...")
        modis_cloud_masks = {}
        orbit_cache_dirs = {}  # Track cache directories for each orbit
        
        # Process each OCO-2 granule with its matched MODIS files
        for oco2_granule_id, matched_modis_files in modis_to_oco2_mapping.items():
            # Extract short orbit ID from granule_id
            # Expected format: oco2_L1bScGL_22845a_181018_B11006r_220921185957.h5
            print(f"\n  Debug - Processing granule_id: {oco2_granule_id}")
            # Use the same helper as Phase 3/4 so the cache folder name always includes
            # the viewing mode (e.g. "22845a_GL") and GL/ND orbits that share an orbit_id
            # are kept in separate directories.
            oco2_orbit_id = processor._extract_short_orbit_id(oco2_granule_id) if oco2_granule_id else None
            print(f"      Extracted orbit_id: {oco2_orbit_id}")
            
            print("oco2_granule_id:", oco2_granule_id)
            print("oco2_orbit_id:", oco2_orbit_id)
            
            print(f"\n  Processing orbit {oco2_orbit_id} with {len(matched_modis_files)} MODIS file(s)...")
            
            # Check if combined cache file already exists
            if oco2_orbit_id and matched_modis_files:
                first_modis = matched_modis_files[0]
                match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', first_modis.filepath.name)
                if match:
                    year = int(match.group(1))
                    doy = int(match.group(2))
                    orbit_folder = Path(processor.data_dir) / "processing" / str(year) / str(doy) / str(oco2_orbit_id)
                    combined_cache_file = orbit_folder / f"granule_combined_{oco2_orbit_id}.pkl"
                    
                    if combined_cache_file.exists():
                        print(f"    ✓ Found existing combined cache file: {combined_cache_file.name}")
                        try:
                            combined_data = processor._load_cached_result(combined_cache_file)
                            
                            # Extract cloud masks from combined data
                            if combined_data.get('lon') is not None:
                                # Reconstruct MODISCloudMask objects for backward compatibility
                                from phase_03_processing import MODISCloudMask
                                for modis_granule_id in combined_data.get('modis_granules', []):
                                    # Create a MODISCloudMask with the combined data
                                    # Note: We're creating a single mask with all data for simplicity
                                    if modis_granule_id not in modis_cloud_masks:
                                        cloud_mask = MODISCloudMask(
                                            granule_id=modis_granule_id,
                                            observation_time=datetime.now(),  # Placeholder
                                            lon=combined_data['lon'],
                                            lat=combined_data['lat'],
                                            cloud_flag=combined_data['cloud_flag']
                                        )
                                        modis_cloud_masks[modis_granule_id] = cloud_mask
                            
                            orbit_cache_dirs[oco2_orbit_id] = orbit_folder
                            
                            pixel_count = len(combined_data['lon']) if combined_data.get('lon') is not None else 0
                            footprint_count = combined_data.get('footprint_count', 0)
                            print(f"    ✓ Loaded from cache:")
                            if pixel_count > 0:
                                print(f"      - {pixel_count} cloud pixels")
                            if footprint_count > 0:
                                print(f"      - {footprint_count} OCO-2 footprints")
                            
                            continue  # Skip extraction and combination for this granule
                        except Exception as e:
                            print(f"    ⚠️  Failed to load cache file: {e}")
                            print(f"    → Re-extracting data...")
            
            # Get matched MYD03 files
            matched_myd03 = []
            for myd35_file in matched_modis_files:
                # Extract time ID from MYD35 filename
                match = re.search(r'A(\d{4}\d{3}\.\d{4})', myd35_file.filepath.name)
                if match:
                    time_id = 'A' + match.group(1)
                    # Find matching MYD03
                    for myd03_file in myd03_files:
                        if time_id in myd03_file.filepath.name:
                            matched_myd03.append(myd03_file)
                            break
            
            # Extract cloud masks for this granule's MODIS files
            # Pass short_orbit_id so all granules from same orbit share cache folder
            granule_cloud_mask = processor.extract_modis_cloud_mask(
                matched_modis_files,  # Only MODIS files that match this granule
                myd03_files=matched_myd03 if matched_myd03 else None,
                oco2_orbit_id=oco2_orbit_id  # Use orbit_id for cache folder (shared across all granules from same orbit)
            )
            modis_cloud_masks.update(granule_cloud_mask)
            
            # Determine cache directory for this orbit and combine data
            if oco2_orbit_id and matched_modis_files:
                # Get cache directory from first MODIS file (follows the same path structure)
                # The orbit folder is created during extract_modis_cloud_mask
                first_modis = matched_modis_files[0]
                match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', first_modis.filepath.name)
                if match:
                    year = int(match.group(1))
                    doy = int(match.group(2))
                    orbit_folder = Path(processor.data_dir) / "processing" / str(year) / str(doy) / str(oco2_orbit_id)
                    
                    # Filter OCO-2 footprints for this specific granule
                    granule_footprints = {
                        sid: fp for sid, fp in oco2_footprints.items()
                        if fp.granule_id == oco2_granule_id
                    }
                    
                    # Combine cloud masks and footprints for this granule
                    if granule_cloud_mask or granule_footprints:
                        print(f"    Combining data for OCO-2 orbit {oco2_orbit_id}...")
                        print(f"      - MODIS granules: {len(granule_cloud_mask)}")
                        print(f"      - OCO-2 footprints: {len(granule_footprints)}")
                        
                        orbit_folder.mkdir(parents=True, exist_ok=True)
                        
                        # Use processor method to combine and save both cloud masks and footprints
                        combined_result = processor.combine_OCO_fp_cloud_masks_by_granule(
                            granule_cloud_mask,
                            granule_footprints,
                            oco2_granule_id,
                            orbit_folder
                        )
                        
                        # Report what was saved
                        pixel_count = len(combined_result['lon']) if combined_result.get('lon') is not None else 0
                        modis_count = len(combined_result.get('modis_granules', []))
                        footprint_count = combined_result.get('footprint_count', 0)
                        
                        if pixel_count > 0 or footprint_count > 0:
                            print(f"      ✓ Saved combined data:")
                            if pixel_count > 0:
                                print(f"        - {pixel_count} cloud pixels from {modis_count} MODIS granule(s)")
                            if footprint_count > 0:
                                print(f"        - {footprint_count} OCO-2 footprints")
                            orbit_cache_dirs[oco2_orbit_id] = orbit_folder
                    else:
                        print(f"    ⚠️  No data to combine for granule {oco2_granule_id}")
        
        # Calculate total pixels handling both new (array) and old (tuple/list) formats
        total_pixels = 0
        for granule_pixels in modis_cloud_masks.values():
            if hasattr(granule_pixels, 'lon') and granule_pixels.lon is not None:
                # New format: MODISCloudMask with numpy arrays
                total_pixels += len(granule_pixels.lon)
            elif hasattr(granule_pixels, 'pixels') and granule_pixels.pixels:
                # Old format: MODISCloudMask with tuples
                total_pixels += len(granule_pixels.pixels)
        
        print(f"\n  ✓ Total: {total_pixels} cloud pixel(s) from {len(modis_cloud_masks)} MODIS granule(s)")
        
        # Summary: Show where MYD35 cache files were saved
        print("\n  [Cache Summary]")
        print(f"  MYD35_L2 granules extracted: {len(modis_cloud_masks)}")
        print(f"  OCO-2 orbits processed: {len(modis_to_oco2_mapping)}")
        print(f"\n  Expected MYD35 cache file locations:")
        processing_dir = Path(args.data_dir) / "processing"
        if processing_dir.exists():
            for oco2_granule_id in modis_to_oco2_mapping.keys():
                actual_orbit_id = processor._extract_short_orbit_id(oco2_granule_id)
                # Find matching cache folders
                for year_dir in processing_dir.glob("*"):
                    for doy_dir in year_dir.glob("*"):
                        orbit_cache_folder = doy_dir / str(actual_orbit_id)
                        if orbit_cache_folder.exists():
                            cached_myd35 = list(orbit_cache_folder.glob("myd35_*.pkl"))
                            if cached_myd35:
                                print(f"    ✓ Orbit {actual_orbit_id}: {len(cached_myd35)} cache file(s)")
                                for f in cached_myd35:
                                    print(f"        - {f.name}")
        
        print("\n[Step 5] Temporal matching (OCO-2 ↔ MODIS)...")
        temporal_matching = processor.match_temporal_windows(
            oco2_footprints,
            modis_files,
            buffer_minutes=20
        )
        matched = sum(1 for m in temporal_matching.values() if len(m) > 0)
        print(f"  ✓ Matched {matched}/{len(oco2_footprints)} soundings")
        
        # Summary
        summary = processor.get_processing_summary(
            oco2_footprints,
            modis_cloud_masks,
            temporal_matching
        )
        
        print("\n" + "=" * 70)
        print("Processing Summary")
        print("=" * 70)
        print(f"OCO-2 footprints:    {summary['oco2_footprints']}")
        print(f"MODIS granules:      {summary['modis_granules']}")
        print(f"Cloud pixels:        {summary['total_cloud_pixels']}")
        print(f"Matched soundings:   {summary['matched_soundings']}")
        print(f"Unmatched soundings: {summary['unmatched_soundings']}")
        
        # Visualization with latitude bands
        if args.visualize:
            print("\n[Step 6] Creating latitude-band visualizations...")
            
            # Find a cache folder from the extracted MODIS data
            # Typically: data/processing/YYYY/DOY/orbit_id/
            data_dir = Path(args.data_dir).resolve()
            processing_dir = data_dir / "processing" / f"{target_date.year}" / f"{target_date.timetuple().tm_yday:03d}"
            
            # Find the first available orbit folder with cache files
            cache_dirs = []
            if processing_dir.exists():
                for orbit_dir in doy_dir.glob("*"):
                    cache_files = list(orbit_dir.glob("granule_combined_*.pkl"))
                    if cache_files:
                        cache_dirs.append(orbit_dir)
            
            cache_dirs = list(set(cache_dirs))  # Unique cache directories
            
                    
            
            if len(cache_dirs)>0:
                for cache_dir in cache_dirs:
                    print(f"\n  Processing cache directory: {cache_dir}")
                    viz_results = processor.plot_by_latitude_bands(
                        cache_dir,
                        # output_dir=args.viz_dir,
                        # use the base name of the cache directory to create a subfolder in visualizations
                        output_dir=Path(args.viz_dir) / cache_dir.name,
                        lat_band_size=10.0,
                        figsize=(16, 10),
                        dpi=100
                    )
                    if viz_results:
                        print(f"  ✓ Created {len(viz_results)} latitude-band visualization(s)")
                        for lat_band, filepath in list(viz_results.items())[:3]:
                            print(f"    - {filepath.name}")
                        if len(viz_results) > 3:
                            print(f"    ... and {len(viz_results) - 3} more")
                    else:
                        print("  ✗ No visualizations created")
            else:
                print("  ✗ No MODIS cache files found in processing directory")
        
        print("\n✓ Phase 3 processing complete!")
        return 0
    
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    exit(main())

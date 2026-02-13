#!/usr/bin/env python3
"""
Demo script for Phase 2: Targeted Data Ingestion
=================================================

This script demonstrates the complete Phase 2 workflow:
1. Retrieve OCO-2 metadata for a target date
2. Download OCO-2 products (L1B, L2 Lite, L2 Met, L2 CO2Prior)
3. Find and download MODIS products (MYD35_L2, MYD03) for the temporal window

Usage:
    python workspace/demo_phase_02.py [--date YYYY-MM-DD] [--no-modis] [--mode GL|ND]

Example:
    python workspace/demo_phase_02.py --date 2018-10-18 --mode GL
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phase_02_ingestion import DataIngestionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Phase 2: Download OCO-2 and MODIS data for a specific date'
    )
    parser.add_argument(
        '--date',
        type=str,
        default='2018-10-18',
        help='Target date in YYYY-MM-DD format (default: 2018-10-18)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['GL', 'ND', 'all'],
        default='GL',
        help='Viewing mode filter: GL (Glint), ND (Nadir), or all (default: GL)'
    )
    parser.add_argument(
        '--orbit',
        type=int,
        help='Specific orbit number to download (optional)'
    )
    parser.add_argument(
        '--no-modis',
        action='store_true',
        help='Skip MODIS downloads'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for downloads (default: ./data)'
    )
    parser.add_argument(
        '--oco2-only',
        action='store_true',
        help='Download only L1B (skip L2 products)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only check file existence without downloading (saves storage)'
    )
    parser.add_argument(
        '--storage-type',
        type=str,
        choices=['default', 'local', 'external', 'scratch', 'shared', 'curc'],
        default='default',
        help='Storage location type (default: default). Use env vars: OCO2_DATA_ROOT, SCRATCH_DIR, SHARED_DATA_ROOT, CURC_DATA_ROOT'
    )
    parser.add_argument(
        '--limit-granules',
        type=int,
        help='Limit to first N granules for testing (optional)'
    )
    return parser.parse_args()


def main():
    """Main demo function."""
    args = parse_arguments()
    
    # Parse date
    try:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
        return 1
    
    print("=" * 70)
    print("Phase 2: OCO-2/MODIS Data Ingestion Demo")
    print("=" * 70)
    print(f"Target Date: {target_date.date()}")
    print(f"Viewing Mode: {args.mode if args.mode != 'all' else 'All modes'}")
    if args.limit_granules:
        print(f"Limit Granules: {args.limit_granules} (testing mode)")
    print(f"Include MODIS: {not args.no_modis}")
    print(f"Storage Type: {args.storage_type}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Mode: {'DRY RUN (check only)' if args.dry_run else 'DOWNLOAD'}")
    print("=" * 70)
    
    # Check for credentials (only needed if not in dry-run mode)
    earthdata_user = os.environ.get('EARTHDATA_USERNAME')
    earthdata_pass = os.environ.get('EARTHDATA_PASSWORD')
    laads_token = os.environ.get('LAADS_TOKEN')
    
    if not args.dry_run:
        if not earthdata_user or not earthdata_pass:
            logger.warning("⚠ EARTHDATA credentials not found in environment!")
            logger.warning("  Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD to enable downloads")
            logger.warning("  Example: export EARTHDATA_USERNAME=your_username")
            print()
        
        if not args.no_modis and not laads_token:
            logger.warning("⚠ LAADS_TOKEN not found in environment!")
            logger.warning("  MODIS downloads may be rate-limited without a token")
            logger.warning("  Get token from: https://ladsweb.modaps.eosdis.nasa.gov/profile/")
            print()
    else:
        logger.info("🔍 Dry-run mode: File existence will be checked, no downloads will occur")
        print()
    
    # Initialize ingestion manager
    try:
        manager = DataIngestionManager(
            output_dir=args.output_dir,
            earthdata_username=earthdata_user,
            earthdata_password=earthdata_pass,
            laads_token=laads_token,
            dry_run=args.dry_run,
            storage_type=args.storage_type
        )
    except Exception as e:
        logger.error(f"Failed to initialize DataIngestionManager: {e}")
        return 1
    
    # Download data
    try:
        mode_filter = None if args.mode == 'all' else args.mode
        
        result = manager.download_all_for_date(
            target_date=target_date,
            orbit_filter=args.orbit,
            mode_filter=mode_filter,
            include_modis=not args.no_modis,
            limit_granules=1,  # Download only first granule for testing
        )
        
        # Print detailed summary
        print("\n" + "=" * 70)
        print("Download Complete!")
        print("=" * 70)
        
        if result['granules']:
            print(f"\nOCO-2 Granules ({len(result['granules'])}):")
            for granule in result['granules']:
                print(f"  • Orbit {granule.orbit_number} ({granule.viewing_mode})")
                print(f"    {granule.granule_id}")
                print(f"    {granule.start_time} - {granule.end_time}")
        
        if result['oco2_files']:
            print(f"\nOCO-2 Files Downloaded ({len(result['oco2_files'])}):")
            by_type = {}
            for f in result['oco2_files']:
                by_type.setdefault(f.product_type, []).append(f)
            for ptype, files in sorted(by_type.items()):
                total_size = sum(f.file_size_mb for f in files)
                print(f"  • {ptype}: {len(files)} file(s), {total_size:.2f} MB")
        
        if result['modis_files']:
            print(f"\nMODIS Files Downloaded ({len(result['modis_files'])}):")
            by_type = {}
            for f in result['modis_files']:
                by_type.setdefault(f.product_type, []).append(f)
            for ptype, files in sorted(by_type.items()):
                total_size = sum(f.file_size_mb for f in files)
                print(f"  • {ptype}: {len(files)} file(s), {total_size:.2f} MB")
        
        # Overall statistics
        stats = manager.get_download_summary()
        print(f"\nOverall Statistics:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Total size: {stats['total_size_gb']:.3f} GB")
        print(f"  Total time: {stats['total_time_seconds']:.1f} seconds")
        if stats['total_time_seconds'] > 0:
            print(f"  Average speed: {stats['average_speed_mbps']:.2f} Mbps")
        if stats['failed_count'] > 0:
            print(f"  ⚠ Failed downloads: {stats['failed_count']}")
        
        print("\n✓ Phase 2 demo completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during download: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

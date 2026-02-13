#!/usr/bin/env python3
"""
Phase 1 Demo: OCO-2 Metadata Acquisition
=========================================

This script demonstrates the Phase 1 functionality:
- Fetching OCO-2 L1B Science metadata from GES DISC
- Parsing orbit information and viewing modes
- Extracting temporal windows for MODIS matching
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phase_01_metadata import OCO2MetadataRetriever
from utils import setup_logging
from config import Config


def demo_phase_01(target_date: datetime, orbit_number: int = None, 
                  viewing_mode: str = None):
    """
    Demonstrate Phase 1 metadata acquisition.
    
    Args:
        target_date: Date to query for OCO-2 data
        orbit_number: Optional specific orbit number
        viewing_mode: Optional viewing mode filter ('GL' or 'ND')
    """
    print("="*70)
    print("Phase 1: OCO-2 Metadata Acquisition and Temporal Filtering")
    print("="*70)
    print(f"\nTarget Date: {target_date.date()}")
    
    if orbit_number:
        print(f"Orbit Filter: {orbit_number}")
    if viewing_mode:
        print(f"Mode Filter: {viewing_mode}")
    
    print("\n" + "-"*70)
    
    # Initialize retriever
    retriever = OCO2MetadataRetriever()
    
    # Step 1: Fetch XML metadata
    print("\n[Step 1] Fetching OCO-2 L1B Science XML from GES DISC...")
    try:
        xml_content = retriever.fetch_oco2_xml(target_date, orbit_number, viewing_mode)
        print(f"✓ Retrieved XML metadata ({len(xml_content):,} bytes)")
    except Exception as e:
        print(f"✗ Error fetching XML: {e}")
        return
    
    # Step 2: Parse orbit information
    print("\n[Step 2] Parsing orbit information...")
    try:
        granules = retriever.parse_orbit_info(xml_content)
        print(f"✓ Found {len(granules)} granule(s)")
    except Exception as e:
        print(f"✗ Error parsing XML: {e}")
        return
    
    if not granules:
        print("\n⚠ No granules found for the specified criteria")
        return
    
    # Display granule details
    print("\n" + "-"*70)
    print("OCO-2 Granule Details:")
    print("-"*70)
    
    # Group by orbit and mode
    orbit_groups = {}
    for g in granules:
        key = (g.orbit_number, g.viewing_mode)
        if key not in orbit_groups:
            orbit_groups[key] = []
        orbit_groups[key].append(g)
    
    for (orbit, mode), group in sorted(orbit_groups.items()):
        print(f"\nOrbit {orbit} - Mode: {mode} ({len(group)} granule(s))")
        for i, g in enumerate(group, 1):
            duration = (g.end_time - g.start_time).total_seconds() / 60
            print(f"  {i}. {g.granule_id}")
            print(f"     Version: {g.version}")
            print(f"     Orbit_number: {g.orbit_number}")
            print(f"     Start: {g.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"     End:   {g.end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"     Duration: {duration:.1f} minutes")
    
    # Step 3: Extract temporal window
    print("\n[Step 3] Extracting temporal window...")
    try:
        start_time, end_time = retriever.extract_temporal_window(
            granules, orbit_number, viewing_mode
        )
        
        duration = (end_time - start_time).total_seconds() / 60
        
        print(f"✓ Temporal window established")
        print(f"\n{'='*70}")
        print("OCO-2 Observation Window:")
        print(f"{'='*70}")
        print(f"  Start: {start_time.isoformat()}")
        print(f"  End:   {end_time.isoformat()}")
        print(f"  Duration: {duration:.2f} minutes")
        
        # Calculate MODIS search window (±20 minutes)
        from datetime import timedelta
        modis_start = start_time - timedelta(minutes=Config.MODIS_TEMPORAL_BUFFER_MINUTES)
        modis_end = end_time + timedelta(minutes=Config.MODIS_TEMPORAL_BUFFER_MINUTES)
        modis_duration = (modis_end - modis_start).total_seconds() / 60
        
        print(f"\n{'='*70}")
        print(f"MODIS Search Window (±{Config.MODIS_TEMPORAL_BUFFER_MINUTES} min buffer):")
        print(f"{'='*70}")
        print(f"  Start: {modis_start.isoformat()}")
        print(f"  End:   {modis_end.isoformat()}")
        print(f"  Duration: {modis_duration:.2f} minutes")
        
        print(f"\n{'='*70}")
        print("Phase 1 Summary:")
        print(f"{'='*70}")
        print(f"  Total granules found: {len(granules)}")
        print(f"  Unique orbits: {len(orbit_groups)}")
        print(f"  Temporal window: {duration:.1f} minutes")
        print(f"  MODIS search window: {modis_duration:.1f} minutes")
        print(f"\n✓ Phase 1 completed successfully!")
        
        return {
            'granules': granules,
            'temporal_window': (start_time, end_time),
            'modis_window': (modis_start, modis_end),
            'orbit_groups': orbit_groups
        }
        
    except ValueError as e:
        print(f"✗ Error extracting temporal window: {e}")
        return


def main():
    """Main entry point."""
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Example 1: Query a specific date
    print("\n" + "="*70)
    print("Example 1: Query all OCO-2 data for a specific date")
    print("="*70)
    
    target_date = datetime(2023, 7, 15)
    result1 = demo_phase_01(target_date)
    
    # Example 2: Query with filters (if there were multiple orbits/modes)
    if result1 and len(result1['orbit_groups']) > 1:
        print("\n\n" + "="*70)
        print("Example 2: Query with orbit/mode filter")
        print("="*70)
        
        # Get first orbit/mode combination
        first_orbit, first_mode = list(result1['orbit_groups'].keys())[0]
        demo_phase_01(target_date, orbit_number=first_orbit, viewing_mode=first_mode)
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == "__main__":
    main()

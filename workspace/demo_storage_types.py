#!/usr/bin/env python3
"""
Demo: Storage Type Configuration
=================================

This script demonstrates how different storage types work.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import Config
from phase_02_ingestion import DataIngestionManager

print("=" * 70)
print("Storage Type Configuration Demo")
print("=" * 70)

print("\n1. Default Storage Paths (without environment variables):")
print("-" * 70)
for storage_type in ['default', 'local', 'external', 'scratch', 'shared', 'curc']:
    path = Config.get_data_path(storage_type)
    print(f"  {storage_type:10s} -> {path}")

print("\n2. Setting Custom Environment Variables:")
print("-" * 70)
os.environ['OCO2_DATA_ROOT'] = '/Volumes/ExternalDrive/oco2_data'
os.environ['SCRATCH_DIR'] = './workspace/scratch_temp'
os.environ['SHARED_DATA_ROOT'] = '/shared/team/oco2_archive'
os.environ['CURC_DATA_ROOT'] = '/projects/user123/oco2_data'

# Note: These are read when Config class is loaded, so we need to reload
# In practice, set env vars before running your script
print("  export OCO2_DATA_ROOT=/Volumes/ExternalDrive/oco2_data")
print("  export SCRATCH_DIR=./workspace/scratch_temp")
print("  export SHARED_DATA_ROOT=/shared/team/oco2_archive")
print("  export CURC_DATA_ROOT=/projects/user123/oco2_data")

# Reload config to pick up new env vars
from importlib import reload
import config as config_module
reload(config_module)
from config import Config

print("\n3. Storage Paths WITH Environment Variables:"), 'curc'
print("-" * 70)
for storage_type in ['default', 'local', 'external', 'scratch', 'shared']:
    path = Config.get_data_path(storage_type)
    print(f"  {storage_type:10s} -> {path}")

print("\n4. Using Storage Types with DataIngestionManager:")
print("-" * 70)

# Example 1: Using storage_type parameter
print("\n  a) Using storage_type='external':")
manager = DataIngestionManager(storage_type='external', dry_run=True)
print(f"     Data directory: {manager.output_dir}")

print("\n  b) Using storage_type='scratch':")
manager = DataIngestionManager(storage_type='scratch', dry_run=True)
print(f"     Data directory: {manager.output_dir}")

print("\n  c) Explicit output_dir overrides storage_type:")
manager = DataIngestionManager(
    output_dir='/custom/path/data',
    storage_type='external',  # This is ignored when output_dir is explicit
    dry_run=True
)
print(f"     Data directory: {manager.output_dir}")

print("\n5. Command-Line Usage Examples:")
print("-" * 70)
print("  # Use external storage")
print("  export OCO2_DATA_ROOT=/Volumes/ExternalDrive/oco2_data")
print("  python demo_phase_02.py --storage-type external --date 2018-10-18")
print()
print("  # Use scratch space")
print("  python demo_phase_02.py --storage-type scratch --date 2018-10-18")
print()
print("  # Use CURC projects storage")
print("  export CURC_DATA_ROOT=/projects/${USER}/oco2_data")
print("  python demo_phase_02.py --storage-type curc --date 2018-10-18")
print()
print("  # Explicit path (overrides storage-type)")
print("  python demo_phase_02.py --output-dir /mnt/data/oco2 --date 2018-10-18")

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print("✓ Storage types are configured via Config.DATA_ROOT_DIRS")
print("✓ Environment variables customize paths per system")
print("✓ --storage-type CLI option selects which to use")
print("✓ --output-dir CLI option overrides everything")
print("=" * 70)

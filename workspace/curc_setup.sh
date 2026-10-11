#!/bin/bash
# CURC (Colorado University Research Computing) Setup for OCO-2 Analysis
# =======================================================================
#
# This script shows how to configure storage paths for CURC/Alpine cluster
#
# Usage:
#   source workspace/curc_setup.sh
#
# Then run:
#   python workspace/demo_phase_02.py --storage-type curc --dry-run --date 2018-10-18

echo "Setting up CURC environment for OCO-2 data analysis..."

# Option 1: Use CURC_DATA_ROOT (recommended)
# This should point to your projects directory or PetaLibrary allocation
export CURC_DATA_ROOT="/projects/${USER}/oco2_data"

# Option 2: If CURC_DATA_ROOT is not set, it will use PROJECTS variable
# export PROJECTS="/projects/${USER}"

# For scratch space (temporary fast storage)
export SCRATCH_DIR="/scratch/alpine/${USER}/oco2_data"

# NASA Earthdata credentials (if you have them)
# export EARTHDATA_USERNAME="your_username"
# export EARTHDATA_PASSWORD="your_password"

# LAADS token for MODIS data
# export LAADS_TOKEN="your_token_here"

echo "✓ CURC environment configured:"
echo "  Data root:  ${CURC_DATA_ROOT}"
echo "  Scratch:    ${SCRATCH_DIR}"
echo ""
echo "Usage examples:"
echo "  # Check what would be downloaded (dry-run)"
echo "  python workspace/demo_phase_02.py --storage-type curc --dry-run --date 2018-10-18"
echo ""
echo "  # Download to CURC projects storage"
echo "  python workspace/demo_phase_02.py --storage-type curc --date 2018-10-18"
echo ""
echo "  # Use scratch for temporary processing"
echo "  python workspace/demo_phase_02.py --storage-type scratch --date 2018-10-18"

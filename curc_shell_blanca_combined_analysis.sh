#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_spectral_anal
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable


module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data
# Prepend conda's libs so Python's netCDF4/h5py loads the conda-compiled
# libhdf5 rather than the system one injected by `module load hdf5/...`.
# Without this, an ABI mismatch causes NC_EHDF (-101) at H5Fopen() time.
# On Linux, $CONDA_PREFIX may be empty when conda activate runs non-interactively
# (SLURM batch context), so hardcode the path as a reliable fallback.
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

# HDF5 file locking is not supported on Lustre (/pl/active/).
# Without this, HDF5 ≥ 1.10 raises NC_EHDF (-101) on any open() call.
export HDF5_USE_FILE_LOCKING=FALSE


cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

# ── Font preflight (manuscript style needs Arial + Times New Roman) ──────────
# The 2026-07-10/11 style pass renders every figure in Arial with the photon
# path symbol l' in Times New Roman italic (workspace/plot_style.py).  Neither
# font ships with Linux: copy the six .ttf files from macOS
#   /System/Library/Fonts/Supplemental/{Arial,Arial Bold,Arial Italic,
#   Arial Bold Italic,Times New Roman,Times New Roman Italic}.ttf
# into <repo>/fonts/ (gitignored).  FAIL FAST rather than render 24 h of
# figures in the DejaVu fallback.
python - <<'PYEOF'
import sys
sys.path.insert(0, 'workspace')
from plot_style import fonts_available
sys.exit(0 if fonts_available() else 1)
PYEOF

python src/analysis/run_all.py --distance-col cld_dist_km --land-class
python src/analysis/spec_sensitivity.py    # defaults to combined_2016_2020_dates.parquet

# Savgol A/B (M9a, appendix A5) — reads the dual-fit fitting_details h5s.
python src/spectral/compare_savgol_fits.py \
    --fitting-dir results/fitting_details \
    --output-dir results/model_comparison/savgol_ab

# python src/analysis/run_all.py --distance-col weighted_cloud_dist_km
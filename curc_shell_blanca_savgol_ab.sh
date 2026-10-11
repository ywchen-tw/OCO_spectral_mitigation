#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=savgol_ab
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable

# SG-vs-no-SG A/B on the spectral fits (review M9a; appendix figure A5).
# Gathers the paired (*_fitting, *_fitting_nosg) parameter sets from every
# fitting_details_*.h5 under the storage root and writes
#   results/model_comparison/savgol_ab/savgol_ab_stats.csv   (per band/param)
#   results/model_comparison/savgol_ab/savgol_ab_scatter.png (6-panel scatter)
# in the repo tree.  Rsync those two files back to the local checkout; the
# stats fill the "X %" placeholder in the A5 caption stub (TODO_ACCOMPLISH §5).
# NOTE: no checkpointing — a preempted job restarts from zero (one-shot
# ~30-60 min, so acceptable; resubmit non-preemptable if preemption bites).
# NOTE: the l' glyph needs Times New Roman (macOS/Windows); on a bare Linux
# node matplotlib warns and substitutes — fine for this appendix QC figure.

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data
# Prepend conda's libs so Python's h5py loads the conda-compiled libhdf5
# rather than the system one injected by `module load hdf5/...` (ABI mismatch
# otherwise raises NC_EHDF (-101) at H5Fopen time).  $CONDA_PREFIX may be
# empty in SLURM batch context, so hardcode the Linux fallback.
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

# HDF5 file locking is not supported on Lustre; the gather loop silently
# skips files whose open() raises, so this is load-bearing for coverage.
export HDF5_USE_FILE_LOCKING=FALSE

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /projects/yuch8913/OCO_spectral_mitigation

# Storage root: env override, else the .bashrc value, else the known scratch
# path (batch shells don't always inherit interactive exports — this was the
# failure mode of the first two attempts).
storage_root="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-/scratch/alpine/yuch8913/oco/spec_anal}}"
storage_root="${storage_root%/}"
fitting_dir="${storage_root}/results/fitting_details"

if [[ ! -d "$fitting_dir" ]]; then
    echo "ERROR: fitting-details dir not found: $fitting_dir"
    echo "Directories named fitting_details under ${storage_root}:"
    find "$storage_root" -maxdepth 3 -type d -name fitting_details 2>/dev/null
    exit 1
fi

n_h5=$(ls "$fitting_dir"/fitting_details_*.h5 2>/dev/null | wc -l)
echo "fitting_dir: $fitting_dir  (${n_h5} h5 files)"
if [[ "$n_h5" -eq 0 ]]; then
    echo "ERROR: no fitting_details_*.h5 files in $fitting_dir"
    exit 1
fi

PYTHONPATH=src python src/spectral/compare_savgol_fits.py \
    --fitting-dir "$fitting_dir" \
    --output-dir results/model_comparison/savgol_ab

# Coverage check: the script's "[N/M files with _nosg]" line should have
# M ≈ n_h5 above; a much smaller M means files were silently skipped
# (locking / corrupt h5) — re-run after investigating.
echo ""
echo "Expected file count: ${n_h5}  (compare against the [N/M] line above)"
echo "Outputs:"
ls -la results/model_comparison/savgol_ab/

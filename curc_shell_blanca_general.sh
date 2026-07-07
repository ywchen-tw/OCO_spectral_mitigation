#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
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

# Prevent MKL/OpenBLAS from creating a thread pool in the parent process.
# fp_abs_coeff.py parallelises at the Python (ProcessPoolExecutor) level, so
# each worker should run single-threaded numpy.  Without this, fork/forkserver
# workers inherit a broken BLAS thread state → deadlock on first numpy call.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Browser-derived GES DISC /data/.TOKEN paths can be stale or context-specific.
# Your CURC curl test showed the plain /data/OCO2_DATA/... route is healthy.
unset GESDISC_DATA_TOKEN
# Prefer ~/.netrc / anonymous-readable GES DISC routes for batch runs.
# Stale EARTHDATA_* environment values force the downloader into a failing
# auth path, even when the plain file URLs are reachable from CURC.
unset EARTHDATA_USERNAME
unset EARTHDATA_PASSWORD

cd /projects/yuch8913/OCO_spectral_mitigation

# ============================================================================
# Single date processing
# ============================================================================
target_year=2018
target_month=3
target_day=15

date=$(printf "%04d-%02d-%02d" "$target_year" "$target_month" "$target_day")
echo "Processing date: $date"

python workspace/oco_modis_cloud_distance.py \
    --date "$date" \
    --force-recompute-if-lite-before 11.2r

if [ $? -ne 0 ]; then
    echo "Failed to process date: $date"
else
    echo "Successfully processed: $date"
    storage_root="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"
    cloud_distance_file="${storage_root%/}/results/results_${date}.h5"
    if [[ ! -s "$cloud_distance_file" ]]; then
        echo "Missing cloud-distance file: $cloud_distance_file"
        echo "Skipping spectral fitting for date: $date"
        exit 1
    fi
    python src/spectral/fitting.py --date "$date" #--delete-ocofiles
fi

echo ""



# ============================================================================
# Option 2: Hard-coded dates (uncomment to use)
# ============================================================================
# Define dates to process (modify as needed)
# dates=(
#     "2018-10-18"
#     "2020-01-04"
#     "2020-01-08"
# )
#
# # Loop through each date
# for date in "${dates[@]}"; do
#     echo "Processing date: $date"
#     python workspace/oco_modis_cloud_distance.py --date "$date" --delete-modis
#     if [ $? -ne 0 ]; then
#         echo "Failed to process date: $date"
#     else
#         echo "Successfully processed: $date"
#     fi
#     echo ""
# done

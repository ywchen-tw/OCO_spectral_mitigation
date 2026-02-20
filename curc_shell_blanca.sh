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
#SBATCH --partition=blanca-airs
#SBATCH --qos=blanca-airs
#SBATCH --cpu-bind=cores


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

# ============================================================================
# Option 1: Loop with year, month, day (ACTIVE)
# ============================================================================
# Specify year, month, and day ranges
start_year=2020
end_year=2020
start_month=2
end_month=2
start_day=1
end_day=1

# Loop through year, month, day
for year in $(seq $start_year $end_year); do
    for month in $(seq $start_month $end_month); do
        for day in $(seq $start_day $end_day); do
            # Format date as YYYY-MM-DD with zero-padding
            date=$(printf "%04d-%02d-%02d" $year $month $day)
            echo "Processing date: $date"
            python workspace/demo_combined.py --date "$date" --delete-modis
            if [ $? -ne 0 ]; then
                echo "Failed to process date: $date"
            else
                echo "Successfully processed: $date"
                python src/oco_fp_spec_anal.py --date "$date" --delete-ocofiles
            fi
            echo ""
        done
    done
done



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
#     python workspace/demo_combined.py --date "$date" --delete-modis
#     if [ $? -ne 0 ]; then
#         echo "Failed to process date: $date"
#     else
#         echo "Successfully processed: $date"
#     fi
#     echo ""
# done
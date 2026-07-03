#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_profile_pca_fit
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable

# Fit ProfilePCA (per-surface EOF compression of the sigma-grid T/q/CO2-prior
# profiles + tropopause scalars) on the FULL 2016-2020 combined parquet and write
# results/profile_pca/profile_pca_{ocean,land}.pkl.  Cheap + CPU-only: the script
# projects to only the ~56 profile/scalar columns, so ~a few GB RAM and ~1-2 min.
# Run this ONCE before curc_shell_blanca_de_profile.sh — the DE trainer auto-loads
# these pkls and hard-fails (FileNotFoundError) if they are absent.

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data

# Prepend conda's libs so h5py/netCDF4 load the conda-compiled libhdf5 (avoids the
# NC_EHDF (-101) ABI mismatch against the module-loaded system hdf5).
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:$PYTHONPATH

# On Linux the script defaults --data to combined_2016_2020_dates.parquet and
# --out-dir to results/profile_pca (both surfaces fit separately, co2-norm=mean so
# the CO2-prior EOFs are trend-invariant / year-generalisable).
python -m models.profile_pca \
    --data results/csv_collection/combined_2016_2020_dates.parquet \
    --out-dir results/profile_pca

echo "=== fitted transformers ==="
ls -la results/profile_pca/profile_pca_ocean.pkl results/profile_pca/profile_pca_land.pkl

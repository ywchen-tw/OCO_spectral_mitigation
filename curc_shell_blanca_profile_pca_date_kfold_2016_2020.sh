#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_foldpca
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --array=0-4
#SBATCH --requeue

# Fit fold-specific ProfilePCA artifacts for leakage-safe 2016-2020 date-kfold
# validation.  Each array task fits both surfaces for one held-date fold, using
# only the proper-training dates: held dates and conformal-calibration dates are
# excluded from the EOF fit.
#
# Output:
#   ${STORAGE}/results/profile_pca_date_kfold_2016_2020/fold${FOLD}/
#       profile_pca_ocean.pkl
#       profile_pca_land.pkl
#       profile_pca_ocean_explained_variance.csv
#       profile_pca_land_explained_variance.csv
#       fold_dates.json
#
# Run before:
#   curc_shell_blanca_train_structured_shared_foldpca_2016_2020.sh

set -euo pipefail

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:${LD_LIBRARY_PATH:-}
else
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

STORAGE=$(python -c "from utils import get_storage_dir; print(get_storage_dir())")
DATA="${STORAGE}/results/csv_collection/combined_2016_2020_dates.parquet"
OUT_DIR="${STORAGE}/results/profile_pca_date_kfold_2016_2020"
NFOLDS=5
FOLD=${SLURM_ARRAY_TASK_ID}

if [[ ! -f "${DATA}" ]]; then
    echo "Missing 2016-2020 dataset: ${DATA}" >&2
    exit 1
fi

echo "Fitting fold-specific ProfilePCA"
echo "fold=${FOLD}/${NFOLDS}"
echo "data=${DATA}"
echo "out_dir=${OUT_DIR}"

python -m models.profile_pca_kfold \
    --data "${DATA}" \
    --out-dir "${OUT_DIR}" \
    --n-folds "${NFOLDS}" \
    --fold "${FOLD}" \
    --calib-frac 0.15 \
    --target 10km \
    --n-components 4 \
    --co2-norm mean \
    --seed 42

echo "=== fitted fold PCA artifacts ==="
ls -la \
    "${OUT_DIR}/fold${FOLD}/profile_pca_ocean.pkl" \
    "${OUT_DIR}/fold${FOLD}/profile_pca_land.pkl" \
    "${OUT_DIR}/fold${FOLD}/fold_dates.json"

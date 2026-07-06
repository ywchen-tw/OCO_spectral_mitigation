#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_struct_foldpca
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%4
#SBATCH --requeue

# Final shared-config structured-residual run on 2016-2020 date-kfold data.
#
# Run stem:
#   de2016_2020_structured_shared_h64x32_b8_foldpca
#
# The shared config was selected from local 2020 M=3 confirmation:
#   hidden_dims = 64,32
#   block_dim   = 8
#   dropout     = 0.10
#   lr          = 0.0005
#
# This launcher uses fold-specific ProfilePCA artifacts fitted by:
#   curc_shell_blanca_profile_pca_date_kfold_2016_2020.sh
#
# Array mapping:
#   task 0,1 → fold 0 ocean,land
#   task 2,3 → fold 1 ocean,land
#   ...
#   task 8,9 → fold 4 ocean,land
#
# Aggregate after completion, for example:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_structured_dcn_ensemble/de2016_2020_structured_shared_h64x32_b8_foldpca_ocean_f*' \
#     --label structured_shared_foldpca_ocean \
#     --out results/model_comparison/de2016_2020_structured_shared_h64x32_b8_foldpca_ocean.md
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_structured_dcn_ensemble/de2016_2020_structured_shared_h64x32_b8_foldpca_land_f*' \
#     --label structured_shared_foldpca_land \
#     --out results/model_comparison/de2016_2020_structured_shared_h64x32_b8_foldpca_land.md

set -euo pipefail

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8 cuda/12.1.1
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

nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv --loop=10 \
    > "gpu_monitor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT

STORAGE=$(python -c "from utils import get_storage_dir; print(get_storage_dir())")
DATA="${STORAGE}/results/csv_collection/combined_2016_2020_dates.parquet"
PCA_ROOT="${STORAGE}/results/profile_pca_date_kfold_2016_2020"
RUN_STEM="de2016_2020_structured_shared_h64x32_b8_foldpca"
NFOLDS=5

TASK_ID=${SLURM_ARRAY_TASK_ID}
FOLD=$((TASK_ID / 2))
SURFACE_ID=$((TASK_ID % 2))

SURFACE_SPECS=(
    "0 ocean"
    "1 land"
)
read -r SFC_TYPE SURFACE <<< "${SURFACE_SPECS[SURFACE_ID]}"

PROFILE_PKL="${PCA_ROOT}/fold${FOLD}/profile_pca_${SURFACE}.pkl"
SUFFIX="${RUN_STEM}_${SURFACE}_f${FOLD}"

if [[ ! -f "${DATA}" ]]; then
    echo "Missing 2016-2020 dataset: ${DATA}" >&2
    exit 1
fi
if [[ ! -f "${PROFILE_PKL}" ]]; then
    echo "Missing fold-specific ProfilePCA: ${PROFILE_PKL}" >&2
    echo "Submit curc_shell_blanca_profile_pca_date_kfold_2016_2020.sh first." >&2
    exit 1
fi

echo "run_stem=${RUN_STEM}"
echo "surface=${SURFACE} sfc_type=${SFC_TYPE} fold=${FOLD}/${NFOLDS}"
echo "data=${DATA}"
echo "profile_pca=${PROFILE_PKL}"
echo "suffix=${SUFFIX}"

python -m models.structured_dcn_ensemble \
    --data "${DATA}" \
    --sfc_type "${SFC_TYPE}" \
    --suffix "${SUFFIX}" \
    --backbone structured_residual \
    --feature_set full \
    --profile-pca "${PROFILE_PKL}" \
    --target 10km \
    --hidden_dims 64,32 \
    --block_dim 8 \
    --loss beta_nll \
    --beta 1.0 \
    --lr 0.0005 \
    --weight_decay 0.0001 \
    --patience 50 \
    --grad_clip 1.0 \
    --n_members 5 \
    --epochs 500 \
    --batch_size 8192 \
    --norm layer \
    --dropout 0.10 \
    --near_cloud_target 0.98 \
    --mondrian_col cld_dist_km \
    --val_split date_kfold \
    --n_folds "${NFOLDS}" \
    --fold "${FOLD}" \
    --seed 42

#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_arch_full
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-39%8
#SBATCH --requeue

# Full-scale per-sounding architecture comparison on the 2016-2020 dataset:
#
#   backbones:    structured_residual, dcn_v2
#   feature sets: full, no_spec
#   surfaces:     ocean, land
#   validation:   five date-blocked folds
#
# 2 backbones x 2 feature sets x 2 surfaces x 5 folds = 40 independent jobs.
# The %8 throttle limits the array to eight concurrent GPUs.
# For an eight-run fold-0 CURC pass before launching all folds:
#   sbatch --array=0-7 curc_shell_blanca_train_structured_dcn_2016_2020.sh
#
# Both feature sets use only compact scalar columns from FeaturePipeline plus the
# profile EOF/tropopause block.  Neither consumes raw radiance spectra, neighboring
# soundings, orbit context, or retrieved examples.  "full" retains the compact
# spectral-fit summaries (k coefficients/intercepts); "no_spec" removes those too.
#
# Architecture is the only intended model change relative to the production deep
# ensemble: beta-NLL(beta=1), M=5, LayerNorm, dropout=0.1, and cloud-Mondrian
# conformal calibration are held fixed.
#
# ProfilePCA is the shared pre-fitted artifact under results/profile_pca.  Its EOF
# basis was fit on all 2016-2020 dates, so this comparison has the same mild
# unsupervised/transductive profile-basis leakage as the existing profile
# experiments. Fit fold-specific EOFs before making a strict held-date headline.
#
# Aggregate after the array finishes, for example:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_structured_dcn_ensemble/de2016_2020_structured_residual_ocean_full_prof_f*' \
#     --label structured_full \
#     --dirs 'results/model_structured_dcn_ensemble/de2016_2020_structured_residual_ocean_no_spec_prof_f*' \
#     --label structured_no_spec \
#     --dirs 'results/model_structured_dcn_ensemble/de2016_2020_dcn_v2_ocean_full_prof_f*' \
#     --label dcn_full \
#     --dirs 'results/model_structured_dcn_ensemble/de2016_2020_dcn_v2_ocean_no_spec_prof_f*' \
#     --label dcn_no_spec \
#     --out results/model_comparison/architectures_2016_2020_ocean.md
# Repeat with "land" in the directory patterns for the land table.

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
if [[ ! -f "${DATA}" ]]; then
    echo "Missing 2016-2020 dataset: ${DATA}" >&2
    exit 1
fi
for PROFILE_SURFACE in ocean land; do
    PROFILE_PKL="${STORAGE}/results/profile_pca/profile_pca_${PROFILE_SURFACE}.pkl"
    if [[ ! -f "${PROFILE_PKL}" ]]; then
        echo "Missing profile transformer: ${PROFILE_PKL}" >&2
        echo "Run curc_shell_blanca_profile_pca_fit.sh first." >&2
        exit 1
    fi
done

COMBINATIONS=(
    "structured_residual full 0 ocean"
    "structured_residual full 1 land"
    "structured_residual no_spec 0 ocean"
    "structured_residual no_spec 1 land"
    "dcn_v2 full 0 ocean"
    "dcn_v2 full 1 land"
    "dcn_v2 no_spec 0 ocean"
    "dcn_v2 no_spec 1 land"
)

NFOLDS=5
TASK_ID=${SLURM_ARRAY_TASK_ID}
NCONFIGS=${#COMBINATIONS[@]}
CONFIG_ID=$((TASK_ID % NCONFIGS))
FOLD=$((TASK_ID / NCONFIGS))

read -r BACKBONE FEATURE_SET SFC_TYPE SURFACE <<< "${COMBINATIONS[CONFIG_ID]}"
SUFFIX="de2016_2020_${BACKBONE}_${SURFACE}_${FEATURE_SET}_prof_f${FOLD}"

echo "backbone=${BACKBONE} feature_set=${FEATURE_SET} surface=${SURFACE} fold=${FOLD}"
echo "data=${DATA}"

python -m models.structured_dcn_ensemble \
    --data "${DATA}" \
    --sfc_type "${SFC_TYPE}" \
    --suffix "${SUFFIX}" \
    --backbone "${BACKBONE}" \
    --feature_set "${FEATURE_SET}" \
    --profile-pca \
    --target 10km \
    --hidden_dims 64,32 \
    --block_dim 16 \
    --cross_layers 2 \
    --cross_rank 16 \
    --loss beta_nll \
    --beta 1.0 \
    --n_members 5 \
    --batch_size 8192 \
    --norm layer \
    --dropout 0.1 \
    --near_cloud_target 0.98 \
    --mondrian_col cld_dist_km \
    --val_split date_kfold \
    --n_folds "${NFOLDS}" \
    --fold "${FOLD}" \
    --seed 42

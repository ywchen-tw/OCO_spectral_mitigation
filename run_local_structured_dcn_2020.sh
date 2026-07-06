#!/usr/bin/env bash

# Local 2020 screening for the standalone structured-residual / DCN-V2 ensemble.
#
# Default matrix:
#   2 backbones x 2 feature sets x 2 surfaces = 8 runs, date_kfold fold 0.
#
# Lightweight defaults (M=1, 30 epochs) test architecture viability before the
# full M=5, five-fold CURC experiment. Override without editing this file:
#
#   N_MEMBERS=3 EPOCHS=100 ./run_local_structured_dcn_2020.sh
#   CONFIG_IDS="0 4" ./run_local_structured_dcn_2020.sh
#   FOLD=2 CONFIG_IDS="2 6" ./run_local_structured_dcn_2020.sh
#
# Config IDs:
#   0 structured/full/ocean       4 dcn_v2/full/ocean
#   1 structured/full/land        5 dcn_v2/full/land
#   2 structured/no_spec/ocean    6 dcn_v2/no_spec/ocean
#   3 structured/no_spec/land     7 dcn_v2/no_spec/land

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH=src:${PYTHONPATH:-}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/oco_fp_matplotlib}

DATA=${DATA:-results/csv_collection/combined_2020_dates.parquet}
NFOLDS=${NFOLDS:-5}
FOLD=${FOLD:-0}
N_MEMBERS=${N_MEMBERS:-1}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-4096}
CONFIG_IDS=${CONFIG_IDS:-"0 1 2 3 4 5 6 7"}

if [[ ! -f "${DATA}" ]]; then
    echo "Missing local 2020 dataset: ${DATA}" >&2
    exit 1
fi

for PROFILE_SURFACE in ocean land; do
    PROFILE_PKL="results/profile_pca/profile_pca_${PROFILE_SURFACE}.pkl"
    if [[ ! -f "${PROFILE_PKL}" ]]; then
        echo "Missing profile transformer: ${PROFILE_PKL}" >&2
        echo "Fit or copy the profile PCA artifacts before running this screen." >&2
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

for CONFIG_ID in ${CONFIG_IDS}; do
    if ((CONFIG_ID < 0 || CONFIG_ID >= ${#COMBINATIONS[@]})); then
        echo "Invalid CONFIG_ID=${CONFIG_ID}; expected 0-7." >&2
        exit 1
    fi

    read -r BACKBONE FEATURE_SET SFC_TYPE SURFACE \
        <<< "${COMBINATIONS[CONFIG_ID]}"
    SUFFIX="local2020_${BACKBONE}_${SURFACE}_${FEATURE_SET}_prof_m${N_MEMBERS}_e${EPOCHS}_f${FOLD}"
    LOG_FILE="log_${SUFFIX}.txt"

    echo "=================================================================="
    echo "config=${CONFIG_ID} backbone=${BACKBONE} feature_set=${FEATURE_SET}"
    echo "surface=${SURFACE} fold=${FOLD} members=${N_MEMBERS} epochs=${EPOCHS}"
    echo "=================================================================="

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
        --n_members "${N_MEMBERS}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --norm layer \
        --dropout 0.1 \
        --near_cloud_target 0.98 \
        --mondrian_col cld_dist_km \
        --val_split date_kfold \
        --n_folds "${NFOLDS}" \
        --fold "${FOLD}" \
        --seed 42 \
        2>&1 | tee "${LOG_FILE}"
done

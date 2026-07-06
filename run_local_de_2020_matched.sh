#!/usr/bin/env bash

# Matched local production-MLP deep-ensemble baseline for the standalone
# structured-residual / DCN-V2 2020 screen.
#
# This holds the experiment protocol fixed:
#   data, target, date fold, full/no_spec, profile PCA, beta-NLL(beta=1),
#   member count, epochs, batch size, LayerNorm, dropout, and conformal settings.
#
# Default matrix:
#   full/no_spec x ocean/land = 4 runs, date_kfold fold 0.
#
# Override without editing:
#   N_MEMBERS=3 EPOCHS=100 ./run_local_de_2020_matched.sh
#   CONFIG_IDS="0 2" ./run_local_de_2020_matched.sh
#   FOLD=2 ./run_local_de_2020_matched.sh
#
# Config IDs:
#   0 full/ocean      2 no_spec/ocean
#   1 full/land       3 no_spec/land

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH=src:${PYTHONPATH:-}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/oco_fp_matplotlib}

PYTHON=${PYTHON:-python}
DATA=${DATA:-results/csv_collection/combined_2020_dates.parquet}
NFOLDS=${NFOLDS:-5}
FOLD=${FOLD:-0}
N_MEMBERS=${N_MEMBERS:-1}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-4096}
CONFIG_IDS=${CONFIG_IDS:-"0 1 2 3"}

if [[ ! -f "${DATA}" ]]; then
    echo "Missing local 2020 dataset: ${DATA}" >&2
    exit 1
fi

for PROFILE_SURFACE in ocean land; do
    PROFILE_PKL="results/profile_pca/profile_pca_${PROFILE_SURFACE}.pkl"
    if [[ ! -f "${PROFILE_PKL}" ]]; then
        echo "Missing profile transformer: ${PROFILE_PKL}" >&2
        exit 1
    fi
done

COMBINATIONS=(
    "full 0 ocean"
    "full 1 land"
    "no_spec 0 ocean"
    "no_spec 1 land"
)

for CONFIG_ID in ${CONFIG_IDS}; do
    if ((CONFIG_ID < 0 || CONFIG_ID >= ${#COMBINATIONS[@]})); then
        echo "Invalid CONFIG_ID=${CONFIG_ID}; expected 0-3." >&2
        exit 1
    fi

    read -r FEATURE_SET SFC_TYPE SURFACE <<< "${COMBINATIONS[CONFIG_ID]}"
    SUFFIX="matched2020_de_${SURFACE}_${FEATURE_SET}_prof_m${N_MEMBERS}_e${EPOCHS}_f${FOLD}"
    LOG_FILE="log_${SUFFIX}.txt"

    echo "=================================================================="
    echo "config=${CONFIG_ID} model=production_de feature_set=${FEATURE_SET}"
    echo "surface=${SURFACE} fold=${FOLD} members=${N_MEMBERS} epochs=${EPOCHS}"
    echo "=================================================================="

    "${PYTHON}" -m models.deep_ensemble \
        --data "${DATA}" \
        --sfc_type "${SFC_TYPE}" \
        --suffix "${SUFFIX}" \
        --feature_set "${FEATURE_SET}" \
        --profile-pca \
        --target 10km \
        --hidden_dims 64,32 \
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

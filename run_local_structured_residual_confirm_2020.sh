#!/usr/bin/env bash

# Local confirmation runs for the five selected structured-residual recipes.
#
# "Five configs" here means:
#   1. separated ocean winner
#   2. separated land winner
#   3. shared best-average recipe
#   4. shared second-best-average recipe
#   5. shared balanced-degradation recipe
#
# Because shared recipes must be evaluated on both surfaces, the default matrix
# is 8 actual runs per fold:
#   2 separated surface-specific runs + 3 shared recipes x 2 surfaces.
#
# Defaults are meant for confirmation after the M=1 tuning screen:
#   M=3, epochs=100, fold 0, full+profile.
#
# Useful overrides:
#
#   DRY_RUN=1 ./run_local_structured_residual_confirm_2020.sh
#   FOLDS="0 1" ./run_local_structured_residual_confirm_2020.sh
#   CONFIG_IDS="0 1" ./run_local_structured_residual_confirm_2020.sh
#   N_MEMBERS=5 EPOCHS=150 ./run_local_structured_residual_confirm_2020.sh
#
# Config IDs:
#   0 separated_best/ocean  h128,64 block16 dropout0.10 lr5e-4
#   1 separated_best/land   h64,32  block8  dropout0.05 lr1e-3
#   2 shared_best_avg/ocean h128,64 block16 dropout0.10 lr5e-4
#   3 shared_best_avg/land  h128,64 block16 dropout0.10 lr5e-4
#   4 shared_second/ocean   h128,64 block24 dropout0.10 lr1e-3
#   5 shared_second/land    h128,64 block24 dropout0.10 lr1e-3
#   6 shared_balanced/ocean h64,32  block8  dropout0.10 lr5e-4
#   7 shared_balanced/land  h64,32  block8  dropout0.10 lr5e-4

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH=src:${PYTHONPATH:-}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/oco_fp_matplotlib}

DATA=${DATA:-results/csv_collection/combined_2020_dates.parquet}
NFOLDS=${NFOLDS:-5}
FOLDS=${FOLDS:-0}
CONFIG_IDS=${CONFIG_IDS:-"0 1 2 3 4 5 6 7"}

FEATURE_SET=${FEATURE_SET:-full}
TARGET=${TARGET:-10km}
N_MEMBERS=${N_MEMBERS:-3}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4096}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
BETA=${BETA:-1.0}
NORM=${NORM:-layer}
PATIENCE=${PATIENCE:-50}
GRAD_CLIP=${GRAD_CLIP:-1.0}
SEED=${SEED:-42}
SKIP_DONE=${SKIP_DONE:-1}
DELETE_WEIGHTS=${DELETE_WEIGHTS:-1}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -f "${DATA}" ]]; then
    echo "Missing local 2020 dataset: ${DATA}" >&2
    exit 1
fi

for PROFILE_SURFACE in ocean land; do
    PROFILE_PKL="results/profile_pca/profile_pca_${PROFILE_SURFACE}.pkl"
    if [[ ! -f "${PROFILE_PKL}" ]]; then
        echo "Missing profile transformer: ${PROFILE_PKL}" >&2
        echo "Fit or copy the profile PCA artifacts before running this confirmation." >&2
        exit 1
    fi
done

token() {
    local value="$1"
    value="${value//,/x}"
    value="${value//./p}"
    value="${value//-/m}"
    echo "${value}"
}

delete_weights_if_done() {
    local output_dir="$1"
    local metrics_file="$2"

    if [[ "${DELETE_WEIGHTS}" != "1" ]]; then
        return
    fi
    if [[ ! -f "${metrics_file}" ]]; then
        echo "Metrics file is missing; keeping weights for inspection: ${output_dir}" >&2
        return
    fi
    if compgen -G "${output_dir}/member_*.pt" > /dev/null; then
        echo "Deleting checkpoint weights from completed confirmation run: ${output_dir}/member_*.pt"
        rm -f "${output_dir}"/member_*.pt
    fi
}

COMBINATIONS=(
    "separated_best ocean 0 128,64 16 0.10 0.0005"
    "separated_best land 1 64,32 8 0.05 0.001"
    "shared_best_avg ocean 0 128,64 16 0.10 0.0005"
    "shared_best_avg land 1 128,64 16 0.10 0.0005"
    "shared_second ocean 0 128,64 24 0.10 0.001"
    "shared_second land 1 128,64 24 0.10 0.001"
    "shared_balanced ocean 0 64,32 8 0.10 0.0005"
    "shared_balanced land 1 64,32 8 0.10 0.0005"
)

run_count=0
skip_count=0

for FOLD in ${FOLDS}; do
    for CONFIG_ID in ${CONFIG_IDS}; do
        if ((CONFIG_ID < 0 || CONFIG_ID >= ${#COMBINATIONS[@]})); then
            echo "Invalid CONFIG_ID=${CONFIG_ID}; expected 0-7." >&2
            exit 1
        fi

        read -r LABEL SURFACE SFC_TYPE HIDDEN_DIMS BLOCK_DIM DROPOUT LR \
            <<< "${COMBINATIONS[CONFIG_ID]}"

        HIDDEN_TOKEN="$(token "${HIDDEN_DIMS}")"
        DROPOUT_TOKEN="$(token "${DROPOUT}")"
        LR_TOKEN="$(token "${LR}")"
        WD_TOKEN="$(token "${WEIGHT_DECAY}")"
        BETA_TOKEN="$(token "${BETA}")"

        SUFFIX="confirm2020_structured_${LABEL}_${SURFACE}_${FEATURE_SET}_prof"
        SUFFIX="${SUFFIX}_h${HIDDEN_TOKEN}_b${BLOCK_DIM}"
        SUFFIX="${SUFFIX}_d${DROPOUT_TOKEN}_lr${LR_TOKEN}"
        SUFFIX="${SUFFIX}_wd${WD_TOKEN}_beta${BETA_TOKEN}"
        SUFFIX="${SUFFIX}_m${N_MEMBERS}_e${EPOCHS}_f${FOLD}"
        OUTPUT_DIR="results/model_structured_dcn_ensemble/${SUFFIX}"
        METRICS_FILE="${OUTPUT_DIR}/structured_residual_mondrian_date_kfold_metrics.json"
        LOG_FILE="log_${SUFFIX}.txt"

        if [[ "${SKIP_DONE}" == "1" && -f "${METRICS_FILE}" ]]; then
            echo "Skipping completed run: ${SUFFIX}"
            delete_weights_if_done "${OUTPUT_DIR}" "${METRICS_FILE}"
            skip_count=$((skip_count + 1))
            continue
        fi

        run_count=$((run_count + 1))
        echo "=================================================================="
        echo "run=${run_count} config_id=${CONFIG_ID} label=${LABEL}"
        echo "surface=${SURFACE} sfc_type=${SFC_TYPE} fold=${FOLD}"
        echo "hidden_dims=${HIDDEN_DIMS} block_dim=${BLOCK_DIM} dropout=${DROPOUT}"
        echo "lr=${LR} weight_decay=${WEIGHT_DECAY} beta=${BETA} norm=${NORM}"
        echo "members=${N_MEMBERS} epochs=${EPOCHS} batch_size=${BATCH_SIZE}"
        echo "suffix=${SUFFIX}"
        echo "=================================================================="

        CMD=(
            python -m models.structured_dcn_ensemble
            --data "${DATA}"
            --sfc_type "${SFC_TYPE}"
            --suffix "${SUFFIX}"
            --backbone structured_residual
            --feature_set "${FEATURE_SET}"
            --profile-pca
            --target "${TARGET}"
            --hidden_dims "${HIDDEN_DIMS}"
            --block_dim "${BLOCK_DIM}"
            --loss beta_nll
            --beta "${BETA}"
            --lr "${LR}"
            --weight_decay "${WEIGHT_DECAY}"
            --patience "${PATIENCE}"
            --grad_clip "${GRAD_CLIP}"
            --n_members "${N_MEMBERS}"
            --epochs "${EPOCHS}"
            --batch_size "${BATCH_SIZE}"
            --norm "${NORM}"
            --dropout "${DROPOUT}"
            --near_cloud_target 0.98
            --mondrian_col cld_dist_km
            --val_split date_kfold
            --n_folds "${NFOLDS}"
            --fold "${FOLD}"
            --seed "${SEED}"
        )

        if [[ "${DRY_RUN}" == "1" ]]; then
            printf '%q ' "${CMD[@]}"
            printf '2>&1 | tee %q\n' "${LOG_FILE}"
        else
            "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
            delete_weights_if_done "${OUTPUT_DIR}" "${METRICS_FILE}"
        fi
    done
done

echo "Completed ${run_count} run(s); skipped ${skip_count} existing run(s)."

#!/usr/bin/env bash

# Local hyperparameter grid for the structured-residual model on the 2020 data.
#
# This is intentionally separate from run_local_structured_dcn_2020.sh.  It is a
# tuning launcher, not a broad architecture screen:
#
#   backbone:    structured_residual only
#   feature set: full by default, retaining compact spectroscopy summaries
#   profiles:    ProfilePCA block enabled
#   surfaces:    ocean and land
#   validation:  date_kfold, fold 0 by default
#
# Defaults run a modest grid:
#   2 surfaces x 2 hidden layouts x 3 block widths x 2 dropout values
#   x 2 learning rates x 1 weight decay = 48 runs per fold.
#
# Useful overrides:
#
#   DRY_RUN=1 ./run_local_structured_residual_tune_2020.sh
#   FOLDS="0 1" N_MEMBERS=1 EPOCHS=80 ./run_local_structured_residual_tune_2020.sh
#   SURFACES="1:land" HIDDEN_DIMS_LIST="64,32 128,64" ./run_local_structured_residual_tune_2020.sh
#   BLOCK_DIMS="16 24" DROPOUTS="0.05" LR_LIST="0.001" ./run_local_structured_residual_tune_2020.sh
#
# Set SKIP_DONE=0 to overwrite/rerun existing output directories.
# Set DELETE_WEIGHTS=0 if you want to keep member_*.pt checkpoints for a
# specific debugging run.  By default, successful tuning runs delete model
# weights after metrics are written to reduce local storage pressure.

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH=src:${PYTHONPATH:-}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/oco_fp_matplotlib}

DATA=${DATA:-results/csv_collection/combined_2020_dates.parquet}
NFOLDS=${NFOLDS:-5}
FOLDS=${FOLDS:-0}
SURFACES=${SURFACES:-"0:ocean 1:land"}

FEATURE_SET=${FEATURE_SET:-full}
TARGET=${TARGET:-10km}
N_MEMBERS=${N_MEMBERS:-1}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4096}
SEED=${SEED:-42}
SKIP_DONE=${SKIP_DONE:-1}
DELETE_WEIGHTS=${DELETE_WEIGHTS:-1}
DRY_RUN=${DRY_RUN:-0}

HIDDEN_DIMS_LIST=${HIDDEN_DIMS_LIST:-"64,32 128,64"}
BLOCK_DIMS=${BLOCK_DIMS:-"8 16 24"}
DROPOUTS=${DROPOUTS:-"0.05 0.1"}
LR_LIST=${LR_LIST:-"0.001 0.0005"}
WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.0001"}
NORMS=${NORMS:-layer}
BETA_LIST=${BETA_LIST:-"1.0"}
PATIENCE=${PATIENCE:-50}
GRAD_CLIP=${GRAD_CLIP:-1.0}

if [[ ! -f "${DATA}" ]]; then
    echo "Missing local 2020 dataset: ${DATA}" >&2
    exit 1
fi

for PROFILE_SURFACE in ocean land; do
    PROFILE_PKL="results/profile_pca/profile_pca_${PROFILE_SURFACE}.pkl"
    if [[ ! -f "${PROFILE_PKL}" ]]; then
        echo "Missing profile transformer: ${PROFILE_PKL}" >&2
        echo "Fit or copy the profile PCA artifacts before running this tune." >&2
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
        echo "Deleting checkpoint weights from completed tuning run: ${output_dir}/member_*.pt"
        rm -f "${output_dir}"/member_*.pt
    fi
}

run_count=0
skip_count=0

for FOLD in ${FOLDS}; do
    for SURFACE_SPEC in ${SURFACES}; do
        SFC_TYPE="${SURFACE_SPEC%%:*}"
        SURFACE="${SURFACE_SPEC#*:}"

        for HIDDEN_DIMS in ${HIDDEN_DIMS_LIST}; do
            HIDDEN_TOKEN="$(token "${HIDDEN_DIMS}")"

            for BLOCK_DIM in ${BLOCK_DIMS}; do
                for DROPOUT in ${DROPOUTS}; do
                    DROPOUT_TOKEN="$(token "${DROPOUT}")"

                    for LR in ${LR_LIST}; do
                        LR_TOKEN="$(token "${LR}")"

                        for WEIGHT_DECAY in ${WEIGHT_DECAYS}; do
                            WD_TOKEN="$(token "${WEIGHT_DECAY}")"

                            for NORM in ${NORMS}; do
                                for BETA in ${BETA_LIST}; do
                                    BETA_TOKEN="$(token "${BETA}")"
                                    SUFFIX="tune2020_structured_${SURFACE}_${FEATURE_SET}_prof"
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
                                    echo "run=${run_count} surface=${SURFACE} sfc_type=${SFC_TYPE} fold=${FOLD}"
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
                        done
                    done
                done
            done
        done
    done
done

echo "Completed ${run_count} run(s); skipped ${skip_count} existing run(s)."

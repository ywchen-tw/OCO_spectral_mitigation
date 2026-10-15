#!/bin/env bash

###SBATCH --nodes=1
###SBATCH --ntasks=4
###SBATCH --ntasks-per-node=4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_de_foldpca_r15
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
###SBATCH --partition=blanca-curc-gpu
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# ── Production DE (land, 15 km target) retrained with FOLD-SPECIFIC ProfilePCA ─
# Identical config to the production run in curc_shell_blanca_de_profile_r15.sh
# (feature_set full, beta_nll beta=1.0, M=5, --norm layer --dropout 0.1,
# near_cloud_target 0.98, Mondrian cld_dist_km, date_kfold), differing ONLY by:
#   --profile-pca <fold-specific pkl> instead of the global (all-2016-2020-dates)
#   results/profile_pca/profile_pca_land.pkl.
#
# WHY: the global ProfilePCA EOF basis was fit on ALL dates, so under date_kfold
# it has seen the held-out fold — a mild unsupervised train/test leak in the
# profile block (flagged in the launcher notes and PROJECT_REVIEW).  The
# fold-specific artifacts (models.profile_pca_kfold; same split_dataframe
# date_kfold blocks, calib block excluded from the EOF fit) close it.
# Suffix carries _foldpca so nothing overwrites the current production runs:
#   de_land_beta_nll_prof_reg_r15_f${F}          (global PCA, production)
#   de_land_beta_nll_prof_reg_foldpca_r15${RV}_f${F}  (this script)
#
# PREREQUISITES:
#   - fold-specific ProfilePCA artifacts under
#     ${STORAGE}/results/profile_pca_date_kfold_2016_2020/fold{0..4}/ .
#     They exist already if the structured foldpca runs succeeded; otherwise
#     submit curc_shell_blanca_profile_pca_date_kfold_2016_2020.sh first.
#   - the combined parquet must contain xco2_bc_anomaly_r15.
#
# FOLD-ARRAY JOB: each array task is an independent 1-GPU job running ONE
# date_kfold fold (--fold $SLURM_ARRAY_TASK_ID); --requeue re-submits a fold if
# the preemptable QOS preempts it.
#
# After ALL array tasks finish, aggregate (no GPU needed):
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_deep_ensemble/de_land_beta_nll_prof_reg_foldpca_r15_v2_f*' \
#     --label DeepEns+foldpca+r15 \
#     --out results/model_comparison/deep_ensemble_land_profile_foldpca_r15_kfold_agg.md

set -euo pipefail

# Modules one by one, data-env python as the fallback (see curc_env.sh).
source /projects/yuch8913/OCO_spectral_mitigation/curc_env.sh
curc_env --torch cuda/12.1.1 || exit 1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

F=${SLURM_ARRAY_TASK_ID}
NFOLDS=5

# Run version (log/PIPELINE_FIX_RERUN_PLAN_2026-10-14.md): v2 = unsmoothed gamma,
# fill/spike imputation, RobustClipScaler, no t700.  The tag sits BEFORE the fold
# tag, so v1 globs (..._f*) never match v2 dirs and v2 runs never overwrite v1
# checkpoints.  RUN_VERSION= (empty) reproduces the v1 names.
export RUN_VERSION=${RUN_VERSION-v2}
RV=${RUN_VERSION:+_${RUN_VERSION}}

STORAGE=$(python -c "from utils import get_storage_dir; print(get_storage_dir())")
PCA_ROOT="${STORAGE}/results/profile_pca_date_kfold_2016_2020"
PROFILE_PKL="${PCA_ROOT}/fold${F}/profile_pca_land.pkl"

if [[ ! -f "${PROFILE_PKL}" ]]; then
    echo "Missing fold-specific ProfilePCA: ${PROFILE_PKL}" >&2
    echo "Submit curc_shell_blanca_profile_pca_date_kfold_2016_2020.sh first." >&2
    exit 1
fi
echo "fold=${F}/${NFOLDS}  profile_pca=${PROFILE_PKL}"

# ── v2 rerun (2026-10-14): every arm of this fold, one after another ────────────
# One array task trains, for fold F: production `full`, the eleven feature-set
# ablations and the ML(raw) arm (v2: 5-30 min each, ~2.5 h in all, inside --time).
# no_fpcloud / no_inhom / no_screen drop one contamination subgroup each
# (split by origin 2026-10-14; see CG_* in src/models/pipeline.py), and the
# *_and_xco2 arms (2026-10-15) drop the same subgroup together with dXCO2.
# All arms share the production config and differ ONLY in feature set, target
# and suffix (log/PIPELINE_FIX_RERUN_PLAN_2026-10-14.md).  An arm whose
# run_summary.json exists is skipped, so a --requeue restart resumes instead of
# retraining finished arms with reshuffled seeds.  Narrow the set with e.g.
#   sbatch --export=ALL,ARMS="full raw" <this script>
#
# Regularization note (2026-07-15): without --norm layer --dropout 0.1, LAND fold
# f2 diverged for every ablation arm (held RMSE 3.8k-43k ppm) and f1/f3
# under-converged, the beta-NLL land instability that lndo01 fixed for
# production; so every arm carries the production regularization.  A diverged
# fold is rerun alone:  sbatch --array=<fold> --export=ALL,ARMS=<arm> after
# adding --seed 43 to train_arm.
#
# ML(raw) differs from production ML(bc) ONLY in the target
# (xco2_raw_anomaly_r15 instead of xco2_bc_anomaly_r15).  Its TCCON tree
# is rebuilt with  bash workspace/build_ablation_variant_trees.sh raw_base.
train_arm() {   # train_arm <suffix> <feature_set> <target>
    local suffix=$1 fs=$2 target=$3
    if [[ -f "${STORAGE}/results/model_deep_ensemble/${suffix}/run_summary.json" ]]; then
        echo "already finished: ${suffix}"
        return 0
    fi
    echo "=== ${suffix}  feature_set=${fs}  target=${target}  $(date)"
    python -m models.deep_ensemble --sfc_type 1 --suffix "${suffix}" \
        --profile-pca "${PROFILE_PKL}" --feature_set "${fs}" --target "${target}" \
        --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
        --norm layer --dropout 0.1 \
        --near_cloud_target 0.98 --mondrian_col cld_dist_km \
        --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
}

ARMS=${ARMS:-"full no_xco2 no_spec no_xco2_and_spec no_contam no_contam_and_xco2 no_fpcloud no_inhom no_screen no_fpcloud_and_xco2 no_inhom_and_xco2 no_screen_and_xco2 raw"}
for ARM in ${ARMS}; do
    case ${ARM} in
        full) train_arm de_land_beta_nll_prof_reg_foldpca_r15${RV}_f${F} full 15km ;;
        raw)  train_arm de_land_beta_nll_prof_reg_raw_foldpca_r15${RV}_f${F} full xco2_raw_anomaly_r15 ;;
        *)    train_arm de_land_${ARM}_prof_foldpca_r15${RV}_f${F} "${ARM}" 15km ;;
    esac
done

kill $GPU_MONITOR_PID 2>/dev/null || true

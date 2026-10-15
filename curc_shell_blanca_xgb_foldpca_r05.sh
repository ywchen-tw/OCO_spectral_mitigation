#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_xgb_foldpca_r05
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --array=0-4
#SBATCH --requeue

# ── Plain XGBoost mean regression (OCEAN, 5 km target) — SAME protocol as the
# production DE, for the reviewer-proof baseline table
# (log/archive/LINREG_XGB_BASELINE_PLAN_2026-07-07.md).
# Mirrors curc_shell_blanca_de_profile_foldpca_r05.sh except:
#   - CPU-only (tree_method=hist; no --gres=gpu)
#   - python -m models.gbdt_baselines --objective mean (ONE reg:squarederror model,
#     prediction = mu; early stopping on the calib block only — NEVER TCCON)
# Fold-specific ProfilePCA (leakage-safe) — SAME artifacts the foldpca DE uses, so
# this is a same-protocol comparison partner to
#   de_ocean_beta_nll_prof_reg_foldpca_r05${RV}_f${F}.
# Suffix: xgb_ocean_full_prof_foldpca_r05${RV}_f${F}
#
# THREADS/DETERMINISM: --n_jobs ${SLURM_NTASKS} uses the full CPU allocation so the
# mean regressor finishes well inside the wall (15M rows × hist). With n_jobs>1 the
# hist builder is NOT bit-reproducible; seeds are still fixed and the xgboost version
# is recorded in run_summary.json (per the plan's determinism note). Pass --n_jobs 1
# for a strictly reproducible (slower) rerun.
#
# PREREQUISITES:
#   - fold-specific ProfilePCA under ${STORAGE}/results/profile_pca_date_kfold_2016_2020/fold{0..4}/ .
#   - the combined parquet must contain xco2_bc_anomaly_r05.
#
# After ALL array tasks finish, aggregate:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_gbdt/xgb_ocean_full_prof_foldpca_r05_v2_f*' \
#     --label XGB+foldpca+r05 \
#     --out results/model_comparison/xgb_ocean_foldpca_r05_kfold_agg.md

set -euo pipefail

# Modules one by one, data-env python as the fallback (see curc_env.sh).
source /projects/yuch8913/OCO_spectral_mitigation/curc_env.sh
curc_env || exit 1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-16}
export MKL_NUM_THREADS=${SLURM_NTASKS:-16}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

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
PROFILE_PKL="${PCA_ROOT}/fold${F}/profile_pca_ocean.pkl"

if [[ ! -f "${PROFILE_PKL}" ]]; then
    echo "Missing fold-specific ProfilePCA: ${PROFILE_PKL}" >&2
    echo "Submit curc_shell_blanca_profile_pca_date_kfold_2016_2020.sh first." >&2
    exit 1
fi
echo "fold=${F}/${NFOLDS}  profile_pca=${PROFILE_PKL}"

python -m models.gbdt_baselines --model xgboost --objective mean \
    --sfc_type 0 --suffix xgb_ocean_full_prof_foldpca_r05${RV}_f${F} \
    --profile-pca "${PROFILE_PKL}" \
    --feature_set full --target 5km --calib_frac 0.15 --n_jobs ${SLURM_NTASKS:-16} \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_feat_importance
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --array=0-9
#SBATCH --requeue

# ── Permutation feature importance for the manuscript model trio (DE / XGB-mean
# / Ridge) on the FULL date_kfold held folds (models.feature_importance).
#
# Array task = surface × fold:  0-4 → ocean folds 0-4,  5-9 → land folds 0-4.
# Each task runs all three models on that fold's full held dates (~1.6-2M rows)
# so the same permutation indices are shared across models (seeded by
# fold/stratum/repeat — cross-model differences are never shuffle noise).
#
# Gates (task FAILS loudly rather than emit wrong numbers):
#   1. the three fold pipelines expose identical feature lists;
#   2. reconstructed held RMSE reproduces run_summary.json per model (rel 2e-3).
#
# PREREQUISITES:
#   - fold dirs de_/xgb_/linreg_{ocean|land}_*_foldpca_{r05|r15}_f{0..4} under
#     ${STORAGE}/results/ (model weights + pipeline pkl + training_dates.json);
#   - the combined parquet with xco2_bc_anomaly_r05/_r15.
#
# After ALL array tasks finish, aggregate into the 3-model comparison table
# (cheap, run locally or on a login node):
#   PYTHONPATH=src python -m models.feature_importance --surface ocean --aggregate --run-version v2
#   PYTHONPATH=src python -m models.feature_importance --surface land  --aggregate --run-version v2
# → results/model_comparison/feature_importance/<surface>/
#     importance_{model}_{surface}_agg.csv + FEATURE_IMPORTANCE_TABLE_<surface>.md

set -euo pipefail

# Modules one by one, data-env python as the fallback (see curc_env.sh).
source /projects/yuch8913/OCO_spectral_mitigation/curc_env.sh
curc_env || exit 1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-8}
export MKL_NUM_THREADS=${SLURM_NTASKS:-8}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

T=${SLURM_ARRAY_TASK_ID}
if (( T < 5 )); then
    SURFACE=ocean; F=${T}
else
    SURFACE=land;  F=$(( T - 5 ))
fi
echo "surface=${SURFACE}  fold=${F}"

# Full held fold (--n-rows 0), 5 shuffle repeats for tight fold-level numbers.
# Run version (log/PIPELINE_FIX_RERUN_PLAN_2026-10-14.md): v2 = unsmoothed gamma,
# fill/spike imputation, RobustClipScaler, no t700.  The tag sits BEFORE the fold
# tag, so v1 globs (..._f*) never match v2 dirs and v2 runs never overwrite v1
# checkpoints.  RUN_VERSION= (empty) reproduces the v1 names.
export RUN_VERSION=${RUN_VERSION-v2}
RV=${RUN_VERSION:+_${RUN_VERSION}}

python -m models.feature_importance \
    --surface ${SURFACE} --fold ${F} \
    --run-version "${RUN_VERSION}" \
    --models de,xgb,linreg \
    --n-rows 0 --n-repeats 5

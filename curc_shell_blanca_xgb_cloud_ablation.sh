#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_xgb_cloud
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --array=0-9
#SBATCH --requeue

# XGBOOST NEAR-CLOUD CLASSIFIER — FINAL config, OOD (date_kfold, full 66-date set).
# A SINGLE XGBoost per fold (no ensemble) → CPU-only, no GPU requested (cheap).
#
# This is the FINAL deployable cloud classifier:
#   * --cloud_features: the deployable cloud-diagnostic block (fit-quality chi2/rms,
#     scattering dp, cloud aerosols aod_ice/water, brightness alb/snr, continuum,
#     glint).  Local lift: ocean OOD 0.69->0.76; land unchanged (already at ceiling).
#   * PER-SURFACE thresholds (Phase 2f/2g): ocean --near_cloud_km 5.0, land 15.0.
#     From the 1km bias-decay (full 66-date) + predictability: ocean bias gone by
#     ~7km & classifier best at <5km -> 5km; land bias persists to ~15km & AUC is
#     threshold-flat -> 15km.  NOT a shared 10km.
#   * regularized config (400 trees, depth 4, strong reg) = best OOD locally.
# cld_dist_km is the LABEL only (never a feature).
#
# Suffix 'xgbcloud_final_*' is distinct from the earlier base (xgbcloud_*) and
# fixed-10km expanded (xgbcloud_exp_*) runs, so all are preserved for comparison.
# Local OOD references: ocean@5km 0.819, land@15km 0.839; base full-66: ocean 0.720
# / land 0.845; exp-10km full-66: ocean 0.781 / land 0.850.
#
# 10 array tasks = {ocean, land} x fold{0..4}, each an independent CPU job:
#   fold    = idx % 5
#   surface = idx / 5            (0 = ocean, 1 = land)
#
# After all finish, average AUC across folds per surface:
#   for S in ocean land; do
#     PYTHONPATH=src python -m models.aggregate_folds \
#       --dirs "results/model_xgb_cloud/xgbcloud_final_${S}_f*" --label "XGBcloud_final_${S}" \
#       --out "results/model_comparison/xgbcloud_final_${S}_kfold_agg.md"
#   done
# (or just read cloud_auc from each run_summary.json)

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8 cuda/12.1.1
conda activate data

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-8}
export MKL_NUM_THREADS=${SLURM_NTASKS:-8}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:$PYTHONPATH

NFOLDS=5
SFC_IDS=(0 1)
SFC_NAMES=(ocean land)
# FINAL per-surface gate thresholds (Phase 2f/2g, from the 1km bias-decay on full
# 66-date data + predictability): ocean bias is gone by ~7km and the classifier is
# most accurate at <5km -> 5km; land bias persists to ~15km (still +0.04 at 14-15km)
# and the AUC is threshold-flat -> 15km.  NOT a shared 10km.
SFC_NEAR_KM=(5.0 15.0)

T=${SLURM_ARRAY_TASK_ID}
F=$((T % 5))
SI=$((T / 5))
SFC=${SFC_IDS[$SI]}
SNAME=${SFC_NAMES[$SI]}
NEAR_KM=${SFC_NEAR_KM[$SI]}

echo "[task ${T}] surface=${SNAME} fold=${F} near_cloud_km=${NEAR_KM}"

python -m models.xgb_cloud_classifier --sfc_type ${SFC} --suffix xgbcloud_final_${SNAME}_f${F} \
  --val_split date_kfold --n_folds ${NFOLDS} --fold ${F} --near_cloud_km ${NEAR_KM} --seed 42 \
  --cloud_features

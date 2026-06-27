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

# XGBOOST CLOUD<10km CLASSIFIER — OUT-OF-DISTRIBUTION (date_kfold, full 66-date set).
# A SINGLE XGBoost per fold (no ensemble) → CPU-only, no GPU requested (cheap).
#
# Question: the local 12-date probe showed the cloud classifier's stellar
# in-distribution AUC (~0.99 land / 0.95 ocean) is mostly spatial/temporal
# autocorrelation leakage and COLLAPSES on unseen dates (~0.84 land / ~0.69 ocean,
# regularized).  That was with only ~9 train dates/fold.  Does the full 66-date set
# (many more train dates/fold) recover OOD AUC by letting the model learn
# date-INVARIANT cloud signatures?  This is the honest cloud-prediction number.
#
# Defaults use the regularized config (400 trees, depth 4, strong reg) that
# generalized best OOD locally.  cld_dist_km is the LABEL only (never a feature).
#
# --cloud_features adds the deployable cloud-diagnostic block (fit-quality chi2/rms,
# scattering dp, cloud aerosols aod_ice/water, brightness alb/snr, continuum, glint).
# Local 12-date OOD: lifts OCEAN 0.69->0.76; LAND unchanged (already at ceiling).
# Suffix 'xgbcloud_exp_*' keeps these separate from the base run (xgbcloud_*) so the
# full-66-date BASE-vs-EXPANDED comparison is preserved.
#
# 10 array tasks = {ocean, land} x fold{0..4}, each an independent CPU job:
#   fold    = idx % 5
#   surface = idx / 5            (0 = ocean, 1 = land)
#
# After all finish, average AUC across folds per surface (compare to base full-66 OOD:
# land 0.845 / ocean 0.720, and local expanded ocean 0.76):
#   for S in ocean land; do
#     PYTHONPATH=src python -m models.aggregate_folds \
#       --dirs "results/model_xgb_cloud/xgbcloud_exp_${S}_f*" --label "XGBcloud_exp_${S}" \
#       --out "results/model_comparison/xgbcloud_exp_${S}_kfold_agg.md"
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

T=${SLURM_ARRAY_TASK_ID}
F=$((T % 5))
SI=$((T / 5))
SFC=${SFC_IDS[$SI]}
SNAME=${SFC_NAMES[$SI]}

echo "[task ${T}] surface=${SNAME} fold=${F}"

python -m models.xgb_cloud_classifier --sfc_type ${SFC} --suffix xgbcloud_exp_${SNAME}_f${F} \
  --val_split date_kfold --n_folds ${NFOLDS} --fold ${F} --near_cloud_km 10.0 --seed 42 \
  --cloud_features

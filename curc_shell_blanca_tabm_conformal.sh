#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_tabm_conformal
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# CONFORMALIZED TabM (production) — closes TabM's only weakness vs the deep ensemble:
# interval calibration.  TabM's raw pinball quantiles under-cover (~0.88); --conformal
# adds a post-hoc CQR layer (carve a calib block from train, train on the rest, shift
# the quantiles to hit nominal 0.90) — mirroring DE's conformal setup so the two are
# directly comparable.  Validated locally: raw cov 0.868 -> CQR 0.900, near-cloud
# 0.852 -> Mondrian 0.950, crossing 0.
#
# DEFAULT TabM structure (--K 16, no --config): the date_kfold HPO showed tuning does
# NOT beat the default (flat landscape).  Mondrian-CQR is binned by cld_dist_km with
# --near_cloud_target 0.95 to over-cover the cloud-contaminated tail (same lever as DE).
#
# Each run writes THREE interval sets (point preds identical, intervals differ):
#   tabm_raw_*       — model's own quantiles (uncalibrated reference)
#   tabm_cqr_*       — global CQR
#   tabm_mondrian_*  — regime-conditional CQR (the HEADLINE; matches de_mondrian_*)
# Suffix: tabm_{surface}_{fs}_conf_f{F}.  One fold per array task; both surfaces x both
# feature sets per task.
#
# NOTE: --conformal carves ~15% of train for calibration, so the model trains on ~85%
# of what the non-conformal tabm_*_f* runs used; expect point R2 a hair lower, intervals
# properly calibrated.  Compare the headline tabm_mondrian_* vs de_mondrian_* (cov90 +
# R2) to pick the production model per surface.
#
# After all folds finish, aggregate the headline (Mondrian) set vs the MATCHED DE
# baseline.  OCEAN -> de_ocean_full_contam (snow N/A).  LAND -> de_land_fc_snowdata
# (snow-INCLUDED, since the TabM land runs include snow; de_land_full_contam is
# snow-excluded and would be an unfair, different-holdout comparison):
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_tabm/tabm_ocean_full_contam_conf_f*'      --label TabM_conf \
#     --dirs 'results/model_deep_ensemble/de_ocean_full_contam_f*'    --label DE \
#     --out results/model_comparison/tabm_conf_vs_de_ocean.md
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_tabm/tabm_land_full_contam_conf_f*'       --label TabM_conf \
#     --dirs 'results/model_deep_ensemble/de_land_fc_snowdata_f*'     --label DE_snow \
#     --out results/model_comparison/tabm_conf_vs_de_land.md

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8 cuda/12.1.1
conda activate data

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:$PYTHONPATH

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv &
GPU_MONITOR_PID=$!

F=${SLURM_ARRAY_TASK_ID}
NFOLDS=5
CONF="--conformal --calib_frac 0.15 --mondrian_col cld_dist_km --near_cloud_target 0.95"

for FS in full full_contam; do
  # Ocean: no snow exists on ocean (all snow is sfc_type=1), so no --include_snow.
  python -m models.tabm --sfc_type 0 --suffix tabm_ocean_${FS}_conf_f${F} --K 16 \
    --feature_set ${FS} ${CONF} --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

  # Land: --include_snow adopts the snow-experiment arm-B win (free high-lat coverage,
  # no cost to the bulk; the snow_flag FEATURE was neutral so we keep feature_set ${FS},
  # NOT full_contam_snow).  Compare these against the snow-INCLUDED DE baseline
  # de_land_fc_snowdata_f* (NOT de_land_full_contam, which is snow-excluded).
  python -m models.tabm --sfc_type 1 --suffix tabm_land_${FS}_conf_f${F} --K 16 \
    --feature_set ${FS} --include_snow ${CONF} --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
done

kill $GPU_MONITOR_PID 2>/dev/null || true

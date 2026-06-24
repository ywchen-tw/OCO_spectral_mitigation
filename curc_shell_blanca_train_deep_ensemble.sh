#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_deep_ensemble
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# FOLD-ARRAY JOB.  Each array task is an INDEPENDENT 1-GPU job that runs ONE
# date_kfold fold (--fold $SLURM_ARRAY_TASK_ID).  This is the same per-task GPU
# ask as a single job (gpu:1) — NOT a larger request — so per-task queue wait is
# unchanged; the scheduler backfills the 5 tasks as GPUs free up (parallel when
# available, graceful when scarce).  --requeue auto-resubmits a fold if the
# preemptable QOS preempts it (only that fold is lost, not the whole run).
#
# After ALL array tasks finish, aggregate (no GPU needed):
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_deep_ensemble/de_ocean_f*' --label DeepEns \
#     --out results/model_comparison/deep_ensemble_ocean_kfold_agg.md
# (aggregate_folds picks the alphabetically-first metrics json = de_mondrian_*.)

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
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

# Deep-ensemble MLP (M Gaussian-NLL members) + conformal calibration.  Each run
# writes THREE interval sets sharing the same point predictions (de_raw / de_split /
# de_mondrian; identical RMSE/R², only intervals differ).  Mondrian is binned by
# cld_dist_km (physical proxy for the cloud-contaminated tail), NOT predicted mu:
# the local check showed mu-deciles do not isolate the y-defined tail.
# M=3 (local check: M=3 already gives global cov90≈0.88) and batch 8192 cut GPU time.
#
# NOTE: under date-blocking the calibration block is different dates from the held
# fold, so conformal coverage is approximate (not exchangeable).  On the 12-date
# local check it held (~0.88); watch de_*_cov90 here.

F=${SLURM_ARRAY_TASK_ID}
NFOLDS=5

# ── Ocean (sfc 0) ─────────────────────────────────────────────────────────────
# python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_f${F} \
#   --n_members 3 --batch_size 8192 --mondrian_col cld_dist_km \
#   --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Land (sfc 1): uncomment to also run land in the same array task ───────────
python -m models.deep_ensemble --sfc_type 1 --suffix de_land_f${F} \
  --n_members 3 --batch_size 8192 --mondrian_col cld_dist_km \
  --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

kill $GPU_MONITOR_PID 2>/dev/null || true

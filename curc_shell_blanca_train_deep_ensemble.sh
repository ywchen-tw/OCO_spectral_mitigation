#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_train_deep_ensemble
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1


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

# Deep-ensemble MLP (M Gaussian-NLL members) + conformal calibration.
# The accuracy-leading MLP ties TabM but has no intervals; this gives it intervals
# and recalibrates them.  Each run writes THREE metric sets sharing the same point
# predictions (identical RMSE/R²; only intervals differ):
#   de_raw_*       — raw Gaussian-mixture 90% interval
#   de_split_*     — global split conformal
#   de_mondrian_*  — regime-conditional (predicted-mu decile) conformal  ← headline
#
# NOTE on conformal under date-blocking: the calibration block is carved from TRAIN
# dates, so it is NOT the same dates as the held fold → calib/test are not strictly
# exchangeable and coverage is approximate (mild on many-date data, watch de_*_cov90).

# ── Block-rotation k-fold over dates, ocean (M=5) ─────────────────────────────
NFOLDS=5
for F in $(seq 0 $((NFOLDS-1))); do
  python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_f${F} \
    --n_members 5 --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
done

# Aggregate (compare to TabM/MLP/XGB k-fold).  Pull the de_mondrian_* metric per fold:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_deep_ensemble/de_ocean_f*' --label DeepEns \
#     --out results/model_comparison/deep_ensemble_ocean_kfold_agg.md
# (aggregate_folds picks the alphabetically-first metrics json: de_mondrian_*; if you
#  want raw/split instead, move/rename or aggregate by hand.)

# ── Land (sfc 1): uncomment to repeat ─────────────────────────────────────────
# NFOLDS=5
# for F in $(seq 0 $((NFOLDS-1))); do
#   python -m models.deep_ensemble --sfc_type 1 --suffix de_land_f${F} \
#     --n_members 5 --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
# done

kill $GPU_MONITOR_PID 2>/dev/null || true

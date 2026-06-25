#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
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
pip install tabulate

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

# Each array task runs BOTH surfaces × BOTH losses sequentially for its fold
# (gpu:1, ~4 runs ≈ 3h, under the 8h wall).  This is the loss ABLATION:
#   gaussian_nll = baseline;  beta_nll (Seitzer) with beta=1.0 = best of a local
#   beta sweep {0,0.3,0.5,0.7,1.0,1.5,2.0} on fold-0 ocean (250ep/3members).
# Plain NLL hides hard near-cloud points behind sigma-inflation instead of
# fitting mu; beta_nll re-weights by stop_grad(var)^beta to restore mu-fitting.
# Because conformal owns calibration, push beta to the point-accuracy optimum:
# beta=1.0 (pure MSE on mu) peaked EVERY metric vs gaussian — global R2 +0.124,
# near-cloud R2 +0.163, near&bottom_5pct R2 +1.40 (still <0 = data floor),
# tail-5% coverage +0.061, 0-2km correction +16pts, global cov held ~0.88.
# beta>1 overfits irreducible noise (R2 & coverage drop) → 1.0 is the ceiling.
# All CSVs carry the near_cloud_tail crossed regime + de_correction_clddist.
# Suffix encodes loss: de_{surface}_{loss}_f{F}.  (Drop the gaussian_nll line to
# run beta-only and halve the cost.)

for LOSS in gaussian_nll beta_nll; do
#   python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_${LOSS}_f${F} \
#     --loss ${LOSS} --beta 1.0 --n_members 3 --batch_size 8192 \
#     --mondrian_col cld_dist_km --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

  python -m models.deep_ensemble --sfc_type 1 --suffix de_land_${LOSS}_f${F} \
    --loss ${LOSS} --beta 1.0 --n_members 3 --batch_size 8192 \
    --mondrian_col cld_dist_km --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
done

kill $GPU_MONITOR_PID 2>/dev/null || true

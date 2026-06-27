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
# pip install tabulate

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
# M=5 members (M=3 already gave global cov90≈0.88; bumped to 5 for accuracy/epistemic
# headroom — ~5/3x member train time, still well under the 16h wall) and batch 8192.
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

# --near_cloud_target raises the conformal target in the near-cloud (<=10km)
# Mondrian bins to over-cover the outcome-defined near-cloud tail (far bins stay
# 0.90, so far intervals stay tight).  Validated under date_kfold (fold-0 ocean):
# near&tail-5% coverage 0.70 -> 0.865 at +51% near-cloud width; 0.975 lands ~0.865
# under date-shift, so 0.98 is used to push the tail nearer 0.90.  Applied to the
# production loss (beta_nll); gaussian stays flat as the clean coverage reference.
for LOSS in gaussian_nll beta_nll; do
  NCT=""; [ "${LOSS}" = "beta_nll" ] && NCT="--near_cloud_target 0.98"
  python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_${LOSS}_f${F} \
    --loss ${LOSS} --beta 1.0 --n_members 5 --batch_size 8192 ${NCT} \
    --mondrian_col cld_dist_km --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

  python -m models.deep_ensemble --sfc_type 1 --suffix de_land_${LOSS}_f${F} \
    --loss ${LOSS} --beta 1.0 --n_members 5 --batch_size 8192 ${NCT} \
    --mondrian_col cld_dist_km --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
done

# ── Feature-set ablations ─────────────────────────────────────────────────────
# Drop XCO2-related features (no_xco2) and spectral features (no_spec) to isolate
# their contribution.  Run on the PRODUCTION config (beta_nll + near_cloud_target),
# ocean, one fold per array task — mirrors the de_ocean_beta_nll_f${F} baseline so
# the only difference is the feature set.  Suffix: de_ocean_{feature_set}_f{F}.
for FS in no_xco2 no_spec no_xco2_and_spec; do
  python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_${FS}_f${F} \
    --feature_set ${FS} --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
    --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

  python -m models.deep_ensemble --sfc_type 1 --suffix de_land_${FS}_f${F} \
    --feature_set ${FS} --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
    --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
done

kill $GPU_MONITOR_PID 2>/dev/null || true

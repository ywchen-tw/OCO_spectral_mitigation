#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_tabm_full_contam
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# TabM x feature-set comparison: does TabM or the Deep Ensemble gain more from the
# cloud/aerosol contamination features (full_contam)?  This script produces the TABM
# half of that comparison; the DEEP-ENSEMBLE half already exists on CURC:
#     full        -> de_{ocean,land}_beta_nll_f{F}        (production baseline)
#     full_contam -> de_{ocean,land}_full_contam_f{F}
# Local 2020 smoke test (date_kfold fold 0, MPS) confirmed the path + that full_contam
# resolves through TabM (sfc_type threaded -> add_per_sfc fires).  DE gain on the full
# 2016-2020 set was +0.040 R2 (ocean) / +0.094 (land); this measures TabM's gain head
# to head on the SAME date_kfold protocol.
#
# FOLD-ARRAY JOB.  Each array task is an INDEPENDENT 1-GPU job running ONE date_kfold
# fold (--fold $SLURM_ARRAY_TASK_ID) of all 4 TabM configs (2 surfaces x 2 feature
# sets), ~4 runs ≈ 3h on a real GPU, under the 16h wall.  --requeue auto-resubmits a
# fold if the preemptable QOS preempts it.
#
# Each model fits its OWN FeaturePipeline on the TRAIN split only (leakage discipline)
# — do NOT pass a pre-fitted full-data --pipeline for blocked splits.  TabM stays at
# its production config (K=16, huber loss); the only thing varied is --feature_set, so
# the full->full_contam delta is the clean feature contribution.  Comparing TabM-huber
# vs DE-beta_nll is the intended "which architecture" question; point metrics (R2/RMSE)
# are loss-agnostic and directly comparable.
#
# After ALL array tasks finish, aggregate the folds (no GPU needed) and build the
# TabM-vs-DE markdown.  aggregate_folds picks the alphabetically-first metrics json:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_tabm/tabm_ocean_full_f*'         --label TabM_full \
#     --dirs 'results/model_tabm/tabm_ocean_full_contam_f*'  --label TabM_contam \
#     --dirs 'results/model_deep_ensemble/de_ocean_beta_nll_f*'    --label DE_full \
#     --dirs 'results/model_deep_ensemble/de_ocean_full_contam_f*' --label DE_contam \
#     --out results/model_comparison/tabm_vs_de_full_contam_ocean.md
#   # repeat with land/sfc1 dirs for the land table.

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8 cuda/12.1.1
conda activate data

# Prepend conda's libs so Python's netCDF4/h5py loads the conda-compiled libhdf5
# rather than the system one injected by `module load hdf5/...` (ABI mismatch ->
# NC_EHDF -101 at H5Fopen).  $CONDA_PREFIX may be empty in the SLURM batch context,
# so hardcode the path on Linux as a reliable fallback.
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_THREADING_LAYER=GNU

# Run from the repo ROOT (get_storage_dir() is cwd-relative -> './results/...'),
# src on PYTHONPATH.  TabM mixes relative + absolute imports, so launch as a package
# module: `python -m models.tabm`.
cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:$PYTHONPATH

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv &
GPU_MONITOR_PID=$!

F=${SLURM_ARRAY_TASK_ID}
NFOLDS=5

# Tuned hyperparameters from the local random-search HPO (winner t09, R²=0.914 of
# 28 trials incl. a 5-trial refinement; see results/model_comparison/hpo_ocean_*
# _trials.csv).  Written to tabm_tuned_ocean.json (repo root), passed via --config;
# DO NOT also pass --K (the flag would override the tuned K).
#   WHAT TRANSFERS as-is: K=8, d_model=192, n_layers=3, dropout, weight_decay,
#     huber_delta=0.63 — architecture/loss/regularization are data-scale-invariant.
#   WHAT IS RESCALED for production: the HPO ran on a 74k-row single-date proxy with
#     batch=1024; the full 2016-2020 set is ~10.4M ocean / 6.8M land rows, where
#     batch=1024 is ~8k steps/epoch (infeasible).  The config's linux_* block uses
#     batch=8192 (codebase default for this data).  The tuned lr=3.8e-3 is KEPT (not
#     up-scaled): a larger batch lowers gradient noise, so the tuned lr is safe-to-
#     conservative there — avoids the lr/big-model divergence that NaN'd one HPO trial.
#   OneCycle re-derives its schedule from total_steps (= linux_epochs x steps/epoch).
# NOTE: tuned on OCEAN — land reuses it as an informed default until its own HPO sweep.
TUNED=tabm_tuned_ocean.json

# This fold, all 4 TabM configs: {ocean, land} x {full, full_contam}.
for FS in full; do
  python -m models.tabm --sfc_type 0 --suffix tabm_ocean_${FS}_f${F} --config ${TUNED} \
    --feature_set ${FS} --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

  python -m models.tabm --sfc_type 1 --suffix tabm_land_${FS}_f${F} --config ${TUNED} \
    --feature_set ${FS} --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
done

kill $GPU_MONITOR_PID 2>/dev/null || true

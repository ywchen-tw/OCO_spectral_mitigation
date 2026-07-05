#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_tabm
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# ── TabM (BatchEnsemble MLP) WITH the vertical-profile EOF block ──────────────
# TabM companion to curc_shell_blanca_de_profile.sh: SAME matrix — full +
# no_xco2 + no_spec + no_xco2_and_spec, BOTH surfaces (ocean sfc 0, land sfc 1) —
# on the 10 km clear-sky reference (default xco2_bc_anomaly), all with the profile
# EOF block appended.  Per fold: 2 (full) + 3 ablations x 2 surfaces = 8 runs.
#
# TabM's CLI is NOT identical to deep_ensemble's, so the DE-only flags are dropped:
#   * TabM has NO --loss beta_nll (only huber/quantile), NO --norm/--dropout,
#     NO --n_members/--batch_size, NO --hidden_dims (K + config set the model).
#     Ensemble size is --K 16 (production TabM); loss is the huber default.
#   * --near_cloud_target is ONLY read under --conformal (it is a silent no-op
#     otherwise), so --conformal is passed to mirror the DE near-cloud tail
#     over-coverage.  This changes intervals only; the point-prediction R^2 that
#     the feature ablations compare is unaffected.
# Kept (all supported by models.tabm): --profile-pca, --feature_set, --target,
#   --val_split date_kfold/--n_folds/--fold, --near_cloud_target/--mondrian_col.
#
# Suffix carries a _prof tag (no _reg, since TabM has no reg-ablation flags) so
# results land in results/model_tabm/tabm_{surface}[_{FS}]_prof_f${F}.
#
# PREREQUISITE: run curc_shell_blanca_profile_pca_fit.sh first so the ProfilePCA
# pkls exist (results/profile_pca/profile_pca_<surface>.pkl), else the pipeline's
# FeaturePipeline.fit raises FileNotFoundError and the fold dies.
#
# FOLD-ARRAY JOB: each array task is an independent 1-GPU job running ONE
# date_kfold fold (--fold $SLURM_ARRAY_TASK_ID); --requeue re-submits a fold if
# the preemptable QOS preempts it.
#
# After ALL array tasks finish, aggregate (no GPU needed), e.g.:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_tabm/tabm_ocean_prof_f*' --label TabM+prof \
#     --out results/model_comparison/tabm_ocean_profile_kfold_agg.md

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8 cuda/12.1.1
conda activate data

# Prepend conda's libs so Python's netCDF4/h5py loads the conda-compiled
# libhdf5 rather than the system one injected by `module load hdf5/...`.
# Without this, an ABI mismatch causes NC_EHDF (-101) at H5Fopen() time.
# On Linux, $CONDA_PREFIX may be empty when conda activate runs non-interactively
# (SLURM batch context), so hardcode the path as a reliable fallback.
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

# HDF5 file locking is not supported on Lustre (/pl/active/).
# Without this, HDF5 >= 1.10 raises NC_EHDF (-101) on any open() call.
export HDF5_USE_FILE_LOCKING=FALSE

# Allow numpy/sklearn (MKL/OpenBLAS) and joblib to use all allocated cores.
# MKL_THREADING_LAYER=GNU forces MKL to use GNU OpenMP (libgomp) instead of
# Intel IOMP5, which otherwise grabs CUDA resources before PyTorch initialises
# cuBLAS — causing CUBLAS_STATUS_NOT_INITIALIZED at the first cublasSgemm call.
export OMP_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_THREADING_LAYER=GNU

# Run from the repo ROOT (get_storage_dir() is cwd-relative → './results/...'),
# with src on PYTHONPATH so the package imports resolve.  TabM mixes relative
# imports (from .pipeline) and absolute (from utils / from search.tracking),
# so it must be launched as a package module: `python -m models.tabm`.
cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:$PYTHONPATH

# Log GPU utilisation every 10 s in background; killed automatically when job ends
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

# Each model fits its own FeaturePipeline ON THE TRAIN SPLIT ONLY (leakage
# discipline) — do NOT pass a pre-fitted full-data --pipeline for blocked splits.

F=${SLURM_ARRAY_TASK_ID}
NFOLDS=5

# ── Full feature set (+profile), both surfaces, 10 km reference ────────────────
python -m models.tabm --sfc_type 0 --suffix tabm_ocean_prof_f${F} \
    --K 16 --profile-pca \
    --conformal --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

python -m models.tabm --sfc_type 1 --suffix tabm_land_prof_f${F} \
    --K 16 --profile-pca \
    --conformal --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Feature-set ablations (+profile), both surfaces ───────────────────────────
# Same config, each with one feature block dropped.  The profile block is
# ORTHOGONAL to --feature_set (the no_xco2/no_spec drops never touch it), so these
# isolate each block's contribution ON TOP OF the profile block.
for FS in no_xco2 no_spec no_xco2_and_spec; do
  python -m models.tabm --sfc_type 0 --suffix tabm_ocean_${FS}_prof_f${F} \
      --K 16 --profile-pca --feature_set ${FS} \
      --conformal --near_cloud_target 0.98 --mondrian_col cld_dist_km \
      --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

  python -m models.tabm --sfc_type 1 --suffix tabm_land_${FS}_prof_f${F} \
      --K 16 --profile-pca --feature_set ${FS} \
      --conformal --near_cloud_target 0.98 --mondrian_col cld_dist_km \
      --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
done

kill $GPU_MONITOR_PID 2>/dev/null || true

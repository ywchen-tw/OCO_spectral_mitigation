#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_train_tabm
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1


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
# Without this, HDF5 ≥ 1.10 raises NC_EHDF (-101) on any open() call.
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

# ── Primary result: TabM K=16, ocean (sfc 0), huber loss ──────────────────────
# # Random split (comparable to FT-Transformer)
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k16_random  --K 16
# # Date-block split (temporal leakage probe)
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k16_date    --K 16 --val_split date

# ── Land (sfc 1) ──────────────────────────────────────────────────────────────
python -m models.tabm --sfc_type 1 --suffix tabm_land_k16_random  --K 16
python -m models.tabm --sfc_type 1 --suffix tabm_land_k16_date    --K 16 --val_split date

# ── K-sweep ablation (K=1 degenerate MLP / 8 / 32) ────────────────────────────
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k1   --K 1
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k8   --K 8
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k32  --K 32

# ── Loss comparison: plain pinball on all three quantiles ─────────────────────
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_quantile --K 16 --loss quantile

# ── Feature-set ablations ─────────────────────────────────────────────────────
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_no_xco2 --K 16 --feature_set no_xco2
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_no_spec --K 16 --feature_set no_spec

# ── Feature-set ablations ─────────────────────────────────────────────────────
# python -m models.tabm --sfc_type 1 --suffix tabm_land_no_xco2 --K 16 --feature_set no_xco2
# python -m models.tabm --sfc_type 1 --suffix tabm_land_no_spec --K 16 --feature_set no_spec

# ── Auxiliary cloud-proximity head (multi-task ablation) ──────────────────────
# (cld_dist_km / near_cloud must be ABSENT from pipeline.features — enforced in code)
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_aux       --K 16 --aux_cloud --lambda_cloud 0.1
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_aux_bins  --K 16 --aux_cloud --cloud_label bins --lambda_cloud 0.1

# ── Repeated seeds for the primary result (report mean ± std) ─────────────────
# NOTE: under date / date_kfold the split is deterministic, so --seed varies only
# model-training stochasticity (init + batch order), not the test set.
# for S in 0 1 2; do
#   python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k16_random_s${S} --K 16 --seed ${S}
#   python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k16_date_s${S}   --K 16 --seed ${S} --val_split date
# done

# ── Block-rotation k-fold over dates (general unseen-date robustness; mean±std) ─
# One fold per invocation → distinct suffix dir; every date is test exactly once.
# This is the PRIMARY robustness probe (lower-variance than a single trailing
# date block).  Aggregate the folds afterwards with:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_tabm/tabm_ocean_kfold_f*'  --label TabM \
#     --dirs 'results/model_gbdt/gbdt_ocean_kfold_f*'  --label XGB \
#     --dirs 'results/model_mlp_baseline/mlp_ocean_kfold_f*' --label MLP \
#     --out results/model_comparison/ocean_kfold_agg.md
# NFOLDS=5
# for F in $(seq 0 $((NFOLDS-1))); do
#   python -m models.tabm --sfc_type 0 --suffix tabm_ocean_kfold_f${F} --K 16 \
#     --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
# done
# # land:
# for F in $(seq 0 $((NFOLDS-1))); do
#   python -m models.tabm --sfc_type 1 --suffix tabm_land_kfold_f${F} --K 16 \
#     --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
# done

kill $GPU_MONITOR_PID 2>/dev/null || true

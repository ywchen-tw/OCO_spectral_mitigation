#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_train_gbdt
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
# GBDT quantile baselines are CPU-only — no GPU requested.


module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data

# Prepend conda's libs so Python's netCDF4/h5py loads the conda-compiled
# libhdf5 rather than the system one injected by `module load hdf5/...`.
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

# HDF5 file locking is not supported on Lustre (/pl/active/).
export HDF5_USE_FILE_LOCKING=FALSE

# Trees are trained single-threaded (n_jobs/num_threads=1) for determinism;
# these vars bound any BLAS used by sklearn during the transform step.
export OMP_NUM_THREADS=${SLURM_NTASKS:-8}
export MKL_NUM_THREADS=${SLURM_NTASKS:-8}
export MKL_THREADING_LAYER=GNU

# Run from the repo ROOT (get_storage_dir() is cwd-relative → './results/...'),
# with src on PYTHONPATH so the package imports resolve (launch via `python -m`).
cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:$PYTHONPATH

# LightGBM is not in the base env; uncomment to enable the --model lightgbm runs.
pip install lightgbm

# Each baseline fits its own FeaturePipeline on the TRAIN split only.
# Quantile crossing is reported both as-is and after monotone rearrangement.

# ── XGBoost (objective=reg:quantileerror), ocean (sfc 0) ──────────────────────
# python -m models.gbdt_baselines --model xgboost --sfc_type 0 --suffix gbdt_ocean_xgb_random
# python -m models.gbdt_baselines --model xgboost --sfc_type 0 --suffix gbdt_ocean_xgb_date   --val_split date

# ── XGBoost, land (sfc 1) ─────────────────────────────────────────────────────
python -m models.gbdt_baselines --model xgboost --sfc_type 1 --suffix gbdt_land_xgb_random
python -m models.gbdt_baselines --model xgboost --sfc_type 1 --suffix gbdt_land_xgb_date   --val_split date

# ── LightGBM (objective=quantile) — requires `pip install lightgbm` above ─────
# python -m models.gbdt_baselines --model lightgbm --sfc_type 0 --suffix gbdt_ocean_lgbm_random
# python -m models.gbdt_baselines --model lightgbm --sfc_type 0 --suffix gbdt_ocean_lgbm_date  --val_split date

# ── Feature-set ablations (XGBoost) ───────────────────────────────────────────
# python -m models.gbdt_baselines --model xgboost --sfc_type 0 --suffix gbdt_ocean_xgb_no_xco2 --feature_set no_xco2
# python -m models.gbdt_baselines --model xgboost --sfc_type 0 --suffix gbdt_ocean_xgb_no_spec --feature_set no_spec

# ── Repeated seeds (report mean ± std; --seed flows into model random_state) ───
# for S in 0 1 2; do
#   python -m models.gbdt_baselines --model xgboost --sfc_type 0 --suffix gbdt_ocean_xgb_random_s${S} --seed ${S}
#   python -m models.gbdt_baselines --model xgboost --sfc_type 0 --suffix gbdt_ocean_xgb_date_s${S}   --seed ${S} --val_split date
# done

# ── Block-rotation k-fold over dates (general unseen-date robustness; mean±std) ─
# Aggregate afterwards with models.aggregate_folds (see the TabM script header).
# NFOLDS=5
# for F in $(seq 0 $((NFOLDS-1))); do
#   python -m models.gbdt_baselines --model xgboost --sfc_type 0 --suffix gbdt_ocean_kfold_f${F} \
#     --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
# done

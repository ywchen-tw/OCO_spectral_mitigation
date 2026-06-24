#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_train_geom_mlp
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1


module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8 cuda/12.1.1
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

# MKL_THREADING_LAYER=GNU avoids Intel IOMP5 grabbing CUDA before PyTorch
# initialises cuBLAS (CUBLAS_STATUS_NOT_INITIALIZED on the first cublasSgemm).
export OMP_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_THREADING_LAYER=GNU

# Run from the repo ROOT (get_storage_dir() is cwd-relative → './results/...'),
# with src on PYTHONPATH so the package imports resolve (launch via `python -m`).
cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:$PYTHONPATH

# Log GPU utilisation every 10 s in background; killed automatically when job ends
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

# Geometry experiment: does a richer representation of geometry help the
# accuracy-leading MLP backbone?  Periodic (sin/cos harmonic) embeddings of raw
# angles (sza, vza, glint_angle, pol_angle, lat, lon) + optional FiLM gate.
# Point predictor — interval metrics are degenerate; compare RMSE/MAE/R².
#
# geom_mode=none reproduces the plain MLP on the same base features → it is the
# control. The advantage of the geometry embedding is (concat|film) MINUS none,
# read per-fold (paired) since folds share the held set.

# ── Block-rotation k-fold over dates: control (none) vs concat vs film ─────────
# NFOLDS=5
# for F in $(seq 0 $((NFOLDS-1))); do
#   python -m models.geom_mlp --sfc_type 0 --suffix geom_ocean_none_f${F}   --geom_mode none   --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
#   python -m models.geom_mlp --sfc_type 0 --suffix geom_ocean_concat_f${F} --geom_mode concat --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
#   python -m models.geom_mlp --sfc_type 0 --suffix geom_ocean_film_f${F}   --geom_mode film   --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
# done

# Aggregate + compare (control vs each fusion) after the run:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_geom_mlp/geom_ocean_none_f*'   --label none \
#     --dirs 'results/model_geom_mlp/geom_ocean_concat_f*' --label concat \
#     --dirs 'results/model_geom_mlp/geom_ocean_film_f*'   --label film \
#     --out results/model_comparison/geom_ocean_kfold_agg.md

# ── Land (sfc 1): uncomment to repeat ─────────────────────────────────────────
for F in $(seq 0 $((NFOLDS-1))); do
  python -m models.geom_mlp --sfc_type 1 --suffix geom_land_none_f${F}   --geom_mode none   --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
  python -m models.geom_mlp --sfc_type 1 --suffix geom_land_concat_f${F} --geom_mode concat --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
  python -m models.geom_mlp --sfc_type 1 --suffix geom_land_film_f${F}   --geom_mode film   --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
done

kill $GPU_MONITOR_PID 2>/dev/null || true

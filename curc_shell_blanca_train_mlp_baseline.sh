#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_train_mlp_baseline
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

# Clean re-implementation of the result_ana.py MLP (Linear64→ReLU→Linear32→ReLU→Linear1)
# in the shared scaffold.  Point predictor: interval metrics are degenerate
# (q05=q50=q95); only RMSE/MAE/R² are meaningful for this baseline.
# Fits its own FeaturePipeline on the TRAIN split only.

# ── Ocean (sfc 0) ─────────────────────────────────────────────────────────────
# python -m models.mlp_baseline --sfc_type 0 --suffix mlp_ocean_random
# python -m models.mlp_baseline --sfc_type 0 --suffix mlp_ocean_date   --val_split date

# ── Land (sfc 1) ──────────────────────────────────────────────────────────────
# python -m models.mlp_baseline --sfc_type 1 --suffix mlp_land_random
# python -m models.mlp_baseline --sfc_type 1 --suffix mlp_land_date   --val_split date

# ── Feature-set ablations ─────────────────────────────────────────────────────
# python -m models.mlp_baseline --sfc_type 0 --suffix mlp_ocean_no_xco2 --feature_set no_xco2
# python -m models.mlp_baseline --sfc_type 0 --suffix mlp_ocean_no_spec --feature_set no_spec

# ── Feature-set ablations ─────────────────────────────────────────────────────
python -m models.mlp_baseline --sfc_type 1 --suffix mlp_land_no_xco2 --feature_set no_xco2
python -m models.mlp_baseline --sfc_type 1 --suffix mlp_land_no_spec --feature_set no_spec

kill $GPU_MONITOR_PID 2>/dev/null || true

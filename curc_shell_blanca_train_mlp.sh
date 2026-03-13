#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_spectral_anal
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

cd /projects/yuch8913/OCO_spectral_mitigation

# Log GPU utilisation every 10 s in background; killed automatically when job ends
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

python src/mlp_lr_models.py --pipeline results/train_data/pipeline_land_2016_2020.pkl \
 --sfc_type 1 --suffix land_2016_2020_4

# python src/mlp_lr_models.py --pipeline results/train_data/pipeline_ocean_2016_2020.pkl \
#  --sfc_type 0 --suffix ocean_2016_2020_4

kill $GPU_MONITOR_PID 2>/dev/null || true


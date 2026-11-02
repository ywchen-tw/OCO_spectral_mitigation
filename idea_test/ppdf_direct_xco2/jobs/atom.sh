#!/bin/bash
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/yuch8913/OCO_spectral_mitigation/idea_test/ppdf_direct_xco2/jobs/logs/%x_%j.log
module load anaconda >/dev/null 2>&1; conda activate data >/dev/null 2>&1
export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src
/projects/yuch8913/software/anaconda/envs/data/bin/python3 idea_test/ppdf_direct_xco2/atom_fit.py "$@" --workers 32
echo "EXIT $?"

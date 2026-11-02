#!/bin/bash
#SBATCH --job-name=ppdf_v2_perdate
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/yuch8913/OCO_spectral_mitigation/idea_test/ppdf_direct_xco2/jobs/logs/v2_perdate_%j.log
export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /projects/yuch8913/OCO_spectral_mitigation
/projects/yuch8913/software/anaconda/envs/data/bin/python3 idea_test/ppdf_direct_xco2/run_v2.py perdate --workers 11 --skip-existing
echo "EXIT $?"

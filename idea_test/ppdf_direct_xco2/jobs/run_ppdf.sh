#!/bin/bash
#SBATCH --job-name=ppdf2016
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/yuch8913/OCO_spectral_mitigation/idea_test/ppdf_direct_xco2/jobs/logs/ppdf2016_%j.log
export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /projects/yuch8913/OCO_spectral_mitigation
/projects/yuch8913/software/anaconda/envs/data/bin/python3 idea_test/ppdf_direct_xco2/run_test.py \
  --all-combined --dates 2018-07-10 2021-09-06 2023-06-26 2019-06-22 2017-10-09 \
  --workers 10 --skip-existing --out-dir idea_test/ppdf_direct_xco2/results_2016_2020
echo "EXIT $?"

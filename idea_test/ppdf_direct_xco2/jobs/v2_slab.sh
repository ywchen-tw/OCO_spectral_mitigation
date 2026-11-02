#!/bin/bash
#SBATCH --job-name=ppdf_v2_slab
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/yuch8913/OCO_spectral_mitigation/idea_test/ppdf_direct_xco2/jobs/logs/v2_slab_%j.log
export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /projects/yuch8913/OCO_spectral_mitigation
/projects/yuch8913/software/anaconda/envs/data/bin/python3 idea_test/ppdf_direct_xco2/slab_mc.py --n-train 4000 --n-test 2000 --workers 12
echo "EXIT $?"

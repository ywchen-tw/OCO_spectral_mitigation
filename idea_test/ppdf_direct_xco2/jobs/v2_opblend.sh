#!/bin/bash
#SBATCH --job-name=ppdf_v2_opblend
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/yuch8913/OCO_spectral_mitigation/idea_test/ppdf_direct_xco2/jobs/logs/v2_opblend_%j.log
export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
cd /projects/yuch8913/OCO_spectral_mitigation
P=/projects/yuch8913/software/anaconda/envs/data/bin/python3
S=idea_test/ppdf_direct_xco2/run_v2.py
$P $S opblend && $P $S summary
echo "EXIT $?"

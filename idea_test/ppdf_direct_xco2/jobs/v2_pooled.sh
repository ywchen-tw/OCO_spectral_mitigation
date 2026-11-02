#!/bin/bash
#SBATCH --job-name=ppdf_v2_pooled
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/yuch8913/OCO_spectral_mitigation/idea_test/ppdf_direct_xco2/jobs/logs/v2_pooled_%j.log
export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12
cd /projects/yuch8913/OCO_spectral_mitigation
P=/projects/yuch8913/software/anaconda/envs/data/bin/python3
S=idea_test/ppdf_direct_xco2/run_v2.py
# the pooled stage rewrites the compact files; a requeue must restart from perdate output
if [ -f /projects/yuch8913/OCO_spectral_mitigation/idea_test/ppdf_direct_xco2/jobs/logs/pooled.done ]; then echo "pooled already ran"; else $P $S pooled || { echo "pooled FAILED"; exit 1; }; fi
$P $S neardiag; echo "neardiag $?"
$P $S tccon; echo "tccon $?"
$P $S summary; echo "summary $?"
echo "EXIT done"

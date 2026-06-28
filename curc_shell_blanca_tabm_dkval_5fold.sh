#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_tabm_dkval
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-14
#SBATCH --requeue

# 5-FOLD VALIDATION of the top-3 date_kfold-HPO configs (ocean, full_contam).
# The HPO ranked on fold 0 only, where the top configs were within 0.003 R²
# (flat landscape).  This re-runs each of the 3 candidates across ALL 5 folds so we
# pick the production config by 5-FOLD MEDIAN — the honest verdict, immune to fold-0
# luck.  Candidates (configs/tabm_dk_*.json), tuned on the full 2016-2020 set:
#   s19: K=32 d=128 L=4 drop=.05 lr=5.3e-4 bs=16384 hub=.93   (fold0 best, expensive)
#   s20: K=32 d=192 L=3 drop=.27 lr=3.3e-3 bs=8192  hub=1.36  (fold0 #2,  expensive)
#   s2 : K=8  d=128 L=2 drop=.14 lr=4.8e-4 bs=8192  hub=.88   (fold0 #3,  CHEAP — if
#        it ties the K=32 configs across folds, it's the production pick at ~1/4 cost)
#
# ONE TASK = ONE (config, fold): 3 configs x 5 folds = 15 tasks, max GPU concurrency.
# task_id -> config = id/5, fold = id%5.  --requeue re-runs only the single lost cell.
#
# AFTER all tasks finish, compare 5-fold medians vs the default-TabM / DE baselines:
#   for C in s19 s20 s2; do
#     PYTHONPATH=src python -m models.aggregate_folds \
#       --dirs "results/model_tabm/tabm_dkval_${C}_full_contam_f*" --label "tuned_${C}" \
#       --out results/model_comparison/tabm_dkval_${C}.md ; done
# then read each median r2 against default TabM (0.628) / DE (0.593) ocean full_contam.

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8 cuda/12.1.1
conda activate data

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:$PYTHONPATH

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv &
GPU_MONITOR_PID=$!

CONFIGS=(s19 s20 s2)
T=${SLURM_ARRAY_TASK_ID}
CIDX=$(( T / 5 ))
FOLD=$(( T % 5 ))
C=${CONFIGS[$CIDX]}

python -m models.tabm --sfc_type 0 --suffix tabm_dkval_${C}_full_contam_f${FOLD} \
  --config configs/tabm_dk_${C}.json --feature_set full_contam \
  --val_split date_kfold --n_folds 5 --fold ${FOLD}

kill $GPU_MONITOR_PID 2>/dev/null || true

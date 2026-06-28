#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_tabm_hpo_dk
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --requeue

# TabM hyperparameter search with DATE-BLOCKED validation, on the FULL 2016-2020 set.
#
# WHY: the local single-date/random-split HPO overfit its proxy — its winner (t09,
# proxy R²=0.914) LOST to the TabM default on the real date_kfold footing
# (full 0.450 vs 0.458; full_contam 0.485 vs 0.506).  Random split leaks autocorrelated
# neighbours into validation, so it rewarded under-regularized configs (wd=8e-6, tiny
# batch) that don't generalize across held-out DATES.  This search fixes the objective:
# every trial is scored by --val_split date_kfold (held-out date block), the deployment
# regime.  Production-scale search space (batch in {4096,8192,16384}; wider wd/dropout
# so date-CV can choose more regularization).
#
# PARALLELISM: array of 4 independent 1-GPU tasks.  Each task t runs --seed t, so it
# samples a DISTINCT slice of the space and writes its own seed-tagged CSV
# (results/model_comparison/hpo_dk_ocean_full_contam_s{t}_trials.csv).  4 tasks x
# N_PER trials = the full budget; --requeue re-runs a preempted task.
#
# Cost: each trial is ONE full date_kfold-fold-0 training on ~8.3M ocean rows.
# At EPOCHS=60 / batch 8192 that's ~1000 steps/epoch; budget ~5 trials/task under the
# 20h wall.  Raise N_PER / the array range for a denser search if GPUs are plentiful.
#
# AFTER all tasks finish (no GPU needed) — aggregate, pick winner, write tuned config:
#   PYTHONPATH=src python tabm_hpo_search.py --aggregate --tag hpo_dk_ocean_full_contam
# That writes tabm_tuned_ocean_datekfold.json; point curc_shell_blanca_tabm_full_contam.sh
# at it (TUNED=tabm_tuned_ocean_datekfold.json) and re-run the 5-fold comparison.

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

SEED=${SLURM_ARRAY_TASK_ID}
N_PER=5
EPOCHS=60

# Ocean search (the surface we have a DE baseline for).  Each task = one search seed.
python tabm_hpo_search.py --seed ${SEED} --n_trials ${N_PER} \
  --sfc_type 0 --feature_set full_contam \
  --val_split date_kfold --n_folds 5 --fold 0 --epochs ${EPOCHS}

kill $GPU_MONITOR_PID 2>/dev/null || true

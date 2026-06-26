#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_tabm_auxcloud
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --requeue

# TabM AUXILIARY-CLOUD MULTI-TASK ABLATION (date_kfold, both surfaces).
# Question: does an auxiliary near-cloud (cld_dist_km <= 10km) classification head
# on the shared TabM backbone lift near-cloud XCO2 accuracy OUT OF DISTRIBUTION
# (held-out dates), not just in-distribution?
#
# Why TabM and why CURC: the local checks showed (a) the deep ensemble's tiny
# 64x32 backbone is capacity-starved and the aux head HURTS it; (b) TabM's larger
# backbone benefits in-distribution (+0.04 near-cloud R2 on a single date); but
# (c) date_kfold on only 12 local dates is degenerate (best epoch 0) and a single
# random split is too noisy to resolve a ~0.03 effect.  Only the full 66-date set
# under real date_kfold, averaged over 5 folds, can answer it.  cld_dist_km is the
# aux TARGET only -- it is banned from pipeline.features (enforced in tabm.py).
#
# 30 array tasks = {ocean, land} x lambda{0.0, 0.1, 0.3} x fold{0..4}, each an
# INDEPENDENT gpu:1 job the scheduler backfills across GPUs (like the DE ablation
# and the fold-array).  lambda=0.0 runs WITHOUT the aux head = the pure-regression
# TabM baseline for that (surface, fold).  Index decode (bash word-splits, so the
# inline ${AUX} flags expand correctly):
#   fold    = idx % 5
#   lambda  = LAMBDAS[(idx / 5) % 3]
#   surface = idx / 15           (0 = ocean, 1 = land)
#
# After all tasks finish, aggregate each (surface, lambda) across folds and read
# off near-cloud XCO2 R2 (aux vs lambda=0) + cloud AUC:
#   for S in ocean land; do for L in 0.0 0.1 0.3; do
#     PYTHONPATH=src python -m models.aggregate_folds \
#       --dirs "results/model_tabm/tabm_${S}_auxL${L}_f*" --label "TabM_${S}_L${L}" \
#       --out "results/model_comparison/tabm_${S}_auxL${L}_kfold_agg.md"
#   done; done
# The aux head's cloud AUC/AP is in each run's metrics json (reported by tabm.py).

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
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

NFOLDS=5
LAMBDAS=(0.0 0.1 0.3)
SFC_IDS=(0 1)
SFC_NAMES=(ocean land)

T=${SLURM_ARRAY_TASK_ID}
F=$((T % 5))
LAM=${LAMBDAS[$(((T / 5) % 3))]}
SI=$((T / 15))
SFC=${SFC_IDS[$SI]}
SNAME=${SFC_NAMES[$SI]}

# lambda=0.0 -> pure-regression baseline (no aux head); lambda>0 -> add the head.
AUX=""
[ "${LAM}" != "0.0" ] && AUX="--aux_cloud --cloud_label binary --lambda_cloud ${LAM}"

echo "[task ${T}] surface=${SNAME} lambda=${LAM} fold=${F}  AUX='${AUX}'"

python -m models.tabm --sfc_type ${SFC} --suffix tabm_${SNAME}_auxL${LAM}_f${F} \
  --K 16 --val_split date_kfold --n_folds ${NFOLDS} --fold ${F} --seed 42 ${AUX}

kill $GPU_MONITOR_PID 2>/dev/null || true

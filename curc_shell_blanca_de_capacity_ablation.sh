#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_de_capacity
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --requeue

# DEEP-ENSEMBLE CAPACITY CONFIRMATION (date_kfold, both surfaces).
# Question: does the near-cloud XCO2 lift from a bigger backbone survive OUT OF
# DISTRIBUTION (held-out dates)?  The local random-split screen showed the 64x32
# backbone badly UNDERFITS -- 128,64,32 gave +0.243 near-cloud R2 (0.510->0.753),
# larger than the MODIS oracle and free.  But random split is in-distribution;
# bigger models can overfit the training dates, so this must be checked under real
# date_kfold on the full 66-date set.
#
# SINGLE-TASK, current style: predicts xco2_bc_anomaly ONLY.  No cloud head
# (cloud_aux_weight=0, default) and no cloud-distance input (cloud_bin_feature=
# none, default).  The ONLY variable swept is --hidden_dims, so any change is
# attributable to backbone capacity alone.  All other settings match production
# (beta_nll beta=1.0, M=5, batch 8192, near_cloud_target 0.98, mondrian cld_dist).
#
# 30 array tasks = {ocean, land} x hidden_dims{64,32 | 128,64,32 | 256,128,64} x
# fold{0..4}, each an independent gpu:1 job backfilled across GPUs.  64,32 is the
# current-production reference; 128,64,32 is the local winner; 256,128,64 is the
# overfit probe.  Index decode (bash):
#   fold   = idx % 5
#   hdims  = HDIMS[(idx / 5) % 3]
#   surface= idx / 15            (0 = ocean, 1 = land)
#
# After all finish, aggregate each (surface, hidden_dims) across folds and read
# off near-cloud XCO2 R2 -- does 128,64,32 still beat 64,32 out-of-distribution,
# and does 256,128,64 overfit (drop below 128,64,32)?
#   for S in ocean land; do for HD in 64_32 128_64_32 256_128_64; do
#     PYTHONPATH=src python -m models.aggregate_folds \
#       --dirs "results/model_deep_ensemble/de_${S}_cap${HD}_f*" --label "DE_${S}_${HD}" \
#       --out "results/model_comparison/de_${S}_cap${HD}_kfold_agg.md"
#   done; done

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
HDIMS=("64,32" "128,64,32" "256,128,64")
SFC_IDS=(0 1)
SFC_NAMES=(ocean land)

T=${SLURM_ARRAY_TASK_ID}
F=$((T % 5))
HD=${HDIMS[$(((T / 5) % 3))]}
SI=$((T / 15))
SFC=${SFC_IDS[$SI]}
SNAME=${SFC_NAMES[$SI]}
HDTAG=${HD//,/_}

echo "[task ${T}] surface=${SNAME} hidden_dims=${HD} fold=${F}"

python -m models.deep_ensemble --sfc_type ${SFC} --suffix de_${SNAME}_cap${HDTAG}_f${F} \
  --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 --near_cloud_target 0.98 \
  --mondrian_col cld_dist_km --val_split date_kfold --n_folds ${NFOLDS} --fold ${F} \
  --hidden_dims "${HD}"

kill $GPU_MONITOR_PID 2>/dev/null || true

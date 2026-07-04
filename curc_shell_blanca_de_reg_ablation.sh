#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_de_reg
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-69
#SBATCH --requeue

# DEEP-ENSEMBLE REGULARIZATION ABLATION (review §7.3 Phase 3; date_kfold, both
# surfaces).  Question: do dropout / normalization improve accuracy or reduce
# overfitting — especially in the data-scarce near-cloud tail — on top of the
# production config?  All arms are the production profile model (beta_nll
# beta=1.0, M=5, --profile-pca, batch 8192, near_cloud_target 0.98, mondrian
# cld_dist_km) with ONLY the regularization/architecture knobs swept:
#
#   arm 0  base      — production reference, retrained under the shared trainer
#                      (this arm doubles as the Phase 5 new-trainer verification:
#                      its fold metrics must land within fold noise of the
#                      existing de_*_beta_nll_prof runs)
#   arm 1  do01      — --dropout 0.1
#   arm 2  do03      — --dropout 0.3
#   arm 3  ln        — --norm layer
#   arm 4  lndo01    — --norm layer --dropout 0.1
#   arm 5  bn        — --norm batch
#   arm 6  cap_lndo  — --hidden_dims 256,128,64 --norm layer --dropout 0.1
#                      (capacity + regularization together: the arm most likely
#                      to win if the 64x32 backbone underfits at 17M rows)
#
# 70 array tasks = {ocean, land} x arm{0..6} x fold{0..4}.  Index decode:
#   fold    = idx % 5
#   arm     = (idx / 5) % 7
#   surface = idx / 35          (0 = ocean, 1 = land)
#
# DECISION RULE (do not adopt on vibes): an arm replaces production only if it
# beats arm 0 by more than the fold-to-fold sigma on BOTH global and near-cloud
# (<=10 km) / bottom-5% tail metrics, on BOTH surfaces, without coverage_90
# regression.  (TabM-HPO lesson: flat landscape — don't chase noise.)
#
# Aggregate after all finish:
#   for S in ocean land; do for A in base do01 do03 ln lndo01 bn cap_lndo; do
#     PYTHONPATH=src python -m models.aggregate_folds \
#       --dirs "results/model_deep_ensemble/de_${S}_reg_${A}_f*" --label "DE_${S}_${A}" \
#       --out "results/model_comparison/de_${S}_reg_${A}_kfold_agg.md"
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
ARM_TAGS=(base do01 do03 ln lndo01 bn cap_lndo)
ARM_FLAGS=(""
           "--dropout 0.1"
           "--dropout 0.3"
           "--norm layer"
           "--norm layer --dropout 0.1"
           "--norm batch"
           "--hidden_dims 256,128,64 --norm layer --dropout 0.1")
SFC_IDS=(0 1)
SFC_NAMES=(ocean land)

T=${SLURM_ARRAY_TASK_ID}
F=$((T % 5))
A=$(((T / 5) % 7))
SI=$((T / 35))
SFC=${SFC_IDS[$SI]}
SNAME=${SFC_NAMES[$SI]}
ATAG=${ARM_TAGS[$A]}

echo "[task ${T}] surface=${SNAME} arm=${ATAG} fold=${F} flags='${ARM_FLAGS[$A]}'"

python -m models.deep_ensemble --sfc_type ${SFC} --suffix de_${SNAME}_reg_${ATAG}_f${F} \
    --profile-pca \
    --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
    --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F} \
    ${ARM_FLAGS[$A]}

kill $GPU_MONITOR_PID 2>/dev/null || true

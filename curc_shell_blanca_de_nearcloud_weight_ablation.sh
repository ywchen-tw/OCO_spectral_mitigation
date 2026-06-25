#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_de_ncw_ablation
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --requeue

# NEAR-CLOUD-WEIGHT ABLATION (fold-0, beta_nll).
# Question: does upweighting the near-cloud (cld_dist_km <= 10km) rows in the
# training loss improve point accuracy ON the near-cloud regime we actually
# deploy on?  Baseline data split is 19% near / 81% far, so the unweighted
# global loss is dominated by the far-cloud majority (de_land_beta_nll_f0:
# near R2~0.53, far R2~0.27).  --near_cloud_weight K multiplies the per-sample
# loss of near rows by K (applied to BOTH train and the early-stopping calib
# block), trading far-cloud accuracy for near-cloud accuracy.  NOTE: cld_dist_km
# is used ONLY for this weighting (and Mondrian binning) -- it is NOT a model
# feature.
#
# 3 array tasks = the weight sweep K={1,3,5}.  Each task runs BOTH surfaces
# sequentially (ocean sfc_type 0 + land sfc_type 1, like the production script),
# gpu:1, ~2 surfaces x 5 members ≈ 10h, under the 16h wall.  As in production,
# the two surfaces are explicit lines below -- comment one out to run a single
# surface.  Read off near-cloud R2/RMSE vs far-cloud cost across both surfaces:
#   awk -F, '$1=="cloud_proximity"{print FILENAME,$2,"r2="$6}' \
#     results/model_deep_ensemble/de_*_beta_nll_ncw*_f0/de_raw_date_kfold_stratified_metrics.csv
# K=1 reproduces the current production baselines (sanity check it matches
# de_ocean/land_beta_nll_f0).  Per surface, pick the K with the best near-cloud
# R2 whose far-cloud R2 has not collapsed, then roll that K into the 5-fold
# production script (surfaces may prefer different K).

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

# Array index -> near-cloud weight (K=1 is the unweighted baseline).
WEIGHTS=(1 3 5)
W=${WEIGHTS[$SLURM_ARRAY_TASK_ID]}
F=0
NFOLDS=5

# Ocean (sfc_type 0)
# python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_beta_nll_ncw${W}_f${F} \
#   --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#   --near_cloud_weight ${W} --near_cloud_km 10.0 --near_cloud_target 0.98 \
#   --mondrian_col cld_dist_km --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# Land (sfc_type 1)
python -m models.deep_ensemble --sfc_type 1 --suffix de_land_beta_nll_ncw${W}_f${F} \
  --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
  --near_cloud_weight ${W} --near_cloud_km 10.0 --near_cloud_target 0.98 \
  --mondrian_col cld_dist_km --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

kill $GPU_MONITOR_PID 2>/dev/null || true

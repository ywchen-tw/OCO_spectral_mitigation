#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_de_land_snow_feat
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
#SBATCH --requeue

# LAND DE feature-set comparison, RE-RUN WITH snow=1 data (--include_snow).
#
# The existing land feature-set comparison (de_land_{full_contam,no_spec,no_xco2,
# no_xco2_and_spec}_f*) was trained on snow_flag==0 only, so it is stale relative
# to how land is now trained (include_snow is the documented high-lat win).  This
# refreshes it: same recipe and same 4 feature sets as the CURRENT OCEAN comparison
# (beta_nll beta=1.0, M=5, date_kfold 5-fold, Mondrian-CQR by cld_dist_km,
# near_cloud_target 0.98), changing ONLY --include_snow so snow/ice footprints are
# kept in train/cal/holdout.  Output suffix: de_land_snow_{feature_set}_f{FOLD}.
#
# Matrix: 4 feature sets x 5 folds = 20 runs, one (fold, feature_set) per array task.
#   fs   = FS_LIST[ TASK % 4 ]
#   fold = TASK / 4            (0..4)
#
# Data: CURC Linux default (combined_2016_2020_dates.parquet) — the full multi-year
# set that the existing land/ocean comparisons used; it contains snow_flag==1 rows.
#
# AFTER the array completes, aggregate with:
#   PYTHONPATH=src python workspace/aggregate_land_snow_features.py
# (compares the 4 snow feature sets, mean +/- std, global + near-cloud).

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

# 4 feature sets (identical to the current ocean comparison).
FS_LIST=(full_contam no_spec no_xco2 no_xco2_and_spec)
NFS=${#FS_LIST[@]}
FS=${FS_LIST[$(( SLURM_ARRAY_TASK_ID % NFS ))]}
FOLD=$(( SLURM_ARRAY_TASK_ID / NFS ))

# Recipe matched to the existing land/ocean DE feature-set comparison.
COMMON="--loss beta_nll --beta 1.0 --n_members 5 --nu 4.0 \
        --near_cloud_target 0.98 --near_cloud_km 10.0 --mondrian_col cld_dist_km \
        --val_split date_kfold --n_folds 5 --fold ${FOLD}"

echo "── DE land (snow=1)  feature_set=${FS}  fold=${FOLD}  (task ${SLURM_ARRAY_TASK_ID}) ──"
python -m models.deep_ensemble --sfc_type 1 --include_snow \
    --feature_set "${FS}" --suffix "de_land_snow_${FS}_f${FOLD}" ${COMMON}

kill $GPU_MONITOR_PID 2>/dev/null || true

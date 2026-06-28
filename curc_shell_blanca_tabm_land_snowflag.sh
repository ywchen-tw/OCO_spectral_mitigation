#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_tabm_snowflag
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# SNOW-FLAG EXPERIMENT for TabM — ARM B (the TabM counterpart of the DE arm B in
# curc_shell_blanca_train_de_snowflag.sh).  LAND only (every snow footprint is
# sfc_type=1, so snow handling changes nothing on ocean).
#
#   A nosnow   = tabm_land_full_contam_f*   PRODUCTION baseline (snow EXCLUDED).
#                Already trained — NOT recomputed.  Status-quo anchor.
#   B snowdata = tabm_land_fc_snowdata_f*   snow INCLUDED, full_contam (no snow_flag
#                feature).  THIS SCRIPT.  Isolates the effect of adding the snow
#                training DATA (B vs A).
#
# Uses the DEFAULT TabM structure (K=16, d_model=256, n_layers=4, huber=1.0, 500ep,
# batch 8192) — the date_kfold HPO showed tuning does NOT beat the default (flat
# landscape), so arm B mirrors the production tabm_land_full_contam config exactly
# except for --include_snow.  date_kfold, one fold per array task.
#
# After ALL array tasks finish, compare B vs the A baseline (no GPU):
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_tabm/tabm_land_full_contam_f*' --label A_nosnow \
#     --dirs 'results/model_tabm/tabm_land_fc_snowdata_f*' --label B_snowdata \
#     --out results/model_comparison/tabm_snowflag_land_kfold_agg.md
#   # plus the high-lat / snow-slice eval used for DE:
#   #   PYTHONPATH=src python workspace/eval_snowflag_highlat.py   (point it at the tabm dirs)

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

F=${SLURM_ARRAY_TASK_ID}
NFOLDS=5

# Arm B: snow INCLUDED, full_contam (default structure, --K 16, no --config).
python -m models.tabm --sfc_type 1 --suffix tabm_land_fc_snowdata_f${F} --K 16 \
  --feature_set full_contam --include_snow \
  --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

kill $GPU_MONITOR_PID 2>/dev/null || true

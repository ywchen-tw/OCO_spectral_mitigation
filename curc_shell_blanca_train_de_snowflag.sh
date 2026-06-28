#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_de_snowflag
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# SNOW-FLAG EXPERIMENT (full-scale, CURC).  Follow-up to the local 2020 fold-0 test
# (workspace/run_snowflag_test.sh).  Trains, per date_kfold fold, the two LAND arms
# that differ from the production de_land_full_contam baseline ONLY in the snow
# handling — same data (combined_2016_2020_dates.parquet, default), loss (beta_nll
# beta=1.0), M=5 members, batch 8192, --near_cloud_target 0.98, mondrian cld_dist_km:
#
#   A nosnow  = de_land_full_contam_f*    PRODUCTION baseline (snow EXCLUDED).  Already
#               trained — NOT recomputed here.  Reused as the status-quo anchor.
#   B snowdata= de_land_fc_snowdata_f*    snow INCLUDED, full_contam (no snow_flag).
#               Isolates the effect of adding the snow training DATA.
#   C snowflag= de_land_fc_snowflag_f*    snow INCLUDED, full_contam_snow (+snow_flag).
#               Isolates the effect of adding the snow_flag FEATURE.
#
# B vs C share an identical holdout (same fold/seed, both snow-included) so their
# difference is purely the snow_flag column → clean feature attribution.
#
# LAND ONLY: every snow footprint is sfc_type=1, so --include_snow changes nothing on
# ocean and full_contam_snow appends snow_flag on land only.  The ocean arm-B/arm-C
# models would be byte-identical to de_ocean_full_contam_f* — recomputing them is pure
# waste, so this script does not.  (Ocean baseline already exists.)
#
# Local fold-0 verdict (3 members, 2020): C made RMSE WORSE than B on the high-lat /
# snow slices (+0.45–0.56 ppm), while B matched-or-beat A even on non-snow rows.  This
# run confirms across all 5 folds on the full 2016–2020 dataset at production M=5.
#
# After ALL array tasks finish, aggregate / evaluate (no GPU):
#   PYTHONPATH=src python workspace/eval_snowflag_highlat.py     # high-lat / snow slices
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_deep_ensemble/de_land_fc_snowflag_f*' --label DE_snowflag \
#     --out  results/model_comparison/de_snowflag_land_kfold_agg.md

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

# ── Arm B: snow INCLUDED, full_contam (no snow_flag) ──────────────────────────
python -m models.deep_ensemble --sfc_type 1 --suffix de_land_fc_snowdata_f${F} \
  --feature_set full_contam --include_snow \
  --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
  --near_cloud_target 0.98 --mondrian_col cld_dist_km \
  --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Arm C: snow INCLUDED, full_contam_snow (+ snow_flag) ──────────────────────
python -m models.deep_ensemble --sfc_type 1 --suffix de_land_fc_snowflag_f${F} \
  --feature_set full_contam_snow --include_snow \
  --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
  --near_cloud_target 0.98 --mondrian_col cld_dist_km \
  --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

kill $GPU_MONITOR_PID 2>/dev/null || true

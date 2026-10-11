#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_de_ablation_extra
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# SURGICAL follow-up to curc_shell_blanca_train_deep_ensemble.sh.  Does ONLY the two
# things that are missing/broken, so the already-good runs are NOT recomputed:
#
#   (1) RE-RUN the one diverged fold: de_ocean_no_xco2_f0 blew up under beta_nll
#       (member log-var ran to the clamp ceiling -> sigma~112, mu over-dispersed,
#       RMSE 2.10 / R2 -11.4 vs ~0.45 on folds 1-4).  Same seed=42 is deterministic
#       and would re-diverge, so re-run with seed=43 (different member inits avoid the
#       pathological start).  Overwrites the bad dir -> the 5-fold mean goes clean.
#       Only array task 0 does this.
#
#   (2) NEW feature group no_xco2_and_spec (drops XCO2_FEATURES | SPEC_FEATURES) for
#       BOTH surfaces, all 5 folds — the combined ablation, mirroring the production
#       config (beta_nll + near_cloud_target 0.98) so the only difference is the
#       feature set.  Suffix: de_{surface}_no_xco2_and_spec_f{F}.
#
# After ALL array tasks finish, re-aggregate as before (median over folds).

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

# ── (1) Re-run ONLY the diverged fold (ocean, no_xco2, fold 0) with a clean seed ──
# if [ "${F}" = "0" ]; then
#   python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_no_xco2_f0 \
#     --feature_set no_xco2 --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#     --near_cloud_target 0.98 --mondrian_col cld_dist_km --seed 43 \
#     --val_split date_kfold --n_folds ${NFOLDS} --fold 0
# fi

# ── (2) New combined-ablation group: no_xco2_and_spec, both surfaces, all folds ───
# python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_no_xco2_and_spec_f${F} \
#   --feature_set no_xco2_and_spec --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#   --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#   --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# python -m models.deep_ensemble --sfc_type 1 --suffix de_land_no_xco2_and_spec_f${F} \
#   --feature_set no_xco2_and_spec --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#   --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#   --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── (3) New feature group full_contam: adds cloud/aerosol contamination features on
#       top of `full` (per-surface; see pipeline.CONTAM_FEATURES).  Date-blocked-CV GBDT
#       screen projected ~+0.06 (ocean) / +0.07 (land) held-out R² over `full`.  Both
#       surfaces, all folds, mirroring the production config (beta_nll + near_cloud_target
#       0.98).  Suffix: de_{surface}_full_contam_f{F}.
# python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_full_contam_f${F} \
#   --feature_set full_contam --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#   --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#   --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

python -m models.deep_ensemble --sfc_type 1 --suffix de_land_full_contam_f${F} \
  --feature_set full --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
  --near_cloud_target 0.98 --mondrian_col cld_dist_km \
  --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

kill $GPU_MONITOR_PID 2>/dev/null || true

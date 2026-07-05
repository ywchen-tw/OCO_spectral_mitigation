#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_tabm_r05
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# ── TabM WITH profile EOF block, targeting the 5 km clear-sky ref (OCEAN) ──────
# TabM companion to curc_shell_blanca_de_profile_r05.sh.  Identical config to
# curc_shell_blanca_train_tabm.sh (K=16, +profile block, conformal near-cloud
# over-coverage), differing ONLY by:
#   (1) --target 5km selects xco2_bc_anomaly_r05 (5 km clear-sky reference)
#       instead of the default xco2_bc_anomaly (10 km), and
#   (2) the suffix carries a _r05 tag so results do NOT overwrite the 10 km
#       _prof runs — e.g. tabm_ocean_prof_f${F} (10 km) vs
#       tabm_ocean_prof_r05_f${F} (this script, 5 km reference).
# Ocean only, matching the r05 DE launcher.  Per fold: 1 (full) + 3 ablations = 4.
#
# NOTE: 5 km is a LOOSER reference than the 10 km default — soundings as close as
# 5 km to a cloud count as clear-sky refs, so the reference pool is larger but
# more likely cloud-contaminated (opposite direction from the 15 km set).
#
# TabM vs DE CLI: no --loss beta_nll / --norm / --dropout / --n_members /
# --batch_size / --hidden_dims (dropped); --K 16 + huber default set the model.
# --near_cloud_target is only read under --conformal, so --conformal is passed to
# mirror the DE near-cloud tail over-coverage (affects intervals only).
#
# PREREQUISITES:
#   - run curc_shell_blanca_profile_pca_fit.sh first so the ProfilePCA pkls exist.
#   - the combined parquet must contain xco2_bc_anomaly_r05 (regenerate via
#     spectral/fitting.py + build_feature_dataset.py if missing), else models.tabm
#     raises ValueError on the target column.
#
# FOLD-ARRAY JOB: each array task is an independent 1-GPU job running ONE
# date_kfold fold (--fold $SLURM_ARRAY_TASK_ID); --requeue re-submits a fold if
# the preemptable QOS preempts it.
#
# After ALL array tasks finish, aggregate (no GPU needed), e.g.:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_tabm/tabm_ocean_prof_r05_f*' --label TabM+prof+r05 \
#     --out results/model_comparison/tabm_ocean_profile_r05_kfold_agg.md

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

F=${SLURM_ARRAY_TASK_ID}
NFOLDS=5

# ── Full feature set (+profile, 5 km reference), ocean ────────────────────────
python -m models.tabm --sfc_type 0 --suffix tabm_ocean_prof_r05_f${F} \
    --K 16 --profile-pca --target 5km \
    --conformal --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Feature-set ablations (+profile, 5 km reference), ocean ───────────────────
for FS in no_xco2 no_spec no_xco2_and_spec; do
  python -m models.tabm --sfc_type 0 --suffix tabm_ocean_${FS}_prof_r05_f${F} \
      --K 16 --profile-pca --feature_set ${FS} --target 5km \
      --conformal --near_cloud_target 0.98 --mondrian_col cld_dist_km \
      --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
done

kill $GPU_MONITOR_PID 2>/dev/null || true

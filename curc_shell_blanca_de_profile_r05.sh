#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_de_profile_r05
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# ── Deep-ensemble WITH profile EOF block, targeting the 5 km clear-sky ref ─────
# Identical config to curc_shell_blanca_de_profile.sh (beta_nll, beta=1.0,
# near_cloud_target 0.98, +profile block), differing ONLY by:
#   (1) --target 5km selects xco2_bc_anomaly_r05 (5 km clear-sky reference)
#       instead of the default xco2_bc_anomaly (10 km), and
#   (2) the suffix carries a _r05 tag so results do NOT overwrite the 10 km
#       _prof runs — e.g. the comparison is
#         de_ocean_beta_nll_prof_f${F}       (10 km reference)
#         de_ocean_beta_nll_prof_r05_f${F}   (this script, 5 km reference).
#
# NOTE: 5 km is a LOOSER reference than the 10 km default — soundings as close
# as 5 km to a cloud count as clear-sky refs, so the reference pool is larger
# but more likely cloud-contaminated (opposite direction from the 15 km set).
#
# PREREQUISITES:
#   - run curc_shell_blanca_profile_pca_fit.sh first so the ProfilePCA pkls exist.
#   - the combined parquet must contain xco2_bc_anomaly_r05 (regenerate via
#     spectral/fitting.py + build_feature_dataset.py if missing), else
#     deep_ensemble raises ValueError on the target column.
#
# FOLD-ARRAY JOB: each array task is an independent 1-GPU job running ONE
# date_kfold fold (--fold $SLURM_ARRAY_TASK_ID); --requeue re-submits a fold if
# the preemptable QOS preempts it.
#
# After ALL array tasks finish, aggregate (no GPU needed), e.g.:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_deep_ensemble/de_ocean_beta_nll_prof_r05_f*' --label DeepEns+prof+r05 \
#     --out results/model_comparison/deep_ensemble_ocean_profile_r05_kfold_agg.md

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

# ── Full feature set (production config + profile, 5 km reference) ─────────────
# --target 5km selects xco2_bc_anomaly_r05.  --profile-pca and _r05 suffix are
# explicit so intent is obvious in the job log.  A/B partner: the 10 km run
# de_{surface}_beta_nll_prof_f${F} in curc_shell_blanca_de_profile.sh.
python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_beta_nll_prof_r05_f${F} \
    --profile-pca \
    --target 5km \
    --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
    --norm layer --dropout 0.1 \
    --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# python -m models.deep_ensemble --sfc_type 1 --suffix de_land_beta_nll_prof_r05_f${F} \
#     --profile-pca \
#     --target 5km \
#     --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#     --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#     --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Feature-set ablations (+profile, 5 km reference) ──────────────────────────
# Same config, each with one feature block dropped, both surfaces.  The profile
# block is ORTHOGONAL to --feature_set.  Suffix: de_{surface}_{FS}_prof_r05_f${F}.
# for FS in no_xco2 no_spec no_xco2_and_spec; do
#   python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_${FS}_prof_r05_f${F} \
#       --profile-pca --feature_set ${FS} --target 5km \
#       --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#       --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#       --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

#   python -m models.deep_ensemble --sfc_type 1 --suffix de_land_${FS}_prof_r05_f${F} \
#       --profile-pca --feature_set ${FS} --target 5km \
#       --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#       --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#       --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
# done

kill $GPU_MONITOR_PID 2>/dev/null || true

#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_de_profile_r15
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# ── Deep-ensemble WITH profile EOF block, targeting the 15 km clear-sky ref ────
# Identical config to curc_shell_blanca_de_profile.sh (beta_nll, beta=1.0,
# near_cloud_target 0.98, +profile block), differing ONLY by:
#   (1) --target 15km selects xco2_bc_anomaly_r15 (15 km clear-sky reference)
#       instead of the default xco2_bc_anomaly (10 km), and
#   (2) the suffix carries a _r15 tag so results do NOT overwrite the 10 km
#       _prof runs — e.g. the comparison is
#         de_ocean_beta_nll_prof_reg_f${F}       (10 km reference)
#         de_ocean_beta_nll_prof_reg_r15_f${F}   (this script, 15 km reference).
#
# PREREQUISITES:
#   - run curc_shell_blanca_profile_pca_fit.sh first so the ProfilePCA pkls exist.
#   - the combined parquet must contain xco2_bc_anomaly_r15 (regenerate via
#     spectral/fitting.py + build_feature_dataset.py if missing), else
#     deep_ensemble raises ValueError on the target column.
#
# FOLD-ARRAY JOB: each array task is an independent 1-GPU job running ONE
# date_kfold fold (--fold $SLURM_ARRAY_TASK_ID); --requeue re-submits a fold if
# the preemptable QOS preempts it.
#
# After ALL array tasks finish, aggregate (no GPU needed), e.g.:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_deep_ensemble/de_ocean_beta_nll_prof_reg_r15_f*' --label DeepEns+prof+r15 \
#     --out results/model_comparison/deep_ensemble_ocean_profile_r15_kfold_agg.md

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

# ── Full feature set (production config + profile, 15 km BC reference) ─────────
# --target 15km selects xco2_bc_anomaly_r15.  --profile-pca and _r15 suffix are
# explicit so intent is obvious in the job log.  A/B partner: the 10 km run
# de_{surface}_beta_nll_prof_reg_f${F} in curc_shell_blanca_de_profile.sh.
# ACTIVATED (was commented) so the arch32 A/B has an in-file bc-target 64,32
# baseline at 15 km, matching the bc×raw × arch32 symmetry in the 10 km launcher.
# python -m models.deep_ensemble --sfc_type 1 --suffix de_land_beta_nll_prof_reg_r15_f${F} \
#     --profile-pca \
#     --target 15km \
#     --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#     --norm layer --dropout 0.1 \
#     --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#     --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Architecture A/B: 32,32,32 vs the default 64,32 (15 km BC target) ──────────
# BC-target companion to the raw_r15 arch32 arm below, completing the bc×raw ×
# arch32 matrix at 15 km (mirrors the 10 km launcher).  ONLY --hidden_dims differs
# from the bc-target 64,32 baseline above; _arch32 tag keeps results distinct.
# A/B partner: de_land_beta_nll_prof_reg_r15_f${F} (64,32).
python -m models.deep_ensemble --sfc_type 1 --suffix de_land_beta_nll_prof_reg_r15_arch32_f${F} \
    --profile-pca \
    --target 15km \
    --hidden_dims 32,32,32 \
    --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
    --norm layer --dropout 0.1 \
    --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Full + profile, RAW-anomaly target (xco2_raw_anomaly_r15, 15 km ref) ──────
# Same production structure (lndo01 + profile) regressing the RAW anomaly instead
# of xco2_bc_anomaly_r15.  A/B partner: de_land_beta_nll_prof_reg_r15_f${F} (bc).
# python -m models.deep_ensemble --sfc_type 1 --suffix de_land_beta_nll_prof_reg_raw_r15_f${F} \
#     --profile-pca \
#     --target xco2_raw_anomaly_r15 \
#     --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#     --norm layer --dropout 0.1 \
#     --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#     --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Architecture A/B: 32,32,32 vs the default 64,32 (15 km raw, land) ──────────
# Mirrors the arch32 confirmation arm in curc_shell_blanca_de_profile.sh, carried
# onto the 15 km RAW-anomaly reference.  NOTE: the local 2020 hidden-dims sweep was
# OCEAN-only; this is the LAND check at the 15 km reference, the surface where the
# correction is actually driven — so the arm is genuinely informative even if the
# ocean edge (~+0.019 R², inside fold noise) doesn't transfer.  Finding: depth not
# width — every 3-layer net beat 64,32, narrowest deep net (32,32,32) won.  ONLY
# --hidden_dims differs from the raw_r15 arm above; _arch32 tag keeps results
# distinct.  A/B partner: de_land_beta_nll_prof_reg_raw_r15_f${F} (64,32).
# python -m models.deep_ensemble --sfc_type 1 --suffix de_land_beta_nll_prof_reg_raw_r15_arch32_f${F} \
#     --profile-pca \
#     --target xco2_raw_anomaly_r15 \
#     --hidden_dims 32,32,32 \
#     --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#     --norm layer --dropout 0.1 \
#     --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#     --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Feature-set ablations (+profile, 15 km reference) ─────────────────────────
# Same config, each with one feature block dropped, land only.  The profile
# block is ORTHOGONAL to --feature_set.  Suffix: de_land_{FS}_prof_r15_f${F}.
# for FS in no_xco2 no_spec no_xco2_and_spec; do
#   python -m models.deep_ensemble --sfc_type 1 --suffix de_land_${FS}_prof_r15_f${F} \
#       --profile-pca --feature_set ${FS} --target 15km \
#       --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#       --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#       --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
# done

kill $GPU_MONITOR_PID 2>/dev/null || true

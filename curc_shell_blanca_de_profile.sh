#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_de_profile
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# ── Deep-ensemble WITH the vertical-profile EOF block ─────────────────────────
# Focused A/B against the profile-less production runs in
# curc_shell_blanca_train_deep_ensemble.sh.  Covers the SAME matrix — full +
# no_xco2 + no_spec + no_xco2_and_spec, both surfaces — on the production config
# (beta_nll, beta=1.0, near_cloud_target 0.98), differing ONLY by:
#   (1) --profile-pca appends the 12 EOF PCs + 2 tropopause scalars (auto-loaded
#       from results/profile_pca/profile_pca_<surface>.pkl), and
#   (2) the suffix carries a _prof tag so results do NOT overwrite the existing
#       profile-less baselines — e.g. the comparison is
#         de_ocean_beta_nll_f${F}        (baseline, no profile)
#         de_ocean_beta_nll_prof_f${F}   (this script, +profile),
#       and likewise de_{surface}_{no_xco2,no_spec,no_xco2_and_spec}_[prof_]f${F}.
# Per fold: 2 (full) + 3 ablations × 2 surfaces = 8 runs (~baseline's 10-run/fold
# cost, comfortably under the 16h wall).
#
# PREREQUISITE: run curc_shell_blanca_profile_pca_fit.sh first so the pkls exist,
# else FeaturePipeline.fit raises FileNotFoundError and the fold dies.
#
# FOLD-ARRAY JOB: each array task is an independent 1-GPU job running ONE
# date_kfold fold (--fold $SLURM_ARRAY_TASK_ID); --requeue re-submits a fold if
# the preemptable QOS preempts it.
#
# NOTE on leakage: ProfilePCA is fit on ALL 2016-2020 dates, so under date_kfold
# its EOF basis has seen the held-out fold — a mild train/test leak in the profile
# block only.  Fine for a feature-contribution measurement; keep in mind for any
# headline held-out R2 claim.
#
# After ALL array tasks finish, aggregate (no GPU needed), e.g.:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_deep_ensemble/de_ocean_beta_nll_prof_f*' --label DeepEns+prof \
#     --out results/model_comparison/deep_ensemble_ocean_profile_kfold_agg.md

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

# ── Full feature set (production baseline + profile) ──────────────────────────
# Production loss (beta_nll, beta=1.0) + near-cloud conformal over-coverage, both
# surfaces, +profile block.  --profile-pca is explicit (opt-in) so intent is
# obvious in the job log.  A/B partner: de_{surface}_beta_nll_f${F} (no profile).
#
# Regularization: --norm layer --dropout 0.1 (arm "lndo01" of the reg ablation,
# curc_shell_blanca_de_reg_ablation.sh).  Adopted for BOTH surfaces on the nosg
# 17.8M-row data: provably better than the unregularized base on land global
# (+0.020 ppm >σ) and the land near-cloud tail (+0.032 ppm >σ), directionally
# better (within fold noise) on ocean, coverage_90 unchanged — i.e. never worse,
# better where the correction matters.  Rejected arms: bn (worse on ocean),
# cap_lndo (extra capacity no help → 64x32 not underfitting at 17.8M rows).
python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_beta_nll_prof_f${F} \
    --profile-pca \
    --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
    --norm layer --dropout 0.1 \
    --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

python -m models.deep_ensemble --sfc_type 1 --suffix de_land_beta_nll_prof_f${F} \
    --profile-pca \
    --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
    --norm layer --dropout 0.1 \
    --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Feature-set ablations (+profile) ──────────────────────────────────────────
# Same production config, each with one feature block dropped, both surfaces.
# The profile block is ORTHOGONAL to --feature_set (the no_xco2/no_spec drops never
# touch it), so these isolate each block's contribution ON TOP OF the profile block.
# A/B partners: de_{surface}_{FS}_f${F} (same ablation, no profile).
# Suffix: de_{surface}_{FS}_prof_f${F}.
# for FS in no_xco2 no_spec no_xco2_and_spec; do
#   python -m models.deep_ensemble --sfc_type 0 --suffix de_ocean_${FS}_prof_f${F} \
#       --profile-pca --feature_set ${FS} \
#       --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#       --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#       --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

#   python -m models.deep_ensemble --sfc_type 1 --suffix de_land_${FS}_prof_f${F} \
#       --profile-pca --feature_set ${FS} \
#       --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
#       --near_cloud_target 0.98 --mondrian_col cld_dist_km \
#       --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
# done

kill $GPU_MONITOR_PID 2>/dev/null || true

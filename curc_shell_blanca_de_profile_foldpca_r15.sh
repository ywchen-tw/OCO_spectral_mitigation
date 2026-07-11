#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_de_foldpca_r15
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# ── Production DE (land, 15 km target) retrained with FOLD-SPECIFIC ProfilePCA ─
# Identical config to the production run in curc_shell_blanca_de_profile_r15.sh
# (feature_set full, beta_nll beta=1.0, M=5, --norm layer --dropout 0.1,
# near_cloud_target 0.98, Mondrian cld_dist_km, date_kfold), differing ONLY by:
#   --profile-pca <fold-specific pkl> instead of the global (all-2016-2020-dates)
#   results/profile_pca/profile_pca_land.pkl.
#
# WHY: the global ProfilePCA EOF basis was fit on ALL dates, so under date_kfold
# it has seen the held-out fold — a mild unsupervised train/test leak in the
# profile block (flagged in the launcher notes and PROJECT_REVIEW).  The
# fold-specific artifacts (models.profile_pca_kfold; same split_dataframe
# date_kfold blocks, calib block excluded from the EOF fit) close it.
# Suffix carries _foldpca so nothing overwrites the current production runs:
#   de_land_beta_nll_prof_reg_r15_f${F}          (global PCA, production)
#   de_land_beta_nll_prof_reg_foldpca_r15_f${F}  (this script)
#
# PREREQUISITES:
#   - fold-specific ProfilePCA artifacts under
#     ${STORAGE}/results/profile_pca_date_kfold_2016_2020/fold{0..4}/ .
#     They exist already if the structured foldpca runs succeeded; otherwise
#     submit curc_shell_blanca_profile_pca_date_kfold_2016_2020.sh first.
#   - the combined parquet must contain xco2_bc_anomaly_r15.
#
# FOLD-ARRAY JOB: each array task is an independent 1-GPU job running ONE
# date_kfold fold (--fold $SLURM_ARRAY_TASK_ID); --requeue re-submits a fold if
# the preemptable QOS preempts it.
#
# After ALL array tasks finish, aggregate (no GPU needed):
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_deep_ensemble/de_land_beta_nll_prof_reg_foldpca_r15_f*' \
#     --label DeepEns+foldpca+r15 \
#     --out results/model_comparison/deep_ensemble_land_profile_foldpca_r15_kfold_agg.md

set -euo pipefail

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8 cuda/12.1.1
conda activate data

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:${LD_LIBRARY_PATH:-}
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

F=${SLURM_ARRAY_TASK_ID}
NFOLDS=5

STORAGE=$(python -c "from utils import get_storage_dir; print(get_storage_dir())")
PCA_ROOT="${STORAGE}/results/profile_pca_date_kfold_2016_2020"
PROFILE_PKL="${PCA_ROOT}/fold${F}/profile_pca_land.pkl"

if [[ ! -f "${PROFILE_PKL}" ]]; then
    echo "Missing fold-specific ProfilePCA: ${PROFILE_PKL}" >&2
    echo "Submit curc_shell_blanca_profile_pca_date_kfold_2016_2020.sh first." >&2
    exit 1
fi
echo "fold=${F}/${NFOLDS}  profile_pca=${PROFILE_PKL}"

python -m models.deep_ensemble --sfc_type 1 --suffix de_land_beta_nll_prof_reg_foldpca_r15_f${F} \
    --profile-pca "${PROFILE_PKL}" \
    --feature_set full --target 15km \
    --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
    --norm layer --dropout 0.1 \
    --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

# ── Feature-set ablations (+fold-specific profile, 15 km reference) ────────────
# Same config as the production run above, each with one feature block dropped,
# land only.  The profile block is ORTHOGONAL to --feature_set; each ablation
# reuses the same fold-specific ProfilePCA pkl.  Mirrors the ablation loops in
# curc_shell_blanca_de_profile_r05.sh, adapted to foldpca (fold PCA + _foldpca
# suffix) and the land/15 km reference.  Suffix: de_land_{FS}_prof_foldpca_r15_f${F}.
for FS in no_xco2 no_spec no_xco2_and_spec; do
  python -m models.deep_ensemble --sfc_type 1 --suffix de_land_${FS}_prof_foldpca_r15_f${F} \
      --profile-pca "${PROFILE_PKL}" --feature_set ${FS} --target 15km \
      --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
      --near_cloud_target 0.98 --mondrian_col cld_dist_km \
      --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
done

for FS in no_contam no_contam_and_xco2; do
  python -m models.deep_ensemble --sfc_type 1 --suffix de_land_${FS}_prof_foldpca_r15_f${F} \
      --profile-pca "${PROFILE_PKL}" --feature_set ${FS} --target 15km \
      --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
      --near_cloud_target 0.98 --mondrian_col cld_dist_km \
      --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}
# done

kill $GPU_MONITOR_PID 2>/dev/null || true

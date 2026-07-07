#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_xgb_foldpca_r15
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --array=0-4
#SBATCH --requeue

# ── Plain XGBoost mean regression (LAND, 15 km target) — SAME protocol as the
# production DE, for the reviewer-proof baseline table
# (log/LINREG_XGB_BASELINE_PLAN_2026-07-07.md).
# Land twin of curc_shell_blanca_xgb_foldpca_r05.sh: --sfc_type 1 --target 15km,
# fold-specific ProfilePCA profile_pca_land.pkl.  Same-protocol comparison partner to
#   de_land_beta_nll_prof_reg_foldpca_r15_f${F}.
# Suffix: xgb_land_full_prof_foldpca_r15_f${F}
#
# PREREQUISITES:
#   - fold-specific ProfilePCA under ${STORAGE}/results/profile_pca_date_kfold_2016_2020/fold{0..4}/ .
#   - the combined parquet must contain xco2_bc_anomaly_r15.
#
# After ALL array tasks finish, aggregate:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_gbdt/xgb_land_full_prof_foldpca_r15_f*' \
#     --label XGB+foldpca+r15 \
#     --out results/model_comparison/xgb_land_foldpca_r15_kfold_agg.md

set -euo pipefail

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:${LD_LIBRARY_PATH:-}
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-16}
export MKL_NUM_THREADS=${SLURM_NTASKS:-16}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

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

python -m models.gbdt_baselines --model xgboost --objective mean \
    --sfc_type 1 --suffix xgb_land_full_prof_foldpca_r15_f${F} \
    --profile-pca "${PROFILE_PKL}" \
    --feature_set full --target 15km --calib_frac 0.15 --n_jobs ${SLURM_NTASKS:-16} \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

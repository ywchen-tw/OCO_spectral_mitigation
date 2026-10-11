#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_train_linreg_foldpca_r05
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --array=0-4
#SBATCH --requeue

# ── Ridge linear baseline (OCEAN, 5 km target) — SAME protocol as the production
# DE, for the reviewer-proof baseline table (log/archive/LINREG_XGB_BASELINE_PLAN_2026-07-07.md).
# Mirrors curc_shell_blanca_de_profile_foldpca_r05.sh exactly except:
#   - CPU-only (no --gres=gpu; Ridge is a closed-form solve, no CUDA)
#   - module load drops cuda
#   - python -m models.linear_baseline (Ridge; alpha grid selected on the calib
#     block only — NEVER TCCON) instead of models.deep_ensemble
# Fold-specific ProfilePCA (leakage-safe) — the SAME artifacts the foldpca DE uses,
# so this is a same-protocol comparison partner to
#   de_ocean_beta_nll_prof_reg_foldpca_r05_f${F}.
# Suffix: linreg_ocean_full_prof_foldpca_r05_f${F}
#
# PREREQUISITES:
#   - fold-specific ProfilePCA artifacts under
#     ${STORAGE}/results/profile_pca_date_kfold_2016_2020/fold{0..4}/ .
#   - the combined parquet must contain xco2_bc_anomaly_r05.
#
# After ALL array tasks finish, aggregate (no GPU needed):
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_linear_baseline/linreg_ocean_full_prof_foldpca_r05_f*' \
#     --label LinReg+foldpca+r05 \
#     --out results/model_comparison/linreg_ocean_foldpca_r05_kfold_agg.md

set -euo pipefail

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:${LD_LIBRARY_PATH:-}
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-8}
export MKL_NUM_THREADS=${SLURM_NTASKS:-8}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

F=${SLURM_ARRAY_TASK_ID}
NFOLDS=5

STORAGE=$(python -c "from utils import get_storage_dir; print(get_storage_dir())")
PCA_ROOT="${STORAGE}/results/profile_pca_date_kfold_2016_2020"
PROFILE_PKL="${PCA_ROOT}/fold${F}/profile_pca_ocean.pkl"

if [[ ! -f "${PROFILE_PKL}" ]]; then
    echo "Missing fold-specific ProfilePCA: ${PROFILE_PKL}" >&2
    echo "Submit curc_shell_blanca_profile_pca_date_kfold_2016_2020.sh first." >&2
    exit 1
fi
echo "fold=${F}/${NFOLDS}  profile_pca=${PROFILE_PKL}"

python -m models.linear_baseline --sfc_type 0 --suffix linreg_ocean_full_prof_foldpca_r05_f${F} \
    --profile-pca "${PROFILE_PKL}" \
    --feature_set full --target 5km \
    --alphas 0.1,1,10,100 --calib_frac 0.15 \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

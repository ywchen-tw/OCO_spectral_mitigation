#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_tabm_ocean_f1
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --requeue

# BACKFILL the one missing fold: tabm_ocean_full_f1 / tabm_ocean_full_contam_f1.
# The existing tabm_ocean_full_f{0,2,3,4} use the ORIGINAL DEFAULT structure
# (K=16, d_model=256, n_layers=4, dropout=0.2, huber=1.0, batch=8192, 500ep) —
# so this uses PURE DEFAULTS (--K 16, no --config) to stay consistent with them.
# Do NOT pass --config tabm_tuned_ocean*.json here (that would make f1 a tuned
# outlier vs the default-structure siblings).  After this lands, re-run:
#   PYTHONPATH=src python -m models.aggregate_folds \
#     --dirs 'results/model_tabm/tabm_ocean_full_f*'              --label 'TabM full' \
#     --dirs 'results/model_tabm/tabm_ocean_full_contam_f*'       --label 'TabM full_contam' \
#     --dirs 'results/model_deep_ensemble/de_ocean_beta_nll_f*'   --label 'DE full' \
#     --dirs 'results/model_deep_ensemble/de_ocean_full_contam_f*' --label 'DE full_contam' \
#     --out results/model_comparison/tabm_vs_de_full_contam_ocean.md
# for the clean 5/5-fold ocean comparison.

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

for FS in full; do
  python -m models.tabm --sfc_type 0 --suffix tabm_ocean_${FS}_f1 --K 16 \
    --feature_set ${FS} --val_split date_kfold --n_folds 5 --fold 1
done

kill $GPU_MONITOR_PID 2>/dev/null || true

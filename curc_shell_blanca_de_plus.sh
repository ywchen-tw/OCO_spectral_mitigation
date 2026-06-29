#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_de_plus
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --requeue

# DE++ : larger + HETEROGENEOUS deep ensemble, vs the M=5 production DE.
#
# Motivation (local 12-date fold-0 ocean gate, beta=1.0, full_contam):
#   tail-5% R2 (near & bottom-5%) is the production DE's residual weakness.  It
#   responds to ENSEMBLE SIZE far more than the saturated bulk — the tail is the
#   high-variance regime where averaging decorrelated members keeps paying:
#       M=3  -> tail5% R2 0.018   (production-ish)
#       M=6  -> 0.043
#       M=10 -> 0.054 (homogeneous)
#       M=10 -> 0.066 (HETEROGENEOUS)   <- best
#   The bulk (global / near-cloud R2) plateaus by ~M=6 (+0.002 past it); only the
#   tail keeps climbing.  HETEROGENEITY (varying per-member width/depth -> more
#   decorrelation) adds +0.013 tail5% R2 at M=10 ON TOP of size, and its benefit
#   GREW with M (M=6 +0.007 -> M=10 +0.013).  Hence M=10 + heterogeneous.
#
# This job runs BOTH arch modes so the heterogeneity effect is measured AT FULL
# SCALE with fold error bars (the local result is single-fold / 275k rows; the
# marginal-member benefit may shrink or grow on the full ~5M-row data):
#   deplus_{surface}_homog_f{F}   — M=10 identical (64,32) members (size-only arm)
#   deplus_{surface}_hetero_f{F}  — M=10 varied-architecture members (DE++)
# Everything else matches production: beta_nll(1.0) + Mondrian-CQR by cld_dist_km
# with --near_cloud_target 0.98; ocean = no snow, land = --include_snow (matches
# the de_land_fc_snowdata baseline).  One fold per array task; 4 runs/task.
#
# After all folds finish (no GPU needed).  BOTH arms (homog M=10 and hetero M=10)
# are full first-class runs — each gets a vs-production verdict, plus the matched
# hetero-vs-homog ablation:
#   # 1) each M=10 arm vs production DE (M=5) — the "did size/diversity help" verdict
#   for ARM in homog hetero; do
#     PYTHONPATH=src python -m models.aggregate_folds \
#       --dirs "results/model_deep_ensemble/deplus_ocean_${ARM}_f*"  --label "DE++_${ARM}" \
#       --dirs 'results/model_deep_ensemble/de_ocean_full_contam_f*' --label DE_prod \
#       --out "results/model_comparison/deplus_${ARM}_vs_prod_ocean.md"
#     PYTHONPATH=src python -m models.aggregate_folds \
#       --dirs "results/model_deep_ensemble/deplus_land_${ARM}_f*"   --label "DE++_${ARM}" \
#       --dirs 'results/model_deep_ensemble/de_land_fc_snowdata_f*'  --label DE_prod \
#       --out "results/model_comparison/deplus_${ARM}_vs_prod_land.md"
#   done
#   # 2) heterogeneity ablation AT SCALE — hetero vs homog at matched M=10
#   for SURF in ocean land; do
#     PYTHONPATH=src python -m models.aggregate_folds \
#       --dirs "results/model_deep_ensemble/deplus_${SURF}_hetero_f*" --label hetero \
#       --dirs "results/model_deep_ensemble/deplus_${SURF}_homog_f*"  --label homog \
#       --out "results/model_comparison/deplus_hetero_ablation_${SURF}.md"
#   done
# NOTE: aggregate_folds reads the alphabetically-first metrics json (de_mondrian_*),
# so compare the per-fold near_cloud_tail stratified CSVs for the tail verdict.

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
# Production-matched config + M=10.  Heterogeneous pool: 6 distinct width/depth
# specs cycled across the 10 members (members 7-10 reuse the first 4) -> diverse
# capacities, more decorrelation.
ARCHS="64,32;128,64;256,128,64;96,48;160,80;64,64,32"
COMMON="--loss beta_nll --beta 1.0 --n_members 10 --batch_size 8192 \
        --feature_set full_contam --near_cloud_target 0.98 --mondrian_col cld_dist_km \
        --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}"

# ── homogeneous M=10 (size-only arm; the heterogeneity-ablation control) ───────
# python -m models.deep_ensemble --sfc_type 0 --suffix deplus_ocean_homog_f${F} ${COMMON}
# python -m models.deep_ensemble --sfc_type 1 --suffix deplus_land_homog_f${F}  --include_snow ${COMMON}

# ── heterogeneous M=10 (DE++; the production candidate) ────────────────────────
python -m models.deep_ensemble --sfc_type 0 --suffix deplus_ocean_hetero_f${F} --member_archs "${ARCHS}" ${COMMON}
# python -m models.deep_ensemble --sfc_type 1 --suffix deplus_land_hetero_f${F}  --include_snow --member_archs "${ARCHS}" ${COMMON}

kill $GPU_MONITOR_PID 2>/dev/null || true

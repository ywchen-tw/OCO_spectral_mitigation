#!/bin/env bash

#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_de_keep_only
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
#SBATCH --requeue

# ── KEEP-ONLY feature-set ablation (KSS review E4, decision D9, 2026-10-09) ────
# The existing drop-only ablation never trains a model on the path-length
# features alone: every variant keeps the retrieved ACOS state of the
# meteorology-surface group, the profile EOF block and the albedo-subtracted
# gamma features.  These arms keep ONLY the named groups (see the KEEP-ONLY note
# above _FEATURE_SETS in src/models/pipeline.py):
#
#   keep_spec_geom  path-length features (gamma, not gamma - albedo) + geometry/L1B + fp one-hot
#   keep_geom       geometry/L1B + fp one-hot                        (skill floor)
#   keep_xco2_geom  xco2_raw_minus_apriori + geometry/L1B + fp one-hot (optional ACOS-side match)
#
# keep_spec_geom - keep_geom = path-length skill with no ACOS state.
#
# Same config as the production / drop-ablation DE (beta_nll beta=1.0, M=5,
# --norm layer --dropout 0.1, near_cloud_target 0.98, Mondrian cld_dist_km,
# date_kfold 5 folds, seed 42; ocean target 5km = xco2_bc_anomaly_r05, land
# target 15km = xco2_bc_anomaly_r15), EXCEPT that the profile EOF block is OFF
# (--no-profile-pca), because it is part of what the keep-only arms remove.
# Suffix: de_{ocean|land}_<variant>_noprof_r{05|15}_f<fold>.
#
# ARRAY LAYOUT: one task = one (variant, surface, fold) deep ensemble.
#   task = 10*variant_index + 5*surface + fold    (surface 0 = ocean, 1 = land)
#   0-9   keep_spec_geom   (0-4 ocean f0-f4, 5-9 land f0-f4)
#   10-19 keep_geom
#   20-29 keep_xco2_geom   (optional; not in the default --array)
# Submit:
#   sbatch curc_shell_blanca_de_keep_only_ablation.sh                  # two core variants
#   sbatch --array=20-29 curc_shell_blanca_de_keep_only_ablation.sh    # optional third
#   sbatch --array=7 curc_shell_blanca_de_keep_only_ablation.sh        # rerun one task
# A task whose run_summary.json already exists exits at once, so a --requeue
# after preemption, or a resubmit of the whole array, never overwrites a finished
# ensemble.  Set FORCE=1 to retrain anyway.
#
# After ALL tasks finish:
#   1. Held-out fold aggregation (no GPU):
#        for V in keep_spec_geom keep_geom; do
#          PYTHONPATH=src python -m models.aggregate_folds \
#            --dirs "results/model_deep_ensemble/de_ocean_${V}_noprof_r05_f*" \
#            --label "DE ${V} ocean" \
#            --out results/model_comparison/deep_ensemble_ocean_${V}_noprof_r05_kfold_agg.md
#          PYTHONPATH=src python -m models.aggregate_folds \
#            --dirs "results/model_deep_ensemble/de_land_${V}_noprof_r15_f*" \
#            --label "DE ${V} land" \
#            --out results/model_comparison/deep_ensemble_land_${V}_noprof_r15_kfold_agg.md
#        done
#   2. TCCON trees (99 station-days):  bash workspace/build_ablation_variant_trees.sh keep_spec_geom
#      (and keep_geom), then point tccon_comparison_report.py --ak-harmonize at
#      results/model_comparison/deep_ensemble/de_prof_mix_<variant>, as for the drop arms.
#   3. Ablation write-up:  PYTHONPATH=src python workspace/make_featureset_ablation_doc.py
#      (picks up any keep_* variant whose TCCON tree exists).

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

VARIANTS=(keep_spec_geom keep_geom keep_xco2_geom)
NFOLDS=5
IDX=${SLURM_ARRAY_TASK_ID}
VI=$(( IDX / 10 ))
SFC=$(( (IDX % 10) / 5 ))
F=$(( IDX % 5 ))
if (( VI >= ${#VARIANTS[@]} )); then
    echo "array index ${IDX} is out of range (0-$(( 10 * ${#VARIANTS[@]} - 1 )))" >&2
    exit 1
fi
V=${VARIANTS[$VI]}

if (( SFC == 0 )); then
    SURF=ocean; TARGET=5km;  RTAG=r05
else
    SURF=land;  TARGET=15km; RTAG=r15
fi
SUFFIX=de_${SURF}_${V}_noprof_${RTAG}_f${F}

STORAGE=$(python -c "from utils import get_storage_dir; print(get_storage_dir())")
DONE_FILE="${STORAGE}/results/model_deep_ensemble/${SUFFIX}/run_summary.json"
if [[ -f "${DONE_FILE}" && "${FORCE:-0}" != 1 ]]; then
    echo "already finished: ${DONE_FILE} (set FORCE=1 to retrain)"
    exit 0
fi
echo "task=${IDX}  variant=${V}  surface=${SURF}  fold=${F}/${NFOLDS}  suffix=${SUFFIX}"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

python -m models.deep_ensemble --sfc_type ${SFC} --suffix ${SUFFIX} \
    --no-profile-pca --feature_set ${V} --target ${TARGET} \
    --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
    --norm layer --dropout 0.1 \
    --near_cloud_target 0.98 --mondrian_col cld_dist_km \
    --val_split date_kfold --n_folds ${NFOLDS} --fold ${F}

kill $GPU_MONITOR_PID 2>/dev/null || true

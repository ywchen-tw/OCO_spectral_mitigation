#!/bin/env bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_tccon_keep_only
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --array=0-2
#SBATCH --requeue

# ── TCCON evaluation of the keep-only deep ensembles (KSS E4 / D9, 2026-10-11) ──
# Step 2 of curc_shell_blanca_de_keep_only_ablation.sh, written as a launcher.
# Same steps as curc_shell_blanca_tccon_full_noprof.sh, one array task per arm:
#   0  keep_spec_geom   path-length features + geometry/L1B + fp one-hot
#   1  keep_geom        geometry/L1B + fp one-hot (skill floor)
#   2  keep_xco2_geom   xco2_raw_minus_apriori + geometry/L1B + fp one-hot
# Each task
#   1. rebuilds every TCCON case's plot_data.parquet with that arm's models
#      -> results/model_comparison/deep_ensemble/de_prof_mix_<arm>/
#   2. runs the AK-harmonized TCCON report on that tree
# Afterwards (any machine with pandas):
#   PYTHONPATH=src python workspace/make_featureset_ablation_doc.py
# which adds every keep_* arm whose tccon_metrics_ak_r100km.csv exists.
#
# Submit:  sbatch curc_shell_blanca_tccon_keep_only.sh
#          sbatch --array=0 curc_shell_blanca_tccon_keep_only.sh            # one arm
#          sbatch --export=ALL,SKIP_BUILD=1 curc_shell_blanca_tccon_keep_only.sh  # report only
# Report only, no batch job (run on a compile node, e.g. acompile, not a login node):
#          SKIP_BUILD=1 SLURM_ARRAY_TASK_ID=0 bash curc_shell_blanca_tccon_keep_only.sh

set -uo pipefail

# Environment.  Lmod failed on the node for this job (2026-10-11, "Please check
# the spelling or version number"), so the modules are tried one by one and the
# job falls back to the data env's own python, which carries its own HDF5/netCDF
# libraries (the LD_LIBRARY_PATH line below already prefers them).
ENV_DIR=/projects/yuch8913/software/anaconda/envs/data
echo "node $(hostname)  $(date)"
module purge 2>/dev/null || true
for MOD in anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8; do
    module load "$MOD" 2>/dev/null || echo "module load $MOD FAILED"
done
conda activate data 2>/dev/null || echo "conda activate data FAILED, using $ENV_DIR/bin directly"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH=$ENV_DIR/lib:${LD_LIBRARY_PATH:-}
echo "python: $(command -v python)"
python -c "import numpy, pandas, pyarrow" || { echo "data env python not usable" >&2; exit 1; }
set -e
export HDF5_USE_FILE_LOCKING=FALSE
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

# Run version (log/PIPELINE_FIX_RERUN_PLAN_2026-10-14.md): v2 = unsmoothed gamma,
# fill/spike imputation, RobustClipScaler, no t700.  The tag sits BEFORE the fold
# tag, so v1 globs (..._f*) never match v2 dirs and v2 runs never overwrite v1
# checkpoints.  RUN_VERSION= (empty) reproduces the v1 names.
export RUN_VERSION=${RUN_VERSION-v2}
RV=${RUN_VERSION:+_${RUN_VERSION}}

VARIANTS=(keep_spec_geom keep_geom keep_xco2_geom)
VARIANT=${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}

DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
TREE="$DATA_ROOT"/results/model_comparison/deep_ensemble/de_prof_mix_${VARIANT}${RV}

# All five folds of both surfaces must exist, or the tree mixes in nothing.
for SURF_TAG in ocean_${VARIANT}_noprof_r05 land_${VARIANT}_noprof_r15; do
    N=$(ls -d "$DATA_ROOT"/results/model_deep_ensemble/de_${SURF_TAG}${RV}_f* 2>/dev/null | wc -l)
    if [[ "$N" -ne 5 ]]; then
        echo "ERROR: found $N of 5 fold dirs for de_${SURF_TAG}${RV}_f*" >&2
        exit 1
    fi
done

# 1. Build the variant tree (replays the active run_case lines of
#    curc_shell_blanca_plot_corr_xco2_deepens.sh, TCCON-available cases only).
if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    bash workspace/build_ablation_variant_trees.sh "$VARIANT"
fi

# 2. TCCON report on the tree, same settings as the drop arms and full_noprof.
python workspace/tccon_comparison_report.py \
    --script curc_shell_blanca_plot_corr_xco2_deepens.sh \
    --out-base "$TREE" --output-dir "$TREE" \
    --ak-harmonize --fname-suffix _r100km

echo "TCCON ${VARIANT} report -> $TREE"

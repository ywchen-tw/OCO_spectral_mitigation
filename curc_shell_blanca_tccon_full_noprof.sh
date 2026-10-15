#!/bin/env bash

#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_tccon_full_noprof
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue

# ── TCCON evaluation of the full no-profile-EOF deep ensemble (2026-10-10) ─────
# Model-size check against Mauceri et al. (2023): the date-kfold CV of the
# full_noprof arm (array 30-39 of curc_shell_blanca_de_keep_only_ablation.sh)
# matches production (ocean RMSE 0.396 vs 0.396, land 0.542 vs 0.536).  This job
# repeats the TCCON side on the same 99 station-days as Table 4:
#   1. rebuild every TCCON case's plot_data.parquet with the full_noprof models
#      -> results/model_comparison/deep_ensemble/de_prof_mix_full_noprof/
#   2. run the AK-harmonized TCCON report on that tree
# Afterwards (any machine with pandas):
#   PYTHONPATH=src python workspace/make_featureset_ablation_doc.py
# which adds full_noprof as a column next to full and the drop/keep arms.
#
# Submit:  sbatch curc_shell_blanca_tccon_full_noprof.sh

set -euo pipefail

# Modules one by one, data-env python as the fallback (see curc_env.sh).
source /projects/yuch8913/OCO_spectral_mitigation/curc_env.sh
curc_env || exit 1
export HDF5_USE_FILE_LOCKING=FALSE
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
# Run version (log/PIPELINE_FIX_RERUN_PLAN_2026-10-14.md): v2 = unsmoothed gamma,
# fill/spike imputation, RobustClipScaler, no t700.  The tag sits BEFORE the fold
# tag, so v1 globs (..._f*) never match v2 dirs and v2 runs never overwrite v1
# checkpoints.  RUN_VERSION= (empty) reproduces the v1 names.
export RUN_VERSION=${RUN_VERSION-v2}
RV=${RUN_VERSION:+_${RUN_VERSION}}

TREE="$DATA_ROOT"/results/model_comparison/deep_ensemble/de_prof_mix_full_noprof${RV}

# 1. Build the variant tree (replays the active run_case lines of
#    curc_shell_blanca_plot_corr_xco2_deepens.sh, TCCON-available cases only).
#    SKIP_BUILD=1 reruns only the report on an existing tree:
#      sbatch --export=ALL,SKIP_BUILD=1 curc_shell_blanca_tccon_full_noprof.sh
if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    bash workspace/build_ablation_variant_trees.sh full_noprof
fi

# 2. TCCON report on the tree, same settings as the drop/keep arms.
python workspace/tccon_comparison_report.py \
    --script curc_shell_blanca_plot_corr_xco2_deepens.sh \
    --out-base "$TREE" --output-dir "$TREE" \
    --ak-harmonize

echo "TCCON full_noprof report -> $TREE"

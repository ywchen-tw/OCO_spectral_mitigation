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

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:${LD_LIBRARY_PATH:-}
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
fi
export HDF5_USE_FILE_LOCKING=FALSE
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:${PYTHONPATH:-}

DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
TREE="$DATA_ROOT"/results/model_comparison/deep_ensemble/de_prof_mix_full_noprof

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

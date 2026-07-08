#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_atom_baseline
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable

# ─────────────────────────────────────────────────────────────────────────────
# ATom ocean-glint BASELINE (linreg | xgb) correction runner — the ocean analog
# of curc_shell_blanca_atom_deepens.sh for the two reviewer baselines.
#
# Applies the ocean r05 baseline (build_baseline_plot_data.py --model-kind) to the
# OCO-2 footprints on each ATom-coincident date, emitting plot_data.parquet with
# <kind>_corrected_xco2, then re-scores the SAME aircraft pseudo-columns as the DE
# run via atom_pseudo_column.py (reusing the DE tree's model-independent
# atom_merged/ profiles).  Output goes to the baseline's own model_comparison
# subtree so it never collides with the DE case.
#
# Pick the model with the first positional arg (default linreg):
#     bash workspace/ATom_analysis/curc_shell_blanca_atom_baseline.sh linreg
#     bash workspace/ATom_analysis/curc_shell_blanca_atom_baseline.sh xgb
#
# Submit from the REPO ROOT.  Runs locally (models live under results/).
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$(uname -s)" == "Linux" ]]; then
    module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
    conda activate data
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$REPO_ROOT}"
export PYTHONPATH=src:${PYTHONPATH:-}

DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
export OCO2_DATAROOT="$DATA_ROOT"

# ─── model kind + dirs ────────────────────────────────────────────────────────
MODEL_KIND="${1:-linreg}"
case "$MODEL_KIND" in
    linreg)
        MODEL_TAG=linreg_prof_foldpca_o05l15
        OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_linear_baseline/linreg_ocean_full_prof_foldpca_r05_f*) ;;
    xgb)
        MODEL_TAG=xgb_prof_foldpca_o05l15
        OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_gbdt/xgb_ocean_full_prof_foldpca_r05_f*) ;;
    *)  echo "unknown model kind '$MODEL_KIND' (expected linreg|xgb)"; exit 2 ;;
esac
CORR_COL=${MODEL_KIND}_corrected_xco2

CSV_DIR="$DATA_ROOT"/results/csv_collection
OUT_BASE="$DATA_ROOT"/results/model_comparison/${MODEL_KIND}/${MODEL_TAG}/atom
# atom_merged/ is model-independent → reuse the DE tree's already-built profiles.
DE_ATOM="$DATA_ROOT"/results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/atom
MERGED_DIR="$DE_ATOM/atom_merged"

# ATom-coincident OCO-2 dates (same set as the DE launcher).
DATES=(2017-01-26 2017-02-04 2017-02-06 2017-02-10 2017-10-09 2017-10-20 2017-10-27 2018-05-12)

echo "ATom baseline run: kind=$MODEL_KIND tag=$MODEL_TAG corr=$CORR_COL"
echo "  ocean model dirs: ${OCEAN_MODEL_DIRS[*]}"
echo "  out base:         $OUT_BASE"

for date in "${DATES[@]}"; do
    input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    outdir="$OUT_BASE/combined_${date}_atom"
    plotdata="$outdir/plot_data.parquet"
    echo ""
    echo "############ ATOM BASELINE $date  (ocean, $MODEL_KIND)  ############"
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"; continue
    fi
    mkdir -p "$outdir"
    python workspace/build_baseline_plot_data.py --model-kind "$MODEL_KIND" \
        --ocean-model-dir "${OCEAN_MODEL_DIRS[@]}" \
        --input "$input" --output "$plotdata" \
        || { echo "  build failed"; continue; }
    echo "  wrote $plotdata"
done

# ─── Stage 2/3 pseudo-column vs ATom (reuse DE atom_merged; baseline corr col) ──
echo ""
echo "############ ATOM pseudo-column ($MODEL_KIND) ############"
python workspace/ATom_analysis/atom_pseudo_column.py \
    --corr-col "$CORR_COL" \
    --out-base "$OUT_BASE" \
    --merged-dir "$MERGED_DIR" \
    --model-label "${MODEL_KIND}-corrected" \
    --radius-km 100 --window-min 120 --min-n 3

echo ""
echo "ATom $MODEL_KIND done → $OUT_BASE/atom_pseudo_column_results.csv"

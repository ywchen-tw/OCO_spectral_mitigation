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
# ATom ocean-glint NON-DE model correction runner — the ocean analog of
# curc_shell_blanca_atom_deepens.sh for the reviewer baselines (linreg/xgb) and the
# alternative correction models (tabm/structured).
#
# Applies the model's ocean r05 correction (the kind-specific build_*_plot_data.py)
# to the OCO-2 footprints on each ATom-coincident date, emitting plot_data.parquet
# with <kind>_corrected_xco2, then re-scores the SAME aircraft pseudo-columns as the
# DE run via atom_pseudo_column.py (reusing the DE tree's model-independent
# atom_merged/ profiles).  Output goes to each model's own model_comparison subtree.
#
# Pick the model with the first positional arg (default linreg):
#     bash workspace/ATom_analysis/curc_shell_blanca_atom_baseline.sh linreg
#     bash workspace/ATom_analysis/curc_shell_blanca_atom_baseline.sh xgb
#     bash workspace/ATom_analysis/curc_shell_blanca_atom_baseline.sh tabm
#     bash workspace/ATom_analysis/curc_shell_blanca_atom_baseline.sh structured
#
# Submit from the REPO ROOT.  Runs locally (models live under results/).
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$(uname -s)" == "Linux" ]]; then
    module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
    conda activate data
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    # macOS: pip xgboost in ml310 lacks libomp.dylib — borrow one from a sibling
    # conda env via DYLD_FALLBACK (nothing installed; no-op if already set).
    if [[ -z "${DYLD_FALLBACK_LIBRARY_PATH:-}" ]]; then
        for _d in "$HOME"/miniforge3/envs/*/lib; do
            [[ -f "$_d/libomp.dylib" ]] && export DYLD_FALLBACK_LIBRARY_PATH="$_d" && break
        done
    fi
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
# Each kind sets: MODEL_TAG, OUT_SUBDIR (model_comparison/<sub>/), CORR_COL,
# BUILDER (the build_*_plot_data.py that emits <corr_col>), OCEAN_MODEL_DIRS (r05).
MODEL_KIND="${1:-linreg}"
case "$MODEL_KIND" in
    linreg)
        MODEL_TAG=linreg_prof_foldpca_o05l15; OUT_SUBDIR=linreg
        CORR_COL=linreg_corrected_xco2
        BUILDER=(workspace/build_baseline_plot_data.py --model-kind linreg)
        OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_linear_baseline/linreg_ocean_full_prof_foldpca_r05_f*) ;;
    xgb)
        MODEL_TAG=xgb_prof_foldpca_o05l15; OUT_SUBDIR=xgb
        CORR_COL=xgb_corrected_xco2
        BUILDER=(workspace/build_baseline_plot_data.py --model-kind xgb)
        OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_gbdt/xgb_ocean_full_prof_foldpca_r05_f*) ;;
    tabm)
        MODEL_TAG=tabm_prof_o05l15; OUT_SUBDIR=tabm
        CORR_COL=tabm_corrected_xco2
        BUILDER=(workspace/build_tabm_plot_data.py)
        OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_tabm/tabm_ocean_prof_r05_f*) ;;
    structured)
        # NOTE: the LOCAL structured ocean r05 folds are the NON-regime variant; the
        # regime TCCON report uses regime_* (whose ocean folds are trained on CURC).
        # On CURC, override OCEAN_STEM/MODEL_TAG to match the regime run.
        MODEL_TAG="${STRUCT_MODEL_TAG:-structured_shared_foldpca_o05l15_m5}"; OUT_SUBDIR=structured_residual
        CORR_COL=structured_residual_corrected_xco2
        BUILDER=(workspace/build_structured_residual_plot_data.py)
        OCEAN_STEM="${STRUCT_OCEAN_STEM:-de2016_2020_structured_shared_h64x32_b8_foldpca_r05_ocean}"
        STRUCT_ROOT="${STRUCT_MODEL_ROOT:-$DATA_ROOT/results/model_structured_dcn_ensemble}"
        OCEAN_MODEL_DIRS=("$STRUCT_ROOT"/"${OCEAN_STEM}"_f*) ;;
    *)  echo "unknown model kind '$MODEL_KIND' (expected linreg|xgb|tabm|structured)"; exit 2 ;;
esac

CSV_DIR="$DATA_ROOT"/results/csv_collection
OUT_BASE="$DATA_ROOT"/results/model_comparison/${OUT_SUBDIR}/${MODEL_TAG}/atom
# atom_merged/ is model-independent → reuse the DE tree's already-built profiles.
DE_ATOM="$DATA_ROOT"/results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_foldpca_o05l15_m5/atom
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
    python "${BUILDER[@]}" \
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

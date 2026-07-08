#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_ship_baseline
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable

# ─────────────────────────────────────────────────────────────────────────────
# Shipborne EM27/SUN ocean-glint BASELINE (linreg | xgb) comparison runner — the
# ocean analog of curc_shell_blanca_ship_deepens.sh for the two reviewer baselines.
#
# For each ship-coincident date it (a) applies the ocean r05 baseline
# (build_baseline_plot_data.py --model-kind) to the OCO-2 ocean footprints, emitting
# plot_data.parquet with <kind>_corrected_xco2, then (b) draws the per-case ship
# figure and (c) the cross-case summary CSV/plot — reading the baseline corr column.
# Output goes to the baseline's own model_comparison subtree.
#
# Pick the model with the first positional arg (default linreg; linreg|xgb|tabm|structured):
#     bash workspace/Ship_analysis/curc_shell_blanca_ship_baseline.sh linreg
#     bash workspace/Ship_analysis/curc_shell_blanca_ship_baseline.sh xgb
#     bash workspace/Ship_analysis/curc_shell_blanca_ship_baseline.sh tabm
#     bash workspace/Ship_analysis/curc_shell_blanca_ship_baseline.sh structured
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
        # LOCAL structured ocean r05 folds are the NON-regime variant; on CURC override
        # STRUCT_OCEAN_STEM/STRUCT_MODEL_TAG/STRUCT_MODEL_ROOT to match the regime run.
        MODEL_TAG="${STRUCT_MODEL_TAG:-structured_shared_foldpca_o05l15_m5}"; OUT_SUBDIR=structured_residual
        CORR_COL=structured_residual_corrected_xco2
        BUILDER=(workspace/build_structured_residual_plot_data.py)
        OCEAN_STEM="${STRUCT_OCEAN_STEM:-de2016_2020_structured_shared_h64x32_b8_foldpca_r05_ocean}"
        STRUCT_ROOT="${STRUCT_MODEL_ROOT:-$DATA_ROOT/results/model_structured_dcn_ensemble}"
        OCEAN_MODEL_DIRS=("$STRUCT_ROOT"/"${OCEAN_STEM}"_f*) ;;
    *)  echo "unknown model kind '$MODEL_KIND' (expected linreg|xgb|tabm|structured)"; exit 2 ;;
esac

CSV_DIR="$DATA_ROOT"/results/csv_collection
OUT_BASE="$DATA_ROOT"/results/model_comparison/${OUT_SUBDIR}/${MODEL_TAG}/ship

RADIUS_KM="${RADIUS_KM:-100}"
WINDOW_MIN="${WINDOW_MIN:-120}"

ship_case() {
    local date="$1" ship="$2" lonmin="$3" lonmax="$4" latmin="$5" latmax="$6"
    local vmin="$7" vmax="$8" note="${9:-}"
    local input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    local outdir="$OUT_BASE/combined_${date}_${ship}"
    local plotdata="$outdir/plot_data.parquet"

    echo ""
    echo "############ SHIP CASE $date  ($ship, ocean, $MODEL_KIND)  ############"
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"; return
    fi
    mkdir -p "$outdir"

    # (a) apply ocean model → plot_data.parquet
    python "${BUILDER[@]}" \
        --ocean-model-dir "${OCEAN_MODEL_DIRS[@]}" \
        --input "$input" --output "$plotdata" \
        || { echo "  build failed"; return; }

    # (b) ship-native comparison figure (baseline corr column)
    python workspace/Ship_analysis/plot_ship_comparison.py \
        --plot-data "$plotdata" --ship-tag "$ship" --date "$date" \
        --corr-col "$CORR_COL" \
        --lon-range "$lonmin" "$lonmax" --lat-range "$latmin" "$latmax" \
        --vmin "$vmin" --vmax "$vmax" \
        --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" \
        --modis-auto --modis-which aqua \
        --output-dir "$outdir" \
        || { echo "  plot failed"; return; }
    echo "  done → $outdir  ($note)"
}

# ═══════════════════════ ship cases (same boxes as the DE launcher) ═══════════
#           DATE(OCO)    SHIP    LON_MIN   LON_MAX  LAT_MIN  LAT_MAX   VMIN   VMAX   NOTE
ship_case  2019-06-09   so268    -152.62  -152.15    29.36    30.94   412.0  414.5  "494 fp, 494 near<=10km — full near-cloud"
ship_case  2019-06-14   so268    -170.75  -170.54    28.94    29.66   409.5  413.0  "111 fp, 73 near<=10km"
ship_case  2019-06-22   so268     152.36   152.80    27.43    28.96   411.5  412.5  "460 fp, 1 near<=10km — clear-sky reference"
ship_case  2021-03-15   mr2101    140.06   140.18    26.24    26.60   416.0  417.5  "6 fp, 5 near<=10km — sparse"

# ── cross-case summary (per-case bias dumbbell + bias-vs-cloud-distance, ±1σ) ──
echo ""
echo "############ SHIP SUMMARY ($MODEL_KIND) ############"
python workspace/Ship_analysis/plot_ship_summary.py --out-base "$OUT_BASE" \
    --corr-col "$CORR_COL" --model-label "${MODEL_KIND}-corrected" \
    --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" || echo "  summary failed"

echo ""
echo "ship $MODEL_KIND comparison done → $OUT_BASE/combined_<date>_<ship>/"
echo "ship summary → $OUT_BASE/ship_comparison_summary.csv"

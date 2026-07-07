#!/bin/env bash

# Common implementation for structured-residual real-date TCCON comparison runs.
# Submit one of the thin wrappers:
#   curc_shell_blanca_plot_corr_xco2_structured_r10.sh
#   curc_shell_blanca_plot_corr_xco2_structured_o05l15.sh
#
# The long case list is intentionally NOT duplicated here.  We evaluate the active
# run_case lines from STRUCT_CASE_SCRIPT (default:
# curc_shell_blanca_plot_corr_xco2_deepens.sh), so DE/TabM/structured comparisons
# stay aligned as the real-date/TCCON case inventory evolves.

set -euo pipefail

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:${LD_LIBRARY_PATH:-}
else
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export PYTHONPATH=src:${PYTHONPATH:-}

DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
export OCO2_DATAROOT="$DATA_ROOT"

MODEL_ROOT="$DATA_ROOT/results/model_structured_dcn_ensemble"
MODEL_TAG="${STRUCT_MODEL_TAG:?STRUCT_MODEL_TAG must be set by the wrapper}"
OCEAN_STEM="${STRUCT_OCEAN_STEM:?STRUCT_OCEAN_STEM must be set by the wrapper}"
LAND_STEM="${STRUCT_LAND_STEM:?STRUCT_LAND_STEM must be set by the wrapper}"

shopt -s nullglob
OCEAN_MODEL_DIRS=("$MODEL_ROOT"/"${OCEAN_STEM}"_f*)
LAND_MODEL_DIRS=("$MODEL_ROOT"/"${LAND_STEM}"_f*)
shopt -u nullglob

if (( ${#OCEAN_MODEL_DIRS[@]} == 0 )); then
    echo "No ocean structured model dirs matched: $MODEL_ROOT/${OCEAN_STEM}_f*" >&2
    exit 2
fi
if (( ${#LAND_MODEL_DIRS[@]} == 0 )); then
    echo "No land structured model dirs matched: $MODEL_ROOT/${LAND_STEM}_f*" >&2
    exit 2
fi

CSV_DIR="$DATA_ROOT"/results/csv_collection
OUT_BASE="$DATA_ROOT"/results/model_comparison/structured_residual/${MODEL_TAG}

RADIUS_KM="${RADIUS_KM:-100}"
WINDOW_MIN="${WINDOW_MIN:-60}"
RADIUS_TAG="r${RADIUS_KM}km"
REQUIRE_TCCON="${REQUIRE_TCCON:-1}"
REBUILD="${REBUILD:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
CASE_SCRIPT="${STRUCT_CASE_SCRIPT:-curc_shell_blanca_plot_corr_xco2_deepens.sh}"
CORR_COL="structured_residual_corrected_xco2"

echo "Structured residual model tag: $MODEL_TAG"
echo "Ocean folds (${#OCEAN_MODEL_DIRS[@]}): ${OCEAN_MODEL_DIRS[*]}"
echo "Land folds  (${#LAND_MODEL_DIRS[@]}): ${LAND_MODEL_DIRS[*]}"
echo "Case script: $CASE_SCRIPT"
echo "OUT_BASE: $OUT_BASE"

run_case() {
    local date="$1" tccon="$2" lonmin="$3" lonmax="$4" latmin="$5" latmax="$6"
    local vmin="$7" vmax="$8" surf="${9:-both}" poster="${10:-}" site="${11:-}"
    local tccon_avail="${12:-yes}"

    local input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    local h5="$DATA_ROOT/results/results_${date}.h5"
    local outdir="$OUT_BASE/combined_${date}_all_orbits"
    [[ -n "$site" ]] && outdir="$OUT_BASE/combined_${date}_${site}"
    local plotdata="$outdir/plot_data.parquet"

    echo ""
    echo "############ STRUCTURED CASE $date  ($surf)  ############"
    if [[ "$REQUIRE_TCCON" == 1 && "$tccon_avail" != yes ]]; then
        echo "  SKIP (REQUIRE_TCCON=1; TCCON_AVAIL=$tccon_avail — no TCCON at OCO pass time)"
        return
    fi
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"
        return
    fi
    mkdir -p "$outdir"

    if [[ "$REBUILD" != 1 && -f "$plotdata" && "$plotdata" -nt "$input" ]]; then
        echo "  (reuse existing plot_data.parquet — REBUILD=1 to force)"
    else
        local model_args=()
        [[ "$surf" == both || "$surf" == ocean ]] && model_args+=(--ocean-model-dir "${OCEAN_MODEL_DIRS[@]}")
        [[ "$surf" == both || "$surf" == land  ]] && model_args+=(--land-model-dir  "${LAND_MODEL_DIRS[@]}")
        python workspace/build_structured_residual_plot_data.py \
            "${model_args[@]}" --input "$input" --output "$plotdata" || { echo "  build failed"; return; }
    fi

    if [[ "$MAKE_PLOTS" != 0 ]]; then
        local h5_arg=(); [[ -f "$h5" ]] && h5_arg=(--results-h5 "$h5")
        local poster_arg=(); [[ "$poster" == poster ]] && \
            poster_arg=(--poster-model "$CORR_COL" --poster-dpi 300)
        python workspace/plot_corrected_xco2.py \
            --plot-data  "$plotdata" \
            --tccon      "$DATA_ROOT/data/TCCON/$tccon" \
            "${h5_arg[@]}" \
            --output-dir "$outdir" \
            --modis-auto \
            --lon-range  "$lonmin" "$lonmax" \
            --lat-range  "$latmin" "$latmax" \
            --date-plot  "$date" \
            --vmin "$vmin" --vmax "$vmax" \
            --hist-radius-km "$RADIUS_KM" \
            "${poster_arg[@]}"

        python workspace/plot_spectral_params.py \
            --input      "$input" \
            --tccon      "$DATA_ROOT/data/TCCON/$tccon" \
            "${h5_arg[@]}" \
            --output-dir "$outdir" \
            --modis-auto \
            --lon-range  "$lonmin" "$lonmax" \
            --lat-range  "$latmin" "$latmax" \
            --date-plot  "$date"
    else
        echo "  (MAKE_PLOTS=0 — skipped per-case MODIS plots; plot_data.parquet written)"
    fi
}

if [[ ! -f "$CASE_SCRIPT" ]]; then
    echo "Case script not found: $CASE_SCRIPT" >&2
    exit 2
fi

while IFS= read -r line; do
    [[ "$line" =~ ^run_case[[:space:]] ]] || continue
    eval "$line"
done < "$CASE_SCRIPT"

echo ""
echo "############ AGGREGATE: tccon_comparison_report ############"
python workspace/tccon_comparison_report.py \
    --script   "$CASE_SCRIPT" \
    --out-base "$OUT_BASE" \
    --output-dir "$OUT_BASE" \
    --corr-col "$CORR_COL" \
    --fname-suffix "_${RADIUS_TAG}" --exclude-sites ny \
    --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" \
    --ak-harmonize

echo ""
echo "############ AGGREGATE: tccon_correction_policy_stats ############"
python workspace/tccon_correction_policy_stats.py \
    --script "$CASE_SCRIPT" \
    --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" \
    --corr-col "$CORR_COL" \
    --plotdata-base "$OUT_BASE" --output-dir "$OUT_BASE" --fname-suffix "_${RADIUS_TAG}"

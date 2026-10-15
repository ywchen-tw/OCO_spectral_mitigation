#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_plot_xgb
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable

# Plain-XGBoost-mean-baseline version of curc_shell_blanca_plot_corr_xco2_tabm.sh.
# For each case it (4) applies the per-surface baseline ensemble on ocean AND land
# footprints (build_baseline_plot_data.py --model-kind xgb) then (5) plots vs
# TCCON + MODIS (plot_corrected_xco2.py, --poster-model xgb_corrected_xco2).
# Cases are replayed from the deepens launcher (see the cases block) and the TCCON
# knobs match it, so the aggregate reports cover the SAME station-days as the DE —
# only the model wiring and MODEL_TAG differ.  See log/archive/LINREG_XGB_BASELINE_PLAN_2026-07-07.md (Phase 4).

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data
if [[ "$(uname -s)" == "Linux" ]]; then
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export PYTHONPATH=src:${PYTHONPATH:-}
# Case list (run_case lines) shared with the DE launcher; see the cases block.
CASES_SCRIPT=curc_shell_blanca_plot_corr_xco2_deepens.sh

DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
export OCO2_DATAROOT="$DATA_ROOT"

# ─── model (xgb-mean fold dirs; per-surface) ──────────────────────────────────
# Pool ALL date_kfold folds (f0..f4) → cross-fold ensemble: the point prediction
# is the mean over folds of each fold's XGBoost-mean mu (each fold transformed by
# its OWN fitted pipeline / scaler), mirroring the cross-fold pooling used for the
# DE/TabM.  Each fold carries the fold-specific ProfilePCA block (leakage-safe),
# embedded in its pipeline pkl.  Mixed near-cloud anomaly-feature radius: OCEAN r05,
# LAND r15 (o05l15), matching the production DE tag.  Trained by
# curc_shell_blanca_xgb_foldpca_r05.sh / _r15.sh (they write under model_gbdt/).
# Run version (log/PIPELINE_FIX_RERUN_PLAN_2026-10-14.md): v2 = unsmoothed gamma,
# fill/spike imputation, RobustClipScaler, no t700.  The tag sits BEFORE the fold
# tag, so v1 globs (..._f*) never match v2 dirs and v2 runs never overwrite v1
# checkpoints.  RUN_VERSION= (empty) reproduces the v1 names.
export RUN_VERSION=${RUN_VERSION-v2}
RV=${RUN_VERSION:+_${RUN_VERSION}}

MODEL_TAG=xgb_prof_foldpca_o05l15${RV}
MODEL_KIND=xgb
OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_gbdt/xgb_ocean_full_prof_foldpca_r05${RV}_f*)
LAND_MODEL_DIRS=("$DATA_ROOT"/results/model_gbdt/xgb_land_full_prof_foldpca_r15${RV}_f*)
CORR_COL=${MODEL_KIND}_corrected_xco2

CSV_DIR="$DATA_ROOT"/results/csv_collection
OUT_BASE="$DATA_ROOT"/results/model_comparison/${MODEL_KIND}/${MODEL_TAG}

RADIUS_KM="${RADIUS_KM:-100}"
WINDOW_MIN="${WINDOW_MIN:-60}"
RADIUS_TAG="r${RADIUS_KM}km"
REQUIRE_TCCON=1

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
    echo "############ CASE $date  ($surf)  ############"
    if [[ "$REQUIRE_TCCON" == 1 && "$tccon_avail" != yes ]]; then
        echo "  SKIP (REQUIRE_TCCON=1; TCCON_AVAIL=$tccon_avail — no TCCON at OCO pass time)"; return
    fi
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"; return
    fi
    mkdir -p "$outdir"

    # (4) apply baseline ensemble → plot_data.parquet
    if [[ "${REBUILD:-0}" != 1 && -f "$plotdata" && "$plotdata" -nt "$input" ]]; then
        echo "  (reuse existing plot_data.parquet — REBUILD=1 to force)"
    else
        local model_args=()
        [[ "$surf" == both || "$surf" == ocean ]] && model_args+=(--ocean-model-dir "${OCEAN_MODEL_DIRS[@]}")
        [[ "$surf" == both || "$surf" == land  ]] && model_args+=(--land-model-dir  "${LAND_MODEL_DIRS[@]}")
        python workspace/build_baseline_plot_data.py --model-kind "$MODEL_KIND" \
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

# ═══════════════════════ cases: replayed from the DE launcher ════════════════════
# The case list lives in ONE place, curc_shell_blanca_plot_corr_xco2_deepens.sh;
# its active run_case lines are replayed here, so the baselines are evaluated on
# exactly the DE's station-days and TCCON file versions.  (Until 2026-10-15 this
# launcher kept its own copy, which missed the 24 cases added 2026-09-15 and kept
# older TCCON releases for bu, oc, et, ci and ka.)
while IFS= read -r line; do
    line="${line%%#*}"
    # shellcheck disable=SC2086
    eval "run_case ${line#run_case}"
done < <(grep -E '^[[:space:]]*run_case[[:space:]]' "$CASES_SCRIPT")

# ═══════════════════════ aggregate reports (run once, after all cases) ════════════
echo ""
echo "############ AGGREGATE: tccon_comparison_report ############"
python workspace/tccon_comparison_report.py \
    --script   "$CASES_SCRIPT" \
    --out-base "$OUT_BASE" \
    --output-dir "$OUT_BASE" \
    --corr-col "$CORR_COL" \
    --fname-suffix "_${RADIUS_TAG}" --exclude-sites ny \
    --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" \
    --cld-edges "0,10,inf" \
    --ak-harmonize

echo ""
echo "############ AGGREGATE: tccon_correction_policy_stats ############"
python workspace/tccon_correction_policy_stats.py \
    --script "$CASES_SCRIPT" \
    --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" \
    --corr-col "$CORR_COL" \
    --plotdata-base "$OUT_BASE" --output-dir "$OUT_BASE" --fname-suffix "_${RADIUS_TAG}"

#!/usr/bin/env bash
# deepens_workflow.sh — end-to-end deep-ensemble correction + TCCON/MODIS plot.
#
# Steps:
#   (1) define target-date parquet
#   (2) define lon/lat range, plot date, vmin/vmax
#   (3) define model (deep-ensemble fold dirs, ocean + land)
#   (4) apply model on BOTH ocean and land footprints  (build_deepens_plot_data.py)
#   (5) plot vs TCCON with MODIS background           (plot_corrected_xco2.py)
#
# Usage:  bash workspace/deepens_workflow.sh
# Edit the CONFIG block below, then run.  All paths are repo-relative.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
export PYTHONPATH=src:${PYTHONPATH:-}

# ─── (1) target-date parquet ──────────────────────────────────────────────────
INPUT_PARQUET=results/csv_collection/combined_2018-09-02_all_orbits.parquet

# ─── (2) plot window ──────────────────────────────────────────────────────────
LON_MIN=-16.48 ; LON_MAX=-16.1
LAT_MIN=27.64  ; LAT_MAX=28.64
DATE_PLOT=2018-09-02
VMIN=402 ; VMAX=408

# ─── (3) model (deep-ensemble fold dirs; per-surface) ─────────────────────────
OCEAN_MODEL_DIR=results/model_deep_ensemble/de_ocean_beta_nll_f0
LAND_MODEL_DIR=results/model_deep_ensemble/de_land_beta_nll_f0

# ─── reference data + output ──────────────────────────────────────────────────
TCCON=data/TCCON/iz20140102_20230830.public.qc.nc
RESULTS_H5=results/results_${DATE_PLOT}.h5      # for MODIS granule date (optional)
OUT_DIR=results/model_comparison/ocean_2016_2020_4_cld/$(basename "${INPUT_PARQUET%.parquet}")/deep_ensemble
PLOT_DATA=$OUT_DIR/plot_data.parquet

mkdir -p "$OUT_DIR"

# ─── (4) apply deep ensemble on ocean + land, build plot_data ─────────────────
echo "── (4) applying deep ensemble (ocean + land) ──"
python workspace/build_deepens_plot_data.py \
    --ocean-model-dir "$OCEAN_MODEL_DIR" \
    --land-model-dir  "$LAND_MODEL_DIR" \
    --input  "$INPUT_PARQUET" \
    --output "$PLOT_DATA"

# ─── (5) plot vs TCCON + MODIS background ──────────────────────────────────────
echo "── (5) plotting ──"
H5_ARG=(); [ -f "$RESULTS_H5" ] && H5_ARG=(--results-h5 "$RESULTS_H5")
python workspace/plot_corrected_xco2.py \
    --plot-data  "$PLOT_DATA" \
    --tccon      "$TCCON" \
    "${H5_ARG[@]}" \
    --output-dir "$OUT_DIR" \
    --modis-auto \
    --lon-range  "$LON_MIN" "$LON_MAX" \
    --lat-range  "$LAT_MIN" "$LAT_MAX" \
    --date-plot  "$DATE_PLOT" \
    --vmin "$VMIN" --vmax "$VMAX" \
    --poster-model deep_ensemble_corrected_xco2 \
    --poster-dpi 300

echo "── done → $OUT_DIR ──"

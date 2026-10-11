#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_ship_deepens
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable

# ─────────────────────────────────────────────────────────────────────────────
# Standalone shipborne-EM27/SUN ocean-glint deep-ensemble comparison runner.
#
# Open-ocean XCO2 truth from two EM27/SUN cruises (the OCEAN anchor the land TCCON
# stations can't give): MORE-2 (RV Sonne, 2019-06) and MR21-01 (RV Mirai, 2021-03).
# For each ship-coincident date it (a) applies the SAME production M=5 deep ensemble
# used by curc_shell_blanca_plot_corr_xco2_deepens.sh to the OCO-2 ocean footprints
# (build_deepens_plot_data.py -> plot_data.parquet), then (b) draws the ship-native
# comparison figure (plot_ship_comparison.py) — OCO-2 corrected vs ship XCO2, with
# proper ship labels (NOT reusing the TCCON machinery).
#
# There is deliberately NO TCCON step and this is NOT wired into the TCCON launcher:
# a ship is a moving platform with no station, and keeping it separate stops the
# TCCON aggregate reports from being polluted by station-less cases (same rationale
# as ATom_analysis/curc_shell_blanca_atom_deepens.sh).
#
# Model = MIXED per-surface radius, matching the TCCON launcher: OCEAN uses the r05
# (0.5°) profile+reg variant.  Ship is ocean-only, so only the ocean model is loaded.
#
# Dates + boxes come from the two-stage screen:
#   1. python workspace/Ship_analysis/ship_lite_collocate.py        (which days overlap)
#   2. python workspace/Ship_analysis/ship_footprint_collocate.py   (box / vmin-vmax)
#
# Submit from the REPO ROOT:  sbatch workspace/Ship_analysis/curc_shell_blanca_ship_deepens.sh
# ─────────────────────────────────────────────────────────────────────────────

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

# Repo root: under SLURM $0/BASH_SOURCE is the spooled copy, so prefer the submit
# dir (submit from repo root); fall back to two levels up from this file locally.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$REPO_ROOT}"
export PYTHONPATH=src:${PYTHONPATH:-}

# ─── data root (mirror the TCCON launcher) ────────────────────────────────────
DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
export OCO2_DATAROOT="$DATA_ROOT"

# ─── model + cloud-classifier dirs (ocean r05, identical to the TCCON launcher) ──
# foldpca (2026-07-15): fold-specific ProfilePCA production models (leakage-safe).
MODEL_TAG=de_beta_nll_prof_reg_foldpca_o05l15_m5
OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_ocean_beta_nll_prof_reg_foldpca_r05_f*)
OCEAN_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_ocean_f*)
# The xgb_cloud classifier only exists on CURC; locally, drop the (diagnostic-only)
# P(near)/gate columns instead of crashing the build on an unexpanded glob.
[[ -d "${OCEAN_CLOUD_DIRS[0]}" ]] && OCEAN_CLOUD_ARGS=(--ocean-cloud-model-dir "${OCEAN_CLOUD_DIRS[@]}") || OCEAN_CLOUD_ARGS=()

CSV_DIR="$DATA_ROOT"/results/csv_collection
# Own subtree under the shared MODEL_TAG dir so ship outputs never collide with
# the TCCON cases' combined_<date>_<site> dirs (mirrors the ATom subtree).
OUT_BASE="$DATA_ROOT"/results/model_comparison/deep_ensemble/${MODEL_TAG}/ship

RADIUS_KM="${RADIUS_KM:-100}"
WINDOW_MIN="${WINDOW_MIN:-120}"

ship_case() {
    local date="$1" ship="$2" lonmin="$3" lonmax="$4" latmin="$5" latmax="$6"
    local vmin="$7" vmax="$8" note="${9:-}"
    local input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    local outdir="$OUT_BASE/combined_${date}_${ship}"
    local plotdata="$outdir/plot_data.parquet"

    echo ""
    echo "############ SHIP CASE $date  ($ship, ocean)  ############"
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"; return
    fi
    mkdir -p "$outdir"

    # (a) apply deep ensemble → plot_data.parquet
    python workspace/build_deepens_plot_data.py \
        --ocean-model-dir "${OCEAN_MODEL_DIRS[@]}" \
        "${OCEAN_CLOUD_ARGS[@]}" \
        --input "$input" --output "$plotdata" \
        || { echo "  build failed"; return; }

    # (b) ship-native comparison figure (with MODIS Aqua true-colour background)
    python workspace/Ship_analysis/plot_ship_comparison.py \
        --plot-data "$plotdata" --ship-tag "$ship" --date "$date" \
        --lon-range "$lonmin" "$lonmax" --lat-range "$latmin" "$latmax" \
        --vmin "$vmin" --vmax "$vmax" \
        --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" \
        --modis-auto --modis-which aqua \
        --output-dir "$outdir" \
        || { echo "  plot failed"; return; }
    echo "  done → $outdir  ($note)"
}

# ═══════════════════════ ship cases ══════════════════════════════════════════
# Boxes/vmin-vmax from ship_footprint_collocate.py (100 km / ±2 h, ocean-glint good-QF).
#           DATE(OCO)    SHIP    LON_MIN   LON_MAX  LAT_MIN  LAT_MAX   VMIN   VMAX   NOTE
ship_case  2019-06-09   so268    -152.62  -152.15    29.36    30.94   412.0  414.5  "494 fp, 494 near<=10km — full near-cloud"
ship_case  2019-06-14   so268    -170.75  -170.54    28.94    29.66   409.5  413.0  "111 fp, 73 near<=10km"
ship_case  2019-06-22   so268     152.36   152.80    27.43    28.96   411.5  412.5  "460 fp, 1 near<=10km — clear-sky reference"
ship_case  2021-03-15   mr2101    140.06   140.18    26.24    26.60   416.0  417.5  "6 fp, 5 near<=10km — sparse"

# 2021-03-18 OMITTED: nearest OCO ocean-glint pass ~212 km, 0 footprints ≤150 km.

# ── cross-case summary (per-case bias dumbbell + bias-vs-cloud-distance, ±1σ) ──
echo ""
echo "############ SHIP SUMMARY ############"
python workspace/Ship_analysis/plot_ship_summary.py --out-base "$OUT_BASE" \
    --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" || echo "  summary failed"

echo ""
echo "ship deep-ensemble comparison done → $OUT_BASE/combined_<date>_<ship>/"
echo "ship summary → $OUT_BASE/ship_comparison_summary.png"

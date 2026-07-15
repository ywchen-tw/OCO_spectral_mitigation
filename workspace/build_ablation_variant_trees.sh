#!/bin/env bash
# Build the feature-set-ablation variant plot_data trees (build-only replay).
#
# For ONE variant (arg 1: no_spec | no_xco2 | no_contam | no_xco2_and_spec |
# no_contam_and_xco2) this replays the ACTIVE run_case lines of the production
# TCCON launcher and rebuilds every case's plot_data.parquet with the variant's
# fold-PCA deep-ensemble models (ocean r05 / land r15), writing to the
# de_prof_mix_<variant> tree that tccon_comparison_report.py is then pointed at:
#
#   results/model_comparison/deep_ensemble/de_prof_mix_<variant>/combined_<date>_<site>/plot_data.parquet
#
# The 'full' reference tree is the production atrain tree (same builds) — no
# separate full-mix build is needed.  Mirrors the replay mechanism of
# curc_shell_blanca_deepens_uncertainty.sh.  Local (macOS) friendly: no module
# loads, cloud-classifier args dropped when the dirs are absent, libomp
# borrowed via DYLD_FALLBACK.
#
# Usage:  bash workspace/build_ablation_variant_trees.sh no_spec

set -u
VARIANT="${1:?usage: build_ablation_variant_trees.sh <variant>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH=src:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ -z "${DYLD_FALLBACK_LIBRARY_PATH:-}" ]]; then
    for _d in "$HOME"/miniforge3/envs/*/lib; do
        [[ -f "$_d/libomp.dylib" ]] && export DYLD_FALLBACK_LIBRARY_PATH="$_d" && break
    done
fi

DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
export OCO2_DATAROOT="$DATA_ROOT"

OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_ocean_${VARIANT}_prof_foldpca_r05_f*)
# Land fold f2 of EVERY variant diverged in the 2026-07 CURC retrain (held-out
# RMSE 3.8k-43k ppm; see FOLDPCA_RERUN_2026-07-15.md) — pool the 4 healthy land
# folds until the retrain lands.  Ocean folds are all healthy.
LAND_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_land_${VARIANT}_prof_foldpca_r15_f{0,1,3,4})
[[ -d "${OCEAN_MODEL_DIRS[0]}" && -d "${LAND_MODEL_DIRS[0]}" ]] || {
    echo "variant model dirs not found for '$VARIANT'" >&2; exit 2; }

CSV_DIR="$DATA_ROOT"/results/csv_collection
OUT_BASE="$DATA_ROOT"/results/model_comparison/deep_ensemble/de_prof_mix_${VARIANT}
SRC_SCRIPT="${SRC_SCRIPT:-curc_shell_blanca_plot_corr_xco2_deepens.sh}"
REQUIRE_TCCON="${REQUIRE_TCCON:-1}"

run_case_build() {
    local date="$1" surf="${9:-both}" site="${11:-}" tccon_avail="${12:-yes}"
    local input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    local outdir="$OUT_BASE/combined_${date}_all_orbits"
    [[ -n "$site" ]] && outdir="$OUT_BASE/combined_${date}_${site}"
    echo ""
    echo "############ [$VARIANT] $date ($surf) ${site:+[$site]} ############"
    if [[ "$REQUIRE_TCCON" == 1 && "$tccon_avail" != yes ]]; then
        echo "  SKIP (TCCON_AVAIL=$tccon_avail)"; return
    fi
    [[ -f "$input" ]] || { echo "  SKIP: no input parquet"; return; }
    mkdir -p "$outdir"
    local model_args=()
    [[ "$surf" == both || "$surf" == ocean ]] && model_args+=(--ocean-model-dir "${OCEAN_MODEL_DIRS[@]}")
    [[ "$surf" == both || "$surf" == land  ]] && model_args+=(--land-model-dir  "${LAND_MODEL_DIRS[@]}")
    python workspace/build_deepens_plot_data.py \
        "${model_args[@]}" --input "$input" --output "$outdir/plot_data.parquet" \
        || echo "  build FAILED"
}

while IFS= read -r line; do
    line="${line%%#*}"
    # shellcheck disable=SC2086
    eval "run_case_build ${line#run_case}"
done < <(grep -E '^[[:space:]]*run_case[[:space:]]' "$SRC_SCRIPT")

echo ""
echo "[$VARIANT] variant tree done → $OUT_BASE"

#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_nassar_de
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable

# Standalone power-plant plume-control runner using Nassar et al. plant locations.
#
# Applies the production deep ensemble to user-selected OCO-2 dates, then the
# downstream screen_nassar_plumes.py script cross-checks those dates against the
# plant lon/lat catalog.  Dates are deliberately NOT taken from the Nassar table:
# the table is only a source-coordinate catalog.
#
# Prerequisite per date:
#   results/csv_collection/combined_<YYYY-MM-DD>_all_orbits.parquet
#
# Submit from repo root:
#   NASSAR_DATES="2018-06-03 2021-04-24" \
#     sbatch workspace/Nassar_plume_analysis/curc_shell_blanca_nassar_deepens.sh
#
# Or:
#   NASSAR_DATES_FILE=path/to/dates_yyyymmdd_or_yyyy-mm-dd.txt \
#     sbatch workspace/Nassar_plume_analysis/curc_shell_blanca_nassar_deepens.sh

if [[ "$(uname -s)" == "Linux" ]]; then
    module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
    conda activate data
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:${LD_LIBRARY_PATH:-}
else
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$REPO_ROOT}"
export PYTHONPATH=src:${PYTHONPATH:-}

DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
export OCO2_DATAROOT="$DATA_ROOT"

MODEL_TAG=de_beta_nll_prof_reg_foldpca_o05l15_m5
OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_ocean_beta_nll_prof_reg_foldpca_r05_f*)
LAND_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_land_beta_nll_prof_reg_foldpca_r15_f*)
OCEAN_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_ocean_f*)
LAND_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_land_f*)
# The xgb_cloud classifier only exists on CURC; locally, drop the (diagnostic-only)
# P(near)/gate columns instead of crashing the build on an unexpanded glob.
[[ -d "${OCEAN_CLOUD_DIRS[0]}" ]] && OCEAN_CLOUD_ARGS=(--ocean-cloud-model-dir "${OCEAN_CLOUD_DIRS[@]}") || OCEAN_CLOUD_ARGS=()
[[ -d "${LAND_CLOUD_DIRS[0]}"  ]] && LAND_CLOUD_ARGS=(--land-cloud-model-dir  "${LAND_CLOUD_DIRS[@]}")  || LAND_CLOUD_ARGS=()

CSV_DIR="$DATA_ROOT"/results/csv_collection
OUT_BASE="$DATA_ROOT"/results/model_comparison/deep_ensemble/${MODEL_TAG}/nassar_plumes
PLANTS_CSV="${PLANTS_CSV:-workspace/Nassar_plume_analysis/nassar_power_plants.csv}"

mkdir -p "$OUT_BASE"

normalize_date() {
    local raw="$1"
    raw="${raw//[$'\r\n\t ']/}"
    if [[ "$raw" =~ ^[0-9]{8}$ ]]; then
        printf "%s-%s-%s\n" "${raw:0:4}" "${raw:4:2}" "${raw:6:2}"
    else
        printf "%s\n" "$raw"
    fi
}

load_dates() {
    if [[ "$#" -gt 0 ]]; then
        printf "%s\n" "$@"
    elif [[ -n "${NASSAR_DATES:-}" ]]; then
        printf "%s\n" $NASSAR_DATES
    elif [[ -n "${NASSAR_DATES_FILE:-}" ]]; then
        sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "$NASSAR_DATES_FILE"
    else
        echo "No dates supplied. Set NASSAR_DATES, NASSAR_DATES_FILE, or pass dates as arguments." >&2
        return 2
    fi
}

nassar_date() {
    local date="$1"
    local input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    local outdir="$OUT_BASE/combined_${date}"
    local plotdata="$outdir/plot_data.parquet"

    echo ""
    echo "############ POWER-PLANT SCREEN DATE $date  ############"
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"
        return
    fi
    mkdir -p "$outdir"

    python workspace/build_deepens_plot_data.py \
        --ocean-model-dir "${OCEAN_MODEL_DIRS[@]}" \
        "${OCEAN_CLOUD_ARGS[@]}" \
        --land-model-dir "${LAND_MODEL_DIRS[@]}" \
        "${LAND_CLOUD_ARGS[@]}" \
        --input "$input" --output "$plotdata" \
        || { echo "  build failed"; return; }
    echo "  wrote $plotdata"
}

# (while-read instead of mapfile: macOS ships bash 3.2, which lacks mapfile)
REQUESTED_DATES=()
while IFS= read -r _d; do
    [[ -n "$_d" ]] && REQUESTED_DATES+=("$_d")
done < <(load_dates "$@")
if [[ "${#REQUESTED_DATES[@]}" -eq 0 ]]; then
    echo "No dates supplied after filtering." >&2
    exit 2
fi

for raw_date in "${REQUESTED_DATES[@]}"; do
    date="$(normalize_date "$raw_date")"
    nassar_date "$date"
done

echo ""
echo "############ NASSAR PLUME SCREEN ############"
python workspace/Nassar_plume_analysis/screen_nassar_plumes.py \
    --plants "$PLANTS_CSV" \
    --dates "${REQUESTED_DATES[@]}" \
    --csv-dir "$CSV_DIR" \
    --plot-base "$OUT_BASE" \
    --output "$OUT_BASE/nassar_plume_screen.csv" \
    --markdown "$OUT_BASE/nassar_plume_screen.md" \
    || echo "  plume screen failed"

echo ""
echo "Power-plant plume-control DE outputs -> $OUT_BASE/combined_<date>/"
echo "Screen summary -> $OUT_BASE/nassar_plume_screen.csv"

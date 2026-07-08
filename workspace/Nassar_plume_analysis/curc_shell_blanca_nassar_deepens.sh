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

# Standalone Nassar et al. power-plant plume-control runner.
#
# Applies the production deep ensemble to the exact OCO-2 dates/source coordinates
# listed in nassar_power_plant_cases.csv.  There is no TCCON step here: these are
# plume negative controls for M1, not station-day validation cases.  The downstream
# screen_nassar_plumes.py script selects footprints near each plant and tests
# whether real XCO2 enhancements are preserved while predicted mu stays inert.
#
# Prerequisite per date:
#   results/csv_collection/combined_<YYYY-MM-DD>_all_orbits.parquet
#
# Submit from repo root:
#   sbatch workspace/Nassar_plume_analysis/curc_shell_blanca_nassar_deepens.sh

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

MODEL_TAG=de_beta_nll_prof_reg_o05l15_m5
OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_ocean_beta_nll_prof_reg_r05_f*)
LAND_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_land_beta_nll_prof_reg_r15_f*)
OCEAN_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_ocean_f*)
LAND_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_land_f*)

CSV_DIR="$DATA_ROOT"/results/csv_collection
OUT_BASE="$DATA_ROOT"/results/model_comparison/deep_ensemble/${MODEL_TAG}/nassar_plumes
CASES_CSV="${CASES_CSV:-workspace/Nassar_plume_analysis/nassar_power_plant_cases.csv}"

mkdir -p "$OUT_BASE"

nassar_case() {
    local case_id="$1" date="$2"
    local input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    local outdir="$OUT_BASE/combined_${date}_${case_id}"
    local plotdata="$outdir/plot_data.parquet"

    echo ""
    echo "############ NASSAR CASE $case_id  $date  ############"
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"
        return
    fi
    mkdir -p "$outdir"

    python workspace/build_deepens_plot_data.py \
        --ocean-model-dir "${OCEAN_MODEL_DIRS[@]}" \
        --ocean-cloud-model-dir "${OCEAN_CLOUD_DIRS[@]}" \
        --land-model-dir "${LAND_MODEL_DIRS[@]}" \
        --land-cloud-model-dir "${LAND_CLOUD_DIRS[@]}" \
        --input "$input" --output "$plotdata" \
        || { echo "  build failed"; return; }
    echo "  wrote $plotdata"
}

python - "$CASES_CSV" <<'PY' | while IFS=$'\t' read -r case_id date; do
import csv
import sys

with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        print(f"{row['case_id']}\t{row['date']}")
PY
    nassar_case "$case_id" "$date"
done

echo ""
echo "############ NASSAR PLUME SCREEN ############"
python workspace/Nassar_plume_analysis/screen_nassar_plumes.py \
    --cases "$CASES_CSV" \
    --csv-dir "$CSV_DIR" \
    --plot-base "$OUT_BASE" \
    --output "$OUT_BASE/nassar_plume_screen.csv" \
    --markdown "$OUT_BASE/nassar_plume_screen.md" \
    || echo "  plume screen failed"

echo ""
echo "Nassar plume-control DE outputs -> $OUT_BASE/combined_<date>_<case_id>/"
echo "Screen summary -> $OUT_BASE/nassar_plume_screen.csv"

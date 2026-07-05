#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_atom_deepens
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable

# ─────────────────────────────────────────────────────────────────────────────
# Standalone ATom ocean-glint deep-ensemble correction runner.
#
# Applies the SAME production M=5 deep ensemble (de_*_beta_nll_prof_reg) used by
# curc_shell_blanca_plot_corr_xco2_deepens.sh to the OCO-2 footprints on each
# ATom-coincident date, emitting plot_data.parquet (deep-ensemble-corrected XCO2
# + sigma/cld_dist + lat/lon/time) per date under an ATom output subtree.
#
# There is deliberately NO TCCON step here: ATom is a moving aircraft with no
# ground station — the aircraft pseudo-column IS the reference and is built
# separately (ATom_analysis Stage 2/3).  Ocean footprints only (ATom glint
# validation).  Kept OUT of the TCCON launcher so the TCCON aggregate reports are
# not polluted by station-less cases.
#
# WHY correct the whole day and not just the box: build_deepens_plot_data.py has
# no lon/lat crop; it corrects every ocean footprint in the input parquet.  The
# per-case collocation box (below) is applied downstream when selecting the
# footprints that coincide with the ATom track — see atom_footprint_collocate.py.
#
# Boxes = ocean-glint good-QF OCO-2 footprints within 100 km / ±2 h of the ATom
# flight track (from atom_lite_collocate.py + atom_footprint_collocate.py).
#
# Submit from the REPO ROOT:  sbatch workspace/ATom_analysis/curc_shell_blanca_atom_deepens.sh
# ─────────────────────────────────────────────────────────────────────────────

# Env: on CURC/Linux load modules + the `data` env; locally (macOS) assume the
# intended conda env (with torch) is already active and just fix the lib path.
if [[ "$(uname -s)" == "Linux" ]]; then
    module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
    conda activate data
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
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

# ─── model + cloud-classifier dirs ────────────────────────────────────────────
# Ocean correction uses the r05 (0.5° clear-sky anomaly) DE model — same as the
# ship/TCCON launcher's OCEAN model (curc_shell_blanca_plot_corr_xco2_deepens.sh,
# which pairs ocean r05 + land r15).  ATom is ocean-only, so only r05 is used here.
# These models live LOCALLY under results/, so this runs on a laptop as well as CURC.
# MODEL_TAG matches the launcher's so ATom + ship outputs share one model-version tree.
MODEL_TAG=de_beta_nll_prof_reg_o05l15_m5
OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_ocean_beta_nll_prof_reg_r05_f*)
OCEAN_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_ocean_f*)

CSV_DIR="$DATA_ROOT"/results/csv_collection
# Own subtree under the shared MODEL_TAG dir so ATom outputs never collide with
# the TCCON cases' combined_<date>_<site> dirs.
OUT_BASE="$DATA_ROOT"/results/model_comparison/deep_ensemble/${MODEL_TAG}/atom

atom_case() {
    local date="$1" lonmin="$2" lonmax="$3" latmin="$4" latmax="$5" note="${6:-}"
    local input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    local outdir="$OUT_BASE/combined_${date}_atom"
    local plotdata="$outdir/plot_data.parquet"

    echo ""
    echo "############ ATOM CASE $date  (ocean)  ############"
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"; return
    fi
    mkdir -p "$outdir"
    python workspace/build_deepens_plot_data.py \
        --ocean-model-dir "${OCEAN_MODEL_DIRS[@]}" \
        --ocean-cloud-model-dir "${OCEAN_CLOUD_DIRS[@]}" \
        --input "$input" --output "$plotdata" \
        || { echo "  build failed"; return; }
    echo "  wrote $plotdata"
    echo "  collocation box: lon[$lonmin,$lonmax] lat[$latmin,$latmax]  $note"
}

# ═══════════════════════ ATom cases ══════════════════════════════════════════
# Boxes/regions from atom_footprint_collocate.py (100 km / ±2 h, ocean-glint good-QF).
#           DATE(OCO)     LON_MIN  LON_MAX  LAT_MIN  LAT_MAX   NOTE
atom_case  2017-01-26   -121.40  -120.94    0.15    2.19   "489 fp, 489 near<=10km — equatorial Pacific"
atom_case  2017-02-10   -141.88  -140.73  -64.58  -62.97   "471 fp, 275 near<=10km — Southern Ocean"
atom_case  2017-10-20    -28.83   -26.28   32.12   39.01   "433 fp, 331 near<=10km — N Atlantic"
atom_case  2017-10-27   -129.88  -129.01   37.92   39.98   "891 fp, 158 near<=10km — NE Pacific"
atom_case  2018-05-12    -40.54   -40.11  -37.57  -36.03   "16 fp, 16 near<=10km — S Atlantic"

# ── stubbed: coincidence falls on the flight's 2nd UTC day, whose OCO-2 parquet
#    is not yet processed.  Process that date through the pipeline, rerun
#    atom_footprint_collocate.py to get the box, then uncomment.
# atom_case  2017-02-06   LON_MIN  LON_MAX  LAT_MIN  LAT_MAX   "ATom 2017-02-05 flight; needs combined_2017-02-06"
# atom_case  2017-10-09   LON_MIN  LON_MAX  LAT_MIN  LAT_MAX   "ATom 2017-10-08 flight; needs combined_2017-10-09"
#
# ── not available: 2017-02-03 (never processed); 2018-05-01 (training data — excluded).

echo ""
echo "ATom deep-ensemble correction done → $OUT_BASE/combined_<date>_atom/plot_data.parquet"

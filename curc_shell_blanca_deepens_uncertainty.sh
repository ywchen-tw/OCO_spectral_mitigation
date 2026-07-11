#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_deepens_uncertainty
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable

# ── Phase-4 uncertainty-aware TCCON comparison: data regen + report ───────────
# Regenerates each case's plot_data.parquet WITH the Phase-0 uncertainty columns
#   de_sigma, de_epistemic_sigma, de_aleatoric_sigma   (build_deepens_plot_data.py)
# plus the per-member mean columns mu_00…  (--emit-members → EXACT case epistemic),
# then runs tccon_comparison_report.py --uncertainty to fill the Phase-4 stats
# (M1 z/CI, M3 TOST equivalence, M4 random-effects, whole-budget ⟨z²⟩).
# See src/analysis/UNCERTAINTY_AWARE_TCCON_COMPARISON.md §9–10.
#
# This does NOT re-run the poster/MODIS/spectral plots — only step (4) build, so
# it is much cheaper than the full curc_shell_blanca_plot_corr_xco2_deepens.sh.
# The rebuilt plot_data.parquet is a strict SUPERSET of the production one (same
# corrected columns + P(near)/gate + xco2_raw, plus the uncertainty columns), so
# overwriting the production tree in place is non-destructive — the other reports
# (policy_stats, plot_corrected_xco2) keep working.
#
# The case list is SOURCED from the production plot script (single source of
# truth); this launcher just replays those cases build-only.  Override which
# script via SRC_SCRIPT=… (e.g. the drift script).
#
#   sbatch curc_shell_blanca_deepens_uncertainty.sh
#   RADIUS_KM=50 sbatch curc_shell_blanca_deepens_uncertainty.sh          # robustness radius
#   SRC_SCRIPT=curc_shell_blanca_plot_corr_xco2_deepens_drift.sh sbatch … # drift cases
#   EMIT_MEMBERS=0 sbatch …                                               # skip mu_NN (fallback epistemic)

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export PYTHONPATH=src:${PYTHONPATH:-}

# ─── data root (mirror the plot script) ───────────────────────────────────────
DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
export OCO2_DATAROOT="$DATA_ROOT"

# ─── model config — KEEP IN SYNC with curc_shell_blanca_plot_corr_xco2_deepens.sh
# Same production M=5 profile+reg mixed-radius DE (ocean r05 / land r15).  MODEL_TAG
# namespaces OUT_BASE so we regenerate exactly the tree the report reads.
MODEL_TAG=de_beta_nll_prof_reg_o05l15_m5
OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_ocean_beta_nll_prof_reg_r05_f*)
LAND_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_land_beta_nll_prof_reg_r15_f*)
OCEAN_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_ocean_f*)
LAND_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_land_f*)

CSV_DIR="$DATA_ROOT"/results/csv_collection

# ─── knobs ────────────────────────────────────────────────────────────────────
SRC_SCRIPT="${SRC_SCRIPT:-curc_shell_blanca_plot_corr_xco2_deepens.sh}"  # case-list source
# Era leaf under MODEL_TAG follows the case-list source, matching the plot
# scripts' output trees: atrain (A-Train era, the default deepens script) vs
# drift (SRC_SCRIPT=…_deepens_drift.sh).
OUT_LEAF=atrain; [[ "$SRC_SCRIPT" == *drift* ]] && OUT_LEAF=drift
OUT_BASE="$DATA_ROOT"/results/model_comparison/deep_ensemble/${MODEL_TAG}/${OUT_LEAF}
RADIUS_KM="${RADIUS_KM:-100}"
WINDOW_MIN="${WINDOW_MIN:-60}"
DECORR_KM="${DECORR_KM:-15}"          # Side-A N_eff block size (the one free knob)
ROPE_DELTA="${ROPE_DELTA:-0.5}"       # TOST/ROPE equivalence margin (ppm)
EMIT_MEMBERS="${EMIT_MEMBERS:-1}"     # 1 → --emit-members (exact epistemic mu_NN)
REQUIRE_TCCON="${REQUIRE_TCCON:-1}"   # 1 → skip cases whose 12th col (TCCON_AVAIL) != yes
FORCE="${FORCE:-0}"                   # 1 → rebuild even if the plot_data already has de_sigma
EXCLUDE_SITES="${EXCLUDE_SITES:-ny}"  # emits an _excl_<sites> report variant (ny = drift-worst)

# build-only replay of one production run_case line (same positional args)
run_case_build() {
    local date="$1" surf="${9:-both}" site="${11:-}" tccon_avail="${12:-yes}"
    local input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    local outdir="$OUT_BASE/combined_${date}_all_orbits"
    [[ -n "$site" ]] && outdir="$OUT_BASE/combined_${date}_${site}"
    local plotdata="$outdir/plot_data.parquet"

    echo ""
    echo "############ REGEN $date  ($surf)  ${site:+[$site]} ############"
    if [[ "$REQUIRE_TCCON" == 1 && "$tccon_avail" != yes ]]; then
        echo "  SKIP (REQUIRE_TCCON=1; TCCON_AVAIL=$tccon_avail)"; return
    fi
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"; return
    fi
    # idempotence: skip if the plot_data already carries de_sigma (and mu_NN when
    # EMIT_MEMBERS=1), unless FORCE=1.
    if [[ "$FORCE" != 1 && -f "$plotdata" ]]; then
        local have
        have=$(python - "$plotdata" "$EMIT_MEMBERS" <<'PY'
import sys, pyarrow.parquet as pq
names = set(pq.ParquetFile(sys.argv[1]).schema_arrow.names)
need = 'de_sigma' in names
if sys.argv[2] == '1':
    need = need and any(n.startswith('mu_') and n[3:].isdigit() for n in names)
print('yes' if need else 'no')
PY
)
        if [[ "$have" == yes ]]; then
            echo "  SKIP: $plotdata already has uncertainty columns (FORCE=1 to rebuild)"; return
        fi
    fi
    mkdir -p "$outdir"

    local model_args=()
    [[ "$surf" == both || "$surf" == ocean ]] && \
        model_args+=(--ocean-model-dir "${OCEAN_MODEL_DIRS[@]}" --ocean-cloud-model-dir "${OCEAN_CLOUD_DIRS[@]}")
    [[ "$surf" == both || "$surf" == land  ]] && \
        model_args+=(--land-model-dir  "${LAND_MODEL_DIRS[@]}"  --land-cloud-model-dir  "${LAND_CLOUD_DIRS[@]}")
    local em=(); [[ "$EMIT_MEMBERS" == 1 ]] && em=(--emit-members)

    python workspace/build_deepens_plot_data.py \
        "${model_args[@]}" "${em[@]}" \
        --input "$input" --output "$plotdata" || { echo "  build FAILED"; return; }
}

# ─── (4) regenerate every production case, build-only ─────────────────────────
if [[ ! -f "$SRC_SCRIPT" ]]; then
    echo "case-list source not found: $SRC_SCRIPT" >&2; exit 1
fi
echo "Replaying active run_case lines from $SRC_SCRIPT (build-only, --emit-members=$EMIT_MEMBERS)"
# Only ACTIVE (uncommented) run_case lines; strip any trailing inline comment.
while IFS= read -r line; do
    line="${line%%#*}"
    # shellcheck disable=SC2086  # deliberate word-split of the positional args
    eval "run_case_build ${line#run_case}"
done < <(grep -E '^[[:space:]]*run_case[[:space:]]' "$SRC_SCRIPT")

# ─── (5) Phase-4 uncertainty-aware report (primary radius + 50 km robustness) ─
for R in "$RADIUS_KM" 50; do
    echo ""
    echo "############ Phase-4 report @ ${R} km ############"
    python workspace/tccon_comparison_report.py \
        --script     "$SRC_SCRIPT" \
        --out-base   "$OUT_BASE" \
        --output-dir "$OUT_BASE" \
        --corr-col   deep_ensemble_corrected_xco2 \
        --uncertainty \
        --radius-km  "$R" \
        --window-min "$WINDOW_MIN" \
        --decorr-km  "$DECORR_KM" \
        --rope-delta "$ROPE_DELTA" \
        --fname-suffix "_r${R}km" \
        --exclude-sites "$EXCLUDE_SITES"
    # only add the 50 km robustness pass when the primary radius isn't already 50
    [[ "$RADIUS_KM" == 50 ]] && break
done

echo ""
echo "DONE.  Phase-4 outputs under $OUT_BASE :"
echo "  tccon_uncertainty_r*.{md,csv}   — per-case D ± u_D, z, TOST equiv, budget parts"
echo "  tccon_comparison_r*.md          — main report now carries the Phase-4 block"
echo "  (read ⟨z²⟩ in the block: ≈1 ⇒ budget calibrated; else tune DECORR_KM)"

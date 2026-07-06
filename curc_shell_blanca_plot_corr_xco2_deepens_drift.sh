#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_plot_deepens_drift
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable

# ══════════════════════════════════════════════════════════════════════════════
# POST-AQUA-DRIFT sibling of curc_shell_blanca_plot_corr_xco2_deepens.sh.
#
# Same deep-ensemble scaffold (same model dirs, same run_case function, same two
# aggregate reports), but for OCO-2 overpasses AFTER Aqua left the A-Train and
# began its free drift (2023 onward).  For these dates the OCO-2↔MODIS temporal
# collocation is no longer tight, so we DO NOT lean on the MODIS-derived
# nearest-cloud distance here.  Per project guidance:
#
#     For drift dates: focus on OCO-2 vs TCCON; use the MODIS Aqua image
#     ONLY as a background reference (--modis-auto), not for cloud distance.
#
# Concretely, relative to the non-drift deepens script:
#   • --modis-auto stays (GIBS Aqua composite as the map backdrop) and we pass
#     --modis-date "$date" so the SAME-DAY Aqua image is fetched regardless of
#     how far Aqua has drifted from the OCO overpass time.
#   • --results-h5 is still passed when present, but only as a diagnostic
#     (it feeds the optional nearest-cloud-distance panel + helps MODIS granule
#     selection).  The headline comparison is OCO-2 corrected XCO2 vs TCCON.
#   • Outputs land in a separate .../deep_ensemble/<MODEL_TAG>/drift tree so the
#     drift aggregate figures never overwrite the main (A-Train-era) ones, and
#     the aggregate reports parse THIS script (SCRIPT_NAME below) so they
#     summarize only the drift cases.
#
# ── HOW THE CASES WERE POPULATED ──────────────────────────────────────────────
# The dates are the post-drift event dates from DATES_EVENTS in
# curc_shell_blanca_fp_anal_perdate.sh / _build_feature_dataset_perdate.sh.
# The run_case lines below were AUTO-DERIVED from the local
# results/csv_collection/combined_<date>_all_orbits.parquet files: for each date
# the TCCON station is the one with the most footprints within 100 km, the LON/LAT
# box bounds those footprints, VMIN/VMAX are the 2nd/98th xco2_bc percentiles, SURF
# is the dominant sfc_type, and TCCON_AVAIL (12th col) is 'yes' iff a station obs
# falls within ±60 min of the overpass.  Regenerate after (re)building parquets
# with the scratch helper gen_drift_cases.py, then paste its output over the block.
#   • 27 of the 40 dates had a local parquet and are emitted as active run_case
#     lines; the other 13 are commented stubs (their parquet isn't downloaded yet).
#   • Hand-tune LON/LAT/VMIN/VMAX to taste; REQUIRE_TCCON=1 skips AVAIL!=yes lines.
#   • To recompute TCCON_AVAIL against the same collocator the reports use:
#       PYTHONPATH=src python workspace/check_tccon_availability.py --emit-flags
#
# Args (positional) — identical to the non-drift script:
#   run_case DATE TCCON_FILE LON_MIN LON_MAX LAT_MIN LAT_MAX VMIN VMAX [SURF] [POSTER] [SITE] [TCCON_AVAIL]
#     SURF   : both | ocean | land     (default both)
#     POSTER : poster | ""             (default "" — no poster figure)
#     SITE   : short station label     (optional; sets output dir combined_DATE_SITE)
#     TCCON_AVAIL : yes | no | unknown (12th col; REQUIRE_TCCON skips non-yes)
# ══════════════════════════════════════════════════════════════════════════════

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

# Under SLURM, $0 is the spooled copy (/var/spool/slurmd/...), so dirname $0 is NOT
# the repo.  Use the submit dir when set; fall back to dirname $0 for local runs.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export PYTHONPATH=src:${PYTHONPATH:-}
# Repo-relative name of THIS script (the aggregate reports parse its run_case lines).
# Don't use basename "$0": under SLURM that's the spooled job name, not this file.
SCRIPT_NAME=curc_shell_blanca_plot_corr_xco2_deepens_drift.sh

# ─── data root ────────────────────────────────────────────────────────────────
# On CURC the results/ tree and data/ (TCCON) live under scratch, not the repo
# checkout.  Prefer CURC_DATA_ROOT, then OCO2_DATAROOT, else '.' (local repo).
DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
# Propagate to the aggregate report scripts so their hardcoded paths follow suit.
export OCO2_DATAROOT="$DATA_ROOT"

# ─── model (deep-ensemble fold dirs; per-surface) ─────────────────────────────
# Production M=5 DE + profile-EOF block (--profile-pca is embedded in each fold's
# pipeline pkl).  OCEAN uses the r05 (0.5°) near-cloud-anomaly variant, LAND uses
# r15 (1.5°) — the per-surface best profile+reg models.  Identical to the
# non-drift deepens script so the correction is the SAME model across eras.
MODEL_TAG=de_beta_nll_prof_reg_o05l15_m5
OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_ocean_beta_nll_prof_reg_r05_f*)
LAND_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_land_beta_nll_prof_reg_r15_f*)

# ─── cloud classifier (xgb_cloud fold dirs; per-surface) ──────────────────────
# Emits P(near) + the two diagnostic correction columns.  Kept for parity with
# the non-drift script (production correction is the full mu, not the gated form).
OCEAN_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_ocean_f*)
LAND_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_land_f*)

CSV_DIR="$DATA_ROOT"/results/csv_collection
# Separate 'drift' subtree so the post-drift aggregate figures never overwrite the
# A-Train-era deepens ones (same MODEL_TAG, different leaf).
OUT_BASE="$DATA_ROOT"/results/model_comparison/deep_ensemble/${MODEL_TAG}/drift

# ─── TCCON collocation knobs (shared by per-case plot + aggregate reports) ─────
RADIUS_KM="${RADIUS_KM:-100}"
WINDOW_MIN="${WINDOW_MIN:-60}"
RADIUS_TAG="r${RADIUS_KM}km"

# REQUIRE_TCCON=1 → SKIP cases whose 12th run_case column (TCCON_AVAIL) is not "yes".
# Recompute flags: PYTHONPATH=src python workspace/check_tccon_availability.py --emit-flags
REQUIRE_TCCON=1

run_case() {
    local date="$1" tccon="$2" lonmin="$3" lonmax="$4" latmin="$5" latmax="$6"
    local vmin="$7" vmax="$8" surf="${9:-both}" poster="${10:-}" site="${11:-}"
    local tccon_avail="${12:-yes}"   # yes|no|unknown — TCCON present at OCO pass time

    local input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    local h5="$DATA_ROOT/results/results_${date}.h5"
    # site label (11th arg) keeps same-date/different-station runs in separate dirs
    local outdir="$OUT_BASE/combined_${date}_all_orbits"
    [[ -n "$site" ]] && outdir="$OUT_BASE/combined_${date}_${site}"
    local plotdata="$outdir/plot_data.parquet"

    echo ""
    echo "############ DRIFT CASE $date  ($surf)  ############"
    if [[ "$REQUIRE_TCCON" == 1 && "$tccon_avail" != yes ]]; then
        echo "  SKIP (REQUIRE_TCCON=1; TCCON_AVAIL=$tccon_avail — no TCCON at OCO pass time)"; return
    fi
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"; return
    fi
    mkdir -p "$outdir"

    # (4) apply deep ensemble → plot_data.parquet
    local model_args=()
    [[ "$surf" == both || "$surf" == ocean ]] && model_args+=(--ocean-model-dir "${OCEAN_MODEL_DIRS[@]}" --ocean-cloud-model-dir "${OCEAN_CLOUD_DIRS[@]}")
    [[ "$surf" == both || "$surf" == land  ]] && model_args+=(--land-model-dir  "${LAND_MODEL_DIRS[@]}" --land-cloud-model-dir "${LAND_CLOUD_DIRS[@]}")
    python workspace/build_deepens_plot_data.py \
        "${model_args[@]}" --input "$input" --output "$plotdata" || { echo "  build failed"; return; }

    # (5) plot — OCO-2 corrected XCO2 vs TCCON, with the SAME-DAY Aqua image as a
    #     background reference only (--modis-date "$date" pins the composite to the
    #     overpass day rather than a drift-shifted MODIS granule).  --results-h5 is
    #     passed when present but only feeds the optional cloud-distance panel.
    local h5_arg=(); [[ -f "$h5" ]] && h5_arg=(--results-h5 "$h5")
    local poster_arg=(); [[ "$poster" == poster ]] && \
        poster_arg=(--poster-model deep_ensemble_corrected_xco2 --poster-dpi 300)
    python workspace/plot_corrected_xco2.py \
        --plot-data  "$plotdata" \
        --tccon      "$DATA_ROOT/data/TCCON/$tccon" \
        "${h5_arg[@]}" \
        --output-dir "$outdir" \
        --modis-auto --modis-date "$date" \
        --lon-range  "$lonmin" "$lonmax" \
        --lat-range  "$latmin" "$latmax" \
        --date-plot  "$date" \
        --vmin "$vmin" --vmax "$vmax" \
        --hist-radius-km "$RADIUS_KM" \
        "${poster_arg[@]}"

    # (6) per-band spectral-fit parameter maps (k1/k2/exp_intercept-alb × o2a/wco2/sco2)
    python workspace/plot_spectral_params.py \
        --input      "$input" \
        --tccon      "$DATA_ROOT/data/TCCON/$tccon" \
        "${h5_arg[@]}" \
        --output-dir "$outdir" \
        --modis-auto \
        --lon-range  "$lonmin" "$lonmax" \
        --lat-range  "$latmin" "$latmax" \
        --date-plot  "$date"
}

# ═══════════════════════ post-drift cases ═════════════════════════════════════
# AUTO-DERIVED from the local combined_<date>_all_orbits.parquet files
# (workspace scratch: gen_drift_cases.py).  For each date the TCCON station is the
# one with the most footprints within 100 km; the LON/LAT box bounds those
# footprints; VMIN/VMAX are the 2nd/98th percentiles of xco2_bc there; SURF is the
# dominant sfc_type; the 12th column (AVAIL) is 'yes' iff a TCCON obs falls within
# ±60 min of the overpass.  Trailing "# n=…, dmin=…" = footprints-in-radius and
# nearest-sounding distance to the station (sanity check).
# Re-derive after (re)building parquets:
#   python <scratch>/gen_drift_cases.py            # re-run the generator
# Hand-tune LON/LAT/VMIN/VMAX per taste; REQUIRE_TCCON=1 skips AVAIL!=yes lines.
#         DATE         TCCON_FILE                          LON_MIN  LON_MAX  LAT_MIN  LAT_MAX   VMIN   VMAX  SURF  POSTER SITE AVAIL

# All 40 requested dates now have a local parquet; all 40 resolve to a station,
# 21 have coincident TCCON (AVAIL=yes).  Grouped by station.

# -- burgos01 (bu) --
run_case  2023-01-23   bu20170303_20250221.public.qc.nc     120.59   120.80    18.36    18.54   416.5  421.5  land  poster bu  yes   # n=3844 dmin=0km
run_case  2023-02-10   bu20170303_20250221.public.qc.nc     120.97   121.19    17.77    18.58   417.5  422.0  land  poster bu  yes   # n=262 dmin=35km
run_case  2023-03-21   bu20170303_20250221.public.qc.nc     120.59   120.78    18.36    18.55   417.0  422.0  land  poster bu  no    # n=4049 dmin=0km
run_case  2023-03-23   bu20170303_20250221.public.qc.nc     120.46   120.91    17.66    19.42   415.0  423.0  both  poster bu  yes   # n=400 dmin=7km
run_case  2023-05-26   bu20170303_20250221.public.qc.nc     120.75   121.22    17.82    19.42   420.0  424.5  ocean poster bu  no    # n=361 dmin=33km
run_case  2023-07-20   bu20170303_20250221.public.qc.nc     120.98   121.17    18.06    18.59   417.0  422.5  land  poster bu  no    # n=34 dmin=35km
run_case  2024-02-06   bu20170303_20250221.public.qc.nc     120.20   120.68    17.63    19.34   420.0  424.5  both  poster bu  no    # n=525 dmin=22km
run_case  2024-04-15   bu20170303_20250221.public.qc.nc     120.59   120.80    18.35    18.54   418.5  427.0  land  poster bu  no    # n=3679 dmin=0km

# -- edwards01 (df) --
run_case  2023-06-26   df20130720_20260121.public.qc.nc    -117.99  -117.76    34.80    35.10   418.0  420.0  land  poster df  yes   # n=6246 dmin=1km

# -- east trout lake (et; overpass 84 km from station — marginal clip) --
run_case  2024-06-26   et20161003_20260326.public.qc.nc    -106.24  -105.97    53.67    54.09   415.0  423.5  both  poster et  yes   # n=35 dmin=84km

# -- izana01 (iz; TCCON record ends 2023-08-30) --
run_case  2023-03-13   iz20140102_20230830.public.qc.nc     -16.60   -16.41    28.14    28.42   413.5  426.5  land  poster iz  no    # n=5337 dmin=0km
run_case  2023-08-04   iz20140102_20230830.public.qc.nc     -16.63   -16.42    28.13    28.42   406.5  424.0  land  poster iz  no    # n=5463 dmin=0km

# -- karlsruhe01 (ka; qc filename ends 2023-06-26 but has coincident 2023 obs) --
run_case  2023-04-04   ka20140115_20230626.public.qc.nc       8.29     8.59    48.95    49.25   420.0  423.0  land  poster ka  yes   # n=5640 dmin=0km
run_case  2023-08-10   ka20140115_20230626.public.qc.nc       8.29     8.60    48.94    49.24   414.5  419.5  land  poster ka  yes   # n=5785 dmin=0km
run_case  2023-09-11   ka20140115_20230626.public.qc.nc       8.17     8.92    48.25    49.99   415.5  421.0  land  poster ka  yes   # n=455 dmin=7km
run_case  2024-10-29   ka20140115_20230626.public.qc.nc       8.28     8.61    48.96    49.23   421.5  426.5  land  poster ka  no    # n=5357 dmin=1km

# -- ny-alesund01 (ny; overpass 25 km — no coincident TCCON) --
run_case  2023-07-04   ny20050316_20250524.public.qc.nc      11.21    14.88    79.11    79.64   408.5  425.0  land  poster ny  no    # n=69 dmin=25km

# -- orleans01 (or; overpass 59 km — marginal, no coincident TCCON) --
run_case  2024-08-26   or20090906_20250411.public.qc.nc       2.53     3.23    47.43    48.81   414.0  419.0  land  poster or  no    # n=535 dmin=59km

# -- parkfalls01 (pa) --
run_case  2023-03-19   pa20040602_20260123.public.qc.nc     -90.42   -90.12    45.80    46.06   419.5  423.0  land  poster pa  yes   # n=5631 dmin=0km
run_case  2024-03-10   pa20040602_20260123.public.qc.nc     -90.39   -90.14    45.78    46.06   422.5  425.5  land  poster pa  yes   # n=4873 dmin=1km

# -- reunion01 (ra; TCCON record ends 2020-07-18 → no coincident data, will SKIP) --
run_case  2023-06-11   ra20150301_20200718.public.qc.nc      55.15    55.63   -21.80   -20.03   415.0  419.0  ocean poster ra  no    # n=424 dmin=19km
run_case  2024-05-10   ra20150301_20200718.public.qc.nc      55.43    55.61   -21.08   -20.88   404.0  425.5  land  poster ra  no    # n=2010 dmin=0km
run_case  2024-07-31   ra20150301_20200718.public.qc.nc      55.20    55.68   -21.79   -20.02   420.5  424.0  ocean poster ra  no    # n=583 dmin=4km

# -- wollongong01 (wg) --
run_case  2023-01-28   wg20130104_20260224.public.qc.nc     150.74   150.92   -34.61   -34.31   404.0  419.0  land  poster wg  yes   # n=4063 dmin=0km
run_case  2023-04-25   wg20130104_20260224.public.qc.nc     150.76   150.92   -34.59   -34.33   397.5  418.5  land  poster wg  yes   # n=2059 dmin=0km
run_case  2023-05-06   wg20130104_20260224.public.qc.nc     150.55   151.05   -35.31   -33.54   411.5  416.5  both  poster wg  yes   # n=523 dmin=3km
run_case  2023-05-20   wg20130104_20260224.public.qc.nc     150.72   150.92   -34.52   -34.30   401.5  418.5  land  poster wg  yes   # n=2170 dmin=0km
run_case  2023-09-02   wg20130104_20260224.public.qc.nc     150.76   150.93   -34.60   -34.31   403.5  420.5  land  poster wg  no    # n=3510 dmin=0km
run_case  2023-10-11   wg20130104_20260224.public.qc.nc     150.73   150.94   -34.62   -34.31   409.0  420.0  land  poster wg  no    # n=4179 dmin=0km
run_case  2023-10-13   wg20130104_20260224.public.qc.nc     150.25   150.75   -35.30   -33.63   412.5  420.5  land  poster wg  no    # n=277 dmin=32km
run_case  2024-02-18   wg20130104_20260224.public.qc.nc     151.08   151.34   -35.25   -34.30   416.5  420.5  ocean poster wg  no    # n=236 dmin=23km
run_case  2024-04-22   wg20130104_20260224.public.qc.nc     150.73   150.94   -34.59   -34.30   410.5  422.0  land  poster wg  yes   # n=4184 dmin=0km
run_case  2024-07-27   wg20130104_20260224.public.qc.nc     150.77   151.11   -34.61   -33.51   410.5  421.5  both  poster wg  yes   # n=89 dmin=13km
run_case  2024-08-03   wg20130104_20260224.public.qc.nc     150.74   150.93   -34.60   -34.29   405.5  424.5  land  poster wg  yes   # n=3427 dmin=0km
run_case  2024-09-20   wg20130104_20260224.public.qc.nc     150.75   150.93   -34.62   -34.31   414.5  425.5  land  poster wg  yes   # n=3929 dmin=0km
run_case  2024-11-23   wg20130104_20260224.public.qc.nc     150.77   150.93   -34.62   -34.31   417.0  427.0  land  poster wg  yes   # n=3837 dmin=0km
run_case  2024-12-02   wg20130104_20260224.public.qc.nc     150.17   150.64   -35.29   -33.68   421.5  428.0  land  poster wg  yes   # n=248 dmin=41km
run_case  2024-12-16   wg20130104_20260224.public.qc.nc     150.73   150.93   -34.60   -34.31   417.0  425.5  land  poster wg  no    # n=4216 dmin=0km

# -- xianghe01 (xh) --
run_case  2023-08-14   xh20180614_20241231.public.qc.nc     116.81   117.38    38.95    40.70   410.0  416.5  land  poster xh  yes   # n=565 dmin=11km
run_case  2024-10-03   xh20180614_20241231.public.qc.nc     116.54   117.10    38.91    40.65   420.0  424.0  land  poster xh  no    # n=666 dmin=9km

# ═══════════════════════ aggregate reports (run once, after all cases) ════════════
# Both parse the active run_case lines above and summarize across the DRIFT cases.

# (7) before/after-vs-TCCON comparison (reads each case's plot_data.parquet).
echo ""
echo "############ AGGREGATE (drift): tccon_comparison_report ############"
python workspace/tccon_comparison_report.py \
    --script   "$SCRIPT_NAME" \
    --out-base "$OUT_BASE" \
    --output-dir "$OUT_BASE" \
    --fname-suffix "_${RADIUS_TAG}" --exclude-sites ny \
    --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" \
    --ak-harmonize

# (8) correction-policy stats — reads step-4 plot_data via --plotdata-base.
echo ""
echo "############ AGGREGATE (drift): tccon_correction_policy_stats ############"
python workspace/tccon_correction_policy_stats.py \
    --script   "$SCRIPT_NAME" \
    --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" \
    --plotdata-base "$OUT_BASE" --output-dir "$OUT_BASE" --fname-suffix "_${RADIUS_TAG}"

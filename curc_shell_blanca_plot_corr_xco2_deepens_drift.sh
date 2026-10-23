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
# foldpca (2026-07-15): fold-specific ProfilePCA production models (leakage-safe).
# Run version (log/PIPELINE_FIX_RERUN_PLAN_2026-10-14.md): v2 = unsmoothed gamma,
# fill/spike imputation, RobustClipScaler, no t700.  The tag sits BEFORE the fold
# tag, so v1 globs (..._f*) never match v2 dirs and v2 runs never overwrite v1
# checkpoints.  RUN_VERSION= (empty) reproduces the v1 names.
export RUN_VERSION=${RUN_VERSION-v2}
RV=${RUN_VERSION:+_${RUN_VERSION}}

MODEL_TAG=de_beta_nll_prof_reg_foldpca_o05l15_m5${RV}
OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_ocean_beta_nll_prof_reg_foldpca_r05${RV}_f*)
LAND_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_land_beta_nll_prof_reg_foldpca_r15${RV}_f*)

# ─── cloud classifier (xgb_cloud fold dirs; per-surface) ──────────────────────
# Emits P(near) + the two diagnostic correction columns.  Kept for parity with
# the non-drift script (production correction is the full mu, not the gated form).
OCEAN_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_ocean_f*)
LAND_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_land_f*)
# The xgb_cloud classifier only exists on CURC; locally, drop the (diagnostic-only)
# P(near)/gate columns instead of crashing the build on an unexpanded glob.
[[ -d "${OCEAN_CLOUD_DIRS[0]}" ]] && OCEAN_CLOUD_ARGS=(--ocean-cloud-model-dir "${OCEAN_CLOUD_DIRS[@]}") || OCEAN_CLOUD_ARGS=()
[[ -d "${LAND_CLOUD_DIRS[0]}"  ]] && LAND_CLOUD_ARGS=(--land-cloud-model-dir  "${LAND_CLOUD_DIRS[@]}")  || LAND_CLOUD_ARGS=()

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
    # SKIP_BUILD=1 (env) → reuse an existing plot_data.parquet and only redo the
    # plots — for local re-plotting (e.g. MODIS/GIBS overlays) against plot_data
    # freshly synced from CURC.  Falls through to a build when the file is absent.
    if [[ "${SKIP_BUILD:-0}" == 1 && -f "$plotdata" ]]; then
        echo "  SKIP build (SKIP_BUILD=1) — reusing $plotdata"
    else
        local model_args=()
        [[ "$surf" == both || "$surf" == ocean ]] && model_args+=(--ocean-model-dir "${OCEAN_MODEL_DIRS[@]}" "${OCEAN_CLOUD_ARGS[@]}")
        [[ "$surf" == both || "$surf" == land  ]] && model_args+=(--land-model-dir  "${LAND_MODEL_DIRS[@]}" "${LAND_CLOUD_ARGS[@]}")
        python workspace/build_deepens_plot_data.py \
            "${model_args[@]}" --input "$input" --output "$plotdata" || { echo "  build failed"; return; }
    fi

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
run_case  2023-01-23   bu20170303_20250409.public.qc.nc     120.59   120.80    18.36    18.54   416.5  421.5  land  poster bu  yes   # n=3844 dmin=0km
run_case  2023-02-10   bu20170303_20250409.public.qc.nc     120.97   121.19    17.77    18.58   417.5  422.0  land  poster bu  yes   # n=262 dmin=35km
run_case  2023-03-21   bu20170303_20250409.public.qc.nc     120.59   120.78    18.36    18.55   417.0  422.0  land  poster bu  no    # n=4049 dmin=0km
run_case  2023-03-23   bu20170303_20250409.public.qc.nc     120.46   120.91    17.66    19.42   415.0  423.0  both  poster bu  yes   # n=400 dmin=7km
run_case  2023-05-26   bu20170303_20250409.public.qc.nc     120.75   121.22    17.82    19.42   420.0  424.5  ocean poster bu  no    # n=361 dmin=33km
run_case  2023-07-20   bu20170303_20250409.public.qc.nc     120.98   121.17    18.06    18.59   417.0  422.5  land  poster bu  no    # n=34 dmin=35km
run_case  2024-02-06   bu20170303_20250409.public.qc.nc     120.20   120.68    17.63    19.34   420.0  424.5  both  poster bu  no    # n=525 dmin=22km
run_case  2024-04-15   bu20170303_20250409.public.qc.nc     120.59   120.80    18.35    18.54   418.5  427.0  land  poster bu  no    # n=3679 dmin=0km

# -- edwards01 (df) --
run_case  2023-06-26   df20130720_20260121.public.qc.nc    -117.99  -117.76    34.80    35.10   418.0  420.0  land  poster df  yes   # n=6246 dmin=1km

# -- east trout lake (et; overpass 84 km from station — marginal clip) --
run_case  2024-06-26   et20161003_20260422.public.qc.nc    -106.24  -105.97    53.67    54.09   415.0  423.5  both  poster et  yes   # n=35 dmin=84km

# -- izana01 (iz; TCCON record ends 2023-08-30) --
run_case  2023-03-13   iz20140102_20230830.public.qc.nc     -16.60   -16.41    28.14    28.42   413.5  426.5  land  poster iz  no    # n=5337 dmin=0km
run_case  2023-08-04   iz20140102_20230830.public.qc.nc     -16.63   -16.42    28.13    28.42   406.5  424.0  land  poster iz  no    # n=5463 dmin=0km

# -- karlsruhe01 (ka; qc filename ends 2023-06-26 but has coincident 2023 obs) --
run_case  2023-04-04   ka20140115_20241227.public.qc.nc       8.29     8.59    48.95    49.25   420.0  423.0  land  poster ka  yes   # n=5640 dmin=0km
run_case  2023-08-10   ka20140115_20241227.public.qc.nc       8.29     8.60    48.94    49.24   414.5  419.5  land  poster ka  yes   # n=5785 dmin=0km
run_case  2023-09-11   ka20140115_20241227.public.qc.nc       8.17     8.92    48.25    49.99   415.5  421.0  land  poster ka  yes   # n=455 dmin=7km
run_case  2024-10-29   ka20140115_20241227.public.qc.nc       8.28     8.61    48.96    49.23   421.5  426.5  land  poster ka  no    # n=5357 dmin=1km

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


# ═══════════════════════ added 2026-09-15: new stations, drift era (3 station-days) ════════════════
# Generated by workspace/newsites_cases.py from the full GGG2020 network download
# (data/TCCON, 35 stations). Box = footprints within 100 km (floor/ceil 0.01°),
# VMIN/VMAX = xco2_bc percentiles, SURF = dominant surface. Stations are not in
# tccon_collocate.SITE_COORDS, so the collocator uses the median-observation position.
#         DATE         TCCON                              LON_MIN  LON_MAX  LAT_MIN  LAT_MAX   VMIN   VMAX  SURF  POSTER SITE  AVAIL
# -- garmisch01 (gm) --
run_case  2023-07-04   gm20070718_20240116.public.qc.nc      11.31    11.62    47.75    48.36   411.0  422.0  land  poster gm  yes   # n=25 dmin=52km
# -- hefei01 (hf) --
run_case  2024-05-10   hf20151102_20251230.public.qc.nc     116.21   116.39    31.63    32.27   424.5  427.5  land  poster hf  yes   # n=244 dmin=79km
# -- lauder03 (lr) --
run_case  2023-03-13   lr20181002_20250629.public.qc.nc     169.04   169.38   -45.38   -44.39   410.5  419.0  land  poster lr  yes   # n=63 dmin=26km


# ═══════════════════════ added 2026-10-21: runner-up stations on already-processed dates ════════════════
# gen_drift_cases.py assigned each date to the single station with the most footprints;
# these are the runner-up stations on the same dates (>= 20 footprints within 100 km,
# >= 1 TCCON observation within +-60 min).  Scan: log/JQSRT_plan/tccon_overpass_candidates_2026-10-21.md.
#         DATE         TCCON                              LON_MIN  LON_MAX  LAT_MIN  LAT_MAX   VMIN   VMAX  SURF  POSTER SITE  AVAIL
# -- karlsruhe01 (ka) --
run_case  2023-10-13   ka20140115_20241227.public.qc.nc       8.43     9.03    48.27    49.65   413.0  419.0  land  poster ka  yes   # n=257 dmin=14km
# -- orleans01 (or) --
run_case  2023-10-11   or20090906_20250411.public.qc.nc       1.68     1.82    47.08    47.28   416.0  419.0  land  poster or  yes   # n=78 dmin=82km
# -- paris01 (pr) --
run_case  2024-08-26   pr20140923_20251120.public.qc.nc       2.18     2.95    48.03    49.49   408.0  419.5  land  poster pr  yes   # n=403 dmin=11km
# -- wollongong01 (wg) --
run_case  2023-03-19   wg20130104_20260224.public.qc.nc     150.73   150.94   -34.59   -34.29   414.5  420.5  land  poster wg  yes   # n=4092 dmin=0km
run_case  2023-04-04   wg20130104_20260224.public.qc.nc     150.65   151.14   -35.28   -33.63   413.0  417.0  both  poster wg  yes   # n=302 dmin=1km
run_case  2024-03-10   wg20130104_20260224.public.qc.nc     150.72   150.93   -34.61   -34.32   413.0  422.5  land  poster wg  yes   # n=3660 dmin=0km

# ═══════════════════════ added 2026-10-22: post-drift TCCON collocations on the v2 rerun dates (24 station-days) ════════════════
# The 2026-10-22 v2 rerun dates (first fitted today; parquets in results/csv_collection).
# Generated by workspace/gen_tccon_cases.py: one line per TCCON station with >= 10
# footprints within 100 km (the TCCON-comparison subset guard) AND >= 1 TCCON
# observation within +-60 min of the overpass (AVAIL=yes).  Box = those footprints
# (floor/ceil 0.01 deg); VMIN/VMAX = xco2_bc 2nd/98th percentiles; SURF = dominant
# surface.  Trailing comment: n = footprints in radius, dmin = nearest footprint,
# tccon_obs = TCCON observations in the window.  6 further station-days
# met the footprint guard but had no TCCON data at pass time and were left out.
# None of these dates is in any production fold's training/calibration manifest.
#         DATE         TCCON                                LON_MIN  LON_MAX  LAT_MIN  LAT_MAX   VMIN   VMAX  SURF  POSTER SITE  AVAIL
# -- burgos01 (bu; lat 18.53, lon 120.65) --
run_case  2025-01-23   bu20170303_20250409.public.qc.nc       120.02   120.47    17.65    19.23   421.0  427.0  ocean poster bu  yes   # n=514 dmin=43km tccon_obs=63
# -- pasadena01 (ci; lat 34.14, lon -118.13) --
run_case  2023-02-20   ci20120920_20260125.public.qc.nc      -117.73  -117.22    33.61    34.98   419.0  423.0  land  poster ci  yes   # n=588 dmin=61km tccon_obs=25
run_case  2024-07-14   ci20120920_20260125.public.qc.nc      -118.17  -117.70    33.45    35.04   422.0  427.5  land  poster ci  yes   # n=473 dmin=20km tccon_obs=27
# -- edwards01 (df; lat 34.96, lon -117.88) --
run_case  2023-02-20   df20130720_20260121.public.qc.nc      -118.03  -117.39    34.14    35.86   419.0  423.0  land  poster df  yes   # n=737 dmin=14km tccon_obs=26
run_case  2023-10-02   df20130720_20260121.public.qc.nc      -118.21  -117.59    34.09    35.83   411.0  420.0  land  poster df  yes   # n=489 dmin=1km tccon_obs=48
run_case  2024-07-14   df20130720_20260121.public.qc.nc      -118.21  -117.87    34.06    35.20   423.0  428.0  land  poster df  yes   # n=327 dmin=22km tccon_obs=38
# -- easttroutlake01 (et; lat 54.35, lon -104.99) --
run_case  2023-07-10   et20161003_20260422.public.qc.nc      -106.40  -105.67    53.54    54.74   413.0  417.5  land  poster et  yes   # n=437 dmin=70km tccon_obs=47
# -- garmisch01 (gm; lat 47.48, lon 11.06) --
run_case  2023-09-04   gm20070718_20240116.public.qc.nc        10.43    11.12    46.58    48.29   413.0  420.0  land  poster gm  yes   # n=380 dmin=20km tccon_obs=32
run_case  2023-09-06   gm20070718_20240116.public.qc.nc        11.36    11.70    48.03    48.32   414.5  417.5  land  poster gm  yes   # n=5610 dmin=70km tccon_obs=33
# -- hefei01 (hf; lat 31.91, lon 117.17) --
run_case  2025-05-29   hf20151102_20251230.public.qc.nc       116.13   116.34    31.38    32.04   426.0  431.0  land  poster hf  yes   # n=146 dmin=90km tccon_obs=74
# -- harwell01 (hw; lat 51.57, lon -1.32) --
run_case  2023-06-10   hw20210530_20260331.public.qc.nc        -0.51    -0.04    51.34    52.32   412.5  418.5  land  poster hw  yes   # n=320 dmin=76km tccon_obs=67
run_case  2024-07-30   hw20210530_20260331.public.qc.nc        -0.57    -0.00    51.16    52.34   416.5  421.5  land  poster hw  yes   # n=453 dmin=73km tccon_obs=84
# -- saga01 (js; lat 33.24, lon 130.29) --
run_case  2023-10-02   js20110728_20231213.public.qc.nc       129.64   130.15    32.50    33.97   416.0  420.5  both  poster js  yes   # n=517 dmin=34km tccon_obs=57
# -- lauder03 (lr; lat -45.04, lon 169.68) --
run_case  2024-07-14   lr20181002_20250629.public.qc.nc       170.28   170.52   -45.18   -44.29   402.5  427.5  land  poster lr  yes   # n=13 dmin=61km tccon_obs=90
run_case  2024-11-03   lr20181002_20250629.public.qc.nc       169.66   170.23   -45.88   -44.16   419.0  427.0  land  poster lr  yes   # n=651 dmin=14km tccon_obs=90
# -- lamont01 (oc; lat 36.60, lon -97.49) --
run_case  2023-07-10   oc20110416_20260122.public.qc.nc       -98.59   -98.31    36.00    36.81   415.5  418.0  land  poster oc  yes   # n=350 dmin=89km tccon_obs=55
run_case  2024-07-30   oc20110416_20260122.public.qc.nc       -97.63   -97.31    36.44    36.74   416.5  423.5  land  poster oc  yes   # n=5650 dmin=0km tccon_obs=56
# -- orleans01 (or; lat 47.96, lon 2.11) --
run_case  2024-07-30   or20090906_20250411.public.qc.nc         1.08     1.23    48.19    48.56   422.5  426.0  land  poster or  yes   # n=45 dmin=71km tccon_obs=7
# -- parkfalls01 (pa; lat 45.94, lon -90.27) --
run_case  2024-06-25   pa20040602_20260123.public.qc.nc       -90.41   -90.12    45.80    46.10   412.5  423.0  land  poster pa  yes   # n=5044 dmin=0km tccon_obs=46
# -- paris01 (pr; lat 48.85, lon 2.36) --
run_case  2024-07-30   pr20140923_20251120.public.qc.nc         1.01     1.16    48.43    48.72   421.5  426.0  land  poster pr  yes   # n=62 dmin=97km tccon_obs=82
# -- wollongong01 (wg; lat -34.41, lon 150.88) --
run_case  2024-06-25   wg20130104_20260224.public.qc.nc       150.64   151.14   -35.30   -33.52   417.0  421.5  both  poster wg  yes   # n=459 dmin=3km tccon_obs=32
# -- xianghe01 (xh; lat 39.80, lon 116.96) --
run_case  2023-09-06   xh20180614_20241231.public.qc.nc       116.84   117.09    39.66    39.90   398.0  421.0  land  poster xh  yes   # n=1638 dmin=1km tccon_obs=8
run_case  2023-09-15   xh20180614_20241231.public.qc.nc       116.71   117.22    38.92    40.47   416.0  419.5  land  poster xh  yes   # n=589 dmin=1km tccon_obs=19
run_case  2024-01-21   xh20180614_20241231.public.qc.nc       116.62   117.21    38.92    40.67   421.0  425.5  land  poster xh  yes   # n=470 dmin=2km tccon_obs=18

# ═══════════════════════ added 2026-10-22 (evening): post-drift TCCON collocations on the 2026-10-22 afternoon batch (36 station-days) ════════════════
# Dates from the active date_list in src/analysis/build_feature_dataset.py (the
# afternoon oco_spectral_anal batch; parquets built by bfd arrays 29013910 and
# 29014520).  Generated by workspace/gen_tccon_cases.py: one line per TCCON station
# with >= 10 footprints within 100 km (the TCCON-comparison subset guard) AND >= 1
# TCCON observation within +-60 min of the overpass (AVAIL=yes).  Box = those
# footprints (floor/ceil 0.01 deg); VMIN/VMAX = xco2_bc 2nd/98th percentiles; SURF =
# dominant surface.  Trailing comment: n = footprints in radius, dmin = nearest
# footprint, tccon_obs = TCCON observations in the window.  9 further station-days
# met the footprint guard but had no TCCON data at pass time and were left out.
# None of these dates is in any production fold's train/calib/held manifest.
#         DATE         TCCON                                LON_MIN  LON_MAX  LAT_MIN  LAT_MAX   VMIN   VMAX  SURF  POSTER SITE  AVAIL
# -- pasadena01 (ci; lat 34.14, lon -118.13) --
run_case  2024-04-09   ci20120920_20260125.public.qc.nc      -117.98  -117.50    33.40    35.03   423.5  427.5  land  poster ci  yes   # n=544 dmin=36km tccon_obs=46
run_case  2024-06-10   ci20120920_20260125.public.qc.nc      -118.21  -117.94    33.99    34.26   424.5  430.5  land  poster ci  yes   # n=5794 dmin=0km tccon_obs=41
run_case  2025-04-28   ci20120920_20260125.public.qc.nc      -118.03  -117.69    33.85    35.04   425.0  431.0  land  poster ci  yes   # n=325 dmin=32km tccon_obs=14
run_case  2025-06-08   ci20120920_20260125.public.qc.nc      -118.28  -117.81    33.56    35.04   422.5  431.5  land  poster ci  yes   # n=607 dmin=11km tccon_obs=41
run_case  2025-07-10   ci20120920_20260125.public.qc.nc      -118.05  -117.52    33.38    35.04   425.0  430.5  land  poster ci  yes   # n=698 dmin=32km tccon_obs=46
run_case  2025-07-15   ci20120920_20260125.public.qc.nc      -118.23  -117.93    33.97    34.28   424.0  430.0  land  poster ci  yes   # n=6053 dmin=0km tccon_obs=68
# -- edwards01 (df; lat 34.96, lon -117.88) --
run_case  2024-04-09   df20130720_20260121.public.qc.nc      -118.21  -117.69    34.07    35.83   423.0  427.5  land  poster df  yes   # n=580 dmin=5km tccon_obs=49
run_case  2024-06-10   df20130720_20260121.public.qc.nc      -118.21  -117.96    34.06    34.26   424.0  431.0  land  poster df  yes   # n=4448 dmin=79km tccon_obs=52
run_case  2025-04-28   df20130720_20260121.public.qc.nc      -118.25  -117.75    34.06    35.78   425.0  430.0  land  poster df  yes   # n=489 dmin=11km tccon_obs=50
run_case  2025-06-08   df20130720_20260121.public.qc.nc      -118.49  -117.95    34.06    35.71   426.5  431.5  land  poster df  yes   # n=618 dmin=32km tccon_obs=47
run_case  2025-07-10   df20130720_20260121.public.qc.nc      -118.29  -117.73    34.07    35.80   424.5  430.5  land  poster df  yes   # n=724 dmin=12km tccon_obs=48
run_case  2025-07-15   df20130720_20260121.public.qc.nc      -118.23  -117.95    34.06    34.28   423.5  430.5  land  poster df  yes   # n=4485 dmin=78km tccon_obs=37
run_case  2025-08-11   df20130720_20260121.public.qc.nc      -118.59  -118.08    34.07    35.66   422.5  428.5  land  poster df  yes   # n=667 dmin=42km tccon_obs=50
run_case  2025-12-08   df20130720_20260121.public.qc.nc      -118.20  -117.66    34.07    35.82   422.5  428.5  land  poster df  yes   # n=533 dmin=3km tccon_obs=51
# -- easttroutlake01 (et; lat 54.35, lon -104.99) --
run_case  2025-05-03   et20161003_20260422.public.qc.nc      -105.49  -104.57    53.48    55.21   427.5  431.0  land  poster et  yes   # n=622 dmin=1km tccon_obs=69
run_case  2025-09-08   et20161003_20260422.public.qc.nc      -105.38  -104.47    53.49    55.22   417.5  422.5  land  poster et  yes   # n=556 dmin=3km tccon_obs=69
# -- harwell01 (hw; lat 51.57, lon -1.32) --
run_case  2025-04-28   hw20210530_20260331.public.qc.nc        -1.49    -1.18    51.46    51.71   424.0  430.0  land  poster hw  yes   # n=5613 dmin=0km tccon_obs=85
# -- karlsruhe01 (ka; lat 49.10, lon 8.44) --
run_case  2024-08-28   ka20140115_20241227.public.qc.nc         8.23     8.99    48.26    49.99   417.0  421.5  land  poster ka  yes   # n=644 dmin=11km tccon_obs=26
# -- nyalesund01 (ny; lat 78.92, lon 11.92) --
run_case  2024-06-10   ny20050316_20250524.public.qc.nc         7.85    16.21    78.55    79.24   419.0  426.5  both  poster ny  yes   # n=122 dmin=56km tccon_obs=46
# -- lamont01 (oc; lat 36.60, lon -97.49) --
run_case  2024-06-10   oc20110416_20260122.public.qc.nc       -97.12   -96.94    36.89    37.44   421.5  426.0  land  poster oc  yes   # n=142 dmin=58km tccon_obs=6
run_case  2024-10-25   oc20110416_20260122.public.qc.nc       -97.61   -97.33    36.46    36.72   423.0  426.0  land  poster oc  yes   # n=5645 dmin=0km tccon_obs=54
run_case  2025-02-28   oc20110416_20260122.public.qc.nc       -97.60   -97.35    36.46    36.75   425.0  428.0  land  poster oc  yes   # n=6252 dmin=0km tccon_obs=61
run_case  2025-05-03   oc20110416_20260122.public.qc.nc       -98.49   -98.15    35.88    37.02   426.5  430.5  land  poster oc  yes   # n=447 dmin=76km tccon_obs=55
run_case  2025-07-15   oc20110416_20260122.public.qc.nc       -98.58   -98.30    35.98    36.83   422.0  425.0  land  poster oc  yes   # n=142 dmin=88km tccon_obs=36
run_case  2025-09-08   oc20110416_20260122.public.qc.nc       -98.29   -98.01    35.80    36.68   418.5  423.0  land  poster oc  yes   # n=343 dmin=66km tccon_obs=45
run_case  2025-12-08   oc20110416_20260122.public.qc.nc       -98.14   -97.43    35.70    37.35   425.5  428.0  land  poster oc  yes   # n=741 dmin=26km tccon_obs=56
# -- orleans01 (or; lat 47.96, lon 2.11) --
run_case  2025-03-06   or20090906_20250411.public.qc.nc         1.86     2.78    47.17    48.85   425.5  428.5  land  poster or  yes   # n=759 dmin=15km tccon_obs=48
run_case  2025-04-07   or20090906_20250411.public.qc.nc         2.31     3.09    47.33    48.85   426.0  429.5  land  poster or  yes   # n=666 dmin=45km tccon_obs=48
# -- paris01 (pr; lat 48.85, lon 2.36) --
run_case  2024-11-28   pr20140923_20251120.public.qc.nc         1.96     2.70    47.97    49.72   420.0  428.0  land  poster pr  yes   # n=577 dmin=1km tccon_obs=32
run_case  2025-03-06   pr20140923_20251120.public.qc.nc         1.48     2.40    47.95    49.57   425.5  428.5  land  poster pr  yes   # n=736 dmin=30km tccon_obs=41
run_case  2025-04-07   pr20140923_20251120.public.qc.nc         1.87     2.78    47.99    49.70   426.0  429.5  land  poster pr  yes   # n=754 dmin=1km tccon_obs=34
run_case  2025-08-11   pr20140923_20251120.public.qc.nc         2.13     2.88    48.01    49.74   421.0  424.0  land  poster pr  yes   # n=675 dmin=10km tccon_obs=32
# -- sodankyla01 (so; lat 67.37, lon 26.63) --
run_case  2024-08-23   so20090516_20251025.public.qc.nc        25.21    26.95    66.57    68.11   414.0  420.0  land  poster so  yes   # n=522 dmin=19km tccon_obs=29
run_case  2025-07-15   so20090516_20251025.public.qc.nc        26.23    27.01    67.22    67.50   420.5  423.5  land  poster so  yes   # n=4971 dmin=0km tccon_obs=43
# -- wollongong01 (wg; lat -34.41, lon 150.88) --
run_case  2025-04-07   wg20130104_20260224.public.qc.nc       150.74   150.93   -34.61   -34.31   415.5  424.0  land  poster wg  yes   # n=4176 dmin=0km tccon_obs=40
run_case  2025-09-14   wg20130104_20260224.public.qc.nc       150.74   150.93   -34.61   -34.31   416.5  424.0  land  poster wg  yes   # n=4174 dmin=0km tccon_obs=48

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

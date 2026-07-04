#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_plot_deepens
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable

# Deep-ensemble version of curc_shell_blanca_plot_corr_xco2_general.sh.
# For each case it (4) applies the deep ensemble on ocean AND land footprints
# (build_deepens_plot_data.py) then (5) plots vs TCCON + MODIS
# (plot_corrected_xco2.py, --poster-model deep_ensemble_corrected_xco2).
#
# Add a case by copying one run_case line.  Args (positional):
#   run_case DATE TCCON_FILE LON_MIN LON_MAX LAT_MIN LAT_MAX VMIN VMAX [SURF] [POSTER] [SITE]
#     SURF   : both | ocean | land     (default both)
#     POSTER : poster | ""             (default "" — no poster figure)
#     SITE   : short station label     (optional; when set, output dir becomes
#              combined_DATE_SITE so same-date/different-station runs don't clash)
# DATE drives the input parquet, results-h5, and output dir automatically.

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
SCRIPT_NAME=curc_shell_blanca_plot_corr_xco2_deepens.sh

# ─── data root ────────────────────────────────────────────────────────────────
# On CURC the results/ tree and data/ (TCCON) live under scratch, not the repo
# checkout.  Mirror the convention in curc_shell_blanca_general.sh: prefer
# CURC_DATA_ROOT, then OCO2_DATAROOT, else '.' (local repo, unchanged).
DATA_ROOT="${CURC_DATA_ROOT:-${OCO2_DATAROOT:-.}}"; DATA_ROOT="${DATA_ROOT%/}"
# Propagate to the aggregate report scripts so their hardcoded paths follow suit.
export OCO2_DATAROOT="$DATA_ROOT"

# ─── model (deep-ensemble fold dirs; per-surface) ─────────────────────────────
# Pool ALL folds (f0..f4) → cross-fold ensemble: mu = mean of 25 members
# (5 folds × M=5), sigma = total predictive std (each fold transformed by its
# own scaler).
# M=5 deep ensemble + the vertical-profile EOF block (--profile-pca: 12 EOF PCs of
# the sigma-grid T/q/CO2-prior profiles + 2 tropopause scalars, appended to the
# `full` feature set — embedded in each fold's pipeline pkl, so no extra build flag
# is needed).  This is the production M=5 DE from curc_shell_blanca_de_profile.sh.
# Profile is a large near-cloud-tail win (global ΔR² ocean +0.033 / land +0.148;
# land near-cloud bottom-5% R² 0.45→0.76) — see de_{ocean,land}_profile_ab.md.
# Snow: both surfaces KEEP snow footprints (CLI default; --exclude_snow would drop
# them), and snow_flag is NOT a feature (dropped in the 2026-07 simplification) —
# so high-latitude/snow land is corrected in-domain, not flagged OOD.
# PREREQUISITE: input parquets must carry the sigma-grid profile + tropopause
# columns (build via curc_shell_blanca_build_feature_dataset_perdate.sh) or the
# profile-PCA transform in build_deepens_plot_data.py raises KeyError.
# MODEL_TAG names the model version; it namespaces OUT_BASE so different DE
# versions never overwrite each other's plot outputs.
MODEL_TAG=de_beta_nll_prof_m5
OCEAN_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_ocean_beta_nll_prof_f*)
LAND_MODEL_DIRS=("$DATA_ROOT"/results/model_deep_ensemble/de_land_beta_nll_prof_f*)

# ─── cloud classifier (xgb_cloud fold dirs; per-surface) ──────────────────────
# When set, build_deepens_plot_data.py also emits P(near) and the two extra
# correction columns deepens_pnear_corrected_xco2 (P*mu) and deepens_gate_corrected_xco2
# (mu*1[P>0.5]).  FT1 verdict (held-out OCO label + TCCON): full mu wins, these two
# under-correct — kept for diagnostics, not the production correction.
OCEAN_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_ocean_f*)
LAND_CLOUD_DIRS=("$DATA_ROOT"/results/model_xgb_cloud/xgbcloud_final_land_f*)

CSV_DIR="$DATA_ROOT"/results/csv_collection
# OUT_BASE is namespaced by MODEL_TAG so each DE model version writes to its own
# tree (e.g. .../deep_ensemble/de_beta_nll_prof_m5/combined_<date>_<site>/).
OUT_BASE="$DATA_ROOT"/results/model_comparison/deep_ensemble/${MODEL_TAG}

# ─── TCCON collocation knobs (shared by per-case plot + aggregate reports) ─────
# Keep these in sync across plot_corrected_xco2.py, tccon_comparison_report.py,
# and tccon_correction_policy_stats.py so all three match the same footprints.
# RADIUS_KM is env-overridable for a radius sweep (e.g. `RADIUS_KM=50 bash <script>`);
# it is stamped into the FINAL aggregate-plot filenames (via RADIUS_TAG below), not
# the output dir, so a 50/100 km sweep's summary plots don't overwrite each other.
RADIUS_KM="${RADIUS_KM:-100}"
WINDOW_MIN="${WINDOW_MIN:-60}"
RADIUS_TAG="r${RADIUS_KM}km"

# REQUIRE_TCCON=1 → SKIP cases whose 12th run_case column (TCCON_AVAIL) is not "yes"
# (stations with no TCCON measurement within ±WINDOW_MIN of the OCO overpass).
# Set to 0 to run every case.  Hardcoded here (not env-overridable) by choice.
# Recompute flags: PYTHONPATH=src python workspace/check_tccon_availability.py
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
    echo "############ CASE $date  ($surf)  ############"
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

    # (5) plot
    local h5_arg=(); [[ -f "$h5" ]] && h5_arg=(--results-h5 "$h5")
    local poster_arg=(); [[ "$poster" == poster ]] && \
        poster_arg=(--poster-model deep_ensemble_corrected_xco2 --poster-dpi 300)
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

# ═══════════════════════ cases, organized by TCCON site then date ════════════════
# (ported from *_general.sh + added sites).  SITE (last arg) sets the output dir
# combined_DATE_SITE so same-date/different-station runs stay separate.
#         DATE         TCCON                              LON_MIN  LON_MAX  LAT_MIN  LAT_MAX   VMIN   VMAX  SURF  POSTER SITE

# ── Burgos, Philippines (bu; lat 18.53, lon 120.65) ──
# run_case  2018-09-01   bu20170303_20250221.public.qc.nc     120.14   121.27    17.41    19.77   404.5  407.5  both  poster  bu   # TRAIN date — excluded (not unseen)
run_case  2018-10-24   bu20170303_20250221.public.qc.nc     120.23   121.13    18.04    18.89   405.5  409.0  land  poster  bu  yes
run_case  2018-11-29   bu20170303_20250221.public.qc.nc     120.30   121.49    18.07    19.25   406.5  409.5  both  poster  bu  no
# run_case  2020-01-15   bu20170303_20250221.public.qc.nc     120.24   121.15    18.00    18.89   411.0  414.5  land  poster  bu  yes   # TRAIN date — excluded (not unseen)
run_case  2020-03-30   bu20170303_20250221.public.qc.nc     120.19   121.21    17.77    19.77   412.0  415.5  both  poster  bu  yes
# run_case  2020-05-01   bu20170303_20250221.public.qc.nc     120.27   121.35    17.76    19.78   414.0  417.0  both  poster  bu   # TRAIN date — excluded (not unseen)
run_case  2020-09-06   bu20170303_20250221.public.qc.nc     120.10   121.03    18.18    19.76   408.0  411.0  both  poster  bu  no

# ── Izaña, Tenerife (iz; lat 28.31, lon -16.50) ──
run_case  2018-09-02   iz20140102_20230830.public.qc.nc     -16.89   -15.80    27.33    29.38   403.0  406.5  both  poster  iz  no
run_case  2018-11-30   iz20140102_20230830.public.qc.nc     -16.88   -15.62    27.18    29.52   406.0  409.5  both  poster  iz  no
run_case  2019-03-13   iz20140102_20230830.public.qc.nc     -16.93   -15.87    27.09    28.95   409.5  412.5  both  poster  iz  yes
run_case  2019-07-10   iz20140102_20230830.public.qc.nc     -16.96   -16.07    27.82    28.79   403.5  412.5  land  poster  iz  yes
run_case  2020-03-31   iz20140102_20230830.public.qc.nc     -16.95   -15.92    27.33    29.40   408.0  416.5  both  poster  iz  no
run_case  2020-04-05   iz20140102_20230830.public.qc.nc     -16.95   -16.07    27.81    28.78   404.5  416.0  both  poster  iz  yes
run_case  2021-03-18   iz20140102_20230830.public.qc.nc     -17.07   -15.87    27.16    29.53   414.5  418.0  both  poster  iz  no
run_case  2021-08-25   iz20140102_20230830.public.qc.nc     -16.85   -15.53    27.24    29.55   411.0  414.0  both  poster  iz  no
run_case  2021-09-26   iz20140102_20230830.public.qc.nc     -17.32   -16.15    27.06    29.21   411.5  415.0  both  poster  iz  no

# ── Lamont, Oklahoma (oc; lat 36.60, lon -97.49) ──
run_case  2018-06-03   oc20110416_20251023.public.qc.nc     -97.94   -96.76    35.65    37.69   406.5  409.0  land  poster  oc  yes
run_case  2021-04-24   oc20110416_20251023.public.qc.nc     -98.21   -96.98    35.55    37.80   415.5  419.0  land  poster  oc  yes
run_case  2021-12-29   oc20110416_20251023.public.qc.nc     -98.24   -96.87    35.40    37.80   416.5  420.0  land  poster  oc  yes

# ── Manaus, Brazil (ma; lat -3.21, lon -60.60) ──
run_case  2014-12-03   ma20140930_20150727.public.qc.nc     -61.05   -60.06    -4.12    -2.36   392.0  399.5  land  poster  ma  yes
run_case  2015-03-23   ma20140930_20150727.public.qc.nc     -61.40   -60.15    -4.06    -2.76   385.0  402.5  land  poster  ma  yes
# 2014-12-17: nearest OCO sounding 263 km from station (overpass orbit not in parquet) — no ≤100km comparison
# run_case  2014-12-17   ma20140930_20150727.public.qc.nc     -61.60   -59.60    -4.21    -2.21   397.0  402.0  land  poster  ma
run_case  2015-06-29   ma20140930_20150727.public.qc.nc     -60.95   -59.06    -3.99    -2.34   397.5  402.0  land  poster  ma  no
run_case  2015-07-06   ma20140930_20150727.public.qc.nc     -60.95   -59.37    -3.77    -2.29   397.5  402.5  land  poster  ma  yes
run_case  2015-07-13   ma20140930_20150727.public.qc.nc     -61.77   -60.25    -4.20    -2.47   397.5  401.5  land  poster  ma  no
run_case  2015-07-15   ma20140930_20150727.public.qc.nc     -60.95   -59.01    -4.14    -1.88   397.5  403.0  land  poster  ma  no

# ── Ny-Ålesund, Svalbard (ny; lat 78.92, lon 11.92) ──
run_case  2015-05-18   ny20050316_20250524.public.qc.nc      11.09    13.09    78.47    79.65   368.0  388.5  land  poster  ny  no
run_case  2015-05-20   ny20050316_20250524.public.qc.nc      11.47    14.61    78.47    79.48   363.0  391.0  land  poster  ny  no
run_case  2015-06-05   ny20050316_20250524.public.qc.nc       9.98    13.62    77.88    79.37   384.0  404.5  both  poster  ny  yes
run_case  2015-07-15   ny20050316_20250524.public.qc.nc      10.50    15.30    78.20    79.50   394.0  402.5  land  poster  ny  yes
# 2016-05-29: nearest OCO sounding 94 km, only 14 footprints ≤100km — marginal clip
# run_case  2016-05-29   ny20050316_20250524.public.qc.nc       8.28    12.37    78.04    79.37   404.5  407.5  both  poster  ny
run_case  2016-07-09   ny20050316_20250524.public.qc.nc       9.17    14.91    78.47    79.93   397.5  405.5  both  poster  ny  no
run_case  2017-04-23   ny20050316_20250524.public.qc.nc      10.46    16.97    78.47    79.37   405.0  412.0  both  poster  ny  no
# 2017-05-16: nearest OCO sounding 99 km, only 2 footprints ≤100km — marginal clip
# run_case  2017-05-16   ny20050316_20250524.public.qc.nc      11.47    14.98    78.47    79.76   406.0  410.0  both  poster  ny
run_case  2017-05-25   ny20050316_20250524.public.qc.nc      11.47    15.34    78.47    79.62   399.5  416.0  both  poster  ny  yes
run_case  2017-06-17   ny20050316_20250524.public.qc.nc      11.12    16.35    77.86    79.78   377.0  409.0  both  poster  ny  yes
# 2020-03-31: nearest OCO sounding 58 km, only 1 footprint ≤100km — marginal clip
# run_case  2020-03-31   ny20050316_20250524.public.qc.nc      11.47    14.62    78.47    79.37   406.0  410.0  both  poster  ny
run_case  2016-09-10   ny20050316_20250524.public.qc.nc      10.31    14.44    77.93    79.27   383.5  401.5  both  poster  ny  yes
run_case  2020-07-11   ny20050316_20250524.public.qc.nc       8.23    13.24    77.69    79.27   408.0  411.5  both  poster  ny  yes
run_case  2020-07-26   ny20050316_20250524.public.qc.nc      10.55    16.59    78.00    79.30   404.0  409.0  both  poster  ny  yes
# 2021-06-21: nearest OCO sounding 165 km from station (overpass orbit not in parquet) — no ≤100km comparison
# run_case  2021-06-21   ny20050316_20250524.public.qc.nc      10.92    12.92    77.92    79.92   413.0  419.0  both  poster  ny
run_case  2021-07-03   ny20050316_20250524.public.qc.nc       9.77    16.72    78.57    80.17   412.0  418.5  both  poster  ny  yes
run_case  2021-09-08   ny20050316_20250524.public.qc.nc      10.21    13.80    78.03    79.27   405.5  412.0  both  poster  ny  yes

# ── Wollongong, Australia (wg; lat -34.41, lon 150.88) ──
run_case  2018-08-17   wg20130104_20260224.public.qc.nc     150.43   151.33   -34.86   -33.96   394.0  408.0  both  poster  wg  yes
run_case  2019-07-30   wg20130104_20260224.public.qc.nc     150.48   151.46   -35.32   -33.73   406.0  409.0  both  poster  wg  yes
run_case  2019-09-14   wg20130104_20260224.public.qc.nc     150.39   151.27   -34.95   -33.98   398.0  410.0  both  poster  wg  yes
run_case  2020-09-16   wg20130104_20260224.public.qc.nc     150.41   151.27   -34.96   -33.97   406.0  412.5  both  poster  wg  yes
run_case  2020-12-23   wg20130104_20260224.public.qc.nc     150.41   151.44   -35.65   -33.73   410.0  413.5  both  poster  wg  yes
run_case  2021-03-29   wg20130104_20260224.public.qc.nc     150.26   151.47   -35.64   -33.17   410.0  413.0  both  poster  wg  yes
run_case  2021-07-03   wg20130104_20260224.public.qc.nc     150.45   151.25   -34.89   -34.03   407.0  415.0  both  poster  wg  yes

# ── East Trout Lake, Canada (et; lat 54.35, lon -104.99) ──
run_case  2018-06-03   et20161003_20260326.public.qc.nc    -105.44  -104.22    53.38    54.80   394.5  412.0  land  poster  et  yes
run_case  2021-05-26   et20161003_20260326.public.qc.nc    -105.84  -104.07    53.16    55.55   416.0  419.5  land  poster  et  yes
# 2021-06-09: nearest OCO sounding 920 km from station (overpass orbit not in parquet) — no nearby pass
# run_case  2021-06-09   et20161003_20260326.public.qc.nc    -105.99  -103.99    53.35    55.35   413.0  418.0  land  poster  et
run_case  2021-07-27   et20161003_20260326.public.qc.nc    -105.81  -104.64    54.00    55.56   406.0  410.0  land  poster  et  no
run_case  2021-09-06   et20161003_20260326.public.qc.nc    -105.50  -104.47    53.87    54.82   406.0  413.0  land  poster  et  yes

# ── Park Falls, USA (pa; lat 45.95, lon -90.27) ──
run_case  2016-11-05   pa20040602_20260123.public.qc.nc     -90.72   -89.17    45.14    46.77   401.5  405.5  land  poster  pa  yes
run_case  2021-03-29   pa20040602_20260123.public.qc.nc     -90.75   -89.77    45.46    46.45   416.5  419.5  land  poster  pa  no
run_case  2021-10-16   pa20040602_20260123.public.qc.nc     -90.74   -89.78    45.46    46.39   408.0  416.0  land  poster  pa  yes

# ── Réunion Island (ra; lat -20.90, lon 55.49) ──
run_case  2015-03-17   ra20150301_20200718.public.qc.nc      54.90    55.94   -21.99   -19.89   395.0  397.5  both  poster  ra  yes
run_case  2015-05-18   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.35   -20.45   376.5  400.5  land  poster  ra  yes
run_case  2015-05-20   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.35   -20.45   375.0  403.5  land  poster  ra  yes
run_case  2015-06-05   ra20150301_20200718.public.qc.nc      55.00    55.94   -21.99   -20.03   390.5  399.5  both  poster  ra  no
run_case  2015-07-14   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.44   -20.45   391.5  402.0  both  poster  ra  yes
run_case  2015-07-23   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.98   -20.03   396.0  400.5  both  poster  ra  yes
run_case  2015-08-01   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.35   -20.45   391.0  404.0  both  poster  ra  yes
run_case  2015-09-25   ra20150301_20200718.public.qc.nc      54.56    55.94   -21.91   -20.11   396.0  400.0  both  poster  ra  no
run_case  2016-03-03   ra20150301_20200718.public.qc.nc      54.95    55.94   -21.99   -19.87   396.5  401.0  both  poster  ra  no
run_case  2016-05-06   ra20150301_20200718.public.qc.nc      54.77    55.94   -21.95   -19.96   398.5  402.0  both  poster  ra  no
run_case  2016-05-29   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.44   -20.45   398.0  404.5  both  poster  ra  no
run_case  2016-06-07   ra20150301_20200718.public.qc.nc      54.95    55.94   -21.53   -19.87   393.0  402.5  both  poster  ra  no
run_case  2016-07-09   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.97   -19.83   399.0  403.0  both  poster  ra  no
run_case  2016-09-11   ra20150301_20200718.public.qc.nc      54.75    55.94   -21.97   -19.97   401.5  404.5  both  poster  ra  yes
run_case  2016-10-13   ra20150301_20200718.public.qc.nc      54.52    55.94   -21.88   -20.18   400.5  404.0  both  poster  ra  no
run_case  2016-11-05   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.46   -20.45   398.0  405.0  both  poster  ra  yes
run_case  2016-11-07   ra20150301_20200718.public.qc.nc      55.04    56.42   -21.67   -19.90   401.0  404.0  both  poster  ra  no
run_case  2017-01-17   ra20150301_20200718.public.qc.nc      54.86    55.94   -21.98   -19.94   399.5  404.0  both  poster  ra  no
run_case  2017-03-22   ra20150301_20200718.public.qc.nc      54.87    55.94   -21.98   -19.92   400.0  404.0  both  poster  ra  yes
run_case  2017-04-23   ra20150301_20200718.public.qc.nc      54.83    55.94   -21.99   -19.93   400.5  403.5  both  poster  ra  yes
run_case  2017-05-16   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.44   -20.45   396.0  406.0  both  poster  ra  yes
run_case  2017-05-25   ra20150301_20200718.public.qc.nc      54.89    55.94   -21.98   -19.89   402.0  404.5  both  poster  ra  yes
run_case  2017-06-17   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.45   -20.45   392.0  407.5  both  poster  ra  no
run_case  2018-04-10   ra20150301_20200718.public.qc.nc      54.77    55.94   -21.97   -19.95   403.0  405.5  both  poster  ra  no
run_case  2018-05-12   ra20150301_20200718.public.qc.nc      54.81    55.94   -21.97   -19.93   402.0  406.5  both  poster  ra  no
run_case  2018-08-07   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.44   -20.45   398.0  411.5  both  poster  ra  no
run_case  2018-09-08   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.47   -20.45   402.5  410.0  both  poster  ra  no
run_case  2018-09-17   ra20150301_20200718.public.qc.nc      54.67    55.94   -21.94   -20.03   403.0  407.0  both  poster  ra  no
run_case  2018-10-10   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.47   -20.45   399.0  410.5  both  poster  ra  no
run_case  2019-04-29   ra20150301_20200718.public.qc.nc      55.01    55.94   -21.97   -20.45   404.5  408.5  both  poster  ra  no
run_case  2020-03-14   ra20150301_20200718.public.qc.nc      55.04    55.94   -21.98   -20.45   406.5  410.5  both  poster  ra  no
# run_case  2020-01-15   ra20150301_20200718.public.qc.nc      55.08    55.95   -21.39   -20.53   399.5  412.0  both  poster  ra  no   # TRAIN date — excluded (not unseen)
run_case  2020-02-11   ra20150301_20200718.public.qc.nc      54.80    55.94   -22.10   -19.70   406.0  410.5  both  poster  ra  no
run_case  2020-03-30   ra20150301_20200718.public.qc.nc      55.10    55.94   -21.31   -20.53   401.5  414.0  both  poster  ra  yes
# run_case  2020-04-15   ra20150301_20200718.public.qc.nc      54.63    55.84   -22.15   -19.79   408.0  411.0  both  poster  ra  yes   # TRAIN date — excluded (not unseen)
run_case  2020-05-17   ra20150301_20200718.public.qc.nc      54.64    55.84   -22.13   -19.75   409.5  413.0  both  poster  ra  no
run_case  2020-07-11   ra20150301_20200718.public.qc.nc      55.10    55.99   -21.72   -20.55   407.5  413.5  both  poster  ra  yes

# ══════════════════ auto-discovered stations (overpasses in existing parquets) ══════════════════

# ── Edwards/AFRC, USA (df; lat 34.96, lon -117.881) ──
run_case  2015-07-06   df20130720_20260121.public.qc.nc    -118.66  -117.43    33.71    36.14   400.0  404.0  land  poster  df  yes
run_case  2015-07-15   df20130720_20260121.public.qc.nc    -118.79  -117.53    33.71    36.08   396.5  401.0  land  poster  df  yes
run_case  2019-03-13   df20130720_20260121.public.qc.nc    -118.23  -116.85    33.90    36.20   410.5  413.5  land  poster  df  no
run_case  2019-07-10   df20130720_20260121.public.qc.nc    -118.34  -117.39    34.46    35.46   408.5  412.0  land  poster  df  yes
run_case  2021-02-10   df20130720_20260121.public.qc.nc    -118.35  -117.40    34.48    35.43   415.0  418.0  land  poster  df  yes
run_case  2021-09-06   df20130720_20260121.public.qc.nc    -118.36  -117.38    34.46    35.42   411.5  414.5  land  poster  df  yes
run_case  2021-09-26   df20130720_20260121.public.qc.nc    -118.49  -117.17    33.76    36.19   411.0  417.5  land  poster  df  yes

# ── Caltech/Pasadena, USA (ci; lat 34.136, lon -118.127) ──
run_case  2015-07-06   ci20120920_20251222.public.qc.nc    -118.48  -117.25    33.06    35.38   399.5  403.5  land  poster  ci  no
run_case  2015-07-15   ci20120920_20251222.public.qc.nc    -118.57  -117.38    33.14    35.38   397.0  401.0  land  poster  ci  no
run_case  2018-06-03   ci20120920_20251222.public.qc.nc    -118.58  -117.68    33.61    34.59   409.5  413.0  land  poster  ci  yes
run_case  2021-04-24   ci20120920_20251222.public.qc.nc    -118.49  -117.64    33.68    34.60   404.5  420.5  land  poster  ci  no

# ── Saga, Japan (js; lat 33.241, lon 130.288) ──
run_case  2018-09-02   js20110728_20231213.public.qc.nc     129.91   131.08    32.20    34.36   403.0  408.0  both  poster  js  yes
run_case  2019-03-13   js20110728_20231213.public.qc.nc     129.68   130.94    32.10    34.41   410.5  414.0  both  poster  js  yes
run_case  2020-10-05   js20110728_20231213.public.qc.nc     129.82   130.79    32.77    33.72   408.5  412.0  both  poster  js  yes
run_case  2021-03-18   js20110728_20231213.public.qc.nc     129.86   130.98    32.06    34.01   416.0  419.0  both  poster  js  yes
run_case  2021-09-26   js20110728_20231213.public.qc.nc     129.41   130.64    31.99    34.27   411.5  416.5  both  poster  js  yes

# ── Karlsruhe, Germany (ka; lat 49.1, lon 8.439) ──
run_case  2018-09-08   ka20140115_20230626.public.qc.nc       7.99     9.96    48.65    50.05   399.0  405.5  land  poster  ka  no
run_case  2018-10-10   ka20140115_20230626.public.qc.nc       7.73     9.02    48.06    50.11   403.0  408.5  land  poster  ka  yes
run_case  2019-07-30   ka20140115_20230626.public.qc.nc       7.98     8.93    48.61    49.60   403.5  411.0  land  poster  ka  yes
# run_case  2020-01-15   ka20140115_20230626.public.qc.nc       7.95     8.94    48.64    49.55   407.0  419.5  land  poster  ka  yes   # TRAIN date — excluded (not unseen)
run_case  2021-03-29   ka20140115_20230626.public.qc.nc       7.97     8.94    48.63    49.59   416.5  419.5  land  poster  ka  no
run_case  2021-07-03   ka20140115_20230626.public.qc.nc       7.99     9.23    48.20    50.16   411.0  416.0  land  poster  ka  no

# ── Orléans, France (or; lat 47.965, lon 2.113) ──
run_case  2015-07-15   or20090906_20250411.public.qc.nc       1.76     3.43    46.95    49.14   396.0  399.5  land  poster  or  yes
run_case  2018-08-17   or20090906_20250411.public.qc.nc       1.11     2.56    47.52    48.86   396.5  404.5  land  poster  or  yes
run_case  2020-04-05   or20090906_20250411.public.qc.nc       1.60     2.62    47.46    48.45   412.5  416.0  land  poster  or  no
run_case  2020-09-16   or20090906_20250411.public.qc.nc       1.43     3.02    46.79    49.19   409.0  416.0  land  poster  or  no

# ── Darwin, Australia (db; lat -12.456, lon 130.926) ──
run_case  2015-05-20   db20130101_20250731.public.qc.nc     130.48   131.38   -12.91   -12.01   388.5  401.5  land  poster  db  no
run_case  2015-06-05   db20130101_20250731.public.qc.nc     130.48   131.38   -12.91   -12.01   396.5  400.5  both  poster  db  yes
run_case  2015-08-01   db20130101_20250731.public.qc.nc     130.48   131.38   -12.91   -12.01   391.0  401.5  both  poster  db  yes
run_case  2016-11-07   db20130101_20250731.public.qc.nc     130.48   131.38   -13.47   -12.01   395.0  404.5  both  poster  db  no
run_case  2018-09-17   db20130101_20250731.public.qc.nc     130.48   131.38   -12.91   -12.01   402.5  409.0  both  poster  db  no
run_case  2021-06-21   db20130101_20250731.public.qc.nc     130.50   131.38   -12.92   -11.99   411.5  414.5  both  poster  db  yes

# ── Rikubetsu, Japan (rj; lat 43.459, lon 143.766) ──
run_case  2020-03-30   rj20140624_20250501.public.qc.nc     143.21   144.21    42.95    43.91   412.5  416.0  land  poster  rj  yes

# ── Xianghe, China (xh; lat 39.798, lon 116.958) ──
run_case  2020-02-11   xh20180614_20241231.public.qc.nc     116.43   117.71    38.60    41.04   414.5  420.5  land  poster  xh  yes
run_case  2020-03-14   xh20180614_20241231.public.qc.nc     116.51   117.46    38.75    40.87   413.0  418.5  land  poster  xh  no

# ── Hefei, China (hf; lat 31.905, lon 117.167) ──
run_case  2020-09-06   hf20151102_20251230.public.qc.nc     116.59   117.61    30.66    32.38   408.0  411.5  land  poster  hf  no
run_case  2021-06-21   hf20151102_20251230.public.qc.nc     116.82   118.27    30.91    33.12   415.5  419.0  land  poster  hf  no
# 2017-06-17: nearest OCO sounding 100 km, only 4 footprints ≤100km — marginal clip
# run_case  2017-06-17   hf20151102_20251230.public.qc.nc     115.92   117.62    31.20    32.36   400.0  406.5  land  poster  hf

# ── Paris, France (pr; lat 48.846, lon 2.356) ──
run_case  2015-07-15   pr20140923_20251024.public.qc.nc       1.98     3.17    47.64    49.41   395.0  400.0  land  poster  pr  yes
run_case  2018-08-17   pr20140923_20251024.public.qc.nc       0.89     2.81    47.90    49.30   395.5  404.5  land  poster  pr  yes
run_case  2020-09-16   pr20140923_20251024.public.qc.nc       1.43     2.71    47.72    49.20   411.5  416.5  land  poster  pr  yes

# ─────────────────────── add your other cases below ──────────────────────────
# run_case  2020-04-15   ra20150301_20200718.public.qc.nc   54.98    55.72    -22.71   -20.32   406     412    both

# ═══════════════════════ aggregate reports (run once, after all cases) ════════════
# Both parse the active run_case lines above and summarize across cases.

# (7) before/after-vs-TCCON comparison (reads each case's plot_data.parquet via the
#     shared collocator).  This ONE script now emits every comparison figure in a
#     single style: all-cases + excl-ny scatter/dumbbell, the ocean/land by-surface
#     version (--exclude-sites), and the per-site bar chart.  --fname-suffix stamps
#     the radius so a sweep's summaries coexist (…_r100km.png vs …_r50km.png).
echo ""
echo "############ AGGREGATE: tccon_comparison_report ############"
python workspace/tccon_comparison_report.py \
    --script   "$SCRIPT_NAME" \
    --out-base "$OUT_BASE" \
    --output-dir "$OUT_BASE" \
    --fname-suffix "_${RADIUS_TAG}" --exclude-sites ny \
    --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN"

# (8) correction-policy stats — READS the step-4 plot_data (no second model run) via
#     --plotdata-base, and writes CSVs to --output-dir (= OUT_BASE), so the policy
#     stats use the SAME processed XCO2 as the plots and everything lands under
#     deep_ensemble/<MODEL_TAG>/.  Footprint-level pooled / near-far-cloud /
#     drop-guards stats (complementary to the per-case figures from step 7).
echo ""
echo "############ AGGREGATE: tccon_correction_policy_stats ############"
python workspace/tccon_correction_policy_stats.py \
    --radius-km "$RADIUS_KM" --window-min "$WINDOW_MIN" \
    --plotdata-base "$OUT_BASE" --output-dir "$OUT_BASE" --fname-suffix "_${RADIUS_TAG}"

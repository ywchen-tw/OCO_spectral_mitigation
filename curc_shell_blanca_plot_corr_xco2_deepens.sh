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

cd "$(dirname "$0")"
export PYTHONPATH=src:${PYTHONPATH:-}

# ─── model (deep-ensemble fold dirs; per-surface) ─────────────────────────────
OCEAN_MODEL_DIR=results/model_deep_ensemble/de_ocean_beta_nll_f0
LAND_MODEL_DIR=results/model_deep_ensemble/de_land_beta_nll_f0

CSV_DIR=results/csv_collection
OUT_BASE=results/model_comparison/deep_ensemble

run_case() {
    local date="$1" tccon="$2" lonmin="$3" lonmax="$4" latmin="$5" latmax="$6"
    local vmin="$7" vmax="$8" surf="${9:-both}" poster="${10:-}" site="${11:-}"

    local input="$CSV_DIR/combined_${date}_all_orbits.parquet"
    local h5="results/results_${date}.h5"
    # site label (11th arg) keeps same-date/different-station runs in separate dirs
    local outdir="$OUT_BASE/combined_${date}_all_orbits"
    [[ -n "$site" ]] && outdir="$OUT_BASE/combined_${date}_${site}"
    local plotdata="$outdir/plot_data.parquet"

    echo ""
    echo "############ CASE $date  ($surf)  ############"
    if [[ ! -f "$input" ]]; then
        echo "  SKIP: input parquet not found: $input"; return
    fi
    mkdir -p "$outdir"

    # (4) apply deep ensemble → plot_data.parquet
    local model_args=()
    [[ "$surf" == both || "$surf" == ocean ]] && model_args+=(--ocean-model-dir "$OCEAN_MODEL_DIR")
    [[ "$surf" == both || "$surf" == land  ]] && model_args+=(--land-model-dir  "$LAND_MODEL_DIR")
    python workspace/build_deepens_plot_data.py \
        "${model_args[@]}" --input "$input" --output "$plotdata" || { echo "  build failed"; return; }

    # (5) plot
    local h5_arg=(); [[ -f "$h5" ]] && h5_arg=(--results-h5 "$h5")
    local poster_arg=(); [[ "$poster" == poster ]] && \
        poster_arg=(--poster-model deep_ensemble_corrected_xco2 --poster-dpi 300)
    python workspace/plot_corrected_xco2.py \
        --plot-data  "$plotdata" \
        --tccon      "data/TCCON/$tccon" \
        "${h5_arg[@]}" \
        --output-dir "$outdir" \
        --modis-auto \
        --lon-range  "$lonmin" "$lonmax" \
        --lat-range  "$latmin" "$latmax" \
        --date-plot  "$date" \
        --vmin "$vmin" --vmax "$vmax" \
        --hist-radius-km 100 \
        "${poster_arg[@]}"
}

# ─────────────────────── cases (ported from *_general.sh) ─────────────────────
#         DATE         TCCON                              LON_MIN  LON_MAX  LAT_MIN  LAT_MAX  VMIN    VMAX   SURF   POSTER
run_case  2020-03-30   bu20170303_20250221.public.qc.nc     120.19   121.21    17.77    19.77   412.0  415.5  both  poster
run_case  2020-09-06   bu20170303_20250221.public.qc.nc     120.10   121.03    18.18    19.76   408.0  411.0  both  poster
run_case  2018-09-01   bu20170303_20250221.public.qc.nc     120.14   121.27    17.41    19.77   404.5  407.5  both  poster
run_case  2018-11-29   bu20170303_20250221.public.qc.nc     120.30   121.49    18.07    19.25   406.5  409.5  both  poster
run_case  2020-05-01   bu20170303_20250221.public.qc.nc     120.27   121.35    17.76    19.78   414.0  417.0  both  poster
run_case  2018-09-02   iz20140102_20230830.public.qc.nc     -16.89   -15.80    27.33    29.38   403.0  406.5  both  poster
run_case  2018-11-30   iz20140102_20230830.public.qc.nc     -16.88   -15.62    27.18    29.52   406.0  409.5  both  poster
run_case  2019-03-13   iz20140102_20230830.public.qc.nc     -16.93   -15.87    27.09    28.95   409.5  412.5  both  poster

# land-region cases (Burgos / Lamont-Oklahoma / Izaña)
run_case  2018-10-24   bu20170303_20250221.public.qc.nc     120.23   121.13    18.04    18.89   405.5  409.0  land  poster
run_case  2020-01-15   bu20170303_20250221.public.qc.nc     120.24   121.15    18.00    18.89   411.0  414.5  land  poster
run_case  2021-04-24   oc20110416_20251023.public.qc.nc     -98.21   -96.98    35.55    37.80   415.5  419.0  land  poster
run_case  2021-12-29   oc20110416_20251023.public.qc.nc     -98.24   -96.87    35.40    37.80   416.5  420.0  land  poster
run_case  2019-07-10   iz20140102_20230830.public.qc.nc     -16.96   -16.07    27.82    28.79   403.5  412.5  land  poster

# ══════════════════════ additional TCCON-site cases ══════════════════════════
# Dates below are confirmed present in fitting_correction.py date_list (parquets
# exist).  LON/LAT box = station coord ±1.0°, VMIN/VMAX are per-year placeholders
# — ADJUST LATER.  Station coords are the standard TCCON site locations.

# ── Manaus, Brazil (ma; lat -3.213, lon -60.599) — READY (TCCON file present) ──
#         DATE         TCCON                              LON_MIN  LON_MAX  LAT_MIN  LAT_MAX  VMIN  VMAX  SURF  POSTER  SITE
# 2014-12-17: nearest OCO sounding is 263 km from station (overpass orbit not in parquet) — no ≤100km comparison
# run_case  2014-12-17   ma20140930_20150727.public.qc.nc   -61.60   -59.60   -4.21    -2.21    397   402   land  poster  ma
run_case  2015-06-29   ma20140930_20150727.public.qc.nc     -60.95   -59.06    -3.99    -2.34   397.5  402.0  land  poster  ma
run_case  2015-07-06   ma20140930_20150727.public.qc.nc     -60.95   -59.37    -3.77    -2.29   397.5  402.5  land  poster  ma
run_case  2015-07-13   ma20140930_20150727.public.qc.nc     -61.77   -60.25    -4.20    -2.47   397.5  401.5  land  poster  ma
run_case  2015-07-15   ma20140930_20150727.public.qc.nc     -60.95   -59.01    -4.14    -1.88   397.5  403.0  land  poster  ma

# ── Ny-Ålesund, Svalbard (ny; lat 78.923, lon 11.923) — READY (TCCON file present) ──
run_case  2016-09-10   ny20050316_20250524.public.qc.nc      10.31    14.44    77.93    79.27   383.5  401.5  both  poster  ny
run_case  2020-07-11   ny20050316_20250524.public.qc.nc       8.23    13.24    77.69    79.27   408.0  411.5  both  poster  ny
run_case  2020-07-26   ny20050316_20250524.public.qc.nc      10.55    16.59    78.00    79.30   404.0  409.0  both  poster  ny
# 2021-06-21: nearest OCO sounding is 165 km from station (overpass orbit not in parquet) — no ≤100km comparison
# run_case  2021-06-21   ny20050316_20250524.public.qc.nc    10.92    12.92    77.92    79.92    413   419   both  poster  ny
run_case  2021-07-03   ny20050316_20250524.public.qc.nc       9.77    16.72    78.57    80.17   412.0  418.5  both  poster  ny
run_case  2021-09-08   ny20050316_20250524.public.qc.nc      10.21    13.80    78.03    79.27   405.5  412.0  both  poster  ny

# ── Wollongong, Australia (wg; lat -34.406, lon 150.879) — READY (TCCON file present) ──
run_case  2019-07-30   wg20130104_20260224.public.qc.nc     150.48   151.46   -35.32   -33.73   406.0  409.0  both  poster  wg
run_case  2019-09-14   wg20130104_20260224.public.qc.nc     150.39   151.27   -34.95   -33.98   398.0  410.0  both  poster  wg
run_case  2020-09-16   wg20130104_20260224.public.qc.nc     150.41   151.27   -34.96   -33.97   406.0  412.5  both  poster  wg
run_case  2020-12-23   wg20130104_20260224.public.qc.nc     150.41   151.44   -35.65   -33.73   410.0  413.5  both  poster  wg
run_case  2021-03-29   wg20130104_20260224.public.qc.nc     150.26   151.47   -35.64   -33.17   410.0  413.0  both  poster  wg
run_case  2021-07-03   wg20130104_20260224.public.qc.nc     150.45   151.25   -34.89   -34.03   407.0  415.0  both  poster  wg

# ── East Trout Lake, Canada (et; lat 54.354, lon -104.987) — READY (TCCON file present) ──
run_case  2021-05-26   et20161003_20260326.public.qc.nc    -105.84  -104.07    53.16    55.55   416.0  419.5  land  poster  et
# 2021-06-09: nearest OCO sounding is 920 km from station (overpass orbit not in parquet) — no nearby pass
# run_case  2021-06-09   et20161003_20260326.public.qc.nc    -105.99  -103.99  53.35    55.35    413   418   land  poster  et
run_case  2021-07-27   et20161003_20260326.public.qc.nc    -105.81  -104.64    54.00    55.56   406.0  410.0  land  poster  et
run_case  2021-09-06   et20161003_20260326.public.qc.nc    -105.50  -104.47    53.87    54.82   406.0  413.0  land  poster  et

# ── Park Falls, USA (pa; lat 45.945, lon -90.273) — READY (TCCON file present) ──
run_case  2021-03-29   pa20040602_20260123.public.qc.nc     -90.75   -89.77    45.46    46.45   416.5  419.5  land  poster  pa
run_case  2021-10-16   pa20040602_20260123.public.qc.nc     -90.74   -89.78    45.46    46.39   408.0  416.0  land  poster  pa

# ─────────────────────── add your other cases below ──────────────────────────
# run_case  2020-04-15   ra20150301_20200718.public.qc.nc   54.98    55.72    -22.71   -20.32   406     412    both

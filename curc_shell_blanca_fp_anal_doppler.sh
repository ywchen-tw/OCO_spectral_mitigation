#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_fp_doppler
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue
# NOTE: no static --array here.  The range is derived from whichever date set is
# active (see ALL= below) and applied via self-resubmission (bootstrap block).

# PER-DATE ARRAY JOB for the TELLURIC-DOPPLER TEST VARIANT of the spectral
# fitting pipeline (src/spectral/fitting.py --telluric-doppler).
#
# It re-computes the slant optical depths with the telluric Doppler shift of
# the ABSCO grid applied (lambda_rest * (1 - v_inst/c)), stores the per-channel
# tau attribution (H2O / Rayleigh fractions, absorption pressure centroid) and
# the extra order-2/3 fit statistics, and writes everything to a SEPARATE tree,
# <storage>/results_doppler, while reading the production cloud distances from
# <storage>/results.  Production files are never touched, so the two versions
# can be compared date by date (workspace/compare_doppler_fits.py) before any
# decision to replace the production tau.
#
# Submit directly (same pattern as curc_shell_blanca_fp_anal_perdate.sh):
#     sbatch curc_shell_blanca_fp_anal_doppler.sh
# Each array task processes ONE date (DATES[$SLURM_ARRAY_TASK_ID]) in parallel.
# Requirements per date on CURC: the OCO-2 L1B/Met/CO2Prior/Lite files under
# data/ and the production cloud-distance file results/results_<date>.h5.
#
# The array RANGE is computed automatically from the active date set: the script
# first runs as a plain (non-array) job, counts the unique dates, then resubmits
# itself with --array=0-(N-1)%MAX_CONCURRENT.  Change the active set on the ALL=
# line and the range follows — no #SBATCH edit needed.  MAX_CONCURRENT (below) is
# the "max jobs at once" throttle.  You can still override at submit time:
#     sbatch --array=0-<N-1>%<MAX> curc_shell_blanca_fp_anal_perdate.sh
# (an explicit --array is honored and skips the auto-resubmit).

# Max concurrent array tasks (the `%N` throttle).
MAX_CONCURRENT=52
# Array tasks per date.  2026-10-30: on the preemptable QOS an 8- or 16-task job with an
# 8 h limit starts in seconds while a 24 h one waits ~1.5 days, and the variant
# tau takes ~1.7 h per orbit with 4 workers (SLURM_NTASKS//2; ~0.9 h with the 8 workers
# of a 16-task allocation, 2026-10-30 evening), i.e. ~15-30 h for a
# 15-20 orbit date.  So each date gets REPLICAS tasks (index -> date = idx % N):
# preprocess() shuffles the orbit list and orbit_lock lets concurrent jobs on the
# same date split the orbits; a task that times out just leaves fewer orbits for
# the others, and resubmitting the date resumes (tau files carry a version stamp,
# finished fitting_details are skipped).
REPLICAS=4

# ── Date sets (YYYYMMDD), same three sets as build_feature_dataset.py main() ────
# Set 1 — 2020-only
DATES_2020=(
    20200101 20200201 20200301 20200401 20200501 20200601
    20200701 20200801 20200903 20201001 20201101 20201201
)
# Set 2 — 2016–2020, 1st + 15th of month
DATES_2016_2020=(
    20160101 20160201 20160301 20160405 20160501 20160601 20160701 20160801
    20160901 20161001 20161101 20161201
    20170101 20170201 20170301 20170401 20170501 20170601 20170701 20171001
    20171105 20171201
    20180101 20180201 20180301 20180401 20180501 20180601 20180701 20180801
    20180901 20181001 20181101 20181201
    20190101 20190201 20190301 20190401 20190501 20190601 20190701 20190801
    20190901 20191001 20191101 20191201
    20200101 20200201 20200301 20200401 20200501 20200601 20200701 20200801
    20200903 20201001 20201101 20201201
    20160115 20160215 20160315 20160415 20160515 20160615 20160715 20160821
    20160915 20161015 20161115 20161215
    20170115 20170215 20170315 20170415 20170515 20170615 20170715 20171015
    20171115 20171215
    20180115 20180212 20180315 20180415 20180515 20180615 20180715 20180815
    20180915 20181015 20181117 20181215
    20190115 20190215 20190315 20190415 20190515 20190615 20190715 20190815
    20190915 20191015 20191115 20191215
    20200115 20200215 20200315 20200415 20200515 20200615 20200715 20200815
    20200915 20201015 20201115 20201215
)
# Set 3 — event-specific dates 2014–2021
DATES_EVENTS=(
    20141203 20141217
    20150213 20150218 20150317 20150323 20150518 20150520 20150605 20150629
    20150706 20150713 20150714 20150715 20150723 20150801 20150925
    20160303 20160506 20160529 20160607 20160709 20160821 20160910 20160911 
    20161013 20161105 20161107
    20170117 20170322 20170423 20170516 20170525 20170617 20170626 20171203
    20180221 20180313 20180410 20180512 20180603 20180710 20180807 20180817
    20180902 20180908 20180917 20181010 20181024 20181129 20181130
    20190313 20190429 20190710 20190730 20190914
    20200211 20200314 20200330 20200331 20200405 20200415 20200517 20200711
    20200726 20200906 20200916 20201005 20201223 20201224
    20210210 20210318 20210329 20210424 20210526 20210609 20210621 20210703
    20210727 20210825 20210906 20210908 20210926 20211016 20211229

    20230123 20230128 20230210 20230313 20230319 20230321 20230323 
    20230404 20230425 20230506 20230520 20230526
    20230611 20230626 20230704 20230720 20230804 20230810 20230814 
    20230902 20230911 20231011 20231013

    20240206 20240218 20240310 20240415 20240422 20240510 20240626 
    20240727 20240731 20240803 20240826 20240920 20241003 20241029 
    20241123 20241202 20241216
)

# Active set(s) — union + de-duplicate (preserve first-seen order).  The test
# starts with the 12 dates of 2020 plus the two orbits studied locally
# (2018-10-18, 2023-08-10); widen to the other arrays once the comparison on
# these dates is satisfactory.
# 2026-10-30: the launcher's original set (DATES_2020 + 20181018 20230810) has no
# production inputs on CURC (no results_<date>.h5, empty orbit dirs).  Substituted
# the 2020 dates that DO have production cloud distances + fits, plus 2018-11-18
# and 2023-09-04 as stand-ins for the two single-orbit dates.
# 2026-10-30 (evening): 20200101 added so the locally studied orbit 29252a_GL
# (reproduction target of the agreed-model handoff) is processed on CURC too.
DATES_DOPPLER_TEST=( 20200227 20200228 20200406 20200411 20200521 20200703
                     20200922 20200923 20201009 20201024 20181118 20230904
                     20200101 )
ALL=( "${DATES_DOPPLER_TEST[@]}" )
# ALL=( "${DATES_2020[@]}" "${DATES_DOPPLER_TEST[@]}" )
# ALL=( "${DATES_2020[@]}" "${DATES_2016_2020[@]}" "${DATES_EVENTS[@]}" )
DATES=()
while IFS= read -r d; do DATES+=("$d"); done < <(printf '%s\n' "${ALL[@]}" | awk '!seen[$0]++')
N=${#DATES[@]}
echo "Total unique dates: $N"

# ── Bootstrap: derive the array range from the active date set ────────────────────
# If we're not inside an array task yet (SLURM_ARRAY_TASK_ID unset), resubmit this
# same script as an array sized to the active date set, then exit.  This keeps the
# --array range in sync with ALL= automatically — no #SBATCH edit needed.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    RANGE="0-$(( N * REPLICAS - 1 ))%${MAX_CONCURRENT}"
    echo "Bootstrapping array job: --array=$RANGE"
    exec sbatch --array="$RANGE" "$0"
fi

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data
# NOTE: do NOT `pip install` here — concurrent array tasks writing the same env
# can race.  Provision pyproj/geopandas/shapely once beforehand (fp_anal.sh).

# Prepend conda's libs so netCDF4/h5py loads the conda-compiled libhdf5 rather
# than the one injected by `module load hdf5/...` (ABI mismatch → NC_EHDF -101).
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE

cd /projects/yuch8913/OCO_spectral_mitigation

# ── This task's date ────────────────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "$(( N * REPLICAS ))" ]; then
    echo "Array index $IDX >= $N dates x $REPLICAS replicas — nothing to do."
    exit 0
fi
RAW="${DATES[$(( IDX % N ))]}"
echo "Replica $(( IDX / N + 1 ))/$REPLICAS for this date"
DATE="${RAW:0:4}-${RAW:4:2}-${RAW:6:2}"
echo "Processing date: $DATE  (array task ${IDX})"
# --telluric-doppler selects the test variant, defaults the output tree to
# <storage>/results_doppler and the cloud-distance source to <storage>/results,
# and turns on --extra-fits.  Pass --output-dir / --cloud-dist-dir to override.
python src/spectral/fitting.py --date "$DATE" --telluric-doppler
if [ $? -ne 0 ]; then
    echo "Failed to process date: $DATE"
    exit 1
fi
echo "Successfully processed: $DATE"

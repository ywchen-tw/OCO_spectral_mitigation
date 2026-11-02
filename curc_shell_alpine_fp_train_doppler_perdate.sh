#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_fp_train_alp
#SBATCH --account=ucb744_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal

# ALPINE COPY of curc_shell_blanca_fp_train_doppler_perdate.sh.
# PER-DATE ARRAY: TELLURIC-DOPPLER VARIANT FITS (src/spectral/fitting.py
# --telluric-doppler: production fit orders + the order-2/3 extras and the agreed-
# model `_t2` / `_am` fields) for the 2016-2020 training dates, reading the v4 tau
# that curc_shell_blanca_tau_only_doppler_perdate.sh wrote to
# <storage>/results_doppler/<date>/<orbit>/fp_tau_combined.h5 and the cloud
# distances in <storage>/results/results_<date>.h5.  Output:
# <storage>/results_doppler/fitting_details/fitting_details_<date>_<orbit>.h5.
# Submit a date only once its tau is complete (otherwise preprocess() computes
# the missing tau itself, ~1 h per orbit); the scratchpad fit feeder does that.
# Same date list and index -> date mapping (idx % N) as the tau-only script;
# REPLICAS tasks per date split the orbits through orbit_lock, and finished
# fitting_details files are skipped on resubmission.
#
# Submit selected dates: sbatch -M alpine --array=<idx>,<idx+N>,... curc_shell_alpine_fp_train_doppler_perdate.sh
#
MAX_CONCURRENT=100
REPLICAS=2

# ── Date sets (YYYYMMDD), same three sets as build_feature_dataset.py main() ────
DATES_2020=(
    20200101 20200201 20200301 20200401 20200501 20200601
    20200701 20200801 20200903 20201001 20201101 20201201
)
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

# Active set: every training date whose year is 2016-2020 (all three sets),
# de-duplicated in first-seen order.  Narrow with YEAR_RE, e.g. '^2020'.
YEAR_RE='^(2016|2017|2018|2019|2020)'
ALL=( "${DATES_2020[@]}" "${DATES_2016_2020[@]}" "${DATES_EVENTS[@]}" )
DATES=()
while IFS= read -r d; do DATES+=("$d"); done < <(printf '%s\n' "${ALL[@]}" | awk '!seen[$0]++' | grep -E "$YEAR_RE")
# DATES_LIST_FILE=<file, one YYYYMMDD per line, '#' comments> replaces the built-in set:
# for dates added to build_feature_dataset.py later, without changing the idx -> date
# mapping of tasks already submitted from the built-in set.
if [ -n "${DATES_LIST_FILE:-}" ]; then
    DATES=()
    while IFS= read -r d; do DATES+=("${d//-/}"); done < <(grep -vE '^\s*(#|$)' "$DATES_LIST_FILE")
fi
N=${#DATES[@]}
echo "Total unique dates: $N  (x $REPLICAS replicas)"

# ── Bootstrap: derive the array range from the active date set ────────────────────
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    RANGE="0-$(( N * REPLICAS - 1 ))%${MAX_CONCURRENT}"
    echo "Bootstrapping array job: --array=$RANGE"
    exec sbatch -M alpine --array="$RANGE" "$0"
fi

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE
# single-threaded BLAS in the parent: the tau step forks a process pool
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /projects/yuch8913/OCO_spectral_mitigation

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "$(( N * REPLICAS ))" ]; then
    echo "Array index $IDX >= $N dates x $REPLICAS replicas — nothing to do."
    exit 0
fi
RAW="${DATES[$(( IDX % N ))]}"
DATE="${RAW:0:4}-${RAW:4:2}-${RAW:6:2}"
echo "Processing date: $DATE  (array task ${IDX}, replica $(( IDX / N + 1 ))/$REPLICAS)"
T0=$(date +%s)
python src/spectral/fitting.py --date "$DATE" --telluric-doppler
RC=$?
echo "Wall time: $(( ($(date +%s) - T0) / 60 )) min"
if [ $RC -eq 0 ]; then
    echo "Successfully processed: $DATE"
else
    echo "Failed to process date: $DATE"
    exit 1
fi

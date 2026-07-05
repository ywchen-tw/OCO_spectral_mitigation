#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=bfd
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue

# PER-DATE ARRAY JOB for the feature-dataset build (build_feature_dataset.py).
#
# Dual-mode, so the array range always matches the active date list — no manual
# --array bookkeeping.  Run it on a LOGIN node:
#     bash curc_shell_blanca_build_feature_dataset_perdate.sh
# In this "controller" pass it counts the active dates and resubmits ITSELF as a
# job array (--array=0-(N-1)%MAX_JOBS).  Each array task then builds ONE date's
# parquet (results/csv_collection/combined_<date>_all_orbits.parquet) via
# `python src/analysis/build_feature_dataset.py --date ...`.
#
# MAX_JOBS caps how many array tasks run at once (SLURM's %N throttle).
# Prerequisite: each date's fitting_details_<date>_*.h5 must already exist
# (produced by curc_shell_blanca_fp_anal_perdate.sh).

MAX_JOBS=36

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
    # 20141203 20141217
    # 20150213 20150218 20150317 20150323 20150518 20150520 20150605 20150629
    # 20150706 20150713 20150714 20150715 20150723 20150801 20150925
    # 20160303 20160506 20160529 20160607 20160709 20160821 20160910 20160911 
    # 20161013 20161105 20161107
    # 20170117 20170322 20170423 20170516 20170525 20170617 20170626 20171203
    # 20180221 20180313 20180410 20180512 20180603 20180710 20180807 20180817
    # 20180902 20180908 20180917 20181010 20181024 20181129 20181130
    # 20190313 20190429 20190710 20190730 20190914
    # 20200211 20200314 20200330 20200331 20200405 20200415 20200517 20200711
    # 20200726 20200906 20200916 20201005 20201223 20201224
    # 20210210 20210318 20210329 20210424 20210526 20210609 20210621 20210703
    # 20210727 20210825 20210906 20210908 20210926 20211016 20211229

    # 20230123 20230128 20230210 20230313 20230319 20230321 20230323 
    # 20230404 20230425 20230506 20230520 20230526
    # 20230611 20230626 20230704 20230720 20230804 20230810 20230814 
    # 20230902 20230911 20231011 20231013

    # 20240206 20240218 20240310 20240415 20240422 20240510 20240626 
    # 20240727 20240731 20240803 20240826 20240920 20241003 20241029 
    # 20241123 20241202 20241216
    
    # 20170126 20170203 20170206
    # 20170204
    20171009
    # 20170205 20170210 20171008 
    # 20171020 20171027 20180512
    # 20190609 20190614 20190622 20210315
)

# Active set(s) — union + de-duplicate (preserve first-seen order).  Currently
# 2020 only; add the other arrays here to widen (the range auto-adjusts).
# ALL=( "${DATES_2016_2020[@]}" )
ALL=( "${DATES_EVENTS[@]}" )
# ALL=( "${DATES_2016_2020[@]}" "${DATES_EVENTS[@]}" )
# ALL=( "${DATES_2020[@]}" "${DATES_2016_2020[@]}" "${DATES_EVENTS[@]}" )
DATES=()
while IFS= read -r d; do DATES+=("$d"); done < <(printf '%s\n' "${ALL[@]}" | awk '!seen[$0]++')
N=${#DATES[@]}

# ── Controller pass: not inside an array task → submit the array and exit ───────
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ "$N" -eq 0 ]]; then echo "No dates selected."; exit 1; fi
    echo "Submitting ${N} per-date build jobs (array 0-$((N - 1)), max ${MAX_JOBS} concurrent)."
    exec sbatch --array=0-$((N - 1))%${MAX_JOBS} "$(readlink -f "$0")"
fi

# ── Array task: build one date's parquet ────────────────────────────────────────
module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data

# Prepend conda's libs so netCDF4/h5py loads the conda-compiled libhdf5 rather
# than the one injected by `module load hdf5/...` (ABI mismatch → NC_EHDF -101).
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE

cd /projects/yuch8913/OCO_spectral_mitigation

RAW="${DATES[$SLURM_ARRAY_TASK_ID]}"
DATE="${RAW:0:4}-${RAW:4:2}-${RAW:6:2}"
echo "Processing date: $DATE  (array task ${SLURM_ARRAY_TASK_ID}/${N})"
python src/analysis/build_feature_dataset.py --date "$DATE"
if [ $? -ne 0 ]; then
    echo "Failed to process date: $DATE"
    exit 1
fi
echo "Successfully processed: $DATE"

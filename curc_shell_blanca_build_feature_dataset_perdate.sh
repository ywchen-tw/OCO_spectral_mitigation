#!/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=bfd_${DATE}
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable


#!/bin/env bash
#
# Per-date launcher for build_feature_dataset.py.
#
# Unlike curc_shell_blanca_fitting_combined_general.sh (one job that processes
# the whole date_list sequentially), this script submits ONE sbatch job PER DATE
# so the per-date parquets build in parallel across the cluster.  Each job runs
#     python src/analysis/build_feature_dataset.py --date <YYYY-MM-DD>
# which writes results/csv_collection/combined_<date>_all_orbits.parquet.
#
# Run it on a LOGIN node (it is a submitter, not itself a batch job):
#     bash curc_shell_blanca_build_feature_dataset_perdate.sh
#     DRY_RUN=1 bash curc_shell_blanca_build_feature_dataset_perdate.sh   # print, don't submit
#
# The three date sets below are copied verbatim from build_feature_dataset.py
# main() (the active list + the two commented-out lists).  They are unioned and
# de-duplicated (order preserved) before submission.  Comment out a set to skip it.

set -euo pipefail

REPO=/projects/yuch8913/OCO_spectral_mitigation
DRY_RUN="${DRY_RUN:-0}"

# ── Set 1 — 2020-only (top, commented-out in build_feature_dataset.py) ──────────
DATES_2020=(
    20200101 20200201 20200301 20200401 20200501 20200601
    20200701 20200801 20200903 20201001 20201101 20201201
)

# ── Set 2 — 2016–2020, 1st + 15th of month (the ACTIVE date_list) ───────────────
DATES_2016_2020=(
    # 1st-of-month block
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
    # 15th-of-month block
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

# ── Set 3 — event-specific dates 2014–2021 (bottom, commented-out) ──────────────
DATES_EVENTS=(
    20141203 20141217
    20150213 20150218 20150317 20150323 20150518 20150520 20150605 20150629
    20150706 20150713 20150714 20150723 20160910 20150925
    20160303 20160506 20160529 20160607 20160709 20160911 20161013 20161105
    20161107
    20170117 20170322 20170423 20170516 20170525 20170617 20170626 20171203
    20180221 20180313 20180410 20180512 20180603 20180710 20180807 20180817
    20180902 20180908 20180917 20181010 20181024 20181129 20181130
    20190313 20190429 20190710 20190730 20190914
    20200211 20200314 20200330 20200331 20200405 20200415 20200517 20200711
    20200726 20200906 20200916 20201005 20201223 20201224
    20210210 20210318 20210329 20210424 20210526 20210609 20210621 20210703
    20210727 20210825 20210906 20210908 20210926 20211016 20211229
)

# ── Union + de-duplicate (preserve first-seen order) ────────────────────────────
# `while read` (not mapfile) so this also runs under the bash 3.2 on macOS.
# ALL=( "${DATES_2020[@]}" "${DATES_2016_2020[@]}" "${DATES_EVENTS[@]}" )
ALL=( "${DATES_2020[@]}")
DATES=()
while IFS= read -r d; do DATES+=("$d"); done < <(printf '%s\n' "${ALL[@]}" | awk '!seen[$0]++')

echo "Submitting ${#DATES[@]} per-date jobs (of ${#ALL[@]} with duplicates)."
[[ "$DRY_RUN" == "1" ]] && echo "DRY_RUN=1 — printing only, not submitting."

for DATE in "${DATES[@]}"; do
    ISO="${DATE:0:4}-${DATE:4:2}-${DATE:6:2}"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  would submit: $ISO  (job bfd_${DATE})"
        continue
    fi

    sbatch <<EOF


module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data
# Prepend conda's libs so netCDF4/h5py loads the conda-compiled libhdf5 rather
# than the one injected by \`module load hdf5/...\` (ABI mismatch → NC_EHDF -101).
if [[ "\$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:\$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
fi
# HDF5 file locking is unsupported on Lustre (/pl/active/) → NC_EHDF (-101).
export HDF5_USE_FILE_LOCKING=FALSE

cd ${REPO}
python src/analysis/build_feature_dataset.py --date ${ISO}
EOF
done

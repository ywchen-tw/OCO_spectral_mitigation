#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_fp_prodextra
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue

# PRODUCTION-TAU EXTRA FITS for the telluric-Doppler comparison (2026-10-30).
#
# The variant (fitting.py --telluric-doppler) stores order-2/3 fit statistics
# and the residual rms of the production-order fit (<band>_rms_prod_nosg).
# Production fitting_details hold none of these, so to compare like with like
# we run the SAME fit step with --extra-fits on the PRODUCTION tau, writing to
# <storage>/results_doppler/prod_extra.  That tree holds symlinks
#   prod_extra/<date>/<orbit>/fp_tau_combined.h5 -> results/<date>/<orbit>/fp_tau_combined.h5
# so fitting.py finds a current production tau (version stamp v2) and never
# recomputes it; the fit step only reads the tau file.  Fits, PNGs and
# fitting_details go to prod_extra/.  Production trees are never written.
# Orbits are restricted (--orbit) to those listed in prod_extra/<date>/orbits.txt,
# i.e. those with a production tau, so no tau is ever computed here.
#
# Submit:  sbatch -M blanca curc_shell_blanca_fp_anal_prodextra.sh
# The array range follows the DATES list below (bootstrap, as in the other launchers).

MAX_CONCURRENT=12
DATES=( 20200227 20200228 20200406 20200411 20200521 20200703
        20200922 20200923 20201009 20201024 20181118 20230904 )
N=${#DATES[@]}

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    RANGE="0-$(( N - 1 ))%${MAX_CONCURRENT}"
    echo "Bootstrapping array job: --array=$RANGE"
    exec sbatch --array="$RANGE" "$0"
fi

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE

cd /projects/yuch8913/OCO_spectral_mitigation
STORAGE=${OCO2_DATAROOT:-/scratch/alpine/yuch8913/oco/spec_anal}
PROD_EXTRA=$STORAGE/results_doppler/prod_extra

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "$N" ]; then echo "Array index $IDX >= $N — nothing to do."; exit 0; fi
RAW="${DATES[$IDX]}"
DATE="${RAW:0:4}-${RAW:4:2}-${RAW:6:2}"
LIST=$PROD_EXTRA/$DATE/orbits.txt
if [ ! -s "$LIST" ]; then echo "No $LIST (symlink farm not built) — abort"; exit 1; fi
ORBIT_ARGS=()
while IFS= read -r o; do [ -n "$o" ] && ORBIT_ARGS+=( --orbit "$o" ); done < "$LIST"
echo "Processing date: $DATE  (array task ${IDX}, $(( ${#ORBIT_ARGS[@]} / 2 )) orbits with production tau)"
T0=$(date +%s)
python src/spectral/fitting.py --date "$DATE" --extra-fits --fit-workers 8 \
    --output-dir "$PROD_EXTRA" --cloud-dist-dir "$STORAGE/results" "${ORBIT_ARGS[@]}"
RC=$?
echo "Wall time: $(( ($(date +%s) - T0) / 60 )) min"
if [ $RC -ne 0 ]; then echo "Failed to process date: $DATE"; exit 1; fi
echo "Successfully processed: $DATE"

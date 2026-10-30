#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_toa_only
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue

# PER-DATE ARRAY JOB: upgrade the telluric-Doppler variant tau files of one date
# from stamp v3 (solar grid shifted by v_solar only) to v4 (solar grid shifted by
# v_solar - v_inst) by recomputing ONLY the <band>_toa_sol_output datasets
# (workspace/compute_tau_only.py --toa-only, abs_util.toa_solar).  tau itself and
# the attribution datasets are unchanged, so this takes minutes per orbit instead
# of the ~1 h of the full tau step.  Orbits with no tau file are only reported
# (exit 2): they get the full v4 tau step from curc_shell_blanca_fp_anal_doppler.sh.
# Run this BEFORE resubmitting the fitting launcher: fitting.py --telluric-doppler
# treats a v3 file as not current and would recompute it from scratch.
#
# Submit:  sbatch -M blanca curc_shell_blanca_toa_only_doppler.sh
# (bootstrap resubmits itself as --array=0-(N-1); an explicit --array is honored)

MAX_CONCURRENT=13
# Same date set as curc_shell_blanca_fp_anal_doppler.sh (DATES_DOPPLER_TEST).
DATES=( 20200227 20200228 20200406 20200411 20200521 20200703
        20200922 20200923 20201009 20201024 20181118 20230904
        20200101 )
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /projects/yuch8913/OCO_spectral_mitigation

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "$N" ]; then echo "Array index $IDX >= $N dates — nothing to do."; exit 0; fi
RAW="${DATES[$IDX]}"
DATE="${RAW:0:4}-${RAW:4:2}-${RAW:6:2}"
STORAGE=/scratch/alpine/yuch8913/oco/spec_anal
echo "Processing date: $DATE  (array task ${IDX})"
python workspace/compute_tau_only.py --date "$DATE" --telluric-doppler \
    --output-dir "$STORAGE/results_doppler" --toa-only --workers "${SLURM_NTASKS:-8}"
RC=$?
case $RC in
  0) echo "Successfully processed: $DATE (all tau files at v4)";;
  2) echo "Partially processed: $DATE (orbits without a tau file or locked by another job; see above)";;
  *) echo "Failed to process date: $DATE (exit $RC)"; exit 1;;
esac
exit $RC

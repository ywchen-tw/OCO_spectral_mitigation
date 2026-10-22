#!/bin/env bash
# Fit one (or a few) orbit(s) of one date.
#   sbatch --clusters=blanca curc_shell_blanca_fp_anal_orbit.sh 2020-09-22 33124a_TG [more orbits...]
# Add OVERWRITE=0 in the environment to reuse an existing complete fit
# instead of refitting (default is to refit the requested orbits).

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_spectral_anal
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data

# Conda-compiled libhdf5 must win over the module one (ABI mismatch -> NC_EHDF).
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE   # Lustre has no HDF5 file locking

cd /projects/yuch8913/OCO_spectral_mitigation

DATE="$1"; shift
if [[ -z "$DATE" || $# -eq 0 ]]; then
    echo "usage: sbatch curc_shell_blanca_fp_anal_orbit.sh YYYY-MM-DD ORBIT_ID [ORBIT_ID ...]"
    exit 2
fi
ORBIT_ARGS=()
for o in "$@"; do ORBIT_ARGS+=(--orbit "$o"); done
OVERWRITE_ARG=""
[[ "${OVERWRITE:-1}" != "0" ]] && OVERWRITE_ARG="--overwrite"

echo "Processing date: $DATE  orbits: $*  ${OVERWRITE_ARG:-(reuse existing fits)}"
python src/spectral/fitting.py --date "$DATE" "${ORBIT_ARGS[@]}" $OVERWRITE_ARG
status=$?
if [ $status -ne 0 ]; then
    echo "Failed to process date: $DATE orbits: $*"
    exit $status
fi
echo "Successfully processed: $DATE orbits: $*"

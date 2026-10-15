# Shared CURC job environment.  Source it, then call curc_env:
#
#   source /projects/yuch8913/OCO_spectral_mitigation/curc_env.sh
#   curc_env --torch cuda/12.1.1 || exit 1     # GPU (deep ensemble) jobs
#   curc_env || exit 1                          # CPU jobs
#
# Why: Lmod sometimes cannot find `anaconda` on a node ("Please check the
# spelling or version number"; keep-only TCCON 2026-10-11, r15 fold 4
# 2026-10-14).  Under `set -euo pipefail` one failed `module load` ended the job
# before python started and left empty logs.  The modules are now loaded one by
# one, a failure is logged, and the data env's own python is used directly; it
# carries its own HDF5/netCDF (and, for torch, CUDA runtime) libraries.
# The python check at the end fails the job loudly if that env is not usable.

CURC_ENV_DIR=/projects/yuch8913/software/anaconda/envs/data
CURC_BASE_MODULES=(anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8)

curc_env() {   # curc_env [--torch] [extra modules...]
    # Module/conda scripts break under -e/-u; the caller's options are restored
    # on return (portable: bash 3.2 has no `local -`).
    local saved_flags=$- torch=0 mod rc=0
    set +eu
    if [[ "${1:-}" == --torch ]]; then torch=1; shift; fi
    echo "node $(hostname)  $(date)"
    # sbatch exports the submitting shell's module state; a stale MODULEPATH
    # there is one way Lmod loses anaconda on the node.
    module purge 2>/dev/null
    for mod in "${CURC_BASE_MODULES[@]}" "$@"; do
        module load "$mod" 2>/dev/null || echo "module load $mod FAILED"
    done
    conda activate data 2>/dev/null || echo "conda activate data FAILED, using $CURC_ENV_DIR/bin directly"
    export PATH="$CURC_ENV_DIR/bin:$PATH"
    # conda's libhdf5 before the module's (ABI mismatch -> NC_EHDF otherwise)
    export LD_LIBRARY_PATH="$CURC_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
    echo "python: $(command -v python)"
    local check='import numpy, pandas, pyarrow'
    if (( torch )); then
        check+='; import torch; assert torch.cuda.is_available(), "torch sees no CUDA device"'
    fi
    python -c "$check" || { echo "data env python not usable ($check)" >&2; rc=1; }
    [[ $saved_flags == *e* ]] && set -e
    [[ $saved_flags == *u* ]] && set -u
    return $rc
}

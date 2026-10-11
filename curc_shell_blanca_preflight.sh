#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_preflight
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable


module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate data

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

export HDF5_USE_FILE_LOCKING=FALSE

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:$PYTHONPATH

# ── Stage 1.5: pre-flight check of the CURC training scripts ─────────────────
# Do NOT run the training scripts themselves — this only validates syntax and
# previews which commands each script will execute.

TRAIN_SCRIPTS=(
    curc_shell_blanca_train_pipeline.sh
    curc_shell_blanca_train_tabm.sh
    curc_shell_blanca_train_gbdt.sh
    curc_shell_blanca_train_mlp_baseline.sh
)

echo "=== 1) Shell syntax check ==="
for f in "${TRAIN_SCRIPTS[@]}"; do
    bash -n "$f" && echo "syntax OK: $f" || echo "SYNTAX ERROR: $f"
done
echo ""

echo "=== 2) Active (uncommented) python commands per script ==="
for f in "${TRAIN_SCRIPTS[@]}"; do
    echo "# $f"
    grep -E '^python' "$f"
    echo ""
done

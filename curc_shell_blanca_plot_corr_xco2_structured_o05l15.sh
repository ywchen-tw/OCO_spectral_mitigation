#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_plot_struct_o05l15
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable

# Structured-residual real-date/TCCON run:
#   ocean uses the r05 cloud-distance-reference model
#   land  uses the r15 cloud-distance-reference model
#
# Useful knobs:
#   MAKE_PLOTS=0 sbatch ...    # build plot_data + aggregate TCCON tables only
#   REBUILD=1 sbatch ...       # force regeneration of existing plot_data.parquet

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_DIR="${PROJECT_DIR:-/projects/yuch8913/OCO_spectral_mitigation}"
else
    PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
fi
cd "${SLURM_SUBMIT_DIR:-$PROJECT_DIR}"

export STRUCT_MODEL_TAG="structured_shared_foldpca_o05l15_m5"
export STRUCT_OCEAN_STEM="de2016_2020_structured_shared_h64x32_b8_foldpca_r05_ocean"
export STRUCT_LAND_STEM="de2016_2020_structured_shared_h64x32_b8_foldpca_r15_land"

exec bash "$PROJECT_DIR/curc_shell_blanca_plot_corr_xco2_structured_common.sh"

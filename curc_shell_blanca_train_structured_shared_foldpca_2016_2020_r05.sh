#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_struct_fp_r05
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%4
#SBATCH --requeue

# 5 km clear-sky reference target.
# Produces suffixes:
#   de2016_2020_structured_shared_h64x32_b8_foldpca_r05_{ocean,land}_f*

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export STRUCT_TARGET_ALIAS="5km"
export STRUCT_TARGET_TAG="r05"
exec "${SCRIPT_DIR}/curc_shell_blanca_train_structured_shared_foldpca_2016_2020.sh"

#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_de_lcurve
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --requeue

# DE LEARNING CURVE — the decisive "more data vs better model" diagnostic.
#
# The near-cloud tail proved DATA-limited: tail-5% R2 went 0.05 (275k rows, local
# 12-date gate) -> ~0.55 ocean / ~0.73 land at full ~5M rows.  Open question: is
# the FULL set on the steep part of the curve (=> ingesting 2014/15/21 dates will
# pay) or the plateau (=> redirect effort to richer cloud features)?
#
# This subsamples ONLY the proper-train (--train_frac; calibration block + held-out
# fold are IDENTICAL across fractions, so tail R2 vs N is measured against the same
# test set).  Production config otherwise (beta_nll=1.0, M=5, full_contam,
# Mondrian-CQR by cld_dist_km, near_cloud_target 0.98).  One FRACTION per array task
# (0.25/0.5/0.75/1.0 of ~5M -> ~1.25M/2.5M/3.75M/5M rows); ocean + land each.
# Fold 0 only (the curve SHAPE needs one consistent fold, not error bars).  Suffix:
# lc_{surface}_fr{FRAC}_f0.
#
# READ: plot tail-5% R2 vs train_N (printed in each run's log as
# "proper-train X -> Y rows").  If the 0.75 -> 1.0 segment is still rising, more
# data helps -> ingest 2014/2015/2021 (already collocated as per-date parquets).
# If it has flattened by ~3.75M, you are near the data ceiling -> features next.
#   for SURF in ocean land; do
#     PYTHONPATH=src python - <<PY
# import json,glob,re,pandas as pd
# B='results/model_deep_ensemble'
# for fr in ['0.25','0.5','0.75','1.0']:
#     d=f"{B}/lc_${SURF}_fr{fr}_f0"
#     s=pd.read_csv(f"{d}/de_mondrian_date_kfold_stratified_metrics.csv")
#     t5=float(s[s.group=='near&bottom_5pct'].iloc[0]['r2'])
#     g=json.load(open(f"{d}/de_mondrian_date_kfold_metrics.json"))['global']['r2']
#     print("${SURF}", fr, "tail5%R2=%.4f globalR2=%.4f"%(t5,g))
# PY
#   done

module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8 cuda/12.1.1
conda activate data

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_NUM_THREADS=${SLURM_NTASKS:-4}
export MKL_THREADING_LAYER=GNU

cd /projects/yuch8913/OCO_spectral_mitigation
export PYTHONPATH=src:$PYTHONPATH

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv --loop=10 > gpu_monitor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv &
GPU_MONITOR_PID=$!

# One fraction per array task.
FRACS=(0.25 0.5 0.75 1.0)
FR=${FRACS[$SLURM_ARRAY_TASK_ID]}
COMMON="--loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
        --feature_set full_contam --near_cloud_target 0.98 --mondrian_col cld_dist_km \
        --val_split date_kfold --n_folds 5 --fold 0 --train_frac ${FR}"

python -m models.deep_ensemble --sfc_type 0 --suffix lc_ocean_fr${FR}_f0 ${COMMON}
python -m models.deep_ensemble --sfc_type 1 --suffix lc_land_fr${FR}_f0  --include_snow ${COMMON}

kill $GPU_MONITOR_PID 2>/dev/null || true

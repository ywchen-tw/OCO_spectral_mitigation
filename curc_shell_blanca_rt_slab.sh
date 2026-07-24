#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_rt_slab
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --array=0-131
#SBATCH --requeue

# ── Appendix G x-z slab RT simulation: production photon budget ──────────────
# One array task = one (surface, solver, wavelength) MCARaTS run at 1e9
# photons x Nrun=3 on the identical slab scene (see workspace/rt_slab_sim/;
# scene: SZA-55 footprint 2020010100281632, cloud 3-4 km / COD 10 at
# x = 9.5-14.5 km in a 32 km Ny=1 slab, solvers 3d vs ipa, dark+bright).
#
# PREREQUISITES (run locally, then rsync to CURC SCRATCH --
# /scratch/alpine/$USER/oco2_data/results/rt_slab_sim/ per workspace/curc_setup.sh):
#   slab_atm.h5   (stage 1: build_atmosphere.py)
#   slab_od.h5    (stage 2: gas_od.py; needs local ABSCO)
# and on CURC:
#   - er3t v0.10-era snapshot        -> export ER3T_DIR=<er3t repo root>
#   - MCARaTS v0.10.4 binary (Linux) -> export MCARATS_V010_EXE=<path>
#   - the er3t water-cloud Mie table cache: submit ONE task first
#     (sbatch --array=0 <this script>) so pha_mie_wc builds its pickle
#     without 132 tasks racing on it, then submit the rest (--array=1-131).
#
# Task index mapping: idx = ((isurf * 2) + isolver) * NWVL + iwvl
#   isurf  0=dark 1=bright ; isolver 0=3d 1=ipa ; iwvl 0..NWVL-1
# (NWVL is read from slab_od.h5 -- do not hardcode; the wavelength selection
#  is data-derived and the array range above assumes 33.)
#
# AFTER the array completes, collect on a login/compile node (cheap; with
# SCRATCH_DIR exported so it finds the runs):
#   python workspace/rt_slab_sim/run_slab.py --collect
# then rsync ${SCRATCH_DIR}/results/rt_slab_sim/slab_rad.h5 back into local
# results/rt_slab_sim/ and rerun fit_and_plot.py / plot_ppdf.py locally.

module purge
module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
#conda activate data
conda activate er3t

if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
fi

# EDIT THESE for the CURC filesystem before first submit:
export ER3T_DIR=${ER3T_DIR:-/projects/yuch8913/er3t}
export MCARATS_V010_EXE=${MCARATS_V010_EXE:-/projects/yuch8913/mcarats/v0.10.4/src/mcarats}

# results live on scratch (same convention as workspace/curc_setup.sh);
# slab_config.py uses SCRATCH_DIR as the parent of results/, so everything
# is under ${SCRATCH_DIR}/results/rt_slab_sim -- rsync slab_atm.h5 +
# slab_od.h5 THERE
export SCRATCH_DIR=${SCRATCH_DIR:-/scratch/alpine/${USER}/oco2_data}
mkdir -p "${SCRATCH_DIR}/results/rt_slab_sim"

NWVL=$(python -c "import sys; sys.path.insert(0,'workspace/rt_slab_sim'); import slab_config as c; import h5py; print(h5py.File(c.OD_FILE,'r')['wvl_nm'].shape[0])")

IDX=${SLURM_ARRAY_TASK_ID}
IWVL=$(( IDX % NWVL ))
ISOLVER=$(( (IDX / NWVL) % 2 ))
ISURF=$(( IDX / (NWVL * 2) ))

SURFACES=(dark bright)
SOLVERS=(3d ipa)

SURF=${SURFACES[$ISURF]}
SOLVER=${SOLVERS[$ISOLVER]}

echo "task ${IDX}: surface=${SURF} solver=${SOLVER} wvl=${IWVL} (NWVL=${NWVL})"

python workspace/rt_slab_sim/run_slab.py \
    --photons 1e9 --nrun 3 \
    --wvl ${IWVL} \
    --surfaces ${SURF} \
    --solvers ${SOLVER} \
    --ncpu ${SLURM_NTASKS}

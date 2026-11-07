#!/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=sbatch-output_%x_%A_%a.txt
#SBATCH --job-name=oco_rt_slab_scenes
#SBATCH --account=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --exclude=bnode0503,bnode0504,bnode0505,bnode0506,bnode0507,bnode0508,bnode0509,bnode0510,blanca-cores1,blanca-cores2,bgpu-biokem1,bgpu-biokem2,bgpu-biokem3

# ── Slab RT condition sweep (2026-11-06): one array task = one
# (scene, surface, solver), all 33 wavelengths in sequence, --skip-existing so
# a requeued task resumes.  STAGE selects what the task does:
#   STAGE=prep    one task per scene: build_atmosphere + gas_od into the
#                 scene's results/rt_slab_sim/scenes/<scene>/  (array 0..NSCENE-1)
#   STAGE=run     MCARaTS runs (array 0..4*NSCENE-1; idx = iscene*4 + isurf*2 + isolver)
#   STAGE=collect one task per scene: assemble slab_rad.h5          (array 0..NSCENE-1)
# Submit the three with dependencies (see the "launch" block at the bottom of
# this file for the exact commands).  Scenes are defined in
# workspace/rt_slab_sim/slab_config.py (SCENES); SCENES_RUN picks the subset.
# PHOTONS defaults to the 1e8 screening budget; PHOTONS=1e9 is production.

set -uo pipefail
module purge
module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
conda activate er3t
export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:${LD_LIBRARY_PATH:-}
PY=/projects/yuch8913/software/anaconda/envs/er3t/bin/python

export ER3T_DIR=${ER3T_DIR:-/projects/yuch8913/wen_soft/er3t}
export MCARATS_V010_EXE=${MCARATS_V010_EXE:-/projects/yuch8913/wen_soft/mcarats/v0.10.4/src/mcarats}
export ABSCO_DIR=${ABSCO_DIR:-/scratch/alpine/${USER}/oco/data/absco/v5.2_final}
export OCO2_DATAROOT=${OCO2_DATAROOT:-/scratch/alpine/${USER}/oco/spec_anal}

STAGE=${STAGE:-run}
PHOTONS=${PHOTONS:-1e8}
NRUN=${NRUN:-3}
NSUB=${NSUB:-1}          # sub-runs per run (PHOTONS/NSUB each, parallel); 1e9 production: PHOTONS=1e9 NSUB=10
WVL=${WVL:-}             # optional wavelength-index subset (smoke tests)
SCENES_RUN=${SCENES_RUN:-"sza25 sza70 vza43_fwd vza43_back cloud_low cod2 cod30 rayleigh_off width1 width15"}
read -r -a SCENE_LIST <<< "$SCENES_RUN"
NSCENE=${#SCENE_LIST[@]}

IDX=${SLURM_ARRAY_TASK_ID}
SURFACES=(dark bright)
SOLVERS=(3d ipa)
if [[ "$STAGE" == "run" && -n "${TASK_LIST:-}" ]]; then
    # explicit task list (scenes with more than two surfaces, e.g. the albedo
    # ladder): line IDX+1 of $TASK_LIST is "scene surface solver"
    read -r SLAB_SCENE SURF SOLVER < <(sed -n "$((IDX + 1))p" "$TASK_LIST")
    [[ -n "${SOLVER:-}" ]] || { echo "task $IDX beyond $TASK_LIST"; exit 1; }
    export SLAB_SCENE
elif [[ "$STAGE" == "run" ]]; then
    ISCENE=$(( IDX / 4 ))
    ISURF=$(( (IDX / 2) % 2 ))
    ISOLVER=$(( IDX % 2 ))
    (( ISCENE < NSCENE )) || { echo "task $IDX beyond scene list"; exit 1; }
    export SLAB_SCENE=${SCENE_LIST[$ISCENE]}
    SURF=${SURFACES[$ISURF]}; SOLVER=${SOLVERS[$ISOLVER]}
else
    ISCENE=$IDX
    (( ISCENE < NSCENE )) || { echo "task $IDX beyond scene list"; exit 1; }
    export SLAB_SCENE=${SCENE_LIST[$ISCENE]}
fi

cd /projects/yuch8913/OCO_spectral_mitigation
hostname; date
echo "stage=$STAGE scene=$SLAB_SCENE task=$IDX photons=$PHOTONS nrun=$NRUN nsub=$NSUB tree=${SLAB_SCENES_SUBDIR:-scenes}"

case "$STAGE" in
  prep)
    $PY workspace/rt_slab_sim/build_atmosphere.py || exit 1
    $PY workspace/rt_slab_sim/gas_od.py || exit 1
    ;;
  run)
    echo "surface=$SURF solver=$SOLVER"
    $PY workspace/rt_slab_sim/run_slab.py \
        --photons "$PHOTONS" --nrun "$NRUN" --nsub "$NSUB" \
        --surfaces "$SURF" --solvers "$SOLVER" \
        --ncpu "$SLURM_NTASKS" --skip-existing ${WVL:+--wvl $WVL} || exit 1
    ;;
  collect)
    $PY workspace/rt_slab_sim/run_slab.py --collect || exit 1
    ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
date
echo "stage $STAGE scene $SLAB_SCENE task $IDX done"

# ── launch (run by hand; NSCENE = number of scenes in SCENES_RUN) ────────────
#   P=$(sbatch --parsable --export=ALL,STAGE=prep    --array=0-$((NSCENE-1))   --ntasks=4 --mem=8G --time=00:30:00 curc_shell_blanca_rt_slab_scenes.sh)
#   R=$(sbatch --parsable --export=ALL,STAGE=run     --array=0-$((4*NSCENE-1)) --dependency=afterok:$P curc_shell_blanca_rt_slab_scenes.sh)
#   (or, for scenes with more surfaces, write a task list -- one "scene surface
#    solver" per line, surfaces from SLAB_SCENE=<s> python -c "import slab_config as c; print(*c.SURFACE_ALBEDOS)"
#    -- and submit --export=ALL,STAGE=run,TASK_LIST=<file> --array=0-$((NLINES-1)))
#   C=$(sbatch --parsable --export=ALL,STAGE=collect --array=0-$((NSCENE-1))   --ntasks=2 --mem=8G --time=00:30:00 --dependency=afterany:$R curc_shell_blanca_rt_slab_scenes.sh)

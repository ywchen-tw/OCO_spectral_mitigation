#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_spectral_anal
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1


module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8 cuda/12.1.1
conda activate data

# Prepend conda's libs so Python's netCDF4/h5py loads the conda-compiled
# libhdf5 rather than the system one injected by `module load hdf5/...`.
# Without this, an ABI mismatch causes NC_EHDF (-101) at H5Fopen() time.
# On Linux, $CONDA_PREFIX may be empty when conda activate runs non-interactively
# (SLURM batch context), so hardcode the path as a reliable fallback.
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH=/projects/yuch8913/software/anaconda/envs/data/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

# HDF5 file locking is not supported on Lustre (/pl/active/).
# Without this, HDF5 ≥ 1.10 raises NC_EHDF (-101) on any open() call.
export HDF5_USE_FILE_LOCKING=FALSE

cd /projects/yuch8913/OCO_spectral_mitigation

# python src/apply_models.py \
#   --pipeline  results/train_data/pipeline_ocean_2017_2020.pkl \
#   --ridge-dir results/model_mlp_lr/ocean_2017_2020/ \
#   --mlp-dir   results/model_mlp_lr/ocean_2017_2020/ \
#   --ft-dir    results/model_ft_transformer/ocean_2017_2020/ \
#   --input     results/csv_collection/combined_2020-05-17_all_orbits.csv  \
#   --output    corrected.csv \
#   --plot-dir  results/model_comparison/ocean_2017_2020_2020-05-17

# python src/apply_models.py \
#   --pipeline  results/train_data/pipeline_land_2017_2020.pkl \
#   --ridge-dir results/model_mlp_lr/land_2017_2020/ \
#   --mlp-dir   results/model_mlp_lr/land_2017_2020/ \
#   --ft-dir    results/model_ft_transformer/land_2017_2020/ \
#   --input     results/csv_collection/combined_2018-10-24_all_orbits.csv  \
#   --output    corrected.csv \
#   --plot-dir  results/model_comparison/land_2017_2020_2018-10-24

python src/apply_models.py \
  --pipeline   results/train_data/pipeline_ocean_2016_2020.pkl \
  --ridge-dir  results/model_mlp_lr/ocean_2016_2020/ \
  --mlp-dir    results/model_mlp_lr/ocean_2016_2020/ \
  --ft-dir     results/model_ft_transformer/ocean_2016_2020/ \
  --xgb-dir    results/model_xgb/ocean_2016_2020/ \
  --hybrid-dir results/model_hybrid/ocean_2016_2020/ \
  --input-dir  results/csv_collection \
  --input     combined_2020-02-11_all_orbits.parquet combined_2018-11-29_all_orbits.parquet combined_2020-05-01_all_orbits.parquet combined_2020-03-30_all_orbits.parquet combined_2020-09-06_all_orbits.parquet combined_2018-09-01_all_orbits.parquet combined_2018-09-02_all_orbits.parquet \
  --output    corrected.csv \
  --plot-dir  results/model_comparison/ocean_2016_2020

  # combined_2020-03-30_all_orbits.parquet combined_2020-09-06_all_orbits.parquet combined_2018-09-01_all_orbits.parquet combined_2018-09-02_all_orbits.parquet
    # combined_2018-02-21_all_orbits.parquet combined_2018-11-30_all_orbits.parquet combined_2020-04-15_all_orbits.parquet
    # combined_2020-02-11_all_orbits.parquet combined_2018-11-29_all_orbits.parquet combined_2020-05-01_all_orbits.parquet

python src/apply_models.py \
  --pipeline   results/train_data/pipeline_land_2016_2020.pkl \
  --ridge-dir  results/model_mlp_lr/land_2016_2020/ \
  --mlp-dir    results/model_mlp_lr/land_2016_2020/ \
  --ft-dir     results/model_ft_transformer/land_2016_2020/ \
  --xgb-dir    results/model_xgb/land_2016_2020/ \
  --hybrid-dir results/model_hybrid/land_2016_2020/ \
  --input-dir  results/csv_collection \
  --input     combined_2021-12-29_all_orbits.parquet combined_2018-10-24_all_orbits.parquet combined_2020-01-15_all_orbits.parquet combined_2021-04-24_all_orbits.parquet \
  --output    corrected.csv \
  --plot-dir  results/model_comparison/land_2016_2020


    # combined_2021-12-29_all_orbits.parquet combined_2018-10-24_all_orbits.parquet combined_2020-01-15_all_orbits.parquet combined_2021-04-24_all_orbits.parquet



python src/apply_models.py \
  --pipeline   results/train_data/pipeline_ocean_2016_2020.pkl \
  --ridge-dir  results/model_mlp_lr/ocean_2016_2020_2/ \
  --mlp-dir    results/model_mlp_lr/ocean_2016_2020_2/ \
  --ft-dir     results/model_ft_transformer/ocean_2016_2020_2/ \
  --xgb-dir    results/model_xgb/ocean_2016_2020_2/ \
  --hybrid-dir results/model_hybrid/ocean_2016_2020_2/ \
  --input-dir  results/csv_collection \
  --input     combined_2020-02-11_all_orbits.parquet combined_2018-11-29_all_orbits.parquet combined_2020-05-01_all_orbits.parquet combined_2020-03-30_all_orbits.parquet combined_2020-09-06_all_orbits.parquet \
  --output    corrected.csv \
  --plot-dir  results/model_comparison/ocean_2016_2020_2


python src/apply_models.py \
  --pipeline   results/train_data/pipeline_ocean_2016_2020.pkl \
  --ridge-dir  results/model_mlp_lr/ocean_2016_2020_4/ \
  --mlp-dir    results/model_mlp_lr/ocean_2016_2020_4/ \
  --ft-dir     results/model_ft_transformer/ocean_2016_2020_4/ \
  --xgb-dir    results/model_xgb/ocean_2016_2020_4/ \
  --hybrid-dir results/model_hybrid/ocean_2016_2020_4/ \
  --input-dir  results/csv_collection \
  --input     combined_2018-09-01_all_orbits.parquet combined_2018-09-02_all_orbits.parquet combined_2019-03-13_all_orbits.parquet \
  --output    corrected.csv \
  --plot-dir  results/model_comparison/ocean_2016_2020_4


python src/apply_models.py \
  --pipeline   results/train_data/pipeline_land_2016_2020.pkl \
  --ridge-dir  results/model_mlp_lr/land_2016_2020_4/ \
  --mlp-dir    results/model_mlp_lr/land_2016_2020_4/ \
  --ft-dir     results/model_ft_transformer/land_2016_2020_4/ \
  --xgb-dir    results/model_xgb/land_2016_2020_4/ \
  --hybrid-dir results/model_hybrid/land_2016_2020_4/ \
  --input-dir  results/csv_collection \
  --input     combined_2021-12-29_all_orbits.parquet combined_2018-10-24_all_orbits.parquet combined_2020-01-15_all_orbits.parquet combined_2021-04-24_all_orbits.parquet combined_2019-07-10_all_orbits.parquet \
  --output    corrected.csv \
  --plot-dir  results/model_comparison/land_2016_2020_4



python src/apply_models.py \
  --pipeline   results/train_data/pipeline_ocean_2016_2020.pkl \
  --ft-dir     results/model_ft_transformer/ocean_2016_2020_4/ \
  --xgb-dir    results/model_xgb/ocean_2016_2020_4/ \
  --hybrid-dir results/model_hybrid/ocean_2016_2020_4/ \
  --input-dir  results/csv_collection \
  --input     combined_2018-09-01_all_orbits.parquet combined_2018-09-02_all_orbits.parquet combined_2019-03-13_all_orbits.parquet \
  --output    corrected.csv \
  --plot-dir  results/model_comparison/ocean_2016_2020_4


python src/apply_models.py \
  --pipeline   results/train_data/pipeline_land_2016_2020.pkl \
  --ft-dir     results/model_ft_transformer/land_2016_2020_4/ \
  --xgb-dir    results/model_xgb/land_2016_2020_4/ \
  --hybrid-dir results/model_hybrid/land_2016_2020_4/ \
  --input-dir  results/csv_collection \
  --input     combined_2021-12-29_all_orbits.parquet combined_2018-10-24_all_orbits.parquet combined_2020-01-15_all_orbits.parquet combined_2021-04-24_all_orbits.parquet combined_2019-07-10_all_orbits.parquet \
  --output    corrected.csv \
  --plot-dir  results/model_comparison/land_2016_2020_4
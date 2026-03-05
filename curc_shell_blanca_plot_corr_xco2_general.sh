#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_spectral_anal
#SBATCH --account=blanca-airs
###SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable


module load anaconda git intel/2024.2.1 hdf5/1.14.5 zlib/1.3.1 netcdf/4.9.2 swig/4.1.1 gsl/2.8
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

# Prevent MKL/OpenBLAS from creating a thread pool in the parent process.
# fp_abs_coeff.py parallelises at the Python (ProcessPoolExecutor) level, so
# each worker should run single-threaded numpy.  Without this, fork/forkserver
# workers inherit a broken BLAS thread state → deadlock on first numpy call.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /projects/yuch8913/OCO_spectral_mitigation

# python src/plot_corrected_xco2.py \
#     --plot-data   results/model_comparison/ocean_2017_2020_2020-05-17/plot_data.csv \
#     --tccon       data/TCCON/ra20150301_20200718.public.qc.nc \
#     --results-h5 results/results_2020-01-01.h5 \
#     --output-dir  results/model_comparison/ocean_2017_2020_2020-05-17/ \
#     --modis-auto \
#     --lon-range   53.67 56.35 \
#     --lat-range   -24.44 -14.98 \
#     --date-plot  2020-05-17 \
#     --vmin 400 --vmax 415

# python src/plot_corrected_xco2.py \
#     --plot-data   results/model_comparison/land_2017_2020_2018-10-24/plot_data.csv \
#     --tccon       data/TCCON/bu20170303_20250221.public.qc.nc \
#     --results-h5 results/results_2018-10-24.h5 \
#     --output-dir  results/model_comparison/land_2017_2020_2018-10-24/ \
#     --modis-auto \
#     --lon-range   120.48  120.87 \
#     --lat-range   18.27 18.61 \
#     --date-plot  2018-10-24 \
#     --vmin 405 --vmax 410

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/ocean_2016_2020/combined_2020-04-15_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/ra20150301_20200718.public.qc.nc \
    --results-h5  results/results_2020-04-15.h5 \
    --output-dir  results/model_comparison/ocean_2016_2020/combined_2020-04-15_all_orbits/ \
    --modis-auto \
    --lon-range   54.98  55.72 \
    --lat-range   -22.71 -20.32 \
    --date-plot  2020-04-15 \
    --vmin 406 --vmax 412

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/ocean_2016_2020/combined_2020-03-30_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/bu20170303_20250221.public.qc.nc \
    --results-h5  results/results_2020-03-30.h5 \
    --output-dir  results/model_comparison/ocean_2016_2020/combined_2020-03-30_all_orbits/ \
    --modis-auto \
    --lon-range   120.50  120.85 \
    --lat-range   18.43  19.30 \
    --date-plot  2020-03-30 \
    --vmin 410 --vmax 417

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/ocean_2016_2020/combined_2020-09-06_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/bu20170303_20250221.public.qc.nc \
    --results-h5 results/results_2020-09-06.h5 \
    --output-dir  results/model_comparison/ocean_2016_2020/combined_2020-09-06_all_orbits/ \
    --modis-auto \
    --lon-range   120.40  120.75 \
    --lat-range   18.46  19.38 \
    --date-plot  2020-09-06 \
    --vmin 405 --vmax 411

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/ocean_2016_2020/combined_2018-02-21_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/bu20170303_20250221.public.qc.nc \
    --results-h5 results/results_2018-02-21.h5 \
    --output-dir  results/model_comparison/ocean_2016_2020/combined_2018-02-21_all_orbits/ \
    --modis-auto \
    --lon-range   120.24  120.78 \
    --lat-range   18.21  19.24 \
    --date-plot  2018-02-21 \
    --vmin 402 --vmax 410

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/ocean_2016_2020/combined_2018-09-01_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/bu20170303_20250221.public.qc.nc \
    --results-h5 results/results_2018-09-01.h5 \
    --output-dir  results/model_comparison/ocean_2016_2020/combined_2018-09-01_all_orbits/ \
    --modis-auto \
    --lon-range   120.33  120.80 \
    --lat-range   18.37  19.71 \
    --date-plot  2018-09-01 \
    --vmin 402 --vmax 408

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/ocean_2016_2020/combined_2018-11-29_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/bu20170303_20250221.public.qc.nc \
    --results-h5 results/results_2018-11-29.h5 \
    --output-dir  results/model_comparison/ocean_2016_2020/combined_2018-11-29_all_orbits/ \
    --modis-auto \
    --lon-range   120.65  121.15 \
    --lat-range   18.47 18.97 \
    --date-plot  2018-11-29 \
    --vmin 405 --vmax 409


python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/ocean_2016_2020/combined_2020-05-01_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/bu20170303_20250221.public.qc.nc \
    --results-h5 results/results_2020-05-01.h5 \
    --output-dir  results/model_comparison/ocean_2016_2020/combined_2020-05-01_all_orbits/ \
    --modis-auto \
    --lon-range   120.10  120.98 \
    --lat-range   18.42 21.08 \
    --date-plot  2020-05-01 \
    --vmin 412 --vmax 418

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/ocean_2016_2020/combined_2018-09-02_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/iz20140102_20230830.public.qc.nc \
    --results-h5 results/results_2018-09-02.h5 \
    --output-dir  results/model_comparison/ocean_2016_2020/combined_2018-09-02_all_orbits/ \
    --modis-auto \
    --lon-range   -16.48 -16.1 \
    --lat-range   27.64  28.64 \
    --date-plot  2018-09-02 \
    --vmin 402 --vmax 408

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/ocean_2016_2020/combined_2018-11-30_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/iz20140102_20230830.public.qc.nc \
    --results-h5 results/results_2018-11-30.h5 \
    --output-dir  results/model_comparison/ocean_2016_2020/combined_2018-11-30_all_orbits/ \
    --modis-auto \
    --lon-range   -16.43 -15.92 \
    --lat-range   27.56  28.70 \
    --date-plot  2018-11-30 \
    --vmin 406.5 --vmax 409

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/land_2016_2020/combined_2018-10-24_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/bu20170303_20250221.public.qc.nc \
    --results-h5 results/results_2018-10-24.h5 \
    --output-dir  results/model_comparison/land_2016_2020/combined_2018-10-24_all_orbits/ \
    --modis-auto \
    --lon-range   120.48  120.87 \
    --lat-range   18.27 18.61 \
    --date-plot  2018-10-24 \
    --vmin 395 --vmax 415

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/land_2016_2020/combined_2020-01-15_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/bu20170303_20250221.public.qc.nc \
    --results-h5 results/results_2020-01-15.h5 \
    --output-dir  results/model_comparison/land_2016_2020/combined_2020-01-15_all_orbits/ \
    --modis-auto \
    --lon-range   120.48  120.87 \
    --lat-range   18.27 18.61 \
    --date-plot  2020-01-15 \
    --vmin 408 --vmax 416

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/land_2016_2020/combined_2021-04-24_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/oc20110416_20251023.public.qc.nc \
    --results-h5 results/results_2021-04-24.h5 \
    --output-dir  results/model_comparison/land_2016_2020/combined_2021-04-24_all_orbits/ \
    --modis-auto \
    --lon-range   -97.86 -97.24 \
    --lat-range   35.82 37.26 \
    --date-plot  2021-04-24 \
    --vmin 412 --vmax 419

python src/plot_corrected_xco2.py \
    --plot-data   results/model_comparison/land_2016_2020/combined_2021-12-29_all_orbits/plot_data.parquet \
    --tccon       data/TCCON/oc20110416_20251023.public.qc.nc \
    --results-h5 results/results_2021-12-29.h5 \
    --output-dir  results/model_comparison/land_2016_2020/combined_2021-12-29_all_orbits/ \
    --modis-auto \
    --lon-range   -97.86 -97.24 \
    --lat-range   35.82 37.26 \
    --date-plot  2021-12-29 \
    --vmin 415.5 --vmax 421

    
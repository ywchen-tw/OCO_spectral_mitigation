#!/bin/env bash

#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=arcsix-lrt_simulation
#SBATCH --account=blanca-airs
#SBATCH --partition=blanca-airs
#SBATCH --qos=blanca-airs


module load anaconda intel/2022.1.2 hdf5/1.10.1 zlib/1.2.11 netcdf/4.8.1 swig/4.1.1 gsl/2.7
conda activate data


cd /projects/yuch8913/OCO_spectral_mitigation


python workspace/demo_combined.py --date 2020-01-08

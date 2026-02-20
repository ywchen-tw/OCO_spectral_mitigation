#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=oco_spectral_anal
#SBATCH --account=blanca-airs
#SBATCH --partition=blanca-airs
#SBATCH --qos=blanca-airs


module load anaconda intel/2022.1.2 hdf5/1.10.1 zlib/1.2.11 netcdf/4.8.1 swig/4.1.1 gsl/2.7
conda activate data


cd /projects/yuch8913/OCO_spectral_mitigation

# ============================================================================
# Option 1: Loop with year, month, day (ACTIVE)
# ============================================================================
# Specify year, month, and day ranges
start_year=2020
end_year=2020
start_month=1
end_month=3
start_day=1
end_day=1

# Loop through year, month, day
for year in $(seq $start_year $end_year); do
    for month in $(seq $start_month $end_month); do
        for day in $(seq $start_day $end_day); do
            # Format date as YYYY-MM-DD with zero-padding
            date=$(printf "%04d-%02d-%02d" $year $month $day)
            echo "Processing date: $date"
            python workspace/demo_combined.py --date "$date" --delete-modis
            if [ $? -ne 0 ]; then
                echo "Failed to process date: $date"
            else
                echo "Successfully processed: $date"
                python src/oco_fp_spec_anal.py --date "$date" --delete-ocofiles
            fi
            echo ""
        done
    done
done



# ============================================================================
# Option 2: Hard-coded dates (uncomment to use)
# ============================================================================
# Define dates to process (modify as needed)
# dates=(
#     "2018-10-18"
#     "2020-01-04"
#     "2020-01-08"
# )
#
# # Loop through each date
# for date in "${dates[@]}"; do
#     echo "Processing date: $date"
#     python workspace/demo_combined.py --date "$date" --delete-modis
#     if [ $? -ne 0 ]; then
#         echo "Failed to process date: $date"
#     else
#         echo "Successfully processed: $date"
#     fi
#     echo ""
# done
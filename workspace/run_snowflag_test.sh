#!/usr/bin/env bash
# Local 3-arm snow_flag test (land, 2020, date_kfold fold 0).
#   A nosnow   : full_contam,      snow EXCLUDED  (status-quo baseline, retrained on 2020)
#   B snowdata : full_contam,      snow INCLUDED  (isolates effect of added snow DATA)
#   C snowflag : full_contam_snow, snow INCLUDED  (treatment: + snow_flag feature)
# B vs C isolates the FEATURE; A anchors the production filter.  3 members for a fast pass.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

DATA=results/csv_collection/combined_2020_dates.parquet
COMMON=(--sfc_type 1 --loss beta_nll --beta 1.0 --n_members 3 --batch_size 8192
        --mondrian_col cld_dist_km --val_split date_kfold --n_folds 5 --fold 0 --data "$DATA")

echo "########## ARM A (nosnow / full_contam / snow excluded) ##########"
python -m models.deep_ensemble "${COMMON[@]}" \
    --feature_set full_contam --suffix de_land_fc_nosnow_f0

echo "########## ARM B (snowdata / full_contam / snow included) ##########"
python -m models.deep_ensemble "${COMMON[@]}" --include_snow \
    --feature_set full_contam --suffix de_land_fc_snowdata_f0

echo "########## ARM C (snowflag / full_contam_snow / snow included) ##########"
python -m models.deep_ensemble "${COMMON[@]}" --include_snow \
    --feature_set full_contam_snow --suffix de_land_fc_snowflag_f0

echo "########## ALL ARMS DONE ##########"

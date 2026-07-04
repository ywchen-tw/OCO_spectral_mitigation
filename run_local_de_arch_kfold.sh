#!/usr/bin/env bash
# ── Local DE architecture sweep, multi-date 2020 + date_kfold ─────────────────
# Same three arms as run_local_de_arch_test.sh, but on the 15-date 2020 ocean
# concatenation (1.39M rows) with --val_split date_kfold so held-out DATES are
# unseen in training — removes the single-date random-split overfit artifact.
# Folds 0-2 per arm (3 held-out dates each); averaged in the summary.
set -euo pipefail
cd /Users/yuch8913/programming/oco_fp_analysis
export PYTHONPATH=src:${PYTHONPATH:-}

DATA=results/csv_collection/combined_2020_ocean_multidate.parquet
NFOLDS=5

run () {
  local suffix=$1 dims=$2 fold=$3
  python -m models.deep_ensemble --sfc_type 0 --suffix "$suffix" \
      --data "$DATA" \
      --profile-pca --feature_set full \
      --hidden_dims "$dims" \
      --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
      --norm layer --dropout 0.1 \
      --near_cloud_target 0.98 --mondrian_col cld_dist_km \
      --val_split date_kfold --n_folds ${NFOLDS} --fold ${fold} \
      --seed 42 2>&1 | tee "log_kf_${suffix}.txt"
}

for F in 0 1 2; do
  echo "############### FOLD $F ###############"
  run "kf_6432_f${F}"    64,32     $F
  run "kf_1286432_f${F}" 128,64,32 $F
  run "kf_646432_f${F}"  64,64,32  $F
done

echo
echo "############## SUMMARY (de_split_date_kfold R²) ##############"
grep -H "de_split_date_kfold" log_kf_*.txt | grep RMSE || true

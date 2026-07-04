#!/usr/bin/env bash
# ── Local DE architecture sweep (2020 data) ───────────────────────────────────
# Compares GaussianMLP hidden-layer geometries, holding EVERYTHING else at the
# production config: ocean, full feature set + profile-PCA, layer-norm + dropout
# 0.1, beta_nll(beta=1.0), M=5, near-cloud target 0.98.  ONLY --hidden_dims varies.
#
#   arm A  64,32       current 2-layer production arch (baseline)
#   arm B  128,64,32   wider+deeper 3-layer
#   arm C  64,64,32    same width, extra 3rd layer
#
# Single local 2020 date (combined_2020-02-01_all_orbits.parquet, the macOS
# default) with a fixed random split so the three arms see the identical
# train/held partition — a clean capacity A/B, not a data comparison.
set -euo pipefail
cd /Users/yuch8913/programming/oco_fp_analysis
export PYTHONPATH=src:${PYTHONPATH:-}

DATA=results/csv_collection/combined_2020-02-01_all_orbits.parquet

run_arm () {
  local suffix=$1 dims=$2
  echo "==================================================================="
  echo ">>> ARM $suffix  hidden_dims=$dims"
  echo "==================================================================="
  python -m models.deep_ensemble --sfc_type 0 --suffix "$suffix" \
      --data "$DATA" \
      --profile-pca --feature_set full \
      --hidden_dims "$dims" \
      --loss beta_nll --beta 1.0 --n_members 5 --batch_size 8192 \
      --norm layer --dropout 0.1 \
      --near_cloud_target 0.98 --mondrian_col cld_dist_km \
      --val_split random --seed 42 2>&1 | tee "log_arch_${suffix}.txt"
}

run_arm de_arch_6432    64,32
run_arm de_arch_1286432 128,64,32
run_arm de_arch_646432  64,64,32

echo
echo "############## SUMMARY (de_split_random R²) ##############"
grep -H "de_split_random" log_arch_de_arch_*.txt || true

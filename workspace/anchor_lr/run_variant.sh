#!/usr/bin/env bash
# run_variant.sh — fit → apply → TCCON report → scatter diagnostic for one term set.
#   bash workspace/anchor_lr/run_variant.sh <term-set> [perfp]   (b11 | nocontam | dp_only | …)
#   'perfp' → a separate regression per footprint (variant name <term-set>_perfp)
# Env: conda er3t_v11 (or any env with torch/xgboost/pyarrow/netCDF4); run from repo root.
set -euo pipefail
V="$1"; MODE="${2:-}"
PY="${PY:-$HOME/miniforge3/envs/er3t_v11/bin/python}"
export PYTHONPATH=src:workspace
TREE=results/model_comparison/deep_ensemble/de_prof_reg_mix_raw${RUN_SUFFIX:-_v2}
FIT_ARGS=()
if [[ "$MODE" == perfp ]]; then
  FIT_ARGS=(--per-footprint); VAR="${V}_perfp"; TAG="_${V}_perfp"
elif [[ "$V" == b11 ]]; then VAR=""; TAG=""
else VAR="$V"; TAG="_$V"; fi
LOG=results/anchor_lr/variant_${VAR:-b11}.log
{
  echo "=== fit $V $MODE"
  $PY workspace/anchor_lr/fit_anchor_lr.py --term-set "$V" ${FIT_ARGS[@]+"${FIT_ARGS[@]}"}
  echo "=== apply $V"
  $PY workspace/anchor_lr/apply_anchor.py --variant "$VAR" --force
  echo "=== report $V"
  $PY workspace/tccon_comparison_report.py \
      --script curc_shell_blanca_plot_corr_xco2_deepens.sh \
      --out-base "$TREE" --output-dir "$TREE/anchor_lr${TAG}" \
      --corr-col "anchor_lr${TAG}_corrected_xco2" --fname-suffix "_anchor_lr${TAG}_r100km" \
      --exclude-sites ny --radius-km 100 --window-min 60 --cld-edges 0,10,inf --ak-harmonize
  echo "=== diagnostic $V"
  $PY workspace/anchor_lr/diagnose_increment_scatter.py --variant "$VAR"
} > "$LOG" 2>&1
echo "done ${VAR:-b11} → $LOG"

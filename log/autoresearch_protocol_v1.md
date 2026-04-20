# Autoresearch Evaluation Protocol v1 (Locked)

Version: v1
Effective date: 2026-04-14
Scope: All autonomous or semi-autonomous experiments for OCO bias-correction models (MLP, FT, Hybrid, XGB).

---

## 1) Protocol intent

This protocol defines a fixed comparison contract so experiment outcomes are scientifically comparable and reproducible.
Any run that does not conform is **non-comparable** and cannot be promoted.

---

## 2) Dataset and filtering contract

- Input data source remains the existing project source configured in each training script.
- Surface filter (`sfc_type`) must be explicit and logged.
- Snow filtering policy is fixed: `snow_flag == 0`.
- Feature transformation must use `FeaturePipeline` and be logged with scaler/PCA settings.

### Required logged fields
- data source path
- sfc_type
- snow filtering rule
- scaler mode
- pca_augment flag
- pipeline artifact path/hash

---

## 3) Split policy (locked)

- Primary split: `train_test_split(test_size=0.2, random_state=42)`
- Stratification:
  - If PC1 index exists (`pipeline.pc1_col_idx`), stratify by PC1 quintiles.
  - Else no stratification.

This split policy is immutable for v1 leaderboard comparisons.

---

## 4) Optimization targets and guardrails

## 4.1 Primary objective

- **Primary metric for model selection**: test-set RMSE on target (`xco2_bc_anomaly`) unless model-specific protocol states otherwise.
- Where scripts currently print R² only, RMSE and MAE must still be included in run summary records.

## 4.2 Required secondary metrics

- MAE (test set)
- R² (test set)
- PC1-stratified RMSE/MAE (if PC1 available)
- Runtime seconds
- Peak memory (if available)

## 4.3 Guardrail policy

A candidate cannot be promoted if any applies:
- Runtime increases > 50% with negligible primary gain.
- PC1 worst-bin RMSE degrades by > 5% relative to incumbent.
- Numerical instability/crash behavior increases.

---

## 5) Reproducibility requirements

Each run must persist:
- fully resolved run config (JSON)
- script version identifier (git commit hash preferred)
- random seeds used
- output directory and artifact paths
- status (`success`, `crash`, `discard`, `keep`)

---

## 6) Promotion semantics (v1)

- Keep if primary metric improves by epsilon or more.
- Keep if primary metric ties within epsilon and complexity/runtime improves.
- Discard otherwise.

Default epsilon for v1: `1e-4` in primary metric units.

---

## 7) Non-comparable changes

The following automatically mark run as non-comparable unless baseline is rerun under same changes:
- split policy change
- target variable definition change
- feature pipeline semantics change
- evaluation subset change

---

## 8) Protocol changes

Any change to this protocol requires:
1. version bump (`v1` -> `v2`)
2. explicit changelog entry
3. fresh baseline under new protocol

---

## 9) Immediate implementation mapping

Phase-1 implementation in `mlp_lr_models.py` should:
- accept config overrides without changing default behavior
- save resolved run config per run
- preserve this protocol’s split and seed defaults unless explicitly overridden and logged

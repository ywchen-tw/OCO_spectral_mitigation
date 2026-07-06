# Autoresearch experimentation loop (historical, consolidated 2026-07-06)

> Consolidated from four docs written 2026-04-14/15: `autoresearch_protocol_v1.md`,
> `autoresearch_phase_plan.md`, `autoresearch_implementation.md`, `autoresearch_usage_guide.md`.
> The tooling was fully implemented and still exists in `src/`, but the production model
> (per-surface deep ensemble, `de_*_beta_nll_prof_reg`) was selected via the near-cloud DE
> experiments + TCCON validation (`log/near_cloud_xco2_plan.md`), not this loop. Kept for reference.

---

# Part 1 — Evaluation protocol (v1, locked)

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

---

# Part 2 — Phase plan

# Autoresearch-Inspired Phase Plan for Model Structure + Hyperparameter Optimization

Date: 2026-04-14
Scope: Integrate an autoresearch-style experimentation loop into the existing OCO model training stack (MLP, FT-Transformer, Hybrid, XGBoost) without rewriting core science logic.

---

## Why this plan

The external reference project (`miolini/autoresearch-macos`) is optimized for a single-file LLM training loop.
Our codebase is multi-model and science-evaluation heavy, so the right adaptation is:

1. Keep current model implementations as source of truth.
2. Add a reproducible orchestration layer over them.
3. Enforce fixed evaluation protocol + keep/discard experiment governance.

This preserves scientific traceability while enabling autonomous tuning.

---

## Phase 0 — Protocol Lock (Before Any Automation)

### Goal
Define a single immutable evaluation protocol so all future experiments are directly comparable.

### Deliverables
- Canonical split policy (including random seeds and stratification policy).
- Primary optimization metric (per model family).
- Secondary guardrail metrics (e.g., regime-level error, calibration, stability).
- Runtime and memory reporting requirements.
- “Regression budget” definition (how much degradation in non-primary metrics is tolerated).

### Acceptance criteria
- A one-page protocol spec exists and is versioned in repo.
- Two consecutive baseline runs produce metric variance within expected tolerance.
- Team agrees no experiment can bypass this protocol.

### Risk notes
- If this phase is skipped, later gains may be artifacts of split drift or metric drift.

---

## Phase 1 — Configuration Surfaces for Existing Trainers

### Goal
Expose architecture and training knobs as explicit config inputs instead of hardcoded values.

### In-scope scripts
- `src/mlp_lr_models.py`
- `src/models_transformer.py`
- `src/models_hybrid.py`
- `src/xgb_models.py`

### Deliverables
- Structured config schema (YAML/JSON/dataclass) with defaults matching current behavior.
- Trainer entry points accept external config overrides.
- Config snapshot saved with each run artifact.

### Candidate knob groups
- **MLP**: hidden width, depth, dropout, lr, weight decay, batch size, max epochs, patience.
- **FT**: d_token, n_heads, n_layers, d_ff, lr schedule params, batch size, loss weighting.
- **Hybrid**: branch widths, fusion_dim, loss component weights, learning-rate policy.
- **XGB**: max_depth, learning_rate, n_estimators, subsample, colsample_bytree, regularization.

### Acceptance criteria
- Running each script with no config reproduces current baseline.
- Running with override config changes only intended hyperparameters.
- Saved output directory contains both metrics and full resolved config.

### Risk notes
- Over-parameterizing too early increases search noise. Start with a narrow, high-impact knob set.

---

## Phase 2 — Unified Results Contract + Ledger

### Goal
Create one consistent machine-readable run output, independent of model family.

### Deliverables
- Unified per-run summary schema (JSON):
  - run_id, timestamp, model_family, config_hash
  - primary_metric, secondary_metrics
  - runtime_seconds, peak_memory, trainable_params (if applicable)
  - status (keep/discard/crash)
  - artifact paths
- Central append-only ledger (`results.tsv` and/or `results.jsonl`).
- Utility that parses model-specific outputs and writes standardized records.

### Acceptance criteria
- MLP/FT/Hybrid/XGB runs all generate valid summary records.
- A simple script can rank runs by primary metric and filter by guardrails.
- Crash runs are recorded consistently with reason metadata.

### Risk notes
- Inconsistent metric naming will silently corrupt ranking logic.

---

## Phase 3 — Keep/Discard Governance (Autoresearch Core Pattern)

### Goal
Implement deterministic promotion rules so autonomous runs cannot “drift” on weak improvements.

### Deliverables
- Promotion policy:
  - Keep if primary metric improves by at least epsilon.
  - Keep if primary metric tie but complexity decreases or guardrails improve.
  - Discard if guardrails regress beyond thresholds.
- Complexity score definition (e.g., params, runtime, memory, code-surface flags).
- Automatic status assignment and rationale string per run.

### Acceptance criteria
- Same inputs always produce same keep/discard decision.
- Promotion rationale is logged in human-readable form.
- Governance policy can be unit-tested without full training.

### Risk notes
- Noisy metrics can trigger oscillation around borderline thresholds; use minimum-effect-size epsilon.

---

## Phase 4 — Search Strategy Layer (Safe Autonomy First)

### Goal
Add exploration methods over config space while preserving reproducibility.

### Strategy order (recommended)
1. **Constrained random search** (fast baseline, low coupling).
2. **TPE/Bayesian search** (after baseline landscape known).
3. **Scheduled local mutation** around best-known configs.

### Deliverables
- Search-space definitions per model family (with hard bounds).
- Sampler/orchestrator that launches one run, ingests summary, updates ledger.
- Seed control for reproducibility.

### Acceptance criteria
- N-run dry test executes end-to-end without manual intervention.
- Best-so-far config is tracked and recoverable at all times.
- Search process can be resumed after interruption.

### Risk notes
- Broad spaces waste budget. Use staged widening after first 20–30 runs.

---

## Phase 5 — Time-Budgeted Autonomous Loop (Overnight Mode)

### Goal
Adopt the fixed-budget loop concept from autoresearch for fair throughput-oriented optimization.

### Deliverables
- Per-run wall-clock budget mode (example: short fixed budget for coarse search).
- Loop controller with stop conditions:
  - max runs
  - max wall-clock hours
  - max consecutive crashes
  - max no-improvement streak
- Periodic checkpointing of best config + ledger snapshot.

### Acceptance criteria
- Overnight loop can run unattended and stop safely on policy triggers.
- Next-day summary report includes:
  - top-k runs
  - promoted winner
  - failure taxonomy
  - suggested next search region

### Risk notes
- Fixed-time budgets are ideal for throughput optimization, but may bias against slower-converging architectures. Keep a separate “full-train confirmation” phase.

---

## Phase 6 — Confirmation, Generalization, and Deployment Gate

### Goal
Prevent overfitting to one split/date regime before adopting new defaults.

### Deliverables
- Confirmatory retraining of top candidates with full epoch budget.
- Cross-date / cross-regime validation comparison against incumbent baseline.
- Model card summary for promoted config:
  - where it improves
  - where it regresses
  - uncertainty/calibration impact

### Acceptance criteria
- Candidate beats incumbent on primary metric and respects all guardrails on confirmation runs.
- Performance is stable across agreed date/regime slices.
- Promotion decision documented and reproducible from ledger.

### Risk notes
- Without this gate, autonomous search can optimize accidental artifacts in one subset.

---

## Recommended Initial Search Space (Narrow Start)

### MLP (Phase 4 initial)
- depth: {3, 4, 5}
- width: {128, 192, 256, 384}
- lr: [1e-4, 2e-3] (log scale)
- weight_decay: [1e-6, 3e-3] (log scale)
- batch_size: {2048, 4096, 8192}
- patience: {20, 30, 40}

### FT-Transformer (Phase 4 initial)
- d_token: {64, 96, 128, 160}
- n_layers: {3, 4, 5}
- n_heads: constrained by d_token divisibility
- d_ff ratio: {2x, 3x, 4x of d_token}
- lr max: [3e-4, 2e-3] (log scale)
- range_loss_weight: {0.0, 0.01, 0.03, 0.05}

### Hybrid (Phase 4 initial)
- mlp_hidden: {128, 192, 256, 384}
- fusion_dim: {64, 96, 128, 192}
- n_layers: {3, 4, 5}
- max_lr: [1e-4, 8e-4] (log scale)
- range_loss_weight: {0.0, 0.01, 0.03}

### XGBoost (Phase 4 initial)
- max_depth: {4, 6, 8, 10}
- learning_rate: [0.01, 0.2] (log scale)
- n_estimators: {300, 600, 900, 1200}
- subsample: [0.6, 1.0]
- colsample_bytree: [0.5, 1.0]
- reg_lambda: [1e-3, 10] (log scale)

---

## Minimal Execution Roadmap (Order of Implementation)

1. Implement Phase 0 protocol doc and lock metrics.
2. Implement Phase 1 config surfaces in MLP first.
3. Implement Phase 2 summary schema + ledger writer.
4. Implement Phase 3 keep/discard policy utility.
5. Add Phase 4 search sampler for MLP.
6. Extend same framework to FT, Hybrid, XGB.
7. Enable Phase 5 overnight controller.
8. Enforce Phase 6 promotion gate before changing defaults.

---

## Definition of Done for the Full Program

The autoresearch adaptation is complete when:
- Every model family can be launched from config only.
- Every run emits standardized metrics + metadata.
- Keep/discard decisions are deterministic and logged.
- Overnight autonomous search can run safely unattended.
- Promoted configurations pass confirmation/generalization gates.

---

## Notes for Future Iteration

After stabilization, consider:
- Multi-objective ranking (Pareto front of error vs calibration vs runtime).
- Meta-learning warm starts from previous date windows.
- Active search-space pruning from feature-importance and sensitivity traces.

---

# Part 3 — Implementation status

# Autoresearch Adaptation — Implementation Tracker

Date started: 2026-04-14
Owner: Copilot + Steven
Reference plan: `log/autoresearch_phase_plan.md`

## Phase Status

| Phase | Name | Status | Completed on | Notes |
|---|---|---|---|---|
| 0 | Protocol Lock | Completed | 2026-04-14 | `log/autoresearch_protocol_v1.md` created and locked |
| 1 | Config Surfaces | Completed | 2026-04-14 | Implemented in `src/mlp_lr_models.py` with JSON config + resolved-config artifact |
| 2 | Unified Results Contract + Ledger | Completed | 2026-04-14 | Added `src/experiment_tracking.py` and MLP success/crash summary logging |
| 3 | Keep/Discard Governance | Completed | 2026-04-14 | Added `src/promotion_policy.py` and `tests/test_promotion_policy.py` |
| 4 | Search Strategy Layer | Completed | 2026-04-14 | Added `src/autoresearch_search.py` and search tests |
| 5 | Time-Budgeted Autonomous Loop | Completed | 2026-04-14 | Added `src/autoresearch_loop.py` and loop tests |
| 6 | Confirmation + Promotion Gate | Completed | 2026-04-14 | Added `src/autoresearch_confirm.py` and confirmation tests |

## Change Log

- 2026-04-14: Initialized implementation tracker.
- 2026-04-14: Completed Phase 0 (Protocol Lock) with `log/autoresearch_protocol_v1.md`.
- 2026-04-14: Completed Phase 1 (MLP config surface) in `src/mlp_lr_models.py`.
- 2026-04-14: Added `--config` support, configurable MLP/Ridge/split/loss/permutation knobs, and persisted `mlp_run_config.json` per run.
- 2026-04-14: Completed Phase 2 (Unified Results Contract + Ledger) with `src/experiment_tracking.py` and run-summary writing from `src/mlp_lr_models.py`.
- 2026-04-14: Completed Phase 3 (Keep/Discard Governance) with `src/promotion_policy.py` and a unit test in `tests/test_promotion_policy.py`.
- 2026-04-14: Completed Phase 4 (Search Strategy Layer) with `src/autoresearch_search.py` and `tests/test_autoresearch_search.py`.
- 2026-04-14: Completed Phase 5 (Time-Budgeted Autonomous Loop) with `src/autoresearch_loop.py` and `tests/test_autoresearch_loop.py`.
- 2026-04-14: Tracker refreshed after Phase 5; Phase 6 was then implemented as the final planned phase.
- 2026-04-14: Completed Phase 6 (Confirmation + Promotion Gate) with `src/autoresearch_confirm.py` and `tests/test_autoresearch_confirm.py`.
- 2026-04-14: Fixed MLP runtime bug by preserving validation arrays before cleanup in `src/mlp_lr_models.py`.

## How to use the implemented features

The examples below use `src/mlp_lr_models.py` as the entry point, because it is the first trainer that now supports the full autoresearch flow.

### 1) Run a single MLP experiment with a config file

Create a JSON file such as `results/model_mlp_lr/demo/mlp_config.json`:

```json
{
	"ridge": {"alpha": 0.5},
	"mlp": {
		"batch_size": 4096,
		"lr": 0.001,
		"weight_decay": 0.0005,
		"max_epochs": 300,
		"patience": 25
	},
	"loss": {
		"huber_delta": 1.0,
		"pred_l1_weight": 0.05
	},
	"importance": {
		"perm_max_rows": 3000,
		"perm_repeats": 5
	}
}
```

Then run:

```bash
python src/mlp_lr_models.py --suffix demo --config results/model_mlp_lr/demo/mlp_config.json
```

What you get:
- `results/model_mlp_lr/demo/mlp_run_config.json` — resolved config used by the run
- `results/model_mlp_lr/demo/run_summary.json` — unified per-run summary
- `results/autoresearch_ledger.tsv` — append-only ledger row
- existing diagnostic plots and adapter artifacts

### 2) Launch a small search over MLP hyperparameters

```bash
python src/autoresearch_search.py --family mlp_lr --n-trials 3 --seed 42
```

This will:
- sample bounded MLP configs
- write each sampled config to the run output directory
- execute `src/mlp_lr_models.py`
- collect run summaries and keep an autoresearch search manifest

### 3) Run the time-budgeted autonomous loop

```bash
python src/autoresearch_loop.py --family mlp_lr --max-trials 12 --max-duration-seconds 300
```

This will:
- repeatedly sample configs
- stop on time budget, crash streak, or no-improvement streak
- reuse the promotion policy to decide whether the incumbent changes
- write a loop manifest under `results/autoresearch_loop/`

### 4) Confirm a candidate against an incumbent

Use two saved run summaries:

```bash
python src/autoresearch_confirm.py \
	--candidate-summary results/model_mlp_lr/<candidate>/run_summary.json \
	--incumbent-summary results/model_mlp_lr/<incumbent>/run_summary.json \
	--family mlp_lr
```

This will:
- rerun candidate and incumbent configs multiple times
- measure stability and runtime/memory guardrails
- write a promotion snapshot manifest for the final keep/discard decision

### 5) Recommended workflow order

1. Baseline `src/mlp_lr_models.py` with no overrides.
2. Tune one config at a time through `--config`.
3. Use `src/autoresearch_search.py` for bounded exploration.
4. Use `src/autoresearch_loop.py` for unattended overnight search.
5. Use `src/autoresearch_confirm.py` before promoting a new default.

---

# Part 4 — Usage guide

# Autoresearch Usage Guide (MLP-first)

This guide records how to use the implemented autoresearch workflow in this repository.

Scope covered:
- Config-driven training run (`mlp_lr_models.py`)
- Search (`autoresearch_search.py`)
- Time-budgeted loop (`autoresearch_loop.py`)
- Confirmation + promotion gate (`autoresearch_confirm.py`)
- How to interpret outputs and decide promotion

---

## 0) Prerequisites

Run from repository root:

```bash
cd /Users/yuch8913/programming/oco_fp_analysis
```

Use your configured Python environment (conda/venv) consistently.

---

## 1) Single MLP run (config-driven baseline)

### 1.1 Prepare a config file

Example path:

# python [mlp_lr_models.py] --suffix demo

```bash
results/model_mlp_lr/demo/mlp_config.json
```

If you already have:

```bash
results/model_mlp_lr/demo/mlp_run_config.json
```

you can copy it to `mlp_config.json` and edit selected fields.

### 1.2 Run

```bash
python src/mlp_lr_models.py --suffix demo --config results/model_mlp_lr/demo/mlp_config.json
```

### 1.3 Outputs to check

- `results/model_mlp_lr/demo/mlp_run_config.json` (resolved config)
- `results/model_mlp_lr/demo/run_summary.json` (standard summary)
- `results/autoresearch_ledger.tsv` (append-only ledger)

### 1.4 Important path rule for `--config`

`--config` must point to a JSON file, not a directory.

Wrong:

```bash
--config results/
```

Correct pattern:

```bash
--config results/model_mlp_lr/demo/mlp_config.json
```

---

## 1b) Config-driven runs for Transformer / Hybrid / XGBoost

All three model scripts now support `--config` and save a resolved run config in each run directory.

### 1b.1 Transformer (`models_transformer.py`)

Generate a default resolved config (one baseline run):

```bash
python src/models_transformer.py --suffix demo_ft
```

Generated file:

```bash
results/model_ft_transformer/demo_ft/ft_run_config.json
```

Create editable config and run:

```bash
cp results/model_ft_transformer/demo_ft/ft_run_config.json \
    results/model_ft_transformer/demo_ft/ft_config.json

python src/models_transformer.py \
   --suffix demo_ft \
   --config results/model_ft_transformer/demo_ft/ft_config.json
```

### 1b.2 Hybrid (`models_hybrid.py`)

Generate a default resolved config:

```bash
python src/models_hybrid.py --suffix demo_hybrid
```

Generated file:

```bash
results/model_hybrid/demo_hybrid/hybrid_run_config.json
```

Create editable config and run:

```bash
cp results/model_hybrid/demo_hybrid/hybrid_run_config.json \
    results/model_hybrid/demo_hybrid/hybrid_config.json

python src/models_hybrid.py \
   --suffix demo_hybrid \
   --config results/model_hybrid/demo_hybrid/hybrid_config.json
```

### 1b.3 XGBoost (`xgb_models.py`)

Generate a default resolved config:

```bash
python src/xgb_models.py --suffix demo_xgb
```

Generated file:

```bash
results/model_xgb/demo_xgb/xgb_run_config.json
```

Create editable config and run:

```bash
cp results/model_xgb/demo_xgb/xgb_run_config.json \
    results/model_xgb/demo_xgb/xgb_config.json

python src/xgb_models.py \
   --suffix demo_xgb \
   --config results/model_xgb/demo_xgb/xgb_config.json
```

---

## 1c) `--family` values and commands (MLP / Transformer / Hybrid / XGBoost)

If you meant the `--family` argument (sometimes typed as `--famalify` by mistake), use:

- Transformer: `ft_transformer`
- Hybrid: `hybrid`
- XGBoost: `xgb`
- MLP (original): `mlp_lr`

Use Sections 2–6 for the actual commands. This section is only the family-name reference.

### 1c.1 Current support note

The orchestration workflow remains MLP-first in maturity, but `autoresearch_search.py` can now execute all four families (`mlp_lr`, `ft_transformer`, `hybrid`, `xgb`).

For non-MLP families, if a trainer does not emit `run_summary.json`, the search runner writes a fallback summary so trials can be tracked consistently.

---

## 2) Hyperparameter search

Run bounded random search by model family:

### 2.1 MLP (`mlp_lr`)

```bash
python src/autoresearch_search.py --family mlp_lr --n-trials 3 --seed 42
```

### 2.2 Transformer (`ft_transformer`)

```bash
python src/autoresearch_search.py --family ft_transformer --n-trials 3 --seed 42
```

### 2.3 Hybrid (`hybrid`)

```bash
python src/autoresearch_search.py --family hybrid --n-trials 3 --seed 42
```

### 2.4 XGBoost (`xgb`)

```bash
python src/autoresearch_search.py --family xgb --n-trials 3 --seed 42
```

Common outputs:
- Per-trial run folders under `results/model_<family>/autoresearch_*`
- Search manifest under `results/autoresearch_search/`

---

## 3) Time-budgeted autonomous loop

Run unattended search with stop guards by family:

### 3.1 MLP (`mlp_lr`)

```bash
python src/autoresearch_loop.py \
   --family mlp_lr \
   --max-trials 12 \
   --max-duration-seconds 300 \
   --max-consecutive-crashes 3 \
   --max-consecutive-non-improve 5
```

### 3.2 Transformer (`ft_transformer`)

```bash
python src/autoresearch_loop.py \
   --family ft_transformer \
   --max-trials 12 \
   --max-duration-seconds 300 \
   --max-consecutive-crashes 3 \
   --max-consecutive-non-improve 5
```

### 3.3 Hybrid (`hybrid`)

```bash
python src/autoresearch_loop.py \
   --family hybrid \
   --max-trials 12 \
   --max-duration-seconds 300 \
   --max-consecutive-crashes 3 \
   --max-consecutive-non-improve 5
```

### 3.4 XGBoost (`xgb`)

```bash
python src/autoresearch_loop.py \
   --family xgb \
   --max-trials 12 \
   --max-duration-seconds 300 \
   --max-consecutive-crashes 3 \
   --max-consecutive-non-improve 5
```

This stops when any configured guardrail is reached.

Outputs:
- Loop manifest under `results/autoresearch_loop/`
- Usual trial summaries in `results/model_<family>/`

---

## 4) Choose candidate and incumbent

List available run summaries and primary metric for each family:

### 4.1 MLP (`mlp_lr`)

```bash
python - <<'PY'
import json,glob
for p in sorted(glob.glob('results/model_mlp_lr/**/run_summary.json', recursive=True)):
    d=json.load(open(p))
    print(f"{p}\t{d.get('primary_metric_name')}={d.get('primary_metric_value'):.6f}")
PY
```

### 4.2 Transformer (`ft_transformer`)

```bash
python - <<'PY'
import json,glob
for p in sorted(glob.glob('results/model_ft_transformer/**/run_summary.json', recursive=True)):
   d=json.load(open(p))
   print(f"{p}\t{d.get('primary_metric_name')}={d.get('primary_metric_value'):.6f}")
PY
```

### 4.3 Hybrid (`hybrid`)

```bash
python - <<'PY'
import json,glob
for p in sorted(glob.glob('results/model_hybrid/**/run_summary.json', recursive=True)):
   d=json.load(open(p))
   print(f"{p}\t{d.get('primary_metric_name')}={d.get('primary_metric_value'):.6f}")
PY
```

### 4.4 XGBoost (`xgb`)

```bash
python - <<'PY'
import json,glob
for p in sorted(glob.glob('results/model_xgb/**/run_summary.json', recursive=True)):
   d=json.load(open(p))
   print(f"{p}\t{d.get('primary_metric_name')}={d.get('primary_metric_value'):.6f}")
PY
```


Pick:
- Candidate: better (lower) primary metric for that family
- Incumbent: the Step 1 baseline run for the same family, using the current running hyperparameters

Important: candidate and incumbent must be different files.

---

## 5) Confirmation + promotion gate

Run confirmatory reruns before promotion (candidate/incumbent must be from the same family):

### 5.1 MLP (`mlp_lr`)

```bash
python src/autoresearch_confirm.py \
  --candidate-summary results/model_mlp_lr/autoresearch_seed42_t003/run_summary.json \
  --incumbent-summary results/model_mlp_lr/demo/run_summary.json \
  --family mlp_lr \
  --repeats 3 \
  --epsilon 1e-4
```

### 5.2 Transformer (`ft_transformer`)

```bash
python src/autoresearch_confirm.py \
   --candidate-summary results/model_ft_transformer/autoresearch_seed42_t001/run_summary.json \
   --incumbent-summary results/model_ft_transformer/demo_ft/run_summary.json \
   --family ft_transformer \
   --repeats 3 \
   --epsilon 1e-4
```

Use the Step 1 baseline run as the incumbent, i.e. the resolved config run that reflects the current hyperparameters you are starting from.
The `results/model_ft_transformer/demo_ft/` folder now contains `run_summary.json` and can be used directly as the FT-Transformer incumbent baseline.

### 5.3 Hybrid (`hybrid`)

```bash
python src/autoresearch_confirm.py \
   --candidate-summary results/model_hybrid/<candidate_suffix>/run_summary.json \
   --incumbent-summary results/model_hybrid/<incumbent_suffix>/run_summary.json \
   --family hybrid \
   --repeats 3 \
   --epsilon 1e-4
```

### 5.4 XGBoost (`xgb`)

```bash
python src/autoresearch_confirm.py \
   --candidate-summary results/model_xgb/<candidate_suffix>/run_summary.json \
   --incumbent-summary results/model_xgb/<incumbent_suffix>/run_summary.json \
   --family xgb \
   --repeats 3 \
   --epsilon 1e-4
```

Note: for non-MLP families, generate these `run_summary.json` files via `autoresearch_search.py` or `autoresearch_loop.py` first.

Optional stricter checks:

```bash
--max-primary-metric-std 0.01 --max-runtime-regression-frac 0.2 --max-memory-regression-frac 0.2
```

Output:
- Promotion snapshot JSON under `results/autoresearch_confirmation/<family>/`

---

## 6) How to evaluate the confirmation result

Open the latest snapshot and read:

- `decision` (`keep` or `discard`)
- `reason`
- `candidate_guardrails_ok`, `incumbent_guardrails_ok`
- `candidate_stats.mean_primary_metric` vs `incumbent_stats.mean_primary_metric`
- `candidate_stats.std_primary_metric` (stability)
- runtime/memory comparisons

### 6.1 MLP (`mlp_lr`)

```bash
python - <<'PY'
import json,glob
p=sorted(glob.glob('results/autoresearch_confirmation/mlp_lr/*.json'))[-1]
d=json.load(open(p))
print('snapshot:', p)
print('decision:', d['decision'])
print('reason:', d['reason'])
print('candidate mean/std:', d['candidate_stats']['mean_primary_metric'], d['candidate_stats']['std_primary_metric'])
print('incumbent mean/std:', d['incumbent_stats']['mean_primary_metric'], d['incumbent_stats']['std_primary_metric'])
print('guardrails candidate/incumbent:', d['candidate_guardrails_ok'], d['incumbent_guardrails_ok'])
PY
```

### 6.2 Transformer (`ft_transformer`)

```bash
python - <<'PY'
import json,glob
p=sorted(glob.glob('results/autoresearch_confirmation/ft_transformer/*.json'))[-1]
d=json.load(open(p))
print('snapshot:', p)
print('decision:', d['decision'])
print('reason:', d['reason'])
print('candidate mean/std:', d['candidate_stats']['mean_primary_metric'], d['candidate_stats']['std_primary_metric'])
print('incumbent mean/std:', d['incumbent_stats']['mean_primary_metric'], d['incumbent_stats']['std_primary_metric'])
print('guardrails candidate/incumbent:', d['candidate_guardrails_ok'], d['incumbent_guardrails_ok'])
PY
```

### 6.3 Hybrid (`hybrid`)

```bash
python - <<'PY'
import json,glob
p=sorted(glob.glob('results/autoresearch_confirmation/hybrid/*.json'))[-1]
d=json.load(open(p))
print('snapshot:', p)
print('decision:', d['decision'])
print('reason:', d['reason'])
print('candidate mean/std:', d['candidate_stats']['mean_primary_metric'], d['candidate_stats']['std_primary_metric'])
print('incumbent mean/std:', d['incumbent_stats']['mean_primary_metric'], d['incumbent_stats']['std_primary_metric'])
print('guardrails candidate/incumbent:', d['candidate_guardrails_ok'], d['incumbent_guardrails_ok'])
PY
```

### 6.4 XGBoost (`xgb`)

```bash
python - <<'PY'
import json,glob
p=sorted(glob.glob('results/autoresearch_confirmation/xgb/*.json'))[-1]
d=json.load(open(p))
print('snapshot:', p)
print('decision:', d['decision'])
print('reason:', d['reason'])
print('candidate mean/std:', d['candidate_stats']['mean_primary_metric'], d['candidate_stats']['std_primary_metric'])
print('incumbent mean/std:', d['incumbent_stats']['mean_primary_metric'], d['incumbent_stats']['std_primary_metric'])
print('guardrails candidate/incumbent:', d['candidate_guardrails_ok'], d['incumbent_guardrails_ok'])
PY
```

Quick terminal check:

```bash
python - <<'PY'
import json, glob

for family in ['mlp_lr', 'ft_transformer', 'hybrid', 'xgb']:
   matches = sorted(glob.glob(f'results/autoresearch_confirmation/{family}/*.json'))
   if not matches:
      print(f'{family}: no confirmation snapshots yet')
      continue
   p = matches[-1]
   d = json.load(open(p))
   print(f'family: {family}')
   print('snapshot:', p)
   print('decision:', d['decision'])
   print('reason:', d['reason'])
   print('candidate mean/std:', d['candidate_stats']['mean_primary_metric'], d['candidate_stats']['std_primary_metric'])
   print('incumbent mean/std:', d['incumbent_stats']['mean_primary_metric'], d['incumbent_stats']['std_primary_metric'])
   print('guardrails candidate/incumbent:', d['candidate_guardrails_ok'], d['incumbent_guardrails_ok'])
   print()
PY
```

Promotion rule of thumb:
- Promote only if `decision == keep` and guardrails pass.

---

## 7) Common failure modes

1. `FileNotFoundError` with `--config ...`
   - Ensure the config file exists exactly at the path.
   - Or run without `--config` for defaults.

2. `FileNotFoundError` with `--candidate-summary ...`
   - Do not pass literal `...`; provide real `run_summary.json` path.

3. Meaningless confirmation run
   - Do not use the same summary file for candidate and incumbent.

4. `AssertionError: embed_dim must be divisible by num_heads`
   - This happens when a Transformer/Hybrid config has incompatible `d_token` and `n_heads`.
   - `autoresearch_search.py` now repairs sampled configs automatically before launch.
   - For manual configs, ensure `d_token % n_heads == 0`.

---

## 8) Recommended daily workflow

1. Baseline run (`mlp_lr_models.py`).
2. Small bounded search (`autoresearch_search.py`).
3. If needed, autonomous loop (`autoresearch_loop.py`).
4. Confirm top candidate against incumbent (`autoresearch_confirm.py`).
5. Promote only after confirmation snapshot says `keep`.

---

## 9) Key implementation files

- `src/mlp_lr_models.py`
- `src/experiment_tracking.py`
- `src/promotion_policy.py`
- `src/autoresearch_search.py`
- `src/autoresearch_loop.py`
- `src/autoresearch_confirm.py`

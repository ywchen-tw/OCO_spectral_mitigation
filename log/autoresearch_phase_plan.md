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

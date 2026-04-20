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

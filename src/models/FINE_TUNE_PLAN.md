# Deep-Ensemble Fine-Tuning Plan (for later implementation)

Status: **planning** — written 2026-06-25. Model is production-usable as-is; this
plan is for squeezing remaining headroom, not a blocker.

## 0. Current locked-in configuration (do NOT re-litigate)
The production model is the deep ensemble in [deep_ensemble.py](deep_ensemble.py):
- **Loss**: `beta_nll`, `beta=1.0` — won a clean sweep {0,0.3,0.5,0.7,1.0,1.5,2.0};
  1.0 is the optimum (β>1 overfits, β<1 underfits the mean). See
  [loss_ablation_verdict.md](../../results/model_comparison/loss_ablation_verdict.md).
- **Architecture**: `GaussianMLP` 35→64→32→(mu, log_var), M=3 members, AdamW
  lr=1e-3, wd=1e-4, OneCycleLR, 500 epochs (Linux), batch 8192.
- **Calibration**: split + cloud-Mondrian conformal, sigma-normalized,
  `near_cloud_target=0.98` (over-covers near-cloud bins to lift the outcome-defined
  tail to ~0.90). See [tail_coverage_verdict.md](../../results/model_comparison/tail_coverage_verdict.md).
- **Validation**: `date_kfold`, 5 folds, ocean + land.

Full-scale results: ocean R²≈0.54 / near-cloud R²≈0.63; land R²≈0.46 / near 0.60.
beta_nll beats gaussian most on **land** (+0.09 near R²) and on the **tail** point
accuracy; see [deep_ensemble_fullscale_verdict.md](../../results/model_comparison/deep_ensemble_fullscale_verdict.md).

## 1. What is already known to be a dead end (do not retry)
- **student_t loss** — robustness down-weights the tail; strictly worse.
- **Asymmetric / raw (non-sigma-normalized) conformal** — worse tail coverage.
- **Geometry embeddings, fit-quality features, extra spectral features** — null
  (the contaminated tail is partly data-irreducible; kNN local Var(y) tail/bulk ≈ 5×).
- **Flat-target conformal for the y-defined tail** — provably unattainable; only
  regime-elevation (`near_cloud_target`) helps.

## 2. Tuning targets — prioritized

### Tier A — highest expected value (do these first, manual, fold-0 ocean+land)
| knob | current | try | hypothesis | links to |
|---|---|---|---|---|
| `n_members` | 3 | 5, 10 | more members → better ensemble σ → better **tail coverage** (the open problem) | tail coverage |
| hidden width | 64→32 | 128→64, 256→128 | more capacity for the near-cloud nonlinearity → near-cloud R² | accuracy |

Method: replicate the beta-sweep pattern — single fold, beta=1.0, 250 epochs, run
with vs without, compare `near_cloud(<=10km)` R² and `near&bottom_5pct` coverage
(crossed regime already in diagnostics). Decision: keep only if Δ is clearly
outside the ±0.10 fold spread.

### Tier B — second-order (only if Tier A shows headroom)
| knob | current | range |
|---|---|---|
| lr | 1e-3 | 3e-4 … 3e-3 (log) |
| weight_decay | 1e-4 | 1e-5 … 1e-3 (log) |
| epochs / patience | 500 / 50 | 300–800 / 30–80 |
| calib_frac | 0.15 | 0.10 … 0.25 |
| mondrian_bins | 10 | 6 … 20 |
| dropout (add) | none | 0 … 0.2 |

### Tier C — calibration refinements (post-hoc, cheap, no retrain)
- Sweep `near_cloud_target` ∈ {0.975, 0.98, 0.985, 0.99} on full-data parquets via
  leave-one-fold-out (already scripted in the tail experiment) — pick the
  width↔coverage point you want for deployment.
- Try **finer near-cloud bins** (more Mondrian bins below 10 km) — the tail
  concentrates in the closest bins.

## 3. Methodology guardrails (critical — easy to get wrong)
1. **Score on a fold-AVERAGED metric, never a single fold.** Fold variance is ±0.10
   in R²; a single-fold win is noise. Use ≥3 folds (or all 5) for any keep decision.
2. **Never touch the held block during search.** The calib block is carved from
   TRAIN dates inside each fold; keep it that way.
3. **Separate point-accuracy from calibration.** Conformal owns coverage, so judge
   architecture/loss changes on R² (and tail R²); judge calibration changes on
   coverage + width. Don't conflate.
4. **Watch the tail, not just global.** The headline metrics are near-cloud R² and
   `near&bottom_5pct` coverage — global R² can improve while the tail does not.
5. **Cost budget**: full-data run ≈ 40 min GPU/fold. Prototype on the 12-date
   subset (`combined_2020_dates.parquet`), confirm survivors on full 2016–2020.

## 4. Autoresearch integration (only after Tier A justifies it)
The harness in `src/search/` (search.py / loop.py / promotion.py / confirm.py)
already runs `mlp_lr`, `ft_transformer`, `hybrid`, `xgb`. To add the deep ensemble:

1. **Add a `--config` JSON interface to deep_ensemble.py** — the loop only executes
   families exposing one (CLI-arg parsing exists; add a JSON loader that maps to the
   same args). deep_ensemble already writes a compatible `RunSummary` ✅.
2. **Define `DEEP_ENSEMBLE_SPACE`** in `src/search/search.py` (Tier A+B bounds) and
   register it in `SUPPORTED_FAMILIES` / `EXECUTABLE_FAMILIES` / `SEARCH_SPACES`.
3. **Add the executor command** mapping family → `python -m models.deep_ensemble
   --config <path> ...` (with fixed `--val_split date_kfold --fold ...`).
4. **Fix the fold-overfitting gap**: the loop scores one run; wrap the executor to
   run all 5 folds and report the *mean* primary metric, or the promotion policy
   will chase fold noise. This is the single most important change for correctness.

Effort: ~1–2 hrs. Payoff: systematic Tier-B search + a defensible "we searched the
space" statement for the writeup. Not worth it unless Tier A shows real headroom.

## 5. Decision gates / stop criteria
- **Stop** if Tier A yields < +0.02 near-cloud R² and < +0.02 tail coverage across
  folds — the model is at its tuning ceiling; ship as-is.
- **Promote** a change only if it improves the headline metric on ≥4/5 folds AND
  does not regress far-cloud coverage or interval width materially.
- **Tail coverage** target is ~0.90 near-cloud via `near_cloud_target`; absolute
  tail point-R² is data-floored (do not chase it past the kNN noise floor).

## 6. Open research lever beyond tuning (separate from this plan)
Tier-B L1b per-pixel **spike residual** features (`spike_eof_weighted_residual_*`)
remain the only untested *admissible* input that could raise tail point-accuracy.
That's a data/feature project, not hyperparameter tuning — track separately.

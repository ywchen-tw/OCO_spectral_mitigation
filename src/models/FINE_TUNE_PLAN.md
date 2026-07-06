# Deep-Ensemble Fine-Tuning Plan (largely executed)

Status: written 2026-06-25 as **planning**; **most of it has since been executed
and adopted** (2026-07). §0 below is updated to the current production config;
the Tier-A/B tables keep their original "current/try" values as a historical
record — adopted knobs are marked.

## 0. Current locked-in configuration (updated 2026-07-06; do NOT re-litigate)
The production model is the deep ensemble in [deep_ensemble.py](deep_ensemble.py),
tag **`de_{ocean,land}_beta_nll_prof_reg_{r05,r15}`**:
- **Loss**: `beta_nll`, `beta=1.0` — won a clean sweep {0,0.3,0.5,0.7,1.0,1.5,2.0};
  1.0 is the optimum (β>1 overfits, β<1 underfits the mean). See
  [loss_ablation_verdict.md](../../results/model_comparison/loss_ablation_verdict.md).
- **Architecture**: `GaussianMLP` n→64→32→(mu, log_var), **M=5 members** (Tier-A
  adopted), **`--norm layer --dropout 0.1`** (lndo01, adopted from the 2026-07
  reg ablation — never worse, land tail +0.032 ppm > fold-σ; BatchNorm rejected;
  the wider-capacity arm did NOT help, 64→32 stands), AdamW lr=1e-3, wd=1e-4,
  OneCycleLR, 500 epochs (Linux), batch 8192, shared trainer `train_common.py`.
- **Features**: `full` + **ProfilePCA block** (`--profile-pca`; adopted 2026-07 —
  land near-cloud tail R² 0.28→0.68); spectroscopy k's are the no-SG fit.
- **Target**: per-surface anomaly radius — **ocean r05 / land r15**.
- **Calibration**: split + cloud-Mondrian conformal, sigma-normalized,
  `near_cloud_target=0.98` (over-covers near-cloud bins to lift the outcome-defined
  tail to ~0.90). See [tail_coverage_verdict.md](../../results/model_comparison/tail_coverage_verdict.md).
- **Validation**: `date_kfold`, 5 folds, ocean + land; independent TCCON/ship/ATom
  chains under `results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/`.

Note: code *defaults* in `deep_ensemble.py` differ from production on purpose
(dropout 0.0 / norm none keep old checkpoints loadable); production flags live in
`curc_shell_blanca_de_profile*.sh`.

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

### Tier A — highest expected value (RESOLVED 2026-07)
| knob | current | try | outcome |
|---|---|---|---|
| `n_members` | 3 | 5, 10 | **M=5 adopted** (production); M=10 tested in the DE++ hetero ablation — within noise of M=5-per-surface at scale |
| hidden width | 64→32 | 128→64, 256→128 | **rejected** — reg-ablation capacity arm and the 32,32,32 arch A/B showed no win; 64→32 stands |

Method: replicate the beta-sweep pattern — single fold, beta=1.0, 250 epochs, run
with vs without, compare `near_cloud(<=10km)` R² and `near&bottom_5pct` coverage
(crossed regime already in diagnostics). Decision: keep only if Δ is clearly
outside the ±0.10 fold spread.

### Tier B — second-order (only if Tier A shows headroom)
| knob | current | range | status |
|---|---|---|---|
| lr | 1e-3 | 3e-4 … 3e-3 (log) | untried (TabM-HPO flat-landscape lesson: don't chase) |
| weight_decay | 1e-4 | 1e-5 … 1e-3 (log) | untried |
| epochs / patience | 500 / 50 | 300–800 / 30–80 | untried |
| calib_frac | 0.15 | 0.10 … 0.25 | untried |
| mondrian_bins | 10 | 6 … 20 | untried |
| dropout (add) | none | 0 … 0.2 | **dropout 0.1 + LayerNorm ADOPTED** (2026-07 reg ablation) |

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

## 4. Autoresearch integration (dormant — Tier A closed without needing it)
The harness in `src/search/` (search.py / loop.py / promotion.py / confirm.py)
now supports only `xgb` (the `mlp_lr`/`ft_transformer`/`hybrid` families were
removed in the 2026-07-03 model consolidation — see PIPELINE_CHANGELOG.md).
If ever revived for the deep ensemble:

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

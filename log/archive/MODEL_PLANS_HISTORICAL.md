# Model plans — historical record (TabM suite + DE fine-tuning)

> **Consolidated 2026-07-09** from three completed plan docs that lived in
> `src/models/` (originals deleted):
>
> 1. `src/models/TABM_PLAN.md` — TabM-style BatchEnsemble MLP implementation plan
> 2. `src/models/TABM_TRAINING_STEPS.md` — its local→CURC execution runbook
> 3. `src/models/FINE_TUNE_PLAN.md` — deep-ensemble hyperparameter-tuning plan
>
> **Why archived:** all three are executed and their verdicts are settled. TabM
> is NOT the production model (the per-surface deep ensemble won the 2026-07-05
> head-to-head; TabM is kept as a maintained comparison model in
> `src/models/tabm.py`). The DE tuning plan's Tier A resolved to the production
> config (M=5, `--norm layer --dropout 0.1`), and its decision gate fired:
> ship as-is.
>
> **Live successors:** `src/models/deep_ensemble.py` +
> `src/models/deep_ensemble_ARCHITECTURE.md` (production model),
> `results/model_comparison/EXPERIMENTS.md` (experiment index),
> `log/near_cloud_xco2_plan.md` (how the production model was actually chosen).

---

# Part 1 — TabM implementation plan (ex `TABM_PLAN.md`)

**Model:** TabM-style BatchEnsemble MLP with monotonic quantile head
**Reference:** Gorishniy et al., "TabM: Advancing Tabular Deep Learning with
Parameter-Efficient Ensembling," 2024. A TabM-*inspired* variant, not a direct
reproduction — deviations: shared input projection (vs per-member), monotonic
quantile output head (vs scalar regression), per-member three-quantile output.
In any paper, call it "TabM-style BatchEnsemble MLP with monotonic quantile
head," not "TabM."

**Research question:** does a TabM-style BatchEnsemble MLP improve *calibrated*
anomaly prediction under physically meaningful holdout regimes (temporal/spatial
blocks) vs strong GBDT and neural baselines?

**Final status (2026-07-06):** TabM ≈ DE overall on validation + 70-case TCCON
(`results/model_comparison/tabm_vs_de_*`, `tabm/tabm_prof_m16/`), with DE clearly
better in the near-cloud land tail — the per-surface deep ensemble
(`de_*_beta_nll_prof_reg`) stays production. `src/models/transformer.py`
(FT-Transformer, referenced throughout the original plan) was deleted 2026-07-03;
its eval utilities live in `src/models/tabm_eval.py`.

## 1.1 What was built (all code-complete, smoke-tested 2026-06-07)

New files: `tabm.py` (TabMLayer/TabMBlock/TabM, monotonic head, `train_tabm`,
CLI), `gbdt_baselines.py` (XGBoost/LightGBM/sklearn fallback), `mlp_baseline.py`
(Linear 64→32→1 point predictor; quantile columns degenerate q05=q50=q95),
`splits.py` (`split_dataframe(mode='random'|'date'|'date_kfold')`),
`diagnostics.py` (coverage / width / pinball / crossing / member-spread,
stratified-by-regime, calibration pass/fail, `monotone_rearrange`).
Modified: `pipeline.py` (`_FEATURE_SETS` + `--feature-set`; sys.modules alias fix
below), `adapters.py` (`TabMAdapter`), `__init__.py` exports.

`splits.py` + `diagnostics.py` were deliberately shared modules so TabM, GBDT,
and MLP score identically and all obey *split-raw-df-then-fit-pipeline-on-
train-only*.

### Bug fix worth remembering — pipeline pickling under `python -m`
Scaler classes in `pipeline.py` force `__module__ = 'pipeline'` for portable
pickles, but fitting+saving a `FeaturePipeline` inline under
`python -m models.tabm` raised `PicklingError` (no top-level `pipeline` module).
Fix: `pipeline.py` registers the `pipeline` sys.modules alias **on import**, not
only under `__main__`:
```python
if __name__ == '__main__':
    sys.modules.setdefault('pipeline', sys.modules['__main__'])
else:
    sys.modules.setdefault('pipeline', sys.modules[__name__])
```

## 1.2 Architecture (as implemented in `tabm.py`)

- **TabMLayer** (BatchEnsemble linear): shared `W [d_out, d_in]` + b, per-member
  scale vectors `r [K, d_in]` / `s [K, d_out]` (init ones + N(0, 0.01)).
  Extra params per layer = `K × (d_in + d_out)` vs `K × d_in × d_out` naive.
- **TabMBlock**: pre-activation residual — LN→GELU→TabMLayer→LN→GELU→Dropout→
  TabMLayer + skip (skip path shared).
- **TabM**: shared input `Linear(n_features, d_model)` (intentional
  simplification; per-member input scaling left as an unrun ablation) → expand
  to `[batch, K, d_model]` → N TabMBlocks → per-member head `[batch, K, 3]` →
  monotonic transform → mean over K. Optional aux cloud head pools K then
  `Linear(d_model, 1)` logit; forward returns a dict when active.
- **Monotonic quantile head**: `q50 = a; q05 = q50 − softplus(b);
  q95 = q50 + softplus(c)` — crossing structurally impossible (crossing_rate 0
  verified everywhere).
- Default hyperparameters: K=16, d_model=256, n_layers=4, dropout 0.2,
  batch 8192 (Linux) / 2048 (Darwin), 500/100 epochs, patience 50, huber loss
  (quantile-pinball mode also available; both reported for research comparison).
- Semantics caveat kept for any writeup: the default forward returns the **mean
  of member quantiles**, which is not the quantile of the ensemble mixture —
  report as "mean member quantile." Member q50 spread is a proxy, not a
  calibrated epistemic uncertainty, unless validated.

## 1.3 Contracts (still true of the shared infrastructure)

- `FeaturePipeline.transform(df) → X float32 [N, n_features]`;
  `pipeline.n_features` / `pipeline.features` are the ONLY authoritative feature
  count/order (never hard-code counts; breaks under PCA augmentation).
- `y = xco2_bc_anomaly`. **Target outlier filter (2026-06-23):**
  `filter_target_outliers()` drops `|y| > MAX_ABS_ANOMALY_PPM (=100)` from the
  RAW df before the split — quality-flag escapees on land dominated squared
  error (RMSE≈30, R²≈0 while MAE ~0.5). Any land run made before that date was
  invalid and was re-run.
- Hard leakage rule: split the raw df first, fit the pipeline on train only,
  persist the train-fitted pipeline with the checkpoint.
- Validation: random split (comparability) + `date` block + **`date_kfold`**
  (block-rotation over dates; the primary unseen-date probe — every date tested
  once, mean±std; one fold per invocation `--n_folds N --fold K`, aggregate via
  `python -m models.aggregate_folds`). Orbit/region blocks were planned
  follow-ons, never needed. Seed sweeps s0/s1/s2 measure training stochasticity
  under the deterministic date splits.
- Calibration pass/fail thresholds (fixed before running): global cov90
  0.87–0.93; near-cloud ≥0.85; high-AOD ≥0.85; left-tail (bottom 10%) ≥0.80;
  crossing rate exactly 0; flag date-block/random RMSE ratio > 1.20.

## 1.4 Results timeline

**Local pre-test (2026-06-07, single date, ocean):** TabM smoke (K=4, 3 ep)
R² 0.32; XGBoost full R² 0.82; MLP R² 0.89 — all paths verified, crossing 0.

**First CURC batch (2026-06-23):** 28 runs clean. Random-split ocean:
TabM K=16 R²=0.821 > MLP 0.672 > XGB 0.577; K barely matters (K=1→32 ≈0.81);
`no_xco2` hurts a lot, `no_spec` barely; date split drops all to R²≈0.51–0.54.
All land runs pre-filter were invalid (re-run); the seed-sweep and aux-cloud
TabM runs were cut by walltime and re-submitted.

**The leakage finding (2026-06-23, `ocean_robustness_comparison.md`):** the
random split overstated everything by ~0.3 R² via same-day leakage. Under
5-fold date_kfold: TabM ≈ MLP > XGB (TabM−MLP +0.007 n.s.; TabM−XGB +0.039
marginal). TabM's random-split left-tail dominance reverses; both interval
models under-cover the left tail (~0.66 vs 0.90). What survives: TabM is the
only single model competitive on both accuracy and calibrated/monotone
intervals. The strongest publishable point is methodological. The decision rule
fired → pivot to calibration, which led to:

1. **Conformal wrapper** — `models/conformal.py` (split + Mondrian, pure numpy).
2. **Deep-ensemble MLP** — `models/deep_ensemble.py`: M Gaussian-NLL members →
   mixture (mu*, sigma*), inline conformal, `cld_dist_km`-Mondrian beats
   mu-Mondrian on every regime (left tail cov 0.58→0.65). **This became the
   production model.**
3. **Geometry experiment (2026-06-24, NULL):** periodic/harmonic embeddings of
   raw angles + FiLM — all deltas within fold noise; geometry already adequately
   encoded (`cos_glint`, `1/cos(sza)`, `airmass`). Track dropped.
   (`geom_experiment_verdict.md`)

**Noise-floor diagnostic (2026-06-24, `src/analysis/noise_floor_diag.py`):**
the left tail is feature-irreducible with the current 35 features —
within-neighbourhood Var(y) = 0.62 in the tail vs 0.13 bulk (5× ratio; tail
local variance exceeds global Var(y)=0.376). Residuals track aerosol+cloud
(+0.17 aod, −0.14 cld_dist), zero geometry. Only *new features* carrying the
cloud-contamination signal can fix the tail.

**New-feature admissibility rule + tiers:** an input must be derivable from the
sounding's own spectrum/retrieval — cloud distance / MODIS features are NOT
admissible (external, circular).
- Tier A, L2-Lite fit-quality scalars (chi2/rms_rel per band, eof3_1_rel,
  diverging_steps, xco2_uncertainty; `--feature_set full_fitqual`): **NULL**
  (ΔR² +0.003, tail −0.001; OCO-2 QC already filters on χ²/dp).
  (`fitqual_feature_verdict.md`)
- Tier B, L1b per-pixel spike residual (`spike_eof_weighted_residual_*`): never
  run — the only untested admissible lever for tail point accuracy.
- Tier C, raw radiance spectra via spectral CNN: not pursued.

## 1.5 Planned ablations never run (deliberately)

Member-input-scaling ablation, monotonic-head-off + crossing-penalty arm,
aux-cloud-head bins variant trained to convergence, LightGBM arm (not installed
on base env), CatBoost, orbit/region block splits, compute-budget table.
Dropped when the DE pivot made them moot.

---

# Part 2 — TabM runbook (ex `TABM_TRAINING_STEPS.md`)

Operational knowledge that outlived the TabM program (applies to all
`models.*` trainers):

- **Always launch from the repo root with `src` on `PYTHONPATH`**
  (`PYTHONPATH=src python -m models.<name>`): `get_storage_dir()` is
  cwd-relative and the modules mix relative + absolute imports.
  CURC root: `/projects/yuch8913/OCO_spectral_mitigation`.
- Data in `results/csv_collection/`: macOS default
  `combined_2020-02-01_all_orbits.parquet` (single date), Linux/CURC
  `combined_2016_2020_dates.parquet`. `--val_split date`/`date_kfold` need
  multi-date data → CURC only.
- Outputs: `results/model_{tabm,gbdt,mlp_baseline}/<suffix>/` with
  `*_metrics.json`, `*_stratified_metrics.csv`, `run_summary.json`, checkpoint
  + pipeline pickle.
- **Pre-fit pipeline leakage caveat:** `python -m models.pipeline` fits on the
  entire file; handing that pickle via `--pipeline` to a blocked-split run leaks
  held-out scaler statistics. For blocked splits, let each model fit its own
  pipeline on the train split (the default).
- Staged validation gates: Stage 0 synthetic smoke (crossing must be 0) →
  Stage 1 local single-date end-to-end (33 s TabM smoke / 17 s XGB / 30 s MLP)
  → Stage 1.5 `curc_shell_blanca_preflight.sh` (syntax + command preview) →
  Stage 2 CURC full runs (`curc_shell_blanca_train_{pipeline,tabm,gbdt,
  mlp_baseline}.sh`; scripts hold commented blocks per ablation; monitor via
  `squeue`, sbatch logs, `gpu_monitor_<JOBID>.csv`).
- Suffix conventions: `tabm_ocean_k16_random`, `..._date`, `..._s{0,1,2}`,
  `tabm_ocean_{k1,k8,k32,quantile,no_xco2,no_spec,aux}`, `gbdt_ocean_xgb_*`,
  `mlp_ocean_*`; swap ocean→land with `--sfc_type 1`.

---

# Part 3 — Deep-ensemble fine-tuning plan (ex `FINE_TUNE_PLAN.md`)

Written 2026-06-25 as planning; executed and adopted 2026-07. Kept as
config-provenance for the production model.

## 3.1 Locked-in production configuration (2026-07-06; do NOT re-litigate)

Tag **`de_{ocean,land}_beta_nll_prof_reg_{r05,r15}`** (`deep_ensemble.py`):
- **Loss** `beta_nll`, β=1.0 — clean sweep over {0,0.3,0.5,0.7,1.0,1.5,2.0};
  β>1 overfits, β<1 underfits the mean (`loss_ablation_verdict.md`).
- **Architecture** `GaussianMLP` n→64→32→(mu, log_var), **M=5**,
  `--norm layer --dropout 0.1` (lndo01; never worse, land tail +0.032 ppm >
  fold-σ; BatchNorm rejected; wider capacity rejected — 64→32 stands), AdamW
  lr=1e-3 wd=1e-4, OneCycleLR, 500 epochs, batch 8192, shared trainer
  `train_common.py`.
- **Features** `full` + ProfilePCA (`--profile-pca`; land near-cloud tail R²
  0.28→0.68); spectroscopy k's are the no-SG fit.
- **Target** per-surface anomaly radius ocean r05 / land r15.
- **Calibration** split + cloud-Mondrian conformal, sigma-normalized,
  `near_cloud_target=0.98` (`tail_coverage_verdict.md`).
- **Validation** `date_kfold` 5 folds; independent TCCON/ship/ATom chains under
  `results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/`.
- Code *defaults* in `deep_ensemble.py` intentionally differ (dropout 0.0 /
  norm none keep old checkpoints loadable); production flags live in
  `curc_shell_blanca_de_profile*.sh`.
- Full-scale: ocean R²≈0.54 / near-cloud ≈0.63; land ≈0.46 / near 0.60.
  beta_nll beats gaussian most on land (+0.09 near R²)
  (`deep_ensemble_fullscale_verdict.md`).

## 3.2 Known dead ends (do not retry)

- student_t loss — down-weights the tail; strictly worse.
- Asymmetric / raw (non-sigma-normalized) conformal — worse tail coverage.
- Geometry embeddings, fit-quality features, extra spectral features — null
  (tail partly data-irreducible; kNN local Var(y) tail/bulk ≈ 5×).
- Flat-target conformal for the y-defined tail — provably unattainable; only
  regime elevation (`near_cloud_target`) helps.

## 3.3 Tuning tiers — outcomes

- **Tier A (RESOLVED 2026-07):** `n_members` 3→**5 adopted**; M=10 within noise
  of M=5-per-surface at scale (DE++ hetero ablation). Hidden width 128→64 /
  256→128 **rejected** — capacity arm and 32,32,32 A/B showed no win.
- **Tier B (untried by design):** lr, weight_decay, epochs/patience, calib_frac,
  mondrian_bins — TabM-HPO flat-landscape lesson: don't chase. Exception:
  dropout+LayerNorm was promoted from here and **adopted** (reg ablation).
- **Tier C (post-hoc, cheap):** `near_cloud_target` sweep {0.975…0.99} and
  finer near-cloud Mondrian bins — available anytime without retraining.

## 3.4 Methodology guardrails (carried forward to any future tuning)

1. Score on a fold-AVERAGED metric, never a single fold (fold σ ≈ ±0.10 R²).
2. Never touch the held block during search (calib block carved from TRAIN
   dates inside each fold).
3. Separate point-accuracy (judge on R²/tail R²) from calibration (coverage +
   width) — conformal owns coverage.
4. Watch the tail: near-cloud R² and `near&bottom_5pct` coverage are the
   headline metrics; global R² can improve while the tail does not.
5. Cost: full-data run ≈ 40 min GPU/fold; prototype on the 12-date subset,
   confirm survivors on full 2016–2020.

## 3.5 Decision gate (fired)

Stop if Tier A yields < +0.02 near-cloud R² and < +0.02 tail coverage across
folds → **the model is at its tuning ceiling; ship as-is.** That is what
happened. Promote only on ≥4/5 folds without regressing far-cloud coverage or
width.

Autoresearch integration (`src/search/`, xgb-only after the 2026-07-03
consolidation) stayed dormant — Tier A closed without needing it. The one
correctness note if ever revived: wrap the executor to run all 5 folds and
score the mean, or promotion chases fold noise.

## 3.6 Open lever beyond tuning

Tier-B L1b per-pixel spike-residual features (`spike_eof_weighted_residual_*`)
remain the only untested *admissible* input that could raise tail
point-accuracy. A data/feature project, not hyperparameter tuning.

# TabM Implementation Plan

> **STATUS (2026-07-06): TabM is NOT the production model.** The 2026-07-05
> head-to-head (validation + 70-case TCCON, `results/model_comparison/tabm_vs_de_*`,
> `tabm/tabm_prof_m16/`) put TabM ≈ DE overall, with DE clearly better in the
> near-cloud land tail — the per-surface deep ensemble (`de_*_beta_nll_prof_reg`)
> stays production. TabM is kept as a maintained comparison model. Note also:
> `src/models/transformer.py` (referenced below) was deleted 2026-07-03 — its
> eval utilities now live in `src/models/tabm_eval.py`.

**Model:** TabM-style BatchEnsemble MLP with monotonic quantile head  
**Reference:** "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling" (Gorishniy et al., 2024)

**Architecture note:** This is a TabM-*inspired* variant, not a direct reproduction. Deviations from the reference include: shared input projection (vs. per-member), monotonic quantile output head (vs. scalar regression), and per-member three-quantile output. In any paper or report, describe it as "TabM-style BatchEnsemble MLP with monotonic quantile head" rather than "TabM" to avoid misrepresenting the original.

**Research question:** Does a TabM-style BatchEnsemble MLP improve *calibrated* anomaly prediction under physically meaningful holdout regimes (temporal/spatial blocks) vs. strong GBDT and neural baselines, and not merely vs. FT-Transformer?

---

## Implementation status (last updated 2026-06-07)

**Code-complete and smoke-tested on synthetic data (CPU).** Not yet trained on
real CURC data / GPU — wall-clock, compute-budget table, multi-seed results, and
actual calibration numbers remain TBD on CURC.

### Files created
| File | Contents | Status |
|---|---|---|
| `tabm.py` | `TabMLayer`, `TabMBlock`, `_monotonic_quantiles`, `TabM`, `train_tabm()`, `_default_run_config`/`_load_run_config`, `main()` | ✅ done |
| `gbdt_baselines.py` | `train_xgboost`, `train_lightgbm`, sklearn fallback, `evaluate_gbdt`, `main()` | ✅ done |
| `mlp_baseline.py` | `MLPBaseline` (Linear64→32→1 point predictor), `train_mlp()`, `main()` | ✅ done |
| `splits.py` | `split_dataframe(mode='random'|'date')` raw-df split (orbit/region = `NotImplementedError`, plan v1) | ✅ done |
| `diagnostics.py` | coverage / width / per-quantile pinball / crossing / member-spread, stratified-by-regime, calibration pass/fail, `monotone_rearrange` | ✅ done |

### Files modified
| File | Change | Status |
|---|---|---|
| `pipeline.py` | `XCO2_FEATURES`, `SPEC_FEATURES`, `_FEATURE_SETS`, `_resolve_feature_set`; `feature_set` arg on `FeaturePipeline.fit()` + persisted on instance; `--feature-set` CLI. **Also: `pipeline` sys.modules alias now registered on import (not only under `__main__`) — see Bug fix below.** | ✅ done |
| `transformer.py` | `plot_permutation_importance(output_prefix="ft")`; `_batched_predict` + permutation `_predict_q50` unwrap `dict["quantiles"]`; module docstring + citation | ✅ done |
| `adapters.py` | `TabMAdapter` (`model_tabm_best.pt` + `tabm_meta.pkl`, `_ARCH_VERSION=1`, `__main__`-aware reconstruction) | ✅ done |
| `__init__.py` | exports `TabM`, `TabMLayer`, `TabMBlock`, `train_tabm`, `MLPBaseline`, `train_mlp`, `train_xgboost`, `train_lightgbm`, `evaluate_gbdt`, `TabMAdapter` | ✅ done |

### Shared infrastructure decisions (additions beyond the original file list)
- **`splits.py` + `diagnostics.py`** were added as shared modules so TabM, GBDT,
  and MLP score identically and all obey the *split-raw-df-then-fit-pipeline-on-
  train-only* rule. (The plan listed diagnostics as "same suite" across models;
  these centralise it rather than duplicating per file.)
- **`--K` CLI override** added to `tabm.py` so the K-sweep needs no per-K config
  JSON (the `--config tabm_kN.json` path still works).
- **MLP baseline is a point predictor**, so the three quantile columns are
  reported as q05=q50=q95 (degenerate); only RMSE/MAE/R² are meaningful for it.

### Verified by smoke test (`tabm`, `mlp_baseline`, `gbdt_baselines`, `splits`, `diagnostics`)
- default forward `[B,3]`; `return_members=True` `[B,K,3]` with `mean(members)==default` (eval mode); per-member monotonicity → `crossing_rate == 0`.
- aux-cloud dict return `{quantiles, members, cloud_logit}`; member-level training loss; checkpoint save/restore; `TabMAdapter` round-trip.
- end-to-end training of TabM (incl. aux-cloud binary), MLP baseline, and XGBoost (raw + rearranged) on synthetic data.
- feature-set drops resolve from the authoritative lists (`pipeline.n_features` remains the single source of truth — counts differ slightly from the approximate numbers in the ablation tables below, which is expected).
- random vs date split is date-disjoint; package imports cleanly; all CLIs parse.

### Local pre-test (2026-06-07, macOS/MPS, `combined_2020-02-01_all_orbits.parquet`, ocean)
All three trained end-to-end and wrote full diagnostics; TabM `crossing_rate==0` throughout.

| Model | Config | Time | RMSE | R² | cov90 | crossing |
|---|---|---|---|---|---|---|
| TabM | smoke (K=4, 3 ep) | 33 s | 0.55 | 0.32 | 0.915 | 0 |
| XGBoost | full (500 trees ×3) | 17 s | 0.285 | 0.82 | 0.829 | 0 |
| MLP baseline | 100 ep | 30 s | 0.224 | 0.89 | — | — |

(TabM R² is low only because of the 3-epoch smoke config; XGBoost/MLP are fully trained.)
`--feature_set no_spec` (35→26 feats) and `--aux_cloud` (leakage check + BCE multi-task)
paths also verified end-to-end.

#### Bug fix — pipeline pickling under `python -m`
The scaler classes in `pipeline.py` force `__module__ = 'pipeline'` so their pickles are
portable. But fitting+saving a `FeaturePipeline` **inline** under `python -m models.tabm`
failed: the live module is `models.pipeline`, and pickle could not find a top-level
`pipeline` module → `PicklingError: Can't pickle <class 'pipeline.RobustStandardScaler'>`.
The legacy FT/XGB flow never hit this because it pre-fits the pipeline via
`python src/models/pipeline.py` (run as `__main__`, where the alias shim was active) and
then loads it with `--pipeline`. Fix: `pipeline.py` now registers the `pipeline` sys.modules
alias **on import** (the `else` branch), not only under `__main__`:
```python
if __name__ == '__main__':
    sys.modules.setdefault('pipeline', sys.modules['__main__'])
else:
    sys.modules.setdefault('pipeline', sys.modules[__name__])
```

### Not yet done / deferred
- Real CURC training runs (data + GPU); compute-budget table; ≥3-seed mean±std tables.
- Orbit-block and region-block splits (plan marks these follow-on after v1).
- Auxiliary cloud **bins** head is wired but only validated structurally, not trained to convergence.
- `lightgbm` is **not installed** in the base env (`gbdt_baselines.py` falls back xgboost→lightgbm→sklearn; `--model lightgbm` errors clearly if absent — `pip install lightgbm` first).

### CURC launch scripts (repo root)
- `curc_shell_blanca_train_tabm.sh` — GPU; primary K=16 ocean random + date, with K-sweep / loss / feature-set / aux / seed / date_kfold examples commented.
- `curc_shell_blanca_train_gbdt.sh` — **CPU-only** (no `--gres`); XGBoost random + date + date_kfold (LightGBM commented behind `pip install`).
- `curc_shell_blanca_train_mlp_baseline.sh` — GPU; MLP baseline random + date + date_kfold.
- `curc_shell_blanca_train_geom_mlp.sh` — GPU; geometry experiment (none/concat/film × 5-fold date_kfold).

All launch from the **repo root** with `src` on `PYTHONPATH`
(`cd <repo> && PYTHONPATH=src python -m models.<name>`): the modules mix relative
(`.pipeline`) and absolute (`utils`, `search.tracking`) imports, and
`get_storage_dir()` is cwd-relative (`./results/...`), so cwd must be the repo root.

---

## Post-kfold findings & next steps (2026-06-23 → open)

The first rigorous unseen-date evaluation is in
`results/model_comparison/ocean_robustness_comparison.md` (5-fold `date_kfold`,
paired per-fold). **Key result:** the random split overstated everything by ~0.3 R²
via same-day leakage. Under k-fold, ocean point accuracy is **TabM ≈ MLP > XGB**
(TabM−MLP Δ=+0.007, n.s.; TabM−XGB Δ=+0.039, marginal; MLP−XGB Δ=+0.033, p≈0.02).
TabM's random-split left-tail dominance **reverses** (MLP best in the tail), and
both interval models under-cover the left tail (~0.66 vs nominal 0.90). What
survives: TabM is the only single model competitive on *both* accuracy and
calibrated/monotone intervals — "no accuracy cost for free calibrated uncertainty,"
not better predictions. The strongest publishable point is methodological
(random-split leakage). Decision rule from the validation section therefore fires →
pivot the claim to calibration, and pursue:

1. **Conformal calibration wrapper** — BUILT: `models/conformal.py` (split + Mondrian,
   pure-numpy, unit-tested: split cov 0.898, per-bin 0.88–0.93). Model-agnostic; bins
   must be observable (predicted-mu deciles by default, never y). Currently wired into
   the deep ensemble; applying it to saved TabM/XGB is a follow-up (needs a calibration
   re-run with their predictions).
2. **Deep-ensemble MLP** — BUILT: `models/deep_ensemble.py` + `curc_shell_blanca_train_deep_ensemble.sh`.
   M Gaussian-NLL members → mixture (mu*, sigma*); conformal calibration inline (calib
   block carved from TRAIN dates). Writes 3 interval variants per run (`de_raw_*`,
   `de_split_*`, `de_mondrian_*`). Mondrian bins are selectable (`--mondrian_col`):
   `mu` deciles vs a physical proxy (`cld_dist_km` / `aod_total`).
   **Local validation on real 12-date data (`combined_2020_dates.parquet`, ocean fold0, M=3):**
   global cov90 ≈ 0.88 (exchangeability caveat does NOT bite at 12 dates, unlike the
   degenerate 6-date fixture). **`cld_dist_km`-Mondrian beats `mu`-Mondrian on every
   regime** and lifts the left tail (bottom-5% cov 0.58→0.65) — set as the CURC default.
   But the y-extreme tail stays ~0.65 vs 0.90: a **residual limit** (a y-defined group
   can't be fully calibrated by an observable proxy). Point R²≈0.40 at M=3/100ep (will
   rise at M=5? — note CURC uses M=3/500ep). CURC script is a **fold-array job**
   (`--array=0-4`, gpu:1/task, `--requeue`, batch 8192) — same per-task queue ask as a
   single job, parallel when GPUs free.
   **➤ UPDATE with the de_mondrian k-fold cov90 / R² vs TabM once the CURC array returns.**
3. **Geometry experiment — `geom_mlp.py` (DONE 2026-06-24 → NULL result).** Tested
   whether a richer geometry representation helps the MLP backbone: periodic (sin/cos
   harmonic, H=4) embeddings of raw angles (`sza, vza, glint_angle, pol_angle, lat,
   lon`) + optional FiLM gate, `none`/`concat`/`film`, 5-fold `date_kfold`, paired.
   **Verdict: no help.** Ocean concat−none ΔR²=−0.008 (t≈−0.87), film−none +0.001
   (t≈0.17); land concat−none −0.011 (t≈−1.83), film−none +0.003 (t≈0.41); all
   per-regime deltas within fold noise, including the left tail. Geometry is already
   adequately encoded by the pipeline (`cos_glint`, `1/cos(sza)`, `airmass`); the new
   sin/harmonic/location content adds no signal. Confirms Lee et al. (MileTS '19) does
   not transfer (their gain was *introducing* geometry + an image attention map;
   neither applies). **Geometry track dropped — do not port to TabM.** Full writeup:
   `results/model_comparison/geom_experiment_verdict.md`.

### Noise-floor diagnostic (2026-06-24) — why the tail is hard

`src/analysis/noise_floor_diag.py` on the real 12-date file (ocean): **the left tail
is feature-irreducible with the current 35 features.** Soundings near-identical in
feature space still have within-neighbourhood Var(y) = **0.62 in the left tail vs 0.13
in the bulk** (tail local variance even exceeds the global Var(y)=0.376). Residual
error is driven by **aerosol + cloud** (|resid| vs `aod_total` +0.17, `cld_dist_km`
−0.14) and **zero by geometry** (sza/vza/glint ~0.03 — independently re-confirms the
geometry null). Implication: no model/calibration trick fixes the tail on the current
features — only **new features** can, and they must carry the cloud-contamination
signal. (The kNN "max R²≈0.60" is a loose lower bound — the base model beats it — so
the robust signal is the 5× tail/bulk variance ratio, not the absolute number.)

### New-feature investigation (admissibility rule + tiers)

**Admissibility rule (hard):** an input must be *known for a given spectra set* — i.e.
derived from the OCO-2 sounding's own spectrum/retrieval. **Cloud distance and any
MODIS-derived cloud-field feature are NOT admissible** (external; unknown at prediction
time; and circular — they are the validation variable). This sharpens the question to:
*can the spectrum itself reveal cloud contamination?*

Candidate unused variables found by inventorying the L1b + L2-Lite files (see chat
2026-06-24), ranked:

- **Tier A — L2-Lite fit-quality scalars (IMPLEMENTED in `spectral/fitting.py`, 2026-06-24):**
  `chi2_{o2a,wco2,sco2}`, `rms_rel_{o2a,wco2,sco2}` (reduced χ² / relative RMS of the L2
  spectral fit per band — the textbook contamination flag), `eof3_1_rel` (systematic
  spectral-residual structure), `diverging_steps` (convergence difficulty),
  `xco2_uncertainty` (posterior error). Added to both the Lite loader and `output_dict`;
  verified all 10 load (100% finite). Exposed to all models as **`--feature_set full_fitqual`**
  (`pipeline.py`: base `full` + the 9 cols = 36 features; an *add*-type set, vs the
  drop-type `no_xco2`/`no_spec`). Compare **paired** vs `--feature_set full` under
  `date_kfold` (advantage = full_fitqual − full per fold).
  **Requires re-running the Phase pipeline to regenerate the combined parquet first**
  (else `df[qt_features]` raises KeyError on the missing columns).
  **PRELIMINARY VERDICT (2026-06-24, NULL):** on the regenerated `combined_2020_dates.parquet`
  (ocean, HistGBR random split), full_fitqual vs full = **ΔR² +0.003 overall, tail R² −0.001**,
  and kNN tail irreducibility unchanged (0.619→0.625). Scale-invariant tree → not a scaling
  artifact. Cause: OCO-2 QC already filters on χ²/dp, so post-QC fit-quality adds ~no signal.
  Converges with the geometry null + noise floor → *the contaminated tail is not predictable
  from the spectrum, its retrieval state, or its fit quality (post-QC)* — a publishable
  negative result. Confirmatory date_kfold pair on CURC optional (expectations low). Full
  writeup: `results/model_comparison/fitqual_feature_verdict.md`. Remaining lever: Tier-B
  L1b per-pixel spike residual.
- **Tier B — L1b spike/residual (not yet done):** `SpikeEOF/spike_eof_weighted_residual_*`
  (per-pixel deviation from clean EOF reconstruction → aggregate mean/max/std per band)
  and `spike_eof_bad_colors_*` (per-sounding scalar), `rad_continuum_*`. Needs per-orbit,
  per-pixel aggregation from the L1b granules.
- **Tier C — raw radiance spectra (deep option):** `radiance_{o2,weak_co2,strong_co2}`
  (1016×3) via a spectral CNN/1-D model — large lift.

**Caveat:** OCO-2 QC already uses χ²/dp to remove bad soundings, so the *surviving*
soundings' χ² has reduced discriminating power — measure, don't assume. **Test plan:**
after regenerating the parquet, add Tier-A to the feature set and re-run the tail-variance /
marginal-value diagnostic; a NULL result is itself publishable (*cloud proximity perturbs
XCO2 in ways not detectable from the single-sounding spectrum or its fit quality*).
**➤ UPDATE with the Tier-A tail-impact numbers once the parquet is rebuilt.**

---

## Data contract (from `pipeline.py` → model)

- `FeaturePipeline.transform(df)` → `X: float32 [N, n_features]`
  - `n_features = pipeline.n_features` — the **only** authoritative feature count
  - `pipeline.features` — the **only** authoritative feature order
  - Never hard-code `len(qt_features) + 8`; this breaks under PCA augmentation or whitening
  - log1p pre-transform already applied to AOD / `fp_area_km2` columns before scaling
  - **Feature provenance audit:** Any feature with `bc` in its name must be verified to be
    available at prediction time and must not encode information derived from or downstream of
    the target construction (`xco2_bc_anomaly`). Audit before finalizing the feature list.
- `y = xco2_bc_anomaly` (ppm, ~zero-centred, heavy left tail from cloud-affected soundings)
- **Target outlier filter (added 2026-06-23):** `filter_target_outliers()` in `pipeline.py`
  drops rows with `|xco2_bc_anomaly| > MAX_ABS_ANOMALY_PPM` (=100 ppm) from the RAW dataframe
  **before** the train/held split, in `tabm.py` / `gbdt_baselines.py` / `mlp_baseline.py`.
  These extremes are quality-flag escapees / fill-value leakage, not the target. On the
  multi-date file a handful of them on **land** dominated squared error → RMSE≈30, R²≈0 while
  MAE stayed ~0.5; the single-date local file has none (max |y| ≈ 8–12 ppm), so the filter is a
  no-op locally and only cleans the CURC multi-date data. NaN targets are kept (handled by the
  per-model `isfinite` mask).
- Loss: see Loss section below — two modes required for research comparison

## Model contract (compatible with existing training infrastructure)

- `forward(x: [batch, n_features]) → [batch, 3]` for `(q05, q50, q95)` — default, drop-in compatible
- `forward(x, return_members=True) → [batch, K, 3]` — member-level outputs for spread/calibration diagnostics
- When the auxiliary cloud head is enabled, `forward()` returns a dict to avoid breaking existing evaluation code:
  ```python
  {
      "quantiles":   torch.Tensor,  # [batch, 3]    — always present
      "members":     torch.Tensor,  # [batch, K, 3] — present if return_members=True
      "cloud_logit": torch.Tensor,  # [batch, 1]    — present if cloud head is active
  }
  ```
  When the cloud head is disabled (default), `forward()` returns `[batch, 3]` unchanged for drop-in compatibility.
- Checkpoint format: `model_tabm_best.pt` with keys `epoch / model_state_dict / val_loss / val_mae / val_r2`

---

## Architecture

### `TabMLayer` — BatchEnsemble linear layer

Each layer has one shared weight matrix and K pairs of lightweight per-member scaling vectors:

```
r : [K, d_in]   — per-member input scale  (init: ones + N(0, 0.01))
s : [K, d_out]  — per-member output scale (init: ones + N(0, 0.01))
W : [d_out, d_in] + b : [d_out]  — shared backbone (standard nn.Linear)
```

Forward pass for input `x: [batch, K, d_in]`:

```
x_scaled = x * r[K, d_in]             # broadcast: [batch, K, d_in]
x_flat   = reshape [batch*K, d_in]
h_flat   = x_flat @ W.T + b           # shared linear: [batch*K, d_out]
h        = reshape [batch, K, d_out]
out      = h * s[K, d_out]            # [batch, K, d_out]
```

Extra parameters per layer vs naive K-fold ensemble:
- `TabMLayer`: `K × (d_in + d_out)` — orders of magnitude cheaper
- Naive ensemble: `K × d_in × d_out`

### `TabMBlock` — residual block

Pre-activation residual (mirrors existing `_ResBlock` in `adapters.py`):

```
skip = x                                         # [batch, K, d]
x    = LayerNorm(d)(x)
x    = GELU(x)
x    = TabMLayer(d, d, K)(x)
x    = LayerNorm(d)(x)
x    = GELU(x)
x    = Dropout(p)(x)
x    = TabMLayer(d, d, K)(x)
out  = skip + x                                  # residual: [batch, K, d]
```

Both inner projections are per-member (TabMLayer); the skip path is shared — parameter-efficient and consistent with the paper's design.

### `TabM` — full model

```
x: [batch, n_features]
│
├─ shared Linear(n_features, d_model)  ← input projection (intentional simplification; see note)
│  LayerNorm(d_model)
│  GELU
│  Dropout(0.15)
│
├─ expand → [batch, K, d_model]
│
├─ N × TabMBlock(d_model, K, dropout)  ← h_members: [batch, K, d_model]
│                                         (K is NOT pooled here — branch below)
├─ anomaly head  (always active)
│   ├─ TabMLayer(d_model, 3, K)(h_members)  ← per-member logits: [batch, K, 3]
│   ├─ monotonic quantile transform          ← enforces q05 ≤ q50 ≤ q95 per member
│   └─ mean over K dim                       ← [batch, 3]  (default)
│                                               or [batch, K, 3] (return_members=True)
│
└─ cloud-proximity head  (ablation only — disabled by default)
    ├─ h_pooled = mean(h_members, dim=1)    ← [batch, d_model]  (pool K for classifier)
    └─ Linear(d_model, 1)(h_pooled)         ← cloud logit: [batch, 1] for near_cloud BCE
```

**Input projection design note:** All K members receive the same pipeline output `X`.
A shared projection is an intentional simplification — it reduces parameters and defers
diversity to the BatchEnsemble hidden layers where the r/s vectors diverge during training.
It does not inherently prevent member collapse; collapse is avoided by the per-member r/s
scaling in the hidden layers, not by the input projection choice. Per-member input scaling
*can* add early diversity (different initialization → different gradient paths from step 1),
but the added complexity is left as an ablation (`ablation_member_input_scale`), not the default.

### Monotonic quantile head

Raw member head outputs `[batch, K, 3]` = `(a, b, c)`. Convert to ordered quantiles:

```python
q50 = a
q05 = q50 - F.softplus(b)   # strictly below q50
q95 = q50 + F.softplus(c)   # strictly above q50
```

This guarantees `q05 < q50 < q95` for every member and every sample, eliminating
quantile crossing without requiring a penalty term in the loss.

**Initialization note:** `softplus(0) ≈ 0.693`, so a zero-initialized head starts with
`q95 − q05 ≈ 1.386 ppm` — plausible but arbitrary. Where practical, initialize from
the training target distribution before training begins:

```python
# run once on y_train (numpy array) before first epoch
q50_init  = np.median(y_train)
dlo_init  = np.median(y_train) - np.quantile(y_train, 0.05)   # ≈ q50 - q05
dhi_init  = np.quantile(y_train, 0.95) - np.median(y_train)   # ≈ q95 - q50

head.bias_q50.data.fill_(q50_init)
# softplus^{-1}(x) = log(exp(x) - 1) for x > 0
head.bias_dlo.data.fill_(math.log(math.expm1(dlo_init)))
head.bias_dhi.data.fill_(math.log(math.expm1(dhi_init)))
```

This is a soft recommendation — skip if the head biases are not exposed as named parameters
in the chosen implementation. Default init is acceptable for ablations.

### `return_members=True` usage

```python
with torch.no_grad():
    preds_members = model(X, return_members=True)   # [batch, K, 3]

spread = preds_members[:, :, 1].std(dim=1)          # member q50 std
interval_from_spread = preds_members[:, :, 2].mean(dim=1) - preds_members[:, :, 0].mean(dim=1)
crossing_rate = (preds_members[:, :, 0] >= preds_members[:, :, 2]).float().mean()
```

**Quantile averaging semantics:** The default `forward()` returns the mean of K member quantile
outputs. The mean of member quantiles is *not* the same as the quantile of the ensemble predictive
mixture distribution. It is a pragmatic predictor that is useful in practice, but do not claim
probabilistic correctness (e.g., "the ensemble q05" is not the 5th percentile of the mixture).
Report it as "mean member quantile" in any paper.

**Member spread as uncertainty proxy:** `spread = std(K q50s)` is a useful diagnostic but is not
a calibrated epistemic uncertainty estimate without further verification. Treat it as a proxy and
validate empirically: check whether spread correlates with absolute error, tail failures, cloud
proximity, and OOD regimes. Only promote it to "uncertainty estimate" if empirical calibration
supports it.

### Default hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| K (ensemble size) | 16 | Sweep 1 / 8 / 16 / 32 (K=1 = plain MLP baseline) |
| d_model | 256 | Wider than plain MLP; K amortises the cost |
| n_layers | 4 | |
| dropout | 0.2 | |
| batch_size (Linux/CURC) | 8192 | Start here; benchmark before scaling up (see memory note) |
| batch_size (Darwin/local) | 2048 | |
| epochs (Linux) | 500 | |
| epochs (Darwin) | 100 | |
| patience | 50 | Early stopping |
| loss | huber | Default for comparability with FT-Transformer |
| huber_delta | 1.0 | |

**Memory note:** With K=16, d_model=256, one activation tensor at batch=32768 is ~134M floats (~536 MB fp32).
Start at batch_size=8192 (~134M/4), benchmark throughput vs memory, then scale to 16384 if headroom allows.
Do not use 32768 as the default.

---

## Loss strategy

Two loss modes — both must be reported for the research comparison:

| Mode | Config key | Description |
|---|---|---|
| `huber` (default) | `loss: "huber"` | Existing `huber_pinball_loss`; median branch uses Huber, not strict L1 |
| `quantile` | `loss: "quantile"` | Plain pinball loss (existing `quantile_loss`) for all three quantiles |

Huber on q50 is not identical to a median quantile objective. For a fair research comparison,
train and evaluate under both losses; report pinball loss per quantile separately.

---

## Files to create / modify

Each model lives in its own file. Never mix model implementations across files — one model class per file keeps checkpoints, adapters, and CLI entry points independently versioned and auditable.

### Code-level citation rule

Every new model file must open with a docstring citing the primary reference(s) for that architecture. Format:

```python
"""
<ModelName> implementation for OCO-2 XCO2 anomaly prediction.

Architecture reference:
  <Author(s)>. "<Title>." <Venue/arXiv>, <Year>.
  <URL or DOI if available>

This implementation deviates from the reference as follows:
  - <deviation 1>
  - <deviation 2>
"""
```

Deviations from the reference paper must be listed explicitly so a future reader can distinguish
what was reproduced and what was modified. If a file has no single reference (e.g., a plain MLP),
note "no external reference" rather than omitting the block.

---

### NEW: `src/models/tabm.py`

Reference: Gorishniy et al., "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling," 2024.  
Deviations to document: shared input projection, monotonic quantile head, three-quantile output (vs. scalar regression).

```
TabMLayer          — BatchEnsemble linear layer (see Architecture section)
TabMBlock          — pre-activation residual block using TabMLayer
TabM               — full model with monotonic head + return_members support
train_tabm()       — training loop; mirrors train_uncertainty_transformer() in transformer.py
_default_run_config() / _load_run_config()  — same JSON config pattern
main()             — CLI entry point; mirrors transformer.py main()
```

Reuses from `transformer.py` (imported, not duplicated):
- `huber_pinball_loss`, `quantile_loss`, `variance_penalty`, `mmd_loss_1d`
- `_batched_predict` — **when the auxiliary cloud head is active, `forward()` returns a dict;
  `_batched_predict` must unwrap `output["quantiles"]` rather than using the raw return value.
  Add a guard: `out = model(x); preds = out["quantiles"] if isinstance(out, dict) else out`
  so evaluation code does not break silently when switching between single- and multi-head modes.**
- `plot_permutation_importance` — **after parameterizing `output_prefix`** (see below)
- `evaluate_model_X_text`, `plot_evaluation_by_regime`

Does NOT include (TabM has no attention mechanism):
- `plot_attention_map`

### NEW: `src/models/gbdt_baselines.py`

Reference: no single paper; cite the libraries used:
- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD 2016.
- Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS 2017.

```
train_xgboost(X_train, y_train, quantiles, cfg) → fitted XGBModel
train_lightgbm(X_train, y_train, quantiles, cfg) → fitted LGBMModel
evaluate_gbdt(model, X_test, y_test, output_prefix, out_dir)  — shared eval; same diagnostics as neural models
main()  — CLI entry point: --model {xgboost,lightgbm} --sfc_type --val_split --suffix
```

Each quantile (q05, q50, q95) requires a separately trained GBDT model (one per `alpha`).  
Persist fitted models with `joblib.dump` to `results/model_gbdt/<suffix>/model_{q}_{algo}.joblib`.  
No GPU needed; runs on CPU only.

### NEW: `src/models/mlp_baseline.py`

Reference: no external reference (standard MLP). Note in docstring.  
This is a clean re-implementation of the existing `result_ana.py` MLP, brought into the same
training scaffold as TabM and FT-Transformer so results are directly comparable (same split,
same pipeline, same diagnostics). Do not modify `result_ana.py` itself.

```
MLPBaseline        — Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) (matches result_ana.py)
train_mlp()        — mirrors train_tabm() structure
main()             — CLI entry point
```

### MODIFY: `src/models/transformer.py`

- Parameterize `plot_permutation_importance(output_prefix: str = "ft")` so it writes
  `{output_prefix}_permutation_importance.csv/.png` instead of the hardcoded `ft_` prefix.
  TabM calls it with `output_prefix="tabm"`. **Do this before reusing the function.**
- Add or verify module-level docstring citing: Gorishniy et al., "Revisiting Deep Learning
  Models for Tabular Data," NeurIPS 2021.

### MODIFY: `src/models/adapters.py`

Add `TabMAdapter` following the exact pattern of `FTAdapter`:

```python
CHECKPOINT_FILE = 'model_tabm_best.pt'
META_FILE       = 'tabm_meta.pkl'
_ARCH_VERSION   = 1
```

Meta dict keys: `n_features, K, d_model, n_layers, dropout, feature_names, arch_version`

`load()` reconstructs `TabM(n_features, K, d_model, n_layers, dropout)` and loads weights.  
`can_load()` checks `arch_version` — stale checkpoints fall through to a fresh training run.

### MODIFY: `src/models/__init__.py`

Export `TabM`, `TabMAdapter`, `MLPBaseline`, and GBDT train functions alongside existing exports.

---

## Training / evaluation scaffold (`tabm.py main()`)

Identical to `transformer.py main()` except:

| Step | Change |
|---|---|
| Output dir | `results/model_tabm/<suffix>/` |
| Config JSON | Adds `K` under `model` key |
| Model construction | `TabM(n_features, K, d_model, n_layers, dropout)` |
| Adapter | `TabMAdapter` instead of `FTAdapter` |
| Checkpoint file | `model_tabm_best.pt` |
| Attention plots | Removed (no attention in TabM) |
| Permutation importance | `plot_permutation_importance(output_prefix="tabm")` |
| Regime eval plots | Kept — same `plot_evaluation_by_regime()` |
| Validation splits | Random split + at least one blocked split (see below) |

---

## Validation strategy

### Random split (existing)
- Current 80/20 random train/test split — keep for comparability with FT-Transformer.

### Blocked splits (new — required)
Random splits overestimate performance when nearby soundings, orbits, or dates share
atmospheric state.

**First implementation: date block only.** Hold out the last N days (e.g., last 10% of
unique dates sorted chronologically). This is the most impactful and simplest to implement
correctly. Orbit and region blocks can follow once the split abstraction is stable.

| Split type | Status | Implementation | Purpose |
|---|---|---|---|
| **Date block** | Implemented | Last 10% of sorted unique dates | Temporal leakage (single draw) |
| **Date k-fold** | **Implemented (2026-06-23)** | Block-rotation over dates (`np.array_split` → N contiguous blocks, hold each out) | General unseen-date robustness, **mean ± std** |
| Orbit block | Follow-on | Every Mth orbit | Within-day autocorrelation |
| Region block | Follow-on | Hold out a lat/lon tile (e.g., 30°×30°) | Spatial leakage |

Expose via `--val_split {random,date,date_kfold}`; for `date_kfold` pass `--n_folds N --fold K`
(one fold per invocation → distinct `--suffix` dir).  Orbit/region remain follow-on.

**Why date_kfold is the primary robustness probe.** A single trailing `date` block is one
draw; the ±0.02–0.05 R² margins TabM shows over GBDT/MLP on that block are within plausible
noise.  `date_kfold` tests every date exactly once across N folds → a lower-variance estimate.
It removes same-day leakage (like `date`) but allows training on later dates to predict earlier
ones — acceptable here because the corrector is applied across the mission record, not as a
forecaster (it is **not** rolling-origin).

**Seed sweep — all three model families, not just TabM.** Under `date`/`date_kfold` the split is
deterministic, so `--seed` varies only model-training stochasticity (init, batch order; GBDT
subsample). Sweep `s0/s1/s2` for TabM **and** XGB **and** MLP so every model carries an error bar
(`--seed` flows into `torch.manual_seed` / XGB `random_state`).

**Aggregation.** `python -m models.aggregate_folds --dirs '<...>/<run>_f*' --label <name> ...`
collects the per-fold/seed `global` blocks and stratified CSVs into mean ± std and a side-by-side
comparison table. Decision rule: if TabM's point-accuracy edge stays inside the fold band, pivot
the claim to calibrated uncertainty (cov90 + zero crossing, which MLP lacks and XGB does worse in
the left tail) rather than a point-accuracy win.

### Hard rule: pipeline fitted on train split only

For all validation modes, the `FeaturePipeline` must **never** see held-out data during `fit()`.
Fitting on the full dataset before splitting leaks test-set statistics (scaler means/variances,
quantile boundaries) into validation — for random splits this is already an issue; for blocked
splits it makes results uninterpretable.

Required order:

```python
train_df, heldout_df = split(df, mode=val_split)                          # split raw dataframe first
pipeline = FeaturePipeline.fit(train_df, sfc_type=sfc_type, ...)          # classmethod; fit on train only
X_train = pipeline.transform(train_df)
X_held  = pipeline.transform(heldout_df)                                  # transform with train-fitted pipeline
```

Persist the train-fitted pipeline alongside the model checkpoint (`tabm_pipeline.pkl`)
so that inference at deployment time uses the same scaler state as training.

---

## Uncertainty diagnostics (required, not optional)

Reported on both random and blocked test sets:

| Metric | Description |
|---|---|
| RMSE / MAE / R² | Standard point metrics |
| Pinball loss (q05, q50, q95) | Per-quantile — report separately for both loss modes |
| 90% empirical coverage | Fraction of y in [q05, q95] — should be ~0.90 for calibrated model |
| Mean interval width | E[q95 − q05] — narrower is better if coverage holds |
| Quantile crossing rate | Fraction of samples where q05 ≥ q95 — should be 0.0 with monotonic head |
| Member spread (std of K q50s) | From `return_members=True` — proxy for epistemic uncertainty |

All metrics also stratified by regime (if `plot_evaluation_by_regime` supports it):
- Cloud proximity (near-cloud / far-cloud)
- Aerosol load (AOD bins)
- Glint angle
- Footprint number
- Surface type (ocean / land)
- **Left-tail performance** — report both bottom 5% and bottom 10% of y; cloud-contaminated anomaly failures are concentrated in the extreme tail and global metrics will miss them

---

## Calibration success criteria (define before running)

Pass/fail thresholds to evaluate before comparing models. Chosen before experiments to prevent
post-hoc model selection by RMSE alone.

| Criterion | Threshold | Why |
|---|---|---|
| 90% interval coverage (global) | 0.87 – 0.93 | Miscalibration outside this range is practically significant |
| 90% coverage in near-cloud regime | ≥ 0.85 | Collapse here is the scientifically important failure mode |
| 90% coverage in high-AOD regime | ≥ 0.85 | Same |
| 90% coverage in left-tail (bottom 10%) | ≥ 0.80 | Looser; tail is harder, but collapse must not be total |
| Quantile crossing rate | 0.0 | Enforced by monotonic head — any non-zero value is a bug |
| Date-block RMSE vs random-split RMSE | Flag if ratio > 1.20 | A larger gap may indicate leakage *or* real temporal distribution shift — flag for investigation, not automatic failure |

A model that passes global coverage but fails stratified coverage should be reported as
uncalibrated in the regimes of interest, not as a calibrated model.

---

## Required baselines

FT-Transformer alone is insufficient for a journal-level comparison. The primary question is
whether any deep model adds value over strong GBDT baselines on this tabular dataset.

| Baseline | File | Library | Why required |
|---|---|---|---|
| **XGBoost** | `gbdt_baselines.py` | `xgboost ≥ 1.7` | Standard tree-ensemble benchmark; quantile regression via `objective="reg:quantileerror"` (added in 1.7 — version-check at import and fall back to LightGBM or sklearn `GradientBoostingRegressor(loss="quantile")` if unavailable on CURC) |
| **LightGBM** | `gbdt_baselines.py` | `lightgbm` | Faster GBDT alternative; native quantile via `objective="quantile"` |
| MLP baseline | `mlp_baseline.py` | PyTorch | Clean re-implementation of `result_ana.py` MLP in the shared scaffold |

Train each baseline on the same train split (post-`FeaturePipeline.transform`) and evaluate
with the same diagnostic suite (coverage, interval width, pinball loss, stratified metrics).
Report under both random and date-block splits.

CatBoost is optional — XGBoost + LightGBM give sufficient coverage unless a reviewer requests it.

**GBDT quantile crossing:** GBDTs train one model per quantile independently, so quantile
crossing (q05 ≥ q95 on some samples) is possible. The diagnostic suite must report the crossing
rate for GBDT outputs. As an optional post-hoc step, apply **monotone rearrangement** (sort the
three quantile outputs per sample) before computing interval-based metrics (coverage, width):

```python
# monotone rearrangement — sort per-sample: preds shape [N, 3] for (q05, q50, q95)
preds_rearranged = np.sort(preds_gbdt, axis=1)
```

Report metrics both before and after rearrangement so GBDT is not penalized for crossing when
the neural baseline has the structural guarantee via the monotonic head.

---

## Repeated seeds

Neural tabular results are seed-sensitive. Require at least **3 independent seeds** for:
- TabM K=16, random split, huber loss (primary result)
- TabM K=16, date-block split, huber loss (primary blocked result)
- FT-Transformer (same two splits, for fair comparison)

Report **mean ± std** (or 95% CI) for RMSE, MAE, 90% coverage, mean interval width, and
pinball loss (q05, q50, q95) in all tables. Single-seed results are acceptable for ablations
(K sweep, loss comparison) to limit compute, but must be labelled as such.

GBDT baselines can be made deterministic by fixing seeds and using deterministic histogram/threading
settings (e.g., `nthread=1`, `subsample=1.0` in XGBoost; `num_threads=1` in LightGBM), but are
not inherently deterministic under parallel execution or sampling. Use fixed seeds and deterministic
settings where available; a single GBDT run is acceptable only if repeated runs are empirically
stable (i.e., verify two independent runs match before relying on single-run results).

---

## Auxiliary cloud-proximity task (ablation)

The shared TabM trunk may become more physically cloud-aware if it must simultaneously predict
both the XCO2 anomaly and whether a sounding is near a cloud. This is an ablation, not the
default, because it is only useful if cloud distance is not already an input feature at
deployment time (see Leakage check below).

### Architecture

```
shared TabM trunk → h_members [batch, K, d_model]
├── anomaly head:         TabMLayer(h_members) → monotonic → mean over K
│                         q05 / q50 / q95 for xco2_bc_anomaly  (primary task)
└── cloud-proximity head: mean(h_members, dim=1) → [batch, d_model]
                          Linear(d_model, 1) → scalar logit     (auxiliary task)
```

### Cloud label

Two variants — try binary first; bins are more informative if binary shows improvement:

| Variant | Label | Notes |
|---|---|---|
| **Binary (default for ablation)** | `near_cloud = cld_dist_km ≤ 10.0` | Hard threshold; simpler to tune |
| Bins | `0–5 / 5–10 / 10–30 / >30 km` as 4-class CE | More physically informative; avoids threshold sensitivity |

### Loss

```python
loss = anomaly_loss + lambda_cloud * BCEWithLogitsLoss(cloud_logit, near_cloud)
```

- Start with `lambda_cloud = 0.1`; sweep 0.05, 0.1, 0.2 if the primary ablation shows signal
- Compute `pos_weight` from `train_df` only — same leakage discipline as the scaler:
  ```python
  n_neg = (train_near_cloud == 0).sum()
  n_pos = (train_near_cloud == 1).sum()
  pos_weight = torch.tensor(n_neg / n_pos)   # passed to BCEWithLogitsLoss
  ```
- Anomaly prediction is the primary task — `lambda_cloud` must keep anomaly loss dominant

### Leakage check (required before running this ablation)

This ablation is only valid in one of two clean cases:

1. **Cloud distance not available at prediction time** — auxiliary task is genuinely predictive
2. **Cloud distance available at prediction time as a feature** — simpler baseline is to include it directly; multi-task is still valid but less necessary

**Hard rule:** `cld_dist_km`, `near_cloud`, and any binned cloud-distance feature must be
**absent from `pipeline.features`** when running the auxiliary cloud head ablation. If they are
present, the model is learning to echo a target-derived input, not to build cloud-awareness
from other physical predictors. If you intentionally want to test the "direct cloud-distance
feature" baseline, run it as a separate named experiment, not as the auxiliary head ablation.

### Success criterion

**Primary:** Does the anomaly RMSE, 90% coverage, and pinball loss improve in the near-cloud
regime and the left-tail (bottom 5% / 10%) compared to the single-task TabM K=16 baseline?

**Not sufficient:** Cloud classification accuracy alone. If the model gets better at predicting
cloud proximity but the anomaly metrics do not improve where clouds matter, the ablation is
negative.

### Cloud-head diagnostics (secondary — compute after anomaly metrics)

| Diagnostic | Description |
|---|---|
| AUROC / AUPRC | Cloud classifier quality on heldout set |
| Precision / recall at 10 km threshold | Operational performance of the binary label |
| Near-cloud prevalence: train vs heldout | Check for distribution shift in the auxiliary label across splits |
| Corr(cloud logit, \|anomaly error\|) | Does higher cloud-proximity score predict larger anomaly residuals? |

These are informative but secondary. A strong AUROC with no anomaly improvement still means
the ablation is negative for the primary task.

---

## Feature set ablations

Three named input feature sets, applied identically across all models (TabM, FT-Transformer,
GBDT baselines). Each set is defined by dropping features from `_FEATURES_SFC{0,1}` in
`pipeline.py`. Add named constants so the active set is explicit and reproducible:

```python
# pipeline.py — add below _FEATURE_MAP
_FEATURE_SETS = {
    "full":    None,                          # sentinel: use _FEATURE_MAP[sfc_type] unchanged
    "no_xco2": {"drop": XCO2_FEATURES},
    "no_spec": {"drop": SPEC_FEATURES},
}
```

Pass via `--feature_set {full,no_xco2,no_spec}` CLI flag and persist the name in the
checkpoint meta dict so results are always traceable to a specific set.

### Set 1 — `full` (current `_FEATURES`, no change)

Default. All currently active features in `_FEATURES_SFC0` / `_FEATURES_SFC1`.

### Set 2 — `no_xco2` (drop xco2-derived features)

**Motivation:** `xco2_raw_minus_apriori` encodes retrieval-model bias in a way that may leak
target information or overfit to retrieval-specific artifacts. Dropping it tests whether the
model can predict anomalies from physical and spectroscopic predictors alone.

Features dropped (active in current sets):

| Feature | Both SFC types |
|---|---|
| `xco2_raw_minus_apriori` | yes |

Remaining active count: SFC0 → 26 features, SFC1 → 30 features (approximate; verify from
`pipeline.n_features` after transform).

### Set 3 — `no_spec` (drop k1/k2/k3 and exp_intercept/exp_intercept-alb)

**Motivation:** The k-basis and intercept terms are spectroscopic retrieval diagnostics that
may capture instrument/retrieval artifacts rather than physical signal. Dropping them tests
whether the spectroscopy retrieval features are necessary for anomaly prediction, and whether
removing them reduces tail failures in near-cloud regimes. Note: some remaining features
(SNR, continuum terms, band ratios) are still retrieval diagnostics — do not describe Set 3
as "purely meteorological/geometric" in any paper.

Features dropped (active in current sets):

| Feature | SFC0 | SFC1 |
|---|---|---|
| `exp_o2a_intercept` | yes | yes |
| `o2a_exp_intercept-alb` | yes | yes |
| `wco2_exp_intercept-alb` | — | yes |
| `o2a_k1`, `o2a_k2` | yes | yes |
| `wco2_k1`, `wco2_k2`, `wco2_k3` | yes | yes |
| `sco2_k1`, `sco2_k2` | yes | yes |

Remaining active count: SFC0 → 17 features, SFC1 → 21 features (approximate).

### Implementation note

`FeaturePipeline.fit()` already accepts a feature list via `_FEATURE_MAP`. Add a
`feature_set` argument that selects from `_FEATURE_SETS` and applies the drop before
fitting. The pipeline's `n_features` and `features` attributes remain the single
authoritative source — no other code needs to know which set is active.

---

## Ablation plan

Priority order for paper scope (run top-to-bottom; stop earlier if compute is limited):

| Priority | Ablation | Config change | Purpose |
|---|---|---|---|
| 1 | K=1 (degenerate MLP) | `K: 1` | BatchEnsemble-degenerate MLP — r/s params present but K=1; not identical to `result_ana.py` MLP |
| 2 | XGBoost baseline | — | Tabular tree-ensemble ceiling; see Required Baselines section |
| 3 | FT-Transformer | — | Neural transformer comparison; already implemented |
| 4 | **TabM K=16** | `K: 16` | **Primary result** |
| 5 | TabM K=8 | `K: 8` | Smaller ensemble |
| 6 | TabM K=32 | `K: 32` | Larger ensemble — diminishing returns? |
| 7 | Huber vs quantile loss | `loss: "quantile"` | Calibration impact of loss choice |
| 8 | Member input scaling | Per-member Linear in proj instead of shared | Does early input diversity help? |
| 9 | Monotonic head off | Raw head + crossing penalty | Architectural guarantee vs penalty |
| 10 | **Feature set: `no_xco2`** | `--feature_set no_xco2` | Drop `xco2_raw_minus_apriori`; test anomaly prediction from spectroscopic + physical features only |
| 11 | **Feature set: `no_spec`** | `--feature_set no_spec` | Drop all k1/k2/k3 and exp_intercept/exp_intercept-alb; test whether spectroscopy retrieval terms are necessary (note: remaining features are not all purely meteorological) |
| 12 | **Auxiliary cloud head (binary)** | `lambda_cloud=0.1`, `near_cloud = cld_dist_km ≤ 10` | Does cloud-awareness improve near-cloud anomaly calibration and left-tail coverage? See Auxiliary cloud-proximity task section. |
| 13 | Auxiliary cloud head (bins) | 4-class CE over 0–5/5–10/10–30/>30 km | More informative than binary threshold — run only if priority 12 shows improvement |

---

## Comparison: model family overview

| | XGBoost / LightGBM | FT-Transformer | TabM (this work) |
|---|---|---|---|
| Type | GBDT | Neural transformer | BatchEnsemble MLP |
| Feature representation | Native tabular | Per-feature token `[batch, F, d_token]` | Flat vector `[batch, K, d_model]` |
| Diversity mechanism | Boosted trees | Multi-head attention | K per-member r/s scalings |
| Quantile crossing | Not guaranteed (need isotonic post-proc) | Not guaranteed | Guaranteed via monotonic head |
| Ensemble spread | N/A | N/A | Member-level `[batch, K, 3]` available |
| Interpretability | SHAP / feature importance | Attention maps + permutation importance | Permutation importance |
| GPU required | No | Yes | Yes |
| Max practical batch size | N/A | ~1024–4096 (attention-limited) | ~8K–16K (benchmark first) |
| Convergence | Mostly deterministic when controlled | Slower (warmup needed) | Faster than FT-Transformer |
| Seed sensitivity | Low when controlled (verify empirically) | High | Moderate (report mean ± std) |

XGBoost and LightGBM are the primary baselines. TabM should beat or meaningfully complement them
to justify the neural overhead; beating only FT-Transformer is insufficient.

### Compute budget table (fill in after first runs)

TabM's primary motivation includes efficiency. Reviewers will expect evidence. Fill in wall-clock
times after the first complete training run on CURC; estimates below are placeholders.

| Model | File | ~Params | Hardware | Batch size | Wall time (500 ep) | Peak memory |
|---|---|---|---|---|---|---|
| XGBoost | `gbdt_baselines.py` | — | CPU | N/A | TBD | TBD |
| LightGBM | `gbdt_baselines.py` | — | CPU | N/A | TBD | TBD |
| MLP baseline | `mlp_baseline.py` | ~10K | GPU | 4096 | TBD | TBD |
| FT-Transformer | `transformer.py` | ~1–2M | GPU | 1024–4096 | TBD | TBD |
| TabM K=1 | `tabm.py` | ~500K | GPU | 8192 | TBD | TBD |
| TabM K=16 | `tabm.py` | ~500K + 16×(d_in+d_out) per layer | GPU | 8192 | TBD | TBD |
| TabM K=32 | `tabm.py` | ~500K + 32×… | GPU | 8192 | TBD | TBD |

---

## CLI usage (after implementation)

```bash
# Basic ocean run (random split, huber loss)
python -m models.tabm --sfc_type 0 --suffix v1_ocean

# Date-blocked validation
python -m models.tabm --sfc_type 0 --suffix v1_ocean_dateblock --val_split date

# K ablation sweep
python -m models.tabm --sfc_type 0 --suffix v1_k1  --config tabm_k1.json
python -m models.tabm --sfc_type 0 --suffix v1_k8  --config tabm_k8.json
python -m models.tabm --sfc_type 0 --suffix v1_k32 --config tabm_k32.json

# Quantile loss comparison
python -m models.tabm --sfc_type 0 --suffix v1_quantile --loss quantile

# Feature set ablations
python -m models.tabm --sfc_type 0 --suffix v1_no_xco2 --feature_set no_xco2
python -m models.tabm --sfc_type 0 --suffix v1_no_spec --feature_set no_spec

# Auxiliary cloud head
python -m models.tabm --sfc_type 0 --suffix v1_cloud_aux --aux_cloud --lambda_cloud 0.1
python -m models.tabm --sfc_type 0 --suffix v1_cloud_aux_bins --aux_cloud --cloud_label bins --lambda_cloud 0.1

# Resume from existing checkpoint
python -m models.tabm --sfc_type 0 --suffix v1_ocean   # can_load() picks up model_tabm_best.pt
```

Example config JSON (`tabm_k32.json`):
```json
{
  "model": {
    "K": 32,
    "d_model": 256,
    "n_layers": 4,
    "dropout": 0.2
  },
  "train": {
    "linux_batch_size": 8192,
    "patience": 50
  }
}
```

---

## Implementation phases

Build and verify one phase before starting the next. Do not implement ablations until the
core model produces sensible outputs under the correct train-only pipeline discipline.

Status legend: ✅ code complete + smoke-tested · ⏳ code ready, awaiting CURC run · ⬜ not started

| Phase | Deliverable | Done when | Status |
|---|---|---|---|
| **1. Core model + pipeline split** | `tabm.py` model, `FeaturePipeline.fit(train_df)` discipline, random-split training loop, basic RMSE/MAE/R² on held-out set | Loss curves converge; held-out RMSE is plausible | ✅ code; ⏳ real convergence run |
| **2. Calibration diagnostics** | 90% coverage, mean interval width, pinball loss per quantile, crossing rate, `return_members` spread | All metrics compute correctly on random split | ✅ (`diagnostics.py`, verified on synthetic) |
| **3. Blocked validation** | Date-block split; pipeline re-fitted on train only; same diagnostics as Phase 2 | Date-block results logged; RMSE ratio to random split within expected range | ✅ code (`--val_split date`); ⏳ logged results |
| **4. GBDT + MLP baselines** | `gbdt_baselines.py`, `mlp_baseline.py`; same split and diagnostics as TabM | All baselines evaluated on both random and date-block splits | ✅ code; ⏳ runs (LightGBM needs install) |
| **5. Feature set ablations** | `--feature_set no_xco2` and `--feature_set no_spec` runs for TabM K=16 | Metric delta vs `full` set documented | ✅ code (`--feature_set`); ⏳ delta table |
| **6. Architecture ablations** | K sweep (1/8/32), loss comparison, member input scaling | Results table complete | ✅ K-sweep (`--K`) + loss (`--loss`); ⬜ `ablation_member_input_scale` (priority 8) and monotonic-head-off + crossing-penalty (priority 9) toggles not yet implemented |
| **7. Auxiliary cloud head** | `--aux_cloud` ablation; cloud-head diagnostics; anomaly metrics in near-cloud and left-tail regimes | Positive/negative result clearly reported | ✅ code (binary + bins, leakage check); ⏳ result + cloud-head AUROC/AUPRC diagnostics |

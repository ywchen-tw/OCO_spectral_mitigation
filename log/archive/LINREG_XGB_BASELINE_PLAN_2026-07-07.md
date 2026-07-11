# Linear-regression & XGBoost baseline comparison vs production DE — implementation plan

> **ARCHIVED 2026-07-09 — plan executed.** The Ridge + XGBoost-mean baselines
> ran to completion: DE > XGBoost-mean > Ridge, decided in the near-cloud land
> tail (fp-RMSE 1.30 < 1.68 < 2.37). Results:
> `results/model_comparison/MODEL_COMPARISON_manuscript_DE_XGB_LinReg.md` and
> the kfold aggregates; status ledger in `TODO_ACCOMPLISH.md` §1.

**Date:** 2026-07-07 · **Status:** PLAN (executed; see banner) · **For:** implementing agent
**Companion docs:** `PROJECT_REVIEW.md` (§2.5 baselines row), `TCCON_PAPERS_AND_ML_IDEAS_2026-07-03.md` §2.5/§2.8-item-7, `TCCON_BIAS_MODEL_IMPROVEMENT_PLAN_2026-07-06.md` (metric caveats + regenerated model table).

## Goal

Reviewer-proof baseline table: quantify what the production deep ensemble buys
over (a) **regularized linear regression** and (b) **XGBoost mean regression**,
trained on the *identical* protocol — feature_set `full` + fold-specific
ProfilePCA, **ocean r05 / land r15 targets**, date_kfold 5 folds, 2016–2020
combined parquet — and judged by the *same* TCCON chain (r100 primary / r50
robustness, AK-harmonized + direct, cloud-distance-grouped near/far split).

Success = the table exists, whatever it shows. Expected (from prior evidence:
TabM≈DE globally but DE wins the near-cloud land tail; profile-EOF gains
concentrated in the tail): baselines competitive on global/far-cloud numbers,
DE clearly ahead on near-cloud land fp-RMSE. If XGBoost ties DE on TCCON,
report it honestly as a parsimony option — the manuscript claim is the
*correction*, not the architecture.

## Fixed design decisions (do not relitigate)

1. **Targets:** ocean `--target 5km` (xco2_bc_anomaly_r05), land `--target 15km`
   (xco2_bc_anomaly_r15) — matches production DE tag `o05l15`.
2. **Fold-specific ProfilePCA** (leakage-safe), same artifacts as the DE
   foldpca retrain: `${STORAGE}/results/profile_pca_date_kfold_2016_2020/fold{F}/profile_pca_{ocean,land}.pkl`.
   The comparison partner is therefore the **foldpca DE**
   (`de_{ocean,land}_beta_nll_prof_reg_foldpca_{r05,r15}_f*`, launchers
   committed 2026-07-07) once its CURC arrays finish — same-protocol
   comparison. If it hasn't finished, compare against current production
   (`_prof_reg_`) but flag the PCA-protocol difference in the writeup.
3. **No HPO.** Lesson recorded in memory (`tabm-hpo-datekfold-lesson`): the
   landscape is flat and single-date/even-correct HPO didn't beat defaults.
   XGBoost: library defaults + `tree_method=hist`, `n_estimators`/early
   stopping on the calibration block only. Linear: Ridge with a small alpha
   grid ({0.1, 1, 10, 100}) selected on the calibration block only. NEVER
   tune on TCCON.
4. **Per-sounding only** (both models trivially satisfy the per-footprint
   deployment constraint — no cloud info, no neighbors at inference).
5. **Correction form:** `corrected = xco2_bc − mu`, same guards as DE plot
   data (climatology 50 ppm, |mu| 25 ppm, kept+flagged).
6. **Splits:** `models.splits.split_dataframe` (mode `date_kfold`, n_folds 5,
   fold F, seed default 42); calib block carved with mode `date`,
   `calib_frac 0.15` — identical to `deep_ensemble.py` / `profile_pca_kfold`.
7. **Reports:** `workspace/tccon_comparison_report.py` post wet/dry fix
   (commit `07822b3`) — model-agnostic via `--corr-col`; always
   `--ak-harmonize --cld-edges 0,10,inf --exclude-sites ny`.

## Existing assets (reuse, don't rewrite)

| asset | what it gives | gaps to close |
|---|---|---|
| `src/models/gbdt_baselines.py` | XGBoost with `--val_split date_kfold --fold`, `--feature_set`, `--target`, `--profile-pca` (accepts a PKL PATH — passes through to `FeaturePipeline.fit`, verified), artifact dir per `--suffix` | trains 3 *quantile* models (q05/q50/q95, `reg:quantileerror`); needs a **mean mode** (see A1.2) or an explicit q50-as-mu decision |
| `src/models/mlp_baseline.py` | pattern for a minimal baseline module on `train_common` | n/a (pattern only) |
| `src/models/pipeline.py` `FeaturePipeline` | feature selection + quantile scaling + profile-PCA block + pkl round-trip (`_resolve_profile_pca` accepts paths) | none |
| `src/models/aggregate_folds.py` | cross-fold CV aggregate tables | none |
| `workspace/build_tabm_plot_data.py` | the adapter pattern for non-DE models: pool fold dirs → per-date `plot_data.parquet` with `<model>_corrected_xco2` + guard flags (`--ocean-model-dir/--land-model-dir`, clim/anomaly guards, `--correction-base bc`) | written for TabM checkpoints; needs a baseline twin (A3.1) |
| `curc_shell_blanca_plot_corr_xco2_tabm.sh` | full TCCON validation launcher pattern for a non-DE model (MODEL_TAG-namespaced OUT_BASE, `--corr-col`, r100/r50 via `RADIUS_KM`) | copy + retag (A4.1) |
| `curc_shell_blanca_de_profile_foldpca_r05/r15.sh` | fold-array training launcher pattern with fold-PCA pkl guard | copy + swap module/flags (A2.x) |

## Action items

### Phase 1 — training modules

- [ ] **A1.1 `src/models/linear_baseline.py`** (new). Ridge regression on the
  standard `FeaturePipeline` features (quantile-transformed continuous +
  fp one-hots + profile-PCA block). CLI mirroring `deep_ensemble.py`:
  `--sfc_type --suffix --target --feature_set --profile-pca <path|auto>
  --val_split date_kfold --n_folds --fold --calib_frac` (+ `--alphas`,
  default `0.1,1,10,100`, selected on the calib block by RMSE).
  Artifacts per run dir `results/model_linear_baseline/<suffix>/`:
  `deep_ensemble_pipeline.pkl`-equivalent (`FeaturePipeline.save`),
  `model.pkl` (or `coef.json` + intercept), `training_dates.json`
  (same schema as DE — feeds the leakage guard), `metrics.json`
  (held-fold global + near-cloud/tail-5% slices via `models.diagnostics`).
  Keep it <300 lines; no torch.
- [ ] **A1.2 `gbdt_baselines.py` mean mode.** Add
  `--objective mean` (single `reg:squarederror` model, prediction = mu;
  early stopping on the calib block) alongside the existing quantile mode,
  writing the same artifact layout. Alternative accepted if simpler:
  document q50-as-mu and reuse the quantile artifacts — but the mean mode is
  preferred (cleaner story: "plain XGBoost regression").
- [ ] **A1.3** Both modules write `training_dates.json` and record the
  resolved profile-PCA pkl path in their run summary (reproducibility).

### Phase 2 — CURC training launchers (pattern: `curc_shell_blanca_de_profile_foldpca_r05.sh`)

- [ ] **A2.1** `curc_shell_blanca_linreg_foldpca_r05.sh` (ocean, `--sfc_type 0
  --target 5km`) and `_r15.sh` (land, `--sfc_type 1 --target 15km`);
  suffixes `linreg_{ocean,land}_full_prof_foldpca_{r05,r15}_f${F}`;
  array 0-4; **CPU-only** (drop `--gres=gpu`, keep mem ≥ 96G — the parquet
  load dominates), time 4 h is plenty.
- [ ] **A2.2** `curc_shell_blanca_xgb_foldpca_r05.sh` / `_r15.sh`; suffixes
  `xgb_{ocean,land}_full_prof_foldpca_{r05,r15}_f${F}`; CPU
  (`tree_method=hist`), mem 128G, time 8 h.
- [ ] **A2.3** Both launchers guard on the fold-PCA pkl (copy the guard block)
  and on the r05/r15 target columns existing in the parquet.
- [ ] **A2.4** After arrays: `models.aggregate_folds` per model × surface →
  `results/model_comparison/{linreg,xgb}_{ocean,land}_foldpca_{r05,r15}_kfold_agg.md`.

### Phase 3 — plot-data adapter

- [ ] **A3.1 `workspace/build_baseline_plot_data.py`** modeled on
  `build_tabm_plot_data.py`: load the 5 fold dirs per surface
  (`--ocean-model-dir .../linreg_ocean_full_prof_foldpca_r05_f*` etc.),
  predict mu per fold and **average across folds** (same pooling philosophy
  as the DE 25-member pool; these models have no members), apply the
  clim-50/|mu|-25 guards (kept + `is_guarded` flag), write per-date
  `plot_data.parquet` with `linreg_corrected_xco2` / `xgb_corrected_xco2`.
  NOTE: no sigma columns exist — the writer and the downstream report must
  tolerate that (the report only needs `--corr-col`; the uncertainty block
  is DE-only and simply skips).
- [ ] **A3.2** Support `--model-kind linreg|xgb` (or two thin wrappers) rather
  than duplicating the file twice.

### Phase 4 — TCCON validation

- [ ] **A4.1** Copy `curc_shell_blanca_plot_corr_xco2_tabm.sh` →
  `..._linreg.sh` / `..._xgb.sh`: `MODEL_TAG=linreg_prof_foldpca_o05l15` /
  `xgb_prof_foldpca_o05l15`,
  `OUT_BASE=$DATA_ROOT/results/model_comparison/{linreg,xgb}/${MODEL_TAG}`,
  step-4 generator → `build_baseline_plot_data.py`, report `--corr-col
  {linreg,xgb}_corrected_xco2`. Same 128 `run_case` lines (copy from the
  deepens launcher or `--script` it), `--ak-harmonize`, `--cld-edges
  0,10,inf`, `--exclude-sites ny`. Run r100, then `RADIUS_KM=50` rerun.
- [ ] **A4.2** Verify case coverage matches DE (75 pre-drift cases; the by-cld
  aggregate should resolve to the same 108 case×bin rows).

### Phase 5 — comparison writeup

- [ ] **A5.1** Extend the "Regenerated results" table in
  `TCCON_BIAS_MODEL_IMPROVEMENT_PLAN_2026-07-06.md` with linreg + xgb rows
  (AK + direct, r100 + r50).
- [ ] **A5.2** Near/far-cloud table (from `tccon_comparison_by_cld_agg_*.csv`):
  the manuscript-relevant cell is **near-cloud (0–10 km) land fp-RMSE** —
  DE's expected win. Include raw → before → after progressions.
- [ ] **A5.3** CV-side table from the fold aggregates (global + tail slices),
  clearly labeled date_kfold. One paragraph: does CV ranking survive TCCON?
  (Precedent: held-out R² over-credits contam features; TCCON is the arbiter.)

## Gotchas for the implementing agent

- The combined parquet must be the **dual-fit (nosg) build** and contain
  `xco2_bc_anomaly_r05`/`_r15`; `pipeline._USE_NOSG_K` governs k-features.
- Snow DATA stays included by default (`--exclude-snow` exists; don't use it).
- Feature sets: only `full` is in scope; ablations are NOT part of this plan.
- Do not add `cld_dist_km` (or any cloud/neighbor info) as a feature — label,
  weighting, and evaluation only (per-footprint deployment constraint).
- Linear model note: features are quantile-transformed by `FeaturePipeline`,
  so "linear regression" here means linear in the *transformed* features —
  state this in the writeup (it is the honest apples-to-apples choice, the
  same representation every other model consumes).
- XGBoost determinism: fix seeds, note runs may not be bit-reproducible under
  parallel hist; record library version in the run summary.
- AK numbers must come from the post-fix `ak_harmonize.py` (commit `07822b3`);
  never quote pre-2026-07-07 AK-referenced tables.
- Guard rates: report `n_guarded` for the baselines — a linear model may emit
  large |mu| more often; if guard rates differ wildly from DE, say so.
- CURC repo path: `/projects/yuch8913/OCO_spectral_mitigation`; launchers use
  `qos=preemptable` + `--requeue`; storage root via `utils.get_storage_dir()`.

## Acceptance criteria

1. 5-fold artifacts exist for 2 models × 2 surfaces with `training_dates.json`.
2. `aggregate_folds` CV tables exist (global + tail slices).
3. TCCON r100 + r50 reports exist under
   `results/model_comparison/{linreg,xgb}/<MODEL_TAG>/` with the same 75-case
   coverage and both references.
4. The improvement-plan comparison table has linreg + xgb rows and a near-cloud
   land fp-RMSE column, with a 3-sentence interpretation.

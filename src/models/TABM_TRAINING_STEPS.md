# TabM model suite — training steps (local test → CURC)

> **STATUS (2026-07-06): TabM is NOT the production model.** The 2026-07-05
> head-to-head (validation + 70-case TCCON, `results/model_comparison/tabm_vs_de_*`,
> `tabm/tabm_prof_m16/`) put TabM ≈ DE overall, with DE clearly better in the
> near-cloud land tail — the per-surface deep ensemble (`de_*_beta_nll_prof_reg`)
> stays production. TabM is kept as a maintained comparison model. Note also:
> `src/models/transformer.py` (referenced below) was deleted 2026-07-03 — its
> eval utilities now live in `src/models/tabm_eval.py`.

Step-by-step plan to validate the TabM / GBDT / MLP-baseline code locally, then
run the real experiments on CURC (Blanca). Plan reference: `src/models/TABM_PLAN.md`.

## Key facts (read first)

- **Always launch from the repo root with `src` on `PYTHONPATH`:**
  ```bash
  cd /Users/yuch8913/programming/oco_fp_analysis     # local  (CURC: /projects/yuch8913/OCO_spectral_mitigation)
  PYTHONPATH=src python -m models.<name> ...
  ```
  Reason: `get_storage_dir()` is **cwd-relative** (`./results/...`), so cwd must be
  the repo root; and the modules mix relative (`from .pipeline`) + absolute
  (`from utils`, `from search.tracking`) imports, so they must run as `python -m`.
- **Data** lives in `results/csv_collection/`. On macOS the default is
  `combined_2020-02-01_all_orbits.parquet` (single date); on Linux/CURC it is
  `combined_2016_2020_dates.parquet` (multi-date, 12 GB).
- **`--val_split date` cannot be tested locally** — the local single-date files
  have only 1 unique date (date-block needs ≥2). Test `date` on CURC only.
- **`--val_split date_kfold` (block-rotation k-fold; 2026-06-23)** is the primary
  unseen-date robustness probe — every date is test once across N folds → mean ± std.
  One fold per invocation: `--val_split date_kfold --n_folds N --fold K` into a distinct
  `--suffix ..._f${K}` dir; aggregate with `python -m models.aggregate_folds`. Supported by
  tabm / gbdt / mlp_baseline; CURC only (needs multi-date data). Seed sweeps (`--seed`, all
  three models) measure training stochasticity only under date/date_kfold (deterministic split).
- Outputs land in `results/model_tabm/<suffix>/`, `results/model_gbdt/<suffix>/`,
  `results/model_mlp_baseline/<suffix>/`.
- **Target outlier filter (2026-06-23):** `filter_target_outliers()` (`pipeline.py`,
  `MAX_ABS_ANOMALY_PPM = 100`) drops `|xco2_bc_anomaly| > 100 ppm` rows from the raw
  dataframe before the split, in tabm / gbdt / mlp_baseline. No-op on the single-date local
  file; only cleans the multi-date CURC data, where unfiltered land extremes produced
  RMSE≈30 / R²≈0 (MAE stayed ~0.5). **Any land run made before this date is invalid — re-run.**

---

## Stage 0 — pure unit/smoke test (no data, seconds)

Confirms the model math, monotonic head, members, adapter round-trip, diagnostics.

```bash
cd /Users/yuch8913/programming/oco_fp_analysis
PYTHONPATH=src python - <<'PY'
import numpy as np, torch
from models.tabm import TabM, _tabm_predict
import models.diagnostics as diag
torch.manual_seed(0); np.random.seed(0)
X = np.random.randn(400, 12).astype('float32')
y = (X[:,0]*0.5 - np.abs(X[:,1])).astype('float32')
m = TabM(n_features=12, K=8, d_model=32, n_layers=2); m.eval()
preds, members = _tabm_predict(m, X, batch_size=128, want_members=True)
g = diag.compute_metrics(y, preds, members=members)
assert g['crossing_rate'] == 0.0, "monotonic head should never cross"
print("Stage 0 OK:", {k: round(v,3) for k,v in g.items() if k in ('rmse','coverage_90','crossing_rate')})
PY
```
Expected: prints `Stage 0 OK: ...` with `crossing_rate` = 0.0.

---

## Stage 0.5 — (optional) fit / inspect the FeaturePipeline standalone

The training modules already fit their own `FeaturePipeline` **on the train split**
and save it inline, so this stage is **optional**. Use it to (a) sanity-check the
feature set / `n_features` for a surface type before training, or (b) pre-fit a
pipeline once to reuse across runs via `--pipeline`.

CLI flags: `--sfc-type {0,1}`, `--feature-set {full,no_xco2,no_spec}`,
`--scaler {robust_standard,pca_whitening}`, `--pca-augment`, `--data`, `--out`
(default `results/model_mlp_lr/<suffix>/pipeline.pkl`).

> ⚠️ **Leakage caveat.** `pipeline.py` fits on the **entire file** you pass it. If you
> then hand that pickle to a model via `--pipeline` under a **blocked (`date`) split**,
> the held-out dates' scaler/quantile statistics leak into the pipeline → results are
> uninterpretable. For blocked-split validation, **let each model fit its own pipeline
> on the train split** (the default — just omit `--pipeline`). Only reuse a pre-fit
> pipeline for random-split experiments or pure inference, and even then inline fitting
> is the cleaner discipline.

Pass a saved pipeline to any model with `--pipeline results/pipelines/pipeline_ocean_full.pkl`
(supported by `models.tabm`, `models.gbdt_baselines`, `models.mlp_baseline`).

### 0.5a — local Mac (all feature-set × sfc-type combinations)

Uses the full multi-date dataset (same file as CURC). Run from the repo root.

```bash
conda activate ml310
cd /Users/yuch8913/programming/oco_fp_analysis
mkdir -p results/pipelines

DATA=results/csv_collection/combined_2016_2020_dates.parquet

# ── ocean (sfc_type=0) ────────────────────────────────────────────────────────
PYTHONPATH=src python -m models.pipeline \
  --data $DATA --sfc-type 0 --feature-set full \
  --out results/pipelines/pipeline_ocean_full.pkl

PYTHONPATH=src python -m models.pipeline \
  --data $DATA --sfc-type 0 --feature-set no_xco2 \
  --out results/pipelines/pipeline_ocean_no_xco2.pkl

PYTHONPATH=src python -m models.pipeline \
  --data $DATA --sfc-type 0 --feature-set no_spec \
  --out results/pipelines/pipeline_ocean_no_spec.pkl

# ── land (sfc_type=1) ─────────────────────────────────────────────────────────
PYTHONPATH=src python -m models.pipeline \
  --data $DATA --sfc-type 1 --feature-set full \
  --out results/pipelines/pipeline_land_full.pkl

PYTHONPATH=src python -m models.pipeline \
  --data $DATA --sfc-type 1 --feature-set no_xco2 \
  --out results/pipelines/pipeline_land_no_xco2.pkl

PYTHONPATH=src python -m models.pipeline \
  --data $DATA --sfc-type 1 --feature-set no_spec \
  --out results/pipelines/pipeline_land_no_spec.pkl
```

Expected `n_features` (ocean, `combined_2016_2020_dates.parquet`):

| `--feature-set` | qt features | fp cols | total |
|---|---|---|---|
| `full`    | 27 | 8 | 35 |
| `no_xco2` | 26 | 8 | 34 |
| `no_spec` | 18 | 8 | 26 |

`pipeline.n_features` / `pipeline.features` are the authoritative source — never hard-code counts.

### 0.5b — CURC (Blanca): SLURM script

The script is `curc_shell_blanca_train_pipeline.sh`. Submit with:

```bash
cd /projects/yuch8913/OCO_spectral_mitigation
sbatch curc_shell_blanca_train_pipeline.sh
```

Output files (same paths on both local and CURC):

| File | sfc_type | feature-set |
|---|---|---|
| `results/pipelines/pipeline_ocean_full.pkl`    | 0 | all features |
| `results/pipelines/pipeline_ocean_no_xco2.pkl` | 0 | drop `xco2_raw_minus_apriori` |
| `results/pipelines/pipeline_ocean_no_spec.pkl` | 0 | drop k1/k2/k3 + exp\_intercept |
| `results/pipelines/pipeline_land_full.pkl`     | 1 | all features |
| `results/pipelines/pipeline_land_no_xco2.pkl`  | 1 | drop `xco2_raw_minus_apriori` |
| `results/pipelines/pipeline_land_no_spec.pkl`  | 1 | drop k1/k2/k3 + exp\_intercept |

---

## Stage 1 — local end-to-end test on REAL data (random split, fast)

Runs the full pipeline → split → train → diagnostics path on one real single-date
file (`combined_2020-02-01_all_orbits.parquet`, ~33k valid ocean rows). Uses the
**local smoke config** (K=4, 3 epochs) for TabM so it finishes in seconds.

> Each model fits its own `FeaturePipeline` on the train split and saves it inline
> (this is what the pre-test validated — earlier this failed to pickle under
> `python -m`; fixed in `pipeline.py`, see TABM_PLAN.md "Bug fix"). You do **not**
> need to pre-fit a pipeline or pass `--pipeline`.

**✅ Verified 2026-06-07 (macOS/MPS) — expected results:**

| Step | Command suffix | Time | RMSE | R² | crossing |
|---|---|---|---|---|---|
| 1a TabM (smoke) | `local_tabm_ocean` | ~33 s | 0.55 | 0.32 | 0 |
| 1c XGBoost (full) | `local_gbdt_ocean` | ~17 s | 0.285 | 0.82 | 0 |
| 1b MLP (100 ep) | `local_mlp_ocean` | ~30 s | 0.224 | 0.89 | n/a |

(TabM R² is low only because the smoke config trains 3 epochs — that is fine for a
code check; bump epochs for a fair local comparison. The non-zero takeaway is that
all three complete and TabM `crossing_rate == 0`.)

### 1a. TabM (ocean), fast config
```bash
cd /Users/yuch8913/programming/oco_fp_analysis
PYTHONPATH=src python -m models.tabm \
  --sfc_type 0 --suffix local_tabm_ocean \
  --config src/models/configs/tabm_local_smoke.json
```
Check it produced, under `results/model_tabm/local_tabm_ocean/`:
- `model_tabm_best.pt`, `tabm_meta.pkl`, `tabm_pipeline.pkl`
- `tabm_random_metrics.json` (look at `global.crossing_rate` → must be `0.0`)
- `tabm_random_stratified_metrics.csv`, `tabm_permutation_importance.{csv,png}`
- `run_summary.json`

### 1b. MLP baseline (point predictor)
```bash
PYTHONPATH=src python -m models.mlp_baseline --sfc_type 0 --suffix local_mlp_ocean
```
(Note: interval metrics are degenerate by design — only RMSE/MAE/R² are meaningful.)

### 1c. GBDT baseline (XGBoost, CPU)
```bash
PYTHONPATH=src python -m models.gbdt_baselines --model xgboost --sfc_type 0 --suffix local_gbdt_ocean
```
Check `results/model_gbdt/local_gbdt_ocean/` has both `xgboost_random_*` and
`xgboost_random_rearranged_*` metric files (crossing reported before/after sort).

### 1d. Feature-set + aux-cloud smoke (optional)
```bash
PYTHONPATH=src python -m models.tabm --sfc_type 0 --suffix local_no_spec \
  --config src/models/configs/tabm_local_smoke.json --feature_set no_spec
PYTHONPATH=src python -m models.tabm --sfc_type 0 --suffix local_aux \
  --config src/models/configs/tabm_local_smoke.json --aux_cloud --lambda_cloud 0.1
```

✅ **Gate:** all of Stage 1 must complete without error and TabM `crossing_rate==0`
before submitting CURC jobs.

Clean up the local test artifacts once inspected:
```bash
rm -rf results/model_tabm/local_* results/model_gbdt/local_* results/model_mlp_baseline/local_* results/pipelines/local_*
```

---

## Stage 1.5 — pre-flight check of the CURC scripts

The script `curc_shell_blanca_preflight.sh` runs the pre-flight checks on CURC
(syntax validation + command preview). Submit it before the real training jobs:

```bash
cd /projects/yuch8913/OCO_spectral_mitigation
sbatch curc_shell_blanca_preflight.sh
```

It checks these four scripts:
- `curc_shell_blanca_train_pipeline.sh`
- `curc_shell_blanca_train_tabm.sh`
- `curc_shell_blanca_train_gbdt.sh`
- `curc_shell_blanca_train_mlp_baseline.sh`

What to confirm in the output (`sbatch-output_oco_preflight_<JOBID>.txt`):
- All scripts print `syntax OK: <name>` — no `SYNTAX ERROR` lines.
- The previewed `python` commands match the forms run in Stage 1 (the scripts add
  `cd <repo-root>` + `export PYTHONPATH=src` for you).
- The active training runs use the **full** config (no `--config tabm_local_smoke.json`)
  and include the `--val_split date` variants — date-block runs only work on CURC
  (multi-date `combined_2016_2020_dates.parquet`), which is why they were skipped locally.

---

## Stage 2 — CURC (Blanca) full runs

On CURC the modules pick up the multi-date `combined_2016_2020_dates.parquet`
(Linux defaults), 500 epochs, batch 8192. Submit with `sbatch` from the repo root.

### 2a. Configure and submit

Each script ships with a default set of active (uncommented) runs. Edit the script
before submitting to enable/disable blocks. Active = uncommented; disabled = prefixed with `# `.

**`curc_shell_blanca_train_tabm.sh`** — currently active:
```
python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k16_random  --K 16
python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k16_date    --K 16 --val_split date
```
Available blocks to uncomment as needed:
```
# python -m models.tabm --sfc_type 1 --suffix tabm_land_k16_random  --K 16
# python -m models.tabm --sfc_type 1 --suffix tabm_land_k16_date    --K 16 --val_split date

# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k1   --K 1
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k8   --K 8
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k32  --K 32

# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_quantile --K 16 --loss quantile

# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_no_xco2 --K 16 --feature_set no_xco2
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_no_spec --K 16 --feature_set no_spec

# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_aux      --K 16 --aux_cloud --lambda_cloud 0.1
# python -m models.tabm --sfc_type 0 --suffix tabm_ocean_aux_bins --K 16 --aux_cloud --cloud_label bins --lambda_cloud 0.1

# for S in 0 1 2; do
#   python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k16_random_s${S} --K 16 --seed ${S}
#   python -m models.tabm --sfc_type 0 --suffix tabm_ocean_k16_date_s${S}   --K 16 --seed ${S} --val_split date
# done
```

**`curc_shell_blanca_train_gbdt.sh`** — currently active:
```
python -m models.gbdt_baselines --model xgboost --sfc_type 0 --suffix gbdt_ocean_xgb_random
python -m models.gbdt_baselines --model xgboost --sfc_type 0 --suffix gbdt_ocean_xgb_date   --val_split date
```
Available blocks to uncomment as needed:
```
# python -m models.gbdt_baselines --model xgboost  --sfc_type 1 --suffix gbdt_land_xgb_random
# python -m models.gbdt_baselines --model xgboost  --sfc_type 1 --suffix gbdt_land_xgb_date   --val_split date

# python -m models.gbdt_baselines --model lightgbm --sfc_type 0 --suffix gbdt_ocean_lgbm_random
# python -m models.gbdt_baselines --model lightgbm --sfc_type 0 --suffix gbdt_ocean_lgbm_date  --val_split date

# python -m models.gbdt_baselines --model xgboost  --sfc_type 0 --suffix gbdt_ocean_xgb_no_xco2 --feature_set no_xco2
# python -m models.gbdt_baselines --model xgboost  --sfc_type 0 --suffix gbdt_ocean_xgb_no_spec --feature_set no_spec
```
> LightGBM first time only: `pip install lightgbm` inside the `data` env.

**`curc_shell_blanca_train_mlp_baseline.sh`** — currently active:
```
python -m models.mlp_baseline --sfc_type 0 --suffix mlp_ocean_random
python -m models.mlp_baseline --sfc_type 0 --suffix mlp_ocean_date   --val_split date
```
Available blocks to uncomment as needed:
```
# python -m models.mlp_baseline --sfc_type 1 --suffix mlp_land_random
# python -m models.mlp_baseline --sfc_type 1 --suffix mlp_land_date   --val_split date

# python -m models.mlp_baseline --sfc_type 0 --suffix mlp_ocean_no_xco2 --feature_set no_xco2
# python -m models.mlp_baseline --sfc_type 0 --suffix mlp_ocean_no_spec --feature_set no_spec
```

Submit after editing:
```bash
cd /projects/yuch8913/OCO_spectral_mitigation
sbatch curc_shell_blanca_train_pipeline.sh       # CPU: fit all 6 feature-pipeline pickles
sbatch curc_shell_blanca_train_tabm.sh           # GPU: TabM
sbatch curc_shell_blanca_train_gbdt.sh           # CPU: XGBoost / LightGBM
sbatch curc_shell_blanca_train_mlp_baseline.sh   # GPU: MLP baseline
```

### 2b. Monitor
```bash
squeue -u $USER
tail -f sbatch-output_oco_train_tabm_*.txt
cat gpu_monitor_<JOBID>.csv          # GPU utilisation log (GPU jobs only)
```

### 2c. Recommended experiment order (phase-gated, per TABM_PLAN.md)

Run top-to-bottom; stop earlier if compute is limited. Edit each script to
uncomment the block you want, or copy the command and change `--suffix`.

| Order | Command (add to the relevant script) | Purpose |
|---|---|---|
| 1 | `... gbdt_baselines --model xgboost --sfc_type 0 --suffix gbdt_ocean_xgb_random` (+ `--val_split date`) | GBDT ceiling (the bar to beat) |
| 2 | `... mlp_baseline --sfc_type 0 --suffix mlp_ocean_random` (+ `--val_split date`) | point-MLP baseline |
| 3 | `... tabm --sfc_type 0 --suffix tabm_ocean_k16_random --K 16` (+ `--val_split date`) | **primary result** |
| 4 | `... tabm --K 1 / --K 8 / --K 32` | K-sweep ablation |
| 5 | `... tabm --loss quantile` | loss comparison |
| 6 | `... tabm --feature_set no_xco2 / no_spec` | feature-set ablations |
| 7 | `... tabm --aux_cloud --lambda_cloud 0.1` | auxiliary cloud head |
| 8 | repeat 3 with `--seed 0/1/2` (random + date) | seed mean ± std (primary) |

Repeat the key rows with `--sfc_type 1` for land.

### 2d. Collect results
Each run writes `results/model_*/<suffix>/`:
- `*_metrics.json` — global RMSE/MAE/R², coverage_90, interval width, per-quantile
  pinball, crossing_rate, member_spread (TabM), plus the calibration pass/fail block.
- `*_stratified_metrics.csv` — per-regime (cloud proximity, AOD, glint, footprint,
  surface, left-tail) metrics.
- `run_summary.json` — one-line tracked summary (RunSummary).

Pull the `global` blocks across suffixes into the comparison + compute-budget
tables in `TABM_PLAN.md`.

### 2e. Run status — 2026-06-23 batch

First full CURC batch synced to `results/`. Status:

- **28 runs completed cleanly** (all `crossing_rate == 0`). Ocean result holds:
  TabM > MLP > GBDT (e.g. `tabm_ocean_k16_random` R²=0.821 vs `mlp_ocean_random` 0.672,
  `gbdt_ocean_xgb_random` 0.577); K barely matters (K=1→32 all ≈0.81); `no_xco2` hurts a lot,
  `no_spec` barely; date-split drops all to ≈0.51–0.54 R².
- **All `*_land_*` runs are INVALID** — produced before the 100 ppm target filter
  (RMSE≈30, R²≈0). **Re-run every land suffix** for tabm / gbdt / mlp_baseline.
- **8 TabM runs incomplete** — only `tabm_pipeline.pkl` + `tabm_run_config.json` written
  (no `model_tabm_best.pt` / metrics): the seed sweep `tabm_ocean_k16_{random,date}_s{0,1,2}`
  and the aux-cloud runs `tabm_ocean_aux`, `tabm_ocean_aux_bins`. Job was cut (likely
  walltime) after the main + ablation blocks. **Re-submit these blocks**; check the SLURM log.
- Ocean re-run is optional (filter is a no-op there — no >100 ppm ocean outliers), but
  re-running guarantees identical filtering provenance across the matrix.

---

## Quick reference — what each `--suffix` should look like

```
tabm_ocean_k16_random   tabm_ocean_k16_date     tabm_ocean_k1 / k8 / k32
tabm_ocean_quantile     tabm_ocean_no_xco2      tabm_ocean_no_spec
tabm_ocean_aux          tabm_ocean_k16_random_s0/_s1/_s2
gbdt_ocean_xgb_random   gbdt_ocean_xgb_date     gbdt_ocean_lgbm_random
mlp_ocean_random        mlp_ocean_date
```
(swap `ocean`→`land` and `--sfc_type 0`→`1` for land.)

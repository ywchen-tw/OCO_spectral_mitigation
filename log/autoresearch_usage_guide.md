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

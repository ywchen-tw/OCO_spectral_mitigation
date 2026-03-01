# Transformer Upgrade: GRN Head + MLP Tokenizer

**Goal**: Improve `UncertainFTTransformerRefined` to reduce scatter on the
predicted-vs-true XCO2 anomaly plot by replacing the linear feature tokenizer
with a per-feature MLP tokenizer and replacing the regression head with a
Gated Residual Network (GRN).

**File**: `src/models_transformer.py` (primary), `src/model_adapters.py`,
`src/fitting_data_correction.py`

---

## Summary of Changes

| # | Component | Before | After | Status |
|---|---|---|---|---|
| 1 | `GatedResidualNetwork` class | (missing) | New class added | [x] |
| 2 | `MLPTokenizer` class | (missing) | New class added | [x] |
| 3a | `UncertainFTTransformerRefined` tokenizer | `FeatureTokenizer` | `MLPTokenizer` (with `tokenizer_type` param) | [x] |
| 3b | `UncertainFTTransformerRefined` head | 2-layer GELU MLP | `GatedResidualNetwork` | [x] |
| 4 | `FTAdapter` | no `tokenizer_type` | propagate `tokenizer_type` in save/load | [x] |
| 5 | `models_transformer.py` call site | no `tokenizer_type` | `tokenizer_type='mlp'` | [x] |

Pre-LayerNorm and GELU in `AdvancedTransformerBlock` are **already implemented** — no change needed.

---

## Detailed Steps

### Step 1 — Add `GatedResidualNetwork` (after `PreNormResidual`, ~line 160)

File: `src/models_transformer.py`

```python
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.skip_connection = nn.Linear(input_size, output_size)
        self.dense = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        residual = self.skip_connection(x)
        gated_output = self.gate(x) * self.dense(x)
        return self.norm(residual + gated_output)
```

**Why**: The gated skip connection lets the Transformer selectively suppress
high-frequency noise in the prediction, tightening the 1:1 scatter alignment.

Status: [ ]

---

### Step 2 — Add `MLPTokenizer` (after `FeatureTokenizer`, ~line 150)

File: `src/models_transformer.py`

Keep `FeatureTokenizer` for backward-compat loading of existing checkpoints.
Add `MLPTokenizer` as a new class:

```python
class MLPTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.tokenizers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_token // 2),
                nn.GELU(),
                nn.Linear(d_token // 2, d_token)
            ) for _ in range(n_features)
        ])

    def forward(self, x):
        # x: [batch, n_features]
        tokens = [self.tokenizers[i](x[:, i:i+1]) for i in range(len(self.tokenizers))]
        return torch.stack(tokens, dim=1)   # [batch, n_features, d_token]
```

**Why**: A single linear projection of a scalar into a 64-D / 256-D token is
unstable for noisy inputs like cloud distance. Two layers with GELU allow each
physical feature to build a richer initial representation before attention.

Status: [ ]

---

### Step 3a — Swap tokenizer in `UncertainFTTransformerRefined.__init__` (~line 201)

File: `src/models_transformer.py`

Add `tokenizer_type: str = 'mlp'` parameter; choose tokenizer accordingly:

```python
# Before:
self.tokenizer = FeatureTokenizer(n_features, d_token)

# After:
if tokenizer_type == 'mlp':
    self.tokenizer = MLPTokenizer(n_features, d_token)
else:
    self.tokenizer = FeatureTokenizer(n_features, d_token)
self.tokenizer_type = tokenizer_type
```

Status: [ ]

---

### Step 3b — Replace head in `UncertainFTTransformerRefined.__init__` (~lines 210-214)

File: `src/models_transformer.py`

```python
# Before:
self.head = nn.Sequential(
    nn.Linear(d_token, d_token // 2),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(d_token // 2, 3),
)

# After:
self.head = GatedResidualNetwork(
    input_size=d_token, hidden_size=d_token // 2,
    output_size=3, dropout=0.1
)
```

`forward()` is **unchanged** — `self.head(x)` still returns `[batch, 3]`.

Status: [ ]

---

### Step 4 — Update `FTAdapter` to propagate `tokenizer_type`

File: `src/model_adapters.py`

Three locations:

**4a. `__init__`** (~line 253): add parameter
```python
def __init__(self, model, n_features, d_token=128, n_heads=8,
             n_layers=4, d_ff=256, tokenizer_type='mlp'):
    ...
    self.tokenizer_type = tokenizer_type
```

**4b. `save()`** (~line 287): add to meta dict
```python
meta = {
    'n_features':     self.n_features,
    'd_token':        self.d_token,
    'n_heads':        self.n_heads,
    'n_layers':       self.n_layers,
    'd_ff':           self.d_ff,
    'tokenizer_type': self.tokenizer_type,   # ← new
}
```

**4c. `load()`** (~line 324): pass to model constructor with safe default for
old checkpoints that pre-date this change:
```python
model = UncertainFTTransformerRefined(
    n_features=meta['n_features'],
    d_token=meta['d_token'],
    n_heads=meta['n_heads'],
    n_layers=meta['n_layers'],
    d_ff=meta['d_ff'],
    tokenizer_type=meta.get('tokenizer_type', 'linear'),  # 'linear' = old FeatureTokenizer
)
```

Status: [ ]

---

### Step 5 — Update training call site in `fitting_data_correction.py` (~line 1435)

File: `src/fitting_data_correction.py`

```python
# Before:
FTAdapter(model, n_features=pipeline.n_features,
          d_token=256, n_heads=8, n_layers=4, d_ff=512).save(output_dir)

# After:
FTAdapter(model, n_features=pipeline.n_features,
          d_token=256, n_heads=8, n_layers=4, d_ff=512,
          tokenizer_type='mlp').save(output_dir)
```

Status: [ ]

---

## Backward Compatibility Notes

- **`FeatureTokenizer` is kept** in the file so that old `model_best.pt`
  checkpoints still load when `ft_meta.pkl` has `tokenizer_type='linear'` or
  is absent (defaults to `'linear'` in `FTAdapter.load`).
- **`plot_attention_map`** accesses `model.tokenizer(sample_batch)` — the
  output shape `[batch, n_features, d_token]` is identical for both tokenizer
  types, so no changes needed there.
- **New `model_best.pt` checkpoints are not backward-compatible** with old
  `FTAdapter.load` builds because the head state-dict keys differ. Existing
  trained models must be retrained after this change.

---

## Parameter Count Impact (d_token=256, n_features≈80)

| Component | Before | After | Delta |
|---|---|---|---|
| Tokenizer | `80 × 256 × 2` ≈ 41 K | `80 × (1×128 + 128×256)` ≈ 2.6 M | +2.6 M |
| Head | `256×128 + 128×3` ≈ 33 K | `256×3 × 3` ≈ 2.3 K | ≈ same |

The MLP tokenizer is the dominant cost — verify GPU memory on Blanca is
sufficient before kicking off a full training run.

---

## Testing Checklist (after implementation)

- [x] `python src/models_transformer.py` imports without error
- [x] `UncertainFTTransformerRefined(n_features=12)` forward pass produces `[batch, 3]`
- [x] `MLPTokenizer` forward produces `[batch, n_features, d_token]`
- [x] `GatedResidualNetwork(D, D//2, 3)` forward produces `[batch, 3]`
- [x] `return_attn=True` path returns correct attn shape `[batch, n_features, n_features]`
- [x] `tokenizer_type='linear'` backward-compat path still works
- [x] `FTAdapter.save` writes `tokenizer_type` into `ft_meta.pkl`
- [x] `FTAdapter.load` on old checkpoint (no `tokenizer_type`) defaults to `'linear'`
- [ ] `plot_attention_map` runs end-to-end without shape errors
- [ ] Training run on local machine (50 epochs) converges — compare val MAE vs old head

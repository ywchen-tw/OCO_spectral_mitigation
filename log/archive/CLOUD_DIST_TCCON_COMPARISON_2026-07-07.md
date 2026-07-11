# Cloud-distance-grouped TCCON comparison

> **ARCHIVED 2026-07-09 — feature landed, numbers superseded.** The near/far
> cloud-distance grouping this doc introduced is now a standing feature of
> `workspace/tccon_comparison_report.py`; the result tables below pre-date the
> wet/dry AK-harmonization fix (`CRITICAL_FIXES.md` #11). Quote the regenerated
> CSVs under
> `results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/` instead.

**Date:** 2026-07-07 · **Commit:** `a1d700c` · **Models:** deep ensemble, TabM, structured residual

> **AK-reference caveat (added later on 2026-07-07):** the numbers below were
> generated against the AK-harmonized reference **before** the
> `ak_harmonize.py` wet/dry fix (CRITICAL_FIXES #11; ak_delta −0.93 → +0.34).
> Since the shift is common-mode per case, the raw→before→after *progressions*
> and model *rankings* are unaffected (all series shift together), but absolute
> bias/MAE/MSE values change. **All reports regenerated with the fix later on
> 2026-07-07** (DE r100+r50, TabM o05l15 r100+r50, TabM m16 r100, structured
> o05l15/o10l10 r100+r50, regime + 3 calibration variants r50) — quote the
> regenerated CSVs, not the tables below; post-fix cross-model summary in
> `TCCON_BIAS_MODEL_IMPROVEMENT_PLAN_2026-07-06.md` §"Regenerated results".

Splits each OCO-2 ↔ TCCON collocation's footprints by **nearest-cloud distance**
and reports the correction quality separately for near-cloud vs far-cloud
footprints. Answers the project's core question — *does the ML correction earn
its keep specifically in the near-cloud regime where the operational bias
correction struggles?* — directly against the TCCON reference.

---

## What was added

`workspace/tccon_comparison_report.py` gained a `cld_group` dimension parallel to
the existing `surface` dimension. Within each case's collocated footprints it
bins by `cld_dist_km` and reuses the **exact same** `_case_metrics` + drawing
machinery as the headline figures, so the cloud-grouped views are statistically
and visually consistent with the `tccon_{ref}_scatter` / `tccon_{ref}_by_surface_*`
figures.

> **Note (paper-ready refactor):** figures were later split one-panel-per-file and
> renamed to the symmetric `tccon_{ak,direct}_*` scheme (scatter and bias emitted
> separately); the `.csv`/`.md` names below are unchanged. See the current
> docstring/`--help` of `tccon_comparison_report.py` for the authoritative list.

New outputs per model (radius suffix `_r100km`):

| file | content |
|---|---|
| `tccon_{ref}_by_cld_scatter_r100km.png`, `tccon_{ref}_by_cld_bias_r100km.png` | one panel per cloud bin (`ref` = `ak`/`direct`) |
| `tccon_{ref}_by_surface_by_cld_bias_r100km.png` | bias, split ocean/land × bin |
| `tccon_comparison_by_cld_r100km.csv` | per-(case, surface, bin) metrics |
| `tccon_comparison_by_cld_agg_r100km.csv` | per-(surface × bin) aggregate, all metrics |
| `tccon_metrics_{ref}_r100km.csv` | comprehensive per-(surface × cloud-group) metrics table |
| "Cloud-distance-grouped aggregate" section in `tccon_comparison_r100km.md` | two markdown tables (below) |

New CLI flags:

- `--cld-edges` — bin edges in km (comma-separated, right-open; `inf` for the open
  tail). Default `0,10,inf` → near (≤10 km) vs far.
- `--cld-all-years` — opt out of the pre-drift restriction (see below).

Wired into the three pre-drift launchers (`curc_shell_blanca_plot_corr_xco2_`
`deepens.sh` / `tabm.sh` / `structured_common.sh`); the **drift** launcher is left
untouched because its cases are all post-drift.

---

## Methodology

- **Pre-drift only.** Restricted to cases with `year < AQUA_FREE_DRIFT_YEAR` (2022),
  since the Aqua-MODIS cloud collocation — hence `cld_dist_km` — is only reliable
  before Aqua entered free drift. `--cld-all-years` overrides.
- **Bins.** `0,10,inf` → **0–10 km** (near-cloud) and **≥10 km** (far). 108 pooled
  `(case × bin)` rows over 75 pre-drift station-days.
- **Series.** Every metric reports the full progression **raw → before → after**:
  - `raw` = pre-bias-correction `xco2_raw`,
  - `before` = `xco2_bc` (operational bias correction),
  - `after` = ML-corrected.
- **Metrics** (all vs the AK/prior-harmonized TCCON reference, ppm unless noted):
  - `mean |bias|` — mean over station-days of the |station-day-mean − TCCON|.
  - `MAE` — footprint-level `mean |XCO₂ − TCCON|`, averaged over station-days.
  - `fp-RMSE` — footprint-level RMSE-to-TCCON, averaged over station-days.
  - **absolute MSE** (ppm²), three variants:
    - `pooled` — footprint-weighted `Σ n·RMSE² / Σ n` (overall squared error; big
      near-cloud cases dominate),
    - `mean per-case` — station-day-weighted mean of per-case MSE (= RMSE²),
    - `station-mean` — mean of squared station-day bias (scatter-plot-level).

---

## Results (radius 100 km, pre-drift)

### Error metrics — raw → before → after

| model | bin | n | mean \|bias\| | MAE | fp-RMSE |
|---|---|--:|---|---|---|
| **DE** | 0–10 km | 70 | 1.34 → 1.44 → **1.06** | 2.46 → 2.28 → **1.26** | 3.13 → 2.95 → **1.46** |
| **DE** | ≥10 km | 38 | 1.22 → 1.24 → **1.07** | 1.68 → 1.37 → **1.13** | 2.01 → 1.58 → **1.21** |
| **Structured** | 0–10 km | 70 | 1.34 → 1.44 → **1.13** | 2.46 → 2.28 → **1.36** | 3.13 → 2.95 → **1.62** |
| **Structured** | ≥10 km | 38 | 1.22 → 1.24 → **1.10** | 1.68 → 1.37 → **1.17** | 2.01 → 1.58 → **1.26** |
| **TabM** | 0–10 km | 70 | 1.34 → 1.44 → **1.15** | 2.46 → 2.28 → **1.50** | 3.13 → 2.95 → **1.82** |
| **TabM** | ≥10 km | 38 | 1.22 → 1.24 → **1.10** | 1.68 → 1.37 → **1.18** | 2.01 → 1.58 → **1.29** |

### Absolute MSE (ppm²) — raw → before → after

| model | bin | pooled fp-MSE | mean per-case MSE | station-mean MSE |
|---|---|---|---|---|
| **DE** | 0–10 km | 10.63 → 12.97 → **1.91** | 11.90 → 12.44 → **2.83** | 3.12 → 3.81 → **1.91** |
| **DE** | ≥10 km | 2.38 → 1.20 → **0.94** | 5.18 → 3.37 → **2.05** | 2.67 → 2.54 → **1.81** |
| **Structured** | 0–10 km | 10.63 → 12.97 → **2.28** | 11.90 → 12.44 → **3.44** | 3.12 → 3.81 → **2.06** |
| **Structured** | ≥10 km | 2.38 → 1.20 → **1.00** | 5.18 → 3.37 → **2.16** | 2.67 → 2.54 → **1.89** |
| **TabM** | 0–10 km | 10.63 → 12.97 → **3.58** | 11.90 → 12.44 → **4.31** | 3.12 → 3.81 → **2.14** |
| **TabM** | ≥10 km | 2.38 → 1.20 → **0.96** | 5.18 → 3.37 → **2.28** | 2.67 → 2.54 → **1.93** |

### Ocean vs land (fp-RMSE / pooled MSE, before → after)

The pooled near-cloud win is **land-driven**; ocean barely moves.

| model | surface | bin | fp-RMSE | pooled MSE |
|---|---|---|---|---|
| DE | land | 0–10 km | 3.41 → **1.39** | 13.50 → **1.84** |
| DE | ocean | 0–10 km | 1.76 → **1.61** | 3.34 → **3.29** |
| Structured | land | 0–10 km | 3.41 → **1.60** | 13.50 → **2.22** |
| TabM | land | 0–10 km | 3.41 → **1.89** | 13.50 → **3.60** |

---

## Key findings

1. **The ML step is what fixes the near-cloud tail.** Near clouds (0–10 km) the
   operational bias correction (raw → before) *does not* reduce pooled MSE — it
   nudges it up (10.63 → 12.97 ppm²), because it is tuned for the clear-sky bulk.
   The ML correction (before → after) collapses it (12.97 → 1.91 for DE). This is
   a cleaner argument for the ML step than before→after alone showed, and only
   became visible once the `raw` series was added.
2. **Correction does most of its work near clouds.** Near-cloud fp-RMSE drops far
   more (2.95 → 1.46, DE) than far-cloud (1.58 → 1.21).
3. **Model ranking near clouds: DE > Structured > TabM** (after-fp-RMSE 1.46 <
   1.62 < 1.82; same order on every MSE variant). Consistent with the prior
   finding that DE owns the near-cloud land tail. Far-cloud, the three are within
   noise.
4. **Land-driven.** Ocean near-cloud error is already small before correction
   (RMSE 1.76) and barely changes; the headline near-cloud improvement is
   essentially the land 3.41 → 1.39 (DE).

---

## Caveats

- Ran locally with `DATA_ROOT=.`; AK harmonization used whatever local Lite /
  parquet-AK columns exist and fell back to raw window means otherwise.
- DE has 108 local `plot_data.parquet` vs 75 each for TabM/structured, but the
  pre-drift cloud aggregate resolves to the same 108 `(case × bin)` rows for all
  three, so case coverage is consistent.
- Only radius 100 km (`_r100km`, launcher primary) regenerated; the 50 km
  robustness variant was not refreshed.
- The **headline** aggregate still shows raw→before→after for |bias|/RMSE but no
  MAE; the MAE addition currently lives only in the cloud-grouped section.

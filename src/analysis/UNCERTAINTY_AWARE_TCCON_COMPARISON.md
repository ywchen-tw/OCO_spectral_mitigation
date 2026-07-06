# Uncertainty-aware OCO-2 (ML-corrected) vs TCCON (AK-harmonized) comparison

**Status:** design note, pre-implementation. Nothing here is coded yet.
**Goal:** stop comparing two bare point values. Attach an uncertainty bar to each
side and test *statistically* whether ML-corrected OCO-2 XCO2 and AK-harmonized
TCCON XCO2 agree (or quantify the confidence of overlap), per case and globally.

Context: current reports (`tccon_comparison_report.py`, driven by
`ak_harmonize.py`) compare a case-level corrected-footprint mean against a
single AK-harmonized TCCON reference — point vs point. This note specifies the
uncertainty budget on each side and the comparison statistics to layer on top.

---

## 1. The two sides

- **Side A — ML-corrected OCO-2**, per case: mean of the deep-ensemble (DE)
  corrected footprints near the station, `x̄_oco`.
- **Side B — AK-harmonized TCCON**, per case: TCCON obs pushed through OCO-2's
  mean column operator (Rodgers & Connor 2003 / Wunch et al. 2017),
  `c̄_TC` (`ak_harmonize.ak_adjusted_ref_from_operator`).

Per case we want `D = x̄_oco − c̄_TC` **with** `u_oco`, `u_TC`, and a test.

---

## 2. Side A uncertainty budget (ML-corrected OCO-2)

Key structural fact: the ML correction is a **deterministic function of
features**, `x_corr = xco2_bc − f(features)`. Random errors not in the feature
set pass straight through the correction.

Candidate components (do **not** assume how they combine — see §4):

| component | source | behavior w/ N | notes |
|---|---|---|---|
| **epistemic** `Var_m(x̄_m)` | DE ensemble | **does NOT shrink** | per-member case mean `x̄_m`, then spread over M members. Correlated across all footprints (shared models). |
| **DE aleatoric** `σ²_alea,DE` | beta-NLL head | shrinks w/ N_eff | predicts `Var(xco2_bc_anomaly − prediction | features)` — residual cloud-contamination scatter. |
| **retrieval posterior** `xco2_uncertainty` | Lite file (in parquet) | shrinks w/ N_eff | OE posterior on **raw** `xco2`. Known optimistic. |

**Critical open question (unresolved — see §4).** `xco2_uncertainty` and the DE
aleatoric are **distinct quantities at opposite ends of the processing chain**
and their relationship has NOT been measured:

```
xco2(raw) ──bias corr (det.)──► xco2_bc ──minus ref──► xco2_bc_anomaly ──DE──► corrected
   │                               │                        │                     │
xco2_uncertainty            +bias-corr unc.          +reference unc.       DE predictive var
(OE posterior, raw,         (NOT in Lite product)    (r05/r15 def.)        (feature-conditioned
 optimistic)                                                               residual scatter)
```

- `xco2_uncertainty` describes RAW `xco2`, **not** `xco2_bc` and **not**
  `xco2_bc_anomaly`. The bias correction has its own uncertainty that the Lite
  product does not fold back in; the anomaly adds reference (r05/r15) uncertainty.
- The DE variance measures near-cloud residual scatter, dominated by scene
  physics, not OE measurement noise.
- They may be ~orthogonal (→ add in quadrature) or correlated via scene
  difficulty (→ partial overlap). **We do not know.** Any assumed decomposition
  (including the earlier "aleatoric ⊇ retrieval noise" guess) is unverified.

Provisional physical budget (to be validated, not adopted blindly):

```
u²_oco,case = Var_m(x̄_m)                                 [epistemic; floor]
            + (Σ_k xco2_uncertainty²_k) / (N · N_eff)     [retrieval]
            + (Σ_k σ²_alea,DE,k)        / (N · N_eff)     [model residual, IF not redundant]
```

`N_eff` = spatial-autocorrelation-reduced footprint count
(`N_eff = N / (1 + (N−1)ρ̄)`, or block bootstrap over orbit segments). Retrieval
error can be spatially correlated (shared aerosol/geometry) → use `N_eff`, not N.

---

## 3. Side B uncertainty budget (AK-harmonized TCCON)

```
u²_TC = σ²_tccon / N_eff,tccon        [temporal scatter; have tccon_sd_ak]
      + ⟨u²_meas⟩ / N_tccon           [TCCON retrieval precision, xco2_error in file]
      + u²_harm                       [AK/prior leakage — kernel-dependent, see below]
      + u²_colloc                     [representativeness: point station vs ≤100 km fp cloud]
      (+ u²_scale)                    [TCCON WMO-scale systematic ~0.4 ppm; ONLY if absolute claim]
```

- **`u_harm`** — prior-shape sensitivity through the `(1−a)·h` weighting.
  Measured empirically so far: ~0.01 ppm at Izaña (kernel ≈ 1 through the
  column), ~0.1–0.3 ppm at Ny-Ålesund (Arctic kernel drops to ~0.26 at TOA,
  OCO-prior "leakage" term +0.5–0.8 ppm). Compute per case; it is NOT constant.
- **`u_colloc` is usually the dominant term** and the most-often-omitted one.
  Estimate from the spatial XCO2 gradient across collocated footprints × mean
  separation, or from a model. Frequently 0.3–1 ppm; ignoring it makes error
  bars unrealistically tight.

---

## 4. The analyses that MUST precede fixing the budget

Do not hard-code how the Side-A components combine. Two studies decide it.

### 4.1 Relationship study (how retrieval σ relates to DE σ)
Per footprint, compare `xco2_uncertainty` vs DE-predicted σ:
- correlation ρ, ratio distribution, **stratified by scene** (cld_dist, AOD,
  SNR, SZA).
- ρ≈0 → orthogonal → quadrature sum. High ρ + similar magnitude → redundant,
  don't sum. DE σ ≫ retrieval σ near clouds → DE dominates where it matters.

### 4.2 Calibration study (authoritative — decides the true total)
Against a truth proxy with "answers" — **r05/r15 clean held-out set** and/or
**TCCON** — form realized error `(corrected − truth)` and the standardized
residual `z = (corrected − truth) / u_candidate` for each candidate budget:
(a) retrieval-only, (b) DE-only, (c) quadrature sum, (d) sum-minus-overlap.

The correct budget is the one whose `z` is **calibrated**: reliability diagram
flat, `⟨z²⟩ ≈ 1`. Expect retrieval-only `⟨z²⟩ ≫ 1` (OE posteriors optimistic).
If the quadrature sum gives `⟨z²⟩ ≈ 1`, that both fixes the total AND proves the
components are additive. This converts "should we double-count?" (unanswerable a
priori) into "which combination is empirically calibrated?" (measurable).

A calibration gap that no combination closes flags the **missing** terms
(bias-correction uncertainty, reference uncertainty) as non-negligible.

---

## 5. Comparison methods (layer on the validated budgets)

1. **Error-propagated z / CI on the case difference** (simplest).
   `D = x̄_oco − c̄_TC`, `u_D = √(u²_oco + u²_TC)`, 95% CI `D ± 1.96 u_D`;
   significant if CI excludes 0. Per-case table verdict. Gaussian/independent.

2. **Monte Carlo overlap / probability of agreement** (best for "confidence of
   overlap"). Sample each side's full distribution — OCO as the ensemble
   *mixture* (member → predictive Gaussian), TCCON as harmonized-obs + prior
   perturbation. Report `P(|D| < δ)` (e.g. δ = 0.5 ppm) or the density overlap
   coefficient. Non-Gaussian-safe.

3. **Equivalence test (TOST / Bayesian ROPE)** — the scientifically correct
   framing for validation (show agreement, not "failed to find a difference").
   Equivalent if the 90% CI of D lies inside ±δ; δ from the science (e.g. 0.5
   ppm accuracy target). Recommended headline.

4. **Hierarchical random-effects model across all cases** (best global answer).
   `D_i ~ N(μ_site, u²_i)`, `μ_site ~ N(μ, τ²)`: overall offset μ + CI, between-
   site heterogeneity τ. Extends the existing site-clustered bootstrap.

5. **Errors-in-both-variables regression** (validation-standard figure).
   York/Deming regression of `x̄_oco` on `c̄_TC`, weights 1/u²; test slope=1,
   intercept=0. Classic scatter + confidence band.

**Recommended combo:** Method 1 for the per-case table (adds `±u` + significance
flag), Method 3 (TOST/ROPE) as headline, Method 4 for the single global
consistency number. Method 2 if a reviewer asks for overlap probability
explicitly. Method 5 only if the scatter figure is wanted.

---

## 6. Invariance caveat (keep the framing honest)

The AK/prior harmonization shifts the **absolute** reference (run-wide
`ak_delta` ≈ −0.96 ± 0.74 ppm) but the **before/after correction improvement is
invariant** to it (same operator both sides). So:
- Uncertainty-aware absolute-agreement tests (this note) live on the AK-
  harmonized side and carry the prior-leakage term `u_harm`.
- The *correction-helps* claim does not depend on any of this and can be shown
  on the direct reference too.

---

## 7. Field/data fields to wire up (reference, for the eventual spec)

- Side A: parquet `xco2_bc`, `xco2_bc_anomaly`, `xco2_uncertainty`, DE per-member
  predictions + beta-NLL σ (from the DE inference), `lon/lat/time`, scene vars
  (`cld_dist_km`, `aod_total`, `snr_*`, `sza`).
- Side B: TCCON `.public.qc.nc` `xco2`, `xco2_error`, `prior_xco2`, `prior_co2`,
  `prior_pressure`, `time`; OCO-2 operator `ak_NN/pwf_NN/co2_ap_NN/plev_NN`,
  `xco2_apriori` (already in parquet via `build_feature_dataset.compute_ak_columns`).
- Truth proxy for calibration (§4.2): r05/r15 clean held-out anomalies; TCCON.

---

## 8. Open items

- [ ] §4.1 relationship study: `xco2_uncertainty` vs DE σ, scene-stratified.
- [ ] §4.2 calibration study: pick the calibrated Side-A budget via `⟨z²⟩`.
- [ ] Per-case `u_harm` from prior perturbation through `(1−a)·h`.
- [ ] `u_colloc` estimator (spatial-gradient based).
- [ ] `N_eff` estimator (spatial autocorrelation / block bootstrap).
- [ ] Bias-correction + reference uncertainty: quantify only if §4.2 gap demands.

---

## 9. Implementation plan

Phased, ordered by dependency. Each phase is independently runnable and gates
the next. Existing entry points reused, not rewritten. **Rule: §4 (Phases 1–2)
decides the Side-A budget shape BEFORE any comparison code (Phase 4) is written.**

### Existing API this builds on (verified)
- `src/models/deep_ensemble.py`
  - `ensemble_predict(members, X, loss, nu) → (mu*, sigma*)` where
    `var* = mean_m(var_m + mu_m²) − mu*²` (epistemic + aleatoric combined).
  - `_member_predict(model, X, …) → (mu_m, var_m)` per member — the hook for the
    **split**: `epistemic = Var_m(mu_m)`, `aleatoric = mean_m(var_m)`.
- `src/apply/apply_deep_ensemble.py`, `workspace/build_deepens_plot_data.py`
  — inference bridge; currently emit the point correction (`mu`) only.
- `workspace/ak_harmonize.py` — `operator_from_dataframe`,
  `ak_adjusted_ref_from_operator` (Side B reference).
- `workspace/tccon_collocate.py` — station collocation.
- `workspace/tccon_comparison_report.py` — report/figure engine (adds columns +
  aggregates here).
- parquet fields via `src/analysis/build_feature_dataset.py`: `xco2_bc`,
  `xco2_bc_anomaly`, `xco2_uncertainty`, `ak_NN/pwf_NN/co2_ap_NN/plev_NN`,
  `xco2_apriori`, scene vars.

### Phase 0 — Uncertainty plumbing (emit the raw components)
**New:** extend `build_deepens_plot_data.py` (or a sibling
`build_deepens_uncertainty_data.py`) to write, per footprint, alongside the
existing correction:
- `de_mu`, `de_sigma` (from `ensemble_predict`),
- `de_epistemic_sigma` = `std_m(mu_m)`, `de_aleatoric_sigma` = `sqrt(mean_m(var_m))`
  (loop `_member_predict` over pooled members — reuse `_predict_pooled`),
- pass through `xco2_uncertainty` from the parquet.
**Out:** `plot_data_unc.parquet` with the three Side-A candidate σ's per footprint.
**Done when:** columns present, `de_sigma² ≈ de_epistemic_sigma² + de_aleatoric_sigma²`
(mixture identity holds to numerical tol).

### Phase 1 — §4.1 Relationship study
**New:** `workspace/uncertainty_relationship.py`.
- Load `plot_data_unc.parquet`; correlate `xco2_uncertainty` vs `de_sigma`,
  `de_aleatoric_sigma`, `de_epistemic_sigma`; ratio distributions; **stratify by
  scene** (`cld_dist_km`, `aod_total`, `snr_*`, `sza`).
**Out:** `results/model_comparison/<tag>/uncertainty_relationship.{md,png}`.
**Done when:** we know ρ and whether retrieval vs DE σ are orthogonal / redundant /
DE-dominated near clouds. Informs — does not finalize — the budget.

### Phase 2 — §4.2 Calibration study (AUTHORITATIVE)
**New:** `workspace/uncertainty_calibration.py`.
- Truth proxy: r05/r15 clean held-out anomalies (primary), TCCON (secondary).
- For candidate budgets {retrieval-only, DE-only, quadrature, sum−overlap}
  compute `z = (corrected − truth)/u_candidate`; reliability diagram, `⟨z²⟩`,
  coverage of nominal 68/95% intervals.
**Out:** `.../uncertainty_calibration.{md,png}` + a chosen `SIDE_A_BUDGET` verdict
string.
**Done when:** one budget gives `⟨z²⟩ ≈ 1` / flat reliability → adopt it. Residual
miscalibration across all candidates → flag missing bias-corr/reference terms
(§2) for Phase 3b.
**GATE:** Phase 4 cannot start until this picks the Side-A total.

### Phase 3 — Side B budget
**New:** `workspace/tccon_uncertainty.py` (or extend `ak_harmonize.py`).
- `u_meas` from TCCON `xco2_error`; `u_tccon` from `tccon_sd_ak` / `N_eff,tccon`.
- **`u_harm`**: MC/analytic prior perturbation through `(1−a)·h` per case
  (kernel-dependent; ~0.01 ppm iz … ~0.1–0.3 ppm ny).
- **`u_colloc`**: spatial XCO2 gradient across collocated footprints × mean
  station separation.
- **`N_eff`** helper (spatial autocorrelation / block bootstrap) — shared by
  Side A and Side B.
**Out:** per-case `u_TC` with component breakdown.
**Done when:** each case has `u_TC` and its components logged; `u_colloc` sanity-
checked against the raw footprint spread.
**Phase 3b (conditional):** bias-correction + reference uncertainty, only if
Phase 2 gap demands.

### Phase 4 — Comparison statistics (gated on Phase 2)
**New:** `workspace/tccon_uncertainty_stats.py`, wired into
`tccon_comparison_report.py`.
- **M1** per-case z / 95% CI on `D` (+ significance flag column in the per-case
  table).
- **M3** TOST / ROPE equivalence at δ = 0.5 ppm — **headline**.
- **M4** hierarchical random-effects across cases (μ, τ, site clustering) —
  extends the existing site-clustered bootstrap in the report.
- **M2** MC overlap `P(|D|<δ)` — optional, ensemble-mixture sampling.
- **M5** York/Deming regression — optional figure.
**Out:** new columns in `tccon_comparison_r{100,50}km.{md,csv}`; new
`tccon_equivalence_r*.{md,csv}`, `tccon_hierarchical_r*.md`.
**Done when:** every case row carries `D ± u_D` + verdict; global μ±CI + TOST
result reported at 100 km (primary) and 50 km (robustness).

### Phase 5 — Report integration & figures
- Per-case table gains `±u_oco`, `±u_TC`, `D±u_D`, sig/equiv flags.
- Headline aggregate gains the global consistency statement (M4) and equivalence
  verdict (M3).
- Optional error-bar scatter (M5) + calibration panel (Phase 2) into the report.
**Done when:** `tccon_comparison_report.py` regenerates the DE production tag
end-to-end with uncertainty-aware numbers.

### Dependency graph
```
P0 ──► P1 ──► P2 (GATE) ──► P4 ──► P5
          └── P3 ──────────┘
```
P3 (Side B) can run in parallel with P1/P2; P4 needs both P2's verdict and P3.

### Suggested milestones
1. **P0+P1** — plumbing + relationship (one PR; answers "are retrieval and DE σ
   redundant?").
2. **P2** — calibration verdict (decides the budget; smallest, highest-leverage).
3. **P3** — Side B budget.
4. **P4+P5** — stats + report integration.

### Scope guards
- Case-level design throughout (per-case means vs one AK-harmonized reference) —
  consistent with the existing report; no per-footprint TCCON matching.
- Keep AK invariance framing (§6): uncertainty work is on the **absolute**
  comparison; the correction-helps claim stays reference-independent.
- No new model training — inference-time uncertainty only.

---

## 10. Results log

### Phase 0 — DONE
`build_deepens_plot_data.py` emits per footprint: `de_sigma`,
`de_epistemic_sigma = std_m(mu_m)`, `de_aleatoric_sigma = sqrt(mean_m(var_m))`,
plus passthrough `xco2_uncertainty` and scene cols (`aod_total`, `sza`,
`snr_*`). Mixture identity `de_sigma² = epi² + alea²` checked at write time
(max|Δvar| ~1e-4, float32).

### Phase 1 — PRODUCTION model, both surfaces, 6 dates (n≈990k) — DONE
Tool: `workspace/uncertainty_relationship.py`. Models: `de_ocean_beta_nll_prof_r05`
(5 folds) + `de_land_beta_nll_prof_r15` (5 folds). Output:
`results/model_comparison/deep_ensemble/uncertainty_phase1_prod_{ocean,land}/`.

Data note: per-date `combined_*_all_orbits.parquet` ARE profile-complete — the
profiles are stored as sigma-grid columns `t_sigma_NN/q_sigma_NN/co2prior_sigma_NN`
(+ `tropopause_sigma`) by `build_feature_dataset.compute_sigma_profile_columns`;
the FeaturePipeline builds the `*_pc01..04` EOF features from those at transform
time. (Earlier note that they lacked profiles was a wrong-column-name error.)

Core result — **the retrieval-vs-DE relationship FLIPS by surface**, and both
surfaces show low footprint-level correlation (ρ ≈ 0.26–0.43):

| | retrieval σ | DE alea σ | alea/retr | verdict |
|---|--:|--:|--:|---|
| **ocean** (median) | 0.42 | 0.36 | 0.83 | retrieval ≥ DE everywhere |
| **land** (median)  | 0.50 | 0.55 | 1.07 | DE ≥ retrieval, esp. near cloud |

By cloud distance:
- **Ocean:** retrieval σ ~flat (0.42–0.45) and *exceeds* DE aleatoric at all
  ranges (ratio 0.93 → 0.70 as you leave the cloud). The DE is more confident
  than the L2 posterior claims; ocean correction is strong (RMS 0.62→0.37). So
  on ocean the **retrieval posterior is the dominant Side-A term.**
- **Land:** DE aleatoric *exceeds* retrieval near clouds (0.87 vs 0.58, ratio
  1.47 at 0–5 km) and converges to it far (0.92 at 20–50 km). The DE captures
  near-cloud contamination scatter the posterior misses. So on land the
  **DE is the dominant term where it matters (near clouds).**

Other:
- Both surfaces: **ρ ≈ 0.26–0.43** retrieval-vs-DE → overlapping magnitude but
  different footprint-level information; NOT redundant, NOT independent, on
  either surface.
- **Epistemic** small (ocean ~0.09, land ~0.17) — distinct, weakly correlated;
  does NOT average down → becomes the case-level floor.

Implication for the budget (per surface, pending §4.2 calibration):
- **Ocean:** Side-A total likely ≈ `quadrature(retrieval, epistemic)`; DE
  aleatoric adds little beyond retrieval.
- **Land:** Side-A total likely ≈ `quadrature(DE_aleatoric, epistemic)` near
  clouds, with retrieval contributing more in the far field.
The per-surface asymmetry means the calibration study must run **separately per
surface**.

**Follow-up (not yet run):**
- [ ] Far-cloud (>50 km) coverage thin in the 6 chosen dates — add far-cloud
  dates so the near→far σ trend extends to the negative-control regime.

### Phase 2 — CALIBRATION (§4.2, the gate) — DONE
Tool: `workspace/uncertainty_calibration.py`. Out-of-sample residual = `y_true − mu`
(corrected − r05/r15 clean-scene truth). Calibrated ⇔ ⟨z²⟩≈1, cov68≈0.683.
Output: `results/model_comparison/deep_ensemble/uncertainty_phase2_{ocean,land}/`.

**Backbone — rigorous OUT-OF-FOLD** (pooled fold `held_out_predictions.parquet`:
7.85M ocean, 3.84M land; physical guard |y|,|mu|≤25 ppm):

| surface | candidate | ⟨z²⟩ | cov68 | cov95 |
|---|---|--:|--:|--:|
| ocean | **de_total** | **1.110** | 0.676 | 0.937 |
| land  | **de_total** | **1.086** | 0.685 | 0.940 |

de_total (DE predictive σ) is **already near-calibrated out-of-sample**, both
surfaces — only ~1.05× from perfect.

By cloud distance (de_total, out-of-fold):

| cld_dist | ocean ⟨z²⟩ | land ⟨z²⟩ |
|---|--:|--:|
| 0–5 km  | 1.232 | **1.590** |
| 5–20 km | 0.997 | 1.259 |
| 20–50 km| 1.026 | 0.998 |

→ σ is calibrated in the far field but **under-confident near clouds**, strongly
so on land (⟨z²⟩=1.59 at 0–5 km ⇒ σ needs ×1.26). This is the regime that drives
the TCCON near-cloud comparison.

**Candidate comparison** (held_out lacks `xco2_uncertainty`, so run on the Phase-1
parquet — IN-SAMPLE, absolute ⟨z²⟩ optimistic by the uniform ~1.2× in-sample→OOF
shift seen in de_total; RELATIVE ranking is valid):

| candidate | ocean ⟨z²⟩ | land ⟨z²⟩ | read |
|---|--:|--:|---|
| de_total | 0.93 | 0.89 | →1.11/1.09 OOF (calibrated) |
| de_aleatoric | 1.00 | 0.99 | in-sample only (no epistemic need) |
| retrieval | 0.77 | 1.13 | worse than de_total |
| **quad_de_retr** | **0.37** | **0.45** | **massively OVER-covers** |
| quad_alea_retr | 0.39 | 0.47 | massively over-covers |

**VERDICT — SIDE_A_BUDGET = `de_total` (DE predictive σ). Do NOT add retrieval σ.**
- Adding `xco2_uncertainty` in quadrature over-covers by ~2× (⟨z²⟩→0.37–0.45):
  the DE predictive σ **already absorbs retrieval noise** — this empirically
  resolves the double-counting question (§2/§4): they are redundant, not additive.
  (Phase-1's "similar magnitude, ρ≈0.4" is explained: the DE learned the
  retrieval-noise component, so re-adding it double-counts.)
- de_total needs a small, **cloud-distance-dependent inflation**: ≈1.05× overall,
  rising to ≈1.11× (ocean) / ≈1.26× (land) at 0–5 km. A near-cloud inflation
  factor is cleaner than retrieval-quadrature (which fixes near-cloud but breaks
  the already-calibrated far field).
- Missing bias-correction / reference uncertainty terms (§2, Phase 3b): NOT needed
  — no calibration gap remains for them to fill (de_total ≈ calibrated).

Implication for Side A (both surfaces):
```
u_oco,footprint = k(cld_dist) · de_sigma          # k≈1.05 far → ~1.1–1.26 near-cloud
u_oco,case²     = Var_m(x̄_m)  +  Σ_k (k·de_sigma_k)²/(N·N_eff)   # epistemic floor + averaged term
```
Retrieval `xco2_uncertainty` drops out of the budget entirely.

### Phase 2 caveats / follow-ups
- Retrieval candidate comparison was IN-SAMPLE (fold held_out set carries no
  `xco2_uncertainty`). The rejection is robust — quad over-covers by ~2× even
  after the ~1.2× OOF shift — but a fully out-of-fold quad test needs
  `xco2_uncertainty` joined onto held_out rows (no key currently; would require
  the per-fold date manifest, absent for these folds).
- Mondrian conformal intervals (`lo_mondrian/hi_mondrian`) already in held_out —
  compare against the k·de_sigma inflation as an alternative calibrated interval.

### Phase 2b — inflation k(cld_dist) fit — DONE
`workspace/sidea_uncertainty.py` fits k = sqrt(⟨z²⟩) per cloud-distance bin from
the pooled out-of-fold set; cached to
`results/model_comparison/deep_ensemble/sidea_inflation_{ocean,land}.json`.

| cld_dist center (km) | 1 | 4 | 8 | 15 | 28 | 42 |
|---|--:|--:|--:|--:|--:|--:|
| k ocean | 1.13 | 1.09 | 1.01 | 0.99 | 0.99 | 1.03 |
| k land  | 1.27 | 1.26 | 1.22 | 1.07 | 0.99 | 1.00 |

Applied as `σ_fp = k(cld_dist)·de_sigma` (clamped linear interp).

### Phase 3 — Side-A case aggregator + Side-B budget — DONE
**Side A** (`sidea_uncertainty.case_uncertainty`):
`u_case² = Var_m(x̄_m) + Σ(k·de_sigma)²/(N·N_eff)`. Test (iz land, 100 km, N=4464):
u_oco = **0.68 ppm** (epistemic 0.44 ⊕ averaged 0.52), N_eff=7.
Two documented knobs, both to be pinned by the Phase-4 end-to-end case calibration:
  - `decorr_km` (N_eff block size, default 15 km) — sets how far the averaged
    term shrinks; N_eff=7 for a compact overpass swath.
  - epistemic term uses the **fully-correlated fallback** (mean per-fp epistemic)²
    here (over-estimate); EXACT `Var_m(x̄_m)` needs per-member μ columns emitted
    from inference (add `mu_00…` to build_deepens_plot_data for the collocated
    subset in Phase 4).

**Side B** (`tccon_uncertainty.side_b_uncertainty`):
`u_TC² = u_meas²/N + u_temporal²/N_eff + u_harm² + u_colloc²`. Tested:

| case | u_TC | u_meas | u_temporal | u_harm | u_colloc |
|---|--:|--:|--:|--:|--:|
| iz | 0.567 | 0.025 | 0.015 | 0.133 | **0.551** |
| ny | 1.915 | 0.028 | 0.042 | 0.155 | **1.908** |

- **u_colloc dominates** (as predicted §3); ny's large value is consistent with
  ny being the worst-RMSE site. `u_harm` (0.13–0.16) matches the §1 trace
  estimates (analytic `Σ[h(1−a)]²σ_prior²`, nominal σ_prior 1 ppm trop / 3 strat).
- Refinement for Phase 4: `u_colloc` here used `xco2_bc` (source parquet); it
  should use the **corrected** XCO2 field (removes cloud-contamination scatter
  from the gradient → likely lowers u_colloc). Needs corrected column joined to
  the collocated footprints.
- Operator (`ak_NN/pwf_NN/…`) must come from the **source combined parquet**, not
  plot_data (which drops those columns) — Phase-4 collocation must carry them.

### Phase 4 — comparison stats + report wiring — DONE (code); GATED on data regen
Built and validated on real data as far as the pre-Phase-0 plot_data allows.

**New / changed code:**
- `workspace/tccon_uncertainty_stats.py` (NEW) — the Phase-4 engine:
  - `compare_case(D, u_oco, u_TC, delta)` → **M1** (z, 95% CI, `significant`) +
    **M3** TOST/ROPE (`tost_p`, `equivalent` ⇔ 90% CI ⊂ [−δ,δ]); δ default 0.5 ppm.
  - `hierarchical(D, u_D)` → **M4** DerSimonian–Laird random-effects: global
    μ ± CI, between-case τ, I², heterogeneity Q p (scipy-free χ² fallback).
  - `case_calibration(D, u_D)` → **end-to-end ⟨z²⟩** + frac-within-95 (the
    whole-budget validator).
  - `side_a_case` / `side_b_case` wrappers + `load_inflation` + `markdown_block`.
- `sidea_uncertainty.py` — `case_uncertainty` now takes `sigma_fp=`/`use_members=`;
  new `calibrated_sigma_by_surface` applies the per-surface k(cld_dist). Exact
  epistemic (mu_NN) is used ONLY on a single-surface frame; a pooled 'all' frame
  forces the fully-correlated fallback (per-surface members are different models,
  so a per-member pooled case-mean is not coherent).
- `build_deepens_plot_data.py` — new `--emit-members` writes per-member μ columns
  `mu_00…` (M extra cols/footprint) for the EXACT epistemic; OFF by default so
  file sizes are unchanged. Enable it on the uncertainty regen.
- `tccon_uncertainty.py` — `side_b_uncertainty(corr_col=…)` so u_colloc uses the
  chosen corrected field; **u_colloc hardened** (see below).
- `tccon_collocate.py` — `collocate()` now also returns `tccon_err_mean` (window
  mean `xco2_error`) for u_meas. Backward-compatible (extra dict key).
- `tccon_comparison_report.py` — new `--uncertainty` (implies `--ak-harmonize`,
  §6), `--rope-delta` (0.5), `--decorr-km` (15). Emits `tccon_uncertainty*.{md,csv}`
  and appends the Phase-4 block (headline M3, global M4, ⟨z²⟩) to the main report.
  All guarded — default runs are byte-for-byte unchanged (verified).

**u_colloc robustness fix (found on the real 70-case run):** the raw OLS
gradient×distance blew up to ~15 ppm on a near-degenerate footprint geometry
(df 2015-07-15). Fixed: 4·(1.4826·MAD) outlier trim before the plane fit, and
**cap u_colloc at the robust field spread** (a point station cannot credibly
differ from the footprint-cloud mean by more than the field's own variability).
After the fix, across all 70 harmonizable cases: u_harm 0.12–0.18, u_colloc
0.03–1.43, **u_TC 0.15–1.45 ppm** (no blow-ups; colloc still dominant). NOTE the
earlier §3 hand-test ny u_TC=1.915 was itself a pre-cap colloc blow-up.

**Validated (real data, tag de_prof_reg_mix_bc, 100 km):** report runs to
completion with `--uncertainty`; AK harmonization + Side B (u_TC) compute on all
70 cases from the real parquet AK operators; the Phase-4 block degrades
gracefully to "no de_sigma → regenerate" because the existing plot_data predates
Phase 0. Default (no `--uncertainty`) run: identical outputs, no new artifacts.

**GATE — remaining to populate the full Phase-4 table (needs a CURC/GPU regen):**
- regenerate the DE-tag plot_data.parquet with the Phase-0 `build_deepens_plot_data.py`
  (`de_sigma` etc.) + `--emit-members` (for exact epistemic). Only then does Side A
  (u_oco) populate, u_D become finite, and M1/M3/M4 + ⟨z²⟩ fill in.
- with that in hand, read ⟨z²⟩: if ≈1 the whole budget is calibrated; if not, tune
  `--decorr-km` (the one free Side-A knob) — it moves only the averaged term, not
  the epistemic floor. That single fit pins the budget and closes the plan.

Two knobs now live CLI flags (`--decorr-km`, `--rope-delta`); the epistemic
fallback-vs-exact choice is automatic (mu_NN presence + single-surface).

**Ready-to-submit regen job:** `curc_shell_blanca_deepens_uncertainty.sh`
(`sbatch` it). Replays the production plot script's active run_case lines
(single source of truth — validated: 128 active cases parse) build-only, writing
plot_data.parquet as a SUPERSET (Phase-0 `de_sigma` etc. + `--emit-members`
mu_NN; keeps the existing corrected/P(near)/raw columns so it overwrites the
production tree non-destructively), then runs `tccon_comparison_report.py
--uncertainty` at 100 km + 50 km. Idempotent (skips a case whose plot_data
already has the uncertainty columns unless `FORCE=1`). Env knobs: `SRC_SCRIPT`
(default the deepens plot script; set to the drift script for drift cases),
`RADIUS_KM`, `DECORR_KM`, `ROPE_DELTA`, `EMIT_MEMBERS`, `EXCLUDE_SITES`.
After it runs, read ⟨z²⟩ in the Phase-4 block to close the GATE above.

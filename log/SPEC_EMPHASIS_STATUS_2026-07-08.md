# Spec-Feature Emphasis & Analysis Redesign — Status

**Date:** 2026-07-08
**Context:** The QF-grouped feature-set ablation
(`results/model_comparison/deep_ensemble/FEATURESET_ABLATION_QF_2026-07-08.md`)
showed the spectral cumulant group is TCCON-parsimony-droppable (no_spec
Δ ≈ −0.006 ppm pooled RMSE) while `xco2_raw − apriori` carries the predictive
load. This effort repositions the spec features on their real strengths —
mechanism evidence, correction safety, MODIS-independent cloud sensitivity —
and rebuilds the phenomenology analysis (`src/analysis/run_all.py`) to be
diagnosable. Companion review: `log/PROJECT_REVIEW_2026-07-03.md` (M1).

**Framing (for the manuscript):** the cumulants establish that the bias is a
photon path-length effect and bound what the correction may touch; the
retrieval-state features are the operationally sufficient predictor of it.
Three spec-feature "jobs": (1) mechanism/phenomenology, (2) plume
safety/selectivity, (3) MODIS-free cloud sensitivity.

---

## 1. Built and tested (all first-look numbers from ~1–6M-row subsets; regenerate on the full 116-date parquet before quoting)

### Land-cover stratification (Job 1)
- `src/analysis/land_cover.py` — MCD12C1 v061 IGBP lookup (0.05° CMG,
  `data/MODIS/MCD12C1/`, years 2016–2020; v006 is the deprecated one).
  Assigns `igbp_class`/`igbp_group` from (lat, lon, date) at load time.
  QC: ocean-glint soundings map to `water` at **99.95%**.
- `src/analysis/land_class.py` — generic group-overlay engine (`group_col`):
  count matrix, per-variable overlay profiles (raw + z-normalized to each
  group's far-cloud 20–50 km baseline), 6-panel k overview,
  effect-size CSV + heatmap (near−far in z-units, 95% CI, n_min guard).
  Includes ref-corrected per-sounding deltas (dk1/dk2/dexp) when ref_* exist.
- Run: `python src/analysis/run_all.py --land-class-only --parquet-fname …`
- **First look (29 dates):** sign flip by class — forest k1/k2 NEGATIVE near
  cloud (strongest WCO₂ ≈ −0.5 z) vs shrubland/cropland/barren positive;
  exp-intercept near-cloud dip scales with surface brightness
  (forest ≈ 0 → barren −0.9 z); XCO₂ BC anomaly positive near-cloud in every
  class (max barren +0.45 z); urban outlier in raw−apriori (−2.2 z, real
  emissions). **TODO:** pre-register the per-band albedo-contrast ordering,
  then regenerate on the full parquet — this is the paper's mechanism figure.
- **K-means native-class robustness: NOT NEEDED** (decision 2026-07-08).
  External-map QC passed; if reviewers push, use MCD12C1's own
  `Land_Cover_Type_1_Percent` purity filter (≥80% cells) instead — it directly
  addresses mixed pixels without the near-cloud circularity of feature-space
  clustering.

### Spec-sensitivity suite (Jobs 2+3) — `src/analysis/spec_sensitivity.py`
Standalone CLI; outputs under `results/figures/cld_dist_analysis/spec_sensitivity/`.
- **Sub-pixel check:** among MODIS-far (≥20 km) soundings, XCO₂ anomaly rises
  monotonically with the cross-band Δk₁ z-index (land −0.01 → +0.04 ppm
  bottom→top decile; σ rises 0.51 → 0.61 ppm). Spectra detect contamination
  below the MODIS 1-km floor. (Caveat to own: far-field Δk₁ can also be
  aerosol; cross-band coherence requirement mitigates.)
- **Shadow vs brightening:** near-cloud split by zexp_o2a ± 0.5 gives coherent
  opposite-signed branches — land anomaly +0.25 (brightened) / −0.16 ppm
  (shadowed); Δk₁ flips with WCO₂ sign-opposed to O₂A/SCO₂.
- **Spec-only near-cloud classifier** (raw per-footprint features, date-blocked
  holdout): AUC land 0.69 (HGB) / 0.66 ocean (logreg). Deployment-compatible
  (no neighbors, no MODIS) → MODIS-free flagging claim.

### Single-case showcase with MODIS RGB (Job 1/3) — `workspace/spec_case_study/`
- `screen_spec_cases.py` — scans the combined parquet for frames where a small
  cloud clips one edge of the swath (same-frame cross-track contrast: min fp
  cld < 2 km, max > 8 km — the swath is only ~10 km wide, so 15 km contrast is
  geometrically impossible), with clear context (baselines) and a short
  near-run (genuinely small cloud). 631 land / 219 ocean candidates on the
  full parquet → `spec_case_candidates_{land,ocean}.csv`.
- `spec_case_figure.py` — per case: Aqua true-color RGB from **NASA GIBS WMS**
  (no login; A-Train ⇒ the daily composite at the overpass IS the matching
  granule; cached under `rgb_cache/`) with footprints colored by cld_dist
  (overview + ±15 km zoom), fp×along-track cross-band Δk1 z heatmap
  ("spectra-only cloud localization"), then along-track traces vs per-fp
  clear baselines: Δk1/Δk2/Δ(exp int−alb)/continuum ratio/Δalbedo/ΔXCO2/cld.
- **102 cases rendered** (56 land / 46 ocean, deduplicated by overpass;
  index with metadata: `case_index.{md,csv}` alongside the figures) —
  awaiting hand-pick. Reading key: real cloud = white in zoom RGB + all-band
  Δk1/Δk2 + continuum/exp-int + XCO2 response; surface feature (river,
  vegetation, bright soil) = Δalbedo & continuum move together, Δk1 quiet —
  valuable as "correction tolerates surface variance" exhibits; RGB-clean +
  spectra-quiet = MYD35 false positive (selectivity exhibit).
- Vetted so far: **Tasman Sea 2018-05-01** (isolated popcorn cumulus,
  brightening + −1.4 ppm) and **N Pacific 2019-02-01** (cloud-street
  approach, continuum ×20, −5 ppm) are showcase-grade; Botswana 2019-08-01
  is a MYD35 **false positive** (dark vegetation patch, no cloud, spectra
  quiet: a selectivity demo in itself).
- Outputs: `results/figures/cld_dist_analysis/spec_case_study/`.

### run_all.py figure-suite trim
Default `full` profile: **~19,968 → ~250 figures** (verified end-to-end on a
1.2M-row subset). fp_0..7 loop → footprint overlay (`run_footprint_overlay`,
land_class engine, `footprints/{ocean,land}/`); stratified overlays-only;
primary XCO₂ target only; sign-split suites dropped (superseded by
shadow/brightening); scatter matrices dropped; ref_corrected default
R1/R2/R3/R5/R10/R14. `--legacy-full` restores everything. Dead r15/r05
`if 0:` blocks deleted. **Column gotcha:** parquet `fp_id` = 16-digit
sounding ID; footprint index is `fp` (0–7).

---

## 2. Plume analyses (Job 2) — Nassar power-plant controls

**Existing** (`workspace/Nassar_plume_analysis/`): screening
(`screen_nassar_plumes.py` → 8 dates processed), preservation diagnostic
(`analyze_nassar_plume_preservation.py` → spread ratios, k-stats, per-fp).
**Key unresolved number:** near-cloud plume cases show
`corrected_over_original_p95_p05` ≈ 0.17–0.26 with corr(xco2_bc, mu) ≈ 0.99
(Lipetsk 2015-08-01, Westar 2023-03-13) — ambiguous between "correction
removes near-cloud noise" and "correction eats plume".

**Plan (status; new scripts in workspace/Nassar_plume_analysis/, outputs in
plume_preservation/):**
1. **Matched control-region null** (`nassar_control_null.py`) ← **DONE
   2026-07-08.** 5/7 cases `consistent_with_controls` — including the two
   clean clear-sky cases (Ghent 2024-04-15 p53, Westar 2023-06-26 p69): the
   spread collapse is generic near-cloud smoothing, preservation concern
   defused there. 2 flagged (`plant_collapses_more`): Lipetsk 2015-08-01
   (p4.8) and Westar 2023-03-13 (p4.0) — both ultra-near-cloud (median cld
   1.3 / 0.75 km). Resolved by items 2–3 below: what collapses there is a
   cloud artifact, not plume.
2. **Along-track transects** (`nassar_plume_transects.py`) ← **DONE.**
   Westar 2023-06-26 (clear): a ~+0.6 ppm bump at closest approach is
   VISIBLY PRESERVED in the corrected trace; mu stays smooth. Lipetsk: the
   removed structure is a −10 ppm dip with clouds < 1 km — cloud bias being
   corrected, not a plume (plumes are +1–3 ppm).
3. **O₂A vs CO₂-band k₁ contrast** (`nassar_k1_contrast.py`) ← **DONE.**
   In both flagged cases o2a_k1 shifts as much as/more than wco2_k1
   (Lipetsk d_o2a_k1 = −0.023 vs CO₂-expected −0.007) = all-band cloud
   fingerprint, confirming the removed structure is cloud-induced. Spec-
   channel plume-removal bound (|∂mu/∂k₁| × expected plume Δk₁): ≤ 0.21 ppm
   worst case, ≤ 0.01 ppm in clear cases — **the spec features are
   plume-safe**; any plume removal must flow through the xco2 features.
   Honest open number: d_mu/d_xco2 ≈ 0.26–0.55 in clear cases (the
   correction removes ~25–55 % of small local clear-sky XCO₂ contrasts via
   the xco2 channel) — refine with items 4–5.
4. **Wind direction** (MERRA-2/ERA5 → `--wind-file`) → downwind−upwind
   enhancement before/after correction ← DEFERRED (user decision 2026-07-08)
5. **Feature-set variants** (full vs no_spec vs no_xco2 through the same
   diagnostic — quantifies item-3's open number) ← BLOCKED on variant
   checkpoints / CURC plot-data for the Nassar dates
6. **Dose-response** (mu vs plant emission rate flat; enhancement not) ←
   needs per-plant emission rates (Nassar 2021 table)

---

## 3. Assessment — what the spec features are actually worth (opinion, Claude 2026-07-08)

**The predictive redundancy is real and should be accepted, not fought.** The
QF ablation is clean: same footprints, same "before", and `no_spec` is TCCON-
neutral (−0.006 ppm pooled) while held-out CV's small spec credit is
in-distribution over-crediting. No amount of re-slicing will turn the
cumulants into a predictive win at the production operating point, because
`xco2_raw − apriori` already *contains* the radiative perturbation the
cumulants measure — the retrieval's own response is the symptom, the
path-length statistics are the mechanism, and a flexible model only needs one
copy. The honest quantitative statement of their information content is the
conditional one: when the xco2 group is absent, dropping spec costs real
skill (held-out land near-cloud RMSE 1.040 → 1.156, ocean 0.495 → 0.582), so
the features are informative, just not *additionally* informative.

**Their real value is established on three other axes, and it is substantial:**

1. **Scientific identity.** Without k1/k2 this is "another ML bias
   correction" competing with Mauceri 2023 / Keely 2023-25 on RMSE tables.
   With them, the paper has a falsifiable physical account: the anomaly
   decays with cloud distance *because* photon path length is perturbed, the
   perturbation flips sign between brightening and shadowing, and its
   amplitude orders by cloud–surface albedo contrast per band (the land-class
   heatmap). The ablation even sharpens this: the fact that retrieval-state
   features suffice *is explained by* the cumulant analysis — k1/k2 show why
   `xco2_raw − apriori` works. That inversion (physics explains the
   predictor, rather than physics competes with it) is the framing I would
   put in the abstract.

2. **Auditability and safety.** The plume verdict was only reachable because
   the cumulants exist: the all-band vs CO₂-band-only k1 contrast is what
   separates "removed a cloud artifact" (Lipetsk) from "removed a plume", and
   the ≤ 0.21 / ≤ 0.01 ppm spec-channel bound is a guarantee no xco2-based
   feature can offer about itself. A correction whose dominant predictor
   contains the signal being corrected NEEDS an independent instrument to
   audit it; the spec features are that instrument. This is, in my view,
   their strongest operational argument.

3. **Coverage where nothing else sees.** Sub-pixel result: among
   MODIS-declared-clear soundings the anomaly rises monotonically with the
   cross-band Δk₁ index — information below the cloud mask's floor, per
   footprint, with no imager (AUC 0.66–0.69 for near-cloud detection). In
   the post-2022 free-drift era this is the only cloud-proximity signal
   available at all. If spec features ever earn a *predictive* keep, it will
   be here (NoMODIS-era QC/flagging, QF=1 recovery), not in the global RMSE.

**Production recommendation:** keep `full` as the production feature set
(continuity, never worse, and the conditional-information argument), report
`no_spec` as the parsimony row, and do not spend more compute trying to make
spec win on TCCON. Preempt the obvious reviewer jab ("your novel features
don't matter") by stating the division of labor plainly: prediction was never
the test of the physics — the cumulants establish the mechanism, bound what
the correction may touch, and extend cloud sensitivity below the imager
floor; the retrieval-state features are merely the operationally sufficient
readout of the same perturbation.

---

## 4. Open items elsewhere
- Regenerate land_class + spec_sensitivity on the full combined parquet
  (CURC or long local run) before quoting numbers.
- MCD12C1 purity-filter robustness row (only if reviewers ask).
- TCCON `run_case` training-date manifest guard; Phase-3 failure accounting
  (§3-Minor of the project review) — unchanged.

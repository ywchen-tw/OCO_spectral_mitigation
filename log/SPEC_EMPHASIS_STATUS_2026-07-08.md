# Spec-Feature Emphasis & Analysis Redesign — Status

**Date:** 2026-07-08
**Context:** The QF-grouped feature-set ablation
(`results/model_comparison/deep_ensemble/FEATURESET_ABLATION_QF_2026-07-08.md`)
showed the spectral cumulant group is TCCON-parsimony-droppable (no_spec
Δ ≈ −0.006 ppm pooled RMSE) while `xco2_raw − apriori` carries the predictive
load. This effort repositions the spec features on their real strengths —
mechanism evidence, correction safety, MODIS-independent cloud sensitivity —
and rebuilds the phenomenology analysis (`src/analysis/run_all.py`) to be
diagnosable. Companion review: `log/PROJECT_REVIEW.md` (M1).

**Framing (for the manuscript):** the cumulants establish that the bias is a
photon path-length effect and bound what the correction may touch; the
retrieval-state features are the operationally sufficient predictor of it.
Three spec-feature "jobs": (1) mechanism/phenomenology, (2) plume
safety/selectivity, (3) MODIS-free cloud sensitivity.

**Update (2026-07-17):** every quantitative anchor of this doc now has a
fold-PCA, reg-consistent edition and the framing SURVIVES unchanged:
- Ablation → `FEATURESET_ABLATION_QF_2026-07-17.md` (lndo01-retrained
  variants, all folds healthy): `no_spec` pooled ΔRMSE **+0.021 ppm**
  (TCCON-neutral confirmed; near-cloud land QF1 +0.041), `no_contam` +0.028;
  the xco2 block carries the correction (`no_xco2` +0.79 pooled / +1.28
  near-cloud land QF1). Quote these instead of the −0.006 below.
- The §2-item-3 open number is CLOSED by the channel attribution
  (`nassar_plumes_variants/nassar_channel_attribution.md`): full removes
  **55 % [40–63]** of control-window clear-sky spread, no_spec **54 %**
  (Δ +1 pp — cumulants do none of the smoothing), no_xco2 **29 %**
  (Δ +26 pp) — the smoothing flows through the xco2-departure channel,
  exactly as §3 argues.
- Related new evidence for §3's "auditability" axis: the raw/bc/ML analysis
  (`RAW_BC_ML_TCCON_2026-07-16.md`) shows the ML layer rediscovers 60–73 % of
  the operational bias-correction variance from footprint-local features and
  partly REVERSES the operational near-cloud over-correction on land — the
  B11-independence companion to the plume-safety bound.

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
  (overview + ±15 km zoom with 5 km scale bar), then along-track traces vs
  per-fp clear baselines: Δk1/Δk2/Δ(exp int−alb)/continuum ratio/Δalbedo/
  absolute X_CO2 BC (robust ±8 ppm y-clamp, clear-median dashline; NOT deltas —
  some fps lack anomaly values)/cld. Journal-figure revision 2026-07-08:
  Okabe-Ito CVD-safe band colors, panel letters (a)–(i), legends outside the
  boxes; the fp×track cross-band Δk1 z heatmap is now opt-in
  (`--with-heatmap`, exploration only). `--fmt pdf` for submission.
- **102 cases rendered + hand-vetted** (56 land / 46 ocean; metadata index
  `case_index.{md,csv}` alongside the figures). Reading key: real cloud =
  white in zoom RGB + all-band Δk1/Δk2 + continuum/exp-int + XCO2 response;
  surface feature (river, vegetation, bright soil) = Δalbedo & continuum move
  together, Δk1 quiet — "correction tolerates surface variance" exhibits;
  RGB-clean + spectra-quiet = MYD35 false positive (selectivity exhibit).
- **Shortlist picked (user, 2026-07-08):** 7 good land + 5 good ocean +
  5 MYD35 false positives + 2 clouds-without-bias →
  `workspace/spec_case_study/shortlist_2026-07-08.csv` (also mirrored as
  `case_shortlist.md` next to the figures). Strongest numbers: land
  2019-10-01 −7.8 ppm; ocean 2016-05-01 −5.7 ppm; Tasman 2018-05-01 −2.0 ppm.
- `spec_case_atlas.py` — **appendix atlas**: one figure per category, one
  column per case, 4 compact rows (zoom RGB / Δk1 / continuum ratio /
  absolute X_CO2 BC with peak-Δ annotation inside the near-cloud shading);
  ≤4 columns per page. 7 pages rendered (`atlas_{category}[_N].png`).
  Appendix plan: one full-detail exemplar (Tasman 2018-05-01) + the three
  category atlases; categories ARE the argument (respond to real
  perturbations / stay quiet on mask false alarms / MODIS proximity alone
  over-predicts bias).
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
   diagnostic — quantifies item-3's open number) ← **DONE 2026-07-17** (on
   the lndo01-retrained foldpca variants; first pass 07-15 was preliminary):
   `nassar_channel_attribution.{csv,md}` — full 55 % [40–63], no_spec 54 %
   (Δ +1 pp), no_xco2 29 % (Δ +26 pp). Item-3's 25–55 % is attributed to the
   xco2 channel; the spec channel does none of the smoothing.
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
  (CURC or long local run) before quoting numbers. ← land_class DONE on the
  full parquet (2026-07: prereg P1–P5 mostly FAIL as written; the sign rule
  lives in WCO2 — dk1 savanna +0.43σ vs barren −0.40σ; GOTCHA: raw vs
  ref-corrected baselining reverses class ordering). spec_sensitivity
  full-parquet regen still pending (needs CURC/large-RAM, with the l′→L′
  figure pass).
- MCD12C1 purity-filter robustness row (only if reviewers ask).
- TCCON `run_case` training-date manifest guard; Phase-3 failure accounting
  (§3-Minor of the project review) — unchanged.

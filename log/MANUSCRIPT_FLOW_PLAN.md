# Manuscript Flow Plan — AMT

**Created:** 2026-07-19  
**Last structural change:** 2026-07-31 (shipborne comparison removed — scope
change entry below; previously 2026-07-26 length trim + Discussion merge +
Appendix B rewrite, 89 → 77 pages)
**Change history:** `log/MANUSCRIPT_FLOW_PLAN_history.md` — 64 dated entries,
2026-07-21 to 2026-07-26. Consult it for *why* a decision was made or whether
an option was already rejected; this document states *what* the manuscript
currently is.

### State as compiled (2026-07-26, 77 pages)

- **Results — 4 subsections.** 4.1 cloud-proximity phenomenology and its
  spectral signature (former 4.1 + 4.2 merged) · 4.2 model comparison and
  feature attribution · 4.3 independent validation (4.3.1 TCCON · 4.3.2
  smoother null · 4.3.3 ocean references · 4.3.4 uncertainty and reliability)
  · 4.4 plume preservation and safety.
- **Main display items: 12 figures, 2 tables** (model comparison, feature-set
  ablation). Figure/table RENUMBERING is deliberately deferred until the
  manuscript is near-final (user decision 2026-07-26).
- **Appendices A–H, all typeset.** A optical-depth/transmittance construction
  (rescoped) · B analysis cohort and target construction (retitled and given
  its prose 2026-07-26) · C model + CV evaluation (now also holds the
  label-noise table beside Fig. C4) · D independent validation (TCCON + ocean
  inventories) · E nulls, negative controls, failure cases · F case studies
  and plume audit · G controlled 3-D RT demonstration · H TEMPO cross-sensor
  feasibility (existence-proof scope).
- **Discussion — 4 subsections** (merged from six on 2026-07-26). 5.1 physical
  interpretation · 5.2 relationship to the operational correction and
  observation recovery · 5.3 meaning and limits of imager independence ·
  5.4 limitations and future work. 5.1 and 5.3 carry their opening paragraphs
  (the skill-versus-trust synthesis, moved out of Results); the rest are
  stubs. Target ≈ 2,000 words total.
- **Journal: AMT.** Copernicus moved to a flat per-paper APC on 2025-01-01
  (€1,800 net; EGU members €1,620; no per-page surcharge, supplements free),
  so length carries no cost penalty — trimming is about reviewability only.
- **Recovery.** `manuscript/` is gitignored. Pre-trim originals are in
  `manuscript/backup/pre_trim_2026-07-26/`; material cut for the dissertation
  is in `manuscript/backup/`.

### SCOPE CHANGE 2026-07-31 — shipborne comparison removed (user decision)

Prompted by the second advisor's scope feedback ("way too much material").
The shipborne EM27/SUN comparison (MORE-2 + MR21-01, 4 cases) is dropped
FULLY, not demoted: it was the sole exception to the AK-harmonized-only
reference basis (2026-07-21 decision), forcing the §3.5 direct-comparison
caveat and the hedged +0.99 → +1.18 ppm offset explanation, and its
headline number was a scatter collapse — the evidence class the §4.3.2
smoother null itself disarms. ATom keeps the ocean story with its own
far-cloud negative control (2017-10-09); the TCCON far-cloud strata are the
second null. Applied batch: log/MANUSCRIPT_REVIEW_SUGGESTIONS_2026-07-28.md
§9 (9.1–9.16). Consequences, superseding any ship mention below:

- **§4.3.3 and Fig. 11 are ATom-only** (fig11a panels a/b; the Fig. 11
  spec and draft text in §4 below are superseded). Fig. E3 (ship clear-day
  control) and Table D4 (ship cases) removed; Table D2 loses its ship
  block (82 → 79 distinct comparison dates); tabB1 loses the ship row;
  tabC4 loses the ship tree.
- **§3.5 is uniformly AK-harmonized** — the direct-comparison sentences
  are gone, and the §7/8.10 direct-comparison anchoring question is
  CLOSED BY SCOPE (no direct comparison remains in the paper).
- **§2 abstract flow move 5/7 and the conclusions outline** must not
  mention shipborne when drafted (validation reads "TCCON and aircraft
  references"); same for the intro ¶6/¶7 (already edited).
- Assets and pre-edit tex in `manuscript/backup/ship_removal_2026-07-31/`;
  commented includes carry `SHIP-REMOVAL 2026-07-31` markers; the
  `\xcoship` macro, fig11a filename, and knapp2020/hanft2021 bib entries
  are cleaned at the final renumbering pass. The terminology-table
  \xcoship row and Supplement S2 ship pages are dissertation-side only.
- MC RT promotion (main-text visibility for the slab experiment) is the
  approved companion direction, pending as §9 item 9.17.

### ADDITION 2026-08-01 — footprint-size analysis (Appendix B, Fig. B5)

User-requested analysis of the influence of footprint size/area on the
near-cloud XCO2 bias and its correction — RAN 2026-08-01 (local, fold-safe;
`workspace/fp_area_analysis.py`, CSVs under
`results/figures/cld_dist_analysis/fp_area/`, headline numbers in
`fp_area_headline.md`). **Figure in the APPENDIX, not the main text (user
decision 2026-08-01):** `manuscript/figures/figB5_fp_area_robustness.{png,pdf}`
(`manuscript/scripts/make_fp_area_figure.py`; file number — print number at
the final renumbering pass). Verdicts: (1) at fixed nearest-cloud distance
the anomaly ATTENUATES with footprint area on both surfaces
(geometry-controlled OLS: ocean +0.079 ± 0.002 ppm km⁻² at 0.5 km, land
−0.271 ± 0.025); (2) the effect is confined to each surface's response zone
(→ real near-cloud interaction, not a scene confound); (3) the
edge-proximity distance-shift hypothesis is REJECTED (best-fit shift ~0 km
— amplitude, not offset); (4) the correction ABSORBS the dependence
(held-out near-cloud residual bias flat across area quartiles: ocean span
0.105 → 0.020 ppm, land 0.105 → 0.051 ppm). Full record + tex edit plan
(10.1–10.5, drafts awaiting approval) in
`log/MANUSCRIPT_REVIEW_SUGGESTIONS_2026-07-28.md` §10. SIDE FINDING —
**§10.6 APPLIED 2026-08-01 (user go-ahead):** Fig. 3's land MEAN curve was
carried by 192 catastrophic QF1 outlier rows (|anomaly| > 100 ppm, all
near-cloud); the generator now screens at the training threshold, the
figures are regenerated, and §4.1/§3.1/Table B2 carry the corrected
median-positive land reading — see the RESOLVED Fig. 3 note in Results 4.1
below and the §10.6 record (incl. the mean–median decomposition).

**Target journal:** *Atmospheric Measurement Techniques* (AMT)  
**Purpose:** Convert the project evidence ledger into a conventional,
reviewer-readable manuscript flow. This document governs narrative order; the
underlying result ledgers remain `log/PROJECT_REVIEW.md` and
`log/TODO_ACCOMPLISH.md`.

## 1. Central manuscript claim

Three-dimensional cloud effects leave a surface-dependent, physically
interpretable fingerprint in individual OCO-2 spectra. Photon path-length
statistics expose that fingerprint, explain the observed cloud-proximity bias,
and provide a trust audit for a probabilistic, per-footprint correction that
requires no cloud information and no neighbor aggregation at inference. (Every
predictor is a per-sounding L2 Lite quantity; a few contamination indicators —
the `h_continuum` band diagnostics — are preprocessor values that internally
summarize local continuum-radiance variability, but the model reads them as
single-sounding inputs and never gathers neighboring footprints itself. Avoid
the stronger "no neighboring-footprint information" wording, which these
features contradict literally.)

The paper should read as one claim with four linked faces:

1. **Diagnose:** XCO2 biases vary systematically with cloud proximity.
2. **Understand:** spectrum-fitted photon path-length statistics connect the
   response to shadowing, brightening, and cloud–surface albedo contrast.
3. **Correct:** a surface-specific probabilistic model reduces independently
   evaluated error.
4. **Audit:** smoothing nulls, feature ablations, plume tests, uncertainty, and
   failure-mode analyses define what the correction does and does not preserve.

The recurring synthesis line is:

> Retrieval-state variables provide most of the predictive skill, whereas
> photon path-length statistics provide physical interpretation, imager-free
> cloud sensitivity, and an independent audit of correction safety.

## 2. Working title and abstract flow

### 2.1 Working title

**Spectrum-internal diagnosis and correction of cloud-proximity biases in
OCO-2 XCO2**

Optional subtitle:

**Photon path-length statistics and validation against TCCON, aircraft, and
shipborne observations**

Keep “deep ensemble” out of the title unless the paper is intentionally
reframed as an ML-method paper. The primary novelty is the spectrum-internal
diagnosis and imager-independent deployment.

### 2.2 Abstract sequence

Write the abstract in seven moves:

1. **Problem:** 3-D cloud radiative effects introduce spatially structured
   XCO2 errors, while quality filtering preferentially removes near-cloud
   observations.
2. **Gap:** existing cloud-aware corrections generally require external cloud
   information; retrieval-only statistical corrections do not by themselves
   explain why they work or whether real CO2 enhancements are preserved.
3. **Method:** derive photon path-length cumulants from individual OCO-2
   spectra, use MODIS cloud distance only for diagnosis and target
   construction, and train surface-specific probabilistic corrections.
4. **Physical result:** near-cloud XCO2 and spectral responses differ in sign
   over land and ocean, consistent with a common cloud–surface albedo-contrast
   mechanism; the WCO2 response changes sign between vegetated and bright
   barren surfaces.
5. **Validation result:** report the current AK-harmonized TCCON headline and
   site-clustered significance. Current production values to verify at writing
   time are mean absolute bias 1.26 to 0.82 ppm, mean footprint RMSE
   2.67 to 1.20 ppm, and improvement in 71 of 75 station-days
   (rounding corrected 2026-07-23l: CSV values 0.816 / 1.196).
6. **Trust boundary:** the correction outperforms a feature-free smoother,
   preserves the tested plume enhancements, and provides calibrated
   per-footprint uncertainty, while representation and high-latitude errors
   remain.
7. **Implication:** MODIS is scaffolding for training and evaluation, but
   deployment is single-footprint and imager-free, supporting application in
   the post-2022 Aqua free-drift period and possible transfer to other
   greenhouse-gas missions.

Do not claim a demonstrated flux-inversion benefit. The supported claim is
that the method creates candidates for observation recovery; downstream
inversion impact remains to be tested.

## 3. Manuscript architecture

### 1 Introduction

**STATUS 2026-07-29: DRAFTED and review-polished — 1_intro.tex is no longer
an outline.** How the draft maps to the plan below:

- **1.1 covered in full.** Opener = "CO2 rises / sources and sinks
  uncertain / satellites provide the constraint" (option B of three
  proposed 2026-07-29). The sub-ppm motivation is anchored to the OCO-2
  mission's ~1 ppm regional-accuracy target, citing eldering2017 +
  crisp2017 already in refs.bib — user decision: frame by the 1-ppm
  requirement, NOT "a few tenths of a ppm"; §1 now states the near-cloud
  biases "can reach about 1 ppm or more, comparable to the accuracy
  requirement". The suggested preferential-screening sentence was applied
  in adapted form ("Quality filtering is therefore not only a loss of
  sample size. It preferentially removes near-cloud observations, and the
  losses concentrate in persistently cloudy regions ..."), with the
  subject "Quality filtering" chosen to match the Massie QF = 0/QF = 1
  evidence and the §5.2 wording — review item 2.20 is closed on both ends.
- **1.2 covered:** operational ACOS chain ("not fully resolved"; the
  cloud-aware guardrail below respected, B10 notation dropped) → RT
  characterization (Massie SHDOM lookup, Chen EaR3T spectra modification,
  Emde MYSTIC) → ML corrections (Mauceri 2023; Mauceri part 1 / Keely
  part 2, 2025).
- **1.3 DEVIATION (accepted):** the four-questions framing and the
  five-item contributions list were not used. The gap and contributions
  live in two compact sentences (physical interpretation + ML +
  single-footprint imager-free inference; independent validation against
  TCCON, aircraft, and shipborne references + explicit plume-preservation
  tests). Deployment independence is stated with the §6-controlled wording
  ("imager-independent at inference"), and the path-length features are
  tied to the 3D-cloud mechanism in the closing paragraph.

Original plan (kept for reference):

#### 1.1 Scientific and measurement problem

- Establish the role of OCO-2 XCO2 in flux inversions, regional carbon
  budgets, trend monitoring, and emission studies.
- Explain why sub-ppm systematic errors matter.
- Introduce cloud-side illumination, shadowing, and altered photon paths as
  mechanisms affecting nominally clear soundings near clouds.
- Reframe screening as spatially correlated sampling, not merely reduced
  sample size: persistently cloudy regions lose observations preferentially.

Suggested agency-level sentence:

> Cloud screening is therefore not only a loss of sample size: it
> preferentially removes observations from persistently cloudy regions,
> potentially reinforcing spatial sampling biases in atmospheric CO2
> inversions.

#### 1.2 Previous approaches and unresolved gap

Organize the literature by approach:

1. 3-D radiative-transfer characterization of cloud adjacency;
2. MODIS- or imager-dependent empirical corrections;
3. general retrieval bias correction and machine learning;
4. the operational OCO-2 bias-correction chain.

Define the gap narrowly: no existing approach is shown here to combine
physical interpretation, single-footprint inference without an imager,
probabilistic uncertainty, independent validation, and explicit
plume-preservation tests.

Avoid claiming that the operational B11 correction is not cloud-aware in any
sense. The supported statement is that it does not fully resolve the observed
surface-dependent near-cloud residuals and can over-correct near-cloud land
soundings in this evaluation.

#### 1.3 Questions and contributions

Frame the study around four questions:

1. Do nominally clear OCO-2 spectra contain a measurable signature of nearby
   clouds?
2. Can photon path-length statistics explain the land–ocean difference in
   XCO2 response?
3. Can a per-footprint model reduce independent-reference error without cloud
   distance at inference?
4. Does the correction preserve real CO2 enhancements, and where does it
   fail?

End with five contributions:

- spectrum-derived photon path-length diagnostics;
- a surface-specific probabilistic correction;
- date-blocked, leakage-guarded evaluation;
- independent TCCON, ATom, and shipborne validation;
- smoother, feature-ablation, plume-preservation, and failure-mode tests.

State deployment independence here. Reserve the full three-tier
imager-independence argument for the Discussion.

### 2 Data

#### 2.1 OCO-2 observations

Report:

- L2 Lite and L1B product versions;
- observing modes and land/ocean definition;
- radiances, retrieval state, meteorology, prior profiles, and quality flags;
- study periods, dates, and sounding counts;
- the datasets used for phenomenology, training, and independent evaluation.

Do not use one sample count for all analyses. The project ledger currently
contains multiple valid-looking cohorts; reconcile and name them explicitly
before drafting.

#### 2.2 MODIS cloud observations

Describe MYD35 Cloudy and Uncertain classes, temporal matching windows,
nearest-cloud-distance calculation, nominal spatial resolution, and the Aqua
free-drift limitation.

State prominently:

> MODIS cloud distance defines and diagnoses cloud proximity and contributes
> to target construction; it is not supplied to the correction model.

Use **imager-independent at inference**, not **imager-independent method**.

#### 2.3 Independent reference observations

Use separate short subsections for:

- TCCON GGG2020;
- ATom aircraft pseudo-columns;
- shipborne EM27/SUN observations.

For TCCON, specify official QC files, averaging-kernel/prior harmonization,
wet-to-dry prior conversion, station coordinates, coincidence criteria, and
the programmatic exclusion of training dates.

#### 2.4 Auxiliary diagnostic datasets

Describe MCD12C1 land cover, GIBS/Aqua RGB imagery, and plume-overpass
catalogues. Clarify that these support interpretation and safety tests and are
not correction inputs.

### 3 Methods

#### 3.1 Cloud distance and within-orbit anomaly target

Define cloud distance before defining the target:

\[
\Delta X_{\mathrm{CO_2},i}
=
X_{\mathrm{CO_2},i}^{\mathrm{bc}}
-
\overline{X_{\mathrm{CO_2}}^{\mathrm{bc}}}_{\mathrm{clear,local}}.
\]

Specify the local latitude window, far-cloud threshold, reference-spread
guard, required reference population, and frozen ocean-r05/land-r15 target
definitions. Explain that the different radii were fixed from development
analysis and assessed on disjoint dates.

Own the limitation immediately: this is an OCO-2-relative target and can
contain real atmospheric gradients.

**Display items:**

- **Fig. 1** — collocation geometry schematic (moved here from Results 4.1,
  2026-07-21c). Exists: `manuscript/figures/fig01_collocation_schematic`
  (`manuscript/scripts/make_collocation_schematic.py`). The anomaly-distance
  decay curves are NOT part of this figure; they open Results 4.1 as Fig. 3.

#### 3.2 Photon path-length statistics

Introduce the transform before empirical results:

\[
\ln T(\tau)=c-k_1\tau+\frac{1}{2}k_2\tau^2-\cdots .
\]

Define \(k_1\) as a relative mean path-length enhancement, \(k_2\) as a
path-length variance, and the fitted intercept as a continuum-reflectance
term. Document band-specific fit orders, channel masks, bounds, exact
least-squares implementation, and the no-Savitzky–Golay production choice.

Use **spectrum-fitted cumulant proxies under the stated transform**. Do not
claim that the fitted values are directly tallied photon-path moments. The
Monte Carlo demonstration supports causal sensitivity to 3-D transport but
does not yet close the fitted-versus-tallied PPDF loop.

Use lowercase math-italic \(l'\) (final 2026-07-22 form, after the
capital-\(L'\) and upright-\(\mathrm{l}'\) trials) throughout text and
figures. In figures the symbol comes from
`plot_style.MEAN_L_LABEL`/`VAR_L_LABEL` (Times italic through the mathtext
cal slot); never retype it. The tex sources currently use \(\ell\) —
unify to plain \(l'\) at writing time.

#### 3.3 Predictors and probabilistic model

Present predictor groups before architecture:

- retrieval-state variables;
- spectral/path-length variables;
- meteorology and profile EOFs;
- geometry and L1B diagnostics;
- contamination indicators;
- footprint identifier.

Then describe separate land/ocean ensembles, architecture, beta-NLL loss,
five folds by five members, regularization, ensemble mean and variance,
Mondrian conformal intervals, and correction guards.

State explicitly that cloud distance is not a predictor (it enters only target
construction and evaluation) and that the correction is applied one footprint at
a time with no neighbor aggregation by the model. Do NOT claim "no
neighboring-footprint information": the `h_continuum` contamination indicators
are per-sounding preprocessor values that summarize local continuum-radiance
variability across adjacent soundings, so the accurate claim is that the model
needs no neighbor *access* at inference, not that it uses no neighbor-derived
information.

**Display items:**

- **Fig. 2** — deep-ensemble architecture schematic, no-cloud variant
  (promoted from Appendix B, 2026-07-21d): panel (a) one Gaussian-head MLP
  member, panel (b) ensemble mixture + conformal calibration, with the
  no-cloud-information-at-inference statement carried in the figure itself.
  Exists: `manuscript/figures/fig02_deep_ensemble_architecture`
  (`PYTHONPATH=src python -m src.models.make_deep_ensemble_figure
  --only no-cloud --out-dir manuscript/figures
  --basename fig02_deep_ensemble_architecture`; the generator stays in
  `src/models/` because `deep_ensemble_ARCHITECTURE.md` documents it).

#### 3.4 Training, validation splits, and leakage control

Describe contiguous date-fold cross-validation, train-only scaling and PCA,
the calibration partition, manifest-based TCCON-date exclusion, and the
label-noise ceiling. The random-vs-date-split comparison is NOT discussed
anywhere in the manuscript (user decision 2026-07-23) — the split-inflation
figure and cv-design schematic are internal reviewer-response material
(`internal_random_split_inflation`, `internal_cv_design`); present the
date-blocked design affirmatively (Fig. C2 fold timeline) without arguing
against the random split.

#### 3.5 Evaluation framework

THE single home of the TCCON comparison protocol (2026-07-23 flow fix):
every Results section that quotes TCCON metrics — §4.3's model/ablation
comparison as much as §4.4's validation — rests on the definitions here,
so no Results section depends on a later one.

Predefine:

- date-blocked target \(R^2\) and RMSE;
- the TCCON sample and coincidence definition (r = 100 km / ±60 min
  primary; 25/50 km and ±30/120 min as sensitivity variants) and the
  AK/prior-harmonised reference as the sole reported comparison basis
  (moved here from §4.4 item 1, 2026-07-23);
- the station-day evaluation unit, station-day bias and absolute bias;
- footprint RMSE and scatter;
- calibration and interval coverage;
- paired Wilcoxon tests;
- site-clustered bootstrap;
- random-effects residual comparison and representation error.

**TCCON evaluation-metric definitions (LaTeX draft, 2026-07-23, moved
here from the Fig. 9 caption block; matches
`tccon_comparison_report._significance` — note "RMS bias" is the quadratic
aggregate of STATION-DAY MEAN biases, deliberately distinct from the
footprint-level RMSE of the third row; do NOT relabel it RMSE):**

```latex
For each station-day $s$ (one TCCON site on one overpass date) with $n_s$
collocated soundings, let
$e_{s,i} = X_{\mathrm{CO2},i} - X^{\mathrm{TCCON}}_{\mathrm{CO2},i}$
denote the footprint residual against the AK-harmonised TCCON reference,
evaluated once for the uncorrected product and once after correction. The
station-day mean bias and per-footprint root-mean-square error are
\begin{equation}
  b_s \;=\; \frac{1}{n_s}\sum_{i=1}^{n_s} e_{s,i},
  \qquad
  R_s \;=\; \Bigl(\frac{1}{n_s}\sum_{i=1}^{n_s} e_{s,i}^{2}\Bigr)^{1/2}.
\end{equation}
Figure~9a (Results 4.4) summarises the $S = 75$ station-days by three aggregates:
the station-day-equal mean absolute bias
$\overline{|b|} = S^{-1}\sum_s |b_s|$, which weights every station-day
equally regardless of its footprint count; the root-mean-square bias
$\mathrm{RMS}_b = (S^{-1}\sum_s b_s^{2})^{1/2}$, a quadratic aggregate of
the same station-day mean biases that emphasises the largest-bias
station-days (it is not a footprint-level error metric); and the mean
per-footprint RMSE $\overline{R} = S^{-1}\sum_s R_s$, which measures
footprint-level scatter about the reference. Each row of Fig.~9a shows
the change $\Delta$ (corrected $-$ uncorrected) of one aggregate, so
negative values are improvements: $\overline{|b|}$ and $\mathrm{RMS}_b$
respond only to systematic station-day offsets, whereas $\overline{R}$
also responds to within-overpass scatter. Confidence intervals and
$p$-values come from a site-clustered bootstrap: the 18 sites are
resampled with replacement, each drawn site contributing all of its
station-days ($B = 10\,000$ resamples), which respects the strong
within-site clustering of the sample (e.g.\ R\'eunion contributes 14
station-days); the interval is the 2.5--97.5 percentile range of the
resampled $\Delta$, and the two-sided
$p = 2\min[P(\Delta \ge 0),\, P(\Delta \le 0)]$, floored at
$2/(B+1)$. The paired Wilcoxon test quoted beneath the panel treats
station-days as exchangeable pairs and is reported as a
clustering-agnostic cross-check.
```


Define the common-protocol baselines: Ridge, XGBoost, the deep ensemble,
the feature-free orbit smoother, and feature-set ablations. TabM and the
structured DCN are EXCLUDED from the main text (user decision
2026-07-23) — they appear only in the Appendix C common-protocol
results.

#### 3.6 Safety and robustness experiments

Describe without reporting outcomes:

- far-cloud and clear-day negative controls;
- Nassar power-plant transects and matched control windows;
- spectral-channel plume-removal bounds;
- target-radius and TCCON-coincidence sensitivity;
- raw versus operational-BC versus ML comparison;
- surface and environmental failure-mode stratification.

### 4 Results

**Structure as compiled (2026-07-26 trim). The `#### 4.x` headings below keep
their ORIGINAL numbers so the drafting notes stay traceable; the arrow gives
the paper section each one became.** The `.tex` files were re-synced to the
printed numbers on 2026-07-26, one file per section, labels unchanged:
`4.1_phenomenology_spectra` · `4.2_model_comparison` · `4.3.1_tccon_val` ·
`4.3.2_correction_vs_smoothing` · `4.3.3_ocean_far_cld` ·
`4.3.4_uncertainty` · `4.4_plume`.

| Plan heading | Paper section |
|---|---|
| 4.1 Cloud-proximity phenomenology | §4.1, first half |
| 4.2 Spectral evidence for a common 3-D mechanism | §4.1, second half (merged) |
| 4.3 Model comparison and feature attribution | §4.2 |
| 4.4 Independent TCCON validation | §4.3.1 |
| 4.5 Distinguishing correction from smoothing | §4.3.2 |
| 4.6 Ocean validation and far-cloud controls | §4.3.3 |
| 4.8 Uncertainty and failure modes | §4.3.4 |
| 4.7 Plume preservation and correction safety | §4.4 — LAST in the paper |

§4.3 opens with a short lead-in naming the three independent references; the
four validation topics are `\subsubsection`s inside it.

#### 4.1 Cloud-proximity phenomenology → paper §4.1 (first half)

Lead with the observation:

- near-cloud coverage statistics of the analysis set (see Display items);
- positive near-cloud response over land and negative response over ocean;
- characteristic land-r15 and ocean-r05 distance scales;
- quality-flag composition and observation loss with cloud distance.

This result establishes the problem before introducing correction skill,
and (2026-07-21g) motivates the surface-specific target radii on the page:
show the common r10 target first, then the adopted r05/r15 targets.

**Draft results text (interpretation moved out of the Fig. 3 caption,
2026-07-22h). SUPERSEDED IN PART 2026-08-01 (§10.6 applied): the "land
response still ~0.5 ppm at the cut" and the closing mean-vs-median /
tail-driven sentences were outlier artifacts — the tex now carries the
screened reading (land median-positive, persisting to ~15 km; QF0
snow-free symmetrically +0.17 ppm; the negative 0–2 km mean carried by
flagged/snow scenes). The ocean sentences and the r05/r15 motivation
stand.**

> Under the common 10-km target (Fig. 3a) the ocean response has decayed
> by ~5 km, well inside the threshold, whereas the land response is still
> ~0.5 ppm when the 10-km reference cut truncates it by construction — the
> common threshold is too tight for land and unnecessarily wide for ocean,
> motivating the surface-specific radii. Under the adopted production
> targets (Fig. 3b) the land response indeed persists to ~15 km while the
> ocean curve is unchanged. The near-cloud response is opposite in sign —
> negative over ocean, positive over land — before any correction is
> applied. Mean and median are drawn separately because they disagree in a
> diagnostic way: over ocean the whole distribution shifts negative near
> cloud, whereas over land the bin mean far exceeds the nearly unmoved
> median — the land bias is carried by a skewed tail of strongly affected
> soundings rather than by a shift of the full population.

**Display items:**

- **Fig. 3** — two-panel anomaly-vs-distance figure (moved here from 4.2,
  2026-07-21g): (a) both surfaces under the common r10 target at 1-km bins
  — ocean decayed by ~5 km, land still ~0.5 ppm at the 10-km reference cut
  — motivating the per-surface radii; (b) the adopted production targets
  (ocean r05, land r15) with their reference thresholds marked. TWO
  candidate renderings pending a choice (2026-07-21h, revised i): compact
  1×2 curves — IQR shading + dashed median + solid mean
  (`manuscript/figures/fig03_anomaly_decay`) — and 2×2 per-bin box plot
  with mean overlay (`manuscript/figures/fig03alt_anomaly_decay_boxplot`);
  both produced by `manuscript/scripts/make_anomaly_decay_figure.py`. Both
  now expose the same key contrast: over land the MEAN (up to ~2 ppm) far
  exceeds the barely-moving MEDIAN — the land bias is tail-driven — while
  over ocean the whole distribution shifts. Anomalies return to zero
  beyond the respective reference threshold by construction — state this
  in the caption.
  **RESOLVED 2026-08-01 (was the OPEN ISSUE from the footprint-size
  analysis; applied as MANUSCRIPT_REVIEW_SUGGESTIONS §10.6, user
  go-ahead):** the "MEAN far exceeds the barely-moving MEDIAN /
  tail-driven land bias" reading above is RETRACTED — it was carried by
  192 catastrophic QF1 rows with |anomaly| > 100 ppm (median ≈ 3,974
  ppm, all within 15 km of cloud; ocean has zero). The generator now
  screens at `models.pipeline.MAX_ABS_ANOMALY_PPM` (training
  population); fig03/fig03alt/figB2 regenerated. CORRECTED reading, now
  in the tex: over land the MEDIAN is positive throughout (+0.13 at
  0–1 km, persisting to ~15 km under r15; +0.05 at the r10 cut — the
  r15 motivation survives via the median), while the 0–2 km bin MEAN is
  negative (−0.51 at 0–1 km). Decomposition (2026-08-01 follow-up): the
  QF0 snow-free stratum is symmetrically positive (mean = median =
  +0.17); the sub-−2-ppm tail carrying the mean is 88 % QF1 / 12 % snow,
  with shadowing the systematic secondary (shadow-branch mean −0.30 vs
  brightening −0.04) — full numbers in REVIEW_SUGGESTIONS §10.6.
  §4.1 prose + caption, §3.1 methods screen sentence, and the new
  Table B2 "Target outlier screen" row are APPLIED (originals in
  `manuscript/backup/fig3_screen_2026-08-01/tex_originals/`; build
  clean, 82 pages). The draft results text above predates this and is
  superseded on the land mean/median sentences.
- **Near-cloud coverage statistics — present as ONE SENTENCE, not a table**
  (four tightly related percentages do not earn a float; a table would also
  duplicate the Appendix B cohort inventory). Computed 2026-07-21 from
  `combined_2016_2020_dates.parquet` (17.75 M footprints with valid cloud
  distance, 99.9 % of the 17.77 M-row / 116-date analysis set): 41.7 % of
  all footprints lie within 4 km of the nearest detected cloud and 60.7 %
  within 10 km; 59.1 % of ocean footprints lie within the 5 km ocean target
  radius and 46.8 % of land footprints within the 15 km land target radius
  (median nearest-cloud distance 3.5 km ocean, 17.9 km land). Re-freeze
  these numbers with the final cohort (§7 item 8).
- No main-text table; the cohort/attrition inventory is Appendix B
  (Table B1), which should carry the same thresholds as a column so the
  coverage sentence is auditable.

#### 4.2 Spectral evidence for a common 3-D mechanism → paper §4.1 (second half; MERGED with 4.1 on 2026-07-26)

Build a three-part evidence chain:

1. band- and surface-resolved \(k_1/k_2\) responses with cloud distance;
2. the WCO2 land-cover sign rule: savanna approximately \(+0.48\sigma\) and
   barren approximately \(-1.29\sigma\) (per-surface reference, 2026-07-22e;
   r10 prereg-scored values were \(+0.43/-0.40\sigma\));
3. shadow and brightening branches with opposite XCO2 responses.

Interpret ocean as the dark endpoint of the cloud–surface contrast axis, but
label this as a synthesis supported by the observations rather than a closed
quantitative derivation.

Use the verified wording:

> The sign of the WCO2 response follows the measured contrast where land
> classes straddle the cloud albedo; response magnitudes do not monotonically
> follow contrast.

**Draft results text (sign-rule interpretation moved out of the Fig. 4/5
captions, 2026-07-22h):**

> The sign of the WCO2 Δ⟨l′⟩ response follows the sign of the
> cloud−surface albedo contrast across the full surface axis of Fig. 4:
> positive over dark ocean (+0.09σ) and vegetated classes (savanna
> +0.48σ), negative over bright barren surfaces (−1.29σ). Magnitudes do
> not rank with contrast, so the albedo-contrast mechanism enters as a
> sign rule. The sparse urban class is thin and is not interpreted. The
> continuum-reflectance response (Δexp-int) is strongly negative over
> ocean, consistent with shadow-dominated darkening of the dark surface.
> The shadow and brightening branches of Fig. 5 carry opposite-signed
> XCO2 anomalies and distinct band-resolved Δ⟨l′⟩ responses, and are
> separable from a single footprint's spectrum alone.

**Draft results text — QF-population paragraph (INSERTED 2026-07-22h,
user-approved; follows the sign-rule paragraph above):**

> Figure 4 is computed from quality-flag-0, snow-free soundings. Because
> the path-length features derive from the L1B radiances, the operational
> quality flag has no mechanical connection to them; its role here is to
> define the physical population, since the flag rate is both
> distance-dependent (rising toward cloud) and class-dependent, and
> flagged scenes carry competing scattering perturbations (in-field-of-view
> cloud, aerosol) that are not the clear-scene proximity effect under
> study. Repeating the analysis on flagged-only and all-flag populations
> (Appendix B, Fig. B4) leaves the sign of every ocean and vegetated-class
> cell unchanged; the single exception is barren, where 67 % of soundings
> are flagged (desert dust and bright-scene retrieval failures) and the
> flagged population responds *positively* — toward the cloud signature —
> diluting the clear-scene response (−1.29σ) to −0.23σ when all flags are
> pooled. This flip corroborates rather than weakens the sign rule: it is
> consistent with the features responding to in-scene scattering
> contamination itself, uniformly positive across all surfaces, while the
> quality-filtered population isolates the clear-scene albedo-contrast
> response. The quality-flag-0 estimates are, if anything, conservative,
> since near-cloud soundings that survive the filter are the
> least-perturbed scenes.

Do not use the superseded “forest sign flip” or “albedo-contrast ordering”
claims. Point to the Appendix F Tasman case study rather than closing with
it in the main text (moved 2026-07-22c: the multi-panel case figure is too
long); category atlases are author-side S3 staging (Supplement
backup-only, 2026-07-28).


**Display items:**

- **Fig. 4** — surface-stratified sign-rule heatmap with an OCEAN column
  (dark endpoint of the albedo-contrast axis):
  `manuscript/figures/fig04_landclass_effect_heatmap`
  (`manuscript/scripts/make_landclass_heatmap_figure.py`; ref-corrected
  delta rows only; population = QF 0 + snow-free — this filter is
  LOAD-BEARING, with QF1/snow included the barren column flips sign).
  Reference: PER-SURFACE, ocean r05 / land r15 (ADOPTED 2026-07-22e,
  consistent with the production target radii). Headline WCO2 Δ⟨l′⟩
  values: ocean +0.09σ, savanna +0.48σ, barren −1.29σ; the small urban
  class is reference-sensitive (−0.89σ under r10 → −0.02σ) and is not
  interpreted. Effect sizes:
  `manuscript/figures/fig04_landclass_effect_sizes.csv`.
- **DECISION 2026-07-22e — per-surface reference ADOPTED as primary**
  (resolves the 2026-07-22d open decision). Rationale: Fig. 3 itself shows
  the land response extends past 10 km, so the common r10 reference is
  contaminated over land and attenuates the land effects (barren −0.40σ →
  −1.29σ once the r15 reference removes the contamination); per-surface is
  also consistent with the production target radii. The common-r10 variant
  is kept as an INTERNAL check only (`--reference common-r10` →
  `internal_landclass_r10` + `internal_landclass_r10_effect_sizes.csv`,
  renamed 2026-07-22p);
  per decision 2026-07-22j it is NOT discussed in the manuscript, main text
  or appendix (the QF0/QF1 sensitivity of B4 is the only Fig. 4 robustness
  presented). Author-side caveats worth remembering: the stricter r15
  reference roughly halves the valid near-cloud land sample (scene
  selection shifts toward less-cloudy orbits), and the small urban class is
  reference-sensitive. NOTE: the preregistration-scored numbers (savanna
  +0.43σ / barren −0.40σ) were computed on r10 — if the scored prereg
  outcome is ever quoted, cite the r10 variant explicitly; the manuscript's
  primary numbers are the per-surface ones.
- **QF sensitivity (2026-07-22g)** — the spectral features are L1B-derived,
  so the L2 quality flag has no mechanical link to them; the QF0 filter
  instead controls the *physical population* (flag rate is distance- and
  class-dependent; flagged scenes carry competing scattering perturbations).
  Variants: `figB4a_landclass_qf1` (QF1-only) and `figB4b_landclass_allqf`
  (no QF filter), CSVs alongside (`figB4a_landclass_qf1_effect_sizes.csv`,
  `figB4b_landclass_allqf_effect_sizes.csv`; renamed to the B4 letters
  2026-07-22p); snow-free always. Result:
  every vegetated/ocean sign is stable across the three populations
  (WCO2 Δ⟨l′⟩ savanna +0.48/+0.54/+0.56σ for QF0/QF1/all); ONLY barren
  flips (−1.29σ QF0 → +0.47σ QF1 → −0.23σ all-QF), and 67 % of barren
  soundings are QF1 (1.34 M of 1.99 M — desert dust + bright-scene
  failures). The QF1 barren response is *positive*, i.e. toward the cloud
  signature, consistent with flagged in-FOV contamination/aerosol
  overwhelming the clear-scene albedo-contrast response. Appendix B
  sensitivity sentence: signs stable except where flagged contamination
  enters (barren), expected because the flag marks scenes with competing
  scattering perturbations; QF0 is also conservative (near-cloud QF0
  survivors are the least-perturbed scenes, attenuating — not inflating —
  the reported effects). All-QF wetland cells (thin class, 17.8 k, appears
  only without the QF filter) are not interpreted.
- **Fig. 5** — shadow vs brightening branches (O2A exp-intercept split) with
  opposite XCO2 responses; single panel since 2026-07-22c (the Tasman RGB
  case study moved to Appendix F — too long for the main text).
  REGENERATED in the locked style 2026-07-22r by
  `manuscript/scripts/make_shadow_brightening_figure.py` (redraws from the
  saved `shadow_brightening_stats.csv` — no parquet run needed; legend
  moved BELOW the panels, the old in-panel legend covered the O2A curves;
  pre-restyle caveat cleared). Ocean companion staged as
  `internal_shadow_brightening_ocean` until it gets a slot.
  **UPDATE 2026-08-01 (§10.7 applied, user request):** the §4.1 tex now
  carries the full three-class definition (z_exp = Δexp-int_O2A/σ_ref;
  shadowed < −0.5, neutral |z| ≤ 0.5, brightened > +0.5, with the
  physical reading of each class), the Fig. 5 caption is rewritten to
  that definition (<10 km window, 10-km reference), and a NEW
  condition-resolved sign paragraph follows the Fig. 5 float
  (albedo-tercile × illumination two-way: dark×shadowed −0.12 ppm is the
  only negative QF0 land cell; ocean all-negative, shadowed deepest;
  AOD/wind/SZA modulate amplitude; aerosol sign flips with surface under
  the contrast rule). Data: `workspace/bias_sign_conditions.py` →
  `results/figures/cld_dist_analysis/bias_sign_conditions/`.
  **UPDATE 2026-08-01b (user decision — Fig. 5 reference unified):** the
  manuscript Fig. 5 now uses the PRODUCTION per-surface reference
  (`spec_sensitivity.py --analyses shadow --reference production`, run
  locally on the full parquet; QF0 snow-free as before, + the 100-ppm
  anomaly screen, land anomaly = r15 target) → stats in
  `spec_sensitivity/prodref/`; the common-r10 CURC stats/figure remain
  as the internal originals (`--reference common-r10` in both scripts →
  `internal_shadow_brightening_r10_*`). CONSEQUENCE for the evidence
  chain: under the production reference the branches ORDER the land
  anomaly (brightened ≈ +0.3 ppm, neutral ≈ +0.14, shadowed ≈ 0, turning
  negative only over dark surfaces per the two-way) rather than carrying
  opposite signs — the "opposite XCO2 responses" phrasing in older plan
  notes/captions is superseded; Δ⟨l′⟩ branch responses stay
  opposite-signed. §4.1 sentence + Fig. 5 caption updated accordingly;
  build clean.
- No main-text table; the Tasman case is Appendix F (Fig. F1); the vetted
  inventory is typeset Table F1; category atlases are author-side S3
  staging (Supplement backup-only, 2026-07-28).

**Effect-size definition for Fig. 4 (LaTeX draft, for Methods 3.2 or the
caption's supporting text; matches `land_class.build_effect_sizes`):**

```latex
For each surface class $g$ and reference-corrected spectral variable
$\Delta v$, the near-cloud effect size shown in Fig.~4 is the
far-field-normalised difference
\begin{equation}
  E_z(\Delta v, g) \;=\;
  \frac{\overline{\Delta v}_{g}^{\,\mathrm{near}}
       - \overline{\Delta v}_{g}^{\,\mathrm{far}}}
       {\sigma_{g}^{\mathrm{far}}(\Delta v)} ,
  \label{eq:effect-size}
\end{equation}
where $\Delta v_i = v_i - \bar{v}_i^{\mathrm{ref}}$ is the per-sounding
departure of $v \in \{\langle l'\rangle,
\mathrm{var}(l'), \text{exp-intercept}\}$ (per band) from the
mean over the same-orbit clear-sky reference population of the
surface-specific production target (cloud distance $> 5$\,km over ocean
and $> 15$\,km over land, within $\pm 0.25^{\circ}$ latitude), the near
and far windows are 0--5\,km and 20--50\,km of
nearest-cloud distance — identical for every class, ocean included, so all
cells are directly comparable (the far window lies beyond the response
scale of both surfaces, and the near window contains the full ocean
response and the strongest part of the land response) — and
$\sigma_{g}^{\mathrm{far}}$ is the standard
deviation of $\Delta v$ in the far window of class $g$, so that $E_z$
reads as the number of far-field standard deviations by which the class
moves near cloud. The analytic 95\,\% confidence interval is
$1.96\,[(\mathrm{SE}^{\mathrm{near}})^2 +
(\mathrm{SE}^{\mathrm{far}})^2]^{1/2} / \sigma_{g}^{\mathrm{far}}$;
asterisks in Fig.~4 mark cells with $|E_z|$ exceeding this interval. Only
quality-flag-0, snow-free soundings enter (sensitivity to the quality-flag
population: Appendix~B); cells with fewer than 500 soundings in either
window are blanked. The surface-specific reference radii match the
production anomaly targets of Sect.~3.1.
```

(Verify the 5/15 km / ±0.25° reference parameters against
`src/constants.py` at writing time. Per-surface reference ADOPTED
2026-07-22e; the common-r10 robustness variant lives in the `_r10` files.)


#### 4.3 Model comparison and feature attribution → paper §4.2

Open with a THREE-sentence date-blocked headline (extended from two,
2026-07-23) so the model-selection result has a stated skill basis:
(1) quote the frozen date-blocked \(R^2\)/RMSE by surface; (2) read them
against the label-noise ceiling (corrected reading 2026-07-23, Fig. C4:
ocean skill exceeds even the posterior-σ noise scenario — the label is
cleaner than the retrieval posterior implies — while land headroom is
concentrated near cloud, achieved 0.72 vs 0.88 stated ceiling); (3) note
that the model ordering on
WITHHELD folds matches the TCCON ordering that follows (deep ensemble ≥
XGBoost ≫ ridge; fold RMSE land 0.54/0.55/0.67 ppm, ocean 0.40/0.42/0.54
— Fig. C5, Table C5; ridge under the production output guard, 2026-07-23)
— internal held-out skill and independent validation
agree, so neither the anomaly target nor the validation chain is driving
the model selection — while the near-cloud land tail separates the top
two models far more strongly at TCCON (1.31 vs 1.56 ppm, land ≤15 km)
than aggregate fold RMSE suggests. Cite Appendix C for the full cross-validation
analysis (fold dispersion, ceilings, and uncertainty calibration; the
random-split comparison is internal-only per 2026-07-23). Do not expand
beyond those three sentences — the detailed fold-level material lives in
Appendix C so it does not interrupt the path to the independent
validation.

Then present the common-protocol baseline table — opening with the
half-sentence pointer "evaluated under the TCCON protocol of
Sect. 3.5" (the protocol is fully defined in Methods, so no forward
reference to §4.4 is needed) — emphasizing the difficult near-cloud
land tail, and report:

- the full ensemble performs best overall;
- `no_spec` is approximately TCCON-neutral;
- `no_xco2` degrades strongly, particularly for near-cloud land QF1;
- spectral features remain conditionally informative when the XCO2-departure
  block is removed.

State the skill-versus-trust synthesis here for the first time.

**Draft discussion text — spec-emphasis synthesis (added 2026-07-23 from
`log/SPEC_EMPHASIS_STATUS_2026-07-08.md`; numbers are the 2026-07-17
fold-PCA edition — quote those, not the doc's superseded first-look
values):**

> The ablation separates two roles a predictor group can play. The
> retrieval-state departure (xco2_raw − a priori) is the operationally
> load-bearing channel: removing it costs 0.8–1.2 ppm at TCCON (up to
> +1.26 ppm for near-cloud land ≤15 km QF1 soundings), whereas removing the
> spectral path-length group is TCCON-neutral (+0.021 ppm pooled) — for
> prediction alone, the cumulants are dispensable. Their value lies in
> three roles the load-bearing channel cannot fill. First, mechanism: the
> sign rule and shadow/brightening branches of Sect. 4.2 establish that
> the corrected signal is a photon path-length effect — something the
> retrieval-state channel exploits but cannot demonstrate. Second,
> safety: the band-resolved fingerprint of Sect. 4.7 bounds what the
> correction may remove; the channel attribution shows the full model
> removes 55 % [40–63 %] of plume-free local clear-sky spread, the
> no-spectral variant removes 54 % (the cumulants contribute essentially
> none of the smoothing), and the no-retrieval-state variant removes
> 29 % — local smoothing flows through the retrieval-state channel while
> the spectral channel supplies the plume-versus-cloud discriminator.
> Third, imager independence: the spectrum's own cloud-proximity
> sensitivity, including below the MODIS resolution floor (Supplement
> S4), is what makes imager-free deployment and cross-sensor transfer
> (Sect. 5.3, Appendix H) conceivable. In short, the cumulants establish
> what the bias is and bound what the correction may touch; the
> retrieval-state features are the operationally sufficient predictor of
> it.

*(Note 2026-07-28: apply this draft without the "(Supplement S4)"
citation — the S4 bulk cite was dropped 2026-07-27 and the Supplement is
backup-only.)*

**Draft results text (moved out of the Fig. 6 caption, 2026-07-22h):**

> The deep ensemble leads at every slice, and the model ordering is
> decided in the near-cloud land tail (Fig. 6a). The same ordering holds
> on the withheld date-blocked folds of the anomaly target (Appendix C,
> Fig. C5): agreement between internal held-out skill and the independent
> validation means neither the target construction nor the TCCON chain is
> driving the selection — though the fold metrics understate the gap,
> since XGBoost trails the deep ensemble by only ~0.02–0.04 ppm in fold
> RMSE but by 0.25 ppm on the near-cloud land (≤15 km) TCCON tail. In the
> feature-group
> ablation (Fig. 6b), dropping the spectral or contamination groups is
> TCCON-neutral, whereas removing the retrieval-state departure
> (xco2_raw − a priori) costs 0.8–1.2 ppm — the operationally
> load-bearing predictor, whose effectiveness the path-length analysis of
> Sect. 4.2 explains. Load-bearing does not mean sufficient: alone, the
> departure explains only a small fraction of the anomaly target
> (univariate R² 0.21 ocean / 0.10 land against the production targets,
> versus 0.71 / 0.55 for the full ensemble on withheld folds), so most
> of the correction skill lies in its nonlinear interactions with the
> meteorological, geometric, contamination, and profile blocks — which
> is also why the linear ridge baseline saturates at fold R²
> 0.46 / 0.30. Per-feature permutation importance (Fig. 7) gives the
> same picture at feature granularity: the xco2 departure leads on both
> surfaces (22 % of the summed importance on ocean, 34 % on land),
> followed by the surface-pressure departure, glint geometry (ocean),
> the CO2-gradient retrieval diagnostic, and the CO2-prior profile
> EOFs, with the spectral cumulants contributing individually small
> terms. Permutation importance is a CV-side diagnostic that
> over-credits blocks the TCCON ablation shows to be droppable
> (contamination), so the retrained group ablation of Fig. 6b remains
> the primary attribution evidence.

**(Numbers for the sufficiency sentence, computed 2026-07-23 on the
17.77 M-row combined parquet, |y| ≤ 100 ppm filter: univariate squared
Pearson R² of xco2_raw − apriori vs the production targets — ocean r05
0.205, land r15 0.099; r10 variants 0.155 / 0.110. The BC-based
departure xco2_bc − apriori, which is NOT a model input, reads higher —
0.323 / 0.191 production, 0.230 / 0.223 r10 — because the operational
correction already puts bias-corrected retrieval-state signal on both
sides; quote the raw-feature numbers in the paper and keep the bc
variant as an author-side footnote if a reviewer asks.)**

**Display items:**

- **Fig. 6** — baseline + feature-ablation composite, decided in the
  near-cloud land tail. Exists:
  `manuscript/figures/fig06_baseline_ablation`
  (`manuscript/scripts/make_baseline_ablation_figure.py`). Panel (a)
  models: DE / XGBoost / Ridge + uncorrected reference — TabM and
  Structured DCN REMOVED from the main text 2026-07-23 (user decision;
  matches Table 1); the 5-model comparison remains in Appendix C.
  Uncorrected row labeled with X_CO2^B11 per the §6 notation.
  REBUILT 2026-07-23e (user request): THREE slices per row — pooled,
  near-cloud OCEAN ≤5 km, near-cloud LAND ≤15 km (near/far split at each
  surface's production target radius, replacing the single land ≤10 km
  tail) — and the numbers now READ FROM the per-surface-edge report CSVs
  (same sources as Tables 1–2) instead of hardcoded values copied from
  the superseded 2026-07-08 docs (that drift is fixed; panel-b deltas
  are now the 2026-07-17 fold-PCA edition).
- **Fig. 7** — per-feature permutation importance (ADDED 2026-07-23j,
  user request; downstream figures renumbered 8–12, main text now
  TWELVE figures). Exists: `manuscript/figures/fig07_feature_importance`
  (`manuscript/scripts/make_feature_importance_figure.py`): top-12
  individual features per surface by DE permutation ΔRMSE (median over
  the five date-blocked held folds ± fold sd, global stratum), bars
  colored by predictor group; from
  `results/model_comparison/feature_importance/*/importance_de_*_agg.csv`.
  CAVEATS carried in the §4.3 draft prose, not the caption: CV-side
  importance over-credits TCCON-neutral blocks (the FI report's own
  warning), and groups are permuted jointly for the honest number under
  collinearity — per-feature rows are relative indications. The
  retrained group ablation (Fig. 6b, Table 2) stays the primary
  attribution evidence.
- **Table 1** — `manuscript/tables/tab_model_comparison.tex`: DE/XGB/Ridge
  fp-RMSE by slice, AK-harmonized only. RE-SLICED 2026-07-23e (user
  request): ocean rows mirror the land rows (all / QF1 / near / near-QF1
  / far) and each surface splits at its production target radius (ocean
  5 km, land 15 km) — sourced from new report editions run with
  `--cld-edges 0,5,inf` / `0,15,inf` (suffixes `_cldo5_r100km` /
  `_cldl15_r100km`, all 9 trees; production `_r100km` files untouched).
- **Table 2** — `manuscript/tables/tab_featureset_ablation.tex`: mix-DE
  feature-group ablation, AK-harmonized. Same 13-slice re-slicing as
  Table 1 (2026-07-23e).
- The date-blocked skill/ceiling figure belongs to Appendix C (Fig. C4,
  `figC4_skill_vs_ceiling`), not here. The CV-design schematic is
  internal-only since 2026-07-23 (`internal_cv_design`,
  `manuscript/scripts/make_cv_design_figure.py`).

#### 4.4 Independent TCCON validation → paper §4.3.1

Use AK-harmonized TCCON as the sole reported reference (decision 2026-07-21:
manuscript tables and figures carry no direct-reference columns). The direct
(non-harmonized) comparison is reduced to one sentence: note that direct
comparisons were run, that the residual approximately 0.3 ppm scale
difference is explained by the documented B7-to-B11 direct-TCCON anchoring
chain, and cite Appendix D. Report the current production values only after
regenerating the manuscript table from the frozen tag.

The paragraph order should be (item 1 re-scoped 2026-07-23 — the
protocol definition lives in Methods §3.5):

1. one-sentence recap of the sample and coincidence criteria, citing
   Sect. 3.5;
2. mean absolute bias and footprint RMSE;
3. number of improved station-days;
4. Wilcoxon and site-clustered bootstrap results;
5. radius/window and QF0/QF1 sensitivity;
6. high-latitude and worsening cases.

Do not tabulate direct-reference results anywhere in the main text; the
single anchoring-chain sentence above is the only place the direct
comparison appears.

**Draft results text (moved out of the Fig. 8 caption, 2026-07-22h; number 2026-07-23j):**

> Across all 75 station-days (18 sites; overpasses from December 2014 to
> December 2021, every evaluation date disjoint from the 116 training
> dates by the manifest-based leakage guard of Sect. 3.4) the
> station-day mean
> |bias| falls from 1.26 to 0.82 ppm and the mean per-footprint RMSE from
> 2.67 to 1.20 ppm, with 71/75 station-days improved (Fig. 8; both
> aggregates are printed in its annotation box). The improvement
> is significant under a paired Wilcoxon test on station-day |bias|
> (p = 0.0064, n = 75) and under site-clustered bootstrap intervals, with
> and without Ny-Ålesund (Fig. 9), and it holds at every collocation
> radius × window combination (25/50/100 km × ±30/60/120 min), largest at
> the tightest radii — as expected of a real, spatially localised
> correction.

**Display items:**

- **Fig. 8** — headline TCCON station-day before/after dumbbell
  (AK-harmonized). Exists: `manuscript/figures/fig08_tccon_dumbbell`
  (copied from the production fold-PCA tree
  `<TAG>/atrain/tccon_ak_bias_dumbbell_label_r100km.png`).
- **Fig. 9** — significance/robustness panel. Exists:
  `manuscript/figures/fig09_significance_robustness`
  (`manuscript/scripts/make_significance_panel.py`). SPLIT from the
  former two-panel Fig. 7 (user decision 2026-07-23); downstream figures
  renumbered 8→9, 9→10, 10→11 (changelog 2026-07-22u).
- **Table 3** — `manuscript/tables/tab_station_equal_bias.tex`:
  station-equal mean |bias| by QF, AK-harmonized only.
- Full coincidence matrix, station audit, and uncertainty budget are
  Appendix D (Figs. D1–D5, Tables D1–D4).

**Metric definitions:** the b_s/R_s/aggregate definitions and the
bootstrap machinery MOVED to Methods §3.5 (2026-07-23) so that §4.3's
TCCON-based comparison can precede §4.4 without forward references —
Fig. 9's caption cites Sect. 3.5.

(No panel assembly — Figs. 7 and 8 are separate single-file floats since
2026-07-23. fig08's internal panel tags from the generator stay.)

#### 4.5 Distinguishing correction from smoothing → paper §4.3.2

Place the feature-free smoother immediately after TCCON:

- it reduces footprint scatter more than the model;
- it leaves station-day bias nearly unchanged;
- the model reduces both bias and RMSE.

This establishes that the validation improvement is not merely local
denoising.

**Draft results text (moved out of the Fig. 10 caption, 2026-07-22h; number 2026-07-23j):**

> The smoother collapses footprint scatter more strongly than the deep
> ensemble (0.35–0.66 vs 0.78 ppm) yet leaves the TCCON station-day mean
> |bias| essentially unmoved (1.20–1.24 ppm, against 0.82 ppm for the
> model): variance removal alone cannot produce the observed bias
> reduction.

**Display items:**

- **Fig. 10** — smoother-null two-panel (scatter collapse vs unmoved
  station-day bias). Copy exists: `manuscript/figures/fig10_smoother_null`
  (from the production fold-PCA tree
  `<TAG>/atrain/smoother_null/smoother_null_r100km.png`,
  `workspace/smoother_null_figure.py`).
- No main-text table; the full smoother numerical table is Appendix E
  (table with Fig. E1).

#### 4.6 Ocean validation and far-cloud controls → paper §4.3.3

Present ATom, shipborne EM27/SUN, and the far-cloud/clear-day cases where the
correction is nearly inert. Treat these as independent corroboration, not as
equivalent in statistical weight to TCCON. State the limited sample and
reference-scale caveats.

**Draft results text (moved out of the Fig. 11 caption, 2026-07-22h; number 2026-07-23j):**

> Across the 17 ATom legs (14 near-cloud), the near-cloud improvement is
> a scatter reduction, not an offset shift: the median |residual| falls
> from 0.53 to 0.45 ppm and the across-leg spread from ±0.72 to
> ±0.58 ppm, while the signed mean bias is essentially unchanged
> (+0.19 → +0.20 ppm, well within the per-leg pseudo-column σ of
> 0.10–0.19 ppm); the far-cloud date (9 October 2017) is a negative
> control that the correction leaves nearly unchanged (Fig. 11a, b).
> Against the shipborne spectrometers (4 cases, 3 near-cloud) the
> correction collapses footprint scatter (mean σ 0.63 → 0.27 ppm,
> against a mean ship reference σ of 0.29 ppm) without moving the
> clear-sky control day (22 June 2019); the residual absolute offset
> (all-case mean +0.99 ± 0.70 before, +1.18 ± 0.81 ppm after) is
> dominated by reference-scale differences, not created by the
> correction (Fig. 11c, d).

**Display items:**

- **Fig. 11** — ATom + shipborne EM27/SUN summary with the far-cloud/clear-day
  negative controls. REGENERATED 2026-07-23 by
  `manuscript/scripts/make_ocean_validation_figure.py` (drives the
  producers' own plotting functions on the production-tree CSVs):
  CONTINUOUS panel letters across the composite — ATom (a)/(b), ship
  (c)/(d) — resolving the duplicate-letter clash of the copied report
  figures; stat-carrying suptitles REMOVED (numbers moved to the draft
  results text below, per the caption rule); legends/titles use
  X_CO2^B11. Files: `fig11a_atom_summary`, `fig11b_ship_summary`
  (png+pdf).
- No main-text table; leg/case inventories and residuals are Appendix D
  (Tables D5–D6).

**Panel assembly, Fig. 11 (LaTeX draft):**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig11a_atom_summary.png}\\[4pt]
\includegraphics[width=0.95\textwidth]{figures/fig11b_ship_summary.png}
\caption{<insert draft caption above>}
\label{fig:ocean-validation}
\end{figure}
```

Panel letters: native and continuous since 2026-07-23 — ATom carries
(a)/(b), ship carries (c)/(d) from the producers' panel_offset kwarg;
nothing to retag when compositing.

#### 4.7 Plume preservation and correction safety → paper §4.4 (LAST section of Results)

Report:

- matched control-window nulls;
- the preserved Westar transect enhancement;
- the all-band versus CO2-band \(k_1\) discriminator;
- the spectral-channel removal bound of at most 0.21 ppm in the worst tested
  case and at most 0.01 ppm in clear cases;
- approximately 55% [40–63%] removal of plume-free local spread by the full
  model, 54% by `no_spec`, and 29% by `no_xco2`.

The required interpretation is:

> Local smoothing is primarily attributable to the retrieval-state channel,
> not to the photon path-length features.

**Draft results text (moved out of the Fig. 12 caption, 2026-07-22h; number 2026-07-23j):**

> The ~+0.6 ppm enhancement at closest approach to the Westar plant
> survives the correction essentially unchanged (Fig. 12a–c). In the two
> flagged removal windows the O2A band shifts as strongly as the CO2
> bands (Fig. 12d) — the all-band fingerprint of cloud contamination
> rather than of a real plume, whose signature would be confined to the
> CO2 bands; the worst-case plume signal removable through the spectral
> channel is ≤ 0.21 ppm.

**Display items:**

- **Fig. 12** — plume-preservation composite with CONTINUOUS panel
  letters (2026-07-23): (a–c) preserved Westar 2023-06-26 corrected
  transect (XCO2, correction μ, nearest-cloud distance) — REGENERATED
  from `nassar_plume_transects.py --pair westar:2023-06-26` with
  X_CO2^B11 / X_CO2^DE legend labels:
  `manuscript/figures/fig12a_westar_transect`; (d) plume-vs-cloud ⟨l′⟩
  fingerprint contrast (retagged from "(b)", k1 → ⟨l′⟩ in title and
  expected-signature legend): `manuscript/figures/fig12b_k1_contrast`
  (`manuscript/scripts/make_k1_contrast_figure.py`).
- **Table 4 → Appendix F (2026-07-22m; letter per 2026-07-22n merge).**
  `manuscript/tables/tab_nassar_attribution.tex` (control/plant removal
  percentages with full / −spec / −xco2 channel attribution) moves to
  Appendix F, merging with the planned Table F3 (same content — keep
  one). §4.7 prose already carries the three attribution percentages and
  the ≤0.21 ppm bound.
- All transects, per-case bounds, and control nulls are Appendix F
  (Figs. F2–F3, Tables F2–F3).

**Panel assembly, Fig. 12 (LaTeX draft):**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig12a_westar_transect.png}\\[4pt]
\includegraphics[width=0.95\textwidth]{figures/fig12b_k1_contrast.pdf}
\caption{<insert draft caption above>}
\label{fig:plume-preservation}
\end{figure}
```

Panel letters: native and continuous since 2026-07-23 — the transect
carries (a)/(b)/(c), the fingerprint carries (d); nothing to retag.

#### 4.8 Uncertainty and failure modes → paper §4.3.4 (precedes the plume section in the paper; now also carries the bright-surface / worsening-case paragraph moved out of §4.3.1)

End Results by defining the boundary of reliability:

- random-effects residual offset and its reference-scale interpretation;
- representation error increasing with coincidence radius;
- under-dispersed whole-budget uncertainty if between-case variance is
  omitted;
- bright surfaces as the main footprint-level failure stratum;
- high-latitude residuals as station-day bias failures;
- predictive uncertainty as a useful land-failure indicator that cannot
  remove reference-scale bias.

**Display items:**

- **NONE (2026-07-22m).** Fig. 11 is DROPPED from the main text (length
  trim): §4.8 stays at two paragraphs of prose (failure strata +
  uncertainty-as-flag), and Discussion 5.4 carries the QF1 candidate
  counts in a single sentence. The QF1-recovery figure copy
  (the former fig11 file was byte-identical to the QF1 report figure
  now staged as `figD2a`/`figD2b`, so the duplicate was deleted
  2026-07-22o — Fig. D2 is the surviving home of that panel; it
  remains available if a reviewer asks.) Worsening-case and strata tables are Appendix E (Tables E1–E2),
  uncertainty components Appendix D (Table D4).

### 5 Discussion

**Structure as compiled (2026-07-26): SIX subsections merged to FOUR**, one
`.tex` file per section, re-synced to the printed numbers on 2026-07-26:
`5.1_physics_interp` · `5.2_opt_bc_recovery` · `5.3_imager_indepent` ·
`5.4_limitations_future` (the former `5.4_recovery.tex` and `5.6_future.tex`
were folded into their hosts and deleted). Section labels are unchanged. The
`#### 5.x` headings below keep their ORIGINAL numbers so the drafting notes
stay traceable; the arrow gives the paper section each became. Two of the
former subsections were never subsection-sized (5.2 = three sentences citing
Appendix C; 5.4 = one sentence of QF1 counts), and 5.5/5.6 duplicated each
other on PPDF closure and local-contrast attenuation.

| Plan heading | Paper section |
|---|---|
| 5.1 Physical interpretation | §5.1 |
| 5.2 Relationship to the operational bias correction | §5.2, first half |
| 5.4 Observation-recovery implications | §5.2, second half (merged) |
| 5.3 Meaning and limits of imager independence | §5.3 |
| 5.5 Limitations | §5.4, first half |
| 5.6 Future work | §5.4, second half (merged) |

Whole-Discussion target: **~2,000 words** (5.1 ≈ 500 · 5.2 ≈ 400 · 5.3 ≈ 500 ·
5.4 ≈ 600). §5.1 and §5.3 already carry their opening paragraphs, moved out of
Results on 2026-07-26 (the skill-versus-trust synthesis); the rest are stubs.

#### 5.1 Physical interpretation → paper §5.1

Synthesize, rather than repeat, the land–ocean response, WCO2 sign rule, and
shadow/brightening bifurcation. Explain why separate land and ocean models are
physically justified. Distinguish the empirical mechanism evidence from full
PPDF moment closure. **2026-08-01:** the MC paragraph now carries one added
synthesis sentence from the condition-resolved sign analysis (§10.7):
dark×shadowed is the only negative quality-passing land population — the
dark endpoint of the contrast axis — and aerosol deepens the ocean deficit
while mildly strengthening the land positive, obeying the same contrast rule.

#### 5.2 Relationship to the operational bias correction → paper §5.2 (first half; MERGED with 5.4 on 2026-07-26)

Discuss the raw/BC/ML experiment:

- an ML model trained from raw retrievals recovers much of the operational
  increment;
- ML applied to bias-corrected XCO2 performs best overall;
- near-cloud land is complementary: the operational correction can
  over-correct, while the learned residual model partly reverses it.

Frame the proposed method as a complementary residual correction, not a
replacement for the operational correction.

**Display items:**

- **Table 5 → Appendix C (2026-07-22m, adopting the standing option;
  letter per 2026-07-22n merge).**
  `manuscript/tables/tabC8_raw_bc_ml.tex` (raw / bc / ML(bc) / ML(raw)
  fp-RMSE and bias by slice, AK-harmonized; renamed from
  `tab_raw_bc_ml.tex` 2026-07-23) moves to Appendix C as
  Table C8, alongside Fig. C7; Discussion 5.2 stays qualitative (three
  sentences citing Appendix C).

#### 5.3 Meaning and limits of imager independence → paper §5.3

Use three tiers:

1. **Deployment:** no imager or neighboring footprints at inference.
2. **Sensitivity:** the spectrum contains cloud-proximity information,
   including a response below the MODIS resolution floor. The bulk S4
   citation (spec-only classifier AUC sentence) was DROPPED 2026-07-27
   (user decision): this tier is carried qualitatively / by the TEMPO
   demonstration, and the main text now cites the Supplement nowhere.
3. **Transferability:** the conceptual requirements are resolved absorption
   bands, channel-level prior optical depth, and single-footprint spectra.

State the caveat once:

> The imager serves as scaffolding for target construction and validation but
> is removed from the deployed correction.

Discuss post-2022 OCO-2 use and OCO-3, CO2M, GOSAT-GW, and TEMPO as prospects,
not demonstrated cross-sensor equivalence.

#### 5.4 Observation-recovery implications → paper §5.2 (second half; MERGED with 5.2 on 2026-07-26)

Use production QF1 counts to quantify candidate recovery by distance, surface,
and region — ONE SENTENCE with the counts, no display item (the former
data-recovery figure dropped 2026-07-22m, pre-22u numbering). Avoid calling observations “usable” solely because RMSE is lower;
formal acceptance also requires retrieval validity, uncertainty calibration,
and downstream application tests.

Recommended language is **candidates for recovery** until an explicit
acceptance criterion and inversion experiment exist.

#### 5.5 Limitations → paper §5.4 (first half; MERGED with 5.6 on 2026-07-26 — attach each future-work item to the limitation it resolves)

Collect the limitations in one subsection:

- anomaly-target circularity and possible real-gradient contamination;
- MYD35 mask semantics, sub-pixel cloud, and parallax limitations;
- land-heavy TCCON sampling and limited ocean-reference sample;
- high-latitude and representation-error residuals;
- absence of fitted-versus-tallied PPDF moment closure;
- possible attenuation of local CO2 contrast by retrieval-state predictors.

#### 5.6 Future work → paper §5.4 (second half; MERGED with 5.5 on 2026-07-26)

Keep this short: Monte Carlo PPDF closure, plume-injection OSSE,
transport-model decomposition of the target, cross-sensor validation, and a
downstream flux-inversion assessment.

### 6 Conclusions

Use four short paragraphs:

1. Cloud adjacency produces coherent surface-dependent XCO2 and spectral
   responses.
2. Photon path-length statistics support a unified cloud–surface contrast
   interpretation.
3. A date-blocked, single-footprint probabilistic correction improves
   independent-reference agreement and outperforms smoothing.
4. Predictive skill comes mainly from retrieval-state variables, while the
   spectral physics provides interpretation, safety tests, and a route toward
   imager-free deployment.

Suggested final sentence:

> These results establish a practical route for correcting cloud-proximity
> errors without an imager at inference, while identifying the reference,
> surface, and representation-error limits that must accompany scientific
> applications.

## 4. Evidence and figure flow

The main text uses exactly ELEVEN figures (2026-07-23: the former
two-panel Fig. 7 split into Fig. 7 dumbbell + Fig. 8 significance, so
8→9, 9→10, 10→11; the former optional-eleventh QF1/failure figure stays
DROPPED per 2026-07-22m — §4.8 carries no display item and Discussion
5.4 carries the QF1 counts in one sentence).
(The earlier option of compositing the decay curves with the k1/k2 distance
responses lapsed when Fig. 3 moved to Results 4.1, 2026-07-21g; if a
figure must be cut, fold the k1/k2 distance responses into Fig. 4
instead.)

**Section assignments updated 2026-07-26 for the 4-subsection Results.**

| Figure | Manuscript role | Primary message |
|---|---|---|
| 1 | Methods 3.1 | Collocation geometry schematic: how cloud distance and the anomaly target are constructed. |
| 2 | Methods 3.3 | Deep-ensemble architecture (no-cloud variant): member MLP, ensemble mixture, conformal calibration; no cloud information at inference. |
| 3 | Results 4.1 | Anomaly–distance decay: (a) common r10 target motivates the surface-specific radii; (b) adopted r05/r15 targets — opposite-sign land/ocean phenomenon. |
| 4 | Results 4.1 | WCO2 land-cover response changes sign across the measured albedo-contrast axis. |
| 5 | Results 4.1 | Shadow and brightening branches connect spectral response to opposite XCO2 anomalies. |
| 6 | Results 4.2 | Baseline/feature-ablation comparison decided in the near-cloud land tail; date-blocked skill and noise ceiling carried as a headline sentence, detail in Appendix C. |
| 7 | Results 4.2 | Per-feature permutation importance of the deep ensemble (DE ΔRMSE, held folds), both surfaces. |
| 8 | Results 4.3.1 | TCCON before/after station-day dumbbell (AK-harmonized). |
| 9 | Results 4.3.1 | Significance/robustness of the TCCON validation (bootstrap CIs, radius × window). |
| 10 | Results 4.3.2 | The feature-free smoother removes scatter but not station-day bias. |
| 11 | Results 4.3.3 | ATom and shipborne ocean validation plus far-cloud negative controls. |
| 12 | Results 4.4 | Plume-preservation transect and channel-attribution safety budget. |
| — | — | The former optional QF1/failure figure stays DROPPED (2026-07-22m); QF1 counts are one sentence in Discussion 5.4. |

Main-text table budget (**TWO tables since 2026-07-26**, all AK-harmonized
only, generated by `manuscript/scripts/make_manuscript_tables.py` into
`manuscript/tables/`; the generator also writes two appendix files and one
backup file):

| Table | Section | File | Content |
|---|---|---|---|
| 1 | Results 4.2 | `tab_model_comparison.tex` | DE/XGB/Ridge fp-RMSE by slice. |
| 2 | Results 4.2 | `tab_featureset_ablation.tex` | Feature-group ablation by slice. |
| — | backup | `tab_station_equal_bias.tex` | Station-equal mean \|bias\| by QF — CUT 2026-07-26 (dissertation material; the generator now writes it to `manuscript/backup/`). |
| — | Appendix F | `tab_nassar_attribution.tex` | Moved 2026-07-22m (merges with planned Table F3). |
| — | Appendix C | `tabC8_raw_bc_ml.tex` | Moved 2026-07-22m (Table C8, with Fig. C7; file renamed 2026-07-23). |

Per-section figure/table assignments, artifact status, and generator scripts
are in the **Display items** blocks under each Results/Discussion section
above.

### Figure captions

**CAPTION RULE (2026-07-22k, applies to EVERY figure and table caption,
main text and appendix):** captions are DESCRIPTIVE ONLY — what is
plotted, from which data/protocol, what each panel shows, and how to
decode the rendering (colors, line styles, symbols, error bars). No
interpretation, no conclusions, and no result numbers. Numbers are
allowed only when needed to identify what is drawn (sample sizes,
thresholds, bin widths, dates, radii); performance values, effect sizes,
and comparative claims belong in the main-text or appendix discussion
prose (the per-section "Draft results text" blocks hold this material
until drafting). Main-text Figs. 3–10 were brought into compliance in the
2026-07-22h slim-down; apply the same rule when drafting every appendix
caption.

Draft captions live under each section's **Display items** block (moved
2026-07-21f for easier reading), together with LaTeX panel-assembly drafts
for the multi-file figures (10, 11). Shared caveats: numerical values
must be re-verified against the frozen fold-PCA tag at writing time (§7);
captions use the lowercase italic l′ notation (2026-07-22 final) and the AK-only reference decision; internal
"Sect. X" references to be resolved in LaTeX.

Supplement contents are governed by the **Supplement plan (S1–S6)** at the
end of §5 (2026-07-22m; this list previously conflicted with the appendix
sections — resolved there: feature lists/hyperparameters STAY in
Appendix C, the Monte Carlo demonstration is a TYPESET appendix
(G — conditional resolved 2026-07-24b), cross-sensor feasibility STAYS in Appendix H, the compact 3×3
coincidence matrix stays in Appendix D with only extended variants in the
Supplement).

Do not place a corrected transect in the early mechanism section. The early
case study should show spectrum-derived variables and uncorrected XCO2 only;
the corrected transect belongs in the plume-preservation result.

## 5. Appendix plan

**ADMISSION RULE (user-set, 2026-07-22n) — decides where any item goes:**

- **Appendices:** ONLY material reviewers must reasonably read to assess
  the method. If a reviewer can fairly judge the paper without reading an
  item, it does not belong in a typeset appendix.
- **Supplement:** unrestricted in logical scope, but NO new central
  scientific conclusions — nothing may appear there that the paper's
  claims depend on.

The appendices should make the main claims auditable without becoming a second
Results section. Each appendix must have one explicit job: document a method,
test robustness, expose individual cases, or bound transferability. The main
text should state the conclusion and cite the relevant appendix; the appendix
should carry the diagnostic detail needed to reproduce that conclusion.

Use lettered topical appendices in the manuscript. The existing A1–A13 labels
below are working figure identifiers from `log/TODO_ACCOMPLISH.md`, not the
recommended final AMT appendix structure.

**Final appendix lettering (2026-07-22n merge; changelog entries dated
before 2026-07-22n use the FORMER letters):**

| Final | Job | Former |
|---|---|---|
| A | Optical-depth and transmittance construction (rescoped 2026-07-26) | A |
| B | Data, collocation, and target construction | B |
| C | Correction model: architecture, training, CV evaluation | C + D |
| D | Independent validation: TCCON protocol + ocean references | E + H |
| E | Null tests, negative controls, failure cases | F |
| F | Case studies and plume-preservation audit | G |
| G | Controlled 3-D RT demonstration (conditional) | J |
| H | Cross-sensor feasibility, TEMPO (KEEP) | K |
| — | Spectrum-internal sensitivity → Supplement S4 | I |

File names were re-synced to the final letters 2026-07-22o:
`figD1b_cv_design`→`figC3b_cv_design`, `figG_case_tasman`→
`figF1_case_tasman`, `fig11_qf1_recovery_candidate`→
`internal_qf1_recovery_candidate` (no manuscript number).

### Appendix A: Optical-depth and transmittance construction for the spectral fit

**RESCOPED 2026-07-26 (title changed).** The derivation half of this appendix
(Beer–Lambert → ensemble average → Laplace transform → gamma model →
cumulant identification) duplicated Methods §3.2, which already carries the
whole chain with the Irvine/Partain/Stephens citations; it was removed, and
the gamma identity κ = k₁²/k₂ went with it (κ is not used anywhere in the
paper). What remains — and what no citation can supply — is the
implementation: how τ_v, the ILS-convolved channel SOD, and the solar
normalization (Earth–Sun distance, Doppler stretch, ILS convolution) are
built from L1B radiances, ABSCO tables, and the solar line list, plus
Fig. A1 and Table A1. Figure A1 uses 2020-01-01 granule 29252a
(`manuscript/scripts/make_appendix_a_fit_figure.py`). Removed block:
`manuscript/backup/pre_trim_2026-07-26/appendix_A_ppdf_derivation_removed.tex`.
The content list below is the PRE-TRIM plan, kept for reference.

**Purpose:** make the physical observable and numerical fit independently
reviewable.

Include:

- derivation of the cumulant expansion from the Laplace-transform view of the
  photon path-length distribution;
- sign and factorial conventions for \(k_1\), \(k_2\), and higher-order terms;
- band windows, optical-depth construction, channel masks, polynomial orders,
  bounds, and fallback behavior;
- examples of accepted and rejected fits;
- convergence-radius and fit-order sensitivity;
- fitting failure counts by band and failure category, if available.

Planned items:

- **Fig. A1:** representative \ce{O2}A, W\ce{CO2}, and S\ce{CO2} production
  fits from 2020-01-01 granule 29252a, including fitted
  \(\langle l'\rangle\) and \(\mathrm{var}(l')\);
- **Table A1:** complete fitting configuration and QC thresholds.

*Status note (2026-07-26): Appendix A retains only generated Fig. A1 and
Table A1; the figure caption lives in the .tex.*

Do not present spectrum-fitted cumulants as directly tallied photon moments.
The mathematical interpretation, numerical estimator, and Monte Carlo causal
test should remain distinct.

### Appendix B: Analysis cohort and target construction

**As compiled 2026-07-26.** Retitled from "Data provenance, cloud collocation,
and target construction": the collocation parameters are one block of
Table B2 and Sect. 3.1 owns that material, so the old title over-promised.
The appendix previously held five display items and NO prose, with only
Fig. B2 referenced from the body; it now opens with ~660 words introducing
each item. Contents: Fig. B1 target-radius sensitivity · Table B1 cohort
attrition · Table B2 target-construction parameters and guards ·
Fig. B2 QF-population land-class heatmaps. **Table B3 (label-noise ceilings)
MOVED to Appendix C** beside Fig. C4, where the achieved-skill discussion
lives (file renamed `tabB3_` → `tabC3_label_noise_ceilings.tex`; it prints as
Table C3). **ADDED 2026-08-01: footprint-size robustness figure**
(`figB5_fp_area_robustness.{png,pdf}`, file number — prints as the next
B-figure at the renumbering pass; generator
`manuscript/scripts/make_fp_area_figure.py`, data
`workspace/fp_area_analysis.py` → `results/figures/cld_dist_analysis/
fp_area/`): (a) footprint-area distributions + quartile edges (L2 Lite
vertex-polygon areas, QC window 0.2–10 km²), (b, c) anomaly-vs-distance by
area quartile per surface (all-QF, |y| ≤ 100 ppm screen), (d, e)
geometry-controlled area coefficient vs distance (attenuation, confined to
each response zone), (f) fold-safe held-out near-cloud residual bias by
quartile before/after correction (flat after — the correction absorbs the
footprint-size dependence). Appendix prose + main-text pointer sentences
drafted as §10.1–10.4 of `MANUSCRIPT_REVIEW_SUGGESTIONS_2026-07-28.md`
(awaiting approval per the no-unasked-tex rule); spectral effect sizes by
quartile stay CSV-only (ocean sign-stable; the land WCO2 swing is the
land-cover mix aliasing through the strata — named as a confound, not
footprint physics). **Body pointers added** so nothing is orphaned: Table B1 from
Sect. 2, Table B2 + Fig. B1 from Sect. 3.1, Fig. B2 already cited from
Sect. 4.1. Surfaced while writing the prose and now stated there: label
retention is 74 % on ocean but only 53 % on land (a 15 km clear-sky floor is
harder to populate than a 5 km one), so the labeled land population
under-represents persistently cloudy scenes — decide at drafting time whether
this also belongs in §5.4 Limitations.

The content list below is the PRE-TRIM plan, kept for reference.

**Purpose:** expose every selection that defines “near cloud” and the anomaly
label.

Include:

- OCO-2 product versions, modes, date inventories, and sample attrition;
- MYD35 bit interpretation, Cloudy+Uncertain pooling, day/night screening,
  temporal buffers, KD-tree geometry, Earth model, and 50 km cap;
- local clear-reference selection and anomaly guards;
- ocean-r05 and land-r15 target definitions;
- target-radius sensitivity and label-noise-ceiling derivation;
- post-2022 NoMODIS sentinel behavior;
- an explicit data-flow diagram distinguishing training-only cloud information
  from inference inputs.

Planned items:

- **Fig. B1:** SUPERSEDED 2026-07-21d — the no-cloud architecture
  schematic is now main-text Fig. 2 (Methods 3.3). Keep a B-figure slot only
  if a more detailed training-vs-inference data-flow variant proves needed;
- **Fig. B2:** target sensitivity for r05/r10/r15 — GENERATED
  2026-07-22p: `figB2_target_sensitivity` (each surface under all three
  reference radii, bin means; `make_anomaly_decay_figure.py`).
  Complementary to Fig. 3's arrangement, no duplication: it shows the
  ocean curve is target-invariant while the land response is truncated
  at whichever radius the reference allows — the label-robustness
  argument for the 15 km land radius;
- **Table B1:** cohort inventory and attrition from raw soundings to fitted,
  labeled, training, and evaluation populations;
- **Table B2:** all target parameters and reference-population guards;
- **Table B3:** label-noise ceilings by surface and cloud-distance regime;
- **Fig./Table B4:** Fig. 4 quality-flag robustness (2026-07-22h,
  user-approved; the ONLY Fig. 4 sensitivity discussed in the manuscript —
  the common-r10 reference variant is NOT discussed anywhere, main text or
  appendix, per decision 2026-07-22j; its `_r10` files remain in the repo
  as an internal check only). QF1-only and all-flag heatmaps
  — staged 2026-07-22p as `figB4a_landclass_qf1` / `figB4b_landclass_allqf`
  (+ `_effect_sizes.csv` each; generator writes these names directly): every ocean and
  vegetated-class sign is stable across QF populations (savanna WCO2 Δ⟨l′⟩
  +0.48/+0.54/+0.56σ for QF0/QF1/all); only barren flips (−1.29σ → +0.47σ
  QF1 → −0.23σ pooled; 67 % of barren soundings are QF1), consistent with
  flagged desert scenes carrying the cloud-signature-positive perturbation
  of in-FOV contamination/aerosol — phrase as "consistent with", the
  mechanism is inferred from sign and class composition, not demonstrated
  cell-by-cell. Note QF0 conservatism (near-cloud survivors are the
  least-perturbed scenes) and do not interpret the thin all-QF-only
  wetland class.

*Caption note (2026-07-23g) — caption text now lives in the .tex.*

This appendix must resolve the currently mixed 116-date/17.8-million and
140-date/21.5-million cohorts by naming the role of each population.

### Appendix C: Correction model — architecture, training, and cross-validated evaluation

**As compiled 2026-07-26.** Gained **Table C3** (label-noise reference levels,
moved from Appendix B) with an introducing paragraph, placed directly after
Fig. C4 (skill vs ceiling) — the table is that figure's numerical companion.
Appendix C tables now run C1–C10.

(MERGED 2026-07-22n: former Appendix C + former Appendix D — "here is
the model" and "why the model and split design are trustworthy" is one
story; a reviewer auditing CV rigor finds the fold design and the fold
results in one place.)

**Purpose:** provide enough detail to reproduce training without interrupting
the scientific narrative.

Include:

- full feature dictionary with units, transformations, and source products;
- land and ocean architecture, beta-NLL definition, regularization, optimizer,
  learning-rate schedule, early stopping, and seeds;
- fold construction, calibration split, scaler and fold-PCA fitting rules;
- ensemble aggregation and predictive-variance decomposition;
- Mondrian conformal procedure and cloud-distance-dependent inflation, where
  applicable;
- leakage-guard logic and the verified training/evaluation date intersection;
- model and feature-set production tags.

Planned items:

- **Fig. C1:** training and evaluation data-flow diagram — GENERATED
  2026-07-22q: `figC1_dataflow` (`make_appendix_c_figures.py`; carries
  the training-only cloud-information boundary, which also satisfies
  Appendix B's data-flow include bullet — B1 slot stays retired);
- **Fig. C2:** fold timeline showing contiguous date blocks — GENERATED
  2026-07-22q: `figC2_fold_timeline` from the REAL per-fold
  `training_dates.json` manifests (fold-PCA fold structure, read from
  the linreg fold dirs; train/calibration/held roles per fold). This
  supersedes the stylized-timeline caveat on the cv-design schematic
  (now `internal_cv_design`) — the manifests ARE
  local. REBUILT 2026-07-23 as a SINGLE panel (user request): the ocean
  and land manifests are identical fold-for-fold (asserted in the
  generator, which also verified linreg == DE manifests), so one panel
  serves both surfaces and all models;
- **Table C1:** full predictor inventory — GENERATED 2026-07-23:
  `manuscript/tables/tabC1_predictor_inventory.tex`
  (`make_appendix_c_figures.py --only tabC1`; built FROM
  `models.pipeline` `_FEATURE_MAP` + ablation sets so it cannot drift
  from the code; ocean 56 / land 67 inputs incl. fp one-hots + 14
  profile-EOF columns; groups map exactly onto the −xco2/−spec/−contam
  ablations. CAVEAT: the per-feature description strings are drafts —
  verify the marked L2-Lite ones, esp. `s31`, `dpfrac`, `fs_rel_0`,
  against the B11 Data User's Guide at writing time);
- **Table C2:** hyperparameters and training configuration — GENERATED
  2026-07-23: `tabC2_training_config.tex`, read from the ten production
  fold `run_summary.json` configs (asserted identical). NOTE this
  exposed a plan error: production β-NLL uses **β = 1.0** (all ten fold
  runs + the launcher), not the β = 0.5 the Fig. 2 draft caption
  claimed — caption fixed in place 2026-07-23;
- **Table C3:** fold-level sample sizes and performance — GENERATED
  2026-07-23: `tabC3_fold_sizes_metrics.tex` (per-fold date counts from
  the manifests; held-out n / RMSE / R² / conformal coverage + width
  from the fold `de_mondrian_date_kfold_metrics.json`);
- **Table C4:** training manifests and zero-overlap verification —
  GENERATED 2026-07-23: `tabC4_manifest_verification.tex`. The overlap
  column is COMPUTED live in the generator (manifest union = 116 dates,
  2016-01-01–2020-12-15, identical both surfaces) against every eval
  tree: TCCON A-train 75 cases / 51 unique dates, TCCON drift 21/21,
  ATom 8/8, ship 4/4 — all intersections empty (asserted). (The "77
  TCCON eval dates" figure in PROJECT_REVIEW counts a different union;
  the table reports the per-tree counts.)

Keep the main text to predictor groups and essential architecture. Individual
features and hyperparameters belong here.

**Second job (former Appendix D, merged 2026-07-22n) — cross-validation,
baselines, and feature attribution:** show that the selected model and
split design are not arbitrary,
and carry the full cross-validated correction-performance analysis (moved out
of the main Results 2026-07-21; the main text keeps only a two-sentence
date-blocked headline in §4.3).

Include:

- ~~random versus date-blocked performance~~ REMOVED 2026-07-23 (user
  decision): the random-vs-date-split comparison is not discussed in the
  manuscript at all; `internal_random_split_inflation` and
  `internal_cv_design` are kept as reviewer-response material only;
- date-blocked skill by surface and near/far-cloud regime, with fold
  dispersion;
- label-noise-ceiling derivation cross-referenced to Appendix B, with the
  CORRECTED ceiling-relative interpretation (2026-07-23, see Fig. C4):
  the reference-sampling term is the only hard ceiling; ocean skill
  exceeds the posterior-σ noise scenario (the anomaly label is cleaner
  than the retrieval posterior implies, since locally common-mode
  retrieval error cancels and feature-predictable retrieval error is
  what the model corrects); land headroom is concentrated in the
  near-cloud regime (0.72 achieved vs 0.88 stated);
- Ridge, XGBoost, TabM, and deep-ensemble common-protocol results;
- full feature-set ablations, including QF and near-cloud strata;
- raw-XCO2 versus operational-BC versus ML comparisons;
- uncertainty calibration against the anomaly target.

Planned items:

- ~~**Fig. C3:** random-split inflation + cv-design schematic~~ —
  REMOVED from the appendix 2026-07-23 (user decision: no
  random-vs-date-split discussion unless reviewers ask). Files renamed
  `figC3a_random_split_inflation` → `internal_random_split_inflation`
  and `figC3b_cv_design` → `internal_cv_design` (generators updated;
  both stay reproducible for a reviewer response). Appendix C figures
  therefore run C1, C2, C4, C5, C7 — renumber at typesetting time;
- **Fig. C4:** date-blocked skill versus the label-noise ceiling by surface
  and cloud-distance regime — GENERATED 2026-07-23:
  `figC4_skill_vs_ceiling` (achieved DE fold R², mean ± std, in
  all/near/far regimes at the production target radii (ocean ≤5 km versus
  >5 km; land ≤15 km versus >15 km), recomputed from the per-fold held-out
  prediction artifacts,
  against the three ceiling columns of the 140-date CSV). The 2026-07-22q
  blocker is RESOLVED from `analysis/label_noise_ceiling.py` itself: the
  three r2max columns deliberately BRACKET the ceiling — `r2max_ref`
  (reference-mean sampling error only) is the only HARD bound (~0.99);
  `r2max_ref_ret` additionally treats the per-sounding retrieval
  posterior σ² as irreducible noise, an assumption the anomaly target
  breaks twice over (locally common-mode retrieval error cancels in the
  within-orbit difference, and feature-predictable retrieval error is
  exactly what the correction removes), so achieved skill legitimately
  exceeds it (ocean all 0.708 > 0.603; the model's held-out MSE 0.157 is
  even below E[σ²_ret] = 0.202 alone); `r2max_emp` (far-field variance
  as noise) is exceedable for the same reason — far-cloud R² > 0 (ocean
  0.26, land 0.40) proves far-field variance contains predictable
  structure. The figure therefore shows three LABELLED reference lines,
  not one "ceiling", and the honest reading replaces the older "ocean ≈
  noise-limited" line: ocean skill exceeds the posterior-σ scenario,
  meaning the label is less noisy than the posterior σ implies; the
  LAND NEAR-CLOUD gap (achieved 0.72 vs ref_ret 0.88) remains the one
  regime with clear stated headroom. Panels lettered (a) ocean / (b)
  land, legend above the axes (2026-07-23 user feedback). Caption must
  say the ceilings are computed on the pooled 140-date population while
  the bars are per-fold held-out estimates;
- **Fig. C5:** model comparison — surface-level version GENERATED
  2026-07-22q, REBUILT 2026-07-23: `figC5_cv_model_comparison` (fold
  mean ± std RMSE/R² for DE/XGBoost/Ridge, both surfaces; now computed
  from the per-fold artifacts directly, no kfold_agg parse; Ridge under
  the production output guard — no median/hatch special case; legend
  above the panels). The by-cloud-distance-stratum variant needs
  per-regime fold metrics (not in the local agg reports);
- **Fig. C6:** feature-ablation changes relative to the full model —
  NO FIGURE, final (user decision 2026-07-23d): the ablation is
  discussed with main-text Fig. 6b and tabulated in Table C7. A
  `figC6_cv_ablation` (paired per-fold ΔRMSE, variant − full) was
  briefly generated 2026-07-23c and REMOVED the same day; its numbers
  survive in Table C7's ΔRMSE column (−spec ≈ +0.01 ppm near-free,
  −contam ≈ +0.04, −xco2 ≈ +0.06, combined +0.10–0.13) and can be
  quoted in the Appendix C prose. Appendix C figure files: C1, C2, C4,
  C5, C7;
- **Fig. C7:** raw/BC/ML increment relationship — GENERATED 2026-07-23:
  `figC7_increment_attribution` (the 2026-07-22q "not local" blocker was
  wrong: the ML-on-raw per-footprint plot_data IS local under
  `de_prof_reg_mix_raw/combined_*`, and the figure reuses
  `make_raw_bc_ml_report._load_pairs` — same 5,912,308-footprint /
  51-date population as the report's §4, verified by count). Panel (a)
  2-D density of Δμ ≡ μ_raw − μ_B11 vs the operational increment
  inc ≡ X_raw − X_B11 with identity + OLS lines (slope +0.53, r +0.79,
  60 % of var(inc) explained); panel (b) r(Δμ, inc) vs r(μ_B11, inc) by
  surface × cloud-distance stratum (near-cloud land: rediscovery +0.86,
  production-overlap −0.34 — the complementarity claim in one panel);
- **Table C5:** date-blocked fold metrics (the frozen values quoted in
  §4.3) — GENERATED 2026-07-23 as
  `manuscript/tables/tabC5_cv_model_comparison.tex`, REVISED same day
  (user request): model label "Deep ensemble" → "DE", and the
  out-of-domain ridge artifact fixed by applying the PRODUCTION output
  guard (|μ| > 25 ppm → correction withheld — the guard the deployed
  chain applies to every model) to the ridge held-out predictions at
  evaluation. All cells are now clean fold mean ± std (guarded ridge:
  ocean 0.537 ± 0.023 / R² 0.464, land 0.673 ± 0.021 / R² 0.299; guard
  trips 18 of ~11.6 M ridge footprints, starred note in the table). A
  RETRAIN was considered and rejected: ridge is deterministic (seed is
  a no-op) and the blowup is test-time feature extrapolation, which
  training-side data guards cannot bound; a true retrain with feature
  clipping would be a CURC job touching the TCCON builders too. NOTE:
  the guard is NOT applied to DE/XGBoost in the CV tables — uniform
  guarding would trip 93/94 land footprints whose large corrections are
  mostly CORRECT and shift the frozen land fold means +0.03 (DE
  0.536→0.562, XGB 0.554→0.574, ordering unchanged) — flagged as an
  author decision if a reviewer asks for symmetric treatment;
- **Table C6:** fold-resolved baseline results — GENERATED 2026-07-23:
  `tabC6_fold_resolved_baselines.tex` (fold-by-fold RMSE/R² for
  DE/XGB/Ridge from the local per-fold artifacts; ridge under the same
  production output guard as Table C5, guarded folds starred with
  per-fold trigger counts — ocean f3: 1, land f2: 1, land f3: 16 — and
  the raw ~10⁶ ppm land-f2 extrapolation stated in the note);
- **Table C7:** complete ablation results, using the 2026-07-17 retrained
  variants only — GENERATED 2026-07-23: `tabC7_cv_ablation.tex` (held-out
  CV per variant × surface, fold mean ± std RMSE/R² + ΔRMSE vs full,
  computed from the local `*_prof_foldpca_*` fold dirs; values match the
  FEATURESET_ABLATION_QF_2026-07-17 per-fold table; the land-f4
  stale-checkpoint caveat carried as a table note). The TCCON-side
  ablation stays in main-text Table 2;
- **Table C8:** raw / bc / ML(bc) / ML(raw) metrics by slice
  (`tabC8_raw_bc_ml.tex` — former main-text Table 5, moved here
  2026-07-22m; file + generator renamed from `tab_raw_bc_ml` 2026-07-23;
  pairs with Fig. C7).

*Caption note (2026-07-23g) — caption text now lives in the .tex.*

The appendix should preserve the null result: `no_spec` is approximately
TCCON-neutral. Do not select only strata that make the spectral block appear
predictively essential.

### Appendix D: Independent validation — TCCON protocol and ocean

**As compiled 2026-07-26.** Section title corrected (it had carried the stale
Appendix-C title "Cross-validation, baselines, and feature attribution").
Contents: Table D1 TCCON stations · Table D2 comparison-date inventory ·
coincidence-criterion sensitivity (Fig. D1) · quality-flag-resolved comparison
(Fig. D2a/b) · ocean inventories (Tables D3 ATom legs, D4 ship cases) ·
DerSimonian–Laird estimator + forest plot · Table D5 training dates.
**REMOVED:** the AK/prior-harmonization operator subsection (standard
Rodgers–Connor/Wunch operator → citation in Methods §3.5, which now also
states the GGG2020 wet→dry prior conversion explicitly, since that step is not
part of the cited procedure and moves the reference by ≈1 ppm); Fig. D3 (r50)
and Fig. D4 (station summary), both unreferenced; a duplicate copy of the
QF0/QF1 figure. Generated-file names re-synced to printed numbers
(`tabD3_atom_legs`, `tabD4_ship_cases`). Dissertation-only, in
`manuscript/backup/`: `tabD2_stationday_metrics`, `tabD3_significance_tests`,
`tabD4_uncertainty_components`. The plan text below predates the trim.
references

(MERGED 2026-07-22n: former Appendix E + former Appendix H. TCCON
protocol/robustness first, ocean references as the closing subsection;
the do-not-pool rule that used to separate the two appendices is now a
statement inside this one.)

**Purpose:** make the independent validation defensible to AMT reviewers.

Include:

- GGG2020 QC statement and station-coordinate provenance;
- AK/prior harmonization equations and wet-to-dry prior conversion;
- the AK-harmonized reference definition, plus a short note documenting the
  direct (non-harmonized) comparison and the B7-to-B11 anchoring offset that
  explains the approximately 0.3 ppm scale difference — no full direct
  metric table (AK-only decision 2026-07-21);
- coincidence radii and time windows;
- station-day aggregation and statistical tests;
- site-clustered bootstrap implementation;
- uncertainty budget and random-effects model;
- complete station-level and station-day results.

Planned items:

- **Fig. D1:** 3-by-3 coincidence sensitivity — GENERATED 2026-07-23p:
  `figD1_coincidence_sensitivity` (before/after dumbbells of station-day
  mean |bias| and fp-RMSE per combo, n annotated;
  `make_appendix_def_figures.py`) from the 3×3 sweep RERUN on the
  fold-PCA production tag (9 local report runs, exact launcher flags,
  into `<TAG>/atrain/coincidence_sensitivity/`; r100/±60 row reproduces
  the production headline 1.26→0.82 / 2.67→1.20 exactly);
- **Fig. D2:** QF0 and QF1 station-day comparisons — staged 2026-07-22o:
  `figD2a_tccon_qf0` + `figD2b_tccon_qf1` (copies of
  `<TAG>/atrain/tccon_ak_bias_qf{0,1}_r100km.png`);
- **Fig. D3:** r50 robustness comparison — staged 2026-07-22o:
  `figD3_tccon_r50` (copy of `<TAG>/atrain/tccon_ak_bias_r50km.png`);
- **Fig. D4:** compact station-grouped summary ONLY (all 75 individual
  station-day before/after panels move to Supplement S1, 2026-07-22m;
  drop D4 entirely if the summary duplicates Fig. 7) — candidate
  staged 2026-07-22o: `figD4_station_summary` (copy of
  `<TAG>/atrain/tccon_ak_by_site_bias_r100km.png`);
- **Fig. D5:** random-effects residual forest plot — GENERATED
  2026-07-23p: `figD5_uncertainty_forest` (75 station-day D ± 95% CI
  from `tccon_uncertainty_r100km.csv`, DL pool recomputed in-script and
  cross-checked EXACTLY against the md: μ = −0.32 ± 0.09 ppm,
  τ = 0.52 ppm, I² = 51%; significant cases in vermillion);
- **Table D1:** station inventory and coincidence counts;
- **Table D2:** complete AK-harmonized metrics;
- **Table D3:** paired Wilcoxon and site-clustered bootstrap results;
- **Table D4:** uncertainty components, \(\tau\), \(I^2\), and coverage.

*Caption note (2026-07-23g; D1/D5 PROVISIONAL — artifacts pending) — caption text now lives in the .tex.*

The main paper should carry the headline and one robustness summary. The full
coincidence matrix and station-level audit belong here.

**Second job (former Appendix H, merged 2026-07-22n) — ocean-reference
details:** document the limited but independent ocean validation without
overloading the TCCON part of this appendix.

Include:

- ATom pseudo-column construction, vertical coverage, prior fill, and AK
  treatment;
- shipborne EM27/SUN processing, scale vintage, and coincidence choices;
- per-date and per-leg results;
- absolute-scale limitations and why the far-cloud cases are interpreted as
  negative controls.

Planned items:

- **Fig. D6:** ATom summary, sourced from working figure A8 — include only
  if it adds beyond main-text Fig. 10a, else the ocean subsection is
  methods + tables only;
- ~~Figs. H2–Hn, ship individual cases~~: all per-date ATom panels and
  individual ship cases move to Supplement S2 (2026-07-22m; the ocean
  subsection shrinks to ~2 typeset pages — the construction methods and the two inventory
  tables are what §4.6's quoted numbers lean on);
- **Table D5:** ATom leg inventory and residuals;
- **Table D6:** ship case inventory and residuals.

*Caption note (2026-07-23g; CONDITIONAL — only if D6 is kept beyond main-text Fig. 10a) — caption text now lives in the .tex.*

Do not pool these observations with TCCON into a single headline metric.

### Appendix E: Null tests, negative controls, and failure cases

**As compiled 2026-07-26.** Figs. E1–E5 unchanged; Tables E2 (driver strata),
E3 (CV albedo cross-check), E4 (smoother null) retained. **Table E1** (per-case
worsening listing) moved to `manuscript/backup/` — its load-bearing number,
"the per-footprint RMSE still improves in 25 of those 29", is now stated in
§4.3.1. Table E2 was considered for the same cut and KEPT: §4.3.4 quotes it
directly (bright-surface RMSE 4.58 → 1.55 ppm, worsened fraction 0.50 vs
0.19–0.39, σ-decile 0.46 / 0.06), and Fig. E4 shows the RMSE curves but not the
worsened-fraction or ⟨z²⟩ columns, so removing it would leave main-text
numbers with no auditable source.
(formerly Appendix F)

**Purpose:** demonstrate that error reduction is not produced by generic
smoothing and expose conditions where the method does not work.

Include:

- feature-free smoother definitions and all tested windows;
- far-cloud ATom and clear-day ship controls;
- bright-surface, high-latitude, snow, and high-AOD stratification;
- all worsening station-days with AK-harmonized interpretation;
- guard activity and predictive-uncertainty diagnostics.

Planned items:

- **Fig. E1:** smoother null, windows not in main-text Fig. 10 —
  GENERATED 2026-07-23p: `figE1_smoother_windows` (2×2: ±10 s top /
  ±100 s bottom rows, same scatter-collapse + |case bias| construction
  as Fig. 10 which shows the ±30 s arm; |bias| 1.26→1.24 / 1.20 vs DE
  0.82 — brackets the main-text numbers);
- **Fig. E2:** ATom far-cloud control (2017-10-09) — SPLIT from the old
  two-case E2 (user decision 2026-07-23s) and REGENERATED with the
  correct MODIS background: the case overpasses lon −175° at 01:32 UTC =
  LOCAL day 2017-10-08, and GIBS daily mosaics are keyed by local day
  near the antimeridian, so the old figure showed the fully-overcast
  scene 24 h later; `plot_atom_comparison.py` (and
  `atom_modis_overlay.py`) now fetch the local-solar date of the
  collocated footprints. New tile (2017-10-08) matches the
  cloud-distance field (clear band, broken cloud N+S, median 18 km);
  comparison numbers unchanged (n=313, +0.29→+0.28 ppm). Staged
  `figE2_atom_farcloud_control` (old figE2a deleted);
- **Fig. E3:** shipborne clear-day control (2019-06-22) — the other
  half of the old E2; file renamed `figE3_ship_clearday_control`
  (content unchanged; lon 152° E, no dateline issue);
- **Fig. E4:** failure rates and residuals by environmental driver —
  staged 2026-07-22o (renamed figE3→figE4 2026-07-23s):
  `figE4_failure_modes` (copy of
  `<TAG>/failure_modes/fig_failure_modes_r100km.png`);
- **Fig. E5:** high-latitude and post-2022 cases — STAGED 2026-07-23p
  (renamed figE4→figE5 2026-07-23s):
  `figE5_drift_tccon` (copy of the drift-tree
  `tccon_ak_bias_dumbbell_label_r100km.png`, RERUN with the current
  report styling — product-label stat box + inside legend; 21 drift-era
  station-days, |bias| 1.24→0.67 ppm, fp-RMSE 2.41→1.01; per-case drift
  comparison CSV verified byte-equal, the significance CSV is the
  current-code deterministic regeneration — no manuscript numbers quote
  it. Annotations moved OUTSIDE the frame 2026-07-23t (new report flag
  `--dumbbell-annotations outside`: stat box above-left, one-row legend
  above-right — the drift dumbbell has data in both inside corners;
  fig08's atrain tree keeps the default inside placement, untouched);
- **Table E1:** all worsening cases and diagnosed cause;
- **Table E2:** performance by bright surface, snow, AOD, latitude, and
  predictive-uncertainty strata;
- **Table E3 (added 2026-07-23r):** held-out CV albedo cross-check — the
  bright-surface TCCON failure signature does NOT reproduce against the
  held-out anomaly label (after-RMSE flat 0.52–0.60 ppm, frac-worse
  0.39–0.41 across all alb_o2a deciles, ⟨z²⟩ ≈ 1.0–1.2, vs TCCON
  bright-bin after-RMSE 1.55 / frac-worse 0.50), establishing the
  bright stratum as UNDER-corrected common-mode bias, not
  mis-correction; source
  `failure_modes/strat_cv_land_alb_o2a_r100km.csv` (stage 6 of
  `analyze_failure_modes.py`, report `FAILURE_MODES_2026-07-23.md`).

*Caption note (2026-07-23r) — caption text now lives in the .tex.*

*Caption note (2026-07-23g; renumbered E2→E2/E3 split, old E3/E4 → E4/E5, 2026-07-23s; all five figures now staged — E1 shows the windows not in main-text Fig. 10) — caption text now lives in the .tex.*

If the smoother null remains a main figure, retain its full numerical table in
this appendix and avoid duplicating the same plot.

### Appendix F: Case studies and plume-preservation audit
(formerly Appendix G)

**Purpose:** allow visual inspection of cloud-mask behavior, spectral response,
and preservation of localized CO2 enhancements.

Include:

- vetted real-cloud land and ocean cases;
- MYD35 false positives;
- visible clouds without a strong XCO2 response;
- all Nassar plume cases, matched controls, and channel-attribution results;
- selection criteria for the showcased Westar case.

Planned items:

- **Fig. F1:** Tasman Sea 2018-05-01 single-case showcase (moved from
  main-text Fig. 5b, 2026-07-22c) — exists:
  `manuscript/figures/figF1_case_tasman` (renamed from figG_*
  2026-07-22o) (2026-07-11 Times-italic l′ —
  consistent with the final convention, no re-render needed);
- ~~Figs. G2–G8~~: the seven case-atlas pages move to Supplement S3
  (2026-07-22m; resolves the previous conflict with the §4 Supplement
  list; renumbering resolved 2026-07-22n: transects/Δl′ are Figs. F2–F3);
- **Fig. F2:** Nassar transects — RE-RENDERED AND REDUCED TO THE SIX
  AUDITED WINDOWS 2026-07-28 (user, two decisions: the 10-case × 3-panel
  mosaic was unreadable, and the three F2a groups carry the story alone):
  `figF2a_nassar_audited` (plume / flagged / clear-sky columns, one XCO2 +
  nearest-cloud-distance panel pair per case) is the ONLY Nassar transect
  figure in the paper. Rendered NATIVELY from the per-case plot_data by
  `make_appendix_def_figures.py` (via the new
  `nassar_plume_transects.load_transect_segment`), not by tiling PNGs —
  fonts stay print-size. The four far-approach cases (14–38 km) render
  author-side as `internal_nassar_null_transects`; the briefly-generated
  `figF2b_nassar_null` and the old `figF2_nassar_transects.png` mosaic are
  DELETED; the full 3-panel per-case dossiers remain under
  `nassar_plumes/plume_preservation/transects/`. Completeness now lives in
  the TABLE F2 NOTE (UPDATED later 2026-07-28, user decision on the
  "§4.4 hard-coded pointers" review item: the testable set is now DEFINED
  by a 12 km closest-approach criterion, so the six shown windows ARE the
  testable set and the main-text count is 4/6; Kozienice 2024-06-26
  (14.3 km) joins Colstrip/Comanche/Vindhyachal (30–38 km) in the
  not-testable list; Matimba no segment. Caveat accepted by user: the
  only cut that yields exactly six sits between Westar-06 11.9 km and
  Kozienice-24 14.3 km). Tex label `app-fig:nassar_transects`
  unchanged. The two §4.4 companion sentences were APPLIED with user
  approval 2026-07-28: the inventory pointer now reads "per-case results
  and accounting in Appendix F, Table F2" (it had pointed at Table F1, the
  case-atlas inventory), and the not-a-selection sentence now reads (after the six-only
  redefinition, later 2026-07-28) "The six testable windows are shown
  ... and the cataloged overpasses without a testable window are
  accounted for in the Table F2 note"; all §4.4 Appendix-F pointers are
  now \ref-based (review item done);
- **Fig. F3:** band-resolved \(\Delta l'\) for plume and cloud-contaminated
  windows — GENERATED 2026-07-23p: `figF3_k1_contrast_full` (all SIX
  audited windows in three groups: plume [Kozienice 2021-09-06, Taean
  2021-09-08] / flagged [Lipetsk, Westar 2023-03-13] / clear-sky
  controls [Westar 2023-06-26, Ghent]; main-text Fig. 12b keeps the
  4-window subset; fold-PCA-tag `nassar_k1_contrast.csv`);
- **Table F1:** vetted case inventory and category assignments;
- **Table F2:** per-case plume-removal bounds and control-null results —
  RESTRUCTURED 2026-07-28: regular table float (was longtable — the 4-in
  \LTcapwidth caption and full-width note looked broken next to the narrow
  table; the longtable helper now sets \LTcapwidth=\linewidth for the
  tables that stay long), SIX rows matching Fig. F2a's panels (letters
  (a)–(f) in the first column, middlehline between groups); the note
  carries the non-testable-case accounting (incl. Kozienice 2024-06-26
  since the 12 km closest-approach criterion, later 2026-07-28);
- **Table F3:** full/no-spec/no-xco2 smoothing attribution — this IS the
  former main-text Table 4 (`tab_nassar_attribution.tex`, moved here
  2026-07-22m); one table, not two.

*Caption note (2026-07-23g; F2/F3 PROVISIONAL — artifacts pending) — caption text now lives in the .tex.*

The main text shows only the Westar preservation case (the Tasman
mechanism showcase moved here as Fig. F1, 2026-07-22c). The appendix must show the full set to avoid
case-selection concerns.

### Appendix G: Controlled 3-D radiative-transfer demonstration
(formerly Appendix J)

**Purpose:** provide a causal mechanism test while keeping its limited scope
explicit.

**PLACEMENT RESOLVED (2026-07-24b): typeset appendix.** The former
conditional (2026-07-22m) is met and exceeded — the simulation is now
SELF-RUN (er3t/MCARaTS v0.10.4 forward MC, `workspace/rt_slab_sim/`,
replacing the cohort backward-MC figure and its provenance/co-authorship
open items), finished to manuscript standard, and includes a quantitative
first-moment closure between the fitted cumulants and directly tallied
photon-path moments — upgrading the appendix from qualitative sensitivity
demonstration to partial PPDF closure (§8d language updated accordingly).

Contents (all implemented; Table G1 auto-generated from the simulation
config so it cannot drift from the code):

- scene: one real OCO-2 sounding (2020010100281632, orbit 29252a,
  1 Jan 2020, 29.9° N ocean glint, SZA 55°) supplies the Met/CO2-prior
  profiles, regridded to 21 layers conserving every gas column exactly
  (above-top O2 residual 0.015 %);
- x–z slab: 32-km periodic domain (Nx 64 × 0.5 km, Ny = 1, medium
  invariant along y — photons keep full 3-D angular freedom), water cloud
  COD 10 / r_eff 10 µm at 3–4 km, x = 9.5–14.5 km; sun along +x; nadir
  sensor; dark (0.03) and bright (0.30) Lambertian surfaces;
- gas optics: ABSCO v5.2 per-layer OD at 33 monochromatic O2A wavelengths
  log-spanning slant τ ≈ 0.03–8 (+3 continuum anchors);
- identical scene under full 3-D transport and the independent-pixel
  approximation (IPA) — a single solver switch, 1e9 photons × 3 runs each;
- refit of every column with the PRODUCTION estimator (order 7, no-SG,
  exact lstsq/BVLS — the same code path as the observations);
- native MCARaTS path-length tally (Rad_mplen=3): per-column,
  radiance-weighted histograms of total geometric photon path = the PPDF,
  reduced to mean/s.d. per column (closure row of Fig. G1);
- results: one-sided shadow-band response (x ≈ 15–22 km) + illuminated-edge
  brightening under 3-D only; IPA exactly flat outside the cloud
  (residual ≤ 0.9 % of the 3-D dynamic range in ⟨l′⟩; the dark-surface
  var(l′) null is 4.6 % — manuscript claims scoped per-feature
  2026-07-29, "within 0.9 % in ⟨l′⟩ and within 5 % in var(l′)");
  the shadow-band ⟨l′⟩
  response REVERSES SIGN between the dark (0.78 → 0.41) and bright
  (0.99 → 1.17) surface — the albedo-contrast mechanism (EMPHASIS 1)
  reproduced causally; 3-D-only var(l′) plateau (~0.45) across the bright
  shadow band;
- closure: r(fitted ⟨l′⟩, tallied mean path) = 0.999 dark / 0.987 bright
  (clear columns); r(var(l′), tallied path variance) = 0.989 bright.
  The dark-surface second moment does NOT close against the geometric
  tally (short-Rayleigh-path dominance; the fit senses absorption-weighted
  moments) — state as the expected limit of a geometric tally;
- limitations: single representative geometry; geometric-vs-
  absorption-weighted path distinction; monochromatic sampling (no ILS).

Display items (EXIST, generators in `workspace/rt_slab_sim/`):

- **Fig. G1:** `manuscript/figures/figG1_mc_3d_vs_ica.{png,pdf}`
  (`make_fig_g1.py`; closure numbers in
  `results/rt_slab_sim/closure_stats.json`);
- **Table G1:** `manuscript/tables/tabG1_slab_config.tex`
  (`make_table_g1.py`);
- optional S6 companions: PPDF heat-map/cut figures
  (`results/rt_slab_sim/figs/slab_ppdf_{dark,bright}.png`,
  `plot_ppdf.py`) and the slab atmosphere profile figure
  (`slab_atm_profiles.png`, `plot_atm_profiles.py`).


State explicitly that the first moment closes quantitatively in shape and
the second moment closes where the cloud-detour population dominates
(bright surface); exact second-moment closure requires absorption-weighted
path tallies. STILL FUTURE WORK (follow-up per §4 of TODO_ACCOMPLISH):
absorption-weighted tallies, the geometry sweep (SZA/COD/albedo), synthetic
full-spectrum generation, and plume injection.

### Appendix H: Cross-sensor feasibility (TEMPO; formerly Appendix K)

**As compiled 2026-07-26: KEPT and SHORTENED** (user decision — the TEMPO
application stays, with less detail). 742 → 464 words: three subsections
collapsed to flat prose, one figure, no table. Retained: what the proof
establishes, the scene, the ⟨l′⟩ 0.91 → 0.76 decay, and both disclaimers
(TEMPO retrieves no XCO2; the trained correction is not claimed to transfer).
Cut: the instrument-specification paragraph, the two CLDO4 caveats (now one
clause), and the "scope and implications" subsection. Discussion §5.3 cites it
as the feasibility anchor. Long version:
`manuscript/backup/pre_trim_2026-07-26/appendix_H_long.tex`.

**Purpose:** bound the transfer claim with one existence proof, not imply a
validated cross-sensor correction.

**KEEP (user decision 2026-07-22):** retained against the length-trim
recommendation — the TEMPO demonstration is the paper's bridge to other
high-spectral-resolution missions, so the transferability claim of
Discussion 5.3 has a concrete anchor. Consequence: hold it to the
existence-proof scope below (one granule, one figure, one table) and make
Discussion 5.3 cite it explicitly as the feasibility anchor for the
prospect list (OCO-3, CO2M, GOSAT-GW, TEMPO).

Include the planned TEMPO O2-B example currently tracked as working figure
A12:

- one pre-specified TEMPO granule;
- O2-B spectral window and optical-depth construction;
- in-scene cloud product and cloud-distance calculation;
- maps of \(l'\) and path-length variance;
- response versus in-scene cloud distance;
- differences from the OCO-2 instrument, geometry, sampling, and retrieval.

Planned items:

- **Fig. H1:** TEMPO RGB/cloud field, fitted cumulants, and distance response
  (LANDED 2026-07-24 — caption block below);
- ~~**Table H1:** OCO-2 and TEMPO inputs required by the spectral fit~~
  **DROPPED (user decision 2026-07-24c — paper length):** the differences
  fold into the Appendix H prose; the side-by-side survives author-side
  only as `manuscript/tables/internal_tempo_fit_inputs.tex`
  (`make_appendix_tables.py --only internal_tempo_inputs`).

*Caption note (updated 2026-07-24 — artifact LANDED as `manuscript/figures/figH1_tempo_o2b_demo.{png,pdf}`, produced by `~/programming/tempo/scripts/make_h1_figure.py`. FINAL scene (author decision 2026-07-24, after a three-scene comparison): **S007G09_160926_o2b_3, Mexico Pacific east, ocean, GOES-West, 2024-07-08 ≈16:09 UTC** — cleanest distance decay (median ⟨l′⟩ 0.91→0.76 over 0–25 km then plateau, N=5005, all 2.5-km bins populated to 50 km, 100% fit success; 0 pixels used the ocean poly_order branch, so no branch caveat needed). Kansas S010G06_160926_o2b_1 kept as rendered all-land alternate in the tempo repo. Corrections vs the 2026-07-23g draft, per the tempo TODO §5 flag: (1) "production estimator" → reference-implementation wording — the tempo repo runs the pre-rewrite curve_fit+SG engine; (2) the var(l′) map was briefly dropped (drop-if-it-crowds rule) then REINSTATED same day by author request in a 2×3 layout, labels column-major — (a) GOES RGB / (b) CLDO4 cloud fraction; (c) ⟨l′⟩ map / (d) var(l′) map; (e) ⟨l′⟩ vs distance / (f) var(l′) vs distance with per-bin sample sizes; rows pair each cumulant's map with its distance curve) — caption text now lives in the .tex.*

Label this appendix **feasibility demonstration**. Do not include EMIT merely
to broaden the sensor list; its sampling may not provide adequate optical-depth
dynamic range. Do not claim transfer of the trained OCO-2 correction.

### Former Appendix I (unlettered): moved wholly to Supplement S4 (2026-07-22m)

The spectrum-internal sensitivity material (spec-only classifier ROC,
sub-pixel response, NoMODIS-era diagnostics) supports only the
"sensitivity" tier of Discussion 5.3 — no Results claim leans on it, and
its own disclaimer states the AUC values do not establish that the
spectral block is required for correction skill. It therefore moves to the
Supplement as a complete section (contents listed under S4 below);
Discussion 5.3 cites it in bulk: "spec-only classifiers recover near-cloud
state from single spectra (AUC 0.72 land / 0.66 ocean; Supplement
Sect. S4)". Letters were reassigned 2026-07-22n (mapping table at the top of §5).

**UPDATE 2026-07-27 (user decision):** the §5.3 bulk citation above was
DROPPED — the main text now cites the Supplement NOWHERE, so the paper can
be submitted without a Supplement. S1–S6 remain author-side (S1/S2/S4
staged locally; S3/S5 never regenerated); if a reviewer asks for the
per-case galleries, a data-repository archive (e.g. Zenodo) referenced
from the data-availability statement is the alternative to a formal
Supplement.

**UPDATE 2026-07-28 (user decision):** the Supplement is now BACKUP-ONLY —
the paper IS submitted without one; S1–S6 stay staged author-side and are
produced/offered only if reviewers ask for more detail (formal Supplement
or Zenodo archive at revision, whichever fits the request). The last
paper-facing citations were removed the same day: tabF1 caption + note no
longer say "Supplement Sect.~S3" (fixed in `make_appendix_tables.py` and
regenerated), and the applied F.1 opening paragraph omits the draft's
"atlas pages are in the Supplement" clause. The S1–S6 plan below is kept
as the activation spec.

### Appendix triage

Resolved 2026-07-22m/n into the appendix/Supplement split above
(letters below are the FINAL 2026-07-22n letters):

1. **Required (typeset):** A–F — method, model+CV, validation (TCCON +
   ocean), nulls/failures, and case/plume audit — plus H (user decision
   2026-07-22: the cross-mission bridge, existence-proof scope).
2. ~~**Conditional:** G (3-D RT)~~ **RESOLVED 2026-07-24b: typeset**
   (self-run simulation finished to manuscript standard; Fig. G1 +
   Table G1 exist; S6 keeps only the optional PPDF companions).
3. **Moved to Supplement:** the former Appendix I (whole section → S4)
   and every per-case gallery (S1–S3).

Never include a placeholder or partially documented simulation. If working
figures A12 or A13 are not completed to manuscript standard, retain their
claims as future work and remove the corresponding appendix references.

### Supplement plan (S1–S6, 2026-07-22m; BACKUP-ONLY since 2026-07-28)

**STATUS 2026-07-28: not submitted with the paper — activate only on
reviewer request (see the UPDATE above).** If activated: published as one
author-formatted PDF with its own DOI. Rules (per the
§5 ADMISSION RULE): unrestricted in logical scope but NO new central
scientific conclusions; cited from the paper in BULK only ("see
Supplement Sect. Sx"), never as the audit trail for a specific main-text
number — everything quantitative the paper leans on stays in the main
text or a lettered appendix. If provided at revision, it is not
copy-edited, so it must be self-contained and carry its own caption
discipline (the §4 caption rule applies).

STAGING (2026-07-22p): `manuscript/scripts/stage_supplement_figures.py`
copies the S1/S2/S4 assets (91 files, ~295 MB) into
`manuscript/supplement/S*_*/` — git-ignored except the tracked
`supplement_manifest.txt`; `manuscript/tex/supplement.tex` assembles the
final PDF from these directories. S3 (atlas pages) and S5 (SG sweep +
fit-example gallery) have no local sources — pull/regenerate on CURC,
then rerun the staging script.

- **S1 — TCCON station-day gallery:** all 75 individual station-day
  before/after panels (working label E4; the typeset remnant is the
  optional compact summary Fig. D4).
- **S2 — Ocean case pages:** per-date ATom panels and individual ship
  cases (working labels H2–Hn).
- **S3 — Case-study atlases:** the seven category atlas pages (working
  labels G2–G8): real-cloud land/ocean, MYD35 false positives, visible clouds
  without XCO2 response, etc.
- **S4 — Spectrum-internal sensitivity (former Appendix I, complete):**
  spec-only classifier definition + held-out ROC, AUC/calibration table by
  surface and fold, sub-pixel spectral-index response, NoMODIS-era
  application examples; keep the sensitivity-vs-skill disclaimer.
- **S5 — Extended fit robustness:** full SG-versus-no-SG sweep and the
  accepted/rejected fit-example gallery (Appendix A keeps one summary
  panel).
- **S6 — Reserve:** secondary uncertainty/failure-mode tables that
  outgrow Appendix D/E, and the OPTIONAL Appendix G companions (the
  PPDF heat-map/cut figures and slab atmosphere profiles; the appendix
  itself is typeset — resolved 2026-07-24b).

*Caption note (2026-07-23g; page-template style — one caption per section, repeated per page with the page's case identifier filled in; the §4 caption rule applies inside the Supplement too) — caption text now lives in the .tex.*

## 6. Terminology and claim controls

Use the following distinctions consistently:

| Prefer | Avoid | Reason |
|---|---|---|
| imager-independent at inference | imager-independent method | MODIS remains part of diagnosis and target construction. |
| spectrum-fitted path-length cumulants/proxies | directly measured photon-path moments | Direct fitted-versus-tallied PPDF closure is not complete. |
| candidates for observation recovery | recovered/usable observations | Downstream acceptance and inversion benefit are not yet demonstrated. |
| WCO2 land-cover sign rule | forest sign flip or albedo ordering | The latter claims were not supported by the full-parquet analysis. |
| complementary residual correction | replacement for B11 | The best result applies ML after the operational correction. |
| independent corroboration | comprehensive ocean validation | ATom and shipborne samples are valuable but limited. |
| deep ensemble (DE) | DE-MLP | Avoids collision with the standalone "MLP" point-predictor baseline (now internal-only split material); matches figure labels and the \(X_{\mathrm{CO2}}^{\mathrm{DE}}\) superscript; the members-are-MLPs detail lives in §3.3. Update the tex sources that currently say DE-MLP. |

Keep XCO2 typesetting and notation consistent with the AMT figure style:
\(X_{\mathrm{CO_2}}\), \(l'\), \(k_1\), and \(k_2\).

**Product-resolved XCO2 notation (user convention 2026-07-23; constants
in `plot_style.py`, macros in `supplement.tex` — never retype):**

| Quantity | Symbol | plot_style constant |
|---|---|---|
| xco2_bc (operational B11 BC) | \(X_{\mathrm{CO2}}^{\mathrm{B11}}\) | `XCO2_BC_LABEL` |
| xco2_raw | \(X_{\mathrm{CO2}}^{\mathrm{raw}}\) | `XCO2_RAW_LABEL` |
| ATom pseudo-column | \(X_{\mathrm{CO2}}^{\mathrm{ATom}}\) | `XCO2_ATOM_LABEL` |
| shipborne EM27/SUN | \(X_{\mathrm{CO2}}^{\mathrm{ship}}\) | `XCO2_SHIP_LABEL` |
| anomaly target (xco2_bc_anomaly) | \(\Delta X_{\mathrm{CO2}}^{\mathrm{B11}}\) | `DXCO2_BC_LABEL` |
| deep-ensemble-corrected product | \(X_{\mathrm{CO2}}^{\mathrm{DE}}\) (define "DE" at first use; chosen over the too-long "DE-corrected" superscript, 2026-07-23) | `XCO2_DE_LABEL` |

Applied 2026-07-23 to fig03/fig03alt/figB2/fig05 (+ ocean companion).
The copied report figures that display XCO2 axes (fig07, fig09,
figD2–D4, figE2) still carry generic labels (fig10a/b adopted the
superscripts at their 2026-07-23 re-render) — adopt the
superscripts when their producers are next re-rendered (ATom/ship
producers live in workspace/ATom_analysis + Ship_analysis).

## 7. Values that must be frozen before prose drafting

Resolve these items from generated artifacts rather than copying historical
summary text:

1. **Analysis cohorts:** distinguish the 116-date/17.8-million-sounding
   manuscript cohort from the 140-date/21.5-million-sounding label-ceiling
   cohort and any 2014–2021 processing inventory.
2. **Production tag:** use only the frozen fold-PCA/lndo01 production tag and
   its regenerated reports.
3. **TCCON values:** regenerate the final table containing AK-harmonized
   results only (direct-reference columns dropped 2026-07-21), qf0/qf1
   strata, and r50/r100 sensitivity.
4. **Cross-validation:** freeze the exact land/ocean date-blocked metrics and
   their fold dispersion.
5. **Uncertainty:** freeze the post-fold-PCA random-effects offset, \(\tau\),
   \(I^2\), and coverage values.
6. **QF1 recovery:** define an acceptance criterion before computing a
   recovery fraction.
7. **Notation:** regenerate remaining figures that still use the retired
   capital \(L'\) (2026-07-22 final form: lowercase Times-italic \(l'\) —
   the 2026-07-11 Times-italic figures are already consistent; fig04 and
   the k1-contrast figure (now fig11b) regenerated).
8. **Near-cloud coverage statistics:** re-freeze the Results 4.1 sentence on
   the final analysis cohort. 2026-07-21 values from
   `combined_2016_2020_dates.parquet` (17.75 M valid-cloud-distance
   footprints, 99.9 % of 17.77 M rows / 116 dates): all footprints 41.7 %
   < 4 km and 60.7 % < 10 km; ocean 59.1 % < 5 km; land 46.8 % < 15 km;
   median nearest-cloud distance 3.5 km (ocean) / 17.9 km (land).

## 8. Recommended writing order

Draft in evidence order rather than manuscript order:

1. Methods 3.1–3.6, freezing definitions and evaluation units.
2. Results 4.1–4.4 directly from final tables and figures (draft the
   Appendix C cross-validation material alongside §4.2, since its conclusions
   feed the two-sentence headline there).
3. Data 2.1–2.4, reconciling cohort provenance and product versions.
4. Discussion 5.1–5.6, constrained to what Results demonstrate.
5. ~~Introduction 1.1–1.3, positioning the now-fixed contribution~~ —
   DRAFTED 2026-07-29 (see the §1 status note above).
6. Conclusions, then Abstract and title.

This order minimizes narrative drift and prevents the Introduction or Abstract
from promising stronger claims than the final evidence supports.

## 9. Final narrative check

Before submission, a reader should be able to answer these questions in order:

1. What cloud-proximity error is observed before any model is applied?
2. What spectral evidence supports a 3-D radiative mechanism?
3. Exactly what information is and is not available to the correction at
   inference?
4. Does performance survive date blocking (headline sentence in §4.2, full
   analysis in Appendix C) and independent validation?
5. Is the gain bias correction rather than smoothing?
6. Are real plume enhancements preserved in the tested cases?
7. Where does the correction remain unreliable?
8. What part of the workflow is genuinely transferable without an imager?

If each answer follows directly from one main figure or table, the manuscript
has the intended AMT flow: **observation → mechanism → method → independent
validation → trust boundary → application scope**.

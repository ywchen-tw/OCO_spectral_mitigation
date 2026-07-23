# Manuscript Flow Plan — AMT

**Created:** 2026-07-19  
**Updated:** 2026-07-21 — cross-validated correction performance moved from the
main Results (former §4.3) into Appendix D; Results renumbered 4.3–4.8. The
main text now carries only a two-sentence date-blocked headline inside the
model-comparison section, so the reader reaches the TCCON payoff without a
detour through split-design diagnostics.  
**Updated:** 2026-07-21b — AK-harmonized TCCON is the sole reported
reference; all direct-comparison tables/columns dropped (generator
`manuscript/scripts/make_manuscript_tables.py` regenerated AK-only). Direct
survives only as a one-sentence anchoring-chain note (§4.4, Appendix E).
Planned figures and tables now listed under each Results/Discussion section
(**Display items** blocks) with their generator scripts where they exist.  
**Updated:** 2026-07-21c — former composite Fig. 1 split: collocation
schematic → Methods 3.1 (Fig. 1), anomaly–distance decay curves → Results
4.2 (Fig. 2); all main figures renumbered (budget now nine + optional
tenth). Results 4.1 becomes prose-led and carries the near-cloud coverage
statistics as a single sentence (computed 2026-07-21, §7 item 8).  
**Updated:** 2026-07-21d — deep-ensemble architecture schematic (no-cloud
variant) promoted from Appendix B (former Fig. B1) to Methods 3.3 as
main-text Fig. 2 (`manuscript/figures/fig02_deep_ensemble_architecture`);
downstream figures renumbered again (decay curves now Fig. 3; budget ten +
optional eleventh).  
**Updated:** 2026-07-21e — every planned main-text figure now has an
artifact in `manuscript/figures/` (Figs. 4, 5a copied from pre-restyle
sources and flagged for regeneration; the rest current); draft captions for
Figs. 1–11 added to §4.  
**Updated:** 2026-07-21f — draft captions moved from the central §4 list to
sit under each section's Display items block; LaTeX panel-assembly drafts
added for the multi-file figures (5, 7, 9, 10).  
**Updated:** 2026-07-21g — Fig. 3 moved from Results 4.2 to Results 4.1 and
rebuilt as a two-panel figure: (a) common r10 target at 1-km bins motivates
the surface-specific radii; (b) adopted r05/r15 production targets.  
**Updated:** 2026-07-21h — two candidate renderings of Fig. 3 generated for
a pending choice: compact curves (`fig03_anomaly_decay`) and per-bin box
plot (`fig03alt_anomaly_decay_boxplot`).  
**Updated:** 2026-07-21i — curve rendering upgraded from mean ± 2 SE to IQR
shading + dashed median + solid mean, so both renderings expose the
tail-driven land bias (mean ≫ median) vs the coherent ocean shift; caption
updated with the interpretation sentence.  
**Updated:** 2026-07-22 — Fig. 4 regenerated in the locked style by a new
local generator (`make_landclass_heatmap_figure.py`) with an ocean column
added (dark endpoint of the contrast axis; WCO2 Δ⟨l′⟩ +0.20σ, sign-rule
consistent); land columns reproduce the scored run exactly under the QF0 +
snow-free filter (load-bearing). Pre-restyle-copy flag on Fig. 4 cleared.  
**Updated:** 2026-07-22b — path-length symbol reverted L′ → l′, rendered as
UPRIGHT serif (the $\mathrm{l}'$ look; plot_style cal slot → upright Times,
labels unchanged in code). fig04 + fig10b regenerated; fig05a/fig05b already
carried lowercase l′. Tex sources use \ell — unify at writing time.  
**Updated:** 2026-07-22c — Tasman case study moved from main-text Fig. 5b to
Appendix G (Fig. G1; atlas pages shift to G2–G8, Nassar items to G9–G10);
Fig. 5 is single-panel (`fig05_shadow_brightening_land`). Effect-size
equation for Fig. 4 drafted in LaTeX under the Fig. 4 caption.  
**Updated:** 2026-07-22d — r05/r15 spectral reference sets found in the
parquet (r05_*/r15_* columns); per-surface Fig. 4 variant computed
(`--reference per-surface`). Sign rule strengthens (barren −1.29σ), urban
cell dissolves; reference-variant choice recorded as an OPEN DECISION under
the Fig. 4 display item.  
**Updated:** 2026-07-22e — per-surface reference (ocean r05 / land r15)
ADOPTED as the Fig. 4 primary; files swapped (`fig04_*` = per-surface,
`fig04_*_r10` = robustness variant), generator default flipped, caption and
evidence-chain numbers updated (savanna +0.48σ, barren −1.29σ, ocean
+0.09σ; urban not interpreted).  
**Updated:** 2026-07-22f — l′ rendering finalized as lowercase Times ITALIC
(reversing the brief upright-\mathrm trial; plot_style cal slot → Times
italic). fig04 (both variants) + fig10b regenerated; the 2026-07-11
Times-italic figures (Tasman case, atlases) are consistent again. Paper
LaTeX: plain math-italic $l'$.  
**Updated:** 2026-07-22g — Fig. 4 QF-sensitivity variants generated
(`--qf {0,1,all}` → `_qf1`/`_allqf` files): every land-class sign is stable
across QF populations EXCEPT barren (−1.29σ QF0 / +0.47σ QF1 / −0.23σ
all-QF; 67 % of barren soundings are QF1) — flagged desert scenes carry the
cloud-signature-positive perturbation of in-FOV contamination/aerosol, which
is exactly why the QF0 filter is load-bearing and the primary population.  
**Updated:** 2026-07-22h — caption slim-down pass: interpretation/results
sentences moved out of the Figs. 3, 4, 5, 6, 7, 8, 9, 10 captions into
"Draft results text" blocks under each section's narrative (captions now
describe only what is shown and how it was computed). The user-approved QF
paragraph inserted as §4.2 draft prose ("consistent with" phrasing), the
Fig. 4 caption carries only a one-line Appendix B pointer, and Appendix B
item B4 extended to cover the QF-variant robustness alongside the
reference variant.  
**Updated:** 2026-07-22i — common-10-km-reference variant removed from ALL
main-text discussion (user: distracting): the §4.2 sign-rule prose keeps
only "urban is thin and not interpreted (Appendix B)", and the effect-size
LaTeX block ends with a one-line Appendix B pointer instead of the r10
attenuation explanation. The r10 material lives ONLY in Appendix B item B4
(which retains the 10–15 km contamination explanation) and the one-line
Fig. 4 caption pointer.  
**Updated:** 2026-07-22j — common-r10 reference variant dropped from the
APPENDIX as well (user: QF0/QF1 sensitivity is enough): B4 is now
QF-robustness only, the Fig. 4 caption pointer reads "Quality-flag
sensitivity: Appendix B", the effect-size LaTeX block attaches its
Appendix B pointer to the QF filter sentence, and the urban prose drops
its appendix pointer. The `_r10` files stay in the repo as an internal
check; author-side caveats preserved in the 2026-07-22e decision block.  
**Updated:** 2026-07-22k — standing CAPTION RULE added under §4 "Figure
captions": all figure/table captions descriptive only, result numbers and
interpretation live in main/appendix discussion prose.  
**Updated:** 2026-07-22l — Appendix K (TEMPO) KEPT by user decision against
the length-trim recommendation: it is the bridge to other
high-spectral-resolution missions; scope stays existence-proof (one
granule, one figure, one table) and Discussion 5.3 should cite it as the
feasibility anchor. Other trim suggestions (drop Fig. 11; Tables 4–5 to
appendices; galleries to Supplement; trim App. I; merge H into E) remain
OPEN pending user decisions.  
**Updated:** 2026-07-22m — appendix/Supplement restructure ADOPTED
(user-approved): main text fixed at ten figures + THREE tables (Fig. 11
dropped; Table 4 → Appendix G merging with G3; Table 5 → Appendix D as
D4); typeset appendices = A–H + K with galleries removed (E4 station-day
panels → S1, H per-date pages → S2, G2–G8 atlases → S3, A extended fit
material → S5); former Appendix I moved WHOLLY to Supplement S4
(Discussion 5.3 cites it in bulk); J conditional (typeset if finished,
else S6); new "Supplement plan (S1–S6)" section added at the end of §5
with the bulk-citation-only rule; §4 Supplement list replaced by a pointer
resolving its four conflicts (C tables stay, J conditional, K stays,
compact coincidence matrix stays in E). H stays a slim appendix (NOT
merged into E): §4.6's quoted numbers lean on its protocol + inventory
tables.  
**Updated:** 2026-07-22n — appendices CONSOLIDATED to eight letters
(user-approved merge): C = former C+D (model + CV evaluation), D =
former E+H (TCCON + ocean validation), E←F, F←G, G←J (conditional),
H←K (TEMPO). Former I is unlettered (Supplement S4). All §5 items
renumbered to the new letters, all body cross-references updated, and a
mapping table added at the top of §5; changelog entries BEFORE this one
keep the former letters. This supersedes the 2026-07-22m statement that
H would not merge into E — the merge keeps the ocean material as a
separate closing subsection with the do-not-pool rule inside.  
**Updated:** 2026-07-22o — figure files re-synced to final letters
(figC3b_cv_design, figF1_case_tasman, internal_qf1_recovery_candidate;
cv_design generator basename updated) and available appendix figures
STAGED into manuscript/figures from the production tree: figD2a/b (QF0/
QF1 TCCON), figD3 (r50), figD4 (station summary, optional), figE2a/b
(far-cloud ATom + clear-day ship controls), figE3 (failure modes).
Still to produce: C1/C2 (diagrams), C3a+C4–C7 (need frozen fold
metrics), D1 (3×3 coincidence composite), D5 (forest plot), D6
(only if beyond Fig. 9a), E4 (high-lat/post-2022 composite), F2
(transect sheet), B2 (likely redundant with Fig. 3), G1 (MC sim), H1
(TEMPO).  
**Updated:** 2026-07-22p — Appendix B display items filled: figB2
GENERATED (target-radius sensitivity, ocean invariant / land truncated
at the reference radius) and the QF heatmaps renamed to their B4 slots
(figB4a_landclass_qf1 / figB4b_landclass_allqf; r10 files →
internal_landclass_r10*, duplicate r10 CSV deleted; generator writes
the final names directly). Supplement staging added:
stage_supplement_figures.py fills manuscript/supplement/ (S1 75
station-day panels, S2 12 ocean case pages, S4 spectrum-internal set;
git-ignored, manifest tracked); S3/S5 sources still on CURC.  
**Updated:** 2026-07-22q — Appendix C figures: figC1_dataflow,
figC2_fold_timeline (REAL fold manifests), figC3a_random_split_inflation,
figC5_cv_model_comparison GENERATED (`make_appendix_c_figures.py`); C4
PENDING on ceiling-column semantics (achieved ocean R² exceeds
r2max_ref_ret — resolve with the original ceiling analysis before
plotting), C6 covered by Fig. 6b + Table C7, C7 pending ML-on-raw
artifacts.  
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
   time are mean absolute bias 1.26 to 0.81 ppm, footprint RMSE 2.67 to
   1.19 ppm, and improvement in 71 of 75 station-days.
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

**Draft caption:**

> **Figure 1.** Collocation geometry for one OCO-2 glint granule (orbit
> 29265, 1 January 2020): Aqua-MODIS MYD35 cloud-mask pixels (Cloudy, blue;
> Uncertain, grey; both classes retained; |Δt| ≤ 10 min) and OCO-2 soundings
> coloured by nearest-cloud distance d (ECEF KD-tree search, capped at
> 50 km). The arrow connects one sounding to its nearest cloudy pixel
> (d = 13.9 km); inset: full granule track.

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

**Draft caption:**

> **Figure 2.** Per-surface probabilistic correction model. (a) One
> Gaussian-head MLP member (64→32 hidden units; layer normalization and
> dropout 0.1; μ, log σ² heads; β-NLL loss, β = 0.5). (b) M = 5 members per
> fold pooled as a Gaussian mixture, followed by split and Mondrian
> conformal calibration of the 90 % intervals. All inputs are per-sounding
> L2 Lite quantities: cloud distance enters only label construction and
> evaluation — never the model input — so the correction runs
> footprint-by-footprint with no imager and no along-track context at
> inference.

#### 3.4 Training, validation splits, and leakage control

Describe contiguous date-fold cross-validation, train-only scaling and PCA,
the calibration partition, manifest-based TCCON-date exclusion, and the
label-noise ceiling. Explain the random split only as a comparator; the
numerical split-inflation result belongs in Appendix C (the main text carries
only the two-sentence headline in Results 4.3).

#### 3.5 Evaluation framework

Predefine:

- date-blocked target \(R^2\) and RMSE;
- station-day bias and absolute bias;
- footprint RMSE and scatter;
- calibration and interval coverage;
- paired Wilcoxon tests;
- site-clustered bootstrap;
- random-effects residual comparison and representation error.

Define the common-protocol baselines: Ridge, XGBoost, TabM, the deep ensemble,
the feature-free orbit smoother, and feature-set ablations.

#### 3.6 Safety and robustness experiments

Describe without reporting outcomes:

- far-cloud and clear-day negative controls;
- Nassar power-plant transects and matched control windows;
- spectral-channel plume-removal bounds;
- target-radius and TCCON-coincidence sensitivity;
- raw versus operational-BC versus ML comparison;
- surface and environmental failure-mode stratification.

### 4 Results

#### 4.1 Cloud-proximity phenomenology

Lead with the observation:

- near-cloud coverage statistics of the analysis set (see Display items);
- positive near-cloud response over land and negative response over ocean;
- characteristic land-r15 and ocean-r05 distance scales;
- quality-flag composition and observation loss with cloud distance.

This result establishes the problem before introducing correction skill,
and (2026-07-21g) motivates the surface-specific target radii on the page:
show the common r10 target first, then the adopted r05/r15 targets.

**Draft results text (interpretation moved out of the Fig. 3 caption,
2026-07-22h):**

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

**Draft caption:**

> **Figure 3.** Within-orbit clear-sky XCO2 anomaly versus nearest-cloud
> distance for the 2016–2020 analysis set (116 dates; 1-km bins). (a) Both
> surfaces under a common anomaly target whose clear-sky reference lies
> beyond 10 km (ocean n = 5.0 M, land n = 4.2 M). (b) The adopted
> production targets: ocean referenced beyond 5 km (r05, n = 7.8 M) and
> land beyond 15 km (r15, n = 3.8 M). Solid lines show the bin mean,
> dashed lines the median, and shading the interquartile range; dotted
> verticals mark each reference threshold, beyond which the anomaly
> returns to zero by construction. [If the box-plot rendering is chosen,
> replace the rendering sentence with: box = interquartile range,
> whiskers = 1.5×IQR, black line = bin mean.]

#### 4.2 Spectral evidence for a common 3-D mechanism

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
long); category atlases are in Supplement S3.


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
  case study moved to Appendix F — too long for the main text). Copy
  exists: `manuscript/figures/fig05_shadow_brightening_land` (from
  `results/figures/cld_dist_analysis/spec_sensitivity/`; PREDATES the
  locked style — regenerate on the next full-parquet pass; ocean variant
  available as appendix companion).
- No main-text table; the Tasman case is Appendix F (Fig. F1); category
  atlases and inventories are Supplement S3.

**Draft captions:**

> **Figure 4.** Surface-stratified near-cloud spectral response: effect
> size (Eq. \ref{eq:effect-size}; near-cloud 0–5 km minus far-cloud
> 20–50 km, in far-field standard deviations) of the reference-corrected
> path-length features for ocean and for each MCD12C1 land-cover class.
> Quality-flag-0, snow-free soundings; per-sounding references use the
> surface-specific clear-sky radii of the production targets (ocean beyond
> 5 km, land beyond 15 km); asterisks mark effects exceeding their
> analytic 95 % confidence interval. ⟨l′⟩ and var(l′) are the fitted mean
> and variance of the relative photon path (the cumulants k1, k2 of
> Sect. 3.2). Quality-flag sensitivity: Appendix B.

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

> **Figure 5.** Shadowing versus brightening. Near-cloud land footprints
> split by the sign of the O2A continuum-reflectance departure from the
> clear-sky reference (exp-intercept low: cloud shadowing; high: side
> illumination / brightening), with each branch's XCO2 anomaly and
> band-resolved Δ⟨l′⟩ response. (A single-overpass illustration against
> Aqua-MODIS true-colour imagery: Appendix F, Fig. F1.)

#### 4.3 Model comparison and feature attribution

Open with a two-sentence date-blocked headline so the model-selection result
has a stated skill basis: quote the frozen date-blocked \(R^2\)/RMSE by
surface, note that performance should be read against the label-noise ceiling
(ocean approximately noise-limited; land headroom concentrated near cloud),
and cite Appendix C for the full cross-validation analysis (random-split
inflation, fold dispersion, ceilings, and uncertainty calibration). Do not
expand beyond those two sentences — the detailed split-design material lives
in Appendix C so it does not interrupt the path to the independent
validation.

Then present the common-protocol baseline table, emphasizing the difficult
near-cloud land tail, and report:

- the full ensemble performs best overall;
- `no_spec` is approximately TCCON-neutral;
- `no_xco2` degrades strongly, particularly for near-cloud land QF1;
- spectral features remain conditionally informative when the XCO2-departure
  block is removed.

State the skill-versus-trust synthesis here for the first time.

**Draft results text (moved out of the Fig. 6 caption, 2026-07-22h):**

> The deep ensemble leads at every slice, and the model ordering is
> decided in the near-cloud land tail (Fig. 6a). In the feature-group
> ablation (Fig. 6b), dropping the spectral or contamination groups is
> TCCON-neutral, whereas removing the retrieval-state departure
> (xco2_raw − a priori) costs 0.7–1.1 ppm — the operationally
> load-bearing predictor, whose effectiveness the path-length analysis of
> Sect. 4.2 explains.

**Display items:**

- **Fig. 6** — baseline + feature-ablation composite, decided in the
  near-cloud land tail. Exists:
  `manuscript/figures/fig06_baseline_ablation`
  (`manuscript/scripts/make_baseline_ablation_figure.py`).
- **Table 1** — `manuscript/tables/tab_model_comparison.tex`: DE/XGB/Ridge
  fp-RMSE by slice, AK-harmonized only.
- **Table 2** — `manuscript/tables/tab_featureset_ablation.tex`: mix-DE
  feature-group ablation, AK-harmonized.
- The date-blocked skill/ceiling figure and CV-design schematic
  (`manuscript/figures/figC3b_cv_design`,
  `manuscript/scripts/make_cv_design_figure.py`) belong to Appendix C
  (Figs. C3–C4), not here.

**Draft caption:**

> **Figure 6.** Model and feature-set comparison under one protocol
> (identical features, date-blocked folds, and TCCON chain; AK-harmonised,
> r = 100 km / ±60 min). (a) TCCON per-footprint RMSE after correction,
> pooled over all quality flags (light, n = 105,683) and for the
> near-cloud land subset (≤ 10 km, dark, n = 75,157); the uncorrected
> product is shown for reference. (b) Feature-group ablation of the deep
> ensemble: ΔRMSE relative to the full feature set.

#### 4.4 Independent TCCON validation

Use AK-harmonized TCCON as the sole reported reference (decision 2026-07-21:
manuscript tables and figures carry no direct-reference columns). The direct
(non-harmonized) comparison is reduced to one sentence: note that direct
comparisons were run, that the residual approximately 0.3 ppm scale
difference is explained by the documented B7-to-B11 direct-TCCON anchoring
chain, and cite Appendix D. Report the current production values only after
regenerating the manuscript table from the frozen tag.

The paragraph order should be:

1. sample and coincidence definition;
2. mean absolute bias and footprint RMSE;
3. number of improved station-days;
4. Wilcoxon and site-clustered bootstrap results;
5. radius/window and QF0/QF1 sensitivity;
6. high-latitude and worsening cases.

Do not tabulate direct-reference results anywhere in the main text; the
single anchoring-chain sentence above is the only place the direct
comparison appears.

**Draft results text (moved out of the Fig. 7 caption, 2026-07-22h):**

> Across all 75 station-days (18 sites, 2016–2020) the station-day mean
> |bias| falls from 1.26 to 0.81 ppm and the per-footprint RMSE from 2.67
> to 1.19 ppm, with 71/75 station-days improved (Fig. 7a). The improvement
> is significant under a paired Wilcoxon test on station-day |bias|
> (p = 0.0064, n = 75) and under site-clustered bootstrap intervals, with
> and without Ny-Ålesund (Fig. 7b), and it holds at every collocation
> radius × window combination (25/50/100 km × ±30/60/120 min), largest at
> the tightest radii — as expected of a real, spatially localised
> correction.

**Display items:**

- **Fig. 7** — TCCON before/after validation (AK-harmonized), with the
  significance/robustness evidence. Both panels exist: (a) headline
  station-day before/after dumbbell —
  `manuscript/figures/fig07a_tccon_dumbbell` (copied from the production
  fold-PCA tree `<TAG>/atrain/tccon_ak_bias_dumbbell_label_r100km.png`);
  (b) significance/robustness —
  `manuscript/figures/fig07b_significance_robustness`
  (`manuscript/scripts/make_significance_panel.py`).
- **Table 3** — `manuscript/tables/tab_station_equal_bias.tex`:
  station-equal mean |bias| by QF, AK-harmonized only.
- Full coincidence matrix, station audit, and uncertainty budget are
  Appendix D (Figs. D1–D5, Tables D1–D4).

**Draft caption:**

> **Figure 7.** Independent TCCON validation (AK-harmonised reference;
> harmonisation follows Rodgers and Connor, 2003, and Wunch et al., 2017;
> before-vs-after differences are invariant to it). (a) Station-day mean
> bias before and after correction for all 75 station-days (18 sites,
> 2016–2020; r = 100 km, ±60 min). (b) Site-clustered bootstrap estimates
> with 95 % confidence intervals for the change in station-day mean
> |bias|, RMS bias, and per-footprint RMSE (all sites and excluding
> Ny-Ålesund), and the change in mean |bias| across collocation radius
> (25/50/100 km) × window (±30/60/120 min).

**Panel assembly, Fig. 7 (LaTeX draft):**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig07a_tccon_dumbbell.png}\\[4pt]
\includegraphics[width=0.95\textwidth]{figures/fig07b_significance_robustness.pdf}
\caption{<insert draft caption above>}
\label{fig:tccon-validation}
\end{figure}
```

Panel letters: fig07b carries its own internal tags from the
generator — retag when composing; fig07a (copied report figure) has
no panel letter yet.

#### 4.5 Distinguishing correction from smoothing

Place the feature-free smoother immediately after TCCON:

- it reduces footprint scatter more than the model;
- it leaves station-day bias nearly unchanged;
- the model reduces both bias and RMSE.

This establishes that the validation improvement is not merely local
denoising.

**Draft results text (moved out of the Fig. 8 caption, 2026-07-22h):**

> The smoother collapses footprint scatter more strongly than the deep
> ensemble (0.35–0.66 vs 0.78 ppm) yet leaves the TCCON station-day mean
> |bias| essentially unmoved (1.20–1.24 ppm, against 0.81 ppm for the
> model): variance removal alone cannot produce the observed bias
> reduction.

**Display items:**

- **Fig. 8** — smoother-null two-panel (scatter collapse vs unmoved
  station-day bias). Copy exists: `manuscript/figures/fig08_smoother_null`
  (from the production fold-PCA tree
  `<TAG>/atrain/smoother_null/smoother_null_r100km.png`,
  `workspace/smoother_null_figure.py`).
- No main-text table; the full smoother numerical table is Appendix E
  (table with Fig. E1).

**Draft caption:**

> **Figure 8.** Correction versus smoothing null test. A feature-free
> orbit-local running-mean smoother (±10/30/100 s half-widths, screened by
> the same input guards as the production correction) compared with the
> deep ensemble on footprint scatter (left) and TCCON station-day mean
> |bias| (right).

#### 4.6 Ocean validation and far-cloud controls

Present ATom, shipborne EM27/SUN, and the far-cloud/clear-day cases where the
correction is nearly inert. Treat these as independent corroboration, not as
equivalent in statistical weight to TCCON. State the limited sample and
reference-scale caveats.

**Draft results text (moved out of the Fig. 9 caption, 2026-07-22h):**

> Near-cloud ATom legs improve in median |residual| from 0.53 to 0.45 ppm,
> while the far-cloud date (9 October 2017) is a negative control that the
> correction leaves nearly unchanged (Fig. 9a). Against the shipborne
> spectrometers the correction collapses footprint scatter
> (σ 0.63 → 0.27 ppm) without moving the clear-sky control day
> (22 June 2019); the residual ~+1 ppm absolute offset is dominated by
> reference-scale differences, not created by the correction (Fig. 9b).

**Display items:**

- **Fig. 9** — ATom + shipborne EM27/SUN summary with the far-cloud/clear-day
  negative controls. Copies exist:
  `manuscript/figures/fig09a_atom_summary` and
  `manuscript/figures/fig09b_ship_summary` (from the production fold-PCA
  tree `<TAG>/atom/atom_pseudo_column_summary.png` and
  `<TAG>/ship/ship_comparison_summary.png`).
- No main-text table; leg/case inventories and residuals are Appendix D
  (Tables D5–D6).

**Draft caption:**

> **Figure 9.** Independent ocean corroboration. (a) ATom aircraft
> pseudo-column comparison (8 dates, 17 collocated legs, AK-smoothed; the
> unmeasured stratosphere is filled with the OCO-2 prior so it cancels in
> the comparison), including the far-cloud negative-control date
> (9 October 2017). (b) Shipborne EM27/SUN comparison (R/V Sonne MORE-2
> and R/V Mirai MR21-01; 100 km / ±2 h), including the clear-sky control
> day (22 June 2019).

**Panel assembly, Fig. 9 (LaTeX draft):**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig09a_atom_summary.png}\\[4pt]
\includegraphics[width=0.95\textwidth]{figures/fig09b_ship_summary.png}
\caption{<insert draft caption above>}
\label{fig:ocean-validation}
\end{figure}
```

Panel letters: both are copied report figures without (a)/(b) tags —
add letters when the panels are regenerated or composite them via a
small `manuscript/scripts/` wrapper.

#### 4.7 Plume preservation and correction safety

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

**Draft results text (moved out of the Fig. 10 caption, 2026-07-22h):**

> The ~+0.6 ppm enhancement at closest approach to the Westar plant
> survives the correction essentially unchanged (Fig. 10a). In the two
> flagged removal windows the O2A band shifts as strongly as the CO2
> bands (Fig. 10b) — the all-band fingerprint of cloud contamination
> rather than of a real plume, whose signature would be confined to the
> CO2 bands; the worst-case plume signal removable through the spectral
> channel is ≤ 0.21 ppm.

**Display items:**

- **Fig. 10** — plume-preservation composite: (a) preserved Westar 2023-06-26
  corrected transect — copy exists: `manuscript/figures/fig10a_westar_transect`
  (from `<TAG>/nassar_plumes/plume_preservation/transects/`); (b) plume-vs-cloud k1
  fingerprint contrast — exists: `manuscript/figures/fig10b_k1_contrast`
  (`manuscript/scripts/make_k1_contrast_figure.py`).
- **Table 4 → Appendix F (2026-07-22m; letter per 2026-07-22n merge).**
  `manuscript/tables/tab_nassar_attribution.tex` (control/plant removal
  percentages with full / −spec / −xco2 channel attribution) moves to
  Appendix F, merging with the planned Table F3 (same content — keep
  one). §4.7 prose already carries the three attribution percentages and
  the ≤0.21 ppm bound.
- All transects, per-case bounds, and control nulls are Appendix F
  (Figs. F2–F3, Tables F2–F3).

**Draft caption:**

> **Figure 10.** Plume preservation and the spectral cloud fingerprint.
> (a) Along-track transect over the Westar power plant (26 June 2023,
> clear sky, nearest cloud ≈ 50 km; overpass from the Nassar et al.
> catalogue) before and after correction; lower panels: predicted
> correction μ and nearest-cloud distance. (b) Band-resolved Δ⟨l′⟩ (plume
> window minus background; ±1 SE) for the two flagged removal windows and
> two clear-sky controls. Black ticks: signature expected of a real CO2
> plume (Δ⟨l′⟩ ≈ ⟨l′⟩·ΔXCO2/XCO2 in the CO2 bands via the prior-based
> optical depth; exactly zero in O2A).

**Panel assembly, Fig. 10 (LaTeX draft):**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig10a_westar_transect.png}\\[4pt]
\includegraphics[width=0.95\textwidth]{figures/fig10b_k1_contrast.pdf}
\caption{<insert draft caption above>}
\label{fig:plume-preservation}
\end{figure}
```

Panel letters: fig10b already carries its "(b)" tag from the
generator; fig10a (copied transect) needs an "(a)" tag added.

#### 4.8 Uncertainty and failure modes

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

#### 5.1 Physical interpretation

Synthesize, rather than repeat, the land–ocean response, WCO2 sign rule, and
shadow/brightening bifurcation. Explain why separate land and ocean models are
physically justified. Distinguish the empirical mechanism evidence from full
PPDF moment closure.

#### 5.2 Relationship to the operational bias correction

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
  `manuscript/tables/tab_raw_bc_ml.tex` (raw / bc / ML(bc) / ML(raw)
  fp-RMSE and bias by slice, AK-harmonized) moves to Appendix C as
  Table C8, alongside Fig. C7; Discussion 5.2 stays qualitative (three
  sentences citing Appendix C).

#### 5.3 Meaning and limits of imager independence

Use three tiers:

1. **Deployment:** no imager or neighboring footprints at inference.
2. **Sensitivity:** the spectrum contains cloud-proximity information,
   including a response below the MODIS resolution floor (cite in bulk:
   Supplement Sect. S4 — the former Appendix I, moved 2026-07-22m).
3. **Transferability:** the conceptual requirements are resolved absorption
   bands, channel-level prior optical depth, and single-footprint spectra.

State the caveat once:

> The imager serves as scaffolding for target construction and validation but
> is removed from the deployed correction.

Discuss post-2022 OCO-2 use and OCO-3, CO2M, GOSAT-GW, and TEMPO as prospects,
not demonstrated cross-sensor equivalence.

#### 5.4 Observation-recovery implications

Use production QF1 counts to quantify candidate recovery by distance, surface,
and region — ONE SENTENCE with the counts, no display item (main-text
Fig. 11 dropped 2026-07-22m). Avoid calling observations “usable” solely because RMSE is lower;
formal acceptance also requires retrieval validity, uncertainty calibration,
and downstream application tests.

Recommended language is **candidates for recovery** until an explicit
acceptance criterion and inversion experiment exist.

#### 5.5 Limitations

Collect the limitations in one subsection:

- anomaly-target circularity and possible real-gradient contamination;
- MYD35 mask semantics, sub-pixel cloud, and parallax limitations;
- land-heavy TCCON sampling and limited ocean-reference sample;
- high-latitude and representation-error residuals;
- absence of fitted-versus-tallied PPDF moment closure;
- possible attenuation of local CO2 contrast by retrieval-state predictors.

#### 5.6 Future work

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

The main text uses exactly ten figures (2026-07-21c/d renumbering; the
optional eleventh figure was DROPPED 2026-07-22m — §4.8 carries no display
item and Discussion 5.4 carries the QF1 counts in one sentence).
(The earlier option of compositing the decay curves with the k1/k2 distance
responses lapsed when Fig. 3 moved to Results 4.1, 2026-07-21g; if a
figure must be cut, fold the k1/k2 distance responses into Fig. 4
instead.)

| Figure | Manuscript role | Primary message |
|---|---|---|
| 1 | Methods 3.1 | Collocation geometry schematic: how cloud distance and the anomaly target are constructed. |
| 2 | Methods 3.3 | Deep-ensemble architecture (no-cloud variant): member MLP, ensemble mixture, conformal calibration; no cloud information at inference. |
| 3 | Results 4.1 | Anomaly–distance decay: (a) common r10 target motivates the surface-specific radii; (b) adopted r05/r15 targets — opposite-sign land/ocean phenomenon. |
| 4 | Results 4.2 | WCO2 land-cover response changes sign across the measured albedo-contrast axis. |
| 5 | Results 4.2 | Shadow and brightening branches connect spectral response to opposite XCO2 anomalies. |
| 6 | Results 4.3 | Baseline/feature-ablation comparison decided in the near-cloud land tail; date-blocked skill and noise ceiling carried as a headline sentence, detail in Appendix C. |
| 7 | Results 4.4 | TCCON before/after validation with AK-harmonized primary reference and uncertainty. |
| 8 | Results 4.5 | The feature-free smoother removes scatter but not station-day bias. |
| 9 | Results 4.6 | ATom and shipborne ocean validation plus far-cloud negative controls. |
| 10 | Results 4.7 | Plume-preservation transect and channel-attribution safety budget. |
| ~~11~~ | — | DROPPED 2026-07-22m (was: failure boundary / QF1 recovery); QF1 counts are one sentence in Discussion 5.4. |

Main-text table budget (THREE tables since 2026-07-22m, all AK-harmonized
only, generated by `manuscript/scripts/make_manuscript_tables.py` into
`manuscript/tables/`; the generator still writes all five files — two are
consumed by appendices):

| Table | Section | File | Content |
|---|---|---|---|
| 1 | Results 4.3 | `tab_model_comparison.tex` | DE/XGB/Ridge fp-RMSE by slice. |
| 2 | Results 4.3 | `tab_featureset_ablation.tex` | Feature-group ablation by slice. |
| 3 | Results 4.4 | `tab_station_equal_bias.tex` | Station-equal mean \|bias\| by QF. |
| — | Appendix F | `tab_nassar_attribution.tex` | Moved 2026-07-22m (merges with planned Table F3). |
| — | Appendix C | `tab_raw_bc_ml.tex` | Moved 2026-07-22m (Table C8, with Fig. C7). |

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
for the multi-file figures (5, 7, 9, 10). Shared caveats: numerical values
must be re-verified against the frozen fold-PCA tag at writing time (§7);
captions use the lowercase italic l′ notation (2026-07-22 final) and the AK-only reference decision; internal
"Sect. X" references to be resolved in LaTeX.

Supplement contents are governed by the **Supplement plan (S1–S6)** at the
end of §5 (2026-07-22m; this list previously conflicted with the appendix
sections — resolved there: feature lists/hyperparameters STAY in
Appendix C, the Monte Carlo demonstration follows the Appendix G
conditional, cross-sensor feasibility STAYS in Appendix H, the compact 3×3
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
| A | Spectral fitting and path-length derivation | A |
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

### Appendix A: Spectral fitting and photon path-length derivation

**Purpose:** make the physical observable and numerical fit independently
reviewable.

Include:

- derivation of the cumulant expansion from the Laplace-transform view of the
  photon path-length distribution;
- sign and factorial conventions for \(k_1\), \(k_2\), and higher-order terms;
- band windows, optical-depth construction, channel masks, polynomial orders,
  bounds, and fallback behavior;
- examples of accepted and rejected fits;
- no-Savitzky–Golay versus Savitzky–Golay comparison;
- convergence-radius and fit-order sensitivity;
- fitting failure counts by band and failure category, if available.

Planned items:

- **Fig. A1:** representative O2A, WCO2, and SCO2 fits, including residuals
  and fitted \(l'\) statistics;
- **Fig. A2:** SG-versus-no-SG comparison (SUMMARY panel only, sourced from
  working figure A5; the full SG robustness sweep and the extended
  accepted/rejected fit-example gallery go to Supplement S5, 2026-07-22m);
- **Table A1:** complete fitting configuration and QC thresholds;
- **Table A2:** fit availability and failure accounting by band and surface.

Do not present spectrum-fitted cumulants as directly tallied photon moments.
The mathematical interpretation, numerical estimator, and Monte Carlo causal
test should remain distinct.

### Appendix B: Data provenance, cloud collocation, and target construction

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

This appendix must resolve the currently mixed 116-date/17.8-million and
140-date/21.5-million cohorts by naming the role of each population.

### Appendix C: Correction model — architecture, training, and
cross-validated evaluation

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
  the linreg fold dirs; train/calibration/held roles per fold, both
  surfaces). This supersedes the stylized-timeline caveat on figC3b —
  the manifests ARE local;
- **Table C1:** full predictor inventory;
- **Table C2:** hyperparameters and training configuration;
- **Table C3:** fold-level sample sizes and performance;
- **Table C4:** training manifests and zero-overlap verification.

Keep the main text to predictor groups and essential architecture. Individual
features and hyperparameters belong here.

**Second job (former Appendix D, merged 2026-07-22n) — cross-validation,
baselines, and feature attribution:** show that the selected model and
split design are not arbitrary,
and carry the full cross-validated correction-performance analysis (moved out
of the main Results 2026-07-21; the main text keeps only a two-sentence
date-blocked headline in §4.3).

Include:

- random versus date-blocked performance by fold and surface, with the
  random-split inflation stated as a cautionary methodological result;
- date-blocked skill by surface and near/far-cloud regime, with fold
  dispersion;
- label-noise-ceiling derivation cross-referenced to Appendix B, and the
  ceiling-relative interpretation: ocean global skill is approximately
  noise-limited, while land headroom is concentrated in the near-cloud
  regime;
- Ridge, XGBoost, TabM, and deep-ensemble common-protocol results;
- full feature-set ablations, including QF and near-cloud strata;
- raw-XCO2 versus operational-BC versus ML comparisons;
- uncertainty calibration against the anomaly target.

Planned items:

- **Fig. C3:** random-split inflation relative to date-blocked estimates
  — panel a GENERATED 2026-07-22q: `figC3a_random_split_inflation`
  (random / date-single / date-kfold R² for TabM/MLP/XGB on the ocean
  testbed, from `ocean_robustness_comparison.md` — caption must say the
  SPLIT effect is the point, the models are the 2026-06 comparison-era
  ones, not the production DE); panel b exists:
  `manuscript/figures/figC3b_cv_design` — renamed from figD1b
  2026-07-22o — `manuscript/scripts/make_cv_design_figure.py`);
- **Fig. C4:** date-blocked skill versus the label-noise ceiling by surface
  and cloud-distance regime — PENDING (2026-07-22q): NOT generatable from
  `label_noise_ceiling_140dates.csv` alone; the achieved ocean fold R²
  (0.708) exceeds that CSV's r2max_ref_ret (0.603), so the ceiling-column
  semantics must come from the original ceiling analysis (and the §7
  frozen metrics) before this figure can be drawn honestly;
- **Fig. C5:** model comparison — surface-level version GENERATED
  2026-07-22q: `figC5_cv_model_comparison` (fold mean ± std RMSE/R² for
  DE/XGBoost/Ridge, both surfaces, parsed from the
  `*_de_vs_baselines_kfold_agg.md` reports; land Ridge shown as fold
  MEDIAN, hatched — outlier fold per the agg report). The
  by-cloud-distance-stratum variant needs per-regime fold metrics (not
  in the local agg reports);
- **Fig. C6:** feature-ablation changes relative to the full model —
  COVERED (2026-07-22q) by main-text Fig. 6b (TCCON ablation) + Table C7
  (complete CV ablations); no separate figure unless a CV-side ablation
  visual proves necessary;
- **Fig. C7:** raw/BC/ML increment relationship — PENDING (2026-07-22q):
  needs the ML-on-raw artifacts (local policy CSVs carry only
  uncorrected/full_mu scenarios);
- **Table C5:** fold-resolved date-blocked metrics (the frozen values quoted
  in §4.3);
- **Table C6:** fold-resolved baseline results;
- **Table C7:** complete ablation results, using the 2026-07-17 retrained
  variants only;
- **Table C8:** raw / bc / ML(bc) / ML(raw) metrics by slice
  (`tab_raw_bc_ml.tex` — former main-text Table 5, moved here
  2026-07-22m; pairs with Fig. C7).

The appendix should preserve the null result: `no_spec` is approximately
TCCON-neutral. Do not select only strata that make the spectral block appear
predictively essential.

### Appendix D: Independent validation — TCCON protocol and ocean
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

- **Fig. D1:** 3-by-3 coincidence sensitivity, sourced from working figure A2;
- **Fig. D2:** QF0 and QF1 station-day comparisons — staged 2026-07-22o:
  `figD2a_tccon_qf0` + `figD2b_tccon_qf1` (copies of
  `<TAG>/atrain/tccon_ak_bias_qf{0,1}_r100km.png`);
- **Fig. D3:** r50 robustness comparison — staged 2026-07-22o:
  `figD3_tccon_r50` (copy of `<TAG>/atrain/tccon_ak_bias_r50km.png`);
- **Fig. D4:** compact station-grouped summary ONLY (all 75 individual
  station-day before/after panels move to Supplement S1, 2026-07-22m;
  drop D4 entirely if the summary duplicates Fig. 7a) — candidate
  staged 2026-07-22o: `figD4_station_summary` (copy of
  `<TAG>/atrain/tccon_ak_by_site_bias_r100km.png`);
- **Fig. D5:** random-effects residual forest plot, derived from working A9;
- **Table D1:** station inventory and coincidence counts;
- **Table D2:** complete AK-harmonized metrics;
- **Table D3:** paired Wilcoxon and site-clustered bootstrap results;
- **Table D4:** uncertainty components, \(\tau\), \(I^2\), and coverage.

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
  if it adds beyond main-text Fig. 9a, else the ocean subsection is
  methods + tables only;
- ~~Figs. H2–Hn, ship individual cases~~: all per-date ATom panels and
  individual ship cases move to Supplement S2 (2026-07-22m; the ocean
  subsection shrinks to ~2 typeset pages — the construction methods and the two inventory
  tables are what §4.6's quoted numbers lean on);
- **Table D5:** ATom leg inventory and residuals;
- **Table D6:** ship case inventory and residuals.

Do not pool these observations with TCCON into a single headline metric.

### Appendix E: Null tests, negative controls, and failure cases
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

- **Fig. E1:** smoother null, sourced from working figure A4;
- **Fig. E2:** far-cloud and clear-day control cases — staged
  2026-07-22o: `figE2a_atom_farcloud_control` (ATom 2017-10-09) +
  `figE2b_ship_clearday_control` (ship 2019-06-22);
- **Fig. E3:** failure rates and residuals by environmental driver —
  staged 2026-07-22o: `figE3_failure_modes` (copy of
  `<TAG>/failure_modes/fig_failure_modes_r100km.png`);
- **Fig. E4:** high-latitude and post-2022 cases, sourced from working A7;
- **Table E1:** all worsening cases and diagnosed cause;
- **Table E2:** performance by bright surface, snow, AOD, latitude, and
  predictive-uncertainty strata.

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
- **Fig. F2:** all Nassar transects with identical axes and annotations;
- **Fig. F3:** band-resolved \(\Delta l'\) for plume and cloud-contaminated
  windows;
- **Table F1:** vetted case inventory and category assignments;
- **Table F2:** per-case plume-removal bounds and control-null results;
- **Table F3:** full/no-spec/no-xco2 smoothing attribution — this IS the
  former main-text Table 4 (`tab_nassar_attribution.tex`, moved here
  2026-07-22m); one table, not two.

The main text shows only the Westar preservation case (the Tasman
mechanism showcase moved here as Fig. F1, 2026-07-22c). The appendix must show the full set to avoid
case-selection concerns.

### Appendix G: Controlled 3-D radiative-transfer demonstration
(formerly Appendix J)

**Purpose:** provide a causal mechanism test while keeping its limited scope
explicit.

**CONDITIONAL PLACEMENT (2026-07-22m):** keep as a typeset appendix ONLY if
finished to manuscript standard (it is the paper's only causal — not
correlational — mechanism evidence, and §3.2's causal-sensitivity sentence
leans on it, so typeset is preferred). If it stays single-geometry-only or
is not polished in time, move the material to Supplement S6 and soften the
§3.2 sentence to cite the Supplement — that landing spot replaces the old
delete-or-delay choice.

Include the backward-Monte-Carlo 3-D-versus-ICA experiment now tracked as
working figure A13:

- model, band, geometry, cloud optical properties, surface albedo, grid, and
  photon-sampling configuration;
- identical cloud scene under full 3-D transport and ICA;
- refitting with the production spectral estimator;
- \(l'\), variance, and intercept response across the cloud boundary;
- the one-sided shadow-edge decay;
- limitations of a single representative geometry.

Planned items:

- **Fig. G1:** controlled 3-D-versus-ICA comparison;
- **Table G1:** simulation configuration and estimator settings.

State explicitly that this validates qualitative sensitivity of the fitted
features to horizontal photon transport. Direct comparison between fitted
cumulants and tallied photon-path moments, a geometry sweep, and plume
injection remain future work.

### Appendix H: Cross-sensor feasibility (TEMPO; formerly Appendix K)

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

- **Fig. H1:** TEMPO RGB/cloud field, fitted cumulants, and distance response;
- **Table H1:** OCO-2 and TEMPO inputs required by the spectral fit.

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

### Appendix triage

Resolved 2026-07-22m/n into the appendix/Supplement split above
(letters below are the FINAL 2026-07-22n letters):

1. **Required (typeset):** A–F — method, model+CV, validation (TCCON +
   ocean), nulls/failures, and case/plume audit — plus H (user decision
   2026-07-22: the cross-mission bridge, existence-proof scope).
2. **Conditional:** G (3-D RT) — typeset appendix if finished to
   manuscript standard, else its material goes to Supplement S6.
3. **Moved to Supplement:** the former Appendix I (whole section → S4)
   and every per-case gallery (S1–S3).

Never include a placeholder or partially documented simulation. If working
figures A12 or A13 are not completed to manuscript standard, retain their
claims as future work and remove the corresponding appendix references.

### Supplement plan (S1–S6, 2026-07-22m)

Published as one author-formatted PDF with its own DOI. Rules (per the
§5 ADMISSION RULE): unrestricted in logical scope but NO new central
scientific conclusions; cited from the paper in BULK only ("see
Supplement Sect. Sx"), never as the audit trail for a specific main-text
number — everything quantitative the paper leans on stays in the main
text or a lettered appendix. Reviewers see the Supplement during review;
it is not copy-edited, so it must be self-contained and carry its own
caption discipline (the §4 caption rule applies).

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
  outgrow Appendix D/E, and the Appendix G (3-D RT) material if it does
  not reach typeset standard.

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

Keep XCO2 typesetting and notation consistent with the AMT figure style:
\(X_{\mathrm{CO_2}}\), \(l'\), \(k_1\), and \(k_2\).

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
   fig10b regenerated).
8. **Near-cloud coverage statistics:** re-freeze the Results 4.1 sentence on
   the final analysis cohort. 2026-07-21 values from
   `combined_2016_2020_dates.parquet` (17.75 M valid-cloud-distance
   footprints, 99.9 % of 17.77 M rows / 116 dates): all footprints 41.7 %
   < 4 km and 60.7 % < 10 km; ocean 59.1 % < 5 km; land 46.8 % < 15 km;
   median nearest-cloud distance 3.5 km (ocean) / 17.9 km (land).

## 8. Recommended writing order

Draft in evidence order rather than manuscript order:

1. Methods 3.1–3.6, freezing definitions and evaluation units.
2. Results 4.1–4.8 directly from final tables and figures (draft the
   Appendix D cross-validation material alongside 4.3, since its conclusions
   feed the two-sentence headline there).
3. Data 2.1–2.4, reconciling cohort provenance and product versions.
4. Discussion 5.1–5.6, constrained to what Results demonstrate.
5. Introduction 1.1–1.3, positioning the now-fixed contribution.
6. Conclusions, then Abstract and title.

This order minimizes narrative drift and prevents the Introduction or Abstract
from promising stronger claims than the final evidence supports.

## 9. Final narrative check

Before submission, a reader should be able to answer these questions in order:

1. What cloud-proximity error is observed before any model is applied?
2. What spectral evidence supports a 3-D radiative mechanism?
3. Exactly what information is and is not available to the correction at
   inference?
4. Does performance survive date blocking (headline sentence in 4.3, full
   analysis in Appendix D) and independent validation?
5. Is the gain bias correction rather than smoothing?
6. Are real plume enhancements preserved in the tested cases?
7. Where does the correction remain unreliable?
8. What part of the workflow is genuinely transferable without an imager?

If each answer follows directly from one main figure or table, the manuscript
has the intended AMT flow: **observation → mechanism → method → independent
validation → trust boundary → application scope**.

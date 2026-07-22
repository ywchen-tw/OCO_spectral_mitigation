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

Use \(L'\), rather than the superseded lowercase \(l'\), throughout text and
figures.

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
numerical split-inflation result belongs in Appendix D (the main text carries
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
> beyond 10 km: the ocean response (n = 5.0 M) has decayed by ~5 km, well
> inside the threshold, whereas the land response (n = 4.2 M) is still
> ~0.5 ppm when the 10-km reference cut truncates it by construction — the
> common threshold is too tight for land and unnecessarily wide for ocean,
> motivating the surface-specific radii. (b) The adopted production
> targets — ocean referenced beyond 5 km (r05, n = 7.8 M) and land beyond
> 15 km (r15, n = 3.8 M): the land response indeed persists to ~15 km and
> the ocean curve is unchanged; dotted lines mark each reference
> threshold, beyond which the anomaly returns to zero by construction. The
> near-cloud response is opposite in sign — negative over ocean, positive
> over land — before any correction is applied. Solid lines show the bin
> mean, dashed lines the median, and shading the interquartile range: over
> ocean the whole distribution shifts negative near cloud, whereas over
> land the mean far exceeds the nearly unmoved median — the land bias is
> carried by a skewed tail of strongly affected soundings rather than a
> shift of the full population. [If the box-plot rendering is chosen,
> replace this sentence with: box = interquartile range, whiskers =
> 1.5×IQR, black line = bin mean.]

#### 4.2 Spectral evidence for a common 3-D mechanism

Build a three-part evidence chain:

1. band- and surface-resolved \(k_1/k_2\) responses with cloud distance;
2. the WCO2 land-cover sign rule: savanna approximately \(+0.43\sigma\) and
   barren approximately \(-0.40\sigma\);
3. shadow and brightening branches with opposite XCO2 responses.

Interpret ocean as the dark endpoint of the cloud–surface contrast axis, but
label this as a synthesis supported by the observations rather than a closed
quantitative derivation.

Use the verified wording:

> The sign of the WCO2 response follows the measured contrast where land
> classes straddle the cloud albedo; response magnitudes do not monotonically
> follow contrast.

Do not use the superseded “forest sign flip” or “albedo-contrast ordering”
claims. Close with one RGB case study; move category atlases to the Supplement.


**Display items:**

- **Fig. 4** — WCO2 land-cover sign rule: ref-corrected effect-size heatmap
  (savanna +0.43σ vs barren −0.40σ). Copy exists:
  `manuscript/figures/fig04_landclass_effect_heatmap` (from
  `results/figures/cld_dist_analysis/land_class/landclass_effect_heatmap.png`;
  PREDATES the locked style and the l′→L′ symbol change — regenerate via
  `run_all.py` on the next full-parquet pass before submission).
- **Fig. 5** — shadow vs brightening branches (O2A exp-intercept split) with
  opposite XCO2 responses, closing with the single vetted RGB case study
  (spectrum-derived variables + uncorrected XCO2 only). Copies exist:
  `manuscript/figures/fig05a_shadow_brightening_land` (from
  `results/figures/cld_dist_analysis/spec_sensitivity/`; PREDATES the
  locked style / l′→L′ — regenerate on the next full-parquet pass; ocean
  variant available as appendix companion) and
  `manuscript/figures/fig05b_case_tasman` (Tasman 2018-05-01, regenerated
  2026-07-11 in the locked style — current).
- No main-text table; case atlases and category inventories are Appendix G.

**Draft captions:**

> **Figure 4.** Land-cover-stratified near-cloud spectral response. Effect
> size (near-cloud minus far-cloud reference, normalised by the far-cloud
> spread) of the reference-corrected path-length features by MCD12C1
> land-cover class. ⟨L′⟩ and var(L′) are the fitted mean and variance of
> the relative photon path (the cumulants k1, k2 of Sect. 3.2). The sign
> of the WCO2 Δ⟨L′⟩ response follows the sign of the cloud−surface albedo
> contrast: positive over vegetated classes (savanna +0.43σ) and negative
> over bright barren surfaces (−0.40σ); magnitudes do not rank with
> contrast, so the albedo-contrast mechanism enters as a sign rule.

> **Figure 5.** Shadowing versus brightening, and a single-overpass
> example. (a) Near-cloud land footprints split by the sign of the O2A
> continuum-reflectance departure from the clear-sky reference
> (exp-intercept low: cloud shadowing; high: side illumination /
> brightening); the two branches carry opposite-signed XCO2 anomalies and
> distinct band-resolved Δ⟨L′⟩ responses, separable from a single
> footprint's spectrum alone. (b) Case study, Tasman Sea, 1 May 2018:
> Aqua-MODIS true-colour image (NASA Worldview/GIBS) with the OCO-2 glint
> track, per-footprint nearest-cloud distance, uncorrected XCO2 anomaly,
> and band-resolved spectral response — coherent path-length and XCO2
> signatures co-located with the imaged cloud edge.

**Panel assembly, Fig. 5 (LaTeX draft):**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig05a_shadow_brightening_land.png}\\[4pt]
\includegraphics[width=0.95\textwidth]{figures/fig05b_case_tasman.png}
\caption{<insert draft caption above>}
\label{fig:shadow-brightening-case}
\end{figure}
```

#### 4.3 Model comparison and feature attribution

Open with a two-sentence date-blocked headline so the model-selection result
has a stated skill basis: quote the frozen date-blocked \(R^2\)/RMSE by
surface, note that performance should be read against the label-noise ceiling
(ocean approximately noise-limited; land headroom concentrated near cloud),
and cite Appendix D for the full cross-validation analysis (random-split
inflation, fold dispersion, ceilings, and uncertainty calibration). Do not
expand beyond those two sentences — the detailed split-design material lives
in Appendix D so it does not interrupt the path to the independent
validation.

Then present the common-protocol baseline table, emphasizing the difficult
near-cloud land tail, and report:

- the full ensemble performs best overall;
- `no_spec` is approximately TCCON-neutral;
- `no_xco2` degrades strongly, particularly for near-cloud land QF1;
- spectral features remain conditionally informative when the XCO2-departure
  block is removed.

State the skill-versus-trust synthesis here for the first time.

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
  (`manuscript/figures/figD1b_cv_design`,
  `manuscript/scripts/make_cv_design_figure.py`) belong to Appendix D
  (Figs. D1–D2), not here.

**Draft caption:**

> **Figure 6.** Model and feature-set comparison under one protocol
> (identical features, date-blocked folds, and TCCON chain; AK-harmonised,
> r = 100 km / ±60 min). (a) TCCON per-footprint RMSE after correction,
> pooled over all quality flags (light, n = 105,683) and for the decisive
> near-cloud land subset (≤ 10 km, dark, n = 75,157); the uncorrected
> product is shown for reference. The deep ensemble leads throughout, and
> the ordering is decided in the near-cloud land tail. (b) Feature-group
> ablation of the deep ensemble (ΔRMSE relative to the full feature set):
> dropping the spectral or contamination groups is TCCON-neutral, whereas
> removing the retrieval-state departure (xco2_raw − a priori) costs
> 0.7–1.1 ppm — the operationally load-bearing predictor, whose
> effectiveness the path-length analysis of Sect. 4.2 explains.

#### 4.4 Independent TCCON validation

Use AK-harmonized TCCON as the sole reported reference (decision 2026-07-21:
manuscript tables and figures carry no direct-reference columns). The direct
(non-harmonized) comparison is reduced to one sentence: note that direct
comparisons were run, that the residual approximately 0.3 ppm scale
difference is explained by the documented B7-to-B11 direct-TCCON anchoring
chain, and cite Appendix E. Report the current production values only after
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
  Appendix E (Figs. E1–E5, Tables E1–E4).

**Draft caption:**

> **Figure 7.** Independent TCCON validation (AK-harmonised reference;
> harmonisation follows Rodgers and Connor, 2003, and Wunch et al., 2017;
> before-vs-after differences are invariant to it). (a) Station-day mean
> bias before and after correction for all 75 station-days (18 sites,
> 2016–2020, r = 100 km, ±60 min): mean |bias| falls from 1.26 to
> 0.81 ppm and per-footprint RMSE from 2.67 to 1.19 ppm, with 71/75
> station-days improved. (b) Significance and robustness: site-clustered
> bootstrap estimates with 95 % confidence intervals for the change in
> station-day mean |bias|, RMS bias, and per-footprint RMSE (all sites and
> excluding Ny-Ålesund; paired Wilcoxon on station-day |bias| p = 0.0064,
> n = 75), and the change in mean |bias| across collocation radius
> (25/50/100 km) × window (±30/60/120 min) — significant at every
> combination and largest at the tightest radii, as expected of a real,
> spatially localised correction.

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

**Display items:**

- **Fig. 8** — smoother-null two-panel (scatter collapse vs unmoved
  station-day bias). Copy exists: `manuscript/figures/fig08_smoother_null`
  (from the production fold-PCA tree
  `<TAG>/atrain/smoother_null/smoother_null_r100km.png`,
  `workspace/smoother_null_figure.py`).
- No main-text table; the full smoother numerical table is Appendix F
  (Table with Fig. F1).

**Draft caption:**

> **Figure 8.** Correction versus smoothing. A feature-free orbit-local
> running-mean smoother (±10/30/100 s half-widths, screened by the same
> input guards as the production correction) collapses footprint scatter
> more than the deep ensemble (0.35–0.66 vs 0.78 ppm) yet leaves the TCCON
> station-day bias essentially unmoved (mean |bias| 1.20–1.24 ppm vs
> 0.81 ppm for the model): variance removal alone cannot produce the
> observed bias reduction.

#### 4.6 Ocean validation and far-cloud controls

Present ATom, shipborne EM27/SUN, and the far-cloud/clear-day cases where the
correction is nearly inert. Treat these as independent corroboration, not as
equivalent in statistical weight to TCCON. State the limited sample and
reference-scale caveats.

**Display items:**

- **Fig. 9** — ATom + shipborne EM27/SUN summary with the far-cloud/clear-day
  negative controls. Copies exist:
  `manuscript/figures/fig09a_atom_summary` and
  `manuscript/figures/fig09b_ship_summary` (from the production fold-PCA
  tree `<TAG>/atom/atom_pseudo_column_summary.png` and
  `<TAG>/ship/ship_comparison_summary.png`).
- No main-text table; leg/case inventories and residuals are Appendix H
  (Tables H1–H2).

**Draft caption:**

> **Figure 9.** Independent ocean corroboration. (a) ATom aircraft
> pseudo-column comparison (8 dates, 17 collocated legs, AK-smoothed; the
> unmeasured stratosphere is filled with the OCO-2 prior so it cancels in
> the comparison): near-cloud legs improve in median |residual| from 0.53
> to 0.45 ppm, while the far-cloud date (9 October 2017) is a negative
> control the correction leaves nearly unchanged. (b) Shipborne EM27/SUN
> comparison (R/V Sonne MORE-2 and R/V Mirai MR21-01; 100 km / ±2 h): the
> correction collapses footprint scatter (σ 0.63 → 0.27 ppm) without
> moving the clear-sky control day (22 June 2019); the residual ~+1 ppm
> absolute offset is dominated by reference-scale differences, not created
> by the correction.

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

**Display items:**

- **Fig. 10** — plume-preservation composite: (a) preserved Westar 2023-06-26
  corrected transect — copy exists: `manuscript/figures/fig10a_westar_transect`
  (from `<TAG>/nassar_plumes/plume_preservation/transects/`); (b) plume-vs-cloud k1
  fingerprint contrast — exists: `manuscript/figures/fig10b_k1_contrast`
  (`manuscript/scripts/make_k1_contrast_figure.py`).
- **Table 4** — `manuscript/tables/tab_nassar_attribution.tex`: control/plant
  removal percentages with full / −spec / −xco2 channel attribution.
- All transects, per-case bounds, and control nulls are Appendix G
  (Figs. G8–G9, Tables G2–G3).

**Draft caption:**

> **Figure 10.** The correction preserves real CO2 enhancements. (a)
> Along-track transect over the Westar power plant (26 June 2023, clear
> sky, nearest cloud ≈ 50 km; overpass from the Nassar et al. catalogue):
> the ~+0.6 ppm enhancement at closest approach survives the correction
> essentially unchanged; lower panels: predicted correction μ and
> nearest-cloud distance. (b) Band-resolved Δ⟨L′⟩ (plume window minus
> background; ±1 SE) for the two flagged removal windows and two clear-sky
> controls. Black ticks: signature expected of a real CO2 plume
> (Δ⟨L′⟩ ≈ ⟨L′⟩·ΔXCO2/XCO2 in the CO2 bands via the prior-based optical
> depth; exactly zero in O2A). In the flagged windows O2A shifts as
> strongly as the CO2 bands — the all-band fingerprint of cloud
> contamination, not a plume; the worst-case plume signal removable
> through the spectral channel is ≤ 0.21 ppm.

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

- **Fig. 11 (optional)** — failure boundary (bright-surface / high-latitude /
  σ-decile strata) or the QF1 candidate-recovery quantification, whichever
  is quantitatively stronger; drop if neither earns main-text space.
  QF1-recovery candidate copy exists:
  `manuscript/figures/fig11_qf1_recovery_candidate` (from
  `<TAG>/atrain/tccon_ak_bias_qf1_r100km.png`); the failure-boundary
  alternative still to compose from the failure-mode stratification.
- No main-text table; worsening-case and strata tables are Appendix F
  (Tables F1–F2), uncertainty components Appendix E (Table E4).

**Draft caption:**

> **Figure 11 (optional).** Quality-flagged near-cloud soundings as
> recovery candidates. Station-day bias against AK-harmonised TCCON before
> and after correction, restricted to QF = 1 (flagged) soundings: the
> correction moves the flagged population toward QF-0 quality, the basis
> of the observation-recovery estimate of Sect. 5.4. (Alternative content
> if quantitatively stronger: failure-boundary strata from Sect. 4.8;
> include only one, or drop.)

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

- **Table 5** — `manuscript/tables/tab_raw_bc_ml.tex`: raw / bc / ML(bc) /
  ML(raw) fp-RMSE and bias by slice, AK-harmonized. If the discussion stays
  qualitative, this table may move to Appendix D alongside Fig. D5 (the
  raw/BC/ML increment relationship).

#### 5.3 Meaning and limits of imager independence

Use three tiers:

1. **Deployment:** no imager or neighboring footprints at inference.
2. **Sensitivity:** the spectrum contains cloud-proximity information,
   including a response below the MODIS resolution floor.
3. **Transferability:** the conceptual requirements are resolved absorption
   bands, channel-level prior optical depth, and single-footprint spectra.

State the caveat once:

> The imager serves as scaffolding for target construction and validation but
> is removed from the deployed correction.

Discuss post-2022 OCO-2 use and OCO-3, CO2M, GOSAT-GW, and TEMPO as prospects,
not demonstrated cross-sensor equivalence.

#### 5.4 Observation-recovery implications

Use production QF1 counts to quantify candidate recovery by distance, surface,
and region. Avoid calling observations “usable” solely because RMSE is lower;
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

The main text should use ten figures (2026-07-21c/d renumbering: the
collocation schematic moved to Methods 3.1, the decay curves to Results
4.2, and the architecture schematic to Methods 3.3). An eleventh figure is
acceptable only if the data-recovery result is quantitatively strong.
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
| 6 | Results 4.3 | Baseline/feature-ablation comparison decided in the near-cloud land tail; date-blocked skill and noise ceiling carried as a headline sentence, detail in Appendix D. |
| 7 | Results 4.4 | TCCON before/after validation with AK-harmonized primary reference and uncertainty. |
| 8 | Results 4.5 | The feature-free smoother removes scatter but not station-day bias. |
| 9 | Results 4.6 | ATom and shipborne ocean validation plus far-cloud negative controls. |
| 10 | Results 4.7 | Plume-preservation transect and channel-attribution safety budget. |
| 11, optional | Results 4.8 or Discussion 5.4 | Failure boundary or quantitatively defined QF1 candidate recovery. |

Main-text table budget (five tables, all AK-harmonized only, generated by
`manuscript/scripts/make_manuscript_tables.py` into `manuscript/tables/`):

| Table | Section | File | Content |
|---|---|---|---|
| 1 | Results 4.3 | `tab_model_comparison.tex` | DE/XGB/Ridge fp-RMSE by slice. |
| 2 | Results 4.3 | `tab_featureset_ablation.tex` | Feature-group ablation by slice. |
| 3 | Results 4.4 | `tab_station_equal_bias.tex` | Station-equal mean \|bias\| by QF. |
| 4 | Results 4.7 | `tab_nassar_attribution.tex` | Smoothing channel attribution at plume cases. |
| 5 | Discussion 5.2 | `tab_raw_bc_ml.tex` | raw/bc/ML(bc)/ML(raw) series (may move to Appendix D). |

Per-section figure/table assignments, artifact status, and generator scripts
are in the **Display items** blocks under each Results/Discussion section
above.

### Figure captions

Draft captions live under each section's **Display items** block (moved
2026-07-21f for easier reading), together with LaTeX panel-assembly drafts
for the multi-file figures (5, 7, 9, 10). Shared caveats: numerical values
must be re-verified against the frozen fold-PCA tag at writing time (§7);
captions use the L′ notation and the AK-only reference decision; internal
"Sect. X" references to be resolved in LaTeX.

Move the following to the Supplement:

- detailed spectral-fit robustness and Savitzky–Golay comparison;
- all category case-study atlases;
- complete coincidence-radius/window matrix;
- all station-day case panels;
- full feature lists and architecture hyperparameters;
- Monte Carlo 3-D-versus-ICA demonstration;
- sub-pixel and NoMODIS-era diagnostics;
- cross-sensor feasibility examples;
- secondary uncertainty and failure-mode tables.

Do not place a corrected transect in the early mechanism section. The early
case study should show spectrum-derived variables and uncorrected XCO2 only;
the corrected transect belongs in the plume-preservation result.

## 5. Appendix plan

The appendices should make the main claims auditable without becoming a second
Results section. Each appendix must have one explicit job: document a method,
test robustness, expose individual cases, or bound transferability. The main
text should state the conclusion and cite the relevant appendix; the appendix
should carry the diagnostic detail needed to reproduce that conclusion.

Use lettered topical appendices in the manuscript. The existing A1–A13 labels
below are working figure identifiers from `log/TODO_ACCOMPLISH.md`, not the
recommended final AMT appendix structure.

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
  and fitted \(L'\) statistics;
- **Fig. A2:** SG-versus-no-SG comparison, sourced from working figure A5;
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
- **Fig. B2:** target sensitivity for r05/r10/r15, if a compact version can be
  generated without duplicating main-text Fig. 3;
- **Table B1:** cohort inventory and attrition from raw soundings to fitted,
  labeled, training, and evaluation populations;
- **Table B2:** all target parameters and reference-population guards;
- **Table B3:** label-noise ceilings by surface and cloud-distance regime.

This appendix must resolve the currently mixed 116-date/17.8-million and
140-date/21.5-million cohorts by naming the role of each population.

### Appendix C: Model architecture, training, and reproducibility

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

- **Fig. C1:** training and evaluation data-flow diagram;
- **Fig. C2:** fold timeline showing contiguous date blocks;
- **Table C1:** full predictor inventory;
- **Table C2:** hyperparameters and training configuration;
- **Table C3:** fold-level sample sizes and performance;
- **Table C4:** training manifests and zero-overlap verification.

Keep the main text to predictor groups and essential architecture. Individual
features and hyperparameters belong here.

### Appendix D: Cross-validation, baselines, and feature attribution

**Purpose:** show that the selected model and split design are not arbitrary,
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

- **Fig. D1:** random-split inflation relative to date-blocked estimates
  (panel b, the CV-design schematic, exists:
  `manuscript/figures/figD1b_cv_design`,
  `manuscript/scripts/make_cv_design_figure.py`);
- **Fig. D2:** date-blocked skill versus the label-noise ceiling by surface
  and cloud-distance regime (absorbs the skill/ceiling panel formerly planned
  for the former main-text skill figure; that panel is now Appendix-D-only);
- **Fig. D3:** model comparison by surface and cloud-distance stratum;
- **Fig. D4:** feature-ablation changes relative to the full model;
- **Fig. D5:** raw/BC/ML increment relationship;
- **Table D1:** fold-resolved date-blocked metrics (the frozen values quoted
  in §4.3);
- **Table D2:** fold-resolved baseline results;
- **Table D3:** complete ablation results, using the 2026-07-17 retrained
  variants only.

The appendix should preserve the null result: `no_spec` is approximately
TCCON-neutral. Do not select only strata that make the spectral block appear
predictively essential.

### Appendix E: TCCON protocol and validation robustness

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

- **Fig. E1:** 3-by-3 coincidence sensitivity, sourced from working figure A2;
- **Fig. E2:** QF0 and QF1 station-day comparisons, sourced from A3;
- **Fig. E3:** r50 robustness comparison, sourced from A3;
- **Fig. E4:** all station-day before/after panels or a compact station-grouped
  summary;
- **Fig. E5:** random-effects residual forest plot, derived from working A9;
- **Table E1:** station inventory and coincidence counts;
- **Table E2:** complete AK-harmonized metrics;
- **Table E3:** paired Wilcoxon and site-clustered bootstrap results;
- **Table E4:** uncertainty components, \(\tau\), \(I^2\), and coverage.

The main paper should carry the headline and one robustness summary. The full
coincidence matrix and station-level audit belong here.

### Appendix F: Null tests, negative controls, and failure cases

**Purpose:** demonstrate that error reduction is not produced by generic
smoothing and expose conditions where the method does not work.

Include:

- feature-free smoother definitions and all tested windows;
- far-cloud ATom and clear-day ship controls;
- bright-surface, high-latitude, snow, and high-AOD stratification;
- all worsening station-days with AK-harmonized interpretation;
- guard activity and predictive-uncertainty diagnostics.

Planned items:

- **Fig. F1:** smoother null, sourced from working figure A4;
- **Fig. F2:** far-cloud and clear-day control cases;
- **Fig. F3:** failure rates and residuals by environmental driver;
- **Fig. F4:** high-latitude and post-2022 cases, sourced from working A7;
- **Table F1:** all worsening cases and diagnosed cause;
- **Table F2:** performance by bright surface, snow, AOD, latitude, and
  predictive-uncertainty strata.

If the smoother null remains a main figure, retain its full numerical table in
this appendix and avoid duplicating the same plot.

### Appendix G: Case atlases and plume-preservation audit

**Purpose:** allow visual inspection of cloud-mask behavior, spectral response,
and preservation of localized CO2 enhancements.

Include:

- vetted real-cloud land and ocean cases;
- MYD35 false positives;
- visible clouds without a strong XCO2 response;
- all Nassar plume cases, matched controls, and channel-attribution results;
- selection criteria for the showcased Westar case.

Planned items:

- **Figs. G1–G7:** seven case-atlas pages from working figure A1;
- **Fig. G8:** all Nassar transects with identical axes and annotations;
- **Fig. G9:** band-resolved \(\Delta L'\) for plume and cloud-contaminated
  windows;
- **Table G1:** vetted case inventory and category assignments;
- **Table G2:** per-case plume-removal bounds and control-null results;
- **Table G3:** full/no-spec/no-xco2 smoothing attribution.

The main text should show only the strongest mechanism case and the Westar
preservation case. The appendix must show the full set to avoid
case-selection concerns.

### Appendix H: Ocean-reference details

**Purpose:** document the limited but independent ocean validation without
overloading the TCCON section.

Include:

- ATom pseudo-column construction, vertical coverage, prior fill, and AK
  treatment;
- shipborne EM27/SUN processing, scale vintage, and coincidence choices;
- per-date and per-leg results;
- absolute-scale limitations and why the far-cloud cases are interpreted as
  negative controls.

Planned items:

- **Fig. H1:** ATom summary, sourced from working figure A8;
- **Figs. H2–Hn:** per-date ATom panels;
- **Fig. Hn+1:** ship summary and individual cases, sourced from A8;
- **Table H1:** ATom leg inventory and residuals;
- **Table H2:** ship case inventory and residuals.

Do not pool these observations with TCCON into a single headline metric.

### Appendix I: Spectrum-internal sensitivity and deployment evidence

**Purpose:** support the inference-independence claim beyond the architecture
diagram.

Include:

- spec-only near/far-cloud classifier definition and held-out ROC curves;
- sub-pixel spectral-index response below the nominal MODIS pixel scale;
- post-2022 NoMODIS-era application examples;
- a clear distinction between spectral sensitivity and correction skill.

Planned items:

- **Fig. I1:** spec-only classifier ROC, sourced from working figure A11;
- **Fig. I2:** sub-pixel anomaly versus spectral index, sourced from A11;
- **Fig. I3:** selected NoMODIS-era corrected cases, if not already used in
  Appendix F;
- **Table I1:** AUC and calibration by surface and fold.

The AUC values demonstrate cloud-proximity information in the spectra; they do
not establish that the spectral block is required for best TCCON correction.

### Appendix J: Controlled 3-D radiative-transfer demonstration

**Purpose:** provide a causal mechanism test while keeping its limited scope
explicit.

Include the backward-Monte-Carlo 3-D-versus-ICA experiment now tracked as
working figure A13:

- model, band, geometry, cloud optical properties, surface albedo, grid, and
  photon-sampling configuration;
- identical cloud scene under full 3-D transport and ICA;
- refitting with the production spectral estimator;
- \(L'\), variance, and intercept response across the cloud boundary;
- the one-sided shadow-edge decay;
- limitations of a single representative geometry.

Planned items:

- **Fig. J1:** controlled 3-D-versus-ICA comparison;
- **Table J1:** simulation configuration and estimator settings.

State explicitly that this validates qualitative sensitivity of the fitted
features to horizontal photon transport. Direct comparison between fitted
cumulants and tallied photon-path moments, a geometry sweep, and plume
injection remain future work.

### Appendix K: Cross-sensor feasibility

**Purpose:** bound the transfer claim with one existence proof, not imply a
validated cross-sensor correction.

Include the planned TEMPO O2-B example currently tracked as working figure
A12:

- one pre-specified TEMPO granule;
- O2-B spectral window and optical-depth construction;
- in-scene cloud product and cloud-distance calculation;
- maps of \(L'\) and path-length variance;
- response versus in-scene cloud distance;
- differences from the OCO-2 instrument, geometry, sampling, and retrieval.

Planned items:

- **Fig. K1:** TEMPO RGB/cloud field, fitted cumulants, and distance response;
- **Table K1:** OCO-2 and TEMPO inputs required by the spectral fit.

Label this appendix **feasibility demonstration**. Do not include EMIT merely
to broaden the sensor list; its sampling may not provide adequate optical-depth
dynamic range. Do not claim transfer of the trained OCO-2 correction.

### Appendix triage

If AMT length or preparation time requires compression, use this priority:

1. **Required:** A–G. These appendices support the method, validation, and
   safety claims made in the main paper.
2. **Strongly recommended:** H–I. These support ocean corroboration and
   imager-independent inference.
3. **Include only when fully prepared:** J–K. These extend mechanism and
   transfer arguments but should not delay or weaken the core manuscript.

Never include a placeholder or partially documented simulation. If working
figures A12 or A13 are not completed to manuscript standard, retain their
claims as future work and remove the corresponding appendix references.

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
\(X_{\mathrm{CO_2}}\), \(L'\), \(k_1\), and \(k_2\).

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
7. **Notation:** regenerate remaining figures that still use lowercase
   \(l'\) or superseded labels.
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

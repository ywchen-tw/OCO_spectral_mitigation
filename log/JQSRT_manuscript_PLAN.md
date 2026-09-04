# JQSRT Manuscript Plan — progress, changes, to-do

**Created:** 2026-08-31 · **Last updated:** 2026-09-01 (flux-inversion
framing, ATom metric parity, R5 closed)
**Compile state:** main.tex 29 pp / si_main.tex 38 pp, both latexmk exit 0,
no undefined refs or citations. Review triage STARTED: R5 and R7 closed,
R1/R2 deferred by author decision; §9 Conclusions now written (no longer a
stub). Remaining review items untriaged.
**Live manuscript:** `manuscript/JQSRT_draft/` (cas-sc single column; `main.tex`
29 pp + `si_main.tex` 38 pp, both compile with latexmk, no undefined refs).
**Frozen source:** `manuscript/AMT_draft/` — read-only archive of the AMT
version; never edit it. Conversion spec: `manuscript/JQSRT_draft/MIGRATION_PLAN.md`.
The AMT-era planning record remains `log/MANUSCRIPT_FLOW_PLAN.md` (audit trail).

## Structure as built (method-first)

| § | Section | File | Main display items |
|---|---|---|---|
| — | Abstract (INLINE in main.tex — cas-sc verbatim-captures it, cannot be \input) | `main.tex` | — |
| 1 | Introduction | `1_intro.tex` | — |
| 2 | Path-length cumulant observables | `2_cumulant_observables.tex` | figA1, tabA1 |
| 3 | Controlled 3-D RT verification (before flight data; predicts the sign rule) | `3_rt_verification.tex` | figG1, tabG1 |
| 4 | Observations, collocation, anomaly target | `4_data_target.tex` | fig01 |
| 5 | Observed cloud-proximity response (confirms §3 prediction) | `5_phenomenology.tex` | fig03, fig04, fig05 |
| 6 | Application: imager-free per-footprint correction | `6_application.tex` | fig02, fig06, fig08, fig13 (test-set anomaly decay, added 2026-08-31), Tables 1–2 |
| 7 | Cross-sensor demonstration: TEMPO O2-B | `7_tempo.tex` | figH1 |
| 8 | Discussion | `8_discussion.tex` | — |
| 9 | Conclusions (STUB — author writes) | `9_conclusion.tex` | — |
| SI | S1 smoother · S2 plume audit · S3 ocean/controls · S4 cohort/target · S5 model/CV · S6 TCCON protocol | `si_main.tex` + `si_S1..S6` | 19 figures, 22 tables |

No typeset appendix (author decision: SI-only, to keep the article PDF short).

## Decision log

- **2026-08-30** — Journal switch to JQSRT with current materials (no new
  simulation/analysis; M9 residue items deferred unless a reviewer asks).
  Structure: slab §3 BEFORE flight data (demonstrates PPDF theory, predicts
  the albedo-contrast sign rule); TEMPO in MAIN text; smoother/plume/controls
  demoted. Class: cas-sc single column (author choice). Demotion target:
  Supporting Information ONLY, no appendix (author choice — appendix pages
  count toward the article PDF; SI does not).
- **2026-08-31** — Conversion executed by agent fleet per MIGRATION_PLAN.md.
- **2026-08-31 (author feedback round 1):**
  1. Abstract and Conclusions: author will write later (current abstract is a
     draft from the AMT bullet outline; §9 is a stub).
  2. Intro path-length lineage citations: standard set inserted
     (irvine1964, partain2000; stephens2000, heidinger2000, funk2003, min2004,
     scholl2006; oshchepkov2008/2012) — author trims as desired.
  3. Slab and TEMPO stay QUALITATIVE in the abstract — no numbers (they are
     demonstrations, not headline results).
  4. Savitzky–Golay is NEVER mentioned — production is no-SG only; the
     robustness discussion and its TODO were removed from §2.3
     (states plainly: coefficients fitted to measured ln T directly, no
     presmoothing).
  5. Coincidence radius/window sensitivity shown ONCE: the §6.4 prose
     sentence + fig09 panel (SI S6). The 3×3 matrix figure (figD1) and the
     reinstated r50/per-station figures (figD3/figD4) are removed from the
     SI; criteria follow community practice (Das et al. window) and the
     sensitivity data remain available for reviewer questions.
  6. Keywords / CRediT / data availability / title: stay as TODOs (below).
- **2026-08-31 (author feedback round 2 — intro/§2 overlap):** the intro and
  §2.1 told the same path-length lineage twice. Division of duties adopted:
  the Introduction motivates and states the gap; §2 owns the full cited
  lineage. Applied: (1) intro lineage paragraph compressed to its motivating
  core (Davis/EPIC citation and the PPDF definition kept; the seven lineage
  citations now live only in §2.1, min2004 moved there so no citation left
  the paper); (2) final intro roadmap sentence deleted (contributions
  paragraph already carries the section pointers); (4) §2's standalone
  2-sentence opener merged into §2.1's first sentence (present tense; the
  imager-independence restatement removed — partially resolves review R25).
  (3) mission list shortened after wording discussion: "in orbit or in
  development" is the status-neutral catch-all ("planned" would mislabel
  GOSAT-2/GOSAT-GW); list now names GOSAT, OCO-2, GOSAT-GW, MicroCarb,
  CO2M, TANGO with all ten citations retained in the bundle (GOSAT-2,
  OCO-3, TanSat names dropped; OCO-3 still named in the following
  resolution sentence — later merged away by T4). Intro further trimmed
  (GRL-lens pass T1–T6, all author-approved): T1 ML-literature detail cut
  (5/4-variable detail out; Mauceri/Keely one sentence; single closing
  judgment); T2 RT literature one sentence per paper (SHDOM/EaR3T/MYSTIC
  expansions dropped; MCARaTS expansion moved to §3 first use); T3 King
  cloud-fraction sentence cut (Massie 40 %/73 % carries the point; MODIS
  expansion moved to the contributions paragraph; king2013 now uncited —
  entry stays in refs.bib harmlessly); T4 point-source clause cut,
  resolution + decade merged (OCO-3 no longer named in intro); T5
  contributions repetition trimmed to "imager-independent at inference"
  (part of review R25); T6 choppy "not only… It preferentially…" joined,
  "The two lines leave a gap between them" slogan deleted (part of R30).
  Intro now 963 words; §2 no longer reads as a second introduction.
- **2026-08-31 (author feedback round 3 — abstract):** author REWROTE the
  abstract (now in `main.tex:104–126`; no longer the outline-derived draft).
  Decisions fixed in that pass: (1) TEMPO is NOT mentioned in the abstract
  (author preference; §7 demo stands on its own); (2) the "71 of 75" clause
  is DROPPED rather than scoped — resolves R7 by omission (per-metric counts
  live in §6.4 only); (3) slab sentence mirrors §3's hedge ("agree with the
  directly tallied path-length moments where the two are comparable" —
  addresses the abstract half of R5); (4) abbreviations follow the JQSRT
  check (guide: define non-standard abbreviations at first mention in the
  abstract; practice from published JQSRT abstracts: expand the paper's own
  instrument, leave MODIS bare): OCO-2 + TCCON expanded, MODIS/RMSE bare,
  "AK-harmonized" → "averaging-kernel-harmonized", WCO2 → "weak CO2 band";
  (5) terminology: "column-averaged CO2 dry air mole fraction" is the
  canonical term (author ruling) — intro updated to match; (6) closing
  sentence must keep BOTH "physically" and "statistically" (prior work was
  one or the other) — final phrasing under discussion.

- **2026-08-31 (terrain/alt_std analysis, author-commissioned):** ran the
  elevation-variation analysis (sub-footprint DEM roughness `alt_std`) on
  the combined parquet: `bias_sign_conditions.py` extended (alt_std in
  NUMERIC, land only) + new `workspace/alt_std_terrain_analysis.py`
  (confound map, albedo-controlled two-way, far-field feature spread) →
  `results/figures/cld_dist_analysis/{bias_sign_conditions,alt_std_terrain}/`.
  VERDICTS (land QF0 snow-free, r15): no terrain–cloud-distance masquerade
  (Spearman −0.044); near-cloud anomaly essentially flat in alt_std
  (|r| < 0.03, no sign flip; albedo axis dominates the two-way; rough↔dark
  confound r = −0.30); far-field clear-sky anomaly scatter GROWS with
  roughness (0.42→0.64 ppm std across quintiles) while locally referenced
  feature departures stay flat (|r| ≤ 0.02; only a −0.23σ zexp_o2a mean
  shift in the roughest, darkest quintile). Reading: terrain XCO2 error
  flows through channels the features barely sense (psurf pathway) —
  refines the §8 "other factors" paragraph toward an orthogonality claim
  rather than "features sense terrain". Caveat: same-orbit references
  share terrain, so static offsets cancel; the test sees only
  roughness-linked scatter. PART 2 (`alt_std_farfield_capture.py`, author
  question round): (Q1) far-field anomaly MAGNITUDE relates to roughness —
  Spearman r(alt_std, |anom|) = +0.14, surviving albedo control
  (+0.13/+0.14/+0.09 per tercile); mean stays ≤0.04 ppm (reference
  cancellation); effect concentrated at alt_std ≳ 10 m, |anom| 0.32→0.54
  ppm flattest→roughest decile. (Q2) features catch it only PARTIALLY:
  anomaly~4-feature OLS R² is 0.029–0.030 in the four flatter quintiles
  and rises to 0.047 in the roughest (WCO2 k1 + O2A continuum lead), but
  that accounts for only ~6 % of the terrain-ADDED variance
  (Δexplained 0.013 ppm² of Δtotal 0.215 ppm²) — the bulk is
  feature-invisible (psurf-prior pathway). alt_std is itself a DE input,
  so the production correction sees terrain through the tabular channel
  regardless. PART 3 (`alt_features_vs_terrain.py`, features vs
  alt/alt_std/alt_std-over-alt, far-field raw + z forms): RAW O2A path
  statistics fall with surface ELEVATION (Spearman k2 −0.26, k1 −0.20,
  exp-int −0.14; WCO2 ≈ 0, SCO2 weak) — Rayleigh column shrinking with
  altitude, physically expected, and it cancels under local referencing
  (all zk |r| ≤ 0.02). ROUGHNESS: after referencing, the only surviving
  signal is the O2A continuum reflectance darkening (zexp_o2a r = −0.085
  vs both alt_std and rel_alt_std; quintile means +0.03σ → −0.21σ) —
  terrain shading imprints on the continuum like a weak shadow while the
  absorption-derived k statistics stay flat. rel_alt_std adds nothing
  beyond alt_std. PART 4 (QF=1 snow-free rerun, `--qf 1` on both scripts):
  terrain-|anom| link weaker relatively (Spearman +0.07 vs +0.14) because
  the QF1 baseline is noisier everywhere (|anom| 0.46 vs 0.32 ppm on flat
  terrain), but the terrain-ADDED variance is similar in absolute terms
  (std 0.62→0.92 vs 0.42→0.63 ppm) and the feature capture is again ~6 %
  of it (R² 0.027→0.039) — the 6 % figure is population-robust. Raw
  elevation Rayleigh response replicates (o2a k1 −0.21 / k2 −0.23).
  Distinct QF1 signature: the roughest quintile carries a coordinated
  ALL-BAND positive k1/k2 bump (WCO2-led, +0.09σ) with continuum
  darkening (zexp −0.11σ) — the in-scene scattering-contamination
  fingerprint (§5's QF1 story) appearing in rough terrain, i.e. flagged
  mountain scenes plausibly contain real unscreened scattering. Mean QF1
  far-field anomaly is slightly negative except the roughest decile,
  which flips positive (+0.005/+0.023 median). SCOPING (author decision):
  terrain-induced biases declared OUT OF SCOPE; manuscript carries only a
  brief QF0-vs-QF1 selectivity passage; the scatter-inflation and
  6 %-capture numbers stay on file for reviewer questions (same policy as
  coincidence sensitivity). APPLIED 2026-08-31: three author-approved
  sentences appended to the §8 "other factors" paragraph
  (`8_discussion.tex:18` — QF0 flat across roughness quintiles / roughest
  QF1 shows the all-band in-scene-scattering signature of
  Sect.~sec:phenom / terrain biases out of scope), closing with
  \citep{kiel2019bias_correction_surface_pressure} (entry completed with
  the full author list; was an "and others" stub). §5 condition-list
  clause NOT added (author chose the brief mention only). main.tex
  rebuilt, exit 0, citation resolves.

- **2026-09-01 (flux-inversion framing + ATom metric parity + R5):** three
  threads, all applied and rebuilt.
  1. **Liu et al. (2026, GRL 53, e2025GL119838)** — OCO-2 seasonally dependent
     sampling over northern tropical Africa. OSSEs (CMS-Flux + TM5-4DVar) show
     the inversion improves every month (monthly RMS 0.26→0.13 GtC/mo) yet
     DEGRADES the year (annual error 0.21→0.36 GtC; posterior uncertainty
     0.37 > prior 0.28), because OCO-2 samples NTA least in the wet growing
     season (JJA). Uniform (Jan-repeated) sampling removes the bias
     (−0.06±0.15 GtC/yr) with ~15 % FEWER observations — the problem is the
     temporal distribution, not the volume. MIP v10 confirms across 10 systems
     (weak-prior-seasonality group +0.99±0.19 GtC/yr). Their only observational
     remedy is future finer-footprint missions with a light-path proxy
     retrieval (frankenberg2024/2025, N2O anchor).
  2. **QF filtering in flux inversions — VERIFIED QF = 0 ONLY.** Baker et al.
     (2022, GMD 15, 649) builds the MIP 10 s product from retrievals "declared
     'good' by the OCO-2 quality screening criteria"; Byrne et al. (2023, ESSD
     15, 963) adds a SECOND near-cloud screen — "only 10 s spans with 10 or
     more good quality retrievals were used (sparser data being thought to be
     more prone to cloud-related biases)"; GONGGA (Jin et al. 2024, ESSD 16,
     2857) independently uses QF = 0 only. Liu §2.3.1 states all MIP systems
     "assimilate an identical set of observations", so one sentence covers all
     ten. Baker's product spans Sept 2014–Oct 2020; MIP LNLG runs 2015–2020;
     Liu's OSSE is 2016 only. This is the citable basis for the intro claim.
  3. **Flux-inversion test declared OUT OF SCOPE (feasibility, measured).**
     A corrected product over the assimilated record needs ~25 TB of downloads
     (L1B 664 MB/granule × ~14/day × 2191 days) plus a full cumulant refit over
     ~30,700 orbits; the `no_spec` model needs no L1B and drops it to ~4.8 TB,
     still heavy. The binding constraint is that an inversion needs a
     CONTIGUOUS year and the whole record is date-sampled (116 train / 96
     held-out). NOTE for any future attempt: the OSSE route needs NO corrected
     data at all — impose the measured bias-vs-cloud-distance curve on Liu's
     pseudo-obs (their OSSE has random noise only, no retrieval bias, so it is
     blind to this effect by construction), and simulate QF = 1 recovery by
     expanding the sampling mask using Lite-file QF = 1 locations (~150 GB)
     with the post-correction error statistics as the noise model. §8 was
     deliberately NOT edited (author decision) — "impact on flux inversions
     remains to be tested" stands.
  4. **Frankenberg framing CORRECTED before it reached the paper.** A drafted
     §8 sentence claiming the finer-footprint route "increases exposure" to 3D
     cloud bias was WRONG and dropped: frankenberg2024 states the proxy
     approach is "robust against … the 3D effects of nearby clouds", and
     frankenberg2025 (GRL, doi:10.1029/2024GL114131) develops the N2O
     light-path proxy. If that literature is ever cited, the defensible axis is
     retrospective (existing record) vs prospective (new hardware), NOT
     efficacy.

## Changes applied post-migration (2026-08-31)

- Abstract inlined into `main.tex` (cas-sc verbatim capture makes
  `\input` impossible there); `0_abstract.tex` removed; README updated.
- `tables/tabG1_slab_config.tex` (JQSRT copy): caption refs retargeted
  `app:rt_demo`→`sec:rtverify`, `sec:methods_photon_path`→`sec:cumulants`.
  NOTE: the AMT-side generator `workspace/rt_slab_sim/make_table_g1.py`
  still emits the old refs — update it if the table is ever regenerated.
- SG mentions scrubbed (§2.3), per decision 4.
- SI S6: coincidence subsection + figD1 + figD3 + figD4 removed, per
  decision 5 (SI 40 → 38 pp).
- ATom statistic mislabel FIXED (found 2026-08-31 while adding RMSE at
  author request): 0.53→0.45 ppm is the MEAN |residual| over the 14
  near-cloud legs (cld_med ≤ 10 km; excluded: 2× 2017-10-09 far-cloud
  controls + 2017-10-27), NOT the median (median is 0.43→0.37) — the
  mislabel originated in MANUSCRIPT_FLOW_PLAN.md §4.6 draft text and was
  carried into both documents. Fixed "median"→"mean" in
  `6_application.tex` (§6.4 sentence, also rescoped "across 17 legs
  (14 near-cloud)" → "across the 14 near-cloud legs (of 17)" and ADDED
  leg RMSE 0.72→0.59 ppm) and in `si_S3_ocean_controls.tex:29`.
  Verified from `atom_pseudo_column_results.csv` (foldpca model dir);
  spread ±0.72→±0.58 and mean bias +0.19→+0.20 confirmed on the same 14
  legs. Both docs rebuilt exit 0.
- "skill" → "performance" sweep (author request 2026-08-31): §6.3 title now
  "Performance and attribution"; §6 intro "the predictive performance of the
  correction and its attribution to predictor groups"; ablation ¶ "attributes
  that performance"; §8 "supplies most of the predictive performance"; SI S5
  "the achieved ocean \Rsq" (names the metric instead). Internal labels
  (`sec:app-skill`, `app-fig:sfc_skill_ceiling`), the figC4 filename, and
  comment lines deliberately unchanged (non-displaying; renaming risks
  broken refs). Both docs rebuilt exit 0.
- NEW DISPLAY ITEM (author-commissioned 2026-08-31): held-out test-set
  anomaly-decay figure added to §6.3 as `fig:test-anomaly-decay`
  (`figures/fig13_test_anomaly_decay.png|.pdf`, copied from
  `results/figures/cld_dist_analysis/test_set_anomaly_decay/`, generator
  `workspace/test_set_anomaly_decay.py`, 96 held-out dates 2014–2021,
  before/after DE, both panels' anomalies recomputed identically). New
  paragraph after the §6.3 model-selection paragraph: near-cloud flattening
  (ocean ≤5 km mean −0.34→−0.04 ppm; innermost land bin −1.15→−0.42 ppm),
  far-field IQR narrowing (ocean 0.41→0.29 / land 0.69→0.47 ppm at
  20–30 km, NOT by construction — the honest far-field claim), +0.7 M land
  labels passing the σ ≤ 1 ppm stability guard (by-elimination reasoning),
  and a link sentence to the §8 non-cloud-structure (terrain) passage.
  AUTHOR DIRECTIVE: no "negative control" language for this figure (the
  far-field mean is zero by construction in both panels). Land ≤15 km MEAN
  deliberately not quoted (two-sided structure makes it uninformative).
  main.tex rebuilt: exit 0, 28 pp, refs resolve.
  QF-SPLIT ADDENDUM (author round, 2026-08-31): internal-only QF=0/QF=1
  split of the same caches via NEW `workspace/test_anomaly_decay_qf_split.py`
  (`internal_test_anomaly_decay_qf.png` + `..._binstats_qf.csv` in the
  test_set_anomaly_decay dir; NOT a manuscript display item — numbers only,
  author directive). Approved sentences appended to the fig13 paragraph,
  MEDIAN-LED because §5's opposite-sign claim rests on the median (author
  rationale): pooled near-window medians ocean QF0 −0.08→−0.03 / QF1
  −0.43→−0.05, land QF0 +0.08→+0.03; QF1 land median ~unchanged (+0.05)
  while its mean −0.15→+0.02 = correction removes the negative tail §5
  attributes to flagged scenes; label recovery mostly QF1 (+0.5 of
  +0.7 M). NOTE: QF0 ∩ snow is EMPTY on the test dates (snow ⇒ QF1), so
  QF0 numbers are automatically snow-free. Rebuilt: exit 0, 28 pp.
- Intro ¶1 terminology aligned with the author abstract (round 3):
  "dry-air column-average CO2 mixing ratio" → "column-averaged CO2 dry air
  mole fraction" (`1_intro.tex:5`). The other two "mixing ratio" uses are
  different quantities and stand: in-situ ATom profiles
  (`4_data_target.tex:17`) and prior trace-gas profiles
  (`8_discussion.tex:96`).
- Intro lineage citations inserted, per decision 2; plus Davis, Yang &
  Marshak (2022, Front. Remote Sens. 3, 796273 — EPIC/DSCOVR differential
  O2-absorption) added at author request (2026-08-31) as the space-based
  member of the lineage paragraph (`davis2022epic_do2as` in refs.bib).
- `\centering` added to 5 figure environments (fig01, fig03, fig04,
  figC1, figC2) — fixes off-center figures.
- natbib `longnamesfirst` option removed (main.tex + si_main.tex) — it was
  spelling out full author lists at first citation ("Kuze, Suto, Nakajima
  and Hamazaki, 2009"); all in-text citations now abbreviate ("Kuze et al.,
  2009"). Not a JQSRT requirement — was our preamble artifact.
- Notation macros converted from trailing-space form to
  `\ensuremath{...}\xspace` (16 per file, main.tex + si_main.tex;
  xspace loaded) — removes the stray space before punctuation
  ("X_CO2 ," → "X_CO2,"); verified by glyph-geometry comparison
  (22 gaps closed, 0 opened). Two call sites needed `\macro{} -`
  (xspace eats the space before a hyphen used as a minus sign):
  `8_discussion.tex:30`, `si_S3_ocean_controls.tex:30` — in-file
  comments mark the `{}` as load-bearing. `\absbias` re-set as
  `\ensuremath{|\text{bias}|}` to keep upright "bias".

## Changes applied 2026-09-01

- **Intro — flux-inversion sampling paragraph** (`1_intro.tex:7`). The closing
  sentence ("Quality filtering therefore not only reduces the sample size
  but…") replaced by three sentences: filtering removes near-cloud
  observations preferentially rather than at random; the ten v10 MIP systems
  assimilate an identical set of 10 s averages built only from QF = 0
  soundings, with spans of fewer than ten good retrievals also discarded
  because sparse retrievals are more prone to cloud-related biases
  (`baker2022…`, `byrne2023…`); losses concentrate where cloud is persistent
  and, because cloud varies seasonally, fall unevenly through the year
  (`liu2026…`). Adds the TEMPORAL axis the intro previously lacked (it said
  "spatial sampling biases" only) and retires a "not only … but" construction
  flagged by R30.
- **Intro — QF = 1 goal sentence** (`1_intro.tex:27`, end of the contributions
  paragraph): "A goal of this work is to make the QF = 1 soundings that
  current flux inversions discard reliable enough to be considered for
  assimilation, so that part of the seasonally uneven sampling loss can be
  recovered." Phrased as an AIM, not an achievement, per the §8 stub rule that
  "usable" is not earned without an acceptance criterion and an inversion
  test. Gives the §8 recovery paragraph an antecedent it previously lacked.
- **refs.bib** — three entries appended with COMPLETE author lists (no new
  "and others" stubs): `baker2022oco2_10s_error_correlation`,
  `byrne2023oco2_mip_national_budgets` (57 authors),
  `liu2026oco2_seasonal_sampling_nta`.
- **ATom metric parity (author decision: option B).** ROOT CAUSE: a leg is the
  ATom analogue of a TCCON station-day (one pseudo-column reference, many
  footprints), so every TCCON metric has a counterpart — but the chain
  reported only leg-level statistics. The published 0.53→0.45 is the mean
  |bias| analogue and 0.72→0.59 is the **RMS bias** analogue, NOT footprint
  RMSE; a draft conclusion had mislabeled the latter as "mean footprint
  RMSE", which would have invited direct comparison with the TCCON 2.67→1.20.
  Footprint-level metrics are recoverable exactly from the stored within-leg
  scatter via RMSE² = resid² + sd² (verified `oco_bc_sd = sub.xco2_bc.std()`).
  - **Code:** `tccon_parallel_metrics()` + `write_tccon_parallel_metrics()`
    added to `workspace/ATom_analysis/atom_pseudo_column.py`, wired into
    `main()`; emits `atom_metrics_tccon_parallel.csv`. Generated for the
    production DE dir; reproduces 0.53→0.45 and 0.72→0.59 exactly.
  - **Numbers (near-cloud n=14 / all n=17 / far-cloud n=3):** mean |bias|
    0.53→0.45 / 0.49→0.43 / 0.31→0.34; RMS bias 0.72→0.59 / 0.67→0.56 /
    0.31→0.34; leg mean per-footprint RMSE 0.84→0.56 / 0.78→0.54 /
    0.52→0.42; pooled footprint RMSE 0.73→0.48 / 0.66→0.47 / 0.53→0.47.
  - **§6.4** (`6_application.tex:383`): 14 near-cloud legs only; "mean
    |residual|" → `\absbias`, "leg RMSE" → "RMS bias", and the leg mean
    per-footprint RMSE (0.84→0.56) added so the sentence mirrors the TCCON
    one.
  - **SI S3**: same rename; new paragraph reporting 14 / 17 / 3 with the
    scoping shown not to be load-bearing (across all 17 legs 0.49→0.43 and
    0.78→0.54; the 3 far-cloud legs move ≤0.03 ppm, as a negative control
    should) — pre-empts an R6-style selection objection. New table
    `tables/tabD4_atom_metric_groups.tex` (`tab:atom-metric-groups`),
    GENERATED from the metrics CSV, not hand-typed; `make_appendix_tables.py`
    lives in the frozen AMT_draft and was deliberately left alone.
  - **SI S3 framing CORRECTED**: "the near-cloud improvement is a scatter
    reduction, not an offset shift" undersold the result and contradicted §8.
    Now: the improvement comes from reduced scatter rather than a shift in the
    mean offset, and since the per-footprint correction targets the relative
    near-cloud anomaly while the operational correction anchors the absolute
    scale, an unchanged offset is the EXPECTED behavior, not a shortfall.
- **§3 R5 fixes** — see the R5 entry in the review list below for the full
  record (both halves closed; all three r values removed from `:35`).
- **§9 Conclusions** — author-drafted over several rounds; applied on request:
  the 1D sentence → "while the same scene under the independent-pixel
  approximation leaves them flat outside the cloud"; "single-footprint-level
  MLP-based ensemble correction models" → "per-surface deep ensembles of
  multilayer perceptrons that correct one footprint at a time"; "mean absolute
  bias" → `\absbias` for parity with the ATom sentence. Author separately
  fixed ppm spacing, adopted "a check on whether real enhancements survive the
  correction", added TEMPO to the closing sentence, and recast the sign
  sentence away from "opposite sign of the median".
  **§9 CLOSED 2026-09-01.** The sign sentence went through three rounds and
  landed at "This difference **is consistent with** the negative \xcobc
  anomaly over ocean and the two-sided \xcobc anomaly over land" — from
  "explains" (the R1/R2 claim stated more strongly than §5 supports) via
  "supports" (right hedge, wrong verb — an anomaly is an observation and
  cannot be supported) to the standard form. Also fixed: "spread … anomaly
  over land" → "two-sided", `\xco`→`\xcobc` so the same quantity keeps one
  macro, the missing comma after "TEMPO \ce{O2} B-band", "utilize"→"use"
  (×2, plain register), and the "validation … against TCCON data" repeat at
  the §9¶2/¶3 boundary trimmed to "The comparison against TCCON shows".
  Every number in §9 verified against §6.4 as edited: 1.26→0.82, 2.67→1.20,
  75 station-days; 0.53→0.45, 0.84→0.56, 14 near-cloud legs; 116 training
  dates, 2014–2021 held out. TEMPO now named in the closing sentence, so §7
  is no longer the one main-text section absent from the conclusion.
  Conclusions are no longer a stub — remove from the TO-DO when re-read.

## TO-DO

- [x] **Abstract** — author-written 2026-08-31 (round 3; see decision log).
  Only the closing sentence ("physically and statistically grounded
  candidate") awaits final phrasing. NOTE: R8 still targets the abstract's
  "preserves the tested plume enhancements" — open for triage.
- [x] **Conclusions (§9)** — WRITTEN 2026-09-01 (author, over three rounds;
  see the §9 entry under Changes applied 2026-09-01). Covers observable →
  slab verification → surface-dependent behavior → model → TCCON + ATom
  validation → transferability (TEMPO named). The `TODO(stub)` comment block
  at the top of `9_conclusion.tex` is now stale and can be deleted.
- [ ] **Title** — draft marked `% DRAFT TITLE — for author approval` in
  main.tex; sync `si_main.tex` title when final.
- [ ] **Keywords** — placeholder list in main.tex frontmatter.
- [ ] **CRediT author statement** — placeholder in main.tex back matter.
- [ ] **Data availability statement** — TODO in main.tex back matter.
- [ ] §2.4 convergence-range caveat: only source is
  `AMT_draft/tex/EQUIVALENCE_THEOREM_FITTING_REPORT.tex` — author confirm
  keep/delete (TODO(source) in file).
- [ ] SI S4 figB5 (footprint-area robustness): no approved prose exists
  (the drafted paragraph is `log/MANUSCRIPT_REVIEW_SUGGESTIONS_2026-07-28.md`
  §10.2, never approved) — approve prose or leave pointer-only; caption's
  0.2–10 km² QC window needs source verification (TODO markers in file).
- [ ] Alias-`\label` cleanup at the final pass (stacked compatibility labels
  under §2/§4/§5/§6 headings; delete once every `\ref` uses canonical keys).
- [ ] Bibliography: cas-model2-names (author–year) is fine for submission;
  switch to JQSRT numeric style at acceptance (one-line change).
- [ ] Bibliography: 41 entries still carry stub author lists
  ("... and others") — complete before submission (kiel2019 done
  2026-08-31; renders as "et al." either way, but the reference list
  should carry real author lists).
- [ ] Intro `%% TODO(verify)` comments (1_intro.tex): the slab-verification
  and TEMPO contribution sentences had no AMT-intro counterpart — confirm
  wording (both kept number-free by decision 3).
- [ ] OPTIONAL (offered, unanswered): trim the 3 mission-bundle citations
  whose missions are no longer named in ¶1 (GOSAT-2/imasu2023,
  OCO-3/eldering2019, TanSat/liu2018) — currently kept since "including"
  is non-exhaustive; OCO-3's cite could move to no sentence now (OCO-3
  name left the intro in T4).
- [ ] Review-comment triage (below).

## Review comments (GRL-style structured review, adapted to JQSRT; length rules ignored)

Recorded 2026-08-31 from the referee-style review pass (all 15 tex files +
tables + compile logs read). Status: ALL OPEN — triage before author edits.
Line numbers are approximate anchors, not exact.

### Major (substance — a referee would raise these)

**Triage status 2026-09-01:** R5 and R7 closed (see entries). **R1 and R2 are
DEFERRED by author decision** — the §5 sign bridge is left as written for now.
The exposure they created in the conclusion has been removed independently
(§9 now says "is consistent with", not "explains"), so nothing downstream
overstates §5; the bridge itself is the remaining work. All other items open
and untriaged.

- [ ] (DEFERRED 2026-09-01) **R1. §3→§5 sign bridge does not close as written**
  (`5_phenomenology.tex:53`, `3_rt_verification.tex:57`, `8_discussion.tex:14`).
  The slab's shadow-band ⟨l′⟩ FALLS over the dark surface (0.78→0.41) and
  RISES over the bright one (0.99→1.17), while §5's WCO2 Δ⟨l′⟩ is POSITIVE
  over dark ocean (+0.09σ) and NEGATIVE over bright barren (−1.29σ) — read
  side by side the polarities look contradictory. State explicitly that the
  slab predicts the EXISTENCE of a reversal along the albedo axis, not its
  polarity, and name why the two are not directly comparable (O2A vs WCO2
  band; Lambertian nadir vs ocean glint; shadow-band-only vs 0–5 km window
  mixing shadowed and brightened footprints). **Top priority.**
- [ ] (DEFERRED 2026-09-01) **R2. "Sign of the albedo contrast" is stated against a quantity that
  never changes sign** (`5_phenomenology.tex:53`): a COT-10 cloud is brighter
  than every surface on the axis, so the contrast changes magnitude, not
  sign. Recast the rule against a threshold surface reflectance (where the
  slab locates the crossing).
- [ ] **R3. Convergence range of the order-7 expansion at τ ≈ 0.03–8**
  (`2_cumulant_observables.tex:145,378`): the caveat sits in §2.4 but the
  first referee test lands on the estimator subsection. Move it there, state
  per-band τ ranges, and justify the truncated polynomial as an empirical
  basis whose coefficients carry cumulant meaning in the small-τ limit,
  citing §3 closure as evidence k₁ tracks the true mean.
- [ ] **R4. Solar-normalization internal inconsistency** — Eq. (7) vs
  Eqs. (13)–(14) differ by cos²θ (`2_cumulant_observables.tex:340–364` vs
  `:117–122`). Fix the geometric factor and note a per-sounding constant is
  absorbed in the intercept so fitted cumulants are unaffected. VERIFY the
  algebra before editing — if real, this is a must-fix.
- [x] **R5. Closure reported as correlation only** (`3_rt_verification.tex:31,35`
  + abstract): r values show co-variation, not moment recovery; "exact null"
  vs 5 % var(l′) residual is self-contradicting. Report slope/offset (or
  bias) and replace "exact null" with the quantitative residual level.
  **RESOLVED 2026-09-01, both halves.** (a) ":31" — "the exact null" →
  "supplies the null … at the percent level … flat to within 0.9 % of the 3D
  dynamic range in ⟨l′⟩ and 5 % in var(l′)"; numbers verified against
  `results/rt_slab_sim/closure_stats.json` (dark 0.87 %/4.58 %, the worst
  case; bright 0.17 %/0.0 %). The abstract half was already handled in the
  round-3 rewrite. (b) ":35" — the fix was NOT "add slope beside r" but
  DROP r, because two diagnostics on `slab_fit.h5` showed r cannot support
  the claim: pooled r is inflated by shadow/far-field cluster structure
  (0.9991 pooled vs 0.9239 within the non-shadow group, between/within
  spread ratio 3.7), and regressing fitted ⟨l′⟩ on the tallied mean path
  gives slope 7.6 (dark) / 5.4 (bright), i.e. r = 0.999 coexists with a
  ~7× scale mismatch. Paragraph rewritten to claim only what is unit-free
  and verified: same SIGN on both surfaces (dark 0.486 vs 0.802 far-field
  fitted, 5.4892 vs 5.5311 tallied; bright 1.136 vs 1.017 and 5.5799 vs
  5.5581) — so the surface-dependent reversal is in the tallied ground
  truth, not just the estimator, which is a STRONGER claim than the r
  values supported — same spatial EXTENT (fitted 15.8–21.2 km vs tallied
  15.8–21.2 km dark; 15.2–21.2 vs 15.8–21.2 bright), and an explicit
  statement that amplitudes are not comparable (geometric vs
  absorption-weighted tally). No other section cited the r values (checked).
  OPEN, now moot for the text: what reference path k₁ is normalized to —
  needed only if an amplitude closure is ever attempted.
- [ ] **R6. Manual TCCON station-day selection unbounded**
  (`6_application.tex:226`): add one sentence that the selection was fixed on
  geographic/coverage grounds before any correction ran (uninformed by model
  performance) and state which way coastal over-sampling could bias the
  aggregate.
- [x] **R7. Abstract "71 of 75" reads as covering both metrics**
  (`main.tex` abstract vs `6_application.tex:234–235`): 71/75 is footprint
  RMSE; station-day mean bias is 46/75. RESOLVED 2026-08-31 by omission:
  the author abstract drops the clause entirely; per-metric counts appear
  only in §6.4, where each is attached to its metric.
- [ ] **R8. "Preserves the tested plume enhancements" overstates**
  (abstract; `6_application.tex:337`; `tabF2`): the two tabulated plume
  windows lose roughly half their contrast (Kozienice +1.94→Δμ +0.87;
  Taean +0.13→Δμ +0.08), and the Westar case quoted in the main text is
  grouped as a clear-sky control in the SI figure. Reword to what was
  tested: plant windows lose no more local contrast than matched plume-free
  controls; spectral-channel worst-case bound 0.21 ppm.
- [ ] **R9. Outlier-screen denominator** (`4_data_target.tex:68`): "192 of the
  17.8 million production labels" — 17.8 M is the fitted-sounding count; the
  labeled population is 7.85 M + 3.84 M = 11.7 M. Fix the denominator or
  name the population screened.
- [ ] **R10. Missing one-sentence mechanism for why a within-orbit anomaly
  correction can move station-day MEAN bias** (`6_application.tex:110–116,
  328–336`): the predicted correction has nonzero overpass mean whenever the
  overpass is dominated by near-cloud footprints — exactly the regime the
  reference set excludes. This is the crux of why TCCON improves and the
  smoother fails; say it explicitly.
- [ ] **R11. ILS gap in the slab claim** (`3_rt_verification.tex:55`): the
  verification validates the estimator in the ILS-free limit; Eq. (8)'s
  ILS-averaging approximation is NOT tested by the experiment — say so
  plainly instead of "regardless of how the optical-depth axis is sampled".
- [ ] **R12. OCO-2 spectral windows/channel counts/resolving power never
  given** (`2_cumulant_observables.tex:253–302`) while §7 gives them for
  TEMPO. Add to §2.2 or Table 1.
- [ ] **R13. TEMPO "same qualitative near-cloud behavior" not checkable**
  (`7_tempo.tex:25–30`): no OCO-2 ocean ⟨l′⟩-vs-distance curve exists in the
  main text, and the comparison crosses bands and decay scales. Add the
  counterpart curve or narrow to the sign of the gradient, naming band and
  scale differences.
- [ ] **R14. Plume completeness accounting does not add up**
  (`si_S2_plume.tex:33` + `tabF2:20`): six audited + five not-testable = 11
  vs "ten usable overpasses of eight plants", with nine plants named.
  Recount and reconcile.
- [ ] **R15. Slab far field undefined under periodic boundaries**
  (`3_rt_verification.tex:13`): 32 km cyclic domain, 5 km cloud → farthest
  clear column ~13 km from the periodic image; define the far-field columns
  and why wrap-around contamination is negligible.

### Minor

- [ ] **R16.** QF "no mechanical connection" too strong (`5_phenomenology.tex:42`)
  — soften to "no direct dependence" + acknowledge indirect selection
  (strengthens the conservative-estimate argument).
- [ ] **R17.** Barren-flip paragraph reads post-hoc (`5_phenomenology.tex:59`)
  — anchor in the falsifiable prediction (QF1−QF0 shift positive in EVERY
  class), which the SI figure already answers.
- [ ] **R18.** l′ < 1 in the slab vs "enhancement" wording
  (`2_cumulant_observables.tex:77–80`) — add clause: l′ < 1 when
  high-altitude backscatter dominates over surface reflection.
- [ ] **R19.** Gamma-model block (8 equations) lacks a purpose sentence up
  front (`2_cumulant_observables.tex:163–250`) — say it fixes the
  coefficient convention and supplies κ as a QC shape diagnostic.
- [ ] **R20.** "costs 0.8–1.2 ppm" ambiguous vs later +0.79/+1.22
  (`6_application.tex:172`) — say whether the range spans slices or arms.
- [ ] **R21.** Ocean near-cloud QF claim not verifiable from Table 2
  (`6_application.tex:155`) — add the row or reword to shown slices.
- [ ] **R22.** Drift-era validation (1.24→0.67 ppm, 20/21) appears only in
  §8 (`8_discussion.tex:80`) — forward-reference from §6.4 or move numbers
  there.
- [ ] **R23.** Far window 20–50 km mixes measured and censored (d=50 km)
  distances (`5_phenomenology.tex:39`) — state it where the window is
  defined.
- [ ] **R24.** Abstract "…used only for diagnosis and target construction,
  SO the correction is applied one footprint at a time" is a non sequitur —
  "and", or state the design choice directly. (Fold into abstract rewrite.)
- [ ] **R25.** Imager-independence asserted ~6× in near-identical wording —
  keep §1, §6.1, §8.3; drop restatements.
- [ ] **R26.** Estimator text vs tabA1 disagree ("singular-value" vs "exact
  linear least squares"; edge-channel mask only in text); no goodness-of-fit
  summary or BVLS-fallback rate anywhere — pre-empt the request.
- [ ] **R27.** Three symbols for the convolved solar term; and state that
  the "direct geometrical slant path L" is the two-way path implied by
  Eq. (6)'s airmass.

### Language / tone (author's plain-declarative, native register)

- [ ] **R28.** `8_discussion.tex:88` — "We demonstrate the ability of the DE
  model correction compared to observation and preserve the \xco
  enhancement" is ungrammatical; rewrite as one declarative sentence.
- [ ] **R29.** `8_discussion.tex:16` — paired tagline ("the channel that can
  erase… is not the channel that judges it" / "…performs the correction,
  while … independently audits…") is AI-sounding; keep one plain statement,
  delete the aphorism. Also "contribute none of it essentially" word order;
  "In other words".
- [ ] **R30.** Recurring tics across files: triadic openers ("Three results
  stand out" ×3 variants), slogan fragments ("…in causal isolation",
  "Improvement is not the same as agreement", "The two lines leave a gap
  between them", "narrow and, more importantly, visible from the inside"),
  "not only … but also" ×2 + the choppy "…not only a loss of sample size.
  It preferentially removes…", boilerplate "show high potential", mixed
  past/present tense through §2, and three consecutive 8–18-line sentences
  closing §6.4. Vary openers, convert slogans to plain statements, settle
  on present tense in §2, split the §6.4 closers one result per sentence.

### Review verdict (recorded)

Structure holds: §2→§3→§4/§5→§6→§7 each delivers what the previous section
promises, and the SI is well organized around one-sentence main-text claims.
NOT yet referee-ready. Top three priorities: (1) reconcile/qualify the §3→§5
sign argument (R1, R2); (2) pre-empt the two estimator questions an RT
referee asks first — convergence range (R3) and the cosθ check (R4) — and
report slab closure as slope+offset, not correlation (R5); (3) align the
abstract's plume and 71-of-75 claims with what the tables support (R7, R8)
and bound the manual TCCON selection (R6).

# Manuscript Flow Plan — change history

**Split out of `log/MANUSCRIPT_FLOW_PLAN.md` on 2026-07-26**, which had grown
to 580 lines of changelog in front of 2,000 lines of plan. This file is the
complete dated record of manuscript-structure decisions; the flow plan itself
now states only what the manuscript currently *is*.

Read this file when you need to know **why** a decision was made, whether an
option was already considered and rejected, or what a superseded numbering
scheme was. Entries are chronological and use the figure/appendix numbering
in force **at the time they were written** — several renumberings have
happened since (notably the 2026-07-22n appendix consolidation and the
2026-07-26 Results restructure), so never take a figure or section number
from an old entry without checking it against the current flow plan.

---

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
**Updated:** 2026-07-22r — Fig. 5 regenerated in the locked style from
`shadow_brightening_stats.csv` (new `make_shadow_brightening_figure.py`;
legend moved below the panels — it covered the O2A curves — and the
pre-restyle/full-parquet caveat is cleared); ocean companion written to
`internal_shadow_brightening_ocean`. Panel (d) label uses
$\Delta X_{\mathrm{CO2}}^{\mathrm{B11}}$ (user request — matches the
supplement.tex \xcobc macro family).  
**Updated:** 2026-07-22s — product-resolved XCO2 notation adopted
paper-wide (B11/raw/ATom/ship superscripts + ΔX for the anomaly; §6
table, plot_style constants XCO2_BC/RAW/ATOM/SHIP_LABEL +
DXCO2_BC_LABEL); fig03, fig03alt, figB2, fig05(+ocean) regenerated with
ΔX_CO2^B11 axes; copied report figures adopt it at their next
re-render.  
**Updated:** 2026-07-22t — TabM and Structured DCN removed from §4.3 /
Fig. 6 (main-text baselines = DE/XGBoost/Ridge, matching Table 1; the
5-model set stays in Appendix C); fig06 regenerated, uncorrected row
relabeled X_CO2^B11.  
**Updated:** 2026-07-22u — Figs. 7a/7b SPLIT into separate floats (user
decision): Fig. 7 = TCCON dumbbell, Fig. 8 = significance/robustness;
downstream renumber 8→9 (smoother), 9→10 (ATom+ship), 10→11 (plume);
main text now ELEVEN figures. Files and generator stems renamed;
captions split; Fig. 7 panel assembly removed. Changelog entries before
this one use the OLD numbers.  
**Updated:** 2026-07-22v — §4.3 gains the spec-emphasis synthesis
draft paragraph (skill-versus-trust: xco2 channel operationally
load-bearing, cumulants carry mechanism/safety/imager-independence;
sourced from SPEC_EMPHASIS_STATUS_2026-07-08.md with the 2026-07-17
fold-PCA numbers).  
**Updated:** 2026-07-22w — Fig. 8a metric-definition LaTeX block added
under the Fig. 8 caption (b_s / R_s, three aggregates, site-clustered
bootstrap, Wilcoxon cross-check). Verified against
tccon_comparison_report._significance: "RMS bias" is the quadratic
aggregate of station-day mean biases — NOT a footprint RMSE — so the
row label stays "Δ RMS bias".  
**Updated:** 2026-07-22x — §4.3 headline extended to THREE sentences:
withheld-fold model ordering (DE ≥ XGB ≫ ridge) agrees with the TCCON
ordering, covering each view's weakness (target-construction vs
validation-chain artifacts), with the tail-divergence clause (fold gap
~0.02–0.04 ppm vs 0.30 ppm TCCON near-cloud land). Matching sentence in
the §4.3 draft results text; Table C5 now GENERATED as
`manuscript/tables/tabC5_cv_model_comparison.tex` by
`make_appendix_c_figures.py` (same kfold_agg parse as Fig. C5).  
**Updated:** 2026-07-22y — TCCON-protocol flow fix (user): the
comparison protocol (sample/coincidence, AK-harmonised reference,
station-day unit, metric + bootstrap definitions incl. the former
Fig. 8a LaTeX block) consolidated into Methods §3.5; §4.3 opens its
baseline table with a Sect. 3.5 pointer; §4.4 item 1 reduced to a
one-sentence recap — no Results section now depends on a later one.  
**Updated:** 2026-07-22z — Fig. 10 regenerated with continuous panel
letters (ATom a/b, ship c/d; duplicate-letter clash resolved),
suptitles removed (numbers moved to the §4.6 draft results text —
including two honest wrinkles: ATom near-cloud signed mean bias
unchanged +0.19→+0.20 ppm, the gain is scatter/|residual|; ship
all-case mean offset +0.99→+1.18 ppm, reference-scale dominated), B11
notation adopted. New `make_ocean_validation_figure.py` drives the
patched producers (panel_offset/suptitle/out_pdf kwargs) on the
production CSVs.  
**Updated:** 2026-07-22aa — Fig. 10 polish: panel titles padded off the
axes box (pad=10 in both producers); "corrected" replaced by the new
X_CO2^DE product label (XCO2_DE_LABEL; chosen over the too-long
DE-corrected superscript), added to the §6 notation table.  
**Updated:** 2026-07-22ab — model name decision: "deep ensemble (DE)",
NOT "DE-MLP" (§6 prefer/avoid row; tex sources to update). All
ATom/ship producers now default their cosmetic label to XCO2_DE_LABEL
(atom_pseudo_column, plot_ship_summary, plot_atom_comparison,
atom_modis_overlay, plot_ship_comparison — suptitles, map titles,
legends, colorbars), so the per-case figures adopt X_CO2^DE at their
next batch re-render; fig10a/b already carry it.  
**Updated:** 2026-07-22ac — Fig. 11 duplicate-letter fix (same problem
as Fig. 10): fingerprint panel retagged (b)→(d) so the composite runs
(a–c) transect + (d) fingerprint; k1 → ⟨l′⟩ in the fingerprint title
and expected-signature legend (MEAN_L_LABEL); Westar transect
regenerated with X_CO2^B11 / X_CO2^DE legend labels
(nassar_plume_transects.py adopts the plot_style constants).  
**Updated:** 2026-07-22ad — Fig. 11 caption: "±1 SE" spelled out
(standard error of the window-mean difference; plume + background
SEMs in quadrature — verified against nassar_k1_contrast.py).  
**Updated:** 2026-07-23 — Appendix C display items COMPLETED
(`make_appendix_c_figures.py`, one generator for all of them, `--only`
selector added). NEW: figC4_skill_vs_ceiling (the 2026-07-22q ceiling
blocker RESOLVED from `analysis/label_noise_ceiling.py` — only
r2max_ref is a hard ceiling; achieved skill legitimately exceeds the
posterior-σ and empirical lines, so the §4.3 "ocean ≈ noise-limited"
reading is CORRECTED in place), figC7_increment_attribution (the
"ML-on-raw artifacts not local" claim was wrong — `de_prof_reg_mix_raw`
plot_data is local; same 5.9 M-footprint population as the RAW_BC_ML
report §4, verified by count), tables tabC1_predictor_inventory
(from `models.pipeline`, cannot drift; description strings DRAFT —
verify `s31`/`dpfrac`/`fs_rel_0` against the B11 DUG),
tabC2_training_config (from the ten fold run_summary configs, asserted
identical — this EXPOSED that production β-NLL uses **β = 1.0**, not
the β = 0.5 in the Fig. 2 draft caption; caption fixed),
tabC3_fold_sizes_metrics, tabC4_manifest_verification (zero overlap
computed live: TCCON atrain 51 dates / drift 21 / ATom 8 / ship 4, all
∩ 116 model dates = ∅), tabC6_fold_resolved_baselines,
tabC7_cv_ablation; tab_raw_bc_ml renamed tabC8_raw_bc_ml (file +
generator). figC2 REBUILT single-panel (user: ocean and land share the
same date folds — verified identical fold-for-fold, and linreg == DE,
now asserted in the generator). User decisions folded in: NO
random-vs-date-split discussion in the manuscript (Fig. C3a/C3b retired
to `internal_random_split_inflation`/`internal_cv_design`; §3.4, §4.3,
and the Appendix C include-list updated; appendix figures now C1, C2,
C4, C5, C7 — renumber at typesetting); figC4 given (a)/(b) panel
letters with the legend outside the axes; Fig. C6 reconfirmed as
deliberately file-less (covered by Fig. 6b + Table C7).  
**Updated:** 2026-07-23b — Table C5 revisions (user request): model
label → "DE", and the land-ridge out-of-domain artifact FIXED by
applying the production output guard (|μ| > 25 ppm → correction
withheld) to the ridge held-out predictions at CV evaluation — the
same guard the deployed chain applies to every model, so the CV and
TCCON protocols now agree. tabC5/tabC6/figC5 recomputed from per-fold
artifacts (kfold_agg parse dropped; recompute verified to reproduce
the stored fold R² exactly); no more median/dagger/hatch special
cases. Guarded ridge: ocean fold RMSE 0.537 ± 0.023, land
0.673 ± 0.021 (18 of ~11.6 M footprints guarded; land f2's raw
extrapolation was ~10⁶ ppm on a −0.2 ppm target). §4.3 quoted ridge
numbers updated 0.69→0.67 land / 0.56→0.54 ocean. Retrain rejected as
the fix (deterministic model — seed no-op; test-time extrapolation
can't be bounded by training-side guards; local retrain infeasible at
26 GB). OPEN author decision: guard is asymmetric by design — uniform
guarding would trip 93/94 mostly-correct DE/XGB land corrections and
shift their frozen fold means +0.03 without changing the ordering.  
**Updated:** 2026-07-23c — Fig. C6 briefly GENERATED
(`figC6_cv_ablation`: paired per-fold ΔRMSE for the CV ablation
variants) after the user twice looked for the file.  
**Updated:** 2026-07-23d — Fig. C6 REMOVED again (user decision, now
final): the ablation is discussed with main-text Fig. 6b, so the
appendix carries Table C7 only; `figC6_cv_ablation` files deleted and
the generator function dropped. Appendix C figure files: C1, C2, C4,
C5, C7.  
**Updated:** 2026-07-23e — TCCON slice symmetry + per-surface radii
(user request): Tables 1, 2, C8 and Fig. 6 now carry OCEAN near-cloud
rows mirroring the land rows, and the near/far split uses each
surface's production target radius (ocean 5 km / land 15 km) instead
of the common 10 km. Implemented by rerunning
`tccon_comparison_report.py` on all 9 model trees (DE, XGB, Ridge, 5
ablation variants, ML-on-raw) with `--cld-edges 0,5,inf` and
`0,15,inf` under new suffixes (`_cldo5_r100km`/`_cldl15_r100km`;
~40 s/run, production `_r100km` files untouched);
`make_manuscript_tables.py` reads ocean slices from the o5 edition and
land from l15 (13 slices total), and
`make_baseline_ablation_figure.py` was rewritten to read the SAME CSVs
(fixing a silent drift: its bars were hardcoded from the superseded
2026-07-08 pre-foldpca docs). Headline consequences: near-cloud ocean
is now a real slice (n = 2,645, before 1.54 → DE 1.12, DE best on
every slice); the near-cloud land tail at ≤15 km reads DE 1.31 / XGB
1.56 / Ridge 2.52 (§4.3 tail-divergence clause updated 1.61→1.56,
0.30→0.25 ppm); spec-emphasis prose updated (no_xco2 up to +1.26 ppm
at land ≤15 km QF1, range 0.8–1.2). Fig. 6 caption ns updated
(2,645 / 81,347).  
**Updated:** 2026-07-23f — Fig. 6b in-panel takeaway text removed
(user request: the "spectral & contam.: free to drop / XCO2 group:
load-bearing" annotation is interpretation, which lives in the §4.3
prose per the caption rule); panel keeps only bars, value labels, and
the full-feature-set reference annotation.  
**Updated:** 2026-07-23g — draft captions added for every APPENDIX and
SUPPLEMENT figure (user request), under each appendix's Display/Planned
items: A1–A2, B2, B4, C1/C2/C4/C5/C7, D1–D6, E1–E4, F1–F3, G1, H1, and
page-template captions for the S1–S5 galleries. Captions for staged
artifacts (B2, B4, C-series, D2–D4, E2–E3, F1) were written against the
actual figures; pending artifacts (A1–A2, D1, D5, D6, E1, E4, F2–F3,
G1, H1) are marked PROVISIONAL — verify panel structure when produced.
All follow the 2026-07-22k caption rule (descriptive only).  
**Updated:** 2026-07-23h — Tables C1 and C2 converted to LONGTABLE
(user: C2 too long in the PDF; C1 at 60+ rows has the same overflow,
so both break across pages now, with repeated headers and wrapping
p-column for the description/value text; `_tex_table` gained a
`longtable` flag). Fig. 6b ablation rows RELABELED and REORDERED to
match Table 2's heads — singles first (−spectral, −contamination,
−X_CO2), then the double drops (−X_CO2−spectral, −X_CO2−contamination)
— after the user briefly could not find the single −X_CO2 row: the old
"−X_CO2 + spectral" phrasing read as add-spectral, and the old
effect-size ordering placed a combo above the single drop.  
**Updated:** 2026-07-23i — §4.3 draft results text gains the
load-bearing-but-not-sufficient sentence: univariate R² of the
xco2_raw − apriori departure against the production anomaly targets is
0.21 (ocean r05) / 0.10 (land r15) versus 0.71 / 0.55 for the full
ensemble — computed 2026-07-23 on the full combined parquet; the
bc-based departure (not a model input) reads 0.32 / 0.19 and is kept
as an author-side note only.  
**Updated:** 2026-07-23j — per-feature permutation importance PROMOTED
to main-text Fig. 7 in §4.3 (user request; previously unassigned — the
manuscript had only group-level attribution). New generator
`make_feature_importance_figure.py` (top-12 features per surface, DE
permutation ΔRMSE from the existing feature_importance agg CSVs, bars
by predictor group). Downstream figures RENUMBERED 7–11 → 8–12 (main
text now TWELVE figures): files and generator stems renamed
(fig08_tccon_dumbbell, fig09_significance_robustness,
fig10_smoother_null, fig11a/b atom+ship, fig12a/b westar+k1-contrast;
make_significance_panel / make_ocean_validation_figure /
make_k1_contrast_figure updated), all live plan references and the §4
figure map updated. §4.3 draft prose gains the feature-granularity
sentence with the standing caveats (CV importance over-credits
TCCON-neutral blocks; joint-group permutation is the honest number
under collinearity — Fig. 6b/Table 2 remain primary). Changelog
entries BEFORE this one use the old numbers.  
**Updated:** 2026-07-23k — Fig. 8 caption + §4.4 prose DATE-RANGE FIX
(user caught it): the 75 TCCON station-days span December 2014 –
December 2021 (51 unique overpass dates, verified from the atrain case
dirs when Table C4 was generated), NOT "2016–2020" — that span is the
TRAINING set, and the evaluation dates are disjoint from it by the
leakage guard, which the corrected sentences now state explicitly (a
stronger claim: the validation includes eras the model never saw).  
**Updated:** 2026-07-23l — Fig. 8 legend adopts the product-resolved
notation (X_CO2^raw / X_CO2^B11 / X_CO2^DE via the plot_style
constants; `tccon_comparison_report._bias_dumbbell`), and the user's
where-is-the-evidence question exposed that the 1.26→0.81 mean-|bias|
aggregate appeared nowhere in Fig. 8: the stat box now prints
"mean |bias|" beside the signed bias and footprint RMSE, and the
quoted 0.81/1.19 turned out to be the SUPERSEDED pre-foldpca tag's
values — the fold-PCA production CSV gives mean |b_s| = 0.816 → 0.82
and mean R_s = 1.196 → 1.20 (exactly the numbers the
tccon-correction-config note says to quote); fixed in the §4.4 prose,
§2.2 abstract values, and §4.5 smoother text. Production report rerun with the EXACT launcher flags
(--exclude-sites ny --cld-edges 0,10,inf, seeded bootstrap) — all
production CSVs verified byte-equal after the rerun; fig08 restaged.  
**Updated:** 2026-07-23m — Fig. 8 annotation iteration (user feedback):
the STAT BOX (not just the legend) now uses the product labels, adds
the std of |bias| (raw 1.34 ± 1.07 / B11 1.26 ± 1.33 / DE 0.82 ± 0.60),
and moved OUTSIDE the axes (above-left) so it cannot cover station-day
markers; the series legend is a one-row band above-right. Both changes
live in the shared report helpers, so the scatter-style D-series
figures inherit them (scatter legend labels updated to the product
notation too); figD2a/b + figD4 restaged from the rerun tree, figD3
(r50 edition) unchanged. CSVs re-verified byte-equal after every
rerun.  
**Updated:** 2026-07-23n — Fig. 8 layout iteration (user feedback): the
series legend moved back INSIDE the frame at bottom-right (framed,
`framealpha=0.85`; the bottom rows are the most-negative biases whose
markers sit left, so the corner is clear), the stat box moved back
INSIDE at top-left (top rows' markers all sit right of the box — nothing
covered), and the labeled-dumbbell canvas grew 6.2→7.8 in tall
(`_one` gained a per-figure `figsize` override) so the ~75 six-pt
'site date' y-tick labels no longer overlap. `_bias_stat_box` is now
style-aware (`inside=` flag): dumbbell styles get the inside box,
scatter_clddist (D-series) keeps the outside-above box unchanged —
figD2a/b + figD4 verified byte-identical after the rerun, only fig08
restaged. Production CSVs re-verified byte-equal.  
**Updated:** 2026-07-23o — Fig. D2a/b (user feedback): the outside stat
box moved from above-left to above-CENTER (`_bias_stat_box` outside
branch, ha='center' at x=0.5), freeing the top-left corner for new
panel labels — the qf0/qf1 single-panel bias views now carry (a)/(b)
via `_emit_figures` (`_qp = {'qf0': '(a)', 'qf1': '(b)'}`; the pooled
'all' view stays unlabeled). Report rerun with exact launcher flags;
production CSVs byte-equal; figD2a/b restaged; fig08 + figD4 verified
byte-identical (dumbbell inside-box path and by-site grid untouched).  
**Updated:** 2026-07-23p — Appendix D/E/F display-item sweep (new
generator `make_appendix_def_figures.py`, item statuses updated in
place): figD1 (3×3 coincidence sweep RERUN on the fold-PCA tag — 9
local report runs + `coincidence_sensitivity_table.py`; production
r100 CSVs verified byte-equal after the sweep), figD5 (DL forest,
pool cross-checked exactly), figE1 (±10/±100 s smoother windows),
figE4 (drift-tree report rerun with current styling, staged), figF2
(transects regenerated on fold-PCA tag then 3-col grid), figF3
(6-window k1 contrast). CONSISTENCY FIX discovered en route: fig09
(significance panel) and fig12b (k1 contrast) still read the
SUPERSEDED pre-foldpca tag — both retargeted to the fold-PCA tag and
regenerated; fig12b byte-identical (k1 columns are model-independent),
fig09 panel-b matrix cells shift ≤0.02 ppm displayed (old-tag sweep vs
fold-PCA sweep ≤0.024 ppm — the fold-PCA no-op bound). NOT produced
(no local data / not yet run): figA1 + figA2 (need fitting_details h5
— CURC), figG1 (3-D RT experiment not run), figH1 (TEMPO not run);
figD6 SKIPPED per plan condition (would duplicate main-text Fig. 11a).  
**Updated:** 2026-07-23q — Fig. 10 panel-a legend overlapped the scatter
(user feedback): moved to the empty upper-left triangle above the 1:1
line, below the mean annotation (`smoother_null_figure.py`; regenerated
from the fold-PCA smoother_null CSVs, figure-only script). figE1
panels a/c share the layout and got the same placement
(`make_appendix_def_figures.py`); both restaged.  
**Updated:** 2026-07-23r — Worsening-case investigation (§4.4/§Appendix
E): (1) §4.4 headline sentence disambiguated — 71/75 is the fp-RMSE
improvement count, station-day |bias| improves 46/75 (29 worsen, median
+0.28 ppm, mostly near-zero starting biases); stale 0.81 → 0.82 and
Wilcoxon p 0.0064 → 0.0063 fixed in the same sentence (sentence now
refs `sec:result-worsening` — label must be added when the worsening
subsection is pasted). (2) NEW stage 6 in `analyze_failure_modes.py`:
held-out CV × albedo cross-check (5 land folds' held_out_predictions
joined to the training parquet on exact lat/aod_total/fp, 100% match,
3.84M footprints, TCCON-matched decile edges) — bright-surface TCCON
failure signature does NOT reproduce against the anomaly label →
verdict revised to UNDER-correction of a within-overpass common-mode
bias (conclusions bullet 7); report regenerated as
FAILURE_MODES_2026-07-23.md (supersedes 2026-07-16 edition), new CSV
strat_cv_land_alb_o2a_r100km.csv; Appendix E gains Table E3 + caption.
Supporting counts for the §4.4 discussion: 12,037 TCCON footprints
(11.4%) have alb_o2a > 0.4, dominated by three Darwin station-days.  
**Updated:** 2026-07-23s — Fig. E2 split + antimeridian RGB bug (user
feedback): the old two-case E2 becomes Fig. E2 (ATom) + Fig. E3
(ship); old E3/E4 renumbered E4 (failure modes) / E5 (drift), files
renamed accordingly (figE2a deleted, figE2b→figE3, figE3→figE4,
figE4→figE5). ROOT CAUSE of the "fully cloud-covered" E2a background:
GIBS daily mosaics are keyed by the LOCAL day near the antimeridian —
the 2017-10-09 case overpasses lon −175° at 01:32 UTC = local day
2017-10-08 (the ATom flight date, as OCO_TO_FLIGHT already encodes),
so the case-date tile showed the overcast scene 24 h later.
`plot_atom_comparison.py` + `atom_modis_overlay.py` now fetch the
local-solar date of the collocated footprints; the 2017-10-08 tile
matches the cloud-distance field (clear band, broken cloud N+S,
median 18 km). figE2 regenerated (numbers unchanged: n=313,
+0.29→+0.28 ppm; also picks up product-label styling); any future
regeneration of the other ATom case figures (Supplement S2) inherits
the fix. E-caption block split/renumbered to match.  
**Updated:** 2026-07-23t — Fig. E5 legend + stat box moved OUTSIDE the
frame (user feedback; the drift dumbbell holds data in both inside
corners, unlike Fig. 8): `tccon_comparison_report.py` gains
`--dumbbell-annotations {inside,outside}` (default inside — Fig. 8's
production layout unchanged, its tree not rerun); `_bias_stat_box`
placement generalized to inside / above-left / above-center. Drift
report rerun with `outside`, drift CSVs byte-equal, figE5 restaged —
all 21 rows now unobstructed.  
**Updated:** 2026-07-24a — ALL missing appendix tables GENERATED (user
request) by the new `manuscript/scripts/make_appendix_tables.py`
(`--only` per table; C-series \tophline style; longtable for the 75-row
listings), 15 files into `manuscript/tables/`: A1 fitting config (from
`constants.FIT_ORDER` + cumulant_fit source), B1 cohort attrition
(116-date parquet counts: 17,769,270 rows, 17,745,005 valid cld-dist,
ocean 10,546,333 / land 7,222,937; ocean r05 labeled 7,848,762 =
fold-held-out sum EXACTLY, land r15 labeled 3,844,864; eval populations
75/21 TCCON, 17 ATom legs, 4 ship), B2 target params + guards, B3
label-noise ceilings (production r05/r15 rows of the 140-date CSV),
D2 complete station-day metrics (75 r100 + 69 r50, longtable), D3
Wilcoxon + bootstrap (r100+r50 × QF × excl-ny; plain 10^{-x} math, no
siunitx), D4 per-case uncertainty budget (label `tab:unc_components`
matching the Appendix D text; DL pool recomputed in-script: r100
μ=−0.32±0.09 / τ=0.52 / I²=51%, r50 μ=−0.45±0.07 / τ=0.22 / I²=14% on
68 of 69 evaluable cases — NaN-budget case dropped, matching the md),
D5 ATom legs, D6 ship cases, E1 all 29 worsening cases with arithmetic
categories (near-zero start / overshoot / amplified; fp-RMSE still
improves in most), E2 driver-strata extremes (alb/snow/AOD/|lat|/σ ×
low/high decile), E3 CV-albedo cross-check (Table E3 of 2026-07-23r),
E4 smoother-null numerical table (|bias| 1.26→0.82 DE vs
1.24/1.20/1.20; scatter 2.23→0.78 vs 0.66/0.51/0.35 — matches §4.5),
F1 case inventory (19 screened cases, 3 RGB-vetted flagged), F2 plume
bounds + control nulls (7 windows, 5 pass / 2 flagged-as-cloud).
All 15 compile clean (scratch pdflatex: 0 errors, 0 overfull).
Table D1 NOT generated — already covered by the hand-written
`tab:tccon-stations-used` + dates tables in appendix_D.tex. Table A2
(fit availability/failure accounting) BLOCKED locally: the combined
parquet holds only successfully fitted soundings (trivially 100%), the
real accounting needs a CURC sweep over per-date fitting_details.h5.
G1/H1 remain artifact-pending with their conditional appendices.
\input wiring into appendix_*.tex left to the author (no-unasked-tex-edit
rule).  
**Updated:** 2026-07-24b — **Appendix G COMPLETED and SELF-RUN; the
conditional placement is RESOLVED → typeset appendix.** The cohort
backward-MC figure (former A13) is replaced by our own er3t/MCARaTS
v0.10.4 x–z slab simulation (`workspace/rt_slab_sim/`; production data =
Blanca 1e9-photon sweep, 33 O2A wavelengths × {3-D, IPA} × {dark 0.03,
bright 0.30}, Nrun 3). Scene: real OCO-2 sounding 2020010100281632
(29252a, SZA 55°), column-conserving 21-layer grid, water cloud COD 10 at
3–4 km in a 32-km Ny=1 periodic slab, sun along +x. The runs are refit
with the PRODUCTION estimator (order 7, no-SG, exact lstsq/BVLS) and —
beyond the original scope — MCARaTS' native path-length tally
(Rad_mplen=3) records the per-column PPDF, giving a quantitative
first-moment closure: r(fitted ⟨l′⟩, tallied mean path) = 0.999 (dark) /
0.987 (bright) over clear columns; r(var(l′), tallied path variance) =
0.989 on the bright surface; IPA-null residual ≤ 0.9 % of the 3-D range.
Display items EXIST: `manuscript/figures/figG1_mc_3d_vs_ica.{png,pdf}`
(AMT style, 4 rows × dark/bright: ⟨l′⟩, var(l′), effective reflectance,
MC path-moment closure row) and `manuscript/tables/tabG1_slab_config.tex`
(auto-generated from the simulation config — cannot drift). Generators:
`workspace/rt_slab_sim/make_fig_g1.py` / `make_table_g1.py`; closure
numbers in `results/rt_slab_sim/closure_stats.json`; supporting PPDF
heat-map/cut figures (`slab_ppdf_{dark,bright}.png`) are S6 candidates.
Appendix G section text + captions updated below (§5).  
**Updated:** 2026-07-24c — **Table H1 DROPPED (user decision; paper
length).** The instrument/geometry/sampling differences from OCO-2 fold
into the Appendix H prose (they were already on its content list), and
the Fig. H1 caption's closing Table-H1 sentence is deleted. The
side-by-side comparison is retained AUTHOR-SIDE ONLY as
`manuscript/tables/internal_tempo_fit_inputs.tex`
(`make_appendix_tables.py --only internal_tempo_inputs`; TEMPO column
from the verified numbers in the tempo TODO §5b, OCO-2 instrument-spec
rows flagged for ATBD verification) — not \input anywhere, kept in case
it proves useful later. Appendix H is now one granule + ONE figure.  
**Updated:** 2026-07-26 — **LENGTH TRIM executed on the compiled manuscript
(89 → 76 pages); this entry supersedes the structural parts of every entry
above.** Trigger: the appendices had grown to 53 of 89 pages (15.5 k words vs
9.5 k in the body) with the Discussion still unwritten. Journal re-confirmed
as AMT — note that Copernicus moved to a FLAT per-paper APC on 2025-01-01
(€1,800 net, EGU members €1,620; no per-page surcharge, supplements free), so
length carries no cost penalty there and the trim is purely about
reviewability. Changes, all reflected in §3–§5 below:
(1) **Results 8 subsections → 4** — 4.1 phenomenology + spectral signature
(former 4.1 + 4.2 MERGED), 4.2 model comparison, 4.3 independent validation
(4.3.1 TCCON / 4.3.2 smoother null / 4.3.3 ocean / 4.3.4 uncertainty and
reliability), 4.4 plume safety. Section labels unchanged, so every `\ref`
still resolves.
(2) The §4.2 "three roles" paragraph MOVED to the Discussion: roles 1
(mechanism) + 2 (safety audit) open §5.1, role 3 (imager-free sensitivity →
transferability) opens §5.3. Results now points forward instead of arguing.
(3) The bright-surface / worsening-case paragraph moved out of §4.3.1 into
§4.3.4 so that stratum is discussed ONCE, with high-AOD and high-latitude, as
one "under-corrected, not mis-corrected" argument.
(4) **Appendix A** cut to the instrument-side construction only (eq. A1–A7 +
Table A1), retitled "Optical-depth and transmittance construction for the
spectral fit"; the Beer-Lambert → gamma → cumulant derivation was a duplicate
of Methods §3.2 and was removed (κ, its only unique product, is not used).
(5) **Appendix D** retitled (it carried the stale Appendix-C title),
tables reordered so D1 = TCCON stations, D2 = comparison dates, D3 = ATom
legs, D4 = ship cases, D5 = training dates; the AK/prior-harmonization
operator subsection REMOVED (standard Rodgers–Connor/Wunch operator, now
carried by citation in Methods §3.5 — with the GGG2020 wet→dry prior
conversion stated explicitly there, since that step is NOT in the cited
procedure and shifts the reference by ≈1 ppm); figures D3 (r50) and D4
(station summary) dropped as unreferenced, plus a duplicate copy of the
QF0/QF1 figure.
(6) **Main-text table budget is now TWO** (model comparison, feature-set
ablation): `tab_station_equal_bias` moved to backup.
(7) Appendix E: Table E1 (per-case worsening listing) moved to backup; its
load-bearing number ("RMSE still improves in 25 of those 29") is now stated
in §4.3.1. Table E2 (driver strata) KEPT — §4.3.4 quotes it directly.
(8) **Appendix H KEPT** (user decision reaffirmed) but shortened 742 → 464
words: flat prose, no subsections, one figure, existence-proof scope.
(9) Bugs fixed while trimming: duplicate `\label{app:null-test}` (E and F —
§4.3.2's smoother pointer had been resolving to the WRONG appendix),
duplicate `\label{app:case_and_plume}` (F and G), two `\ref{a,b}` two-label
refs printing as `??`, `\ref{tab:model_comparison}` → `tab:model-comparison`,
hard-coded "Sect. 4.4"/"Appendix D, Table D4"/"Fig. D5" pointers (Table D4 was
not even in the document), undefined `\ref{fig:unc_forest}`, and a verbatim
duplicated paragraph at the end of §4.3.2. Four generated tables had section
numbers hard-coded in their captions; the generators now emit `\ref`.
Originals of every edited file: `manuscript/backup/pre_trim_2026-07-26/`;
dropped generated tables: `manuscript/backup/`. NOTE `manuscript/` is
gitignored — those snapshots are the only recovery path.
Deferred by user decision: figure/table renumbering (do it when the
manuscript is near-final).  

**Updated:** 2026-07-26b — **second trim pass, same day: Discussion merge,
one-file-per-section, filename re-sync, Appendix B rewrite (76 → 77 pages).**
(1) **Discussion 6 subsections → 4**: 5.1 physical interpretation · 5.2
relationship to the operational correction AND observation recovery (former
5.2 + 5.4) · 5.3 meaning and limits of imager independence · 5.4 limitations
AND future work (former 5.5 + 5.6). Rationale: two of the six were never
subsection-sized by their own plan (5.2 = three sentences citing Appendix C;
5.4 = one sentence of QF1 counts), and 5.5/5.6 duplicated each other on PPDF
moment closure and local-contrast attenuation. Word targets recorded in
`5_discussion.tex`: 500 / 400 / 500 / 600 ≈ 2,000.
(2) **One `.tex` file per section** (user rule): merged sections are single
files, not a host plus a label-only stub. `4.2_spectral_features.tex`,
`5.4_recovery.tex`, and `5.6_future.tex` were folded into their hosts and
deleted; the absorbed section's `\label` stays at the merge point under a
`% --- former X.Y ... ---` banner, so every `\ref` still resolves.
(3) **File names re-synced to printed section numbers**:
`4.1_phenomenology_spectra` · `4.2_model_comparison` · `4.3.1_tccon_val` ·
`4.3.2_correction_vs_smoothing` · `4.3.3_ocean_far_cld` ·
`4.3.4_uncertainty` · `4.4_plume` · `5.1_physics_interp` ·
`5.2_opt_bc_recovery` · `5.3_imager_indepent` · `5.4_limitations_future`.
The two merged files got content-matching names rather than a bare renumber.
`log/CONTAM_REGROUPING_2026-07-25.md` repointed to `4.2_model_comparison.tex`.
(4) **Appendix B rewritten** — retitled "Analysis cohort and target
construction", ~660 words of prose added (it previously had none, with 5 of 6
display items unreferenced), Table B3 moved to Appendix C as Table C3 beside
Fig. C4, and the missing body pointers added (Table B1 from Sect. 2, Table B2
+ Fig. B1 from Sect. 3.1). A `\FloatBarrier` now keeps B's figures inside
Appendix B instead of drifting into C, which is what had scrambled the
B/C numbering. New fact surfaced and stated in the prose: label retention is
74 % ocean vs 53 % land, so the labeled land population under-represents
persistently cloudy scenes.
(5) Earlier claim CORRECTED: the Appendix B title was said to promise
cloud-collocation content the appendix lacked. Wrong — Table B2's first block
IS the collocation parameter set. The retitle was made on emphasis grounds
(Sect. 3.1 owns collocation), not because the old title was inaccurate.
(6) Caption drafts (21 blocks, ~450 lines) removed from the flow plan; where a
label carried a decision rather than provenance it survives as a one-line
`*Caption note (...)*`, including the full TEMPO scene-selection rationale.
This history file was split out of the flow plan in the same pass.

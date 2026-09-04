# Manuscript review suggestions — Sections 2–5 and Appendices A–H (2026-07-28)

Scope: every `manuscript/*.tex` for Sections 2–5 and Appendices A–H, plus the
`tables/*.tex` they input. Extended 2026-07-31: §7 covers `1_intro.tex`,
reviewed against `manuscript/ADVISOR_REVIEW_CRITERIA.md`. Drafted manuscript
text follows your style rules (no colon, no dash, plain professional tone).
OPEN items only — applied items were removed 2026-07-28 and again 2026-07-29
(each removal verified against the tex first); item numbers keep their
original gaps so earlier discussion still cross-references.

**Removed as done (for the record):** all §1 broken refs (1.1–1.5; a
comment-stripped scan finds zero unresolved `\ref`); §2 items 2.1–2.10 and
2.17 (β = 1, batch 8192, seed s₀ = 42, inline hyperparams table deleted,
cohort wording/counts, aggregate naming, ablation deltas, m₁ = 0.5 prose);
the tabB1 "glint" row labels and the Table 2 caption generator (deltas now
computed from rounded cells; caption reads +0.02/−0.03, max +0.10 on ocean
≤5 km QF = 1, +1.22 pooled); tabC2/tabC3 captions now emit
`Fig.~\ref{fig:5fold_illustration}`; the "Table 1 from 4.2" pointer (§3.1);
the 5.3 wording block; item 4.14 (resolved by deleting the inline table).

**Verified and removed 2026-07-29:** 2.11, 2.14, 2.19 (figC5 caption and
tabC6 generator both emit `\ref{tab:cv-model-comparison}`); ALL of §3 — the
appendix C/D/E/F prose is applied (C intro reworded with "model predictors"
added), every orphan pointer is live, and the §4.4 pointers are `\ref`-based
with the six-window recount (12 km closest-approach criterion, four-of-six);
§4 items 4.1–4.6, 4.7 (full Unicode sweep incl. ⊕/·/°/µ/³/thin-spaces),
4.8, 4.9–4.13, 4.15, 4.16 (generators emit `\ref`; internal_tempo table
stays author-side), 4.18 (training-dates table now precedes comparison
dates), 4.19 (all four labels renamed with their 12 `\ref` sites); §5 items
for 2_data (opening sentence, A-Train sentence + spelling, bare `\ref`),
3.1 (Fig. 1 caption, emission-source clause), 3.2 (l′ definition sentence),
3.3 (Fig. 2 caption rewritten dash-free; the cloud-distance clause moved to
prose), 3.5 (both), 3.6 (dropped from 3_method.tex; the placeholder file
stays on disk author-side, unreferenced), 4.1 (2.14 parenthetical, the
$\thicksim$0.5 sentence removed in a rewrite, dotted threshold lines now
defined in the fig03 caption, \textbf → \emph). Finishing touches applied
during the 2026-07-29 verification itself: band macros standardized in the
generated tabC1/E2/E3/G1 (generators now emit `\ce{O2}A`/`W\ce{CO2}`/
`S\ce{CO2}`; tabG1 also 3-D → 3D), the 10 remaining straight-quote pairs in
4.2 → ``...'', `Eq.~\ref` → `Eq.~\eqref` at 4 sites (3.1, 3.2 ×2, 5.4), and
two extra bare/long refs on 2_data line 16 → `Sect.~\ref{...}`.
Later on 2026-07-29: 2.18 (appendix G follow-up list trimmed to match 5.4 —
angle/optical-depth/albedo sweep plus absorption-weighted path tallies; the
synthetic-spectra and plume-injection promises dropped), 2.20 (introduction
drafted and states the QF-screening claim that 5.2 cites via
`\ref{sec:intro}`), and the stale supplement.tex compile bullet in §6
removed (Supplement backup-only, paper cites it nowhere). 2.15 (Eq. 3
integrand now $\alpha_\lambda$; user) and 2.16 (user restructured Eqs. 5–6 to
$\tau_\lambda(l') \simeq \mathrm{\overline{SOD}}_\lambda l'$ with a two-step
Laplace reduction; the remaining bare $\tau$ in Eqs. 8–9 resolved by the
"writing $\tau \equiv \mathrm{\overline{SOD}}_\lambda$ for brevity" clause —
consistent with the appendix "order 7 in slant $\tau$" vernacular). Same
pass: line 43's \meanl slip → $l'$ (Eq. 4 defines the variable, not its
mean), and $\ln T$/$T_\lambda$ → $\ln \mathcal T_\lambda$/$\mathcal
T_\lambda$ in Eqs. 8–9 and the channel mask ($\mathcal T_\lambda = \gamma\,
\mathcal T$ carries the $\ln\gamma$ intercept, so $\mathcal T_\lambda$ is
the correct LHS object). 2.13 verified with no edit — fig01 is generated
from granule 29265a_GL (`make_collocation_schematic.py` default, annotated
sounding 2020010121413432), so the caption's orbit 29265 is correct
alongside appendix A/G's 29252a (different orbits of the same day). 2.12
resolved with a scope correction beyond rounding: `closure_stats.json`
gives IPA-null residuals of 0.87 % for ⟨l′⟩ (max over surfaces) but 4.6 %
for dark-surface var(l′), so 3.2 and appendix G now both read "within
0.9 % of the 3D dynamic range in \meanl and within 5 % in \varl" instead
of the blanket "below 1 %"; bright-surface variance closure unified at
r = 0.989 (JSON 0.9888). §2 is now empty and removed.

**Applied and removed 2026-07-31 (pre-advisor-send batch, user-approved):**
all §7 introduction items 7.1–7.13 — the ¶7 rewrite of 7.2/7.3 as drafted;
7.9 via the make-intent-explicit route (a transferability sentence in ¶7,
"in principle transferable ... including several of those listed above",
pointing to `Sect.~\ref{sec:disc-img-free}`; trimming the mission list
remains an open alternative if SS still flags it); 7.7 as drafted, closing
the v11 product-naming vagueness. Also the 4.3.4 duplicate-paragraph merge
(the 1.27 near-cloud σ scaling folded into the paragraph-1 $u_s$ definition
so it survives the cut; the weights $1/(u_s^2+\tau^2)$ and the τ-would-be-
zero explanation kept once, in paragraph 2), the 5.4 "parallel effect" →
parallax and "otherviwse" fixes, and the 4.1 `$\thicksim15$` → `$\sim15$`.
The 2_data.tex and 3.1 typo lines were found already fixed in the tex
before this pass (stale entries dropped from §5).

**Applied and verified 2026-07-31, second batch (dash/colon cleanup + the
checking items, user-approved):** all remaining §5 and §8 dash/colon items
([x] below) — 4.2 ablation-paragraph colons/dashes, 4.3.2 structural
sentence, 4.3.4 dashes + agreement colon, 4.4 bound sentence, 5.1 drifted
colon, appendix G colons/dashes (align/rounding sub-item verified already
applied; double space fixed), appendix H dashes + colon — plus the §8.25
unicode-dash sweep (live text now clean; leftovers only in the
0_abstract/6_conclusion/5.4 comment blocks) and the 4.2 unbalanced
parenthesis (part of 8.24, same clause). Verification outcomes: **8.1
CONFIRMED and fixed** (recomputed on the 17.77M-row parquet: raw−prior R²
= 0.205 ocean / 0.099 land; the quoted 0.323/0.191 are exactly the
bc−prior values; tex now 0.205/0.099). **8.4 CONFIRMED and fixed**
(tabC8: QF=1 pooled fp-RMSE raw 4.04 → bc 4.68 is a degradation; wording
now says so). **8.14 fixed** (figC4 caption → Table~\ref{tab:label-noise}).
**8.16 tex correct** (CSV wilcoxon_p = 0.00633; the plan's 0.0064 was
stale). **8.17 all verified correct** against tabC8 + the 2026-07-27
report (slope +0.53, r +0.79, 61 %; land ≤10 km r +0.866 → +0.87, 74 %;
1.15 vs 1.22; 3.26→3.71, 4.20→4.94, −0.85→−1.20; QF0 0.85). **8.18
resolved** (tabG1: 33 total = 3 anchors + 30 log-spaced τ 0.03–8; G prose
aligned). **8.19 tex correct** (50 km × ±30 min is the largest cell,
Δ|bias| −0.93 vs −0.77 at 25 km). **8.20 fully verified**: headline
1.26→0.82 / 2.67→1.20 / 71/75 / 46/75; QF1 1.57±1.44(ddof 1)→0.89±0.62 and
3.19→1.31 / 4.68→1.54; Ny-Ålesund 2.18→1.31, 4.74→1.95, −8.20→−0.86
(2017-06-17), 7/10 positive, max +2.89; far-cloud strata 1.02→0.86 /
0.95→0.79 (tabC8 rows); drift 1.24→0.67, 2.41→1.01, 20/21, exact Wilcoxon
p = 0.0263 and 1.34e-5 (the md's "< 1e-4" display; tex's 1.3e-5 exact);
poleward fractions 936,178 (5.3 %) / 411,649 (2.3 %) exact. One fix from
the check: excl-Ny bootstrap max p is 0.0162, so "p ≤ 0.016" → "p ≤ 0.017".
**7.14 mostly confirmed via publisher pages**: King 2013 land ≈55 % /
ocean ≈72 % ✓ (IEEE TGRS abstract), Mauceri 2023 20 %/40 % ✓ (AMT
abstract); Massie 2021's 40 %/73 %-within-4-km still needs the paper body
(the project's own 41.7 % agrees with the 40 %). **7.15 resolved and
applied**: Emde Part 1 is "Synthetic dataset for validation of trace gas
retrieval algorithms" (Part 2 carries the NO2-retrieval impact); the intro
sentence now describes Part 1 accurately. Full compile clean after all
edits (zero LaTeX warnings).

Legend: **[fix]** = objective error, should apply. **[verify]** = needs your
confirmation of which number/fact is right. **[draft]** = proposed new text.
**[style]** = wording/notation, optional.

---

## 4. Notation and style consistency — [style]

- [ ] **4.17 Appendix C printed numbers vs file names** (cosmetic; shifted
  again when the inline hyperparams table was deleted): with the current
  order, tables print as C1 = tabC1_predictor_inventory,
  C2 = tabC3_label_noise_ceilings, C3 = tabC2_training_config,
  C4 = tabC3_fold_sizes_metrics, C5 = tabC4_manifest_verification,
  C6 = tabC5_cv_model_comparison, C7 = tabC6_fold_resolved_baselines,
  C8 = tabC7_cv_ablation, C9 = tabC8_raw_bc_ml. Consider renaming the
  generated files at the final renumbering pass.

---

## 5. Grammar and wording, per file — [style] with proposed rewrites

Rules applied: no colon or dash where avoidable, plain direct professional
tone, keep your voice. Only load-bearing sentences listed; typos are grouped.

### 4.2_model_comparison.tex
- [ ] "Each model has $5$ members for their ocean and land models, which are
  trained on different folds of data" conflates members and folds → suggest

  > Each model is trained separately for ocean and land and five times under
  > the date-blocked folds (Fig.~\ref{fig:5fold_illustration}); the \emph{DE}
  > additionally carries five members per fold.

- [ ] "(See detailed features in Table~\ref{tab:training-config})" →
  "(see the feature groups in Table~\ref{tab:predictor-inventory})". The
  training-config table does not list features; this is the same mispointing
  family as the fixed 1.4.
- [ ] "which $\Delta \mathrm{X_{CO2}^{predicted}}$ is predicted \xco anomaly"
  → "where $\Delta \mathrm{X_{CO2}^{predicted}}$ is the predicted \xco
  anomaly".
- [ ] "Same as the results in Fig.~\ref{fig:model_baseline_ablation}b, the
  ..." → "Consistent with Fig.~\ref{fig:model_baseline_ablation}b, the ...".
- [x] Colon/dash cleanup in the long ablation paragraph (line 19), e.g.
  "...radiance-variability diagnostic (Table~\ref{tab:predictor-inventory}):
  the correction does not require any explicit description of the scene's
  scattering content." → "...radiance-variability diagnostic
  (Table~\ref{tab:predictor-inventory}). The correction therefore does not
  require any explicit description of the scene's scattering content." and
  "in every slice — $+1.22$ ppm pooled, ..." → "in every slice, with $+1.22$
  ppm pooled, ...", and "does not make it dispensable: the roles it plays
  beyond prediction — mechanism, correction-safety audit, and imager-free
  cloud sensitivity — are taken up in ..." → "does not make it dispensable.
  Its roles beyond prediction, namely mechanism, correction-safety audit, and
  imager-free cloud sensitivity, are taken up in ...".
- [ ] "carries the most abundant cloud-induced \xco bias information" →
  "carries the largest share of the cloud-induced \xco bias information".

### 4.3.1_tccon_val.tex
- [ ] "Note that these 75 station-days for 18 sites from December 2014 to
  December 2021 are chosen manually, as we want to have more available data
  for stations near the coastline to validate ocean footprints, and we are
  unable to run all OCO-2 overpass cases." → suggest

  > These 75 station-days at 18 sites from December 2014 to December 2021 are
  > selected manually. We prioritize stations near coastlines so that ocean
  > footprints are represented, and processing every OCO-2 overpass is not
  > computationally feasible.

- [ ] "For all TCCON stations we used, Ny-Ålesund is the only station in our
  comparison poleward of 60$^\circ$ latitude" → "Among the 18 stations,
  Ny-Ålesund is the only one poleward of 60$^\circ$ latitude".
- [ ] Colons in the aggregate-definition paragraph (line 26) are mostly
  list-defining and defensible; if you want them out, split "...so negative
  values are improvements: $\overline{|b|}$ and $\mathrm{RMS}_b$ respond only
  to..." into "...so negative values are improvements. $\overline{|b|}$ and
  $\mathrm{RMS}_b$ respond only to...".

### 4.3.2_correction_vs_smoothing.tex
- [ ] "it is the strongest form of the ``the correction only smooths''
  hypothesis" → "it is the strongest form of the hypothesis that the
  correction only smooths".
- [x] "The reason is structural: because the running mean approximately
  preserves the overpass-mean \xco, it can only redistribute the error among
  footprints, not remove it, while the deep ensemble shifts the mean itself."
  → "The reason is structural. Because the running mean approximately
  preserves the overpass-mean \xco, it can only redistribute the error among
  footprints, whereas the deep ensemble shifts the mean itself."

### 4.3.3_ocean_far_cld.tex
- [x] **LAPSED 2026-07-31 (ship removal, 9.7)** — the "two different
  in-situ measurements" sentence was rewritten ATom-only; the in-situ
  mislabel no longer occurs.
- [x] **LAPSED 2026-07-31 (ship removal, 9.7)** — the ship-offset sentence
  (+0.99 → +1.18 ppm) was deleted with the shipborne comparison.
- [x] **LAPSED 2026-07-31 (ship removal, 9.7)** — the "limited temporal and
  spatial overpass samples" sentence was deleted with the shipborne
  comparison.

### 4.3.4_uncertainty.tex
- [x] Dash cleanup: "Because each station-day can carry its own small true
  offset — different air masses, seasons, and site conditions — we
  summarize..." → "Because each station-day can carry its own small true
  offset from different air masses, seasons, and site conditions, we
  summarize...". Similarly "the per-case error bars alone are too
  optimistic — the residuals are..." → "the per-case error bars alone are too
  optimistic. The residuals are...", and "treat $\sigma$ as a per-footprint
  reliability indicator — with the explicit limit that..." → "treat $\sigma$
  as a per-footprint reliability indicator, with the explicit limit that...".
- [x] "First, improvement is not the same as agreement: only 1 of the 75
  station-days passes..." → "First, improvement is not the same as agreement.
  Only 1 of the 75 station-days passes...".

### 4.4_plume.tex
- [x] "is at most 0.21~ppm in the worst tested window and below 0.01~ppm in
  the clear cases — small compared with the 0.6--2.5~ppm plant-window
  enhancements in the catalog." → "...in the clear cases, which is small
  compared with the 0.6 to 2.5~ppm plant-window enhancements in the catalog."

### 5.1_physics_interp.tex
- [ ] "the spectral features contribute none of it essentially" →
  "the spectral features contribute essentially none of it".
- [x] "not why it drifted: a nearby cloud produces such a departure, but a
  real \ce{CO2} enhancement produces one as well." → "not why it drifted. A
  nearby cloud produces such a departure, but a real \ce{CO2} enhancement
  produces one as well."
- [ ] "where the well-known \ce{O2} abundance cannot be the cause" — the
  argument is that O2 is well mixed with a known column, so a CO2 change
  cannot move it. Suggest "where the \ce{O2} column is well mixed and known,
  so a \ce{CO2} change cannot be the cause".

### 5.2_opt_bc_recovery.tex
- [ ] "(Sect.~1)" → `(Sect.~\ref{sec:intro})`, see 2.20.
- [ ] Open item from the merge comment: the planned one-sentence statement of
  the production QF1 counts (how many flagged soundings exist by distance,
  surface, region) is still not in the text. If you want it, the counts need
  to be pulled from the production inference output; currently the paragraph
  quantifies recovery only through Table~\ref{tab:raw-bc-ml}.

### 5.4_limitations_future.tex
- [ ] "We assume clouds remain in similar locations, but there is actually
  about a 6-minute difference between OCO-2 and MODIS Aqua." → "We assume the
  clouds do not move during the roughly 6-minute separation between the OCO-2
  and the MODIS Aqua overpasses."
- [ ] "Pixels with sub-grid clouds categorized as clear-sky will also lead to
  analysis uncertainty." → "Pixels with sub-pixel clouds that are classified
  as clear also add uncertainty to the analysis."
- [ ] "low interference from Rayleigh scattering by air molecules, which
  could cause additional significant multiple scattering. In other words,
  absorption properties near short visible and ultraviolet ranges are not
  ideal" → suggest

  > low interference from Rayleigh scattering, which adds its own multiple
  > scattering. Absorption bands in the short visible and ultraviolet ranges
  > are therefore not ideal for this spectral fitting analysis.

### Appendix A
- [x] Unclosed dash parenthetical in the opening sentence ("The path-length
  formalism itself — the equivalence theorem, ... $k_2 =$ \varl is given
  in..."). Dash-free rewrite:

  > The path-length formalism, namely the equivalence theorem, the Laplace
  > transform of the path distribution, and the cumulant identification
  > $k_1 =$ \meanl and $k_2 =$ \varl, is given in
  > Sect.~\ref{sec:methods_photon_path} and follows
  > \citet{irvine1964,partain2000,stephens2000}.

### Appendix B
- [x] Style only: three "---" constructions, e.g. "is a direct consequence of
  the wider land reference radius --- a 15 km clear-sky floor is harder to
  populate within the latitude window than a 5 km one --- and it is the
  selection effect..." → "...radius, since a 15 km clear-sky floor is harder
  to populate within the latitude window than a 5 km one. This is the
  selection effect...". Same treatment for the other two, and the two colons
  ("...construction: the ocean curve is...").

### Appendix C
- [x] Subsection title "Deep-ensemble Member architecture and probabilistic
  head" → lowercase "member".
- [x] `\subsection*{XGBoost}` should be `\subsubsection*` to match
  `\subsubsection*{Ridge}` (currently XGBoost renders one level too high).
- [x] "The LR and XGBoost models are trained with..." → "The \emph{Ridge} and
  \emph{XGBoost} models are trained with..." (LR is never defined).

### Appendix D
- The "---" instances are inside longtable group headers (structural), fine
  to keep.

### Appendix E
- [x] Subsection title "Failure cases analysis" → "Failure-case analysis"
  ([appendix_E.tex:46](manuscript/appendix_E.tex#L46); the §3.4 prose is
  applied, only this rename remains).

### Appendix G
- [x] "We simulate one explicitly defined cloud scene with both
  three-dimensional (3D) and the independent-pixel approximation (IPA) mode"
  → "...with both a full three-dimensional (3D) solver and a one-dimensional
  (1D) solver under the independent-pixel approximation (IPA)". This also
  introduces 1D, which G currently never expands, and fixes the
  singular/plural ("mode").
- [x] "Here we use the simulation to provide insights under ideal and single
  conditions about the change of spectral features near clouds." → "Here we
  use a simulation with one controlled condition to isolate how the spectral
  features change near a cloud."
- [x] Citation form: "(v0.10.4; \citet{iwabuchi2006efficient})" and
  "(similar to  \citet{chen2025ear3t_oco2_cloud_bias})" → use `\citealp`
  inside parentheses, and remove the double space.
- [x] "at relative paths up to l $\approx$ 1.3" → "$l' \approx 1.3$" (the
  variable is the relative path).
- [x] Colon cleanup in Results, e.g. "the IPA run is the exact null that the
  causal argument requires: outside the cloud, every fitted feature..." →
  "...requires. Outside the cloud, every fitted feature...". Same for "The
  underlying distributions show why: in the shadow band, ...".
- [x] Dashes: "$\tau \approx 10^{-4}$ to $8$ — the same range sampled by the
  real spectra" → "...to 8, the same range sampled by the real spectra";
  "contributing photons — a direct sample of the photon path-length
  distribution (PPDF) whose cumulants..." → "contributing photons, which is a
  direct sample of the photon path-length distribution (PPDF) whose
  cumulants...".
- [x] Align follow-up list with 5.4 (2.18); rounding alignment with 3.2
  (2.12) — verified already applied (G follow-up list matches 5.4;
  0.9 %/5 % rounding matches 3.2).

### Appendix H
- [x] Dash cleanup: "with a single existence proof --- the same cumulant
  model applied to..." → "with a single existence proof, the same cumulant
  model applied to..."; "and is flat beyond --- the same qualitative
  near-cloud behaviour that..." → "and is flat beyond, which is the same
  qualitative near-cloud behavior that...". Colon in the opening ("consumes
  only three ingredients: a spectrally resolved...") → "consumes only three
  ingredients, namely a spectrally resolved...".

---

## 6. Out-of-scope observations (no action requested here)

- 6_conclusion.tex is still an outline (1_intro.tex was drafted and
  review-polished 2026-07-29 — it now states the near-cloud QF-screening
  claim that 5.2 cites, closing 2.20); abstract, runningtitle,
  runningauthor, author-contribution and availability blocks are
  placeholders.
- After applying any batch of fixes, one full `pdflatex`+`bibtex` cycle is
  worth running to confirm zero "??" references; I checked labels statically,
  not by compiling.

## 7. Introduction (1_intro.tex) — reviewed against ADVISOR_REVIEW_CRITERIA (2026-07-31)

Criteria references are to `manuscript/ADVISOR_REVIEW_CRITERIA.md`. Checked
and clean: no results stated in the introduction; no overturning language,
prior work credited with specific gaps (needs cloud distance / needs cloud
context and heavy RT / statistics-based); register ban list clean apart from
the items below; every acronym defined at first use; all 22 citation keys
resolve in `refs.bib` with matching titles and years; all six `\ref` section
labels exist. The plan-mandated elements are present (option-B opener, 1-ppm
framing, preferential-screening sentence, "independently of the imager at
inference" wording).

**Items 7.1–7.13 APPLIED and removed 2026-07-31** (see the header record for
the choices made on 7.7 and 7.9). The full original findings with their
criteria mappings survive in this conversation's review and in git history
of this file. Only the citation-freeze items below remain OPEN.

### Citation verification — [verify]

- [ ] **7.14 Literature numbers (criteria §9) — NARROWED 2026-07-31.**
  CONFIRMED via publisher pages: King 2013 land ≈55 % / ocean ≈72 % (IEEE
  TGRS abstract) and Mauceri 2023 20 % land / 40 % ocean (AMT abstract).
  STILL OPEN: Massie 2021's "40 % of QF = 0 and 73 % of QF = 1 within
  4 km" — the exact sentence needs the paper body (the project's own
  41.7 %-within-4-km statistic agrees with the 40 %).
- [x] **7.15 Emde Part 1 vs Part 2 — RESOLVED and applied 2026-07-31.**
  Part 1 confirmed as "Synthetic dataset for validation of trace gas
  retrieval algorithms" (O2A + 400--500 nm spectra); the NO2-retrieval
  impact is Part 2. The intro sentence now reads "building a synthetic
  dataset for quantifying 3D cloud effects on ultraviolet--visible trace
  gas retrievals such as \ce{NO2}", accurate for the cited Part 1.

Mechanical note, no action needed: `sec:methods_photon_path` is defined in
both `3.2_PPDF.tex` and `tex/MANUSCRIPT_METHODS_PHOTON_PATH.tex`; harmless
as long as the `tex/` standalone stays out of the manuscript build.

---

## 8. Full-draft criteria review — Sections 2–5 and Appendices A–H (2026-07-31)

The ADVISOR_REVIEW_CRITERIA pass applied to the whole body (the intro was §7).
Checked and clean: "cloud radiative forcing" appears nowhere; the terminology
controls hold in live text (DE-MLP survives only in comments and the unbuilt
3.6 file; "candidates"/"in principle transferable"/"band-effective cumulant"
wording all correct); the headline chain 1.26→0.82 / 2.67→1.20 / 71 of 75 /
3.29→1.22 is identical everywhere it appears (4.2, 4.3.1, 4.3.2); the
55/54/29 % attribution matches across 4.4 and 5.1; appendix G/H disclaimers
match the plan; bibtex resolves every citation. Items below are ordered
substantive first.

### Substantive — [fix] unless marked

- [x] **8.1 CONFIRMED and FIXED 2026-07-31** (recomputed from the parquet;
  tex now reads 0.205/0.099) **— 4.2 quoted the wrong univariate R² pair**
  ([4.2_model_comparison.tex:20](manuscript/4.2_model_comparison.tex#L20)).
  "the \Rsq between ``\xcoraw $-$ \xcoprior'' and $\Delta$ \xcobc is only
  0.323 for ocean and 0.191 for land" — per the flow plan's 2026-07-23
  computation note, 0.323/0.191 belong to the BC-based departure
  (xco2_bc − apriori), which is NOT a model input; the raw-feature values
  are 0.205 (ocean) / 0.099 (land). Quoting the raw numbers also
  strengthens the "only" argument. Verify against the computation, then fix.
- [ ] **8.2 [fix] Date-split sentence asserts the leakage it prevents**
  ([3.4-data_split.tex:11](manuscript/3.4-data_split.tex#L11)). "Splitting
  by date keeps soundings from the same orbit in two different subsets"
  says the opposite of the design. Suggest "Splitting by date prevents
  soundings from the same orbit from appearing in two different subsets".
- [x] **8.3 APPLIED 2026-07-31 via 9.4** (bridge clause added to 3.4 line 15
  in the ship-removal batch) **— The 96-vs-75 station-day bridge is
  missing** (criteria §3
  referent). 3.4 line 15 introduces "96 TCCON station-days" and 4.3.1
  line 12 then opens with "these 75 station-days" — a demonstrative with
  no antecedent; the A-Train/free-drift split (75 + 21, Table D2) is never
  stated in the main text before Sect. 5.3. Suggest in 3.4: "96 TCCON
  station-days (75 during the A-Train period, which form the primary
  comparison of Sect.~\ref{sec:result-tccon_val}, and 21 during the Aqua
  free-drift period, evaluated in Sect.~\ref{sec:disc-img-free})".
- [x] **8.4 CONFIRMED and FIXED 2026-07-31** (tabC8 verifies 4.04→4.68;
  wording now "left slightly worse by the operational correction alone")
  **— 5.2 wording contradicted its own numbers**
  ([5.2_opt_bc_recovery.tex:36](manuscript/5.2_opt_bc_recovery.tex#L36)).
  "Flagged soundings ... gain the least from the operational correction
  (4.04 to 4.68\,ppm pooled)" — 4.04→4.68 is a degradation, not a small
  gain (4.68 is the bc value that 4.3.1 also quotes). Either the direction
  words or the numbers are wrong. If the numbers are right, suggest "and
  are left slightly worse by the operational correction alone (footprint
  RMSE 4.04 to 4.68\,ppm pooled)".
- [ ] **8.5 [fix] 5.4 paragraph 1 grammar failures**
  ([5.4_limitations_future.tex:5](manuscript/5.4_limitations_future.tex#L5)).
  The opening sentence is broken ("We demonstrate the ability of the
  \emph{DE} model correction compared to observation and preserve the \xco
  enhancement.") and the paragraph's last sentence makes the cloud
  properties the analyst and ends with a comma ("...cloud top height, have
  the potential to further analyze the relationship ... for future
  work,"). Suggest "The correction improves agreement with the independent
  references and preserves the tested plume enhancements. A few
  limitations should be noted." and "Beyond location and identification
  uncertainty, relating the spectral features to MODIS-retrieved cloud
  properties such as cloud optical thickness, effective radius, and
  cloud-top height is left for future work."
- [ ] **8.6 [fix] Gamma clause misstates the cumulant expansion** (criteria
  §2; [3.2_PPDF.tex:89](manuscript/3.2_PPDF.tex#L89)). "With the
  assumption of a gamma path model, expanding the log Laplace transform
  about zero absorption gives ..." — the expansion of $\ln \mathcal
  T_\lambda$ in cumulants is the general cumulant-generating-function
  expansion and requires no gamma model (the gamma assumption mattered
  only for the retired $\kappa$ identity). Drop the clause, or replace
  with a convergence qualifier ("for a path distribution with finite
  cumulants, within the convergence radius of the expansion").
- [ ] **8.7 [fix] Significance tools listed under the CV evaluation**
  ([3.5_evaluation.tex:9](manuscript/3.5_evaluation.tex#L9)). The Wilcoxon
  test, site-clustered bootstrap, and random-effects comparison are
  TCCON-side machinery, but they close the "Predicted Δ evaluation"
  subsubsection, where sites do not exist. Move the sentence to
  Sect. 3.5.2 or scope it ("For the observation comparison of the next
  subsection, statistical significance is established with ...").
- [ ] **8.8 [fix] Two broken sentences in 4.1.** Line 39: "This analysis
  defines the physical population since the flag rate is..." — the
  subject is the QF = 0 filter, not the analysis; suggest "The QF = 0
  filter instead defines the physical population, since the flag rate
  is...". Line 49: "separable from a single spectrum from a footprint
  spectrum alone" (duplicated phrase) → "separable from a single
  footprint's spectrum alone".
- [ ] **8.9 [draft] 4.2 opening lacks the planned date-blocked headline**
  (flow-plan §4.3: three sentences — numbers, ceiling, ordering). The
  current opening quotes no fold R²/RMSE values and never mentions the
  label-noise ceiling; the ceiling reading exists only in Appendix C. A
  reviewer reaching Table 1 has no absolute-skill anchor. Suggest adding
  after the first Table~\ref{tab:cv-model-comparison} sentence: "On the
  withheld folds the deep ensemble reaches an anomaly-target RMSE of
  0.40/0.54~ppm (R² 0.71/0.55) over ocean/land. Read against the
  label-noise reference levels of Appendix~\ref{app:model}
  (Fig.~\ref{app-fig:sfc_skill_ceiling}), ocean skill exceeds even the
  retrieval-posterior noise scenario, while the near-cloud land regime
  (achieved R² 0.72 against a stated 0.88) is the one regime with clear
  headroom." (Verify the exact frozen values from Table C5/Fig. C4 when
  applying.)
- [x] **8.10 CLOSED BY SCOPE 2026-07-31** (ship removal, §9: no direct
  comparison remains anywhere in the paper, so the anchoring sentence is
  moot) **— The direct-comparison anchoring sentence was never
  written.** Flow-plan §4.4 requires ONE sentence in 4.3.1 noting that
  direct (non-harmonized) TCCON comparisons were run and that the ~0.3 ppm
  scale difference is explained by the documented B7→B11 direct-TCCON
  anchoring chain, citing Appendix D — and Appendix D likewise carries no
  such note. Either add the sentence + short appendix note, or record a
  decision that the direct comparison is dropped from the paper entirely.
- [ ] **8.11 [draft] The mean-vs-median interpretation of Fig. 3 exists only
  as a comment** ([4.1_phenomenology_spectra.tex:14](manuscript/4.1_phenomenology_spectra.tex#L14)).
  The figure draws mean and median separately, and the planned diagnostic
  reading (ocean: whole distribution shifts; land: mean far exceeds the
  nearly unmoved median, so the land bias is carried by a skewed tail) is
  never stated in prose. Promote the commented sentence into the paragraph
  at line 7 or 19; it is load-bearing for the later tail-driven model
  comparison.
- [ ] **8.12 [fix] Appendix A uses undefined symbols.** $R_{cq}$ (the ILS
  weight), the channel index $c$, and the angles $\theta$/$\phi$ in
  Eqs. (A2)–(A3) are never defined in the appendix (the definitions were
  lost with the commented-out old draft). One sentence after
  Eq.~\eqref{eq:hr_sod} covers all four.
- [ ] **8.13 [fix] TCCON is "corrected"**
  ([4.3.1_tccon_val.tex:12](manuscript/4.3.1_tccon_val.tex#L12)). "all
  TCCON \xco are corrected following the description in Sect. 3.5.2" —
  the reference is harmonized, not corrected; the current verb invites
  confusion with the DE correction. → "harmonized following".
- [x] **8.14 FIXED 2026-07-31** (caption now cites
  `Table~\ref{tab:label-noise}`) **— figC4 caption pointer** ([appendix_C.tex:174](manuscript/appendix_C.tex#L174)).
  The caption's closing "(Sect.~\ref{sec:method-dataset-split})" points at
  the data-split section, which says nothing about noise scenarios; the
  supporting text is the appendix-C paragraph and Table~\ref{tab:label-noise}.
  Probably a mispointer.
- [ ] **8.15 [style] "truth" overstates twice.** 3.3 line 7 "The \delxcobc
  is used as the truth for training" (3.1 itself owns that the target can
  contain real gradients) → "as the training target"; 2_data line 12
  MYD35 "as the truth for cloud locations" (Appendix F documents MYD35
  false positives) → "as the cloud-location reference".

### Numbers to re-freeze — [verify] (ALL CHECKED 2026-07-31 against the
production artifacts; details in the header record)

- [x] **8.16 Wilcoxon p — tex correct** (CSV 0.00633 → 0.0063; the
  flow-plan/§7 0.0064 was stale).
- [x] **8.17 The 5.2 raw/BC/ML paragraph — all values verified correct**
  against tabC8 and the RAW_BC_ML_TCCON_2026-07-27 report (the plan's
  60 %/+0.86 and the ledger's 3.35→3.84 were earlier editions).
- [x] **8.18 Appendix G wavelength set — resolved** (tabG1: 33 total = 3
  continuum anchors + 30 log-spaced, τ ≈ 0.03--8; G prose aligned to the
  table in the dash edit).
- [x] **8.19 Largest-improvement cell — tex correct** (50 km × ±30 min,
  Δ|bias| −0.93, vs −0.77 at 25 km; the plan's "tightest radii" was loose).
- [x] **8.20 Freeze-time sweep — fully verified** (headline, QF-resolved,
  Ny-Ålesund incl. per-case extremes, far-cloud strata, drift chain with
  exact Wilcoxon p 0.0263 / 1.34e-5, poleward counts exact). One fix
  applied: excl-Ny "p ≤ 0.016" → "p ≤ 0.017" (actual max bootstrap p
  0.0162).
- [ ] **8.21 [verify] Land label-retention limitation.** Appendix B states
  the labeled land population under-represents persistently cloudy scenes
  (53 % retention); the flow plan deferred whether 5.4 should carry this
  limitation too. 5.4 currently does not. Decide.

### Register and style — [style]

- [ ] **8.22 Ban-list sweep** (criteria §6). "notable because"
  (4.2:20 → "matters because"), "the most essential role" and "also
  crucial for the prediction" (4.2:36), "In other words" (4.1:7, 4.2:20,
  5.1:12; the 5.4:12 instance is covered by the existing Rayleigh
  rewrite), "significantly decreases" without a test (4.3.1:30 — quote
  the magnitude and drop the adverb, or attach the test), "more
  importantly" (4.3.4:17 → drop), "Notably," (appendix G:28 → drop),
  "is worth stating" (3.5:15 → "One implementation detail is not part of
  the cited procedure and moves the reference by about 1 ppm"), "That is
  to say," (4.4:6 → drop or "In other words" family — just delete).
- [ ] **8.23 Hedges where the evidence is sharp** (criteria §4). 4.1:7 "the
  cloud-influence distance could be different over ocean and land" — the
  preceding sentences just showed it is → "therefore differs between ocean
  and land"; 5.4:9 "which could be wrong if a local \xco gradient exists"
  → "which fails where a real local gradient exists".
- [ ] **8.24 Grammar/wording batch, per file.** (PARTIAL 2026-07-31: the
  ship-removal batch applied the 4.3.1:4 modifier fix (9.6), the 4.4:6
  opening rewrite in its airborne-only form (9.8), the 4.3.3:5
  "considering that" fix (9.7), and subsumed the 2_data:16 and
  appendix_D:14 items in their ship-free rewrites (9.3/9.11); the rest of
  the batch is still open.)
  2_data:16 "as additional OCO-2 ocean footprints comparison" → "for
  additional comparison of OCO-2 ocean footprints".
  3.1:61 + 3.4:7 "Sect.~\ref{...}, ~\ref{...}" → "Sects.~\ref{...}
  and~\ref{...}".
  3.2:1 subsection title sentence case + missing article ("Spectrum-internal
  features from the photon path-length distribution function"); 3.2 is in
  past tense (derived/normalized/fitted/solved) while 3.1/3.3 are present —
  unify; "conservative reference atmosphere" → gloss "(non-absorbing)".
  3.3:7 stray subscript comma "$\mathrm{X_{CO2,}^{raw}}$" (×2, and use the
  \xcoraw/\xcoprior macros); "normalized and standardized" → "standardized".
  4.2:18 unbalanced parenthesis "departure \xcoraw $-$ \xcoprior)".
  4.2:36 "increase in $\Delta$RMSE" — the increase IS the ΔRMSE → "increase
  in held-out RMSE (ΔRMSE)"; "to see the increase" → "to measure".
  4.1:19 "The signature of opposite sign of ocean and land \delxcobc also
  remains the same" → "The opposite-signed land and ocean responses remain".
  4.3.1:4 "over the ocean, where TCCON has no coverage in
  Sect.~\ref{...}" — misplaced modifier → "over the ocean, where TCCON has
  no coverage (Sect.~\ref{...})".
  4.3.1:12 "After understanding the general performance" → "Having
  established the model ordering in Sect.~\ref{sec:result-model_comp}";
  "to obtain more statistics" → "for sample size".
  4.3.1:26 aggregate symbol mismatch $\overline{RMSE}$ vs $\overline{R}$ in
  one paragraph — pick one.
  4.3.1:30 "confirming that our correction works in the right direction" →
  "consistent with a cloud-proximity-driven correction".
  4.3.3:5 "considering that all TCCON stations are still over land and only
  part of them are close to the ocean coastline" → "since all TCCON
  stations are on land and only some sit near coastlines"; ship "legs" here
  vs "cases"/"days" elsewhere — unify.
  4.4:6 opening two sentences ("demonstrate the ability of the \emph{DE}
  model correction" / "The concerns of changes in far-cloud footprints are
  also minor.") → "The TCCON, airborne, and shipborne comparisons establish
  that the correction improves agreement with independent references, and
  the far-cloud controls show it is nearly inert where it should be.";
  4.4:13 "the four cases shown here are not a selection" — referent: name
  them ("the four windows of Fig.~\ref{fig:plume-preservation}d").
  5.1:6 "bound the correction as a safety guard" → "bound what the
  correction may remove"; "cannot simply fill" → "cannot fill".
  5.1:8 "only tells that the retrieval drifted" → "indicates only that".
  5.1:15 "They can provide more information than cloud influence." → "They
  carry information beyond cloud influence."
  5.3:4 "demonstrates the validation of our correction method" →
  "validates the correction against".
  5.3:6 + H:4 "spectrally resolved high-resolution absorption band" —
  redundant doubling, once per file → "a spectrally resolved absorption
  band"; 5.3:6 "show high potential for application to" → "are applicable
  in principle to".
  appendix_A:21 "on finer wavelength grids" → "on the high-resolution
  wavelength grid".
  appendix_D:14 "with each station or ATom, or shipborne data" → "with
  each TCCON station, with ATom, and with the shipborne references".
- [x] **8.25 APPLIED 2026-07-31 — leftover unicode dash sweep** (survivors/regressions after the
  4.7 sweep). En dashes: 4.2:18 "0.8–1.2 ppm", fig08 caption "2014–2021",
  figB2 caption "anomaly–distance", figD5 caption "DerSimonian–Laird",
  appendix G "x–z"/"15–22 km"/"3–4 km"/"9.5–14.5 km", 4.3.3:7 "0.10–0.19
  ppm". Em dashes in captions/prose: figE2 caption "— the overpass
  crosses", figF3 caption "— all-band response", appendix G lines 13/15/30
  (the G dash item above covers two of the three). Replace with `--`/`---`
  or reword per the no-dash rule.
- [ ] **8.26 "sign flip" wording** ([5.1_physics_interp.tex:10](manuscript/5.1_physics_interp.tex#L10)).
  "the vegetated-versus-barren sign flip in the W\ce{CO2} band" — the
  terminology table prefers "sign rule" (the retired claim was the "forest
  sign flip"); suggest "the vegetated-versus-barren sign rule in the
  W\ce{CO2} band".
- [x] **8.27 APPLIED 2026-07-31 — G double-space residue** fixed.
- [ ] **8.28 3.5 does not predefine the sensitivity variants.** The
  25/50/100 km × ±30/60/120 min grid first appears in Results 4.3.1; the
  flow plan makes Methods 3.5 the single home of the coincidence protocol.
  One sentence after the 100 km/±1 h definition ("Sensitivity of the
  validation to this choice is tested over 25/50/100 km and ±30/60/120
  min") closes it.

---

## 9. Scope decision — shipborne removal and MC RT promotion (2026-07-31)

### Discussion record

Second advisor's feedback: the manuscript carries too much material
(think about scope), and the Monte Carlo RT simulation should be
explained more. User direction: drop the shipborne comparison, since it
alone forces the direct-vs-AK reference distinction to be explained.
Assessment recorded from the discussion (agreed by user):

- **Drop ship fully, not demote.** Its headline number is a scatter
  collapse (0.63 → 0.27 ppm), which is the evidence class §4.3.2's
  smoother null itself disarms; it is the sole exception to the
  AK-harmonized-only reference basis (2026-07-21 decision), forcing the
  §3.5 caveat and the hedged +0.99 → +1.18 ppm offset explanation; a
  demoted appendix remnant would still need the direct-reference
  explanation. Removing it also resolves **8.10 by scope** — no direct
  comparison remains anywhere in the paper, so the B7→B11 anchoring
  sentence is moot. What is lost: one of two ocean references and the
  clear-day negative-control type (2019-06-22); ATom keeps the ocean
  story with its own far-cloud null (2017-10-09), and the TCCON
  far-cloud strata provide a second null.
- **MC RT is a visibility problem, not a detail problem.** Appendix G is
  thorough, but the main text gives the experiment ~10 lines at the end
  of 3.2, so a reader meets the closure numbers before the experiment.
  Plan: promote (9.17 below). The sign reversal (⟨l′⟩ 0.78 → 0.41 dark /
  0.99 → 1.17 bright, only albedo changed) is the paper's strongest
  causal demonstration and deserves main-text presence. With ship
  dropped, Fig. 11 shrinks to ATom-only, so a promoted MC figure keeps
  the main-text figure count at ~12.
- **Next trim candidates if more is needed** (not decided): Appendix H
  (TEMPO; weakest by the admission rule, kept 2026-07-22 by user
  decision) and Fig. 7 permutation importance (prose half-disavows it;
  demote to Appendix C). Protect the smoother null, plume audit, and
  uncertainty section.

### DONE 2026-07-31, second pass (prose batch 9.1–9.16 APPLIED, user-approved)

- Original tex of every touched file preserved in
  `manuscript/backup/ship_removal_2026-07-31/tex_originals/` (11 tex files
  + the pre-regeneration `tabB1_cohort_attrition.tex`).
- Applied 9.1 and 9.3–9.12 as drafted. The 9.4 recount was re-verified by
  script from Table D2 before editing: 82 distinct dates with ship, 79
  without (ship-unique 2019-06-14, 2019-06-22, 2021-03-15; 2019-06-09
  shared with ny/xh; FYI 2017-02-06 is shared between ATom and et). The
  8.3 bridge clause went in with the same edit.
- Sweep extras not in the original plan, also applied: the
  3.1_cloud_dis_xco2_anom.tex line 61 "TCCON, aircraft, and shipborne
  references" sentence (now "TCCON and aircraft references"), and the ship
  row in generated `tabB1_cohort_attrition.tex`. 4.3.1 line 4 "three
  independent references" → "two". `tex/MANUSCRIPT_APPENDIX_DATES_TABLE.tex`
  still carries a ship block but is not \input anywhere (archive copy);
  left alone.
- 9.13: generators edited with SHIP-REMOVAL markers
  (`make_appendix_c_figures.py` EVAL_TREES ship entry removed;
  `make_appendix_tables.py` tabB1 ship row removed and the tabD4 registry
  entry commented out so a default run cannot resurrect the ship table;
  tabD4/tabD6 functions kept for the dissertation). Regenerated tabC4 and
  tabB1; diff vs backup confirms the only change is the ship row/line.
- Full pdflatex ×3 + bibtex rebuild: exit 0, 81 pages, zero undefined
  references or citations (`grep -a` on main.log), no multiply-defined
  labels. The four staged dangling ship references are cleared.
- Remaining ship residue is intentional and 9.14-scoped: the commented
  fig11b/figE3/tabD4 includes, the unused `\xcoship` macro (main.tex:94),
  the fig11a filename, and the uncited knapp2020/hanft2021 bib entries.

### DONE 2026-07-31 (mechanical staging; prose batch below awaits go-ahead)

- Moved to `manuscript/backup/ship_removal_2026-07-31/`:
  `fig11b_ship_summary.{png,pdf}`, `figE3_ship_clearday_control.png`,
  `tables/tabD4_ship_cases.tex`.
- Commented the three includes (marker `SHIP-REMOVAL 2026-07-31` in
  4.3.3 / appendix_D / appendix_E).
- Interim build compiles; exactly FOUR dangling references remain
  (`app-fig:ship_far_cld` in 4.3.3:10 + appendix_E:24;
  `tab:ship-cases` in 4.3.3:5 + appendix_D:183) — cleared by 9.7/9.11/9.12.
  NOTE discovered doing this: `main.log` contains non-text bytes, so
  plain `grep` silently returns nothing — use `grep -a` on it. Re-checked
  with `-a`: no other undefined references or citations exist.

### 9.x — Ship-removal edit plan (drafts dash/colon-free)

- [x] **9.1 APPLIED 2026-07-31 — 1_intro.tex** ¶6 + ¶7: "against TCCON, aircraft, and shipborne
  references" → "against TCCON and aircraft references" (both places).
- [x] **9.2 NO ACTION NEEDED 2026-07-31 (the abstract and conclusions outlines contain no ship text; flow plan updated per 9.16) — Drafting notes** (no tex yet): abstract move 5/7 and the
  conclusions outline drop "shipborne"; flow-plan §2.3/§4.6 entries
  updated per 9.16.
- [x] **9.3 APPLIED 2026-07-31 — 2_data.tex** §2.3 line 16: remove the shipborne EM27/SUN
  sentence part (MORE-2, MR21-01, \citep{knapp2020more2_pangaea,
  hanft2021mr2101_pangaea}); suggest "In addition to TCCON data, we also
  used airborne in-situ measurements from the Atmospheric Tomography
  Mission (ATom) to create pseudo-column \xco (\xcoatom) for comparison
  with OCO-2 ocean footprints."
- [x] **9.4 APPLIED 2026-07-31 — 3.4-data_split.tex**: line 5 drop "and from two shipborne
  campaigns (MORE-2, MR21-01)"; line 15 "96 TCCON station-days, 8 ATom
  flight days, and 4 shipborne days" → drop ship and RECOUNT "82
  distinct comparison dates" → **79** (ship-unique dates 2019-06-14,
  2019-06-22, 2021-03-15; 2019-06-09 is shared with the ny/xh TCCON
  cases — confirm the arithmetic when applying). Same edit should add
  the 8.3 bridge clause (75 A-Train + 21 free-drift).
- [x] **9.5 APPLIED 2026-07-31 — 3.5_evaluation.tex**: remove \xcoship from the harmonization
  opening (line 15); delete the two ship sentences of lines 17–18
  ("Since the averaging kernel and prior for shipborne ... directly. As
  a result ... shipborne comparison."); remove "or the ship-track
  center" from the collocation definition (line 20). The AK-only
  comparison basis is then uniform.
- [x] **9.6 APPLIED 2026-07-31 — 4.3.1 lead-in** (lines 4–5): "We also compare the correction
  with airborne and shipborne column references over the ocean, where
  TCCON has no coverage" → "We also compare the correction with an
  airborne column reference over the ocean, where TCCON has no
  coverage" (keep the 8.24 modifier fix: "(Sect.~\ref{...})").
- [x] **9.7 APPLIED 2026-07-31 — 4.3.3_ocean_far_cld.tex** (largest edit): line 5 remove the
  ship identification and the \xcoship sentence, pointer → "(Table~\ref{tab:atom-legs})";
  line 7 delete the ship result sentences (scatter collapse, offset,
  small-sample note — the §5 rewrite drafts for them lapse); line 10
  drop `\ref{app-fig:ship_far_cld}` ("Figure~\ref{app-fig:atom_far_cld}
  illustrates ..."); Fig. 11 caption → ATom-only (a, b); subsection
  title can stay ("Ocean references and far-cloud controls").
- [x] **9.8 APPLIED 2026-07-31 — 4.4_plume.tex** line 6: "Validation with TCCON stations and
  additional comparisons with airborne and shipborne measurements
  demonstrate ..." → fold into the 8.24 rewrite of the same sentence,
  airborne only.
- [x] **9.9 APPLIED 2026-07-31 — 5.3_imager_indepent.tex** line 4: "with TCCON, ATom, and
  shipborne observations" → "with TCCON and ATom observations".
- [x] **9.10 APPLIED 2026-07-31 — 5.4_limitations_future.tex** line 7: "which is why we decided
  to include ATom and shipborne comparisons in addition to TCCON" →
  "which is why we include the ATom comparison in addition to TCCON",
  PLUS the candid selectivity clause (criteria §4) "the ocean validation
  therefore rests on a single aircraft campaign".
- [x] **9.11 APPLIED 2026-07-31 — appendix_D.tex**: subsection title D1 and intro line 14 drop
  shipborne; Table D2 remove the Shipborne block (4 dates) and update
  its caption count 82 → 79; app:ocean_inventory text line 183 →
  "Table~\ref{tab:atom-legs} lists the collocated ATom legs ...".
- [x] **9.12 APPLIED 2026-07-31 — appendix_E.tex**: line 24 → single control ("Figure~\ref{app-fig:atom_far_cld}
  documents the ocean negative control of Sect.~... at the footprint
  level."); figE3 block already commented; E-figure renumbering joins
  the final 4.17 pass.
- [x] **9.13 APPLIED 2026-07-31 — tabC4_manifest_verification**: regenerate without the ship
  row (`make_appendix_c_figures.py --only tabC4`, or equivalent flag).
- [ ] **9.14 Cosmetic at the final renumbering pass**: fig11a filename,
  freed figE3/tabD4 numbers, `\xcoship` macro in main.tex, and the
  now-uncited knapp2020/hanft2021 bib entries (harmless meanwhile).
- [x] **9.15 APPLIED 2026-07-31 — §5/§8 items that lapse with the ship text**: the 4.3.3
  "in-situ measurements" and ship-offset rewrite items, and the
  "limited temporal and spatial overpass samples" item (its sentence is
  deleted); strike them when applying 9.7.
- [x] **9.16 APPLIED 2026-07-31 — MANUSCRIPT_FLOW_PLAN.md**: record the scope change (ATom-only
  §4.3.3 and Fig. 11, §2.3/D/E content, abstract/conclusion notes,
  8.10 closed by scope) as a dated entry.
- [ ] **9.17 MC RT promotion (companion batch, approve separately):**
  (a) expand the 3.2 closing paragraph into a short named subsection
  stating the design (one scene, 3D vs IPA solver switch, production
  estimator refit) before the closure numbers; (b) NEW main-text figure,
  condensed from Fig. G1's ⟨l′⟩ row (dark/bright × 3D/IPA, sign reversal
  visible), generator `workspace/rt_slab_sim/make_fig_g1.py` extended
  with a `--main-panel` variant; (c) add a PPDF-distribution panel to
  Appendix G from the existing S6 assets
  (`results/rt_slab_sim/figs/slab_ppdf_{dark,bright}.png`,
  `plot_ppdf.py`) so the "photons detouring at l′ up to 1.3" sentence
  has its figure.

---

## 10. Addition 2026-08-01 — footprint-size (spatial-resolution) analysis (Appendix B, Fig. B5)

### Analysis record (RAN 2026-08-01, local; no CURC cycle)

User-requested addition: influence of the OCO-2 footprint size/area on the
near-cloud XCO2 bias and its correction. Figure goes in the APPENDIX, not
the main text (user decision 2026-08-01). Code:
`workspace/fp_area_analysis.py` (steps qc/decay/spectral/residual/numbers;
CSV outputs under `results/figures/cld_dist_analysis/fp_area/`, headline
numbers in `fp_area_headline.md`) + figure generator
`manuscript/scripts/make_fp_area_figure.py` →
`manuscript/figures/figB5_fp_area_robustness.{png,pdf}` (file number; the
print number joins the final renumbering pass).

Axis provenance and QC: `fp_area_km2` is the equal-area shoelace area of
the L2 Lite vertex polygons (`src/spectral/fitting.py`). It is real
geometry, not vertex noise (within-frame std ~0.02 km2 across the 8
footprints vs ~0.84 km2 across frames; along-track lag-10 autocorrelation
0.997), essentially uncorrelated with nearest-cloud distance
(|r| < 0.03 both surfaces), but correlated with SZA/VZA (ocean r ~ 0.34)
and, on land, the area strata sample different scene mixes (median cloud
distance 36 km in Q1 vs 6 km in Q3; `fp_area_qc_confounds.csv`) — so the
controlled regression below, not the raw stratification, carries the
quantitative claim. QC window [0.2, 10] km2 screens 5.3 % (the small-area
tail is uniform in year and footprint index and is not interpreted).
Target screen |y| <= 100 ppm everywhere, matching
`models.pipeline.filter_target_outliers` (training population).

Verdicts (numbers in `fp_area_headline.md`):

1. **The near-cloud anomaly ATTENUATES with footprint area at fixed
   nearest-cloud distance on both surfaces.** Geometry-controlled OLS per
   1-km distance bin (controls SZA + |lat|): ocean +0.079 ± 0.002
   ppm km-2 at 0.5 km (the anomaly is negative, so positive = weaker);
   land −0.271 ± 0.025 ppm km-2 at 0.5 km (anomaly positive near cloud in
   the screened population's median sense; negative = weaker).
2. **Zone-confined, therefore a real near-cloud interaction, not a scene
   confound:** the coefficient collapses to |≤0.004| (ocean) / |≤0.02|
   (land) ppm km-2 beyond each surface's response zone (~4 km / ~8 km).
3. **The edge-proximity (distance-shift) hypothesis is REJECTED:** decay
   curves of different area quartiles do not collapse under a distance
   shift (best-fit shift ~0 km); the size effect is amplitude, not
   offset. Larger footprints dilute rather than amplify the perturbation
   at fixed center distance.
4. **The correction ABSORBS the dependence (fold-safe held-out mu, all
   116 dates, 10 fold models, leakage discipline as everywhere):**
   near-cloud residual bias across area quartiles is flat after
   correction — ocean span 0.105 → 0.020 ppm (before −0.377…−0.272,
   after −0.022…−0.003), land span 0.105 → 0.051 ppm; near-cloud fp-RMSE
   reduction is uniform-to-growing with area (land Q4 1.57 → 0.67 ppm,
   the largest improvement).
5. Spectral effect sizes by area quartile (Fig. 4 machinery, QF0
   snow-free): ocean sign-stable in every quartile (Δexp ~ −0.25σ, mild
   Q4 attenuation, consistent with verdict 1); the land WCO2 swing
   across quartiles (−0.21σ → +0.45σ) is the land-cover sign rule
   aliasing through the strata scene mix — state as a confound, do not
   interpret as footprint physics. CSVs `fp_area_spec_{ocean,land}_
   effect_sizes.csv`; not a figure panel.

### 10.x — Edit plan (tex drafts await approval per the no-unasked-tex rule)

- [ ] **10.1 — 4.1_phenomenology_spectra.tex**: one robustness sentence
  where Fig. B4 is cited, e.g. "The response is likewise robust to
  footprint-size variation. Stratifying by the reported footprint area
  leaves both response zones unchanged, and the residual dependence, an
  attenuation of the near-cloud anomaly with footprint area, is confined
  to each surface's response zone (Appendix B)."
- [ ] **10.2 — appendix_B.tex**: Fig. B5 with a ~150-word paragraph
  covering the axis provenance and QC window, the confound table, the
  controlled-coefficient result, the rejected shift hypothesis, and the
  residual-flatness panel. Caption descriptive-only per the caption rule.
- [ ] **10.3 — 3.1 (or 2.2)**: one sentence owning the center-based
  distance definition: the footprint extent gives the distance axis an
  ambiguity of order the footprint semi-axis; Appendix B quantifies the
  resulting footprint-size dependence and shows the correction absorbs it.
- [ ] **10.4 — 5.3_imager_indepent.tex (Tier 3)**: one clause — the weak,
  zone-confined footprint-area dependence suggests footprint scale is a
  second-order factor for transfer among few-km2 missions (OCO-3, CO2M);
  extrapolation to much larger footprints (GOSAT-class) is untested.
- [ ] **10.5 — bookkeeping**: figB5 files exist; add to the appendix-B
  figure list at the final renumbering pass (print number TBD there).
- [x] **10.6 [fix] APPLIED 2026-08-01 (option a, user go-ahead "fix 10.6")
  — Fig. 3 land MEAN curve was carried by 192 catastrophic outlier
  rows.** Original finding: the unscreened land bin means at 1–5 km
  (+1.2 to +1.8 ppm, the "mean far exceeds the nearly unmoved median"
  contrast) collapse when the 192 rows with |anomaly| > 100 ppm are
  removed — 192 of 7.2 M land soundings, ALL QF1, ALL within 15 km of
  cloud, median |anomaly| ≈ 3,974 ppm (max 6,529): catastrophic
  retrieval failures, the class that poisoned the Ridge CV mean. The
  training chain always screened them (`filter_target_outliers`,
  |y| <= 100 ppm), so no model or skill number moves; ocean has zero
  such rows; medians/IQR untouched.
  **Deeper correction found while applying: the within-screen land story
  REVERSES the drafted skew.** Screened land r15: the MEDIAN is positive
  throughout (+0.13 at 0–1 km, +0.15 at 3–4 km, +0.04 at 14–15 km — the
  r15 motivation survives via the median; r10 median +0.05 at the 10-km
  cut), but the 0–2 km bin MEAN is NEGATIVE (−0.51 at 0–1 km). The old
  "0.5 ppm at the 10-km cut" and "tail-driven land bias" sentences were
  outlier artifacts.
  **Mean–median decomposition (user follow-up, RAN 2026-08-01; land
  <3 km screened, n = 92,849; mean −0.16 / median +0.13):** the
  QF0 snow-free population (47 %) is SYMMETRICALLY positive (mean =
  median = +0.17 ppm — no skew at all in the quality-passing scenes);
  the negative mean is carried by a heavier negative tail — 10.4 % of
  soundings below −2 ppm (band mean −5.1, contributing −0.53 ppm to the
  overall mean vs +0.25 from the +2 ppm tail) — that is 88 %
  quality-flagged and 12 % snow (QF1 snow-free stratum: mean −0.33,
  median +0.04; snow: mean −1.24, median +0.07). Shadowing is the
  systematic secondary axis (shadow-branch mean −0.30 vs brightening
  −0.04; medians +0.04 vs +0.19) but only mildly enriches the deep tail
  (52 % vs 47 % baseline shadow fraction) — the deep tail is a flagged
  in-FOV-scattering population, consistent with the §4.1 QF-population
  paragraph's own framing. The §4.1 sentence was refined accordingly
  (quality-passing soundings consistently positive; sub-−2-ppm tail 88 %
  flagged / 12 % snow; shadowed systematically lower than brightened).
  Applied (originals in
  `manuscript/backup/fig3_screen_2026-08-01/tex_originals/`):
  (i) `make_anomaly_decay_figure.py` screens all three target columns at
  `models.pipeline.MAX_ABS_ANOMALY_PPM` (prints 236/164/192 screened for
  r10/r05/r15); fig03, fig03alt, figB2 regenerated; (ii)
  `4.1_phenomenology_spectra.tex` prose rewritten to the median-positive
  / mean-negative-under-2-km reading with the Fig. 5 forward link, the
  0.5-ppm claim replaced by the median +0.05 value, "land median
  response" in the panel-b sentence, caption gains the screen sentence,
  stale comment replaced by the corrected note; (iii)
  `3.1_cloud_dis_xco2_anom.tex` gains the magnitude-screen sentence
  after the n_min/σ_max guards (192 labels, all flagged land, none
  ocean; cites Table B2); (iv) `tabB2_target_params.tex` regenerated
  with a "Target outlier screen" row (generator reads
  MAX_ABS_ANOMALY_PPM — cannot drift). Full pdflatex+bibtex rebuild:
  exit 0, 82 pages, no errors, no undefined references. Also 2026-08-01
  (user request): the panel-b legend-key annotation moved lower
  (axes y 0.55 → 0.42) clear of the curves; figures regenerated and the
  build re-verified after the sentence refinement. NOTE for 10.1: its
  §4.1 robustness sentence should attach to the NEW prose.

- [x] **10.7 APPLIED TO MANUSCRIPT 2026-08-01 (user request "add this into
  manuscript" + define bright/neutral/shadow).** The script was RERUN with
  the manuscript's THREE-CLASS branch definition (z_exp = Δexp-int_O2A /
  σ_ref, thresholds ±0.5, neutral kept — spec_sensitivity's z_thresh; the
  binary-split numbers below are superseded for branch rows by:
  ocean QF0-snowfree shadowed −0.24 / neutral −0.08 / brightened −0.06;
  land QF0-snowfree +0.06 / +0.15 / +0.26; two-way land dark×shadowed
  −0.12 is the only negative land cell, bright×brightened +0.28; ocean
  QF0 modulators AOD −0.07→−0.21, ws −0.09→−0.18). Tex applied
  (originals + README in `manuscript/backup/bias_sign_2026-08-01/`):
  (i) 4.1 gains the full z_exp classification definition (physical
  reading of each class, the ±0.5σ rationale) before the Fig. 5
  sentence + a condition-resolved sign paragraph after the Fig. 5 float
  (two-way cells, AOD/wind/SZA modulators, aerosol sign flip under the
  contrast rule) + the branch-sentence typo fix; (ii) Fig. 5 caption
  rewritten to the three-class definition (<10 km window, 10-km
  reference); (iii) 5.1 gains one synthesis sentence after the MC
  paragraph (dark×shadow = ocean-like endpoint; aerosol obeys the same
  contrast rule). Build clean, 82 pages. Original analysis record
  (binary split) kept below for provenance.
  **ADDENDUM 2026-08-01b (user decision): Fig. 5 unified to the
  production reference.** `spec_sensitivity.py` gained `--reference
  production` (shadow-only; per-surface r05/r15 aliasing via the
  `_R05/_R15_PAIRS` machinery + the 100-ppm anomaly screen; writes to
  `spec_sensitivity/prodref/`; the r10 default path is unchanged) and
  the full-parquet shadow analysis was RERUN LOCALLY; the fig05
  generator (`make_shadow_brightening_figure.py`) defaults to the
  prodref stats, with `--reference common-r10` preserving the CURC
  originals as `internal_shadow_brightening_r10_*`. RESULT CHANGE: the
  land branches now ORDER the anomaly (brightened +0.31 / neutral +0.14
  / shadowed +0.01 ppm at 0–1 km) instead of carrying opposite signs
  (r10 shadowed was −0.08); Δ⟨l′⟩ branch responses remain
  opposite-signed (O₂A brightened +0.006 vs shadowed −0.004; WCO₂
  orientation flipped — shadowed positive — consistent with the WCO₂
  sign rule). §4.1 branch sentence reworded (ordered branches, ~0.3 ppm
  separation, negative only over dark surfaces) and the Fig. 5 caption
  now names the r15 reference + screen. Fig. 5 + internal ocean
  companion regenerated; build clean, 82 pages. Tex snapshot before this
  batch: `manuscript/backup/bias_sign_2026-08-01/tex_originals/
  4.1_phenomenology_spectra_post10.7.tex`. `workspace/bias_sign_conditions.py`
  → `results/figures/cld_dist_analysis/bias_sign_conditions/` (binned stats,
  Spearman table, two-way splits; near-cloud ≤5 km, screened targets,
  all-flag / QF0-snow-free / QF1 populations). Verdicts: (1) OCEAN negative
  everywhere, amplified by shadowing (−0.43 vs −0.20 brightening), aerosol
  (top AOD quintile −0.58 vs −0.15), wind speed / dim glint (−0.40 to −0.45
  at 5–7.6 m s⁻¹ vs −0.15 below 3.9), and dark glint (−0.44 darkest vs
  −0.15 brightest albedo quintile). (2) LAND QF0 snow-free positive in
  every condition bin EXCEPT dark-surface shadow: the two-way
  albedo-tercile × branch split flips sign only in the dark × shadow cell
  (−0.10; all other cells +0.12 to +0.28, max bright × brightening) — the
  albedo-contrast mechanism at XCO2 level. Positive bias grows with
  surface brightness (alb_wco2 darkest +0.02 → brightest +0.30) and high
  sun (SZA 16–30° +0.26 vs 56–81° +0.11). (3) AOD flips sign with surface
  (ocean more negative, land mildly more positive) — aerosol behaves like
  contamination following the same contrast rule. (4) The all-flag land
  negative mean is QF1/snow tails (10.6 record); bright QF1 scenes
  (alb_o2a > 0.47) carry the largest negative tail (mean −0.62, median
  +0.08) — consistent with the bright-surface failure stratum. Caveat:
  associations, not partialled; conditions inter-correlate (ws↔glint
  albedo, alb↔land cover, SZA↔latitude/season).

---

## Suggested application order (remaining work)

1. DONE 2026-07-31: §9 ship-removal batch 9.1–9.13 + 9.15–9.16 (the four
   staged dangling references are cleared; build clean). Next: the 9.17
   MC-promotion batch (awaits separate go-ahead).
2. §8 substantive items still open: 8.2, 8.5–8.9, 8.11–8.13, 8.15, plus
   the author decision 8.21 (land label-retention limitation in 5.4).
   8.3 applied via 9.4; 8.10 closed by scope with §9.
3. Remaining §5 + §8 style batches: 8.22 ban-list sweep, 8.23 hedges,
   8.24 grammar batch (minus the parts applied via §9; see its PARTIAL
   note), 8.26 sign-rule wording, 8.28 sensitivity-grid predefinition
   in 3.5.
4. 7.14 residual (Massie 2021 40 %/73 % sentence, needs the paper body).
5. §10 footprint-size batch: 10.6 APPLIED 2026-08-01 (screen + reworded
   §4.1/§3.1 + tabB2 row; build clean). Remaining: 10.1–10.4 tex drafts
   await approval (attach 10.1 to the NEW §4.1 prose; analysis + figB5
   artifacts are DONE 2026-08-01).
6. Final appendix renumbering pass (4.17 + 9.14 + 10.5), then a full
   `pdflatex`+`bibtex` cycle (grep the log with `-a`).

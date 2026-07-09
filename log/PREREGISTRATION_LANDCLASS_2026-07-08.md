# Pre-registered predictions — land-class × spec-feature analysis (full parquet)

**Date frozen:** 2026-07-08 (commit `543d6cf`). **Timeline correction (same
day, before scoring):** the full-parquet CURC run had in fact ALREADY
completed and been downloaded to `results/figures/cld_dist_analysis/` when
these predictions were committed — so this is a **pre-inspection**
registration, not a pre-run one: the predictions were written and committed
before any person or tool read `landclass_effect_sizes.csv`, the effect
heatmap, or any per-class profile from that run (only directory listings and
the bin-count totals were touched to establish scale). The only stratified
result known when P1–P5 were written was the earlier subset-scale first look
(forest-vs-bright-surface flip), disclosed below as P1's provenance.
Predictions may NOT be edited after this note — score them as written.
(TODO_ACCOMPLISH §2-2.)

**Scored against:** `land_class` outputs `*_effect_sizes.csv` /
`*_effect_heatmap.png` — near-cloud (0–5 km) − far (20–50 km) effects per
(variable × land-cover group), z-units with analytic 95 % CI; groups =
forest / savanna / shrubland / grassland / cropland / barren / wetland /
urban / snow_ice (water + fill excluded). Preference: the ref-corrected
Δ-variables (`dk1_*`, `dk2_*`, `dexp_*`) where populated; raw far-baselined
variables otherwise.

## Physical premise

The near-cloud 3D-RT perturbation of a footprint's spectrum is driven by the
**cloud–surface albedo contrast in each band**,
`C_band(class) = α_cloud(band) − α_class(band)`.
Liquid-water cloud albedo falls steeply with wavelength across our bands
(≈ 0.6–0.7 at O2A 0.76 µm; ≈ 0.4–0.55 at WCO2 1.61 µm; ≈ 0.2–0.4 at SCO2
2.06 µm, where droplets absorb), while surface albedo moves the other way
for barren/desert (rising into the SWIR) and drops steeply for vegetation
(bright NIR plateau at 0.76 µm, dark at 2.06 µm). Brightening (C > 0) adds
scattered photons and path-length spread; when C ≤ 0 the cloud is no brighter
than the surface and the brightening channel shuts off (shadowing remains).

## Predictions

- **P1 — SCO2 sign flip (headline).** Near-cloud Δk1 and Δ(exp-intercept) in
  the **SCO2 band** are **positive over dark-in-SWIR classes** (forest,
  wetland, cropland) and **near zero or negative over bright-in-SWIR classes**
  (barren, shrubland), because α_cloud(2.06 µm) exceeds vegetation SWIR albedo
  but not desert albedo. (Consistent with the subset-scale first look; the
  full-parquet run is the confirmatory test.)
- **P2 — no flip in O2A.** O2A-band Δk1 is positive for **every** snow-free
  class (clouds at 0.76 µm are brighter than all snow-free land surfaces).
  Exception allowed: `snow_ice` (C ≤ 0 there; sign unconstrained).
- **P3 — monotone contrast ordering.** Across snow-free classes, per band,
  the near-cloud Δk1 effect (z-units) increases with the measured contrast:
  Spearman rank correlation between class effect size and
  `α_cloud(band) − median clear-sky α_class(band)` (class-median band albedo
  from our own parquet, cld_dist ≥ 20 km) is **positive** in all three bands,
  taking α_cloud = 0.65 / 0.50 / 0.30 (O2A/WCO2/SCO2). Only the *ordering*
  is claimed; the α_cloud constants just fix the ranking direction.
- **P4 — k2 is non-negative.** Δk2 near cloud is ≥ 0 (within CI) for all
  classes in all bands — added geometric spread can only widen the
  path-length distribution — with magnitude increasing with |C_band| (same
  Spearman test as P3 on |effect|).
- **P5 — WCO2 is intermediate.** The class-ordering in WCO2 sits between O2A
  and SCO2: no (or marginal) sign flip, but a compressed forest-vs-barren
  spread relative to SCO2 in z-units.

**Falsification:** any CI-significant class×band cell with sign opposite to
P1/P2, or a non-positive Spearman in P3, counts against the mechanism claim
and must be reported in the manuscript as such — no post-hoc class
re-grouping. If effects are CI-consistent with zero at full scale, the
mechanism section reverts to the pooled (non-stratified) evidence only.

**Prediction provenance:** albedo spectra premise from standard liquid-cloud
/ vegetation / desert reflectance behaviour; P1's direction was suggested by
the subset-scale first look (forest-vs-bright-surface flip), so P1 is a
*confirmatory replication* at scale, while P3–P5 are novel orderings not yet
inspected in any output.

---

## SCORED (2026-07-08, same day — full-parquet run, `landclass_effect_sizes.csv`)

Scored on the ref-corrected Δ-variables as pledged (dk1_*/dk2_*/dexp_*,
z-units, 95 % CI); class-median clear-sky band albedos measured from 40 local
date parquets (1.15 M clear land soundings, MCD12C1 v061 classes):
forest/savanna dark → barren bright in SWIR (alb_wco2 0.15/0.20 → 0.61;
alb_sco2 0.06/0.11 → 0.59), exactly the premise — and barren is the only
class whose WCO2 albedo EXCEEDS the assumed cloud albedo (0.61 > 0.50),
i.e. the contrast changes sign in WCO2, not SCO2.

- **P1 — FAIL as written.** The vegetated-vs-barren sign flip is real but
  lives in **WCO2**, not SCO2: dk1_wco2 savanna +0.43σ / grassland +0.27 /
  shrubland +0.23 / forest +0.09 vs barren **−0.40** (all CI-significant).
  In SCO2 the effects are an order weaker and partly reversed (forest
  −0.029 ± 0.025 — CI-significant with the WRONG sign; barren −0.021).
  dexp_sco2 is also sign-reversed vs prediction (forest −0.41, barren +0.06).
- **P2 — PARTIAL.** dk1_o2a ≥ 0 (CI) for savanna/shrubland/grassland/
  cropland/barren; forest consistent with zero; **urban fails** (−0.26,
  CI-significant; urban is anomalous in every variable — n_near = 3555,
  likely aerosol/3D-structure confound — but it was not exempted, so P2
  is a partial fail as written).
- **P3 — FAIL formally.** Spearman(effect_z, α_cloud − α_class): O2A +0.25
  (p = 0.59), WCO2 +0.07 (p = 0.88), SCO2 −0.04 (p = 0.94), n = 7 — none
  positive-significant. Salvage worth reporting: in WCO2 the **sign** of
  dk1 follows the sign of the measured contrast for every non-urban class
  (barren is the only negative-contrast class and the only negative dk1),
  but magnitudes do not rank with contrast within or across bands.
- **P4 — FAIL.** dk2 is strongly class-signed, mirroring dk1: dk2_wco2
  vegetated +0.10…+0.44 vs barren **−0.37** and urban −0.86 (CI-significant
  negatives); small significant negatives also in forest o2a/sco2. Added
  path-length *variance* is not non-negative for bright surfaces.
- **P5 — FAIL.** WCO2 has the LARGEST class spread (it is where the
  contrast changes sign), not an intermediate one; SCO2 is weak.

**Interpretation for the manuscript (per the falsification clause):** the
stratified evidence supports the qualitative albedo-contrast mechanism only
as a *sign* rule, and only where the measured surface albedo actually
straddles the cloud albedo (WCO2). The naive per-band contrast-ordering
model is rejected: SCO2 — where the contrast is most negative over barren —
shows almost no k1 response, so band-dependent measurement sensitivity
(SNR/absorption depth) must co-determine magnitudes. Report P1–P5 as
scored; do not soften. Full-scale spec-sensitivity survives independently:
spec-only near-cloud classifier AUC land 0.718 / ocean 0.664 (HGB,
full_spec; n_train 2 M per surface), sub-pixel monotonicity and
shadow/brightening branches regenerated at scale.

**Post-scoring methodological note (raw vs ref-corrected — explains the
first look):** at full scale the RAW far-baselined variables reproduce the
subset first look (wco2_k1: forest −0.20σ, barren +0.17σ) while the
ref-corrected deltas show the OPPOSITE class signs (dk1_wco2: forest +0.09,
savanna +0.43, barren −0.40). The raw near−far contrast within a class is
confounded by population shift (near-cloud forest scenes differ in
region/season/moisture from far-cloud forest scenes); the per-sounding
clear-sky-neighbor deltas remove that scene baseline and are the
defensible cloud response — they are what the prereg pledged and what the
manuscript should use. The subset "first look" direction was an artifact
of raw baselining, which is itself worth one manuscript sentence.

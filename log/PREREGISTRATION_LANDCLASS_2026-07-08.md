# Pre-registered predictions — land-class × spec-feature analysis (full parquet)

**Date frozen:** 2026-07-08, BEFORE submitting the full-combined-parquet run
(`sbatch curc_shell_blanca_combined_analysis.sh`, which regenerates
`run_all.py --land-class` + `spec_sensitivity.py` on
`combined_2016_2020_dates.parquet`, 17.8 M soundings).
The git commit containing this file predates the run; that hash is the
timestamp. Predictions may NOT be edited after submission — score them as
written. (TODO_ACCOMPLISH §2-2.)

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

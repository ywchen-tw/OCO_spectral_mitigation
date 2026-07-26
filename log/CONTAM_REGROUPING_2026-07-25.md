# CONTAM_FEATURES regrouped by physics (2026-07-25)

**Trigger:** re-reading Mauceri, Massie & Schmidt (AMT 16, 1461, 2023) against
`src/models/pipeline.py` showed the `CONTAM_FEATURES` ablation group was defined by
*provenance* (whatever the forward-selection round happened to add) rather than by
*physics*. The group was wrong in both directions, which undermines the
`no_contam` conclusion quoted in `manuscript/4.3_model_comparison.tex`
("dropping the spectral or contamination groups is TCCON-neutral").

**Evidence base:** 3 × 2018 per-date parquets (`combined_2018-02-21`, `-03-13`,
`-04-10`), 465 067 soundings — 170 825 land / 294 242 ocean. Spearman throughout;
cloud-distance correlations computed on soundings with a valid `cld_dist_km`.

---

## 1. Membership rule (new)

A feature belongs to `CONTAM_FEATURES` iff it diagnoses the **scene's scattering
content or its spatial inhomogeneity**:

| | Class | Members |
|---|---|---|
| (a) | retrieved cloud/aerosol optical depth | `aod_water`, `aod_ice`, `aod_dust`, `aod_oc`, `aod_seasalt`, `aod_sulfate`, `aod_strataer` |
| (b) | layer height retrieved with it | `water_height`, `ice_height`, `dust_height` |
| (c) | inter-sounding radiance variability | `h_cont_o2a/wco2/sco2` (Massie HC), `csnr_o2a/sco2` (CSNoiseRatio), `max_declock_wco2` |
| (d) | operational cloud screens | `dp_abp` (A-band preprocessor), `co2_ratio_bc`, `h2o_ratio_bc` (IMAP-DOAS) |

Everything else — met prior, surface state, geometry, retrieval-state departures —
stays in the base list but is *attributed* to the retrieval-state/surface group.

**Boundary cases deliberately excluded** (brightness/consistency terms, surface-driven
as much as scattering-driven): `s31`, `alb_sco2_over_wco2`.

### 1b. Addendum (same day) — the IMAP-DOAS ratios were moved IN

The first cut of criterion (d) read "the A-band cloud screen → `dp_abp`" and left
`co2_ratio_bc` / `h2o_ratio_bc` out, on the grounds that a band-ratio *consistency*
diagnostic is one causal step removed from "there are scatterers here", and that
Mauceri et al. (2023) list `h2o_ratio` among retrieval-state/bias-correction features
rather than among their four 3D-cloud metrics. Both points stand, but they are
outweighed:

1. **The pair cannot be split.** OCO-2 screens clouds with *two* preprocessors
   (Taylor et al., 2016): the ABP, which yields `dp_abp`, and IMAP-DOAS, which yields
   `co2_ratio` / `h2o_ratio`. Keeping one half of the operational cloud screen inside
   the group and the other half in the base set is not defensible under any rule.
2. **Measured, on ocean it is a first-order leak.** Partial Spearman with
   `cld_dist_km` (rank-residual after controls):

   | | raw \|ρ\| | ctrl `tcwv`,\|lat\| | + ctrl `aod_water`,`dp_abp` |
   |---|---|---|---|
   | `h2o_ratio_bc` ocean | 0.293 | **0.327** | **0.256** |
   | `co2_ratio_bc` ocean | 0.168 | 0.185 | 0.127 |
   | `h2o_ratio_bc` land | 0.086 | 0.183 | — |
   | `co2_ratio_bc` land | 0.055 | 0.136 | — |
   | *scale:* `aod_water` ocean / `dp_abp` ocean / `h_cont_o2a` land | 0.252 / 0.232 / 0.524 | | |

   On ocean `h2o_ratio_bc` is an independent proximity channel as strong as `aod_water`
   and `dp_abp` — both already in the group — and ocean is where the group is thinnest
   (no `h_cont_o2a/sco2`, no `aod_ice`, no ice/dust heights). On land it is second order
   (partial 0.18 / 0.14 vs 0.49 for `h_cont_o2a`), where `|ratio−1|` instead tracks
   retrieved scatterer load (ρ 0.36 with `aod_water`).
3. **Mechanism note worth one sentence in the paper.** On ocean the signal lives in the
   *signed* ratio (0.29) and not in `|ratio−1|` (0.035); on land the reverse. That is
   what a directional, wavelength-dependent path-length difference over a dark surface
   should look like, versus an unsigned "the two-band solution is inconsistent"
   indicator over bright land.

**Cost accepted:** the IDP ratios are also genuine operational *bias-correction*
predictors, so a **degraded** `no_contam` would be ambiguous between "contamination
information matters" and "we removed a bias-correction predictor". Tie-break if that
happens: one extra arm = current group minus the two ratios. A **neutral** result needs
no caveat and is strictly stronger for having removed both operational screens.

## 2. Moved OUT (were in the group, are not contamination indicators)

| Feature | L2 Lite source | Why it does not belong | Number |
|---|---|---|---|
| `t700` | `Retrieval/t700` | GEOS met-prior 700 hPa temperature — a prior field cannot diagnose scene contamination; it is a latitude/airmass proxy | ρ(t700, \|lat\|) = **−0.85** land, −0.73 ocean; ρ(t700, tcwv) = 0.70 |
| `alt_std` | `Sounding/altitude_stddev` | DEM sub-footprint terrain roughness — static per location, identical clear or cloudy | ρ with `alt` = 0.39 land; its ρ = 0.26 with cloud distance is geographic confounding |
| `fs_rel_0` | `Retrieval/fs_rel` | Solar-induced chlorophyll fluorescence relative to Band-1 continuum — biosphere/surface state (**not** relative humidity; the comment in `build_feature_dataset.py` said so and was wrong, fixed 2026-07-25) | ρ with cloud distance = **0.08** — carries essentially no proximity information |
| `dpfrac` | `Retrieval/dpfrac` | Duplicate of `dp_psfc_prior_ratio`, which sits in the retrieval-state group — one variable split across two ablation groups | ρ(dpfrac, dp_psfc_prior_ratio) = **0.965** land / **0.978** ocean |
| `alb_sco2_over_wco2` | derived from `Retrieval/albedo_*` | Band-albedo ratio = surface brightness; Mauceri et al. (2023) use `albedo_wco2` as a bias-correction feature, never as a contamination indicator | ρ with cloud distance 0.46 land / 0.14 ocean |

## 3. Pulled IN (are contamination diagnostics, were left in the base)

This is the load-bearing half: with the old grouping, `no_contam` **did not remove
the contamination channel on land**.

| Feature | Why it belongs | Number |
|---|---|---|
| `h_cont_o2a`, `h_cont_sco2` | The same Massie HC metric as `h_cont_wco2`, which *was* in the group. Dropping one of three near-collinear siblings removes no information. They are also the two strongest cloud-proximity features in the entire predictor table. | ρ(h_cont_wco2, h_cont_o2a) = **0.79**, ρ(h_cont_wco2, h_cont_sco2) = **0.93** (land); \|ρ\| with cloud distance 0.52 / 0.50 — vs **0.08** for `aod_water`, which was dropped |
| `csnr_o2a`, `csnr_sco2` | `color_slice_noise_ratio_*` = Mauceri's `CSNoiseRatio`, one of their four named 3D-cloud-effect variables | \|ρ\| with cloud distance 0.40 land (o2a) / 0.31 ocean (sco2) |
| `aod_dust`, `aod_oc`, `aod_seasalt`, `aod_sulfate`, `aod_strataer` | The aerosol half of "cloud/aerosol contamination"; previously split from `dust_height`, which was inside the group | \|ρ\| with cloud distance up to 0.36 (sulfate, land) / 0.29 (seasalt, ocean) |
| `co2_ratio_bc`, `h2o_ratio_bc` | The IMAP-DOAS half of the operational cloud screen, whose ABP half (`dp_abp`) was already in the group — see §1b | ocean `h2o_ratio_bc` partial ρ **0.26** after controlling `tcwv`, \|lat\|, `aod_water`, `dp_abp` |

**Land cloud-proximity ranking, old grouping** (`in` = dropped by old `no_contam`):
`h_cont_o2a` 0.52 (out), `h_cont_sco2` 0.50 (out), `t700` 0.49 (in), `h_cont_wco2`
0.49 (in), `alb_sco2_over_wco2` 0.46 (in), `s31` 0.46 (out), `csnr_o2a` 0.40 (out),
`max_declock_wco2` 0.39 (out — ocean-only feature), … `aod_water` 0.08 (in),
`fs_rel_0` 0.08 (in), `ice_height` 0.08 (in). The old set dropped weak members and
retained strong ones — a null result was close to guaranteed.

## 4. What changed in code

- `src/models/pipeline.py` — `CONTAM_FEATURES` redefined (20 names, union over both
  surfaces); base `_FEATURES_SFC0/1` **byte-identical** (34 ocean / 45 land, same
  order), so `full` and every production checkpoint are unaffected. Comment blocks in
  the base lists relabelled "forward-selected additions" so position is no longer read
  as group membership.
- `src/analysis/build_feature_dataset.py` — `fs_rel_0` comment corrected
  (fluorescence, not relative humidity). No data change.

Drop counts, old → new (ocean drops one fewer IDP ratio because `co2_ratio_bc` is a
land-only feature):

| Set | Ocean | Land |
|---|---|---|
| `no_contam` | 7 → **12** (34→22 features) | 12 → **17** (45→28) |
| `no_contam_and_xco2` | 8 → 13 | 13 → 18 |

`full`, `no_xco2`, `no_spec`, `no_xco2_and_spec` are unchanged.

## 5. What has to be redone

1. **Retrain** the `no_contam` / `no_contam_and_xco2` arms, both surfaces, 5 folds:
   `curc_shell_blanca_de_profile_foldpca_r05.sh` (ocean, the loop at :125) and
   `curc_shell_blanca_de_profile_foldpca_r15.sh` (land, :125). Suffixes are unchanged,
   so **archive the old dirs first** to keep the 2026-07-17 numbers reproducible:
   `for d in results/model_deep_ensemble/de_*_no_contam*_prof_foldpca_r*_f*; do mv "$d" "${d}_oldcontam"; done`
2. **Rebuild** the variant plot-data trees (`workspace/build_ablation_variant_trees.sh
   no_contam` / `no_contam_and_xco2`) and regenerate
   `results/model_comparison/deep_ensemble/FEATURESET_ABLATION_QF_*.md`
   (`workspace/make_featureset_ablation_doc.py`). Archive
   `FEATURESET_ABLATION_QF_2026-07-17.md` — its `no_contam` column is now historical.
3. **Manuscript:** Table C1 (`manuscript/tex/MANUSCRIPT_APPENDIX_PREDICTOR_TABLE.tex`)
   re-sectioned to the new grouping, and the §4.3 sentence rewritten once the new
   numbers land. Expect the contamination ablation to stop being neutral — the point of
   the fix is that the land model can no longer fall back on `h_cont_o2a/sco2`.
4. **Attribution note** (`log/no-contam-ablation-result` memory, `PROJECT_REVIEW.md`
   §3.2 M6 bonus finding, `SPEC_EMPHASIS_STATUS`): the claim "contamination features
   are free to drop" is **withdrawn pending the rerun** — it was measured with a group
   that left the strongest contamination diagnostics in place.

## 5b. RESULT OF THE RERUN (2026-07-25, same day)

Retrained on CURC: `no_contam` + `no_contam_and_xco2`, both surfaces, all 5 folds,
lndo01 + fold-PCA (the 07-17 land-f4 config leftover is gone). Every fold verified
against the new grouping before use (`n_qt` 22/21 ocean, 28/27 land; no
contamination feature survives as an input anywhere). Both variant trees rebuilt
(75/75 cases, zero failures, leakage guard clean) and all three report editions
regenerated → `FEATURESET_ABLATION_QF_2026-07-25.md` (QUOTABLE; supersedes the
07-17 edition's two contamination columns only). Old trees kept as
`de_prof_mix_no_contam{,_and_xco2}_oldgroup`.

**Verdict: `no_contam` is still TCCON-neutral — and now the claim means something.**

ΔRMSE vs full (ppm, AK reference, footprint-weighted; + = worse):

| slice | no_spec | **no_contam (new)** | *no_contam (old group)* | no_xco2 | no_contam_and_xco2 |
|---|---|---|---|---|---|
| pooled, QF 0+1 | +0.021 | **−0.025** | *+0.028* | +0.793 | +1.226 |
| pooled, QF=1 | +0.036 | **−0.057** | *+0.033* | +1.180 | +1.779 |
| land ≤10 km, QF 0+1 | +0.029 | **−0.052** | *+0.030* | +0.973 | +1.494 |
| land ≤10 km, QF=1 | +0.041 | **−0.083** | *+0.034* | +1.282 | +1.926 |
| land ≥10 km, QF 0+1 | −0.013 | **+0.079** | — | +0.067 | +0.112 |
| ocean, QF 0+1 | +0.015 | **+0.036** | *+0.009* | +0.152 | +0.188 |

Station-equal mean |bias| (AK): full 0.731 → `no_contam` **0.687**; QF1 0.819 →
**0.771**. On the *direct* reference it goes the other way (0.502 → 0.546).

Case-level check (station-day rows, site-clustered bootstrap 10k, paired Wilcoxon;
this weights every station-day equally, unlike the footprint-weighted table above):

| variant vs full | per-case fp-RMSE Δ | 95 % CI | Wilcoxon p | \|bias\| Δ |
|---|---|---|---|---|
| `no_spec` | +0.001 | [−0.020, +0.023] | 0.76 | −0.002 |
| `no_contam` | +0.043 | [−0.036, +0.125] | 0.002 | −0.040 |
| `no_contam_and_xco2` | +0.876 | [+0.616, +1.131] | <1e-4 | +0.253 |

So the sign of the `no_contam` effect **flips with the weighting** (footprint-weighted
−0.025, case-weighted +0.043) and every estimate is ≤0.05 ppm against a correction
that moves RMSE 3.29 → 1.22. That is the definition of neutral. The Wilcoxon p is
significant only because the small per-case differences are consistent in sign within
a few data-rich sites; the site-clustered CI straddles zero.

**Two findings that are new, and quotable:**

1. **The parsimony claim is now defensible.** Under the old group, "contamination is
   droppable" was near-guaranteed by construction (land kept `h_cont_o2a/sco2`).
   The regrouped `no_contam` removes **both operational cloud screens, all seven
   AODs, all three layer heights, and every inter-sounding variability metric** — and
   TCCON still does not move. That is a much stronger statement of the same claim.
2. **Contamination and XCO₂-departure are mutually redundant but jointly essential.**
   `no_xco2` costs +0.793 pooled; `no_contam` costs ~0; but `no_contam_and_xco2`
   costs **+1.226** — i.e. the contamination block adds +0.43 ppm of damage *on top of*
   removing xco2, having added nothing on its own. Under the old grouping this
   interaction was invisible (`no_contam_and_xco2` +0.896 ≈ `no_xco2` +0.793). The
   two blocks encode overlapping information about the same physical scene; either
   one suffices, losing both does not. Held-out R² agrees and is starker: land
   `no_contam_and_xco2` 0.232 vs `no_xco2` 0.420.

Held-out date-kfold (median over 5 healthy folds, R²): ocean full 0.727 →
`no_contam` 0.647 → `no_xco2` 0.607 → `no_contam_and_xco2` 0.506; land full 0.547 →
`no_contam` 0.499 → `no_xco2` 0.420 → `no_contam_and_xco2` 0.232. The 2026-07-08
lesson survives intact and is sharpened: **held-out anomaly R² over-credits the
contamination block relative to TCCON truth** — it is the single largest
validation-vs-TCCON discrepancy in the whole ablation.

## 5c. Downstream manuscript artifacts regenerated (2026-07-25)

All under `manuscript/`, which is gitignored — this log entry is the only tracked
record.

- **§4.3** rewritten around the new numbers. The old sentence "dropping
  *\xcoraw − \xcoprior* with spectral-fitting features shows the largest ΔRMSE
  increase" was **inverted by the rerun** and is corrected: the contamination
  combination is now the largest in every slice (+1.23 pooled / +1.43 land ≤15 km /
  +0.26 ocean ≤5 km, against +0.93 / +1.08 / +0.20 for the spectral combination).
- **Table C1** (`tex/MANUSCRIPT_APPENDIX_PREDICTOR_TABLE.tex`) re-sectioned to the
  new groups; verified row-by-row against `pipeline.py`.
- **Fig. 6** (`make_baseline_ablation_figure.py`) and the generated tables
  (`make_manuscript_tables.py`) rebuilt — they read the variant trees directly, so
  they picked up the new numbers automatically. The Table 2 caption now states the
  group's contents and the interaction result.
- **Fig. 7** (`make_feature_importance_figure.py`) needed a code fix: its bar
  colours came from a `group` column frozen into `importance_de_*_agg.csv` at
  permutation time, and the regrouping invalidated it for **15 of the plotted
  features** (`alt_std`, `fs_rel_0`, `dpfrac`, `alb_sco2_over_wco2`, `t700` out;
  the five aerosol AODs, `h_cont_o2a/sco2`, `csnr_o2a`, `co2_ratio_bc`,
  `h2o_ratio_bc` in). The figure now resolves the group from
  `src/models/pipeline.py` at plot time, so it cannot drift from the code again.
  The permutation **values** are untouched and remain valid — they are per-feature
  permutations of the unchanged production model.

**Known stale, deliberately not fixed:** the `scope=='group'` rows in
`results/model_comparison/feature_importance/*/importance_de_*_agg.csv` were
produced by permuting the OLD groups *jointly*, so they cannot be relabelled —
only recomputed. No manuscript figure or table reads them (checked); they would
need a CURC re-run of `models.feature_importance` before anyone quotes them.

**Caption mismatch spotted, NOT edited** (predates this work, needs an author
decision): the Fig. 6 caption in `4.3_model_comparison.tex` still says the dark
bars are "near-cloud land subset (≤ 10 km, n = 75,157)", but the figure has shown
three series at the production radii since 2026-07-23 — pooled (n = 105,683),
near-cloud ocean ≤5 km (n = 2,645) and near-cloud land ≤15 km (n = 81,347).

## 6. Reference

Taylor, T. E., et al.: Orbiting Carbon Observatory-2 (OCO-2) cloud screening
algorithms: validation against collocated MODIS and CALIOP data, *Atmos. Meas. Tech.*,
9, 973–989, 2016 — the operational screen is the ABP/IDP **pair**, which is why
`dp_abp`, `co2_ratio_bc` and `h2o_ratio_bc` share a group (§1b).

Mauceri, S., Massie, S., and Schmidt, S.: Correcting 3D cloud effects in XCO2
retrievals from the Orbiting Carbon Observatory-2 (OCO-2), *Atmos. Meas. Tech.*, 16,
1461–1476, 2023. Relevant points: their four dedicated 3D-cloud metrics (H3D, HC,
CSNoiseRatio, cloud distance) are *removed by recursive feature elimination*, and
their retained land/ocean features are retrieval-state (`dp`, `dp_abp`,
`co2_grad_del`, `h2o_ratio`, `albedo_wco2`) plus two cloud terms (`aod_water`,
`aod_ice`) — i.e. they explicitly separate "measures 3D cloud effects" from
"correlates with the residual bias". Our groups now follow the same distinction.

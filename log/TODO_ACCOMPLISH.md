# Manuscript Readiness Tracker — Done / To-Do / Storyline

**Created:** 2026-07-08 (from the advisor-style readiness assessment)
**Companions:** `log/PROJECT_REVIEW_2026-07-03.md` (reviewer concerns M1–M9, engineering §7),
`log/SPEC_EMPHASIS_STATUS_2026-07-08.md` (spec-feature repositioning),
`log/TCCON_PAPERS_AND_ML_IDEAS_2026-07-03.md` (bibliography + ML ideas).
**Verdict (2026-07-08):** the core TCCON-validated result is manuscript-ready;
start writing NOW in parallel with §2. The mechanism half (spec features) must
not be quoted until the full-parquet regeneration (§2-2) lands. Scoop risk
(Mauceri/Keely 2025 Parts 1+2) argues against gold-plating.

---

## 1. ACCOMPLISHED (evidence chain in place)

### Validation core
- [x] AK/prior harmonization (Rodgers–Connor; wet/dry bug found+fixed, CRITICAL_FIXES #11);
      reports against BOTH references: direct mean |bias| 1.08→**0.63** ppm,
      AK 1.26→**0.81**; fp-RMSE improved 71/75.
- [x] Significance surviving site clustering: station-day |bias| Wilcoxon
      **p = 0.0064**; bootstrap Δ mean |bias| −0.43 [−0.71, −0.13] p = 0.004;
      Δ fp-RMSE −1.46 [−1.93, −0.97] p = 0.0002.
- [x] Honest CV: date_kfold adopted; random-split inflation (ΔR² ≈ 0.29) documented.
- [x] Ocean validation at reduced n: ship EM27 (4 station-days, MORE-2 + MR21-01)
      + ATom pseudo-columns (8 dates / 17 legs); offsets explained
      (direct-vs-AK ~0.5, EM27 scale vintage, B11 regional).
- [x] TCCON report triples every metric by xco2_qf (all/qf0/qf1).

### Negative controls / correction safety (M1)
- [x] Far-cloud negative controls: ATom 2017-10-09; ship clear day 2019-06-22.
- [x] Nassar plume suite (`workspace/Nassar_plume_analysis/`): control-region
      null 5/7 consistent; 2 flags resolved as cloud (all-band k1 fingerprint);
      spec-channel plume-removal bound ≤ 0.21 ppm worst / ≤ 0.01 clear;
      Westar 2023-06-26 transect = preserved +0.6 ppm bump (money figure).

### Models / ablations
- [x] Production frozen: `de_beta_nll_prof_reg_o05l15_m5` (per-surface DE, M=5,
      beta-NLL, lndo01 reg, ProfilePCA, no-SG k, ocean r05 / land r15 targets).
- [x] Same-protocol baseline table: DE > XGB-mean > Ridge (near-cloud land tail
      fp-RMSE 1.30 < 1.68 < 2.37); TabM ≈ DE except that tail; 5-model
      land+ocean writeup (`MODEL_COMPARISON_land_ocean_2026-07-08.md`).
- [x] Feature-set ablation incl. QF-grouped: `no_spec` TCCON-neutral; xco2
      block the only one whose importance survives on TCCON → division-of-labor
      framing adopted (spec = mechanism / safety / MODIS-free sensitivity).

### Mechanism / phenomenology tools (numbers still subset-scale — see §2-2)
- [x] MCD12C1 v061 land-cover stratification (`land_cover.py`/`land_class.py`);
      first look: forest vs bright-surface near-cloud k1/k2 sign flip.
- [x] Spec-sensitivity suite (`spec_sensitivity.py`): sub-pixel monotonicity,
      shadow/brightening opposite branches, spec-only classifier AUC 0.66–0.69.
- [x] Case-study toolchain (`workspace/spec_case_study/`): cross-swath screener,
      GIBS-RGB single-case figures, vetted 19-case shortlist
      (7 land + 5 ocean + 5 MYD35 false positives + 2 cloud-no-bias),
      per-category appendix atlases; `run_all.py` trimmed ~20k → ~250 figs.

### Engineering / reproducibility
- [x] constants.py; shared trainer + training_dates.json manifests (written);
      reg ablation; fitting fast path (×14) verified; file splits; download
      stability; atomic writes.

---

## 2. MUST-DO BEFORE SUBMISSION (ranked; ~2–3 weeks incl. one CURC cycle)

1. [x] **Pure-smoother null (M4). DONE 2026-07-08.** `workspace/smoother_null.py`
   (feature-free orbit-local running mean, per segment × surface, input-screened
   with the same climatology criterion as the production guard; ±10/30/100 s
   columns) — wired into `build_deepens_plot_data.py` (`--smoother-windows-s`)
   AND retrofitted into all 75 local production case dirs. Report rerun locally
   4× (DE control + 3 windows, identical flags; DE control reproduces the
   production headline exactly). **Result (75 cases, AK ref):** smoother
   collapses footprint scatter MORE than the DE (2.23 → 0.35–0.66 vs 0.78 ppm)
   but leaves the case bias untouched (|bias| 1.26 → 1.20–1.24 vs DE 0.81 ppm);
   fp-RMSE DE 1.19 < smoother 1.34–1.52 ppm. Figure:
   `smoother_null/smoother_null_r100km.png` (`workspace/smoother_null_figure.py`).
2. [~] **Regenerate land-class + spec-sensitivity on the full combined parquet.**
   Pre-registration WRITTEN 2026-07-08 (`log/PREREGISTRATION_LANDCLASS_2026-07-08.md`:
   P1 SCO2 sign flip, P2 O2A no-flip, P3 Spearman contrast ordering, P4 k2 ≥ 0,
   P5 WCO2 intermediate; scored against `*_effect_sizes.csv`). REMAINING: commit,
   then user submits `sbatch curc_shell_blanca_combined_analysis.sh` on CURC.
3. [x] **Training-date leakage guard. DONE 2026-07-08.** `src/models/leakage_guard.py`
   (manifest union train+calib; filename-label date identity, time-column
   fallback; refuse by default, `--allow-train-overlap` loud override; missing
   manifest → warn "unverifiable") wired into all four builders
   (`build_deepens/baseline/tabm/structured_residual_plot_data.py`). Verified:
   all 77 TCCON eval dates ∩ 116 manifest dates = ∅ for BOTH production
   surfaces — the paper's one-sentence guarantee now runs on every `run_case`.
4. [~] **Label-noise ceiling.** `src/analysis/label_noise_ceiling.py` +
   `compute_xco2_anomaly(..., return_ref_stats=True)`; recomputed per-sounding
   ref stats reproduce the stored anomaly to ≤1e-6 ppm (grouped by orbit_id).
   Three bracketing ceilings: ref-only (weak), ref+retrieval-σ (headline),
   empirical far-field (no error model). Single-date check: land r15 all-rows
   ceiling ≈ 0.55 (achieved 0.39 → decent headroom framing TBD), ref+ret ≈
   empirical far-field on land (posterior σ explains the far-field variance).
   RUNNING: 140-local-date aggregate → `results/label_noise_ceiling_140dates.csv`.
5. [x] **M2 polish. DONE 2026-07-08 (except the manuscript sentence placement).**
   (a) Coincidence table `coincidence_sensitivity/coincidence_sensitivity.{csv,md}`
   (3 radii × 3 windows, rerun after the coordinate change): improvement
   significant at EVERY combo (site-clustered bootstrap p ≤ 0.004, Wilcoxon
   p ≤ 0.007); Δ|bias| −0.43 (100 km) to −0.95 ppm (50 km) — tighter
   coincidence improves MORE. (b) Published station coordinates: frozen
   `SITE_COORDS` table in `tccon_collocate.py` (from the GGG2020 files'
   constant per-spectrum metadata), preferred over median-obs with >5 km
   discrepancy warning; metrically a no-op (r100/w60 reproduces the production
   headline bit-for-bit), pure provenance. (c) QC sentence for the manuscript:
   *"All TCCON references are the GGG2020 `*.public.qc.nc` release files,
   which contain only spectra passing the network's official quality control
   (Laughner et al., 2024); we apply no additional screening beyond a gross
   validity window (300–550 ppm) and the stated coincidence criteria."*
6. [x] **25–55 % xco2-channel number OWNED. DONE 2026-07-08.** Variant plot_data
   (no_spec / no_xco2, prof_foldpca r05/r15 5-fold pools) built locally for all
   8 Nassar dates (leakage guard: all disjoint from training) + control-null
   rerun per variant. `nassar_plumes_variants/nassar_channel_attribution.{csv,md}`:
   full removes **54 % [36–64 %]** of plume-free control-window clear-sky
   spread; **no_spec 53 % (Δ +1 pp — the cumulants do NONE of the smoothing,
   consistent with the ≤0.21 ppm spec-channel plume bound); no_xco2 31 %
   (Δ −23 pp — the xco2 block carries ≈ half; the remaining ~31 % is the other
   retrieval-state/MET/geometry features).** Manuscript statement for §8a:
   quote 54 % [36–64 %] as the local-contrast smoothing plume/flux users must
   expect, attributed to the xco2-departure channel and NOT to the spectral
   features.

## 3. SHOULD-DO (strengthens, not blocking)

- [ ] **MODIS cloud-product dependence (2026-07-08 decision: literature-first).**
  The Discussion paragraph is carried by (a) the Cluster 9 citations
  (TCCON_PAPERS doc: Ackerman 1998/2008, Frey 2008, Holz 2008, Stubenrauch
  2013, Platnick 2017 — mask semantics, ~85–90 % lidar agreement weakest for
  thin cirrus / sub-pixel cumulus, product-to-product spread), (b) the
  structural argument (distance never a model input; TCCON/ATom/ship
  validation MODIS-independent; r05/r15 target variants = built-in criteria
  sensitivity), and (c) our MYD35 false-positive case studies (mask
  imperfection shown + spectra quiet there). **Time-permitting only:**
  Cloudy-only vs Cloudy+Uncertain distance rerun on a one-year subset →
  stability of anomaly-vs-distance curves. MYD06 cloud-top height = optional
  parallax bound (M5); MYD04 is aerosol (different purpose — relevant only to
  the far-field Δk1-vs-aerosol caveat).
- [ ] **Data-recovery paragraph:** read QF1-recovery numbers straight off the
  production report's qf tables ("N% of QF1 near-cloud soundings usable
  post-correction"). The "so-what" for flux people.
- [ ] Uncertainty-aware layer regen (`curc_shell_blanca_deepens_uncertainty.sh`
  pending submit) → per-case error bars + TOST equivalence.
- [ ] Writing-time subsections: Ny-Ålesund/high-latitude; five worsening sites;
  M5 parallax/advection bound sentences; M9(f) convergence-radius caveat.

## 4. EXPLICITLY DEFERRED (state in reply-to-reviewers if asked)

- Transport-model (CAMS/CT) regression of the label (ceiling + r05/r15
  sensitivity covers most of it) · EaR3T OSSE (follow-up paper) ·
  wind-resolved plume enhancement · σ(k1)/σ(k2) export (appendix QC) ·
  MCD12C1 purity-filter robustness row · FT-Transformer/TabPFN baselines.

---

## 5. MANUSCRIPT STORYLINE (planned 2026-07-08; supersedes PROJECT_REVIEW §5)

Target: **AMT**. Title shape: *"Correcting cloud-proximity biases in OCO-2
XCO2 with photon path-length statistics and deep ensembles, validated against
TCCON, aircraft, and shipborne observations."*

1. **Introduction.** OCO-2 XCO2 widely used (flux inversion, plume/emission
   quantification, trend monitoring) at sub-ppm accuracy requirements → but
   cloud-induced 3D-RT biases documented (Merrelli 2015; Massie 2017/2021/2023;
   Mauceri 2023; Chen 2025), and QF filtering discards near-cloud soundings
   disproportionately (cloudy tropics). Gap: operational B11 correction is not
   cloud-proximity-aware; prior ML corrections are general-purpose
   (Keely 2023; Mauceri/Keely 2025) or imager-dependent. Contributions: physics
   features from the photon path-length PDF; per-footprint, imager-free
   deployment (abstract sentence); triple independent validation; audited
   plume safety.
2. **Data.** OCO-2 L2 Lite v11 glint (2016–2020, 116 dates, 17.8M soundings),
   L1B radiances, MET/CO2-prior profiles; Aqua-MODIS MYD35 collocation and the
   nearest-cloud-distance definition (±time buffers, Cloudy+Uncertain, 1-km
   floor). TCCON GGG2020, ATom, shipborne EM27/SUN; MCD12C1 for stratification.
   **State here:** cloud distance is a *diagnostic and label-construction*
   quantity only — never a model input.
3. **Phenomenology.** Anomaly-vs-cloud-distance decay: **positive bias over
   land (within ~15 km), negative over ocean (within ~5 km)** — motivates the
   per-surface targets (land r15 / ocean r05). QF throughput vs distance.
4. **Photon path-length statistics.** Cumulant expansion of ln T vs slant
   optical depth → k1 (mean path enhancement), k2 (variance), exp-intercept −
   albedo (reflectance mismatch); no-SG fit in production, SG as robustness.
   Show the mechanism: k-vs-distance by band/surface; **land-cover-stratified
   effect sizes** (sign flip forest vs bright surfaces; albedo-contrast
   ordering — pre-registered); shadow vs brightening split (opposite-signed
   XCO2 branches). **Single-case showcase with Aqua RGB overlay** (Tasman
   exemplar in main text; category atlases in appendix: real clouds / MYD35
   false positives / clouds-without-bias). **Ordering decision (user,
   2026-07-08): this section stays BEFORE the correction results** — the
   exemplar figure therefore shows spec variables + raw/BC XCO2 response
   only (no corrected trace, no mu); the corrected transect appears later
   in the plume-preservation section (§8a).
5. **ML correction pipeline.** Within-orbit clear-sky anomaly target
   (definition, guards, r05/r15; limitations owned); feature groups (spec,
   retrieval-state, MET + profile EOFs, L1B/geometry, contamination);
   **methodological interlude:** random-split R² 0.82 → 0.53 under date-blocked
   CV; label-noise ceiling (§2-4). Per-surface deep ensemble, beta-NLL,
   Mondrian conformal.
6. **Model comparison and feature attribution.** Ridge → XGB → TabM → DE table
   (decided in the near-cloud land tail); feature-set ablation with the
   division-of-labor framing: retrieval-state features are the operationally
   sufficient predictor; the cumulants explain *why* they work, and are
   conditionally informative (no-xco2 numbers) though parsimony-droppable —
   their jobs are mechanism, safety audit, MODIS-free sensitivity
   (spec-only classifier AUC; sub-pixel result; NoMODIS era).
7. **Independent validation.** TCCON headline (both references, significance,
   qf0/qf1, calibration); high-latitude subsection (Ny-Ålesund owned);
   **pure-smoother null** (§2-1); ocean: ATom + ship with far-cloud/clear-day
   negative controls.
8. **Correction safety and limitations.**
   (a) **Plume preservation:** control-region null, Westar preserved transect,
   O2A-vs-CO2-band k1 discriminator, spec-channel bound ≤ 0.21 ppm; the
   25–55 % xco2-channel smoothing number stated for plume users (§2-6).
   (b) **MODIS cloud-product dependence (user point, 2026-07-08):** the
   *diagnosed bias curves and label construction* inherit MYD35's definition
   of "cloud" — Cloudy+Uncertain pooling, 1-km resolution floor, false
   positives over bright/dark surface features (our case studies show both
   the false positives and that the spectra stay quiet there); stricter
   criteria or MYD06-based masks would shift the distance axis and reference
   purity, NOT the correction itself (per-footprint, no cloud input) nor the
   TCCON/ATom/ship validation (independent of MODIS). Backing: Cluster 9
   citations (mask validation ~85–90 %, weakest exactly for thin cirrus /
   sub-pixel cumulus; product spread per GEWEX assessment), r05/r15 target
   sensitivity, parallax/advection bounds (M5 sentences); Cloudy-only rerun
   only if time permits (§3).
   (c) Label circularity/selection (M1 residuals), land-driven TCCON weighting.
9. **Data recovery and applications.** QF1 near-cloud recovery numbers;
   throughput in the cloudy tropics → flux-inversion relevance; OCO-3 /
   GOSAT-GW / CO2M transferability; post-2022 NoMODIS-era deployability.
10. **Conclusions.**

**Main-figure budget (~9):** (1) collocation/geometry schematic + decay curves,
(2) land-cover effect-size heatmap, (3) shadow/brightening split,
(4) case-study exemplar with RGB, (5) pipeline/CV design schematic,
(6) TCCON before/after headline, (7) significance/robustness panel,
(8) baseline + ablation table-figure, (9) plume preservation (transect + k1
contrast). Appendix: category atlases, coincidence-sensitivity, qf tables,
smoother null, SG-vs-no-SG, MODIS-sensitivity, high-lat.

# Manuscript Readiness Tracker — Done / To-Do / Storyline

**Created:** 2026-07-08 (from the advisor-style readiness assessment)
**Companions:** `log/PROJECT_REVIEW.md` (reviewer concerns M1–M9, engineering §7),
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
2. [x] **Full-parquet land-class + spec-sensitivity: RAN AND SCORED 2026-07-08.**
   The CURC job had already completed (downloaded to
   `results/figures/cld_dist_analysis/`; 2.44 M binned land rows, classifier
   n_train 2 M/surface — full scale). Pre-registration
   (`log/PREREGISTRATION_LANDCLASS_2026-07-08.md`) was therefore corrected to
   a PRE-INSPECTION registration (committed before anyone read the effect
   sizes) and scored as written: **P1 FAIL as written (the vegetated-vs-barren
   flip is real but lives in WCO2 — dk1_wco2 savanna +0.43σ vs barren −0.40σ —
   not SCO2; forest dk1_sco2/dexp_sco2 CI-significantly sign-reversed);
   P2 PARTIAL (urban −0.26σ fails; others hold); P3 FAIL formally (Spearman
   n.s. all bands; salvage: in WCO2, where measured albedos actually straddle
   the cloud albedo — barren 0.61 vs 0.50 — the SIGN of dk1 follows the sign
   of the measured contrast for every non-urban class); P4 FAIL (dk2_wco2
   barren −0.37σ, urban negative everywhere); P5 FAIL (WCO2 has the LARGEST
   spread, not intermediate).** Manuscript consequence: present the
   stratification as a sign rule valid where the contrast changes sign
   (WCO2), reject the naive per-band ordering, name band-dependent
   measurement sensitivity as co-determinant, flag urban as anomalous
   (n = 3.5 k, aerosol/3D-structure confound). Independently robust at scale:
   spec-only classifier AUC land 0.718 / ocean 0.664; sub-pixel monotonicity
   + shadow/brightening branches regenerated.
3. [x] **Training-date leakage guard. DONE 2026-07-08.** `src/models/leakage_guard.py`
   (manifest union train+calib; filename-label date identity, time-column
   fallback; refuse by default, `--allow-train-overlap` loud override; missing
   manifest → warn "unverifiable") wired into all four builders
   (`build_deepens/baseline/tabm/structured_residual_plot_data.py`). Verified:
   all 77 TCCON eval dates ∩ 116 manifest dates = ∅ for BOTH production
   surfaces — the paper's one-sentence guarantee now runs on every `run_case`.
4. [x] **Label-noise ceiling. DONE 2026-07-08.** `src/analysis/label_noise_ceiling.py`
   + `compute_xco2_anomaly(..., return_ref_stats=True)`; recomputed reference
   stats reproduce the stored anomaly to ≤ 4e-6 ppm (≤ 0.02 ppm worst single
   row on ocean r05, from the extra-vars reference-pool screening at build
   time — negligible for variance statistics). 21.5 M soundings / 140 dates →
   `results/label_noise_ceiling_140dates.csv`. **Production-target ceilings
   (max R² = 1 − E[σ²_noise]/Var(y); bracket = [empirical far-field,
   ref + retrieval-σ]):** ocean r05 all-rows **[0.52, 0.60]** vs achieved
   date-kfold ≈ 0.53 — **at the ceiling**; land r15 all-rows **[0.46, 0.61]**
   vs achieved ≈ 0.39 — 64–84 % of it. Near-cloud (< 10 km) ceilings are much
   higher (ocean 0.70, land 0.88–0.89) because signal variance dominates
   there: the global R² is noise-limited, not model-limited, and the headroom
   sits exactly where the signal is. Reference-mean sampling error
   (ref_std²/n_ref ≈ 0.003–0.014 ppm²) is negligible next to the retrieval
   posterior (σ² ≈ 0.20 ocean / 0.26–0.33 land ppm²). Manuscript sentence:
   "the date-blocked R² of 0.53/0.39 should be read against a label-noise
   ceiling of ≈ 0.5–0.6, not against 1."
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
- [x] **Uncertainty-aware layer: RAN AND READ 2026-07-08**
  (`tccon_uncertainty_r{100,50}km.{csv,md}`, qf-tripled, downloaded from CURC).
  DL random-effects D = corrected − AK TCCON: **−0.32 ± 0.10 ppm** (r100;
  r50 −0.43 ± 0.08, qf0 −0.21 ± 0.10) = the known ~0.3 ppm anchoring offset
  → globally consistent up to the documented absolute-scale chain.
  Between-case τ = 0.29 (r50) → 0.60 ppm (r100), I² 23 → 61 %: the
  unbudgeted systematic grows with radius ⇒ representation error (name as
  u_rep). Whole-budget ⟨z²⟩ = 1.90/3.04 (r50/r100): per-case budgets
  under-disperse ×1.4–1.7 because τ is missing; per-footprint σ already
  calibrated vs the anomaly target (Phase 2b k(cld_dist)). TOST δ=0.5:
  1/75 equivalent, 14/75 differ — quote the DL CI as the global statement,
  not per-case equivalence. Full read in `log/PROJECT_REVIEW.md` §3.1.
- [ ] **Manuscript figure pass (style LOCKED 2026-07-09; user-approved on 3
  vetted samples: bu 2018-10-24, iz 2019-03-13, ra 2016-09-11).**
  Shared module `workspace/plot_style.py`: Arial + Arial mathtext (AMT;
  unicode ₂ has no Arial glyph → mathtext $X_{CO2}$ everywhere), base 10 pt,
  CVD-safe maps (plasma XCO2/spec, magma σ, cividis cld-dist, RdBu_r μ),
  spectral labels ⟨l′⟩ / var(l′) (caption defines k1/k2 = cumulants of
  relative photon path). `plot_corrected_xco2.py` rebuilt: compact 2×3
  (Lite | ML-corrected | σ / μ | histogram | TCCON) + `_full` 3×3 (adds
  ideal-corrected + cloud-distance row) per run; per-map horizontal
  colorbars width-snapped to the aspect-locked (equal-km) maps; separate
  (a)/(b) bars sharing one norm; station ±100 km view extent
  (`--extent-radius-km`) + dashed 100-km collocation circle
  (`--hist-radius-km`); histogram legend in the cell below the panel, no
  ideal in histogram, TCCON-shading-driven ylim, no monthly mean.
  `plot_spectral_params.py` same conventions. **Remaining figure edits:**
  - [ ] ATom figures → same style (`workspace/ATom_analysis/plot_atom_comparison.py`,
        `atom_modis_overlay.py`, `atom_pseudo_column.py`)
  - [ ] Ship figures (`workspace/Ship_analysis/plot_ship_comparison.py`,
        `plot_ship_summary.py`)
  - [ ] Arial/style pass on remaining manuscript producers:
        `tccon_comparison_report.py`, `smoother_null_figure.py`,
        `spec_case_study` figure tools, `run_all.py` figures used in the paper
  - [ ] Batch re-render all 75 TCCON cases locally:
        `SKIP_BUILD=1 bash curc_shell_blanca_plot_corr_xco2_deepens.sh`
        (SKIP_BUILD guards the synced mu_XX member columns)
  - [ ] Pick compact vs `_full` per manuscript slot; poster figure still
        legacy styling (only relabeled to $X_{CO2}$)
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
   (d) **Future validation of the PPDF with a Monte Carlo model (user,
   2026-07-09):** the cumulant fit *interprets* k1/k2 as mean and variance of
   the photon path-length distribution function, but the paper never verifies
   that the fitted moments equal the physical ones — a 3D Monte Carlo RT model
   (EaR3T; Chen et al. 2025) can tally the actual per-photon path-length
   histogram in a cloud-adjacent scene, generate the synthetic OCO-2 spectra
   from the same photon ensemble, and compare the spectrum-fitted k1/k2
   against the directly tallied MC moments — closing the loop between the Laplace-transform
   derivation and the retrieval. Same machinery doubles as the plume OSSE
   (inject ΔCO2, verify k1/k2 invariance rigorously). State as future work in
   the Discussion (deferred to a follow-up paper per §4; one paragraph +
   citation here, not a result).
9. **Data recovery and applications.** QF1 near-cloud recovery numbers;
   throughput in the cloudy tropics → flux-inversion relevance; OCO-3 /
   GOSAT-GW / CO2M transferability; post-2022 NoMODIS-era deployability.
10. **Conclusions.** Close with the outlook sentence pair: MC/PPDF validation
    (§8d) and transfer to imager-free correction for OCO-3 / GOSAT-GW / CO2M.

**Main-figure budget (~9):** (1) collocation/geometry schematic + decay curves,
(2) land-cover effect-size heatmap, (3) shadow/brightening split,
(4) case-study exemplar with RGB, (5) pipeline/CV design schematic,
(6) TCCON before/after headline, (7) significance/robustness panel,
(8) baseline + ablation table-figure, (9) plume preservation (transect + k1
contrast). Appendix: category atlases, coincidence-sensitivity, qf tables,
smoother null, SG-vs-no-SG, MODIS-sensitivity, high-lat.

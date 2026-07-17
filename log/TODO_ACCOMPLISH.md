# Manuscript Readiness Tracker — Done / To-Do / Storyline

**Created:** 2026-07-08 (from the advisor-style readiness assessment)
**Companions:** `log/PROJECT_REVIEW.md` (reviewer concerns M1–M9, engineering §7),
`log/SPEC_EMPHASIS_STATUS_2026-07-08.md` (spec-feature repositioning),
`log/TCCON_PAPERS_AND_ML_IDEAS_2026-07-03.md` (bibliography + ML ideas).
**Verdict (2026-07-08):** the core TCCON-validated result is manuscript-ready;
start writing NOW in parallel with §2. The mechanism half (spec features) must
not be quoted until the full-parquet regeneration (§2-2) lands. Scoop risk
(Mauceri/Keely 2025 Parts 1+2) argues against gold-plating.
**Update (2026-07-13):** cross-sensor decision recorded — §9 transferability
paragraph + ONE appendix demo figure (A12, TEMPO O2-B); EMIT rejected for this
paper; CH4 handled in text (TROPOMI/GOSAT-GW); journal stays AMT; full
multi-scene demonstration deferred to a follow-up letter (§4).
**Update (2026-07-13b):** cohort's backward-MC 3D-vs-ICA coefficient demo
ACCEPTED for the appendix (A13) — the causal mechanism experiment the
observational stratifications can't provide; §8d reframed from pure future
work to "first demonstration included, PPDF closure + plume OSSE deferred".
**Update (2026-07-15 — fold-PCA production rerun + L′ relabel; full detail in
`log/FOLDPCA_RERUN_2026-07-15.md`):** production DE/XGB/LinReg retrained on
CURC with FOLD-SPECIFIC ProfilePCA (closes the global-PCA train/test leak) and
the entire local stack regenerated — new production tag
**`de_beta_nll_prof_reg_foldpca_o05l15_m5`**. The leakage fix is metrically a
near no-op: AK mean |bias| 1.26→**0.82** (was 0.81), fp-RMSE 2.67→**1.20**
(was 1.19), 71/75 improved, Wilcoxon p=0.0063 — quote foldpca numbers + a
one-sentence reproduction note. Regenerated: TCCON chains ×3 models (+drift),
ATom (near-cloud 0.532→**0.445**), Ship, kfold aggregates, smoother-null
(reproduced), uncertainty layer (locally, refit Side-A inflation; DL
−0.32±0.09 r100 / −0.45±0.07 r50 — conclusions carry over), Nassar suite
(control null again 5/7 + same 2 flags), comparison pptx (now scripted:
`workspace/make_comparison_pptx.py`). Symbol changed **l′ → L′**
(plot_style.py, plain Arial-italic mathtext; all figures regenerated today
carry it — remaining l′ figures listed in the rerun doc). **RESOLVED
2026-07-17: the feature-set variants were RETRAINED with lndo01 on CURC and
the full downstream stack rerun (see §2-7) — the preliminary 2026-07-15
ablation/attribution numbers are superseded; verdicts revert to the
2026-07-08 story (no_spec TCCON-neutral; attribution 55/54/29 %).**
**Update (2026-07-16 — advisor big-picture discussion):** framing decisions
recorded in the new §5 "Framing spine" block: (1) unifying claim =
spectrum-internal cloud awareness (the photon path-length PDF as the single
object behind phenomenology / mechanism / correction / safety); (2)
**land/ocean sign difference ELEVATED** from stratification detail to
mechanism evidence (ocean = dark endpoint of the albedo-contrast axis; one
mechanism, three manifestations: Fig 1b/c + Fig 2 + Fig 3); (3) **imager
independence stated as three tiers** (deployment / spectral sensitivity /
cross-sensor transfer) with the honest training-vs-inference caveat; (4)
skill-vs-trust line resolves the ablation tension in abstract + conclusions;
(5) both emphasis points go into title/abstract keywords (differentiation vs
Mauceri/Keely 2025 under scoop risk).

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
- [x] Production frozen: `de_beta_nll_prof_reg_foldpca_o05l15_m5` (2026-07-15
      fold-PCA retrain, metrically ≤0.01 ppm from the 07-08 freeze; per-surface DE, M=5,
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
   **Regenerated 2026-07-15 under the foldpca tag — conclusion identical**
   (σ 2.23 → 0.35–0.66 vs DE 0.78; |bias| 1.26 → 1.20–1.24 vs DE 0.82;
   fp-RMSE DE 1.20 < smoother 1.34–1.52).
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
   features. **2026-07-17 lndo01-retrained-variant rerun CONFIRMS: full 55 %
   [40–63], no_spec 54 % (Δ +1 pp), no_xco2 29 % (Δ +26 pp) — quote the
   2026-07-17 edition (same conclusions, fold-PCA + reg-consistent variants).**

7. [x] **Retrain the LAND feature-set-ablation variants on CURC — DONE
   2026-07-17.** Both r15 + r05 variant arms retrained with lndo01 and
   downloaded; ALL folds healthy (land f2 restored: RMSE 0.53–0.67, R²
   0.28–0.53; no divergence anywhere). Full downstream stack rerun: f2
   exclusion reverted in `build_ablation_variant_trees.sh`, 5 variant trees
   rebuilt (75 cases each, 0 failures) + reports, ablation doc regenerated →
   `FEATURESET_ABLATION_QF_2026-07-17.md` (QUOTABLE; supersedes 07-15
   preliminary + 07-08 global-PCA editions), Nassar no_spec/no_xco2 rebuilt +
   control nulls + `nassar_channel_attribution.{csv,md}`. **Verdicts revert
   to the 2026-07-08 story:** no_spec TCCON-neutral (pooled ΔRMSE +0.021 ppm;
   near-cloud land QF1 +0.041), no_contam neutral (+0.028), xco2 block carries
   the correction (no_xco2 +0.79 pooled / +1.28 near-cloud land QF1);
   attribution full 55 % [40–63] / no_spec 54 % (Δ +1 pp) / no_xco2 29 %
   (Δ +26 pp). Held-out ordering full > no_spec > no_contam > no_xco2 >
   combos on both surfaces. Known leftover (minor, documented in the doc):
   land f4 of no_contam and no_contam_and_xco2 kept the old unreg checkpoint
   (healthy fold; preempted array tail) — optional `sbatch --array=4` top-up
   for config purity.

## 3. SHOULD-DO (strengthens, not blocking)

- [x] **Failure-mode analysis: where the ML correction does NOT work well —
  DONE 2026-07-16.** →
  `results/model_comparison/deep_ensemble/FAILURE_MODES_2026-07-16.md`
  (generator `workspace/analyze_failure_modes.py`; artifacts incl. the
  163,917-footprint driver table + per-driver figure under
  `<TAG>/failure_modes/`; atrain 75 + drift 21 station-days, AK primary /
  direct check, Ny-Ålesund included). Verdicts: (1) failure rare & small —
  5/96 fp-RMSE worseners (max +0.44 ppm), zero guard activity; (2) **high AOD
  is NOT a relative failure mode** — top land decile has the BEST skill
  (6.10→1.88) but the largest remaining residual (strongest land OLS driver
  +0.26 ppm/SD, t≈98): under-corrected in amplitude, not mis-corrected;
  (3) **bright surfaces are the per-footprint failure stratum** — top alb_o2a
  decile (>0.40) frac-worse 0.47 (df 2021-02-10 dossier = its case instance;
  ties to WCO2 albedo sign rule); (4) **snow is NOT a failure mode**
  (snow_flag=1: 9.39→2.73; biggest improver ny 2017-05-25 is the snow case);
  (5) **high lat fails at the BIAS level, not scatter** (ny 2020-07-11
  correction inert on +1.7 ppm bias; largest worsener = high-lat drift
  ka 2023-09-11; abs_lat tiny in footprint OLS) — station-day anchoring, not
  footprint noise; (6) **σ self-awareness**: land frac-worse falls 0.46→0.06
  from lowest to highest σ decile — worsening is small corrections dithering
  small residuals; large-correction footprints almost never worsen. Feeds M3
  worsening-sites + §7 high-lat: both now "diagnosed limitation".
- [x] **Raw-vs-BC-vs-ML against TCCON: does the ML correction explain part of
  the operational bias correction? — DONE 2026-07-16.** →
  `results/model_comparison/deep_ensemble/RAW_BC_ML_TCCON_2026-07-16.md`
  (generator `workspace/make_raw_bc_ml_report.py`; ML-on-raw tree
  `de_prof_reg_mix_raw` REBUILT over the 75 current cases with
  `build_ablation_variant_trees.sh raw_base` + its own r100 report; raw-model
  folds all healthy, lndo01). Verdicts: (1) keep correcting bc — ML-on-bc
  best station-equal |bias| everywhere, best direct RMSE, far better ocean
  (AK 0.98 vs 1.55); (2) the ML layer CAN subsume most of the operational
  correction — ML-on-raw pooled AK 1.18 vs 1.22, and the raw-trained DE
  rediscovers the increment with r(Δmu, inc)=0.79 pooled / 0.86 near-cloud
  land at 53–66 % amplitude (60 % / 73 % of var(inc)) — B11-independence
  Discussion point secured; (3) near-cloud LAND the operational BC
  over-corrects (raw→bc AK RMSE 3.35→3.84, bias −0.90→−1.22, while helping
  QF=0 pool 1.47→1.01) and the production correction is ANTI-correlated with
  the increment there (r=−0.34) — the two layers are complementary, ML partly
  reverses the near-cloud over-application. Supersedes the four-series part
  of `reg_mix_bc_vs_raw/BC_VS_RAW_COMPARISON.md` (2026-07-04).
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
  **Regenerated LOCALLY 2026-07-15 on the foldpca tag** (Side-A inflation
  refit from the new held-out predictions; runs locally now — the new atrain
  plot_data carries mu_NN): DL −0.32 ± 0.09 (r100) / −0.45 ± 0.07 (r50),
  τ 0.22 → 0.52 ppm with radius, ⟨z²⟩ 1.80/2.39, TOST 1/75 equivalent —
  every conclusion of the CURC edition carries over.
- [ ] **Manuscript figure pass (style LOCKED 2026-07-09; user-approved on 3
  vetted samples: bu 2018-10-24, iz 2019-03-13, ra 2016-09-11).
  SYMBOL CHANGE 2026-07-15: l′ → L′** (plot_style.py MEAN_L_LABEL/VAR_L_LABEL,
  now plain Arial-italic mathtext — the Times \mathcal hack was only needed
  for the bad lowercase glyph; paper LaTeX should use $L'$). Carrying L′
  already: all 75 foldpca TCCON cases + aggregates, drift, ATom, Ship, Nassar,
  smoother-null, Fig 2 heatmap, Fig 4 case figures, A1 atlases. Still l′:
  run_all heavy figures (Fig 1b,c / Fig 3 / A11 — CURC), A5 savgol (CURC),
  composed make_*.py figures (fig01a/05b/07/08/09b — re-check numbers against
  the NEW reports before regen), poster.
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
  - [x] ATom figures → same style **(DONE 2026-07-10:** all three scripts +
        outputs regenerated in place — comparison 4-panel, MODIS overlay,
        pseudo-column summary; stats unchanged, 17 legs 0.53→0.46 ppm**)**
  - [x] Ship figures **(DONE 2026-07-10:** both scripts + all 4 case PNGs +
        summary regenerated; turbo→plasma, aspect-locked maps, (a)–(d) tags,
        ⊕→mathtext; panel-(d) ylim now includes the corrected-median line**)**
  - [x] Arial/style pass on remaining manuscript producers **(DONE
        2026-07-10/11:** `smoother_null_figure.py` (+PNG regenerated),
        `spec_case_study` figure tools (Tasman case + cloud_no_bias atlas
        regenerated; screener is CSV-only), Nassar plume suite
        (`nassar_plume_transects.py`, `nassar_control_null.py`,
        `plot_nassar_plumes.py` — Westar money transect + control null +
        6-panel maps regenerated, verdicts unchanged; k1-contrast is
        tables-only, no figure exists yet), `land_class.py` +
        `spec_sensitivity.py` (effect heatmap regenerated from cached CSV;
        shadow/subpixel smoke-tested), and the full `src/analysis` suite for
        `run_all.py` (style constants re-exported via `utils.py`, ⟨l′⟩/var(l′)
        + mathtext bands everywhere, jet purged from legacy `results.py`,
        `_save()` dpi 150→rcParam 300; decay curves regenerated on
        2019-07-10 and inspected). `tccon_comparison_report.py` was already
        styled. All 23 edited files py_compile.**)**
  - [x] Batch re-render all 75 TCCON cases locally **(DONE 2026-07-10:**
        `SKIP_BUILD=1` run — 128 run_case lines, 0 errors; aggregates
        reproduce production (calib R² 0.935→0.983, mean resid −0.37→+0.007);
        Burgos sample visually verified**)**
  - [ ] Pick compact vs `_full` per manuscript slot; poster figure still
        legacy styling (only relabeled to $X_{CO2}$)
  - [x] **Missing budget figures CREATED (2026-07-11)** — five new composed
        figures via `workspace/manuscript_figures/make_*.py`, all rendered,
        style-locked, and number-checked against their source CSVs/MDs:
        fig01a collocation schematic (real 2020-01-01 granule), fig05b
        date-blocked CV design (real fold manifests), fig07 significance +
        coincidence panel, fig08 baseline + ablation table-figure, fig09b
        plume-vs-cloud k1 fingerprint. Paths in the §5 Main-figure list.
        Note: fig08's ablation numbers use the canonical QF-grouped doc
        (`FEATURESET_ABLATION_QF_2026-07-08.md`; the older 64x32 summary was
        archived in the 07-09 docs sweep). Also restyled
        `src/spectral/compare_savgol_fits.py` (A5 producer; regen on CURC).
- [ ] **Cross-sensor transferability (decided 2026-07-13: mention + one
  appendix demo; TEMPO only, EMIT rejected; journal call unchanged — AMT).**
  (a) Extend the §9 transferability paragraph: abstract requirements of the
  method (resolved absorption band + per-channel prior τ + single-footprint
  spectrum — no imager, no retrieval internals) → OCO-3/CO2M (same bands,
  direct transfer) → TEMPO O2-B (geostationary AND an imager: its own cloud
  product gives in-scene proximity, no cross-platform collocation — the
  strongest form of the NoMODIS argument) → CH4 in TEXT only (TROPOMI
  2305–2385 nm @ ~0.25 nm, GOSAT-GW are the resolution-viable candidates;
  EMIT-class ~7.4 nm band-integrated sampling stated as an open question —
  it collapses the channel-to-channel τ dynamic range the cumulant fit
  feeds on). Add the CH4 hook sentence (plume quantification near broken
  cloud fields is where emission estimates get made or discarded).
  (b) NEW appendix figure A12 (see §5 appendix list): one TEMPO O2-B granule
  demo from `~/programming/tempo` (WP1–WP8 pipeline complete; 4-panel figs
  exist for 3 vetted scenes) with an existence-proof caption ONLY — no bias
  or validation claim. Step-by-step task list lives in the tempo repo:
  `~/programming/tempo/TODO_OCO_APPENDIX_A12.md` (scene selection → in-scene
  cloud distance → decay panel → style → caption handshake). Add the in-scene ⟨l′⟩-vs-cloud-distance decay panel
  (threshold CLDO4 cloud fraction + KD-tree distance, reuse step_04 geometry
  logic; ~1 day) so the panel echoes Fig 1b,c; restyle to `plot_style.py`.
  Caveat to state: tempo still runs the pre-§7.2 fit engine (curve_fit + SG)
  — caption as "same model, reference implementation".
  (c) EMIT demo explicitly REJECTED for this paper: no pipeline exists
  (weeks of new work), the resolution physics is likely fatal, and a weak
  panel next to a clean TEMPO panel would undermine the transferability
  claim it is meant to support.
- [ ] **Backward-MC 3D-vs-ICA mechanism demo → appendix A13 (decided
  2026-07-13; cohort's model).** Source figure:
  `~/Downloads/01_radtoa_fitting_coefficients.png` (X–Z scene, cloud at
  x = 10–15 km; fitted ln-reflectance-vs-τ coefficients C2/C1/C0 along track,
  3D vs ICA). Why included: the ICA curve is a perfect null (same cloud, no
  horizontal transport) — the 3D-only adjacency response in clear columns
  (x ≈ 15–18 km: C2 ×~5, C1 −0.45 excursion, C0 −2.3) is a CAUSAL
  demonstration that the cumulants respond to 3D transport; one-sided
  (shadow-side) response corroborates the Fig 3 shadow/brightening split;
  ~3 km enhancement scale echoes the observed ocean decay. Conditions before
  it ships:
  - [ ] Refit the synthetic spectra with the PRODUCTION estimator (order 7,
        exact lstsq, no-SG) — removes the order-2-vs-order-7 reviewer
        question entirely (fallback: one sentence on why order 2 suffices
        at the simulated τ range).
  - [ ] Relabel to paper notation (C1 → −⟨l′⟩, C2 → ½·var(l′)) + restyle to
        `plot_style.py`; caption in locked conventions.
  - [ ] Methods paragraph (cohort sign-off): model name + citation (EaR3T /
        Chen et al. 2025 lineage?), backward MC, X–Z (2D) domain, cloud
        COD/height/placement, SZA + viewing geometry, band, surface albedo,
        photon-noise level.
  - [ ] Understand + dismiss the far-field 3D-vs-ICA offset (C1 −1.07 vs
        −1.05) in half a caption sentence (ask cohort: domain-average side
        illumination? boundary conditions?).
  - [ ] Scope sentence pre-empting the parameter-sweep request: single
        representative geometry, mechanism demonstration only — validates
        the FEATURE physics, not the correction μ; sensitivity across
        SZA/COD/albedo belongs to the OSSE follow-up.
  - [ ] Co-authorship / provenance settled with the cohort member.
- [ ] Writing-time subsections: Ny-Ålesund/high-latitude; five worsening sites;
  M5 parallax/advection bound sentences; M9(f) convergence-radius caveat.
- [ ] **Framing-spine writing tasks (advisor discussion 2026-07-16, see §5
  spine):** (a) abstract second sentence = imager-independence tiers; (b)
  land/ocean albedo-contrast unification sentence in §3 + §4 + Discussion
  echo; (c) skill-vs-trust line in abstract + conclusions; (d) "so what"
  sampling-bias sentence in the Introduction; (e) title/abstract keywords
  carry both emphasis points.

## 4. EXPLICITLY DEFERRED (state in reply-to-reviewers if asked)

- Transport-model (CAMS/CT) regression of the label (ceiling + r05/r15
  sensitivity covers most of it) · EaR3T OSSE (follow-up paper; the A13
  single-scene 3D-vs-ICA demo is now IN the appendix — what stays deferred
  is the PPDF tally closure, plume injection, and the geometry sweep) ·
  wind-resolved plume enhancement · σ(k1)/σ(k2) export (appendix QC) ·
  MCD12C1 purity-filter robustness row · FT-Transformer/TabPFN baselines ·
  **cross-sensor demonstration letter** (2026-07-13: multi-scene TEMPO O2-B
  statistics + TROPOMI CH4 feasibility; the A12 appendix demo is the teaser;
  natural GRL/AMT-letter shape; port the exact-lstsq fitter to tempo first) ·
  EMIT CH4 fitting (~7.4 nm sampling likely collapses the τ dynamic range;
  revisit only if the band-integrated open question gets answered).

---

## 5. MANUSCRIPT STORYLINE (planned 2026-07-08; supersedes PROJECT_REVIEW §5)

Target: **AMT**. Title shape: *"Correcting cloud-proximity biases in OCO-2
XCO2 with photon path-length statistics and deep ensembles, validated against
TCCON, aircraft, and shipborne observations."*

**Framing spine (advisor big-picture discussion, 2026-07-16) — the paper must
read as one claim with four faces, not a pipeline + validations:**

- **Unifying claim:** 3D cloud effects leave a physically interpretable
  fingerprint — the photon path-length PDF, read as cumulants — in every
  single-footprint spectrum, sufficient to *diagnose, understand, correct,
  and audit* cloud-proximity bias. Phenomenology (§3), mechanism (§4),
  correction (§5–7), and safety (§8) are four faces of that one object.
- **"So what" sentence (agency-level reader):** QF filtering is a spatially
  CORRELATED sampling bias — it preferentially deletes the cloudy tropics
  where flux inversions are hungriest, at the ~1 ppm level of the signals
  they chase; the correction converts that sampling bias into usable data.
- **EMPHASIS 1 — land/ocean sign difference = mechanism evidence, not a
  stratification detail:** ocean is the DARK ENDPOINT of the same
  cloud−surface albedo-contrast axis behind the WCO2 land-cover sign rule;
  one mechanism, three independent manifestations (Fig 1b/c opposite-sign
  decays, Fig 2 sign rule, Fig 3 shadow/brightening split). State the
  unification explicitly in §3 + §4 and echo in the Discussion — it
  justifies the per-surface model split physically, not just empirically.
  Supporting operational hook: raw-vs-BC (2026-07-16) found B11
  OVER-corrects near-cloud land — the operational correction does not
  resolve the surface dependence; ours does.
- **EMPHASIS 2 — imager independence in three tiers (claim precisely):**
  Tier 1 *deployment* — no cloud input, no neighboring footprints at
  inference; runs unchanged in the post-2022 Aqua free-drift NoMODIS era
  (A7 = proof by existence). Tier 2 *sensitivity* — the spectrum itself
  carries cloud-proximity information (spec-only classifier AUC 0.72/0.66,
  sub-pixel monotonicity below the 1-km MODIS floor — A11); the claim no
  imager-dependent method (Massie/Mauceri line) can make. Tier 3
  *transfer* — requirements are only a resolved band + per-channel prior τ
  + a single-footprint spectrum → OCO-3/CO2M/GOSAT-GW/TEMPO (A12). Honest
  caveat that makes it bulletproof: TRAINING used MODIS once (labels +
  validation stratification), INFERENCE never does — "the imager is
  scaffolding: used to build and verify the correction, then removed."
  Structure §1 contributions, §9, and the abstract's second sentence
  around the tiers.
- **Skill-vs-trust line (resolves the ablation tension head-on):** the
  retrieval-state feature supplies the skill; the path-length physics
  supplies the trust — mechanism, plume-safety audit (the ≤0.21 ppm bound
  and the all-band k1 discriminator, which no xco2-based feature can
  provide about itself), and the imager-free sensitivity floor. Skill
  without trust is a black box nobody deploys on a climate record. One
  line in abstract + conclusions.
- Both emphasis points → title/abstract keywords.

1. **Introduction.** OCO-2 XCO2 widely used (flux inversion, plume/emission
   quantification, trend monitoring) at sub-ppm accuracy requirements → but
   cloud-induced 3D-RT biases documented (Merrelli 2015; Massie 2017/2021/2023;
   Mauceri 2023; Chen 2025), and QF filtering discards near-cloud soundings
   disproportionately (cloudy tropics). Gap: operational B11 correction is not
   cloud-proximity-aware; prior ML corrections are general-purpose
   (Keely 2023; Mauceri/Keely 2025) or imager-dependent. Contributions: physics
   features from the photon path-length PDF; per-footprint, imager-free
   deployment (abstract sentence); triple independent validation; audited
   plume safety. **Structure the contribution list around the three
   imager-independence tiers (framing spine, EMPHASIS 2).**
2. **Data.** OCO-2 L2 Lite v11 glint (2016–2020, 116 dates, 17.8M soundings),
   L1B radiances, MET/CO2-prior profiles; Aqua-MODIS MYD35 collocation and the
   nearest-cloud-distance definition (±time buffers, Cloudy+Uncertain, 1-km
   floor). TCCON GGG2020, ATom, shipborne EM27/SUN; MCD12C1 for stratification.
   **State here:** cloud distance is a *diagnostic and label-construction*
   quantity only — never a model input.
3. **Phenomenology.** Anomaly-vs-cloud-distance decay: **positive bias over
   land (within ~15 km), negative over ocean (within ~5 km)** — motivates the
   per-surface targets (land r15 / ocean r05). QF throughput vs distance.
   **State here (and unify in §4) that the land/ocean sign difference is
   mechanism evidence — the dark-vs-bright endpoints of the albedo-contrast
   axis (framing spine, EMPHASIS 1).**
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
   (d) **PPDF validation with a Monte Carlo model — REFRAMED 2026-07-13
   (first demonstration now included as appendix A13):** the cumulant fit
   *interprets* k1/k2 as mean and variance of the photon path-length
   distribution function. A cohort backward-MC X–Z simulation (cloud at
   x = 10–15 km) now demonstrates causally that the fitted coefficients
   respond to cloud adjacency ONLY under 3D transport — the ICA null (same
   cloud, no horizontal transport) shows zero adjacency response while the
   3D run shows the k1/k2 enhancement in clear columns, one-sided on the
   shadow side, decaying within ~3 km (appendix Fig A13; conditions
   checklist in §3). Cite it here as the first step. STILL FUTURE WORK
   (follow-up paper per §4): the full PPDF closure — tally the per-photon
   path-length histogram, generate synthetic OCO-2 spectra from the same
   ensemble, compare spectrum-fitted k1/k2 against the directly tallied MC
   moments (closing the Laplace-transform loop) — and the plume OSSE
   (inject ΔCO2, verify k1/k2 invariance rigorously) across
   SZA/COD/albedo geometries.
9. **Data recovery and applications.** QF1 near-cloud recovery numbers;
   throughput in the cloudy tropics → flux-inversion relevance; **cross-sensor
   transferability paragraph (user decision 2026-07-13, see §3):** abstract
   method requirements → OCO-3/CO2M → TEMPO O2-B (geostationary imager,
   in-scene cloud proximity; points to appendix Fig A12) → CH4 in text only
   (TROPOMI/GOSAT-GW viable; EMIT-class band-integrated sampling an open
   question) + the CH4 plume-community hook sentence; post-2022 NoMODIS-era
   deployability. **Present the whole section as Tier 3 of the
   imager-independence framing (Tier 1 deployment in §5/§7, Tier 2
   spectral sensitivity in §6/A11 — framing spine, EMPHASIS 2), with the
   training-vs-inference "imager as scaffolding" caveat stated once here.**
10. **Conclusions.** Restate the skill-vs-trust line (framing spine) and the
    land/ocean-unifying mechanism sentence; close with the outlook sentence
    pair: MC/PPDF validation (§8d) and transfer to imager-free correction
    for OCO-3 / GOSAT-GW / CO2M.

**Main-figure budget (~9) — paths (all figures EXIST as of 2026-07-11; all in
the locked AMT style; new composed figures generated by
`workspace/manuscript_figures/make_*.py`):**

1. **Collocation/geometry schematic + decay curves**
   - (a) schematic (real data, orbit 29265 2020-01-01, annotated nearest-cloud
     arrow): `results/figures/manuscript/fig01a_collocation_schematic.png`
   - (b,c) anomaly-vs-distance decay:
     `results/figures/cld_dist_analysis/{ocean,land}/xco2_bc_anomaly_vs_cld_dist_binned.png`
     (local copies predate the 2026-07-10 restyle AND the 2026-07-11 l′
     typesetting change (Times italic via plot_style) — refresh via
     `run_all.py` on the next full-parquet pass; needs > 34 GB RAM or CURC.
     Same applies to every run_all/l′ figure: Fig 2 heatmap, Fig 3
     shadow/brightening, k1_k2_binned_profile, A5 savgol A/B, A11
     sub-pixel. Locally REGENERATED with the new l′ 2026-07-11: fig09b,
     Tasman + N-Pacific case figures, 11 Nassar case maps, all
     feature-importance figures.)
   - *Suggested caption:* **Figure 1.** (a) Collocation geometry for one
     OCO-2 glint granule (orbit 29265, 1 January 2020): Aqua-MODIS MYD35
     cloud-mask pixels (Cloudy, blue; Uncertain, grey; both classes
     retained; |Δt| ≤ 10 min) and OCO-2
     soundings coloured by nearest-cloud distance d (great-circle KD-tree
     search, capped at 50 km). The arrow connects one sounding to its
     nearest cloudy pixel (d = 13.9 km); inset: full granule track.
     (b, c) Binned median within-orbit clear-sky XCO2 anomaly versus
     nearest-cloud distance for ocean (b) and land (c) glint soundings
     (2016–2020, 116 dates, 17.8 M soundings; shading: interquartile
     range) — the bias grows systematically as clouds are approached,
     before any correction is applied.
2. **Land-cover effect-size heatmap** (WCO2 Δ⟨l′⟩ sign rule):
   `results/figures/cld_dist_analysis/land_class/landclass_effect_heatmap.png`
   - *Suggested caption:* **Figure 2.** Land-cover-stratified near-cloud
     spectral response. Effect size (near-cloud minus far-cloud reference,
     normalised by the far-cloud spread) of the reference-corrected
     path-length features by MCD12C1 land-cover class. ⟨l′⟩ and var(l′)
     are the fitted mean and variance of the relative photon path (the
     cumulants k1, k2 defined in Sect. X). The sign of the WCO2 Δ⟨l′⟩
     response follows the sign of the cloud−surface albedo contrast:
     positive over vegetated classes (savanna +0.43σ) and negative over
     bright barren surfaces (−0.40σ); magnitudes do not rank with
     contrast, so the albedo-contrast mechanism enters as a sign rule.
3. **Shadow/brightening split** (O2A exp-intercept branches):
   `results/figures/cld_dist_analysis/spec_sensitivity/shadow_brightening_land.png`
   (+ `_ocean.png`; predate restyle — refresh on next heavy run). Optional
   companion: `results/figures/cld_dist_analysis/{ocean,land}/k1_k2_binned_profile.png`
   - *Suggested caption:* **Figure 3.** Shadowing versus brightening.
     Near-cloud footprints are split by the sign of the O2A
     continuum-reflectance departure from the clear-sky reference
     (exp-intercept low: cloud shadowing; high: side illumination /
     brightening); panels show the binned XCO2 anomaly and band-resolved
     Δ⟨l′⟩ versus cloud distance for the two branches (land). The two
     branches carry opposite-signed XCO2 anomalies, consistent with the
     3-D radiative-transfer mechanism, and are separable from a single
     footprint's spectrum alone.
4. **Case-study exemplar with RGB** (vetted Tasman):
   `results/figures/cld_dist_analysis/spec_case_study/case_2018-05-01_201805010338467_fp2.png`
   (alternate: N-Pacific 2019-02-01 via `spec_case_figure.py`)
   - *Suggested caption:* **Figure 4.** Single-overpass case study, Tasman
     Sea, 1 May 2018. (a) Aqua-MODIS true-colour image (NASA Worldview /
     GIBS) with the OCO-2 glint track; remaining panels: per-footprint
     nearest-cloud distance, XCO2 anomaly, band-resolved Δ⟨l′⟩ and
     Δvar(l′), continuum-reflectance mismatch, and the cross-band Δ⟨l′⟩
     z-score. Footprints approaching the cloud deck show coherent
     path-length and XCO2 signatures across the ~10-km swath, co-located
     with the imaged cloud edge.
5. **Pipeline/CV design schematic**
   - (a) DE architecture: `results/figures/deep_ensemble_architecture.png`
   - (b) date-blocked CV (real fold manifests, random-split leakage contrast):
     `results/figures/manuscript/fig05b_cv_design.png`
   - *Suggested caption:* **Figure 5.** Model and validation design.
     (a) Per-surface deep ensemble: M = 5 Gaussian MLPs (64→32 hidden
     units; μ, log σ² heads; β-NLL loss, β = 0.5; layer-norm + dropout 0.1)
     trained on per-sounding features only — no cloud information and no
     neighbouring-footprint information enters at inference — pooled as a
     mixture, with split + Mondrian conformal 90 % intervals. (b)
     Validation design on the 116 semi-monthly training dates (2016–2020):
     a sounding-level random split leaves test days interleaved with
     training days and inflates skill (R² = 0.82) because same-day
     soundings share weather and orbit state; date-blocked 5-fold CV
     (contiguous year blocks, trailing calibration block for early
     stopping and conformal calibration) gives the honest estimate
     (held-out R² 0.53 ocean / 0.39 land).
6. **TCCON before/after headline** (production tag, AK reference)
   - per-case compact 2×3 (user-approved sample):
     `results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_foldpca_o05l15_m5/atrain/combined_2018-10-24_bu/corrected_xco2_vs_tccon.png`
     (alternates: `combined_2019-03-13_iz`, `combined_2016-09-11_ra`; `_full`
     3×3 variant in the same dirs)
   - all-75-case dumbbell:
     `results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_foldpca_o05l15_m5/atrain/tccon_ak_bias_dumbbell_label_r100km.png`
   - *Suggested caption:* **Figure 6.** TCCON validation. (a–f) One
     overpass (Burgos, 24 October 2018; station ±100 km view, dashed
     100-km collocation circle): OCO-2 Lite XCO2, ML-corrected XCO2,
     predicted uncertainty σ, predicted correction μ, footprint
     distributions against coincident TCCON, and the TCCON time series
     around the overpass. (g) Station-day mean bias against AK-harmonised
     TCCON before and after correction for all 75 station-days (18 sites,
     2016–2020, r = 100 km, ±60 min): mean |bias| falls from 1.26 to
     0.81 ppm and per-footprint RMSE from 2.67 to 1.19 ppm, with 71/75
     station-days improved. Averaging-kernel/prior harmonisation follows
     Rodgers and Connor (2003) and Wunch et al. (2017); before-vs-after
     differences are invariant to the harmonisation.
7. **Significance/robustness panel** (bootstrap forest + 3×3 coincidence
   matrix): `results/figures/manuscript/fig07_significance_robustness.png`
   - *Suggested caption:* **Figure 7.** Statistical significance and
     robustness of the TCCON improvement (AK-harmonised reference,
     production configuration). (a) Site-clustered bootstrap estimates
     (dots) with 95 % confidence intervals for the corrected-minus-
     uncorrected change in station-day mean |bias|, RMS bias, and
     per-footprint RMSE at r = 100 km / ±60 min, for all 18 sites and
     excluding Ny-Ålesund; bootstrap p-values at right; the paired
     Wilcoxon test on station-day |bias| gives p = 0.0064 (n = 75).
     (b) Change in mean |bias| across collocation radius (25/50/100 km)
     × coincidence window (±30/60/120 min); the improvement is
     significant at every combination and largest at the tightest radii,
     the behaviour expected of a real, spatially localised correction
     rather than a selection artefact. Primary criteria outlined in black.
8. **Baseline + ablation table-figure** (5 models + 5 feature-set deltas,
   pooled vs near-cloud land): `results/figures/manuscript/fig08_baseline_ablation.png`
   - *Suggested caption:* **Figure 8.** Model and feature-set comparison
     under one protocol (identical features, date-blocked folds, and TCCON
     chain; AK-harmonised, r = 100 km / ±60 min). (a) TCCON per-footprint
     RMSE after correction, pooled over all quality flags (light,
     n = 105,683) and for the decisive near-cloud land subset (≤ 10 km,
     dark, n = 75,157); the uncorrected product is shown for reference.
     The deep ensemble leads throughout, and the ordering is decided in
     the near-cloud land tail. (b) Feature-group ablation of the deep
     ensemble (ΔRMSE relative to the full feature set): dropping the
     spectral or contamination groups is TCCON-neutral, whereas removing
     the retrieval-state departure (xco2_raw − a priori) costs
     0.7–1.1 ppm — the operationally load-bearing predictor, whose
     effectiveness the path-length analysis of Sect. X explains.
9. **Plume preservation**
   - (a) transect (Westar 2023-06-26 money figure):
     `results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_foldpca_o05l15_m5/nassar_plumes/plume_preservation/transects/nassar_transect_westar_2023-06-26.png`
   - (b) band-resolved k1 fingerprint (cloud vs plume discriminator):
     `results/figures/manuscript/fig09b_k1_contrast.png`
   - *Suggested caption:* **Figure 9.** The correction preserves real CO2
     enhancements. (a) Along-track transect over the Westar power plant
     (26 June 2023, clear sky, nearest cloud ≈ 50 km; overpass from the
     Nassar et al. catalogue): the ~+0.6 ppm enhancement at closest
     approach survives the correction essentially unchanged; lower
     panels: predicted correction μ and nearest-cloud distance. (b)
     Band-resolved Δ⟨l′⟩ (plume window minus background; ±1 SE) for the
     two flagged removal windows and two clear-sky controls. Black ticks:
     signature expected of a real CO2 plume (Δ⟨l′⟩ ≈ ⟨l′⟩·ΔXCO2/XCO2 in
     the CO2 bands via the prior-based optical depth; exactly zero in
     O2A). In the flagged windows O2A shifts as strongly as the CO2
     bands — the all-band fingerprint of cloud contamination, not a
     plume; the worst-case plume signal removable through the spectral
     channel is ≤ 0.21 ppm.

**Appendix / supporting figures (proposed set, with paths;** base
`results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_foldpca_o05l15_m5`
(the 2026-07-15 fold-PCA production tree; the old
`de_beta_nll_prof_reg_o05l15_m5` tree is retained for reference)
abbreviated `<TAG>`**):**

- **A1 case-study category atlases:** `results/figures/cld_dist_analysis/spec_case_study/atlas_{good_land_1,good_land_2,good_ocean_1,good_ocean_2,false_positive_1,false_positive_2,cloud_no_bias}.png`
  (ALL seven pages regenerated 2026-07-11 in the locked style incl. the
  Times-italic l′ — `spec_case_atlas.py`, defaults)
  *Caption stub:* "Case atlas, category X (n cases from the vetted 19-case
  shortlist): GIBS true-colour, footprint Δ⟨l′⟩, and XCO2 anomaly per case —
  real cloud signatures / MYD35 false positives / clouds without XCO2 bias."
- **A2 coincidence sensitivity:** table
  `<TAG>/atrain/coincidence_sensitivity/coincidence_sensitivity.md` (numbers
  are already in main fig 7b; keep the 9 per-combo dumbbells as supplement)
- **A3 QF-stratified TCCON:** `<TAG>/atrain/tccon_ak_bias_qf0_r100km.png` +
  `_qf1` (the §4-8 data-recovery hook) and the r50 robustness row
  `<TAG>/atrain/tccon_ak_bias_r50km.png`
  *Caption stub:* "Station-day bias before/after correction restricted to
  QF = 0 (best) and QF = 1 (flagged) soundings: the correction improves the
  flagged population toward QF-0 quality — the basis of the data-recovery
  estimate — and the r = 50 km row shows the headline is not
  collocation-radius dependent."
- **A4 smoother null:** `<TAG>/atrain/smoother_null/smoother_null_r100km.png`
  *Caption stub:* "Pure-smoother null (orbit-local running mean, feature-free,
  ±10/30/100 s): the smoother collapses footprint scatter more than the deep
  ensemble (0.35–0.66 vs 0.78 ppm) yet leaves the TCCON station-day bias
  unmoved (1.20–1.24 vs 0.81 ppm) — variance removal alone cannot produce
  the observed bias reduction."
- **A5 SG-vs-no-SG (M9a):** `results/model_comparison/savgol_ab/savgol_ab_scatter.png`
  (script restyled 2026-07-11; regenerate on CURC where `fitting_details_*.h5`
  live)
  *Caption stub:* "Fitted ⟨l′⟩ and var(l′) from Savitzky-Golay-smoothed vs
  raw ln T per band: parameters agree to within X % (r > 0.99); the no-SG fit
  is the production default and smoothing is a robustness choice, not a
  result driver."
- **A6 MODIS-sensitivity / product dependence:** the MYD35 false-positive
  atlas (A1 `atlas_false_positive.png`) + Cluster 9 literature bounds — no
  extra figure needed (bounds-only sentences per M5 resolution)
- **A7 high-latitude / drift era:** Ny-Ålesund + post-2022 cases under
  `<TAG>/drift/combined_*/corrected_xco2_vs_tccon.png` (pairs with the
  Ny-Ålesund writing-time subsection)
  *Caption stub:* "High-latitude and post-2022 (Aqua free-drift, NoMODIS)
  overpasses: the per-footprint correction requires no imager collocation,
  so it applies unchanged in the drift era; Ny-Ålesund illustrates the
  high-SZA/snow limit discussed in Sect. X."
- **A8 ocean validation:** ATom summary `<TAG>/atom/atom_pseudo_column_summary.png`
  + per-date 4-panels `<TAG>/atom/combined_<date>_atom/atom_comparison_<date>.png`;
  ship summary `<TAG>/ship/ship_comparison_summary.png` + 4 case figures
  `<TAG>/ship/combined_<date>_<ship>/ship_comparison_*.png`
  *Caption stub (ATom):* "ATom aircraft pseudo-column comparison (8 dates,
  17 collocated legs, AK-smoothed; stratosphere above the ceiling filled
  with the OCO-2 prior so it cancels in the comparison): near-cloud legs
  improve |residual| 0.53 → 0.46 ppm; 2017-10-09 is the far-cloud negative
  control." *Caption stub (ship):* "Shipborne EM27/SUN comparison (R/V
  Sonne MORE-2, R/V Mirai MR21-01; 100 km/±2 h): the correction collapses
  footprint scatter (σ 0.63 → 0.27 ppm) without moving the clear-sky
  control (2019-06-22); the residual ~+1 ppm absolute offset is dominated
  by the un-harmonised direct comparison and EM27 scale vintage."
- **A9 uncertainty-aware layer:** tables `<TAG>/atrain/tccon_uncertainty_r{100,50}km.md`
  (optionally compose a DL-mean forest figure later; CSVs ready)
- **A10 architecture no-cloud variant (deployment story):**
  `results/figures/deep_ensemble_architecture_no_cloud.png`
  *Caption stub:* "Inference-time data flow: cloud distance enters only
  label construction, loss weighting, and evaluation — never the model
  input — so the correction runs footprint-by-footprint with no imager and
  no along-track context."
- **A11 spec-only classifier ROC (MODIS-free sensitivity):**
  `results/figures/cld_dist_analysis/spec_sensitivity/spec_cloud_classifier_roc.png`
  + sub-pixel check `subpixel_anomaly_vs_specidx.png` (predate restyle —
  refresh on next heavy run)
  *Caption stub:* "Cloud-proximity information in the spectral features
  alone: a classifier on the cumulant block separates near-cloud from
  far-cloud soundings (AUC 0.72 land / 0.66 ocean), and the anomaly rises
  monotonically with the spectral index below the 1-km MODIS pixel floor —
  the only proximity signal available in the post-2022 NoMODIS era."
- **A12 cross-sensor demo — TEMPO O2-B (decided 2026-07-13; TO CREATE):**
  one granule from `~/programming/tempo` (existing 4-panel: GOES ABI RGB +
  CLDO4 cloud fraction + ⟨l′⟩ + var(l′) maps; 3 vetted scenes to pick from)
  + NEW in-scene ⟨l′⟩-vs-cloud-distance decay panel (CLDO4 cloud-fraction
  threshold + KD-tree, reuse step_04 geometry; ~1 day); restyle to
  `plot_style.py`. Existence-proof caption ONLY; EMIT explicitly excluded
  (no pipeline; ~7.4 nm sampling collapses the τ dynamic range; a weak
  panel would undermine the claim). Pairs with the §9 transferability
  paragraph and the A10 deployment-story figure.
  *Caption stub:* "Instrument transfer existence proof: the cumulant fit of
  Sect. X applied to one TEMPO O2-B granule (683–697 nm; same model,
  reference implementation): ⟨l′⟩ and var(l′) are retrievable from a
  geostationary UV-Vis imager whose own cloud product provides in-scene
  proximity — no cross-platform collocation — and rise toward the imaged
  cloud field; shown as feasibility, not a validated correction."
- **A13 backward-MC 3D-vs-ICA mechanism demo (decided 2026-07-13; TO
  PREPARE — cohort figure, conditions checklist in §3):** source
  `~/Downloads/01_radtoa_fitting_coefficients.png`, to be refit with the
  production estimator, relabeled (C1 → −⟨l′⟩, C2 → ½·var(l′)), and
  restyled; final path
  `results/figures/manuscript/fig_a13_mc_3d_vs_ica.png`. Pairs with §4
  (mechanism) and the §8d paragraph; together with A12 it makes the
  "features are physics, and the physics transfers" bookend.
  *Caption stub:* "Controlled 3-D radiative-transfer demonstration of the
  path-length mechanism. Fitted ⟨l′⟩, var(l′), and intercept along an X–Z
  backward Monte Carlo scene (cloud at x = 10–15 km; [model, geometry,
  COD, band]) for full 3-D transport and the independent-column
  approximation (ICA). The ICA run — same cloud, no horizontal photon
  transport — shows no response outside the cloud, whereas the 3-D run
  shows enhanced ⟨l′⟩ and var(l′) in clear columns within ~3 km of the
  shadow-side cloud edge: the fitted cumulants respond to cloud adjacency
  only under 3-D transport, corroborating the shadow/brightening asymmetry
  of Fig. 3. Single representative geometry, validating the feature
  physics, not the correction; the full path-length-histogram closure and
  plume OSSE are future work (Sect. 8d)."

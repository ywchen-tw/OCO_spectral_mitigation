# Fold-PCA production rerun + L′ relabel — 2026-07-15

**What happened:** the production DE, XGBoost-mean, and Ridge/LinReg models were
retrained on CURC with **fold-specific ProfilePCA** (EOF basis fit on each
date_kfold fold's TRAIN dates only — closes the mild unsupervised leak of the
former global-PCA basis) and downloaded to
`results/model_{deep_ensemble,gbdt,linear_baseline}/`. The full local
comparison + validation stack was regenerated against them, and the photon
path-length symbol was changed **l′ → L′** everywhere via
`workspace/plot_style.py` (single source; now plain Arial-italic mathtext —
the Times `\mathcal` workaround was only needed for the lowercase glyph).

**New production DE tag:** `de_beta_nll_prof_reg_foldpca_o05l15_m5`
(ocean `de_ocean_beta_nll_prof_reg_foldpca_r05_f0..4`, land
`..._foldpca_r15_f0..4`). Baseline tags unchanged
(`xgb_prof_foldpca_o05l15`, `linreg_prof_foldpca_o05l15`) but their trees were
REBUILT with the updated fold-PCA models. The old tag tree
(`de_beta_nll_prof_reg_o05l15_m5`) is retained untouched for reference.

## Headline: the leakage fix is metrically a near no-op

TCCON r=100 km / ±60 min, 75 station-days, AK-harmonized reference:

| metric | old (global PCA) | new (fold PCA) |
|---|---|---|
| mean \|bias\| before → after | 1.26 → **0.81** | 1.26 → **0.82** |
| fp-RMSE before → after | 2.67 → **1.19** | 2.67 → **1.20** |
| improved fp-RMSE | 71/75 | 71/75 |
| station-day \|bias\| Wilcoxon | p = 0.0064 | p = 0.0063 |
| bootstrap Δ mean \|bias\| | −0.43 [−0.71, −0.13], p=0.004 | −0.43 [−0.70, −0.13], p=0.0042 |
| bootstrap Δ fp-RMSE | −1.46 [−1.93, −0.97] | −1.45 [−1.91, −0.97] |

Manuscript can quote the foldpca numbers with the one-sentence note that the
fold-safe PCA reproduces the global-PCA results to ≤0.01 ppm.

## Model comparison (same-protocol, regenerated)

- **date_kfold aggregates** (`results/model_comparison/*_kfold_agg.md`, incl.
  `{ocean_r05,land_r15}_de_vs_baselines_kfold_agg.md`): DE > XGB > LinReg —
  ocean R² 0.708 / 0.675 / 0.417; land 0.552 / 0.522 / 0.24 (robust median;
  LinReg land keeps the known single-fold OOD artifact).
- **TCCON chains rerun for all three models** (75 cases each; XGB/LinReg with
  `REBUILD=1 MAKE_PLOTS=0`): AK mean |bias| after — DE **0.82** < XGB **0.88**
  < LinReg **1.03** ppm; fp-RMSE after — **1.20 < 1.41 < 1.95**.
- **Decisive near-cloud (≤10 km) land tail** (by_cld_agg, mean_rmse_after):
  DE **1.30** < XGB **1.66** < LinReg **2.39** ppm — ordering identical to the
  2026-07-08 verdict (1.30/1.68/2.37).

## Independent validation (regenerated on the new tag)

- **ATom** (8 dates / 17 legs, AK pseudo-columns): |resid| all legs
  0.493 → **0.426** ppm (was → 0.443); near-cloud (n=14) 0.532 → **0.445**
  (was → 0.460). 2017-10-09 far-cloud control: +0.25 (barely moved). Baselines
  also rerun under `{linreg,xgb}/..._o05l15/atom/`.
- **Ship EM27** (4 station-days): scatter still collapses (2019-06-09 σ
  0.75→0.30); corrected residuals +1.17 / +0.44 / +0.80 / +2.30 ppm
  (old +1.13/+0.47/+0.81/+2.23); clear-sky control 2019-06-22 ~inert
  (+0.77→+0.80). Baselines under `{linreg,xgb}/..._o05l15/ship/`.
- **Drift era** (21 post-2022 station-days): near-cloud fp-RMSE 3.04 → 1.39;
  station-day RMS resid 1.52 → 0.97.

## Launcher/infra changes (all committed to the scripts)

- TCCON atrain + drift launchers, `atom/ship_deepens.sh`, and the ATom/Ship
  python `TAG` constants now point at the foldpca dirs / new tag.
- The atrain DE build now passes `--emit-members` (mu_NN columns), so the
  uncertainty layer can rerun on this tree WITHOUT a rebuild; smoother-null
  columns are emitted by default.
- Launchers are laptop-safe: the CURC-only `xgb_cloud` classifier args are
  dropped when the dirs are absent (P(near)/gate are diagnostics only), and a
  `DYLD_FALLBACK_LIBRARY_PATH` fallback borrows `libomp.dylib` from a sibling
  conda env so ml310's pip xgboost loads on macOS.
- `atom_baseline.sh` reuses `atom_merged/` from the NEW tag tree
  (model-independent inputs copied over from the old tree).
- Env note: ml310 has scikit-learn 1.7.2 vs the 1.8.0 used to pickle the CURC
  pipelines → InconsistentVersionWarning on unpickle (scaler/PCA behavior
  stable; DE control numbers reproduce, so accepted).

## L′ status — regenerated vs still carrying l′

Regenerated with L′ today: all 75 atrain TCCON case figures + aggregate
figures, 21 drift cases, ATom (pseudo-column summary, 4-panel comparisons,
MODIS overlays), Ship (4 cases + summary), Fig 2 land-class effect heatmap
(from cached CSV; CSV labels refreshed too), Fig 4 Tasman + N-Pacific case
figures, A1 category atlases.

Still on the old l′ (need their own regen pass):
- `run_all.py` full-parquet figures (Fig 1b,c decay, Fig 3 shadow/brightening,
  k1_k2 profiles, A11 sub-pixel/ROC) — needs >34 GB RAM or CURC.
- A5 savgol A/B (`compare_savgol_fits.py`) — needs `fitting_details_*.h5` on CURC.
- Composed manuscript figures `make_*.py` (fig01a, fig05b, fig07, fig08,
  fig09b) — fig07/fig08 numbers should be re-checked against the NEW reports
  before regen; fig09b (k1 contrast) can regen from the new
  `nassar_k1_contrast.csv` once the variant retrain lands.
- Poster figure (legacy styling by decision).
- (Second pass, below, regenerated the Nassar case plots + transects and the
  smoother-null figure under the new tag — those now carry L′.)

## Second pass (same day): ablation, Nassar, smoother-null, uncertainty, pptx

All four deferred layers were rerun locally, plus the comparison slide deck.

- **⚠ VARIANT MODEL-HEALTH FINDING (needs a CURC retrain):** in the fold-PCA
  retrain, **land fold f2 of EVERY feature-set variant diverged** (held-out
  RMSE 3.8k–43k ppm, R² ≈ −10⁸; ocean folds all healthy; production full folds
  all healthy). Land f1/f3 of the variants also look under-converged (R²
  0.01–0.24 vs 0.44–0.50 for f0/f4) — the variants train WITHOUT lndo01, i.e.
  exactly the beta-NLL land instability the reg ablation fixed. All variant
  trees here pool land f0,f1,f3,f4 (20 members; builder
  `workspace/build_ablation_variant_trees.sh` documents the exclusion).
  **Recommendation:** retrain the 5 land variants (consider `--norm layer
  --dropout 0.1`) and rerun `make_featureset_ablation_doc.py` +
  the Nassar variant builds before quoting variant numbers.
- **Feature-set ablation** (5 variants × 75 cases rebuilt + per-tree reports;
  combiner `workspace/make_featureset_ablation_doc.py` →
  `FEATURESET_ABLATION_QF_2026-07-15.md`). PRELIMINARY per the caveat above:
  with the current variant models `no_spec` is NO LONGER TCCON-neutral
  (pooled ΔRMSE ≈ +0.8 ppm with the healthy-fold pools) — but this is confounded by the under-converged
  land variant folds, so treat deltas as upper bounds until the retrain.
- **Nassar plume suite** (new tag tree; 8 dates full + no_spec/no_xco2
  variants, land f2 excluded): control-region null reproduces **5/7 consistent
  with controls, same 2 near-cloud flags** (Lipetsk 2015-08-01, Westar
  2023-03-13); transects, k1 contrast, preservation tables + 11 case plots
  regenerated (L′). **Channel attribution moved:** full removes 55 % [40–63]
  of control-window clear-sky spread; no_spec 44 % (Δ +11 pp — was +1 pp);
  no_xco2 28 % (Δ +27 pp). The spec share is inflated by the weak land variant
  folds — re-read after the variant retrain before revising the §8a statement.
- **Smoother-null** (4 reports + figure under `<TAG>/atrain/smoother_null/`):
  M4 conclusion reproduced exactly — smoother collapses σ MORE than the DE
  (2.23 → 0.35–0.66 vs 0.78) but leaves TCCON |bias| at 1.20–1.24 vs DE 0.82;
  fp-RMSE DE 1.20 < smoother 1.34–1.52 ppm.
- **Uncertainty layer** (Side-A inflation REFIT from the foldpca held-out
  predictions → `<TAG>/sidea_inflation_{ocean,land}.json`; Phase-4 reports at
  r100 + r50): DL random-effects D = **−0.32 ± 0.09 ppm** (r100; r50
  −0.45 ± 0.07) — the known ~0.3 ppm anchoring offset; τ 0.22 → 0.52 ppm and
  I² 14 → 51 % growing with radius (u_rep reading unchanged); ⟨z²⟩ 1.80 (r50)
  / 2.39 (r100); TOST δ=0.5: 1/75 equivalent, 13/75 differ. All conclusions
  of the 2026-07-08 CURC edition carry over.
- **PowerPoint** regenerated scripted (`workspace/make_comparison_pptx.py` →
  `results/model_comparison/tccon_r100_comparison_tables.pptx`, 6 slides,
  same structure as the 07-10 deck, fold-PCA numbers: pooled AK DE 1.22 <
  XGB 1.43 < LinReg 2.27; near-cloud land QF1 1.60 < 1.96 < 3.36 ppm).
  python-pptx installed into ml310 for this.
- The r50 robustness report for the new tag also now exists
  (`tccon_comparison_r50km.*`, from the uncertainty launcher pass).

## Follow-up 2026-07-17: variant retrain landed — preliminary numbers superseded

The lndo01 variant retrain (r15 + r05) downloaded and the whole variant stack
was rerun (trees + reports + ablation doc + Nassar builds + attribution).
**All folds healthy** (land f2 restored; no divergence). The 2026-07-15
PRELIMINARY numbers above are superseded and the verdicts revert to the
2026-07-08 story: `no_spec` TCCON-neutral again (pooled ΔRMSE +0.021 ppm, was
"+0.8"), and the channel attribution is full 55 % [40–63] / no_spec 54 %
(Δ +1 pp, was +11) / no_xco2 29 % (Δ +26 pp). Quotable editions:
`FEATURESET_ABLATION_QF_2026-07-17.md` + the regenerated
`nassar_channel_attribution.{csv,md}`. Minor leftover: land f4 of no_contam
and no_contam_and_xco2 kept the old unreg checkpoint (healthy fold; preempted
array tail) — optional `--array=4` top-up.

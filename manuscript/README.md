# Manuscript — AMT submission

Collection point for everything that goes into the paper. The narrative plan
governing section order is `log/MANUSCRIPT_FLOW_PLAN.md`; the evidence ledger
is `log/PROJECT_REVIEW.md`.

Everything under `manuscript/` is **tracked by git** (the repo-global
`*.png`/`*.csv`/`*.json`/`*.txt` ignores are overridden here; only LaTeX
build intermediates stay ignored — see the manuscript block in `.gitignore`).

## Layout

- `tex/` — manuscript LaTeX sources, bibliography, and compiled PDFs.
  - `MANUSCRIPT_METHODS_PHOTON_PATH.tex` — methods section body
    (`\input` by `MANUSCRIPT_METHODS_PHOTON_PATH_DOCUMENT.tex`, the
    standalone compile wrapper).
  - `MANUSCRIPT_APPENDIX_MODEL.tex`, `MANUSCRIPT_APPENDIX_DATES_TABLE.tex`,
    `MANUSCRIPT_APPENDIX_PREDICTOR_TABLE.tex` — appendix sections/tables.
  - `EQUIVALENCE_THEOREM_FITTING_REPORT.tex` — expanded derivation report
    (referenced from `src/spectral/FITTING_DERIVATION.md`).
  - `equivalence_theorem_fitting.bib` — shared bibliography.
  - All `\input`/`\bibliography` references are same-directory relative;
    compile from inside `tex/`.
- `figures/` — final manuscript figures (pdf+png), written directly by the
  generators in `scripts/` where one exists. Figures without a local
  generator are COPIES from the analysis trees (sources and status in the
  Display items blocks of `log/MANUSCRIPT_FLOW_PLAN.md`): figF1_case_tasman
  (Appendix F Tasman showcase, Times-italic l′ — final convention),
  fig07 (dumbbell), fig09 (smoother), fig10a/fig10b (ATom/ship),
  fig11a (Westar), and the appendix copies staged
  2026-07-22o — figD2a/figD2b (QF0/QF1 TCCON), figD3 (r50), figD4
  (station summary), figE2a/figE2b (far-cloud/clear-day controls), figE3
  (failure modes) — all from the production fold-PCA report tree
  `de_beta_nll_prof_reg_foldpca_o05l15_m5`.
  Appendix letters follow the 2026-07-22n mapping table
  in the flow plan. Draft
  captions for all figures: `log/MANUSCRIPT_FLOW_PLAN.md` §4.
- `supplement/` — GIT-IGNORED staging area for the Supplement PDF
  (S1 station-day panels, S2 ocean cases, S4 spectrum-internal set),
  rebuilt by `scripts/stage_supplement_figures.py`; assembled by
  `tex/supplement.tex`. Only `supplement_manifest.txt` is tracked.
- `tables/` — generated LaTeX tables (`tab_*.tex`), written directly by
  `scripts/make_manuscript_tables.py` from the report CSVs under
  `results/model_comparison/`. Don't hand-edit; regenerate.
- `scripts/` — manuscript figure/table generators (moved from
  `workspace/manuscript_figures/` + `workspace/make_manuscript_tables.py`
  2026-07-21). All output into `figures/` and `tables/` above; inputs are
  the git-ignored analysis trees under `results/` and `data/`. Shared
  styling still comes from `workspace/plot_style.py` (sys.path-inserted by
  each script). Production analysis code stays in `src/` and `workspace/`;
  put here only what exists purely to produce a manuscript artifact.

  Current generators (filenames follow the 2026-07-21c plan numbering —
  see the figure table in `log/MANUSCRIPT_FLOW_PLAN.md` §4):
  - `make_collocation_schematic.py` → fig01 (Methods 3.1; needs
    `data/processing/` pickles for the case date)
  - `src/models/make_deep_ensemble_figure.py` → fig02 (Methods 3.3
    architecture schematic; lives in `src/models/` because
    `deep_ensemble_ARCHITECTURE.md` documents it — run
    `PYTHONPATH=src python -m src.models.make_deep_ensemble_figure
    --only no-cloud --out-dir manuscript/figures
    --basename fig02_deep_ensemble_architecture`)
  - `make_anomaly_decay_figure.py` → fig03 (Results 4.1; two panels —
    common r10 target motivating the radii, then adopted r05/r15 targets)
    + figB2_target_sensitivity (Appendix B2: each surface under all three
    reference radii; reads the combined 2016–2020 parquet directly)
  - `make_landclass_heatmap_figure.py` → fig04 (Results 4.2 sign-rule
    heatmap, ocean column + IGBP land classes; needs the combined parquet
    and `data/MODIS/MCD12C1/` HDFs; QF0 + snow-free filter is
    load-bearing; default reference is PER-SURFACE r05/r15 — the common-r10
    robustness variant writes `internal_landclass_r10*` via
    `--reference common-r10` (internal check, not in the manuscript);
    `--qf {0,1,all}` selects the quality-flag population, writing the
    Appendix B4 files `figB4a_landclass_qf1`/`figB4b_landclass_allqf` —
    only the barren column is QF-sensitive, see plan 2026-07-22g)
  - `make_shadow_brightening_figure.py` → fig05 (Results 4.2; redraws
    from spec_sensitivity/shadow_brightening_stats.csv, bottom legend;
    --surface ocean → internal_shadow_brightening_ocean)
  - `make_baseline_ablation_figure.py` → fig06 (Results 4.3)
  - `make_significance_panel.py` → fig08 (Results 4.4 significance figure
    — split from the former two-panel Fig. 7, 2026-07-23; Fig. 7 is the
    copied station-day dumbbell `fig07_tccon_dumbbell`)
  - `make_k1_contrast_figure.py` → fig11b (Results 4.7 plume-safety panel;
    11a is the copied Westar transect `fig11a_westar_transect`)
  - `make_cv_design_figure.py` → figC3b (Appendix C CV-design schematic;
    falls back to a stylized timeline when the fold manifests are not
    local — regenerate on a machine with the production fold dirs before
    submission)
  - `make_manuscript_tables.py` → all `tables/tab_*.tex` (AK-harmonized
    only since 2026-07-21)
  - `make_appendix_c_figures.py` → figC1_dataflow, figC2_fold_timeline
    (real training_dates.json fold manifests), figC3a_random_split_inflation
    (ocean-testbed numbers from ocean_robustness_comparison.md),
    figC5_cv_model_comparison + tables/tabC5_cv_model_comparison.tex
    (parsed from the kfold_agg reports);
    C4/C7 pending — see plan 2026-07-22q
  - `stage_supplement_figures.py` → `manuscript/supplement/S*_*/` (bulk
    Supplement assets, ~295 MB; git-ignored except the tracked
    `supplement_manifest.txt`; S3/S5 sources still on CURC)

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
  Display items blocks of `log/MANUSCRIPT_FLOW_PLAN.md`): fig04 + fig05a
  (pre-restyle `results/figures/cld_dist_analysis/` outputs — regenerate
  via `run_all.py` on the next full-parquet pass), fig05b (Tasman case,
  current style), fig07a, fig08, fig09a, fig09b, fig10a, fig11 (production
  fold-PCA report tree `de_beta_nll_prof_reg_foldpca_o05l15_m5`). Draft
  captions for all figures: `log/MANUSCRIPT_FLOW_PLAN.md` §4.
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
    common r10 target motivating the radii, then adopted r05/r15 targets;
    reads the combined 2016–2020 parquet directly)
  - `make_baseline_ablation_figure.py` → fig06 (Results 4.3)
  - `make_significance_panel.py` → fig07b (Results 4.4 significance panel;
    the before/after headline panel 7a is still to compose)
  - `make_k1_contrast_figure.py` → fig10b (Results 4.7 plume-safety panel;
    10a is the Westar transect, still to compose)
  - `make_cv_design_figure.py` → figD1b (Appendix D CV-design schematic;
    falls back to a stylized timeline when the fold manifests are not
    local — regenerate on a machine with the production fold dirs before
    submission)
  - `make_manuscript_tables.py` → all `tables/tab_*.tex` (AK-harmonized
    only since 2026-07-21)

# anchor_lr — TCCON-anchored ML(raw) test (land only)

Idea test (2026-10-22): the production chain is
`raw → B11 operational BC → ML(bc)`; the ML(raw) arm is `raw → ML(raw)` and
has never passed through a TCCON anchoring step, so its overpass mean |bias|
(1.12 ppm) is not comparable with ML(bc) (0.74 ppm) even though its fp-RMSE
is.  This adds the missing step as one linear regression (LR) with the B11
land predictor family, fit ONCE to TCCON on the training-date pool, and
evaluates `raw_corr_anchor = (xco2_raw − mu_raw) − LR(f)` through the
unchanged TCCON report.

Steps (env: `conda activate er3t_v11`; run from the repo root with
`PYTHONPATH=src:workspace`):

| step | script | writes |
|---|---|---|
| 1 | `build_tccon_pool.py` | `results/anchor_lr/pool/combined_<date>_<site>/*_source.parquet`, `pool_manifest.csv` |
| 2 | `predict_pool.py` | `…/plot_data.parquet` (out-of-fold mu_raw, one held fold per date) |
| 3 | `fit_anchor_lr.py` | `results/anchor_lr/anchor_lr_land.{json,md}`, `anchor_lr_fit_cases.csv` |
| 4 | `apply_anchor.py` | `anchor_lr_{increment,guard,corrected_xco2}` columns in the ML(raw) tree; `apply_diagnostics.csv` |
| 5 | `workspace/tccon_comparison_report.py --corr-col anchor_lr_corrected_xco2 …` | `<raw tree>/anchor_lr/` |
| 6 | `workspace/make_raw_bc_ml_report.py` | five-series table (`RAW_BC_ML_TCCON_<date>.md`) |

Design decisions (see `anchor_common.py` docstring): additive form (B11 scale
fixed at 1), predictors centered at pool medians, footprint offsets as
sum-to-zero contrasts, station-day-equal weighting, fit on QF = 0 and applied
to QF 0 + 1, ocean rows NaN.  Caveat for the write-up: the intercept absorbs
the ~0.3 ppm gap between B11's direct anchoring and our AK-harmonized
reference, so part of any mean-bias gain is re-calibration; fp-RMSE and the
near-cloud / QF = 1 land slices are the cleaner readouts.  To verify against
the Lite DUG Table 5-4: `aod_fine` composition and the `albedo_quad_wco2` form.

**Outcome (2026-10-22, land, 102 cases):** the anchor closes most of the
station-day |bias| gap (AK 0.906 → 0.629 ppm vs ML(bc) 0.608) but raises
near-cloud fp-RMSE (land ≤10 km 1.226 → 1.561 ppm); the per-footprint LR
structure is cloud-contaminated, only its station-day mean is useful.
Full write-up: `log/ANCHOR_LR_TEST_2026-10-22.md`.

**Term-set sweep (same day):** `TERM_SETS` in `anchor_common.py` (b11, nocontam,
dp_only, alb_only, fp_only, geom, met, geom_met, geom_met_alb, b11_plus, spec,
spec_geom_met, spec_b11), `--per-footprint` in the fit, driver
`run_variant.sh <set> [perfp]`, cross-variant table `summarize_variants.py` →
`results/anchor_lr/variant_summary.md`.  Every retrieval-state / aerosol /
cumulant term injects near-cloud scatter (dpfrac worst); only the
cloud-insensitive geometry + prior-met terms improve fp-RMSE (1.078 vs ML(bc)
1.144 pooled) but with a site-memorization caveat (20/21 fit sites are
evaluation sites; LOSO not run).  Per-footprint coefficients never help.

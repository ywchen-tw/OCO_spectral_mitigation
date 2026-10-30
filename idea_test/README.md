# idea_test — side studies outside the manuscript scope

Each folder is a self-contained exploratory test: a `PLAN.md` stating the
hypothesis, the estimators or experiment, and pass/fail bars written *before*
running, plus the scripts and a small `results/` tree. Nothing here feeds the
production pipeline or the manuscript unless promoted explicitly.

| folder | question | status |
|---|---|---|
| `ppdf_direct_xco2/` | Can the fitted path cumulants (k1, k2 per band) give a direct XCO2 estimate without a full-physics retrieval? | closed 2026-10-29: 5–30 ppm scatter, no near-cloud signal (see PLAN.md) |
| `fit_order_study/` | Which truncation order, weighting and model form give the most stable k1, k2 per band? Instrument-noise σ(k) and residual diagnostics per variant. | done 2026-10-29 (one orbit): production order 7 inflates k1 spread 8×; order 2 or weighted gamma gives ~1 %; O2A cores sit at the series' convergence radius; residual is misfit, not noise; variance-function/Huber weighting rejected (cuts near-cloud k1 signal); slab closure: k1 = absorption-weighted mean path (ols_3 within 2–4 %), no fitter recovers var(l'), gamma shape rejected; τ-cut sweep: keep the full range, order-3 residual flat to τ 8, cuts inflate k1 scatter and cannot reach var A; replication (2018 glint ×2, 2023 target land): order 7 worst everywhere, order 2 vs 3 split; residual decomposition: the structured misfit is the missing telluric Doppler shift in the τ step (fp_abs_coeff.py, commented out), confirmed sign and 0.174 px/(km/s) slope on the target orbit, rms drops 5–13×; production fix on hold pending Steven (see PLAN.md) |

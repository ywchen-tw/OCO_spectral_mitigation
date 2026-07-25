# MCARaTS-Derived Photon Path Lengths — Mechanics, Fidelity, Truncation

**Created:** 2026-07-24 (consolidates the Appendix G photon-path discussion)
**Companions:** `workspace/rt_slab_sim/` (pipeline; docstrings in
`run_slab.py`/`fit_and_plot.py`/`make_fig_g1.py` carry the same content at
the code sites), `log/MANUSCRIPT_FLOW_PLAN.md` §Appendix G (2026-07-24b),
`results/rt_slab_sim/closure_stats.json` (quotable closure numbers).
**Purpose:** one place answering (1) how the photon path is obtained from
MCARaTS output, (2) whether it represents "real" photon paths, and (3)
whether the histogram truncation affects the Appendix G conclusions.

---

## 1. How the path is obtained (Fortran tally → Fig. G1g/h)

1. **Internal recording.** MCARaTS carries a per-photon array
   `Pho_plen(0:nz+1)` — geometric distance travelled in each model layer,
   accumulated by the ray tracer (`mcarAtm.F90`). Radiance uses local
   estimation: at every scattering event a virtual copy of the photon is
   traced toward the sensor (`mcarRad__samp1`), extending its own copy
   `PhoV_plen`. When a contribution reaches a radiance pixel,
   `sum(PhoV_plen)` is the TOTAL geometric path of that contribution:
   TOA entry → all scattering events → sensor (including the constant
   vacuum leg to the 705-km sensor).
2. **Enabling the tally.** Built-in "pathlength statistics", namelist-only
   (no source edit), injected via the `SlabMcarats.extra_nml` subclass in
   `run_slab.py`: `Rad_mplen = 3` (histogram of total path per
   contribution), `Rad_ntp = 2500`, `Rad_tpmin = 640 km`,
   `Rad_tpmax = 1140 km` (200-m bins). Each contribution adds its radiance
   weight into the bin containing its total path (`mcarRad.F90:803`); on
   output the histogram is normalized by the pixel radiance
   (`mcarRad__normal`) → the written field is the FRACTION of each pixel's
   radiance contributed by each total-path bin: a per-column,
   radiance-weighted photon path-length distribution (PPDF), Σbins ≈ 1.
3. **Output/reading.** Written next to the radiance in the GrADS-style
   output (`.bin` + `.ctl`). v0.10.4 names the ctl variable
   `b1 … Pathlength Statistics` (v0.11: `pln_0001`). `read_plen()` uses
   er3t's generic `mca_out_raw` (parses whatever the ctl lists) and selects
   by name; `collect()` stacks into `slab_rad.h5` as `plen_hist`
   `(n_wavelength, n_run, Nx, 2500)`.
4. **Reduction.** Moments per column: mean = Σhᵢ·Lᵢ/Σhᵢ (+ s.d.), at the
   CONTINUUM wavelength (slant τ ≈ 1e-4) where no absorption weighting
   exists → the pure geometric PPDF. Constant offsets (≈645 km vacuum leg
   + injection-plane accounting) are identical for all columns/solvers:
   Fig. G1g/h plots anomalies vs the clear-sky far field (x < 5 km); the
   PPDF figure's panel (d) instead anchors the clear-sky direct-bounce
   peak at relative path l = 1 and scales by
   L_direct = 60 km/cos 55° + 60 km ≈ 165 km.

## 2. Is this the path of "real" photons?

**Yes, in expectation — with precise qualifiers.** An analog estimator
(trace photons until one physically enters the sensor aperture) would
sample the same distribution with hopeless variance (satellite solid
angle). Local estimation instead adds, at every scattering event, the
probability-weighted virtual completion of the path toward the sensor;
because the weight is the probability of exactly that completion (phase
function × escape transmittance), the weighted histogram converges to the
true path-length distribution of detected photons. Standard property:
any per-path functional tallied with the contribution weight is unbiased
for the detector ensemble.

Qualifiers:

- **Radiance-weighted, not photon-count-weighted** — which is the RIGHT
  weighting for spectroscopy (the PPDF whose Laplace transform the
  spectrum samples). At the continuum the two coincide.
- **No single simulated photon "ends" at the sensor** — one trajectory
  yields contributions at every scattering event; ensemble identical in
  expectation, conceptually different.
- **Numerical caveats (all checked, none load-bearing):** hot-spot /
  numerical-diffusion smoothing (`Rad_difr*`) blurs per-PIXEL attribution
  near sharp gradients (path values untouched); 200-m bin discretization
  negligible at km-scale features; out-of-range truncation → §3; constant
  offset → interpreted only through anomalies/differences/variances.
- "Real" means real within the discretized scene (grid cells, tabulated
  Mie phase functions), not nature.
- **Geometric vs absorption-weighted (the G3 caveat):** the tally is
  geometric path; the spectral fit senses absorption-weighted path
  (layers ∝ absorber density). They agree where surface-reflected photons
  dominate — hence r(⟨l′⟩, tallied mean) = 0.999 dark / 0.987 bright and
  r(var(l′), tallied var) = 0.989 bright (clear columns) — and diverge
  where high-altitude scattering carries the radiance (dark-surface
  shadow band, in-cloud). Exact second-moment closure needs an
  absorption-weighted accumulator in `mcarRad.F90` (identified future
  work; minimal-edit anchors documented in the 2026-07-24 exploration).

## 3. Truncation (paths > Rad_tpmax dropped): measured impact

Measured on the production (1e9-photon) continuum histograms:

- **Dropped radiance fraction:** 0.1–0.3 % in clear/IPA columns (a nearly
  uniform floor of genuinely long multiple-scattering paths; larger
  relative floor over the dark surface because the normalizing radiance
  is smaller) rising to **0.7–1.1 % in the 3-D shadow band** (worst:
  1.06 % at x = 15.8 km, dark). IPA missing fraction is flat across all
  columns.
- **Tail shape:** e-folding length ~50–84 km near the cap; measured
  missing mass ≈ 4× the exponential extrapolation → the true tail is
  heavier than exponential, so effects are BOUNDED, not corrected.
- **Moment bias (exponential-fit estimate; heavier-tail bound ≲ 2×):**
  mean +0.2 km (clear) to +0.5–0.6 km (shadow); s.d. +1.5–3 km against a
  shadow-band s.d. of 26–29 km (≈5–10 %). Differential (anomaly-relevant)
  part: ~+0.3 km mean, ~+1.5 km s.d. in the shadow band.

**Why the conclusions are unaffected:**

1. Bias direction is CONSERVATIVE for every claim: truncation removes
   long-path weight preferentially in the 3-D shadow band, so the true
   3-D shadow-band moment excursions are slightly LARGER than plotted —
   it weakens the signal, it cannot create it.
2. The IPA null is untouched: the uniform missing-fraction floor adds no
   spatial structure; "flat outside the cloud" stands exactly.
3. The closure correlations are shape statistics; a few-percent smooth
   damping does not move r = 0.999/0.987, and the fitted-cumulant axis
   comes from radiances, which are complete (truncation touches only the
   tally, never the radiance).

**Mitigation knob (if ever needed):** raise `PLEN_MAX_M` 1140 → 2140 km
and `PLEN_NBIN` 2500 → 7500 (keeps 200-m bins) and rerun the Blanca sweep
(output ×3, still small). NOT worth a rerun for Appendix G; instead one
appendix/caption sentence: *"the path histogram truncates 0.1–1.1 % of the
radiance weight at 500 km beyond the direct path, largest in the shadow
band; this slightly underestimates the tallied shadow-band moments and is
conservative with respect to the 3-D–IPA contrast."* Fold the wider cap
into any future absorption-weighted-tally rerun.

## 4. Manuscript-safe summary sentence

> The tallied histogram is a statistically exact (importance-weighted)
> estimate of the geometric path-length distribution of the photons
> contributing to each pixel's radiance, within the discretized scene;
> the fitted cumulants sense the corresponding absorption-weighted
> distribution, and the two agree wherever surface-reflected photons
> dominate the signal.

# fit_order_study — how should the three bands be fitted, and at what truncation order?

Side study (2026-10-29), outside the manuscript scope. Nothing in `src/` is
changed; the production knob is `FIT_ORDER = (7, 3, 7)` in `src/constants.py`
and the `fit_order` argument of `spectral.fitting.process_orbit`.

## Motivation

The direct-XCO2 test (`../ppdf_direct_xco2/`) found a 6–10 % per-footprint
scatter of the cross-band k1 ratio after removing per-footprint-index means.
The formal variance amplification of the cumulant design matrix on real τ
grids (one orbit, 2018-10-18) is

| band, order | σ(k1) per unit σ(ln T) | σ(k2) per unit σ(ln T) | cond(A) |
|---|---|---|---|
| SCO2, 2 | 0.12 | 0.07 | 1e1 |
| SCO2, 3 | 0.28 | 0.39 | 6e1 |
| SCO2, 7 (production) | 2.2–2.7 | 14–15 | 4e5 |
| O2A, 7 (production) | 1.2–1.3 | 6.5–7.5 | 1e6 |
| WCO2, 3 (production) | 1.5–1.6 | 14–16 | 1e3 |

so a 0.01 scatter in ln T gives ~2.5 % in k1 at order 7 but ~0.1 % at order 2,
and σ(k2) at order 7 exceeds k2 itself (hence the zero-bound pinning).

## Hypotheses

H1. Lowering the truncation order (2–3) cuts the per-footprint k1 scatter by
    several times at the cost of a small, stable bias in k1 that the
    far-clear calibration absorbs.
H2. Instrument-noise weighting helps less than the order change, because the
    residual is misfit-dominated (reduced χ² ≫ 1).
H3. Dropping channels beyond the series' convergence range (τ > τ_cut) lets
    a low order fit O2A without the line-core bias.
H4. The two-parameter gamma form (k1, κ) is at least as stable as order 2 and
    keeps a physically bounded k2 = k1²/κ.

## Variants (per band, per sounding, raw ln T = production `_nosg` target)

| tag | model | weighting | channels |
|---|---|---|---|
| `ols_N` | cumulant order N = 1..7, k1,k2 ≥ 0 (BVLS fallback, as production) | none | all valid, edge channels dropped |
| `wls_N` | same | 1/σ²(ln T), σ = noise/rad from the L1B SNR model | same |
| `cut_N` | same as `ols_N` | none | τ ≤ τ_cut (3.0) |
| `gamma` | ln T = c − κ ln(1 + k1 τ/κ) | 1/σ² | all valid (subsample) |

Noise model (verified against the per-sounding `snr_*_l1b` continuum SNR,
form D in the notebook check): with x = 100·rad/MaxMS,
σ_rad = (MaxMS/100)·sqrt(c_photon²·x + c_bg²), c = `InstrumentHeader/snr_coef[..., (0, 1)]`.

## Metrics

Per sounding/band/variant: k1, k2, intercept, formal σ(k1), σ(k2) from the
noise covariance (sandwich for OLS, (AᵀΣ⁻¹A)⁻¹ for WLS), RSS-based σ, reduced
χ², residual rms, lag-1 residual autocorrelation in channel order, bound flag,
channel count.

Aggregated on far-clear (cld_dist > 20 km, QF0) footprints:
median formal σ(k1)/k1; robust spread of k1 after per-fp-index mean removal;
per-fp-index mean spread; robust spread of k1_S/k1_O2A and k1_W/k1_O2A after
per-fp mean removal (direct-XCO2 proxy, ×400 ppm); k1 median relative to the
production reference `ols_7` (bias); k2 ≤ 1e-6 fraction; median χ²; median
residual rms; median lag-1 autocorrelation. Near-cloud (< 3 km): corr(k1,
xco2_bc_anomaly) as a secondary, single-orbit indication.

## Pass/fail written before running

- H1 holds if the far-clear robust spread of k1_S/k1_O2A (per-fp removed) for
  `ols_2` or `ols_3` is ≤ half of `ols_7`.
- H2 holds if `wls_N` improves that spread by < 20 % relative to `ols_N`, and
  median χ² of `ols_7` > 3.
- H3 holds if `cut_2`/`cut_3` on O2A has |k1 bias| vs `ols_7` < 1 % and a
  spread ≤ `ols_2`.
- H4 holds if `gamma` matches or beats `ols_2` on spread and has κ finite
  (k2 > 0) in > 90 % of far-clear footprints.

## Run

```
python idea_test/fit_order_study/run_study.py \
    --l1b data/OCO2/2020/001/29252a_GL/oco2_L1bScGL_29252a_191231_B11006r_220715045716.h5 \
    --fp-tau results/2020-01-01/29252a_GL/fp_tau_combined.h5 \
    --fit-details results/fitting_details/fitting_details_2020-01-01_29252a_GL.h5
```
Outputs in `results/`: `per_sounding_<orbit>.parquet`, `summary_<orbit>.csv`,
`summary.md`, figures `fig_*.png`.

## Result 2026-10-29 — orbit 29252a_GL (2020-01-01, glint, all ocean, 52°S–24°N), 10 365 soundings, 1 240 far-clear

Reproduction: the production-order variants (`ols_7`, `ols_3`, `ols_7`)
reproduce the stored `_nosg` k1 to ~1e-6, so the study runs on the production
target. Full tables: `results/summary.md`; per-sounding: `results/per_sounding_29252a_GL.parquet`.

Far-clear k1 spread after per-fp-index mean removal (robust SD, %), and the
spread of the per-fp means:

| band | production | best low order | gamma | per-fp mean spread: production → low order |
|---|---|---|---|---|
| O2A | ols_7: 5.0 | wls_2: 0.11, ols_3: 0.69 | 0.32 | 4.2 → 0.3 |
| WCO2 | ols_3: 5.6 | ols_1: 0.68, ols_2: 3.4, wls_4: 2.4 | 5.0 | 2.6 → 0.6 |
| SCO2 | ols_7: 8.0 | ols_3: 0.58, ols_2: 1.4 | 1.1 | 22.6 → 0.3 |

SCO2/O2A ratio spread (the direct-XCO2 proxy): ols_7 9.0 % → ols_3 0.94 %,
ols_2 1.2 %, gamma 1.1 % (≈ 4–5 ppm per footprint instead of 36).

- **H1 PASS.** The production scatter is variance amplification of the
  order-7 (and WCO2 order-3) design, not scene or instrument. The
  footprint-index systematic reported in `../ppdf_direct_xco2/` is an
  artefact of the same amplification (different fp have different τ grids)
  and disappears at order ≤ 3.
- **H2 holds.** Reduced χ² is 340–720 for every variant: the residual is
  misfit, not photon noise (residual rms 0.05 WCO2, 0.17 O2A, 0.23 SCO2 in
  ln T against a noise floor of ~0.005). Noise weighting changes the spread
  by < ±30 % except O2A order 2, where it helps (0.84 → 0.11 %). Formal
  σ(k1) from noise is 0.05–0.3 % at low order: not the limiting term.
- **H3 FAIL.** τ-cut at 3.0 makes every band worse; the line cores constrain
  the curvature and should stay.
- **H4 partial.** Gamma matches order 2–3 on spread, keeps k2 > 0 always,
  κ never at bound; the noise weighting pulls it toward the low-τ channels so
  it misses the deep-line trend (example figure). Bias −2.5 % vs production.
- **k2.** At order 3 the fit drives k2 to the zero bound in 97 % (SCO2) and
  63 % (O2A) of footprints: with a k3 term present, the data prefer concave-down
  curvature. At order 2 k2 is positive everywhere (SCO2 median 0.065, O2A
  0.12, WCO2 0.58). The two are not the same quantity; k2 is only
  interpretable once the order is fixed and the spread understood.
- **The spread of ln T at fixed τ is the dominant term** and is structured:
  lag-1 autocorrelation 0.4–0.7, visible streaks of channel populations in
  WCO2 and SCO2 (example figure). This is the next object of study (species
  split, vertical weighting, dispersion), not the solver.

Recommendation for the production knob: `FIT_ORDER = (2, 2, 2)` or
`(3, 2, 3)` for k1 with `_nosg`; order 2 if k2 is to be used. Decide after
the same study on two more orbits (land nadir, ocean glint) and after the
spread diagnostics, because the choice of order changes the k2 definition
that downstream models were trained on.

### Addendum 2026-10-29 — does order 2 break the expansion? (unweighted gamma added, 4 000 soundings)

The cumulant series is the Taylor expansion of the log Laplace transform of
p(l'); for a gamma path distribution it converges only for τ·k1/κ < 1 with
κ = k1²/k2, and truncating at order 2 is the Gaussian approximation of p(l').
Measured with the unit-weight gamma fit (`gamma_u`, all orders resummed):

| band | κ median (IQR) | τ_max·k1/κ | ols_2 / gamma_u − 1 (median, fp-to-fp robust SD) | ols_3 / gamma_u − 1 |
|---|---|---|---|---|
| O2A | 5.9 (4.9–6.9) | 1.06 | −4.9 %, 1.8 % | −13.2 %, 1.4 % |
| WCO2 | 1.9 (1.5–3.7) | 0.40 | −1.6 %, 1.5 % | +4.0 %, 3.3 % |
| SCO2 | 14.1 (11.7–18.5) | 0.30 | −0.7 %, 0.5 % | −4.1 %, 1.9 % |

- SCO2: the series converges fast; order 2 is a 0.7 % effective-coefficient
  bias with 0.5 % scene variation. Acceptable.
- O2A: the line cores sit at the convergence radius. No truncation order is a
  Taylor approximation there (the M9 note); order 2 is 5 % low, order 3 13 %
  low, and both vary ~1.5 % footprint to footprint. The closed-form model is
  required for O2A if k1 is to mean ⟨l'⟩.
- WCO2: κ ≈ 2 is not credible as path variance at τ < 0.7; the curvature is
  the channel-population misfit (streaks), so neither order nor model choice
  fixes WCO2 until the spread is understood.
- Order 3 is worse than order 2 in every band: with k2 bounded at zero the
  cubic is no longer a truncated series but a distorted fit.

Conclusion: choose the model, not the order. Unit-weight gamma (k1, κ, c)
is the principled replacement for the (7, 3, 7) polynomial; order 2 is an
acceptable stand-in for SCO2 only.

Far-clear spread of the two gamma variants (500 far-clear of the 4 000 subsample):

| band | gamma (noise-weighted) k1, spread | gamma_u (unit weights) k1, spread | ols_2 k1, spread |
|---|---|---|---|
| O2A | 1.01, 0.32 % | 1.16, 1.28 % | 1.10, 0.84 % |
| WCO2 | 1.03, 4.4 % | 1.12, 3.9 % | 1.08, 3.4 % |
| SCO2 | 0.99, 1.25 % | 1.06, 1.8 % | 1.05, 1.4 % |

The resummed model is *less* stable than order 2 under unit weights and
*more* stable under noise weights, because the weights decide how much the
deep-line channels, where the ln T spread is largest, pull k1 and κ. The
same band yields k1 from 1.00 to 1.16 depending on that choice, so the
remaining ±8 % is interpretational and set by the spread, not by the solver.
Stability (for features) and accuracy (for ⟨l'⟩) are different targets:
order 2 or weighted gamma for the former; the latter waits on the spread
diagnostics.

## Weighting test (Opus agent) — 2026-10-29, orbit 29252a_GL

Question: does a pooled, scene-independent misfit variance function σ²(τ)
(optionally + Huber IRLS) give a k1 that is more stable far from clouds
*without* weakening the near-cloud contrast that anomaly features rely on?

### Implementation

- `weighting.py`: `VarFunc` (σ²_noise + a + b·τ^p, fitted in log space to
  equal-count τ bins, p free on 0.5–4 or fixed), `TabVarFunc`
  (non-parametric log-log interpolation of the binned variance),
  `fit_weighted` (one routine for order 2 and gamma, arbitrary channel
  variance, optional Huber IRLS with 2 reweightings, noise and calibrated
  sandwich covariances, AR(1) inflation n/n_eff = (1+ρ)/(1−ρ), χ² against the
  weight variance and against a common calibration, down-weight mask),
  `k1_kernel` (equivalent kernel of k1).
- `run_weighting.py`: stage 1 first-pass `ols_2`/`gamma_u` + production-order
  refs; stage 2 variance functions from **far-clear soundings only**, per
  model × band × fp (and band-only); stage 3 all variants on all 10 365
  soundings (gamma on all, no subsample); stage 4 tables, bootstrap, figures.
  Production reproduction: median |Δk1|/k1 = 1.7e-6 / 1.6e-7 / 2.4e-6.
- `weight_extra.py`: systematic-vs-random misfit split, σ(k1) calibration,
  latitude-local near/far contrast for k1 and k2 with bootstrap CIs, joint
  Fisher distance, solar-line share of Huber events.

Variants (suffix `_2` = order 2, `_g` = gamma; all fixed per orbit × band × fp unless stated):

| tag | weights |
|---|---|
| `ols_2`, `gamma_u` | unit (first pass) |
| `wls_2`, `gamma` | 1/σ²_noise |
| `vf_*` | 1/(σ²_noise + a + b·τ^p), robust (MAD) bin variance, per fp, p free — **the plan** |
| `vfb_*` | same, one function per band (all fp pooled) |
| `vfp2_*` | same, p = 2 |
| `vfn_*` | non-parametric robust bin variance (tests the functional form) |
| `vfm_*` | non-parametric **mean-square** bin variance (χ² ≈ 1 by construction; added) |
| `vfh25_*`, `vfh_*`, `vfh4_*` | `vf` + Huber IRLS, cutoff 2.5 / 3 / 4 — **the plan** |
| `vfs_*` | adaptive: a, b from the sounding's own first-pass residuals (p from the pooled fit) |
| `vfc_*` | per-channel pooled far-clear mean-square residual (fixed alternative to Huber; added) |
| `vfch_2` | `vfc` + Huber 3 (what is left for Huber after fixed per-channel weights; added) |
| `sm_*` | unit weights, channels with solar-line depth > 5 % removed (fixed mask; added) |

Common calibration for χ²_cal and σ(k): the per-fp mean-square function
(`vfm`), i.e. the actual far-clear error variance, whatever the weights.

### Variance function (step 2)

Far-clear robust bin variance, all fp pooled (`results/weight_varfunc_29252a_GL.csv`, figure `weight_varfunc_29252a_GL.png`):

| band | p (free) | a | b | log-rms misfit (p free / p = 2) | per-fp p range |
|---|---|---|---|---|---|
| O2A | 2.25 | ≈ 0 | 0.017 | 0.71 / 0.81 | 2.20–2.35 |
| WCO2 | 2.15 | 5.0e-5 (σ ≈ 0.007, 3× noise) | 0.18 | 0.41 / 0.42 | 2.10–2.20 |
| SCO2 | 2.30 | ≈ 0 | 0.042 | 0.40 / 0.49 | 2.25–2.35 |

σ_misfit ∝ τ^1.1 holds to within a factor 1.5–2 over two decades of τ;
O2A flattens above τ ≈ 2 and every band drops in the deepest bin. The
first-pass model (order 2 vs gamma) does not change the function. Below
τ ≈ 0.1 in O2A and WCO2 the bin mean square is 3–10× the robust variance:
the low-τ residual is heavy-tailed (a channel population, see Huber below).

### Results (full tables: `results/weight_summary_29252a_GL.md`)

Table W1 — far-clear stability (n = 1240). Per band: k1 median / robust spread after per-fp mean removal (%) / spread of the per-fp means (%). Last columns: spread of k1_band/k1_O2A (%).

| variant | O2A k1 | O2A spread | O2A fp-mean | WCO2 k1 | WCO2 spread | WCO2 fp-mean | SCO2 k1 | SCO2 spread | SCO2 fp-mean | SCO2/O2A | WCO2/O2A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prod (7,3,7) | 1.049 | 4.97 | 4.17 | 1.126 | 5.64 | 2.63 | 1.009 | 8.04 | 22.60 | 9.03 | – |
| ols_2 | 1.102 | 0.84 | 0.28 | 1.083 | 3.42 | 0.62 | 1.050 | 1.39 | 0.33 | 1.22 | 2.98 |
| wls_2 | 0.999 | 0.11 | 0.30 | 1.003 | 3.86 | 0.63 | 0.969 | 1.46 | 0.89 | 1.39 | 3.81 |
| vf_2 | 1.006 | 1.08 | 0.56 | 1.006 | 3.04 | 5.05 | 0.988 | 1.38 | 0.45 | 0.94 | 1.90 |
| vfb_2 | 1.006 | 1.09 | 0.65 | 1.004 | 3.08 | 5.93 | 0.988 | 1.41 | 0.45 | 0.93 | 1.88 |
| vfp2_2 | 1.017 | 1.04 | 0.50 | 1.007 | 2.81 | 4.96 | 0.993 | 1.28 | 0.36 | 1.03 | 1.57 |
| vfn_2 | 1.014 | 1.27 | 0.93 | 1.004 | 3.60 | 5.93 | 0.991 | 1.11 | 0.84 | 1.15 | 2.31 |
| vfm_2 | 1.052 | 0.95 | 0.51 | 1.039 | 4.80 | 4.63 | 0.997 | 1.06 | 0.33 | 1.25 | 2.75 |
| vfh_2 | 0.999 | 0.88 | 0.68 | 1.015 | 5.16 | 3.94 | 0.988 | 1.38 | 0.45 | 0.98 | 4.13 |
| vfs_2 | 1.006 | 1.08 | 0.66 | 1.001 | 2.46 | 4.88 | 0.989 | 1.28 | 0.41 | 0.97 | 1.48 |
| vfc_2 | 1.072 | 1.36 | 0.99 | 1.009 | 7.13 | 2.07 | 1.012 | 2.60 | 0.64 | 0.90 | 3.46 |
| sm_2 | 1.117 | 0.95 | 0.36 | 1.126 | 3.05 | 0.98 | 1.048 | 1.15 | 0.37 | 2.66 | 2.60 |
| gamma_u | 1.155 | 1.41 | 0.49 | 1.115 | 5.24 | 0.91 | 1.057 | 1.85 | 0.34 | 2.24 | 4.47 |
| gamma | 1.010 | 0.34 | 0.36 | 1.029 | 6.22 | 0.73 | 0.992 | 1.14 | 0.58 | 1.12 | 5.82 |
| vf_g | 1.008 | 1.12 | 0.42 | 1.006 | 2.96 | 5.11 | 0.988 | 1.39 | 0.46 | 0.94 | 1.77 |
| vfh_g | 1.001 | 0.89 | 0.58 | 1.014 | 5.16 | 4.04 | 0.988 | 1.39 | 0.46 | 0.97 | 4.11 |
| vfs_g | 1.007 | 1.10 | 0.73 | 1.001 | 2.38 | 4.86 | 0.989 | 1.29 | 0.42 | 0.97 | 1.46 |
| sm_g | 1.181 | 1.63 | 0.63 | 1.165 | 4.53 | 1.33 | 1.056 | 1.51 | 0.40 | 4.00 | 4.48 |

Table W2 — fit diagnostics (far-clear medians). χ²_own: against the variant's own weight variance; χ²_cal: against the common mean-square calibration σ²_msq(τ); σ_infl: calibrated sandwich σ(k1)/k1 × sqrt((1+ρ)/(1−ρ)); k2 ≤ 0: fraction at the bound.

| variant | χ²_own O2A/WCO2/SCO2 | χ²_cal O2A/WCO2/SCO2 | σ_infl(k1) % O2A/WCO2/SCO2 | k2 median O2A/WCO2/SCO2 | k2 ≤ 0 WCO2/SCO2 |
|---|---|---|---|---|---|
| ols_2 | 0.0/0.0/0.1 | 1.19/1.17/1.14 | 4.0/12.6/8.3 | 0.116/0.584/0.065 | 0.00/0.00 |
| wls_2 | 441.5/337.1/577.8 | 1.10/1.17/1.27 | 4.0/12.9/7.0 | 0.080/0.440/0.100 | 0.00/0.00 |
| vf_2 | 3.3/2.9/1.0 | 1.13/1.17/1.10 | 3.7/8.2/4.4 | 0.061/0.194/0.021 | 0.15/0.20 |
| vfb_2 | 3.4/2.8/0.9 | 1.13/1.17/1.10 | 3.7/8.2/4.5 | 0.061/0.186/0.021 | 0.15/0.23 |
| vfp2_2 | 2.8/2.9/1.0 | 1.11/1.17/1.10 | 3.3/8.1/4.0 | 0.069/0.216/0.026 | 0.12/0.06 |
| vfn_2 | 3.6/2.9/0.9 | 1.08/1.17/1.10 | 3.7/8.2/3.8 | 0.084/0.261/0.035 | 0.09/0.00 |
| vfm_2 | 1.1/1.2/1.1 | 1.07/1.17/1.10 | 2.7/7.0/3.6 | 0.097/0.432/0.036 | 0.00/0.00 |
| vfh_2 | 3.3/2.9/1.0 | 1.14/1.17/1.10 | 3.7/8.2/4.4 | 0.057/0.235/0.021 | 0.15/0.20 |
| vfs_2 | 3.5/2.6/0.9 | 1.13/1.17/1.10 | 3.7/8.4/4.4 | 0.061/0.167/0.021 | 0.15/0.16 |
| vfc_2 | 0.9/1.0/1.1 | 1.09/1.17/1.12 | 10.0/14.8/9.6 | 0.105/0.364/0.054 | 0.09/0.00 |
| sm_2 | 0.0/0.0/0.1 | 1.17/1.18/1.13 | 4.0/14.0/8.4 | 0.122/0.699/0.064 | 0.00/0.00 |
| gamma_u | 0.0/0.0/0.1 | 1.19/1.17/1.14 | 5.8/14.8/9.4 | 0.210/0.908/0.080 | 0.00/0.00 |
| gamma | 442.3/336.6/576.9 | 1.06/1.17/1.26 | 4.8/14.4/8.8 | 0.107/0.661/0.152 | 0.00/0.00 |
| vf_g | 3.1/2.9/1.0 | 1.09/1.17/1.10 | 4.2/8.4/4.5 | 0.071/0.193/0.022 | 0.00/0.00 |
| vfh_g | 3.1/2.9/1.0 | 1.10/1.17/1.10 | 4.2/8.4/4.5 | 0.065/0.236/0.022 | 0.00/0.00 |
| vfs_g | 3.5/2.6/0.9 | 1.09/1.17/1.10 | 4.2/8.6/4.6 | 0.069/0.160/0.022 | 0.00/0.00 |
| sm_g | 0.0/0.0/0.1 | 1.20/1.18/1.14 | 6.1/16.8/9.5 | 0.235/1.156/0.079 | 0.00/0.00 |

Table W3 — near-cloud signal. "local": each footprint minus the median of reference footprints (cld_dist > 10 km, QF0) within ±1° latitude, per-fp means removed; σ_loc = robust SD of far-clear local values; SNR = median local shift of cld_dist < 3 km footprints / σ_loc (n = 1053 with a local reference); [95 % bootstrap CI]. "global": shift vs far-clear without the local reference (confounded by latitude/airmass, shown for completeness).

| variant | O2A σ_loc % | O2A SNR k1 | O2A SNR k2 | WCO2 σ_loc % | WCO2 SNR k1 | WCO2 SNR k2 | SCO2 σ_loc % | SCO2 SNR k1 | SCO2 SNR k2 | global SNR k1 O2A/WCO2/SCO2 |
|---|---|---|---|---|---|---|---|---|---|---|
| ols_2 | 0.14 | -1.47 [-1.85, -1.23] | 0.68 | 0.43 | 2.72 [2.36, 3.01] | 2.42 | 0.74 | 3.47 [3.03, 3.95] | 3.40 | -2.72/-0.27/0.81 |
| wls_2 | 0.11 | -0.60 [-0.81, -0.31] | 1.31 | 0.66 | 1.97 [1.68, 2.26] | 2.06 | 0.83 | 3.74 [3.21, 4.26] | 3.74 | -5.33/0.28/1.53 |
| vf_2 | 0.30 | -0.28 [-0.48, -0.13] | 0.78 | 1.12 | 1.62 [1.32, 1.85] | 1.11 | 0.74 | 3.60 [3.10, 4.07] | 3.20 | -1.48/0.02/2.08 |
| vfb_2 | – | -0.30 | – | – | 1.59 | – | – | 3.64 | – | -1.48/0.02/2.05 |
| vfp2_2 | – | -0.44 | – | – | 1.62 | – | – | 3.65 | – | -1.57/-0.01/2.22 |
| vfn_2 | – | -0.37 | – | – | 1.57 | – | – | 3.50 | – | -1.52/0.19/2.41 |
| vfm_2 | 0.18 | -0.67 [-0.92, -0.48] | 0.73 | 0.94 | 2.15 [1.83, 2.44] | 1.97 | 0.75 | 3.70 [3.17, 4.24] | 3.29 | -1.98/0.02/2.33 |
| vfh_2 | 0.29 | -0.23 [-0.40, -0.04] | 0.80 | 1.06 | 1.84 [1.60, 2.08] | 1.24 | 0.75 | 3.60 [3.10, 4.07] | 3.17 | -1.53/0.01/2.07 |
| vfs_2 | 0.29 | -0.27 [-0.48, -0.02] | 0.88 | 1.12 | 1.97 [1.71, 2.24] | 1.43 | 0.79 | 3.36 [2.93, 3.80] | 2.86 | -1.39/0.62/2.23 |
| vfc_2 | 0.28 | -0.53 [-0.73, -0.34] | 0.59 | 1.00 | 2.31 [2.05, 2.73] | 2.02 | 0.72 | 3.34 [2.87, 3.89] | 2.05 | -1.32/0.36/0.59 |
| sm_2 | 0.17 | -1.26 [-1.47, -1.05] | 0.56 | 0.70 | 1.25 [1.14, 1.42] | 1.21 | 0.80 | 3.29 [2.96, 3.68] | 3.27 | -1.93/-0.88/0.98 |
| gamma_u | 0.22 | -0.24 [-0.43, -0.06] | 0.81 | 0.54 | 2.76 [2.38, 3.09] | 2.40 | 0.83 | 3.47 [3.11, 3.95] | 3.06 | -0.16/-0.33/0.64 |
| gamma | 0.13 | 0.11 [-0.07, 0.29] | 1.41 | 0.75 | 2.22 [1.87, 2.47] | 2.02 | 0.89 | 3.89 [3.27, 4.35] | 3.34 | 0.51/0.10/1.99 |
| vf_g | 0.31 | -0.23 [-0.37, -0.00] | 0.79 | 1.12 | 1.60 [1.32, 1.82] | 0.97 | 0.75 | 3.57 [3.07, 4.03] | 3.38 | -1.28/0.02/2.09 |
| vfh_g | – | -0.18 | – | – | 1.82 | – | – | 3.58 | – | -1.33/0.01/2.09 |
| vfs_g | 0.32 | -0.13 [-0.32, 0.08] | 0.93 | 1.12 | 1.98 [1.71, 2.26] | 1.31 | 0.79 | 3.36 [2.96, 3.80] | 2.82 | -1.18/0.64/2.25 |
| sm_g | 0.28 | -0.15 [-0.35, 0.01] | 0.70 | 0.84 | 1.39 [1.22, 1.56] | 1.38 | 0.88 | 3.36 [3.04, 3.77] | 3.01 | -0.42/-0.94/0.78 |

Table W4 — joint near/far separability (local Fisher distance on far-clear covariance, cld_dist < 3 km) and corr(k1, xco2_bc_anomaly) near clouds (n = 566).

| variant | 6 coef [CI] | 3 × k1 [CI] | O2A (k1,k2) | WCO2 (k1,k2) | SCO2 (k1,k2) | corr(k1, anom) O2A/WCO2/SCO2 |
|---|---|---|---|---|---|---|
| ols_2 | 4.58 [4.23, 5.04] | 3.51 [3.19, 3.91] | 1.80 | 2.21 | 2.95 | 0.05/-0.20/0.08 |
| wls_2 | 4.82 [4.12, 5.34] | 3.24 [2.87, 3.59] | 1.52 | 1.93 | 3.05 | 0.20/-0.22/0.10 |
| vf_2 | 4.04 [3.76, 4.56] | 2.98 [2.66, 3.38] | 1.59 | 1.66 | 2.86 | 0.03/-0.20/0.08 |
| vfm_2 | 4.06 [3.74, 4.60] | 3.30 [2.91, 3.79] | 1.20 | 1.58 | 2.78 | 0.03/-0.21/0.09 |
| vfh_2 | 4.01 [3.68, 4.47] | 2.98 [2.58, 3.32] | 1.50 | 1.83 | 2.84 | 0.04/-0.23/0.08 |
| vfs_2 | 3.86 [3.63, 4.34] | 2.73 [2.41, 3.07] | 1.69 | 1.88 | 2.69 | 0.03/-0.19/0.08 |
| vfc_2 | 3.41 [3.02, 4.00] | 2.59 [2.26, 3.00] | 1.38 | 1.55 | 2.29 | -0.06/-0.21/-0.01 |
| gamma_u | 3.95 [3.56, 4.48] | 2.46 [2.20, 2.78] | 1.53 | 2.46 | 2.40 | -0.06/-0.21/0.07 |
| gamma | 5.24 [4.48, 5.85] | 2.90 [2.51, 3.21] | 1.50 | 1.93 | 3.44 | 0.12/-0.23/0.08 |
| vf_g | 4.05 [3.76, 4.58] | 2.97 [2.63, 3.35] | 1.64 | 1.91 | 2.83 | 0.02/-0.20/0.08 |
| vfm_g | 4.85 [4.11, 5.41] | 3.12 [2.69, 3.71] | 1.29 | 3.00 | 2.63 | -0.03/-0.21/0.09 |
| vfs_g | 3.90 [3.65, 4.34] | 2.70 [2.42, 3.03] | 1.76 | 2.11 | 2.67 | 0.02/-0.19/0.08 |
| sm_2 | 4.12 [3.78, 4.47] | 3.11 [2.83, 3.37] | 1.76 | 1.27 | 2.90 | -0.06/-0.25/0.09 |
| sm_g | 3.43 [3.03, 3.94] | 2.43 [2.15, 2.73] | 1.37 | 1.37 | 2.42 | -0.16/-0.25/0.08 |

Table W5 — Huber cutoff sensitivity (order 2; gamma within 0.1 of these). Down-weighted channel fraction far / near (%), far-clear k1 spread (%), local SNR k1 <3 km.

| variant | O2A down far/near | O2A spread | O2A SNR | WCO2 down far/near | WCO2 spread | WCO2 SNR | SCO2 down far/near | SCO2 spread | SCO2 SNR |
|---|---|---|---|---|---|---|---|---|---|
| vf_2 | 0.00/0.00 | 1.08 | -0.28 | 0.00/0.00 | 3.04 | 1.63 | 0.00/0.00 | 1.38 | 3.59 |
| vfh25_2 | 9.21/8.00 | 0.94 | -0.28 | 9.31/8.44 | 5.26 | 1.90 | 0.39/1.78 | 1.36 | 3.60 |
| vfh_2 | 6.80/5.75 | 0.88 | -0.23 | 6.83/5.94 | 5.16 | 1.84 | 0.01/1.05 | 1.38 | 3.58 |
| vfh4_2 | 3.85/3.24 | 0.94 | -0.30 | 3.87/3.23 | 4.44 | 1.75 | 0.00/0.52 | 1.38 | 3.59 |
| vfc_2 | 0.00/0.00 | 1.36 | -0.53 | 0.00/0.00 | 7.13 | 2.28 | 0.00/0.00 | 2.60 | 3.35 |
| vfch_2 | 0.07/0.86 | 1.36 | -0.52 | 0.02/2.04 | 7.13 | 2.32 | 0.03/1.82 | 2.61 | 3.88 |

k1 equivalent kernel, share of Σ|h| in the lowest / highest τ tercile
(one far-clear sounding, `results/weight_k1_kernel_29252a_GL.csv`,
figure `weight_k1_kernel_29252a_GL.png`): unit weights 0.27–0.38 / 0.49–0.57;
noise 0.34–0.46 / 0.19–0.54; `vf` 0.50–0.76 / 0.03–0.15; `vfc` 0.40–0.64 / 0.04–0.17.

Misfit split (`results/weight_extra_sysrand_29252a_GL.csv`): a fixed
per-channel pattern (across-sounding mean²) is 16–18 % of the far-clear
residual mean square (25–35 % in the lowest τ tercile); the rest varies
between soundings.

Huber events (`results/weight_downweight_29252a_GL.csv`,
`weight_extra_solar_29252a_GL.csv`, figures `weight_downweight_{vfh_2,vfh_g,vfch_2}_29252a_GL.png`):
in O2A and WCO2 the down-weighted channels are a fixed population (near-cloud
vs far-clear frequency correlation 0.96; 44 / 66 channels down-weighted in
> 50 % of soundings; clustered along the detector, lag-1 0.86 / 0.57).
Channels with solar-line depth > 5 % are 10 % (O2A) / 21 % (WCO2) of the
channels but carry 55 % / 85 % of the far-clear events; the most frequent
O2A channels are 770.10–770.14 nm (K I D1, vacuum) and 758.8 nm. In SCO2
Huber removes 0.01 % of channels far from clouds but 1.0 % near clouds,
mostly at the short-wavelength continuum edge (2043–2050 nm), and only 5 %
of those near-cloud events are solar lines. After fixed per-channel weights
(`vfch_2`) Huber removes 0.02–0.07 % far and 0.9–2.0 % near: what remains
for Huber is near-cloud-specific.

### Verdict on each premise

1. **"Weights must not adapt per sounding."** Holds for Huber, moot for the
   variance function. The adaptive variance function (`vfs`) is
   indistinguishable from the pooled one in stability and local SNR (all
   within the CIs), because the variance shape is nearly the same in every
   sounding. Huber is adaptive by construction and behaves differently near
   clouds (SCO2: 100× more events; `vfch`: 10–100×). Huber therefore
   violates the plan's own premise.
2. **σ² = σ²_noise + a + b·τ^p.** Adequate to a factor 1.5–2; p ≈ 2.2 (not
   2) with a ≈ 0. p, the per-fp split and the parametric vs non-parametric
   form change far-clear spread by ≤ 0.6 percentage points and local SNR by
   ≤ 0.2: none of them matters. Per band is enough (weights are scale-free,
   and p varies by ≤ 0.15 between fp).
3. **"If σ_misfit ∝ τ every channel contributes equally."** False. With c and
   k2 free, 1/τ² weights put 50–76 % of the k1 kernel in the lowest τ tercile
   and 3–15 % in the highest (unit weights: 49–57 % in the highest). The
   information τ²/σ² is per channel *before* profiling the intercept; k1 has
   to be separated from c, which only the low-τ channels do. `vf` turns k1
   into a continuum-side slope. Side effects: order 2 and gamma give the same
   k1 to 0.1–0.3 % (the truncation question becomes moot because the cores
   no longer count), k2 is poorly constrained (k2 at the zero bound in 15–20 %
   of WCO2/SCO2 footprints), and WCO2 acquires a 5 % per-fp k1 offset
   (0.6 % with unit weights), presumably because fp-dependent continuum
   effects now dominate.
4. **"Reduced χ² ≈ 1 by construction."** Fails with the robust variance
   (χ²_own 3.3 / 2.9 / 1.0), because the low-τ residual is heavy-tailed.
   It holds with the mean-square function (`vfm`: 1.1 / 1.2 / 1.1).
5. **σ(k) from the weighted covariance with AR(1) inflation.** Not
   calibrated. σ_infl(k1) = 3.7 / 8.2 / 4.4 % for `vf`, against an observed
   far-clear spread of 1.1 / 3.0 / 1.4 % (ratio 0.03–0.7 across variants) and
   a latitude-local spread of 0.1–1.1 %. The noise-only σ is 2–26× too small.
   The misfit is coherent over line populations and largely absorbed by c
   and k2, so neither a white nor an AR(1) model of it predicts the k1
   scatter. The empirical local far-clear spread per orbit × band × fp is the
   usable uncertainty.
6. **Huber removes streaking populations.** It removes an identifiable,
   fixed population in O2A and WCO2 (solar lines, above). A fixed population
   should be treated by a fixed rule, not by a per-sounding robust weight.
   The cutoff (2.5 / 3 / 4) changes the down-weighted fraction (9 / 7 / 4 %)
   but not the conclusions; in WCO2 any Huber step raises the far-clear
   spread from 3.0 % to 4.4–5.3 %.
7. **Decisive test (near-cloud signal).** `vf`/`vfh` wash out the signal in
   two of three bands. Local k1 SNR (< 3 km) `ols_2` → `vf_2`: O2A −1.47 →
   −0.28, WCO2 2.72 → 1.62, SCO2 3.47 → 3.60 (n.s.); the local far-clear
   noise rises from 0.14 → 0.30 % (O2A) and 0.43 → 1.12 % (WCO2). The joint
   Fisher distance on all six coefficients drops from 4.6 to 4.0 (CIs
   overlap). By the criterion set in advance, **`vf` and `vfh` are not an
   improvement.** The global (non-local) SNR suggests the opposite for SCO2
   (0.81 → 2.08); near-cloud and far-clear footprints sit at different
   latitudes (median 6° vs 22°) and airmass (2.50 vs 2.77), and the weighting
   changes k1's airmass sensitivity, so the global contrast is confounded
   and not used for the decision.
8. **Order 2 vs gamma × weighting.** They interact. Under `vf` the model
   choice is irrelevant (same k1, same SNR). Under unit or noise weights
   the model choice changes k1 by 5–15 % and moves the O2A near-cloud
   information between k1 and k2: `ols_2` carries it in k1 (local SNR −1.47,
   k2 0.68), noise-weighted `gamma` in k2 (k1 0.11, k2 1.41), with a similar
   O2A (k1, k2) Fisher distance (1.80 vs 1.50). Weighting schemes should be
   compared on (k1, k2) jointly, not on k1 alone.

### Recommendation

Reject the variance-function + Huber scheme as the default weighting for
anomaly features. Keep fixed, scene-independent weights that do not model
the misfit:

- **k1-only features: unit-weight order 2 (`ols_2`)**: best or tied local
  k1 SNR in every band (−1.47 / 2.72 / 3.47; 3-band k1 Fisher 3.51 [3.19,
  3.91], highest of all variants); local far-clear noise 0.14 / 0.43 / 0.74 %;
  per-fp offsets ≤ 0.6 %.
- **(k1, k2) features: noise-weighted gamma (`gamma`)** is the candidate to
  beat: highest 6-coefficient Fisher distance (5.24 [4.48, 5.85] vs 4.58
  [4.23, 5.04] for `ols_2`), O2A local noise 0.13 % (`wls_2` 0.11 %), best SCO2 k1 SNR (3.89).
  It is worse for WCO2 k1 (2.22 vs 2.72). The CIs overlap; do not switch on
  one orbit.
- No Huber step. If the solar-line population is to be treated, do it by a
  fixed rule. The fixed mask (`sm_2`) does not help on this orbit: it lowers
  the far-clear spread only in SCO2 (1.39 → 1.15 %) and halves the WCO2 local
  SNR (2.72 → 1.25), so the solar-line channels carry part of the WCO2
  near-cloud contrast. This must be understood before any treatment.

### Additional tests (and why)

- `vfm`, `vfn`: separate "functional form" from "robust vs mean-square";
  needed because χ² ≠ 1 under the plan.
- `vfc`, `vfch`: a fixed per-channel weight is the scene-independent
  equivalent of Huber, and `vfch` shows what Huber does beyond it.
- Latitude-local contrast: the global near/far contrast is confounded by
  latitude/airmass (see 7).
- k2 contrast and joint Fisher distance: to tell lost information from
  information moved between k1 and k2 (see 8).
- k1 equivalent kernel: direct test of the "equal contribution" rationale.
- Systematic/random split and σ calibration: test of step 4.
- Solar-line depth per channel (TOA solar spectrum, upper-envelope depth)
  and the fixed solar mask `sm_*`: identification of the Huber population.
- Paired bootstrap CIs (400 draws; 100 for Fisher) on every SNR.

### Open questions / next

1. Repeat on a land nadir orbit and a second ocean glint orbit (same
   commands); the `ols_2` vs `gamma` choice for (k1, k2) features waits on it.
2. Only 1 053 of 5 863 near-cloud footprints have a local far-clear reference
   (±1°, ≥ 15 references); the local metric should be rebuilt on the
   project's own nearby-footprint anomaly definition.
3. Why do solar-line channels carry WCO2 near-cloud contrast? Test whether
   their residual tracks continuum radiance (zero-level offset,
   non-linearity) or the solar Doppler shift along the orbit; if
   brightness-driven, part of the WCO2 "cloud signal" is instrumental.
4. The SCO2 short-edge continuum (2043–2050 nm) has a near-cloud-specific
   negative residual (10th percentile −0.095 vs −0.021 far-clear,
   `weight_extra_edge_29252a_GL.csv`): a continuum spectral slope that the
   model has no term for. A linear-in-wavelength continuum term may be worth
   testing before any reweighting.
5. Per-species τ (plan step 5) remains the way to address the H2O part of
   the misfit; this orbit cannot separate H2O from solar-line channels.

### Files

Code: `weighting.py`, `run_weighting.py`, `weight_extra.py`. Results:
`results/weight_per_sounding_29252a_GL.parquet` (long format, FIELDS + extra
columns), `weight_summary_29252a_GL.{csv,md}`, `weight_signal_29252a_GL.csv`,
`weight_varfunc_29252a_GL.csv`, `weight_downweight_29252a_GL.csv`,
`weight_downweight_freq_29252a_GL.npz`, `weight_k1_kernel_29252a_GL.csv`,
`weight_extra_{sysrand,edge,sigma_calibration,local_signal,solar}_29252a_GL.csv`;
figures `weight_varfunc_`, `weight_k1_kernel_`, `weight_snr_vs_spread_`,
`weight_downweight_{vfh_2,vfh_g,vfch_2}_` (all `_29252a_GL.png`).
Run (≈ 3 min with 8 workers, then ≈ 3 min for the extra diagnostics):

```
python idea_test/fit_order_study/run_weighting.py --l1b <L1B> --fp-tau <fp_tau> --fit-details <fit_details> --workers 8
python idea_test/fit_order_study/weight_extra.py  --l1b <L1B> --fp-tau <fp_tau> --fit-details <fit_details>
```

## Slab closure test (Opus agent) — 2026-10-29

Question: which fitter's (k1, k2) is the PPDF? Closure against the Appendix-G
x-z slab (`results/rt_slab_sim`, read-only), where the photon path tally is
known per column. Code: `slab_truth.py` (inputs, reference PPDFs),
`slab_fitters.py` (wrappers over `variants.py`), `slab_closure.py` (T0–T6,
tables, figures). Run: `python idea_test/fit_order_study/slab_closure.py`
(≈ 4 min with 8 workers; `--reuse-t3` re-summarizes the stored T3 table).

### T0 — pipeline reproduction

`ols_7` reproduces `slab_fit.h5` to |Δk1| ≤ 2e-7, |Δk2| ≤ 9e-7 in all four
cases (T = π·rad/(μ0·toa), all 33 channels, BVLS k1,k2 ≥ 0), and the
closure_stats.json correlations to the 4th decimal: r(k1, ⟨L⟩) 0.9989 / 0.9890,
r(k2, var L) −0.499 / +0.610 (dark / bright), clear-column −0.163 / +0.989
(`results/slab_closure_repro.csv`). l' = 1 + (L − L_peak)/l_direct with
l_direct = 164.6 km and L_peak = 914.1 km (far-field direct-bounce peak, as
`make_fig_g1.py`); the existing closure used absolute km, which is irrelevant
for r.

### Which reference PPDF? (decided before T1)

The MCARaTS tally is **geometric** path. The spectrum senses
**absorption-weighted** path: per km, the lowest kilometre holds 19 % of the
O2 column against 1/60 for a uniform absorber. So geometric l' is not the
quantity of the equivalence theorem, at any wavelength. Measured (3-D medians):

| case, region | ⟨l'⟩ geo784 | ⟨l'⟩ geo_rw | var l' geo784 | ⟨A⟩ | var A (between-bin) | K1(0.3) | K2(0.3) | K2(1.0) | tally mass lost |
|---|---|---|---|---|---|---|---|---|---|
| dark clear | 0.978 | 0.959 | 0.010 | 0.862 | 0.281 (0.204) | 0.800 | 0.175 | 0.180 | 0.3 % |
| dark cloud | 0.936 | 0.933 | 0.003 | 0.399 | 0.076 (0.061) | 0.384 | 0.041 | 0.019 | 0.1 % |
| dark shadow | 0.934 | 0.902 | 0.028 | 0.644 | 0.711 (0.487) | 0.503 | 0.282 | 0.165 | 0.9 % |
| bright clear | 1.004 | 1.000 | 0.004 | 1.042 | 0.274 (0.193) | 1.001 | 0.081 | 0.057 | 0.1 % |
| bright cloud | 0.964 | 0.955 | 0.007 | 0.651 | 0.403 (0.329) | 0.571 | 0.194 | 0.098 | 0.1 % |
| bright shadow | 1.027 | 1.000 | 0.020 | 1.261 | 1.353 (0.960) | 1.016 | 0.447 | 0.251 | 0.6 % |

- `geo784`: the three τ ≈ 1e-4 anchor channels (784 nm; the closure_stats
  reference). `geo_rw`: radiance-weighted average of all 33 channel
  histograms. They differ by ≤ 3.5 % in mean and 10–40 % in variance. Both
  are immaterial next to the geometric/absorption difference.
- **`abs` (adopted reference)**: for each geometric-path bin b, the 30
  absorbing channels' radiance-weighted histograms give ln[T_λ h_λ(b)], the
  log Laplace transform of A within the bin. Fitted per bin as
  α_b − κ_b ln(1 + τ(a_b + β_b η_λ)/κ_b), η_λ = leading SVD mode of the
  channel's normalized layer-OD profile (96 % of the profile variance; zero
  for the band-mean profile). p(A) is the mixture of these gammas at η = 0.
  This is photon-resolved information from the tally, not a fit to the column
  spectrum. Self-consistency: the bins rebuild the MC ln T to 0.15–1.0 %
  rms. Sensitivity: K1, K2 at τ = 0.3, 1 change < 0.5 % under finer bins or
  under η = above-5-km fraction; var A changes ≈ 4 %; MC-run σ of ⟨A⟩ ≤ 0.001.
- Cloud-top photons (l'_geo ≈ 0.92) have A ≈ 0.30 and β ≈ 1.0 (A equals the
  channel's above-cloud OD fraction). Long low-level paths (l'_geo = 1.5) have
  A ≈ 4.7. Absorption weighting stretches p(l') by a factor 5–10.
- **var A is not identifiable.** It is tail-dominated (skew 6–15, excess
  kurtosis 50–600), 25–30 % of it is within-bin, and 0.1–0.9 % of the
  radiance lies beyond the tally range (l'_geo > 2.37, i.e. A ≳ 8). The same
  truncation biases the geometric var l' low by 30–60 %. The identifiable
  targets are the **local cumulants** K1(τ) = −d ln T/dτ, K2(τ) = d² ln T/dτ²
  of the mean-profile spectrum (moments of p(A) tilted by e^{−τA}), with
  K(0) = (⟨A⟩, var A). K2 drops 2–5× between τ = 0 and τ = 1.

### T1 — closure on the MC spectra (33 channels, 3-D, 64 columns)

k1 vs ⟨A⟩ (all columns): r [95 % column-bootstrap CI], slope, median relative
bias (± MC-run σ of the median bias), rms. k2 vs var A and vs K2(τ = 1).
`results/slab_closure_stats.csv` holds all targets, subsets (all / clear =
outside cloud / far = x < 5 or > 25 km) and κ.

| fitter | case | k1 vs ⟨A⟩ | k2 vs var A: r, bias | k2 vs K2(1.0): r, slope, bias | k2 ≤ 0 |
|---|---|---|---|---|---|
| ols_7 | dark | 0.916 [0.90, 0.94], 0.87, −7.2 % (±0.05 %), 0.093 | −0.52, −94 % | −0.95, −2.83, −91 % | 19 % |
| ols_3 | dark | 0.965 [0.96, 0.98], 1.21, **+1.6 %**, 0.072 | −0.00, −22 % | **0.95**, 1.34, +23 % | 0 |
| ols_2 | dark | 0.930 [0.91, 0.95], 0.93, −14.5 %, 0.147 | −0.06, −62 % | 0.92, 0.67, −40 % | 16 % |
| ols_2u | dark | 0.936 [0.92, 0.95], 0.94, −14.5 %, 0.147 | −0.05, −62 % | 0.93, 0.69, −40 % | 16 % (k2 < 0) |
| gamma_u | dark | 0.977 [0.97, 0.99], 1.87, +34.8 %, 0.246 | −0.15, +253 % | 0.89, 6.38, +462 % | 0 |
| gamma_w | dark | 0.957 [0.94, 0.97], 1.53, +13.4 %, 0.146 | −0.07, +95 % | 0.85, 4.14, +211 % | 0 |
| ols_7 | bright | 0.985 [0.98, 0.99], 0.70, −2.7 %, 0.071 | 0.60, −91 % | 0.65, 1.81, −57 % | 5 % |
| ols_3 | bright | 0.864 [0.75, 0.92], 0.84, **+2.0 %**, 0.103 | 0.97, −68 % | **0.97**, 0.97, +60 % | 0 |
| ols_2 | bright | 0.533 [0.20, 0.75], 0.65, +6.8 %, 0.212 | 0.06, −54 % | 0.00, 0.00, +131 % | 0 |
| ols_2u | bright | 0.533 [0.16, 0.74], 0.65, +6.8 %, 0.212 | 0.06, −54 % | 0.00, 0.00, +131 % | 0 |
| gamma_u | bright | 0.954 [0.93, 0.97], 1.70, +28.5 %, 0.279 | 0.91, +75 % | 0.89, 6.17, +773 % | 0 |
| gamma_w | bright | 0.972 [0.95, 0.98], 1.29, +4.9 %, 0.088 | 0.92, −32 % | 0.91, 4.69, +244 % | 0 |

ICA columns, k1 median bias: ols_3 +2.3 / +4.5 %, ols_7 −7.3 / −1.6 %, ols_2
−14.0 / +10.0 %, gamma_u +35.7 / +32.0 %, gamma_w +13.8 / +3.8 % (dark / bright).
Against the geometric reference every fitter has k1 slope 7–17 vs ⟨l'⟩_geo
(k1 spans 0.4–1.2 while ⟨l'⟩_geo spans 0.93–1.03): the r = 0.999 of
closure_stats is a correlation with an incommensurate quantity. MC noise is
negligible (run σ of the median bias ≤ 5e-4).

**Verdict T1.** No fitter recovers var A (expected; see above). ols_3 is the
only fitter with |k1 bias| ≤ 2 % in both surfaces and both solvers, and its k2
tracks K2(τ ≈ 1) (r 0.95 / 0.97). The negative dark-case r(k2, var l') is not
PPDF physics: it appears with the correct reference too (r = −0.52), and T2/T6
show it is ols_7 amplifying two non-PPDF features of the spectrum.

### T2 — theorem error vs fitter error

Theorem residual (ln T_MC − ln T_eq), median rms by region
(`results/slab_closure_theorem_resid.csv`, figure `slab_theorem_resid.png`):

| reference | dark clear / cloud / shadow | bright clear / cloud / shadow | corr(residual/τ, η) in cloud |
|---|---|---|---|
| geo784 p(l') | 1.26 / 1.35 / 1.51 (max 4.9) | 0.74 / 1.25 / 1.13 | – |
| abs p(A), mean profile | 0.048 / 0.093 / 0.053 (max 0.18–0.40) | 0.039 / 0.091 / 0.044 | −0.98 / −0.99 |

- Geometric p(l'): the theorem fails by e^{1–5} at τ = 8. It is the wrong variable.
- Absorption-weighted p(A): what remains is the **channel vertical-shape
  effect** (each channel has its own profile; deep line cores weight
  absorption aloft, η > 0). It is systematic in τ: −0.05 to −0.4 at τ = 3–8
  in every column, i.e. a **concave-down** tail. This is the slab's version
  of the real-data "spread at fixed τ" and of the concave-down curvature that
  pins ols_3 k2 at zero on real orbits (hypothesis for the real data).
- The 784-nm anchors carry a Rayleigh step relative to the band
  ((784/760)⁴ = 1.13): −0.023 (dark clear) to −0.060 (dark shadow) in ln T.

Fitters on the exact ln T_eq,A (33 slab τ, no noise): pure truncation/shape error.

| fitter | k1 vs ⟨A⟩ dark / bright: r, bias | k2 vs var A dark / bright: r, bias | τ* of k1 / of k2 (K(τ*) = fitted) |
|---|---|---|---|
| ols_7 | 0.999, −1.1 % / 0.995, −1.2 % | 1.00, −39 % / 0.99, −60 % | 0.03–0.05 / 0.13–0.35 |
| ols_3 | 0.975, +3.2 % / 0.872, +2.3 % | 0.06, −17 % / 0.98, −66 % | 0.2–0.35 (cloud) / 0.05–1.25 |
| ols_2 | 0.936, −14.5 % / 0.527, +7.4 % | −0.08, −60 % / 0.04, −51 % | 0.6–0.8 / 2.3–3.6 |
| gamma_u | 0.993, +46 % / 0.978, +32 % | 0.03, +377 % / 0.93, +102 % | > K1(0) in clear/shadow / 1.5–1.8 |
| gamma_w | 0.978, +19 % / 0.977, +6 % | 0.27, +140 % / 0.92, −23 % | – |

(τ* in `results/slab_closure_tau_star.csv`; a fitter with k1 > K1(0) has no τ*.)

**Verdict T2.** Every fitter's k1, k2 equals a local cumulant at its own τ*;
the k2 of different fitters are different quantities. With 33 noise-free
channels, ols_7 is a Taylor estimator (k1 = ⟨A⟩ to 1 %; k2 = K2(0.1–0.35),
r = 1.00 with var A). Its MC failure is therefore fitter amplification of the
non-PPDF features, not truncation. The gamma closed form is biased high in k1
(+6 to +46 %) even on exact spectra: p(A) is not gamma-shaped (T4).

### T3 — dense synthetic spectra on real τ grids

Per 3-D column, ln T_eq of the true PPDF on the real 2020-01-01 29252a_GL
sounding-5000 grids (O2A n = 1016, τ 0.06–5.4; SCO2 n = 1016, τ 0.06–4.2;
wavelength order kept), white noise σ = 0.005·T^{−1/2} (continuum 0.005,
photon scaling), plus a misfit ∝ τ with AR(1) ρ = 0.5 and rms 0.05 / 0.1 / 0.2;
12 realizations. Medians over dark + bright (`results/slab_closure_t3.csv`,
figures `slab_t3_abs.png`, `slab_t3_geo784.png`). abs p(A), O2A grid:

| fitter | k1 bias noiseless / 0.1 / 0.2 | k1 scatter white / 0.1 / 0.2 | k1 r, slope (0.1) | k2 vs K2(1.0): r, bias (0.1) | k2 scatter 0.1 / 0.2 | k2 ≤ 0 (0.1 / 0.2) |
|---|---|---|---|---|---|---|
| ols_7 | −0.9 / +3.7 / +10.1 % | 1.2 / 9.1 / 15.9 % | 0.986, 0.97 | 0.53, +333 % | 182 / 312 % | 38 / 43 % |
| ols_3 | −1.2 / −1.2 / −0.5 % | 0.2 / 3.2 / 6.2 % | 0.974, 0.99 | **0.98, −8 %** | 13 / 25 % | 15 / 20 % |
| ols_2 | −0.1 / −0.4 / −0.6 % | 0.1 / 2.3 / 4.3 % | 0.861, 0.95 | 0.87, +42 % | 5 / 8 % | 1 / 2 % |
| gamma_u | +15.6 / +15.2 / +14.6 % | 0.2 / 4.5 / 8.5 % | 0.996, 1.45 | 0.97, +279 % | 25 / 53 % | 0 |
| gamma_w | +8.0 / +7.7 / +7.6 % | 0.2 / 2.5 / 4.6 % | 0.998, 1.26 | 0.96, +169 % | 14 / 28 % | 0 |

SCO2 grid: same ranking (ols_3 k1 −1.9 %, k2 vs K2(1.0) r 0.94, bias 0 %;
ols_2 −0.9 %, r 0.96, +24 %; gamma_u +9.4 %, gamma_w +5.4 %; ols_7 k1 +10.9 %,
scatter 18 %). With the narrow geometric p(l') (κ 30–300, near-Gaussian)
every fitter except ols_7 recovers k1 to ≤ 2 %, but k2 scatter is 100–800 % at
misfit 0.1: a var of 0.004–0.03 is below the misfit floor for every fitter.

**Verdict T3.** ols_7 does not survive realistic misfit (k1 scatter 9–33 %,
k2 at the bound in ~40 %). ols_3's k2 is the tilted variance K2(τ ≈ 1) to
−8 % / 0 % with r 0.94–0.98, at 2–3× the k2 scatter of ols_2 and with 15–23 %
of k2 at the bound. ols_2 gives the most stable k1 and k2, but its k2 is a
blend (K2(1) × 1.2–1.4, r 0.87–0.96) and its k1 accuracy depends on the grid
(≈ 0 % here, −15 % / +7 % on the slab grid). The gamma fits have the highest
k1 correlation (0.996–0.998) and a slope of 1.2–1.45: a scene-dependent bias
that a constant calibration cannot remove.

### T4 — is p(A) a gamma?

No (`results/slab_closure_shape.csv`, figure `slab_ppdf_gamma.png`). p(A) is a
mixture: a spike at A = 1 (direct surface bounce), a low-A mode (Rayleigh
scatter aloft in clear columns; cloud-top reflection at A ≈ 0.3 in cloud) and
a long multiple-scattering tail to A > 3. Median skew 6–15 against the gamma
prediction 2/√κ = 1.0–2.7; excess kurtosis 50–600 against 6/κ = 1.5–11. The
gamma best fit misses the true ln T_eq,A by up to 0.17–0.34 (clear), 0.00–0.04
(cloud) and 0.03–0.22 (shadow) in ln T. The gamma shape is worst in clear columns, where the
direct spike and the Rayleigh mode are both present (bimodal), and at the
illuminated cloud edge. It is adequate only where one population dominates
(dark in-cloud).

### Extra tests and why

- **T5, vertical-shape regressor** (`ols_2_eta`, `gamma_u_eta`: −β·τ·η added).
  Needed to test whether the T2 shape residual can be absorbed by one term.
  It cannot: k1 changes < 2 %, k2 improves only in cloud (dark k2/K2(1):
  0 → 0.5). The shape sensitivity is population-dependent (β ≈ 1 for cloud-top
  photons, ≈ 0 for surface photons, < 0 for long low-level paths), so a
  column-level linear term is the wrong form.
- **T6, 784-nm anchors dropped** (30 absorbing channels only). Needed because
  T2 shows a Rayleigh step at the anchors. It is the main driver of the ols_7
  dark failure: k1 r 0.92 → 0.996, bias −7.2 → +1.1 %; k2 vs var A
  r −0.52 → +0.55; k2 pinning 19 → 0 %. Other fitters change by < 1 %.
  The real O2A continuum has the same kind of spectral slope (Rayleigh
  (772/757)⁴ = 1.08, plus surface albedo slope) concentrated in the few
  lowest-τ channels where order 7 has the most leverage.
- **Local cumulants and τ*.** Needed because var A is not identifiable. They
  give each fitter's k2 a definite meaning (figure `slab_local_cumulants.png`).
- **Absorption-weighted reference** (`slab_truth.reconstruct_abs`) and its
  sensitivity checks. Needed because the tallied geometric PPDF fails the
  theorem by e^{1–5}.

### Recommendation

- **Adopt unit-weight order 3 (`ols_3`, k1, k2 ≥ 0 as production) for
  (k1, k2) as PPDF descriptors**, with k2 defined as the tilted variance
  K2(τ ≈ 1), not var l'. Measured closure: k1 vs ⟨A⟩ median bias +1.6 / +2.0 %
  (3-D dark / bright), +2.3 / +4.5 % (ICA), r 0.965 / 0.864, rms 0.07 / 0.10;
  on realistic grids and misfit −0.5 to −1.9 %, slope 0.96–0.99, r 0.97,
  scatter 3–8 %; k2 vs K2(1.0) r 0.95 / 0.97 (MC) and 0.94–0.98 (T3).
- **What it fails at:** var A itself (−20 to −70 %; no fitter succeeds);
  k2 scatter 13–37 % and 15–23 % of k2 at the bound under misfit 0.1–0.2;
  bright-surface k1 r only 0.86 (cloud/shadow columns biased −11 to −13 %).
  On real orbits order 3 pins k2 at zero in 63 % (O2A) / 97 % (SCO2) of
  footprints, which no T3 setting reproduces. The T2 concave-down
  vertical-shape tail is the candidate cause. Until that is resolved, ols_3's
  k2 is not usable on real data; k1 from ols_3 is (real-data spread 0.6–0.7 %).
- **Fallback for a non-pinned k2: `ols_2`.** k1 unbiased on the real grid
  (−0.4 to −0.9 %), but −15 % / +7 % on the slab grid and r = 0.53 on bright,
  i.e. its k1 is a grid-dependent K1(τ ≈ 0.7). k2 ≈ 1.2–1.4·K2(1), r 0.87–0.96.
  The bound is irrelevant (`ols_2u` identical).
- **Reject the gamma closed form as a PPDF estimator**, unit- or
  noise-weighted. p(A) is not gamma-shaped, and the forced mean–variance link
  biases k1 by +5 to +46 % with a scene-dependent slope of 1.2–1.9, and k2 by
  2–9× K2(1). Its high correlation (0.99) is precision, not accuracy. This
  supersedes the addendum's "gamma is the principled replacement": the
  ols_2/ols_3 "bias" quoted there was relative to gamma_u, which is itself
  +29 to +35 % high here.
- **Keep ols_7 out.** It is the best Taylor estimator on noise-free exact
  spectra, and nothing else.
- κ = k1²/k2 inherits all k2 problems and should not be used as a feature.

### Limitations

- One scene (SZA 55°, nadir, one 5-km cloud, COD 10), O2A optics only. The
  SCO2 T3 run uses O2A-band PPDFs on the SCO2 grid.
- p(A) is a reconstruction (gamma within geometric bins, one shape mode), not a
  tally. An absorption-weighted (per-layer path) accumulator in MCARaTS and a
  wider plen range (> 2.4·l_direct) would remove both caveats.
- The slab τ grid (30 log-spaced channels to τ = 8 plus three 784-nm anchors)
  is not the real grid. Absolute k1 biases of ols_2 depend on the grid.
- The T3 misfit is random AR(1) ∝ τ. The real misfit has fixed channel
  populations and a concave component; T3 does not reproduce the real
  k2 pinning, so the real misfit is not of this kind.

### Next

1. Test the concave-tail hypothesis on real data: correlate the order-2
   residual at τ > 2 with each channel's absorber-weighted altitude (needs
   per-layer OD per channel from the ABSCO step).
2. Repeat T0–T3 for a second slab scene (bright surface + lower SZA) and an
   SCO2 run, if the CURC budget allows.
3. Re-examine real-data k1 spread for ols_3 vs ols_2 with the anchor issue in
   mind: compare fits with and without the lowest-τ continuum channels.

### Files

Code: `slab_truth.py`, `slab_fitters.py`, `slab_closure.py`. Results:
`results/slab_closure_{repro,truth,stats,fits,theorem_resid,tau_star,shape,t3}.csv`,
`results/slab_closure_t3_raw.csv.gz`, `results/slab_closure_summary.md`
(auto-generated tables); figures `results/slab_closure_scatter_mc.png`,
`slab_closure_scatter_teq.png`, `slab_theorem_resid.png`,
`slab_local_cumulants.png`, `slab_ppdf_gamma.png`, `slab_t3_abs.png`,
`slab_t3_geo784.png`.

## τ-cut sweep — 2026-10-29 (does the full τ range break the truncated expansion?)

Scripts `cut_sweep_real.py` (orbit 29252a_GL, orders 2–3, τ_cut ∈ {0.5, 1, 2, 3, 5, 8, none})
and `cut_sweep_slab.py` (slab MC spectra vs absorption-weighted truth, orders 2/3/7, τ_cut ∈
{0.3 … 5, none}); outputs `results/cut_*`.

1. **Full-range residual vs τ (real, far-clear).** Order 2 leaves a systematic wave in O2A
   (median −0.009 at τ < 0.1, +0.03 at τ 0.5–2, −0.02 at 2–5, +0.09 at 5–8; SCO2 same shape,
   smaller). Order 3 flattens it to |median| ≤ 0.01 up to τ 8, IQR straddling zero above τ 3.
   The high-τ channels are therefore within what a cubic represents; there is no breakdown of
   the truncated model in the mean. WCO2 has τ ≤ 1, so cuts there are no-ops.
2. **k1 scatter grows monotonically as τ is cut** (far-clear robust SD, %):

   | band, order | none | ≤5 | ≤3 | ≤2 | ≤1 | ≤0.5 |
   |---|---|---|---|---|---|---|
   | O2A, 3 | 0.68 | 0.78 | 1.41 | 1.82 | 2.83 | 4.08 |
   | O2A, 2 | 0.84 | 0.98 | 1.37 | 0.92 | 1.50 | 2.73 |
   | SCO2, 3 | 0.57 | 0.57 | 1.40 | 2.40 | 5.84 | 6.70 |
   | SCO2, 2 | 1.37 | 1.37 | 1.46 | 1.64 | 1.31 | 3.10 |

   Near-cloud k1 correlation with the XCO2 anomaly does not improve under any cut. The order-2
   full-range O2A k1 is +10 % above order 3; a cut at τ ≤ 1–2 removes that bias (−0.5 %) at
   the cost of 10–80 % more scatter — a legitimate robustness check for order-2 k1, nothing more.
3. **k2 cannot be pushed toward var A by range selection.** Slab truth: K2(τ)/var A is
   0.55–0.75 at τ = 0.1 and 0.30–0.62 at τ = 0.3 (bright/dark). The variance lives at τ < 0.1,
   where the curvature signal ½K2τ² ≲ 1e-3 in ln T, below the instrument noise (~5e-3) and
   30–100× below the misfit (0.05–0.2). On the real orbit, cuts ≤ 2 give O2A k2 ≈ 0.03 with
   20–43 % at the zero bound and 45–93 % spread. In the noise-free slab a cut at τ ≤ 1–2 does
   raise r(k2, var A) to 0.95–0.98 (bright) but recovers only 25–60 % of it; τ* stays 0.3–0.4.
4. **Slab k1.** Bright: orders 2/3 with τ ≤ 1–3 are within 1.5–2.7 % of ⟨A⟩ (r 0.95–0.98 over
   all columns) against +7 % (order 2) / +2 % (order 3) full range. Dark: cuts ≤ 1 bias k1 by
   −6 to −20 % because the 784-nm Rayleigh anchors dominate the short range (same anchor effect
   as the closure test); full-range order 3 is +1.6 %. The slab grid has 33 channels (15 at
   τ ≤ 0.3), so its cut penalties are not quantitatively transferable to the 1000-channel grid.

**Verdict.** Keep the full τ range; use order 3 for k1. The "expansion validity" worry is in
practice an identification question (least-squares coefficients equal local cumulants at a
fitter-specific τ*), which is addressed by fixing the order and reporting τ*, not by truncating
the τ range. A cut is not a route to var A; the k2 question remains the concave-tail diagnostic.

## Residual decomposition (Opus agent) — 2026-10-29

Question: which channel attribute carries the structured far-clear misfit of the
unit-weight order-3 (and order-2) cumulant fit on orbit 29252a_GL, and is it path
physics (vertical weighting, H2O) or spectroscopy / solar / instrument? Code:
`resid_attrs.py` (attributes + validation), `resid_decompose.py` (analysis),
`resid_fitters.py` (conditional fitters). Outputs `results/resid_*`.

### Pre-registered definitions and bars (written before running)

Data: far-clear = cld_dist > 20 km and QF0 (n ≈ 1240); edge channels [1:-1] dropped;
orders 3 and 2, unit weights, as `variants.py`.

- **Structured residual r**: residual of the *unconstrained* unit-weight order-N
  polynomial fit, i.e. exactly orthogonal per sounding to [1, τ, …, τ^N]. The
  bounded (k1, k2 ≥ 0) residual is reported alongside; the difference is the part
  the bound refuses to absorb.
- **Attributes** (per band, channel, sounding), each regressor column residualized
  per sounding against the same polynomial (Frisch–Waugh–Lovell), so an attribute
  only gets credit for what the τ polynomial cannot represent. Groups:
  - VERT: absorber-weighted altitude η (exact integrated-gradient split of the
    stored transmission-space convolution into layers; see method); forms η⊥ and
    (τ·η)⊥, plus a cubic B-spline in η.
  - H2O: f_H2O = τ_H2O/τ from the same split; forms f⊥, (τ·f)⊥, spline.
  - SOLAR: solar-line depth 1 − toa/(running max ±8 channels), per sounding.
  - CONT: continuum flag (τ < band 5th percentile), Rayleigh fraction of τ.
  - INSTR: smooth channel index (B-spline, 10 interior knots), distance to band
    edge, dispersion-shift signature ∂τ/∂(pixel shift), ILS-width signature
    ∂τ/∂ln(ILS width). The last two are added because the example residuals
    split into ± symmetric branches that grow with τ, the signature of a
    wavelength-registration error; they are computed in the same convolution.
  - TAU: cubic B-spline in ln τ (smooth τ shape beyond the polynomial = truncation).
  - FP: share of the fixed channel pattern that differs between footprints
    (per-fp minus pooled per-channel mean), reported separately.
- **Metrics** (per band × order, pooled over far-clear soundings):
  R²_alone(g) and **partial R²(g) = (RSS_without g − RSS_all)/RSS_without g**, with
  pooled coefficients (a fixed relationship), and the same with one coefficient per
  sounding per linear regressor (per-sounding amplitude). Fixed-pattern share =
  Σ_c n_c·mean_c² / Σ r² (per fp). τ-dependence: pooled η⊥ slope in τ bins.
- **Dominant**: partial R² > 50 % (pooled or per-sounding model); **partial**:
  20–50 %; **none dominant**: R²_alone < 20 % for every group in both models.
  Per-sounding models report the chance level p/n.
- **Attribute validation (must pass before VERT/H2O are used)**: my reproduction of
  `{band}_tau_output` from ABSCO + met profiles: median |Δτ|/τ ≤ 1e-3 per band
  and 99th percentile ≤ 1e-2; Σ_layers contribution + Rayleigh = τ_c to ≤ 1e-5
  relative. If the reproduction fails, VERT/H2O are declared untrusted and the
  proxy 1b (τ rank within the local line) is used and labelled as such.
- **Decision → action**:
  - VERT dominant → fit ols_3 + (−β τ η) per sounding (β free sign). PASS if
    order-3 k2 zero fraction < 20 % (from 63 % O2A / 97 % SCO2) *and* far-clear
    k1 spread not worse than ols_3 by > 10 % relative; also report SCO2/O2A
    ratio spread, k2 spread, resid rms, lag-1, near-cloud corr(k1, xco2_bc_anomaly)
    and latitude-local near-cloud SNR (`weight_extra` definition).
  - H2O dominant → two-τ fit (−k1_dry τ_dry − k1_w τ_H2O + ½k2 τ² − ⅓k3 τ³) on
    WCO2/SCO2, same bars.
  - SOLAR / CONT / INSTR dominant → name which with evidence (sign stability across
    soundings, correlation of per-sounding coefficients with Doppler velocity,
    continuum radiance, footprint). As an extra, test the corresponding regressor in
    the fit with the same metrics (labelled as not in the decision table).
  - TAU dominant → truncation / model shape, not a channel attribute.

## Replication on 2018-10-18 (Opus agent) — 2026-10-29

Goal: test whether the 29252a_GL truncation-order result (order 3 ≪ order 7 in
far-clear k1 scatter, without losing near-cloud signal) holds on other orbits.
Scripts `rep_orbit_io.py`, `rep_proxy.py`, `rep_run.py`; outputs `results/rep_*`.

### Data situation (checked before writing the bars)

- 22845a_GL: fp_tau (10 811 soundings) = exactly the Lite soundings of orbit 22845;
  L1B local. Only 0.2 % of these soundings have Lite land_fraction ≥ 50 % (the
  ~15 % land quoted earlier is over all L1B soundings; land glint soundings
  rarely reach the Lite file). A land/ocean split on 22845a is therefore
  ocean-only in practice.
- 22846a_GL: fp_tau (15 485) = Lite soundings of orbit 22846, 6.4 % land.
  Its L1B was not local; it was fetched from GES DISC (CMR + ~/.netrc, the
  repo's ingestion route) into the session scratchpad, not into `data/`.
  22846a is run as a second replication orbit and carries the land split.
- The local `L2CPr` files are the **CO2 prior** product (CO2Prior, Meteorology,
  SoundingGeometry groups), not a cloud preprocessor; there is no ABO2/IDP cloud
  flag anywhere locally. The only cloud information is which L1B soundings
  reached the Lite file (they passed the ABP/IDP prescreen; all L1B soundings on
  29252a missing from the set have sounding_qual_flag 0 in 99 % of cases).
- No MODIS cloud-distance file for 2018-10-18; no fitting-details file.

### Proxy definitions (fixed before the fits)

- Cloud-proximity proxy `cld_dist_km` = great-circle distance from each sounding
  to the nearest L1B sounding (any footprint, valid geolocation, whole orbit)
  that is **not in the Lite file** (proxy A). Variant B also counts Lite
  soundings with QF1 as cloudy. B is used only if its far-clear precision on
  29252a (definition below) exceeds A's at 20 km; otherwise A.
- `xco2_bc_anomaly` proxy = production `src/spectral/anomaly.compute_xco2_anomaly`
  (imported, unchanged; 0.25° lat window, ref std ≤ 1 ppm, refs cld_dist > 10 km,
  ≥ 5 refs) applied to Lite `xco2` (= production `xco2_corr`) with the proxy
  cloud distance. Difference from production: the cloud distance, and the
  reference pool is not narrowed by the validity of the production k-fit/albedo
  extra variables (all finite in practice).
- `xco2_qf` = Lite `xco2_quality_flag`; land = Lite `Sounding/land_fraction` ≥ 50 %.

Disclosure: one exploratory probe of proxy A against the true MODIS distance on
29252a was run before these bars were written: Spearman 0.59; of proxy > 20 km
soundings (all QF) 67 % were true > 20 km (76 % at 30 km); of proxy < 3 km, 79 %
were true < 3 km. The fallback rule below was written knowing this.

### Pass/fail bars (written before running the fits)

P0 — proxy validity, on 29252a_GL (true cld_dist available):
- P0a far: precision = fraction of proxy far-clear soundings (proxy > T, QF0) that
  are true far-clear (true > 20 km). Bar ≥ 80 % at T = 20 km. If it fails, use the
  smallest T in {30, 40, 50} km with precision ≥ 80 % and ≥ 300 soundings on
  29252a, and apply the same T on 2018-10-18. If none, report far-clear statistics
  as "proxy-far" with the measured contamination.
- P0b near: precision of proxy < 3 km ≥ 70 %, **and** the local near-cloud SNR
  of ols_3 and ols_7 recomputed on 29252a with proxy labels has the same sign
  and is within a factor 1.5 of the true-label value for O2A and SCO2.
  Otherwise the near-cloud metrics on 2018-10-18 are declared not reproducible.
- P0c metric transfer: on 29252a the far-clear SCO2/O2A ratio spread of ols_2,
  ols_3, ols_7 under proxy labels is within ±25 % (relative) of the true-label
  value.

R — replication, on 22845a_GL far-clear (proxy labels; same bars applied to
22846a as a second orbit):
- R1: ols_3 k1 spread ≤ 1.0 % for O2A and for SCO2 (the order-7 bands under
  decision; WCO2 is order 3 in production and had 5.6 % on 29252a, so it is
  reported without a bar), and SCO2/O2A ratio spread ≤ 1.5 %.
- R2: ols_7 SCO2/O2A ratio spread ≥ 4 × ols_3.
- R3: |local near-cloud SNR| of ols_3 ≥ that of ols_7, for O2A and SCO2 k1
  (only if P0b passes). Paired bootstrap CI of the difference is reported.
- R4 (report, no bar): order-3 k2 zero-bound fraction, compared with 63 % (O2A) /
  97 % (SCO2); "reproduces" means within ±15 percentage points.
- R5: does the SCO2/O2A ratio-spread ordering ols_3 < ols_2 < gamma < ols_7
  reproduce (gamma = noise-weighted, ~4000-sounding subsample)?
- R6: land vs ocean agree = R1 and R2 give the same verdict in each subset with
  ≥ 100 far-clear soundings; a subset with < 100 is "insufficient".
- *Addition before any residual analysis (after the 16-sounding attribute test only)*:
  TEMP group = ∂τ/∂T (±2 K shift of the ABSCO-lookup temperature; n·dz is
  T-invariant at fixed pressure). It counts as spectroscopy/met (not path physics).

### Results — Replication on 2018-10-18 (Opus agent), written after the runs

(The italic "Addition before any residual analysis … TEMP group" bullet just
above was appended by the residual-decomposition agent while this section was
open; it belongs to "Residual decomposition", not to this replication.)

Runs: `rep_run.py` on 29252a_GL (validation, true + proxy labels), 22845a_GL and
22846a_GL (proxy labels), all orders 1–7 ols/wls, gamma/gamma_u on a 4 000-sounding
subsample, no cut_N. Loader check: `load_orbit_proxy` gives bit-identical ln T and
σ(ln T) to `orbit_io.load_orbit` on 29252a; the production-order fits reproduce the
stored `_nosg` k1 to 2e-6. With true labels, every 29252a number in "Result
2026-10-29" is reproduced (ols_3 0.69 / 0.57 %, ratio 0.94 %, ols_7 ratio 9.03 %,
k2 = 0 in 63 / 97 %).

**P0 — proxy validation on 29252a** (`results/rep_proxy_validation_29252a_GL.{md,csv}`,
`rep_proxy_scatter_29252a_GL.png`):

| proxy | Spearman | far (>20 km, QF0) precision / recall / Jaccard | near (<3 km) precision / recall | local-ref (>10 km, QF0) precision |
|---|---|---|---|---|
| A (not in Lite) | 0.59 | 0.68 / 0.95 / 0.65 (n 1735) | 0.79 / 0.76 | 0.68 |
| B (A + Lite QF1) | 0.70 | **0.81** / 0.43 / 0.39 (n 658 vs 1240 true) | 0.75 / 0.94 | 0.83 |

- Rule → proxy B, T = 20 km. **P0a PASS** (0.81 ≥ 0.80), at the cost of recall 0.43.
- Anomaly: the imported production function with the true distance reproduces the
  stored anomaly to 7e-5 ppm; with proxy B it gives r = 0.992 against the stored
  anomaly (n = 2 769 common, finite fraction 31.6 vs 33.6 %, median |Δ| 0.017 ppm).
- **P0b FAIL.** Near precision 0.75 passes; SNR transfer: ols_3 O2A −0.78 → −0.82,
  ols_3 SCO2 3.33 → 2.65, ols_7 SCO2 0.39 → 0.31 pass; ols_7 O2A −0.11 → −0.03
  fails the factor-1.5 test (both values ≈ 0, so the relative test is ill-posed
  there, but the bar stands). Per the rule, the near-cloud SNR on 2018-10-18 is
  reported below as unvalidated and R3 is not scored.
- **P0c FAIL.** SCO2/O2A ratio spread true → proxy: ols_2 1.22 → 1.70 (+40 %),
  ols_3 0.94 → 1.11 (+18 %), ols_7 9.03 → 9.46 (+5 %). The far-set *composition*
  changes (658 of the 1240, different latitude mix) and the global spread contains
  along-track geophysical variation, so absolute spreads are not transferable at
  ±25 %. The 29252a verdicts themselves survive the proxy: R1 (0.48 / 0.82 / 1.11 %),
  R2 (×8.5) and the R5 ordering (1.11 < 1.70 < 1.92 < 9.46) all hold with proxy labels.

**Replication tables** (proxy B, far = proxy > 20 km & QF0, subset all; robust SD %
after per-fp mean removal; full tables incl. ocean/land, wls_N, orders 4–6 in
`results/rep_summary_{orbit}.md`):

| orbit (n_far) | variant | O2A | WCO2 | SCO2 | SCO2/O2A | k2=0 O2A / SCO2 |
|---|---|---|---|---|---|---|
| 29252a true (1240) | ols_2 / ols_3 / ols_7 / gamma | 0.84 / 0.69 / 4.97 / 0.32 | 3.4 / 5.6 / 17.1 / 4.4 | 1.39 / 0.57 / 8.04 / 1.25 | 1.22 / 0.94 / 9.03 / 1.24 | ols_3: 0.63 / 0.97 |
| 22845a (164) | ols_2 | 0.14 | 0.19 | 0.66 | 0.70 | 0.00 / 0.80 |
| | ols_3 | 0.18 | 0.63 | **1.20** | 1.28 | **1.00 / 1.00** |
| | ols_7 | 1.42 | 2.05 | 2.07 | 2.47 | 0.02 / 0.67 |
| | gamma (n 56) | 0.17 | 0.27 | 1.01 | 1.01 | 0 / 0 |
| 22846a (598) | ols_2 | 0.53 | 1.25 | 1.67 | 1.27 | 0 / 0 |
| | ols_3 | 0.40 | 1.81 | **1.18** | **1.64** | 0.78 / 0.99 |
| | ols_7 | 4.76 | 14.1 | 7.59 | 7.23 | 0.23 / 0.49 |
| | gamma (n 165) | 0.18 | 2.55 | 1.55 | 1.40 | 0 / 0 |

Ocean = all on both 2018 orbits; land: 22845a has 27 land soundings and 0 far-clear,
22846a 986 land soundings and 0 far-clear (proxy B removes every land footprint
from the far set; Lite land glint soundings sit next to non-Lite ones).

Local near-cloud SNR of k1 (unvalidated, P0b failed; `results/rep_snr_{orbit}.csv`):
22845a O2A ols_2 / 3 / 7 = −2.0 / −2.3 / 0.0, SCO2 2.6 / 1.4 / 1.0 (ols_3 − ols_7 in
|SNR|, 95 % CI [0.14, 0.84]); 22846a O2A −0.74 / −0.42 / 0.06, SCO2 0.01 / −0.08 / 0.05
(no near-cloud signal at all in SCO2 with the proxy labels on this orbit).

**Verdict against the bars**

| bar | 22845a | 22846a |
|---|---|---|
| R1 ols_3 ≤ 1 % (O2A, SCO2), ratio ≤ 1.5 % | **FAIL** (0.18, 1.20; 1.28) | **FAIL** (0.40, 1.18; 1.64) |
| R2 ols_7 ratio ≥ 4 × ols_3 | **FAIL** (2.47 / 1.28 = 1.9×) | PASS (7.23 / 1.64 = 4.4×) |
| R3 |SNR| ols_3 ≥ ols_7 | not scored (P0b fail); point values would pass | not scored; would pass in O2A, SCO2 ≈ 0 for all |
| R4 order-3 k2 = 0 (63 / 97 %) | 100 / 100 % (O2A not within ±15 pp) | 78 / 99 % (O2A not within ±15 pp; SCO2 reproduces) |
| R5 ols_3 < ols_2 < gamma < ols_7 | **No** (0.70 < 1.01 < 1.28 < 2.47: ols_2 best) | **No** (1.27 < 1.40 < 1.64 < 7.23: ols_2 best) |
| R6 land vs ocean | insufficient (0 land far-clear) | insufficient (0 land far-clear) |

**Sensitivity to the far-set definition** (not a bar; `rep_sensitivity.py`,
`results/rep_sensitivity.md`): proxy A/B × 10/20/30 km, plus a latitude-local
ratio spread (minus the ±1° far-set median, which removes along-track gradients).

| orbit | SCO2/O2A ratio spread ols_2 / ols_3 / ols_7, global, range over 4 far definitions | same, latitude-local |
|---|---|---|
| 29252a | 1.37–2.13 / 0.98–1.43 / 6.2–9.5 (true: 1.22 / 0.94 / 9.03) | 0.58–0.98 / 0.38–0.66 / 5.7–8.1 (true 0.74 / 0.47 / 7.37) |
| 22845a | 0.70–2.28 / 1.28–1.53 / 2.5–5.2 | 0.47–0.91 / 0.75–0.99 / 2.1–5.0 |
| 22846a | 1.17–1.73 / 1.10–2.08 / 5.3–7.2 | 0.57–0.77 / 0.45–0.63 / 4.8–6.1 |

- Order 7 has the largest ratio spread of orders 1–7 on all three orbits with the
  pre-registered far set, and of ols_2/3/7 under every far definition, but
  its amplification is orbit-dependent: ×5–13 (local) on 29252a/22846a, only ×3–5 on
  22845a (×1.9 global on the pre-registered 164-sounding set).
- Order 3 vs order 2 does **not** replicate. On 22846a they are equal within the
  definition scatter (ols_3 better locally, worse globally on B20); on 22845a ols_2 is
  better in SCO2 (local ratio 0.47–0.91 vs 0.75–0.99; SCO2 k1 local SD 0.56 vs 0.87 %)
  and SCO2 order 3 has k2 pinned at zero in 100 % of far-clear footprints (order 2
  also in 80 %), i.e. this orbit's SCO2 curvature is concave-down even at order 2.
- ols_3 SCO2 k1 spread > 1 % on both 2018 orbits under every far definition
  (1.14–3.5 %); the 29252a value 0.57 % is the low end, not typical.

**Conclusion for the FIT_ORDER gate.** Replicated: dropping order 7 for O2A and SCO2
(its far-clear ratio scatter is the largest on every orbit and definition tested). Not
replicated: order 3 as the specific replacement — on 2018-10-18 order 2 is as good
or better in the SCO2/O2A ratio, and the order-3 k2 pinning is stronger (O2A 78–100 %).
The evidence favours `(2, ·, 2)` or a k1-only order-3 decision only after the
SCO2 concave-curvature question (k2 pinned at order 2 on 22845a) is resolved. This
is ocean-glint only.

**Caveats.** (1) No true cloud distance on 2018-10-18; the proxy keeps 43 % of the
true far-clear set and changes its composition, which moves absolute spreads by up
to 40 % on 29252a (P0c fail). (2) The global spread mixes sounding noise with
along-track geophysical variation; the latitude-local version is the fairer
cross-orbit metric but was added after the bars. (3) 22845a far-clear n = 164 (one
latitude band, −27 to −9°), gamma n = 56. (4) Near-cloud metrics unvalidated (P0b).
(5) WCO2 is reported without a bar.

**Not done.** Land nadir replication (no land-nadir orbit locally) — still open.
Land glint: no far-clear land soundings on either orbit, so no land/ocean comparison.
cut_N variants were not run (optional; H3 already failed on 29252a). 22846a L1B is in
the session scratchpad only (not copied into `data/`); `rep_sensitivity.py` points
to that path and will need a local copy to be re-run.

Files: `rep_orbit_io.py`, `rep_proxy.py`, `rep_run.py`, `rep_sensitivity.py`;
`results/rep_per_sounding_{29252a,22845a,22846a}_GL.parquet`,
`rep_summary_{orbit}.{csv,md}`, `rep_snr_{orbit}.csv`,
`rep_proxy_validation_29252a_GL.{csv,md,json}`, `rep_proxy_choice.json`,
`rep_sensitivity.{csv,md}`, figures `rep_spread_vs_order.png`, `rep_land_ocean.png`,
`rep_proxy_scatter_29252a_GL.png`.

### Validation note on the replication (main session) — 2026-10-29

Checked against `results/rep_summary_*.csv` and `rep_per_sounding_*.parquet`. The 22845a far-clear
set is one patch of 164 soundings (8.5–27 °S) whose residual rms (0.030 O2A, 0.036 SCO2) is 3–5×
below the orbit's own QF0 median (0.096 / 0.115) and the other orbits (0.14–0.22); its O2A
order-3 k1 is 0.97 orbit-wide (other orbits 0.99–1.00). Bootstrap 90 % intervals of the far-clear
spreads (400 resamples of soundings, per-footprint means removed):

| orbit (n_far) | order | SCO2 k1 spread % | SCO2/O2A ratio spread % |
|---|---|---|---|
| 29252a (658) | 2 / 3 / 7 | 1.27 [1.27, 1.71] / 0.82 [0.78, 1.02] / 9.0 [7.9, 9.7] | 1.70 [1.49, 1.92] / 1.11 [1.01, 1.33] / 9.5 [8.3, 10.0] |
| 22845a (164) | 2 / 3 / 7 | 0.66 [0.63, 1.16] / 1.20 [1.01, 1.49] / 2.1 [1.6, 2.9] | 0.70 [0.56, 0.87] / 1.28 [1.03, 1.42] / 2.5 [1.9, 3.0] |
| 22846a (598) | 2 / 3 / 7 | 1.67 [1.53, 1.86] / 1.18 [1.08, 1.28] / 7.6 [6.8, 8.6] | 1.27 [1.16, 1.35] / 1.64 [1.50, 1.74] / 7.2 [6.3, 7.5] |

Reading: order 7 is worst everywhere (×2–8, outside all intervals). For single-band k1, order 3
is better on two of three orbits; for the SCO2/O2A ratio, order 2 is better on two of three, because
the order-2 scene-dependent slope bias is common to both bands and cancels in the ratio. Neither
order is settled for PPDF inference, where k2 matters and order 3 pins it (78–100 % here). All
evidence remains ocean glint; the 22845a verdicts rest on one atypical clear patch.

## Target-mode land orbit 48443a_TG (Opus agent) — 2026-10-29

Goal: first land test of the truncation-order result. Orbit 48443a, target mode over
Salt Lake City (2023-08-10, L1B B11008r, Lite B11.2.10 `oco2_LtCO2_230810_B11210Ar_240919170825s.nc4`).
6 096 L1B soundings, 5 631 in the Lite file (the τ step only processes Lite soundings), 3 247
QF0, 100 % land, SZA 28–29°, VZA 0–57°. Scripts `tg_*.py`; outputs `results/tg_*`,
pipeline outputs under `results/pipeline_2023-08-10/`.

### Data situation (checked before writing the bars)

- **No collocated MODIS cloud field.** The local MYD35/MYD03 granules (20:15–20:55 UTC) are
  night passes over 65–110 °E; OCO-2 sees the target at 20:33–20:37 UTC. A CMR query gives the
  nearest Aqua day granule over the target at 21:25 UTC (+50 min, outside the production ±20 min
  window; Aqua free-drift era). The production cloud-distance step (`workspace/oco_modis_cloud_distance.py`,
  phases 3–5, run on a symlinked copy of the inputs so `data/processing/` is untouched) therefore
  writes `nearest_cloud_distance_km = −999` (NoMODIS) for all 5 631 soundings. The existing
  `data/processing/2023/222/48443a_TG/phase4_results.pkl` is the same placeholder, pickled under
  an old module path, so it could not be reused.
- **Lite-membership proxy fallback** (`rep_orbit_io.cloud_proxy_km`): distance to the nearest
  L1B sounding not in the Lite file (465 such soundings). In a ~30 × 20 km target scene the
  proxy is bounded by the scene: proxy A max 4.2 km (median 0.8), proxy B (A + Lite QF1) max
  1.1 km. **No sounding has proxy > 20 km (or > 5 km), and 93 % (A) / 100 % (B) are < 3 km.**
  The far-clear / near-cloud split of the ocean study is therefore undefined here; the run_study
  far-clear rows will be empty (n_far = 0) and near-cloud correlations are not computed.
  (Disclosure: these label distributions were computed before the bars; no fit had been run.)

### Pre-registered definitions (written before any fit)

- Sample S = Lite QF0 soundings of 48443a_TG with finite k1 in all three bands for the variant
  (all land). Variants from `run_study.py`: ols_1..7, wls_1..7, gamma/gamma_u on a 3 000-sounding
  subsample (`--gamma-subsample 3000`).
- **Within-scene (VZA-detrended) spread** of y ∈ {k1_O2A, k1_WCO2, k1_SCO2, k1_SCO2/k1_O2A}:
  OLS of y/median(y) − 1 on 8 footprint-index intercepts + a linear VZA term; spread = 1.4826·MAD
  of the residuals, in %. Ratio spreads ×400 ppm/100 give the ppm equivalent.
- **Ocean-comparable spread**: the run_study metric (per-fp mean removed, no VZA term) on S.
- **VZA slope**: coefficient of the same regression, in % of median per 10° VZA; SE from a frame-block
  bootstrap (blocks of 20 consecutive frames, 1 000 resamples); "significant" if |slope| > 2 SE.
  Robustness (not a bar): add surface altitude (Lite `Sounding/altitude`) as a covariate, since
  the scene spans the Salt Lake valley and the Wasatch front.
- **Geometry tables**: binned medians of k1 (ols_2, ols_3, ols_7, gamma) vs VZA (6 bins), vs
  scattering angle (from Lite SZA/VZA/SAA/VAA) and vs Lite `glint_angle`, per band.

### Pass/fail bars (written before running the fits)

- T1: within-scene SCO2/O2A ratio spread of ols_3 ≤ 1.5 %.
- T2: within-scene SCO2/O2A ratio spread of ols_7 ≥ 4 × ols_3.
- T3 (report): ordering of ols_3 vs ols_2 for the ratio and for single-band k1 (O2A, SCO2), with
  a sounding-bootstrap 90 % interval on the spread ratio ols_2/ols_3.
- T4 (report): order-3 k2 zero-bound fraction (k2 ≤ 1e-6) on S for O2A and SCO2, compared with the
  ocean range 63–100 % (O2A) / 97–100 % (SCO2). "Ocean-like" = O2A ≥ 60 % and SCO2 ≥ 90 %.
- T5: ols_3 k1-vs-VZA slope, per band: |slope| < 1 % per 10° = no detectable scattering trend
  (PASS); slope and its significance reported for ols_2, ols_7, gamma too.
- Pipeline provenance: wall times of each step, files created, and whether the order-7/3/7
  production fit reproduces the stored `_nosg` k1 (run_study repro check < 1e-4).

### Results — 48443a_TG (Opus agent), written after the runs

Correction to "Data situation": the local MODIS granules span 22–110 °E (plus two polar
granules), not 65–110 °E; the conclusion (no overlap with the target) is unchanged.

**Pipeline provenance** (all outputs under `results/pipeline_2023-08-10/`; nothing written to
`data/`, `results/` or `src/`, checked by directory listings before/after):

| step | command | wall time | outputs |
|---|---|---|---|
| cloud distance (phases 3–5; phase 2 download skipped) | `python workspace/oco_modis_cloud_distance.py --date 2023-08-10 --skip-phase 2 --data-dir <P>/data_links --output-dir <P> --viz-dir <P>/viz` | 11 s | `results_2023-08-10.{h5,json}` (5 631 soundings, all −999 NoMODIS), `viz/kdtree_range_2023-08-10.png` |
| benchmark, 100 soundings (`OCO_FP_MAX_SOUNDINGS=100`) | `python src/spectral/fitting.py --date 2023-08-10 --data-dir <P>/data_links --output-dir <P>/bench_100 --orbit 48443a_TG --fit-workers 6` | 111 s (τ 101 s, mostly ABSCO set-up) | `bench_100/…` |
| full τ + fits | same, `--output-dir <P>` | 1 601 s (τ 25.9 min in six 1 000-sounding chunks of 2.7–5.9 min; fit + write ≈ 25 s) | `2023-08-10/48443a_TG/fp_tau_combined.h5` (412 MB), `fitting_details/fitting_details_2023-08-10_48443a_TG.h5`, 18 example-fit PNGs |
| order study | `python run_study.py --l1b … --fp-tau … --fit-details … --gamma-subsample 3000 --workers 6 --out-dir results/tg_run` | 59 s | `tg_run/{per_sounding_48443a_TG.parquet, summary_48443a_TG.csv, summary.md, fig_*.png}` |
| TG analysis | `python tg_analysis.py --per-sounding … --fp-tau … --lite …` | ~6 min (bootstrap) | `tg_within_48443a_TG.csv`, `tg_vza_48443a_TG.csv`, `tg_summary.md`, `tg_k1_vs_vza_48443a_TG.png`, `tg_spread_vs_order_48443a_TG.png`, `tg_geometry_48443a_TG.png` |

`<P>/data_links/` holds file-level symlinks to the L1B/Met/CPr/Lite/MYD35/MYD03 inputs so the
phase-3 cache lands in `<P>/data_links/processing/`. `fitting.py` needs `pyproj`, which the default
miniforge `python` lacks; it was run with `~/miniforge3/envs/ml310/bin/python` (3.10.20, pyproj
3.7.1, numpy 2.2.6). The study scripts ran with the default `python`. The τ log repeats
"nu2=4759.6 cm⁻¹ out of range (too low)" twice per chunk (also in the benchmark); not investigated.
Fitting details: 5 631 soundings, `cld_dist_km` = −999 everywhere, `xco2_bc_anomaly` all NaN
(no clear-sky references). Production reproduction: `ols_7/ols_3/ols_7` reproduce the stored
`_nosg` k1 to 2.7e-8 / 2.3e-8 / 2.3e-8 (bar < 1e-4: PASS).

**Cloud labels.** Production: 0 collocated soundings. Proxy A: median 0.8 km, max 4.2 km,
93 % < 3 km; proxy B max 1.1 km. Far-clear (> 20 km, QF0) = 0 under every definition, so
`tg_run/summary.md` is all-NaN (n_far = 0) and no near-cloud statistic is computed. All results
below are on the 3 247 QF0 soundings (1 728 for gamma).

**Within-scene spreads** (robust SD %, fp intercepts + linear VZA removed; [frame-block 90 %];
cubic = cubic-VZA robustness; fp-only = run_study metric):

| variant | O2A | WCO2 | SCO2 | SCO2/O2A ratio [90 %] | ratio cubic | ratio fp-only |
|---|---|---|---|---|---|---|
| ols_1 | 1.27 | 1.65 | 0.79 | 1.07 [0.73, 1.48] | 1.15 | 1.98 |
| ols_2 | 0.85 | 3.75 | 1.21 | **1.28** [1.14, 1.38] | 1.27 | 2.08 |
| ols_3 | 1.41 | 3.07 | 0.91 | **1.73** [1.42, 1.92] | 1.61 | 2.55 |
| ols_7 | 5.99 | 8.01 | 13.0 | **15.0** [12.5, 16.8] | 15.4 | 12.1 |
| wls_2 | 0.91 | 3.72 | 2.21 | 1.91 | 1.57 | 3.51 |
| gamma | 0.82 | 4.58 | 1.89 | 1.79 | 1.81 | 1.96 |
| gamma_u | 1.16 | 5.66 | 1.82 | 1.33 | — | 2.03 |
| ocean far-clear ols_2 / 3 / 7 (29252a, 22845a, 22846a) | | | | 1.22, 0.70, 1.27 / 0.94, 1.28, 1.64 / 9.0, 2.5, 7.2 | | |

Spread ratios (sounding bootstrap 90 %): ols_2/ols_3 = 0.74 [0.71, 0.77] (ratio), 0.60
[0.58, 0.63] (O2A), 1.34 [1.28, 1.38] (SCO2); ols_7/ols_3 (ratio) = 8.7 [8.2, 9.2].

**k1 vs VZA** (slope % per 10°, ± frame-block SE; all |t| > 2 unless marked):

| variant | O2A | WCO2 | SCO2 | ratio |
|---|---|---|---|---|
| ols_1 | −1.36 ± 0.18 | −4.06 ± 0.24 | −2.41 ± 0.11 | −1.05 ± 0.13 |
| ols_2 | +2.33 ± 0.10 | +4.06 ± 0.30 | +3.38 ± 0.12 | +1.03 ± 0.09 |
| ols_3 | **+2.71 ± 0.16** | **+8.69 ± 0.38** | **+1.09 ± 0.09** | −1.52 ± 0.13 |
| ols_7 | +1.86 ± 0.12 | +4.69 ± 0.57 | +4.34 ± 0.35 | +2.46 ± 0.37 |
| wls_2 | −0.13 ± 0.11 (n.s.) | +1.00 ± 0.28 | −2.19 ± 0.23 | −2.08 ± 0.16 |
| gamma | +0.66 ± 0.09 | +4.96 ± 0.44 | +0.16 ± 0.14 (n.s.) | −0.50 ± 0.09 |

Surface altitude as a covariate changes no slope by more than 0.3 %/10° (VZA–altitude correlation
−0.01; VZA–latitude +0.19; VZA–scattering angle −0.43). The binned profiles are curved and
monotone: ols_3 O2A goes from −2.6 % (VZA 0–10°) to +9.1 % (50–60°), SCO2 −2.4 → +2.8 %, WCO2
−13.5 → +25.4 %; versus scattering angle (104–160°) ols_3 O2A falls from +6.8 % to −1.6 % with no
hot-spot feature near 160°. Noise-weighted gamma is nearly flat in O2A and SCO2 (±2.3 % over all
geometry bins).

**k2 zero-bound fraction (QF0)**: ols_3 O2A 39 %, SCO2 77 %, WCO2 2 %; ols_2 0 / 0 / 12 %;
ols_7 15 / 47 / 81 %; gamma 0.

**Verdict against the bars**

| bar | result | verdict |
|---|---|---|
| T1 ols_3 ratio spread ≤ 1.5 % | 1.73 % [1.42, 1.92] (cubic 1.61 %) | **FAIL** |
| T2 ols_7 ratio ≥ 4 × ols_3 | 15.0 / 1.73 = 8.7× [8.2, 9.2] | **PASS** |
| T3 ols_3 vs ols_2 | ratio: ols_2 better (×0.74); O2A: ols_2 better (×0.60); SCO2: ols_3 better (×1.34^-1) | same split as 2 of 3 ocean orbits |
| T4 order-3 k2 = 0, ocean-like (O2A ≥ 60 %, SCO2 ≥ 90 %) | 39 % / 77 % | **not ocean-like** (less pinning on land) |
| T5 ols_3 |slope| < 1 %/10° per band | O2A 2.7, WCO2 8.7, SCO2 1.1 %/10°, all significant | **FAIL** (detectable VZA trend in every band) |

**Reading.**
- Order 7 is again the worst by far; its ratio amplification on land (×8.7) is at the top of
  the ocean range (×1.9–9.6). Dropping order 7 for O2A and SCO2 replicates on land.
- Order 2 vs 3 reproduces the ocean pattern of the validation note: order 3 for single-band SCO2
  k1, order 2 for O2A and for the SCO2/O2A ratio.
- The VZA trend is fitter-dependent in sign and size (ols_1 negative, ols_2/3/7 positive,
  noise-weighted wls_2/gamma ≈ flat in O2A). τ already includes the airmass factor, so a physical
  path-lengthening trend would appear in every fitter. Most of the T5 trend is therefore
  identification: the τ grid stretches with airmass, and each fitter's coefficients are local
  cumulants at a fitter-specific τ*. A scattering contribution cannot be separated from this
  with the present data; the gamma O2A slope (+0.66 ± 0.09 %/10°) bounds it if the gamma fit
  is taken as identification-free.
- Order 3 pins k2 less on land (39 / 77 %) than on ocean (63–100 / 97–100 %).

**Caveats.** (1) No cloud labels: the sample is all QF0 soundings of a single 30 × 20 km scene,
not a far-clear set. Within-scene spread and ocean far-clear spread are different metrics:
the VZA term removes one geometry trend, and the ocean spreads mix along-track variation. Absolute
comparisons are indicative only. (2) The linear VZA detrend leaves curvature; the cubic check
moves the ratio spreads by ≤ 0.4 percentage points and changes no verdict. (3) One orbit, one
city, one day; frame-block bootstrap with 38 blocks. (4) `fitting.py` ran in the ml310 env (numpy 2),
not the default env; the production-order fits reproduce to 3e-8, so the fit path is consistent.

**Not done.** Near-cloud SNR (no labels). A clean scattering-vs-identification separation (needs
RT simulation of the TG geometry, or a fitter whose τ* is fixed across airmass).

Files: `tg_analysis.py`; `results/tg_run/`, `results/tg_within_48443a_TG.csv`,
`results/tg_vza_48443a_TG.csv`, `results/tg_summary.md`, `results/tg_*.png`, logs
`results/tg_run_study.log`, `results/tg_analysis.log`; pipeline tree `results/pipeline_2023-08-10/`.

### Method (as run)

- `resid_attrs.py` re-runs the production τ step (ABSCO v5.2 trilinear lookup, GEOS
  72-level met + CO2 prior, 71 layers, Rayleigh, transmission-space ILS convolution;
  numba-free copies of the `src/abs_util` helpers) for all 10 365 soundings (45 min,
  8 workers). Layer/species split by integrated gradients along s ∈ [0, 1]
  (τ_c(s) = −ln⟨e^{−sτ_hr}⟩_ILS, 32 Gauss–Legendre nodes in √s); η = absorber-weighted
  mean altitude, f_H2O, f_Rayleigh, boundary-layer share, ∂τ/∂pix (±0.1 px),
  ∂τ/∂ln w (ILS abscissa ±2 %), ∂τ/∂T (±2 K lookup T).
- **Validation PASS**: |Δτ|/τ vs stored `{band}_tau_output` median 6.1e-5 / 2.1e-4 /
  3.0e-4 (O2A / WCO2 / SCO2), p99 1.0e-3 / 1.9e-3 / 1.6e-3, max ≤ 4.5e-3; layer-sum
  closure ≤ 2e-7. VERT and H2O attributes are trustworthy; proxy 1b not needed.
- `resid_decompose.py`: pooled and per-sounding regressions from per-sounding Gram
  matrices on polynomial-residualized columns (≈ 75 columns; per-sounding model 20
  columns, chance R² 0.02), fixed-pattern share, SVD of the residual matrix, streak
  AUCs, τ-binned slopes, per-sounding coefficients vs L2 variables.
- `resid_fitters.py`: ols_N + {τη, τ_H2O, ∂τ/∂pix, ∂τ/∂ln w, ∂τ/∂T, combinations}.
- Extra (not in the bars): `resid_doppler.py` recomputes τ with the ABSCO grid shifted
  by λ(1 − v_inst/c) (the sign was chosen on 24 soundings: rms 0.23 → 0.019 vs
  0.23 → 0.42 for the opposite sign), refits, and the decomposition and fitters are
  re-run on that τ (`results/resid_doppler_decomp/`, section "— doppler").

### Result 1 — production τ: the misfit is a Doppler wavelength-registration error

Group R² of the structured (unconstrained order-3) far-clear residual, alone / partial
(order 2 within ±0.02 except TAU; full tables `results/resid_summary.md`):

| group | O2A pooled | O2A per-snd | WCO2 pooled | WCO2 per-snd | SCO2 pooled | SCO2 per-snd |
|---|---|---|---|---|---|---|
| all groups | 0.970 | 0.967 | 0.872 | 0.871 | 0.972 | 0.978 |
| VERT (η) | 0.013 / 0.019 | 0.006 / 0.006 | 0.024 / 0.011 | 0.002 / 0.003 | 0.009 / 0.049 | 0.006 / 0.021 |
| H2O | 0.012 / 0.012 | 0.018 / 0.003 | 0.011 / 0.011 | 0.013 / 0.004 | 0.006 / 0.020 | 0.007 / 0.089 |
| SOLAR | 0.020 / 0.241 | 0.026 / 0.167 | 0.017 / 0.055 | 0.013 / 0.046 | 0.002 / 0.021 | 0.004 / 0.026 |
| CONT | 0.000 / 0.002 | 0.000 / 0.006 | 0.003 / 0.002 | 0.002 / 0.003 | 0.000 / 0.001 | 0.000 / 0.001 |
| CHAN | 0.005 / 0.030 | 0.002 / 0.008 | 0.007 / 0.017 | 0.002 / 0.016 | 0.003 / 0.014 | 0.002 / 0.023 |
| **DISP (∂τ/∂pix)** | **0.937 / 0.968** | **0.937 / 0.962** | **0.823 / 0.862** | **0.823 / 0.857** | **0.943 / 0.971** | **0.943 / 0.976** |
| ILSW | 0.017 / 0.165 | 0.019 / 0.159 | 0.033 / 0.015 | 0.036 / 0.043 | 0.028 / 0.269 | 0.032 / 0.337 |
| TEMP | 0.013 / 0.004 | 0.017 / 0.015 | 0.014 / 0.007 | 0.015 / 0.008 | 0.006 / 0.052 | 0.006 / 0.109 |
| TAU (ln τ spline) | 0.001 / 0.004 | 0.001 / 0.016 | 0.003 / 0.006 | 0.003 / 0.007 | 0.001 / 0.018 | 0.000 / 0.004 |

Evidence that DISP is the telluric Doppler shift, not an ILS or calibration offset:
- the implied per-sounding shift −β/k1 correlates with `sounding_relative_velocity` at
  r = 0.999 / 0.999 / 0.995, slope 0.117 / 0.144 / 0.149 px per km/s against λ/(c·Δλ)
  = 0.174 (the linear signature under-reads shifts of up to 0.6 px);
- SVD mode 1 carries 71 / 92 / 56 % of Σr²; its channel pattern correlates with
  ∂τ/∂pix at 0.97 / 0.93 / 0.99 and its amplitude with v_inst at −0.99 / 1.00 / −0.96;
- the residual "streaks" are the two signs of ∂τ/∂pix: the AUC that separates the upper
  from the lower branch is 1.00 in every band (η, f_H2O, solar depth: 0.52–0.60);
- recomputing τ with the ABSCO grid shifted by −v_inst/c and no fitted parameter cuts
  the far-clear residual rms from 0.170 / 0.054 / 0.232 to 0.031 / 0.019 / 0.018 (×5 / 3 / 13).
  Production comments out this shift (`wvldat_obs = wvldat  # * (1.0 + v_inst / C_LIGHT)`,
  `src/abs_util/fp_abs_coeff.py`); the L1B `dispersion_coef_samp` is one per footprint
  per file and cannot follow the ±5.3 km/s along-orbit Doppler (0.17 px per km/s).
- Caveat: far-clear soundings occupy only two v_inst clusters (−3.6…−3.1 and
  +2.4…+2.6 km/s), so v_inst is collinear with latitude/airmass here (|r| 0.97–0.99).
  The recomputed-τ test does not depend on this, because its shift is fixed by v/c.

Fixed per-channel pattern: 9–17 % of Σr² (pooled), 14–18 % per fp. The ILS-width
signature is a secondary, mostly footprint-dependent pattern (SVD mode 2, 23 % O2A /
32 % SCO2, channel corr with ∂τ/∂ln w 0.75–0.77, amplitude corr with fp 0.32–0.35);
its partial R² (0.16–0.34) makes it "partial" by the bar.

**Decision on production τ: instrument category dominant (DISP), i.e. not path
physics.** VERT and H2O are below 5 % alone in every band and order: the
vertical-weighting hypothesis is rejected as the carrier of the misfit. The η slope
does not grow with τ (O2A pooled slope −0.011 → −0.021 per km from τ 0.06 to 5;
SCO2 changes sign).

### Result 2 — conditional fitters (production τ; full tables in resid_summary.md)

Order 3, far-clear (n = 1240): k1 spread % / k2 zero fraction / resid rms / lag-1 /
local near-cloud k1 SNR.

| variant | O2A | WCO2 | SCO2 | SCO2/O2A ratio spread |
|---|---|---|---|---|
| ols_3 | 0.69 / 0.63 / 0.170 / 0.64 / −0.78 | 5.64 / 0.01 / 0.054 / 0.50 / 2.40 | 0.58 / 0.97 / 0.232 / 0.62 / 3.34 | 0.94 |
| ols_3_disp (+∂τ/∂pix) | 0.72 / 0.66 / 0.055 / 0.37 / −0.80 | 4.82 / 0.00 / 0.023 / 0.48 / 3.07 | 0.55 / 0.97 / 0.057 / 0.22 / 3.38 | 0.95 |
| ols_3_eta (−βτη) | 1.77 / 0.38 / 0.170 / 0.64 / −0.48 | 2.77 / 0 / 0.054 / 0.50 / 3.46 | 4.66 / 1.00 / 0.232 / 0.62 / 3.13 | 5.02 |
| ols_3_h2o (two-τ) | 0.56 / 0.84 / 0.170 / 0.64 / −0.89 | 3.22 / 0 / 0.054 / 0.50 / −0.32 | 0.25 / 0.99 / 0.232 / 0.62 / −0.96 | 0.54 |

Near-cloud corr(k1, xco2_bc_anomaly), SCO2: ols_3 0.084, _disp 0.083, _eta −0.010, _h2o 0.231.

The linear Doppler term removes 75 % of the SCO2 rms and two thirds of the lag-1, but
k1, its spread and k2 hardly move: ∂τ/∂pix is odd in line position, so it is nearly
orthogonal to the τ polynomial. **The Doppler misfit is not what pins k2.**

### Result 3 — Doppler-corrected τ (extra; `resid_doppler.py`)

| order 3 | O2A | WCO2 | SCO2 |
|---|---|---|---|
| resid rms, production → Doppler τ | 0.170 → 0.031 | 0.054 → 0.019 | 0.232 → 0.018 |
| lag-1 | 0.64 → 0.71 | 0.50 → 0.55 | 0.62 → 0.62 |
| k1 median | 1.00 → 0.993 | 1.13 → 0.988 | 1.01 → 0.986 |
| k1 spread % | 0.69 → 0.41 | 5.64 → 2.44 | 0.58 → 0.71 |
| k2 zero fraction | 0.63 → **1.00** | 0.01 → **0.91** | 0.97 → **1.00** |
| local near-cloud k1 SNR | −0.78 → −2.11 | 2.40 → 3.04 | 3.34 → 3.95 |

Order 2 on Doppler τ: k1 spread 0.86 / 0.71 / 1.01 %; k2 zero fraction 0 / 0.80 / 0.11;
SCO2/O2A ratio spread 1.22 → 2.39 %; local SNR 1.32 / 1.08 / 1.45 (down from
1.47 / 2.72 / 3.47 in magnitude). WCO2 k1 now agrees with O2A/SCO2 (0.99 instead of 1.13).

Decomposition of the remaining misfit (Doppler τ, order 3; `results/resid_doppler_decomp/`):
- O2A: all groups 0.56 pooled; fixed per-channel pattern 59 % pooled / 72 % per fp;
  SOLAR partial 0.37 (pooled) / 0.24 (per-sounding), ILSW 0.17 per-sounding → partial,
  spectroscopy/solar/ILS, not path (VERT 0.12 alone, 0.01 partial).
- WCO2: all groups explain 0.10; SVD mode 1 (81 %) tracks v_inst/latitude (r 1.00)
  with a channel pattern unrelated to every attribute (|r| ≤ 0.12). Unexplained.
- SCO2: **path category dominant in the per-sounding model (partial 0.54)**; within
  it H2O alone 0.52 (VERT alone 0.44, collinear: H2O channels sit low, η ≈ 2 km).
  The per-sounding H2O coefficient tracks L2 `h2o_scale` at r = 0.991 (implied median
  scale 0.987); SVD mode 1 (50 %): channel corr with f_H2O 0.82, amplitude corr with
  h2o_scale −0.98. This is an H2O amount error of the met prior, not a photon-path
  effect.
- Pre-registered H2O action, two-τ fit on Doppler τ (order 3): SCO2 k1 spread
  0.71 → 0.15 %, SCO2/O2A ratio spread 1.10 → 0.52 %, WCO2 2.44 → 0.57 %; k2 zero
  fraction 1.00 / 1.00 (**FAIL**, bar < 20 %); far-clear k1_w/k1_dry vs L2 h2o_scale
  r = 0.998. Near-cloud: SCO2 local k1 SNR 3.95 → −1.41, WCO2 3.04 → −0.41;
  corr(k1, xco2_bc_anomaly) SCO2 0.11 → 0.26. The SCO2 near-cloud k1_dry shift vanishes
  (median far 0.9897, near 0.9895), and the contrast moves into k1_w (0.975 → 0.980).
- Vertical term on Doppler τ (order 3): k2 zero fraction 1.00 / 0.99 / 1.00 (**FAIL**);
  k1 spread worsens 4× (SCO2 0.71 → 3.43 %).

### Decision

1. **The structured misfit on production τ is instrument-side: the telluric Doppler
   shift that the τ step does not apply.** It carries 82–94 % of the residual variance
   alone (partial 0.86–0.98) in all bands and both orders. Applying λ(1 − v/c) removes it
   (rms ×3–13).
2. **Vertical weighting is rejected** as the driver, before and after the correction
   (partial R² ≤ 0.05 before; ≤ 0.16 pooled / 0.03 per-sounding after). The −βτη term
   raises the k1 spread 2–6× and does not release k2.
3. **After the Doppler correction, SCO2's leading misfit is the per-sounding H2O amount**
   (met-prior error, r = 0.99 with L2 h2o_scale). O2A is left with a fixed solar/ILS
   channel pattern, and WCO2 with an unexplained latitude/Doppler-tracking pattern.
4. **k2 pinning is not a channel-attribute misfit.** It gets *worse* when the misfit is
   removed (order 3: 1.00 / 0.91 / 1.00). The unconstrained cubic is concave-down in
   nearly every footprint. The τ-polynomial shape itself, i.e. the mean ln T(τ)
   curvature, drives it (TAU spline partial ≤ 0.02 at order 3); no regressor tested here fixes it.

### Caveats

- One orbit, ocean glint, two v_inst clusters only; the Doppler sign and slope must be
  confirmed on an orbit with a continuous v_inst range (nadir land).
- Attributes (η, f_H2O, ∂τ/∂·) were computed on the unshifted grid and reused for
  the Doppler-τ analyses (shift < 0.7 px); the second-stage DISP group is therefore a
  test of residual mis-registration only.
- The partial-R² bar is met at category level for SCO2 path (0.54) but not by a single
  group (H2O 0.145, VERT 0.028 per-sounding) because η and f_H2O are collinear. The
  H2O attribution rests on the h2o_scale correlation and the SVD mode.
- Pooled models include Doppler-scaled columns (v·∂τ/∂pix) that use v_inst from L1B,
  which is metadata, not a fitted quantity.
- The two-τ fit removes the SCO2 near-cloud k1 contrast. On this orbit, part of what
  looked like a near-cloud path signal in SCO2 k1 is therefore indistinguishable from
  an H2O amount change near clouds. This needs its own test before any feature is
  built on it.

### Recommended next step

1. Fix the τ step: apply the telluric Doppler shift (λ_obs = λ(1 − v_inst/c) on the
   ABSCO grid, sign verified on one orbit) in `src/abs_util/fp_abs_coeff.py`, behind a
   version bump of `TAU_CONVOLUTION_VERSION`. This is a production change for the user to
   decide; the sign/slope check on a nadir-land orbit with continuous v_inst comes first.
2. Re-run the fit-order and weighting conclusions on Doppler τ. WCO2 k1 moves 1.13 → 0.99
   and its spread 5.6 → 2.4 % (order 3) / 0.7 % (order 2), so earlier WCO2 statements
   ("κ ≈ 2 not credible", streaks) were Doppler artefacts.
3. For SCO2/WCO2, prescribe the H2O column from L2 `h2o_scale` (or fit k1_w) only after
   the near-cloud H2O question is resolved, since the two-τ fit takes the near-cloud
   SCO2 contrast.
4. k2: pursue the τ-shape question (local cumulants / τ*), not channel regressors.

### Files

Code: `resid_attrs.py`, `resid_decompose.py`, `resid_fitters.py`, `resid_doppler.py`.
Results: `results/resid_attrs_29252a_GL.npz` (1.1 GB, all soundings × 3 bands × 1016
channels), `resid_summary.md` (all tables, including both fitter sections and the Doppler
refit), `resid_decomp_29252a_GL.csv`, `resid_svd_*.csv`, `resid_fixed_pattern_*.csv`,
`resid_tau_slopes_*.csv`, `resid_streak_auc_*.csv`, `resid_ps_coefs_*.parquet`,
`resid_ps_coef_stats_*.csv`, `resid_fitters[_doppler]_*.csv/.parquet`,
`resid_doppler_*.csv/.parquet`, `resid_doppler_tau[_far]_29252a_GL.npz`; figures
`resid_partial_r2.png`, `resid_vs_tau_attrs.png`, `resid_mean_channel.png`,
`resid_streaks.png`, `resid_tau_binned_slopes.png`; second-stage copies of all of these in
`results/resid_doppler_decomp/`. Run order: resid_attrs → resid_decompose → resid_fitters →
resid_doppler (--sign -1; --save-far) → resid_decompose --tau … --out-dir
results/resid_doppler_decomp → resid_fitters --tau results/resid_doppler_tau_29252a_GL.npz
--tag doppler. Re-running resid_decompose rewrites resid_summary.md, so re-run the later steps after it.

### Follow-up 2026-10-29 — Doppler sign and slope on target orbit 48443a_TG

Why: on 29252a_GL v_inst sat in two clusters collinear with latitude. In target mode
`sounding_relative_velocity` sweeps −5.6…+5.7 km/s inside one scene (|v| vs VZA r = 0.998).
Script `resid_doppler_tg.py` (inputs from `results/pipeline_2023-08-10/`, read-only;
`xco2_qf == 0` only, since cld_dist is −999). 600 QF0 soundings, 12 signed-v bins × 50,
round-robin over the 8 fp. For each: unshifted τ (reproduces stored τ to median
6e-5 / 2e-4 / 4e-4, O2A / WCO2 / SCO2), corrected τ (ABSCO λ(1 − v/c), the 29252a sign),
wrong-sign τ (λ(1 + v/c)), and a scan of the ABSCO-grid shift −1.3…+1.3 px (0.1 px
steps, parabolic refinement of the order-3 rms minimum). Outputs
`results/resid_doppler_tg_{summary_48443a_TG.md, rms_by_v, shift, k, per_sounding}_48443a_TG.*`,
figure `results/resid_doppler_tg_48443a_TG.png`.

**Sign: confirmed in all three bands.** Corrected < production < wrong in every bin
with |v| > 0.5 km/s (5 positive, 5 negative bins); per-sounding: corrected better in
100 % of soundings in those bins (90–98 % in the bin straddling 0). Median order-3 rms:

| band | v −5.6…−4.6: prod / corr / wrong | v −0.8…0.1 | v +5.0…+5.7 | all 600 |
|---|---|---|---|---|
| O2A | 0.255 / 0.032 / 0.440 | 0.032 / 0.021 / 0.052 | 0.245 / 0.032 / 0.420 | 0.142 / 0.025 / 0.264 |
| WCO2 | 0.069 / 0.030 / 0.100 | 0.016 / 0.015 / 0.019 | 0.071 / 0.031 / 0.100 | 0.039 / 0.020 / 0.063 |
| SCO2 | 0.293 / 0.020 / 0.495 | 0.032 / 0.018 / 0.056 | 0.307 / 0.021 / 0.511 | 0.162 / 0.018 / 0.303 |

**Slope: physical.** Best ABSCO-grid shift vs signed v (no soundings at the scan edge):

| band | expected (px per km/s) | fitted slope | intercept (px) | scale = fit/expected | resid SD (px) | r | linear ∂τ/∂pix estimator: slope (scale) |
|---|---|---|---|---|---|---|---|
| O2A | −0.1737 | −0.1739 | +0.002 | 1.001 | 0.008 | −1.000 | 0.121 (0.70) |
| WCO2 | −0.1745 | −0.1760 | −0.006 | 1.009 | 0.004 | −1.000 | 0.113 (0.65) |
| SCO2 | −0.1740 | −0.1738 | −0.006 | 0.999 | 0.011 | −1.000 | 0.142 (0.81) |

Per-sounding scale (|v| > 1 km/s) median 1.001 / 1.011 / 1.001, IQR ±0.01–0.015, and the
same in every footprint (O2A 0.999–1.003). The 0.12–0.15 px per km/s found on 29252a_GL is
**attenuation of the linear estimator**: a first-order ∂τ/∂pix term under-reads shifts of
0.3–1 px. The same estimator on these soundings gives 0.65–0.81 of the physical value.
It is not partial compensation in the L1B dispersion: the intercept is ≤ 0.006 px and the
scale is 1.00. A free scale factor on v gains nothing over v/c, so the physical correction
λ_ABSCO(1 − v/c) needs no tuning.

k1, k2 and the k1-vs-VZA slope (% per 10°, pooled k1/median), production → corrected:

| band, order | k1 median | k2 median | k2 zero fraction | k1–VZA slope |
|---|---|---|---|---|
| O2A, 3 | 1.03 → 0.997 | 0.017 → 0 | 0.38 → **1.00** | +2.81 → +0.05 |
| O2A, 2 | 1.07 → 1.03 | 0.089 → 0.056 | 0 → 0 | +2.42 → +0.34 |
| WCO2, 3 | 1.20 → 1.02 | 2.20 → 0.27 | 0.01 → 0.23 | +8.57 → −0.28 |
| WCO2, 2 | 1.07 → 1.00 | 0.57 → 0 | 0.11 → 0.57 | +4.20 → +0.31 |
| SCO2, 3 | 1.04 → 1.01 | 0 → 0.005 | 0.76 → 0.44 | +1.09 → −0.16 |
| SCO2, 2 | 1.08 → 1.02 | 0.087 → 0.017 | 0 → 0.02 | +3.39 → −0.05 |

The production VZA slope on all QF0 soundings (tg_run parquet) is O2A ols_3 +2.70,
matching the other agent's +2.7 %/10° and this sample (+2.81). After the correction the
k1–VZA dependence collapses to |slope| ≤ 0.34 %/10° in every band and order: the
production k1 "geometry" trend was the Doppler misregistration (|v| ∝ VZA here). The
k2 picture is as on 29252a_GL. The correction lowers k2 everywhere and pins O2A order-3
k2 at zero in all soundings; WCO2 order-2 k2 goes to zero in 57 %. The k2 pinning is not
caused by the misregistration.

Remaining caveat: target mode over one scene; the sign/slope should hold for any mode
(it is λv/c), but the near-cloud contrast after the correction has only been checked on
29252a_GL.

## H2O vs cloud-path test (Opus agent) — 2026-10-29

Question: is the near-cloud SCO2/WCO2 signal on Doppler-corrected τ (k1 +2.8 % / +1.2 %,
SCO2 order-2 k2 0.008 → 0.022 within 1 km; O2A k1 −0.2 %) a photon-path effect of the
cloud, a water-vapour amount error in τ (moister air near clouds than the met prior), or
a mix? Code: `h2o_split.py` (species split), `h2o_analysis.py` (fits, contrasts,
correlations, figures). Outputs `results/h2o_*`.

### Pre-registered definitions and bars (written before running)

Data: orbit 29252a_GL, corrected τ from `results/resid_doppler_tau_29252a_GL.npz`.
All channels with finite ln T, τ, σ (as `near_cloud_doppler.py`), unit weights, all QF.

- **Species split**: integrated-gradient attribution of τ_c = −ln⟨e^{−τ_hr}⟩_ILS into dry
  gas (O2 for O2A, CO2 for WCO2/SCO2), H2O and Rayleigh, as `resid_attrs.py` but on the
  Doppler-shifted ABSCO grid λ(1 − v_inst/c) of `resid_doppler.tau_shifted`. Validation:
  Σ components = stored corrected τ to ≤ 1e-5 relative per channel (max over channels and
  soundings, per band); if it fails, the two-τ results are labelled untrusted.
- **Fits** per sounding and band:
  - S2: single-τ order 2, unconstrained (k1, k2); S3: single-τ order 3, k1, k2 ≥ 0 (BVLS).
  - **T2 (reported)**: ln T = c − k1_dry τ_dry − k1_w τ_h2o + ½ k2 τ_tot², unconstrained,
    with τ_dry = dry gas + Rayleigh (Rayleigh is a well-mixed path; it is nearly flat in
    channel and mostly aliases into c) and τ_tot = τ_dry + τ_h2o = corrected τ.
  - T2d (variant): k2 on τ_dry² only. T3 (link to the residual agent): T2 plus −⅓k3 τ_tot³,
    k1_dry, k2 ≥ 0.
  - Reported: k1_dry, k1_w, k2, k1_w/k1_dry, residual rms.
- **Contrast** (identical to `near_cloud_doppler.py`): value − latitude-local far reference
  (median of cld_dist > 10 km, QF0, |Δlat| ≤ 1°, same footprint, ≥ 5 soundings); relative
  for k1-type quantities and ratios, absolute for k2 and for the H2O variables (also
  relative for h2o_scale/tcwv). SNR = bin median contrast / robust SD of far-clear
  (> 20 km, QF0) contrasts. Bins 0–1, 1–3, 3–10, 10–20, > 20 km.
- **Bar A (SCO2 k1, order 2)**: R_A = con₀₋₁(k1_dry, T2) / con₀₋₁(k1, S2).
  R_A ≥ 0.50 with SNR(k1_dry) ≥ 2 → path dominant; R_A < 0.25 with k1_w con₀₋₁ > 0 at
  SNR ≥ 2 → H2O dominant; otherwise mixed. The order-3 pair (S3 vs T3) and WCO2 are
  reported with the same rule, not part of the decision.
- **Bar B (SCO2 k2, order 2)**: same rule with R_B = con₀₋₁(k2, T2) / con₀₋₁(k2, S2)
  (absolute contrasts), SNR of the T2 k2 contrast.
- **Bar C (O2A control)**: PASS if |con₀₋₁(k1_dry, T2) − con₀₋₁(k1, S2)| ≤ 0.5 × far
  robust SD of the S2 k1 contrast, and k1_w is either ill-determined (far-clear robust SD
  of k1_w > 10 × that of k1_dry) or has |SNR₀₋₁| < 2. FAIL means the two-τ fit changes a
  band with no H2O signal, which would make it an unreliable discriminator.
- **Bar D (H2O attribution)**: among soundings < 3 km, OLS of the S2 SCO2 k1 contrast on
  [h2o_scale contrast, tcwv contrast]; R² ≥ 0.50 → H2O implicated. Also reported: the same
  for WCO2 and for k2; partial R² of the H2O pair after ln(cld_dist), and of ln(cld_dist)
  after the H2O pair; Pearson r with h2o_scale, tcwv, h2o_ratio_bc, cld_dist, aod_total,
  aod_water, xco2_bc_anomaly (values and contrasts) for < 3 km and far-clear; and an
  extra (not a bar) far-clear-calibrated share: slope of k1 contrast on h2o_scale contrast
  fitted on far-clear, applied to the near-cloud h2o_scale contrasts, predicted median /
  observed median.
- **Mechanism**: latitude-local contrast of h2o_scale, tcwv and the met-prior column
  tcwv_met = ∫q dp/g (from q_profile, p_profile) by bin.
- Interpretation caveat fixed in advance: h2o_scale and tcwv are L2 retrieval outputs from
  the same spectra (tcwv/tcwv_met tracks h2o_scale at r = 0.992), and H2O absorption sits
  low (η ≈ 2 km), so the k1_w degree of freedom also absorbs a lower-tropospheric path
  change. The bars measure attribution to the H2O-shaped degree of freedom; a pure amount
  error additionally predicts that the near-cloud rise of k1_w/k1_dry equals the
  h2o_scale rise and is the same in WCO2 and SCO2.

### Results — H2O vs cloud-path test (Opus agent), written after the runs

Species split (`h2o_split.py`, all 10 365 soundings, 23 min with 8 workers; 0.12 s per sounding
once BLAS is pinned to one thread per worker — with threaded BLAS the first 100-sounding timing run
did not finish in 10 min): Σ(dry + H2O + Rayleigh) vs stored corrected τ, max relative error 2.3e-7 / 6.0e-8 /
6.0e-8 (O2A / WCO2 / SCO2), median 2e-8: **PASS** (bar 1e-5). Median f_H2O 2.3e-4 / 0.15 / 0.16.
The single-τ fits reproduce `near_cloud_doppler.py` exactly (S2/S3 k1 identical; SCO2 0–1 km
contrast +2.83 %, SNR 3.57).

0–1 km contrasts (n = 421 with a latitude-local reference; relative for k1, absolute for k2):

| band | single-τ k1 o2 (SNR) | two-τ k1_dry (SNR) | k1_w (SNR) | single-τ k2 o2 (SNR) | two-τ k2 (SNR) |
|---|---|---|---|---|---|
| O2A | −0.18 % (−1.39) | −0.18 % (−1.40) | −2.8 % (−0.12), ill-determined | +0.0020 (1.25) | +0.0020 (1.26) |
| WCO2 | +1.19 % (3.18) | −0.04 % (−0.17) | +14.6 % (5.14) | +0.035 (2.76) | −0.003 (−0.44) |
| SCO2 | +2.83 % (3.57) | +0.10 % (0.42) | +7.9 % (3.41) | +0.0145 (3.65) | +0.0028 (1.40) |

1–3 km: SCO2 k1 +2.44 % → k1_dry +0.03 %; WCO2 +1.29 % → −0.02 %. Order 3: SCO2 S3 +2.98 %
(5.95) → T3 k1_dry −0.03 % (−0.19). T2d (k2 on τ_dry²) gives the same picture (SCO2 k1_dry
−0.12 %, k2 +0.0014). All contrasts vanish beyond 3–10 km.

**Bar A: H2O dominant** (R_A = 0.035, k1_w +7.9 % at SNR 3.4). **Bar B: H2O dominant**
(R_B = 0.19). **Bar C: PASS** (O2A k1_dry − k1 = 1.6e-5 ≪ 6.6e-4; k1_w far SD 0.24 vs k1_dry
0.0013, i.e. degenerate, so the two-τ fit leaves O2A unchanged). **Bar D: H2O implicated**:
R² of the < 3 km SCO2 k1 contrast on (h2o_scale, tcwv) contrasts 0.885 (WCO2 0.680, SCO2 k2
0.826); partial R² H2O | ln cld 0.887, ln cld | H2O 0.017; partial r(contrast, h2o_scale | ln cld)
0.94. Far-clear R² is the same (0.906), so the H2O relation is not near-cloud specific.

Mechanism: h2o_scale is elevated near clouds (+0.037 at 0–1 km, SNR 1.4; +0.075 at 1–3 km,
SNR 2.8), while the met-prior column is lower (tcwv_met −3.2 % / −4.6 %). Near-cloud air is
moister than the prior says. AOD_total also rises (+0.034, SNR 4.8).

Pre-registered consistency check (pure amount error: k1_w/k1_dry rise = h2o_scale rise, same in
WCO2 and SCO2), with the far-clear slope applied (`h2o_extra.py`):
- 1–3 km: **consistent** for SCO2 (ratio +6.4 %, predicted +6.0 %, residual SNR −0.14); WCO2
  +9.2 % vs +7.2 % (residual SNR 2.95).
- 0–1 km: **not consistent**. SCO2 ratio +6.2 % vs predicted +3.0 % (residual SNR 2.0); WCO2
  +14.7 % vs +3.6 % (residual SNR 9.1). The far-calibrated h2o_scale share of the single-τ SCO2
  k1 median contrast is 36 % at 0–1 km and 83 % at 1–3 km.

Extra (post hoc, `h2o_lat_breakdown_29252a_GL.csv`): the 0–1 km bin mixes two scenes.
- 15–25°N (n = 253, 34 % QF1): h2o_scale +0.147 / +0.047 (two 5° bands), SCO2 ratio +14.3 % /
  +2.4 %, SCO2 k1_dry −0.1 %, O2A k1 −0.4 %. This is the water-vapour amount case.
- 55–50°S (n = 126, 93 % QF1): h2o_scale +0.023, but O2A k1 +1.5 %, SCO2 k1 +2.9 % with
  k1_dry +1.7 % (survives the two-τ fit), WCO2 k1_dry +0.2 % with WCO2 ratio +14.4 %. This is a
  path signal in O2A and SCO2, plus an H2O-shaped WCO2 excess that L2 does not report.
The pooled median follows the larger tropical group, which is why bars A/B read "H2O dominant".

### Decision

1. By the pre-registered bars, the pooled near-cloud SCO2/WCO2 k1 and k2 signal on this orbit is
   carried by the H2O degree of freedom (A, B: H2O dominant; D: H2O implicated, R² 0.885). The
   two-τ fit removes it and leaves O2A untouched (C passes).
2. It is not a pure amount error everywhere. At 1–3 km it matches the L2 h2o_scale rise. At
   0–1 km the H2O degree of freedom takes 2× (SCO2) to 4× (WCO2) more than h2o_scale predicts.
   The excess comes from a 55–50°S cloud-edge scene where O2A k1 and SCO2 k1_dry also rise by
   ~1.5 %, i.e. a photon-path effect. **Verdict: a mix, regime-dependent: tropical near-cloud
   contrast = moister air than the met prior; Southern Ocean sub-km contrast = path effect.**
3. Consequence: a two-τ fit (or prescribing τ_H2O from h2o_scale) is the right tool to separate
   the two, but its k1_w must not be read as water amount within 1 km of cloud, and the WCO2
   two-τ split is unreliable there (k1_w absorbs a WCO2-only excess).

### Caveats

- One orbit; the 0–1 km contrast rests on 421 soundings in essentially two scenes (the latitude-
  local reference exists for only 421 of 3 272 soundings < 1 km; most of the 62–52°S cloud deck
  has no QF0 reference within ±1°). The scene split is post hoc.
- h2o_scale and tcwv are L2 outputs from the same spectra (tcwv/tcwv_met vs h2o_scale r = 0.992),
  so "H2O implicated" means the H2O-shaped signal, not independent proof of more vapour. The
  opposite sign of the met-prior contrast makes a real moisture difference plausible.
- The Southern Ocean group is 93 % QF1, so it is outside the operational QF0 product; its O2A
  rise also coincides with the AOD_total rise.
- SNRs use the far-clear robust SD of single soundings; the 0–1 km bin medians are not
  independent across 421 soundings in two scenes, so SNR overstates significance.

### Files

Code: `h2o_split.py`, `h2o_analysis.py`, `h2o_extra.py`. Results: `results/h2o_tau_split_29252a_GL.npz`
(327 MB), `h2o_per_sounding_29252a_GL.parquet`, `h2o_contrasts_29252a_GL.csv`,
`h2o_correlations_29252a_GL.csv`, `h2o_extra_29252a_GL.csv`, `h2o_lat_breakdown_29252a_GL.csv`,
`h2o_summary.md`, logs `h2o_split.log`, `h2o_analysis.log`; figures `h2o_k1_contrast_bins_29252a_GL.png`,
`h2o_scatter_scale_29252a_GL.png`, `h2o_ratio_vs_scale_29252a_GL.png`. Run order: h2o_split
(--subset all) → h2o_analysis → h2o_extra → h2o_extra --lat-breakdown.

## Cross-band k1 ratio scatter and continuum-polynomial test (main session) — 2026-10-30

Scripts `xr_ratio_scatter.py`, `alb_poly_test.py`; results `results/xr_ratio_scatter_29252a_GL.{csv,png}`,
`results/alb_poly_29252a_GL.csv`, `results/alb_poly_48443a_TG_smoke100.csv` (100 Doppler-variant
soundings of the land target orbit, from the local smoke run). Far-clear = cld > 20 km, QF0 (n = 1240 on
29252a_GL, in three latitude clusters: 52–49°S, 18°N, 22–24°N). Spread = robust SD (%) after removing
per-footprint medians.

### SCO2/O2A k1 ratio (the path proxy of the direct-XCO2 side study), 29252a_GL far-clear

| fit | ratio spread % | O2A k1 % | SCO2 k1 % | lag-1 | note |
|---|---|---|---|---|---|
| production τ, prod order (7,3,7) | 5.75 | 2.8 | 5.3 | −0.31 | noise-like |
| production τ, order 2 / 3 | 1.23 / 0.94 | 0.84 / 0.67 | 1.35 / 0.58 | 0.96 | 18°N cluster −8 % / −5 % (H2O) |
| corrected τ, order 2 / order 3 (bounded) | 2.38 / 1.09 | 0.84 / 0.41 | 1.01 / 0.72 | 0.97 | 18°N cluster −9 % / −5 % |
| corrected τ, two-τ k1_dry | 1.15 | 0.84 | 0.25 | 0.93 | 18°N cluster at 0; +2 % step at 50°S |

The two-τ fit removes the tropical (moist) cluster's 5–9 % single-τ offset. What remains is a 2 % step
between the Southern Ocean (airmass 2.8) and the tropical (2.35) groups, carried entirely by O2A k1
(r = +0.99 with airmass; SCO2 k1_dry r = +0.34). Within clusters the ratio scatters ~0.3 % (≈ 1.2 ppm)
and SCO2 k1_dry alone 0.25 %, which is comparable to the real XCO2 variation across the same soundings
(xco2_bc robust SD 0.27 %, 1.1 ppm, r = 0.97 with latitude). Airmass and scene are confounded on this
orbit, so the step cannot be assigned; it needs the CURC dates (handoff step 5).

### Continuum polynomial (L2-style linear/quadratic albedo per band)

Model: ln T = c + a1 x (+ a2 x²) − k1 τ + ½ k2 τ² (…), x = channel index scaled to [−1, 1]; applied to
S2, unconstrained S3 and T2 on corrected τ.

Ocean glint (29252a_GL far-clear): a1 is small and nearly the same for every sounding (O2A −0.012 ± 0.002,
WCO2 +0.0035 ± 0.0015, SCO2 −0.010 ± 0.002); k1 shifts by a near-constant +0.5 % (O2A), −0.3 % (WCO2),
+0.4 % (SCO2) with sounding-to-sounding variation of ±0.1 %; k1 spreads and the SCO2/O2A ratio spread are
unchanged; rms drops ≤ 6 %. The quadratic term is partly degenerate with the τ polynomial in WCO2
(k1 change p05–p95 −1 to +3 % for S2xx) and in unconstrained S3.

Land (48443a_TG, 100 soundings, August): a1 is larger and scene-dependent (WCO2 +0.020 ± 0.010,
SCO2 −0.003 ± 0.020, O2A +0.003 ± 0.005). The linear term shifts WCO2 k1 by −2 % (range −3.3 to −1 %)
and SCO2 k1 by −1 to +1.7 % per sounding, and reduces the k1 spread: SCO2 S2 0.85 → 0.54 %, T2 1.02 →
0.50 % (quadratic 0.22 %); WCO2 S2 1.09 → 0.90 % (quadratic 0.45 %); O2A unchanged (0.28 → 0.33 %).
rms drops 7–10 % in the CO2 bands. Unconstrained order 3 is the most sensitive (WCO2 k1 −3.9 %).

Reading: over ocean the continuum slope is a fixed instrument/solar/glint trend that cancels in contrasts;
over land it is a real surface-reflectance degree of freedom that currently leaks into the CO2-band k1 at
the 1–3 % level and would be read as a surface-type anomaly. A linear term per band is cheap and safe
(condition number 8 → 9); the quadratic term should only be used with the two-τ model and after checking
degeneracy. Not yet tested: whether near-cloud contrasts survive the extra column (the 3D-cloud signal acts
on τ, the column on x; cond ≈ 10 says they are separable, but this needs the near-cloud metrics on CURC).

### Identifiability of the continuum polynomial (main session) — 2026-10-30

Quasi-continuum channels (corrected τ < 0.1) per band: O2A ≈ 420–450 but only at the blue edge and the red
third (none in the band-head fifth); WCO2 ≈ 800–840 spread evenly; SCO2 12–37, all at the red edge.
A continuum-only quadratic is therefore impossible in SCO2 (σ(a2) 0.03–0.06 at rms 0.02) and marginal in
O2A. In the joint fit with τ known, x and x² are nearly orthogonal to the τ columns (VIF 1.0–1.4) and
σ(a2) = 0.0023 in every band; σ(k1) grows by ≤ 10 %. The price of omitting the quadratic when it is
present (a2 = 0.005–0.02 at the band edge) is a k1 bias of 0.3–0.4 × a2 in O2A/SCO2 but 2.7–3.0 × a2 in
WCO2 (τ small and hump-shaped), i.e. 1.5–6 % — the band in which L2 uses albedo_quad in its QF and bias
correction.

Per-sounding check against L2 (same sounding ids, L2 slope/quad converted to ln T change per unit x /
at the band edge): on 48443a_TG land the fitted a1 tracks the L2 albedo slope at r = 0.80 (O2A), 0.94
(WCO2), 0.80 (SCO2) with regression slope 0.8–1.0; the fitted a2 tracks L2 albedo_quad at r = 0.8 in SCO2
(scene SD 0.017), but only 0.4 (O2A) and 0.2 (WCO2), where the true quadratic (≈ 0.003 at the edge) is
below the per-sounding σ. Over ocean (29252a_GL far-clear) a1 vs L2 slope r = 0.5–0.66 with slope 0.75–0.8
(L2 has no quadratic over ocean). Conclusion: linear term in all bands; quadratic justified in SCO2 over
land (constrained by the line channels, as in L2), noise-level in O2A/WCO2 where a prior or pooled value
would be better than a free per-sounding term.

### Shape model comparison on corrected τ: polynomial vs gamma vs additive-floor mixture (main session) — 2026-10-30

Results `results/gm_models_29252a_GL_farclear.csv` (207 far-clear soundings) and
`results/gm_models_48443a_TG_smoke100.csv` (100 land soundings). All models include the two-τ split
(k1_dry, k1_w) and a linear continuum term a1·x. P2x = order 2 unconstrained; P3x = order 3 with
k1, k2 ≥ 0; Gx = closed-form gamma ln T = c + a1x − κ ln(1 + (k1_dry τ_dry + k1_w τ_h2o)/κ);
Mx = mixture ln T = c + a1x + ln[(1−f) e^{−(k1_dry τ_dry + k1_w τ_h2o)} + f e^{−k_s τ}], f ∈ [0, 0.5].

| band / sample | P2x rms, k1 spread | P3x | Gx | Mx |
|---|---|---|---|---|
| O2A ocean | 0.040, 0.90 % (r airmass 0.99) | 0.031, 0.50 % (k2 pinned) | 0.047, 1.18 % (κ ≈ 7.7) | 0.029, 0.20 % (f = 0.94 %, k_s → 0) |
| O2A land | 0.038, 0.33 % | 0.035, 0.36 % | 0.040, 0.44 % (κ ≈ 9.4) | 0.034, 0.33 % (f = 1.08 %) |
| WCO2 ocean | 0.0194, 0.84 % (k2 −0.03) | 0.0194, 0.52 % | 0.0194, 0.47 % (κ → bound) | 0.0194, 1.12 % (f unconstrained) |
| WCO2 land | 0.030, 0.93 % (k2 −0.21) | 0.030, 0.59 % | 0.030, 0.26 % (κ → bound) | 0.030, 0.24 % |
| SCO2 ocean | 0.0167, 0.26 % (k2 0.011) | 0.0164, 0.15 % | 0.0166, 0.27 % (κ 84, p05–p95 61–131) | 0.0163, 0.16 % (f 0.18 %) |
| SCO2 land | 0.0183, 0.50 % | 0.0182, 0.39 % | 0.0183, 0.49 % (κ 83, 41–286) | 0.0182, 0.52 % |

Findings.
- O2A is not gamma-shaped: the gamma form fits worse than order 2 (curvature in the wrong place). The
  shape is an additive floor: a ~1 % unabsorbed component with k_s → 0, f = 0.94 % ± 0.04 % (ocean) and
  1.08 % ± 0.06 % (land), r(f, albedo_o2a) = −0.94, no dependence on airmass, SZA, glint or AOD. This is an
  additive radiance offset relative to the continuum (the "zero-level offset" the L2 preprocessor fits
  for stray light and SIF), not Rayleigh (λ⁻⁴ predicts O2A/SCO2 ratio 53, observed 5.3). With the floor in
  the model, O2A k1 spread drops from 0.90 % to 0.20 % and its airmass correlation from 0.99 to 0.56;
  the order-2 "k2 ≈ 0.10" in O2A has been measuring this floor, not path variance.
- WCO2 is linear in τ: κ runs to its upper bound, mixture f is unconstrained, and the free order-2 k2 is
  negative (−0.03 ocean, −0.21 land) through continuum/curvature aliasing. k2 is not identifiable in WCO2.
- SCO2: order 2, gamma and mixture are equivalent (κ ≈ 80, i.e. k2 ≈ 0.012, but κ is poorly determined per
  sounding). Bounded order 3 pins k2 at 0 and gives the tightest k1 but a slight k1 offset.

### Sign of a free k2 under the full measurement model (main session) — 2026-10-30

Same samples as the shape comparison. Models: order-2 two-τ fit with linear (P2_lin) or quadratic
(P2_quad) continuum, free vs BVLS-bounded (k1, k2 ≥ 0); for O2A also the floor model with a free k2 in
the path part, ln[(1−f) e^{−k1 τ_dry − k1_w τ_h2o + ½ k2 τ²} + f].

| band | sample | free k2 median (frac < 0) | bound effect on k1 |
|---|---|---|---|
| O2A, polynomial | ocean / land | +0.100 (0 %) / +0.085 (0 %) | never active |
| O2A, floor model | ocean / land | −0.005 (95 %) / −0.001 (59 %) | n/a; k1 spread 0.4 % vs 0.2 % with k2 fixed at 0 |
| WCO2, lin continuum | ocean / land | −0.030 (80 %) / −0.212 (100 %) | k1 0.983 → 0.991 / 0.928 → 0.983; spread 0.84 → 0.46 % / 0.93 → 0.26 % |
| WCO2, quad continuum | ocean / land | +0.022 (41 %) / −0.200 (100 %) | ocean 41 % pinned; land k1 0.935 → 0.987, spread 0.47 → 0.18 % |
| SCO2 | ocean / land | +0.011 (0 %) / +0.012 (0 %) | never active |

Reading: positivity of k2 is a property of the log-Laplace transform (convexity), not of the gamma form,
so a negative free k2 flags a measurement-model error. With floor, continuum and H2O terms in place:
SCO2 k2 is positive everywhere (bound irrelevant); O2A path curvature is ≈ 0 (−0.005 ± 0.005, the
residual channel-weighting effect, A3) once the floor is modelled; WCO2 over land has a large, consistent
negative curvature (−0.2) that neither the continuum quadratic nor the H2O split removes — an open model
deficiency (candidates: different vertical weighting of H2O and CO2 lines, line mixing / ILS core in the
strong CO2 lines, residual τ scale). Bounding there is a band-aid that stabilizes k1 but hides the misfit;
fixing k2 = 0 in WCO2 is the honest version of the same choice until the cause is found.

### Stray light vs SIF in the O2A floor (main session) — 2026-10-30

The flat floor f in ln T = c + a1x + ln[(1−f)L(τ) + f] is an instrument/additive term relative to the
continuum. SIF is emitted at the surface and attenuated along the viewing path, so it is a component
f_SIF·exp(−k_s k1 τ) with k_s = μ_v/airmass (0.46 ocean glint, 0.62 on 48443a_TG), not a flat floor.
Tests (O2A, same samples): a free SIF-like component is assigned zero on both samples, including the land
target where L2 reports fs_rel = 0.86 % (p05–p95 0.2–1.5 %); the flat floor stays 0.94 % (ocean, no SIF)
vs 1.07 % (land). Synthetic injection of a 0.86 % SIF term into the land spectra and refitting the
floor-only model changes k1 by −0.45 % (p05–p95 −0.46 to −0.44 %) while f0 and the rms barely move: SIF
is not separable at the current residual level and hides in k1. Prescribing f_SIF = fs_rel from L2
changes k1 by +0.46 % (0.10–0.79 %) with r = 1.00 against fs_rel (Δk1 ≈ +0.53 × fs_rel); rms unchanged;
k1 spread 0.33 → 0.40 %. With the SIF term prescribed, a free k2 is −0.003 (≈ 0). Conclusion: the floor
handles stray light; SIF must be prescribed (land, O2A only) from fs_rel or the IMAP-DOAS SIF, otherwise
O2A k1 carries a vegetation-correlated error of about −0.5 % per 1 % SIF.

Constraint (Steven, 2026-10-30): no retrieved L2 quantity enters the fit. SIF therefore cannot be
prescribed from fs_rel. The only non-retrieved lever is Fraunhofer-line in-filling in the O2A continuum
channels (τ < 0.1, ≈ 440 channels, of which ≈ 88 have ILS-convolved solar lines deeper than 2 %, max 35 %):
an additive line-free term (stray light + SIF) can be estimated per sounding to σ ≈ 0.37 % of the continuum
at a ln T noise of 0.005, i.e. SNR ≈ 2 against a vegetation SIF of 0.9 %, which maps to a k1 correction
noise of ≈ 0.2 %. The O2-line floor at high τ sees stray light only (SIF attenuated along the viewing
path), so the two can be separated by their τ dependence once the floor is modelled line-free
(T_obs = e^{c+a1x}[(1−z−s)L(τ) + (z + s e^{−k_s k1 τ})/n_sun(λ)], n_sun = solar spectrum / envelope).
Not yet implemented; the current flat floor absorbs stray light and leaves a −0.5 % k1 error per 1 % SIF
over vegetation, smooth in space and absent over ocean, desert and snow.

Decision (Steven, 2026-10-30): option 1. SIF is not modelled; the O2A fit carries a flat floor for stray
light only, and the vegetation-correlated k1 error (≈ −0.5 % per 1 % SIF, O2A, land) is documented as a
limitation. In-filling (option 2) stays on the follow-up list, to be taken up only if O2A land k1
anomalies become decision-relevant.


## Agreed fit model: local test before the CURC run (main session) — 2026-10-30

Script `final_model_test.py`; results `results/final_model_per_sounding_{29252a_GL,48443a_TG_smoke100}.parquet`,
`final_model_summary_*.md`, `final_model_contrasts_29252a_GL.{csv,png}`. Model: corrected τ; O2A
ln[(1−f)e^{−k1τ}+f] + a1x, k2 = 0; WCO2 two-τ, k1 only, + a1x; SCO2 two-τ order 2 + a1x (+a2x² land).
Free k2 stored as diagnostic. Channel mask as the local studies (finite ln T, τ, σ > 0).

Far-clear 29252a_GL (n = 1240), k1 median (spread %): O2A 0.987 (0.17), WCO2 0.991 (0.25), SCO2 0.999
(0.28); SCO2/O2A ratio 0.32 %, WCO2/SCO2 0.50 %. For comparison: production 5.0 / 5.6 / 8.1 % and ratios
9.0 / 11.3 %; two-τ without continuum 0.86 / 0.84 / 0.25 % and ratios 1.10 / 1.25 %. Terms: f = 0.94 %
[0.87, 1.33], a1 −0.006 / +0.004 / −0.010, SCO2 k2 +0.011 [0.007, 0.016]; free-k2 diagnostic O2A −0.005
(95 % < 0), WCO2 −0.03 (80 % < 0), SCO2 never negative; no failed fits, f never at its bound; rms
0.029 / 0.019 / 0.017. Land (48443a_TG, 100): k1 spreads 0.40 / 0.29 / 0.24 %, f = 1.08 %, WCO2 a1 +0.020,
WCO2 free k2 −0.21 (100 % < 0, unchanged), SCO2 a2 ≈ 0 with wide scatter.

Near-cloud 0–1 km contrasts (n = 421, SNR): k1 O2A −0.18 % (−1.7), WCO2 −0.08 % (−1.0), SCO2 −0.28 %
(−1.1) — the single-τ SCO2 +2.8 % (3.6) and WCO2 +1.2 % (3.2) signals are gone; k1_w SCO2 +6.2 % (2.7);
SCO2 k2 +0.0004 (0.2). New near-cloud signals appear in the nuisance terms: SCO2 continuum slope a1
+0.0048 (SNR 5.3, i.e. a 1 % change in ln T across the band), WCO2 a1 +0.0011 (2.5), O2A floor f
+0.0005 (1.7), O2A residual rms +2.6 % (1.9). All vanish beyond 3 km. Cost: 2.5 ms per sounding for the
O2A nonlinear fit, < 0.2 ms for the linear bands (10 365 soundings in 4 s wall with 8 workers).

Reading: the model does what the tests predicted (k1 scatter 0.2–0.3 % in all bands, cross-band ratio
0.3–0.5 %), and the pooled near-cloud CO2-band k1 signal on this orbit is fully accounted for by H2O and
the continuum slope. Whether a path signal survives in specific scenes (the 55°S deck) needs the latitude
breakdown on CURC. The continuum-slope contrast near clouds is itself a candidate 3-D-cloud diagnostic.

## Pseudo-XCO2 from the agreed model: local check (2026-10-30)

Question: how would the agreed fit perform as a direct XCO2 estimator? Script `pseudo_xco2_test.py`, outputs
`results/pseudo_xco2_29252a_GL.{md,csv,png}` and `pseudo_xco2_clusters_29252a_GL.csv`. Nothing is scaled to
xco2_bc or to the prior (Steven's correction, 2026-10-30; a first version had normalised the far-clear
medians). τ_S is computed with the prior CO2 profile, so xco2_est = prior × k1_S (E0, geometric path assumed),
prior × k1_S/k1_O2A (E1, l_S = l_O2A assumed), prior × k1_W/k1_O2A (E1w); E2 is a held-out ridge of k1_S on
the gas-free fit terms and is calibrated to the prior by construction (stand-in for a path forward model).
Benchmarks: prior alone and the stored L2 xco2_strong_idp. Target xco2_bc, far-clear n = 1240.

| estimator | bias vs bc | SD(est − bc) | white noise | corr(est − prior, bc − prior) | −1.5 ppm hemispheric step |
|---|---|---|---|---|---|
| production E1 | −9.8 ppm | 44 ppm | 52 ppm | 0.11 | wrong sign, 28 ppm |
| two-τ T2 E1 | −28.7 ppm | 3.6 ppm | 1.1 ppm | 0.88 (slope 4.7) | 8 ppm, 5× too large |
| agreed E0 (SCO2 alone) | +0.0 ppm | 1.8 ppm | 1.1 ppm | −0.33 | wrong sign, 0.9 ppm |
| agreed E1 (SCO2/O2A) | +5.7 ppm | 1.7 ppm | 1.2 ppm | 0.15 (slope 0.31) | absent |
| agreed E2 ridge | +1.0 ppm | 0.64 ppm | 0.46 ppm | 0.62 (slope 0.48) | −0.9 of −1.5 ppm |
| prior alone | +0.8 ppm | 0.81 ppm | 0.29 ppm | 0 | 0 |
| L2 strong IDP (as stored) | −13.8 ppm | 1.05 ppm | 0.71 ppm | 0.72 (slope 1.3) | −2.5 of −1.5 ppm |

Reading. (1) Absolute level: the far-clear k1 medians 0.987 (O2A) / 0.999 (SCO2) put E1 at +5.7 ppm and E0
at 0.0 ppm. The 1.2 % band difference is the sum of the band-dependent path factor and the band-dependent
absolute τ accuracy (ABSCO scale, O2 continuum/CIA, solar model); L2 absorbs the same thing through its
ABSCO scaling and bias correction. An unscaled pseudo-retrieval therefore carries a spectroscopy-level
offset of order 1 % that cannot be separated from the path with one orbit. (2) Scatter: the agreed model
moves the pseudo-XCO2 from ±44 ppm (production) and ±3.6 ppm (two-τ) to ±1.7 ppm per sounding, ≈ 1.15 ppm
white plus ≈ 1.2 ppm scene-correlated structure (within clusters: wind speed +0.58, albedo −0.55, psurf
−0.60, airmass −0.37) that does not average away (block-24 SD 1.18 vs 0.75 for the prior). (3) There is
almost no XCO2 signal on this orbit to detect: the prior is within 0.8 ppm of xco2_bc and the near-cloud bc
anomaly is ≈ 0; the one feature, a −1.5 ppm hemispheric increment (bc − prior), is reproduced by IDP but not
by the k1 ratio (SCO2 and O2A k1 both step by +0.2 %, so the ratio is flat). (4) The within-cluster
correlation of E1 with bc − prior (0.48, slope 1.7) is common-mode scene dependence shared with the L2
increment (bc − prior itself correlates −0.72 with airmass within clusters); against xco2_raw − prior it is
0.24. (5) E2 beats the prior only by shrinking toward it (within-cluster slope 0.15). (6) Sensitivity:
1 ppm = 0.245 % of the k1 ratio; with a 0.32 % ratio spread the single-sounding 1σ limit is ≈ 1.3 ppm, so a
+4 ppm plume is a 3σ single-sounding feature and sub-ppm cloud anomalies need averaging that the structured
part limits.

Verdict: usable as a path diagnostic at the ppm level, not as an XCO2 estimator competitive with L2 on this
orbit; the absolute level needs a band-to-band τ accuracy budget before it means anything. The decisive
test needs land dates with real XCO2 variation (Kozienice 2021-09-06 plume, +4 ppm) and the multi-date CURC
run; the proper bar is slope ≈ 1 of (est − prior) on (xco2_raw − prior) after removing scene covariates, with
the bias reported unscaled.

## Kozienice plume (2021-09-06, orbit 38200a_ND): known-answer test on land (2026-10-30)

Steven asked for the land test locally. Orbit files (L1B, L2Met, L2CPr, Lite) downloaded to
`data/OCO2/2021/249/`; `kozienice_pseudo_xco2.py` recomputes everything for the 1057 soundings within
150 km of the plant with the study's copy of the τ step (Doppler-corrected, dry/H2O split, TOA solar as
production), fits the agreed model and compares the unscaled estimates prior × k1_S (E0), prior × k1_S/k1_O2A
(E1), prior × k1_W/k1_O2A (E1w) with the Lite xco2 (bc), xco2_raw and strong IDP. The prior is the CPr column
mean (agrees with the Lite prior to 0.03 ppm) so the estimate exists for all soundings, not only the 393 that
L2 retrieved. Outputs `results/kozienice_38200a_ND.{md,csv,png}`, `kozienice_38200a_ND_per_sounding.parquet`,
`kozienice_38200a_ND_spectra.npz` (cache), `kozienice_bl_kernel.{md,csv}`.

Background (beyond 40 km, QF0, n = 254): k1 0.985 / 0.987 / 1.003 (rSD 0.38 / 0.60 / 0.51 %); E0 bias vs bc
+0.6 ppm, rSD 2.1 ppm; E1 bias +7.0 ppm (O2A k1 0.985 with Lite fs_rel 1.6 %: the documented SIF
limitation), rSD 2.8 ppm; L2 raw rSD 1.4, bc 0.6. The plain single-τ fit on the same τ: E0 −8.9 ppm, E1 +38 ppm,
rSD 2.7 / 4.6 ppm. Soundings that L2 did not retrieve (cloudy, e.g. −12 km bin) give k1 drops of 3–6 % in all
bands and estimates of 350–390 ppm: a cloud screen is a precondition.

Plume (Lite bc > background + 2 ppm within 30 km, n = 14, 4 of them QF0): Δbc +5.3 ppm (QF0: +4.3). Agreed
model: Δk1 SCO2 +1.87 % (QF0 +1.99 %), WCO2 +1.07 % (QF0 +1.61 %), O2A −1.0 % (QF0 −0.4 %); E0 − bg +7.8 ppm,
E1 +9.7, E1w +8.0, raw +4.9. The plume is detected with the right sign in both CO2 bands and in the 5-km bins
(+2 km bin: bc 414.7, E0 417.5, E1 426.8 vs backgrounds 409.3 / 409.7 / 416.2), with amplitude 1.5–2× L2.

Why the amplitude: `kozienice_bl_kernel.py` regresses the ln T change from a 1-km boundary-layer CO2
enhancement (same column amount as a uniform increase) on the agreed design. BL kernel of k1: SCO2 1.60
[1.59, 1.60], WCO2 1.01 (24 background + 14 plume soundings; BL holds 12 % of the column). The strong band
over-responds to near-surface CO2 by 1.6× (saturated lines, pressure-broadened wings carry the τ change
into the channels that set k1); the weak band is column-mean. Observed SCO2/WCO2 contrast ratio 1.24–1.75
vs 1.58 predicted. WCO2 still exceeds bc by ≈ 1.5× (+6.6 vs +4.3 ppm on 4 QF0 soundings, 2σ of its noise),
with the L2 averaging kernel and bias correction as the remaining candidates. QF1 plume soundings
(cld < 2.6 km) also show O2A k1 −1.4 % and f up: a scene/condensed-plume effect that E1 adds to the signal.

Verdict for the pseudo-XCO2 idea: on land the agreed model detects a +5 ppm plume at single-sounding
level (noise 2–3 ppm, so ≈ 2σ per sounding, clear in 5-km bins) and is quantitatively ≈ 1.5× too strong in
WCO2 and 2× in SCO2, the latter understood (BL kernel). E0 (SCO2 alone) is unbiased in the background
(+0.6 ppm) by the coincidence noted on 29252a_GL; E1 carries the SIF-induced O2A bias (+7 ppm) and the
scene term. A pseudo-XCO2 would need (i) a cloud screen, (ii) the WCO2 band as the CO2 band (flat kernel)
or a kernel correction for SCO2, (iii) the SIF/floor problem solved before O2A can serve as the path
reference over vegetation. For the dissertation this is the right framing: the k-coefficients are path
diagnostics with a band-dependent vertical kernel, not a retrieval.

## O2A path coefficient vs the L2 surface-pressure adjustment (2026-10-30)

Question (Steven): can the O2A PPDF fit capture dp, the L2 retrieved-minus-prior surface pressure? Context:
L2 retrieves psurf from the O2A band with a 4 hPa prior sigma and uses it in xco2_raw; the B11.2 bias
correction then adds back 0.82 × dpfrac on land (0.25 ppm/hPa over ocean, ≈ 60 %), so xco2 (bc) leans on
the prior pressure. Script `o2a_dp_test.py`, outputs `results/o2a_dp_test.{md,csv,png}`,
`results/o2a_dp_kernel.csv`. x = 100·dp/psurf_prior, y = 100·(k1 − 1); both datasets of the pseudo-XCO2
tests (29252a_GL ocean, 38200a_ND Kozienice land).

- Expected slope for a pure psurf change: surface-layer kernel of the O2A k1 (air added at the surface vs
  uniformly, linearised floor model, 24 land QF0 soundings) = 1.80 [1.80, 1.81], i.e. k1 responds 1.8× to
  near-surface air because the saturated band carries the change into the pressure-broadened wings.
- Clear sky: ocean far-clear (n 1240) corr −0.47, slope −1.8; WCO2 and SCO2 k1 correlate the same way
  (−0.56, −0.41) and the SCO2/O2A ratio not at all → common-mode scene dependence, not pressure. Land
  background (n 254) corr +0.27, slope +1.1, r² 0.07. dp spread in clear sky is 0.06–0.12 % (0.6–1.2 hPa),
  at or below the L2 dp noise (≈ 1.4 hPa), while k1 scatters 0.3–0.5 % (1.7–2.8 hPa equivalent after the
  kernel) and does not average down in 24/96-sounding blocks (block corr ±0.7–0.8 with slopes 3.5–6, all
  three bands alike).
- Cloud-affected: ocean QF1 corr +0.77, slope 2.4; the 55°S deck cluster (L2 dp +1.17 %) has Δk1/Δdp =
  1.16 with SCO2 and WCO2 k1 going down. Both methods see the same path change; L2 books it as +11 hPa of
  surface pressure.
- f_O2A anti-correlates with dp where clouds are present (−0.56 QF1 ocean, −0.32 land background).

Verdict: no. In clear sky the clear-sky dp carries no detectable surface-pressure signal for either method
(the met prior is better than both), and the O2A k1 scatter is scene-correlated 3–5× larger than the dp
spread. The two agree only where dp is a cloud/path artefact. The L2 analogue of dividing by k1_O2A (E1) is
the dp adjustment, which L2 itself undoes by 60–80 % in the bias correction; this supports keeping the O2A
coefficient as a diagnostic rather than as a divisor in a pseudo-XCO2.

### Follow-up: what the dp finding changes in the pseudo-XCO2 (2026-10-30)

Script `o2a_anomaly_estimator.py`, output `results/o2a_anomaly_estimator.md`. Candidates: E0 = prior·k1_S;
E1 = prior·k1_S/k1_O2A; EA(g) = prior·k1_S/(1 + g·Δk1_O2A) with Δk1_O2A the anomaly against the running
median of the same footprint (±25 soundings, self excluded); g = 1 or 1.6/1.8 (SCO2 BL kernel over O2A
surface kernel); WCO2 variants.

| estimator | ocean far-clear bias / rSD (ppm) | ocean 0–1 km − far (bc +0.8) | land bg bias / rSD | plume − bg (bc +5.3, QF0 +4.3) |
|---|---|---|---|---|
| E0 | 0.0 / 1.92 | +1.15 | +0.6 / 1.95 | +7.8 / +8.3 |
| E1 | +5.7 / 1.69 | +7.0 | +7.0 / 3.04 | +9.7 / +10.0 |
| EA g=1 | 0.0 / 1.88 | +1.74 | +0.5 / 2.86 | +7.9 / +8.8 |
| E0 WCO2 | −2.5 / 2.27 | −0.48 | −6.1 / 2.07 | +4.5 / +6.7 |

- The local anomaly removes the E1 bias completely (SIF, τ scale) but adds the O2A noise (land Δk1_O2A rSD
  0.39 %) and does not improve the near-cloud or plume response: in clear sky there is no O2A path
  information to add, as the dp test showed.
- As a screen, |Δk1_O2A| < 1 % keeps 87 % of QF0, 72 % of QF1 and 36 % of non-retrieved land soundings
  (baseline from all neighbours); E0 rSD 2.11 vs 2.14 unscreened. A clear-neighbour baseline (MODIS
  distance) is needed for a useful flag.
- WCO2 as the CO2 band reproduces the plume amplitude (flat kernel) but carries a −6 ppm absolute offset
  over land (k1_W background 0.987) and 2.1 ppm scatter.

Conclusion: the dp finding improves the pseudo-XCO2 by subtraction. Use the prior air column (E0), as L2
effectively does after bias correction; keep the O2A coefficient as a flag/diagnostic; put the remaining
effort into the CO2-band side (WCO2 kernel, absolute τ scale, cloud screen), not into the O2A divisor.

### Kozienice re-test with the revised design (2026-10-30)

Script `kozienice_plume_retest.py`, outputs `results/kozienice_plume_retest.{md,png}`. Design: prior air
column; XCO2 anomaly = prior × Δk1_CO2band / K_BL (SCO2 1.60, WCO2 1.01); anomalies against the running
median of the same footprint (±25 soundings, self excluded) among clear neighbours (MODIS cloud distance
> 5 km); O2A anomaly |Δk1_O2A| > 1 % as path flag only. Known answer: L2 bc anomaly computed the same way
among QF0 neighbours.

| quantity | background rSD (ppm) | all plume (n 14) | QF0 plume (n 4) | unflagged plume (n 12) |
|---|---|---|---|---|
| ΔX SCO2 / 1.60 | 1.32 | +5.0 | +5.6 | +5.4 |
| ΔX WCO2 / 1.01 | 2.35 | +6.0 | +8.5 | +7.4 |
| ΔX SCO2, no kernel | 2.11 | +8.0 | +8.9 | +8.6 |
| E1-style (SCO2 − O2A) | 2.87 | +8.1 | +8.8 | +8.1 |
| L2 bc anomaly | 0.65 | +5.4 | +4.5 | +5.4 |
| L2 raw anomaly | 1.28 | +5.1 | +5.4 | +5.1 |

- The kernel-corrected SCO2 anomaly reproduces the plume amplitude (+5.0 vs +5.4 ppm) with a background
  scatter of 1.3 ppm, the same as L2 raw (1.28) and twice L2 bc (0.65); corr with the bc anomaly 0.30.
  WCO2 is noisier (2.35) and overshoots within its noise.
- The O2A flag removes 87 % of the non-retrieved (cloudy) soundings, 26 % of QF1 and 5 % of background
  QF0; 2 of the 14 plume soundings (condensed plume, Δk1_O2A −1.4 %) are flagged. Upwind bins (−20..−10 km)
  are entirely flagged, consistent with the cloud there.
- Diagnostics in the plume: Δk1_w SCO2 −4.5 to −5.9 % (H2O coefficient drops), Δf_O2A +0.3×10⁻², Δk1_O2A
  −0.6 % (all), +0.2 % (QF0).
- Caveats: the anomaly definition relies on along-track baselines, so it measures contrasts, not absolute
  XCO2 (the −6 ppm WCO2 and the band τ-scale offsets are removed by construction); 4 QF0 plume soundings
  only; the 1.60 kernel is the boundary-layer kernel for a 1-km layer, appropriate for a fresh plume.

### Absolute pseudo-XCO2 on the Kozienice orbit (2026-10-30)

Script `kozienice_absolute.py`, output `results/kozienice_absolute.md`. Unscaled; L2 as stored.

| quantity | background median | − L2 bc | − L2 raw | rSD vs bc | fp-median spread | plume (all / QF0) |
|---|---|---|---|---|---|---|
| E_S = prior·k1_S | 409.70 | +0.56 | −0.02 | 1.95 | 2.7 | 417.5 / 418.0 |
| E_S,kern (baseline + Δ/1.60) | 409.57 | +0.39 | −0.24 | 1.27 | — | 414.0 / 414.7 |
| E_W = prior·k1_W | 403.20 | −6.10 | −6.75 | 2.07 | — | 407.7 / 409.9 |
| L2 raw | 409.78 | +0.57 | 0 | 1.02 | 1.8 | 414.7 / 414.9 |
| L2 bc | 409.28 | 0 | −0.57 | 0 | 0.3 | 414.6 / 413.5 |

- The absolute SCO2 level agrees with L2 raw to 0.02 ppm on this orbit and with bc to +0.6 ppm (ocean orbit:
  0.0 ppm). WCO2 sits 6 ppm low (k1_W 0.987), a band τ-scale problem, not a path problem.
- Footprint-dependent offsets: E_S per-footprint medians spread 2.7 ppm (k1_S 0.9998–1.0063), L2 raw 1.8,
  L2 bc 0.3 after its footprint correction. A footprint-dependent τ scale (ILS) is the first absolute-level
  correction a pseudo-XCO2 would need; it is not L2-derived.
- The scatter of E_S − bc is uncorrelated with airmass, latitude, SIF, k1_O2A, f, k1_w, a1 (|r| ≤ 0.14) and
  correlates +0.60 with the free SCO2 k2. Fixing k2 at the background median makes everything worse (rSD
  3.3 ppm, plume contrast lost: −4.6 ppm), so the curvature is real scene information and k2 stays free.
- The kernel-scaled version shrinks the local anomalies (and their noise) by 1.6; its 1.27 ppm is only
  validated where the anomaly is real (plume).

### Do the agreed coefficients correlate with the L2 XCO2 anomaly better than the production ones? (2026-10-30)

Script `coef_anomaly_corr.py`, outputs `results/coef_anomaly_corr.{md,csv}` (29252a_GL; target the production
`xco2_bc_anomaly`; coefficients as stored and as local anomalies; held-out ridge R² over 10 along-track
blocks). Validation only.

| subset | best production coefficient (r) | best agreed coefficients (r) | ridge R² production set | ridge R² agreed set | agreed k1 ×3 only |
|---|---|---|---|---|---|
| QF0 cld < 1 km (n 101) | wco2 k1 −0.26, wco2 k2 −0.25 | wco2 a1 +0.38, SCO2/O2A k1 +0.34, o2a k1 −0.28, o2a f +0.25 | — | — | — |
| QF0 cld < 3 km (n 328) | wco2 k1 −0.13 | wco2 a1 +0.31, SCO2/O2A k1 +0.28, sco2 k1 +0.20 | −0.11 | 0.26 | −0.01 |
| QF0+1 cld < 3 km (n 566) | intercepts +0.16 | sco2 k1 +0.31, wco2 k2_diag −0.37, wco2 a1 +0.24 | −0.27 | 0.48 | 0.03 |
| QF0 cld < 10 km (n 1076) | — | — | −0.02 | 0.20 | 0.03 |
| QF0 all (n 3005) | sco2 k1 +0.08 | sco2 k2 +0.16, wco2 a1 +0.14 | 0.02 | 0.16 | 0.02 |

Against the raw anomaly (QF0+1 cld < 3): agreed sco2 a1 −0.71, wco2 k1_w −0.41, o2a k2_diag −0.37;
production best wco2 k2 −0.37. Reading: yes, clearly better, and the information is in the new terms
(continuum slope a1, floor f, H2O coefficient k1_w, free-k2 diagnostic, the SCO2/O2A ratio), not in k1
alone (R² ≈ 0 for either k1 triple). The production ridge has no held-out skill on this orbit.

Addendum (same day): the running-median anomaly above is NOT the pipeline/ML definition. Pipeline target
= XCO2 − mean of clear soundings (cld_dist > r) within ±0.25° latitude, all footprints, kept where the
reference std ≤ 1 ppm (src/spectral/anomaly.py); production ML targets use r = 5 km over ocean and 15 km
over land (`xco2_bc_anomaly_r05` / `_r15`), the stored `xco2_bc_anomaly` is the 10 km default. The pipeline
coefficient anomalies dk1_* use the same window via `compute_xco2_anomaly(extra_vars)`. Rerun with those
definitions (`coef_anomaly_corr_pipeline.py`, `results/coef_anomaly_corr_pipeline.md`), held-out ridge R²:

| target / subset | production set, stored → pipeline anomaly | agreed set, stored → pipeline anomaly | agreed k1 ×3 (anomaly) |
|---|---|---|---|
| r05 (ML ocean) QF0 cld < 1 (n 390) | −0.02 → 0.26 | 0.10 → 0.42 | 0.07 |
| r05 QF0 cld < 3 (n 1235) | 0.00 → 0.20 | 0.18 → 0.43 | 0.09 |
| r05 QF0 cld < 10 (n 2997) | −0.02 → 0.10 | 0.11 → 0.37 | 0.08 |
| r05 QF0 all (n 4928) | −0.01 → 0.07 | 0.02 → 0.33 | 0.07 |
| r10 QF0 cld < 3 (n 328) | −0.11 → 0.02 | 0.26 → 0.30 | 0.05 |
| r15 (ML land) QF0 cld < 3 (n 172) | −0.03 → 0.00 | 0.16 → 0.28 | 0.00 |

Conclusion unchanged and stronger on the ML target: agreed set 0.33–0.43 vs production 0.07–0.26 with
the pipeline's own anomaly construction. Best single coefficients on r05: SCO2 k1 anomaly +0.34 (cld < 1),
+0.53 (QF0+1 cld < 3), SCO2 k1_w +0.27/+0.47, WCO2 a1 +0.28, WCO2 free-k2 diagnostic −0.27.

### Third-order term to constrain the sign of k2? (2026-10-30)

Steven's question: add a k3 term, free or tied to (k1, k2) by the gamma relation k3 = 2k2²/k1, so that the
k1–k2–k3 relationships fix the sign of k2. Script `cubic_term_test.py`, output `results/cubic_term_test.md`
(agreed CO2-band design; Kozienice background/plume and 29252a_GL far-clear subsample).

| band / set | A agreed k1, spread, k2 | B free k3: k1, spread, k2 (<0 %), k3 | C gamma-tied: k1, spread, k2 (at 0 %), k3 | D k2 ≥ 0: spread, at 0 % |
|---|---|---|---|---|
| WCO2 land bg | 0.9868, 0.52 %, 0 (fixed) | 0.903, 3.95 %, −0.58 (96 %), −1.48 | 0.9868, 0.52 %, 0 (99 %), 0 | 0.52 %, 96 % |
| WCO2 ocean far | 0.9912, 0.25 %, 0 (fixed) | 0.949, 1.89 %, −0.28 (100 %), −0.76 | 0.9912, 0.58 %, 0 (72 %), 0 | 0.58 %, 65 % |
| SCO2 land bg | 1.0026, 0.48 %, +0.014 | 0.978, 0.90 %, −0.025 (98 %), −0.023 | 1.0029, 0.50 %, +0.015 (0 %), +0.0004 | 0.47 %, 0 % |
| SCO2 ocean far | 0.9988, 0.27 %, +0.011 | 0.972, 0.49 %, −0.030 (100 %), −0.023 | 0.9989, 0.28 %, +0.012 (0 %), +0.0004 | 0.27 %, 0 % |

Plume Δk1 (all / QF0): WCO2 A +1.07/+1.61, B +3.6/+8.9, C +1.32/+1.61; SCO2 A +1.87/+1.99, B +2.19/+2.20,
C +2.00/+2.05. rms changes by < 2 % in every case.

- A free cubic does not constrain k2, it destabilises it: τ² and τ³ are nearly collinear over the band's τ
  range, so B drives k2 and k3 negative in both bands (negative skew, unphysical), lowers k1 by 2.5 % (SCO2)
  to 8 % (WCO2) and doubles to quadruples the spread, for no rms gain. Same as the 2026-10-28 order-3
  result on the production setup (k2 at the zero bound in 97 %).
- The gamma-tied cubic is physically negligible in SCO2: k3 = 2k2²/k1 ≈ 4×10⁻⁴ (κ = k1²/k2 ≈ 70, so the
  series converges fast); k1, k2 and the plume contrast move by < 0.1 %. In WCO2 it collapses onto the k2 = 0
  bound (99 % land, 72 % ocean), identical to a plain k2 ≥ 0 bound, which in turn reproduces the agreed k1
  but with a larger spread where the bound is sometimes inactive (ocean 0.58 vs 0.25 %).
- The sign of k2 is set by the shape of ln T against τ at the band's high-τ channels, and in WCO2 that shape is
  concave-down: a misfit (channel-population streaks, continuum/H2O structure), not path statistics. No
  expansion order changes it; the WCO2 forward model does.

Decision: keep the agreed model. k2 free in SCO2 (a k2 ≥ 0 bound is never active there), k2 = 0 in WCO2, no
k3 anywhere; store nothing new.

## WCO2 spread at low SOD (2026-10-30, Opus agent)

Question: why does WCO2 ln T fan out at SOD < 0.1 (≈ ±0.05 against noise 0.005) in the revised fit, and can it be
reduced? Scripts `wco2_lowsod_{common,solar,analysis,kozienice,fig}.py`; report `results/wco2_lowsod_report.md`;
tables `results/wco2_lowsod_summary_29252a_GL.md`, `results/wco2_lowsod_kozienice_38200a_ND.md`; figures
`results/wco2_lowsod_patterns_29252a_GL.png`, `results/wco2_lowsod_example_2020010100214501.png`.

Cause: the TOA solar spectrum. Production shifts the solar lines by v_sol only (`fp_abs_coeff.py`
`beta_total = v_solar / C_LIGHT`; the v_inst term is commented out). The spacecraft–surface velocity (±5.3 km/s on
glint, up to 0.9 px) is missing, as it was for the telluric τ. At low SOD the solar lines are the only spectral
structure, so the misregistration shows there. Evidence (far-clear 29252a_GL, n = 1240): 99 % of the residual at
SOD < 0.1 is one fixed pattern per (footprint, v_inst cluster); the two cluster patterns anticorrelate (r −0.6 to −0.7),
and their difference matches −Δv·∂lnS/∂v at r = 0.99 in every footprint; a free per-sounding shift scale gives α̂ = 0.99.
This also explains the "unexplained" WCO2 SVD mode of 2026-10-29 (the solar test there used line depth, not ∂lnS/∂λ).

| WCO2 candidate (29252a_GL far-clear) | rms SOD<0.1 | k1 median | spread % | within-cluster % | v-cluster step % | Δk1 0–1 km % (SNR) |
|---|---|---|---|---|---|---|
| agreed model | 0.0193 | 0.9914 | 0.253 | 0.078 | −0.86 | −0.098 (−1.3) |
| solar lines at λ(1 + v_sol/c − v_inst/c), no fit parameter | 0.0048 | 0.9876 | 0.119 | 0.079 | +0.15 | −0.099 (−1.3) |
| + L1B bad_sample_list ≠ 0 dropped | 0.0036 | 0.9902 | 0.127 | 0.072 | +0.17 | −0.061 (−0.9) |
| + per-fp fixed pattern subtracted (cross-fitted) | 0.0026 | 0.9885 | 0.089 | 0.079 | +0.09 | −0.066 (−0.9) |
| solar + Legendre P2, P3 | 0.0043 | 0.9923 | 0.127 | 0.080 | +0.15 | −0.030 (−0.4) |
| solar + ∂τ/∂pix + ∂τ/∂lnw | 0.0047 | 0.9845 | 0.174 | 0.102 | +0.23 | −0.096 (−1.0) |

- The solar error biased WCO2 k1 by ≈ 0.17 % per km/s of v_inst, i.e. a latitude/geometry-dependent step along glint
  orbits. Sounding-to-sounding scatter and near-cloud contrasts are unchanged (v_inst is smooth along track). O2A rms
  0.029 → 0.025, SCO2 0.0167 → 0.0158 with the same fix.
- After the fix, L1B-flagged channels are the outliers: rms at SOD < 0.1 is 0.010 (isolated bad pixels) and 0.0086
  (edge channels 0–89) against 0.0037 for unflagged channels. Dropping them lowers the within-cluster spread 0.079 → 0.072 %
  (ocean) and the background rSD 0.60 → 0.40 % (land 38200a_ND). The k1 level moves +0.26 / +0.48 %, mostly from the
  edge channels.
- What remains is a detector-fixed per-footprint pattern (rms 0.003–0.006). It is the same in both velocity clusters,
  so it is not tied to the line frame. It is stable across orbits (land 2021 vs ocean 2020 pattern r = 0.80–0.89), and
  it is uncorrelated with solar depth, ∂lnS/∂v, f_H2O, ∂τ/∂pix and ∂τ/∂lnw (|r| ≤ 0.2). Subtracting it lowers the rms only;
  no k1 metric moves.
- The WCO2–SCO2 k1 correlation (0.51, land) is carried by the edge channels: their k1 lever tracks the mean
  edge-channel residual (r = 0.9997) and correlates +0.61 with SCO2 k1. On ocean, the within-cluster correlation is ≈ 0
  in every variant. It is not a usable quality criterion on these orbits.

Recommendation: (1) add the solar −v_inst/c shift to the τ/TOA step under the same `--telluric-doppler` switch
(production change, pending Steven); (2) exclude `bad_sample_list ≠ 0` channels (changes the k1 level; Steven's call;
O2A/SCO2 effect not yet checked); (3) do not adopt the fixed-pattern subtraction, Legendre continuum or ILS columns for
k1. Open: source of the detector-fixed pattern (per-pixel gain or ILS shape?); confirmation on an orbit with
continuous v_inst; CH4 not tested.

### Addendum: O2A and SCO2, and Kozienice plume check of the bad-sample mask (2026-10-30)

Script `wco2_lowsod_allbands.py`; tables `results/wco2_lowsod_allbands_{29252a_GL,kozienice}.md`.

| band, 29252a_GL far-clear | rms SOD<0.1 | rms all | k1 median | spread % | within % | step % | Δk1 0–1 km % (SNR) |
|---|---|---|---|---|---|---|---|
| O2A agreed | 0.0183 | 0.0293 | 0.9873 | 0.169 | 0.130 | −0.19 | −0.30 (−2.7) |
| O2A solar fix | 0.0040 | 0.0249 | 0.9872 | 0.158 | 0.130 | −0.16 | −0.30 (−2.7) |
| O2A solar + drop bad | 0.0036 | 0.0270 | 0.9888 | 0.166 | 0.140 | −0.16 | −0.27 (−2.2) |
| SCO2 agreed | 0.0059 | 0.0167 | 0.9987 | 0.279 | 0.242 | −0.22 | −0.11 (−0.4) |
| SCO2 solar fix | 0.0055 | 0.0158 | 0.9987 | 0.259 | 0.244 | −0.14 | −0.11 (−0.4) |
| SCO2 solar + drop bad | 0.0054 | 0.0161 | 1.0022 | 0.269 | 0.265 | −0.04 | −0.03 (−0.1) |

O2A f 0.0094 in all three; free-k2 diagnostic < 0 in 95 / 96 / 91 %. SCO2 k2 +0.011 / +0.011 / +0.013; k1_w
0.983 / 0.983 / 0.986. Flagged channels at SOD ≥ 1 per footprint: O2A 0.1 (no line cores, max 1.14), WCO2 0,
SCO2 13.8 (the SCO2 edge channels lie at SOD 0.1–1.9).

Kozienice (land nadir, no solar error), dropping the bad ≠ 0 channels: k1 median +0.20 / +0.49 / +0.24 % and
background rSD 0.38 → 0.38 / 0.60 → 0.40 / 0.51 → 0.44 % (O2A / WCO2 / SCO2). Plume Δk1 (all 14 / QF0 4)
falls: WCO2 +1.48/+2.10 → +1.00/+1.27 %, SCO2 +1.95/+2.18 → +1.41/+1.41 %. All of the change comes from the edge
channels (value 2); value 1 is neutral.

Reading: the solar fix is safe in every band. It reduces rms and the latitude step and moves nothing else. The
edge-channel exclusion is not a free noise reduction: it lowers the scatter and the plume response by similar
amounts. It is a continuum-model choice and is deferred. This supersedes the earlier "exclude bad ≠ 0" recommendation.

### Addendum: free k2 after the solar fix? (2026-10-30)

Script `wco2_lowsod_k2_after_fix.py`, output `results/wco2_lowsod_k2_after_fix.md` (29252a_GL far-clear, solar-corrected
ln T; Kozienice background for WCO2).

| band / variant | k1 median | spread % | within % | k2 median (frac < 0 or at bound) | rms | Δk1 0–1 km % (SNR) |
|---|---|---|---|---|---|---|
| WCO2 k2 = 0 (current) | 0.9876 | 0.119 | 0.079 | 0 | 0.0058 | −0.099 (−1.3) |
| WCO2 free k2 | 0.9607 | 0.353 | 0.240 | −0.099 (1.00) | 0.0056 | −0.183 (−0.8) |
| WCO2 k2 ≥ 0 | 0.9876 | 0.119 | 0.079 | 0 (1.00 at bound) | 0.0058 | −0.099 (−1.3) |
| O2A floor, k2 = 0 (current) | 0.9872 | 0.158 | 0.130 | 0; f 0.0094 | 0.0249 | −0.300 (−2.7) |
| O2A floor + free k2 | 0.9806 | 0.248 | 0.176 | −0.0055 (0.96); f 0.0097 | 0.0249 | −0.189 (−1.2) |
| O2A no floor, order 2 | 1.0800 | 0.886 | 0.185 | +0.099 (0.00) | 0.0370 | −0.293 (−2.1) |
| O2A no floor, order 2 + k3 | 0.9442 | 0.264 | 0.231 | −0.056 (1.00), k3 −0.063 | 0.0251 | −0.096 (−0.4) |

WCO2 free k2 before/after the fix: −0.033 (80 % < 0) → −0.099 (100 %). r(k2, k1) is +1.00/+0.94 raw and +0.94/+0.92
within cluster; r(k2, a1) is −0.35/−0.67. Kozienice: free k2 −0.155 (100 % < 0), r(k2, k1) +0.98, rSD 0.60 → 1.67 %.
Verdict: the solar fix does not make k2 physical in either band. Keep WCO2 k2 = 0 and the O2A floor with k2 = 0
(free k2 as a diagnostic only).

## Origin of the O2A floor: scattered light, not an instrument offset (CURC, 2026-11-05)

Supersedes the 2026-10-30 notes that called the O2A floor stray light / a zero-level offset, rejected Rayleigh
via the O2A/SCO2 floor ratio, and found no SZA/airmass dependence (single orbit, narrow geometry).

- No-floor O2A fit (6000 soundings, 2018-07-01, two glint orbits): k2 +0.061 land / +0.110 water, never negative;
  rms 0.0336 vs 0.0250; k1 +8 %; residual is a wave in τ. r(k2, τ p95) +0.89 land / +0.79 water (partial, f fixed,
  0.96 / 0.86), so no-floor k2 is set by floor size and sampled τ range. Floor + free k2: median +0.001.
- Saturation argument fails: k1·τ > 5.6 essentially never; O2A has ~2× (not 10×) SCO2's fraction of T < 5 % channels.
- Floors are start-independent (SCO2 f 0.0019 / 0.0021, O2A 0.0080 / 0.0121 land / water). SCO2 floor-only fit
  leaves the same residual as k2 (degenerate).
- Real-data geometry test (6 dates 2016–2020, QF0, cloud distance > 10 km, 15k land + 15k water), X = τ_R P(Θ)/(4μμ0),
  τ_R = 0.0255: spread of f·A·μ0 (offset) vs f·A/X (Rayleigh) across μ0 quintiles 52 / 16 % land, 47 / 16 % water;
  across obs modes (land) 33 / 7 %; per-sounding IQR/median 33 / 16 %. f·A/X = 0.22 land, 0.11 water. f·A vs AOD
  Spearman +0.4. Far-clear land: f vs albedo −0.84, f vs fs_rel ≈ 0.
- Slab Rayleigh-off (1e6 photons × 3, 3D, far columns): k1 1.001 / 1.005, f 0 / 1e-5, no-floor k2 0.002 / 0.004
  - Local reproduction 2026-11-05 (run_slab.py --variant norayleigh, results/rt_slab_sim/*_norayleigh.h5):
    far k1 1.015 / 1.017, f 3e-5 / ~0, no-floor k2 0.013 / -0.004, dark continuum -24 %. At 1e6 the
    run-to-run k1 spread is 1-4e-2 (CURC "agree to 2e-4" does not hold at 1e6). Manuscript now says
    "within 0.02 of 1" and quotes the 0.04 run spread. A 1e7 rerun (~30-50 min) would cut noise ~3x.
- Correction size by QF x cloud distance: workspace/correction_size_by_qf.py ->
  results_doppler/model_comparison/xgb/fm1620k5v2xt41_eval_dates/correction_size/, SI Table S14.
  (A 0.03 / 0.30). Rayleigh on: f 0.083 / 0.011, f·γ = 0.0034 on both, f·A/X 0.22 / 0.29.
- Consequence: fit form unchanged. Flat floor ignores the partial O2 absorption of scattered light, so O2A k1_am reads
  low where scattered light is a large share (slab 0.80 at A 0.03, 0.97 at 0.30), incl. cloud shadows.
- Manuscript (JQSRT) updated 2026-11-05: Sect. 2 model paragraph and k1 caveat, Sect. 3 Rayleigh-off sentence,
  SI S6 Rayleigh-off paragraph, new SI S1.6 "Origin of the O2 A-band floor".

# PPDF-based direct XCO2 estimate — first feasibility test

**Started:** 2026-10-29. Side study, outside the JQSRT/AMT paper scope.

## Status (2026-11-01): paused — read this first

**Goal (user, 2026-11-01).** Find out whether the per-band cumulant fits can give a fast
far-cloud (> 20 km, QF0) XCO2 retrieval that replaces a full-physics forward model. The
user wants a **non-ML, physics-based** estimator. A trained path model (round 2, HGB) works
but was set aside as "another XCO2 emulator". TCCON is out of scope for now.

**Where it stands.**

| round | approach | ML? | far-cloud result | status |
|---|---|---|---|---|
| 1 | per-date ridge path model (E2) on `_am` k1 | ridge | 1.3–1.5 ppm vs bc, prior + offset null 1.08 | does not beat the prior |
| 2 | dates-pooled path model (HGB), S+W combined, training-date prior blend | yes | **0.78 ppm total vs prior 1.07**; beats the prior on all held-out years | works, rejected as non-physical |
| 3 | single-layer PPDF-S-like physical model fitted to cumulants (LM/OE) | no | ties the prior on 2018-07-01 (ocean 0.93 vs 0.92; land 1.54 vs 1.59) | **fails the O2A truth check** |
| 4 | layer-free two-kernel (shortening/lengthening) path transfer across bands | no | 2–5× worse than the prior | failed |
| 5 | per-band direct-path atom (δ at l = 1) + gamma diffuse part, no cross-band link | no | O2A ŝ rsd 1.0 % (bar 0.3 %) | failed |
| 3-truth | round 3 with s_p freed (prior σ 5 %) | no | O2A ŝ_p rsd 0.6 % land / 0.9 % ocean; CO2 worse than the prior | failed |

**What was learned (keep these).**
1. **Degeneracy.** With T = ∫p_b(l)·exp(−g·τ·l)dl, g enters only as g·l. So within one band
   g is exactly degenerate with the PPDF's scale (k_n = gⁿκ_n). Any g estimate needs either
   a cross-band path link or an absolute path anchor. The user's framework assumes neither,
   and O2A, WCO2 and SCO2 photons need not share paths.
2. **The O2A truth check** is the decisive real-data test with no bc or TCCON: retrieve the
   O2 scale freely and require rsd(ŝ − 1) ≲ 0.3 % (≈ 1 ppm) on far-cloud soundings. Run it
   first on any new path model. A tight surface-pressure prior hides the failure
   (round 3: ±0.08 % with σ = 0.2 %, against 0.6–0.9 % freed).
3. **The limit is the path physics of the forward model, not τ resolution.** The round-3
   O2A error is the same on the slab (no sub-channel saturation, no spectroscopy, no
   ZLO/SIF) as on real data. So recomputing τ at sub-channel resolution is not the first fix.
4. **Real-data systematics seen along the way:**
   - SCO2/WCO2 dry k1 ratio offset of ~1 % (median 1.009–1.012), scene scatter 0.55–0.7 %,
     barely correlated (0.09–0.15) with the water-k1 ratio;
   - O2A `_am` residual rms 0.024, far above noise;
   - O2A fill f is mis-modelled by effective τ (predicted 0.023–0.038 vs observed 0.008–0.013)
     and may include SIF/ZLO;
   - observed O2A/WCO2 k2diag ≤ 0, which no photon-path mixture can produce.
5. **Noise is not the limit.** Formal σ from the fits (0.3–1 ppm) is 2–10× below the actual
   errors in every round, i.e. model bias dominates.
6. **Real far-cloud O2A k1 varies ~1 % sounding to sounding.** That is the path variation a
   non-ML method must explain to 0.25 %.

**If resumed, candidate directions (none started).**
- A forward model that passes the O2A truth check on the slab first: multi-layer / two-stream
  scattering, with the aerosol vertical profile as state. Accept only if slab and real O2A
  rsd ≤ 0.3 %.
- Diagnose the SCO2/WCO2 k1 inconsistency (τ spectroscopy, temperature, band-specific fit
  terms, ZLO/ILS). This matters for any two-band method.
- Use the O2A path error as a screen. Round 3 over ocean at |ŝ_p − 1| < 0.3 % gave
  0.71 vs prior 1.01 ppm, but on 28 % of soundings and one date only.
- Expand round 2 (HGB) to 116 dates with TCCON, if a trained path model becomes acceptable.

**File map.**
- Code (all in this folder; nothing under `src/` was changed, `src/spectral/cumulant_fit.py` is imported):
  - `estimators.py`, `run_test.py`: round 1;
  - `estimators_v2.py`, `run_v2.py`: round 2;
  - `slab_mc.py`: slab Monte Carlo scene generator and identifiability test;
  - `phys_ppdf.py`, `run_phys.py`: round 3 and its truth check (`--sp-sd`, `--exclude-obs o2a_f_am`, `--out-name`);
  - `round3truth.py`: truth-check tables;
  - `path_transfer.py`: round 4;
  - `atom_fit.py`: round 5.
- Results:
  - `results_2016_2020/`: round 1;
  - `results_2016_2020_v2/`: round 2;
  - `results_phys/`: rounds 3–5 and truth check (`summary_phys.md`, `round4_*`, `round5_*`, `round3truth_*`).
- Slurm wrappers in `jobs/` (Blanca preemptable, 32 cores, logs in `jobs/logs/`, git-ignored). Run from this folder:
  - round 3: `sbatch -M blanca jobs/phys.sh slab|real [--date 2018-07-01] [--sp-sd 0.05] [--exclude-obs o2a_f_am] [--out-name NAME]`
  - round 4: `sbatch -M blanca jobs/pt.sh slabgen|slab|real`
  - round 5: `sbatch -M blanca jobs/atom.sh slab|real --bands o2a [wco2 sco2]`
  - round 2: `jobs/v2_perdate.sh` → `v2_pooled.sh` (afterok) → `v2_opblend.sh` (afterok); `v2_slab.sh`
  - round 1: `jobs/run_ppdf.sh`
- Inputs:
  - `results_doppler/csv_collection/combined_2016_2020_dates.parquet`, plus per-date parquets;
  - τ files `results_doppler/<date>/<orbit>/fp_tau_combined.h5` (real runs need these, keep them);
  - real ln T from L1B with the τ-file `toa_sol`.
- Real test set: 2018-07-01, 1000 ocean + 1000 land far-cloud QF0 soundings, fixed seed.

## Hypothesis

The per-band cumulant fit returns `k1_b = <l'>_b · g_b` and `k2_b = var(l')_b · g_b²`,
where `g_b` is the gas scale factor of the band's absorber (air for O2A, CO2 for
WCO2/SCO2) relative to the prior. If the band path enhancement `<l'>_b` can be
predicted from gas-free observables — O2A `k1`, the shape ratios `k2/k1²` (invariant
to `g_b`), the continuum intercepts, and geometry — then
`f = k1_S / <l'>_S,pred` is the CO2 scale factor and `XCO2 = XCO2_prior · f`.

The assumption `<l'>_W = <l'>_S` is explicitly **not** made; the two CO2 bands are
treated as two estimates whose disagreement is itself a result.

## What is used

Per-date feature parquets `results/csv_collection/combined_<date>_all_orbits.parquet`
(no-SG cumulants `*_k1_nosg`, `*_k2_nosg`, `*_intercept_nosg`; `xco2_apriori`,
`xco2_bc`, `xco2_bc_anomaly`, `xco2_strong_idp`, `cld_dist_km`, `sfc_type`,
`xco2_qf`, geometry). No per-species optical depth is available, so the H2O
lever (WCO2 → H2O scale) is deferred.

## Estimators

Calibrated per date × surface on far-cloud (> 20 km) QF0 footprints where `f = 1`
is assumed, with 2-fold hold-out by orbit so far-cloud scores are out-of-sample.

| id | CO2 factor `f` | what it tests |
|---|---|---|
| E0 | `k1_S / mean_far(k1_S)` | naive SOD rescaling |
| E1 | `(k1_S/k1_O2A) / mean_far(k1_S/k1_O2A)` | proxy assumption `<l'>_S ∝ <l'>_O2A` |
| E2 | `k1_S / ridge(k1_S | k1_O2A, k2/k1² ×3, exp-intercepts ×3, geometry)` | k1+k2 scheme with an empirical path model |
| E2w | same with WCO2 as the CO2 band | band consistency (`<l'>_W` vs `<l'>_S`) |
| B-IDP | `xco2_strong_idp − mean_far(idp − bc)` | ideal non-scattering line-by-line benchmark |

## Pass/fail bars (fixed before running)

- **Retrieval bar:** far-cloud held-out SD(est − xco2_bc) ≤ 2 ppm for any estimator.
- **Diagnostic bar:** near-cloud (< 3 km) corr(est − bc, xco2_bc_anomaly) ≥ 0.3 with SD ≤ 5 ppm.
- **Plume bar:** Kozienice 2021-09-06 core-minus-background enhancement ≥ +2 ppm in E2
  (xco2_bc shows ≈ +4 ppm).

Expected from the 2018-07-10 pre-check: E0/E1 SD 30–60 ppm; E2 lower but far above
2 ppm; B-IDP ≈ 2 ppm far, worse near. Whatever comes out is recorded below.

## Dates (first pass)

`2018-07-10` mixed · `2021-09-06` Kozienice plume (land) · `2023-06-26` Westar
plume (land) · `2019-06-22` ship clear-sky control (ocean) · `2017-10-09` ATom
far-cloud control (ocean).

## How to run

```
python idea_test/ppdf_direct_xco2/run_test.py \
    --dates 2018-07-10 2021-09-06 2023-06-26 2019-06-22 2017-10-09
```
Outputs: `results/metrics_<date>.csv`, `results/summary.md`,
`results/hist_<date>.png`, `results/plume_<plant>_<date>.png`.

## Result log

(appended after each run)

### Result 2026-10-29 — first pass, 5 dates (`results/summary.md`, `metrics_all.csv`)

Robust SD (1.4826·MAD) of estimate − `xco2_bc`, median over date × surface:

| estimator | far held-out | near < 3 km | near corr with anomaly |
|---|---|---|---|
| E0 naive SCO2 | 30.0 ppm | 35.9 ppm | −0.04 |
| E1 O2A-normalized | 32.0 ppm | 35.7 ppm | −0.06 |
| E2 k1+k2 path model (SCO2) | 20.0 ppm | 28.8 ppm | −0.09 |
| E2w k1+k2 path model (WCO2) | 8.8 ppm | 12.1 ppm | −0.06 |
| B-IDP benchmark | 1.8 ppm | 3.6 ppm | +0.13 |

- **Retrieval bar (≤ 2 ppm far): FAILED** by every cumulant estimator (best E2w
  5–24 ppm per date×surface). Only the line-by-line IDP benchmark reaches ~2 ppm,
  and it degrades to 3–6 ppm near clouds — the band-dependent path effect that
  full physics exists to absorb.
- **Diagnostic bar (near corr ≥ 0.3): FAILED** — |corr| ≤ 0.2 everywhere; the
  residuals carry no cloud-anomaly information.
- **Plume bar: FAILED** — Kozienice 2021-09-06: E2 core−bg −0.5 ppm with ±25 ppm
  footprint scatter (E2w shows a +4 ppm bump at 0–10 km but inside its ±5 ppm
  noise); Westar 2023-06-26: E2 +7.5 ppm where `xco2_bc` has +0.4 — noise.
- O2A normalization (E1) does not improve on E0 anywhere: `<l'>_O2A` is not a
  usable proxy for `<l'>_SCO2`, as argued.
- Adding k2 shape + intercepts (E2) halves the scatter vs E1 and WCO2 (E2w) halves
  it again, so the empirical path model does capture some scene dependence — but
  the floor is still 5–10 ppm on ocean and 10–24 ppm on land, and 10–25 % of
  footprints fall outside |f − 1| ≤ 0.2 (`frac_valid`). E2 − E2w disagree by
  ~10–30 ppm robust SD: `<l'>_W ≠ <l'>_S` confirmed in numbers.
- Caveats: 2017-10-09 land has too few orbits for the 2-fold path model (E2 rows
  missing / E2w 70 valid); the plume core−bg metric uses a symmetric ±10 km core,
  so `xco2_bc`'s one-sided Kozienice peak reads +0.8 rather than the +4 ppm max.

**Verdict:** with the current fit products (no-SG k1/k2 from the (7,3,7)
cumulant fit on total-gas τ) a direct XCO2 estimate is 5–30 ppm, i.e. 10–100×
short of the 1 ppm target, and carries no near-cloud information. The idea is
closed at this level. Reopening would require, in order: σ(k1)/σ(k2) export to
separate fit noise from scene variation; per-species τ (CO2/H2O split) in the
fit; a slab-MC identifiability test of whether (shape ×3, O2A k1, intercepts)
can pin `<l'>_S` to 0.25 % even noise-free. None of this is scheduled.

### Result 2026-11-01 — 2016–2020 rerun on fit schema doppler_am_t2_v2 (`results_2016_2020/`)

**What changed.** `estimators.py` / `run_test.py` ported from the retired `*_nosg`
columns to the v2 schema (first-pass code is in git history). Estimator definitions,
calibration (per date × surface, far-cloud > 20 km QF0, 2-fold by orbit), validity cuts
(|f − 1| ≤ 0.2, path prediction 0.5–1.5 × median) and bars are unchanged. Each of
E0/E1/E2/E2w is evaluated on four fits: `_o2` (order-2 total τ, the nearest port of
`*_nosg`), `_o3`, `_t2` (two-τ: dry-gas `k1_dry_t2`, shape `k2_t2/k1_dry_t2²`, `exp(c_t2)`)
and `_am` (agreed model: dry `k1_am`, `k2diag_am/k1_am²`, `exp(c_am)`, plus `a1_am` ×3,
`o2a_f_am`, `sco2_a2_am`). New: `*_ivw` (ridge weighted by 1/σ(k1_target)²),
`*_scr` (drop rows with σ(k1)/k1 above the calib-far 75th percentile), `E2h*`
(adds WCO2/SCO2 water k1 as path predictors), and the null baseline **`N0_prior`**
= prior + calib-far mean(bc − prior). Extra metrics: `sig_fit` (σ(k1)/k1 × prior),
pull SD, `corr_bcdev` = corr(est − prior, bc − prior). No `ref_*`/`r15_*`/`r05_*`
statistic enters any estimator. Data: all 116 dates of `combined_2016_2020_dates.parquet`
(231 date × surface cells) plus first-pass dates 2018-07-10, 2019-06-22, 2021-09-06,
2023-06-26 from per-date files (2017-10-09 has no v2 file). Run:
`run_test.py --all-combined --dates 2018-07-10 2021-09-06 2023-06-26 2019-06-22 --workers 10`
(Blanca job, ~6 min).

Median over the 231 cells (robust SD of est − `xco2_bc`, ppm; first pass = 5 dates):

| estimator | first pass far / near / corr | far held-out | near < 3 km | near corr | far fvalid | far rsd ÷ N0 |
|---|---|---|---|---|---|---|
| E0 (`_o2`) | 30.0 / 35.9 / −0.04 | 7.8 | 10.2 | +0.09 | 1.00 | 7.7 |
| E1 (`_o2`) | 32.0 / 35.7 / −0.06 | 10.6 | 12.7 | −0.01 | 1.00 | 9.9 |
| E2 (`_o2`) | 20.0 / 28.8 / −0.09 | 2.47 | 4.17 | +0.02 | 1.00 | 2.7 |
| E2w (`_o2`) | 8.8 / 12.1 / −0.06 | 1.44 | 2.89 | −0.07 | 1.00 | 1.5 |
| E0_t2 (dry k1) | — | 2.86 | 5.47 | +0.04 | 1.00 | 2.5 |
| E2_t2 / E2w_t2 | — | 1.64 / 1.52 | 3.48 / 2.42 | −0.02 / −0.08 | 1.00 | 1.7 / 1.5 |
| E0_am | — | 1.81 | 3.48 | +0.24 | 1.00 | 1.7 |
| E2_am / E2w_am | — | 1.45 / 1.31 | 2.73 / 2.31 | −0.00 / −0.18 | 1.00 | 1.5 / 1.3 |
| E2_am_ivw / E2w_am_ivw | — | 1.46 / 1.33 | 2.67 / 2.46 | −0.01 / −0.18 | 1.00 | 1.5 / 1.3 |
| E2_am_scr / E2w_am_scr | — | 1.30 / 1.30 | 1.85 / 2.17 | −0.01 / −0.17 | 0.71 / 0.74 (near 0.34 / 0.46) | 1.3 / 1.3 |
| E2h_am (water lever) | — | 1.43 | 2.63 | −0.01 | 1.00 | 1.4 |
| B-IDP | 1.8 / 3.6 / +0.13 | 1.76 | 3.16 | +0.12 | 1.00 | 1.7 |
| **N0_prior (null)** | — | **1.08** | **1.71** | −0.75 | 1.00 | 1.0 |

- **The k1 products are 5–15× better than in the first pass.** On the same four
  first-pass dates E2/E2w (`_o2`) give 2.8 / 2.2 ppm against 20 / 8.8 then, while
  B-IDP is unchanged (2.2 vs 1.8), so the gain is in the fit (Doppler τ, low-order
  no-SG fit), not in the data. Removing H2O from k1 is the largest single step:
  naive SCO2 E0 goes 7.8 (total τ) → 2.9 (`k1_dry_t2`) → 1.8 ppm (`k1_am`). Order 3 is
  worse than order 2 everywhere.
- **σ products:** fit noise is not the limit. For `_am` the formal σ is 0.6–0.75 ppm
  and the pull SD is 1.4–1.6 over ocean and 2.3–4.6 over land. For WCO2 `_o2`/`_t2` the
  formal σ (2–5 ppm) overstates the scatter (pull ≈ 0.55): the misfit is systematic.
  Inverse-variance weighting changes nothing (±0.02 ppm). σ-screening helps a little
  (E2_am 1.45 → 1.30 far, 2.73 → 1.85 near), but it keeps 71 % of far footprints and
  only 34 % of near ones, so most of the near gain comes from rejecting cloudy scenes.
  The water lever (E2h) adds nothing.
- **Ocean vs land:** over ocean the path models reach about the null: E2w_am 0.91,
  E2w_am_scr 0.89 ppm against N0 0.90 (ratio 1.00–1.05; near-cloud ratio 0.96–1.06),
  and 97–99 % of ocean cells are ≤ 2 ppm. Over land they fail: E2_am 2.98 / E2w_am 2.27
  against N0 1.25, with 6–35 % of rows invalid. That is held-out-orbit extrapolation of
  the ridge path model. The best land estimator is the plain E0_am (1.75 ppm, ratio 1.36).
  The SCO2 − WCO2 band disagreement is now 1.1 ppm (ocean) / 3.1 ppm (land) for `_am`,
  against 10–30 ppm in the first pass.
- **Per year** (far rsd median): E2_am 1.33–1.63 and E2w_am 1.25–1.45 for 2016–2020,
  against N0 0.93–1.18. The year-to-year spread is small; the cell-to-cell spread
  (q25–q75 ≈ 1.0–3) is dominated by land.

**Bars.**
- **Retrieval (far ≤ 2 ppm): nominally PASSED** by 21 estimators. Every `_t2`/`_am` E2
  variant, E0_am and E2w_o2 qualify, and so does B-IDP.
- **The bar no longer discriminates.** The null N0_prior (no spectral information) scores
  1.08 ppm, and **no estimator beats it** (best median ratio 1.27, E2w_am / E2w_am_scr).
  The estimates still add more noise than they carry signal. They do carry some:
  corr(est − prior, bc − prior) = 0.4–0.5 for `_am`/`_t2` (0.05 for E0_o2), i.e. ~20 %
  of the variance of the retrieved departure from the prior.
- **Diagnostic (near corr ≥ 0.3, SD ≤ 5): FAILED.** The best is E0_am at +0.24 (its ocean
  subset alone, +0.31 at 2.9 ppm, would pass). The path models sit at −0.2 to 0, between
  the null (−0.75, which does not track the anomaly at all) and "tracks bc".
- **Plume (pre-registered: Kozienice E2 symmetric core − bg ≥ +2): FAILED.** E2_am gives
  −0.32 and E2_o2 −3.7; `xco2_bc` gives +0.83. E2_am_scr (+2.8) and E0_am_scr (+3.5)
  pass, but on only 9 core soundings, so they are not credible. A post-hoc one-sided
  check (0 to +10 km, where `xco2_bc` shows +4.7) gives E0_am +4.4, E2w_am +3.9, and
  E2_am +0.8 against a background rsd of 1–2 ppm. Westar 2023-06-26: E2_am +0.15,
  E2w_am +0.45 against bc +0.40, so the first pass's spurious +7.5 is gone.

**Verdict:** **reopened, not yet feasible.** The two-τ dry-gas k1 and the agreed model
take the cumulant estimate from 5–30 ppm to ~1.3–1.8 ppm (≈ 0.9 over ocean). That is
within 20–30 % of the prior-plus-offset null and positively correlated with the
retrieved XCO2 departure. But no estimator beats the prior, the land path model breaks
on held-out orbits, and near-cloud information is weak. The σ products show the
remaining error is scene/path-model variation (pull ≈ 2), not fit noise. The fixed
2 ppm retrieval bar is now too loose. Any next round should require far rsd ÷ N0 < 1
(or, equivalently, a gain in explained variance of bc − prior). Next steps, in order:
(1) a land path model that generalises across orbits (surface-class or albedo-binned
calibration, or a pooled multi-date model with date-held-out CV instead of per-date
ridge); (2) a combined SCO2+WCO2 `_am` estimate (their errors differ: band rsd
1.1 ppm ocean) and an optimal blend with the prior to test whether the ~20 %
explained variance survives out of sample; (3) the slab-MC identifiability test from
the first-pass verdict, now with the `_am` model, to find the noise-free floor;
(4) for the near-cloud diagnostic, test E0_am / k1_am directly against the anomaly
over ocean, the only place a ≥ 0.3 correlation appeared.

### Result 2026-11-01 (round 2) — can a fast far-cloud retrieval approach full physics? (`results_2016_2020_v2/`)

**Scope** (user, 2026-11-01): far-cloud (> 20 km) QF0 soundings only. The question is
whether the cumulant/PPDF route can replace a full-physics retrieval there. The TCCON
comparison was dropped (its stage is disabled in `run_v2.py`; no results), and the
near-cloud diagnostic is low priority. The pre-registered bars are kept, and one
criterion is added: **far error ÷ prior-only null < 1**.

**Code.** All new code is in this folder:
- `estimators_v2.py`: surface-binned path models, SCO2+WCO2 combination, prior blends;
- `run_v2.py`: stages `perdate` → `pooled` → `opblend` → `neardiag` → `summary`;
- `slab_mc.py`: the identifiability test.

`estimators.py` / `run_test.py` (round 1) are unchanged. Jobs (Blanca): 29264259
(perdate), 29264262 (pooled), 29264268 (slab MC), 29264291 (opblend + summary). Data:
116 dates of the 2016–2020 combined file, 232 date × surface cells. No `ref_*`/`r15_*`/`r05_*`
statistic enters any estimator.

What was tried, by step:

1. **Land path model that carries over.**
   - (a) Per date, path models in tertiles of continuum reflectance (`_ab`) or of the
     spectral class exp(c_SCO2)/exp(c_O2A) (`_sc`).
   - (b) One model per surface pooled over dates (`E2P*`: ridge; `*_hgb`:
     HistGradientBoosting), trained on far QF0 rows of the fm1620k5 fold's
     train+calib dates and scored on its held-out date block (≈ one year per fold).
     `_rc` rescales by the same date's far-cloud mean f.
2. **Combination and blends.**
   - `E2c*` is the convex minimum-variance SCO2+WCO2 combination of f; it uses no bc.
   - Blends have the form prior + a + w(est − prior):
     - `BL*`: (a, w) from the same date's other orbit fold, with a = N0's same-date
       bc offset;
     - `BLP*`: w pooled over training dates, a = same-date bc offset;
     - **`OB*` (operational)**: a and w both fitted on training dates only, so nothing
       from the scored date's xco2_bc enters. This is the only blend a fast
       retrieval could actually run. N0, BL* and BLP* need a same-date bc offset,
       so they are information diagnostics, not retrievals.
   - The fair null for an operational estimator is therefore **the prior itself**:
     rsd(bc − prior) on the same footprints.
3. **Slab Monte Carlo** (`slab_mc.py`): a 1D plane-parallel MC (HG aerosol layer plus
   cloud in 30 % of scenes, Lambertian surface, local estimates toward the sensor).
   It uses real Doppler τ spectra, frac_h2o and p_abs from 450 soundings, refits the
   `_am` models, and predicts k1 from the same gas-free features. 4000 training
   scenes (g = 1), 2000 test scenes (g ∈ 1 ± 1 %).

**Far-cloud results** (116 dates; median over cells; ppm; `fast_retrieval_far_table.csv`, `summary.md`).
Columns:
- `rsd vs bc` = robust SD of est − xco2_bc.
- `total` = median over cells of √(rsd² + bias²).
- `R²` = 1 − (rsd vs bc ÷ rsd(bc − prior))², the robust out-of-sample share of the
  full-physics departure from the prior that is reproduced.
- `corr` = corr(est − prior, bc − prior).

| estimator | held-out unit | rsd vs bc (ocean / land) | total | rsd vs prior | R² | corr | cells < null |
|---|---|---|---|---|---|---|---|
| prior only (xco2_apriori) | — | 0.94 (0.80 / 1.11) | 1.07 | 0 | 0 | — | — |
| N0_prior (prior + same-date bc offset) | orbit | 1.08 (0.90 / 1.25) | 1.10 | 0 | −0.18 | −0.22 | — |
| E2w_am (round-1 best, per-date) | orbit | 1.31 (0.91 / 2.27) | — | 1.40 | −0.85 | 0.42 | 31 % |
| E2_am_ab / E2_am_sc (binned, per-date) | orbit | 1.65 / 1.70 (land 4.5 / 5.2) | — | 1.8 | < −2 | 0.2 | 13 % |
| E2c_am (S+W combined, per-date) | orbit | 1.13 (0.91 / 2.37) | 1.02 | 1.04 | −0.57 | 0.39 | 33 % |
| E2P_am (pooled ridge, SCO2) | **date block** | 0.89 (0.78 / 1.01) | — | 1.07 | 0.13 | 0.61 | 72 % |
| E2wP_am (pooled ridge, WCO2) | date block | 0.80 (0.67 / 1.01) | — | 1.04 | 0.25 | 0.68 | 78 % |
| **E2cP_am_hgb** (pooled HGB, S+W) | date block | **0.79 (0.67 / 0.90)** | 0.86 | 0.92 | 0.31 | 0.68 | 82 % |
| **OB_E2cP_am_hgb** (+ training-date blend) | date block | **0.74 (0.64 / 0.86)** | **0.78** | 0.70 | **0.39** | 0.68 | 93 % |
| OB2P_am_hgb (S, W separately blended) | date block | 0.72 (0.62 / 0.85) | 0.76 | 0.72 | 0.38 | 0.68 | 93 % |
| OB_E0_am / OB_E2c_am (per-date inputs) | date block | 0.93 / 0.88 | 1.03 / 1.02 | 0.39 / 0.32 | 0.10 / 0.08 | 0.38 | 78 % |
| B-IDP (offset-calibrated IMAP-DOAS) | orbit | 1.76 | — | 1.91 | −2.45 | 0.45 | 9 % |

- **Step 1: pooling across dates is what makes the path model carry over.**
  - Per-date surface binning makes land worse (4.5–5 ppm), because the bins starve
    the per-date calibration.
  - The pooled models, tested on held-out date blocks with no information from the
    scored date, reach land 0.90–1.01 ppm against the prior's 1.11. Round-1 per-date
    E2_am was 2.98.
  - Ocean: 0.67 against the prior's 0.80.
  - Ridge and HGB are close (0.80–0.89); combining SCO2+WCO2 gives a further ~0.05–0.1 ppm.
  - Per held-out year, far rsd ÷ N0 for E2cP_am_hgb is 0.79 / 0.82 / 0.84 / 0.67 / 0.65
    (2016–2020); its total error is 0.82–0.91 against the prior's 0.96–1.16. Every
    year beats the prior.
- **Step 2: blending.**
  - The pure pooled estimate inherits the prior's date-mean bias against bc (land
    median +0.47 ppm, as the prior's +0.50). The operational blend (w ≈ 0.75, a ≈
    +0.2 ocean / −0.4 land from training dates) removes it (land bias +0.08) and
    shrinks the scatter.
  - Result: total 0.78 ppm against the prior's 1.07; R² 0.39 (ocean 0.33, land 0.40);
    93 % of cells beat the prior.
  - Per-date blends (BL*) only tie the null (ratio 0.97–1.05): one date's 2-orbit-fold
    calibration is too noisy. The round-1 "~20 % explained variance" therefore holds
    out of sample only through the pooled path model, where it rises to 30–40 %.
- **Step 3: noise-free identifiability floor** (slab MC; error of ĝ in ppm, `slab_mc_floor.csv`):

  | path model | all scenes | clear (no cloud, τa < 0.1) | cloudy | land | ocean |
  |---|---|---|---|---|---|
  | E0 (k1 / mean) | 4.3 | 1.5 | 25 | 2.8 | 5.2 |
  | ridge (as per-date E2) | 6–8 | 2.9–3.5 | 11–17 | 5.6–6.7 | 4.6–8.8 |
  | HGB, noise-free | 1.1–1.3 | **0.5–0.6** | 4.8–5.7 | 0.85–0.9 | 1.8–2.2 |
  | HGB, with photon noise | 1.4 | 0.7–0.75 | 4.7–6.0 | 0.9–1.2 | 2.1–2.5 |

  The gas-free `_am` observables pin `<l'>_S` to ≈ 0.15 % (0.5–0.6 ppm) in near-clear
  scenes. That is under the 0.25 % target, but only with a non-linear path model; a
  linear one is 5× worse. The floor degrades to 1–2 ppm over dark ocean, where aerosol
  dominates the path, and 5 ppm under thin cloud. Instrument noise adds little, so the
  limit is path-model physics, not SNR.
  - Caveats: 1D only (no 3D/near-cloud effects); a single vertical profile shape per
    channel for dry gas and H2O; no glint BRDF, polarization or in-band scattering
    dispersion; only 4000 training scenes. It is an optimistic lower bound.
  - The real far-cloud result (≈ 0.7 ppm vs bc, which includes bc's own
    single-sounding noise) is consistent with this floor.

**Bars** (116-date medians):
- **Retrieval (≤ 2 ppm):** passed by most `_am` estimators, but not discriminating.
- **Beats the prior-only null:**
  - **passed** by every pooled estimator: E2P*, E2wP*, E2cP*, the BL*P / BLP*P blends
    of them, and all OB* on pooled inputs (far/N0 0.71–0.89);
  - **failed** by every raw per-date estimator (best E2c_am 1.15). Per-date inputs pass
    only after a training-date blend (OB_E0_am / OB_E2c_am 0.86, R² ≈ 0.1); the
    same-date-offset blends BL* / BLP* sit at 0.94–1.0.
- **Diagnostic (near corr ≥ 0.3):** failed. Not pursued.
- **Plume:** not re-tested (far-cloud scope).

**Cost** (order of magnitude):
- The fit is ~1 min per orbit of ~5–10k soundings, i.e. ~10 ms per sounding.
- The pooled path model and blend take microseconds.
- The dominant cost is the per-sounding prior τ: the Doppler line-by-line τ takes
  ~0.9 h per orbit on 8 cores (curc tau-only script), ≈ 5 core-s per sounding with the
  current Python LBL code. A lookup-table τ could cut that by 10–100×.
- A full-physics OCO-2 retrieval is of order minutes of CPU per sounding (outside
  estimate, not measured here).
- So the PPDF route is ~10–100× cheaper as implemented, and potentially ~10⁴× with
  tabulated τ.

(Near-cloud, low priority, from the job already run: over ocean, corr(E0_am − prior,
`xco2_bc_anomaly_r05`) is 0.59 at 0–1 km and falls to 0.09 beyond 20 km; bc − prior
gives 0.81. `neardiag_ocean.csv`.)

**Verdict:** **promising as a fast far-cloud retrieval, not yet a full-physics
replacement.**
- A dates-pooled path model on the two-τ / agreed-model cumulants, combined over SCO2
  and WCO2 and blended with the prior using training-date coefficients, is the first
  estimator in this study to beat the prior on held-out dates and years:
  - total error vs bc 0.78 ppm against the prior's 1.07;
  - 93 % of date × surface cells beat the prior;
  - it reproduces ~30–40 % of the variance of the full-physics departure (bc − prior)
    and correlates with it at 0.68.
- It is cheap: ~10 ms per sounding plus the τ cost.
- What limits it:
  1. It is trained against far-cloud k1 with f = 1 assumed, so it can only learn the
     path; its information on XCO2 comes from k1 residuals. Its departure from the
     prior correlates only 0.68 with bc − prior and has to be shrunk (w ≈ 0.75).
  2. Dark-ocean aerosol paths, where the slab floor is 2 ppm even noise-free.
  3. The prior's date-mean bias, which only the training-date offset removes.
  4. It is validated against xco2_bc, not truth. The 0.74 ppm residual includes bc's
     own noise, so the true gap to full physics is unknown without an independent
     reference (TCCON was dropped from scope).
- Next steps:
  - a larger, physically richer simulator (3D-free but with glint BRDF and separate
    H2O profile) to train the path model on simulations instead of on far-cloud data;
  - tabulated τ to make the cost argument real;
  - an independent truth (TCCON) once back in scope.

### Result 2026-11-01 (round 3, physical PPDF retrieval) — slab MC + one real date (`results_phys/`)

**Scope** (user, 2026-11-01): a non-ML, physics-based retrieval of XCO2 from the `_am`
cumulant observables, as a candidate fast retrieval. Far-cloud (> 20 km) QF0 only. The user
then cut Test 2 to **one date** (2018-07-01) before any scale-up. The 116-date run was **not**
done, so the training-date offset could not be fitted (see below). Hard rules kept: no
regression trained on xco2_bc or on far-cloud k1, no ref_/r15_/r05_ inputs, Lite aerosol
fields as diagnostics only, xco2_bc only for scoring.

**Code** (new, this folder): `phys_ppdf.py` (forward model + optimal estimation),
`run_phys.py` (stages `slab`, `real`, `summary`). The `_am` fits are imported from
`src/spectral/cumulant_fit.py` (`_fit_am_o2a/_wco2/_sco2`), not reimplemented. Blanca jobs:
29264692 (slab), 29264707 (slab, no O2A f), 29264701 / 29264702 (2018-07-01, with / without
O2A f). Each took 1–3 min on 32 cores.

**Forward model** (per sounding, per band, per channel, reflectance-factor units):
- **Gas.** The sounding's own slant τ comes from `fp_tau_combined.h5`, split into Rayleigh
  (`frac_ray`), H2O (`frac_h2o`) and dry gas.
  - Dry gas is scaled by s_p (O2A) or g·s_p (WCO2/SCO2); H2O is scaled by h.
  - The vertical distribution F(σ) = σ^(m+1) comes from each channel's `p_abs_hpa`, using
    slab_mc's construction. ps = `psfc_prior` (a met prior, not a retrieved quantity).
- **Scattering layer.** One layer at σ_H with OD τ_L(765)·(λ/0.765)^−α. Fixed ω = 0.95 and
  HG g = 0.7: the phase function is exact for single scattering, and δ-M scaling is used
  for attenuation and the diffuse terms.
- **Components:**
  - (a) direct surface path;
  - (b) layer single scattering, which sees only the gas above σ_H (path shortening);
  - (c) surface–layer multiple reflections: geometric series in ρ_f·r_L·exp(−2D(1−F)τ_vert),
    with D = 5/3 (path lengthening);
  - (e) one diffuse leg through the layer below σ_H, in both orders;
  - (d) Rayleigh single scattering integrated over σ (real data only; the slab has no Rayleigh).
- **Surface.** The direct surface reflectance ρ is set per band from the **observed**
  continuum exp(c_am) by a fixed point. ρ_f = ρ for land and for the Lambertian slab, and
  0.06 for ocean glint. The scale T = μ0²·RF follows from the fitting.py T definition
  (toa_sol = 0.5·F0/μ0, one polarization).
- **Synthetic fit.** The synthetic ln T is fitted with the same `_am` functions on the same
  channels (1..1014, τ-sorted). The slab uses its every-4th-channel set.

**Inversion.** Levenberg–Marquardt optimal estimation (Rodgers §5.7) with forward-difference
Jacobians.
- **State and priors:**

  | element | prior | σ |
  |---|---|---|
  | g | 1 | 1 % (weak, ≈ 4 ppm) |
  | s_p | 1 | 0.2 % |
  | h | 1 | 20 % |
  | ln τ_L | ln 0.08 | 1 |
  | α | 1 | 0.7 |
  | σ_H | 0.85 | 0.15 |

  All are bounded.
- **Observables (8):** O2A k1_am, f_am, k2diag_am; WCO2 k1_am, k1w_am; SCO2 k1_am, k1w_am,
  k2diag_am.
  - WCO2 k2diag_am is **excluded**: it is −0.08 to −0.19 in the real data, which no positive
    photon-path distribution can produce. It is τ-model curvature, not path.
  - The a1/a2 slopes and c are not matched: c is used for ρ, and a log-linear albedo slope
    changes only a1.
- **Error covariance.** S_y = S_formal + diag(floor²).
  - S_formal is the full formal covariance of the observed coefficients, rebuilt by linear
    propagation of iid residuals (s² = rms_am²·n/(n−p)) through each fit, including the O2A
    k1/f correlation (+0.71). Its diagonal reproduces the stored sig_* exactly (ratio 1.000).
  - Forward-model floors, set before any run: O2A k1 0.004, f 0.002, k2diag 0.005; WCO2/SCO2
    k1 0.002; k1w 0.03; SCO2 k2diag 0.003.
  - No inflation factor was fitted.
- **Output.** XCO2 = xco2_apriori·ĝ, with posterior σ_g and averaging kernel. DOF_g = A_gg.

**Test 1: slab MC, no training.** The round-2 test scenes are regenerated exactly (same RNG
stream; k1 matches `slab_mc_scenes.csv` to 1e-9). There are 2000 scenes with g ∈ 1 ± 1 %.
Table: rsd of ĝ error in ppm, noise-free / noisy; HGB floor from round 2 for comparison
(`slab_phys_floor.csv`):

| subset | physical OE (8 obs) | physical OE, no O2A f | round-2 HGB (trained on 4000 scenes) |
|---|---|---|---|
| all | 1.59 / 1.68 | 1.62 / 1.73 | 1.1–1.3 / 1.4 |
| clear (no cloud, τa < 0.1) | **0.67 / 0.74** | 0.70 / 0.77 | **0.5–0.6 / 0.7–0.75** |
| cloudy | 7.5 / 7.3 | 7.7 / 7.3 | 4.8–5.7 / 4.7–6.0 |
| land | 1.44 / 1.48 | 1.41 / 1.50 | 0.85–0.9 / 0.9–1.2 |
| ocean | 1.89 / 1.98 | 2.07 / 2.21 | 1.8–2.2 / 2.1–2.5 |

- Convergence: 98 % overall, 100 % clear, 92 % cloudy. Median DOF_g 0.90 (0.93 clear);
  posterior σ_g 1.06 ppm in clear scenes. Median χ²: 0.7 in clear scenes, 19 in cloudy ones.
- **A 6-parameter physical model with no training matches the trained HGB floor in clear
  scenes (0.67 vs 0.5–0.6 ppm) and over ocean.** It is worse over land (1.4 vs 0.9) and under
  cloud (7.5 vs 5), where the single layer cannot represent the slab's aerosol + cloud pair.
  This is forward-model mismatch, not non-identifiability: DOF_g stays at 0.8–0.9 in cloudy
  scenes, and χ² flags them.
- A χ² < 5 screen keeps 68 % of scenes (95 % of clear, 23 % of cloudy) and gives
  0.93 / 1.02 ppm overall.
- The error grows with aerosol OD, as a bias: median error +0.3, +0.8, +1.4, +2.5 and +3.4 ppm
  for τa < 0.05, 0.05–0.1, 0.1–0.2, 0.2–0.5 and > 0.5. Path modification is under-corrected
  by the few-component model; the single-scattering/δ-M approximation is the obvious suspect.
- Retrieved ln τ_L correlates with the true ln τa at 0.55 in cloud-free scenes.
- Instrument noise adds only ~0.07 ppm.

**Test 2: real far-cloud QF0 soundings, 2018-07-01.**
- **Why this date:** it is in the combined 2016–2020 file; it has 9258 ocean and 14425 land
  far-cloud QF0 soundings over 14 orbits (GL and ND); all 17 τ files exist; slab_mc's τ
  bank comes from it.
- 1000 per surface were drawn at random, with ok_am = 1 in all bands and nchan = 1014; all
  were matched to τ by sounding id.
- Round-2 rows are on the same soundings: E0_am and E2cP_am_hgb from the round-2 compact
  frame (fold 2 holds this date out); OB_E2cP_am_hgb from fold 2's training-date
  coefficients.
- **No training-date offset was possible with a single date.** The raw result is shown, plus
  the same-date mean offset as a **diagnostic only**.

Table: ppm; `rsd` is the robust SD, `total` = √(rsd² + bias²), `R²` = 1 − (rsd vs bc ÷
rsd(bc − prior))², `corr` = corr(est − prior, bc − prior). Primary run (8 observables):

| estimator | rsd vs bc ocean / land | bias vs bc ocean / land | total ocean / land | rsd vs prior ocean / land | R² ocean / land | corr ocean / land |
|---|---|---|---|---|---|---|
| prior only | 0.92 / 1.59 | +0.19 / +1.53 | 0.94 / 2.20 | 0 / 0 | 0 / 0 | — |
| **PHYS (raw, uncalibrated)** | **0.93 / 1.54** | −2.80 / −3.87 | 2.95 / 4.16 | 1.06 / 2.61 | −0.02 / 0.07 | **0.72 / 0.80** |
| PHYS + same-date offset [diagnostic only] | 0.93 / 1.54 | 0 / 0 | 0.93 / 1.54 | 1.06 / 2.61 | −0.02 / 0.07 | 0.72 / 0.80 |
| PHYS, σ_a(g) = 0.25 % (analytic re-prior) | 0.89 / **1.01** | −1.18 / −0.84 | 1.48 / 1.32 | 0.52 / 1.20 | 0.07 / **0.60** | 0.68 / 0.77 |
| PHYS without O2A f (7 obs), raw | 1.09 / 1.51 | −2.58 / −3.80 | 2.80 / 4.09 | 1.12 / 2.51 | −0.41 / 0.10 | 0.70 / 0.80 |
| E0_am | 2.39 / 2.01 | +0.49 / +1.85 | 2.44 / 2.74 | 2.44 / 2.81 | −5.8 / −0.6 | 0.41 / 0.68 |
| E2cP_am_hgb (round 2, ML) | 0.79 / 1.28 | −0.11 / +0.86 | 0.80 / 1.54 | 0.98 / 1.30 | 0.26 / 0.35 | 0.79 / 0.54 |
| OB_E2cP_am_hgb (round 2, ML + blend) | 0.67 / 1.24 | +0.15 / +0.61 | 0.69 / 1.38 | 0.77 / 1.02 | 0.47 / 0.39 | 0.79 / 0.54 |

**Notes on the table.**
- **Re-prior row.** It recombines the weak-prior posterior with σ_a(g) = 0.25 % (≈ 1 ppm, the
  typical spread of bc − prior). This is exact for a linear-Gaussian problem; per sounding it
  shrinks ĝ − 1 by ≈ 0.45. The value was fixed before the run, but it is informed by the
  round-2 spread of bc − prior, so read it as a sensitivity, not a tuned blend.
- **"% of cells beating the prior"** is not meaningful with two cells. Here the raw estimate
  ties the prior over ocean (0.93 vs 0.92) and beats it slightly over land (1.54 vs 1.59).

**Diagnostics** (primary run; `real_2018-07-01_diag.csv`, `summary_phys.md`).
- **Convergence and DOF:** 100 % converged, median 3–4 iterations, 30–36 forward calls.
  DOF_g median 0.93 / 0.92 (q10–q90 0.88–0.94); total DOF 2.7–2.8; posterior σ_g 1.07 /
  1.13 ppm.
- **χ²/obs:** median 20 (ocean) / 7.9 (land), almost entirely from O2A f_am. The model fill
  is 0.023–0.038 against 0.008–0.013 observed; f pulls are −5 to −14.
  - Cause: the τ files hold only ILS-effective τ. Exp(−a·τ_eff) overestimates the transmission
    of short-path components in saturated O2A lines (Jensen), and `mean_ext` carries no
    sub-channel information.
  - Without f, χ²/obs is 1.25 / 1.53. The g result changes little over land (1.51) and gets
    worse over ocean (1.09), so f still carries useful shortening information despite its
    bias.
  - Other pulls: rsd 0.7–1.7. Systematic residuals: O2A k1 obs − F = +0.004 to +0.005; O2A
    k2diag −0.008 to −0.011, because observed O2A k2diag is ≤ 0. WCO2/SCO2 k1 are matched to
    ≤ 0.002.
- **Bias** −2.8 ppm (ocean) and −3.9 ppm (land), constant across aerosol OD and airmass bins
  (bin medians within ±0.6 ppm), with orbit-to-orbit scatter ~±0.7 ppm. This is what a
  training-date offset would absorb. The remaining land error correlates with albedo
  (corr(est − bc, alb_sco2) = 0.44), pointing to the surface/multiple-reflection terms.
- **Retrieved parameters vs Lite (diagnostic only):**
  - τ_L vs aod_total: Spearman 0.58 over land, 0.02 over ocean. Median τ_L 0.07–0.08.
  - σ_H vs dust_height: ≈ 0. σ_H mostly sits near the surface (median 0.94–0.99) and is
    poorly constrained.
  - h vs h2o_scale: 0.98–0.99.
  - s_p vs psfc/psfc_prior: 0.05 (ocean) / 0.33 (land).
  - χ² screening (best half) does not improve land.
- **Cost.**
  - Inversion: 213 / 262 ms median per sounding on one core (ocean / land; 123–150 ms
    without f). The O2A nonlinear `_am` fit (≈ 6 ms) is called ~30 times; WCO2/SCO2 fits are
    0.2 ms. Excludes τ.
  - The current LBL τ costs ≈ 5 core-s per sounding, so τ is still ~95 % of the total. A τ
    lookup table at ~ms per sounding would make the inversion the bottleneck (~0.2 s).
    Analytic Jacobians and linear projection for the two linear-fit bands could take that
    to ~20–50 ms.

**Verdict.** **A non-ML physical PPDF retrieval works in principle, but on real data it does
not beat the prior on its own.**
- **What works:**
  - On slab MC it reaches the trained-HGB clear-sky floor with no training. g is well
    determined (DOF ≈ 0.9).
  - On 2018-07-01 its departure from the prior tracks the full-physics departure strongly:
    corr 0.72 (ocean) / 0.80 (land), higher than the round-2 ML model on land (0.54).
- **What does not:**
  - Its amplitude is too noisy. Scatter about bc ties the prior over ocean (0.93 vs 0.92) and
    barely beats it over land (1.54 vs 1.59).
  - It carries a −3 to −4 ppm constant bias, which needs the allowed per-surface offset
    (not fittable with one date).
  - Only with a ~1 ppm prior on g (OE shrinkage) does it clearly beat the prior over land
    (1.01 vs 1.59, R² 0.60). Over ocean that gives 0.89 vs 0.92 (R² 0.07), still behind the
    round-2 ML blend (0.67).
- **Limits, in order:**
  1. ILS-effective τ cannot represent sub-channel saturation for short/long-path components.
     O2A f is badly mis-modelled, and the same error enters SCO2 strong channels.
  2. The single-layer, single-scattering/δ-M component model under-corrects path
    modification as aerosol grows (slab bias +2.5 ppm at τa 0.2–0.5) and fails for
    multi-layer or cloud scenes.
  3. Unmodelled spectroscopic/instrument systematics: the constant bias; observed O2A and
     WCO2 k2diag ≤ 0; an albedo-correlated land error.
  4. One date only, so neither the per-surface offset nor the per-year stability is tested.
- **Not run:** the 116-date evaluation and per-year table, the training-date offset, and the
  % of cells beating the prior.
- **Possible next steps if expanded:**
  - run the 116 dates with the training-date offset and the analytic σ_a(g) sensitivity;
  - a sub-channel τ distribution (two-stream k-distribution from the LBL step) to fix O2A f;
  - a two-layer or 2-stream adding model in place of the single-scattering components.

### Result 2026-11-01 (round 4, layer-free path-transfer retrieval) (`results_phys/round4_*`)

**Why this round.** The user rejected PPDF-S (round 3), because the cumulant framework does not
assume a scattering layer. This round tests a layer-free, non-ML path-transfer method that works
on the fitted `_am` coefficients only: no τ files, no spectrum synthesis. Code is in
`path_transfer.py`, with stages `slabgen` (Blanca job 29265004, about 1 min), `slab` and `real`.
The method is vectorised algebra and runs in seconds. The data are the same 2000 slab-MC test
scenes as rounds 2–3, regenerated with identical coefficients, and the same 1000 + 1000
far-cloud QF0 soundings of 2018-07-01 as round 3. Rules as before: no ML, xco2_bc only for
scoring, no ref_ statistics, Lite fields only as diagnostics.

**Method.**
- **Path model:** k1_O = s_p⟨l⟩_O, k1_W = g⟨l⟩_W, k1_S = g⟨l⟩_S, with
  δ_b = ⟨l⟩_b − 1 = w_b(c_L·A_b − c_S/A_b) and w_b = (λ_b/0.765)^−α.
  - A_b = exp(c_am)/K is the band continuum reflectance factor: K = μ0² for the OCO-2 T,
    N_PHOT/π in the slab.
  - Only band ratios of A matter for g; a common scale is absorbed by c_L and c_S.
- **Per sounding, with α fixed:**
  - eq1: δ_O = k1_O − 1 (s_p = 1);
  - eq2: δ_S − r·δ_W = r − 1, with r = k1_S/k1_W;
  - 2×2 linear solve for (c_L, c_S), then g = k1_S/(1 + δ_S). g_W is identical by construction.
- **Variants:**
  - (a) two kernels (TK), α ∈ {0, 0.5, 1, 1.5, 2};
  - (b) one kernel (OK): δ_b = δ_O·w_b, albedo ignored;
  - (c) `_ray`: a known Rayleigh single-scattering shortening −½ε/(1+ε) is removed from every
    band first. ε = τ_R·P_R(Θ)/(4μ0μv·A_b), with Hansen–Travis τ_R scaled by psfc_prior. Real
    data only; the slab has no Rayleigh.
  - (d) TKfree: α free per sounding (grid 0–3), using SCO2 k2diag/k1² as a third equation.
    This needs an extra assumption: each kernel is a single path shift of fixed size
    Δ_S = Δ_L = 1 slant column, so var(l)_b = w_b(c_L·A_b + c_S/A_b).
- **Errors:** linear propagation of the three bands' sig_k1_am gives σ_g. Systems with
  condition number > 1e3 are flagged.

**Slab MC** (error of ĝ in ppm; noise-free / noisy; `round4_slab_floor.csv`):

| estimator | all | clear | cloudy | land | ocean |
|---|---|---|---|---|---|
| TK α = 1 | 6.3 / 7.2 | 1.73 / 2.50 | 31 / 30 | 4.7 / 6.0 | 9.2 / 9.3 |
| TK α = 2 (best TK) | 5.2 / 5.7 | 1.46 / 1.91 | 31 / 31 | 3.9 / 4.6 | 7.4 / 7.5 |
| OK α = 1 | 3.2 / 3.5 | 1.19 / 1.56 | 19 / 19 | 3.2 / 3.4 | 5.1 / 5.1 |
| OK α = 2 | 3.7 / 3.8 | 1.16 / 1.48 | 22 / 22 | 2.7 / 3.1 | 4.9 / 4.8 |
| TKfree (k2 third eq.) | 4.5 / 4.9 | 1.59 / 1.96 | 23 / 23 | 3.0 / 3.6 | 8.5 / 8.6 |
| round 3 physical OE | 1.59 / 1.68 | 0.67 / 0.74 | 7.5 / 7.3 | 1.44 / 1.48 | 1.89 / 1.98 |
| round-2 HGB (trained) | 1.1–1.3 / 1.4 | 0.5–0.6 / 0.7–0.75 | ~5 | 0.9 | 2 |

- **Aerosol dependence** (cloud-free, TK α = 2, noise-free): 1.09, 3.6 and 8.7 ppm for
  τa < 0.05, 0.1–0.2 and 0.2–0.5. OK α = 2 gives 0.74, 2.7 and 5.0. Median biases are
  +0.4 to +2.5 ppm.
- **Conditioning:** 54–61 % of scenes are ill-conditioned, 97–99 % over ocean. Where all band
  albedos are nearly equal, the two equations become proportional; the well-conditioned rows
  are only a little better.
- **σ_g:** the formal σ_g (0.6–1.2 ppm) understates the actual scatter by a factor of 2–7
  (pull rsd).
- **Is the two-kernel form adequate?** I tested it against the true path factors
  ⟨l⟩_b = k1_b/g_true.
  - Taking kernels from the true δ_O and δ_W and predicting δ_S gives 1.07–1.32 ppm in clear
    scenes, 2.4–3.0 over land, 5–17 over ocean, 5.3–7.7 at τa 0.2–0.5 and about 18 in cloudy
    scenes. Even with perfect W-band information, the two-kernel λ-scaling cannot carry the
    path to SCO2 better than ~1 ppm in clear scenes, and it breaks for ocean (dark, flat-albedo),
    higher aerosol and cloud.
  - The one-kernel oracle (true δ_O → δ_S) is similar: 1.1–1.2 clear, 2.7–3.2 land, 4.6–5.0
    ocean at α 1–2.
  - With α free per scene the form fits all three true δ's (median residual 0.08 ppm), but the
    best-fit α does not track the true Ångström exponent (corr 0.17). α acts as a nuisance
    parameter, not a physical one. The k2-based α of TKfree has corr −0.04.
  - With retrieved kernels, eq2 amplifies errors in r: TK goes from 1.46 noise-free to 1.91 ppm
    with noise in clear scenes.

**Real data, 2018-07-01** (rsd vs bc, ocean / land, ppm; `round4_real_2018-07-01_table.csv`).
The prior gives 0.92 / 1.59; round-3 physical OE gives 0.93 / 1.53.
- **TK (α 0–2):** 3.4–5.1 / 4.9–5.8. corr(est − prior, bc − prior) is about 0 over ocean and
  0.2–0.5 over land; R² is −12 to −29.
- **TK with Rayleigh removed:** 3.5–4.0 / 3.8–6.3.
- **TKfree:** 1.93 / 3.37.
- **OK (α 0–2):** 2.2–4.7 / 1.87–5.0. Best is OK α = 2 at 2.24 / 1.87, corr 0.44 / 0.63.
- **OK with Rayleigh removed:** best α = 2 at 2.17 / 1.80. The Rayleigh term removes most of
  the ocean bias of OK (median ε_O = 0.064 ocean, 0.035 land; δ_R,O ≈ −3.0 % / −1.7 %).
- **Well-conditioned ocean subset** (TK α = 2, 23 % of soundings): 1.27 ppm.
- No variant beats the prior or round 3.
- Same-date offsets [diagnostic only] are −4 to +7 ppm depending on the variant; the offset
  does not change rsd.
- Formal σ_g is about 0.9–1.1 ppm, with pull rsd vs bc 2.2–4.4.

**Model-free consistency check (real data).**
- The dry SCO2/WCO2 path ratio r = k1_S/k1_W has median 1.012 (ocean) / 1.009 (land) and
  rsd 0.0055 / 0.0070. This offset is a 1 % spectroscopic/τ inconsistency between the bands.
  eq2 reads it as path, and it is 10× larger than any plausible two-kernel path difference.
- The water ratio k1w_S/k1w_W has median 0.994 / 0.970. Its correlation with the dry ratio is
  only 0.09 / 0.15.
- So the scene-to-scene variation of the band ratio is mostly not a common photon-path signal,
  which is what eq2 requires.

**Kernels vs Lite (diagnostic only, TK α = 1).**
- Ocean: Spearman(c_S, aod_total) = +0.31 (physical sign); c_L vs albedo +0.43.
- Land: c_S vs aod −0.22 (unphysical sign); c_L vs aod +0.47.

**Cost.** One vectorised 2×2 solve takes about 0.05 µs per sounding. All 30+ variants with
error propagation take about 10 µs per sounding. This is essentially free; τ is not needed
at all.

**Verdict.**
- **The layer-free path-transfer retrieval does not work at the needed level.** In slab MC its
  best clear-sky error is 1.2–1.5 ppm (one kernel) and 1.5–1.9 ppm (two kernels), against
  0.67–0.74 for round 3 and 0.5–0.6 for the trained HGB. It is 5–9 ppm over ocean and
  20–30 ppm under cloud.
- On real data it is 2–5× worse than the prior.
- **Why:**
  1. The first-order two-kernel λ-scaling is inadequate even with oracle W-band information
     (≥ 1 ppm clear, 5–17 ppm ocean).
  2. The system is singular whenever the band albedos are similar (most ocean).
  3. On real data eq2 is dominated by a 1 % SCO2/WCO2 k1 inconsistency and by ratio scatter that
     the water path does not share. The method assumes that k1_S/k1_W carries pure path
     information, and in these products it does not.
- The two-kernel form adds nothing over the one-kernel baseline, which is itself well short of
  round 3.
- The physical OE of round 3 stays the best non-ML result: it ties or barely beats the prior
  raw, and beats it over land with an OE g-prior.

### Result 2026-11-01 (round 5, per-band direct-path atom retrieval, no cross-band link) (`results_phys/round5_*`)

**Why this round.** The user's premise: O2A, WCO2 and SCO2 light need not share photon paths,
so rounds 3–4, which tie paths across bands, violate it. Under the equivalence theorem g enters
only as g·l, and an arbitrary per-band PPDF makes g exactly degenerate. The only absolute anchor
is the direct surface-reflected beam, whose path is exactly l = 1.

**Model** (`atom_fit.py`). Each band is fitted independently on the observed ln T, using the
same channels, x and T convention as the `_am` fits:
- ln T = c + a1·x [+ a2·x² for SCO2 land] + ln{(1−f)[(1−ε)e^(−t) + ε(1 + t·θ/κ)^(−κ)] + f}.
  - CO2 bands: t = g·τ_dry + h·τ_h2o. O2A: t = s·τ.
  - H2O choice: one gamma PPDF per band, applied to the band's total scaled gas τ; the direct
    term is exact for both gases.
- Free parameters: g (s), h, ε ∈ [0, 1], θ ∈ [0.05, 5] (shortening or lengthening),
  κ ∈ [0.3, 1000], c, a1 (a2). f ∈ [0, 0.2] only in the `_f` variant (O2A).
- Fitting: bounded TRF least squares with an analytic Jacobian and 9 starts (3 g × 3 (ε, θ)).
  A sounding counts as multimodal if starts within Δχ² < 1 of the best disagree by > 1 ppm.
- Profile likelihood: refit on a g grid of ĝ ± 2 % (12 points), with χ² = RSS/s²_best. The
  minimum is "well-defined" if it sits at ĝ and Δχ² = 1 is reached on both sides; σ_g is the
  half-width of that interval.
- Real ln T comes from L1B radiance / (0.5·toa_sol)·π. The src `_am` fit on this ln T reproduces
  the stored k1_am exactly (max diff 0), so the inputs are identical to production.
- Jobs: Blanca 29265401 (slab, about 15 min on 32 cores) and 29265413 (real O2A).

**Test 1: slab MC, same 2000 scenes** (error of ĝ in ppm-equivalent, 0.25 % = 1 ppm;
noise-free / noisy; `round5_slab_floor.csv`):

| band | all | clear | cloudy | land | ocean | τa < 0.05 | 0.1–0.2 | 0.2–0.5 |
|---|---|---|---|---|---|---|---|---|
| O2A (s vs 1) | 6.2 / 7.8 | 2.77 / 3.71 | 21 / 22 | 2.9 / 4.3 | 14 / 16 | 1.8 / 2.5 | 7.1 / 8.4 | 16 / 18 |
| O2A + f | 8.3 / 10.0 | 3.28 / 4.05 | 31 / 30 | 4.0 / 5.7 | 18 / 19 | 2.4 / 2.7 | 8.4 / 10.5 | 16 / 19 |
| WCO2 | 4.1 / 5.6 | 1.65 / 2.22 | 16 / 20 | 3.9 / 4.2 | 4.4 / 6.7 | 1.3 / 1.8 | 3.3 / 3.6 | 5.5 / 6.7 |
| SCO2 | 4.7 / 5.4 | **1.27 / 1.74** | 39 / 40 | 1.7 / 2.7 | 13 / 13 | 0.86 / 1.32 | 3.2 / 4.3 | 7.1 / 8.6 |
| round 3 physical OE | 1.59 / 1.68 | 0.67 / 0.74 | 7.5 | 1.44 | 1.89 | | | |
| round-2 HGB (trained) | 1.1–1.3 / 1.4 | 0.5–0.6 / 0.7–0.75 | ~5 | 0.9 | 2 | | | |

- **Convergence and modes:** convergence is 92–100 %. Multimodality is rare except for WCO2
  (23–40 %) and O2A + f (14–19 %).
- **Well-defined profile minimum** (noise-free / noisy): O2A 77 / 71 %, SCO2 68 / 59 %,
  WCO2 63 / 8 %, O2A + f 59 / 52 %.
- **The error is set by ε̂** (SCO2, noisy):

  | ε̂ | rsd error | well-defined |
  |---|---|---|
  | < 0.003 | 0.66 ppm | 89 % |
  | 0.003–0.01 | 1.5 ppm | |
  | 0.01–0.03 | 3.3 ppm | |
  | 0.03–0.1 | 8.5 ppm | |
  | 0.1–0.3 | 23 ppm | |
  | > 0.3 | 54 ppm | 1 % |

  The other bands follow the same pattern.
- **Error vs max channel τ:** the top quartile of τ_max roughly doubles the error.
- **σ_g:** the profile σ_g (median 0.3–0.8 ppm for O2A/SCO2) is 3–10× smaller than the actual
  error. The minima are mostly sharp but in the wrong place.
- **Interpretation:** in the slab the effective τ is used consistently as if it were
  monochromatic, so sub-channel saturation is absent. The failure here is therefore the atom
  premise itself: diffuse light scattered close to the surface has l ≈ 1 and smears the atom.
  The gamma q cannot represent that mixture, so ε and g trade off and the bias grows quickly
  with aerosol.
- The method only works (≤ 1 ppm) when ε ≲ 0.01, essentially aerosol-free scenes.

**Test 2: real O2A truth check, 2018-07-01** (same 1000 + 1000 soundings;
`round5_real_2018-07-01_table.csv`):

| variant | median ŝ − 1 (ocean / land) | rsd(ŝ) | well-defined | profile σ |
|---|---|---|---|---|
| no f | −1.92 % / −0.52 % | 0.99 % / 1.01 % (≈ 4 ppm-eq) | 98 % / 97 % | 0.44–0.47 ppm-eq |
| with f | −2.61 % / −0.90 % | 1.12 % / 1.18 % | 53 % / 46 % | 1.7–2.3 ppm-eq |

- **The bar (rsd(ŝ) ≲ 0.3 %) fails by about 3.3×.** The profile σ again understates the
  actual scatter by about 9×.
- **No f:** the diffuse term collapses to the θ = 0.05, κ = 0.3 bounds in > 90 % of soundings.
  It acts as a zero-absorption fill (median ε 1–1.5 %), not as a scattered-light PPDF.
- **With f:** f takes the fill (median 0.012 ocean / 0.008 land, as in `_am`), and ε → 0 in
  31 % / 61 %.
- **ŝ adds nothing beyond k1_am:** corr(ŝ, k1_am) = 0.72–0.77, and the rsd is the same as k1_am
  (≈ 1 %).
- **Diagnostics only:**
  - ŝ vs psfc/psfc_prior: corr 0.26–0.52, but that ratio scatters only 0.19–0.24 %.
  - ŝ vs aod_total: Spearman −0.43 (ocean) / +0.46 (land).
  - ε vs aod: +0.21 (ocean) / −0.34 (land); no consistent physical sign.
- **Step 3 (CO2 ĝ vs xco2_bc) was not run,** as instructed, because steps 1–2 are clearly not
  viable.

**Cost.** One free fit (9 starts, 1014 channels) takes about 240 ms per band per sounding,
about 25 ms per start. With the 12-point profile it is about 0.4 s for O2A. The slab bands
(254 channels) take 0.2–0.6 s.

**What limits it.**
- **Primarily, near-surface or low aerosol smearing the l = 1 atom.** This shows directly in the
  slab, which has no sub-channel saturation: error rises steeply with ε̂ and aerosol OD, the
  minima are sharp but biased, and O2A in the slab already scatters 0.9 % in clear scenes,
  about the same as real data.
- **On real data, two further effects:**
  - the channel-averaged (ILS-effective) τ makes short-path light in saturated lines look like
    zero-absorption fill: θ is pinned at its lower bound, and f/ε trade off;
  - τ-model and spectroscopy misfit: residual rms 0.024 in O2A, far above noise.
  These cannot be separated from zero-level offset or SIF with the information available.

**Verdict.** The per-band atom retrieval removes the cross-band assumption, but the direct beam
is not a usable absolute anchor at the 0.25 % level. Only scenes with ε ≲ 0.01 get close.
Clear-sky slab error is 1.3–1.7 ppm (SCO2) against 0.67 for round 3. The real O2A self-check is
1 % against the 0.3 % bar. Not pursued further.

### Result 2026-11-01 (round-3 O2A truth check) (`results_phys/round3truth_*`)

**What was run.** The round-3 OE was rerun with the O2 scale freed: s_p prior σ 0.2 % → 5 %,
bounds 0.85–1.15, everything else unchanged. It was run with and without O2A f, on the same
2018-07-01 1000 + 1000 soundings and the 2000 slab scenes.
- Command: `run_phys.py slab|real --sp-sd 0.05 [--exclude-obs o2a_f_am] --out-name round3truth_*`.
- Tables come from `round3truth.py`. Blanca jobs 29265624–29265627.
- O2 is known (true s_p = 1), so ŝ_p − 1 is the method's own path error at O2A.

**O2A truth** (rsd of ŝ_p − 1; % / ppm-eq; bar ≲ 0.3 %):

| | clear | land | ocean | cloudy | median ŝ_p − 1 |
|---|---|---|---|---|---|
| slab, freed, with f (noisy) | 0.53 % / 2.1 | 0.68 % / 2.7 | 1.13 % / 4.5 | 3.7 % / 15 | +0.52 % (+2.1 ppm-eq) clear |
| slab, freed, no f (noisy) | 0.54 % / 2.2 | 0.68 % / 2.7 | 1.17 % / 4.7 | 3.5 % / 14 | +0.52 % clear |
| real 2018-07-01, freed, with f | — | 0.63 % / 2.5 | 0.94 % / 3.7 | — | +0.70 % land / +0.91 % ocean |
| real 2018-07-01, freed, no f | — | 0.61 % / 2.4 | 0.86 % / 3.4 | — | +0.51 % land / +0.57 % ocean |

- The tight-prior round-3 runs keep ŝ_p within ±0.08 %. The 0.2 % prior held it there.
- ŝ_p vs psfc/psfc_prior (diagnostic only; that ratio scatters 0.19–0.24 %): corr −0.11 to 0.08
  over ocean and 0.34–0.38 over land.
- **Freeing s_p reduces DOF_g** from 0.92 to 0.67 (ocean) / 0.78 (land).

**CO2 with s_p freed** (rsd vs bc, ocean / land, ppm):

| run | rsd vs bc | bias | rsd vs prior | corr(est − prior, bc − prior) |
|---|---|---|---|---|
| prior only | 0.92 / 1.59 | +0.19 / +1.53 | 0 | — |
| round 3, s_p 0.2 %, with f | 0.93 / 1.54 | −2.8 / −3.9 | 1.06 / 2.61 | 0.72 / 0.80 |
| round 3, s_p 0.2 %, no f | 1.09 / 1.51 | −2.6 / −3.8 | 1.12 / 2.51 | 0.70 / 0.80 |
| freed s_p, with f | 3.00 / 2.01 | −5.3 / −5.7 | 2.69 / 2.54 | 0.19 / 0.56 |
| freed s_p, no f | 2.91 / 1.99 | −4.3 / −5.0 | 2.60 / 2.42 | 0.24 / 0.58 |

- corr(ŝ_p − 1, ĝ − 1) is −0.75 to −0.77 (ocean) and −0.34 to −0.41 (land).
- corr(ŝ_p − 1, ĝ − bc/prior) is −0.85 / −0.70: the O2A path error maps almost one-for-one into
  the CO2 error.
- corr(ŝ_p − 1, (bc − prior)/prior) is only 0.22–0.34.
- The product g·s_p is no better: 1.40–1.46 / 1.81–1.96 ppm.

**Screen on |ŝ_p − 1|** (freed run, no f; thresholds fixed in advance):

| threshold | kept (ocean / land) | rsd vs bc (ocean / land) | prior on the same soundings |
|---|---|---|---|
| < 0.3 % | 28 % / 25 % | 0.71 / 1.75 | 1.01 / 1.65 |
| < 0.5 % | 41 % / 41 % | 0.77 / 1.78 | 1.16 / 1.78 |
| < 1 % | 62 % / 83 % | 1.01 / 1.77 | 1.35 / 1.73 |

- Over ocean the kept soundings beat the prior on the same soundings (0.71 vs 1.01), but only
  28 % are kept.
- Over land the screen gives no gain.
- The with-f variant is similar (ocean 0.81 vs 1.03 at 19 % kept).

**Slab CO2 with s_p freed:**
- ĝ error is 1.3 ppm clear (0.74 with the tight prior), 1.7 land, 2.4 ocean and 4.3 cloudy
  (7.3 tight).
- corr(ŝ_p error, ĝ error) is −0.41 to −0.46 in clear scenes and about 0 overall.

**Verdict.**
- **The O2A truth test fails** at about 0.6 % (land) to 0.9 % (ocean), 2–3× over the 0.3 % bar.
- **The round-3 land gain does not survive it.** With O2 freed, CO2 degrades to 2.0 ppm (land)
  and 3.0 ppm (ocean), worse than the prior. The tight s_p prior was hiding a 0.5–0.9 % path
  error at O2A, which the aerosol state absorbed.
- **The real-data O2A error is about the same size as the slab's:**
  - real 0.61–0.63 % vs slab 0.68 % (land);
  - real 0.86–0.94 % vs slab 1.1 % (ocean);
  - median bias +0.5 to +0.9 % in both.
  The few-component scattering model's mismatch alone, in a world with no sub-channel
  saturation, spectroscopy error or zero-level offset (ZLO)/SIF, already produces it.
- **So sub-channel τ (option 2) does not look like the first-order fix.** It cannot remove an
  error that the slab shows without it. The O2A f misfit (χ²/obs 20 / 7 with f) is a separate
  real-data effect, but dropping f leaves ŝ_p almost unchanged.
- The forward-model path physics is the limit.

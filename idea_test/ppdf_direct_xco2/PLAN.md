# PPDF-based direct XCO2 estimate — first feasibility test

**Started:** 2026-10-29. Side study, outside the JQSRT/AMT paper scope.

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

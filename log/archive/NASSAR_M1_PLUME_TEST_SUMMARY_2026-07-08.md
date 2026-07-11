# Nassar Power-Plant Plume Test for M1

> **ARCHIVED 2026-07-09 — verdict superseded.** This first-pass summary
> concluded the diagnostic did "not yet" validate plume preservation. The
> subsequent passes (control-region null, transects, channel attribution)
> resolved it: the spec channel is plume-safe (≤0.21 ppm bound), 5/7 control
> nulls pass, the 2 flags were real cloud (all-band k1 fingerprint). Current
> record: `PROJECT_REVIEW.md` M1, `SPEC_EMPHASIS_STATUS_2026-07-08.md` §2, and
> `results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/nassar_plumes/`.

Date: 2026-07-08

## Purpose

This note summarizes the current power-plant plume diagnostic for **M1 — target
circularity + selection effect**.  The M1 concern is that the correction target
is defined from OCO-2 minus nearby OCO-2 clear-sky neighbors, both using
`xco2_bc`, so real sub-0.25 degree CO2 gradients can be labeled as cloud
anomalies.  A useful negative control should show that, for clear-sky
power-plant plumes, real XCO2 enhancements remain in `xco2_bc` while predicted
`mu` is approximately inert.

## Current Diagnostic

The Nassar plume diagnostic uses selected power-plant cases from the local
Nassar workflow and summarizes footprints within 25 km and 50 km of each source.
For each case it reports:

- original `xco2_bc` range;
- corrected XCO2 range;
- predicted correction `mu` range;
- `corr(xco2_bc, mu)`;
- cloud-distance context;
- footprint-number grouped spectral spread for `o2a_k1`, `o2a_k2`,
  `wco2_k1`, and `wco2_k2`.

The diagnostic is intentionally not a flux inversion.  It is a plume-preservation
stress test for whether the model correction might remove real local CO2
structure.

## Headline Results

For the three main clear-sky plume candidates, the current results do **not**
yet satisfy the desired negative-control behavior.  `mu` is not flat across the
local XCO2 structure; it is strongly correlated with `xco2_bc`, and the
correction compresses the local XCO2 range.

| Case | Radius | Clear n | XCO2 p95-p5 | Corrected p95-p5 | mu p95-p5 | mu/original | Corrected/original | corr(XCO2, mu) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ghent 2024-04-15 | 25 km | 66 | 3.79 ppm | 1.78 ppm | 2.20 ppm | 0.58 | 0.47 | 0.94 |
| Ghent 2024-04-15 | 50 km | 151 | 2.46 ppm | 1.35 ppm | 1.00 ppm | 0.41 | 0.55 | 0.92 |
| Comanche 2020-09-06 | 50 km | 94 | 1.60 ppm | 0.92 ppm | 0.83 ppm | 0.52 | 0.58 | 0.89 |
| Colstrip 2015-08-01 | 50 km | 201 | 1.55 ppm | 0.72 ppm | 1.01 ppm | 0.65 | 0.46 | 0.90 |

Interpretation: these cases are currently better described as a **stress test or
possible failure mode** than as a successful plume-preservation negative
control.  The model reduces roughly 40-55% of the local XCO2 p95-p5 range in
these windows.

## Cloud-Distance and Spectral Context

The interpretation is not yet final because the local windows can mix plume
structure with cloud-distance gradients, footprint geometry, surface type, and
along-track variation.  This matters because absolute values of `o2a_k1`,
`o2a_k2`, `wco2_k1`, and `wco2_k2` can vary with geometry and surface type.
Therefore the more defensible quantity is the **within-group std**, especially
within footprint-number groups.

Using footprint-number groups with at least five soundings per group:

| Group | Cloud-distance filter | n footprint groups | median std O2A k1 | median std O2A k2 | median std WCO2 k1 | median std WCO2 k2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clear, `cld_dist_km >= 20` | none | 48 | 0.0195 | 0.0880 | 0.0174 | 0.136 |
| Cloudy median `< 20 km` | none | 87 | 0.0261 | 0.111 | 0.0245 | 0.155 |
| Clear, `cld_dist_km >= 20` | `cld p95-p5 >= 5 km` | 32 | 0.0245 | 0.106 | 0.0166 | 0.118 |
| Cloudy median `< 20 km` | `cld p95-p5 >= 5 km` | 39 | 0.0228 | 0.0870 | 0.0292 | 0.203 |

Interpretation:

- The cloud-related spread signal is stronger and more consistent in
  `wco2_k1` and `wco2_k2`.
- `o2a_k1` and `o2a_k2` std are not consistently larger in cloudy cases after
  accounting for cloud-distance variation and footprint number.
- Small spectral std is only meaningful evidence for stability when the local
  footprints also sample enough cloud-distance variation.  If all footprints are
  similarly far from clouds or similarly close to clouds, the spectral std can
  be small even if the case is not a strong diagnostic.

## Current Conclusion

The Nassar power-plant analysis should not yet be presented as evidence that the
correction preserves real power-plant plume signals.  The current result is more
cautious:

> Initial Nassar cases show clear local XCO2 structure, but predicted `mu` is
> not inert.  It correlates strongly with `xco2_bc` and compresses the
> plume-like XCO2 range.  This does not yet validate plume preservation.  The
> next pass should isolate far-cloud and cloud-distance-homogeneous footprints,
> stratify by footprint number, and test whether `mu` still tracks XCO2 after
> controlling for cloud distance, footprint number, and along-track position.

## Recommended Next Pass

1. Restrict to far-cloud soundings, for example `cld_dist_km > 30 km`.
2. Within those, separately test cloud-distance-homogeneous subsets, for example
   `cld_dist_km_p95_minus_p05 < 5-10 km`.
3. Stratify by footprint number (`fp=0..7`) before comparing spectral std.
4. Regress `mu` against `xco2_bc` while controlling for:
   - `cld_dist_km`;
   - footprint number;
   - source distance or along-track coordinate;
   - optionally latitude/longitude trend terms.
5. Treat cases where `mu` remains correlated with `xco2_bc` after these controls
   as plume-removal risk, not as negative-control success.

## Related Outputs

Current local outputs are under:

```text
results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/nassar_plumes/
```

Key files:

```text
plume_preservation/nassar_plume_preservation_summary.csv
plume_preservation/nassar_plume_preservation_summary.md
plume_preservation/nassar_plume_preservation_by_footprint.csv
plume_preservation/nassar_plume_preservation_by_footprint.md
plots/
```

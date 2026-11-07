# Slab 3-D vs IPA condition sweep: summary for the manuscript (2026-11-06)

Written for the manuscript agent. Purpose: decide whether any of the new
slab-simulation results should be added to Appendix G (or the SI) of the
AMT/JQSRT draft, and which existing statements need correcting. Everything
below was produced on CURC from the same slab code that generates Fig. G1
and Table G1 (`workspace/rt_slab_sim/`), with one analysis script
(`fit_scenes_am.py`) that applies the production agreed-model (AM) fit of
`src/spectral/cumulant_fit.py` to every simulated column.

## 1. What the manuscript currently has, and what changed

The appendix shows one scene: SZA 55, nadir view, water cloud 4–5 km
(effective, er3t +1 km offset), COD 10, r_eff 10 µm, 1 km thick × 5 km wide
in a 32 km periodic x–z slab, Lambertian dark (0.03) and bright (0.30)
surfaces, O2 A-band, 1e9 photons, MCARaTS 3-D vs IPA, with path-length
moments from the MC path record (Fig. G1) and the configuration in Table G1.

The sweep adds 50 scenes (1e8 photons each unless noted) in three groups,
plus a glint group, all fitted with the production AM model so the metric
is the retrieved quantity the paper actually uses: the AM path-length
coefficient k1 (and k2diag) per 0.5 km column.

Metric definitions used in the tables:
- `k1_far`: IPA k1 averaged over the far field (x < 5 km, no cloud influence).
- `Δk1_shadow`: mean (3-D − IPA) k1 over the shadow columns (continuum
  radiance ratio 3-D/IPA < 0.7, outside the cloud).
- `Δk1_cloud`: same over the cloud columns (parallax-corrected for slant views).
- `Δk1_far`: same over the far field (cloud-glow / horizontal-transport offset).
- Footprint rows: radiances and path histograms averaged with a circular
  2.25 km top-hat before fitting (OCO-2 footprint scale).

Data and figures: `/scratch/alpine/yuch8913/oco/spec_anal/results/rt_slab_sim/scenes/_summary/`
(`summary_am.md`, `summary_am_fp2.25.md`, `scene_<scene>_am.png`,
`summary_delta_{k1,k2diag,exp_int}_am[_band][_fp2.25].png`) and
`.../scenes_1e9/_summary/` for the 1e9 copies.

## 2. Two corrections to existing text

1. **Phase function.** Table G1 says "Mie phase function (water cloud)". er3t
   writes the Mie asymmetry parameter (g = 0.86 for r_eff 10 µm at 768 nm)
   into the MCARaTS phase-function field, which MCARaTS interprets as a
   Henyey–Greenstein parameter. Every production run, including Fig. G1,
   used HG with the Mie g. Suggested wording: "Henyey–Greenstein phase
   function with the Mie asymmetry parameter (g = 0.86)". The `mie` scene
   below quantifies the effect (shadow Δk1 −0.34 → −0.30, i.e. 10 %).
   `make_table_g1.py` still has the old wording.
2. **Slant-view naming.** The sweep scenes `vza43_fwd` (sensor azimuth 270)
   and `vza43_back` (azimuth 90) are physically swapped: er3t's sensor
   azimuth is the direction of the view ray (sensor → ground), so azimuth 270
   puts the sensor up-sun (backscatter) and azimuth 90 down-sun (forward
   scatter, glint side). This was established by the glint smoke test
   (specular peak at azimuth 90). Only relevant if slant-view results are
   quoted; the manuscript scene is nadir.

## 3. O2 A-band condition ladder (shadow Δk1, dark / bright surface)

| scene | change vs base | Δk1 shadow dark | Δk1 shadow bright | Δk1 cloud dark/bright |
|---|---|---|---|---|
| ref | none (1e8 control of Fig. G1) | −0.342 | −0.025 | 0.00 / 0.05 |
| sza25 | SZA 25 | −0.170 | +0.111 | 0.00 / 0.03 |
| sza70 | SZA 70 | −0.452 | −0.082 | 0.01 / 0.06 |
| vza43_fwd (backscatter) | VZA 43 | −0.290 | −0.155 | 0.01 / 0.05 |
| vza43_back (forward) | VZA 43 | −0.233 | −0.051 | 0.00 / 0.03 |
| cloud_low | cloud 1–2 km | −0.212 | −0.034 | 0.00 / 0.01 |
| cod2 | COD 2 | −0.131 | −0.018 | 0.03 / 0.06 |
| cod30 | COD 30 | −0.405 | +0.004 | 0.00 / 0.01 |
| rayleigh_off | no Rayleigh | +0.020 | +0.053 | 0.00 / 0.05 |
| width1 | cloud 1 km wide | −0.328 | −0.070 | 0.00 / 0.06 |
| width15 | cloud 15 km wide | −0.324 | −0.037 | 0.00 / 0.03 |

Reading: the 3-D signature is a shadow effect, not an in-cloud effect.
|Δk1| in the cloud columns never exceeds 0.09 (bright) or 0.03 (dark); the
shadow Δk1 over a dark surface is −0.13 to −0.45 and scales with SZA and
COD (how much direct beam the cloud removes), is independent of cloud
width, and vanishes when Rayleigh scattering is switched off. Over a bright
surface the sign flips at low SZA (+0.11 at SZA 25) because cloud-side
illumination adds to the surface term. The far field carries a small
positive offset (Δk1_far +0.005 to +0.014) from cloud glow.

Mechanism (consistent across the sweep): in the shadow the direct beam is
gone, so the column radiance is dominated by Rayleigh/aerosol path
radiance from above the absorbing layers; the AM fit reads that as a shorter
effective path (lower k1) and a larger curvature term (k2diag +0.04 to
+0.09 in the shadow, all surfaces).

## 4. Surface types and CO2 bands (base geometry)

Far-field k1 and shadow Δk1 by band and surface (Lambertian albedos:
ocean 0.03/0.02/0.02, veg 0.45/0.25/0.12, soil 0.28/0.42/0.32, snow
0.90/0.12/0.03 for O2A/WCO2/SCO2):

| surface | O2A k1_far / Δk1_shadow | WCO2 | SCO2 |
|---|---|---|---|
| ocean | 0.790 / −0.341 | 0.940 / −0.203 | 1.258 / −0.288 |
| veg | 0.968 / +0.042 | 0.994 / −0.013 | 1.169 / +0.097 |
| soil | 0.954 / −0.037 | 0.997 / +0.020 | 1.098 / +0.240 |
| snow | 0.983 / +0.167 | 0.988 / −0.063 | 1.249 / −0.204 |

The dark-surface shadow signal transfers to both CO2 bands (−0.20 WCO2,
−0.29 SCO2 over ocean). Bright SWIR surfaces (soil, veg) give a positive
shadow Δk1 in SCO2 (+0.10 to +0.24) and a positive in-cloud Δk1 (soil
+0.085). The CO2-band ladders (same ten conditions as Sect. 3, ocean and
soil) reproduce the O2A ordering: SCO2 ocean −0.56 at SZA 70 and −0.54 at
COD 30, −0.02 at SZA 25 and −0.04 at COD 2; WCO2 ocean −0.32 at SZA 70,
−0.28 at COD 30, −0.05 at SZA 25; Rayleigh-off removes the negative signal
in all three bands. Full rows in `summary_am.md`.

## 5. Physics variants (O2A, base geometry; shadow Δk1 dark / bright)

| scene | variant | Δk1 shadow dark | bright | comment |
|---|---|---|---|---|
| ref | HG, r_eff 10 | −0.342 | −0.025 | |
| mie | tabulated Mie phase function | −0.303 | −0.015 | ~10 % weaker shadow |
| cer5 / cer20 | r_eff 5 / 20 µm | −0.354 / −0.323 | −0.021 / −0.024 | ±5 % |
| high_thin | cloud 9–10 km, COD 1 | −0.119 | −0.021 | thin cirrus-like layer still −0.12 |
| high_cod10 | cloud 9–10 km, COD 10 | −0.453 | −0.053 | higher cloud → longer, darker shadow |
| aerosol | BL aerosol AOD 0.15 (1–3 km, SSA 0.95, g 0.7) | −0.134 | +0.001 | aerosol path radiance fills the shadow; far-field k1 0.790 → 0.754 |
| coast | no cloud, albedo step 0.03 → 0.30 at 16 km | – | – | no 3-D k1 signature at the step (|Δk1| ≤ 0.015) |

Cloud microphysics moves the shadow signal by ≤12 %; cloud height, COD and
boundary-layer aerosol move it by factors of 2–3. The aerosol case matters
for framing: the shadow Δk1 is a Rayleigh-plus-aerosol path-radiance
effect and a hazy boundary layer damps it. The coast case shows that a
surface-albedo contrast alone produces no 3-D path-length artefact.

## 6. Footprint averaging (2.25 km circular top-hat)

Shadow Δk1 shrinks by 15–30 % and in-cloud and far-field values are
unchanged: ref dark −0.342 → −0.286; sza70 dark −0.452 → −0.388; cod30 dark
−0.405 → −0.362; width1 dark −0.328 → −0.160 (shadow only 4 columns wide);
vza43 backscatter dark −0.290 → −0.188; wco2 ocean −0.203 → −0.156; sco2
ocean −0.288 → −0.201. Sign and ordering of every scene are preserved.
Narrow-shadow scenes lose the most, as expected.

## 7. Photon budget: 1e8 vs 1e9

wco2, sco2 and o2a_sfc were rerun at 1e9 photons (3 runs × 10 pooled
sub-runs, explicit seeds). Every tabulated Δk1 agrees with the 1e8 value to
≤ 0.005 (e.g. o2a ocean shadow −0.342 vs −0.341, wco2 ocean −0.203 vs
−0.203, sco2 ocean −0.292 vs −0.288); only the far-field run-to-run scatter
tightens (sig_dk1_far 0.006 → 0.002 for SCO2). 1e8 is sufficient for
tabulated numbers; 1e9 is only needed for figure-quality curves.

## 8. Sun glint (OCO-2 glint mode: VZA = SZA, sensor at the specular point)

Surface: MCARaTS diffuse–specular mixture, Fresnel reflection off Cox–Munk
facets (slope variance 0.003 + 0.00512 U) with a Lambertian water-leaving
term of 0.02; water refractive index 1.33 / 1.318 / 1.30 per band. A
cloud-free control (`glint_cod0`) gives 3-D = IPA within 0.002.

| scene | 3 m/s | 7 m/s | 12 m/s | note |
|---|---|---|---|---|
| far-field k1, SZA 30 (O2A) | 0.969 | 0.945 | 0.925 | Lambertian ocean: 0.790 |
| glint, SZA 30, COD 10: Δk1 shadow | −0.18 | −0.17 | −0.15 | shadow reflectance −78 to −90 % |
| glint_sza15 | −0.11 | −0.10 | −0.09 | shadow 5 columns |
| glint_sza45 | −0.16 | −0.16 | −0.12 | far field contaminated by parallax (see caveats) |
| glint_cod2 | −0.04 | −0.04 | −0.04 | direct beam survives |
| glint_cloud_low | −0.15 | −0.13 | −0.11 | shadow 3 columns |
| wco2_glint | −0.03 | −0.02 | −0.02 | far-field k1 0.99–1.00 |
| sco2_glint (shadow plateau / edge column) | +0.08 / −0.50 | +0.07 / −0.61 | +0.06 / −0.65 | k1 rises 1.05 → 1.17 in the shadow; one negative column at the cloud/shadow boundary |

Reading: a glint ocean behaves like a bright surface (k1_far 0.93–0.97, not
0.79), because the specular beam dominates the Rayleigh floor. The cloud
shadow removes the specular beam, so the shadow column reverts to a
Rayleigh-dominated spectrum: Δk1 −0.15 to −0.18, about half the
Lambertian-ocean value, with in-cloud Δk1 ≤ 0.01 and a shadow k2diag of
+0.06 to +0.09. At 2.25 km footprint: glint 7 m/s −0.168 → −0.147;
glint_cloud_low −0.13 → −0.08; the others change by < 0.02.

1e9-photon copies of `glint` and `sco2_glint` at 7 m/s
(`scenes_1e9/_summary/`, same wavelength grid) agree with the 1e8 rows:
glint far-field k1 0.945 both, shadow Δk1 −0.173 vs −0.168, in-cloud 0.003
vs 0.002, far-field scatter 0.000 vs 0.001; sco2_glint far-field k1 1.050
both, shadow plateau +0.069 both. In the SCO2 band the glint shadow is a
*positive* k1 anomaly (1.05 → 1.17 across the whole shadow, k2diag +0.08),
opposite in sign to the Lambertian-ocean SCO2 shadow (−0.29); the only
negative SCO2 column is the one at the cloud/shadow boundary. The 1e9
`scene_glint_am.png` and `scene_sco2_glint_am.png` are figure-ready: the
a1 and k2diag curves are smooth outside the cloud-edge columns, and the
run-to-run shading is invisible at plot scale. If a glint figure is wanted,
use the 1e9 tree.

## 9. Recommendation on what to include

Worth adding (in order of value for the paper's argument):
1. **Shadow-vs-cloud statement with the ladder numbers** (Sect. 3): one
   sentence plus a small table or the `summary_delta_k1_am.png` panel. It
   turns Fig. G1's single case into a sensitivity result (SZA, COD, Rayleigh
   on/off) and supports the "shadow, not in-cloud" interpretation.
2. **Band transfer** (Sect. 4, ocean row): the shadow signature exists in
   all three bands with the same ordering; this backs the use of CO2-band
   fit coefficients in the model.
3. **Aerosol and Mie rows** (Sect. 5): the aerosol row is the most useful
   caveat (real boundary layers damp the effect by about half); the Mie row
   is the honest companion to the phase-function correction.
4. **Footprint sentence** (Sect. 6): at OCO-2 footprint scale the signal
   survives at 70–85 % of the column value.
5. **Glint** (Sect. 8): one paragraph if the paper discusses glint
   soundings over ocean; otherwise SI. The key numbers are k1_far 0.93–0.97
   and shadow −0.15 to −0.18 in O2A, and the sign reversal of the SCO2
   shadow anomaly (+0.07 for glint vs −0.29 for Lambertian ocean), which
   is the one glint result that changes a qualitative statement. 1e9
   figure-quality versions of both scenes exist.

Probably not worth adding: cloud width, droplet size, high thin cloud,
coast (null result), the slant-view scenes (parallax makes the zone
statistics less clean), and the 1e9 comparison (methods sentence at most:
"1e8-photon sensitivity runs agree with the 1e9 base case to 0.005 in k1").

## 10. Caveats the agent should know before quoting numbers

- All sweep numbers are 1e8 photons unless stated; run-to-run scatter of
  k1 is 0.001 (O2A), 0.001–0.003 (WCO2) and 0.004–0.006 (SCO2) in the far
  field, so differences < 0.01 are not significant.
- Slant-view scenes (vza43_*, glint_sza45, sco2_vza43_back) have the 3-D
  cloud displaced by h·tan(VZA); the zone statistics roll the IPA profile
  onto the detected cloud, but in glint_sza45 and sco2_vza43_back the
  displaced cloud touches the 0–5 km far-field window, so their
  `Δk1_far` (−0.04, −0.08) is cloud-edge contamination, not a far-field effect.
- Shadow zones are defined by a radiance-ratio threshold (0.7); zone means
  therefore depend slightly on shadow width (3–15 columns). In slant-view
  scenes the detected parallax offset can differ by one column between
  fits (e.g. −5 vs −6 columns for glint), which moves one cloud-edge
  column in or out of the shadow zone; this is why the footprint-averaged
  glint shadow reads −0.147 (1e8) vs −0.179 (1e9) while the full-resolution
  values agree to 0.005. Quote full-resolution zone means, or the 1e9 rows,
  for slant-view scenes.
- The slab cloud is a single homogeneous block; no cloud-edge roughness or
  3-D cloud shape effects are represented.
- The slab uses a one-tau design (total gas + H2O optical depth from the
  footprint profile), and x = band-window coordinate; k1 values are
  comparable across scenes within a band but not to the production fit
  values on real spectra.
- The production fit uses the HG phase function (Sect. 2). If the Mie row
  is quoted, say so.

## 11. Reproduction

```
# any scene: prep -> run -> collect -> fit (Slurm launcher, Blanca)
sbatch -M blanca --export=ALL,STAGE=prep,SCENES_RUN="<scene>" --array=0-0 curc_shell_blanca_rt_slab_scenes.sh
sbatch -M blanca --export=ALL,STAGE=run,TASK_LIST=<scene surface solver lines>,PHOTONS=1e8,NRUN=3,NSUB=10 --array=0-N curc_shell_blanca_rt_slab_scenes.sh
sbatch -M blanca --export=ALL,STAGE=collect,SCENES_RUN="<scene>" --array=0-0 --ntasks=2 --mem=8G --time=01:00:00 curc_shell_blanca_rt_slab_scenes.sh
python workspace/rt_slab_sim/fit_scenes_am.py [--boxcar-km 2.25]      # SLAB_SCENES_SUBDIR=scenes_1e9 for the 1e9 tree
```
Scene definitions: `slab_config.py` (SCENES dict, glint_surface/glint_set).

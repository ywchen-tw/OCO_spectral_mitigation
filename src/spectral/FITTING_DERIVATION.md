# Spectral transmittance fitting — derivation, assumptions, and code map

This note records the physical derivation behind the photon-path cumulant fit
used by this project. The expanded manuscript-style treatment, literature
discussion, and bibliography are in
[`EQUIVALENCE_THEOREM_FITTING_REPORT.tex`](../../manuscript/tex/EQUIVALENCE_THEOREM_FITTING_REPORT.tex).

The central idea is the radiative-transfer **equivalence theorem**: within a
narrow spectral band whose continuum scattering properties are approximately
wavelength-independent, molecular absorption reweights one conservative photon
trajectory ensemble by the Beer–Lambert survival probability. Under an
additional scalar-path approximation, the normalized spectrum is the Laplace
transform of a single conservative photon path distribution.

The active estimator does **not** invert a complete PDF and does **not** impose a
gamma PDF. It fits a truncated log-Laplace expansion. Under the stated
assumptions, `k1` is the mean normalized absorber path and `k2` is its variance.

---

## 1. Define the conservative photon trajectory ensemble

Let `Γ` denote a complete photon trajectory from solar illumination to the
detector, including atmospheric scattering and surface interactions. Define

```text
P₀(Γ)
```

as the distribution of trajectories reaching the detector in a conservative
reference problem with:

- the same illumination and viewing geometry;
- the same surface boundary;
- the same cloud, aerosol, Rayleigh, and polarization scattering properties;
- no absorption by the target molecular line.

For a sufficiently narrow band such as O2A, the continuum scattering properties
are assumed to change slowly with wavelength compared with the large spectral
variation of molecular absorption. Therefore, one `P₀(Γ)` is used for all
retained channels in that band.

**Assumption A1 — wavelength-invariant conservative ensemble:** introducing the
target absorption, or moving between channels within the band, does not change
the underlying scattering trajectories. Wavelength changes only their survival
weights.

This is the project-specific use of the equivalence theorem developed in the
literature cited in the LaTeX report (Irvine, 1964; Partain et al., 2000;
Stephens and Heidinger, 2000).

---

## 2. Add molecular absorption along each trajectory

For wavelength `λ`, the absorption optical depth accumulated along trajectory
`Γ` is

```text
A_λ(Γ) = ∫_Γ α_λ(r) ds,
```

where `α_λ(r)` contains the absorber number density, line strength, and local
pressure and temperature dependence. The Beer–Lambert survival weight is

```text
w_λ(Γ) = exp[−A_λ(Γ)].
```

The normalized radiance of the absorbing problem is the conservative-ensemble
average of this weight:

```text
𝒯_λ = E_{P₀}[w_λ]
    = ∫ P₀(Γ) exp[−A_λ(Γ)] dΓ.                     (1)
```

Equation (1) is the general equivalence-theorem expression used here. It does
not require a plane-parallel atmosphere or a gamma path distribution.

**Assumption A2 — passive absorption:** molecular absorption removes photon
weight but does not change the trajectory-generating scattering problem.
Inelastic wavelength-changing processes are neglected or assumed corrected.

---

## 3. Reduce each trajectory to a scalar normalized absorber path

Equation (1) is a functional average over complete, vertically structured
trajectories. The project reduces it to one scalar path coordinate by assuming

```text
A_λ(Γ) ≈ τ_λ · l′(Γ),                              (2)
```

where:

- `τ_λ` is the modeled optical depth for the direct geometric slant path at
  channel `λ` (`od["tau"]` in the code);
- `l′(Γ)` is the effective absorber-path enhancement relative to that direct
  slant reference.

The scalar `l′` is not required to exceed one:

- `l′ ≈ 1` corresponds to the selected direct surface-slant reference;
- multiple scattering and cloud–surface interactions can produce `l′ > 1`;
- cloud-top or aerosol reflection can remove below-scatter travel and produce
  `0 ≤ l′ < 1`.

This is consistent with the implementation, which constrains `k1 ≥ 0`, not
`k1 ≥ 1`.

**Assumption A3 — scalar-path approximation:** the layer-by-layer absorber path
of a trajectory can be represented by one channel-independent multiplier
`l′(Γ)`. This is stronger than assuming the absorber is well mixed. Pressure-
and temperature-dependent line absorption can make two geometrically similar
paths accumulate different absorption if they occupy different altitude
ranges.

---

## 4. Induce the conservative scalar path distribution

The conservative trajectory ensemble induces the normalized path distribution

```text
p₀(l′) = ∫ P₀(Γ) δ[l′ − l′(Γ)] dΓ,
```

with

```text
∫₀^∞ p₀(l′) dl′ = 1.
```

Substituting Equation (2) into Equation (1) gives

```text
𝒯(τ_λ) = ∫₀^∞ p₀(l′) exp(−τ_λ l′) dl′
        = ℒ{p₀}(τ_λ).                              (3)
```

Thus the absorption spectrum samples the Laplace transform of **one
wavelength-independent conservative path distribution**. Wavelength changes
the transform coordinate `τ_λ`; it does not introduce a separate fitted PDF for
each channel.

### Conservative paths versus detected paths

The paths represented among photons detected inside an absorption channel have
the conditional distribution

```text
p_det(l′ | τ) = p₀(l′) exp(−τl′) / 𝒯(τ).           (4)
```

Consequently, weak and strong absorption channels naturally detect different
path populations. Strong absorption preferentially removes long paths. This is
predicted by Equation (3) and does **not** violate the common-`p₀` assumption.

The assumption fails only if the underlying conservative ensemble changes with
wavelength, or if no common scalar `l′(Γ)` satisfies Equation (2) for all
channels.

---

## 5. Include the continuum/surface factor

Under the idealized Lambertian single-reflection interpretation,

```text
πI_λ = F₀,λ μ₀ · γ · 𝒯(τ_λ),
```

so the normalized quantity fitted by the project is

```text
T_λ ≡ πI_λ / (F₀,λ μ₀)
    = γ ∫₀^∞ p₀(l′) exp(−τ_λ l′) dl′.              (5)
```

Therefore `T(0) = γ` and the ideal intercept is `ln γ`.

In the implementation, `compute_transmittance` evaluates

```text
T = radiances / toa_sol · π
```

and masks `T > 1`. For real OCO-2 spectra, `exp(intercept)` should be described
as an **effective continuum reflectance**, not necessarily the exact Lambertian
surface albedo. It can absorb BRDF, polarization, multiple surface–atmosphere
interactions, continuum spectral structure, and normalization errors.

**Assumption A4 — adequate continuum normalization:** remaining channel-to-
channel continuum structure is small enough that the fitted absorption
curvature primarily represents photon-path effects.

---

## 6. Expand the log Laplace transform in cumulants

Let `C_n` be the ordinary cumulants of the conservative distribution `p₀(l′)`.
Its cumulant-generating function is

```text
K(t) = ln E_{p₀}[exp(tl′)]
     = Σₙ₌₁^∞ C_n tⁿ/n!.
```

Since Equation (3) is the moment-generating function evaluated at `t = −τ`,
Equation (5) gives

```text
ln T(τ) = ln γ + Σₙ₌₁^∞ (−1)ⁿ C_n τⁿ/n!.          (6)
```

The code uses the parameterization

```text
ln T(τ) ≈ intercept + Σₙ₌₁ᴺ (−1)ⁿ (k_n/n) τⁿ.     (7)
```

Matching Equations (6) and (7) gives the exact coefficient convention

```text
intercept = ln γ,
k_n = C_n/(n−1)!.                                  (8)
```

Therefore:

| Project coefficient | Exact relation | Interpretation under A1–A4 |
|---|---|---|
| `k1` | `C1` | `E_{p₀}[l′]`, mean conservative path enhancement |
| `k2` | `C2` | `var_{p₀}(l′)`, conservative path variance |
| `k3` | `C3/2!` | scaled third cumulant |
| `k4` | `C4/3!` | scaled fourth cumulant |
| `k_n` | `C_n/(n−1)!` | scaled nth cumulant |

The factorial distinction matters: `k1` and `k2` are exactly the ordinary first
two cumulants, whereas `k3…` are scaled higher cumulants in the code's `/n`
parameterization.

The local derivatives at finite optical depth describe the absorption-selected
distribution from Equation (4):

```text
−d ln 𝒯/dτ  = E_det,τ[l′],
 d²ln 𝒯/dτ² = var_det,τ(l′).
```

By contrast, the fitted `k1` and `k2` target the derivatives at `τ = 0`, hence
the moments of the conservative `p₀`.

---

## 7. Truncate the expansion and solve the project fit

For truncation order `N`, the design matrix is

```text
[1, −τ, +τ²/2, −τ³/3, …, (−1)ᴺτᴺ/N].
```

The model is linear in

```text
[intercept, k1, …, kN].
```

`fit_spectral_model` solves it with SVD least squares. If the unconstrained
solution violates `k1 ≥ 0` or `k2 ≥ 0`, `_solve_cumulant` uses bounded-variable
least squares (BVLS). Higher coefficients are allowed to be negative because
higher cumulants of a general distribution need not be positive.

The production band orders are

```text
(O2A, WCO2, SCO2) = (7, 3, 7).
```

WCO2 is limited to order 3 because its optical-depth range is much smaller than
those of O2A and SCO2; higher powers are poorly identifiable and strongly
collinear. See [`FIT_ORDER_EXPERIMENT.md`](FIT_ORDER_EXPERIMENT.md).

The fitter produces both Savitzky–Golay-smoothed and raw (`_nosg`) coefficient
sets. The raw/no-SG coefficients are the production ML inputs; the smoothed
twins are retained for robustness analysis.

**Assumption A5 — adequate finite-order approximation:** the retained channels
constrain the derivatives of `ln T` about `τ = 0` sufficiently well. Strong line
centers can be outside the useful Taylor region; increasing polynomial order
does not guarantee physical validity.

**Assumption A6 — existence and identifiability:** channel noise, continuum
errors, and polynomial collinearity do not dominate the inferred `k1` and `k2`.
The constraints `k1,k2 ≥ 0` are necessary for a physical PDF but do not ensure
that the complete fitted polynomial is the log-Laplace transform of any
nonnegative PDF.

---

## 8. Gamma PDF: an optional interpretive special case

If, additionally, the conservative path enhancement follows a gamma PDF with
mean `m` and shape `κ`,

```text
p₀(l′) = (κ/m)^κ / Γ(κ) · l′^(κ−1) exp[−(κ/m)l′],
```

then

```text
T(τ) = γ [1 + (m/κ)τ]^(−κ).                        (9)
```

The ordinary gamma cumulants are

```text
C_n = (n−1)! mⁿ/κ^(n−1),
```

so the project coefficients become

```text
k_n = mⁿ/κ^(n−1),
k1 = m,
k2 = m²/κ,
κ = k1²/k2,
k_n = k1 (k2/k1)^(n−1).                            (10)
```

The active fit does **not** enforce Equation (9) or the coefficient sequence in
Equation (10). The closed-form `transmittance_model` exists in
`cumulant_fit.py` but is unused by production. The stored `k1…k5` come from the
free polynomial fit. Consequently, `k1²/k2` is only a gamma-equivalent shape
diagnostic, not evidence that the path PDF is gamma.

Gamma should therefore not be listed as a required assumption of the active
estimator. It is a useful two-parameter physical reference that cannot represent
all multilayer, direct-beam-plus-tail, or other multimodal path populations.

---

## 9. When the scalar path distribution is insufficient

For a vertically structured atmosphere, describe trajectory `Γ` by its path
lengths in atmospheric layers:

```text
L(Γ) = (L1, …, LZ).
```

The general conservative-ensemble expression becomes

```text
T_λ = γ_λ ∫ p₀(L1,…,LZ)
      exp[−Σ_z α_λ,z L_z] dL1…dLZ.                 (11)
```

Equation (11) is a multidimensional Laplace transform. It reduces to the
project's Equation (3) only if the layer paths are approximately proportional
to a common reference profile:

```text
L_z(Γ) ≈ l′(Γ) L̄_z.
```

Possible violations include:

- cloud-top reflection truncating the below-cloud path;
- 3-D horizontal transport concentrated near cloud altitude;
- lower-tropospheric aerosol scattering;
- repeated cloud–surface interactions;
- pressure- and temperature-dependent differences between line centers and
  wings;
- wavelength-dependent surface BRDF, polarization, or aerosol scattering.

These effects do not invalidate the general equivalence theorem in Equation
(1). They invalidate only the reduction to one scalar, wavelength-independent
`p₀(l′)`. In that case, the fitted coefficients remain useful **band-effective
absorber-path summaries**, but they are not exact geometrical-path cumulants.

---

## 10. Assumption ledger

| ID | Assumption | Needed for | If it fails |
|---|---|---|---|
| A1 | Conservative scattering ensemble is constant within a band | One `P₀(Γ)` for all channels | Spectrum combines different underlying trajectory ensembles |
| A2 | Absorption passively applies Beer–Lambert weights | General equivalence theorem | Requires wavelength-changing/coupled radiative transfer |
| A3 | `A_λ(Γ) ≈ τ_λ l′(Γ)` | One-dimensional Laplace transform | `k1`, `k2` become effective absorber- and channel-weighted coefficients |
| A4 | Continuum and surface normalization is adequate | Intercept and curvature interpretation | BRDF/instrument/continuum structure leaks into fitted coefficients |
| A5 | Truncated expansion represents the retained `τ` range | Recover zero-absorption derivatives | Strong channels bias or destabilize cumulants |
| A6 | Moments are identifiable and a physical PDF is locally plausible | Probabilistic interpretation | Polynomial may fit well without corresponding to a nonnegative PDF |
| Optional | Gamma form | Closed-form Equation (9), `κ = k1²/k2` interpretation | No impact on active free-polynomial estimator |

---

## 11. Code map

The fit core is in [`cumulant_fit.py`](cumulant_fit.py). [`fitting.py`](fitting.py)
is the orchestration facade and re-exports the public fitting names.

| Derivation step | Code |
|---|---|
| `T = πI/toa_sol` normalized radiance, Equation (5) | `compute_transmittance`; masks `T > 1` (`cumulant_fit.py`) |
| Per-channel direct-slant optical depth `τ_λ` | `od["tau"]` from `oco_fp_atm_abs`; edge channels `[1:-1]` are dropped (`orbit_data.py`, `fitting.py`) |
| Project expansion, Equation (7) | `log_transmittance_model_1 … _9` and `LOG_TRANSMITTANCE_MODELS` (`cumulant_fit.py`) |
| Design matrix `[1, −τ, +τ²/2, …]` | `get_design_matrix` (`cumulant_fit.py`) |
| Exact linear solve | `_solve_cumulant` and `fit_spectral_model`: `lstsq`, with BVLS fallback (`cumulant_fit.py`) |
| `k1,k2 ≥ 0` | Bounds applied only to the first two coefficients in `_solve_cumulant` |
| Raw and SG fit products | Dual fit in `_fit_chunk`; `_nosg` coefficients are production inputs (`models/pipeline._USE_NOSG_K`) |
| Band orders `(7,3,7)` | `constants.FIT_ORDER` |
| Store `k1…k5` and intercept | `kappa_fitting`, `intercept_fitting` → `output_dict` (`fitting.py`) |
| Gamma-equivalent `κ = k1²/k2` | `gamma_kappa`, NaN where `k2 ≤ 0` (`fitting.py`) |
| Closed-form gamma Equation (9) | `transmittance_model`; present but unused by the active fit (`cumulant_fit.py`) |

---

## 12. Recommended manuscript wording

> Following the equivalence theorem, we define `p₀(l′)` as the normalized
> absorber-path distribution induced by the conservative photon trajectory
> ensemble for a fixed geometry and continuum scattering problem. Within each
> narrow OCO-2 band, wavelength is assumed to modify photon contributions only
> through the Beer–Lambert weight `exp(−τ_λl′)`, giving
> `T(τ_λ) = γ∫p₀(l′)exp(−τ_λl′)dl′`. Expanding the log Laplace transform yields
> `ln T = intercept + Σ_n(−1)^n(k_n/n)τ^n`, where `k1` and `k2` are respectively
> the mean and variance of `p₀(l′)` under the scalar-path approximation. The
> fitted distribution is the common conservative ensemble, not the
> absorption-selected path population within each channel; the latter naturally
> varies with optical depth.

The main caveat should follow immediately:

> Because real line absorption is vertically pressure- and temperature-
> dependent, the scalar relation `A_λ(Γ) ≈ τ_λl′(Γ)` is approximate. We therefore
> interpret the fitted coefficients as band-effective absorber-path cumulants;
> exact geometrical-path moments require closure against layer-resolved photon
> trajectories.

# Spectral transmittance fitting — derivation, assumptions, and code map

This note records the physical hypothesis behind the cumulant/gamma transmittance
fit in [`fitting.py`](fitting.py) and points to where each step is implemented.

The idea: the absorption spectrum within an OCO-2 band is a sampling of the
**Laplace transform of the photon path-length distribution** `p(l')`. Sweeping
wavelength across an absorption line scans the slant optical depth `S̄ŌD̄` from
small to large, tracing out `T(S̄ŌD̄)`. Fitting `ln T` vs. optical depth recovers
the cumulants of `p(l')` — chiefly the mean path enhancement and its variance.

---

## 1. Single photon — Beer–Lambert

For one photon travelling a path of length `L` to the surface:

```
T_λ(L) = exp{ −∫₀ᴸ σ(l) n(l) dl } = exp{ −∫₀ᴸ α(l) dl }
```

with `α(l) = σ(l) n(l)` the absorption coefficient.

---

## 2. Many photons — ensemble average → Laplace transform

Average single-photon transmittance over `N` photons, each with its own path
length `Lᵢ` and absorption profile `αᵢ`:

```
T̄(αᵢ, Lᵢ) = (1/N) Σᵢ exp{ −∫₀^Lᵢ αᵢ(l) dl }
```

**Mean-absorption approximation** — replace the spatially varying `αᵢ(l)` with an
effective constant `ᾱ`, so the only remaining randomness is the path length:

```
≈ (1/N) Σᵢ exp{ −ᾱ Lᵢ } = ∫₀^∞ p(l) exp{ −ᾱ l } dl
```

The sum over photons becomes an integral over the path-length PDF `p(l)`. The
right-hand side is exactly the **Laplace transform of `p(l)`** evaluated at `ᾱ`.

---

## 3. Normalize to the slant optical depth

Let `L̄` be the **direct geometric slant path** to the surface and normalize
`l' = l / L̄` (so `dl' = dl / L̄`). Then `∫₀^Lᵢ ᾱ dl = ᾱ L̄ · lᵢ'`, and defining
the **slant optical depth**

```
S̄ŌD̄ = ᾱ · L̄
```

gives

```
T̄(S̄ŌD̄) = ∫₀^∞ p(l') exp{ −S̄ŌD̄ · l' } dl'
```

`l'` is the **path-length enhancement** relative to the direct slant path
(`l' ≈ 1` with no scattering, `> 1` when scattering lengthens the path), so
`⟨l'⟩` is a genuine free quantity ≥ 1 (not identically 1).

---

## 4. Add surface reflectance

For a Lambertian surface of reflectance `γ`:

```
πI = f₀ cos θ · γ · T̄
⇒  πI / (f₀ cos θ) = γ T̄ = γ ∫₀^∞ p(l') exp{ −S̄ŌD̄ · l' } dl'
```

The measured "transmittance" used in the fit is therefore `γ T̄`, which → `γ` as
`S̄ŌD̄ → 0`.

---

## 5. Gamma assumption for `p(l')`

Assume the path-length enhancement follows a gamma distribution:

```
p(l') = (κ/⟨l'⟩)^κ / Γ(κ) · l'^{κ−1} · exp{ −(κ/⟨l'⟩) l' }     with   κ = ⟨l'⟩² / var(l')
```

Its Laplace transform is closed-form:

```
∫₀^∞ p(l') exp{ −S̄ŌD̄ · l' } dl'  =  1 / (1 + (⟨l'⟩/κ) S̄ŌD̄)^κ
```

so

```
T(S̄ŌD̄) = πI / (f₀ cos θ) = γ / (1 + (⟨l'⟩/κ) S̄ŌD̄)^κ
```

---

## 6. Log + Taylor → cumulant expansion

```
ln T(S̄ŌD̄) = ln γ − κ ln[ 1 + (⟨l'⟩/κ) S̄ŌD̄ ]
```

Expanding `ln(1 + x) = x − x²/2 + x³/3 − …` with `x = (⟨l'⟩/κ) S̄ŌD̄`:

```
ln T = ln γ − [ ⟨l'⟩·S̄ŌD̄ − (var(l')/2)·S̄ŌD̄² + (κ(⟨l'⟩/κ)³/3)·S̄ŌD̄³ − … ]
```

The first two coefficients are exactly the **mean** and **variance** of `p(l')`.

### Coefficient identities

Matching to the code's parameterization
`ln T = intercept + Σᵢ (−1)ⁱ (kᵢ / i) S̄ŌD̄ⁱ` gives, for the gamma model:

```
kₙ = ⟨l'⟩ⁿ / κ^{n−1}
```

| coefficient | meaning                | sign under gamma |
|-------------|------------------------|------------------|
| `intercept` | `ln γ`                 | —                |
| `k1`        | `⟨l'⟩`  (mean)         | > 0              |
| `k2`        | `var(l')`  (variance)  | > 0              |
| `k3 …`      | higher cumulants `⟨l'⟩ⁿ/κⁿ⁻¹` | > 0 (gamma) |

Consequences:
- **κ recovered from the fit:** `κ = ⟨l'⟩² / var(l') = k1² / k2`.
- Under a *strict* gamma the `kₙ` are not independent — they form a geometric
  sequence `kₙ = k1 (k2/k1)^{n−1}`. Fitting them freely (orders 7/3) is therefore
  **more general than gamma**; in that general case only `k1` (mean) and `k2`
  (variance) must be positive — higher cumulants of an arbitrary distribution may
  be negative.

---

## 7. Assumptions (where the model can break)

1. **Mean-absorption** `αᵢ(l) → ᾱ` (Section 2). Valid for a well-mixed absorber
   weakly correlated with the scattering geometry; approximate for real O₂/CO₂.
2. **`L̄` is the direct geometric slant path** (Section 3), so `S̄ŌD̄` is the known
   geometric slant optical depth and `⟨l'⟩ = k1` is a free enhancement ≥ 1.
3. **`p(l')` is wavelength-independent within a band** — required so the channels
   of one band all sample the same Laplace transform.
4. **Gamma form for `p(l')`** (Section 5) — a tractable 2-parameter choice; cannot
   capture a bimodal direct-beam-plus-tail distribution.
5. **Truncation convergence** — the Taylor series only converges for
   `(⟨l'⟩/κ) S̄ŌD̄ < 1`. At line centres (large S̄ŌD̄) the truncated polynomial
   departs from the true curve regardless of order.
6. **Single Lambertian surface reflection** (Section 4).

---

## 8. Code map

All in [`fitting.py`](fitting.py).

| Derivation step | Code |
|-----------------|------|
| `T = γ T̄ = πI/(f₀cosθ)` measured transmittance (Section 4) | `compute_transmittance` — `T = radiances / toa_sol * π`, masks `T > 1` ([fitting.py:408-423](fitting.py#L408-L423)) |
| `ln T` cumulant polynomial, order `N` (Section 6) | `log_transmittance_model_1 … _9` ([fitting.py:46-93](fitting.py#L46-L93)) |
| Order → model dispatch | `LOG_TRANSMITTANCE_MODELS` dict ([fitting.py:125-133](fitting.py#L125-L133)); selected in `fit_spectral_model` ([fitting.py:516](fitting.py#L516)) |
| Closed-form gamma `T(S̄ŌD̄)` (Section 5) | `transmittance_model(tau, l_mean, kappa, intercept)` ([fitting.py:95-102](fitting.py#L95-L102)) — `l_mean = ⟨l'⟩`, `kappa = κ` (unused in active fit) |
| Design matrix `1, −τ, +½τ², …` (Section 6) | `get_design_matrix` ([fitting.py:426-442](fitting.py#L426-L442)) |
| Fit `ln T` vs. `τ` (Section 6) | `fit_spectral_model` — savgol smooth + `curve_fit` ([fitting.py:500-531](fitting.py#L500-L531)) |
| **k1, k2 ≥ 0 bounds** (Section 6 positivity) | `n_pos = min(2, fit_order)`; `lb = [0]*n_pos + [-inf]*…`; passed via `bounds=(lb, ub)` ([fitting.py:526-531](fitting.py#L526-L531)) |
| `S̄ŌD̄ ≡ τ` per channel (Section 3) | `od["tau"]` from `oco_fp_atm_abs`; edge channels dropped `[1:-1]` ([fitting.py:872](fitting.py#L872)) |
| Store k1…k5 + intercept | `kappa_fitting`, `intercept_fitting` → `output_dict` ([fitting.py:886](fitting.py#L886), [fitting.py:1011-1036](fitting.py#L1011-L1036)) |
| **κ = k1²/k2** (Section 6 identity) | `gamma_kappa = kappa_fitting[:,:,0]² / kappa_fitting[:,:,1]`, NaN where `k2 ≤ 0`; written as `{band}_kappa` ([fitting.py:929-932](fitting.py#L929-L932), [fitting.py:1030-1032](fitting.py#L1030-L1032)) |
| Per-band orders `(o2a, wco2, sco2)` | `fit_order = (7, 3, 7)` in `run_simulation` ([fitting.py:1584](fitting.py#L1584)) |

> Note: `transmittance_model` (closed-form gamma) is **not** used in the active
> fit — it appears only in commented-out plotting. The stored `k1…k5` and the
> derived `κ` all come from the polynomial `log_transmittance_model_*` fits.

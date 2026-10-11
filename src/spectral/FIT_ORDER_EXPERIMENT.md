# Cumulant Fit-Order Selection Experiment

**Question:** What polynomial (cumulant) truncation order should the
log-transmittance fit use for each band — O2A, WCO2, SCO2?

**Conclusion:** Keep `fit_order = (7, 3, 7)`. The current setting is
physically justified; do **not** raise the WCO2 order.

- **O2A → 7**
- **WCO2 → 3** (2–3; capped by optical-depth range, not by fit statistics)
- **SCO2 → 7**

---

## Setup

- **Orbit:** `22845a_GL`, date `2018-10-18` (OCO-2 glint mode).
- **Sample:** 300 soundings (of 10 811), every *n*-th valid sounding.
- **Candidate orders:** {1, 2, 3, 4, 5, 7, 9} — the keys of
  `LOG_TRANSMITTANCE_MODELS` in [`fitting.py`](fitting.py).
- **Model:** cumulant expansion of `ln(T)` vs optical depth τ,
  `ln T = -k1·τ + ½k2·τ² - ⅓k3·τ³ + … + intercept`.
- **Procedure:** reused the production functions (`preprocess`,
  `load_orbit_data`, `compute_transmittance`, `fit_spectral_model`) so the
  study matches real fitting exactly — including the Savitzky–Golay smoothing
  (window 51, polyorder 3) applied to `ln(T)` before fitting.
- **Scores per sounding × order:** RSS, adjusted-R², and BIC
  `= n·ln(RSS/n) + k·ln(n)` with `k = order + 1`, plus the fraction of
  soundings where each order minimizes BIC (`%best_BIC`).

---

## Result 1 — Naive BIC on the smoothed curve (MISLEADING)

Scoring RSS/BIC against the *same savgol-smoothed* `ln(T)` that is fit:

| Band | med adjR² @1 | BIC bottoms at | %best_BIC winner |
|------|-------------|----------------|------------------|
| O2A  | 0.9945 | **order 7** (−8054); 9 rises to −8046 | 7 (34%) ≈ 9 (33%) |
| WCO2 | 0.9981 | order 9 (−10996), still falling | 9 (**95%**) |
| SCO2 | 0.9989 | order 9 (−8119), still falling | 9 (**84%**) |

Taken at face value this suggests `(7, 9, 9)`. **That is wrong for WCO2** — see
below. adjusted-R² is useless here (saturated ≥0.99 from order 1); only BIC and
`%best_BIC` carry signal, and even those are biased when the τ range is small.

## Result 2 — Optical-depth (SOD) range per band (DECISIVE)

Per-sounding **maximum** τ, edge channels dropped (as the fitter does):

| Band | median max τ | 99th-pct max τ | global max | dynamic range |
|------|-------------|----------------|-----------|---------------|
| O2A  | 5.32 | 8.44 | 8.75 | wide |
| WCO2 | **0.66** | **1.09** | **1.12** | **tiny** |
| SCO2 | 3.96 | 5.02 | 5.12 | wide |

WCO2 optical depth barely reaches ~1.

---

## Interpretation

Over τ ∈ [0, ~0.7] (WCO2) the high-order cumulant terms are negligible and
mutually collinear — τ⁷ ≈ 0.08, τ⁹ ≈ 0.04 — so a 9th-order fit has ~7 free
parameters chasing wiggles in the **smoothed** curve rather than physical
curvature. That is exactly why BIC "kept falling" to order 9 for WCO2: it was
rewarding flexibility against smoothed residual, not signal. The τ dynamic range
physically caps the number of identifiable cumulants at ~2–3 for WCO2.

O2A (τ → 8.7) and SCO2 (τ → 5) have wide range that genuinely supports high
order. O2A's BIC turned back **up** at 9 → a real plateau at 7. SCO2's edge of 9
over 7 was marginal and driven by a handful of deep line-core channels, so 7 is
the safe, defensible choice.

**Lesson / gotcha:** an order-selection criterion that fits the savgol-smoothed
`ln(T)` and scores BIC on that same smoothed curve over-rewards flexibility for
**low-τ-range** bands. To settle O2A/SCO2 7-vs-9 honestly, use a range-aware or
held-out criterion (fit on the smoothed curve, score on unsmoothed held-out
channels, or cap candidate order by τ dynamic range). Not pursued because 7 is
already the safe pick and WCO2's answer is unambiguous from the τ range alone.

---

## Reproduce

Study scripts live in the session scratchpad (not committed):
`order_study.py` (per-order BIC/adjR²/`%best_BIC`) and `tau_range.py`
(per-band SOD percentiles). Both import directly from
[`fitting.py`](fitting.py) and require the orbit's `fp_tau_combined.h5`
(built by `preprocess`) plus `results_2018-10-18.h5`.

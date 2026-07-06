"""sidea_uncertainty.py — Side-A (ML-corrected OCO-2) per-footprint and per-case
uncertainty for the uncertainty-aware TCCON comparison.

Phase-2 verdict (src/analysis/UNCERTAINTY_AWARE_TCCON_COMPARISON.md §10):
  SIDE_A_BUDGET = de_total (DE predictive σ), NOT de_total ⊕ retrieval.
  de_total is near-calibrated out-of-sample but UNDER-confident near clouds, so a
  cloud-distance inflation k(cld_dist) is applied:

    σ_footprint = k(cld_dist) · de_sigma          (k≈1.0 far → ~1.13 ocean /
                                                   ~1.27 land at 0 km)

Case-level (correlated epistemic floor + averaged predictive term):

    u_case² = Var_m(x̄_m)  +  Σ_k (k_k · de_sigma_k)² / (N · N_eff)

  Var_m(x̄_m) is the spread of the per-member CASE MEAN — the epistemic part that
  does NOT average down (all footprints share the M members).  It needs the
  per-member μ (columns mu_00…); when absent, a documented fully-correlated
  fallback (mean-per-footprint epistemic)² is used (a mild over-estimate).

k(cld_dist) is fit from the pooled out-of-fold `held_out_predictions.parquet`
(k = sqrt(⟨z²⟩) per cloud-distance bin) and cached to JSON per surface.
"""
from __future__ import annotations

import glob as _glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

# cloud-distance bin edges (km) for the k fit; centers used for interpolation.
_K_EDGES = [0, 2, 5, 10, 20, 35, 50, 1e9]
_FAR_CENTER = 75.0   # nominal center for the open last bin


def fit_inflation(held_out_glob, max_abs_ppm=25.0):
    """Fit k(cld_dist) = sqrt(⟨z²⟩) per cloud-distance bin from out-of-fold rows.

    Returns dict(centers=[km], k=[factor]) with the far field pinned so that
    np.interp clamps to the last measured k beyond 50 km.
    """
    paths = sorted(_glob.glob(held_out_glob))
    if not paths:
        raise SystemExit(f"no held-out parquet matched {held_out_glob!r}")
    df = pd.concat([pd.read_parquet(p, columns=['y_true', 'mu', 'sigma', 'cld_dist_km'])
                    for p in paths], ignore_index=True)
    df = df[(df['y_true'].abs() <= max_abs_ppm) & (df['mu'].abs() <= max_abs_ppm)]
    z = (df['y_true'] - df['mu']) / df['sigma']
    lab = pd.cut(df['cld_dist_km'], bins=_K_EDGES, include_lowest=True)
    centers, ks = [], []
    for iv, s in z.groupby(lab, observed=True):
        z2 = float(np.mean(s.to_numpy() ** 2))
        lo, hi = iv.left, iv.right
        c = _FAR_CENTER if not np.isfinite(hi) else 0.5 * (max(lo, 0) + hi)
        centers.append(float(c)); ks.append(float(np.sqrt(z2)))
    order = np.argsort(centers)
    return dict(centers=[centers[i] for i in order], k=[ks[i] for i in order])


class InflationModel:
    """k(cld_dist) lookup with clamped linear interpolation."""
    def __init__(self, centers, k):
        self.centers = np.asarray(centers, float)
        self.k = np.asarray(k, float)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d['centers'], d['k'])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(
            {'centers': self.centers.tolist(), 'k': self.k.tolist()}, indent=2))

    def __call__(self, cld_dist_km):
        d = np.asarray(cld_dist_km, float)
        return np.interp(d, self.centers, self.k)   # clamps outside range


def calibrated_sigma(de_sigma, cld_dist_km, infl):
    """Per-footprint calibrated Side-A σ = k(cld_dist) · de_sigma."""
    return infl(cld_dist_km) * np.asarray(de_sigma, float)


def calibrated_sigma_by_surface(df, infl_by_sfc):
    """Per-footprint calibrated σ using a SEPARATE inflation model per surface.

    infl_by_sfc : {sfc_type -> InflationModel}, e.g. {0: ocean, 1: land}.  Each
    footprint's de_sigma is inflated by the k(cld_dist) fit for its own surface —
    the k curves differ (land is more under-confident near clouds).  Footprints
    whose sfc_type has no model fall back to k=1 (raw de_sigma).
    """
    de_sigma = df['de_sigma'].to_numpy(float)
    cld = df['cld_dist_km'].to_numpy(float)
    sfc = df['sfc_type'].to_numpy() if 'sfc_type' in df.columns else np.zeros(len(df))
    out = de_sigma.copy()
    for s, infl in infl_by_sfc.items():
        m = sfc == s
        if m.any():
            out[m] = infl(cld[m]) * de_sigma[m]
    return out


def _n_eff(lon, lat, decorr_km=15.0):
    """Effective independent count for spatially-correlated footprints.

    Blocks the footprints onto a ~decorr_km grid and counts occupied cells —
    a simple, documented proxy for the number of independent samples (adjacent
    OCO-2 soundings are ~2 km apart and correlated over tens of km, so raw N
    badly over-counts).  N_eff is capped to [1, N].
    """
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    m = np.isfinite(lon) & np.isfinite(lat)
    if m.sum() == 0:
        return 1
    lat0 = np.nanmean(lat[m])
    kx = 111.32 * np.cos(np.radians(lat0))   # km per deg lon
    ky = 111.32                              # km per deg lat
    ix = np.floor(lon[m] * kx / decorr_km).astype(np.int64)
    iy = np.floor(lat[m] * ky / decorr_km).astype(np.int64)
    n_cells = len({(a, b) for a, b in zip(ix, iy)})
    return int(min(max(n_cells, 1), int(m.sum())))


def case_uncertainty(df, infl=None, *, sigma_fp=None, use_members=True,
                     member_prefix='mu_', decorr_km=15.0):
    """Per-case Side-A uncertainty u_oco (ppm) for one case's footprint frame.

    df must carry de_sigma, cld_dist_km, lon, lat (+ optional mu_00… per-member
    columns for the exact epistemic term).  Returns dict with u_oco and parts.

    infl      : InflationModel for the (single-surface) frame; ignored when
                sigma_fp is given directly (use for a mixed-surface frame via
                calibrated_sigma_by_surface()).
    sigma_fp  : precomputed per-footprint calibrated σ (overrides infl).
    use_members : when False, force the fully-correlated epistemic fallback even
                if mu_NN columns are present — REQUIRED for a pooled multi-surface
                frame, where per-surface members are different models and the
                per-member case mean is not coherent.
    """
    N = len(df)
    if sigma_fp is not None:
        sig = np.asarray(sigma_fp, float)
    else:
        if infl is None:
            raise ValueError("case_uncertainty needs infl or sigma_fp")
        sig = calibrated_sigma(df['de_sigma'].to_numpy(float),
                               df['cld_dist_km'].to_numpy(float), infl)
    neff = _n_eff(df['lon'], df['lat'], decorr_km)

    # averaged predictive term (aleatoric + within-case): Σ σ² / (N·N_eff)
    avg_var = float(np.nansum(sig ** 2) / (N * neff)) if N else np.nan

    # epistemic case term Var_m(x̄_m)
    mem_cols = sorted(c for c in df.columns
                      if c.startswith(member_prefix) and c[len(member_prefix):].isdigit())
    if mem_cols and use_members:
        M = df[mem_cols].to_numpy(float)                 # [N, M]
        xbar_m = np.nanmean(M, axis=0)                   # per-member case mean
        epi_var = float(np.nanvar(xbar_m))
        epi_src = f'exact ({len(mem_cols)} members)'
    else:
        # fully-correlated fallback: (mean per-footprint epistemic)²  [over-estimate]
        epi = df['de_epistemic_sigma'].to_numpy(float) if 'de_epistemic_sigma' in df.columns \
            else np.zeros(N)
        epi_var = float(np.nanmean(epi) ** 2)
        epi_src = 'fallback (fully-correlated)'

    u = float(np.sqrt(max(epi_var + avg_var, 0.0)))
    return dict(u_oco=u, epi_sigma=float(np.sqrt(max(epi_var, 0))),
                avg_sigma=float(np.sqrt(max(avg_var, 0))),
                N=N, N_eff=neff, epi_src=epi_src)


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Fit + cache k(cld_dist) per surface.")
    ap.add_argument('--held-out-glob', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    d = fit_inflation(args.held_out_glob)
    InflationModel(d['centers'], d['k']).save(args.out_json)
    print(f"  k(cld_dist) centers={['%.0f'%c for c in d['centers']]}")
    print(f"              k      ={['%.3f'%k for k in d['k']]}")
    print(f"  [saved] {args.out_json}")


if __name__ == '__main__':
    _main()

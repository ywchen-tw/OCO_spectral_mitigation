"""Effect of a wavelength-polynomial continuum (L2-style linear/quadratic albedo per band)
on the cumulant fit, Doppler-corrected τ.

Models per sounding and band (x = channel index scaled to [−1, 1]; a λ polynomial is
equivalent to a channel-index polynomial at these orders):
    S2   : ln T = c − k1 τ + ½ k2 τ²                       (+ a1 x  [S2x], + a1 x + a2 x²  [S2xx])
    S3   : S2 − ⅓ k3 τ³, unconstrained                     (S3x, S3xx)
    T2   : c − k1_dry τ_dry − k1_w τ_h2o + ½ k2 τ²         (T2x, T2xx), τ_h2o = frac_h2o · τ
Reported: k1 (k1_dry for T2), k2, rms, a1, a2, design condition number.

Usage: python alb_poly_test.py GL     (29252a_GL, all soundings, local corrected τ + H2O split)
       python alb_poly_test.py TG     (48443a_TG, 100-sounding Doppler-variant smoke run, stored frac_h2o)
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
from pathlib import Path
import h5py, numpy as np, pandas as pd
from cut_sweep_real import DEFAULTS
from orbit_io import load_orbit, BANDS

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SMOKE = Path("/private/tmp/claude-370367/-Volumes-YWC-SSD-programming-oco-fp-analysis/106fa6df-e817-4e62-8f1a-c441f33123ad/scratchpad/smoke/dop")
_G = {}


def _fit(cols, y):
    a = np.column_stack(cols)
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    rms = float(np.sqrt(np.mean((y - a @ c) ** 2)))
    cond = float(np.linalg.cond(a / np.linalg.norm(a, axis=0)))
    return c, rms, cond


def fit_sounding(j):
    rows = []
    x_all = (np.arange(1016) - 507.5) / 507.5
    for ib, b in enumerate(BANDS):
        t = _G["tau"][ib, j].astype(float); y = _G["ln_t"][ib, j].astype(float); s = _G["sig"][ib, j].astype(float)
        tw = _G["tau_h2o"][ib, j].astype(float)
        ok = np.isfinite(y) & np.isfinite(t) & np.isfinite(s) & (s > 0) & np.isfinite(tw)
        if ok.sum() < 20:
            continue
        t, y, tw, x = t[ok], y[ok], tw[ok], x_all[ok]
        one = np.ones_like(t)
        base = {"S2": [one, -t, 0.5 * t**2], "S3": [one, -t, 0.5 * t**2, -t**3 / 3],
                "T2": [one, -(t - tw), -tw, 0.5 * t**2]}
        for name, cols in base.items():
            for suffix, extra in (("", []), ("x", [x]), ("xx", [x, x**2])):
                c, rms, cond = _fit(cols + extra, y)
                k1 = c[1]; k2 = c[3] if name == "T2" else c[2]
                na = len(cols)
                rows.append(dict(j=j, band=b, model=name + suffix, k1=k1, k2=k2, rms=rms, cond=cond,
                                 k1_w=c[2] if name == "T2" else np.nan,
                                 a1=c[na] if extra else np.nan, a2=c[na + 1] if len(extra) == 2 else np.nan))
    return rows


def _chunk(js):
    return [r for j in js for r in fit_sounding(j)]


def main(which):
    if which == "GL":
        orb = load_orbit(str(DEFAULTS["l1b"]), str(DEFAULTS["fp_tau"]), str(DEFAULTS["fit_details"]))
        zt = np.load(OUT / "resid_doppler_tau_29252a_GL.npz")
        z = np.load(OUT / "h2o_tau_split_29252a_GL.npz")
        tau = np.transpose(zt["tau"], (1, 0, 2)).astype(np.float32)          # [3, N, 1016]
        tau_h2o = np.stack([z[f"tau_h2o_{b}"] for b in BANDS]).astype(np.float32)
        tag = "29252a_GL"
    else:
        l1b = str(HERE.parents[1] / "data/OCO2/2023/222/48443a_TG/oco2_L1bScTG_48443a_230810_B11008r_230912173118.h5")
        orb = load_orbit(l1b, str(SMOKE / "2023-08-10/48443a_TG/fp_tau_combined.h5"),
                         str(SMOKE / "fitting_details/fitting_details_2023-08-10_48443a_TG.h5"))
        with h5py.File(SMOKE / "2023-08-10/48443a_TG/fp_tau_combined.h5") as f:
            frac = np.stack([f[f"{b}_frac_h2o"][...] for b in BANDS]).astype(np.float32)
        tau = orb.tau                                                        # corrected τ from the variant
        tau_h2o = frac * tau
        tag = "48443a_TG_smoke100"
    _G.update(tau=tau, ln_t=orb.ln_t, sig=orb.sigma_ln_t, tau_h2o=tau_h2o)
    n = tau.shape[1]
    chunks = np.array_split(np.arange(n), max(1, n // 50))
    with mp.get_context("fork").Pool(8) as pool:
        rows = [r for part in pool.imap(_chunk, chunks) for r in part]
    df = pd.DataFrame(rows)
    meta = pd.DataFrame({k: v for k, v in orb.meta.items() if k in ("cld_dist_km", "xco2_qf", "fp_number", "lat", "sfc_type")})
    meta["j"] = np.arange(n)
    df = df.merge(meta, on="j", how="left")
    df.to_csv(OUT / f"alb_poly_{tag}.csv", index=False)
    print(f"{tag}: {n} soundings, {len(df)} rows -> results/alb_poly_{tag}.csv")
    return df


if __name__ == "__main__":
    main(sys.argv[1])

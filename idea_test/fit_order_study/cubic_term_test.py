"""Does a third-order (k3) term, free or tied to (k1, k2) by the gamma relation, fix the sign of k2?

Variants on the agreed CO2-band design D = [1, x, (x² land), −τ_dry, −τ_h2o]:
  A  agreed      : D + ½k2τ² (SCO2); D only (WCO2)                      [k2 free / 0]
  A2 order 2 free: D + ½k2τ²                                            [k2 free, both bands]
  B  order 3 free: D + ½k2τ² − (1/6)k3τ³                                [k2, k3 free]
  C  gamma-tied  : D + ½k2τ² − (1/3)(k2²/k1)τ³, k2 ≥ 0                  [k3 = 2k2²/k1, nonlinear]
  D  bounded     : D + ½k2τ², k2 ≥ 0                                    [bounded linear]
Ocean 29252a_GL far-clear QF0 subsample and Kozienice 38200a_ND (background QF0 + plume soundings).
Reports k1 median, far-clear/background spread after per-footprint medians, k2/k3 sign fractions, rms,
and the Kozienice plume Δk1.  Outputs results/cubic_term_test.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_model_test as fm  # noqa: E402
from final_model_test import X_ALL  # noqa: E402
from o2a_anomaly_estimator import rsd  # noqa: E402

OUT = HERE / "results"
IB = {"wco2": 1, "sco2": 2}


def design(td, tw, x, land, b):
    cols = [np.ones_like(td), x] + ([x**2] if (land and b == "sco2") else []) + [-td, -tw]
    return np.column_stack(cols), 3 if (land and b == "sco2") else 2


def fit_variants(td, tw, x, y, land, b):
    t = td + tw
    D, kc = design(td, tw, x, land, b)
    out = {}
    # A2 / A
    c = np.linalg.lstsq(np.column_stack([D, 0.5 * t**2]), y, rcond=None)[0]
    r2 = np.sqrt(np.mean((y - np.column_stack([D, 0.5 * t**2]) @ c) ** 2))
    out["A2 order2 free"] = dict(k1=c[kc], k2=c[-1], k3=np.nan, rms=r2)
    if b == "wco2":
        c0 = np.linalg.lstsq(D, y, rcond=None)[0]
        out["A agreed"] = dict(k1=c0[kc], k2=0.0, k3=np.nan, rms=np.sqrt(np.mean((y - D @ c0) ** 2)))
    else:
        out["A agreed"] = out["A2 order2 free"]
    # B
    A3 = np.column_stack([D, 0.5 * t**2, -t**3 / 6])
    c = np.linalg.lstsq(A3, y, rcond=None)[0]
    out["B order3 free"] = dict(k1=c[kc], k2=c[-2], k3=c[-1], rms=np.sqrt(np.mean((y - A3 @ c) ** 2)))
    # D bounded k2 >= 0
    A2 = np.column_stack([D, 0.5 * t**2])
    lb = np.full(A2.shape[1], -np.inf); lb[-1] = 0.0
    res = lsq_linear(A2, y, bounds=(lb, np.inf))
    out["D order2 k2>=0"] = dict(k1=res.x[kc], k2=res.x[-1], k3=np.nan, rms=np.sqrt(np.mean((y - A2 @ res.x) ** 2)))
    # C gamma-tied cubic, k2 >= 0
    def resid(p):
        lin, k2 = p[:-1], p[-1]
        k1 = lin[kc]
        return D @ lin + 0.5 * k2 * t**2 - (k2**2 / max(k1, 1e-6)) * t**3 / 3 - y
    p0 = np.r_[res.x[:-1], max(res.x[-1], 1e-4)]
    lo = np.full(len(p0), -np.inf); lo[-1] = 0.0; lo[kc] = 0.0
    sol = least_squares(resid, p0, bounds=(lo, np.inf), max_nfev=200)
    k1, k2 = sol.x[kc], sol.x[-1]
    out["C gamma-tied k3"] = dict(k1=k1, k2=k2, k3=2 * k2**2 / max(k1, 1e-6), rms=np.sqrt(np.mean(sol.fun**2)), ok=sol.success)
    return out


def run(tau, tau_h2o, ln_t, sig, land, idx, label):
    rows = []
    for j in idx:
        for b, ib in IB.items():
            t, tw, y, s = tau[ib, j].astype(float), tau_h2o[ib, j].astype(float), ln_t[ib, j].astype(float), sig[ib, j].astype(float)
            ok = np.isfinite(y) & np.isfinite(t) & np.isfinite(s) & (s > 0) & np.isfinite(tw)
            if ok.sum() < 20:
                continue
            for v, r in fit_variants(t[ok] - tw[ok], tw[ok], X_ALL[ok], y[ok], bool(land[j]), b).items():
                rows.append(dict(set=label, j=j, band=b, variant=v, **r))
    return pd.DataFrame(rows)


def summarize(df, fp, groups, lines):
    for b in IB:
        lines += [f"### {b}", "", "| variant | " + " | ".join(f"{g}: k1 median / spread % / k2 median / k2<0 or at 0 % / k3 median / rms" for g in groups) + (" | plume Δk1 % (all / QF0)" if "plume" in groups else "") + " |",
                  "|---|" + "---|" * (len(groups) + (1 if "plume" in groups else 0))]
        for v in ("A agreed", "A2 order2 free", "B order3 free", "D order2 k2>=0", "C gamma-tied k3"):
            cells = []
            base = None
            for g, m in groups.items():
                d = df[(df.band == b) & (df.variant == v) & df.j.isin(np.flatnonzero(m))]
                if len(d) == 0:
                    cells.append("—"); continue
                k1 = d.k1.values
                fpm = pd.Series(k1).groupby(fp[d.j.values]).transform("median").values
                spread = 100 * rsd(k1 - fpm) / np.median(k1)
                neg = 100 * np.mean(d.k2.values <= 1e-9) if v in ("D order2 k2>=0", "C gamma-tied k3") else 100 * np.mean(d.k2.values < 0)
                cells.append(f"{np.median(k1):.4f} / {spread:.2f} / {np.median(d.k2):+.3f} / {neg:.0f} / {np.nanmedian(d.k3):+.3f} / {np.median(d.rms):.4f}")
                if g in ("background", "far-clear"):
                    base = np.median(k1)
            if "plume" in groups:
                dp = df[(df.band == b) & (df.variant == v) & df.j.isin(np.flatnonzero(groups["plume"]))]
                dq = df[(df.band == b) & (df.variant == v) & df.j.isin(np.flatnonzero(groups["plume QF0"]))]
                cells.append(f"{100*(np.median(dp.k1)/base-1):+.2f} / {100*(np.median(dq.k1)/base-1):+.2f}")
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        lines.append("")


def main():
    lines = ["# Third-order term and the sign of k2 (agreed CO2-band design)", ""]
    # Kozienice
    import kozienice_pseudo_xco2 as kz
    z = np.load(OUT / f"{kz.TAG}_spectra.npz", allow_pickle=True)
    df = pd.read_parquet(OUT / f"{kz.TAG}_per_sounding.parquet")
    land = df.land.values
    bg = ((df.qf == 0) & (df.dist_km >= 40)).values
    bgm = df.xco2_bc[bg].median()
    plume = ((df.xco2_bc - bgm > 2) & (df.dist_km < 30)).values
    idx = np.flatnonzero(bg | plume)
    res = run(z["tau"], z["tau_h2o"], z["ln_t"], z["sig"], land, idx, "kozienice")
    lines += ["## Kozienice 38200a_ND (land): background QF0 ≥ 40 km (n 254) and plume soundings (n 14, QF0 4)", ""]
    summarize(res, df.fp.values, {"background": bg, "plume": plume, "plume QF0": plume & (df.qf == 0).values}, lines)
    # ocean
    orb, tau, tau_h2o, meta, tag = fm.load("GL")
    far = (meta["cld_dist_km"] > 20) & (meta["xco2_qf"] == 0)
    idx = np.flatnonzero(far)[::2]
    land_o = meta["sfc_type"] == 1
    res_o = run(tau, tau_h2o, orb.ln_t, orb.sigma_ln_t, land_o, idx, "ocean")
    lines += [f"## 29252a_GL (ocean): far-clear QF0 subsample (n {len(idx)})", ""]
    summarize(res_o, meta["fp_number"].astype(int), {"far-clear": far}, lines)
    pd.concat([res, res_o]).to_csv(OUT / "cubic_term_test.csv", index=False)
    (OUT / "cubic_term_test.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

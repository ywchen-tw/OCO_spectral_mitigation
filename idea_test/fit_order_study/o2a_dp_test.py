"""Does the O2A path coefficient of the agreed model track the L2 surface-pressure adjustment?

L2 retrieves psurf from the O2A band (4 hPa prior sigma); dp = psurf − psurf_prior scales the O2 column
by dp/psurf.  The agreed-model O2A k1 scales the O2A optical depth built with the prior psurf, so
k1 − 1 should follow dp/psurf if the PPDF fit sees what L2 sees.  Tested on 29252a_GL (ocean) and the
Kozienice orbit 38200a_ND (land) with the production single-τ k1 and a plain single-τ fit as references.
A surface-layer kernel (Δk1 for air added at the surface vs the same amount spread uniformly, linearised
about the fitted floor model) gives the slope expected for a pure psurf change.
Outputs results/o2a_dp_test.{md,csv,png}.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import h5py  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import resid_attrs as ra  # noqa: E402
import kozienice_pseudo_xco2 as kz  # noqa: E402
from final_model_test import DEFAULTS, X_ALL  # noqa: E402

OUT = HERE / "results"
EPS = 0.01
C_LIGHT = 299792458.0


# ----------------------------------------------------------------------------- data assembly
def ocean_table() -> pd.DataFrame:
    with h5py.File(DEFAULTS["fit_details"], "r") as f:
        fpk = "fp_number" if "fp_number" in f else "fp_id"
        d = {k: f[k][...].astype(float) for k in ("dp", "dp_o2a", "psfc", "psurf_met", "o2a_k1_fitting_nosg",
                                                  "xco2_qf", "cld_dist_km", "lat", "xco2_raw", "xco2_apriori",
                                                  "fs_rel", "aod_total", fpk)}
    df = pd.DataFrame(d).rename(columns={fpk: "fp"})
    df["psurf_prior"] = df.psurf_met / 100.0
    df["dp_frac_pct"] = 100 * df.dp / df.psurf_prior
    ag = pd.read_parquet(OUT / "final_model_per_sounding_29252a_GL.parquet")
    for b in ("o2a", "wco2", "sco2"):
        s = ag[ag.band == b].set_index("j")
        df[f"k1_{b}"] = s.k1.reindex(df.index).values
    df["f_o2a"] = ag[ag.band == "o2a"].set_index("j").f.reindex(df.index).values
    h2 = pd.read_parquet(OUT / "h2o_per_sounding_29252a_GL.parquet")
    df["k1plain_o2a"] = h2[h2.band == "o2a"].set_index("j").S2_k1.reindex(df.index).values
    df["k1prod_o2a"] = df.o2a_k1_fitting_nosg
    df["land"] = False
    df["qf"] = df.xco2_qf
    df["far"] = df.cld_dist_km > 20
    return df


def land_table() -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"{kz.TAG}_per_sounding.parquet")
    with h5py.File(kz.FILES["lite"], "r") as f:
        lsid = f["sounding_id"][...].astype(np.int64)
        pos = pd.Series(np.arange(len(lsid)), index=lsid)
        idx = pos.reindex(df.sounding_id.values).values
        ok = np.isfinite(idx)
        ii = idx[ok].astype(int)
        for k, src in (("dp", "Retrieval/dp"), ("dp_o2a", "Retrieval/dp_o2a"), ("psurf", "Retrieval/psurf"),
                       ("psurf_prior", "Retrieval/psurf_apriori"), ("aod_total", "Retrieval/aod_total")):
            v = np.full(len(df), np.nan)
            v[ok] = f[src][...][ii]
            df[k] = v
    df["dp_frac_pct"] = 100 * df.dp / df.psurf_prior
    df["k1prod_o2a"] = np.nan                      # no production fit for this orbit
    df["far"] = df.dist_km >= 40                   # background, as in the Kozienice test
    return df


# ----------------------------------------------------------------------------- statistics
def _demean(v, groups):
    s = pd.Series(v)
    return (s - s.groupby(np.asarray(groups)).transform("mean")).values


def stats(x, y, fp, lat=None):
    m = np.isfinite(x) & np.isfinite(y)
    x, y, fp = x[m], y[m], np.asarray(fp)[m]
    if m.sum() < 10:
        return dict(n=int(m.sum()))
    r = np.corrcoef(x, y)[0, 1]
    slope = np.polyfit(x, y, 1)[0]
    xd, yd = _demean(x, fp), _demean(y, fp)
    rd = np.corrcoef(xd, yd)[0, 1]
    sd = np.polyfit(xd, yd, 1)[0]
    res = y - x
    return dict(n=int(m.sum()), corr=r, slope=slope, corr_fp=rd, slope_fp=sd, sd_x=x.std(), sd_y=y.std(),
                sd_resid_unit=res.std(), sd_resid_fit=(y - np.polyval(np.polyfit(x, y, 1), x)).std(),
                r2=r**2, mean_x=x.mean(), mean_y=y.mean())


def subsets(df):
    qf0 = df.qf == 0
    out = [("QF0 far-clear", qf0 & df.far), ("QF0 all", qf0), ("QF1 (retrieved)", df.qf == 1)]
    if "cld_dist_km" in df:
        out.append(("QF0+1 cld < 3 km", np.isfinite(df.qf) & (df.cld_dist_km < 3)))
    return out


def evaluate(df, label):
    rows = []
    x = df.dp_frac_pct.values
    for name, m in subsets(df):
        d = df[m]
        for coef, col in (("agreed k1_O2A", "k1_o2a"), ("plain k1_O2A", "k1plain_o2a"),
                          ("production k1_O2A", "k1prod_o2a"), ("agreed k1_WCO2", "k1_wco2"),
                          ("agreed k1_SCO2", "k1_sco2"), ("agreed SCO2/O2A", None)):
            y = (d.k1_sco2 / d.k1_o2a).values if col is None else d[col].values
            s = stats(x[m], 100 * (y - 1), d.fp.values)
            rows.append(dict(dataset=label, subset=name, coef=coef, **s))
        s = stats(x[m], 100 * d.f_o2a.values, d.fp.values)
        rows.append(dict(dataset=label, subset=name, coef="agreed f_O2A (×100)", **s))
        pri = d.xco2_prior_cpr.values if "xco2_prior_cpr" in d else d.xco2_apriori.values
        s = stats(x[m], 100 * (d.xco2_raw.values / pri - 1), d.fp.values)
        rows.append(dict(dataset=label, subset=name, coef="L2 xco2_raw/prior − 1", **s))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- surface-layer kernel (O2A)
def o2a_kernel(i, ln_t, sig, k1, f):
    """Δk1 per unit relative O2 column increase, air added at the surface vs uniformly; floor-model Jacobian."""
    atm, b = ra._G["atm"], ra._G["bands"][0]
    fp = int(ra._G["fp"][i])
    airmass = 1 / np.cos(np.deg2rad(ra._G["sza"][i])) + 1 / np.cos(np.deg2rad(ra._G["vza"][i]))
    beta = ra._G["sign"] * atm["v_inst"][i] / C_LIGHT
    u, wq = np.polynomial.legendre.leggauss(ra.N_QUAD)
    u = 0.5 * (u + 1)
    s_nodes, s_w = u**2, wq * u
    dz = atm["dz"][i]
    e_gas, e_h2o = ra._ext_layers(b, atm, i, atm["o2"][i])
    ray = (b["ray_xsec"][:, None] * 1e-28 * atm["air"][i] * dz * 1e5).sum(axis=1)
    col = (atm["o2"][i] * dz)
    low = np.argmax(atm["p"][i])                       # highest-pressure layer = surface layer
    comp = np.stack([(e_gas * dz).sum(axis=1), (e_h2o * dz).sum(axis=1), ray,
                     e_gas[:, low] * dz[low] * col.sum() / col[low]]) * airmass
    tau_hr = comp[:3].sum(axis=0)
    wl, xx, yy = b["wl"][fp], b["xx"][fp], b["yy"][fp]
    wv = b["wvldat"] * (1.0 + beta)
    start, end = ra.conv_indices(wl, xx, wv)
    xw = xx + wl[:, None]
    tau = np.full(1016, np.nan); d_col = np.full(1016, np.nan); d_sfc = np.full(1016, np.nan)
    for c in range(len(wl)):
        s, e = start[c], end[c]
        w = np.interp(wv[s:e], xw[c], yy[c])
        th = tau_hr[s:e]
        ex = np.exp(-np.outer(s_nodes, th - th.min())) * w
        p_bar = s_w @ (ex / ex.sum(axis=1, keepdims=True))
        v = comp[:, s:e] @ p_bar
        tau[c], d_col[c], d_sfc[c] = v[:3].sum(), v[0], v[3]
    y, sg = ln_t[0, i].astype(float), sig[0, i].astype(float)
    ok = np.isfinite(y) & np.isfinite(tau) & np.isfinite(sg) & (sg > 0)
    t, x = tau[ok], X_ALL[ok]
    g = (1 - f) * np.exp(-k1 * t)
    w = g / (g + f)                                     # ∂lnT/∂(k1 τ) = −w
    jac = np.column_stack([np.ones(ok.sum()), x, -t * w, (1 - np.exp(-k1 * t)) / (g + f)])
    out = {}
    for name, d in (("col", d_col), ("sfc", d_sfc)):
        dy = -k1 * w * EPS * d[ok]
        out[name] = np.linalg.lstsq(jac, dy, rcond=None)[0][2] / (k1 * EPS)
    return dict(j=i, dk1_col=out["col"], dk1_sfc=out["sfc"], kernel_sfc=out["sfc"] / out["col"])


def _kwork(a):
    return o2a_kernel(*a)


def land_kernel(df, n=24):
    z = np.load(OUT / f"{kz.TAG}_spectra.npz", allow_pickle=True)
    ln_t, sig = z["ln_t"], z["sig"]
    sel = kz.select_soundings()
    for k in ("l1b", "met", "cpr"):
        setattr(ra, k.upper(), kz.FILES[k])
    bands, h2o_meta = ra.load_tables()
    atm = ra.build_atmosphere(sel["sid"])
    ra._G.update(bands=bands, h2o_meta=h2o_meta, atm=atm, fp=sel["fp"], sza=sel["sza"], vza=sel["vza"],
                 sign=kz.DOPPLER_SIGN)
    rng = np.random.default_rng(1)
    cand = np.flatnonzero((df.qf == 0).values & np.isfinite(df.k1_o2a.values))
    pick = rng.choice(cand, min(n, len(cand)), replace=False)
    jobs = [(int(i), ln_t, sig, float(df.k1_o2a[i]), float(df.f_o2a[i])) for i in pick]
    with mp.get_context("fork").Pool(8) as pool:
        return pd.DataFrame(pool.map(_kwork, jobs))


# ----------------------------------------------------------------------------- main
def main():
    oc, ld = ocean_table(), land_table()
    res = pd.concat([evaluate(oc, "29252a_GL ocean"), evaluate(ld, "38200a_ND Kozienice land")], ignore_index=True)
    res.to_csv(OUT / "o2a_dp_test.csv", index=False)
    ker = land_kernel(ld)
    ker.to_csv(OUT / "o2a_dp_kernel.csv", index=False)

    lines = ["# O2A path coefficient vs L2 surface-pressure adjustment", "",
             "x = 100·dp/psurf_prior (%), dp = L2 retrieved − prior surface pressure (Lite `dp`). y = 100·(k1 − 1) (%).",
             "`slope` = OLS of y on x (1 = k1 follows the O2 column change one-to-one); `_fp` after removing per-footprint",
             "means; `sd_resid_unit` = SD(y − x); `sd_resid_fit` = SD about the OLS line.", ""]
    for ds, g in res.groupby("dataset", sort=False):
        lines += [f"## {ds}", "", "| subset | coefficient | n | corr | slope | corr_fp | slope_fp | SD x | SD y | SD(y−x) | SD resid | r² |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            if r.n < 10 or not np.isfinite(r.get("corr", np.nan)):
                lines.append(f"| {r.subset} | {r.coef} | {r.n} | — | | | | | | | | |")
                continue
            v = r.to_dict()
            lines.append(f"| {v['subset']} | {v['coef']} | {v['n']} | {v['corr']:+.2f} | {v['slope']:+.2f} | {v['corr_fp']:+.2f} | "
                         f"{v['slope_fp']:+.2f} | {v['sd_x']:.2f} | {v['sd_y']:.2f} | {v['sd_resid_unit']:.2f} | "
                         f"{v['sd_resid_fit']:.2f} | {v['r2']:.2f} |")
        lines.append("")
    q = np.percentile(ker.kernel_sfc, [10, 90])
    lines += ["## Surface-layer kernel of the O2A k1 (Kozienice, QF0 soundings)", "",
              f"n = {len(ker)}: Δk1 per unit relative column for a uniform increase {ker.dk1_col.median():.3f}, "
              f"for the same amount added at the surface {ker.dk1_sfc.median():.3f}; kernel {ker.kernel_sfc.median():.2f} "
              f"[{q[0]:.2f}, {q[1]:.2f}]. A pure psurf change of dp/psurf therefore moves k1 by kernel × dp/psurf.", ""]
    (OUT / "o2a_dp_test.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, df, ttl in zip(axes, (oc, ld), ("29252a_GL ocean", "38200a_ND Kozienice land")):
        for qf, col, lab in ((1, "0.75", "QF1"), (0, "C0", "QF0")):
            m = (df.qf == qf) & np.isfinite(df.dp_frac_pct) & np.isfinite(df.k1_o2a)
            ax.scatter(df.dp_frac_pct[m], 100 * (df.k1_o2a[m] - 1), s=4, c=col, alpha=0.5, label=lab)
        m = (df.qf == 0) & np.isfinite(df.dp_frac_pct) & np.isfinite(df.k1_o2a)
        p = np.polyfit(df.dp_frac_pct[m], 100 * (df.k1_o2a[m] - 1), 1)
        xx = np.linspace(np.nanpercentile(df.dp_frac_pct[m], 1), np.nanpercentile(df.dp_frac_pct[m], 99), 2)
        ax.plot(xx, np.polyval(p, xx), "k-", label=f"QF0 OLS slope {p[0]:+.2f}")
        ax.plot(xx, xx + np.median(100 * (df.k1_o2a[m] - 1)) - np.median(df.dp_frac_pct[m]), "k--", lw=0.8, label="slope 1")
        ax.set(title=ttl, xlabel="L2 dp / psurf_prior (%)", ylabel="agreed O2A k1 − 1 (%)")
        ax.legend(fontsize=8)
    fig.savefig(OUT / "o2a_dp_test.png", dpi=150)


if __name__ == "__main__":
    main()

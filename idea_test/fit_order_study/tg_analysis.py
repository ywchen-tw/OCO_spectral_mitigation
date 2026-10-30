"""Target-mode land analysis of the truncation-order study (orbit 48443a_TG).

The target scene is fixed while the viewing zenith angle sweeps 0–57°, so the
far-clear / near-cloud split of the ocean study is replaced by a within-scene
analysis of QF0 soundings (definitions pre-registered in PLAN.md, section
"Target-mode land orbit 48443a_TG"):

* within-scene spread: OLS of y/median(y) − 1 on 8 footprint intercepts plus a
  linear VZA term; spread = 1.4826·MAD of the residuals (%);
* VZA slope: the VZA coefficient in % of the median per 10°, with a
  frame-block bootstrap SE (blocks of 20 consecutive frames);
* ocean-comparable spread: the run_study metric (per-fp mean removed only);
* k2 zero-bound fraction per variant;
* binned k1 medians versus VZA, scattering angle and glint angle.

Inputs are the run_study per-sounding parquet (row index ``j`` = fp_tau order),
the production fp_tau file (sounding order, VZA) and the Lite file (QF and
geometry).  Outputs: ``results/tg_vza_<orbit>.csv``, ``results/tg_within_<orbit>.csv``,
``results/tg_summary.md`` and ``results/tg_*.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_study import robust_sd, spread_pct  # noqa: E402

BANDS = ("o2a", "wco2", "sco2")
TARGETS = ("o2a", "wco2", "sco2", "ratio")  # ratio = k1_SCO2 / k1_O2A
FRAME_BLOCK = 20
N_BOOT = 1000
GEOM_VARIANTS = ("ols_2", "ols_3", "ols_7", "gamma")
OCEAN_ORBITS = {  # orbit -> (summary csv, label used in the replication)
    "29252a_GL (true labels)": ("summary_29252a_GL.csv", None),
    "22845a_GL (proxy B)": ("rep_summary_22845a_GL.csv", "proxy"),
    "22846a_GL (proxy B)": ("rep_summary_22846a_GL.csv", "proxy"),
}


# --------------------------------------------------------------------------- inputs


def load_meta(fp_tau_file: str, lite_file: str) -> pd.DataFrame:
    """Per-sounding geometry/QF in fp_tau order (row = run_study ``j``)."""
    with h5py.File(fp_tau_file, "r") as f:
        sid = f["sounding_id"][...].astype(np.int64)
        fp = f["fp_number"][...].astype(int)
        vza_tau = f["vza"][...].astype(float)
    with netCDF4.Dataset(lite_file, "r") as nc:
        lt_id = np.asarray(nc.variables["sounding_id"][:], dtype=np.int64)
        order = np.argsort(lt_id)
        pos = np.clip(np.searchsorted(lt_id[order], sid), 0, len(lt_id) - 1)
        if not np.array_equal(lt_id[order][pos], sid):
            raise ValueError("fp_tau soundings missing from the Lite file")
        row = order[pos]

        def root(v):
            return np.asarray(nc.variables[v][:], dtype=float)[row]

        def grp(g, v):
            return np.asarray(nc.groups[g].variables[v][:], dtype=float)[row]

        meta = pd.DataFrame({
            "sid": sid, "fp": fp, "vza_tau": vza_tau,
            "qf": root("xco2_quality_flag").astype(int),
            "lat": root("latitude"), "lon": root("longitude"),
            "sza": root("solar_zenith_angle"), "vza": root("sensor_zenith_angle"),
            "saa": grp("Sounding", "solar_azimuth_angle"),
            "vaa": grp("Sounding", "sensor_azimuth_angle"),
            "glint_angle": grp("Sounding", "glint_angle"),
            "altitude": grp("Sounding", "altitude"),
            "land_fraction": grp("Sounding", "land_fraction"),
        })
    meta.loc[meta.altitude < -1000, "altitude"] = np.nan   # Lite fill value
    th0, th = np.radians(meta.sza), np.radians(meta.vza)
    dphi = np.radians(meta.saa - meta.vaa)
    cos_sca = -np.cos(th0) * np.cos(th) - np.sin(th0) * np.sin(th) * np.cos(dphi)
    meta["scat_angle"] = np.degrees(np.arccos(np.clip(cos_sca, -1, 1)))
    meta["frame"] = meta.sid // 10
    frames = np.sort(meta.frame.unique())
    meta["block"] = np.searchsorted(frames, meta.frame) // FRAME_BLOCK
    return meta


def load_wide(parquet: Path) -> pd.DataFrame:
    long = pd.read_parquet(parquet, columns=["j", "band", "variant", "k1", "k2"])
    wide = long.set_index(["j", "variant", "band"])[["k1", "k2"]].unstack(["variant", "band"])
    return wide.sort_index(axis=1)


# --------------------------------------------------------------------------- statistics


def design(fp: np.ndarray, vza: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    cols = [(fp == i).astype(float) for i in range(8)]
    cols.append(vza / 10.0)
    if extra is not None:
        cols.append((extra - extra.mean()) / 1000.0)
    x = np.column_stack(cols)
    return x[:, x.any(axis=0)]


def detrended(y: np.ndarray, x: np.ndarray, vza_col: int) -> tuple[float, float]:
    """(spread %, VZA slope % per 10°) of y relative to its median."""
    y_rel = y / np.median(y) - 1.0
    coef, *_ = np.linalg.lstsq(x, y_rel, rcond=None)
    resid = y_rel - x @ coef
    return 100.0 * robust_sd(resid), 100.0 * coef[vza_col]


def target_values(wide: pd.DataFrame, variant: str, target: str) -> np.ndarray:
    k1 = lambda b: wide[("k1", variant, b)].to_numpy()  # noqa: E731
    if target == "ratio":
        return k1("sco2") / k1("o2a")
    return k1(target)


def within_scene(wide, meta, variants, rng) -> pd.DataFrame:
    recs = []
    qf0 = (meta.qf == 0).to_numpy()
    for v in variants:
        if ("k1", v, "o2a") not in wide.columns:
            continue
        all_ok = np.all([np.isfinite(wide[("k1", v, b)].to_numpy()) for b in BANDS], axis=0)
        s = qf0 & all_ok
        m = meta[s]
        fp, vza, blocks = m.fp.to_numpy(), m.vza.to_numpy(), m.block.to_numpy()
        x = design(fp, vza)
        vcol = x.shape[1] - 1
        x_alt = design(fp, vza, m.altitude.to_numpy())
        ublk = np.unique(blocks)
        blk_rows = {b: np.flatnonzero(blocks == b) for b in ublk}
        for t in TARGETS:
            y = target_values(wide, v, t)[s]
            spread, slope = detrended(y, x, vcol)
            _, slope_alt = detrended(y, x_alt, vcol)
            # Robustness (not pre-registered as a bar): cubic VZA trend, since
            # the binned k1-vs-VZA profiles are curved.
            x_cub = np.column_stack([x, (vza / 10.0) ** 2, (vza / 10.0) ** 3])
            spread_cubic, _ = detrended(y, x_cub, vcol)
            boot_slope, boot_spread = [], []
            for _ in range(N_BOOT):
                pick = np.concatenate([blk_rows[b] for b in rng.choice(ublk, len(ublk))])
                xb = design(fp[pick], vza[pick])
                sp_b, sl_b = detrended(y[pick], xb, xb.shape[1] - 1)
                boot_slope.append(sl_b)
                boot_spread.append(sp_b)
            se = float(np.std(boot_slope, ddof=1))
            rec = {"variant": v, "target": t, "n": int(s.sum()),
                   "median": float(np.median(y)),
                   "spread_detrended_pct": spread,
                   "spread_block_p05": float(np.percentile(boot_spread, 5)),
                   "spread_block_p95": float(np.percentile(boot_spread, 95)),
                   "spread_fp_only_pct": spread_pct(y, fp),
                   "spread_cubic_pct": spread_cubic,
                   "vza_slope_pct_per10": slope, "vza_slope_se": se,
                   "vza_slope_sig": bool(abs(slope) > 2 * se),
                   "vza_slope_alt_pct_per10": slope_alt}
            if t != "ratio":
                k2 = wide[("k2", v, t)].to_numpy()[s]
                rec["k2_zero_frac"] = float(np.mean(k2 <= 1e-6))
                rec["k2_median"] = float(np.median(k2))
            recs.append(rec)
    return pd.DataFrame(recs)


def sounding_bootstrap_ratio(wide, meta, num, den, target, rng):
    """90 % interval of spread(num)/spread(den) on the common QF0 sample."""
    ok = (meta.qf == 0).to_numpy()
    for v in (num, den):
        ok &= np.all([np.isfinite(wide[("k1", v, b)].to_numpy()) for b in BANDS], axis=0)
    m = meta[ok]
    x = design(m.fp.to_numpy(), m.vza.to_numpy())
    vcol = x.shape[1] - 1
    yn, yd = target_values(wide, num, target)[ok], target_values(wide, den, target)[ok]
    n = len(yn)
    point = detrended(yn, x, vcol)[0] / detrended(yd, x, vcol)[0]
    boot = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        xb = x[pick]
        boot.append(detrended(yn[pick], xb, vcol)[0] / detrended(yd[pick], xb, vcol)[0])
    return point, float(np.percentile(boot, 5)), float(np.percentile(boot, 95)), n


def geometry_table(wide, meta) -> pd.DataFrame:
    qf0 = (meta.qf == 0).to_numpy()
    axes = {
        "vza": np.arange(0, 70, 10.0),
        "scat_angle": np.quantile(meta.scat_angle[qf0], np.linspace(0, 1, 7)),
        "glint_angle": np.quantile(meta.glint_angle[qf0], np.linspace(0, 1, 7)),
    }
    recs = []
    for v in GEOM_VARIANTS:
        for b in BANDS:
            k1 = wide[("k1", v, b)].to_numpy()
            k2 = wide[("k2", v, b)].to_numpy()
            ok = qf0 & np.isfinite(k1)
            med_all = np.median(k1[ok])
            for ax, edges in axes.items():
                val = meta[ax].to_numpy()
                for lo, hi in zip(edges[:-1], edges[1:]):
                    sel = ok & (val >= lo) & (val <= hi if hi == edges[-1] else val < hi)
                    if sel.sum() == 0:
                        continue
                    recs.append({"variant": v, "band": b, "axis": ax, "bin_lo": lo, "bin_hi": hi,
                                 "n": int(sel.sum()),
                                 "k1_median": float(np.median(k1[sel])),
                                 "k1_rel_median_pct": 100 * (np.median(k1[sel]) / med_all - 1),
                                 "k1_robust_sd_pct": 100 * robust_sd(k1[sel] / med_all),
                                 "k2_median": float(np.median(k2[sel])),
                                 "vza_median": float(np.median(meta.vza[sel]))})
    return pd.DataFrame(recs)


def ocean_table(res_dir: Path) -> pd.DataFrame:
    recs = []
    for name, (fn, labels) in OCEAN_ORBITS.items():
        p = res_dir / fn
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if labels is not None:
            df = df[(df["labels"] == labels) & (df["subset"] == "all")]
        for v in ("ols_2", "ols_3", "ols_7", "gamma"):
            d = df[df.variant == v].set_index("band")
            if d.empty:
                continue
            recs.append({"orbit": name, "variant": v, "n_far": int(d.loc["o2a", "n_far"]),
                         "o2a": d.loc["o2a", "k1_spread_pct"], "wco2": d.loc["wco2", "k1_spread_pct"],
                         "sco2": d.loc["sco2", "k1_spread_pct"],
                         "ratio": d.loc["sco2", "ratio_to_o2a_spread_pct"],
                         "k2zero_o2a": d.loc["o2a", "k2_zero_frac"],
                         "k2zero_sco2": d.loc["sco2", "k2_zero_frac"]})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- figures


def fig_vza(wide, meta, out: Path, orbit: str):
    qf0 = (meta.qf == 0).to_numpy()
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), sharex=True)
    colors = {"ols_2": "#1f77b4", "ols_3": "#2ca02c", "ols_7": "#d62728", "gamma": "#9467bd"}
    edges = np.arange(0, 62.5, 5.0)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for ax, t in zip(axs, TARGETS):
        for v, c in colors.items():
            y = target_values(wide, v, t)
            ok = qf0 & np.isfinite(y)
            yr = 100 * (y[ok] / np.median(y[ok]) - 1)
            vz = meta.vza.to_numpy()[ok]
            if v in ("ols_3", "ols_7"):
                ax.scatter(vz, yr, s=2, color=c, alpha=0.12, lw=0)
            med = [np.median(yr[(vz >= lo) & (vz < hi)]) if ((vz >= lo) & (vz < hi)).sum() > 10 else np.nan
                   for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(mid, med, "-o", ms=3, color=c, label=v)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-15, 15)
        ax.set_title({"ratio": "k1 SCO2 / O2A"}.get(t, f"k1 {t.upper()}"))
        ax.set_xlabel("VZA (deg)")
    axs[0].set_ylabel("relative to scene median (%)")
    axs[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{orbit}: QF0 k1 vs VZA (lines: 5° bin medians; dots: ols_3, ols_7)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_spread(within: pd.DataFrame, ocean: pd.DataFrame, out: Path, orbit: str):
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, t in zip(axs, TARGETS):
        for kind, c in (("ols", "k"), ("wls", "0.6")):
            d = within[(within.target == t) & within.variant.str.startswith(kind + "_")].copy()
            d["order"] = d.variant.str.split("_").str[1].astype(int)
            d = d.sort_values("order")
            ax.plot(d.order, d.spread_detrended_pct, "-o", color=c, label=f"{orbit} {kind} (detrended)")
            if kind == "ols":
                ax.plot(d.order, d.spread_fp_only_pct, "--", color=c, lw=0.8, label=f"{orbit} ols (fp only)")
        g = within[(within.target == t) & (within.variant == "gamma")]
        if len(g):
            ax.axhline(float(g.spread_detrended_pct.iloc[0]), color="#9467bd", ls=":", label="gamma (TG)")
        for (orb, d), mk in zip(ocean.groupby("orbit"), ("s", "^", "v")):
            d = d[d.variant.str.startswith("ols")]
            ax.scatter(d.variant.str.split("_").str[1].astype(int), d[t], marker=mk, color="#1f77b4",
                       label=f"ocean far-clear {orb.split(' ')[0]}")
        ax.set_yscale("log")
        ax.set_xlabel("truncation order")
        ax.set_title({"ratio": "SCO2/O2A ratio"}.get(t, t.upper()))
    axs[0].set_ylabel("robust spread (%)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_geometry(geo: pd.DataFrame, out: Path, orbit: str):
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharey="row")
    for i, ax_name in enumerate(("scat_angle", "glint_angle")):
        for j, b in enumerate(BANDS):
            ax = axs[i, j]
            for v in GEOM_VARIANTS:
                d = geo[(geo.axis == ax_name) & (geo.band == b) & (geo.variant == v)]
                x = 0.5 * (d.bin_lo + d.bin_hi)
                ax.plot(x, d.k1_rel_median_pct, "-o", ms=3, label=v)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel(f"{ax_name.replace('_', ' ')} (deg)")
            if i == 0:
                ax.set_title(b.upper())
        axs[i, 0].set_ylabel("k1 bin median vs scene median (%)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{orbit}: QF0 k1 vs scattering and glint angle (sextile bins)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- report


def fmt_table(df: pd.DataFrame, cols, floatfmt="{:.3g}") -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sounding", required=True)
    ap.add_argument("--fp-tau", required=True)
    ap.add_argument("--lite", required=True)
    ap.add_argument("--orbit", default="48443a_TG")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    rng = np.random.default_rng(0)

    meta = load_meta(args.fp_tau, args.lite)
    wide = load_wide(Path(args.per_sounding))
    if len(wide) != len(meta):
        raise ValueError(f"per-sounding rows {len(wide)} != fp_tau rows {len(meta)}")
    meta = meta.iloc[wide.index.to_numpy()].reset_index(drop=True)
    wide = wide.reset_index(drop=True)
    variants = sorted({c[1] for c in wide.columns})
    variants = [v for v in variants if not v.startswith("cut_")]

    within = within_scene(wide, meta, variants, rng)
    within.insert(0, "orbit", args.orbit)
    within.to_csv(out / f"tg_within_{args.orbit}.csv", index=False)

    geo = geometry_table(wide, meta)
    geo.insert(0, "orbit", args.orbit)
    geo.to_csv(out / f"tg_vza_{args.orbit}.csv", index=False)

    ocean = ocean_table(out)
    boot = {}
    for t in ("ratio", "o2a", "sco2"):
        boot[t] = sounding_bootstrap_ratio(wide, meta, "ols_2", "ols_3", t, rng)
    boot73 = sounding_bootstrap_ratio(wide, meta, "ols_7", "ols_3", "ratio", rng)

    fig_vza(wide, meta, out / f"tg_k1_vs_vza_{args.orbit}.png", args.orbit)
    fig_spread(within, ocean, out / f"tg_spread_vs_order_{args.orbit}.png", args.orbit)
    fig_geometry(geo, out / f"tg_geometry_{args.orbit}.png", args.orbit)

    qf0 = meta.qf == 0
    lines = [f"# Target-mode land test — {args.orbit}", "",
             f"Soundings with τ: {len(meta)}; QF0: {int(qf0.sum())}; land_fraction min "
             f"{meta.land_fraction.min():.0f} %; VZA {meta.vza.min():.1f}–{meta.vza.max():.1f}°; "
             f"SZA {meta.sza.min():.1f}–{meta.sza.max():.1f}°; scattering angle "
             f"{meta.scat_angle.min():.0f}–{meta.scat_angle.max():.0f}°; glint angle "
             f"{meta.glint_angle.min():.0f}–{meta.glint_angle.max():.0f}°; surface altitude "
             f"{meta.altitude[qf0].min():.0f}–{meta.altitude[qf0].max():.0f} m (QF0).", "",
             f"Frame blocks for the bootstrap: {meta.block.nunique()} blocks of {FRAME_BLOCK} frames. "
             f"QF0 correlation of VZA with surface altitude {meta[qf0][['vza', 'altitude']].corr().iloc[0, 1]:+.2f}, "
             f"with latitude {meta[qf0][['vza', 'lat']].corr().iloc[0, 1]:+.2f}, "
             f"with scattering angle {meta[qf0][['vza', 'scat_angle']].corr().iloc[0, 1]:+.2f}.", "",
             "## Within-scene statistics (QF0, VZA-detrended)", "",
             "spread = robust SD (%) of residuals after fp intercepts + linear VZA; "
             "[p05, p95] = frame-block bootstrap; fp-only = run_study metric; slope in % per 10° VZA "
             "(± frame-block SE); alt = slope with surface altitude as an extra covariate; "
             "cubic = spread after fp intercepts + cubic VZA polynomial (robustness, not a bar).", ""]
    show = within[within.variant.isin(["ols_1", "ols_2", "ols_3", "ols_4", "ols_5", "ols_6", "ols_7",
                                       "wls_2", "wls_3", "wls_7", "gamma", "gamma_u"])]
    cols = ["variant", "target", "n", "median", "spread_detrended_pct", "spread_block_p05",
            "spread_block_p95", "spread_fp_only_pct", "spread_cubic_pct", "vza_slope_pct_per10", "vza_slope_se",
            "vza_slope_sig", "vza_slope_alt_pct_per10", "k2_zero_frac", "k2_median"]
    lines += [fmt_table(show, cols), ""]
    lines += ["## Spread ratios (sounding bootstrap, 90 %)", ""]
    for t, (p, lo, hi, n) in boot.items():
        lines.append(f"- ols_2 / ols_3 spread, {t}: {p:.2f} [{lo:.2f}, {hi:.2f}] (n {n})")
    p, lo, hi, n = boot73
    lines.append(f"- ols_7 / ols_3 spread, ratio: {p:.2f} [{lo:.2f}, {hi:.2f}] (n {n})")
    lines += ["", "## Ocean far-clear reference (per-fp mean removed, %)", "",
              fmt_table(ocean, list(ocean.columns)), "",
              "## Geometry tables", "",
              f"Binned medians in `tg_vza_{args.orbit}.csv` (axes vza, scat_angle, glint_angle).", ""]
    gv = geo[(geo.axis == "vza") & geo.variant.isin(["ols_2", "ols_3", "ols_7"])]
    piv = gv.pivot_table(index=["band", "variant"], columns="bin_lo", values="k1_rel_median_pct")
    piv.columns = [f"VZA {int(c)}–{int(c) + 10}" for c in piv.columns]
    lines += ["k1 bin median relative to the scene median (%), QF0:", "",
              fmt_table(piv.reset_index(), ["band", "variant", *piv.columns], "{:.2f}"), ""]
    (out / "tg_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

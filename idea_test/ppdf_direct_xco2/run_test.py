"""Feasibility test: direct XCO2 from path cumulants vs the bias-corrected product.

Usage:
    python idea_test/ppdf_direct_xco2/run_test.py --dates 2018-07-10 2021-09-06 ...

Per date x surface x estimator it reports far-cloud (held-out) and near-cloud
agreement with xco2_bc, the correlation of the residual with the clear-sky
anomaly, and the along-track lag-1 autocorrelation of the residual.  Plume
dates get a transect figure around the plant's closest approach.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "workspace" / "Nassar_plume_analysis"))
from estimators import ESTIMATORS, LOAD_COLUMNS, far_clear_mask, prepare, run_all  # noqa: E402
from analyze_nassar_plume_preservation import haversine_km  # noqa: E402

PARQUET_DIR = ROOT / "results" / "csv_collection"
PLANTS_CSV = ROOT / "workspace" / "Nassar_plume_analysis" / "nassar_power_plants.csv"
PLUME_CASES = {"2021-09-06": "kozienice", "2023-06-26": "westar", "2024-04-15": "ghent"}
SURFACE = {0.0: "ocean", 1.0: "land"}
SUBSETS = {
    "far_heldout": lambda d, far_km: far_clear_mask(d, far_km),
    "near_0_3": lambda d, far_km: (d.cld_dist_km > 0) & (d.cld_dist_km < 3),
    "near_3_10": lambda d, far_km: (d.cld_dist_km >= 3) & (d.cld_dist_km < 10),
}
BAR_FAR_SD = 2.0
BAR_NEAR_CORR = 0.3
BAR_NEAR_SD = 5.0


def load_date(date: str) -> pd.DataFrame:
    path = PARQUET_DIR / f"combined_{date}_all_orbits.parquet"
    df = pd.read_parquet(path, columns=LOAD_COLUMNS)
    df["date"] = date
    return prepare(df)


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Minimal markdown table (avoids the optional tabulate dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def lag1_autocorr(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.autocorr(1)) if len(x) > 10 else np.nan


def metrics_for(df: pd.DataFrame, date: str, far_km: float) -> pd.DataFrame:
    rows = []
    for sfc, g in df.groupby("sfc_type"):
        for sub, fn in SUBSETS.items():
            s = g[fn(g, far_km)]
            for e in ESTIMATORS:
                r = (s[f"est_{e}"] - s["xco2_bc"]).dropna()
                if len(r) < 20:
                    continue
                anom = s.loc[r.index, "xco2_bc_anomaly"]
                ok = anom.notna()
                rows.append({
                    "date": date, "surface": SURFACE.get(sfc, str(sfc)), "subset": sub,
                    "estimator": e, "n": len(r), "frac_valid": len(r) / max(len(s), 1),
                    "bias_ppm": r.mean(), "sd_ppm": r.std(),
                    "robust_sd_ppm": 1.4826 * (r - r.median()).abs().median(),
                    "frac_within_2ppm": (r.abs() <= 2).mean(),
                    "corr_anomaly": float(np.corrcoef(r[ok], anom[ok])[0, 1]) if ok.sum() > 20 else np.nan,
                    "lag1_autocorr": lag1_autocorr(r),
                })
        # band consistency: E2 (SCO2) vs E2w (WCO2) on everything
        d = (g["est_E2"] - g["est_E2w"]).dropna()
        if len(d) > 20:
            rows.append({"date": date, "surface": SURFACE.get(sfc, str(sfc)), "subset": "all",
                         "estimator": "E2_minus_E2w", "n": len(d), "frac_valid": len(d) / len(g),
                         "bias_ppm": d.mean(), "sd_ppm": d.std(),
                         "robust_sd_ppm": 1.4826 * (d - d.median()).abs().median(),
                         "frac_within_2ppm": (d.abs() <= 2).mean(), "corr_anomaly": np.nan,
                         "lag1_autocorr": lag1_autocorr(d)})
    return pd.DataFrame(rows)


def hist_figure(df: pd.DataFrame, date: str, far_km: float, out: Path) -> None:
    surfaces = sorted(df.sfc_type.dropna().unique())
    fig, axes = plt.subplots(len(surfaces), 2, figsize=(10, 3.2 * len(surfaces)), squeeze=False)
    bins = np.linspace(-60, 60, 121)
    for i, sfc in enumerate(surfaces):
        g = df[df.sfc_type == sfc]
        for j, (sub, fn) in enumerate([("far_heldout", SUBSETS["far_heldout"]), ("near_0_3", SUBSETS["near_0_3"])]):
            ax = axes[i, j]
            s = g[fn(g, far_km)]
            for e in ESTIMATORS:
                r = (s[f"est_{e}"] - s["xco2_bc"]).dropna()
                if len(r):
                    ax.hist(r.clip(-60, 60), bins=bins, histtype="step", lw=1.2,
                            label=f"{e} (robust sd {1.4826 * (r - r.median()).abs().median():.1f})")
            ax.set_yscale("log")
            ax.set_title(f"{date} {SURFACE.get(sfc, sfc)} — {sub} (n={len(s)})", fontsize=9)
            ax.set_xlabel("estimate − xco2_bc (ppm)")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plume_figure(df: pd.DataFrame, date: str, plant: str, out: Path, max_km: float = 100.0) -> dict:
    plants = pd.read_csv(PLANTS_CSV).set_index("plant_id")
    lat0, lon0 = plants.loc[plant, ["lat", "lon"]]
    d = df.copy()
    d["src_km"] = haversine_km(d["lat"].to_numpy(), d["lon"].to_numpy(), lat0, lon0)
    i_min = d["src_km"].idxmin()
    seg = d[(d.orbit_id == d.loc[i_min, "orbit_id"]) & (d.src_km < max_km)].copy()
    seg["x_km"] = (seg["lat"] - d.loc[i_min, "lat"]) * 111.0
    seg = seg.sort_values("x_km")
    cols = {"xco2_bc": "xco2_bc", "E1": "est_E1", "E2": "est_E2", "E2w": "est_E2w", "B_IDP": "est_B_IDP"}
    enh = {}
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 2.1 * len(cols)), sharex=True)
    for ax, (name, col) in zip(axes, cols.items()):
        y = seg[col]
        core = y[seg.x_km.abs() <= 10].median()
        bg = y[(seg.x_km.abs() >= 20) & (seg.x_km.abs() <= 60)].median()
        enh[name] = core - bg
        ax.scatter(seg.x_km, y, s=4, alpha=0.4, color="0.5")
        roll = y.rolling(25, center=True, min_periods=5).median()
        ax.plot(seg.x_km, roll, color="C3", lw=1.3)
        ax.set_ylabel(name, fontsize=8)
        lo, hi = np.nanpercentile(y, [2, 98])
        pad = max(1.0, 0.1 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(f"core−bg {enh[name]:+.2f} ppm", fontsize=8, loc="right")
    axes[-1].set_xlabel("along-track distance from closest approach (km)")
    fig.suptitle(f"{plant} {date} — closest approach {d.loc[i_min, 'src_km']:.1f} km, "
                 f"median cld_dist in core {seg[seg.x_km.abs() <= 10].cld_dist_km.median():.1f} km", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return enh


def write_summary(metrics: pd.DataFrame, plume_enh: dict, out: Path) -> None:
    lines = ["# Summary — direct XCO2 from path cumulants", ""]
    lines += ["Residual = estimate − xco2_bc. `robust_sd` = 1.4826·MAD; `frac_valid` = share of "
              "footprints whose |f − 1| ≤ 0.2 (±80 ppm) and whose path prediction was in range.", ""]
    for sub, cols in [("far_heldout", ["robust_sd_ppm", "sd_ppm", "bias_ppm", "frac_within_2ppm", "frac_valid", "lag1_autocorr"]),
                      ("near_0_3", ["robust_sd_ppm", "bias_ppm", "corr_anomaly", "frac_valid"]),
                      ("near_3_10", ["robust_sd_ppm", "corr_anomaly"])]:
        m = metrics[metrics.subset == sub]
        if m.empty:
            continue
        lines += [f"## {sub}", ""]
        for col in cols:
            piv = m.pivot_table(index=["date", "surface"], columns="estimator", values=col)
            lines += [f"**{col}**", "", md_table(piv.round(2)), ""]
    bc = metrics[metrics.estimator == "E2_minus_E2w"]
    if not bc.empty:
        lines += ["## Band consistency: E2 (SCO2) − E2w (WCO2), all footprints", "",
                  md_table(bc[["date", "surface", "n", "bias_ppm", "sd_ppm", "robust_sd_ppm"]].round(2), index=False), ""]
    if plume_enh:
        lines += ["## Plume core − background (ppm)", "",
                  md_table(pd.DataFrame(plume_enh).T.round(2)), ""]
    far = metrics[metrics.subset == "far_heldout"]
    near = metrics[metrics.subset == "near_0_3"]
    lines += ["## Bars", "",
              f"- Retrieval bar (far robust SD ≤ {BAR_FAR_SD} ppm), median over date×surface: "
              f"{far.groupby('estimator').robust_sd_ppm.median().round(2).to_dict()}",
              f"- Diagnostic bar (near corr ≥ {BAR_NEAR_CORR}, robust SD ≤ {BAR_NEAR_SD}): "
              f"corr = {near.groupby('estimator').corr_anomaly.median().round(2).to_dict()}, "
              f"robust SD = {near.groupby('estimator').robust_sd_ppm.median().round(2).to_dict()}", ""]
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--far-km", type=float, default=20.0)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics, plume_enh = [], {}
    for date in args.dates:
        print(f"[{date}] loading", flush=True)
        df = run_all(load_date(date), far_km=args.far_km)
        m = metrics_for(df, date, args.far_km)
        m.to_csv(args.out_dir / f"metrics_{date}.csv", index=False)
        all_metrics.append(m)
        if not args.no_fig:
            hist_figure(df, date, args.far_km, args.out_dir / f"hist_{date}.png")
            if date in PLUME_CASES:
                plant = PLUME_CASES[date]
                plume_enh[f"{plant}_{date}"] = plume_figure(df, date, plant, args.out_dir / f"plume_{plant}_{date}.png")
        print(m[m.subset.isin(["far_heldout", "near_0_3"])][["surface", "subset", "estimator", "n", "frac_valid", "robust_sd_ppm", "sd_ppm", "corr_anomaly"]]
              .to_string(index=False), flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(args.out_dir / "metrics_all.csv", index=False)
    write_summary(metrics, plume_enh, args.out_dir / "summary.md")
    print(f"summary -> {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

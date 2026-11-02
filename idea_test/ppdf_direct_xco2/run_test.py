"""Feasibility test: direct XCO2 from path cumulants vs the bias-corrected product.

Usage (fit schema doppler_am_t2_v2, results_doppler parquets):
    python idea_test/ppdf_direct_xco2/run_test.py --all-combined --workers 8 \
        --out-dir idea_test/ppdf_direct_xco2/results_2016_2020
    python idea_test/ppdf_direct_xco2/run_test.py --dates 2018-07-10 2021-09-06 ...
    python idea_test/ppdf_direct_xco2/run_test.py --summarize-only --out-dir ...

Dates in the combined 2016-2020 file are read from its row group (one per date);
other dates from combined_<date>_all_orbits.parquet.  Per date x surface x
estimator it reports far-cloud (held-out) and near-cloud agreement with xco2_bc,
the correlation of the residual with the clear-sky anomaly, the along-track lag-1
autocorrelation of the residual, and the fit-noise-only sigma (from sig_*) with
the pull SD.  Plume dates get a transect figure around the plant's closest approach.

Ported 2026-11-01 from the first pass (see git history); bars and metrics unchanged.
"""
from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from estimators import (ESTIMATORS, SPEC_BY_ID, far_clear_mask, load_columns,  # noqa: E402
                        prepare, run_all)

PARQUET_DIR = Path("/scratch/alpine/yuch8913/oco/spec_anal/results_doppler/csv_collection")
COMBINED = PARQUET_DIR / "combined_2016_2020_dates.parquet"
PLANTS_CSV = ROOT / "workspace" / "Nassar_plume_analysis" / "nassar_power_plants.csv"
PLUME_CASES = {"2021-09-06": "kozienice", "2023-06-26": "westar", "2024-04-15": "ghent"}
FIRST_PASS_DATES = ("2018-07-10", "2021-09-06", "2023-06-26", "2019-06-22", "2017-10-09")
SURFACE = {0.0: "ocean", 1.0: "land"}
SUBSETS = {
    "far_heldout": lambda d, far_km: far_clear_mask(d, far_km),
    "near_0_3": lambda d, far_km: (d.cld_dist_km > 0) & (d.cld_dist_km < 3),
    "near_3_10": lambda d, far_km: (d.cld_dist_km >= 3) & (d.cld_dist_km < 10),
}
BAR_FAR_SD = 2.0
BAR_NEAR_CORR = 0.3
BAR_NEAR_SD = 5.0
FIG_ESTIMATORS = ("E2_o2", "E2w_o2", "E0_am", "E2_am", "E2w_am", "E2_t2", "B_IDP", "N0_prior")
PLUME_FIG_ESTIMATORS = ("E2_o2", "E2w_o2", "E0_am", "E2_am", "E2w_am", "E2_t2", "B_IDP", "N0_prior")
BAND_PAIRS = [("E2_o2", "E2w_o2"), ("E2_t2", "E2w_t2"), ("E2_am", "E2w_am")]


def combined_index() -> dict[str, int]:
    f = pq.ParquetFile(COMBINED)
    i = f.schema_arrow.names.index("date")
    return {f.metadata.row_group(g).column(i).statistics.min.decode(): g
            for g in range(f.metadata.num_row_groups)}


def load_date(date: str, rg_index: dict[str, int] | None = None) -> pd.DataFrame:
    if rg_index and date in rg_index:
        f = pq.ParquetFile(COMBINED)
        cols = load_columns(set(f.schema_arrow.names))
        t = f.read_row_group(rg_index[date], columns=cols + ["date"]).to_pandas()
        got = set(t.pop("date").unique())
        assert got == {date.encode()}, f"row group {rg_index[date]} holds {got}, expected {date}"
        df = t
    else:
        path = PARQUET_DIR / f"combined_{date}_all_orbits.parquet"
        f = pq.ParquetFile(path)
        df = f.read(columns=load_columns(set(f.schema_arrow.names))).to_pandas()
    df["date"] = date
    return prepare(df)


def haversine_km(lat, lon, src_lat, src_lon):
    lat1 = np.radians(np.asarray(lat, float))
    lat2 = math.radians(src_lat)
    dlon = np.radians(np.asarray(lon, float) - src_lon)
    a = np.sin((lat1 - lat2) / 2.0) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(a))


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


def rsd(x) -> float:
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def _corr(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 20 else np.nan


def metrics_for(df: pd.DataFrame, date: str, far_km: float) -> pd.DataFrame:
    rows = []
    for sfc, g in df.groupby("sfc_type"):
        surf = SURFACE.get(sfc, str(sfc))
        for sub, fn in SUBSETS.items():
            s = g[fn(g, far_km)]
            for e in ESTIMATORS:
                r = (s[f"est_{e}"] - s["xco2_bc"]).dropna()
                if len(r) < 20:
                    continue
                idx = r.index
                sig = s.loc[idx, f"sig_{e}"].to_numpy(float)
                pull = r.to_numpy() / sig
                pull = pull[np.isfinite(pull)]
                rows.append({
                    "date": date, "surface": surf, "subset": sub,
                    "estimator": e, "n": len(r), "frac_valid": len(r) / max(len(s), 1),
                    "bias_ppm": r.mean(), "sd_ppm": r.std(), "robust_sd_ppm": rsd(r),
                    "frac_within_2ppm": (r.abs() <= 2).mean(),
                    "corr_anomaly": _corr(r.to_numpy(), s.loc[idx, "xco2_bc_anomaly"].to_numpy(float)),
                    "corr_anomaly_r15": _corr(r.to_numpy(), s.loc[idx, "xco2_bc_anomaly_r15"].to_numpy(float)),
                    "lag1_autocorr": lag1_autocorr(r),
                    # does est - prior track bc - prior? (0 for the null baseline by construction)
                    "corr_bc_dev": _corr((s.loc[idx, f"est_{e}"] - s.loc[idx, "xco2_apriori"]).to_numpy(float),
                                         (s.loc[idx, "xco2_bc"] - s.loc[idx, "xco2_apriori"]).to_numpy(float)),
                    "sig_fit_ppm": float(np.nanmedian(sig)) if np.isfinite(sig).any() else np.nan,
                    "pull_robust_sd": rsd(pull) if len(pull) > 20 else np.nan,
                })
        for a, b in BAND_PAIRS:   # band consistency on everything
            d = (g[f"est_{a}"] - g[f"est_{b}"]).dropna()
            if len(d) > 20:
                rows.append({"date": date, "surface": surf, "subset": "all",
                             "estimator": f"{a}_minus_{b}", "n": len(d), "frac_valid": len(d) / len(g),
                             "bias_ppm": d.mean(), "sd_ppm": d.std(), "robust_sd_ppm": rsd(d),
                             "frac_within_2ppm": (d.abs() <= 2).mean(), "lag1_autocorr": lag1_autocorr(d)})
    return pd.DataFrame(rows)


def hist_figure(df: pd.DataFrame, date: str, far_km: float, out: Path) -> None:
    surfaces = sorted(df.sfc_type.dropna().unique())
    fig, axes = plt.subplots(len(surfaces), 2, figsize=(11, 3.4 * len(surfaces)), squeeze=False)
    bins = np.linspace(-60, 60, 121)
    for i, sfc in enumerate(surfaces):
        g = df[df.sfc_type == sfc]
        for j, sub in enumerate(("far_heldout", "near_0_3")):
            ax = axes[i, j]
            s = g[SUBSETS[sub](g, far_km)]
            for e in FIG_ESTIMATORS:
                r = (s[f"est_{e}"] - s["xco2_bc"]).dropna()
                if len(r):
                    ax.hist(r.clip(-60, 60), bins=bins, histtype="step", lw=1.2,
                            label=f"{e} (robust sd {rsd(r):.1f})")
            ax.set_yscale("log")
            ax.set_title(f"{date} {SURFACE.get(sfc, sfc)} — {sub} (n={len(s)})", fontsize=9)
            ax.set_xlabel("estimate − xco2_bc (ppm)")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plume_enhancement(df: pd.DataFrame, date: str, plant: str, out: Path | None,
                      max_km: float = 100.0) -> dict:
    plants = pd.read_csv(PLANTS_CSV).set_index("plant_id")
    lat0, lon0 = plants.loc[plant, ["lat", "lon"]]
    d = df.copy()
    d["src_km"] = haversine_km(d["lat"].to_numpy(), d["lon"].to_numpy(), lat0, lon0)
    i_min = d["src_km"].idxmin()
    seg = d[(d.orbit_id == d.loc[i_min, "orbit_id"]) & (d.src_km < max_km)].copy()
    seg["x_km"] = (seg["lat"] - d.loc[i_min, "lat"]) * 111.0
    seg = seg.sort_values("x_km")
    core_m = seg.x_km.abs() <= 10
    bg_m = (seg.x_km.abs() >= 20) & (seg.x_km.abs() <= 60)
    cols = {"xco2_bc": "xco2_bc", **{e: f"est_{e}" for e in ESTIMATORS}}
    enh, noise = {}, {}
    for name, col in cols.items():
        y = seg[col]
        enh[name] = y[core_m].median() - y[bg_m].median()
        noise[name] = rsd(y[bg_m].dropna()) if y[bg_m].notna().sum() > 5 else np.nan
    if out is not None:
        show = ["xco2_bc", *PLUME_FIG_ESTIMATORS]
        fig, axes = plt.subplots(len(show), 1, figsize=(8, 1.9 * len(show)), sharex=True)
        for ax, name in zip(axes, show):
            y = seg[cols[name]]
            ax.scatter(seg.x_km, y, s=4, alpha=0.4, color="0.5")
            ax.plot(seg.x_km, y.rolling(25, center=True, min_periods=5).median(), color="C3", lw=1.3)
            ax.set_ylabel(name, fontsize=8)
            if y.notna().sum() > 5:
                lo, hi = np.nanpercentile(y, [2, 98])
                pad = max(1.0, 0.1 * (hi - lo))
                ax.set_ylim(lo - pad, hi + pad)
            ax.axvline(0, color="k", lw=0.6, ls="--")
            ax.set_title(f"core−bg {enh[name]:+.2f} ppm (bg robust sd {noise[name]:.1f})", fontsize=8, loc="right")
        axes[-1].set_xlabel("along-track distance from closest approach (km)")
        fig.suptitle(f"{plant} {date} — closest approach {d.loc[i_min, 'src_km']:.1f} km, "
                     f"median cld_dist in core {seg[core_m].cld_dist_km.median():.1f} km", fontsize=9)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
    return {"enh": enh, "bg_rsd": noise}


def process_date(date: str, rg_index, far_km: float, out_dir: Path, make_fig: bool,
                 skip_existing: bool) -> str:
    mpath = out_dir / f"metrics_{date}.csv"
    if skip_existing and mpath.exists():
        return f"[{date}] exists, skipped"
    df = run_all(load_date(date, rg_index), far_km=far_km)
    m = metrics_for(df, date, far_km)
    if date in PLUME_CASES:
        plant = PLUME_CASES[date]
        p = plume_enhancement(df, date, plant, out_dir / f"plume_{plant}_{date}.png" if make_fig else None)
        pd.DataFrame(p).rename_axis("estimator").to_csv(out_dir / f"plume_{plant}_{date}.csv")
    if make_fig:
        hist_figure(df, date, far_km, out_dir / f"hist_{date}.png")
    m.to_csv(mpath.with_suffix(".tmp"), index=False)
    mpath.with_suffix(".tmp").rename(mpath)
    far = m[(m.subset == "far_heldout") & m.estimator.isin(["E2_o2", "E2w_o2", "E0_am", "E2_am", "E2w_am", "B_IDP", "N0_prior"])]
    return f"[{date}] n={len(df)}\n" + far[["surface", "estimator", "n", "frac_valid", "robust_sd_ppm", "corr_bc_dev", "sig_fit_ppm"]].to_string(index=False)


# ── summary ─────────────────────────────────────────────────────────────────────

def _est_order(names):
    order = {e: i for i, e in enumerate(ESTIMATORS)}
    return sorted(names, key=lambda e: order.get(e, 999))


def _over_null(sub: pd.DataFrame) -> pd.Series:
    """Per-cell robust SD divided by the N0_prior robust SD of the same cell."""
    null = sub[sub.estimator == "N0_prior"].set_index(["date", "surface"]).robust_sd_ppm
    key = pd.MultiIndex.from_frame(sub[["date", "surface"]])
    return pd.Series(sub.robust_sd_ppm.to_numpy() / null.reindex(key).to_numpy(), index=sub.index)


def headline_table(metrics: pd.DataFrame) -> pd.DataFrame:
    far = metrics[metrics.subset == "far_heldout"].copy()
    near = metrics[metrics.subset == "near_0_3"].copy()
    far["over_null"] = _over_null(far)
    near["over_null"] = _over_null(near)
    t = pd.DataFrame({
        "far_rsd": far.groupby("estimator").robust_sd_ppm.median(),
        "far_rsd/N0": far.groupby("estimator").over_null.median(),
        "near_rsd": near.groupby("estimator").robust_sd_ppm.median(),
        "near_rsd/N0": near.groupby("estimator").over_null.median(),
        "near_corr": near.groupby("estimator").corr_anomaly.median(),
        "near_corr_r15": near.groupby("estimator").corr_anomaly_r15.median(),
        "far_corr_bcdev": far.groupby("estimator").corr_bc_dev.median(),
        "far_fvalid": far.groupby("estimator").frac_valid.median(),
        "near_fvalid": near.groupby("estimator").frac_valid.median(),
        "far_sig_fit": far.groupby("estimator").sig_fit_ppm.median(),
        "far_pull_rsd": far.groupby("estimator").pull_robust_sd.median(),
        "far_le2_share": far.groupby("estimator").robust_sd_ppm.apply(lambda x: (x <= BAR_FAR_SD).mean()),
        "n_cells": far.groupby("estimator").size(),
    })
    return t.loc[_est_order(t.index)]


def write_summary(metrics: pd.DataFrame, out: Path, plume: dict) -> None:
    metrics = metrics.copy()
    metrics["year"] = metrics.date.str[:4]
    main = metrics[metrics.date.str[:4].isin([str(y) for y in range(2016, 2021)])
                   & ~metrics.date.isin(["2018-07-10", "2019-06-22"])]
    fp = metrics[metrics.date.isin(FIRST_PASS_DATES)]
    lines = ["# Summary — direct XCO2 from path cumulants (fit schema doppler_am_t2_v2)", "",
             "Residual = estimate − xco2_bc. `rsd` = robust SD 1.4826·MAD (ppm). Medians over "
             "date × surface cells. `fvalid` = share of footprints with |f − 1| ≤ 0.2 and a path "
             "prediction in range (and, for *_scr, passing the σ screen). `sig_fit` = median "
             "fit-noise-only σ of the estimate from sig_* (ppm); `pull_rsd` = robust SD of "
             "residual/σ_fit (≫ 1 ⇒ scene variation, not fit noise, dominates). `far_corr_bcdev` = "
             "corr(est − prior, xco2_bc − prior) on far held-out (does the estimate track the "
             "retrieved departure from the prior?). `N0_prior` = prior + calib-far mean(bc − prior), "
             "the no-information null that every estimator must beat; `*/N0` = per-cell rsd ÷ N0 rsd "
             "(< 1 ⇒ better than the prior). `far_le2_share` "
             "= share of date × surface cells with far rsd ≤ 2 ppm. No ref_*/r15_*/r05_* "
             "statistics enter any estimator; xco2_bc_anomaly(_r15) is used only as the "
             "diagnostic target.", ""]
    n_dates = main.date.nunique()
    lines += [f"## Main: {n_dates} dates of the 2016–2020 combined file", "",
              md_table(headline_table(main).round(3)), ""]
    for surf in ("ocean", "land"):
        lines += [f"### {surf} only", "", md_table(headline_table(main[main.surface == surf])
                                                    [["far_rsd", "far_rsd/N0", "near_rsd", "near_rsd/N0", "near_corr", "far_corr_bcdev", "far_fvalid", "far_sig_fit", "far_pull_rsd", "far_le2_share"]].round(3)), ""]
    far = main[main.subset == "far_heldout"]
    lines += ["## Per-year far held-out rsd: median [q25–q75] over date × surface", ""]
    py = far.groupby(["estimator", "year"]).robust_sd_ppm.agg(
        lambda x: f"{x.median():.2f} [{x.quantile(.25):.1f}–{x.quantile(.75):.1f}]").unstack()
    lines += [md_table(py.loc[_est_order(py.index)]), ""]
    nr = main[main.subset == "near_0_3"]
    py2 = nr.groupby(["estimator", "year"]).robust_sd_ppm.median().unstack()
    lines += ["## Per-year near (< 3 km) rsd median", "", md_table(py2.loc[_est_order(py2.index)].round(2)), ""]
    bc = main[main.subset == "all"]
    if not bc.empty:
        lines += ["## Band consistency (SCO2 − WCO2 path-model estimates), all footprints", "",
                  md_table(bc.groupby(["estimator", "surface"])[["robust_sd_ppm", "bias_ppm"]].median().round(2)), ""]
    if not fp.empty:
        lines += [f"## First-pass dates present: {sorted(fp.date.unique())}", "",
                  md_table(headline_table(fp)[["far_rsd", "far_rsd/N0", "near_rsd", "near_corr", "far_corr_bcdev", "far_fvalid", "far_sig_fit", "far_pull_rsd"]].round(3)), ""]
        piv = fp[fp.subset == "far_heldout"].pivot_table(index="estimator", columns=["date", "surface"], values="robust_sd_ppm")
        lines += ["far rsd per first-pass date × surface", "", md_table(piv.loc[_est_order(piv.index)].round(2)), ""]
    if plume:
        lines += ["## Plume core − background (ppm; bg rsd in parentheses)", ""]
        cols = {}
        for k, p in plume.items():
            cols[k] = {e: f"{p['enh'][e]:+.2f} ({p['bg_rsd'][e]:.1f})" for e in p["enh"].index}
        tab = pd.DataFrame(cols)
        lines += [md_table(tab.loc[["xco2_bc"] + _est_order([e for e in tab.index if e != "xco2_bc"])]), ""]
    h = headline_table(main)
    passed_far = h.index[h.far_rsd <= BAR_FAR_SD].tolist()
    beats_null = h.index[h["far_rsd/N0"] < 1].tolist()
    passed_diag = h.index[(h.near_corr >= BAR_NEAR_CORR) & (h.near_rsd <= BAR_NEAR_SD)].tolist()
    lines += ["## Bars (main set, medians)", "",
              f"- Retrieval bar (far rsd ≤ {BAR_FAR_SD} ppm): passed by {passed_far or 'none'}",
              f"- Far rsd below the N0_prior null (median per-cell ratio < 1): {beats_null or 'none'}",
              f"- Diagnostic bar (near corr ≥ {BAR_NEAR_CORR} and near rsd ≤ {BAR_NEAR_SD}): passed by {passed_diag or 'none'}",
              ""]
    out.write_text("\n".join(lines))


def load_plumes(out_dir: Path) -> dict:
    plume = {}
    for p in sorted(out_dir.glob("plume_*.csv")):
        plume[p.stem.replace("plume_", "")] = pd.read_csv(p, index_col=0)
    return plume


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="*", default=[])
    ap.add_argument("--all-combined", action="store_true", help="add every date of the 2016-2020 combined file")
    ap.add_argument("--far-km", type=float, default=20.0)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results_2016_2020")
    ap.add_argument("--fig-dates", nargs="*", default=list(FIRST_PASS_DATES))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        rg_index = combined_index()
        dates = list(dict.fromkeys(args.dates + (sorted(rg_index) if args.all_combined else [])))
        missing = [d for d in dates if d not in rg_index
                   and not (PARQUET_DIR / f"combined_{d}_all_orbits.parquet").exists()]
        if missing:
            print(f"no file for {missing}; skipped", flush=True)
        dates = [d for d in dates if d not in missing]
        print(f"{len(dates)} dates, {len(ESTIMATORS)} estimators", flush=True)
        jobs = {}
        with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as ex:
            for d in dates:
                jobs[ex.submit(process_date, d, rg_index, args.far_km, args.out_dir,
                               d in args.fig_dates, args.skip_existing)] = d
            for fut in as_completed(jobs):
                try:
                    print(fut.result(), flush=True)
                except Exception as exc:  # keep the other dates going
                    print(f"[{jobs[fut]}] FAILED: {exc!r}", flush=True)
    paths = sorted(args.out_dir.glob("metrics_????-??-??.csv"))
    metrics = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    metrics.to_csv(args.out_dir / "metrics_all.csv", index=False)
    write_summary(metrics, args.out_dir / "summary.md", load_plumes(args.out_dir))
    print(f"{len(paths)} dates -> {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

"""analyze_failure_modes.py — where does the ML correction NOT work well, and why?

Four stages (TCCON truth, AK-harmonized reference primary / direct as check):

  1. WORSENERS (station-day level): station-days whose fp-RMSE or |bias| got
     WORSE after correction, from the per-case report CSVs of the production
     atrain + drift trees; persistence cross-checked against the r50 report.
  2. FOOTPRINT TABLE: every TCCON-collocated footprint (100 km / ±60 min,
     shared collocator) with resid_before/after under both references and the
     candidate failure drivers merged from plot_data + the source parquet
     (AOD block, SZA/latitude/airmass, albedos + snow_flag, SNR, dpfrac, tcwv,
     DE sigma split, guards).
  3. STRATIFICATION: per-surface decile bins of each driver (fp-RMSE before →
     after, mean Δ|resid|, ⟨z²⟩ self-consistency), a standardized multivariate
     OLS of |resid_after| on the drivers jointly, and the σ-self-awareness
     check (is failure flagged by the model's own uncertainty?).
  4. DOSSIERS: per worsening station-day, driver z-scores vs the full cohort —
     which stratum of stage 3 is this case an instance of?

Reference invariance note: ak_delta is a per-station-day CONSTANT, so
footprint-level Δ|resid| and scatter metrics are identical under AK and
direct; only bias-anchored metrics differ (reported under both).

Outputs (under <TAG>/failure_modes/ + one top-level md):
  footprints_r100km.parquet      stage-2 table (one row per collocated footprint)
  worseners_r100km.csv           stage-1 list
  strat_<surface>_r100km.csv     stage-3 binned tables
  multivariate_r100km.csv        stage-3 OLS coefficients
  fig_failure_modes_r100km.png   stage-3 driver panels
  ../FAILURE_MODES_<date>.md     the writeup skeleton with all tables

Usage:  PYTHONPATH=src:workspace python workspace/analyze_failure_modes.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as apq

from plot_style import apply_manuscript_style
from tccon_collocate import collocate, find_plotdata, get_storage_dir, load_tccon

TAG = "de_beta_nll_prof_reg_foldpca_o05l15_m5"
BASE = Path("results/model_comparison/deep_ensemble")
CORR = "deep_ensemble_corrected_xco2"
TREES = {  # tree label -> (out_base subdir, launcher with run_case lines)
    "atrain": (BASE / TAG / "atrain", "curc_shell_blanca_plot_corr_xco2_deepens.sh"),
    "drift": (BASE / TAG / "drift", "curc_shell_blanca_plot_corr_xco2_deepens_drift.sh"),
}

# Driver columns taken from the source parquet (merged on time/lon/lat).
SRC_DRIVERS = ["xco2_qf", "snow_flag", "alb_o2a", "alb_wco2", "alb_sco2",
               "aod_dust", "aod_seasalt", "aod_sulfate", "aod_bc", "aod_oc",
               "aod_ice", "aod_water", "aod_strataer", "airmass", "dpfrac",
               "tcwv"]
# Driver columns already in plot_data.
PD_DRIVERS = ["aod_total", "sza", "snr_o2a", "snr_wco2", "snr_sco2",
              "de_sigma", "de_epistemic_sigma", "de_aleatoric_sigma"]

# Stage-3 binned drivers (column, pretty label, log-x?).
BIN_DRIVERS = [
    ("aod_total", "total AOD", False),
    ("aod_dust", "dust AOD", False),
    ("sza", "SZA (°)", False),
    ("abs_lat", "|latitude| (°)", False),
    ("airmass", "airmass", False),
    ("alb_o2a", "O2A albedo", False),
    ("alb_sco2", "SCO2 albedo", False),
    ("tcwv", "TCWV (kg m$^{-2}$)", False),
    ("dpfrac", "dpfrac", False),
    ("snr_o2a", "O2A SNR", False),
    ("cld_dist_km", "cloud distance (km)", False),
    ("de_sigma", "DE σ (ppm)", False),
]
# Multivariate OLS drivers (standardized; cld_dist only meaningful pre-drift).
OLS_DRIVERS = ["aod_total", "aod_dust", "sza", "abs_lat", "airmass",
               "alb_o2a", "alb_sco2", "snow_flag", "tcwv", "dpfrac",
               "snr_o2a", "cld_dist_km"]


def parse_cases(script: str) -> list[dict]:
    cases = []
    for ln in Path(script).read_text().splitlines():
        if not re.match(r"^run_case\s", ln):
            continue
        t = ln.split()
        rest = t[9:]
        cases.append(dict(
            date=t[1], tccon=t[2],
            box=tuple(map(float, t[3:7])),
            site=(rest[2] if len(rest) >= 3 else "") or t[2][:2],
            avail=(rest[3] if len(rest) >= 4 else "yes")))
    return cases


def source_parquet(date: str, storage: Path) -> Path | None:
    name = f"combined_{date}_all_orbits.parquet"
    for base in (Path("results/csv_collection"), storage / "results/csv_collection"):
        if (base / name).exists():
            return base / name
    return None


# ── stage 1 ──────────────────────────────────────────────────────────────────

def stage1_worseners() -> pd.DataFrame:
    frames = []
    for tree, (out_base, _) in TREES.items():
        p = out_base / "tccon_comparison_r100km.csv"
        d = pd.read_csv(p)
        d["tree"] = tree
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.surface == "all") & (d.qf_group == "all")].copy()
    d["d_rmse"] = d.rmse_after - d.rmse_before
    d["d_absbias"] = d.bias_after.abs() - d.bias_before.abs()
    d["d_absbias_direct"] = (d.bias_after_direct.abs()
                             - d.bias_before_direct.abs())
    d["worse_rmse"] = d.d_rmse > 0
    d["worse_bias"] = d.d_absbias > 0

    # r50 persistence (atrain tree only — the robustness report exists there).
    r50p = TREES["atrain"][0] / "tccon_comparison_r50km.csv"
    if r50p.exists():
        r50 = pd.read_csv(r50p)
        r50 = r50[(r50.surface == "all") & (r50.qf_group == "all")]
        r50 = r50.assign(worse_rmse_r50=(r50.rmse_after - r50.rmse_before) > 0,
                         worse_bias_r50=(r50.bias_after.abs()
                                         - r50.bias_before.abs()) > 0)
        d = d.merge(r50[["site", "date", "worse_rmse_r50", "worse_bias_r50"]],
                    on=["site", "date"], how="left")
    else:
        d["worse_rmse_r50"] = np.nan
        d["worse_bias_r50"] = np.nan
    return d


# ── stage 2 ──────────────────────────────────────────────────────────────────

def stage2_footprints(radius_km: float, window_min: float) -> pd.DataFrame:
    storage = get_storage_dir()
    tccon_cache: dict[str, pd.DataFrame] = {}
    frames = []
    for tree, (out_base, script) in TREES.items():
        percase = pd.read_csv(out_base / "tccon_comparison_r100km.csv")
        percase = percase[(percase.surface == "all")
                          & (percase.qf_group == "all")]
        deltas = {(r.site, r.date): r.ak_delta for r in percase.itertuples()}
        for c in parse_cases(script):
            if c["avail"] != "yes":
                continue
            pq = find_plotdata(out_base, c["date"], c["site"])
            if pq is None:
                continue
            oco = pd.read_parquet(pq)
            sp = source_parquet(c["date"], storage)
            if sp is not None:
                avail = set(apq.ParquetFile(sp).schema_arrow.names)
                want = [col for col in SRC_DRIVERS + ["xco2_raw"]
                        if col in avail and col not in oco.columns]
                src = (pd.read_parquet(sp, columns=["time", "lon", "lat"] + want)
                         .drop_duplicates(["time", "lon", "lat"]))
                oco = oco.merge(src, on=["time", "lon", "lat"], how="left")
            name = c["tccon"]
            if name not in tccon_cache:
                pth = Path("data/TCCON") / name
                if not pth.exists():
                    pth = storage / "data/TCCON" / name
                tccon_cache[name] = load_tccon(str(pth))
            col = collocate(oco, tccon_cache[name], box=c["box"],
                            radius_km=radius_km, window_min=window_min,
                            site=c["site"])
            near, tref = col["near"], col["tccon_ref"]
            if not len(near) or not np.isfinite(tref):
                continue
            delta = deltas.get((c["site"], c["date"]), np.nan)
            ak_fallback = not np.isfinite(delta)
            if ak_fallback:
                delta = 0.0
            keep = [k for k in (["time", "lon", "lat", "sfc_type",
                                 "cld_dist_km", "xco2_bc", CORR, "is_guarded",
                                 "pred_anomaly"] + PD_DRIVERS + SRC_DRIVERS)
                    if k in near.columns]
            f = near[keep].copy()
            f["site"], f["date"], f["tree"] = c["site"], c["date"], tree
            f["ak_fallback"] = ak_fallback
            ref_ak, ref_dir = tref + delta, tref
            f["resid_before"] = f.xco2_bc - ref_ak
            f["resid_after"] = f[CORR] - ref_ak
            f["resid_before_direct"] = f.xco2_bc - ref_dir
            f["resid_after_direct"] = f[CORR] - ref_dir
            frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d["abs_lat"] = d.lat.abs()
    d["imp"] = d.resid_after.abs() - d.resid_before.abs()   # <0 ⇒ improved
    # cloud distance is only reliable pre-drift (Aqua free-drift ≥2023)
    d.loc[d.tree == "drift", "cld_dist_km"] = np.nan
    print(f"stage 2: {len(d):,} collocated footprints, "
          f"{d.groupby('tree').size().to_dict()}, "
          f"{d[['site', 'date']].drop_duplicates().shape[0]} station-days")
    return d


# ── stage 3 ──────────────────────────────────────────────────────────────────

def _rmse(x: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan


def stage3_bins(d: pd.DataFrame, surface: str, nbins: int = 10) -> pd.DataFrame:
    rows = []
    for colname, label, _ in BIN_DRIVERS:
        if colname not in d.columns:
            continue
        s = d[np.isfinite(d[colname])]
        if len(s) < 500:
            continue
        try:
            b = pd.qcut(s[colname], nbins, duplicates="drop")
        except ValueError:
            continue
        for iv, g in s.groupby(b, observed=True):
            z2 = ((g.resid_after / g.de_sigma) ** 2).mean() \
                if "de_sigma" in g.columns else np.nan
            rows.append(dict(
                surface=surface, driver=colname, label=label,
                lo=iv.left, hi=iv.right, center=g[colname].median(),
                n=len(g),
                rmse_before=_rmse(g.resid_before),
                rmse_after=_rmse(g.resid_after),
                bias_after=g.resid_after.mean(),
                mean_imp=g.imp.mean(),
                frac_worse=(g.imp > 0).mean(),
                z2=z2))
    # snow_flag as categorical
    if "snow_flag" in d.columns:
        s = d[np.isfinite(d.snow_flag)]
        for v, g in s.groupby(s.snow_flag.astype(int)):
            if len(g) < 50:
                continue
            rows.append(dict(
                surface=surface, driver="snow_flag", label="snow flag",
                lo=v, hi=v, center=v, n=len(g),
                rmse_before=_rmse(g.resid_before),
                rmse_after=_rmse(g.resid_after),
                bias_after=g.resid_after.mean(),
                mean_imp=g.imp.mean(),
                frac_worse=(g.imp > 0).mean(),
                z2=((g.resid_after / g.de_sigma) ** 2).mean()))
    return pd.DataFrame(rows)


def stage3_ols(d: pd.DataFrame, surface: str) -> pd.DataFrame:
    cols = [c for c in OLS_DRIVERS if c in d.columns]
    X = d[cols].astype(float)
    y = d.resid_after.abs().astype(float)
    m = np.isfinite(y)
    for c in cols:
        m &= np.isfinite(X[c])
    X, y = X[m], y[m]
    if len(y) < 1000:
        return pd.DataFrame()
    cols = [c for c in cols if X[c].std(ddof=0) > 0]      # drop constants
    X = X[cols]
    Xs = (X - X.mean()) / X.std(ddof=0)
    A = np.column_stack([np.ones(len(Xs)), Xs.to_numpy()])
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    resid = y.to_numpy() - A @ coef
    sigma2 = resid @ resid / (len(y) - A.shape[1])
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    out = pd.DataFrame(dict(
        surface=surface, driver=["intercept"] + cols,
        coef_ppm_per_sd=coef, se=se, t=coef / se))
    out["n"] = len(y)
    out["r2"] = r2
    return out


def sigma_awareness(d: pd.DataFrame) -> list[str]:
    """Is failure flagged by the model's own σ?"""
    lines = ["| surface | σ̄ improved fp | σ̄ worsened fp | frac worse, σ decile 1 | "
             "σ decile 10 | ⟨z²⟩ improved | ⟨z²⟩ worsened |",
             "|---|---|---|---|---|---|---|"]
    for surface, g in (("ocean", d[d.sfc_type == 0]), ("land", d[d.sfc_type == 1])):
        g = g[np.isfinite(g.de_sigma) & np.isfinite(g.imp)]
        w, i = g[g.imp > 0], g[g.imp <= 0]
        dec = pd.qcut(g.de_sigma, 10, labels=False, duplicates="drop")
        fw1 = (g.imp[dec == 0] > 0).mean()
        fw10 = (g.imp[dec == dec.max()] > 0).mean()
        z2i = ((i.resid_after / i.de_sigma) ** 2).mean()
        z2w = ((w.resid_after / w.de_sigma) ** 2).mean()
        lines.append(f"| {surface} | {i.de_sigma.mean():.2f} | "
                     f"{w.de_sigma.mean():.2f} | {fw1:.2f} | {fw10:.2f} | "
                     f"{z2i:.2f} | {z2w:.2f} |")
    return lines


def stage3_figure(bins: pd.DataFrame, out: Path):
    apply_manuscript_style()
    drivers = [b for b in BIN_DRIVERS if b[0] in set(bins.driver)]
    if "snow_flag" in set(bins.driver):
        drivers.append(("snow_flag", "snow flag", False))
    n = len(drivers)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    colors = {"ocean": "tab:blue", "land": "tab:red"}
    for k, (colname, label, _) in enumerate(drivers):
        ax = axes[k // ncol, k % ncol]
        for surface in ("ocean", "land"):
            g = bins[(bins.surface == surface) & (bins.driver == colname)]
            if not len(g):
                continue
            ax.plot(g.center, g.rmse_before, "--", color=colors[surface],
                    alpha=0.45, lw=1.2)
            ax.plot(g.center, g.rmse_after, "-o", color=colors[surface],
                    ms=2.5, lw=1.4, label=surface)
        ax.set_xlabel(label)
        ax.set_ylabel("fp-RMSE vs TCCON (ppm)")
        ax.grid(alpha=0.25, lw=0.4)
        if k == 0:
            ax.legend(frameon=False, fontsize=8,
                      title="after (solid) / before (dashed)",
                      title_fontsize=7)
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].set_visible(False)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ── stage 4 ──────────────────────────────────────────────────────────────────

def stage4_dossiers(d: pd.DataFrame, worse: pd.DataFrame) -> list[str]:
    drivers = [c for c in ["aod_total", "aod_dust", "sza", "abs_lat", "airmass",
                           "alb_o2a", "alb_sco2", "snow_flag", "tcwv", "dpfrac",
                           "snr_o2a", "cld_dist_km", "de_sigma"]
               if c in d.columns]
    stats = {s: (d[s].mean(), d[s].std()) for s in drivers}
    lines = ["| site | date | tree | ΔRMSE | Δ|bias| | n_fp | guard% | "
             "top anomalous drivers (z vs cohort) |",
             "|---|---|---|---|---|---|---|---|"]
    ww = worse[worse.worse_rmse | worse.worse_bias].sort_values(
        "d_rmse", ascending=False)
    for r in ww.itertuples():
        g = d[(d.site == r.site) & (d.date == r.date)]
        if not len(g):
            continue
        zs = {}
        for s in drivers:
            mu, sd = stats[s]
            if sd and np.isfinite(sd) and np.isfinite(g[s].mean()):
                zs[s] = (g[s].mean() - mu) / sd
        top = sorted(zs.items(), key=lambda kv: -abs(kv[1]))[:3]
        top_s = ", ".join(f"{k} {v:+.1f}σ" for k, v in top)
        lines.append(
            f"| {r.site} | {r.date} | {r.tree} | {r.d_rmse:+.2f} | "
            f"{r.d_absbias:+.2f} | {len(g):,} | "
            f"{100 * g.is_guarded.mean():.1f} | {top_s} |")
    return lines


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-km", type=float, default=100.0)
    ap.add_argument("--window-min", type=float, default=60.0)
    ap.add_argument("--out-dir", default=str(BASE / TAG / "failure_modes"))
    args = ap.parse_args()
    today = dt.date.today().isoformat()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rtag = f"r{int(args.radius_km)}km"

    worse = stage1_worseners()
    worse.to_csv(out_dir / f"worseners_{rtag}.csv", index=False)
    n_w = int(worse.worse_rmse.sum()), int(worse.worse_bias.sum()), len(worse)

    d = stage2_footprints(args.radius_km, args.window_min)
    d.to_parquet(out_dir / f"footprints_{rtag}.parquet", index=False)

    bins_all = []
    ols_all = []
    for surface, g in (("ocean", d[d.sfc_type == 0]),
                       ("land", d[d.sfc_type == 1])):
        b = stage3_bins(g, surface)
        b.to_csv(out_dir / f"strat_{surface}_{rtag}.csv", index=False)
        bins_all.append(b)
        ols_all.append(stage3_ols(g, surface))
    bins = pd.concat(bins_all, ignore_index=True)
    ols = pd.concat([o for o in ols_all if len(o)], ignore_index=True)
    ols.to_csv(out_dir / f"multivariate_{rtag}.csv", index=False)
    fig_p = out_dir / f"fig_failure_modes_{rtag}.png"
    stage3_figure(bins, fig_p)

    # ── writeup ──
    def bins_md(surface, drivers):
        lines = [f"| driver | bin | n | RMSE before | RMSE after | bias after | "
                 f"frac worse | ⟨z²⟩ |", "|---|---|---|---|---|---|---|---|"]
        g = bins[bins.surface == surface]
        for drv in drivers:
            gg = g[g.driver == drv]
            for r in gg.itertuples():
                lines.append(f"| {drv} | {r.lo:.3g}–{r.hi:.3g} | {r.n:,} | "
                             f"{r.rmse_before:.2f} | {r.rmse_after:.2f} | "
                             f"{r.bias_after:+.2f} | {r.frac_worse:.2f} | "
                             f"{r.z2:.1f} |")
        return lines

    def ols_md():
        lines = ["| surface | driver | coef (ppm / SD) | t | |",
                 "|---|---|---|---|---|"]
        for r in ols[ols.driver != "intercept"].itertuples():
            flag = "◀" if abs(r.t) > 10 else ""
            lines.append(f"| {r.surface} | {r.driver} | "
                         f"{r.coef_ppm_per_sd:+.3f} | {r.t:+.1f} | {flag} |")
        return lines

    # md table: RMSE worseners + MATERIAL |bias| worseners (Δ|bias| ≥ 0.25 ppm);
    # the full 35-row list (incl. near-zero-before sign flips) is in the CSV.
    w_lines = ["| site | date | tree | RMSE before→after | bias before→after | "
               "Δ|bias| direct | persists @r50 (rmse/bias) | n_oco |",
               "|---|---|---|---|---|---|---|---|"]
    ww = worse[worse.worse_rmse | (worse.d_absbias >= 0.25)].sort_values(
        ["worse_rmse", "d_rmse"], ascending=False)
    for r in ww.itertuples():
        p50 = (f"{r.worse_rmse_r50}/{r.worse_bias_r50}"
               if r.tree == "atrain" else "n/a")
        w_lines.append(
            f"| {r.site} | {r.date} | {r.tree} | "
            f"{r.rmse_before:.2f}→{r.rmse_after:.2f} | "
            f"{r.bias_before:+.2f}→{r.bias_after:+.2f} | "
            f"{r.d_absbias_direct:+.2f} | {p50} | {r.n_oco:,} |")

    md = [
        f"# ML-correction failure modes vs TCCON — {today}",
        "",
        f"Production `{TAG}` atrain (75) + drift (21) station-days, "
        f"{args.radius_km:.0f} km / ±{args.window_min:.0f} min, AK-harmonized "
        "reference (per-footprint Δ|resid| and scatter metrics are "
        "reference-invariant because ak_delta is a per-case constant; "
        "bias-anchored numbers under direct are in the CSVs). Ny-Ålesund is "
        "INCLUDED here (it is excluded from the headline aggregates).",
        "",
        "## 1. Station-days that got worse",
        "",
        f"fp-RMSE worseners: **{n_w[0]}/{n_w[2]}**; |bias| worseners: "
        f"**{n_w[1]}/{n_w[2]}** (surface=all, QF=all rows; both trees). The "
        "|bias| count reads worse than it is: it includes every sign flip from "
        "a near-zero before-bias, and the improvers improve by much more "
        "(pooled mean |bias| still drops 1.26 → 0.82). Table below = RMSE "
        "worseners + material |bias| worseners (Δ|bias| ≥ 0.25 ppm); full "
        "list in `worseners_r100km.csv`.",
        "",
        *w_lines, "",
        "## 2. Driver stratification (decile bins)", "",
        f"Footprint table: `failure_modes/footprints_{rtag}.parquet` "
        f"({len(d):,} rows). Figure: `{fig_p.name}` (per-driver fp-RMSE "
        "before/after, ocean vs land). Full bin tables in "
        f"`strat_{{ocean,land}}_{rtag}.csv`; selected land drivers:", "",
        *bins_md("land", ["aod_total", "sza", "alb_o2a", "snow_flag"]), "",
        "## 3. Multivariate view (standardized OLS of |resid_after|)", "",
        "Coefficient = ppm change in |corrected residual| per +1 SD of driver, "
        "all drivers jointly (footprint-pooled; big station-days dominate — "
        "read as within-sample attribution, not causal). sza / airmass / "
        "abs_lat are strongly collinear — interpret that trio as one "
        "geometry block, not separately.", "",
        *ols_md(), "",
        "## 4. Is failure flagged by the model's own σ?", "",
        *sigma_awareness(d), "",
        "## 5. Worsener dossiers (driver z-scores vs full cohort)", "",
        *stage4_dossiers(d, worse), "",
        "## Conclusions (2026-07-16 edition)", "",
        "1. **Failure is rare and small-amplitude.** 5/96 station-days worsen "
        "in fp-RMSE (max +0.44 ppm, vs improvements up to −2.9); material "
        "|bias| worseners are a minority and the pooled mean |bias| still "
        "improves 1.26 → 0.82. No worsener has any guard activity — these are "
        "in-distribution misses, not OOD blowups.",
        "2. **High AOD is NOT a relative failure mode — it is where the "
        "correction works best but the most error remains.** Top land-AOD "
        "decile (AOD > 0.13): before 6.10 → after 1.88 ppm (best skill, "
        "lowest frac-worse 0.19), yet AOD is the strongest land driver of the "
        "remaining |residual| (+0.26 ppm/SD, t≈98). Aerosol scenes are "
        "under-corrected in amplitude, not mis-corrected.",
        "3. **Bright surfaces are the per-footprint failure stratum.** Top "
        "alb_o2a decile (> 0.40): frac-worse 0.47 (highest of any bin), "
        "after-RMSE 1.39, and the most negative after-bias (−0.70). The "
        "df 2021-02-10 worsener (alb_sco2 +2.1σ) is its case instance — "
        "consistent with the WCO2 albedo-contrast sign rule.",
        "4. **Snow is NOT a failure mode.** snow_flag=1 (n=505): 9.39 → 2.73 "
        "ppm, frac-worse 0.17; the single biggest improver case "
        "(ny 2017-05-25, −2.87 ppm) is the snow case. Keeping snow data in "
        "training paid off.",
        "5. **High latitude fails at the BIAS level, not scatter.** ny "
        "2020-07-11: correction inert on a +1.7 ppm bias (1.78 → 1.78); the "
        "largest worsener overall is high-lat drift ka 2023-09-11 "
        "(−1.05 → −2.00). abs_lat barely drives footprint |residual| on land "
        "(+0.04 ppm/SD) — the high-lat problem is station-day anchoring "
        "(reference/airmass), not footprint noise. Matches the known "
        "~−0.3 ppm DL anchoring offset growing at high lat.",
        "6. **The model's σ advertises where it can fail — in the right "
        "direction.** On land, worsened footprints have LOWER σ than improved "
        "ones (0.67 vs 0.90) and frac-worse falls from 0.46 in the lowest σ "
        "decile to 0.06 in the highest: worsening is small corrections "
        "dithering already-small residuals, while high-σ (large-correction) "
        "footprints almost never get worse. On ocean σ is uninformative for "
        "worsening (frac-worse flat 0.22 → 0.24), but amplitudes there are "
        "small.",
        "",
    ]
    out_md = BASE / f"FAILURE_MODES_{today}.md"
    out_md.write_text("\n".join(md))
    print(f"wrote {out_md}")
    print(f"artifacts in {out_dir}")


if __name__ == "__main__":
    main()

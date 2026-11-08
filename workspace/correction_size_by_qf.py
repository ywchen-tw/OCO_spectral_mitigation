#!/usr/bin/env python3
"""Size of the applied production XGBoost correction per footprint.

Correction definition (sign convention): delta = xco2_corrected - xco2_bc = -mu,
where mu is the predicted near-cloud anomaly (xco2_corrected = xco2_bc - mu).
A positive delta raises XCO2 above the operational B11 value. Guarded
footprints keep xco2_bc, i.e. delta = 0, and stay in the sample.

Statistics are split by surface x xco2_qf (0, 1, all) x cloud distance:
  zone : near = cld_dist_km <= 5 km over ocean, <= 15 km over land; far =
         beyond; footprints without a cloud distance form a "nan" group.
  bin  : 0-2, 2-5, 5-10, 10-20, >20 km.

Populations
  eval   : 139 non-training 2014-2021 dates, per_date_nok2diag/ (production
           feature set no_xco2_sig_shape_nok2diag), all footprints QF 0/1.
  atrain : TCCON-collocated footprints 2014-2021 (cases of the production
           tccon_comparison_r100km.csv; box + site radius + 50 ppm sanity band
           vs the raw TCCON window mean, as in tccon_collocate.collocate).
  drift  : same for 2022-2025 (no cloud-distance split).

Outputs (default --out-dir: the correction_size/ sibling of per_date_nok2diag):
  {eval,atrain,drift}_correction_stats[_snowfree].csv, *_ratios*.csv,
  eval_coverage_check.csv, {atrain,drift}_collocation_check.csv, report.md.
  manuscript/JQSRT_draft/scripts/make_table_correction_size.py reads
  eval_correction_stats.csv and atrain_correction_stats.csv.

Usage:  python workspace/correction_size_by_qf.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
XGB = REPO / "results_doppler/model_comparison/xgb"
EVAL_DIR = XGB / "fm1620k5v2xt41_eval_dates/per_date_nok2diag"
PROD = XGB / "fm1620k5v2xt41_am_no_xco2_sig_shape_nok2diag"
SRC_DIR = REPO / "results_doppler/csv_collection"
DEFAULT_OUT = EVAL_DIR.parent / "correction_size"

SFC = {0: "ocean", 1: "land"}
BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, np.inf)]
BIN_LABELS = ["0-2", "2-5", "5-10", "10-20", ">20"]
NEAR_KM = {"ocean": 5.0, "land": 15.0}
MAX_ABS_ANOM = 100.0


def stats(delta: np.ndarray) -> dict:
    delta = delta[np.isfinite(delta)]
    n = delta.size
    if n == 0:
        return dict(n=0)
    a = np.abs(delta)
    med = np.median(delta)
    return dict(
        n=n,
        median=med,
        mean=delta.mean(),
        median_abs=np.median(a),
        mean_abs=a.mean(),
        robust_sd=1.4826 * np.median(np.abs(delta - med)),
        frac_gt05=(a > 0.5).mean(),
        frac_gt1=(a > 1.0).mean(),
        sum_abs=a.sum(),
    )


def zone_labels(df: pd.DataFrame) -> pd.Series:
    thr = df["surface"].map(NEAR_KM)
    z = np.where(df["cld_dist_km"] <= thr, "near", "far")
    return pd.Series(np.where(df["cld_dist_km"].isna(), "nan", z), index=df.index)


def bin_labels(d: pd.Series) -> pd.Series:
    out = pd.Series("nan", index=d.index, dtype=object)
    for (lo, hi), lab in zip(BINS, BIN_LABELS):
        out[(d >= lo) & (d < hi)] = lab
    return out


def table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, k)), **stats(g["delta"].to_numpy(float))})
    return pd.DataFrame(rows)


def all_tables(df: pd.DataFrame, with_cld: bool) -> pd.DataFrame:
    """Rows for surface x QF (QF 0, 1, all) x {bin, zone, all distances}."""
    df = df[["surface", "qf", "zone", "cbin", "delta"]]
    dfa = pd.concat([df, df.assign(qf="all")], ignore_index=True)
    parts = [table(dfa, ["surface", "qf"]).assign(split="all", group="all")]
    if with_cld:
        parts.append(table(dfa, ["surface", "qf", "zone"])
                     .rename(columns={"zone": "group"}).assign(split="zone"))
        parts.append(table(dfa, ["surface", "qf", "cbin"])
                     .rename(columns={"cbin": "group"}).assign(split="bin"))
    t = pd.concat(parts, ignore_index=True)
    order = {"all": 0, "near": 1, "far": 2, **{b: 3 + i for i, b in enumerate(BIN_LABELS)},
             "nan": 99}
    t["_o"] = t["group"].map(order)
    t["_s"] = t["surface"].map({"ocean": 0, "land": 1})
    t["_q"] = t["qf"].map({"0": 0, "1": 1, "all": 2})
    t = t.sort_values(["_s", "_q", "_o"]).drop(columns=["_o", "_s", "_q"])
    cols = ["surface", "qf", "split", "group"] + [c for c in t.columns
                                                 if c not in ("surface", "qf", "split", "group")]
    return t[cols].reset_index(drop=True)


def finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["xco2_qf"].isin([0, 1]) & df["sfc_type"].isin([0, 1])].copy()
    df["surface"] = df["sfc_type"].astype(int).map(SFC)
    df["qf"] = df["xco2_qf"].astype(int).astype(str)
    df["zone"] = zone_labels(df)
    df["cbin"] = bin_labels(df["cld_dist_km"])
    return df


# ── evaluation dates ─────────────────────────────────────────────────────────
def load_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["sounding_id", "sfc_type", "cld_dist_km", "xco2_qf", "xco2_bc",
            "guard_nok2diag", "mu_nok2diag", "corrected_nok2diag",
            "anom_bc_r05", "anom_nok2diag_r05", "anom_bc_r15", "anom_nok2diag_r15"]
    frames, cov = [], []
    for p in sorted(EVAL_DIR.glob("*.parquet")):
        d = pd.read_parquet(p, columns=cols)
        d["date"] = p.stem
        src = SRC_DIR / f"combined_{p.stem}_all_orbits.parquet"
        if src.exists():
            s = pd.read_parquet(src, columns=["fp_id", "snow_flag"])
            d = d.merge(s.rename(columns={"fp_id": "sounding_id"}), on="sounding_id",
                        how="left")
        else:
            d["snow_flag"] = np.nan
        delta = (d["corrected_nok2diag"] - d["xco2_bc"]).to_numpy(float)
        resid = np.nanmax(np.abs(delta + np.where(d["guard_nok2diag"], 0.0,
                                                  d["mu_nok2diag"])))
        cov.append(dict(date=p.stem, rows=len(d), guarded=int(d["guard_nok2diag"].sum()),
                        snow_src=src.exists(), snow_nan=int(d["snow_flag"].isna().sum()),
                        max_resid_delta_plus_mu=resid))
        d["delta"] = delta
        frames.append(d)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(cov)


# ── TCCON-collocated footprints ──────────────────────────────────────────────
def haversine_km(lon, lat, lon0, lat0):
    lon, lat = np.radians(lon), np.radians(lat)
    lon0, lat0 = np.radians(lon0), np.radians(lat0)
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def run_cases(script: Path) -> pd.DataFrame:
    rows = []
    for ln in script.read_text().splitlines():
        if not re.match(r"^run_case\s", ln):
            continue
        t = ln.split("#")[0].split()
        rest = t[9:]
        site = rest[2] if len(rest) >= 3 else ""
        rows.append(dict(date=t[1], tccon=t[2], lonmin=float(t[3]), lonmax=float(t[4]),
                         latmin=float(t[5]), latmax=float(t[6]),
                         site=site or t[2][:2]))
    return pd.DataFrame(rows)


def load_tccon(base: Path, script: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(REPO / "workspace"))
    from tccon_collocate import SITE_COORDS  # noqa: E402

    comp = pd.read_csv(base / "tccon_comparison_r100km.csv")
    comp_all = comp[(comp["qf_group"] == "all") & (comp["surface"] == "all")]
    cases = run_cases(script).drop_duplicates(["date", "site"])
    frames, chk = [], []
    for _, r in comp_all.iterrows():
        c = cases[(cases["date"] == r["date"]) & (cases["site"] == r["site"])]
        if c.empty:
            print("no run_case for", r["site"], r["date"])
            continue
        c = c.iloc[0]
        pq = base / f"combined_{r['date']}_{r['site']}" / "plot_data.parquet"
        oco = pd.read_parquet(pq)
        src = pd.read_parquet(SRC_DIR / f"combined_{r['date']}_all_orbits.parquet",
                              columns=["time", "lon", "lat", "xco2_qf", "snow_flag"]
                              ).drop_duplicates(["time", "lon", "lat"])
        oco = oco.merge(src, on=["time", "lon", "lat"], how="left")
        box = ((oco["lon"] >= c.lonmin) & (oco["lon"] <= c.lonmax)
               & (oco["lat"] >= c.latmin) & (oco["lat"] <= c.latmax))
        oco = oco[box]
        st_lon, st_lat = SITE_COORDS[c.tccon[:2].lower()]
        oco = oco[haversine_km(oco["lon"].to_numpy(), oco["lat"].to_numpy(),
                               st_lon, st_lat) <= r["radius_km"]]
        if np.isfinite(r["tccon_mu_raw"]):
            oco = oco[np.abs(oco["xco2_bc"] - r["tccon_mu_raw"]) < 50.0]
        oco = oco.assign(site=r["site"], date=r["date"])
        oco["delta"] = oco["xgb_corrected_xco2"] - oco["xco2_bc"]
        guard = oco["clim_guard"].astype(bool) | oco["anomaly_guard"].astype(bool)
        resid = (np.nanmax(np.abs(oco["delta"] + np.where(guard, 0.0, oco["pred_anomaly"])))
                 if len(oco) else np.nan)
        chk.append(dict(site=r["site"], date=r["date"], n_report=int(r["n_oco"]),
                        n_mine=len(oco), qf_nan=int(oco["xco2_qf"].isna().sum()),
                        guarded=int(guard.sum()), max_resid_delta_plus_mu=resid))
        frames.append(oco)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(chk)


def ratios(t: pd.DataFrame, df: pd.DataFrame, with_cld: bool) -> pd.DataFrame:
    def get(sfc, qf, split, group, col):
        v = t[(t.surface == sfc) & (t.qf == qf) & (t.split == split) & (t.group == group)]
        return float(v[col].iloc[0]) if len(v) else np.nan

    rows = []
    for sfc in ("ocean", "land"):
        sub = df[df.surface == sfc]
        a = sub["delta"].abs()
        r = dict(surface=sfc,
                 qf1_share_of_footprints=(sub.qf == "1").mean(),
                 qf1_share_of_sum_abs_delta=a[sub.qf == "1"].sum() / a.sum(),
                 qf1_share_of_abs_gt1=((sub.qf == "1") & (a > 1)).sum() / (a > 1).sum(),
                 medabs_qf0_over_qf1=get(sfc, "0", "all", "all", "median_abs")
                 / get(sfc, "1", "all", "all", "median_abs"))
        if with_cld:
            r.update(
                medabs_qf0far=get(sfc, "0", "zone", "far", "median_abs"),
                medabs_qf0near=get(sfc, "0", "zone", "near", "median_abs"),
                medabs_qf1near=get(sfc, "1", "zone", "near", "median_abs"),
                medabs_qf1far=get(sfc, "1", "zone", "far", "median_abs"),
                medabs_qf0far_over_qf1near=get(sfc, "0", "zone", "far", "median_abs")
                / get(sfc, "1", "zone", "near", "median_abs"),
                medabs_qf0far_over_qf0near=get(sfc, "0", "zone", "far", "median_abs")
                / get(sfc, "0", "zone", "near", "median_abs"),
                fgt05_qf0far=get(sfc, "0", "zone", "far", "frac_gt05"),
                fgt05_qf1near=get(sfc, "1", "zone", "near", "frac_gt05"),
            )
        rows.append(r)
    return pd.DataFrame(rows)


def md(t: pd.DataFrame, cols: list[str]) -> str:
    t = t[cols].copy()
    for c in t.columns:
        if c in ("frac_gt05", "frac_gt1"):
            t[c] = (100 * t[c]).map(lambda v: f"{v:.1f}%")
        elif c == "n":
            t[c] = t[c].map(lambda v: f"{int(v):,}")
        elif t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda v: f"{v:+.3f}" if c in ("median", "mean") else f"{v:.3f}")
    head = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in t.values) + "\n"


MDCOLS = ["surface", "qf", "group", "n", "median", "median_abs", "robust_sd",
          "frac_gt05", "frac_gt1"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default: {DEFAULT_OUT.relative_to(REPO)})")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    report = []
    # ── eval dates
    raw, cov = load_eval()
    cov.to_csv(out / "eval_coverage_check.csv", index=False)
    ev = finish(raw)
    report.append(f"# eval dates: {cov.date.nunique()} dates, {cov.rows.sum():,} rows "
                  f"({len(ev):,} with QF 0/1 and sfc 0/1); guarded {cov.guarded.sum():,}; "
                  f"max |delta+mu| {cov.max_resid_delta_plus_mu.max():.2e}; "
                  f"dates without snow source {int((~cov.snow_src).sum())}; "
                  f"snow NaN rows {cov.snow_nan.sum():,}; "
                  f"cld_dist NaN rows {int(ev.cld_dist_km.isna().sum()):,}; "
                  f"cld_dist max {ev.cld_dist_km.max():.2f}\n")
    for sfc in ("ocean", "land"):
        report.append(f"rows {sfc}: {(raw.sfc_type == (0 if sfc == 'ocean' else 1)).sum():,}\n")

    t_eval = all_tables(ev, True)
    t_eval.to_csv(out / "eval_correction_stats.csv", index=False)
    t_eval_ns = all_tables(ev[ev.snow_flag != 1], True)
    t_eval_ns.to_csv(out / "eval_correction_stats_snowfree.csv", index=False)
    t_eval_nonan = all_tables(ev[ev.cld_dist_km.notna()], True)
    r_eval = ratios(t_eval, ev, True)
    r_eval.to_csv(out / "eval_ratios.csv", index=False)
    ratios(t_eval_ns, ev[ev.snow_flag != 1], True).to_csv(
        out / "eval_ratios_snowfree.csv", index=False)

    # sign check vs fig03 headline: ocean <=5 km median anomaly before/after
    oc = ev[(ev.surface == "ocean") & (ev.cld_dist_km <= 5)]
    for col in ("anom_bc_r05", "anom_nok2diag_r05"):
        v = oc[col].to_numpy(float)
        v = v[np.isfinite(v) & (np.abs(v) <= MAX_ABS_ANOM)]
        report.append(f"ocean <=5 km {col}: n={v.size:,} mean={v.mean():+.3f} "
                      f"median={np.median(v):+.3f}\n")
    report.append(f"snow share land: {(ev[ev.surface == 'land'].snow_flag == 1).mean():.4f}\n")
    diff = (t_eval.set_index(["surface", "qf", "split", "group"])[["n", "median", "median_abs"]]
            - t_eval_nonan.set_index(["surface", "qf", "split", "group"])[["n", "median",
                                                                           "median_abs"]])
    report.append(f"drop-NaN-distance max abs diff:\n{diff.abs().max().to_string()}\n")

    report.append("\n## eval: all distances + near/far\n")
    report.append(md(t_eval[t_eval.split.isin(["all", "zone"])], MDCOLS))
    report.append("\n## eval: by cloud-distance bin\n")
    report.append(md(t_eval[t_eval.split == "bin"], MDCOLS))
    report.append("\n## eval snow-free (land changes only): all + near/far\n")
    report.append(md(t_eval_ns[t_eval_ns.split.isin(["all", "zone"])
                               & (t_eval_ns.surface == "land")], MDCOLS))
    report.append("\n## eval ratios\n" + r_eval.T.to_string() + "\n")
    del raw, ev

    # ── TCCON 2014-2021 (atrain) and 2022-2025 (drift)
    for name, script, with_cld in (
            ("atrain", REPO / "curc_shell_blanca_plot_corr_xco2_deepens.sh", True),
            ("drift", REPO / "curc_shell_blanca_plot_corr_xco2_deepens_drift.sh", False)):
        raw, chk = load_tccon(PROD / name, script)
        chk.to_csv(out / f"{name}_collocation_check.csv", index=False)
        tc = finish(raw)
        t = all_tables(tc, with_cld)
        t.to_csv(out / f"{name}_correction_stats.csv", index=False)
        r = ratios(t, tc, with_cld)
        r.to_csv(out / f"{name}_ratios.csv", index=False)
        report.append(f"\n# {name}: {len(chk)} cases, n_mine={chk.n_mine.sum():,} "
                      f"n_report={chk.n_report.sum():,} "
                      f"(cases mismatched: {(chk.n_mine != chk.n_report).sum()}); "
                      f"QF NaN {chk.qf_nan.sum():,}; guarded {chk.guarded.sum():,}; "
                      f"max |delta+mu| {chk.max_resid_delta_plus_mu.max():.2e}; "
                      f"snow share land {(tc[tc.surface == 'land'].snow_flag == 1).mean():.4f}\n")
        report.append(md(t[t.split.isin(["all", "zone"])], MDCOLS))
        report.append(f"\n## {name} ratios\n" + r.T.to_string() + "\n")
        tns = all_tables(tc[tc.snow_flag != 1], with_cld)
        tns.to_csv(out / f"{name}_correction_stats_snowfree.csv", index=False)

    (out / "report.md").write_text("".join(report))
    print("".join(report))


if __name__ == "__main__":
    main()

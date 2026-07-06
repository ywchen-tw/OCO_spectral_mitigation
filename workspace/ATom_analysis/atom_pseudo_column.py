#!/usr/bin/env python3
"""Stage 2/3 — ATom XCO2 pseudo-columns vs OCO-2 (raw/bc/DE-corrected), AK-harmonized.

For each usable ATom date and each vertical profile leg that has collocated OCO-2
ocean-glint footprints (100 km / ±2 h), this:

  1. builds the aircraft pseudo-column CO2 profile from the leg (measured p, CO2),
     extended below the floor (hold lowest value) and ABOVE the ceiling with the
     OCO-2 prior profile — so unmeasured stratosphere contributes only the prior;
  2. applies the OCO-2 column operator (Rodgers & Connor 2003 / Wunch et al. 2017;
     reusing workspace/ak_harmonize.py) to get the AK-smoothed pseudo-column
        c_ak = c_a + Σ_j h_j a_j (x_j − x_a,j)
     with h/a/x_a/p_levels averaged over the collocated footprints (the parquet's
     ak_NN/pwf_NN/co2_ap_NN/plev_NN columns);
  3. compares c_ak to the collocated OCO-2 xco2_bc and DE-corrected XCO2, overall
     and split by cloud distance — the near-cloud test of the correction.

Inputs (all local):
  ATom profiles : $OUT/atom_merged/atom_merged_<date>.parquet (merge_atom_profiles.py)
  OCO operator  : results/csv_collection/combined_<date>_all_orbits.parquet
  DE-corrected  : results/model_comparison/deep_ensemble/<TAG>/atom/combined_<date>_atom/plot_data.parquet

Output: $OUT/atom_pseudo_column_results.csv  (one row per collocated leg)

Usage: python atom_pseudo_column.py [--radius-km 100] [--window-min 120] [--min-n 3]
"""
from __future__ import annotations
import argparse, datetime as dt, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # for ak_harmonize (workspace/)
sys.path.insert(0, os.path.join(HERE, ".."))
from ak_harmonize import operator_from_dataframe, _haversine_km  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV_DIR = os.path.join(REPO, "results", "csv_collection")
TAG = "de_beta_nll_prof_reg_o05l15_m5"
OUT_BASE = os.path.join(REPO, "results", "model_comparison", "deep_ensemble", TAG, "atom")
PLOT_BASE = OUT_BASE                              # per-case combined_<date>_atom/ dirs
MERGED_DIR = os.path.join(OUT_BASE, "atom_merged")  # merged profiles (input)
EPOCH = dt.datetime(1970, 1, 1)

DATES = ["2017-01-26", "2017-02-04", "2017-02-06", "2017-02-10", "2017-10-09",
         "2017-10-20", "2017-10-27", "2018-05-12"]
CORR_COL = "deep_ensemble_corrected_xco2"

# Flights whose OCO coincidence day (key everywhere) is the flight's 2nd UTC day, so
# the merged ATom profile lives under the flight (1st) date.
OCO_TO_FLIGHT = {"2017-02-04": "2017-02-03", "2017-02-06": "2017-02-05",
                 "2017-10-09": "2017-10-08"}


def load_atom(date: str) -> pd.DataFrame:
    ymd = OCO_TO_FLIGHT.get(date, date).replace("-", "")
    p = os.path.join(MERGED_DIR, f"atom_merged_{ymd}.parquet")
    df = pd.read_parquet(p, columns=["time_utc_s", "lat", "lon", "p_hpa", "co2_ppm",
                                      "profile_id", "leg_dir"])
    base = (dt.datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])) - EPOCH).total_seconds()
    df["epoch"] = base + df["time_utc_s"].to_numpy()
    return df[df["profile_id"] >= 0]


def load_oco(date: str) -> pd.DataFrame:
    """Ocean good-QF OCO-2 footprints with AK operator cols + DE-corrected XCO2."""
    op_cols = (["time", "lon", "lat", "sfc_type", "xco2_qf", "xco2_bc", "xco2_raw",
                "xco2_apriori", "cld_dist_km"]
               + [f"{p}_{i:02d}" for p in ("ak", "pwf", "co2_ap", "plev") for i in range(20)])
    oco = pd.read_parquet(os.path.join(CSV_DIR, f"combined_{date}_all_orbits.parquet"),
                          columns=op_cols)
    oco = oco[(oco.sfc_type == 0) & (oco.xco2_qf == 0)].copy()
    pd_path = os.path.join(PLOT_BASE, f"combined_{date}_atom", "plot_data.parquet")
    corr = pd.read_parquet(pd_path, columns=["time", "lon", "lat", CORR_COL])
    # exact merge on (time,lat,lon) — plot_data is derived from combined, bytes match
    oco = oco.merge(corr, on=["time", "lat", "lon"], how="left")
    return oco


def pseudo_profile_on_grid(leg: pd.DataFrame, op: dict, n_bins: int = 60):
    """ATom leg CO2(p) → OCO-2 20-level grid. Below floor: hold lowest; above
    ceiling: OCO-2 prior (op['xa']). Returns x_on_oco (ppm, len 20) or None."""
    p = leg["p_hpa"].to_numpy(); c = leg["co2_ppm"].to_numpy()
    m = np.isfinite(p) & np.isfinite(c) & (p > 0)
    p, c = p[m], c[m]
    if p.size < 20:
        return None
    # bin into monotonic profile in pressure (also de-noises the 1-Hz data)
    edges = np.linspace(p.min(), p.max(), n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    pb = np.array([p[idx == b].mean() for b in range(n_bins) if (idx == b).any()])
    cb = np.array([c[idx == b].mean() for b in range(n_bins) if (idx == b).any()])
    order = np.argsort(pb)                       # ascending p → ascending log p
    logp_atom, c_atom = np.log(pb[order]), cb[order]

    pl = op["pl"]                                # OCO 20-level pressures (hPa)
    x = np.interp(np.log(pl), logp_atom, c_atom)  # clamps: below-floor→lowest, above-ceil→ceil
    above = pl < p.min()                         # OCO levels above the aircraft ceiling
    x[above] = op["xa"][above]                   # replace clamped-ceiling with OCO prior
    return x, float(p.min()), float(p.max())


def process(date, oco, atom, radius_km, twin_s, min_n):
    rows = []
    olon = oco.lon.to_numpy(); olat = oco.lat.to_numpy(); ot = oco.time.to_numpy()
    for pid, leg in atom.groupby("profile_id"):
        alon = leg.lon.to_numpy(); alat = leg.lat.to_numpy(); at = leg.epoch.to_numpy()
        # OCO soundings within radius of ANY leg point AND within twin of nearest leg pt
        # (chunk over OCO to bound memory)
        hit = np.zeros(len(oco), bool); dmin = np.full(len(oco), np.inf)
        for i0 in range(0, len(oco), 400):
            sl = slice(i0, i0 + 400)
            d = _haversine_km(olon[sl][:, None], olat[sl][:, None], alon[None, :], alat[None, :])
            tg = np.abs(ot[sl][:, None] - at[None, :])
            dm = np.where(tg <= twin_s, d, np.inf).min(axis=1)
            dmin[sl] = dm; hit[sl] = dm <= radius_km
        sub = oco[hit]
        if len(sub) < min_n:
            continue
        op = operator_from_dataframe(sub, min_n=min_n)
        if op is None:
            continue
        prof = pseudo_profile_on_grid(leg, op)
        if prof is None:
            continue
        x_on_oco, p_ceiling, p_floor = prof
        c_ak = op["ca"] + np.nansum(op["h"] * op["a"] * (x_on_oco - op["xa"]))
        c_direct = float(np.nansum(op["h"] * x_on_oco))     # pressure-weighted, no AK
        cld = sub.cld_dist_km.to_numpy()
        rows.append(dict(
            date=date, profile_id=int(pid), leg_dir=leg.leg_dir.iloc[0],
            lat=float(leg.lat.mean()), lon=float(leg.lon.mean()),
            n_oco=len(sub), n_near=int((cld <= 10).sum()),
            cld_min=float(np.nanmin(cld)), cld_med=float(np.nanmedian(cld)),
            p_ceiling_hpa=round(p_ceiling, 1), p_floor_hpa=round(p_floor, 1),
            atom_xco2_ak=round(float(c_ak), 3), atom_xco2_direct=round(c_direct, 3),
            oco_xco2_apriori=round(op["ca"], 3),
            oco_xco2_raw=round(float(sub.xco2_raw.mean()), 3),
            oco_xco2_bc=round(float(sub.xco2_bc.mean()), 3),
            oco_xco2_corr=round(float(sub[CORR_COL].mean()), 3),
            # 1σ spread of the collocated OCO-2 soundings (error bars when comparing bias)
            oco_bc_sd=round(float(sub.xco2_bc.std()), 3),
            oco_corr_sd=round(float(sub[CORR_COL].std()), 3),
            resid_bc=round(float(sub.xco2_bc.mean()) - float(c_ak), 3),
            resid_corr=round(float(sub[CORR_COL].mean()) - float(c_ak), 3),
        ))
    return rows


def make_summary_plot(df, out_png):
    """Two-panel bias summary with ±1σ (spread of collocated OCO-2 soundings) error
    bars, mirroring tccon_comparison_report's dumbbell (errorbar = OCO-2 σ)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.sort_values("cld_med").reset_index(drop=True)
    y = np.arange(len(d))
    lbl = [f"{r.date[5:]} L{r.profile_id}" for r in d.itertuples()]
    RED, BLUE = "#d62728", "#1f77b4"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # (A) per-leg signed bias bc→corr, xerr = 1σ of the collocated OCO-2 soundings
    for yi, rb, rc in zip(y, d.resid_bc, d.resid_corr):
        ax1.plot([rb, rc], [yi, yi], "-", color="0.75", zorder=1)
    ax1.errorbar(d.resid_bc, y, xerr=d.oco_bc_sd, fmt="o", ms=6, color=RED, ecolor=RED,
                 elinewidth=1, capsize=2, zorder=2, label="xco2_bc − ATom")
    ax1.errorbar(d.resid_corr, y, xerr=d.oco_corr_sd, fmt="o", ms=6, color=BLUE, ecolor=BLUE,
                 elinewidth=1, capsize=2, zorder=3, label="DE-corrected − ATom")
    ax1.axvline(0, color="k", lw=0.7)
    ax1.set_yticks(y); ax1.set_yticklabels(lbl, fontsize=8); ax1.invert_yaxis()
    ax1.set_xlabel("OCO-2 − ATom pseudo-column (ppm)   [±1σ of collocated soundings]")
    ax1.set_title("Per-leg bias: xco2_bc → DE-corrected"); ax1.legend(fontsize=8)

    # (B) bias vs cloud distance, yerr = 1σ of the collocated OCO-2 soundings
    ax2.axhline(0, color="k", lw=0.7)
    ax2.errorbar(d.cld_med, d.resid_bc, yerr=d.oco_bc_sd, fmt="o", color=RED, mfc="none",
                 ms=7, elinewidth=0.8, capsize=2, label="xco2_bc")
    ax2.errorbar(d.cld_med, d.resid_corr, yerr=d.oco_corr_sd, fmt="o", color=BLUE,
                 ms=7, elinewidth=0.8, capsize=2, label="DE-corrected")
    for r in d.itertuples():
        ax2.plot([r.cld_med, r.cld_med], [r.resid_bc, r.resid_corr], color="0.8", zorder=0)
    ax2.set_xlabel("median cloud distance of collocated OCO-2 (km)")
    ax2.set_ylabel("OCO-2 − ATom (ppm)"); ax2.set_title("Bias vs cloud distance")
    ax2.legend(fontsize=8)

    nc = d[d.cld_med <= 10]
    fig.suptitle(
        f"ATom pseudo-column vs OCO-2 ocean-glint (AK-harmonized) — "
        f"{len(d)} legs, {len(nc)} near-cloud\n"
        f"near-cloud mean bias {nc.resid_bc.mean():+.2f}±{nc.resid_bc.std():.2f} → "
        f"{nc.resid_corr.mean():+.2f}±{nc.resid_corr.std():.2f} ppm  (±1σ across legs)",
        fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius-km", type=float, default=100)
    ap.add_argument("--window-min", type=float, default=120)
    ap.add_argument("--min-n", type=int, default=3)
    args = ap.parse_args()
    twin = args.window_min * 60

    all_rows = []
    for date in DATES:
        try:
            oco = load_oco(date); atom = load_atom(date)
        except FileNotFoundError as e:
            print(f"{date}: skip ({e})"); continue
        rows = process(date, oco, atom, args.radius_km, twin, args.min_n)
        print(f"{date}: {len(rows)} collocated leg(s)")
        all_rows += rows

    if not all_rows:
        print("no collocated legs"); return
    df = pd.DataFrame(all_rows)
    os.makedirs(OUT_BASE, exist_ok=True)
    out = os.path.join(OUT_BASE, "atom_pseudo_column_results.csv")
    df.to_csv(out, index=False)

    show = ["date", "profile_id", "leg_dir", "n_oco", "n_near", "cld_med",
            "p_ceiling_hpa", "atom_xco2_ak", "oco_xco2_bc", "oco_bc_sd",
            "oco_xco2_corr", "oco_corr_sd", "resid_bc", "resid_corr"]
    print("\n" + df[show].to_string(index=False))
    print(f"\n=== residual vs ATom pseudo-column (ppm; ±1σ across legs) ===")
    for lab, d in [("ALL legs", df),
                   ("near-cloud legs (cld_med<=10km)", df[df.cld_med <= 10])]:
        if len(d):
            print(f"{lab}  (n={len(d)}):  |resid_bc| mean {d.resid_bc.abs().mean():.3f}"
                  f"  ->  |resid_corr| mean {d.resid_corr.abs().mean():.3f}"
                  f"   bias_bc {d.resid_bc.mean():+.3f}±{d.resid_bc.std():.3f} -> "
                  f"bias_corr {d.resid_corr.mean():+.3f}±{d.resid_corr.std():.3f}")
    print(f"\nwrote {out}")
    make_summary_plot(df, os.path.join(OUT_BASE, "atom_pseudo_column_summary.png"))


if __name__ == "__main__":
    main()

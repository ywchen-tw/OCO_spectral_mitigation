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
from plot_style import XCO2_BC_LABEL, apply_manuscript_style, panel_label  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV_DIR = os.path.join(REPO, "results", "csv_collection")
TAG = "de_beta_nll_prof_reg_foldpca_o05l15_m5"
DE_ATOM = os.path.join(REPO, "results", "model_comparison", "deep_ensemble", TAG, "atom")
# The following four are DEFAULTS (production DE); --out-base/--plot-base/--merged-dir/
# --corr-col override them so the SAME pipeline scores the linreg/xgb baselines against
# the SAME aircraft pseudo-columns.  atom_merged is model-independent, so baselines
# reuse the DE tree's merged profiles (default MERGED_DIR) rather than re-deriving them.
OUT_BASE = DE_ATOM                                # where results CSV/plot are written
PLOT_BASE = DE_ATOM                               # per-case combined_<date>_atom/ dirs
MERGED_DIR = os.path.join(DE_ATOM, "atom_merged")  # merged profiles (input)
EPOCH = dt.datetime(1970, 1, 1)

DATES = ["2017-01-26", "2017-02-04", "2017-02-06", "2017-02-10", "2017-10-09",
         "2017-10-20", "2017-10-27", "2018-05-12"]
CORR_COL = "deep_ensemble_corrected_xco2"
MODEL_LABEL = "DE-corrected"                      # cosmetic (plot legends/titles)

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
    ceiling: OCO-2 prior (op['xa']). Returns (x_on_oco, sig_meas_on_oco,
    p_ceiling, p_floor) or None, where sig_meas is the per-level within-layer
    CO2 scatter (0 above the aircraft ceiling — the stratospheric term is added
    separately in process())."""
    p = leg["p_hpa"].to_numpy(); c = leg["co2_ppm"].to_numpy()
    m = np.isfinite(p) & np.isfinite(c) & (p > 0)
    p, c = p[m], c[m]
    if p.size < 20:
        return None
    # bin into monotonic profile in pressure (also de-noises the 1-Hz data);
    # per-bin std = within-layer CO2 scatter (the measured-column uncertainty)
    edges = np.linspace(p.min(), p.max(), n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    filled = [b for b in range(n_bins) if (idx == b).any()]
    pb = np.array([p[idx == b].mean() for b in filled])
    cb = np.array([c[idx == b].mean() for b in filled])
    sb = np.nan_to_num(np.array([c[idx == b].std() for b in filled]))
    order = np.argsort(pb)                       # ascending p → ascending log p
    logp_atom, c_atom, s_atom = np.log(pb[order]), cb[order], sb[order]

    pl = op["pl"]                                # OCO 20-level pressures (hPa)
    logpl = np.log(pl)
    x = np.interp(logpl, logp_atom, c_atom)      # clamps: below-floor→lowest, above-ceil→ceil
    sig = np.interp(logpl, logp_atom, s_atom)    # per-level within-layer scatter
    above = pl < p.min()                         # OCO levels above the aircraft ceiling
    x[above] = op["xa"][above]                   # replace clamped-ceiling with OCO prior
    sig[above] = 0.0                             # measured term = 0 above ceiling
    return x, sig, float(p.min()), float(p.max())


def process(date, oco, atom, radius_km, twin_s, min_n, strat_prior_sd=1.0):
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
        x_on_oco, sig_meas, p_ceiling, p_floor = prof
        w = op["h"] * op["a"]                                # AK-weighted pressure weights
        c_ak = op["ca"] + np.nansum(w * (x_on_oco - op["xa"]))
        c_direct = float(np.nansum(op["h"] * x_on_oco))     # pressure-weighted, no AK
        # pseudo-column-average uncertainty: within-layer aircraft scatter propagated
        # through the AK weights ⊕ the stratospheric prior-fill above the ceiling
        # (AK-weighted column fraction there × a nominal prior σ).
        sigma_meas = float(np.sqrt(np.nansum((w * sig_meas) ** 2)))
        frac_strat = float(np.nansum(w[op["pl"] < p_ceiling]))
        sigma_strat = frac_strat * strat_prior_sd
        atom_ak_sd = float(np.hypot(sigma_meas, sigma_strat))
        cld = sub.cld_dist_km.to_numpy()
        rows.append(dict(
            date=date, profile_id=int(pid), leg_dir=leg.leg_dir.iloc[0],
            lat=float(leg.lat.mean()), lon=float(leg.lon.mean()),
            n_oco=len(sub), n_near=int((cld <= 10).sum()),
            cld_min=float(np.nanmin(cld)), cld_med=float(np.nanmedian(cld)),
            p_ceiling_hpa=round(p_ceiling, 1), p_floor_hpa=round(p_floor, 1),
            atom_xco2_ak=round(float(c_ak), 3), atom_xco2_direct=round(c_direct, 3),
            # ATom pseudo-column-average uncertainty (measured ⊕ stratospheric-fill)
            atom_ak_sd=round(atom_ak_sd, 3),
            atom_sd_meas=round(sigma_meas, 3), atom_sd_strat=round(sigma_strat, 3),
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


def make_summary_plot(df, out_png, panel_offset=0, suptitle=True,
                      out_pdf=None):
    """Two-panel bias summary. Residual error bars = the collocated OCO-2 sounding
    spread (per leg). Each leg's OWN ATom pseudo-column-average ±1σ is drawn as a
    grey bar around the reference (0) — individual per leg, not a pooled mean."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    apply_manuscript_style()   # Arial (AMT), Arial mathtext, thin axes, 300 dpi
    d = df.sort_values("cld_med").reset_index(drop=True)
    y = np.arange(len(d))
    lbl = [f"{r.date[5:]} L{r.profile_id}" for r in d.itertuples()]
    RED, BLUE, GREY = "#d62728", "#1f77b4", "0.45"
    sd = d.atom_ak_sd.to_numpy()                 # per-leg pseudo-column σ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # (A) per-leg signed bias bc→corr; xerr = OCO spread; grey box per row = that
    #     leg's own ATom pseudo-column ±1σ around the reference (0)
    for yi, s in zip(y, sd):
        ax1.fill_betweenx([yi - 0.4, yi + 0.4], -s, s, color=GREY, alpha=0.25, zorder=0)
    for yi, rb, rc in zip(y, d.resid_bc, d.resid_corr):
        ax1.plot([rb, rc], [yi, yi], "-", color="0.75", zorder=1)
    ax1.errorbar(d.resid_bc, y, xerr=d.oco_bc_sd, fmt="o", ms=6, color=RED, ecolor=RED,
                 elinewidth=1, capsize=2, zorder=2,
                 label=f"{XCO2_BC_LABEL} − ATom")
    ax1.errorbar(d.resid_corr, y, xerr=d.oco_corr_sd, fmt="o", ms=6, color=BLUE, ecolor=BLUE,
                 elinewidth=1, capsize=2, zorder=3, label=f"{MODEL_LABEL} − ATom")
    ax1.axvline(0, color="k", lw=0.7)
    ax1.set_yticks(y); ax1.set_yticklabels(lbl, fontsize=8); ax1.invert_yaxis()
    ax1.set_xlabel("OCO-2 − ATom pseudo-column (ppm)   [error bars = OCO sounding spread]")
    ax1.set_title(f"Per-leg bias: {XCO2_BC_LABEL} → {MODEL_LABEL}")
    ax1.legend(handles=[Patch(facecolor=GREY, alpha=0.25, label="ATom pseudo-column ±1σ (per leg)"),
                        *ax1.get_legend_handles_labels()[0]], fontsize=7)
    panel_label(ax1, f"({chr(ord('a') + panel_offset)})")

    # (B) bias vs cloud distance; per-leg grey bar at each x = that leg's pseudo-column ±1σ
    ax2.errorbar(d.cld_med, np.zeros(len(d)), yerr=sd, fmt="none", ecolor=GREY,
                 elinewidth=6, alpha=0.3, capsize=0, zorder=0)
    ax2.axhline(0, color="k", lw=0.7)
    ax2.errorbar(d.cld_med, d.resid_bc, yerr=d.oco_bc_sd, fmt="o", color=RED, mfc="none",
                 ms=7, elinewidth=0.8, capsize=2, label=XCO2_BC_LABEL)
    ax2.errorbar(d.cld_med, d.resid_corr, yerr=d.oco_corr_sd, fmt="o", color=BLUE,
                 ms=7, elinewidth=0.8, capsize=2, label=MODEL_LABEL)
    for r in d.itertuples():
        ax2.plot([r.cld_med, r.cld_med], [r.resid_bc, r.resid_corr], color="0.8", zorder=0)
    ax2.set_xlabel("median cloud distance of collocated OCO-2 (km)")
    ax2.set_ylabel("OCO-2 − ATom (ppm)"); ax2.set_title("Bias vs cloud distance")
    ax2.legend(handles=[Patch(facecolor=GREY, alpha=0.3, label="ATom pseudo-column ±1σ (per leg)"),
                        *ax2.get_legend_handles_labels()[0]], fontsize=8)
    panel_label(ax2, f"({chr(ord('a') + panel_offset + 1)})")

    nc = d[d.cld_med <= 10]
    if suptitle:
        fig.suptitle(
            f"ATom pseudo-column vs OCO-2 ocean-glint (AK-harmonized) — "
            f"{len(d)} legs, {len(nc)} near-cloud\n"
            f"near-cloud mean bias {nc.resid_bc.mean():+.2f}±{nc.resid_bc.std():.2f} → "
            f"{nc.resid_corr.mean():+.2f}±{nc.resid_corr.std():.2f} ppm  "
            f"(±1σ across legs;  pseudo-column σ {sd.min():.2f}–{sd.max():.2f} per leg)",
            fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png)
    if out_pdf:
        fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    # module globals used by load_oco/load_atom/process/make_summary_plot; declared
    # up front (before argparse reads them as defaults) so the overrides below are legal
    global CORR_COL, OUT_BASE, PLOT_BASE, MERGED_DIR, MODEL_LABEL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius-km", type=float, default=100)
    ap.add_argument("--window-min", type=float, default=120)
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--strat-prior-sd", type=float, default=1.0,
                    help="nominal stratospheric CO2 prior σ (ppm) for the pseudo-column "
                         "uncertainty's prior-fill term above the aircraft ceiling")
    ap.add_argument("--corr-col", default=CORR_COL,
                    help="corrected-XCO2 column in each plot_data.parquet "
                         "(deep_ensemble_corrected_xco2 | linreg_corrected_xco2 | xgb_corrected_xco2)")
    ap.add_argument("--out-base", default=OUT_BASE,
                    help="dir for the results CSV/plot AND (default) the per-case "
                         "combined_<date>_atom/plot_data.parquet inputs")
    ap.add_argument("--plot-base", default=None,
                    help="dir holding combined_<date>_atom/plot_data.parquet (default: --out-base)")
    ap.add_argument("--merged-dir", default=MERGED_DIR,
                    help="atom_merged/ profiles (model-independent; default reuses the DE tree)")
    ap.add_argument("--model-label", default=MODEL_LABEL, help="cosmetic label for plot legends")
    args = ap.parse_args()
    twin = args.window_min * 60

    # override module globals from CLI (declared global at top of main())
    CORR_COL = args.corr_col
    OUT_BASE = args.out_base
    PLOT_BASE = args.plot_base if args.plot_base is not None else args.out_base
    MERGED_DIR = args.merged_dir
    MODEL_LABEL = args.model_label

    all_rows = []
    for date in DATES:
        try:
            oco = load_oco(date); atom = load_atom(date)
        except FileNotFoundError as e:
            print(f"{date}: skip ({e})"); continue
        rows = process(date, oco, atom, args.radius_km, twin, args.min_n,
                       strat_prior_sd=args.strat_prior_sd)
        print(f"{date}: {len(rows)} collocated leg(s)")
        all_rows += rows

    if not all_rows:
        print("no collocated legs"); return
    df = pd.DataFrame(all_rows)
    os.makedirs(OUT_BASE, exist_ok=True)
    out = os.path.join(OUT_BASE, "atom_pseudo_column_results.csv")
    df.to_csv(out, index=False)

    show = ["date", "profile_id", "leg_dir", "n_oco", "n_near", "cld_med",
            "p_ceiling_hpa", "atom_xco2_ak", "atom_ak_sd", "oco_xco2_bc", "oco_bc_sd",
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

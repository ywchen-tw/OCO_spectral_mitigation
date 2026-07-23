#!/usr/bin/env python3
"""Summary plot across all ship comparison cases — mirrors atom_pseudo_column_summary.png.

Two panels with ±1σ (spread of the collocated OCO-2 soundings) error bars, matching
the ATom summary / tccon_comparison_report dumbbell convention:
  (A) per-case signed bias  xco2_bc → DE-corrected  (dumbbell), sorted by cloud distance
  (B) bias vs median cloud distance of the collocated footprints

Reference = ship EM27/SUN median XCO2 in the OCO-2 overpass window (±window).
Reads each case's plot_data.parquet from the OUT_BASE tree written by
curc_shell_blanca_ship_deepens.sh.

Usage: python plot_ship_summary.py [--out-base <dir>] [--radius-km 100] [--window-min 120]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_ship_comparison import load_ship, haversine_km, TABS  # noqa: E402

ROOT_WORKSPACE = os.path.abspath(os.path.join(HERE, ".."))
if ROOT_WORKSPACE not in sys.path:
    sys.path.insert(0, ROOT_WORKSPACE)
from plot_style import XCO2_BC_LABEL, apply_manuscript_style, panel_label  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_OUT = os.path.join(REPO, "results/model_comparison/deep_ensemble/"
                           "de_beta_nll_prof_reg_foldpca_o05l15_m5/ship")
CORR = "deep_ensemble_corrected_xco2"   # overridable via --corr-col (linreg/xgb baselines)
MODEL_LABEL = "DE-corrected"            # cosmetic (plot legends/titles)

# (date, ship, (lon_min,lon_max), (lat_min,lat_max)) — same 4 cases as the runner
CASES = [
    ("2019-06-09", "so268",  (-152.62, -152.15), (29.36, 30.94)),
    ("2019-06-14", "so268",  (-170.75, -170.54), (28.94, 29.66)),
    ("2019-06-22", "so268",  (152.36, 152.80),   (27.43, 28.96)),
    ("2021-03-15", "mr2101", (140.06, 140.18),   (26.24, 26.60)),
]


def case_stats(date, ship, lonr, latr, out_base, radius_km, twin_s):
    pq = os.path.join(out_base, f"combined_{date}_{ship}", "plot_data.parquet")
    if not os.path.exists(pq):
        return None
    oco = pd.read_parquet(pq)
    oco = oco[(oco.sfc_type == 0) & oco.lon.between(*lonr) & oco.lat.between(*latr)]
    sh = load_ship(ship, date)
    if oco.empty or sh.empty:
        return None
    olon, olat, ot = oco.lon.to_numpy(), oco.lat.to_numpy(), oco.time.to_numpy()
    slon, slat, st = sh.lon.to_numpy(), sh.lat.to_numpy(), sh.epoch.to_numpy()
    dmin = np.full(len(oco), np.inf)
    for i0 in range(0, len(oco), 200):
        sl = slice(i0, i0+200)
        d = haversine_km(olon[sl][:, None], olat[sl][:, None], slon[None, :], slat[None, :])
        tg = np.abs(ot[sl][:, None] - st[None, :])
        dmin[sl] = np.where(tg <= twin_s, d, np.inf).min(axis=1)
    near = oco[dmin <= radius_km]
    if near.empty:
        return None
    ow = (st >= ot.min() - twin_s) & (st <= ot.max() + twin_s)
    ship_ref = float(np.nanmedian(sh.xco2.to_numpy()[ow]))
    # ship reference uncertainty = reported measurement error ⊕ temporal variability in window
    ship_meas = float(np.nanmedian(sh.xco2_err.to_numpy()[ow]))  # instrument-reported σ
    ship_var = float(np.nanstd(sh.xco2.to_numpy()[ow]))          # ship XCO2 spread in window
    ship_err = float(np.hypot(ship_meas, ship_var))              # combined
    o_bc, o_cc = near.xco2_bc.to_numpy(), near[CORR].to_numpy()
    return dict(
        date=date, ship=ship, n=len(near),
        cld_med=float(np.nanmedian(near.cld_dist_km)),
        n_near=int((near.cld_dist_km <= 10).sum()),
        ship_xco2=ship_ref, ship_meas=ship_meas, ship_var=ship_var, ship_err=ship_err,
        oco_bc=float(np.nanmedian(o_bc)),   oco_bc_sd=float(np.nanstd(o_bc)),
        oco_corr=float(np.nanmedian(o_cc)), oco_corr_sd=float(np.nanstd(o_cc)),
        resid_bc=float(np.nanmedian(o_bc) - ship_ref),
        resid_corr=float(np.nanmedian(o_cc) - ship_ref),
    )


def make_summary_plot(df, out_png, panel_offset=0, suptitle=True,
                      out_pdf=None):
    d = df.sort_values("cld_med").reset_index(drop=True)
    y = np.arange(len(d))
    lbl = [f"{r.date[5:]} {r.ship}" for r in d.itertuples()]
    RED, BLUE, GREY = "#d62728", "#1f77b4", "0.82"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # (A) per-case signed bias bc→corr, xerr = 1σ of the collocated OCO-2 soundings.
    # Grey band = ship measured XCO2 uncertainty about the (zero) reference, per case.
    for k, (yi, err) in enumerate(zip(y, d.ship_err)):
        ax1.fill_betweenx([yi - 0.35, yi + 0.35], -err, err, color=GREY, zorder=0,
                          label=r"ship σ (meas$\oplus$var)" if k == 0 else None)
    for yi, rb, rc in zip(y, d.resid_bc, d.resid_corr):
        ax1.plot([rb, rc], [yi, yi], "-", color="0.75", zorder=1)
    ax1.errorbar(d.resid_bc, y, xerr=d.oco_bc_sd, fmt="o", ms=7, color=RED, ecolor=RED,
                 elinewidth=1, capsize=3, zorder=2,
                 label=f"{XCO2_BC_LABEL} − ship")
    ax1.errorbar(d.resid_corr, y, xerr=d.oco_corr_sd, fmt="o", ms=7, color=BLUE, ecolor=BLUE,
                 elinewidth=1, capsize=3, zorder=3, label=f"{MODEL_LABEL} − ship")
    ax1.axvline(0, color="k", lw=0.7)
    ax1.set_yticks(y); ax1.set_yticklabels(lbl, fontsize=9); ax1.invert_yaxis()
    ax1.set_xlabel("OCO-2 − ship EM27/SUN (ppm)   [±1σ of collocated soundings]")
    ax1.set_title(f"Per-case bias: {XCO2_BC_LABEL} → {MODEL_LABEL}", pad=10)
    ax1.legend(fontsize=8)
    panel_label(ax1, f"({chr(ord('a') + panel_offset)})")

    # (B) bias vs cloud distance, yerr = 1σ of the collocated OCO-2 soundings.
    # Grey band = ship measured XCO2 uncertainty about the (zero) reference, per case.
    ax2.axhline(0, color="k", lw=0.7)
    xw = max((d.cld_med.max() - d.cld_med.min()) * 0.02, 1.0)
    for k, r in enumerate(d.itertuples()):
        ax2.fill_between([r.cld_med - xw, r.cld_med + xw], -r.ship_err, r.ship_err,
                         color=GREY, zorder=0, label=r"ship σ (meas$\oplus$var)" if k == 0 else None)
    ax2.errorbar(d.cld_med, d.resid_bc, yerr=d.oco_bc_sd, fmt="o", color=RED, mfc="none",
                 ms=8, elinewidth=0.8, capsize=3, label=XCO2_BC_LABEL)
    ax2.errorbar(d.cld_med, d.resid_corr, yerr=d.oco_corr_sd, fmt="o", color=BLUE,
                 ms=8, elinewidth=0.8, capsize=3, label=MODEL_LABEL)
    for r in d.itertuples():
        ax2.plot([r.cld_med, r.cld_med], [r.resid_bc, r.resid_corr], color="0.8", zorder=0)
        ax2.annotate(r.date[5:], (r.cld_med, r.resid_corr), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    ax2.set_xlabel("median cloud distance of collocated OCO-2 (km)")
    ax2.set_ylabel("OCO-2 − ship (ppm)"); ax2.set_title("Bias vs cloud distance", pad=10)
    ax2.legend(fontsize=8)
    panel_label(ax2, f"({chr(ord('a') + panel_offset + 1)})")

    nc = d[d.cld_med <= 10]
    if suptitle:
        sub = f"{len(nc)} near-cloud: mean bias {nc.resid_bc.mean():+.2f}→{nc.resid_corr.mean():+.2f} ppm  " if len(nc) else ""
        fig.suptitle(
            f"OCO-2 ocean-glint vs shipborne EM27/SUN — {len(d)} cases\n"
            f"{sub}all: mean bias {d.resid_bc.mean():+.2f}±{d.resid_bc.std():.2f} → "
            f"{d.resid_corr.mean():+.2f}±{d.resid_corr.std():.2f} ppm   "
            f"(mean OCO σ {d.oco_bc_sd.mean():.2f}→{d.oco_corr_sd.mean():.2f}; "
            f"mean ship σ {d.ship_err.mean():.2f})",
            fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png)
    if out_pdf:
        fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    global CORR, MODEL_LABEL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-base", default=DEFAULT_OUT)
    ap.add_argument("--radius-km", type=float, default=100)
    ap.add_argument("--window-min", type=float, default=120)
    ap.add_argument("--corr-col", default=CORR,
                    help="corrected-XCO2 column in plot_data (deep_ensemble_/linreg_/xgb_corrected_xco2)")
    ap.add_argument("--model-label", default=MODEL_LABEL, help="cosmetic label for plot legends")
    args = ap.parse_args()
    apply_manuscript_style()   # Arial (AMT), Arial mathtext, thin axes, 300 dpi
    CORR = args.corr_col
    MODEL_LABEL = args.model_label
    twin = args.window_min * 60

    rows = []
    for date, ship, lonr, latr in CASES:
        r = case_stats(date, ship, lonr, latr, args.out_base, args.radius_km, twin)
        if r is None:
            print(f"{date} {ship}: skip (no plot_data / no collocation)"); continue
        rows.append(r)
        print(f"{date} {ship}: n={r['n']:4d} cld_med={r['cld_med']:5.1f}  "
              f"bias_bc {r['resid_bc']:+.2f}±{r['oco_bc_sd']:.2f} → "
              f"corr {r['resid_corr']:+.2f}±{r['oco_corr_sd']:.2f}  "
              f"ship σ {r['ship_err']:.2f} (meas {r['ship_meas']:.2f}⊕var {r['ship_var']:.2f})")
    if not rows:
        print("no cases with plot_data — run curc_shell_blanca_ship_deepens.sh first"); return
    df = pd.DataFrame(rows)
    csv = os.path.join(args.out_base, "ship_comparison_summary.csv")
    df.to_csv(csv, index=False); print(f"wrote {csv}")
    make_summary_plot(df, os.path.join(args.out_base, "ship_comparison_summary.png"))


if __name__ == "__main__":
    main()

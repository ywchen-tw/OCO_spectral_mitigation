#!/usr/bin/env python3
"""Per-date ATom↔OCO-2 comparison figure — the ATom analog of plot_ship_comparison.py.

Same 4-panel layout as the ship figure, adapted for an aircraft reference (the ATom
AK pseudo-column per profile leg, not a continuous XCO2 field):

  (1) map  : DeepEns-corrected XCO2 footprints + ATom track (MODIS Aqua bg);
             collocated legs marked by a diamond coloured by their AK pseudo-column
  (2) hist : OCO-2 original vs DeepEns-corrected (pooled collocated footprints), with
             each leg's AK pseudo-column as a vertical reference line; Δmedian in title
  (3) map  : original XCO2_bc (before correction), same colour scale (side-by-side)
  (4) prof : the aircraft CO2 profile(s) that built the pseudo-column (CO2 vs pressure)
             + the OCO-2 prior profile + column-value reference lines

Reuses the collocation/AK machinery in atom_pseudo_column.py and the GIBS Aqua fetch
in plot_corrected_xco2.py. Writes atom_comparison_<date>.png into each
combined_<date>_atom/ (like ship_comparison_<tag>_<date>.png).

Usage: python plot_atom_comparison.py [--date 2017-10-20] [--no-modis]
                                      [--radius-km 100] [--window-min 120]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from atom_pseudo_column import (load_atom, load_oco, pseudo_profile_on_grid,   # noqa: E402
                                PLOT_BASE, MERGED_DIR, DATES, CORR_COL)
from ak_harmonize import operator_from_dataframe, _haversine_km               # noqa: E402
from plot_corrected_xco2 import download_modis_rgb                            # noqa: E402

TILES_DIR = os.path.join(PLOT_BASE, "_modis_tiles")


def collocate_date(date, radius_km, twin_s, min_n=3):
    """Return (oco, atom, legs, pooled) — legs is a list of per-leg dicts with the
    collocated footprint subset, AK pseudo-column, operator and leg frame."""
    oco, atom = load_oco(date), load_atom(date)
    olon, olat, ot = oco.lon.values, oco.lat.values, oco.time.values
    legs, hit_any = [], np.zeros(len(oco), bool)
    for pid, leg in atom.groupby("profile_id"):
        alon, alat, at = leg.lon.values, leg.lat.values, leg.epoch.values
        hit = np.zeros(len(oco), bool)
        for i0 in range(0, len(oco), 400):
            sl = slice(i0, i0 + 400)
            d = _haversine_km(olon[sl][:, None], olat[sl][:, None], alon[None, :], alat[None, :])
            tg = np.abs(ot[sl][:, None] - at[None, :])
            hit[sl] = np.where(tg <= twin_s, d, np.inf).min(axis=1) <= radius_km
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
        c_ak = float(op["ca"] + np.nansum(op["h"] * op["a"] * (x_on_oco - op["xa"])))
        legs.append(dict(pid=int(pid), sub=sub, op=op, leg=leg, c_ak=c_ak,
                         lat=float(leg.lat.mean()), lon=float(leg.lon.mean())))
        hit_any |= hit
    return oco, atom, legs, oco[hit_any]


def bin_profile(leg, n=40):
    p = leg["p_hpa"].to_numpy(); c = leg["co2_ppm"].to_numpy()
    m = np.isfinite(p) & np.isfinite(c) & (p > 0)
    p, c = p[m], c[m]
    edges = np.linspace(p.min(), p.max(), n + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n - 1)
    pb = np.array([p[idx == b].mean() for b in range(n) if (idx == b).any()])
    cb = np.array([c[idx == b].mean() for b in range(n) if (idx == b).any()])
    o = np.argsort(pb)
    return pb[o], cb[o]


def make_fig(date, radius_km, twin_s, do_modis):
    oco, atom, legs, pooled = collocate_date(date, radius_km, twin_s)
    if not legs:
        print(f"{date}: no collocated legs"); return
    o_orig = pooled.xco2_bc.to_numpy()
    o_corr = pooled[CORR_COL].to_numpy()
    c_aks = np.array([L["c_ak"] for L in legs])
    ref = float(c_aks.mean())                       # date reference = mean leg pseudo-column
    sd_ref = float(c_aks.std())                     # spread across legs (0 if single leg)
    d_orig = float(np.nanmedian(o_orig) - ref)
    d_corr = float(np.nanmedian(o_corr) - ref)
    e_orig = float(np.hypot(np.nanstd(o_orig), sd_ref))
    e_corr = float(np.hypot(np.nanstd(o_corr), sd_ref))

    # map extent from pooled footprints + nearby ATom track
    pad = 0.5
    lon0, lon1 = pooled.lon.min() - pad, pooled.lon.max() + pad
    lat0, lat1 = pooled.lat.min() - pad, pooled.lat.max() + pad
    at_in = atom[(atom.lon.between(lon0, lon1)) & (atom.lat.between(lat0, lat1))]
    inx = oco[(oco.lon.between(lon0, lon1)) & (oco.lat.between(lat0, lat1))]
    # keep only footprints with BOTH bc and corrected finite so the corrected (panel 1)
    # and original-bc (panel 3) maps show the identical set of soundings
    inx = inx[np.isfinite(inx[CORR_COL]) & np.isfinite(inx.xco2_bc)]
    vmin = float(np.floor(np.nanpercentile(o_corr, 2) * 2) / 2)
    vmax = float(np.ceil(np.nanpercentile(o_corr, 98) * 2) / 2)

    bg = ext = None
    if do_modis:
        try:
            os.makedirs(TILES_DIR, exist_ok=True)
            ext = [lon0, lon1, lat0, lat1]
            cached = download_modis_rgb(pd.Timestamp(date), ext, which="aqua",
                                        fdir=TILES_DIR, run=False)
            png = cached if os.path.exists(cached) else \
                download_modis_rgb(pd.Timestamp(date), ext, which="aqua", fdir=TILES_DIR)
            bg = plt.imread(png)
        except Exception as e:
            print(f"{date}: MODIS unavailable ({e})")

    def _bg(a):
        if bg is not None:
            a.imshow(bg, extent=ext, aspect="auto", origin="upper", zorder=0)

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"OCO-2 (DeepEns-corrected) vs ATom aircraft pseudo-column — {date}\n"
                 f"{len(pooled)} ocean-glint footprints ≤{radius_km:.0f} km / "
                 f"±{twin_s/60:.0f} min, {len(legs)} profile leg(s)",
                 fontsize=15, weight="bold")

    # (1) corrected map + track + leg pseudo-column diamonds
    a = ax[0, 0]; _bg(a)
    a.scatter(inx.lon, inx.lat, c=inx[CORR_COL], s=10, cmap="turbo", vmin=vmin, vmax=vmax,
              alpha=0.5, zorder=2, edgecolors="none")
    sc = a.scatter(pooled.lon, pooled.lat, c=pooled[CORR_COL], s=16, cmap="turbo",
                   vmin=vmin, vmax=vmax, zorder=3, edgecolors="none", label="OCO-2 corrected")
    a.plot(at_in.lon, at_in.lat, "-", color="magenta", lw=1.2, alpha=0.9, zorder=4, label="ATom track")
    a.scatter([L["lon"] for L in legs], [L["lat"] for L in legs], c=c_aks, cmap="turbo",
              vmin=vmin, vmax=vmax, s=180, marker="D", edgecolor="k", linewidth=1.2,
              zorder=6, label="ATom pseudo-column")
    a.set(xlim=(lon0, lon1), ylim=(lat0, lat1), xlabel="Lon (°E)", ylabel="Lat (°N)",
          title="DeepEns-corrected XCO₂ + ATom track" + (" (MODIS Aqua)" if bg is not None else ""))
    fig.colorbar(sc, ax=a, label="XCO₂ (ppm)"); a.legend(loc="upper left", fontsize=8)

    # (2) histogram OCO original vs corrected + pseudo-column reference lines
    a = ax[0, 1]
    lo = min(np.nanmin(o_orig), np.nanmin(o_corr), c_aks.min())
    hi = max(np.nanmax(o_orig), np.nanmax(o_corr), c_aks.max())
    bins = np.linspace(lo, hi, 50)
    a.hist(o_orig, bins, density=True, alpha=0.45, color="tab:blue",
           label=f"OCO-2 original  μ={np.nanmean(o_orig):.2f} σ={np.nanstd(o_orig):.2f}")
    a.hist(o_corr, bins, density=True, alpha=0.45, color="tab:green",
           label=f"OCO-2 DeepEns   μ={np.nanmean(o_corr):.2f} σ={np.nanstd(o_corr):.2f}")
    for i, L in enumerate(legs):
        a.axvline(L["c_ak"], color="tab:red", lw=2,
                  label="ATom pseudo-column" if i == 0 else None)
    a.set(xlabel="XCO₂ (ppm)", ylabel="Density",
          title=f"Δmedian(OCO−ATom):  original {d_orig:+.2f}±{e_orig:.2f}  →  "
                f"corrected {d_corr:+.2f}±{e_corr:.2f} ppm")
    a.legend(fontsize=9)

    # (3) original XCO2_bc map, same scale + same footprint set as panel (1)
    a = ax[1, 0]; _bg(a)
    a.scatter(inx.lon, inx.lat, c=inx.xco2_bc, s=10, cmap="turbo", vmin=vmin, vmax=vmax,
              alpha=0.5, zorder=2, edgecolors="none")
    sc = a.scatter(pooled.lon, pooled.lat, c=pooled.xco2_bc, s=16, cmap="turbo",
                   vmin=vmin, vmax=vmax, zorder=3, edgecolors="none")
    a.plot(at_in.lon, at_in.lat, "-", color="magenta", lw=1.2, alpha=0.9, zorder=4)
    a.scatter([L["lon"] for L in legs], [L["lat"] for L in legs], c=c_aks, cmap="turbo",
              vmin=vmin, vmax=vmax, s=180, marker="D", edgecolor="k", linewidth=1.2, zorder=6)
    a.set(xlim=(lon0, lon1), ylim=(lat0, lat1), xlabel="Lon (°E)", ylabel="Lat (°N)",
          title="Original XCO₂_bc (before correction)" + (" (MODIS Aqua)" if bg is not None else ""))
    fig.colorbar(sc, ax=a, label="XCO₂ (ppm)")

    # (4) aircraft CO2 profile(s) + OCO prior + column-value reference lines
    a = ax[1, 1]
    cmap = plt.get_cmap("cool")
    for i, L in enumerate(legs):
        pb, cb = bin_profile(L["leg"])
        col = cmap(i / max(1, len(legs) - 1))
        a.plot(cb, pb, "-", color=col, lw=1.5, label=f"ATom leg {L['pid']} (in-situ CO₂)")
        a.plot(L["op"]["xa"], L["op"]["pl"], ":", color=col, lw=1.2,
               label="OCO-2 prior" if i == 0 else None)
    a.axvline(ref, color="tab:red", lw=2, ls="--", label="ATom pseudo-column XCO₂")
    a.axvline(np.nanmedian(o_orig), color="tab:blue", lw=1.5, label="OCO orig median")
    a.axvline(np.nanmedian(o_corr), color="tab:green", lw=1.5, label="OCO corrected median")
    a.invert_yaxis()
    a.set(xlabel="CO₂ / XCO₂ (ppm)", ylabel="Pressure (hPa)",
          title="Aircraft profile → pseudo-column (vs OCO-2 prior)")
    a.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(PLOT_BASE, f"combined_{date}_atom", f"atom_comparison_{date}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"{date}: n_fp={len(pooled)} legs={len(legs)}  "
          f"Δmedian orig {d_orig:+.2f} → corr {d_corr:+.2f} ppm  → {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="single date YYYY-MM-DD (default: all 5)")
    ap.add_argument("--radius-km", type=float, default=100)
    ap.add_argument("--window-min", type=float, default=120)
    ap.add_argument("--no-modis", action="store_true")
    args = ap.parse_args()
    for d in ([args.date] if args.date else DATES):
        make_fig(d, args.radius_km, args.window_min * 60, not args.no_modis)


if __name__ == "__main__":
    main()

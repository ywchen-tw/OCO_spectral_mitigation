"""T4: f, gamma, f*gamma and k1 across the 1e9-photon slab (3D and IPA).

Reads results/rt_slab_sim/slab_fit_am.h5 the way workspace/rt_slab_sim/make_fig_g1.py
does (gamma = exp(intercept); far = x < 5 km; shadow band 15 <= x <= 22 km;
cloud at slab_config.CLOUD_X_KM = 9.5-14.5 km, sun on the -x side, so the
sunlit / brightened ground is 5 <= x < 9.5 km).

Outputs: out/t4_slab_profiles.png, out/t4_slab_values.csv, out/t4_slab_columns.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace" / "rt_slab_sim"))
sys.path.insert(0, str(REPO / "workspace"))
sys.path.insert(0, str(REPO / "src"))
import slab_config as cfg  # noqa: E402
from spectral.cumulant_fit import _AM_F_MAX  # noqa: E402
from plot_style import MEAN_L_LABEL, apply_manuscript_style, panel_label  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
FIT = Path(cfg.OUT_DIR) / "slab_fit_am.h5"
CLIP_TOL = 1e-4      # f within this of the bound counts as clipped


def main() -> None:
    with h5py.File(FIT, "r") as f:
        x = f["x_km"][...]
        fit = {(s, v): {n: f[f"{s}/{v}"][n][...] for n in ("k1", "f", "intercept", "ok")}
               for s in cfg.SURFACE_ALBEDOS for v in cfg.SOLVERS}
    c0, c1 = cfg.CLOUD_X_KM
    regions = {
        "far (x<5, median)": x < 5.0,
        "sunlit side 5-9.5 (mean)": (x >= 5.0) & (x < c0),
        "sunlit edge (last clear col)": x == x[x < c0][-1],
        "in cloud 9.5-14.5 (mean)": (x >= c0) & (x < c1),
        "shadow 15-22 (mean)": (x >= 15.0) & (x <= 22.0),
    }
    rows, cols = [], []
    for (s, v), r in fit.items():
        g = np.exp(r["intercept"])
        fg = r["f"] * g
        prof = {"k1": r["k1"], "f": r["f"], "gamma": g, "fg": fg}
        clipped = r["f"] >= _AM_F_MAX - CLIP_TOL
        for i in range(len(x)):
            cols.append({"surface": s, "solver": v, "x_km": x[i], "k1": r["k1"][i],
                         "f": r["f"][i], "gamma": g[i], "fg": fg[i],
                         "f_clipped": bool(clipped[i])})
        for name, m in regions.items():
            agg = np.nanmedian if name.startswith("far") else np.nanmean
            rec = {"surface": s, "solver": v, "region": name, "n_col": int(m.sum()),
                   "n_clipped": int((m & clipped).sum())}
            rec.update({k: float(agg(val[m])) for k, val in prof.items()})
            rows.append(rec)
        # shadow extremes: the column of minimum gamma, and max f*gamma
        sh = regions["shadow 15-22 (mean)"]
        i_g = np.flatnonzero(sh)[np.nanargmin(g[sh])]
        i_fg = np.flatnonzero(sh)[np.nanargmax(fg[sh])]
        sl = regions["sunlit side 5-9.5 (mean)"]
        i_gs = np.flatnonzero(sl)[np.nanargmax(g[sl])]
        for lbl, i in (("shadow min-gamma col", i_g), ("shadow max-fg col", i_fg),
                       ("sunlit max-gamma col", i_gs)):
            rows.append({"surface": s, "solver": v, "region": f"{lbl} x={x[i]:.2f}",
                         "n_col": 1, "n_clipped": int(clipped[i]),
                         "k1": r["k1"][i], "f": r["f"][i], "gamma": g[i], "fg": fg[i]})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "t4_slab_values.csv", index=False)
    pd.DataFrame(cols).to_csv(OUT / "t4_slab_columns.csv", index=False)
    with pd.option_context("display.width", 200):
        print(tab.round(5).to_string(index=False))

    apply_manuscript_style(base_size=9.0)
    names = [("k1", f"fitted {MEAN_L_LABEL}"), ("f", "floor $f$"),
             ("gamma", r"$\gamma$"), ("fg", r"$f\gamma$")]
    surfaces = list(cfg.SURFACE_ALBEDOS)
    fig, axes = plt.subplots(len(names), len(surfaces), figsize=(6.2, 6.4),
                             sharex=True, sharey="row", constrained_layout=True)
    for j, s in enumerate(surfaces):
        for i, (k, lab) in enumerate(names):
            ax = axes[i, j]
            for v, col in (("3d", "#c1272d"), ("ipa", "#0000a7")):
                r = fit[(s, v)]
                g = np.exp(r["intercept"])
                y = {"k1": r["k1"], "f": r["f"], "gamma": g, "fg": r["f"] * g}[k]
                ax.plot(x, y, color=col, lw=1.2, label=v.upper())
                if k == "f":
                    clip = r["f"] >= _AM_F_MAX - CLIP_TOL
                    ax.plot(x[clip], y[clip], "o", ms=3, mfc="none", color=col)
            ax.axvspan(c0, c1, color="0.85", lw=0)
            ax.axvspan(15.0, 22.0, color="#1b9e77", alpha=0.10, lw=0)
            if k == "f":
                ax.axhline(_AM_F_MAX, color="0.4", lw=0.7, ls="--")
            if k == "fg":
                # linear, clear-sky range; dark in-cloud f*gamma ~ 0 (f at its lower bound)
                ax.set_ylim(0.0, 0.009)
                ax.axhline(0.0034, color="0.4", lw=0.7, ls=":")
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f"{s} surface (A = {cfg.SURFACE_ALBEDOS[s]:.2f})")
            panel_label(ax, f"({chr(97 + i * len(surfaces) + j)})", size=9)
    for ax in axes[-1]:
        ax.set_xlabel("x (km)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "t4_slab_profiles.png", dpi=200)
    print("wrote", OUT / "t4_slab_profiles.png")


if __name__ == "__main__":
    main()

"""
Plot the slab atmosphere profiles (fine 72-layer GEOS vs coarse 21-layer
slab grid) -- verification / appendix-G companion figure.

Layout follows er3t.pre.atm.util.zpt_plot / create_modis_dropsonde_atm:
  (a) SkewT: temperature + dew point vs pressure (metpy),
  (b) H2O volume mixing ratio + CO2 vs pressure (log-p axis),
  (c) number densities vs altitude with the slab level grid and cloud layer.
Fine-grid profiles are lines; the coarse slab layers are step/markers so the
column-conserving aggregation is visible.

Run:  python workspace/rt_slab_sim/plot_atm_profiles.py
"""
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

FIG_DIR = os.path.join(cfg.OUT_DIR, "figs")

EPSILON = 0.622


def dew_point_k(p_hpa, h2o_vmr):
    """Dew point (K) from vapor pressure via inverted Magnus (WMO constants)."""
    e_hpa = np.clip(p_hpa * h2o_vmr / (1.0 + h2o_vmr), 1e-10, None)
    ln_ratio = np.log(e_hpa / 6.112)
    return 273.15 + 243.5 * ln_ratio / (17.67 - ln_ratio)


def main():
    with h5py.File(cfg.ATM_FILE, "r") as f:
        lay = {k: f["lay"][k][...] for k in f["lay"]}
        lev = {k: f["lev"][k][...] for k in f["lev"]}
        fine = {k: f["fine"][k][...] for k in f["fine"]}
        meta = dict(f["meta"].attrs)

    p_f = fine["p_lay_pa"] / 100.0                       # hPa
    t_f = fine["t_lay_k"]
    h_f = (fine["h_edge_m"][:-1] + fine["h_edge_m"][1:]) / 2.0 / 1e3   # km
    vmr_f = fine["h2o_vmr"]
    co2_f = fine["d_co2"] / fine["d_dry"] * 1e6          # ppm

    p_c = lay["pressure_hpa"]
    t_c = lay["temperature_k"]
    vmr_c = lay["h2o_vmr"]
    co2_c = lay["d_co2_cm3"] / lay["d_dry_cm3"] * 1e6
    alt_c = lay["altitude_km"]

    pmax = float(np.ceil(p_f.max() / 100.0) * 100.0)
    pmin = 100.0

    from metpy.plots import SkewT
    from metpy.units import units

    fig = plt.figure(figsize=(16, 6.5))

    # (a) SkewT (aspect='auto' so the panel fills the subplot and matches
    # the height of panels b/c) -------------------------------------------
    skew = SkewT(fig, subplot=(1, 3, 1), rotation=45, aspect="auto")
    skew.plot(p_f * units.hPa, (t_f * units.kelvin).to(units.degC),
              "r", lw=1.5, label="T (fine)")
    skew.plot(p_f * units.hPa,
              (dew_point_k(p_f, vmr_f) * units.kelvin).to(units.degC),
              "b", lw=1.5, label="Td (fine)")
    skew.plot(p_c * units.hPa, (t_c * units.kelvin).to(units.degC),
              "ro", ms=5, mfc="none", label="T (slab layers)")
    skew.ax.set_ylim(pmax, pmin)
    skew.ax.set_xlim(-80, 40)
    skew.plot_dry_adiabats(alpha=0.25)
    skew.plot_moist_adiabats(alpha=0.25)
    skew.ax.legend(loc="lower left", fontsize=9)
    skew.ax.set_title("(a) temperature / dew point")
    skew.ax.set_xlabel("Temperature (degC)")
    skew.ax.set_ylabel("Pressure (hPa)")

    # (b) H2O VMR + CO2 vs log-p ------------------------------------------
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(vmr_f, p_f, "b-", lw=1.5, label="H$_2$O (fine)")
    ax.plot(vmr_c, p_c, "bo", ms=5, mfc="none", label="H$_2$O (slab layers)")
    ax.set_xscale("log")
    ax.set_ylim(pmax, pmin)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.set_ylabel("Pressure (hPa)")
    ax.set_xlabel("H$_2$O volume mixing ratio", color="b")
    ax2 = ax.twiny()
    ax2.plot(co2_f, p_f, "g-", lw=1.5, label="CO$_2$ (fine)")
    ax2.plot(co2_c, p_c, "gs", ms=5, mfc="none", label="CO$_2$ (slab layers)")
    ax2.set_xlabel("CO$_2$ prior (ppm)", color="g")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower left",
              fontsize=9)
    ax.set_title("(b) H$_2$O and CO$_2$")

    # (c) number densities vs altitude + slab grid -------------------------
    ax = fig.add_subplot(1, 3, 3)
    for name, col, lab in (("d_air", "k", "air"), ("d_o2", "tab:orange", "O$_2$"),
                           ("d_h2o", "b", "H$_2$O")):
        ax.plot(fine[name], h_f, "-", color=col, lw=1.5, label=f"{lab} (fine)")
        ckey = {"d_air": "d_air_cm3", "d_o2": "d_o2_cm3", "d_h2o": "d_h2o_cm3"}[name]
        ax.plot(lay[ckey], alt_c, "o", color=col, ms=5, mfc="none")
    for z in lev["altitude_km"]:
        ax.axhline(z, color="0.85", lw=0.5, zorder=0)
    ax.axhspan(cfg.CLOUD_BASE_KM, cfg.CLOUD_TOP_KM, color="tab:cyan",
               alpha=0.25, label="cloud layer")
    ax.set_xscale("log")
    ax.set_xlim(1e10, 1e20)
    ax.set_ylim(0, cfg.LEVELS_KM[-1])
    ax.set_xlabel("Number density (cm$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("(c) densities + slab grid (grey lines = levels)")

    fig.suptitle(
        f"Slab atmosphere -- sounding {meta['sounding_id']}  "
        f"({meta['lat']:.1f}N, {meta['lon']:.1f}E)  SZA {meta['sza']:.1f}, "
        f"ocean glint; 72-layer GEOS -> {alt_c.size}-layer column-conserving slab grid",
        y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "slab_atm_profiles.png")
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

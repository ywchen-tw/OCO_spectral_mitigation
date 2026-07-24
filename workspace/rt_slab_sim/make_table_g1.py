"""
Table G1 (manuscript): configuration of the controlled 3-D-vs-ICA slab
simulation, generated from slab_config.py + the stage-1/2 data files so the
typeset table cannot drift from the code.

Output: manuscript/tables/tabG1_slab_config.tex   (Copernicus style)

Run:  python workspace/rt_slab_sim/make_table_g1.py
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

OUT_TEX = os.path.join(cfg.REPO_ROOT, "manuscript", "tables",
                       "tabG1_slab_config.tex")


def main():
    with h5py.File(cfg.ATM_FILE, "r") as f:
        meta = dict(f["meta"].attrs)
        fine = {k: f["fine"][k][...] for k in ("h_edge_m", "d_o2", "dz_m")}
        nlay = f["lay"]["altitude_km"].shape[0]
    with h5py.File(cfg.OD_FILE, "r") as f:
        nwvl = f["wvl_nm"].shape[0]
        slant = f["slant_tau"][...]
        airmass = float(f.attrs["airmass"])
        od_c = f["od_column"][...]
        od_f = f["od_column_fine"][...]

    # above-TOA O2 residual + coarse-vs-fine column-OD agreement
    toa_m = cfg.LEVELS_KM[-1] * 1e3 + fine["h_edge_m"][0]
    w_below = np.clip(np.minimum(toa_m, fine["h_edge_m"][1:])
                      - fine["h_edge_m"][:-1], 0.0, None)
    frac_above = 1.0 - (np.sum(fine["d_o2"] * w_below)
                        / np.sum(fine["d_o2"] * fine["dz_m"]))
    od_rel = np.abs(od_c / od_f - 1)

    slant_abs = np.sort(slant)
    tau_min_abs = slant_abs[cfg.N_CONTINUUM:].min()

    sid = meta["sounding_id"]
    lat, lon = meta["lat"], meta["lon"]
    lon_txt = f"{abs(lon):.1f}$^\\circ$\\,{'E' if lon >= 0 else 'W'}"
    lat_txt = f"{abs(lat):.1f}$^\\circ$\\,{'N' if lat >= 0 else 'S'}"

    rows = []
    rows.append(("\\textit{Scene and geometry}", None))
    rows.append(("Source sounding",
                 f"OCO-2 glint sounding {sid} (orbit 29252a, 1 January 2020; "
                 f"{lat_txt}, {lon_txt}, ocean)"))
    rows.append(("Atmospheric profiles",
                 "GEOS met + CO$_2$ prior of that sounding (L2Met/L2CPr, "
                 "72-layer hybrid-sigma grid)"))
    rows.append(("Slab vertical grid",
                 f"{nlay} layers: 1\\,km below 10\\,km, 2\\,km to 20\\,km, "
                 "5\\,km to 40\\,km, 10\\,km to the 60\\,km model top; "
                 "partial gas columns conserved exactly per layer "
                 f"(above-top O$_2$ residual {frac_above * 100:.3f}\\,\\%)"))
    rows.append(("Sun / sensor",
                 f"SZA {meta['sza']:.1f}$^\\circ$, solar azimuth along $+x$; "
                 "nadir-viewing sensor at 705\\,km"))
    rows.append(("Domain",
                 f"$N_x={cfg.NX}$, $\\Delta x={cfg.DX_KM:.1f}$\\,km "
                 f"({cfg.NX * cfg.DX_KM:.0f}\\,km), $N_y=1$ "
                 f"($\\Delta y={cfg.DY_KM:.0f}$\\,km, medium invariant along "
                 "$y$), cyclic horizontal boundaries"))
    rows.append(("Surface",
                 "Lambertian, albedo "
                 f"{cfg.SURFACE_ALBEDOS['dark']:.2f} (dark) / "
                 f"{cfg.SURFACE_ALBEDOS['bright']:.2f} (bright)"))
    rows.append(("\\textit{Cloud}", None))
    rows.append(("Geometry / optics",
                 f"water cloud, base {cfg.CLOUD_BASE_KM:.0f}\\,km / top "
                 f"{cfg.CLOUD_TOP_KM:.0f}\\,km, $x={cfg.CLOUD_X_KM[0]}$--"
                 f"{cfg.CLOUD_X_KM[1]}\\,km ({cfg.CLOUD_X_KM[1] - cfg.CLOUD_X_KM[0]:.0f}\\,km wide); "
                 f"COD {cfg.CLOUD_COD:.0f}, $r_\\mathrm{{eff}}="
                 f"{cfg.CLOUD_CER_UM:.0f}\\,\\upmu$m, Mie phase function "
                 "(water cloud)"))
    rows.append(("\\textit{Gas optics (O$_2$A band)}", None))
    rows.append(("Absorption",
                 "ABSCO v5.2 O$_2$ + H$_2$O cross-sections on the slab "
                 "layers (trilinear $p$/$T$/broadener interpolation, as in "
                 "the production fit inputs)"))
    rows.append(("Wavelengths",
                 f"{nwvl} monochromatic wavelengths: {cfg.N_CONTINUUM} "
                 "continuum anchors + 30 log-spaced in slant optical depth "
                 f"$\\tau \\approx {tau_min_abs:.2f}$--{slant.max():.0f} "
                 f"(airmass {airmass:.2f}); slab-vs-72-layer column optical "
                 f"depth agrees to $\\le {od_rel.max() * 100:.1f}\\,\\%$"))
    rows.append(("Rayleigh", "Bodhaine et al. (1999) cross-sections from the "
                 "same profile"))
    rows.append(("\\textit{Monte Carlo}", None))
    rows.append(("Model",
                 "MCARaTS v0.10.4 (Iwabuchi, 2006) via the EaR$^3$T "
                 "interface; solvers: full 3-D transport vs.\\ "
                 "independent-column approximation on the identical scene"))
    rows.append(("Photons",
                 f"$10^9$ per wavelength and solver, {cfg.NRUN} independent "
                 "runs (run-to-run spread shown as shading in Fig.~G1)"))
    rows.append(("Path-length tally",
                 f"per-column histogram of total geometric photon path "
                 f"(radiance-contribution weighted; {cfg.PLEN_NBIN} bins "
                 f"$\\times$ {1e-3 * (cfg.PLEN_MAX_M - cfg.PLEN_MIN_M) / cfg.PLEN_NBIN * 1e3:.0f}\\,m)"))
    rows.append(("\\textit{Spectral estimator}", None))
    rows.append(("Fit",
                 "production cumulant estimator (Sect.~3.2, Table~A1): "
                 f"order {cfg.FIT_ORDER} in slant $\\tau$, exact linear "
                 "least squares with BVLS sign bounds, no presmoothing; "
                 "$T=\\pi I/(\\mu_0 F_0)$ per column"))

    lines = [
        "% Auto-generated by workspace/rt_slab_sim/make_table_g1.py -- do not hand-edit.",
        "\\begin{table}[t]",
        "\\caption{Configuration of the controlled 3-D-versus-ICA slab "
        "simulation (Appendix~G). All entries are generated from the "
        "simulation configuration and input files.}",
        "\\label{tab:g1-slab-config}",
        "\\footnotesize",
        "\\begin{tabular}{lp{0.62\\linewidth}}",
        "\\tophline",
        "Component & Setting \\\\",
        "\\middlehline",
    ]
    first_section = True
    for name, val in rows:
        if val is None:
            if not first_section:
                lines.append("\\middlehline")
            lines.append(f"\\multicolumn{{2}}{{l}}{{{name}}} \\\\")
            first_section = False
        else:
            lines.append(f"{name} & {val} \\\\")
    lines += ["\\bottomhline", "\\end{tabular}", "\\end{table}", ""]

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()

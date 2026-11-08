"""
Table G1 (manuscript): configuration of the controlled 3D-vs-IPA slab
simulation, generated from slab_config.py + the stage-1/2 data files so the
typeset table cannot drift from the code.  The Rayleigh-off row is read from
slab_rad_norayleigh.h5 (run_slab.py --variant norayleigh).

Output: manuscript/AMT_draft/tables/tabG1_slab_config.tex   (Copernicus style,
        AMT-era cross-references).  The JQSRT copy is produced by
        manuscript/JQSRT_draft/scripts/make_table_g1.py, which imports this
        module, runs main() into a temporary file and retargets the refs.

Run:  python workspace/rt_slab_sim/make_table_g1.py [--out PATH]
      (--out writes elsewhere, e.g. for a verification diff; the default
       stays manuscript/AMT_draft/tables/tabG1_slab_config.tex)
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

sys.path.insert(0, os.path.join(cfg.REPO_ROOT, "src"))
from spectral.cumulant_fit import _AM_F_MAX  # noqa: E402

OUT_TEX = os.path.join(cfg.REPO_ROOT, "manuscript", "AMT_draft", "tables",
                       "tabG1_slab_config.tex")


def _k1_run_spread_ceiling():
    """Upper bound, one significant figure rounded up, of the run-to-run std
    (ddof 1, fit_and_plot.py) of the fitted k1 = <l'> over every column,
    both surfaces and both solvers of the am slab refit."""
    with h5py.File(os.path.join(cfg.OUT_DIR, "slab_fit_am.h5"), "r") as f:
        top = max(float(np.nanmax(f[f"{s}/{v}/k1_std"][...]))
                  for s in cfg.SURFACE_ALBEDOS for v in cfg.SOLVERS)
    step = 10.0 ** np.floor(np.log10(top))
    return float(np.ceil(top / step) * step)


def _norayleigh_row():
    """Setting text of the Rayleigh-off sensitivity run, read from its
    radiance file slab_rad_norayleigh.h5 (run_slab.py --variant norayleigh):
    photon count and run count from the sweep attributes, solvers and
    surfaces from the groups that hold every wavelength."""
    path = cfg.variant_path(cfg.RAD_FILE, "norayleigh")
    with h5py.File(path, "r") as f:
        if int(f.attrs["rayleigh"]) != 0:
            raise ValueError(f"{path}: rayleigh attribute is not 0")
        photons, nrun = float(f.attrs["photons"]), int(f.attrs["nrun"])
        nwvl = f["wvl_nm"].shape[0]
        done = [(s, v) for s in cfg.SURFACE_ALBEDOS for v in cfg.SOLVERS
                if f"{s}/{v}" in f and f[f"{s}/{v}/iw"].shape[0] == nwvl]
    solvers = sorted({v for _, v in done})
    surfaces = {s for s, _ in done}
    if solvers != ["3d"] or surfaces != set(cfg.SURFACE_ALBEDOS):
        raise ValueError(f"{path}: expected the 3D solver over both surfaces, "
                         f"found {done}")
    expo = np.log10(photons)
    if expo != round(expo):
        raise ValueError(f"{path}: photon count {photons:g} is not a power of 10")
    return ("Rayleigh extinction set to zero, gas absorption and cloud "
            "unchanged, 3D solver over both surfaces, "
            f"$10^{int(round(expo))}$ photons per wavelength, {nrun} "
            "independent runs")


def main(out_tex=None):
    # module-level OUT_TEX is read at call time so importers (the JQSRT
    # wrapper) can redirect the output by patching it
    out_tex = out_tex or OUT_TEX
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
                 "GEOS meteorology and \\ce{CO2} prior of that sounding "
                 "(OCO-2 Met and CO2Prior files, 72-layer hybrid-sigma grid)"))
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
                 f"water cloud, base {cfg.CLOUD_BASE_EFF_KM:.0f}\\,km / top "
                 f"{cfg.CLOUD_TOP_EFF_KM:.0f}\\,km, $x = {cfg.CLOUD_X_KM[0]}$--"
                 f"{cfg.CLOUD_X_KM[1]}\\,km ({cfg.CLOUD_X_KM[1] - cfg.CLOUD_X_KM[0]:.0f}\\,km wide); "
                 f"COT {cfg.CLOUD_COD:.0f}, $r_\\mathrm{{eff}} = "
                 f"{cfg.CLOUD_CER_UM:.0f}\\,\\mu\\mathrm{{m}}$, "
                 "Henyey--Greenstein phase function with the Mie "
                 "asymmetry parameter ($g = 0.86$)"))
    rows.append(("\\textit{Gas optics (\\ce{O2}~A-band)}", None))
    rows.append(("Absorption",
                 "ABSCO v5.2 O$_2$ + H$_2$O cross-sections on the slab "
                 "layers (trilinear $p$/$T$/broadener interpolation, as in "
                 "the OCO-2 fit inputs)"))
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
                 "MCARaTS v0.10.4 \\citep{iwabuchi2006efficient} via the EaR$^3$T "
                 "interface; solvers: full 3D transport vs.\\ "
                 "independent-pixel approximation (IPA) on the identical scene"))
    # 2026-11-03: the k1 run-to-run std (shaded in the figure) is narrower
    # than the line, so the row states its size instead of pointing at shading
    rows.append(("Photons",
                 f"$10^9$ per wavelength and solver, {cfg.NRUN} independent "
                 "runs (run-to-run spread of the fitted \\meanl below "
                 f"{_k1_run_spread_ceiling():g})"))
    rows.append(("Path-length record",
                 f"per-column histogram of total geometric photon path "
                 f"(radiance-contribution weighted; {cfg.PLEN_NBIN} bins "
                 f"$\\times$ {1e-3 * (cfg.PLEN_MAX_M - cfg.PLEN_MIN_M) / cfg.PLEN_NBIN * 1e3:.0f}\\,m)"))
    # 2026-11-05: Rayleigh-off sensitivity run (slab_config.VARIANTS)
    rows.append(("Rayleigh-off run", _norayleigh_row()))
    rows.append(("\\textit{Path-length fit}", None))
    # 2026-11-02: the O2A agreed model (fit_and_plot.py --estimator am, fit
    # schema doppler_am_t2_v2) replaces the retired order-7 cumulant fit
    rows.append(("Fit",
                 "\\ce{O2}~A-band path-length fit of Sect.~\\ref{sec:methods_photon_path} "
                 "(Table~\\ref{tab:fit-config}): floor form with $k_2$ fixed at zero, "
                 "$\\ln\\gamma + a_1 x_\\lambda + "
                 "\\ln[(1-f)\\,\\exp(-k_1 \\tau) + f]$ in slant $\\tau$, bounded "
                 f"nonlinear least squares with $k_1 \\ge 0$ and $0 \\le f \\le {_AM_F_MAX:g}$; "
                 f"$x_\\lambda$, the wavelength scaled to $[-1, 1]$ over the {nwvl} "
                 "simulated wavelengths; "
                 "$\\mathcal T=\\pi I/(\\mu_0 F_0)$ per column"))

    lines = [
        "% Auto-generated by workspace/rt_slab_sim/make_table_g1.py -- do not hand-edit.",
        "\\begin{table}[t]",
        "\\caption{Configuration of the controlled 3D and IPA "
        "simulation (Appendix~\\ref{app:rt_demo}).}",
        "\\label{tab:g1-slab-config}",
        "\\footnotesize",
        "\\begin{tabular}{@{}p{0.22\\linewidth}p{0.72\\linewidth}@{}}",
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

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="write the .tex here instead of the default "
                         "manuscript/AMT_draft/tables/tabG1_slab_config.tex")
    main(ap.parse_args().out)

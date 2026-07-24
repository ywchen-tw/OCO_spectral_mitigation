"""
Stage 2: monochromatic O2A gas absorption optical depth on the slab grid.

Reuses the ABSCO v5.2 machinery from src/abs_util (same tables, same
trilinear P/T/broadener interpolation as production) to compute the
per-layer absorption OD for O2 + H2O at every ABSCO wavenumber in the O2A
window, on both the coarse 21-layer slab grid and the fine 72-layer grid
(verification), then selects the simulation wavelengths so their slant
column tau log-spans the production fit range plus continuum anchors.

Output: slab_od.h5
    wvl_nm, wvl_um, nu_cm1        (selected wavelengths)
    od_layer   (nwvl, nlay)       coarse-grid absorption OD per layer
    od_column  (nwvl,)            vertical column OD
    od_column_fine (nwvl,)        72-layer verification column
    slant_tau  (nwvl,)            od_column * airmass
    full_grid/{nu_cm1, od_column} full-window column OD (diagnostics)

Run:  python workspace/rt_slab_sim/gas_od.py
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

sys.path.insert(0, os.path.join(cfg.REPO_ROOT, "src"))
from abs_util.abs.calc_ext_absco import calc_ext_absco
from abs_util.abs.find_bound import find_boundary
from abs_util.abs.get_index import get_P_index, get_T_index
from abs_util.abs.rdabs_gas_absco import rdabs_species_absco
from abs_util.abs.rdabsco_gas_absco import rdabsco_species_absco

ABSCO_DIR = os.path.join(cfg.REPO_ROOT, "src", "abs_util", "abs")

TRILINEAR_MATRIX = np.array([[ 1,  0,  0,  0,  0,  0,  0,  0],
                             [-1,  0,  0,  0,  1,  0,  0,  0],
                             [-1,  0,  1,  0,  0,  0,  0,  0],
                             [-1,  1,  0,  0,  0,  0,  0,  0],
                             [ 1,  0, -1,  0, -1,  0,  1,  0],
                             [ 1, -1, -1,  1,  0,  0,  0,  0],
                             [ 1, -1,  0,  0, -1,  1,  0,  0],
                             [-1,  1,  1, -1,  1, -1, -1,  1]], dtype=np.float64)


def _read_absco(path, gas_key):
    with h5py.File(path, "r") as f:
        return {
            "Pressure": f["Pressure"][...],
            "Temperature": f["Temperature"][...],
            "Wavenumber": f["Wavenumber"][...],
            "Broadener_01_VMR": f["Broadener_01_VMR"][...],
            "Gas_Absorption": f[gas_key][...],
        }


def layer_od(absco_gas, absco_h2o, p_hpa, t_k, d_o2, d_h2o, h2o_vmr, dz_km):
    """Absorption OD per (wavenumber, layer) for O2 + H2O, O2A window."""
    nu_gas, _, tk_gas, broad_gas, hpa_gas, _ = rdabs_species_absco(
        absco_data=absco_gas, species="o2")
    nu_h2o_all, _, tkh2o, broadh2o, hpah2o, _ = rdabs_species_absco(
        absco_data=absco_h2o, species="h2o")

    wl1, wl2 = cfg.BAND_WVL_RANGE_UM
    nu2, nu1 = 1.0e4 / wl1, 1.0e4 / wl2
    inu1, inu2, nwav, nudat, wvldat = find_boundary(nu1, nu2, nu_gas)
    inu1h, inu2h, nwavh, nudath, _ = find_boundary(nu1, nu2, nu_h2o_all)
    iw_h2o_start, iw_h2o_end = np.searchsorted(nudat, nudath[[0, -1]], side="left")
    h2o_pad = (nwav != nwavh)

    nlay = dz_km.size
    od = np.zeros((nwav, nlay))
    P_inds = get_P_index(p_hpa, hpa_gas, trilinear=True)
    P_inds_h2o = get_P_index(p_hpa, hpah2o, trilinear=True)
    ext1 = np.zeros(nwav)
    for iz in range(nlay):
        T_ind = get_T_index(t_k[iz], tk_gas, P_inds[iz], trilinear=True)
        T_ind_h2o = get_T_index(t_k[iz], tkh2o, P_inds_h2o[iz], trilinear=True)
        absco = rdabsco_species_absco(
            absco_gas, None, tk_gas, broad_gas, hpa_gas,
            t_k[iz], p_hpa[iz], T_ind, P_inds[iz], h2o_vmr[iz],
            inu1, inu2, species="o2",
            trilinear_matrix=TRILINEAR_MATRIX, mode="trilinear")
        abscoh2o = rdabsco_species_absco(
            absco_h2o, None, tkh2o, broadh2o, hpah2o,
            t_k[iz], p_hpa[iz], T_ind_h2o, P_inds_h2o[iz], h2o_vmr[iz],
            inu1h, inu2h, species="h2o",
            trilinear_matrix=TRILINEAR_MATRIX, mode="trilinear")
        ext0 = calc_ext_absco(d_o2[iz], absco)          # 1/km
        ext1_ = calc_ext_absco(d_h2o[iz], abscoh2o)
        if h2o_pad:
            ext1[:] = 0.0
            ext1[iw_h2o_start:iw_h2o_end + 1] = ext1_
        else:
            ext1 = ext1_
        od[:, iz] = (ext0 + ext1) * dz_km[iz]
    return nudat, wvldat, od


def select_wavelengths(nu, slant_tau_full):
    """Continuum anchors + log-spaced slant-tau targets -> unique indices."""
    idx = set()
    order = np.argsort(slant_tau_full)
    idx.update(order[:cfg.N_CONTINUUM])                # lowest-tau anchors
    targets = np.geomspace(*cfg.SLANT_TAU_RANGE, cfg.N_TAU_SAMPLES)
    for t in targets:
        idx.add(int(np.argmin(np.abs(slant_tau_full - t))))
    return np.array(sorted(idx))


def main():
    with h5py.File(cfg.ATM_FILE, "r") as f:
        lay = {k: f["lay"][k][...] for k in f["lay"]}
        fine = {k: f["fine"][k][...] for k in f["fine"]}
        sza = f["meta"].attrs["sza"]

    absco_gas = _read_absco(os.path.join(ABSCO_DIR, "o2_v52.hdf"), "Gas_07_Absorption")
    absco_h2o = _read_absco(os.path.join(ABSCO_DIR, "h2o_v52.hdf"), "Gas_01_Absorption")

    print("Coarse-grid OD ...")
    nu, wvl_um, od = layer_od(
        absco_gas, absco_h2o,
        lay["pressure_hpa"], lay["temperature_k"],
        lay["d_o2_cm3"], lay["d_h2o_cm3"], lay["h2o_vmr"],
        lay["thickness_km"])
    od_col = od.sum(axis=1)

    # simulation airmass: footprint SZA, nadir sensor
    mu0 = np.cos(np.deg2rad(sza))
    muv = np.cos(np.deg2rad(cfg.SENSOR_ZENITH))
    airmass = 1.0 / mu0 + 1.0 / muv
    slant_tau = od_col * airmass

    sel = select_wavelengths(nu, slant_tau)
    print(f"Selected {sel.size} wavelengths; slant tau "
          f"{slant_tau[sel].min():.4f} - {slant_tau[sel].max():.3f}")

    print("Fine-grid verification OD (selected wavelengths only) ...")
    # fine grid: densities & thickness below slab TOA (partial top layer)
    toa_m = cfg.LEVELS_KM[-1] * 1e3 + fine["h_edge_m"][0]
    dz_below = np.clip(np.minimum(toa_m, fine["h_edge_m"][1:])
                       - fine["h_edge_m"][:-1], 0.0, None) / 1e3   # km
    keep = dz_below > 0
    nu_f, _, od_fine = layer_od(
        absco_gas, absco_h2o,
        fine["p_lay_pa"][keep] / 100.0, fine["t_lay_k"][keep],
        fine["d_o2"][keep], fine["d_h2o"][keep], fine["h2o_vmr"][keep],
        dz_below[keep])
    assert np.array_equal(nu, nu_f)
    od_col_fine = od_fine.sum(axis=1)

    rel = np.abs(od_col[sel] / od_col_fine[sel] - 1)
    print(f"Column-OD coarse-vs-fine rel diff: median {np.median(rel):.3e}  "
          f"max {rel.max():.3e}")

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    with h5py.File(cfg.OD_FILE, "w") as f:
        f["nu_cm1"] = nu[sel]
        f["wvl_um"] = wvl_um[sel]
        f["wvl_nm"] = wvl_um[sel] * 1e3
        f["od_layer"] = od[sel, :]
        f["od_column"] = od_col[sel]
        f["od_column_fine"] = od_col_fine[sel]
        f["slant_tau"] = slant_tau[sel]
        f.attrs["airmass"] = airmass
        f.attrs["sza"] = sza
        f.attrs["sensor_zenith"] = cfg.SENSOR_ZENITH
        g = f.create_group("full_grid")
        g["nu_cm1"] = nu
        g["od_column"] = od_col
    print(f"Wrote {cfg.OD_FILE}")


if __name__ == "__main__":
    main()

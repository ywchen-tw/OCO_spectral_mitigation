"""
Stage 1: build the slab atmosphere from one 29252a_GL footprint.

Reads the OCO-2 Met + CO2-prior granule files, reconstructs the 72-layer
GEOS hybrid-sigma profile for one sounding (same construction as
src/abs_util/fp_atm.py), then regrids to the coarse slab grid
(slab_config.LEVELS_KM) conserving the partial column of every gas, so the
nadir column optical depth is preserved by construction.

Output: slab_atm.h5
    lev/{altitude_km,pressure_hpa}
    lay/{altitude_km,thickness_km,pressure_hpa,temperature_k,
         d_air_cm3,d_o2_cm3,d_co2_cm3,d_h2o_cm3,h2o_vmr}
    fine/  (72-layer originals, for verification)
    meta/  (sounding_id, lat, lon, sza, vza, doy, dist_au, ...)

Run:  python workspace/rt_slab_sim/build_atmosphere.py
"""
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

# Constants (identical to fp_atm.py)
RD = 287.052874
EPSILON = 0.622
KB = 1.380649e-23
G0 = 9.81
AU_M = 149597870700.0

# GEOS-Chem 72-layer hybrid grid (Ap hPa, Bp unitless) -- copied from
# src/abs_util/fp_atm.py (module-local there, not importable)
AP = np.array([
    0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01, 1.961311e+01, 2.609201e+01,
    3.257081e+01, 3.898201e+01, 4.533901e+01, 5.169611e+01, 5.805321e+01, 6.436264e+01,
    7.062198e+01, 7.883422e+01, 8.909992e+01, 9.936521e+01, 1.091817e+02, 1.189586e+02,
    1.286959e+02, 1.429100e+02, 1.562600e+02, 1.696090e+02, 1.816190e+02, 1.930970e+02,
    2.032590e+02, 2.121500e+02, 2.187760e+02, 2.238980e+02, 2.243630e+02, 2.168650e+02,
    2.011920e+02, 1.769300e+02, 1.503930e+02, 1.278370e+02, 1.086630e+02, 9.236572e+01,
    7.851231e+01, 6.660341e+01, 5.638791e+01, 4.764391e+01, 4.017541e+01, 3.381001e+01,
    2.836781e+01, 2.373041e+01, 1.979160e+01, 1.645710e+01, 1.364340e+01, 1.127690e+01,
    9.292942e+00, 7.619842e+00, 6.216801e+00, 5.046801e+00, 4.076571e+00, 3.276431e+00,
    2.620211e+00, 2.084970e+00, 1.650790e+00, 1.300510e+00, 1.019440e+00, 7.951341e-01,
    6.167791e-01, 4.758061e-01, 3.650411e-01, 2.785261e-01, 2.113490e-01, 1.594950e-01,
    1.197030e-01, 8.934502e-02, 6.600001e-02, 4.758501e-02, 3.270000e-02, 2.000000e-02,
    1.000000e-02])
BP = np.array([
    1.000000e+00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
    8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
    7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
    5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
    2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
    6.960025e-03, 8.175413e-09] + [0.0] * 41)


def load_footprint_profile(sounding_id=None):
    """Return the 72-layer profile dict for one sounding (surface -> top)."""
    with h5py.File(cfg.OCO_FILES["met"], "r") as f:
        lon = f["SoundingGeometry/sounding_longitude"][...]
        lat = f["SoundingGeometry/sounding_latitude"][...]
        valid = np.isfinite(lon) & np.isfinite(lat)
        qprf = f["Meteorology/specific_humidity_profile_met"][...][valid][:, ::-1]
        sfc_p = f["Meteorology/surface_pressure_met"][...][valid]
        tprf = f["Meteorology/temperature_profile_met"][...][valid][:, ::-1]
        pprf = f["Meteorology/vector_pressure_levels_met"][...][valid][:, ::-1]
        sfc_gph = f["Meteorology/gph_met"][...][valid]
        lon_v, lat_v = lon[valid], lat[valid]

    with h5py.File(cfg.OCO_FILES["co2prior"], "r") as f:
        lon_c = f["SoundingGeometry/sounding_longitude"][...]
        lat_c = f["SoundingGeometry/sounding_latitude"][...]
        valid_c = np.isfinite(lon_c) & np.isfinite(lat_c)
        co2_prf = f["CO2Prior/co2_prior_profile_cpr"][...][valid_c][:, ::-1]
        snd_id = f["SoundingGeometry/sounding_id"][...][valid_c]
        sza = f["SoundingGeometry/sounding_solar_zenith"][...][valid_c]
        vza = f["SoundingGeometry/sounding_zenith"][...][valid_c]

    with h5py.File(cfg.OCO_FILES["l1b"], "r") as f:
        lon_l = f["SoundingGeometry/sounding_longitude"][...]
        lat_l = f["SoundingGeometry/sounding_latitude"][...]
        valid_l = np.isfinite(lon_l) & np.isfinite(lat_l)
        dist_m = f["SoundingGeometry/sounding_solar_distance"][...][valid_l]
        land_frac = f["SoundingGeometry/sounding_land_fraction"][...][valid_l]

    for arr in (qprf, sfc_p, tprf, pprf, sfc_gph, co2_prf):
        arr[arr == -999999] = np.nan

    n = sfc_p.shape[0]
    assert co2_prf.shape[0] == n and dist_m.shape[0] == n, "sounding count mismatch"

    row_ok = (
        np.isfinite(sfc_p)
        & np.all(np.isfinite(tprf), axis=1)
        & np.all(np.isfinite(pprf), axis=1)
        & np.all(np.isfinite(qprf), axis=1)
        & np.all(np.isfinite(co2_prf), axis=1)
        & np.isfinite(sfc_gph)
    )
    if sounding_id is not None:
        idx_cand = np.where(snd_id == sounding_id)[0]
        if idx_cand.size == 0 or not row_ok[idx_cand[0]]:
            raise ValueError(f"sounding_id {sounding_id} not found or invalid")
        i = idx_cand[0]
    else:
        ok_inds = np.where(row_ok)[0]
        i = ok_inds[len(ok_inds) // 2]  # median-index valid sounding

    p_edge = AP * 100 + BP * sfc_p[i]                       # Pa
    log_p_ratio = np.log(p_edge[:-1] / p_edge[1:])
    r = qprf[i] / (1 - qprf[i])                             # mass mixing ratio
    e = pprf[i] * r / (EPSILON + r)                         # vapor pressure Pa
    tv = tprf[i] / (1 - (r / (r + EPSILON)) * (1 - EPSILON))
    dz = (RD * tv) / G0 * log_p_ratio                       # m

    h_edge = np.empty(73)
    h_edge[0] = sfc_gph[i]
    h_edge[1:] = sfc_gph[i] + np.cumsum(dz)

    d_air = pprf[i] / (KB * tprf[i]) / 1e6                  # molec/cm3
    d_dry = (pprf[i] - e) / (KB * tprf[i]) / 1e6
    d_h2o = e / (KB * tprf[i]) / 1e6
    d_o2 = d_dry * cfg.O2MIX
    d_co2 = d_dry * co2_prf[i]

    date = os.path.basename(cfg.OCO_FILES["co2prior"]).split("_")[3]
    doy = pd.to_datetime("20" + date, format="%Y%m%d").dayofyear

    return {
        "p_edge_pa": p_edge, "h_edge_m": h_edge, "dz_m": dz,
        "p_lay_pa": pprf[i], "t_lay_k": tprf[i],
        "d_air": d_air, "d_dry": d_dry, "d_h2o": d_h2o,
        "d_o2": d_o2, "d_co2": d_co2,
        "h2o_vmr": d_h2o / d_dry,
        "meta": {
            "sounding_id": int(snd_id[i]),
            "lat": float(lat_v[i]), "lon": float(lon_v[i]),
            "sza": float(sza[i]), "vza": float(vza[i]),
            "sfc_gph_m": float(sfc_gph[i]), "sfc_p_pa": float(sfc_p[i]),
            "land_fraction": float(land_frac[i]),
            "dist_au": float(dist_m[i] / AU_M), "doy": int(doy),
        },
    }


def regrid_conserving(prof):
    """Aggregate the 72-layer profile onto LEVELS_KM conserving partial columns.

    Coarse levels are km above the footprint surface; densities become
    partial-column / thickness means, temperature is air-column weighted.
    Above the fine-grid top the coarse layers are empty -- the fine top
    (~80 km) exceeds the 60 km slab TOA, so every coarse layer is covered.
    """
    lev_m = cfg.LEVELS_KM * 1e3 + prof["h_edge_m"][0]
    fine_lo, fine_hi = prof["h_edge_m"][:-1], prof["h_edge_m"][1:]
    n_coarse = lev_m.size - 1

    if lev_m[-1] > prof["h_edge_m"][-1]:
        raise ValueError("slab TOA above fine-grid top")

    def overlap(lo, hi):
        return np.clip(np.minimum(hi, fine_hi) - np.maximum(lo, fine_lo), 0.0, None)

    out = {k: np.empty(n_coarse) for k in
           ("d_air", "d_dry", "d_o2", "d_co2", "d_h2o", "t_lay_k", "p_lay_pa")}
    for j in range(n_coarse):
        w = overlap(lev_m[j], lev_m[j + 1])          # m of each fine layer inside
        dz_c = lev_m[j + 1] - lev_m[j]
        col_air = np.sum(prof["d_air"] * w)
        for gas in ("d_air", "d_dry", "d_o2", "d_co2", "d_h2o"):
            out[gas][j] = np.sum(prof[gas] * w) / dz_c
        out["t_lay_k"][j] = np.sum(prof["t_lay_k"] * prof["d_air"] * w) / col_air
        out["p_lay_pa"][j] = np.sum(prof["p_lay_pa"] * prof["d_air"] * w) / col_air

    # level pressures: log-p interpolation of the fine edges onto coarse edges
    p_lev = np.exp(np.interp(lev_m, prof["h_edge_m"], np.log(prof["p_edge_pa"])))

    return {
        "lev_alt_km": cfg.LEVELS_KM,
        "lev_p_hpa": p_lev / 100.0,
        "lay_alt_km": 0.5 * (cfg.LEVELS_KM[:-1] + cfg.LEVELS_KM[1:]),
        "lay_dz_km": np.diff(cfg.LEVELS_KM),
        "lay_p_hpa": out["p_lay_pa"] / 100.0,
        "lay_t_k": out["t_lay_k"],
        "d_air": out["d_air"], "d_dry": out["d_dry"],
        "d_o2": out["d_o2"], "d_co2": out["d_co2"], "d_h2o": out["d_h2o"],
        "h2o_vmr": out["d_h2o"] / out["d_dry"],
    }


def column_cm2(dens_cm3, dz_m):
    """Partial column in molec/cm2 from density [cm^-3] and thickness [m]."""
    return np.sum(dens_cm3 * dz_m * 100.0)


def main():
    prof = load_footprint_profile(cfg.SOUNDING_ID)
    m = prof["meta"]
    print(f"Selected sounding {m['sounding_id']}  "
          f"lat {m['lat']:.2f}  lon {m['lon']:.2f}  SZA {m['sza']:.2f}  "
          f"VZA {m['vza']:.2f}  land_frac {m['land_fraction']:.0f}  "
          f"sfc_gph {m['sfc_gph_m']:.1f} m")

    coarse = regrid_conserving(prof)

    # --- verification: partial columns conserved up to slab TOA ---
    toa_m = cfg.LEVELS_KM[-1] * 1e3 + prof["h_edge_m"][0]
    fine_lo, fine_hi = prof["h_edge_m"][:-1], prof["h_edge_m"][1:]
    w_below = np.clip(np.minimum(toa_m, fine_hi) - fine_lo, 0.0, None)
    print("\nColumn conservation (fine grid below slab TOA vs coarse grid):")
    for gas in ("d_air", "d_o2", "d_co2", "d_h2o"):
        col_f = np.sum(prof[gas] * w_below * 100.0)
        col_c = column_cm2(coarse[gas], coarse["lay_dz_km"] * 1e3)
        frac_above = 1.0 - np.sum(prof[gas] * w_below) / np.sum(prof[gas] * prof["dz_m"])
        print(f"  {gas:6s}: fine {col_f:.6e}  coarse {col_c:.6e}  "
              f"rel diff {abs(col_c / col_f - 1):.2e}  above-TOA {frac_above:.2e}")

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    with h5py.File(cfg.ATM_FILE, "w") as f:
        g = f.create_group("lev")
        g["altitude_km"] = coarse["lev_alt_km"]
        g["pressure_hpa"] = coarse["lev_p_hpa"]
        g = f.create_group("lay")
        g["altitude_km"] = coarse["lay_alt_km"]
        g["thickness_km"] = coarse["lay_dz_km"]
        g["pressure_hpa"] = coarse["lay_p_hpa"]
        g["temperature_k"] = coarse["lay_t_k"]
        g["d_air_cm3"] = coarse["d_air"]
        g["d_dry_cm3"] = coarse["d_dry"]
        g["d_o2_cm3"] = coarse["d_o2"]
        g["d_co2_cm3"] = coarse["d_co2"]
        g["d_h2o_cm3"] = coarse["d_h2o"]
        g["h2o_vmr"] = coarse["h2o_vmr"]
        g = f.create_group("fine")
        g["h_edge_m"] = prof["h_edge_m"]
        g["p_edge_pa"] = prof["p_edge_pa"]
        g["p_lay_pa"] = prof["p_lay_pa"]
        g["t_lay_k"] = prof["t_lay_k"]
        g["dz_m"] = prof["dz_m"]
        for gas in ("d_air", "d_dry", "d_o2", "d_co2", "d_h2o"):
            g[gas] = prof[gas]
        g["h2o_vmr"] = prof["h2o_vmr"]
        g = f.create_group("meta")
        for k, v in m.items():
            g.attrs[k] = v
    print(f"\nWrote {cfg.ATM_FILE}")


if __name__ == "__main__":
    main()

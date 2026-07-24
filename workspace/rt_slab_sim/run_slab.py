"""
Stage 3: MCARaTS x-z slab runs via er3t (installed er3t_mca_v11 fork).

For each surface albedo (dark/bright) x solver (3d/ipa) x selected
wavelength: one MCARaTS radiance run on the identical scene -- a 24 km
periodic slab (Nx=48, dx=0.5 km, Ny=1) with a boundary-layer water cloud
occupying the 1-2 km layer at x = 9.5-14.5 km. Gas absorption is the
footprint-derived monochromatic OD from slab_od.h5 (Ng=1 duck-typed abs
object); Rayleigh comes from the same footprint profile via the er3t
atmosphere object; cloud scattering uses the water-cloud Mie table.

Usage:
  python run_slab.py --photons 1e6                    # full local sweep
  python run_slab.py --photons 1e6 --wvl 0 16 32      # subset of wavelengths
  python run_slab.py --collect                        # assemble slab_rad.h5
"""
import argparse
import datetime
import os
import sys

import h5py
import numpy as np

# NumPy 2.0 compatibility shims for the er3t_mca_v11 fork (np.string_ /
# np.float_ were removed in NumPy 2.0; restore the aliases in-process
# instead of editing the er3t library)
if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "float_"):
    np.float_ = np.float64

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

# MCARaTS v0.10.4 with the matching er3t snapshot at ~/programming/er3t/er3t
# (takes import precedence over the installed er3t_mca_v11 fork).
# Override via MCARATS_V010_EXE / ER3T_DIR on CURC.
os.environ.setdefault("MCARATS_V010_EXE", os.path.expanduser(
    "~/programming/er3t/mcarats/v0.10.4/src/mcarats"))
sys.path.insert(0, os.environ.get(
    "ER3T_DIR", os.path.expanduser("~/programming/er3t/er3t")))

from er3t.pre.pha import pha_mie_wc
from er3t.rtm.mca import (mca_atm_1d, mca_atm_3d, mca_out_raw, mca_sca,
                          mcarats_ng, mca_out_ng)

DATE = datetime.datetime(2020, 1, 1)


class SlabMcarats(mcarats_ng):
    """mcarats_ng with extra namelist entries injected before input-file
    generation (e.g. Rad_mplen path-length statistics, Rad_difr* overrides)."""

    extra_nml = {}

    def gen_mca_inp(self, comment=False):
        for ig in range(self.Ng):
            self.nml[ig].update(self.extra_nml)
        super().gen_mca_inp(comment=comment)


# --------------------------------------------------------------------------
# Duck-typed er3t objects
# --------------------------------------------------------------------------
class SlabAbs:
    """Monochromatic (Ng=1) absorption object satisfying the er3t abs contract.

    mca_atm_1d consumes coef['abso_coef']['data'] (nlay, Ng) as per-layer OD;
    mca_out_ng consumes coef['solar'] and coef['weight'] for normalization
    (solar=1.0 -> output radiance is in reflectance-normalized units; the
    cumulant fit intercept absorbs any constant scale).
    """

    def __init__(self, wvl_nm, od_layer):
        self.Ng = 1
        self.wvl = float(wvl_nm)
        self.wvl_info = f"{wvl_nm:.4f} nm (slab monochromatic)"
        self.coef = {
            "abso_coef": {"data": np.asarray(od_layer, dtype=np.float64)[:, None]},
            "weight": {"data": np.array([1.0])},
            "solar": {"data": np.array([1.0])},
            "slit_func": {"data": np.ones((2, 1))},
            "wavelength": {"data": np.array([float(wvl_nm)])},
        }


class SlabCloud:
    """Minimal cloud object for mca_atm_3d: one cloudy model layer."""

    def __init__(self, atm, lat):
        nx, ny = cfg.NX, cfg.NY
        alt_lay = atm.lay["altitude"]["data"]
        thick_lay = atm.lay["thickness"]["data"]
        t_lay = atm.lay["temperature"]["data"]

        in_cloud = (alt_lay > cfg.CLOUD_BASE_KM) & (alt_lay < cfg.CLOUD_TOP_KM)
        iz = np.where(in_cloud)[0]
        if iz.size == 0:
            raise ValueError("no model layer inside the cloud altitude range")

        nz = iz.size
        geom_m = np.sum(thick_lay[iz]) * 1e3
        ext = np.zeros((nx, ny, nz), dtype=np.float64)
        x_centers = (np.arange(nx) + 0.5) * cfg.DX_KM
        in_x = (x_centers >= cfg.CLOUD_X_KM[0]) & (x_centers < cfg.CLOUD_X_KM[1])
        ext[in_x, :, :] = cfg.CLOUD_COD / geom_m          # m^-1

        cer = np.full((nx, ny, nz), cfg.CLOUD_CER_UM, dtype=np.float64)
        tmp = np.broadcast_to(t_lay[iz], (nx, ny, nz)).copy()

        self.lay = {
            "nx": {"data": nx}, "ny": {"data": ny},
            "dx": {"data": cfg.DX_KM}, "dy": {"data": cfg.DY_KM},
            "altitude": {"data": alt_lay[iz]},
            "thickness": {"data": thick_lay[iz]},
            "extinction": {"data": ext},
            "temperature": {"data": tmp},
            "cer": {"data": cer},
        }
        self.lev = {"altitude": {"data": np.append(
            alt_lay[iz] - 0.5 * thick_lay[iz], alt_lay[iz[-1]] + 0.5 * thick_lay[iz[-1]])}}
        self.in_x = in_x
        self.lat = lat


class SlabAtm:
    """Duck-typed er3t atmosphere on the slab grid (footprint profile).

    Provides exactly what mca_atm_1d / mca_atm_3d / cal_mol_ext_atm consume:
    lev altitude+pressure, lay altitude/thickness/pressure/temperature and
    air/o2/co2/h2o number densities (cm-3), plus .lat for gravity.
    (atm_atmmod is bypassed: its AFGL interpolation breaks under NumPy 2
    and every field would be overwritten with the footprint profile anyway.)
    """

    def __init__(self, lev, lay, lat):
        self.lev = {
            "altitude": {"data": lev["altitude_km"], "units": "km"},
            "pressure": {"data": lev["pressure_hpa"], "units": "mb"},
        }
        self.lay = {
            "altitude": {"data": lay["altitude_km"], "units": "km"},
            "thickness": {"data": lay["thickness_km"], "units": "km"},
            "pressure": {"data": lay["pressure_hpa"], "units": "mb"},
            "temperature": {"data": lay["temperature_k"], "units": "K"},
            "air": {"data": lay["d_air_cm3"], "units": "cm-3"},
            "o2": {"data": lay["d_o2_cm3"], "units": "cm-3"},
            "co2": {"data": lay["d_co2_cm3"], "units": "cm-3"},
            "h2o": {"data": lay["d_h2o_cm3"], "units": "cm-3"},
        }
        self.lat = lat


def build_atm():
    """Slab atmosphere with the footprint profile injected."""
    with h5py.File(cfg.ATM_FILE, "r") as f:
        lev = {k: f["lev"][k][...] for k in f["lev"]}
        lay = {k: f["lay"][k][...] for k in f["lay"]}
        meta = dict(f["meta"].attrs)
    os.makedirs(cfg.TMP_DIR, exist_ok=True)
    return SlabAtm(lev, lay, meta["lat"]), meta


def run_dir(surface, solver, iw):
    return os.path.join(cfg.TMP_DIR, surface, solver, f"w{iw:03d}")


def out_file(surface, solver, iw):
    return os.path.join(run_dir(surface, solver, iw), "mca-out-rad.h5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=float, default=cfg.PHOTONS_LOCAL)
    ap.add_argument("--nrun", type=int, default=cfg.NRUN)
    ap.add_argument("--wvl", type=int, nargs="*", default=None,
                    help="wavelength indices (default: all)")
    ap.add_argument("--surfaces", nargs="*", default=list(cfg.SURFACE_ALBEDOS),
                    choices=list(cfg.SURFACE_ALBEDOS))
    ap.add_argument("--solvers", nargs="*", default=list(cfg.SOLVERS),
                    choices=list(cfg.SOLVERS))
    ap.add_argument("--ncpu", default="auto")
    ap.add_argument("--no-plen", action="store_true",
                    help="disable the Rad_mplen path-length histogram tally")
    ap.add_argument("--collect", action="store_true",
                    help="assemble slab_rad.h5 from completed runs, no new runs")
    args = ap.parse_args()

    with h5py.File(cfg.OD_FILE, "r") as f:
        wvl_nm = f["wvl_nm"][...]
        od_layer = f["od_layer"][...]
        slant_tau = f["slant_tau"][...]
    iw_all = list(range(wvl_nm.size)) if args.wvl is None else args.wvl

    if args.collect:
        collect(wvl_nm, slant_tau)
        return

    atm, meta = build_atm()
    sza = float(meta["sza"])
    cld = SlabCloud(atm, meta["lat"])
    ncpu = args.ncpu if args.ncpu == "auto" else int(args.ncpu)

    if not args.no_plen:
        SlabMcarats.extra_nml = {
            "Rad_mplen": cfg.PLEN_MODE,
            "Rad_ntp": cfg.PLEN_NBIN,
            "Rad_tpmin": cfg.PLEN_MIN_M,
            "Rad_tpmax": cfg.PLEN_MAX_M,
        }

    # per-invocation dir for the shared scattering/3D-atm binaries: Slurm
    # array tasks run concurrently and must not rebuild the same files
    bins_dir = os.path.join(
        cfg.TMP_DIR, f"bins_{os.environ.get('SLURM_ARRAY_TASK_ID', 'local')}")
    os.makedirs(bins_dir, exist_ok=True)
    pha0 = pha_mie_wc(wavelength=cfg.PHA_MIE_WVL_NM, overwrite=False)
    sca = mca_sca(pha_obj=pha0, fname=os.path.join(bins_dir, "mca_sca.bin"),
                  overwrite=True)
    atm3d = mca_atm_3d(cld_obj=cld, atm_obj=atm, pha_obj=pha0,
                       fname=os.path.join(bins_dir, "mca_atm_3d.bin"),
                       overwrite=True)

    n_total = len(args.surfaces) * len(args.solvers) * len(iw_all)
    i_run = 0
    for surface in args.surfaces:
        albedo = cfg.SURFACE_ALBEDOS[surface]
        for solver in args.solvers:
            for iw in iw_all:
                i_run += 1
                fdir = run_dir(surface, solver, iw)
                os.makedirs(fdir, exist_ok=True)
                print(f"[{i_run}/{n_total}] {surface} {solver} "
                      f"w{iw:03d} ({wvl_nm[iw]:.3f} nm, slant tau "
                      f"{slant_tau[iw]:.3f})", flush=True)

                abs0 = SlabAbs(wvl_nm[iw], od_layer[iw])
                atm1d = mca_atm_1d(atm_obj=atm, abs_obj=abs0)
                mca0 = SlabMcarats(
                    date=DATE,
                    atm_1ds=[atm1d], atm_3ds=[atm3d], sca=sca,
                    Ng=1, weights=abs0.coef["weight"]["data"],
                    target="radiance",
                    surface_albedo=albedo,
                    solar_zenith_angle=sza,
                    solar_azimuth_angle=cfg.SOLAR_AZIMUTH,
                    sensor_zenith_angle=cfg.SENSOR_ZENITH,
                    sensor_azimuth_angle=cfg.SENSOR_AZIMUTH,
                    sensor_altitude=cfg.SENSOR_ALTITUDE_M,
                    fdir=fdir, Nrun=args.nrun,
                    photons=args.photons, solver=solver,
                    Ncpu=ncpu, mp_mode="py", overwrite=True)
                mca_out_ng(fname=out_file(surface, solver, iw), mca_obj=mca0,
                           abs_obj=abs0, mode="all", squeeze=True,
                           overwrite=True)

    # auto-collect only for a full standalone sweep -- concurrent Slurm array
    # tasks (or subset runs) must not race on slab_rad.h5; run --collect once
    # after the array completes instead
    if args.wvl is None and "SLURM_ARRAY_TASK_ID" not in os.environ:
        collect(wvl_nm, slant_tau)
    else:
        print("Subset/array run complete -- run with --collect afterwards "
              "to assemble slab_rad.h5")


def read_plen(fdir):
    """Per-run path-length histograms (Nrun, Nx, Ntp) from the raw GrADS
    output, or None if the pln variable is absent.  MCARaTS mode-3 output is
    the fraction of each pixel's radiance contributed by each total-path bin
    (already radiance-normalized per pixel)."""
    import glob
    hists = []
    for path in sorted(glob.glob(os.path.join(fdir, "r*.out.bin"))):
        out0 = mca_out_raw(path)
        # v0.10.4 names the variable 'b1 (Pathlength Statistics)'; v0.11 'pln_0001'
        pln = [d for d in out0.data
               if "pathlength" in d["name"].lower() or d["name"].startswith("pln")]
        if not pln:
            return None
        hists.append(np.squeeze(pln[0]["data"]))     # (Nx, Ntp)
    return np.stack(hists) if hists else None


def collect(wvl_nm, slant_tau):
    """Assemble every completed run into slab_rad.h5."""
    nw = wvl_nm.size
    with h5py.File(cfg.RAD_FILE, "w") as f:
        f["wvl_nm"] = wvl_nm
        f["slant_tau"] = slant_tau
        f.attrs["plen_min_m"] = cfg.PLEN_MIN_M
        f.attrs["plen_max_m"] = cfg.PLEN_MAX_M
        f.attrs["plen_nbin"] = cfg.PLEN_NBIN
        n_found = 0
        for surface in cfg.SURFACE_ALBEDOS:
            for solver in cfg.SOLVERS:
                rad, rad_runs, toa, plen = {}, {}, {}, {}
                for iw in range(nw):
                    path = out_file(surface, solver, iw)
                    if not os.path.exists(path):
                        continue
                    with h5py.File(path, "r") as g:
                        r = g["mean/rad"][...] if "mean/rad" in g else g["all/rad"][...]
                        if "all/rad" in g:
                            rad_runs[iw] = g["all/rad"][...]
                            r = rad_runs[iw].mean(axis=-1)
                        rad[iw] = r
                        toa[iw] = g["mean/toa"][...] if "mean/toa" in g else g["all/toa"][...]
                    p = read_plen(run_dir(surface, solver, iw))
                    if p is not None:
                        plen[iw] = p
                    n_found += 1
                if not rad:
                    continue
                grp = f.create_group(f"{surface}/{solver}")
                iws = sorted(rad)
                grp["iw"] = np.array(iws)
                grp["rad"] = np.stack([rad[i] for i in iws])
                if rad_runs:
                    grp["rad_runs"] = np.stack([rad_runs[i] for i in iws])
                grp["toa"] = np.array([np.atleast_1d(toa[i])[0] for i in iws])
                if plen:
                    grp.create_dataset(
                        "plen_hist", data=np.stack([plen[i] for i in iws]),
                        compression="gzip", compression_opts=4)
        print(f"Collected {n_found} runs -> {cfg.RAD_FILE}")


if __name__ == "__main__":
    main()

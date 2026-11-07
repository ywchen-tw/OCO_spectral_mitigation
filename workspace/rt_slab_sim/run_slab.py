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
from er3t.rtm.mca import (mca_atm_1d, mca_atm_3d, mca_out_raw, mca_sca, mca_sfc_2d,
                          mcarats_ng, mca_out_ng)

DATE = datetime.datetime(2020, 1, 1)


class SlabMcarats(mcarats_ng):
    """mcarats_ng with extra namelist entries injected before input-file
    generation (e.g. Rad_mplen path-length statistics, Rad_difr* overrides).

    How the photon path-length tally works inside MCARaTS
    -----------------------------------------------------
    MCARaTS carries a per-photon array ``Pho_plen(0:nz+1)`` -- the geometric
    distance travelled in each model layer, accumulated by the ray tracer
    (mcarAtm.F90).  Radiance is computed by local estimation: at every
    scattering event a virtual copy of the photon is traced toward the
    sensor (mcarRad__samp1) and keeps extending its own copy ``PhoV_plen``,
    so when a contribution reaches a radiance pixel, sum(PhoV_plen) is the
    TOTAL geometric path of that contribution: TOA entry -> all scattering
    events -> sensor (including the constant vacuum leg above the
    atmosphere to the 705-km sensor).

    Setting ``Rad_mplen = 3`` (a built-in feature -- no source edit) makes
    the sampler bin every contribution's total path into ``Rad_ntp`` bins
    spanning [Rad_tpmin, Rad_tpmax] meters, weighted by the contribution's
    radiance weight (mcarRad.F90:803).  On output the histogram is
    normalized by the pixel radiance (mcarRad__normal), so the written
    field is the FRACTION of each pixel's radiance contributed by each
    total-path bin -- a per-column, radiance-weighted photon path-length
    distribution (PPDF) that sums to ~1 over bins.  NOTE this is the
    GEOMETRIC path; the spectral fit senses the absorption-weighted path
    (layers weighted by absorber density) -- see fit_and_plot.plen_moments.
    """

    extra_nml = {}
    seed_base = None    # explicit Wld_jseed base; run ir gets seed_base + ir

    def gen_mca_inp(self, comment=False):
        for ig in range(self.Ng):
            self.nml[ig].update(self.extra_nml)
        super().gen_mca_inp(comment=comment)
        if self.seed_base is not None:
            # er3t seeds every run with int(time.time()) + shuffle(0..Nrun-1);
            # concurrent array tasks started in the same second would share
            # seeds, so runs are seeded explicitly per (scene, surface,
            # solver, wavelength, run) -- unique within a scene and
            # reproducible when a preempted task is requeued
            from er3t.rtm.mca.mca_inp import mca_inp_file
            for ir in range(self.Nrun):
                for ig in range(self.Ng):
                    self.nml[ig]["Wld_jseed"] = int(self.seed_base + ir)
                    mca_inp_file(self.fnames_inp[ir][ig], self.nml[ig],
                                 comment=comment)


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

        # the 3-D block spans the cloud layers and, with aerosol, down to the
        # aerosol base (ext = 0 outside the cloud; SlabAtm3d adds the aerosol)
        lo = cfg.CLOUD_BASE_KM
        if cfg.AEROSOL is not None:
            lo = min(lo, cfg.AEROSOL["base_km"])
        in_block = (alt_lay > lo) & (alt_lay < cfg.CLOUD_TOP_KM)
        iz = np.where(in_block)[0]
        if iz.size == 0:
            raise ValueError("no model layer inside the cloud altitude range")
        in_cloud = (alt_lay[iz] > cfg.CLOUD_BASE_KM) & (alt_lay[iz] < cfg.CLOUD_TOP_KM)

        nz = iz.size
        geom_m = np.sum(thick_lay[iz][in_cloud]) * 1e3
        ext = np.zeros((nx, ny, nz), dtype=np.float64)
        x_centers = (np.arange(nx) + 0.5) * cfg.DX_KM
        in_x = (x_centers >= cfg.CLOUD_X_KM[0]) & (x_centers < cfg.CLOUD_X_KM[1])
        ext[in_x, :, :] = (cfg.CLOUD_COD / geom_m) * in_cloud[None, None, :]   # m^-1

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


class SlabAtm3d(mca_atm_3d):
    """mca_atm_3d with the slab's phase-function and aerosol options.

    er3t writes the Mie asymmetry parameter into Atm_apfp3d (-> MCARaTS
    Henyey-Greenstein).  CLOUD_PHASE == "mie" replaces it in the cloud cells
    by the 1-based index of the tabulated Mie phase function closest to
    CLOUD_CER_UM (apf >= 1 selects the table written by mca_sca).  AEROSOL
    adds a second 3-D constituent: a horizontally homogeneous layer with
    ext = AOD / thickness, the given single-scattering albedo and HG g.
    """

    def pre_mca_3d_atm(self):
        super().pre_mca_3d_atm()
        ext = self.nml["Atm_extp3d"]["data"]
        apf = self.nml["Atm_apfp3d"]["data"]
        self.mie_index = None
        if cfg.CLOUD_PHASE == "mie":
            ref = self.pha.data["ref"]["data"]
            self.mie_index = int(np.argmin(np.abs(ref - cfg.CLOUD_CER_UM))) + 1
            apf[ext > 0] = float(self.mie_index)
        if cfg.AEROSOL is not None:
            a = cfg.AEROSOL
            alt = self.cld.lay["altitude"]["data"]
            thick = self.cld.lay["thickness"]["data"]
            in_aer = (alt > a["base_km"]) & (alt < a["top_km"])
            if not in_aer.any():
                raise ValueError("no 3-D layer inside the aerosol altitude range")
            geom_m = float(np.sum(thick[in_aer])) * 1e3
            ext3d = np.zeros(ext.shape[:3], dtype=np.float32)
            ext3d[:, :, in_aer] = cfg.AEROSOL_AOD / geom_m
            self.add_mca_3d_atm(ext3d=ext3d,
                                omg3d=np.full(ext3d.shape, a["ssa"], dtype=np.float32),
                                apf3d=np.full(ext3d.shape, a["g"], dtype=np.float32))


class SlabSurface:
    """Minimal 2-D surface object for SlabSfc2d: an (nx, ny) Lambertian albedo
    map, or an (nx, ny, 5) MCARaTS BRDF parameter map with model id jsfc
    (1 Lambertian, 2 DSM diffuse-specular mixture)."""

    def __init__(self, albedo_2d, jsfc=1):
        nx, ny = albedo_2d.shape[:2]
        self.Nx, self.Ny = nx, ny
        self.data = {"nx": {"data": nx}, "ny": {"data": ny}, "jsfc": jsfc,
                     "sfc": {"name": "surface BRDF parameters", "data": albedo_2d}}


def dsm_params(spec):
    """MCARaTS DSM parameter packet (mcarSfc.F90): alpha_d, f_d, n_real,
    n_imag, sigma^2 (Cox-Munk facet-slope variance)."""
    return np.array([spec["alpha_d"], spec["f_d"], spec["n_real"],
                     spec["n_imag"], spec["sigma2"]], dtype=np.float32)


class SlabSfc2d(mca_sfc_2d):
    """mca_sfc_2d for an (nx, 1) Lambertian albedo map (er3t's own
    pre_mca_2d_sfc squeezes the ny = 1 axis away and then rejects the map)."""

    def pre_mca_2d_sfc(self):
        arr = np.asarray(self.sfc.data["sfc"]["data"], dtype=np.float32)
        jsfc = int(self.sfc.data.get("jsfc", 1))
        nx, ny = arr.shape[:2]
        if arr.ndim == 2:                       # Lambertian albedo map
            psfc = np.zeros((nx, ny, 5), dtype=np.float32)
            psfc[:, :, 0] = np.clip(arr, 0.0, 1.0)
        else:                                   # (nx, ny, 5) BRDF parameters
            psfc = arr
        self.nml = {
            "Sfc_nxb": {"data": nx}, "Sfc_nyb": {"data": ny},
            "Sfc_tmps2d": {"data": np.zeros((nx, ny), dtype=np.float32)},
            "Sfc_jsfc2d": {"data": np.full((nx, ny), jsfc, dtype=np.int16)},
            "Sfc_psfc2d": {"data": psfc},
        }


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
    ap.add_argument("--nrun", type=int, default=cfg.NRUN,
                    help="independent runs (spread estimate)")
    ap.add_argument("--nsub", type=int, default=1,
                    help="MCARaTS sub-runs per independent run, executed in "
                         "parallel with --photons/nsub photons each and pooled "
                         "by --collect (er3t runs one serial process per run, "
                         "so e.g. --photons 1e9 --nsub 10 costs the wall time "
                         "of one 1e8 run per wavelength)")
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
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip wavelengths whose mca-out-rad.h5 exists "
                         "(requeued preemptable array tasks resume)")
    args = ap.parse_args()
    print(f"scene {cfg.SCENE}: OUT_DIR {cfg.OUT_DIR}", flush=True)

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
    if not cfg.RAYLEIGH:
        # zero the 1-D Rayleigh extinction before any mca_atm_1d is built;
        # the cloud-only 3-D field carries no Rayleigh (verified 2026-11-05:
        # Atm_ext1d all zeros in the namelist, k1 = 1.000 on both surfaces)
        import er3t.rtm.mca.mca_atm as _mca_atm
        _mca_atm.cal_mol_ext_atm = (
            lambda wvl, atm_obj: np.zeros(atm_obj.lay["thickness"]["data"].shape))
        print("Rayleigh OFF: er3t cal_mol_ext_atm -> zeros", flush=True)
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
    atm3d = SlabAtm3d(cld_obj=cld, atm_obj=atm, pha_obj=pha0,
                      fname=os.path.join(bins_dir, "mca_atm_3d.bin"),
                      overwrite=True)
    print(f"cloud phase {cfg.CLOUD_PHASE} (Mie table index {atm3d.mie_index}); "
          f"aerosol {cfg.AEROSOL} AOD(band) {cfg.AEROSOL_AOD}; "
          f"3-D constituents {atm3d.nml['Atm_np3d']['data']}; surface step {cfg.SURFACE_STEP}",
          flush=True)
    x_centers = (np.arange(cfg.NX) + 0.5) * cfg.DX_KM

    n_total = len(args.surfaces) * len(args.solvers) * len(iw_all)
    i_run = 0
    import zlib
    scene_code = zlib.crc32(cfg.SCENE.encode()) % 97
    for surface in args.surfaces:
        albedo = cfg.SURFACE_ALBEDOS[surface]
        if isinstance(albedo, dict):            # glint: DSM BRDF, uniform in x
            if albedo.get("model") != "dsm":
                raise ValueError(f"unknown surface model {albedo}")
            par2d = np.broadcast_to(dsm_params(albedo), (cfg.NX, cfg.NY, 5)).copy()
            print(f"surface {surface}: DSM {albedo}", flush=True)
            albedo = SlabSfc2d(atm_obj=atm, sfc_obj=SlabSurface(par2d, jsfc=2),
                               fname=os.path.join(bins_dir, f"mca_sfc_2d_{surface}.bin"),
                               overwrite=True)
        elif cfg.SURFACE_STEP is not None:
            alb2d = np.full((cfg.NX, cfg.NY), albedo, dtype=np.float32)
            alb2d[x_centers >= cfg.SURFACE_STEP["x_km"], :] = cfg.SURFACE_STEP["albedo_right"]
            albedo = SlabSfc2d(atm_obj=atm, sfc_obj=SlabSurface(alb2d),
                               fname=os.path.join(bins_dir, f"mca_sfc_2d_{surface}.bin"),
                               overwrite=True)
        isurf = list(cfg.SURFACE_ALBEDOS).index(surface)
        for solver in args.solvers:
            isolver = cfg.SOLVERS.index(solver)
            for iw in iw_all:
                SlabMcarats.seed_base = ((scene_code * 8 + isurf * 2 + isolver)
                                         * 100_000 + iw * 1000 + 1)
                i_run += 1
                fdir = run_dir(surface, solver, iw)
                if args.skip_existing and os.path.exists(out_file(surface, solver, iw)):
                    print(f"[{i_run}/{n_total}] {surface} {solver} w{iw:03d} exists, skipped",
                          flush=True)
                    continue
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
                    fdir=fdir, Nrun=args.nrun * args.nsub,
                    photons=args.photons / args.nsub, solver=solver,
                    Ncpu=ncpu, mp_mode="py", overwrite=True)
                mca_out_ng(fname=out_file(surface, solver, iw), mca_obj=mca0,
                           abs_obj=abs0, mode="all", squeeze=True,
                           overwrite=True)
                with h5py.File(out_file(surface, solver, iw), "a") as g:
                    g.attrs["nrun"] = args.nrun
                    g.attrs["nsub"] = args.nsub
                    g.attrs["photons_per_run"] = args.photons
                    g.attrs["seed_base"] = SlabMcarats.seed_base

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
    output, or None if the pln variable is absent.

    MCARaTS writes the Rad_mplen=3 histogram next to the radiance in its
    GrADS-style output (.bin + .ctl): v0.10.4 declares it in the ctl as
    variable 'b1 ... Pathlength Statistics' with Rad_ntp Z-levels (v0.11
    names it 'pln_0001').  er3t's mca_out_raw parses whatever variables the
    ctl lists, so we just select the pathlength one by name.  Values are
    already radiance-normalized per pixel (fraction of the pixel's radiance
    per total-path bin, ~sums to 1 over bins); the constant vacuum leg to
    the sensor is included in the path totals and cancels in anomalies."""
    import glob
    from concurrent.futures import ThreadPoolExecutor

    def _one(path):
        out0 = mca_out_raw(path)
        # v0.10.4 names the variable 'b1 (Pathlength Statistics)'; v0.11 'pln_0001'
        pln = [d for d in out0.data
               if "pathlength" in d["name"].lower() or d["name"].startswith("pln")]
        return np.squeeze(pln[0]["data"]) if pln else None   # (Nx, Ntp)

    paths = sorted(glob.glob(os.path.join(fdir, "r*.out.bin")))
    if not paths:
        return None
    # 2026-11-06: opening a file on /scratch costs ~0.45 s (metadata latency),
    # not the read itself; with NSUB=10 that is 30 opens per run dir and the
    # serial collect blew its 30 min limit.  Threaded opens: 13 s -> 2 s per dir.
    with ThreadPoolExecutor(max_workers=16) as ex:
        hists = list(ex.map(_one, paths))
    if any(h is None for h in hists):
        return None
    return np.stack(hists)


def collect(wvl_nm, slant_tau):
    """Assemble every completed run into slab_rad.h5."""
    nw = wvl_nm.size
    with h5py.File(cfg.RAD_FILE, "w") as f:
        f["wvl_nm"] = wvl_nm
        f["slant_tau"] = slant_tau
        f.attrs["plen_min_m"] = cfg.PLEN_MIN_M
        f.attrs["plen_max_m"] = cfg.PLEN_MAX_M
        f.attrs["plen_nbin"] = cfg.PLEN_NBIN
        # scene record (2026-11-06): the geometry/cloud/Rayleigh settings the
        # runs were made with, so analysis never has to trust SLAB_SCENE
        f.attrs["scene"] = cfg.SCENE
        f.attrs["sza"] = cfg.SCENE_SZA
        f.attrs["sensor_zenith"] = cfg.SENSOR_ZENITH
        f.attrs["sensor_azimuth"] = cfg.SENSOR_AZIMUTH
        f.attrs["solar_azimuth"] = cfg.SOLAR_AZIMUTH
        f.attrs["nx"] = cfg.NX
        f.attrs["dx_km"] = cfg.DX_KM
        f.attrs["cloud_x_km"] = np.array(cfg.CLOUD_X_KM)
        f.attrs["cloud_base_eff_km"] = cfg.CLOUD_BASE_EFF_KM
        f.attrs["cloud_top_eff_km"] = cfg.CLOUD_TOP_EFF_KM
        f.attrs["cloud_cod"] = cfg.CLOUD_COD
        f.attrs["cloud_cer_um"] = cfg.CLOUD_CER_UM
        f.attrs["rayleigh"] = bool(cfg.RAYLEIGH)
        f.attrs["band"] = cfg.BAND
        f.attrs["mie_wvl_nm"] = cfg.PHA_MIE_WVL_NM
        f.attrs["cloud_phase"] = cfg.CLOUD_PHASE
        f.attrs["aerosol"] = "none" if cfg.AEROSOL is None else repr(cfg.AEROSOL)
        f.attrs["aerosol_aod_band"] = -1.0 if cfg.AEROSOL_AOD is None else cfg.AEROSOL_AOD
        f.attrs["surface_step"] = "none" if cfg.SURFACE_STEP is None else repr(cfg.SURFACE_STEP)
        f.attrs["surfaces"] = repr(cfg.SURFACE_ALBEDOS)
        n_found = 0
        for surface in cfg.SURFACE_ALBEDOS:
            for solver in cfg.SOLVERS:
                rad, rad_runs, toa, plen = {}, {}, {}, {}
                for iw in range(nw):
                    path = out_file(surface, solver, iw)
                    if not os.path.exists(path):
                        continue
                    nsub, raw = 1, None
                    with h5py.File(path, "r") as g:
                        nsub = int(g.attrs.get("nsub", 1))
                        r = g["mean/rad"][...] if "mean/rad" in g else g["all/rad"][...]
                        if "all/rad" in g:
                            raw = g["all/rad"][...]            # (Nx, nrun*nsub)
                            runs = raw
                            if nsub > 1:
                                # pool the nsub equal-photon sub-runs of each
                                # independent run (mean of per-photon
                                # estimators with equal counts = one big run)
                                runs = raw.reshape(raw.shape[0], -1, nsub).mean(axis=-1)
                            rad_runs[iw] = runs
                            r = runs.mean(axis=-1)
                        rad[iw] = r
                        toa[iw] = g["mean/toa"][...] if "mean/toa" in g else g["all/toa"][...]
                    p = read_plen(run_dir(surface, solver, iw))
                    if p is not None:
                        if nsub > 1 and raw is not None:
                            # (nrun*nsub, Nx, Ntp) radiance-fraction histograms
                            # -> radiance-weighted mean over the sub-runs
                            ntot, nx, ntp = p.shape
                            p = p.reshape(ntot // nsub, nsub, nx, ntp)
                            w = raw.T.reshape(ntot // nsub, nsub, nx)[..., None]
                            p = (p * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-30)
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

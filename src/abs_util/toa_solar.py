"""TOA-only recompute of the solar irradiance in telluric-Doppler tau files.

The telluric-Doppler test variant of the tau step (fp_abs_coeff,
``telluric_doppler=True``) originally shifted the TOA solar spectrum by the
Sun -> sounding-location velocity only (stamp
``transmission_ils_rayleigh_hr_v3_telluric_doppler``).  From stamp v4
(``ils_tau.TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER``) the solar grid also
gets the spacecraft term, beta = (v_solar - v_inst)/c
(fp_abs_coeff.solar_doppler_beta).  Tau and the attribution datasets do not
depend on the solar spectrum, so a v3 file is upgraded by recomputing only the
three ``<band>_toa_sol_output`` datasets -- minutes per orbit instead of the
hours of the full ABSCO convolution.

Inputs per sounding are looked up in the orbit's L1B by sounding_id (frame x
footprint flattened, the layout the full step uses); musolzen comes from the
tau file's own ``sza`` (the full step uses the CO2-prior solar zenith, which is
what that dataset holds).  The per-band ILS / solar arrays and the per-track
convolution are the very functions the full step calls
(fp_abs_coeff.band_ils_solar_setup / toa_sol_track), so with
``solar_vinst=False`` this module reproduces the stored v3 values exactly.

Entry point: recompute_toa_sol(); CLI: workspace/compute_tau_only.py --toa-only.
"""

import glob
import multiprocessing
import os
import re
import time

import h5py
import numpy as np

from abs_util.fp_abs_coeff import absco_input_dir, band_ils_solar_setup, toa_sol_track
from abs_util.fp_atm import _COMPONENT_DATASETS, _TAU_DATASETS, _write_tau_file_atomic
from abs_util.ils_tau import (
    TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER,
    TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER_V3,
)

AU_M = 149597870700.0          # same constant as fp_atm.oco_fp_atm_abs
BANDS = ("o2a", "wco2", "sco2")
TOA_DATASETS = tuple(f"{band}_toa_sol_output" for band in BANDS)
ACCEPTED_STAMPS = (TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER_V3,
                   TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER)
_CORE_ATTRS = ("tau_convolution", "use_ring", "telluric_doppler")


# --------------------------------------------------------------------------- inputs

def chunk_files(tau_file):
    """``<stem>_tmp_<i>.h5`` chunk files next to ``tau_file``, sorted by i."""
    stem = os.path.splitext(tau_file)[0]

    def _key(path):
        m = re.search(r"_tmp_(\d+)\.h5$", path)
        return int(m.group(1)) if m else -1

    return sorted((p for p in glob.glob(f"{glob.escape(stem)}_tmp_*.h5")
                   if re.search(r"_tmp_\d+\.h5$", p)), key=_key)


def tau_file_stamp(path):
    """(tau_convolution, telluric_doppler) attrs of a tau file."""
    with h5py.File(path, "r") as f:
        stamp = f.attrs.get("tau_convolution")
        if isinstance(stamp, bytes):
            stamp = stamp.decode()
        return stamp, bool(f.attrs.get("telluric_doppler", False))


def load_l1b_inputs(oco_l1b_file):
    """Per-sounding Doppler inputs and the InstrumentHeader ILS from an L1B file.

    Arrays are flattened [frame, footprint] (row-major), exactly the order in
    which fp_atm.oco_fp_atm_abs selects them with its finite-lon/lat mask.
    """
    with h5py.File(oco_l1b_file, "r") as f:
        sg = f["SoundingGeometry"]
        snd = sg["sounding_id"][...]
        out = {
            "sounding_id": snd.reshape(-1).astype(np.int64),
            "fp_number": np.broadcast_to(np.arange(snd.shape[1]), snd.shape).reshape(-1),
            "v_solar": sg["sounding_solar_relative_velocity"][...].reshape(-1),
            "v_inst": sg["sounding_relative_velocity"][...].reshape(-1),
            # same expression / dtype as the full step (dist_m / AU_M)
            "dist_au": (sg["sounding_solar_distance"][...] / AU_M).reshape(-1),
            "wvl_coef": f["InstrumentHeader/dispersion_coef_samp"][...],
            "del_lambda_all": f["InstrumentHeader/ils_delta_lambda"][...],
            "rel_lresponse_all": f["InstrumentHeader/ils_relative_response"][...],
        }
    return out


def match_soundings(sounding_id, fp_number, l1b):
    """Indices into the flattened L1B arrays for each tau-file sounding.

    Raises ValueError when a sounding is missing from the L1B, appears twice
    there, or its footprint number disagrees.
    """
    sid = np.asarray(sounding_id)
    if sid.dtype.kind == "f":          # combined files store sounding_id as float64
        rid = np.rint(sid)
        if not np.all(rid == sid):
            raise ValueError("non-integer sounding_id in tau file")
        sid = rid.astype(np.int64)
    sid = sid.astype(np.int64)

    l1b_id = l1b["sounding_id"]
    order = np.argsort(l1b_id, kind="stable")
    sorted_id = l1b_id[order]
    pos = np.searchsorted(sorted_id, sid)
    pos_c = np.clip(pos, 0, len(sorted_id) - 1)
    found = sorted_id[pos_c] == sid
    if not np.all(found):
        bad = sid[~found]
        raise ValueError(f"{bad.size} sounding(s) not found in L1B, e.g. {bad[:5].tolist()}")
    dup = (pos_c + 1 < len(sorted_id)) & (sorted_id[np.minimum(pos_c + 1, len(sorted_id) - 1)] == sid)
    if np.any(dup):
        raise ValueError(f"{int(dup.sum())} sounding id(s) appear more than once in the L1B")
    idx = order[pos_c]
    fp_l1b = l1b["fp_number"][idx]
    if not np.array_equal(fp_l1b, np.asarray(fp_number).astype(np.int64)):
        n_bad = int(np.sum(fp_l1b != np.asarray(fp_number)))
        raise ValueError(f"{n_bad} sounding(s) have a footprint number different from the L1B")
    return idx


# --------------------------------------------------------------------------- compute

_WORKER = {}


def default_workers():
    """CPUs this process may use (Slurm cgroup / affinity), at least 1."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    ntasks = int(os.environ.get("SLURM_NTASKS", 0) or 0)
    ncpu_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or 0)
    if ntasks > 0:
        n = min(n, ntasks * max(1, ncpu_task))
    return max(1, n)


def _band_setups(l1b, solar_h5_path):
    setups = []
    for iband in range(3):
        su = band_ils_solar_setup(iband, l1b["wvl_coef"], l1b["del_lambda_all"],
                                  l1b["rel_lresponse_all"], solar_h5_path)
        # identical to the per-band ``xx + wloco[:, np.newaxis]`` of the full step
        su["xx_wloco_fps"] = [su["xx_fps"][fp] + su["wloco_fps"][fp][:, np.newaxis]
                              for fp in range(8)]
        setups.append(su)
    return setups


def _toa_rows(setups, fp, sza, v_solar, v_inst, dist_au, solar_vinst):
    """TOA rows (3, n, 1016) for per-sounding scalar arrays."""
    convr = np.pi / 180.0
    n = len(fp)
    out = np.empty((3, n, 1016))
    for i in range(n):
        # Python floats, as in fp_abs_coeff.oco_fp_abs_all_bands' track_args
        solzen = float(sza[i])
        musolzen = 1.0 / np.cos(solzen * convr)
        f = int(fp[i])
        vs, vi, da = float(v_solar[i]), float(v_inst[i]), float(dist_au[i])
        for iband, su in enumerate(setups):
            out[iband, i] = toa_sol_track(
                su["wloco_fps"][f], su["xx_fps"][f], su["yy_fps"][f],
                su["sol_nu"], su["sol_abs_val"], su["sol_cont_val"],
                vs, vi, da, musolzen, solar_vinst,
                xx_wloco=su["xx_wloco_fps"][f])
    return out


def _worker_rows(bounds):
    lo, hi = bounds
    w = _WORKER
    return lo, _toa_rows(w["setups"], w["fp"][lo:hi], w["sza"][lo:hi], w["v_solar"][lo:hi],
                         w["v_inst"][lo:hi], w["dist_au"][lo:hi], w["solar_vinst"])


def compute_toa_sol(sounding_id, fp_number, sza, l1b, solar_h5_path=None,
                    solar_vinst=True, workers=None, setups=None):
    """Recompute the three TOA datasets for the given soundings.

    l1b         : load_l1b_inputs() dict
    solar_vinst : True  -> beta = (v_solar - v_inst)/c (v4 variant);
                  False -> beta = v_solar/c (v3 variant and production) -- for
                  validating the re-implementation against stored values.
    workers     : processes (fork) over sounding blocks; None -> default_workers().
                  ~0.16 s CPU per sounding (3 bands), so a 10k-sounding orbit
                  is ~27 min on one core.
    Returns {"<band>_toa_sol_output": (n, 1016) float64}.
    """
    if solar_h5_path is None:
        solar_h5_path = absco_input_dir() + "l2_solar_model.h5"
    idx = match_soundings(sounding_id, fp_number, l1b)
    if setups is None:
        setups = _band_setups(l1b, solar_h5_path)
    fp = np.asarray(fp_number).astype(np.int64)
    sza = np.asarray(sza)
    v_solar, v_inst, dist_au = (l1b[k][idx] for k in ("v_solar", "v_inst", "dist_au"))
    n = len(fp)
    workers = default_workers() if workers is None else max(1, int(workers))

    if workers <= 1 or n < 2 * workers:
        rows = _toa_rows(setups, fp, sza, v_solar, v_inst, dist_au, solar_vinst)
    else:
        global _WORKER
        _WORKER = dict(setups=setups, fp=fp, sza=sza, v_solar=v_solar, v_inst=v_inst,
                       dist_au=dist_au, solar_vinst=solar_vinst)
        block = max(1, min(200, n // (4 * workers)))
        bounds = [(lo, min(lo + block, n)) for lo in range(0, n, block)]
        rows = np.empty((3, n, 1016))
        try:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                for lo, part in pool.imap_unordered(_worker_rows, bounds):
                    rows[:, lo:lo + part.shape[1]] = part
        finally:
            _WORKER = {}
    return {name: rows[iband] for iband, name in enumerate(TOA_DATASETS)}


# --------------------------------------------------------------------------- files

def _check_input(path):
    stamp, td = tau_file_stamp(path)
    if not td:
        raise ValueError(f"{path}: production tau file (telluric_doppler=False); refusing")
    if stamp not in ACCEPTED_STAMPS:
        raise ValueError(f"{path}: unexpected tau_convolution {stamp!r}; "
                         f"expected one of {ACCEPTED_STAMPS}")
    with h5py.File(path, "r") as f:
        missing = [d for d in _TAU_DATASETS if d not in f]
    if missing:
        raise ValueError(f"{path}: missing tau datasets {missing}")
    return stamp


def recompute_toa_sol_file(path, l1b, overwrite_stamp=True, workers=None, solar_vinst=True,
                           write=True, solar_h5_path=None, setups=None, return_arrays=False):
    """Recompute the TOA datasets of one tau (or chunk) file; see recompute_toa_sol."""
    if write and not solar_vinst:
        raise ValueError("solar_vinst=False is for validation only (write=False)")
    t0 = time.time()
    old_stamp = _check_input(path)
    with h5py.File(path, "r") as f:
        sid = f["sounding_id"][...]
        fpn = f["fp_number"][...]
        sza = f["sza"][...]
        old = {name: f[name][...] for name in TOA_DATASETS}
        has_components = all(d in f for d in _COMPONENT_DATASETS)
    new = compute_toa_sol(sid, fpn, sza, l1b, solar_h5_path=solar_h5_path,
                          solar_vinst=solar_vinst, workers=workers, setups=setups)

    with np.errstate(divide="ignore", invalid="ignore"):
        dln = {name: np.log(new[name]) - np.log(old[name]) for name in TOA_DATASETS}
    summary = {
        "path": path, "n": int(len(sid)), "old_stamp": old_stamp,
        "has_components": has_components,
        "max_abs_dln": {name: float(np.nanmax(np.abs(d))) if d.size else 0.0
                        for name, d in dln.items()},
    }
    if return_arrays:
        summary.update(old=old, new=new, sounding_id=sid, fp_number=fpn, sza=sza)

    if write:
        with h5py.File(path, "r") as f:
            datasets, dataset_attrs = {}, {}
            for name in f:
                if not isinstance(f[name], h5py.Dataset):
                    raise ValueError(f"{path}: unexpected group {name!r}; refusing to rewrite")
                if name in TOA_DATASETS:
                    datasets[name] = new[name].astype(f[name].dtype, copy=False)
                else:
                    datasets[name] = f[name][...]
                if len(f[name].attrs):
                    dataset_attrs[name] = dict(f[name].attrs)
            use_ring = bool(f.attrs.get("use_ring", False))
            extra = {k: v for k, v in f.attrs.items() if k not in _CORE_ATTRS}
        new_stamp = TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER if overwrite_stamp else old_stamp
        extra["toa_sol_solar_doppler"] = "beta=(v_solar-v_inst)/c"
        if old_stamp != TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER:
            extra["toa_sol_recomputed_from"] = old_stamp
        extra["toa_sol_recomputed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_tau_file_atomic(path, datasets, use_ring=use_ring, telluric_doppler=True,
                               stamp=new_stamp, extra_attrs=extra, dataset_attrs=dataset_attrs)
        summary["new_stamp"] = new_stamp
    summary["seconds"] = time.time() - t0
    return summary


def recompute_toa_sol(tau_file, oco_l1b_file, overwrite_stamp=True, workers=None,
                      solar_vinst=True, write=True, include_chunks=True, skip_current=False,
                      solar_h5_path=None, return_arrays=False, verbose=True):
    """Upgrade a telluric-Doppler tau file (and its chunk files) to the v4 TOA.

    For ``tau_file`` (if it exists) and, with include_chunks, every
    ``<stem>_tmp_<i>.h5`` next to it: re-compute the three
    ``<band>_toa_sol_output`` datasets with beta = (v_solar - v_inst)/c and
    rewrite the file atomically (fp_atm._write_tau_file_atomic) with all other
    datasets and attrs preserved and the v4 stamp (overwrite_stamp=False keeps
    the input stamp).  Inputs must carry telluric_doppler=True and the v3 or v4
    variant stamp; every sounding must be found in the L1B with the same
    footprint number, otherwise ValueError and nothing is written for that
    file.

    solar_vinst=False, write=False : validation mode, reproduces the v3 numbers.
    skip_current : leave files already at the v4 stamp untouched.
    Returns a list of per-file summaries (path, n, max_abs_dln per band,
    seconds, ...; with return_arrays also old/new TOA arrays).
    """
    paths = [tau_file] if os.path.isfile(tau_file) else []
    if include_chunks:
        paths += chunk_files(tau_file)
    if not paths:
        raise FileNotFoundError(f"no tau file or chunk files for {tau_file}")
    if solar_h5_path is None:
        solar_h5_path = absco_input_dir() + "l2_solar_model.h5"

    todo = []
    for p in paths:
        stamp = _check_input(p)
        if skip_current and stamp == TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER:
            if verbose:
                print(f"[toa-only] {p}: already at {stamp}; skipped", flush=True)
            continue
        todo.append(p)
    if not todo:
        return []

    l1b = load_l1b_inputs(oco_l1b_file)
    # Validate the sounding match of every file before writing any of them.
    for p in todo:
        with h5py.File(p, "r") as f:
            try:
                match_soundings(f["sounding_id"][...], f["fp_number"][...], l1b)
            except ValueError as exc:
                raise ValueError(f"{p}: {exc}; nothing written") from None
    setups = _band_setups(l1b, solar_h5_path)
    summaries = []
    for p in todo:
        s = recompute_toa_sol_file(p, l1b, overwrite_stamp=overwrite_stamp, workers=workers,
                                   solar_vinst=solar_vinst, write=write,
                                   solar_h5_path=solar_h5_path, setups=setups,
                                   return_arrays=return_arrays)
        if verbose:
            dl = ", ".join(f"{k.split('_')[0]} {v:.2e}" for k, v in s["max_abs_dln"].items())
            print(f"[toa-only] {os.path.basename(p)}: {s['n']} soundings, "
                  f"{s['seconds']:.0f} s, max|dln toa| {dl}"
                  + (f", stamp -> {s['new_stamp']}" if write else " (not written)"), flush=True)
        summaries.append(s)
    return summaries

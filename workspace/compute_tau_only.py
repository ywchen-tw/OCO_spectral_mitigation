"""Compute the per-footprint optical depths (tau) for one date WITHOUT fitting.

Thin driver around ``spectral.fitting.preprocess`` so the tau step of the
production pipeline can be run on its own (for example to pre-compute tau for
a training set whose fits will be run later).  It discovers the date's orbits
under ``<storage>/data/OCO2/<YYYY>/<DOY>``, and for every orbit (or only the
``--orbit`` ones) computes ``<storage>/results/<date>/<orbit>/fp_tau_combined.h5``
unless a current one exists.  Same locking as fitting.py, so several jobs on
the same date split the orbits.  Nothing else is written: no cloud distances
are read, no fitting_details, no plots.

The output is the PRODUCTION tau (rest-wavelength ABSCO grid, version stamp
``transmission_ils_rayleigh_hr_v2``); pass ``--telluric-doppler`` together with
``--output-dir <storage>/results_doppler`` to compute the test-variant tau
instead (never mix the two in one tree).

``--toa-only`` (with ``--telluric-doppler`` and ``--output-dir``) upgrades
existing variant tau files from stamp ``transmission_ils_rayleigh_hr_v3_telluric_doppler``
to the current variant stamp by recomputing only the three
``<band>_toa_sol_output`` datasets (solar grid shifted by the spacecraft
velocity too; ``abs_util.toa_solar.recompute_toa_sol``), including any
``fp_tau_combined_tmp_*.h5`` chunk files.  No ABSCO convolution is run.  Orbits
without a tau file are reported (they need the full tau step; exit status 2).

Workers: ``oco_fp_abs_all_bands`` uses SLURM_NTASKS // 2 processes, so an
8-task Slurm allocation gives 4 tau workers.

Usage
-----
    python workspace/compute_tau_only.py --date 2020-01-01
    python workspace/compute_tau_only.py --date 2020-01-01 --orbit 29262a_ND
    python workspace/compute_tau_only.py --date 2020-05-21 --telluric-doppler \
        --output-dir <storage>/results_doppler --toa-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from spectral.fitting import (  # noqa: E402
    _fp_tau_path, _select_orbits, get_storage_dir, preprocess, validate_date)
from spectral.orbit_data import (  # noqa: E402
    _discover_orbit_files, fp_tau_file_is_current, search_oco2_orbit)
from spectral import orbit_lock  # noqa: E402


def run_toa_only(date, data_dir, result_dir, orbits=None, limit_granules=-1,
                 lock_stale_seconds=orbit_lock.DEFAULT_STALE_SECONDS, workers=None):
    """Upgrade the date's variant tau files to the current TOA (see module doc).

    Returns the exit status: 0 all done, 2 some orbit has no tau file or is
    locked by another job, 1 some orbit failed.
    """
    from abs_util.ils_tau import TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER as NEW_STAMP
    from abs_util.toa_solar import chunk_files, recompute_toa_sol, tau_file_stamp

    orbit_list = _select_orbits(search_oco2_orbit(date, data_dir=str(data_dir)),
                                orbits, limit_granules)
    if not orbit_list:
        raise FileNotFoundError(f"No OCO-2 orbit data found for {date} under {data_dir}")
    oco2_dir = f"{data_dir}/OCO2/{date.year}/{date.timetuple().tm_yday:03d}"

    upgraded, current, no_tau, busy, failed = [], [], [], [], []
    for orbit_id in orbit_list:
        tau_file = _fp_tau_path(result_dir, date, orbit_id)
        files = ([tau_file] if os.path.isfile(tau_file) else []) + chunk_files(tau_file)
        if not files:
            print(f"toa-only: {orbit_id}: no tau file -> needs the full tau step", flush=True)
            no_tau.append(orbit_id)
            continue
        if all(tau_file_stamp(f)[0] == NEW_STAMP for f in files):
            print(f"toa-only: {orbit_id}: already at {NEW_STAMP}; skipped", flush=True)
            current.append(orbit_id)
        else:
            with orbit_lock.held(tau_file, lock_stale_seconds) as got:
                if not got:
                    print(f"toa-only: {orbit_id}: locked by another job; skipped", flush=True)
                    busy.append(orbit_id)
                    continue
                t0 = time.time()
                try:
                    l1b = _discover_orbit_files(f"{oco2_dir}/{orbit_id}")["oco_l1b"]
                    summaries = recompute_toa_sol(tau_file, l1b, workers=workers,
                                                  skip_current=True)
                except (OSError, ValueError, KeyError) as exc:
                    print(f"toa-only: {orbit_id}: FAILED: {exc}", flush=True)
                    failed.append(orbit_id)
                    continue
                n = sum(s["n"] for s in summaries)
                print(f"toa-only: {orbit_id}: {len(summaries)} file(s), {n} soundings, "
                      f"{time.time() - t0:.0f} s", flush=True)
                upgraded.append(orbit_id)
        if not os.path.isfile(tau_file):
            # only chunk files: they are now current, but the combined file
            # still has to come from the full tau step
            print(f"toa-only: {orbit_id}: chunk files only -> needs the full tau step", flush=True)
            no_tau.append(orbit_id)

    print(f"toa-only: upgraded {len(upgraded)}, already current {len(current)}, "
          f"no tau file {len(no_tau)}, locked {len(busy)}, failed {len(failed)} "
          f"(of {len(orbit_list)} orbits)", flush=True)
    if no_tau:
        print(f"toa-only: need the full tau step: {no_tau}", flush=True)
    if busy:
        print(f"toa-only: locked by another job: {busy}", flush=True)
    if failed:
        print(f"toa-only: failed: {failed}", flush=True)
        return 1
    return 2 if (no_tau or busy) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--data-dir", default=None, help="default <storage>/data")
    ap.add_argument("--output-dir", default=None, help="tau tree, default <storage>/results (production)")
    ap.add_argument("--orbit", action="append", default=None, metavar="ORBIT_ID",
                    help="restrict to these orbit ids (repeatable)")
    ap.add_argument("--limit-granules", type=int, default=-1)
    ap.add_argument("--lock-stale-hours", type=float, default=4.0)
    ap.add_argument("--telluric-doppler", action="store_true",
                    help="test-variant tau; requires --output-dir outside the production tree")
    ap.add_argument("--toa-only", action="store_true",
                    help="only recompute <band>_toa_sol_output of existing variant tau files "
                         "(v3 -> current stamp); requires --telluric-doppler and --output-dir")
    ap.add_argument("--workers", type=int, default=None,
                    help="--toa-only processes (default: CPUs of this job)")
    args = ap.parse_args(argv)

    date = validate_date(args.date)
    storage = get_storage_dir()
    data_dir = Path(args.data_dir) if args.data_dir else storage / "data"
    result_dir = Path(args.output_dir) if args.output_dir else storage / "results"
    if args.telluric_doppler and result_dir.resolve() == (storage / "results").resolve():
        ap.error("--telluric-doppler must not write into the production tree; give --output-dir")
    if args.toa_only:
        if not args.telluric_doppler or args.output_dir is None:
            ap.error("--toa-only requires --telluric-doppler and --output-dir")
        print(f"toa-only: date {date:%Y-%m-%d}, data {data_dir}, tau tree {result_dir}, "
              f"orbits={args.orbit or 'all'}", flush=True)
        t0 = time.time()
        status = run_toa_only(date, data_dir, result_dir, orbits=args.orbit,
                              limit_granules=args.limit_granules,
                              lock_stale_seconds=args.lock_stale_hours * 3600.0,
                              workers=args.workers)
        print(f"toa-only: done in {(time.time() - t0) / 60.0:.1f} min", flush=True)
        return status

    print(f"tau-only: date {date:%Y-%m-%d}, data {data_dir}, tau tree {result_dir}, "
          f"telluric_doppler={args.telluric_doppler}, orbits={args.orbit or 'all'}", flush=True)
    t0 = time.time()
    sat0 = preprocess(date, data_dir=str(data_dir), result_dir=str(result_dir),
                      limit_granules=args.limit_granules,
                      lock_stale_seconds=args.lock_stale_hours * 3600.0,
                      orbits=args.orbit, telluric_doppler=args.telluric_doppler)
    dt_min = (time.time() - t0) / 60.0

    # Report: which orbits now have a current tau, which are still missing
    # (held by another job, or failed).
    done, missing = [], []
    for o in sat0["orbit_list"]:
        f = sat0[o]["fp_tau_file"]
        (done if fp_tau_file_is_current(f, telluric_doppler=args.telluric_doppler) else missing).append(o)
    print(f"tau-only: {len(done)}/{len(sat0['orbit_list'])} orbits have a current tau after {dt_min:.1f} min", flush=True)
    if missing:
        print(f"tau-only: not current (another job may still hold them): {missing}", flush=True)
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())

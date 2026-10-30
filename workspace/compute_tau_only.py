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

Workers: ``oco_fp_abs_all_bands`` uses SLURM_NTASKS // 2 processes, so an
8-task Slurm allocation gives 4 tau workers.

Usage
-----
    python workspace/compute_tau_only.py --date 2020-01-01
    python workspace/compute_tau_only.py --date 2020-01-01 --orbit 29262a_ND
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from spectral.fitting import get_storage_dir, preprocess, validate_date  # noqa: E402
from spectral.orbit_data import fp_tau_file_is_current  # noqa: E402


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
    args = ap.parse_args(argv)

    date = validate_date(args.date)
    storage = get_storage_dir()
    data_dir = Path(args.data_dir) if args.data_dir else storage / "data"
    result_dir = Path(args.output_dir) if args.output_dir else storage / "results"
    if args.telluric_doppler and result_dir.resolve() == (storage / "results").resolve():
        ap.error("--telluric-doppler must not write into the production tree; give --output-dir")

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

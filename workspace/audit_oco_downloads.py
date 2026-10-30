"""Audit the completeness of per-date OCO-2 downloads made by oco_modis_cloud_distance.py.

For every date given, under ``<data-dir>/OCO2/<YYYY>/<DOY>/``:

* the day's Lite file(s) ``oco2_LtCO2_*.nc4`` must exist; the unique
  ``Sounding/orbit`` values in the newest one give the orbits that have
  retrievals that day, i.e. the granules the fitting step will look for;
* every granule directory ``<orbit>a_<GL|ND|TG>`` must hold exactly one
  ``oco2_L1bSc*``, one ``oco2_L2Met*`` and one ``oco2_L2CPr*`` .h5 file, no
  leftover ``*.part`` partial download, and each .h5 must open with h5py and
  contain its signature group (``SoundingGeometry`` / ``Meteorology`` /
  ``CO2Prior``);
* ``sat_data_status.json`` (written at the end of Phase 2) is read for
  ``l1b_ready``, ``downloading_completed`` and ``failed_download_count``.  The
  file checks above are the authoritative test; the status flags are reported
  as WARNINGS only, because ``failed_download_count`` is one per-date number
  shared by every granule's status file and counts MODIS downloads too (an
  audit of the 92 production dates on 2026-10-30 found 97 granules flagged by
  it and zero with a missing, partial or unreadable OCO file).  A missing
  status file (Phase 2 still running, or an old download) is ``status=absent``.
* every orbit with retrievals in the Lite file must have a granule directory
  (``missing orbits``); extra directories without Lite retrievals are fine.

Optionally (``--logs-dir``), the Slurm logs ``sbatch-output_oco_cld_dist_*.txt``
and ``sbatch-output_oco_spectral_anal_*.txt`` whose first "Processing date:"
line matches are scanned for download retries, incomplete downloads, 503s and
the final "Successfully processed" / "Failed to process" line.

Usage
-----
    python workspace/audit_oco_downloads.py --dates 2020-01-01 2020-02-27
    python workspace/audit_oco_downloads.py --dates-file dates.txt --logs-dir . --out audit/

Exit status 0 when every date is complete, 1 otherwise.  Read-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd

PRODUCTS = {"L1B": ("oco2_L1bSc*.h5", "SoundingGeometry"),
            "Met": ("oco2_L2Met*.h5", "Meteorology"),
            "CPr": ("oco2_L2CPr*.h5", "CO2Prior")}
MODE_NAMES = {0: "ND", 1: "GL", 2: "TG", 3: "XS"}
LOG_PATTERNS = {
    "downloaded": re.compile(r"✓ Downloaded"),
    "retrying": re.compile(r"retrying in", re.I),
    "http_5xx": re.compile(r"returned 5\d\d"),
    "incomplete": re.compile(r"Incomplete download"),
    "redownload": re.compile(r"re-downloading", re.I),
    "failed_lines": re.compile(r"(?<!Successfully )\b(Failed|FAILED|Traceback)\b"),
}


def storage_root():
    return os.environ.get("OCO2_DATAROOT", "/scratch/alpine/yuch8913/oco/spec_anal")


def parse_date(s):
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"bad date {s!r}")


def h5_ok(path, group):
    try:
        with h5py.File(path, "r") as f:
            return group in f
    except OSError:
        return False


def lite_orbits(lite_path):
    """{orbit_number: mode} for the orbits that have retrievals in the Lite file."""
    try:
        with h5py.File(lite_path, "r") as f:
            orb = f["Sounding/orbit"][...].astype(int)
            mode = f["Sounding/operation_mode"][...].astype(int) if "Sounding/operation_mode" in f else None
    except (OSError, KeyError) as exc:
        return None, f"lite unreadable: {exc}"
    out = {}
    for o in np.unique(orb):
        m = np.unique(mode[orb == o]) if mode is not None else []
        out[int(o)] = "/".join(MODE_NAMES.get(int(x), str(x)) for x in m)
    return out, None


def audit_date(date, data_dir, open_h5=True):
    doy_dir = os.path.join(data_dir, "OCO2", f"{date:%Y}", f"{date.timetuple().tm_yday:03d}")
    rows = []
    summary = {"date": f"{date:%Y-%m-%d}", "doy_dir": doy_dir, "exists": os.path.isdir(doy_dir)}
    if not summary["exists"]:
        summary.update(n_lite=0, n_granule_dirs=0, n_complete=0, n_incomplete=0, n_expected=np.nan,
                       missing_orbits="", status_absent=0, complete=False, note="no DOY directory")
        return summary, rows

    lites = sorted(glob.glob(os.path.join(doy_dir, "oco2_LtCO2_*.nc4")))
    summary["n_lite"] = len(lites)
    expected, lite_err = (lite_orbits(lites[-1]) if lites else (None, "no Lite file"))
    summary["lite_file"] = os.path.basename(lites[-1]) if lites else ""

    gdirs = sorted(d for d in glob.glob(os.path.join(doy_dir, "*_??")) if os.path.isdir(d))
    present_orbits = {}
    for g in gdirs:
        name = os.path.basename(g)
        m = re.match(r"(\d+)a?_(\w\w)$", name)
        orbit = int(m.group(1)) if m else None
        present_orbits[orbit] = name
        row = {"date": summary["date"], "granule": name, "orbit": orbit}
        problems, warns = [], []
        for prod, (pat, group) in PRODUCTS.items():
            files = glob.glob(os.path.join(g, pat))
            row[f"n_{prod}"] = len(files)
            row[f"size_{prod}_MB"] = round(sum(os.path.getsize(f) for f in files) / 1e6, 1)
            if len(files) != 1:
                problems.append(f"{prod}x{len(files)}")
            elif open_h5:
                ok = h5_ok(files[0], group)
                row[f"h5ok_{prod}"] = ok
                if not ok:
                    problems.append(f"{prod} unreadable")
        parts = glob.glob(os.path.join(g, "*.part"))
        row["n_part"] = len(parts)
        if parts:
            problems.append(f"{len(parts)} .part")
        sp = os.path.join(g, "sat_data_status.json")
        if os.path.isfile(sp):
            try:
                st = json.load(open(sp))
                row["status"] = "complete" if st.get("downloading_completed") else "incomplete"
                row["status_l1b_ready"] = st.get("l1b_ready")
                row["status_failed_downloads"] = st.get("failed_download_count")
                row["status_time"] = st.get("download_timestamp", "")
                if st.get("l1b_ready") is False:
                    warns.append("status: l1b_ready False")
                elif not st.get("downloading_completed"):
                    warns.append("status: incomplete")
                if st.get("failed_download_count"):
                    warns.append(f"status: {st['failed_download_count']} failed downloads on this date (any product)")
            except (OSError, ValueError):
                row["status"] = "unreadable"
                warns.append("status json unreadable")
        else:
            row["status"] = "absent"
        if expected is not None and orbit is not None and orbit not in expected:
            row["in_lite"] = False
        else:
            row["in_lite"] = expected is not None
        row["complete"] = not problems
        row["problems"] = "; ".join(problems)
        row["warnings"] = "; ".join(warns)
        rows.append(row)

    missing = [] if expected is None else sorted(o for o in expected if o not in present_orbits)
    summary.update(
        n_granule_dirs=len(gdirs),
        n_complete=sum(r["complete"] for r in rows),
        n_incomplete=sum(not r["complete"] for r in rows),
        n_expected=len(expected) if expected is not None else np.nan,
        missing_orbits=" ".join(f"{o}({expected[o]})" for o in missing),
        status_absent=sum(r.get("status") == "absent" for r in rows),
        n_status_warn=sum(bool(r.get("warnings")) for r in rows),
        note=lite_err or "",
    )
    summary["complete"] = (summary["n_lite"] > 0 and summary["n_granule_dirs"] > 0
                           and summary["n_incomplete"] == 0 and not missing and not lite_err)
    return summary, rows


def scan_logs(logs_dir, dates_iso):
    """Per date: counts of download events in matching Slurm logs."""
    out = {}
    pats = ["sbatch-output_oco_cld_dist_*.txt", "sbatch-output_oco_spectral_anal_*.txt"]
    files = sorted({f for p in pats for f in glob.glob(os.path.join(logs_dir, p))}, key=os.path.getmtime)
    for f in files:
        try:
            with open(f, errors="replace") as fh:
                head = "".join(fh.readline() for _ in range(5))
                m = re.search(r"Processing date:\s*(\d{4}-\d{2}-\d{2})", head)
                if not m or m.group(1) not in dates_iso:
                    continue
                text = head + fh.read()
        except OSError:
            continue
        d = m.group(1)
        rec = {"log": os.path.basename(f), "log_mtime": dt.datetime.fromtimestamp(os.path.getmtime(f)).isoformat(timespec="minutes")}
        for k, p in LOG_PATTERNS.items():
            rec[k] = len(p.findall(text))
        if "Successfully processed" in text:
            rec["outcome"] = "success"
        elif "Failed to process" in text:
            rec["outcome"] = "failed"
        elif "Pipeline Complete" in text:
            rec["outcome"] = "complete"
        else:
            rec["outcome"] = "running/unknown"
        out[d] = rec            # newest log for the date wins (sorted by mtime)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="*", default=[])
    ap.add_argument("--dates-file", help="one date per line (YYYY-MM-DD or YYYYMMDD)")
    ap.add_argument("--data-dir", default=os.path.join(storage_root(), "data"))
    ap.add_argument("--logs-dir", default=None, help="directory with sbatch-output_*.txt logs to scan")
    ap.add_argument("--out", default=None, help="write audit_granules.csv and audit_dates.csv here")
    ap.add_argument("--no-open-h5", action="store_true", help="skip opening each .h5 (faster)")
    ap.add_argument("--quiet", action="store_true", help="only print dates with problems")
    ap.add_argument("--verbose", action="store_true", help="also print status-file warnings per granule")
    args = ap.parse_args(argv)

    dates = [parse_date(d) for d in args.dates]
    if args.dates_file:
        dates += [parse_date(l) for l in open(args.dates_file) if l.strip() and not l.startswith("#")]
    if not dates:
        ap.error("no dates given")
    dates = sorted(set(dates))
    dates_iso = {f"{d:%Y-%m-%d}" for d in dates}

    summaries, granules = [], []
    for d in dates:
        s, rows = audit_date(d, args.data_dir, open_h5=not args.no_open_h5)
        summaries.append(s)
        granules += rows
    logs = scan_logs(args.logs_dir, dates_iso) if args.logs_dir else {}
    for s in summaries:
        s.update({f"log_{k}": v for k, v in logs.get(s["date"], {}).items()})

    sdf = pd.DataFrame(summaries)
    gdf = pd.DataFrame(granules)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        sdf.to_csv(os.path.join(args.out, "audit_dates.csv"), index=False)
        gdf.to_csv(os.path.join(args.out, "audit_granules.csv"), index=False)

    n_ok = int(sdf["complete"].sum())
    print(f"{n_ok}/{len(sdf)} dates complete; {int(gdf['complete'].sum()) if len(gdf) else 0}/{len(gdf)} granules complete")
    for s in summaries:
        if args.quiet and s["complete"]:
            continue
        flag = "OK  " if s["complete"] else "BAD "
        logtxt = ""
        if s.get("log_log"):
            logtxt = (f" | log: {s['log_outcome']}, downloaded {s['log_downloaded']}, retries {s['log_retrying']}, "
                      f"5xx {s['log_http_5xx']}, incomplete {s['log_incomplete']}, redownload {s['log_redownload']}")
        exp = "" if pd.isna(s.get("n_expected", np.nan)) else f"/{int(s['n_expected'])} in Lite"
        print(f"{flag}{s['date']}: lite {s.get('n_lite', 0)}, granule dirs {s.get('n_granule_dirs', 0)}{exp}, "
              f"complete {s.get('n_complete', 0)}, incomplete {s.get('n_incomplete', 0)}, "
              f"status absent {s.get('status_absent', 0)}, status warnings {s.get('n_status_warn', 0)}"
              + (f", missing orbits: {s['missing_orbits']}" if s.get("missing_orbits") else "")
              + (f", note: {s['note']}" if s.get("note") else "") + logtxt)
        for r in granules:
            if r["date"] == s["date"] and not r["complete"]:
                print(f"      {r['granule']}: {r['problems']}")
        if args.verbose:
            for r in granules:
                if r["date"] == s["date"] and r["complete"] and r.get("warnings"):
                    print(f"      {r['granule']}: files OK; {r['warnings']}")
    return 0 if n_ok == len(sdf) else 1


if __name__ == "__main__":
    sys.exit(main())

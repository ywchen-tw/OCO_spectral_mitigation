#!/usr/bin/env python
"""Merge ATom NOAA-Picarro CO2 with ATom navigation, and segment vertical profiles.

Stage 1 of the ATom ocean-glint pseudo-column comparison (PROJECT_REVIEW M7-3, item 3).

This script resolves the two blockers found while inspecting the raw download:
  1. The NOAA-Picarro `.ict` files carry only ``UTC_Start, CO2, CH4, CO`` -- no
     position. This joins them to the ATom flight-track nav (lat/lon/alt + *measured*
     static pressure) on seconds-since-midnight UTC, which both products share.
  2. ATom "porpoises" continuously between ~0.2 and ~12 km. We segment each flight
     into individual ascending/descending legs -> one leg == one pseudo-column site.

Output per flight (parquet, in ``$OUT/atom_merged/``):
    time_utc_s, date, lat, lon, alt_m, p_hpa, t_k, co2_ppm, profile_id, leg_dir

It deliberately stops *before* the hard half (stratospheric extension above the
aircraft ceiling, pressure-weighting, and OCO-2 averaging-kernel application). Those
come next; this gives clean profiles-on-a-pressure-grid to eyeball first.

Usage:
    python merge_atom_profiles.py                 # all flights
    python merge_atom_profiles.py --date 20171001 # one flight
    python merge_atom_profiles.py --plot          # also drop a per-flight profile PNG
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# --- paths -------------------------------------------------------------------
# Outputs live under results/model_comparison (like the TCCON comparison), namespaced
# by the DE MODEL_TAG. Merged ATom profiles (this script's product) go in atom_merged/.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_ROOT = os.path.join(REPO, "data", "Other")
PICARRO_DIR = os.path.join(DATA_ROOT, "ATom_Picarro_Instrument_Data_1732", "data")
NAV_DIR = os.path.join(DATA_ROOT, "ATom_nav_1613", "data")
TAG = "de_beta_nll_prof_reg_o05l15_m5"
OUT_BASE = os.path.join(REPO, "results", "model_comparison", "deep_ensemble", TAG, "atom")
OUT_DIR = os.path.join(OUT_BASE, "atom_merged")   # merged profiles + profile plots

FILL = -99999.0  # ICARTT / nav missing-data sentinel

# --- profile segmentation knobs ---------------------------------------------
MIN_LEG_SPAN_M = 3000.0    # a leg must span >= this vertical extent to count as a profile
SMOOTH_WIN_S = 60          # rolling-median window (seconds) to de-noise altitude before peak find
PEAK_PROMINENCE_M = 2000.0 # altitude turning points must be this prominent


@dataclass
class Flight:
    date: str            # YYYYMMDD
    picarro_path: str


# -----------------------------------------------------------------------------
# Parsers
# -----------------------------------------------------------------------------
def parse_picarro(path: str) -> pd.DataFrame:
    """Read an ICARTT (FFI 1001) NOAA-Picarro file -> DataFrame[time_utc_s, co2_ppm].

    Header length is declared on line 1 as ``<n_header>,1001`` and varies per file,
    so we read it dynamically rather than hard-coding an offset.
    """
    with open(path) as fh:
        n_header = int(fh.readline().split(",")[0])
    # Column names live on the last header line (1-indexed line n_header).
    df = pd.read_csv(path, skiprows=n_header - 1)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "time_utc_s": df["UTC_Start"].astype(float),
            "co2_ppm": df["CO2_NOAA"].astype(float),
        }
    )
    out = out[out["co2_ppm"] != FILL].reset_index(drop=True)
    return out


def load_nav() -> pd.DataFrame:
    """Concatenate the four ATom deployment flight-track CSVs.

    Columns (per the file header block):
        index, RF, YYYYMMDD, UTC_Start, UTC_Stop, Latitude, Longitude,
        Altitude(m masl GPS), P(hPa), T(K), CumDist(km)
    Data rows start with the deployment tag (A1_/A2_/...); comment lines start '#'.
    """
    names = [
        "index", "rf", "date", "utc_start", "utc_stop",
        "lat", "lon", "alt_m", "p_hpa", "t_k", "cumdist_km",
    ]
    frames = []
    for csv in sorted(glob.glob(os.path.join(NAV_DIR, "ATom*_flight_tracks.csv"))):
        d = pd.read_csv(csv, comment="#", header=None, names=names)
        frames.append(d)
    nav = pd.concat(frames, ignore_index=True)
    nav["date"] = nav["date"].astype(int).astype(str)
    # Time coordinate for a 10-s interval = its midpoint.
    nav["time_utc_s"] = (nav["utc_start"].astype(float) + nav["utc_stop"].astype(float)) / 2.0
    for c in ("lat", "lon", "alt_m", "p_hpa", "t_k"):
        nav.loc[nav[c] == FILL, c] = np.nan
    return nav


# -----------------------------------------------------------------------------
# Merge + segment
# -----------------------------------------------------------------------------
def merge_flight(pic: pd.DataFrame, nav_day: pd.DataFrame) -> pd.DataFrame:
    """Interpolate nav lat/lon/alt/P/T onto the 1-Hz Picarro CO2 timestamps."""
    nav_day = nav_day.sort_values("time_utc_s")
    t_nav = nav_day["time_utc_s"].to_numpy()
    out = pic.copy()
    for c in ("lat", "lon", "alt_m", "p_hpa", "t_k"):
        out[c] = np.interp(out["time_utc_s"], t_nav, nav_day[c].to_numpy(),
                           left=np.nan, right=np.nan)
    # Restrict to the interval actually covered by nav (interp clamps outside).
    out = out[(out["time_utc_s"] >= t_nav.min()) & (out["time_utc_s"] <= t_nav.max())]
    out = out.dropna(subset=["lat", "lon", "alt_m", "p_hpa"]).reset_index(drop=True)
    return out


def segment_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Label ascending/descending legs. Adds ``profile_id`` and ``leg_dir`` columns.

    Turning points = prominent local maxima/minima of a median-smoothed altitude
    series. Each span between consecutive turning points is one leg; legs whose
    vertical extent is below MIN_LEG_SPAN_M (level flight, taxi) get profile_id = -1.
    """
    df = df.sort_values("time_utc_s").reset_index(drop=True)
    alt = df["alt_m"].to_numpy()
    # Median smooth on a per-second grid (data is ~1 Hz but may have gaps; use index window).
    win = max(3, SMOOTH_WIN_S)
    alt_s = pd.Series(alt).rolling(win, center=True, min_periods=1).median().to_numpy()

    hi, _ = find_peaks(alt_s, prominence=PEAK_PROMINENCE_M)
    lo, _ = find_peaks(-alt_s, prominence=PEAK_PROMINENCE_M)
    turns = np.array(sorted(set([0, len(df) - 1]) | set(hi.tolist()) | set(lo.tolist())))

    profile_id = np.full(len(df), -1, dtype=int)
    leg_dir = np.array([""] * len(df), dtype=object)
    pid = 0
    for a, b in zip(turns[:-1], turns[1:]):
        seg = slice(a, b + 1)
        span = np.nanmax(alt_s[seg]) - np.nanmin(alt_s[seg])
        if span < MIN_LEG_SPAN_M:
            continue
        profile_id[seg] = pid
        leg_dir[seg] = "asc" if alt_s[b] > alt_s[a] else "desc"
        pid += 1
    df["profile_id"] = profile_id
    df["leg_dir"] = leg_dir
    return df


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def discover_flights(date_filter: str | None) -> list[Flight]:
    flights = []
    for p in sorted(glob.glob(os.path.join(PICARRO_DIR, "*.ict"))):
        m = re.search(r"_(\d{8})_", os.path.basename(p))
        if not m:
            continue
        date = m.group(1)
        if date_filter and date != date_filter:
            continue
        flights.append(Flight(date=date, picarro_path=p))
    return flights


def plot_flight(df: pd.DataFrame, date: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 7))
    prof = df[df["profile_id"] >= 0]
    for pid, g in prof.groupby("profile_id"):
        ax.plot(g["co2_ppm"], g["alt_m"] / 1000.0, lw=0.8,
                label=f"{pid}:{g['leg_dir'].iloc[0]}")
    ax.set_xlabel("CO2 (ppm)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"ATom {date}  ({prof['profile_id'].nunique()} profiles)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"profiles_{date}.png"), dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="single flight YYYYMMDD (default: all)")
    ap.add_argument("--plot", action="store_true", help="also save per-flight profile PNG")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    nav = load_nav()
    flights = discover_flights(args.date)
    if not flights:
        raise SystemExit("No matching Picarro flights found.")

    summary = []
    for fl in flights:
        pic = parse_picarro(fl.picarro_path)
        if pic.empty:
            summary.append((fl.date, 0, 0, "no valid CO2 (flagged flight)"))
            continue
        nav_day = nav[nav["date"] == fl.date]
        if nav_day.empty:
            summary.append((fl.date, len(pic), 0, "no nav rows for this date"))
            continue
        merged = merge_flight(pic, nav_day)
        if merged.empty:
            summary.append((fl.date, len(pic), 0, "no time overlap after merge"))
            continue
        seg = segment_profiles(merged)
        n_prof = int((seg["profile_id"] >= 0).any() and seg["profile_id"].max() + 1 or 0)
        out_path = os.path.join(OUT_DIR, f"atom_merged_{fl.date}.parquet")
        seg.to_parquet(out_path, index=False)
        summary.append((fl.date, len(seg), n_prof, os.path.basename(out_path)))
        if args.plot:
            plot_flight(seg, fl.date)

    print(f"{'date':>10} {'rows':>8} {'profiles':>9}  note")
    print("-" * 60)
    for date, rows, nprof, note in summary:
        print(f"{date:>10} {rows:>8} {nprof:>9}  {note}")
    tot_rows = sum(s[1] for s in summary)
    tot_prof = sum(s[2] for s in summary)
    print("-" * 60)
    print(f"{'TOTAL':>10} {tot_rows:>8} {tot_prof:>9}  files in {OUT_DIR}")


if __name__ == "__main__":
    main()

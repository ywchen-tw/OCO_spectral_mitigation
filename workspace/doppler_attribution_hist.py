"""Histograms of the per-channel tau attribution written by the telluric-Doppler variant.

``fitting.py --telluric-doppler`` stores, next to ``<band>_tau_output`` in
``<dop-dir>/<date>/<orbit>/fp_tau_combined.h5``, three (N, 1016) float32
attribution datasets per band (src/abs_util/ils_tau.py
``attribute_optical_depth``; names from fp_abs_coeff.COMPONENT_NAMES):

* ``<band>_frac_h2o``  — share of the channel's ILS-effective tau from H2O,
* ``<band>_frac_ray``  — share from Rayleigh,
* ``<band>_p_abs_hpa`` — pressure centroid of the gas absorption (hPa),

each weighted by w = ILS * exp(-tau_hr), NaN where the channel has no
absorption.  For every orbit of every date this script streams the nine
datasets (one band at a time, never concatenating orbits) and accumulates,
per band and land/ocean, histograms of

* per-channel pooled values (every finite channel of every sounding), and
* per-sounding channel medians (nanmedian over the sounding's finite channels),

with frac_h2o and frac_ray on 0–1 (50 bins) and p_abs_hpa on 0–1100 hPa (55 bins).
Finiteness is applied per quantity (p_abs_hpa can be NaN where the fractions are
finite).  All 1016 channels are used, including the two edge channels the
cumulant fit drops.  Values outside the histogram range are counted as
under/overflow in the counts CSV.

Surface class
-------------
``sfc_type`` in fitting_details is the Lite ``Retrieval/surface_type`` copied
verbatim (src/spectral/orbit_data.py); its Lite definition is "0=ocean (scaled
Cox-Munk); 1=land (scaled soil BRDF)", and src/ uses it the same way
(profile_pca.SURFACE_NAMES = {0: 'ocean', 1: 'land'}).  It is used when finite.
Soundings with no Lite match have sfc_type NaN; for those the L1B
``SoundingGeometry/sounding_land_fraction`` is used if the L1B is found under
``--data-dir`` (the dataset is in percent, so land = land fraction > 50 %).
Soundings without either are counted as "unknown" and excluded.  QF0 soundings
always have a Lite match, so the QF0 histograms use sfc_type only.

Counts are reported with and without the QF0 cut (``xco2_qf == 0`` from the
fit file); the figure and the markdown medians use QF0.  Soundings are joined
tau ``sounding_id`` <-> fit ``fp_id`` by id (row order not assumed).

Pooled-channel medians in the markdown are interpolated from a fine internal
histogram (1000 bins for fractions, 1100 for pressure, i.e. 0.001 / 1 hPa
resolution); per-sounding-median medians are exact.

Usage
-----
    python workspace/doppler_attribution_hist.py --dates 2020-02-27 \\
        --dop-dir $OCO2_DATAROOT/results_doppler --out <analysis dir>

Outputs in ``--out``: ``doppler_attribution_hist.csv`` (histogram counts, long
format), ``doppler_attribution_counts.csv`` (sounding/channel counts incl.
under/overflow), ``doppler_attribution_hist.png`` and
``doppler_attribution_hist.md``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doppler_rms_vs_velocity import (  # noqa: E402
    find_l1b, fit_path, join_ids, load_l1b_geometry, md_table, storage_root)

BANDS = ("o2a", "wco2", "sco2")
QUANTITIES = ("frac_h2o", "frac_ray", "p_abs_hpa")
# (range, output bins, fine bins for medians)
HIST = {"frac_h2o": ((0.0, 1.0), 50, 1000),
        "frac_ray": ((0.0, 1.0), 50, 1000),
        "p_abs_hpa": ((0.0, 1100.0), 55, 1100)}
UNITS = {"frac_h2o": "frac_h2o (fraction of τ)", "frac_ray": "frac_ray (fraction of τ)",
         "p_abs_hpa": "p_abs (hPa)"}
SURFACES = ("land", "ocean")
SUBSETS = ("all", "qf0")
KINDS = ("channel", "sounding_median")
LAND_FRACTION_PCT = 50.0
C_LAND, C_OCEAN = "#E69F00", "#0072B2"   # Okabe-Ito


def fine_edges(q):
    (lo, hi), _, nf = HIST[q]
    return np.linspace(lo, hi, nf + 1)


class Accumulator:
    """Fine-binned counts per (band, quantity, surface, subset, kind) plus under/overflow and N."""

    def __init__(self):
        self.counts, self.under, self.over, self.n = {}, {}, {}, {}
        self.n_snd = {}
        self.snd_medians = {}   # QF0 per-sounding medians, kept for exact medians (1 float per sounding)

    def add(self, key, values):
        q = key[1]
        lo, hi = HIST[q][0]
        h, _ = np.histogram(values, bins=fine_edges(q))
        self.counts[key] = self.counts.get(key, 0) + h
        self.under[key] = self.under.get(key, 0) + int((values < lo).sum())
        self.over[key] = self.over.get(key, 0) + int((values > hi).sum())
        self.n[key] = self.n.get(key, 0) + int(values.size)


def surface_class(sfc_type, land_fraction_pct):
    """'land' / 'ocean' / 'unknown' per sounding and the source used ('sfc_type' / 'l1b' / 'none')."""
    sfc = np.full(len(sfc_type), "unknown", dtype=object)
    src = np.full(len(sfc_type), "none", dtype=object)
    st = np.asarray(sfc_type, float)
    m = st == 1
    sfc[m], src[m] = "land", "sfc_type"
    m = st == 0
    sfc[m], src[m] = "ocean", "sfc_type"
    if land_fraction_pct is not None:
        lf = np.asarray(land_fraction_pct, float)
        need = (src == "none") & np.isfinite(lf)
        sfc[need & (lf > LAND_FRACTION_PCT)] = "land"
        sfc[need & (lf <= LAND_FRACTION_PCT)] = "ocean"
        src[need] = "l1b"
    return sfc, src


def orbits_for(dates, dop_dir):
    out = []
    for date in dates:
        for p in sorted(glob.glob(os.path.join(dop_dir, date, "*", "fp_tau_combined.h5"))):
            out.append((date, os.path.basename(os.path.dirname(p)), p))
    return out


def load_meta(date, orbit, dop_dir, prod_dir):
    """fp_id, sfc_type, xco2_qf from the variant fit file (production fit file as fallback)."""
    for tree in (dop_dir, prod_dir):
        if tree is None:
            continue
        p = fit_path(tree, date, orbit)
        if os.path.isfile(p):
            with h5py.File(p, "r") as f:
                return {k: f[k][...] for k in ("fp_id", "sfc_type", "xco2_qf")}, p
    raise FileNotFoundError(f"no fitting_details for {date} {orbit} in {dop_dir}" +
                            (f" or {prod_dir}" if prod_dir else ""))


def process_orbit(date, orbit, tau_file, acc, args):
    meta, meta_file = load_meta(date, orbit, args.dop_dir, args.prod_dir)
    with h5py.File(tau_file, "r") as f:
        names = [f"{b}_{q}" for b in BANDS for q in QUANTITIES]
        absent = [n for n in names if n not in f]
        if absent:
            raise KeyError(f"{tau_file}: no attribution datasets ({', '.join(absent[:3])}...)")
        sid = f["sounding_id"][...]
        it, im = join_ids(sid, meta["fp_id"])
        n_tau = len(sid)
        sfc_type = np.full(n_tau, np.nan)
        qf = np.full(n_tau, np.nan)
        sfc_type[it] = meta["sfc_type"][im]
        qf[it] = meta["xco2_qf"][im]

        lf = None
        if np.isnan(sfc_type).any() and args.data_dir:
            l1b_file = find_l1b(args.data_dir, date, orbit)
            if l1b_file is not None:
                g = load_l1b_geometry(l1b_file, land_fraction=True)
                lf = np.full(n_tau, np.nan)
                a, b = join_ids(sid, g["sounding_id"])
                lf[a] = g["land_fraction_pct"][b]
        sfc, src = surface_class(sfc_type, lf)
        stats = {"date": date, "orbit": orbit, "n_tau": n_tau, "n_meta_matched": len(it),
                 "meta_file": os.path.basename(meta_file),
                 "n_land": int((sfc == "land").sum()), "n_ocean": int((sfc == "ocean").sum()),
                 "n_unknown": int((sfc == "unknown").sum()),
                 "n_sfc_from_l1b": int((src == "l1b").sum()), "n_qf0": int((qf == 0).sum())}

        masks = {}
        for s in SURFACES:
            masks[(s, "all")] = sfc == s
            masks[(s, "qf0")] = (sfc == s) & (qf == 0)
        for b in BANDS:
            for q in QUANTITIES:
                arr = f[f"{b}_{q}"][...]                 # (N, 1016) float32, one at a time
                fin = np.isfinite(arr)
                has = fin.any(axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(arr, axis=1)
                for (s, sub), m in masks.items():
                    key = (b, q, s, sub)
                    acc.add(key + ("channel",), arr[m][fin[m]])
                    acc.add(key + ("sounding_median",), med[m & has])
                    acc.n_snd[key] = acc.n_snd.get(key, 0) + int((m & has).sum())
                    if sub == "qf0":
                        acc.snd_medians.setdefault(key, []).append(med[m & has].astype(np.float32))
                del arr, fin
    return stats


def hist_median(counts, edges):
    c = np.cumsum(counts)
    if c[-1] == 0:
        return np.nan
    half = 0.5 * c[-1]
    i = int(np.searchsorted(c, half))
    prev = c[i - 1] if i > 0 else 0
    frac = (half - prev) / counts[i] if counts[i] else 0.0
    return float(edges[i] + frac * (edges[i + 1] - edges[i]))


def coarse(counts, q):
    _, nb, nf = HIST[q]
    return np.asarray(counts).reshape(nb, nf // nb).sum(axis=1)


def coarse_edges(q):
    (lo, hi), nb, _ = HIST[q]
    return np.linspace(lo, hi, nb + 1)


def write_outputs(acc, orbit_stats, args):
    rows, crow = [], []
    for key, cnt in acc.counts.items():
        b, q, s, sub, kind = key
        e = coarse_edges(q)
        cc = coarse(cnt, q)
        rows += [{"band": b, "quantity": q, "surface": s, "subset": sub, "kind": kind,
                  "bin_lo": e[i], "bin_hi": e[i + 1], "count": int(cc[i])} for i in range(len(cc))]
        crow.append({"band": b, "quantity": q, "surface": s, "subset": sub, "kind": kind,
                     "n_values": acc.n[key], "n_soundings": acc.n_snd[key[:4]],
                     "underflow": acc.under[key], "overflow": acc.over[key]})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "doppler_attribution_hist.csv"), index=False)
    counts = pd.DataFrame(crow).sort_values(["band", "quantity", "kind", "subset", "surface"])
    counts.to_csv(os.path.join(args.out, "doppler_attribution_counts.csv"), index=False)

    d0, d1 = min(args.dates), max(args.dates)
    drange = d0 if d0 == d1 else f"{d0} to {d1}"
    ost = pd.DataFrame(orbit_stats)
    # sounding counts for the title: QF0 soundings with >= 1 finite o2a frac_h2o channel
    n_land = acc.n_snd.get(("o2a", "frac_h2o", "land", "qf0"), 0)
    n_ocean = acc.n_snd.get(("o2a", "frac_h2o", "ocean", "qf0"), 0)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for i, q in enumerate(QUANTITIES):
        e = coarse_edges(q)
        for j, b in enumerate(BANDS):
            ax = axes[i, j]
            for s, col in (("land", C_LAND), ("ocean", C_OCEAN)):
                for kind, ls in (("channel", "-"), ("sounding_median", "--")):
                    key = (b, q, s, "qf0", kind)
                    if key not in acc.counts:
                        continue
                    cc = coarse(acc.counts[key], q).astype(float)
                    if cc.sum() == 0:
                        continue
                    dens = cc / (cc.sum() * np.diff(e))
                    lab = f"{s}, {'per channel' if kind == 'channel' else 'per-sounding median'}"
                    ax.stairs(dens, e, color=col, ls=ls, lw=1.3, label=lab)
            ax.set_xlabel(UNITS[q])
            ax.set_ylabel("probability density")
            if i == 0:
                ax.set_title(b.upper())
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Telluric-Doppler τ attribution, QF0, {drange}, {len(ost)} orbits, "
                 f"N = {n_land:,} land / {n_ocean:,} ocean soundings", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "doppler_attribution_hist.png"), dpi=130)
    plt.close(fig)

    med_rows = []
    for b in BANDS:
        for q in QUANTITIES:
            rec = {"band": b, "quantity": q}
            for s in SURFACES:
                key = (b, q, s, "qf0")
                cnt = acc.counts.get(key + ("channel",))
                rec[f"{s}_channel_median"] = hist_median(cnt, fine_edges(q)) if cnt is not None else np.nan
                parts = acc.snd_medians.get(key, [])
                v = np.concatenate(parts) if parts else np.array([])
                rec[f"{s}_snd_median_median"] = float(np.median(v)) if v.size else np.nan
                rec[f"{s}_n_channels"] = acc.n.get(key + ("channel",), 0)
                rec[f"{s}_n_soundings"] = acc.n_snd.get(key, 0)
            med_rows.append(rec)
    med = pd.DataFrame(med_rows)

    cnt_tbl = counts[(counts.kind == "channel")].pivot_table(
        index=["band", "quantity", "surface"], columns="subset",
        values=["n_soundings", "n_values"], aggfunc="sum").reset_index()
    cnt_tbl.columns = ["_".join(c).strip("_") for c in cnt_tbl.columns]
    tot = {k: int(ost[k].sum()) for k in ("n_tau", "n_meta_matched", "n_land", "n_ocean", "n_unknown",
                                          "n_sfc_from_l1b", "n_qf0")}
    lines = [f"# Telluric-Doppler τ attribution — {drange}, {len(ost)} orbits", "",
             f"Soundings in tau files: {tot['n_tau']:,}; matched to fit metadata: {tot['n_meta_matched']:,}; "
             f"land {tot['n_land']:,}, ocean {tot['n_ocean']:,}, unknown {tot['n_unknown']:,} "
             f"(surface from L1B land fraction for {tot['n_sfc_from_l1b']:,}); QF0 {tot['n_qf0']:,}.", "",
             "Surface: Lite sfc_type (0 ocean, 1 land), L1B land fraction > 50 % where sfc_type is missing. "
             "Channel medians interpolated from 0.001 / 1 hPa histograms; per-sounding values are the "
             "nanmedian over each sounding's finite channels.", "",
             "## Per-band medians, QF0, land vs ocean", "",
             md_table(med, ["band", "quantity", "land_channel_median", "ocean_channel_median",
                            "land_snd_median_median", "ocean_snd_median_median",
                            "land_n_soundings", "ocean_n_soundings"]), "",
             "## Counts with and without the QF0 cut (finite channels / soundings with >= 1 finite channel)", "",
             md_table(cnt_tbl, list(cnt_tbl.columns), fmt="{:.0f}"), ""]
    with open(os.path.join(args.out, "doppler_attribution_hist.md"), "w") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    root = storage_root()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--dop-dir", default=os.path.join(root, "results_doppler"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prod-dir", default=None,
                    help="fallback location of fitting_details for sfc_type/xco2_qf if the variant file is absent")
    ap.add_argument("--data-dir", default=os.path.join(root, "data"),
                    help="L1B root for the land-fraction fallback ('' to disable)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    acc, orbit_stats = Accumulator(), []
    for date, orbit, tau_file in orbits_for(args.dates, args.dop_dir):
        try:
            st = process_orbit(date, orbit, tau_file, acc, args)
        except (KeyError, OSError, ValueError) as exc:
            print(f"[skip] {date} {orbit}: {exc}", file=sys.stderr)
            continue
        orbit_stats.append(st)
        print(f"{date} {orbit}: {st['n_tau']} soundings, meta matched {st['n_meta_matched']}, "
              f"land {st['n_land']}, ocean {st['n_ocean']}, unknown {st['n_unknown']}, QF0 {st['n_qf0']}",
              flush=True)
    if not orbit_stats:
        print("no variant tau file with attribution datasets found", file=sys.stderr)
        return 1
    write_outputs(acc, orbit_stats, args)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Cumulant-fit residual rms against |sounding_relative_velocity|, production vs telluric-Doppler tau.

For every date given, every orbit that has ``fitting_details_<date>_<orbit>.h5``
in BOTH the production tree and the telluric-Doppler tree (written by
``fitting.py --telluric-doppler``) is joined by sounding id to the L1B
``SoundingGeometry/sounding_relative_velocity`` (m/s, converted to km/s).
Far-clear soundings (``cld_dist_km > 20`` and ``xco2_qf == 0``, from the
production fit file) are binned in |v| (0.5 km/s bins, 0 to ``--max-v``) and,
per band and bin, the median and 16/84th percentiles of the residual rms are
reported for the production side and for the variant's ``<band>_rms_prod_nosg``,
``<band>_rms_o2_nosg`` and ``<band>_rms_o3_nosg``.  Statistics are pooled over
all orbits of all dates; a per-orbit Spearman correlation of rms with |v| is
reported separately.

Which production dataset is compared (resolved 2026-10-29)
----------------------------------------------------------
The variant's ``<band>_rms_prod_nosg`` is written by
``spectral.cumulant_fit._extra_fit_chunk``: the production-order cumulant fit
(k1, k2 >= 0 bounds, BVLS fallback) of the RAW (no Savitzky-Golay) ln T against
the Doppler-corrected tau, channels [1:-1], residual rms
``sqrt(mean((ln T - model)^2))`` in ln T units (dimensionless).  Those extra
fields are produced only with ``--extra-fits`` / ``--telluric-doppler``; the
production fitting_details files hold no residual rms of the cumulant fit at all.

The only residual-rms datasets in production fitting_details are
``rms_rel_<band>``, copied verbatim from the OCO-2 Lite file
``Retrieval/rms_rel_<band>`` (src/spectral/orbit_data.py, fitting.py
output_dict).  Lite definition: "RMS of the L2 fit residuals in the <band> band,
relative to the continuum signal, times 100 [so it is in percent]".  That is the
ACOS full-physics radiance-fit residual, not the ln T residual of the cumulant
fit on this project's tau, so it is NOT the same quantity as
``<band>_rms_prod_nosg`` (different model, different units, and it does not
depend on the tau this project computes).

``--prod-rms auto`` (default) therefore uses ``<band>_rms_prod_nosg`` from the
production files only if every paired production file has it (i.e. production
was rerun with ``--extra-fits``); otherwise it falls back to ``rms_rel_<band>``,
which is the closest stored dataset.  The chosen dataset is named in every
output (CSV column ``prod_source``, figure axis label, markdown header).  When
the fallback is used the production curve is drawn on its own y axis (percent)
because the two quantities are not on the same scale.  No fit is recomputed
from tau here.

Joins
-----
* variant fit file <-> production fit file: inner join on ``fp_id`` (the tau
  file ``sounding_id``, stored row-for-row by fitting.py); row order is not
  assumed to match.
* fit files <-> L1B: inner join of ``fp_id`` with the flattened
  ``SoundingGeometry/sounding_id`` (frames x 8).  L1B is looked up at
  ``<data-dir>/OCO2/<YYYY>/<DOY>/<orbit>/oco2_L1bSc*.h5`` for the date's DOY,
  then DOY-1 and DOY+1.  Fill values (|v| > 100 km/s) become NaN.
Matched counts per orbit are printed and written to the per-orbit CSV.

Usage
-----
    python workspace/doppler_rms_vs_velocity.py --dates 2020-02-27 \\
        --prod-dir $OCO2_DATAROOT/results --dop-dir $OCO2_DATAROOT/results_doppler \\
        --out <analysis dir>

Outputs in ``--out``: ``doppler_rms_vs_velocity.csv`` (pooled bin table, long
format), ``doppler_rms_vs_velocity_per_orbit.csv`` (join counts and Spearman
rho per orbit/band/quantity), ``doppler_rms_vs_velocity.png``,
``doppler_rms_vs_velocity.md`` and, if under 200 MB in memory,
``doppler_rms_vs_velocity_soundings.parquet`` (far-clear soundings only;
``.npz`` if pyarrow is unavailable).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BANDS = ("o2a", "wco2", "sco2")
FAR_KM = 20.0
BIN_KMS = 0.5
DOP_QUANTITIES = ("rms_prod", "rms_o2", "rms_o3")
META_KEYS = ("fp_id", "fp_number", "lat", "lon", "cld_dist_km", "xco2_qf", "sfc_type")
SUMMARY_RANGES = ((0.0, 0.5), (4.5, 5.5))
PARQUET_LIMIT_BYTES = 200 * 1024**2
V_FILL_KMS = 100.0
# Okabe-Ito colour-blind-safe pair (production, corrected)
C_PROD, C_DOP = "#E69F00", "#0072B2"


def storage_root():
    return os.environ.get("OCO2_DATAROOT", "/scratch/alpine/yuch8913/oco/spec_anal")


def fit_path(tree, date, orbit):
    return os.path.join(tree, "fitting_details", f"fitting_details_{date}_{orbit}.h5")


def paired_orbits(dates, prod_dir, dop_dir):
    """[(date, orbit)] present in both fitting_details trees, and those missing in one of them."""
    pairs, missing = [], []
    for date in dates:
        pat = f"fitting_details_{date}_*.h5"
        prod = {os.path.basename(p)[len(f"fitting_details_{date}_"):-3]
                for p in glob.glob(os.path.join(prod_dir, "fitting_details", pat))}
        dop = {os.path.basename(p)[len(f"fitting_details_{date}_"):-3]
               for p in glob.glob(os.path.join(dop_dir, "fitting_details", pat))}
        pairs += [(date, o) for o in sorted(prod & dop)]
        missing += [(date, o, "dop") for o in sorted(prod - dop)]
        missing += [(date, o, "prod") for o in sorted(dop - prod)]
    return pairs, missing


def read_datasets(path, keys, required=()):
    out = {}
    with h5py.File(path, "r") as f:
        for k in keys:
            if k in f:
                out[k] = f[k][...]
            elif k in required:
                raise KeyError(f"{path}: missing {k}")
            else:
                out[k] = None
    return out


def resolve_prod_source(pairs, prod_dir, choice):
    """'rms_prod_nosg' or 'rms_rel' (see module docstring)."""
    if choice != "auto":
        return choice
    for date, orbit in pairs:
        with h5py.File(fit_path(prod_dir, date, orbit), "r") as f:
            if not all(f"{b}_rms_prod_nosg" in f for b in BANDS):
                return "rms_rel"
    return "rms_prod_nosg" if pairs else "rms_rel"


def prod_key(band, source):
    return f"{band}_rms_prod_nosg" if source == "rms_prod_nosg" else f"rms_rel_{band}"


def prod_label(source):
    if source == "rms_prod_nosg":
        return "production τ: <band>_rms_prod_nosg"
    return "production: rms_rel_<band> (Lite L2, %, right axis)"


def unique_ids(ids):
    """int64 ids and a boolean keep-mask that drops repeated ids (first occurrence kept)."""
    ids = np.asarray(ids).astype(np.int64)
    _, first = np.unique(ids, return_index=True)
    keep = np.zeros(len(ids), bool)
    keep[first] = True
    return ids, keep


def join_ids(a, b):
    """Inner join of two id arrays; returns (ia, ib) index arrays with a[ia] == b[ib], in a's order."""
    a, ka = unique_ids(a)
    b, kb = unique_ids(b)
    ra = np.flatnonzero(ka)
    rb = np.flatnonzero(kb)
    _, ia, ib = np.intersect1d(a[ra], b[rb], assume_unique=True, return_indices=True)
    ia, ib = ra[ia], rb[ib]
    order = np.argsort(ia)
    return ia[order], ib[order]


def find_l1b(data_dir, date, orbit):
    d0 = dt.datetime.strptime(date, "%Y-%m-%d")
    for off in (0, -1, 1):
        d = d0 + dt.timedelta(days=off)
        hits = sorted(glob.glob(os.path.join(data_dir, "OCO2", f"{d:%Y}", f"{d.timetuple().tm_yday:03d}",
                                             orbit, "oco2_L1bSc*.h5")))
        if hits:
            return hits[-1]
    return None


def load_l1b_geometry(path, land_fraction=False):
    """Flattened L1B sounding_id, relative velocity (km/s) and optionally land fraction (%)."""
    with h5py.File(path, "r") as f:
        g = f["SoundingGeometry"]
        sid = g["sounding_id"][...].ravel().astype(np.int64)
        v = g["sounding_relative_velocity"][...].ravel().astype(np.float64) / 1000.0
        lf = g["sounding_land_fraction"][...].ravel().astype(np.float64) if land_fraction else None
    v[~np.isfinite(v) | (np.abs(v) > V_FILL_KMS)] = np.nan
    ok = sid > 0
    out = {"sounding_id": sid[ok], "v_kms": v[ok]}
    if land_fraction:
        lf[(lf < 0) | (lf > 100)] = np.nan
        out["land_fraction_pct"] = lf[ok]
    return out


def velocity_for(ids, l1b):
    """|v|-ready velocity (km/s) aligned to ids; NaN where the id is not in the L1B."""
    v = np.full(len(ids), np.nan)
    ia, ib = join_ids(ids, l1b["sounding_id"])
    v[ia] = l1b["v_kms"][ib]
    return v, len(ia)


def orbit_table(date, orbit, prod_dir, dop_dir, data_dir, source):
    """Per-sounding DataFrame (all matched soundings) and a join-statistics dict for one orbit."""
    pkeys = list(META_KEYS) + [prod_key(b, source) for b in BANDS]
    prod = read_datasets(fit_path(prod_dir, date, orbit), pkeys, required=pkeys)
    dkeys = ["fp_id"] + [f"{b}_{q}_nosg" for b in BANDS for q in DOP_QUANTITIES + ("nchan",)]
    dop = read_datasets(fit_path(dop_dir, date, orbit), dkeys,
                        required=["fp_id"] + [f"{b}_rms_prod_nosg" for b in BANDS])
    stats = {"date": date, "orbit": orbit, "n_prod": len(prod["fp_id"]), "n_dop": len(dop["fp_id"])}
    ip, idp = join_ids(prod["fp_id"], dop["fp_id"])
    stats["n_prod_dop_matched"] = len(ip)

    l1b_file = find_l1b(data_dir, date, orbit)
    if l1b_file is None:
        raise FileNotFoundError(f"no L1B under {data_dir}/OCO2 for {date} {orbit}")
    v, n_l1b = velocity_for(prod["fp_id"][ip], load_l1b_geometry(l1b_file))
    stats["n_l1b_matched"] = n_l1b
    stats["n_v_finite"] = int(np.isfinite(v).sum())
    stats["l1b_file"] = os.path.basename(l1b_file)

    df = pd.DataFrame({
        "date": date, "orbit": orbit,
        "sounding_id": prod["fp_id"][ip].astype(np.int64),
        "fp_number": prod["fp_number"][ip].astype(np.int8),
        "lat": prod["lat"][ip].astype(np.float32),
        "lon": prod["lon"][ip].astype(np.float32),
        "cld_dist_km": prod["cld_dist_km"][ip].astype(np.float32),
        "xco2_qf": prod["xco2_qf"][ip].astype(np.float32),
        "sfc_type": prod["sfc_type"][ip].astype(np.float32),
        "v_kms": v,                                   # float64: keeps bin-edge assignment exact
    })
    df["abs_v_kms"] = np.abs(df["v_kms"])
    for b in BANDS:
        df[f"prod_{b}"] = prod[prod_key(b, source)][ip].astype(np.float32)
        for q in DOP_QUANTITIES + ("nchan",):
            arr = dop[f"{b}_{q}_nosg"]
            df[f"dop_{q}_{b}"] = (arr[idp].astype(np.float32) if arr is not None
                                  else np.full(len(ip), np.nan, np.float32))
    df["far_clear"] = (df["cld_dist_km"] > FAR_KM) & (df["xco2_qf"] == 0)
    stats["n_far_clear"] = int(df["far_clear"].sum())
    stats["n_far_clear_v"] = int((df["far_clear"] & np.isfinite(df["v_kms"])).sum())
    return df, stats


def bin_edges(max_v, width=BIN_KMS):
    edges = np.arange(0.0, max_v + 1e-9, width)
    if edges[-1] < max_v - 1e-9:
        edges = np.append(edges, max_v)
    return edges


def quantity_columns(source):
    cols = [("prod", "prod_{b}", prod_key("{b}", source))]
    cols += [(f"dop_{q}", f"dop_{q}_{{b}}", f"{{b}}_{q}_nosg") for q in DOP_QUANTITIES]
    return cols


def binned_stats(df, edges, source):
    """Long-format table: band, quantity, bin, n, median, p16, p84 (pooled)."""
    rows = []
    av = df["abs_v_kms"].to_numpy()
    idx = np.digitize(av, edges) - 1          # bin i holds edges[i] <= |v| < edges[i+1]
    for b in BANDS:
        for qname, col, dataset in quantity_columns(source):
            x = df[col.format(b=b)].to_numpy(float)
            for i in range(len(edges) - 1):
                m = (idx == i) & np.isfinite(x)
                n = int(m.sum())
                p16, p50, p84 = np.percentile(x[m], [16, 50, 84]) if n else (np.nan,) * 3
                rows.append({"band": b, "quantity": qname, "dataset": dataset.format(b=b),
                             "bin_lo_kms": edges[i], "bin_hi_kms": edges[i + 1],
                             "bin_center_kms": 0.5 * (edges[i] + edges[i + 1]),
                             "n": n, "median": p50, "p16": p16, "p84": p84})
    out = pd.DataFrame(rows)
    out.insert(2, "prod_source", source)
    return out


def range_medians(df, source):
    rows = []
    av = df["abs_v_kms"].to_numpy()
    for b in BANDS:
        for qname, col, dataset in quantity_columns(source):
            x = df[col.format(b=b)].to_numpy(float)
            rec = {"band": b, "quantity": qname, "dataset": dataset.format(b=b)}
            for lo, hi in SUMMARY_RANGES:
                m = (av >= lo) & (av < hi) & np.isfinite(x)
                rec[f"n_{lo:g}-{hi:g}"] = int(m.sum())
                rec[f"median_{lo:g}-{hi:g}"] = float(np.median(x[m])) if m.any() else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def orbit_spearman(df, stats, source, min_n=10):
    rows = []
    far = df[df["far_clear"] & np.isfinite(df["abs_v_kms"])]
    av = far["abs_v_kms"].to_numpy(float)
    for b in BANDS:
        for qname, col, dataset in quantity_columns(source):
            x = far[col.format(b=b)].to_numpy(float)
            m = np.isfinite(x)
            rec = dict(stats)
            rec.update({"band": b, "quantity": qname, "dataset": dataset.format(b=b),
                        "n_far_used": int(m.sum()),
                        "abs_v_min": float(av[m].min()) if m.any() else np.nan,
                        "abs_v_max": float(av[m].max()) if m.any() else np.nan})
            if m.sum() >= min_n and np.ptp(av[m]) > 0 and np.ptp(x[m]) > 0:
                r = spearmanr(av[m], x[m])
                rec["spearman_rho"], rec["spearman_p"] = float(r.statistic), float(r.pvalue)
            else:
                rec["spearman_rho"], rec["spearman_p"] = np.nan, np.nan
            rows.append(rec)
    return rows


def md_table(df, cols, fmt="{:.4g}"):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _use_log(lo, hi, ratio=20.0):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ok = np.isfinite(lo) & np.isfinite(hi) & (lo > 0)
    return bool(ok.any() and np.nanmax(hi[ok]) / np.nanmin(lo[ok]) > ratio)


def plot(bins, source, title, path):
    same_axis = source == "rms_prod_nosg"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, b in zip(axes, BANDS):
        pb = bins[(bins.band == b) & (bins.quantity == "prod")]
        db = bins[(bins.band == b) & (bins.quantity == "dop_rms_prod")]
        axp = ax if same_axis else ax.twinx()
        handles = []
        for a, d, col, lab in ((axp, pb, C_PROD, prod_label(source).replace("<band>", b)),
                               (ax, db, C_DOP, f"corrected τ: {b}_rms_prod_nosg")):
            ok = d["n"].to_numpy() > 0
            x = d["bin_center_kms"].to_numpy()[ok]
            a.fill_between(x, d["p16"].to_numpy()[ok], d["p84"].to_numpy()[ok], color=col, alpha=0.25, lw=0)
            handles += a.plot(x, d["median"].to_numpy()[ok], "-o", ms=3, color=col, label=lab)
        if same_axis:
            both = pd.concat([pb, db])
            if _use_log(both.p16, both.p84):
                ax.set_yscale("log")
        else:
            if _use_log(db.p16, db.p84):
                ax.set_yscale("log")
            if _use_log(pb.p16, pb.p84):
                axp.set_yscale("log")
            axp.set_ylabel(f"rms_rel_{b}: L2 radiance residual / continuum (%)", color=C_PROD)
            axp.tick_params(axis="y", colors=C_PROD)
        ax.set_ylabel("ln T residual rms (dimensionless)")
        ax.set_xlabel("|sounding_relative_velocity| (km s$^{-1}$)")
        n_far = int(db["n"].sum())
        ax.set_title(f"{b.upper()}  (far-clear N in bins = {n_far:,})", fontsize=10)
        ax.legend(handles=handles, fontsize=7, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_soundings(far, out_dir):
    """Far-clear per-sounding table: parquet (or npz) if under PARQUET_LIMIT_BYTES in memory."""
    nbytes = int(far.memory_usage(deep=True).sum())
    if nbytes > PARQUET_LIMIT_BYTES:
        return f"not written: far-clear table is {nbytes / 1024**2:.0f} MB in memory (> 200 MB)"
    try:
        path = os.path.join(out_dir, "doppler_rms_vs_velocity_soundings.parquet")
        far.to_parquet(path, index=False)
    except ImportError:
        path = os.path.join(out_dir, "doppler_rms_vs_velocity_soundings.npz")
        np.savez_compressed(path, **{c: far[c].to_numpy() for c in far.columns})
    return f"{os.path.basename(path)} ({len(far):,} far-clear soundings, {os.path.getsize(path) / 1024**2:.1f} MB)"


def main(argv=None):
    root = storage_root()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--prod-dir", default=os.path.join(root, "results"))
    ap.add_argument("--dop-dir", default=os.path.join(root, "results_doppler"))
    ap.add_argument("--data-dir", default=os.path.join(root, "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-v", type=float, default=7.0, help="upper |v| bin edge (km/s)")
    ap.add_argument("--prod-rms", choices=("auto", "rms_prod_nosg", "rms_rel"), default="auto",
                    help="production residual dataset (see docstring); auto = rms_prod_nosg if present else rms_rel")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    pairs, missing = paired_orbits(args.dates, args.prod_dir, args.dop_dir)
    for d, o, tree in missing:
        print(f"[missing in {tree}] {d} {o}", file=sys.stderr)
    if not pairs:
        print("no orbit present in both trees", file=sys.stderr)
        return 1
    source = resolve_prod_source(pairs, args.prod_dir, args.prod_rms)
    print(f"production residual dataset: {prod_key('<band>', source)}", flush=True)

    far_parts, orbit_rows, skipped = [], [], []
    for date, orbit in pairs:
        try:
            df, stats = orbit_table(date, orbit, args.prod_dir, args.dop_dir, args.data_dir, source)
        except (KeyError, OSError, ValueError) as exc:
            print(f"[skip] {date} {orbit}: {exc}", file=sys.stderr)
            skipped.append(f"{date} {orbit}: {exc}")
            continue
        print(f"{date} {orbit}: prod {stats['n_prod']}, dop {stats['n_dop']}, matched "
              f"{stats['n_prod_dop_matched']}, L1B matched {stats['n_l1b_matched']}, "
              f"far-clear {stats['n_far_clear']}", flush=True)
        orbit_rows += orbit_spearman(df, stats, source)
        far_parts.append(df[df["far_clear"]].copy())
        del df
    if not far_parts:
        print("nothing joined", file=sys.stderr)
        return 1
    far = pd.concat(far_parts, ignore_index=True)
    del far_parts
    edges = bin_edges(args.max_v)
    n_over = int((far["abs_v_kms"] >= edges[-1]).sum())
    n_nan_v = int(far["abs_v_kms"].isna().sum())

    bins = binned_stats(far, edges, source)
    bins.to_csv(os.path.join(args.out, "doppler_rms_vs_velocity.csv"), index=False)
    per_orbit = pd.DataFrame(orbit_rows)
    per_orbit.insert(2, "prod_source", source)
    per_orbit.to_csv(os.path.join(args.out, "doppler_rms_vs_velocity_per_orbit.csv"), index=False)

    d0, d1 = min(args.dates), max(args.dates)
    drange = d0 if d0 == d1 else f"{d0} to {d1}"
    n_orb = per_orbit[["date", "orbit"]].drop_duplicates().shape[0]
    title = (f"Cumulant-fit residual rms vs |v|, far-clear (cld > {FAR_KM:g} km, QF0), {drange}, "
             f"{n_orb} orbits, N = {len(far):,}")
    plot(bins, source, title, os.path.join(args.out, "doppler_rms_vs_velocity.png"))
    snd_msg = save_soundings(far, args.out)

    rm = range_medians(far, source)
    os_ = per_orbit.drop_duplicates(["date", "orbit"])
    tot = {k: int(os_[k].sum()) for k in ("n_prod", "n_dop", "n_prod_dop_matched", "n_l1b_matched", "n_far_clear")}
    rho = (per_orbit.groupby(["band", "quantity"])["spearman_rho"]
           .agg(["count", "median", "min", "max"]).reset_index()
           .rename(columns={"count": "n_orbits", "median": "rho_median", "min": "rho_min", "max": "rho_max"}))
    lines = [f"# Residual rms vs |sounding_relative_velocity| — {drange}, {n_orb} orbits", "",
             f"Production residual dataset: `{prod_key('<band>', source)}`. "
             + ("Same quantity as the variant's `<band>_rms_prod_nosg` (production-order fit of raw ln T, "
                "residual rms in ln T)." if source == "rms_prod_nosg" else
                "NOT the same quantity as the variant's `<band>_rms_prod_nosg`: production fitting_details "
                "store no cumulant-fit residual; `rms_rel_<band>` is the Lite L2 full-physics radiance-fit "
                "residual rms relative to the continuum, in percent. Closest stored dataset, shown for context only."),
             "",
             f"Far-clear: cld_dist_km > {FAR_KM:g} and xco2_qf == 0 (production fit file). "
             f"Joins: prod {tot['n_prod']:,} / dop {tot['n_dop']:,} soundings, prod-dop matched "
             f"{tot['n_prod_dop_matched']:,}, L1B velocity matched {tot['n_l1b_matched']:,}, far-clear "
             f"{tot['n_far_clear']:,}. Far-clear with |v| >= {edges[-1]:g} km/s (outside bins): {n_over}; "
             f"with no velocity: {n_nan_v}.", "",
             "## Pooled medians at |v| < 0.5 km/s and 4.5–5.5 km/s", "",
             md_table(rm, list(rm.columns)), "",
             "## Per-orbit Spearman rho(|v|, rms), far-clear", "",
             md_table(rho, list(rho.columns), fmt="{:.3f}"), "",
             f"Per-sounding table: {snd_msg}."]
    if skipped:
        lines += ["", "Skipped: " + "; ".join(skipped)]
    with open(os.path.join(args.out, "doppler_rms_vs_velocity.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

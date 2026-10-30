"""Compare production fitting details with the telluric-Doppler test variant.

For every date given, pairs ``fitting_details_<date>_<orbit>.h5`` found in the
production tree with the same file in the test tree (written by
``fitting.py --telluric-doppler``) and reports, per orbit and band:

* far-clear (cld_dist > 20 km, QF0) scatter of the production-order ``_nosg``
  k1 and of the SCO2/O2A k1 ratio, production vs corrected tau;
* the same for the unconstrained order-2 / order-3 k1 stored by the variant;
* residual rms of the variant's fits (production-order, order 2, order 3);
* order-3 k2 zero fraction, production vs corrected;
* near-cloud (< 1 km, < 3 km) k1 contrast against latitude-local far
  references, with its signal-to-noise ratio, production vs corrected;
* correlation of the near-cloud k1 contrast with ``xco2_bc_anomaly``.

Usage
-----
    python workspace/compare_doppler_fits.py --dates 2020-01-01 2020-02-01 \\
        --prod-dir results --dop-dir results_doppler --out results_doppler/compare

Outputs: ``compare_doppler_<first>_<last>.csv``, ``compare_doppler_summary.md``
and ``compare_doppler_spread.png`` in ``--out``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BANDS = ("o2a", "wco2", "sco2")
FAR_KM = 20.0
REF_KM = 10.0
LAT_HALF_WIDTH = 1.0
NEAR_BINS = ((0.0, 1.0), (1.0, 3.0))
META_KEYS = ("cld_dist_km", "xco2_qf", "lat", "fp_number", "xco2_bc_anomaly", "fp_id")


def robust_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def spread_pct(x, fp):
    """Robust SD (%) after removing per-footprint-index means."""
    s = pd.Series(np.asarray(x, float))
    rel = s / s.groupby(np.asarray(fp)).transform("mean") - 1.0
    return 100.0 * robust_sd(rel.values)


def local_contrast(v, lat, fp, cld, qf):
    """value / latitude-local far reference − 1 (same footprint, ±1° lat, cld > 10 km, QF0)."""
    n = len(v)
    ref_ok = (cld > REF_KM) & (qf == 0) & np.isfinite(v)
    ref = np.full(n, np.nan)
    for f in np.unique(fp):
        m_f = fp == f
        rl, rv = lat[ref_ok & m_f], v[ref_ok & m_f]
        if len(rl) < 5:
            continue
        order = np.argsort(rl)
        rl, rv = rl[order], rv[order]
        for j in np.flatnonzero(m_f):
            lo = np.searchsorted(rl, lat[j] - LAT_HALF_WIDTH)
            hi = np.searchsorted(rl, lat[j] + LAT_HALF_WIDTH, side="right")
            if hi - lo >= 5:
                ref[j] = np.nanmedian(rv[lo:hi])
    return v / ref - 1.0


def load_details(path, extra=False):
    out = {}
    with h5py.File(path, "r") as f:
        for k in META_KEYS:
            out[k] = f[k][...] if k in f else None
        for b in BANDS:
            out[f"{b}_k1"] = f[f"{b}_k1_fitting_nosg"][...]
            out[f"{b}_k2"] = f[f"{b}_k2_fitting_nosg"][...]
            if extra:
                for name in ("k1_o2", "k2_o2", "k1_o3", "k2_o3", "rms_prod", "rms_o2", "rms_o3", "nchan"):
                    key = f"{b}_{name}_nosg"
                    out[f"{b}_{name}"] = f[key][...] if key in f else None
    return out


def orbit_rows(date, orbit, prod, dop):
    # fp_id is the sounding id in fitting_details; both trees must list the same soundings in order
    if prod["fp_id"] is None or dop["fp_id"] is None or not np.array_equal(prod["fp_id"], dop["fp_id"]):
        raise ValueError(f"{date} {orbit}: sounding order differs between the two trees")
    cld, qf, lat, fp = (prod["cld_dist_km"], prod["xco2_qf"], prod["lat"], prod["fp_number"].astype(int))
    anom = prod["xco2_bc_anomaly"]
    far = (cld > FAR_KM) & (qf == 0)
    rows = []

    def add(src, band, quantity, v):
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        rec = {"date": date, "orbit": orbit, "src": src, "band": band, "quantity": quantity,
               "n_far": int((far & ok).sum()), "far_median": np.nanmedian(v[far & ok]) if (far & ok).any() else np.nan}
        rec["far_spread_pct"] = spread_pct(v[far & ok], fp[far & ok]) if (far & ok).sum() >= 10 else np.nan
        con = local_contrast(v, lat, fp, cld, qf)
        far_sd = robust_sd(con[far])
        for lo, hi in NEAR_BINS:
            m = (cld >= lo) & (cld < hi) & np.isfinite(con)
            lab = f"{lo:g}-{hi:g}km"
            rec[f"n_{lab}"] = int(m.sum())
            rec[f"contrast_{lab}_pct"] = 100 * np.nanmedian(con[m]) if m.sum() > 20 else np.nan
            rec[f"snr_{lab}"] = (np.nanmedian(con[m]) / far_sd) if (m.sum() > 20 and far_sd > 0) else np.nan
        mn = (cld < 3) & np.isfinite(con) & (anom is not None) & (np.isfinite(anom) if anom is not None else False)
        rec["corr_anom_lt3km"] = float(np.corrcoef(con[mn], anom[mn])[0, 1]) if mn.sum() > 30 else np.nan
        rows.append(rec)

    for src, d in (("prod", prod), ("dop", dop)):
        for b in BANDS:
            add(src, b, "k1_prod_order", d[f"{b}_k1"])
        add(src, "ratio", "k1_prod_order", d["sco2_k1"] / d["o2a_k1"])
    for name in ("k1_o2", "k1_o3"):
        if dop.get(f"o2a_{name}") is None:
            continue
        for b in BANDS:
            add("dop", b, name, dop[f"{b}_{name}"])
        add("dop", "ratio", name, dop[f"sco2_{name}"] / dop[f"o2a_{name}"])

    # k2 zero fraction (production order, bounded) and variant rms, far-clear
    extra = []
    for src, d in (("prod", prod), ("dop", dop)):
        for b in BANDS:
            k2 = d[f"{b}_k2"]
            m = far & np.isfinite(k2)
            rec = {"date": date, "orbit": orbit, "src": src, "band": b,
                   "k2_zero_frac_far": float(np.mean(k2[m] <= 1e-6)) if m.any() else np.nan}
            if src == "dop":
                for name in ("rms_prod", "rms_o2", "rms_o3", "k2_o2", "k2_o3"):
                    v = d.get(f"{b}_{name}")
                    rec[f"{name}_far_median"] = float(np.nanmedian(v[far])) if v is not None else np.nan
                k2o3 = d.get(f"{b}_k2_o3")
                rec["k2_o3_neg_frac_far"] = float(np.nanmean(k2o3[far] < 0)) if k2o3 is not None else np.nan
            extra.append(rec)
    return rows, extra


def md_table(df, cols, fmt="{:.3g}"):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--prod-dir", default="results")
    ap.add_argument("--dop-dir", default="results_doppler")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.dop_dir, "compare")
    os.makedirs(out_dir, exist_ok=True)

    rows, extra, missing = [], [], []
    for date in args.dates:
        for pf in sorted(glob.glob(os.path.join(args.prod_dir, "fitting_details", f"fitting_details_{date}_*.h5"))):
            orbit = os.path.basename(pf)[len(f"fitting_details_{date}_"):-3]
            df_ = os.path.join(args.dop_dir, "fitting_details", os.path.basename(pf))
            if not os.path.isfile(df_):
                missing.append((date, orbit))
                continue
            try:
                r, e = orbit_rows(date, orbit, load_details(pf), load_details(df_, extra=True))
            except (KeyError, ValueError, OSError) as exc:
                print(f"[skip] {date} {orbit}: {exc}", file=sys.stderr)
                continue
            rows += r
            extra += e
            print(f"compared {date} {orbit}", flush=True)
    if not rows:
        print("nothing compared", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    dx = pd.DataFrame(extra)
    tag = f"{args.dates[0]}_{args.dates[-1]}"
    df.to_csv(os.path.join(out_dir, f"compare_doppler_{tag}.csv"), index=False)
    dx.to_csv(os.path.join(out_dir, f"compare_doppler_k2rms_{tag}.csv"), index=False)

    # summary over orbits: median of each metric per (src, band, quantity)
    num = [c for c in df.columns if c not in ("date", "orbit", "src", "band", "quantity")]
    summ = df.groupby(["quantity", "band", "src"])[num].median().reset_index()
    summ_x = dx.groupby(["band", "src"])[[c for c in dx.columns if c not in ("date", "orbit", "src", "band")]].median().reset_index()
    lines = [f"# Production vs telluric-Doppler fits — {len(df.orbit.unique())} orbits, dates {', '.join(args.dates)}", "",
             f"Far-clear: cld_dist > {FAR_KM:g} km and QF0; spread = robust SD (%) after removing per-footprint means; "
             "contrast = k1 relative to latitude-local far references (same footprint, ±1° lat, cld > 10 km, QF0); "
             "SNR = median contrast / robust SD of far-clear contrasts. Medians over orbits.", ""]
    if missing:
        lines += ["Missing in the test tree: " + ", ".join(f"{d} {o}" for d, o in missing), ""]
    cols = ["quantity", "band", "src", "n_far", "far_median", "far_spread_pct", "contrast_0-1km_pct", "snr_0-1km",
            "contrast_1-3km_pct", "snr_1-3km", "corr_anom_lt3km"]
    lines += ["## k1 scatter and near-cloud contrast", "", md_table(summ, cols), ""]
    lines += ["## k2 zero fraction and variant residual rms (far-clear medians)", "",
              md_table(summ_x, list(summ_x.columns)), ""]
    with open(os.path.join(out_dir, "compare_doppler_summary.md"), "w") as fh:
        fh.write("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = df[df.quantity == "k1_prod_order"]
    for i, (metric, ylabel) in enumerate((("far_spread_pct", "far-clear k1 spread (%)"), ("snr_0-1km", "near-cloud (<1 km) k1 SNR"))):
        ax = axes[i]
        for src, col in (("prod", "0.4"), ("dop", "C3")):
            g = d[d.src == src].groupby("band")[metric].median().reindex(list(BANDS) + ["ratio"])
            ax.bar(np.arange(4) + (0.2 if src == "dop" else -0.2), g.values, width=0.4, color=col, label="production τ" if src == "prod" else "Doppler-corrected τ")
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(list(BANDS) + ["sco2/o2a"])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_doppler_spread.png"), dpi=130)
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

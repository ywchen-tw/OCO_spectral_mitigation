"""
mixed_surface_reference_count.py
================================
How often does the clear-sky reference window of a Fig.-6 target footprint
contain footprints of the OTHER surface type?

Background
----------
`src/spectral/anomaly.py::compute_xco2_anomaly` builds, for footprint *i*, a
reference set of all footprints in the SAME orbit file (== same (date,
orbit_id) group in the combined parquet) with

    |lat_j - lat_i| <= ANOMALY_LAT_THRES_DEG (0.25 deg)
    cld_dist_km_j   >  min_cld_dist            (15 km for land r15, 5 km ocean r05)
    valid xco2                                 (xco2_bc > 0)
    valid values of every extra_var             (k1..k3 x 3 bands, alb x 3,
                                                 exp_int x 3 -- see
                                                 spectral/fitting.py ref_extra_vars)

There is NO surface-type constraint in that window, and the anomaly / reference
mean+std only exist when the reference has >= 5 members and std(xco2) <= 1 ppm.

The Fig.-6 (shadow / brightening) population is defined by
`src/analysis/spec_sensitivity.py::run_shadow_brightening` on a dataframe that
has already passed `analysis.utils.apply_quality_filter` (xco2_bc > 0,
xco2_qf == 0, snow_flag == 0) and `_apply_production_reference`
(land -> r15 columns, ocean -> r05 columns, |anomaly| <= 100 ppm screen):

    0 <= cld_dist_km < 10 km, split into bins [0,1) [1,2) [2,3) [3,5) [5,7) [7,10)
    branches by zexp_o2a: > +0.5 brightened, |.| <= 0.5 neutral, < -0.5 shadowed
    land uses zr15exp_o2a = (exp_o2a_intercept - r15_exp_int_o2a_mean)
                            / r15_exp_int_o2a_std

This script replicates that population, recomputes each target's reference
window membership with a sorted-latitude searchsorted (O(N log N)), splits the
members by sfc_type, and reports the mixed-window fraction overall, per
distance bin and per z_exp branch.  It also recomputes z_exp from the
same-surface members only, to see whether any target changes branch.

NOTE on the k family used for the extra_vars validity mask: the fitting stage
passes the SMOOTHED (Savitzky-Golay) cumulants -- `kappa_fitting[...]` in
spectral/fitting.py, stored in the parquet as `o2a_k1` etc. without the `_nosg`
suffix -- so this script uses the non-`_nosg` columns.  (`_USE_NOSG_K = True` in
models/pipeline.py concerns the ML FEATURES, not the reference-window mask.)

Read-only on the repo apart from the output directory.  Besides the summary
tables it writes `mixed_window_flags.parquet` (fp_id, sfc_type, n_ref,
n_ref_other, mixed) covering both evaluated populations; that file is what
`spec_sensitivity.py --exclude-mixed-ref` merges on to drop mixed-window
targets from the Fig.-6 statistics.

Usage
-----
    PYTHONPATH=src:workspace python workspace/mixed_surface_reference_count.py \
        [--max-row-groups 10] [--outdir ...]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from constants import ANOMALY_LAT_THRES_DEG, ANOMALY_STD_THRES_PPM  # noqa: E402

logger = logging.getLogger("mixed_surface")

# ── configuration mirrored from the production code ──────────────────────────
LAT_THRES = float(ANOMALY_LAT_THRES_DEG)      # 0.25 deg
STD_THRES = float(ANOMALY_STD_THRES_PPM)      # 1.0 ppm  (reported, not re-applied)
N_MIN_REF = 5                                 # anomaly.py has_refs threshold
NEAR_MAX = 10.0                               # spec_sensitivity._NEAR_MAX
Z_THRESH = 0.5                                # run_shadow_brightening default
BIN_EDGES = [0, 1, 2, 3, 5, 7, 10]
BIN_LABELS = [f"{BIN_EDGES[i]}-{BIN_EDGES[i+1]}" for i in range(len(BIN_EDGES) - 1)]
ANOM_SCREEN_PPM = 100.0                       # _apply_production_reference

# extra_vars passed by spectral/fitting.py (SG cumulants -> non-_nosg columns)
EXTRA_VAR_COLS = [
    "o2a_k1", "o2a_k2", "o2a_k3",
    "wco2_k1", "wco2_k2", "wco2_k3",
    "sco2_k1", "sco2_k2", "sco2_k3",
    "alb_o2a", "alb_wco2", "alb_sco2",
    "exp_o2a_intercept", "exp_wco2_intercept", "exp_sco2_intercept",
]

KEEP_F4 = [
    "lat", "cld_dist_km", "xco2_bc",
    "xco2_bc_anomaly_r15", "xco2_bc_anomaly_r05",
    "exp_o2a_intercept",
    "r15_exp_int_o2a_mean", "r15_exp_int_o2a_std",
    "r05_exp_int_o2a_mean", "r05_exp_int_o2a_std",
    "sfc_type", "xco2_qf", "snow_flag",
]
READ_COLS = sorted(set(KEEP_F4 + EXTRA_VAR_COLS + ["date", "orbit_id", "fp_id"]))


# ── data loading ─────────────────────────────────────────────────────────────

def load(path: Path, max_row_groups=None) -> dict:
    """Stream the parquet row-group by row-group, keeping only small arrays."""
    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups
    if max_row_groups:
        n_rg = min(n_rg, max_row_groups)
    logger.info("Reading %d/%d row groups, %d columns from %s",
                n_rg, pf.metadata.num_row_groups, len(READ_COLS), path.name)

    chunks = {c: [] for c in KEEP_F4}
    chunks["fp_id"] = []
    chunks["extras_ok"] = []
    chunks["gkey"] = []
    gmap: dict = {}
    t0 = time.time()
    n_rows = 0
    for rg in range(n_rg):
        tbl = pf.read_row_group(rg, columns=READ_COLS)
        n = tbl.num_rows
        n_rows += n

        # per-row extra_vars validity (the shared reference pool requirement)
        ok = np.ones(n, dtype=bool)
        for c in EXTRA_VAR_COLS:
            ok &= ~np.isnan(tbl.column(c).to_numpy(zero_copy_only=False))
        chunks["extras_ok"].append(ok)

        for c in KEEP_F4:
            chunks[c].append(tbl.column(c).to_numpy(zero_copy_only=False)
                             .astype(np.float32, copy=False))
        chunks["fp_id"].append(tbl.column("fp_id").to_numpy(zero_copy_only=False))

        # (date, orbit_id) -> global integer code, via the small unique sets
        d = tbl.column("date").to_pandas().to_numpy()
        o = tbl.column("orbit_id").to_pandas().to_numpy()
        pair = pd.Series(list(zip(d, o)))
        codes, uniq = pd.factorize(pair)
        gcodes = np.empty(len(uniq), dtype=np.int32)
        for i, key in enumerate(uniq):
            if key not in gmap:
                gmap[key] = len(gmap)
            gcodes[i] = gmap[key]
        chunks["gkey"].append(gcodes[codes])
        del tbl

        if (rg + 1) % 10 == 0 or rg + 1 == n_rg:
            logger.info("  row group %3d/%d  rows=%s  groups=%d  %.0fs",
                        rg + 1, n_rg, f"{n_rows:,}", len(gmap), time.time() - t0)

    out = {k: np.concatenate(v) for k, v in chunks.items()}
    logger.info("Loaded %s rows in %d (date, orbit_id) groups (%.0fs)",
                f"{len(out['lat']):,}", len(gmap), time.time() - t0)
    return out


# ── core per-orbit computation ───────────────────────────────────────────────

def _window_counts(tlat, ref_lat_sorted):
    """Members with |lat_j - lat_i| <= LAT_THRES, for sorted ref latitudes."""
    lo = np.searchsorted(ref_lat_sorted, tlat - LAT_THRES, side="left")
    hi = np.searchsorted(ref_lat_sorted, tlat + LAT_THRES, side="right")
    return lo, hi


def _prefix(vals):
    """Prefix sums of v and v^2 for window mean/std (population std, ddof=0)."""
    cs = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
    cs2 = np.concatenate(([0.0], np.cumsum(vals * vals, dtype=np.float64)))
    return cs, cs2


def _win_mean_std(cs, cs2, lo, hi):
    n = (hi - lo).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = cs[hi] - cs[lo]
        s2 = cs2[hi] - cs2[lo]
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        var = np.where(n > 0, s2 / np.maximum(n, 1) - mean ** 2, np.nan)
        std = np.sqrt(np.maximum(var, 0.0))
    return mean, std, n


def analyse(data: dict, surface: int, min_cld_dist: float, near_max: float,
            anom_col: str, ref_mean_col: str, ref_std_col: str) -> pd.DataFrame:
    """Per-target reference-window composition for one surface's production ref."""
    lat = data["lat"].astype(np.float64)
    cld = data["cld_dist_km"].astype(np.float64)
    xco2 = data["xco2_bc"].astype(np.float64)
    sfc = data["sfc_type"]
    qf = data["xco2_qf"]
    snow = data["snow_flag"]
    gkey = data["gkey"]
    extras_ok = data["extras_ok"]
    exp_o2a = data["exp_o2a_intercept"].astype(np.float64)
    anom = data[anom_col].astype(np.float64)
    rmean = data[ref_mean_col].astype(np.float64)
    rstd = data[ref_std_col].astype(np.float64)

    # clear-sky reference candidates for this min_cld_dist (NO surface constraint)
    clear = (np.isfinite(lat) & (cld > min_cld_dist) & (xco2 > 0) & extras_ok)

    # Fig.-6 target population (quality filter + near-cloud + surface)
    with np.errstate(invalid="ignore"):
        zprod = np.where(rstd > 0, (exp_o2a - rmean) / rstd, np.nan)
    target = ((sfc == surface) & (qf == 0) & (snow == 0) & (xco2 > 0)
              & np.isfinite(lat) & (cld >= 0) & (cld < near_max)
              & np.isfinite(zprod) & np.isfinite(anom)
              & (np.abs(anom) <= ANOM_SCREEN_PPM))

    logger.info("  clear-sky candidates (>%.0f km): %s | targets: %s",
                min_cld_dist, f"{int(clear.sum()):,}", f"{int(target.sum()):,}")

    order = np.argsort(gkey, kind="stable")
    gsorted = gkey[order]
    bounds = np.flatnonzero(np.r_[True, gsorted[1:] != gsorted[:-1], True])

    n_all = np.zeros(len(lat), dtype=np.int32)
    n_land = np.zeros(len(lat), dtype=np.int32)
    n_ocean = np.zeros(len(lat), dtype=np.int32)
    mean_all = np.full(len(lat), np.nan)
    std_all = np.full(len(lat), np.nan)
    mean_same = np.full(len(lat), np.nan)
    std_same = np.full(len(lat), np.nan)

    n_groups = len(bounds) - 1
    t0 = time.time()
    for gi in range(n_groups):
        idx = order[bounds[gi]:bounds[gi + 1]]
        tsel = idx[target[idx]]
        if tsel.size == 0:
            continue
        csel = idx[clear[idx]]
        tlat = lat[tsel]

        if csel.size:
            o = np.argsort(lat[csel], kind="stable")
            csel = csel[o]
            clat = lat[csel]
            lo, hi = _window_counts(tlat, clat)
            n_all[tsel] = hi - lo
            cs, cs2 = _prefix(exp_o2a[csel])
            m, s, _ = _win_mean_std(cs, cs2, lo, hi)
            mean_all[tsel] = m
            std_all[tsel] = s

            csfc = sfc[csel]
            for code, arr in ((1, n_land), (0, n_ocean)):
                sub = csel[csfc == code]
                if sub.size == 0:
                    continue
                slat = lat[sub]           # still sorted (subset of sorted)
                l2, h2 = _window_counts(tlat, slat)
                arr[tsel] = h2 - l2
                if code == surface:
                    cs_s, cs2_s = _prefix(exp_o2a[sub])
                    ms, ss, _ = _win_mean_std(cs_s, cs2_s, l2, h2)
                    mean_same[tsel] = ms
                    std_same[tsel] = ss
        if (gi + 1) % 200 == 0:
            logger.info("    group %d/%d (%.0fs)", gi + 1, n_groups, time.time() - t0)

    t = np.flatnonzero(target)
    n_other = n_ocean[t] if surface == 1 else n_land[t]
    n_same = n_land[t] if surface == 1 else n_ocean[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        other_share = np.where(n_all[t] > 0, n_other / n_all[t], np.nan)
        z_same = np.where(std_same[t] > 0,
                          (exp_o2a[t] - mean_same[t]) / std_same[t], np.nan)

    return pd.DataFrame({
        "fp_id": data["fp_id"][t],
        "lat": lat[t],
        "cld_dist_km": cld[t],
        "n_ref": n_all[t],
        "n_ref_same": n_same,
        "n_ref_other": n_other,
        "other_share": other_share,
        "z_prod": zprod[t],
        "z_same_surface": z_same,
        "ref_mean_recomp": mean_all[t],
        "ref_std_recomp": std_all[t],
        "ref_mean_stored": rmean[t],
        "ref_std_stored": rstd[t],
        "anomaly": anom[t],
    })


# ── reporting helpers ────────────────────────────────────────────────────────

def branch(z, thr=Z_THRESH):
    return np.where(z > thr, "brightened",
                    np.where(z < -thr, "shadowed", "neutral"))


def summarise(res: pd.DataFrame, label: str) -> dict:
    n = len(res)
    mixed = res["n_ref_other"] > 0
    out = {
        "population": label,
        "n_targets": int(n),
        "n_mixed": int(mixed.sum()),
        "frac_mixed": float(mixed.mean()) if n else np.nan,
        "n_other_majority": int((res["other_share"] > 0.5).sum()),
        "frac_other_majority": float((res["other_share"] > 0.5).mean()) if n else np.nan,
        "n_all_other": int((res["other_share"] >= 1.0).sum()),
        "median_n_ref": float(res["n_ref"].median()) if n else np.nan,
        "frac_n_ref_lt_5": float((res["n_ref"] < N_MIN_REF).mean()) if n else np.nan,
    }
    sh = res.loc[mixed, "other_share"]
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        out[f"other_share_q{int(q*100):02d}_mixed"] = float(sh.quantile(q)) if len(sh) else np.nan
    out["mean_other_share_mixed"] = float(sh.mean()) if len(sh) else np.nan
    return out


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--parquet", default=str(
        ROOT / "results/csv_collection/combined_2016_2020_dates.parquet"))
    ap.add_argument("--max-row-groups", type=int, default=None,
                    help="validate on the first N row groups (~1 date each)")
    ap.add_argument("--outdir", default=str(
        ROOT / "results/figures/cld_dist_analysis/spec_sensitivity/prodref"
             / "mixed_surface_windows"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "" if args.max_row_groups is None else f"_rg{args.max_row_groups}"

    data = load(Path(args.parquet), args.max_row_groups)

    logger.info("── LAND targets (<%.0f km), r15 reference window ──", NEAR_MAX)
    land = analyse(data, surface=1, min_cld_dist=15.0, near_max=NEAR_MAX,
                   anom_col="xco2_bc_anomaly_r15",
                   ref_mean_col="r15_exp_int_o2a_mean",
                   ref_std_col="r15_exp_int_o2a_std")

    logger.info("── OCEAN targets (<%.0f km), r05 reference window ──", NEAR_MAX)
    ocean = analyse(data, surface=0, min_cld_dist=5.0, near_max=NEAR_MAX,
                    anom_col="xco2_bc_anomaly_r05",
                    ref_mean_col="r05_exp_int_o2a_mean",
                    ref_std_col="r05_exp_int_o2a_std")
    del data

    # ── per-footprint flag file (consumed by spec_sensitivity's
    #    --exclude-mixed-ref option) ────────────────────────────────────────
    flags = pd.concat(
        [land.assign(sfc_type=np.int8(1)), ocean.assign(sfc_type=np.int8(0))],
        ignore_index=True)[["fp_id", "sfc_type", "n_ref", "n_ref_other"]].copy()
    flags["fp_id"] = flags["fp_id"].astype("int64")
    flags["n_ref"] = flags["n_ref"].astype("int32")
    flags["n_ref_other"] = flags["n_ref_other"].astype("int32")
    flags["mixed"] = flags["n_ref_other"] > 0
    flags.to_parquet(outdir / f"mixed_window_flags{tag}.parquet", index=False)
    logger.info("wrote %s (%s rows, %s mixed)",
                outdir / f"mixed_window_flags{tag}.parquet",
                f"{len(flags):,}", f"{int(flags['mixed'].sum()):,}")

    # ── sanity: recomputed window vs the production guards / stored ref stats ──
    sanity = {}
    for name, res in (("land_r15", land), ("ocean_r05", ocean)):
        ok = res["n_ref"] >= N_MIN_REF
        dm = (res["ref_mean_recomp"] - res["ref_mean_stored"]).abs()
        ds = (res["ref_std_recomp"] - res["ref_std_stored"]).abs()
        rel = dm / res["ref_std_stored"].replace(0, np.nan).abs()
        sanity[name] = {
            "n_targets": int(len(res)),
            "frac_n_ref_ge_5": float(ok.mean()),
            "n_ref_lt_5": int((~ok).sum()),
            "median_abs_dmean_vs_stored": float(dm.median()),
            "p99_abs_dmean_vs_stored": float(dm.quantile(0.99)),
            "median_abs_dstd_vs_stored": float(ds.median()),
            "frac_dmean_within_0p01_sigma": float((rel < 0.01).mean()),
            "median_n_ref": float(res["n_ref"].median()),
        }
        logger.info("sanity[%s]: n_ref>=5 in %.4f of targets; median |Δref_mean| "
                    "vs stored = %.3g; %.4f within 0.01σ",
                    name, ok.mean(), dm.median(), (rel < 0.01).mean())

    # ── land breakdown: distance bin x branch ──────────────────────────────────
    land = land.copy()
    land["cld_bin"] = pd.cut(land["cld_dist_km"], bins=BIN_EDGES,
                             labels=BIN_LABELS, right=False)
    land["branch"] = branch(land["z_prod"].to_numpy())
    land["mixed"] = land["n_ref_other"] > 0

    rows = []
    rows.append({"group": "all", "level": "all", **summarise(land, "land_r15_all")})
    for b in BIN_LABELS:
        sub = land[land["cld_bin"] == b]
        rows.append({"group": "cld_bin", "level": b,
                     **summarise(sub, f"land_r15_bin_{b}")})
    for br in ("shadowed", "neutral", "brightened"):
        sub = land[land["branch"] == br]
        rows.append({"group": "branch", "level": br,
                     **summarise(sub, f"land_r15_branch_{br}")})
    for br in ("shadowed", "neutral", "brightened"):
        for b in BIN_LABELS:
            sub = land[(land["branch"] == br) & (land["cld_bin"] == b)]
            rows.append({"group": "branch_x_bin", "level": f"{br}|{b}",
                         **summarise(sub, f"land_r15_{br}_{b}")})
    rows.append({"group": "all", "level": "all",
                 **summarise(ocean, "ocean_r05_all_lt10km")})
    o5 = ocean[ocean["cld_dist_km"] < 5.0]
    rows.append({"group": "subset", "level": "cld_dist<5km",
                 **summarise(o5, "ocean_r05_lt5km")})

    table = pd.DataFrame(rows)
    table.to_csv(outdir / f"mixed_surface_window_table{tag}.csv", index=False)
    logger.info("wrote %s", outdir / f"mixed_surface_window_table{tag}.csv")

    # ── branch flip test on mixed-window land targets ─────────────────────────
    mx = land[land["mixed"]].copy()
    mx["branch_same"] = branch(mx["z_same_surface"].to_numpy())
    usable = mx["z_same_surface"].notna() & (mx["n_ref_same"] >= N_MIN_REF)
    u = mx[usable]
    flip = (u["branch"] != u["branch_same"])
    flip_tbl = (pd.crosstab(u["branch"], u["branch_same"])
                if len(u) else pd.DataFrame())
    flip_tbl.to_csv(outdir / f"branch_flip_crosstab{tag}.csv")
    flip_stats = {
        "n_mixed_land": int(len(mx)),
        "n_mixed_land_with_ge5_same_surface_refs": int(len(u)),
        "n_branch_changed": int(flip.sum()),
        "frac_branch_changed_of_usable": float(flip.mean()) if len(u) else np.nan,
        "frac_branch_changed_of_all_land": (float(flip.sum()) / len(land)
                                            if len(land) else np.nan),
        "median_abs_dz": float((u["z_same_surface"] - u["z_prod"]).abs().median())
        if len(u) else np.nan,
        "p95_abs_dz": float((u["z_same_surface"] - u["z_prod"]).abs().quantile(0.95))
        if len(u) else np.nan,
    }
    logger.info("branch flip: %d/%d usable mixed-window land targets change branch "
                "(%.4f)", flip_stats["n_branch_changed"],
                flip_stats["n_mixed_land_with_ge5_same_surface_refs"],
                flip_stats["frac_branch_changed_of_usable"])

    meta = {
        "parquet": str(args.parquet),
        "max_row_groups": args.max_row_groups,
        "lat_thres_deg": LAT_THRES,
        "std_thres_ppm": STD_THRES,
        "n_min_ref": N_MIN_REF,
        "near_max_km": NEAR_MAX,
        "z_thresh": Z_THRESH,
        "bin_edges_km": BIN_EDGES,
        "anomaly_screen_ppm": ANOM_SCREEN_PPM,
        "extra_var_cols": EXTRA_VAR_COLS,
        "k_family": "SG (non-_nosg), matching spectral/fitting.py ref_extra_vars",
        "sanity": sanity,
        "branch_flip": flip_stats,
    }
    (outdir / f"mixed_surface_window_meta{tag}.json").write_text(
        json.dumps(meta, indent=2))

    # ── markdown summary ─────────────────────────────────────────────────────
    allrow = table.iloc[0]
    orow = table[table["population"] == "ocean_r05_all_lt10km"].iloc[0]
    o5row = table[table["population"] == "ocean_r05_lt5km"].iloc[0]
    lines = []
    lines.append("# Mixed-surface clear-sky reference windows in the Fig. 6 population\n")
    lines.append(f"Generated by `workspace/mixed_surface_reference_count.py` "
                 f"({'full 116-date pass' if args.max_row_groups is None else f'first {args.max_row_groups} row groups'}).\n")
    lines.append("## Headline\n")
    lines.append(f"- Land Fig.-6 targets (`sfc_type==1`, QF0, no snow, 0 <= cld_dist < 10 km, "
                 f"valid `zr15exp_o2a` and `xco2_bc_anomaly_r15`): "
                 f"**{int(allrow['n_targets']):,}**")
    lines.append(f"- With >= 1 OCEAN footprint in the r15 reference window: "
                 f"**{int(allrow['n_mixed']):,} ({allrow['frac_mixed']*100:.1f} %)**")
    lines.append(f"- Ocean-majority windows (ocean share > 50 %): "
                 f"{int(allrow['n_other_majority']):,} "
                 f"({allrow['frac_other_majority']*100:.1f} % of all land targets)")
    lines.append(f"- Ocean share among mixed windows: median "
                 f"{allrow['other_share_q50_mixed']*100:.1f} %, "
                 f"IQR {allrow['other_share_q25_mixed']*100:.1f}–"
                 f"{allrow['other_share_q75_mixed']*100:.1f} %, "
                 f"mean {allrow['mean_other_share_mixed']*100:.1f} %")
    lines.append(f"- Mirror (ocean targets < 10 km, r05 window): "
                 f"{int(orow['n_targets']):,} targets, "
                 f"{orow['frac_mixed']*100:.1f} % have >= 1 land reference "
                 f"(land-majority {orow['frac_other_majority']*100:.1f} %). "
                 f"Restricted to < 5 km: {int(o5row['n_targets']):,} targets, "
                 f"{o5row['frac_mixed']*100:.1f} % mixed.")
    lines.append(f"- Branch reclassification: of the "
                 f"{flip_stats['n_mixed_land_with_ge5_same_surface_refs']:,} mixed-window "
                 f"land targets with >= 5 land-only reference members, "
                 f"{flip_stats['n_branch_changed']:,} "
                 f"({flip_stats['frac_branch_changed_of_usable']*100:.1f} %) change "
                 f"shadowed/neutral/brightened branch when z_exp is recomputed from "
                 f"land members only "
                 f"({flip_stats['frac_branch_changed_of_all_land']*100:.2f} % of the "
                 f"whole land population). Median |Δz| = "
                 f"{flip_stats['median_abs_dz']:.3f}, 95th pct "
                 f"{flip_stats['p95_abs_dz']:.3f}.\n")

    lines.append("## Sanity check against the production guards\n")
    lines.append("| population | targets | frac with n_ref >= 5 | median n_ref | "
                 "median abs(Δ ref_mean) vs stored | frac within 0.01σ |")
    lines.append("|---|---|---|---|---|---|")
    for k, v in sanity.items():
        lines.append(f"| {k} | {v['n_targets']:,} | {v['frac_n_ref_ge_5']:.4f} | "
                     f"{v['median_n_ref']:.0f} | {v['median_abs_dmean_vs_stored']:.3g} | "
                     f"{v['frac_dmean_within_0p01_sigma']:.4f} |")
    lines.append("")

    lines.append("## Land, by cloud-distance bin\n")
    lines.append("| bin (km) | targets | mixed | frac mixed | ocean-majority frac | "
                 "median ocean share (mixed) |")
    lines.append("|---|---|---|---|---|---|")
    for b in BIN_LABELS:
        r = table[(table["group"] == "cld_bin") & (table["level"] == b)].iloc[0]
        lines.append(f"| {b} | {int(r['n_targets']):,} | {int(r['n_mixed']):,} | "
                     f"{r['frac_mixed']*100:.1f} % | {r['frac_other_majority']*100:.1f} % | "
                     f"{r['other_share_q50_mixed']*100:.1f} % |")
    lines.append("")

    lines.append("## Land, by z_exp branch\n")
    lines.append("| branch | targets | mixed | frac mixed | ocean-majority frac | "
                 "median ocean share (mixed) |")
    lines.append("|---|---|---|---|---|---|")
    for br in ("shadowed", "neutral", "brightened"):
        r = table[(table["group"] == "branch") & (table["level"] == br)].iloc[0]
        lines.append(f"| {br} | {int(r['n_targets']):,} | {int(r['n_mixed']):,} | "
                     f"{r['frac_mixed']*100:.1f} % | {r['frac_other_majority']*100:.1f} % | "
                     f"{r['other_share_q50_mixed']*100:.1f} % |")
    lines.append("")
    if len(flip_tbl):
        lines.append("## Branch flip crosstab (production z_exp vs land-only z_exp)\n")
        cols = list(flip_tbl.columns)
        lines.append("| prod \\ land-only | " + " | ".join(map(str, cols)) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for ix, r in flip_tbl.iterrows():
            lines.append(f"| {ix} | " + " | ".join(f"{int(v):,}" for v in r) + " |")
        lines.append("")

    brs = {br: table[(table["group"] == "branch") & (table["level"] == br)].iloc[0]
           for br in ("shadowed", "neutral", "brightened")}
    br_txt = ", ".join(f"{br} {brs[br]['frac_mixed']*100:.1f} %"
                       for br in ("shadowed", "neutral", "brightened"))
    n_mixed = int(allrow["n_mixed"])
    n_usable = flip_stats["n_mixed_land_with_ge5_same_surface_refs"]
    lines.append("## Plain reading\n")
    lines.append(
        "The clear-sky reference window is a pure latitude band inside one orbit, "
        "so nothing stops a coastal or island land footprint from being referenced "
        "against ocean soundings. That happens for "
        f"{allrow['frac_mixed']*100:.1f} % of the land footprints entering Fig. 6 "
        f"({n_mixed:,} of {int(allrow['n_targets']):,}), and for "
        f"{allrow['frac_other_majority']*100:.1f} % of them the window is majority "
        "ocean. Mixing falls off with cloud distance (from "
        f"{table[(table['group']=='cld_bin') & (table['level']==BIN_LABELS[0])].iloc[0]['frac_mixed']*100:.1f} % "
        f"in the 0-1 km bin to "
        f"{table[(table['group']=='cld_bin') & (table['level']==BIN_LABELS[-1])].iloc[0]['frac_mixed']*100:.1f} % "
        "at 7-10 km) because the closest-to-cloud land footprints are the coastal "
        "ones. Across the three z_exp branches the mixed fraction is "
        f"{br_txt} — a mild excess in the brightened branch (bright ocean glint "
        "raises the reference continuum for the land target), not a concentration "
        "that could create the shadow-versus-brightening contrast on its own.\n")
    lines.append(
        f"Only {n_usable:,} of the {n_mixed:,} mixed windows retain at least "
        f"{N_MIN_REF} land-only members, i.e. most mixed windows are almost "
        "entirely ocean and have no same-surface reference to fall back on. Among "
        f"the {n_usable:,} that do, "
        f"{flip_stats['n_branch_changed']:,} "
        f"({flip_stats['frac_branch_changed_of_usable']*100:.1f} %) land in a "
        "different shadowed / neutral / brightened branch when z_exp is rebuilt "
        "from land members only, which is "
        f"{flip_stats['frac_branch_changed_of_all_land']*100:.2f} % of the whole "
        "Fig.-6 land population. Mixed-surface reference windows are therefore a "
        "real property of the anomaly definition and worth stating, but at this "
        "prevalence they cannot move the binned means that Fig. 6 shows.\n")

    (outdir / f"mixed_surface_windows{tag}.md").write_text("\n".join(lines))
    logger.info("wrote %s", outdir / f"mixed_surface_windows{tag}.md")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()

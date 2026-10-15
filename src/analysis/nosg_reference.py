"""Clear-sky reference statistics of the unsmoothed (``_nosg``) path-length fit.

The per-orbit fit (``spectral.fitting``) stores the clear-sky reference mean and
std of the SG-smoothed k1-k3 and gamma only.  This module recomputes the same
statistics for the unsmoothed twin from the combined training parquet, so the
phenomenology analyses can use the fit that Sect. 2.2 defines and the models use.

The reference rules are those of ``spectral.anomaly.compute_xco2_anomaly``:
per orbit, the reference of sounding i is every clear-sky sounding with
|lat_i - lat_j| <= lat_thres, cld_dist_km > min_cld_dist, finite xco2_bc and
finite values of every reference variable (one shared pool); at least five
references and a reference xco2 std <= std_thres are required.  The pool is
built from the fit-time variables (smoothed k1-k3, albedo, gamma) plus the
_nosg variables, whose NaN pattern is identical, so it equals the fit-time pool.
``clear_sky_reference`` gets the same answer in O(N log N) from sorted
latitudes and prefix sums instead of the O(N^2) pairwise window.

Every date is gated: the recomputed xco2_bc anomaly and smoothed statistics
must reproduce the stored columns, or nothing is written.

Output: a row-aligned sidecar ``<parquet stem>_nosg_ref.parquet`` (one row group
per date, ``fp_id`` + float32 columns) and ``<stem>_nosg_ref_gate.csv``.
Columns, for prefix p in {ref (10 km), r05, r15} and band b:
    exp_{b}_intercept_nosg
    {p}_{b}_k{1,2,3}_nosg_{mean,std}
    {p}_exp_int_{b}_nosg_{mean,std}
``read_combined`` joins the sidecar back onto the main parquet by row, after
checking that ``fp_id`` agrees.

Run:
    PYTHONPATH=src python -m analysis.nosg_reference            # all dates
    PYTHONPATH=src python -m analysis.nosg_reference --dates 2  # gate only, no output
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from constants import anomaly_args  # noqa: E402

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = REPO / "results" / "csv_collection" / "combined_2016_2020_dates.parquet"

BANDS = ("o2a", "wco2", "sco2")
MOMENTS = (1, 2, 3)
# prefix of the stored column -> min_cld_dist (km) of that reference set
REFERENCE_SETS = {"ref": 10.0, "r05": 5.0, "r15": 15.0}
# stored anomaly column of each reference set (the gate target)
ANOMALY_COLUMNS = {"ref": "xco2_bc_anomaly", "r05": "xco2_bc_anomaly_r05",
                   "r15": "xco2_bc_anomaly_r15"}

# Gate tolerances.  The stored columns are float32 and the window rule compares
# float32 latitudes, so a few soundings may sit exactly on a window or std edge.
_ANOMALY_ATOL = 5e-4          # ppm (float32 ulp of ~400 ppm is 3e-5)
_STAT_RTOL, _STAT_ATOL = 1e-4, 1e-6
# A window of identical values has std 0; prefix-sum rounding leaves var ~1e-12
# there, i.e. std ~1e-6.
_STD_ATOL = 1e-5
# Per (reference, column), over all dates: soundings whose validity differs from
# the stored one, and soundings outside the tolerance.  The first full run
# (2026-10-14) had 0 validity differences and, beyond the std rounding above,
# one orbit (2019-11-01 28369a, 143 soundings at 36.9-37.3 N) whose stored 10 km
# ('ref') statistics differ by up to 0.025 ppm while its r05/r15 statistics
# reproduce; that is 1.6e-5 of the 10 km set.  Any date with mismatches is logged.
_MAX_EDGE_FRACTION = 1e-4
_MAX_OUT_FRACTION = 1e-4


def _stat_stem(column: str) -> str:
    """Reference-stat stem of an observation column (fit.py naming)."""
    if column.startswith("exp_") and "_intercept" in column:
        band = column.split("_")[1]                 # exp_o2a_intercept[_nosg]
        return f"exp_int_{band}" + ("_nosg" if column.endswith("_nosg") else "")
    if column.startswith("alb_"):
        return column
    return column                                   # o2a_k1[_nosg]


def smoothed_columns() -> list[str]:
    """Fit-time reference variables: they define the shared reference pool."""
    return ([f"{b}_k{k}" for b in BANDS for k in MOMENTS]
            + [f"alb_{b}" for b in BANDS]
            + [f"exp_{b}_intercept" for b in BANDS])


def nosg_columns() -> list[str]:
    return ([f"{b}_k{k}_nosg" for b in BANDS for k in MOMENTS]
            + [f"exp_{b}_intercept_nosg" for b in BANDS])


def output_columns() -> list[str]:
    cols = [f"exp_{b}_intercept_nosg" for b in BANDS]
    for p in REFERENCE_SETS:
        for c in nosg_columns():
            cols += [f"{p}_{_stat_stem(c)}_mean", f"{p}_{_stat_stem(c)}_std"]
    return cols


# ── reference statistics ──────────────────────────────────────────────────────

def _window_bounds(sorted_lat: np.ndarray, q: np.ndarray,
                   lat_thres: float) -> tuple[np.ndarray, np.ndarray]:
    """[lo, hi) of the sorted references with |q - lat| <= lat_thres.

    The predicate is evaluated exactly as compute_xco2_anomaly does (in the
    dtype of the latitudes), so window edges match it bit for bit.  The
    searchsorted bounds are widened by a margin far above float32 rounding and
    then tightened with that predicate.
    """
    thr = sorted_lat.dtype.type(lat_thres)
    margin = 1e-3
    lo = np.searchsorted(sorted_lat, q - (lat_thres + margin), side="left")
    hi = np.searchsorted(sorted_lat, q + (lat_thres + margin), side="right")
    n = len(sorted_lat)
    while True:
        idx = np.minimum(lo, n - 1)
        step = (lo < hi) & (np.abs(q - sorted_lat[idx]) > thr)
        if not step.any():
            break
        lo = lo + step
    while True:
        idx = np.maximum(hi - 1, 0)
        step = (hi > lo) & (np.abs(q - sorted_lat[idx]) > thr)
        if not step.any():
            break
        hi = hi - step
    return lo, hi


def _window_mean_std(values: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Population mean and std of values[lo:hi] for every window (n > 0).

    Centering on the pool mean keeps the prefix-sum cancellation error at
    ~1e-16 of the pool variance, far below float32 storage precision.
    """
    x = values.astype(np.float64)
    c = x.mean() if len(x) else 0.0
    x = x - c
    s1 = np.concatenate(([0.0], np.cumsum(x)))
    s2 = np.concatenate(([0.0], np.cumsum(x * x)))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (s1[hi] - s1[lo]) / n
        var = (s2[hi] - s2[lo]) / n - m * m
    return m + c, np.sqrt(np.maximum(var, 0.0))


def clear_sky_reference(lat: np.ndarray, cld_dist_km: np.ndarray,
                        xco2: np.ndarray, extra_vars: Mapping[str, np.ndarray],
                        *, lat_thres: float, std_thres: float,
                        min_cld_dist: float,
                        ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Same result as ``compute_xco2_anomaly(..., extra_vars=extra_vars)``.

    Every array is one orbit.  Returns (anomaly, means, stds), NaN where the
    reference is missing (< 5 soundings) or unstable (xco2 std > std_thres).
    """
    N = len(lat)
    valid_lat = ~np.isnan(lat)
    pool = valid_lat & (cld_dist_km > min_cld_dist) & ~np.isnan(xco2)
    for v in extra_vars.values():
        pool &= ~np.isnan(np.asarray(v))

    order = np.flatnonzero(pool)
    order = order[np.argsort(lat[order], kind="stable")]
    ref_lat = lat[order]

    anomaly = np.full(N, np.nan)
    means = {k: np.full(N, np.nan) for k in extra_vars}
    stds = {k: np.full(N, np.nan) for k in extra_vars}
    if len(order) == 0:
        return anomaly, means, stds

    q_idx = np.flatnonzero(valid_lat)
    lo, hi = _window_bounds(ref_lat, lat[q_idx], lat_thres)
    n = (hi - lo).astype(np.float64)
    has_refs = n >= 5

    x_mean, x_std = _window_mean_std(xco2[order], lo, hi, n)
    ok = has_refs & (x_std <= std_thres)
    rows = q_idx[ok]
    anomaly[rows] = xco2[rows] - x_mean[ok]
    for k, v in extra_vars.items():
        m, s = _window_mean_std(np.asarray(v)[order], lo, hi, n)
        means[k][rows] = m[ok]
        stds[k][rows] = s[ok]
    return anomaly, means, stds


# ── per-date driver ───────────────────────────────────────────────────────────

def _close(new: np.ndarray, stored: np.ndarray, rtol: float, atol: float) -> dict:
    """Agreement of a recomputed column with the stored one."""
    fn, fs = np.isfinite(new), np.isfinite(stored)
    both = fn & fs
    err = np.abs(new[both] - stored[both])
    tol = atol + rtol * np.abs(stored[both])
    return {"n": int(fs.sum()), "edge": int((fn != fs).sum()),
            "max_abs": float(err.max()) if err.size else 0.0,
            "n_out": int((err > tol).sum())}


def process_date(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """New columns (row-aligned with df) and the gate rows of one date."""
    out = {c: np.full(len(df), np.nan) for c in output_columns()}
    for b in BANDS:
        out[f"exp_{b}_intercept_nosg"] = np.exp(df[f"{b}_intercept_nosg"].to_numpy(np.float64))

    sources = {c: df[c].to_numpy() for c in smoothed_columns()}
    for b in BANDS:
        for k in MOMENTS:
            sources[f"{b}_k{k}_nosg"] = df[f"{b}_k{k}_nosg"].to_numpy()
        # exp computed in float32, as the stored smoothed gamma is
        sources[f"exp_{b}_intercept_nosg"] = np.exp(df[f"{b}_intercept_nosg"].to_numpy())

    xco2 = df["xco2_bc"].to_numpy().copy()
    xco2[xco2 <= 0] = np.nan
    lat = df["lat"].to_numpy()
    cld = df["cld_dist_km"].to_numpy()
    gate_acc: dict[tuple[str, str], list[np.ndarray]] = {}

    for _, idx in df.groupby("orbit_id", sort=False).indices.items():
        for p, d in REFERENCE_SETS.items():
            a, m, s = clear_sky_reference(
                lat[idx], cld[idx], xco2[idx], {k: v[idx] for k, v in sources.items()},
                **anomaly_args(min_cld_dist=d))
            for c in nosg_columns():
                st = _stat_stem(c)
                out[f"{p}_{st}_mean"][idx] = m[c]
                out[f"{p}_{st}_std"][idx] = s[c]
            gate_acc.setdefault((p, "anomaly"), []).append((idx, a))
            for c in smoothed_columns():
                st = _stat_stem(c)
                gate_acc.setdefault((p, f"{st}_mean"), []).append((idx, m[c]))
                gate_acc.setdefault((p, f"{st}_std"), []).append((idx, s[c]))

    gate = []
    for (p, what), parts in gate_acc.items():
        new = np.full(len(df), np.nan)
        for idx, v in parts:
            new[idx] = v
        if what == "anomaly":
            stored_col = ANOMALY_COLUMNS[p]
            res = _close(new, df[stored_col].to_numpy(np.float64), 0.0, _ANOMALY_ATOL)
        else:
            stored_col = f"{p}_{what}"
            atol = _STD_ATOL if what.endswith("_std") else _STAT_ATOL
            res = _close(new, df[stored_col].to_numpy(np.float64), _STAT_RTOL, atol)
        gate.append({"reference": p, "column": stored_col, **res})

    frame = pd.DataFrame({"fp_id": df["fp_id"].to_numpy()})
    for c in output_columns():
        frame[c] = out[c].astype(np.float32)
    return frame, gate


def gate_failures(gate: pd.DataFrame) -> pd.DataFrame:
    g = gate.groupby(["reference", "column"], as_index=False)[["n", "edge", "n_out"]].sum()
    g["max_abs"] = gate.groupby(["reference", "column"])["max_abs"].max().to_numpy()
    n = g["n"].clip(lower=1)
    bad = (g["n_out"] > _MAX_OUT_FRACTION * n) | (g["edge"] > _MAX_EDGE_FRACTION * n)
    return g[bad]


def _input_columns() -> list[str]:
    return sorted(set(["date", "orbit_id", "fp_id", "lat", "cld_dist_km", "xco2_bc"]
                      + list(ANOMALY_COLUMNS.values()) + smoothed_columns()
                      + [f"{b}_k{k}_nosg" for b in BANDS for k in MOMENTS]
                      + [f"{b}_intercept_nosg" for b in BANDS]
                      + [f"{p}_{_stat_stem(c)}_{s}" for p in REFERENCE_SETS
                         for c in smoothed_columns() for s in ("mean", "std")]))


def sidecar_path(parquet: Path) -> Path:
    return parquet.with_name(parquet.stem + "_nosg_ref.parquet")


def build(parquet: Path, n_dates: int | None = None) -> Path | None:
    """Write the sidecar (all dates) or, with n_dates, only run the gate."""
    pf = pq.ParquetFile(parquet)
    cols = _input_columns()
    n_groups = pf.metadata.num_row_groups if n_dates is None else min(n_dates, pf.metadata.num_row_groups)
    side = sidecar_path(parquet)
    tmp = side.with_suffix(".parquet.partial")
    writer = None
    gates = []
    t0 = time.time()
    try:
        for i in range(n_groups):
            df = pf.read_row_group(i, columns=cols).to_pandas()
            date = df["date"].iloc[0]
            date = date.decode() if isinstance(date, bytes) else str(date)
            frame, gate = process_date(df)
            gates += [{"date": date, **g} for g in gate]
            if n_dates is None:
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=len(frame))
            logger.info("%3d/%d %s  %7d rows  %.0f s", i + 1, n_groups, date,
                        len(df), time.time() - t0)
    finally:
        if writer is not None:
            writer.close()

    gate_df = pd.DataFrame(gates)
    gate_csv = parquet.with_name(parquet.stem + "_nosg_ref_gate.csv")
    gate_df.to_csv(gate_csv, index=False)
    flagged = gate_df[(gate_df["n_out"] > 0) | (gate_df["edge"] > 0)]
    for (date, ref), g in flagged.groupby(["date", "reference"]):
        logger.warning("%s %s: %d column(s) with mismatches, up to %d soundings, max |diff| %.2g",
                       date, ref, len(g), max(g["n_out"].max(), g["edge"].max()),
                       g["max_abs"].max())
    bad = gate_failures(gate_df)
    if len(bad):
        logger.error("gate FAILED (see %s):\n%s", gate_csv, bad.to_string(index=False))
        if tmp.exists():
            tmp.unlink()
        raise SystemExit(1)
    logger.info("gate passed for %d dates (%s)", n_groups, gate_csv)
    if n_dates is not None:
        return None
    tmp.replace(side)
    logger.info("wrote %s", side)
    return side


# ── reader ────────────────────────────────────────────────────────────────────

def read_combined(parquet: Path | str, columns: Iterable[str]) -> pd.DataFrame:
    """Read columns of the combined parquet, taking _nosg reference columns
    from its sidecar when the main file does not have them.

    The sidecar is row-aligned with the parquet it was built from; fp_id is
    compared row by row so a rebuilt main parquet cannot be silently misjoined.
    """
    parquet = Path(parquet)
    columns = list(dict.fromkeys(columns))
    main_names = set(pq.read_schema(parquet).names)
    from_side = [c for c in columns if c not in main_names]
    from_main = [c for c in columns if c in main_names]
    if not from_side:
        return pq.read_table(parquet, columns=from_main).to_pandas()

    side = sidecar_path(parquet)
    if not side.exists():
        raise FileNotFoundError(
            f"{from_side[:3]}... are not in {parquet.name} and {side.name} is missing; "
            "run  PYTHONPATH=src python -m analysis.nosg_reference")
    side_names = set(pq.read_schema(side).names)
    missing = [c for c in from_side if c not in side_names]
    if missing:
        raise KeyError(f"columns in neither {parquet.name} nor {side.name}: {missing}")

    key_main = "fp_id" in from_main
    main = pq.read_table(parquet, columns=from_main + ([] if key_main else ["fp_id"]))
    extra = pq.read_table(side, columns=["fp_id"] + from_side)
    if main.num_rows != extra.num_rows or not np.array_equal(
            main.column("fp_id").to_numpy(), extra.column("fp_id").to_numpy()):
        raise ValueError(f"{side.name} is not row-aligned with {parquet.name}; rebuild it")
    df = main.to_pandas()
    if not key_main:
        df = df.drop(columns="fp_id")
    for c in from_side:
        df[c] = extra.column(c).to_numpy()
    return df[columns]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--dates", type=int, default=None,
                    help="gate the first N dates only; write nothing")
    args = ap.parse_args(argv)
    build(args.parquet, args.dates)


if __name__ == "__main__":
    main()

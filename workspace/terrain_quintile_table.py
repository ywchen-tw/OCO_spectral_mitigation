#!/usr/bin/env python3
"""Far-cloud land footprints: path-length departures by terrain-roughness quintile.

Population: land, snow-free, nearest-cloud distance >= 30 km, training dates
(combined_2016_2020_dates.parquet). Roughness is alt_std, the sub-footprint
elevation spread (m). Quintile edges are common to QF = 0 and QF = 1 (computed
on the pooled population).

Each statistic is the departure from the same-orbit r15 clear-sky reference
mean, in the units of the statistic (not standardized), so the values compare
directly with the near-cloud departures of Sect. 5 of the main article.
Standard errors are clustered by date: SE = SD of the per-date means / sqrt(n_dates).

FIT SCHEMA (2026-11-02): ``--fit-schema v2`` (default) reads the agreed-model
``_am`` coefficients of the Doppler refit (results_doppler/): <l'> = <band>_k1_am,
var(l') = <band>_k2diag_am (O2A, WCO2) / sco2_k2_am, gamma = o2a_exp_int_am, with
the r15_<band>_<stub>_am_mean references (the mapping of
analysis.spec_sensitivity.V2_COEFS).  A footprint needs finite printed
statistics (k1 of the three bands, O2A gamma) and alt_std; var(l') is kept in
the CSV but may be NaN (the O2A k2diag refit can fail on its own).
``--fit-schema v1`` reproduces the pre-2026-11-02 edition (unsmoothed
order-7/3/7 fit, ``_nosg`` sidecar of results/csv_collection), with its
all-statistics finiteness rule.

Outputs (v2; v1 under results/ instead of results_doppler/):
  results_doppler/figures/cld_dist_analysis/alt_std_terrain/farfield_dk1_by_altstd_quintile_qf01.csv
  manuscript/JQSRT_draft/tables/tabA2_terrain_quintiles.tex   (SI table)

Run:  python workspace/terrain_quintile_table.py              (parquet -> CSV -> .tex)
      python workspace/terrain_quintile_table.py --from-csv   (CSV -> .tex only)
      [--tex PATH]  write the table elsewhere, e.g. for a verification diff
      [--fit-schema v1|v2] [--parquet PATH] [--csv PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Volumes/YWC_SSD/programming/oco_fp_analysis")
sys.path.insert(0, str(REPO / "src"))
FIT = "_nosg"  # v1 only: constants.PATH_LENGTH_FIT_SUFFIX of the v1 edition
_CSV_REL = "figures/cld_dist_analysis/alt_std_terrain/farfield_dk1_by_altstd_quintile_qf01.csv"
# results tree of each fit schema: v1 = order-7/3/7 fit, v2 = agreed model (Doppler refit)
RESULTS = {"v1": REPO / "results", "v2": REPO / "results_doppler"}
PARQUETS = {k: v / "csv_collection/combined_2016_2020_dates.parquet" for k, v in RESULTS.items()}
OUTS = {k: v / _CSV_REL for k, v in RESULTS.items()}
TEX = REPO / "manuscript/JQSRT_draft/tables/tabA2_terrain_quintiles.tex"
FAR_KM = 30.0
BANDS = ["o2a", "wco2", "sco2"]
# v2 var(l') stub per band (analysis.spec_sensitivity._V2_VAR_L): O2A and WCO2 fix
# k2 = 0 in the fit, so their var(l') is the k2diag refit; SCO2 fits k2.
V2_VAR_L = {"o2a": "k2diag", "wco2": "k2diag", "sco2": "k2"}

def _tex_rows(k1_exp):
    """(CSV statistic, row label, scale to the printed unit); the k1 rows in
    units of 10^-k1_exp."""
    unit = f"($\\times10^{{-{k1_exp}}}$)"
    return [
        ("d_k1_o2a", f"\\ce{{O2}}~A-band $\\Delta\\meanl$ {unit}", 10.0 ** k1_exp),
        ("d_k1_wco2", f"W\\ce{{CO2}} $\\Delta\\meanl$ {unit}", 10.0 ** k1_exp),
        ("d_k1_sco2", f"S\\ce{{CO2}} $\\Delta\\meanl$ {unit}", 10.0 ** k1_exp),
        ("d_gamma_rel_o2a", "\\ce{O2}~A-band $\\Delta\\gamma/\\gamma$ (\\%)", 1e2),
    ]


# The agreed-model k1 departures far from cloud are about ten times smaller than
# those of the order-7/3/7 fit (v1: 1e-4..4e-3; v2: 1e-5..3e-4), so v2 prints
# them in 10^-4 to keep one significant decimal.
TEX_ROWS_BY_SCHEMA = {"v1": _tex_rows(3), "v2": _tex_rows(4)}
TEX_ROWS = TEX_ROWS_BY_SCHEMA["v2"]


def _stat_columns(fit_schema):
    """{stat: (observed column, r15 reference-mean column)} of a fit schema;
    'gamma' is the O2A continuum reflectance."""
    if fit_schema == "v1":
        cols = {f"k{m}_{b}": (f"{b}_k{m}{FIT}", f"r15_{b}_k{m}{FIT}_mean")
                for b in BANDS for m in (1, 2)}
        cols["gamma"] = (f"exp_o2a_intercept{FIT}", f"r15_exp_int_o2a{FIT}_mean")
        return cols
    cols = {}
    for b in BANDS:
        for m, stub in ((1, "k1"), (2, V2_VAR_L[b])):
            cols[f"k{m}_{b}"] = (f"{b}_{stub}_am", f"r15_{b}_{stub}_am_mean")
    cols["gamma"] = ("o2a_exp_int_am", "r15_o2a_exp_int_am_mean")
    return cols


def _read(path, cols, fit_schema):
    if fit_schema == "v1":
        from analysis.nosg_reference import read_combined  # joins the _nosg sidecar
        return read_combined(path, cols)
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=cols).to_pandas()


def compute(fit_schema="v2", parquet=None, out_csv=None):
    parquet = parquet or PARQUETS[fit_schema]
    out_csv = out_csv or OUTS[fit_schema]
    sc = _stat_columns(fit_schema)
    cols = (["date", "sfc_type", "cld_dist_km", "xco2_qf", "snow_flag", "alt_std"]
            + [c for pair in sc.values() for c in pair])
    df = _read(parquet, cols, fit_schema)
    print(f"{len(df):,} rows read from {parquet} (fit schema {fit_schema})")
    df = df[(df.sfc_type == 1) & (df.snow_flag == 0) & (df.cld_dist_km >= FAR_KM)]
    if fit_schema == "v2":
        # alt_std carries the Lite fill -999999 on a few footprints (351 of the
        # v2 population), which v1 sorted into the lowest quintile.
        df = df[df.alt_std > -9.9e4]

    stats = {}
    for b in BANDS:
        for m in (1, 2):
            obs, ref = sc[f"k{m}_{b}"]
            stats[f"d_k{m}_{b}"] = df[obs] - df[ref]
    obs, ref = sc["gamma"]
    stats["d_gamma_rel_o2a"] = df[obs] / df[ref] - 1.0
    s = pd.DataFrame(stats)
    s["date"] = df["date"].values
    s["qf"] = df["xco2_qf"].values
    s["alt_std"] = df["alt_std"].values
    # v1: every statistic finite (k2 is NaN exactly where k1 is).  v2: only the
    # printed ones -- a failed O2A k2diag refit must not drop the footprint.
    need = ([c for c in s.columns if c != "date"] if fit_schema == "v1"
            else [stat for stat, *_ in TEX_ROWS_BY_SCHEMA[fit_schema]]
            + ["qf", "alt_std"])
    s = s[np.isfinite(s[need].astype(float)).all(axis=1)]

    edges = np.quantile(s.alt_std, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    s["q"] = pd.cut(s.alt_std, edges, labels=False, include_lowest=True)
    print("alt_std quintile edges (m):", np.round(edges, 2))

    rows = []
    value_cols = [c for c in stats]
    for (qf, q), sub in s.groupby(["qf", "q"]):
        per_date = sub.groupby("date")[value_cols].mean()
        row = {"qf": int(qf), "quintile": int(q) + 1, "n": len(sub),
               "n_dates": len(per_date),
               "alt_std_lo": edges[int(q)], "alt_std_hi": edges[int(q) + 1]}
        for c in value_cols:
            row[f"{c}_mean"] = sub[c].mean()
            row[f"{c}_se"] = per_date[c].std(ddof=1) / np.sqrt(len(per_date))
        rows.append(row)
    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250)
    for qf in (0, 1):
        t = out[out.qf == qf]
        print(f"\nQF = {qf}")
        for c in value_cols:
            print(f"  {c:18s}", "  ".join(
                f"{m:+.4f}±{e:.4f}" for m, e in zip(t[f"{c}_mean"], t[f"{c}_se"])),
                "  n =", list(t.n))
    print("written", out_csv)
    return out


def _signed(x):
    """One-decimal value with a true minus; 0.0 unsigned."""
    mag = f"{abs(x):.1f}"
    if float(mag) == 0.0:
        return mag
    return ("$-$" if x < 0 else "+") + mag


def tex_table(out, source_csv=OUTS["v2"], fit_schema="v2"):
    out = out.sort_values(["qf", "quintile"])
    q0 = out[out.qf == 0]
    # inner edges only: the outer ones are the sample extremes. pd.cut bins
    # are right-closed: [min, e1], (e1, e2], ..., (e4, max]
    e = [f"{x:.1f}" for x in q0.alt_std_hi.to_numpy()[:-1]]
    heads = ([f"$\\leq$\\,{e[0]}"] + [f"{a}--{b}" for a, b in zip(e[:-1], e[1:])]
             + [f"$>$\\,{e[-1]}"])
    n_dates = int(out.n_dates.max())
    lines = [
        "% Generated by workspace/terrain_quintile_table.py -- do not hand-edit.",
        f"% Source: {Path(source_csv).relative_to(REPO)}",
        "% (land, snow-free, nearest-cloud distance >= 30 km, training dates; "
        "SE clustered by date).",
        "\\begin{table}[t]",
        "\\caption{Departures of \\meanl and $\\gamma$ from the clear-sky "
        "reference ($d_0$ = 15\\,km) for snow-free land footprints with a "
        "nearest-cloud distance of at least 30\\,km, by quintile of the "
        "sub-footprint surface-elevation standard deviation (SD). Values are "
        "the mean\\,$\\pm$\\,one standard error, computed from the spread of "
        f"the daily means over the {n_dates} training dates. The quintile "
        "edges are computed on the footprints of both quality flags together.}",
        "\\label{tab:terrain}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{lccccc}",
        "\\tophline",
        "Elevation SD (m) & " + " & ".join(heads) + " \\\\",
    ]
    for qf in (0, 1):
        t = out[out.qf == qf]
        lines += ["\\middlehline",
                  f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{QF~=~{qf}}}}} \\\\",
                  "Footprints ($10^3$) & "
                  + " & ".join(f"{n / 1e3:.0f}" for n in t.n) + " \\\\"]
        for stat, label, scale in TEX_ROWS_BY_SCHEMA[fit_schema]:
            cells = [f"{_signed(m * scale)}\\,$\\pm$\\,{se * scale:.1f}"
                     for m, se in zip(t[f"{stat}_mean"], t[f"{stat}_se"])]
            lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomhline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-csv", action="store_true",
                    help="skip the parquet pass and build the table from the CSV")
    ap.add_argument("--tex", type=Path, default=TEX, help="output .tex path")
    ap.add_argument("--fit-schema", choices=sorted(RESULTS), default="v2",
                    help="v2 (default): agreed-model _am fit of results_doppler/; "
                         "v1: the pre-2026-11-02 order-7/3/7 _nosg fit of results/")
    ap.add_argument("--parquet", type=Path, default=None,
                    help="training parquet (default: the fit schema's "
                         "csv_collection/combined_2016_2020_dates.parquet)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="statistics CSV (default: the fit schema's "
                         + _CSV_REL + ")")
    args = ap.parse_args()
    csv = args.csv or OUTS[args.fit_schema]
    out = (pd.read_csv(csv) if args.from_csv
           else compute(args.fit_schema, args.parquet, csv))
    args.tex.write_text(tex_table(out, csv, args.fit_schema))
    print("written", args.tex)


if __name__ == "__main__":
    main()

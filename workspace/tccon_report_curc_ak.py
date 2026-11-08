"""tccon_report_curc_ak.py — rerun tccon_comparison_report.py locally on a CURC
evaluation tree, taking the AK-harmonized TCCON reference of every station-day
from that tree's own CURC per-case CSV.

Why: the AK harmonization needs the column operators of the day OCO-2 Lite
files, which are not on the local disk (a plain local run harmonizes only a few
cases).  The AK step only replaces the TCCON window mean of a station-day by a
scalar (tccon_mu = tccon_mu_raw + ak_delta), so a local rerun that reads that
scalar from <tree>/tccon_comparison_r100km.csv is otherwise the report itself:
collocation, xco2_qf merge, guards, --min-fp and the cloud-distance split are
the report's own code.  Station-days without a CURC AK reference (ak_delta NaN)
fall back to the raw window mean, as on CURC.

Use it to add cloud-distance editions a tree was not delivered with, e.g. the
per-surface target-radius editions of the ablation arms:

    PYTHONPATH=src:workspace python workspace/tccon_report_curc_ak.py \
        --tree results_doppler/model_comparison/xgb/<tag>_no_contam/atrain \
        --cld-edges 0,5,inf --fname-suffix _cldo5_r100km

Checked 2026-11-03 against CURC output: the base tree's _cldo5 edition and the
no_contam arm's default (0,10,inf) edition are reproduced to <= 2e-14 ppm in
every RMSE / residual cell, with identical footprint and station-day counts.
Only CSV outputs are kept in --output-dir (default: the tree); the figures go
to a temporary directory unless --keep-figures is given.

--ak-csv takes the AK reference from another tree's per-case CSV, for a tree
that has plot_data but no report of its own (e.g. the smoother-null arms:
smoother columns added to the model tree's plot_data with
smoother_null.add_smoother_columns, --corr-col smoother_w30_corrected_xco2,
--ak-csv <model tree>/tccon_comparison_r100km.csv).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "workspace"), str(REPO / "src")]
import tccon_comparison_report as report  # noqa: E402


def _install_curc_ak(per_case_csv: Path) -> dict:
    """Patch the report's AK hooks so they return the CURC reference."""
    cmp = pd.read_csv(per_case_csv).drop_duplicates(["site", "date"])
    lookup = {(r.site, r.date): r for r in cmp.itertuples()}
    state = {"key": None, "missing": []}
    find_plotdata = report.find_plotdata

    def _find(out_base, date, site):
        state["key"] = (site, date)          # the case the loop is on
        return find_plotdata(out_base, date, site)

    def _ref(op, tpath, t0, t1, **kw):
        r = lookup.get(state["key"])
        if r is None or not np.isfinite(r.ak_delta):
            state["missing"].append(state["key"])
            return None
        return {"tccon_ref_ak": r.tccon_mu, "ak_delta": r.ak_delta,
                "n_lite": r.ak_n_lite}

    report.find_plotdata = _find
    report.operator_from_dataframe = lambda near: {"source": "curc"}
    report.ak_adjusted_ref_from_operator = _ref
    report.find_lite_file = lambda *a, **k: None
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, required=True,
                    help="CURC tree with combined_<date>_<site>/plot_data.parquet "
                         "and tccon_comparison_r100km.csv (e.g. .../atrain)")
    ap.add_argument("--cld-edges", required=True)
    ap.add_argument("--fname-suffix", required=True)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="where the CSVs go (default: --tree)")
    ap.add_argument("--corr-col", default="xgb_corrected_xco2")
    ap.add_argument("--ak-csv", type=Path, default=None,
                    help="per-case CSV holding the CURC AK reference "
                         "(default: <tree>/tccon_comparison_r100km.csv)")
    ap.add_argument("--script", type=Path,
                    default=REPO / "curc_shell_blanca_plot_corr_xco2_deepens.sh")
    ap.add_argument("--csv-collection", type=Path,
                    default=REPO / "results_doppler" / "csv_collection",
                    help="source parquets (xco2_raw, xco2_qf), linked as "
                         "results/csv_collection in the working directory")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--keep-figures", action="store_true")
    args = ap.parse_args()

    tree = args.tree.resolve()
    out_dir = (args.output_dir or tree).resolve()
    state = _install_curc_ak(args.ak_csv or tree / "tccon_comparison_r100km.csv")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # the report looks for source parquets under ./results/csv_collection
        (work / "results").mkdir()
        (work / "results" / "csv_collection").symlink_to(args.csv_collection.resolve())
        run_dir = out_dir if args.keep_figures else work / "out"
        import os
        cwd = os.getcwd()
        os.chdir(work)
        try:
            sys.argv = ["tccon_comparison_report.py",
                        "--script", str(args.script.resolve()),
                        "--out-base", str(tree), "--output-dir", str(run_dir),
                        "--corr-col", args.corr_col,
                        "--fname-suffix", args.fname_suffix,
                        "--exclude-sites", "ny", "--radius-km", "100",
                        "--window-min", "60", "--cld-edges", args.cld_edges,
                        "--ak-harmonize",
                        "--tccon-dir", str(REPO / "data" / "TCCON"),
                        "--n-boot", str(args.n_boot)]
            report.main()
        finally:
            os.chdir(cwd)
        if not args.keep_figures:
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted(run_dir.glob(f"*{args.fname_suffix}.csv")):
                shutil.copy2(p, out_dir / p.name)
                print("wrote", out_dir / p.name)
    print("station-days without a CURC AK reference (raw mean used):",
          sorted(set(state["missing"])))


if __name__ == "__main__":
    main()

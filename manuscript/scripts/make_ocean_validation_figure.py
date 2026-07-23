#!/usr/bin/env python3
"""Manuscript Fig. 10a/10b — ocean validation summaries (Results 4.6).

Regenerates the ATom and ship summary figures from the production-tree CSVs
with CONTINUOUS panel letters across the two-file composite — ATom (a)/(b),
ship (c)/(d) — and the stat-carrying suptitles removed (2026-07-23: the
suptitle numbers live in the §4.6 draft results text of the flow plan; the
manuscript caption rule forbids them in figures). Legends use the
X_CO2^B11 notation (the producers' next-re-render obligation, §6).

Reuses the producers' own plotting functions (patched 2026-07-23 with
panel_offset / suptitle / out_pdf kwargs), so the report-tree versions and
the manuscript versions cannot drift apart.

Inputs: <TAG>/atom/atom_pseudo_column_results.csv
        <TAG>/ship/ship_comparison_summary.csv
Output: manuscript/figures/fig11a_atom_summary.{png,pdf}
        manuscript/figures/fig11b_ship_summary.{png,pdf}
Also prints the suptitle-equivalent summary numbers for the draft text.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace"))  # plot_style, ak_harmonize
TAG = (REPO / "results" / "model_comparison" / "deep_ensemble"
       / "de_beta_nll_prof_reg_foldpca_o05l15_m5")
OUT = REPO / "manuscript" / "figures"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    atom = _load("atom_pseudo_column",
                 REPO / "workspace" / "ATom_analysis" / "atom_pseudo_column.py")
    ship = _load("plot_ship_summary",
                 REPO / "workspace" / "Ship_analysis" / "plot_ship_summary.py")
    from plot_style import XCO2_DE_LABEL
    atom.MODEL_LABEL = XCO2_DE_LABEL
    ship.MODEL_LABEL = XCO2_DE_LABEL

    OUT.mkdir(parents=True, exist_ok=True)
    da = pd.read_csv(TAG / "atom" / "atom_pseudo_column_results.csv")
    atom.make_summary_plot(da, OUT / "fig11a_atom_summary.png",
                           panel_offset=0, suptitle=False,
                           out_pdf=OUT / "fig11a_atom_summary.pdf")
    ds = pd.read_csv(TAG / "ship" / "ship_comparison_summary.csv")
    ship.make_summary_plot(ds, OUT / "fig11b_ship_summary.png",
                           panel_offset=2, suptitle=False,
                           out_pdf=OUT / "fig11b_ship_summary.pdf")

    # summary numbers formerly carried by the suptitles -> draft text
    nca = da[da.cld_med <= 10]
    print(f"\nATom: {len(da)} legs, {len(nca)} near-cloud; near-cloud mean "
          f"bias {nca.resid_bc.mean():+.2f}±{nca.resid_bc.std():.2f} → "
          f"{nca.resid_corr.mean():+.2f}±{nca.resid_corr.std():.2f} ppm; "
          f"pseudo-column σ {da.atom_ak_sd.min():.2f}–{da.atom_ak_sd.max():.2f}")
    ncs = ds[ds.cld_med <= 10]
    print(f"Ship: {len(ds)} cases, {len(ncs)} near-cloud "
          f"(near-cloud bias {ncs.resid_bc.mean():+.2f}→"
          f"{ncs.resid_corr.mean():+.2f} ppm); all-case mean bias "
          f"{ds.resid_bc.mean():+.2f}±{ds.resid_bc.std():.2f} → "
          f"{ds.resid_corr.mean():+.2f}±{ds.resid_corr.std():.2f} ppm; "
          f"mean OCO σ {ds.oco_bc_sd.mean():.2f}→{ds.oco_corr_sd.mean():.2f}; "
          f"mean ship σ {ds.ship_err.mean():.2f}")


if __name__ == "__main__":
    main()

"""nassar_channel_attribution.py — which feature channel does the clear-sky
contrast smoothing? (TODO_ACCOMPLISH §2-6)

The control-region null (nassar_control_null.py) measures, in plume-free
cloud-matched windows, how much of the local clear-sky XCO2 spread the
correction removes: removal = 1 − (corrected spread / original spread).
For the production model this is the 25–55 % number plume/flux users must
know about.  Rerunning the null with the feature-ablation variants
attributes it:

  full      production de_beta_nll_prof_reg_o05l15_m5 (xco2 + spec + rest),
  no_spec   cumulants dropped  → removal unchanged ⇒ spec channel not the smoother,
  no_xco2   xco2 block dropped → removal collapses ⇒ the xco2_raw−apriori
                                  channel carries the smoothing.

Reads the three plume_preservation/nassar_control_null.csv files and writes
nassar_channel_attribution.{csv,md} next to the variants.

Usage:
  python workspace/Nassar_plume_analysis/nassar_channel_attribution.py \
      [--base results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BASE = Path("results/model_comparison/deep_ensemble/"
                    "de_beta_nll_prof_reg_o05l15_m5")


def load(csv: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[["plant_id", "date", "radius_km", "plant_ratio_corr_over_bc",
             "control_ratio_median", "n_controls"]].copy()
    return df.rename(columns={
        "plant_ratio_corr_over_bc": f"plant_ratio_{tag}",
        "control_ratio_median": f"ctrl_ratio_{tag}",
        "n_controls": f"n_ctrl_{tag}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = ap.parse_args()

    paths = {
        "full": args.base / "nassar_plumes/plume_preservation/nassar_control_null.csv",
        "no_spec": args.base / "nassar_plumes_variants/no_spec/plume_preservation/nassar_control_null.csv",
        "no_xco2": args.base / "nassar_plumes_variants/no_xco2/plume_preservation/nassar_control_null.csv",
    }
    dfs = None
    for tag, p in paths.items():
        d = load(p, tag)
        dfs = d if dfs is None else dfs.merge(d, on=["plant_id", "date", "radius_km"],
                                              how="outer")
    out_dir = args.base / "nassar_plumes_variants"
    dfs.to_csv(out_dir / "nassar_channel_attribution.csv", index=False)

    lines = [
        "# Clear-sky contrast smoothing — channel attribution (§2-6)",
        "",
        "Control-window removal = 1 − median(corrected/original spread ratio) "
        "over plume-free, cloud-matched windows (same dates/windows per row). "
        "`plant removal` is the same quantity on the plant disk.",
        "",
        "| case | date | ctrl removal full | ctrl removal no_spec | "
        "ctrl removal no_xco2 | plant removal full | plant removal no_xco2 |",
        "|---|---|---|---|---|---|---|",
    ]
    def pct(r):
        return f"{100*(1-r):.0f} %" if np.isfinite(r) else "—"
    for _, x in dfs.iterrows():
        lines.append(
            f"| {x['plant_id']} | {x['date']} | {pct(x['ctrl_ratio_full'])} | "
            f"{pct(x['ctrl_ratio_no_spec'])} | {pct(x['ctrl_ratio_no_xco2'])} | "
            f"{pct(x['plant_ratio_full'])} | {pct(x['plant_ratio_no_xco2'])} |")
    med = {t: float(np.nanmedian(dfs[f'ctrl_ratio_{t}'])) for t in
           ("full", "no_spec", "no_xco2")}
    rng = {t: (float(np.nanmin(dfs[f'ctrl_ratio_{t}'])),
               float(np.nanmax(dfs[f'ctrl_ratio_{t}']))) for t in med}
    lines += [
        "",
        "## Pooled (median across cases; range in brackets)",
        "",
    ]
    for t in ("full", "no_spec", "no_xco2"):
        lo, hi = rng[t]
        lines.append(f"- **{t}**: removes {100*(1-med[t]):.0f} % "
                     f"[{100*(1-hi):.0f}–{100*(1-lo):.0f} %] of control-window "
                     f"clear-sky spread")
    lines += [
        "",
        f"Attribution: dropping the cumulants (no_spec) changes the removal by "
        f"{100*(med['no_spec']-med['full']):+.0f} pp; dropping the xco2 block "
        f"(no_xco2) changes it by {100*(med['no_xco2']-med['full']):+.0f} pp. "
        "The channel whose removal tracks the full model is the one doing the "
        "smoothing.",
        "",
    ]
    (out_dir / "nassar_channel_attribution.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {out_dir / 'nassar_channel_attribution.csv'} and .md")


if __name__ == "__main__":
    main()

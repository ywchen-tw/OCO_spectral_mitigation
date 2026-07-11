"""coincidence_sensitivity_table.py — M2 appendix table (TODO §2-5).

Aggregates the 3 × 3 coincidence-criteria sweep of tccon_comparison_report.py
(radius 25/50/100 km × window ±30/60/120 min) into one compact table showing
that the TCCON improvement is not an artifact of the collocation choice.

Per combo (qf 'all', surface 'all' rows): number of cases and footprints,
per-case mean |bias| and fp-RMSE before → after (AK reference), the
site-clustered bootstrap Δ mean |bias| with CI/p, and the station-day
Wilcoxon p.  Writes coincidence_sensitivity.{csv,md}.

Usage:
  python workspace/coincidence_sensitivity_table.py \
      [--dir results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/atrain/coincidence_sensitivity]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path("results/model_comparison/deep_ensemble/"
                   "de_beta_nll_prof_reg_o05l15_m5/atrain/coincidence_sensitivity")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--radii", default="25,50,100")
    ap.add_argument("--windows", default="30,60,120")
    args = ap.parse_args()

    rows = []
    for r in (int(x) for x in args.radii.split(",")):
        for w in (int(x) for x in args.windows.split(",")):
            sfx = f"_r{r}km_w{w}min"
            cmp_p = args.dir / f"tccon_comparison{sfx}.csv"
            sig_p = args.dir / f"tccon_significance{sfx}.csv"
            if not cmp_p.exists():
                print(f"missing {cmp_p} — skipped")
                continue
            c = pd.read_csv(cmp_p)
            if "qf_group" in c.columns:
                c = c[c["qf_group"] == "all"]
            c = c[c["surface"] == "all"]
            c = c[np.isfinite(c["bias_before"]) & np.isfinite(c["bias_after"])]
            rec = dict(
                radius_km=r, window_min=w, n_cases=len(c),
                n_fp=int(c["n_oco"].sum()),
                absbias_before=float(c["bias_before"].abs().mean()),
                absbias_after=float(c["bias_after"].abs().mean()),
                rmse_before=float(c["rmse_before"].mean()),
                rmse_after=float(c["rmse_after"].mean()),
            )
            if sig_p.exists():
                s = pd.read_csv(sig_p)
                s = s[(s["qf_group"] == "all") & (s["subset"] == "all")]
                if len(s):
                    s = s.iloc[0]
                    rec.update(
                        d_absbias=float(s["d_mean_absbias_mean"]),
                        d_absbias_ci=(float(s["d_mean_absbias_ci_lo"]),
                                      float(s["d_mean_absbias_ci_hi"])),
                        d_absbias_p=float(s["d_mean_absbias_p"]),
                        wilcoxon_p=float(s["wilcoxon_absbias_p"]))
            rows.append(rec)

    df = pd.DataFrame(rows)
    out_csv = args.dir / "coincidence_sensitivity.csv"
    df.to_csv(out_csv, index=False)

    lines = [
        "# Coincidence-criteria sensitivity (M2)",
        "",
        "Per-case mean |bias to TCCON| and fp-RMSE (AK reference, qf all, "
        "surface all), before → after correction, across collocation radius "
        "and time window.  Δ|bias| is the site-clustered bootstrap mean "
        "difference (after − before) with 95 % CI and p; Wilcoxon is the "
        "station-day paired test on |bias|.",
        "",
        "| radius (km) | window (min) | cases | footprints | mean \\|bias\\| "
        "(ppm) | mean fp-RMSE (ppm) | Δ\\|bias\\| [95 % CI] | boot p | "
        "Wilcoxon p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, x in df.iterrows():
        ci = x.get("d_absbias_ci")
        lines.append(
            f"| {x['radius_km']:.0f} | ±{x['window_min']:.0f} | "
            f"{x['n_cases']:.0f} | {x['n_fp']:,.0f} | "
            f"{x['absbias_before']:.2f} → {x['absbias_after']:.2f} | "
            f"{x['rmse_before']:.2f} → {x['rmse_after']:.2f} | "
            + (f"{x['d_absbias']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
               f"{x['d_absbias_p']:.4f} | {x['wilcoxon_p']:.4f} |"
               if isinstance(ci, tuple) else "— | — | — |"))
    out_md = args.dir / "coincidence_sensitivity.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(df.to_string(index=False))
    print(f"\nwrote {out_csv}\nwrote {out_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage the Supplement (S1–S6) figure assets into manuscript/supplement/.

The staging area is GIT-IGNORED except for the manifest this script writes
(supplement_manifest.txt): fully staged it holds ~250 MB of bulk case
panels, which do not belong in git. Rerun this script to rebuild after a
report-tree refresh; `manuscript/tex/supplement.tex` assembles the final
Supplement PDF from these directories.

Sections (per the Supplement plan S1–S6 in log/MANUSCRIPT_FLOW_PLAN.md):
  S1_station_days/      all TCCON station-day before/after panels (atrain)
  S2_ocean_cases/       per-date ATom panels + individual ship cases
  S3_case_atlases/      seven category atlas pages — NO LOCAL SOURCE yet
                        (working figure A1 pages; pull/regenerate on CURC)
  S4_spectrum_internal/ former Appendix I: spec-only classifier ROC,
                        sub-pixel response (+ metrics CSVs)
  S5_fit_robustness/    extended SG sweep + fit-example gallery — NO LOCAL
                        SOURCE yet (savgol A/B lives on CURC)
  S6_reserve/           populated only if Appendix G (3-D RT) demotes
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TAG = (REPO / "results" / "model_comparison" / "deep_ensemble"
       / "de_beta_nll_prof_reg_foldpca_o05l15_m5")
SPEC = REPO / "results" / "figures" / "cld_dist_analysis" / "spec_sensitivity"
OUT = REPO / "manuscript" / "supplement"

manifest: list[str] = []


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(f"{dst.relative_to(OUT)}  <-  {src.relative_to(REPO)}")


def main() -> None:
    # S1 — TCCON station-day panels (compact 2x3 version per case)
    for case in sorted((TAG / "atrain").glob("combined_*")):
        src = case / "corrected_xco2_vs_tccon.png"
        if src.exists():
            copy(src, OUT / "S1_station_days"
                 / f"{case.name.removeprefix('combined_')}.png")

    # S2 — ocean case pages
    for case in sorted((TAG / "atom").glob("combined_*_atom")):
        date = case.name.split("_")[1]
        for src in sorted(case.glob("atom_comparison_*.png")):
            copy(src, OUT / "S2_ocean_cases" / f"atom_{date}.png")
    for case in sorted((TAG / "ship").glob("combined_*")):
        _, date, cid = case.name.split("_")
        for src in sorted(case.glob("ship_comparison_*.png")):
            copy(src, OUT / "S2_ocean_cases" / f"ship_{date}_{cid}.png")

    # S4 — spectrum-internal sensitivity (former Appendix I)
    for name in ("spec_cloud_classifier_roc.png",
                 "subpixel_anomaly_vs_specidx.png",
                 "spec_cloud_classifier_metrics.csv",
                 "subpixel_anomaly_vs_specidx.csv"):
        src = SPEC / name
        if src.exists():
            copy(src, OUT / "S4_spectrum_internal" / name)

    pending = [
        "S3_case_atlases: no local source (working figure A1 atlas pages; "
        "pull or regenerate on CURC before Supplement assembly)",
        "S5_fit_robustness: no local source (savgol A/B sweep + fit-example "
        "gallery live on CURC)",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "supplement_manifest.txt").write_text(
        "# Supplement staging manifest — rebuilt by "
        "manuscript/scripts/stage_supplement_figures.py\n"
        "# (staging area is git-ignored; this manifest is tracked)\n\n"
        + "\n".join(manifest)
        + "\n\n# PENDING sources:\n" + "\n".join(f"# {p}" for p in pending)
        + "\n")
    print(f"staged {len(manifest)} files into {OUT}")
    for p in pending:
        print("PENDING:", p)


if __name__ == "__main__":
    main()

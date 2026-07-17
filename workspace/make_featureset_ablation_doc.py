"""make_featureset_ablation_doc.py — regenerate the QF-grouped feature-set
ablation writeup from the per-tree TCCON report CSVs.

Inputs (all produced by tccon_comparison_report.py --ak-harmonize):
  full     : <DE tag>/atrain/tccon_metrics_{ak,direct}_r100km.csv (+ per-case CSV)
  variants : deep_ensemble/de_prof_mix_<v>/tccon_metrics_{ak,direct}_r100km.csv
Held-out date-kfold table: run_summary.json of each variant's fold dirs.

All trees are built from the same input parquets and collocated identically, so
the footprint sets match row-for-row (asserted on n_footprints); the tables
therefore isolate the value of each feature group exactly as the 2026-07-08
edition did.

Usage:  PYTHONPATH=src python workspace/make_featureset_ablation_doc.py \
            [--out results/model_comparison/deep_ensemble/FEATURESET_ABLATION_QF_<date>.md]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

TAG = "de_beta_nll_prof_reg_foldpca_o05l15_m5"
BASE = Path("results/model_comparison/deep_ensemble")
FULL_DIR = BASE / TAG / "atrain"
VARIANTS = ["no_spec", "no_contam", "no_xco2", "no_xco2_and_spec",
            "no_contam_and_xco2"]
MODELS = ["full"] + VARIANTS

# (label, qf_group, surface, cld_group) rows for the two slice tables
POOLED_SLICES = [
    ("pooled, QF 0+1", "all", "all", "all"),
    ("pooled, QF=0 (good)", "qf0", "all", "all"),
    ("pooled, QF=1", "qf1", "all", "all"),
    ("ocean, QF 0+1", "all", "ocean", "all"),
    ("land, QF 0+1", "all", "land", "all"),
    ("land, QF=1", "qf1", "land", "all"),
]
NEARCLOUD_SLICES = [
    ("land ≤10 km, QF 0+1", "all", "land", "0–10 km"),
    ("land ≤10 km, QF=1", "qf1", "land", "0–10 km"),
    ("land ≥10 km, QF 0+1", "all", "land", "≥10 km"),
]

FOLD_STEM = {
    "ocean": "results/model_deep_ensemble/de_ocean_{v}_prof_foldpca_r05_f*",
    "land": "results/model_deep_ensemble/de_land_{v}_prof_foldpca_r15_f*",
}
FOLD_STEM_FULL = {
    "ocean": "results/model_deep_ensemble/de_ocean_beta_nll_prof_reg_foldpca_r05_f*",
    "land": "results/model_deep_ensemble/de_land_beta_nll_prof_reg_foldpca_r15_f*",
}


def tree_dir(model: str) -> Path:
    return FULL_DIR if model == "full" else BASE / f"de_prof_mix_{model}"


def load_metrics(model: str, ref: str) -> pd.DataFrame:
    p = tree_dir(model) / f"tccon_metrics_{ref}_r100km.csv"
    d = pd.read_csv(p)
    d["model"] = model
    return d


def slice_row(d: pd.DataFrame, qf: str, surface: str, cld: str) -> pd.Series:
    m = d[(d.qf_group == qf) & (d.surface == surface) & (d.cld_group == cld)]
    if len(m) != 1:
        raise SystemExit(f"slice ({qf},{surface},{cld}) matched {len(m)} rows")
    return m.iloc[0]


def station_equal_absbias(model: str) -> dict:
    """{(ref, qf): (before, after)} station-equal-weighted mean |bias|."""
    d = pd.read_csv(tree_dir(model) / "tccon_comparison_r100km.csv")
    d = d[d.surface == "all"]
    out = {}
    for qf in ("all", "qf0", "qf1"):
        g = d[d.qf_group == qf]
        for ref, (b_col, a_col) in {
                "ak": ("bias_before", "bias_after"),
                "direct": ("bias_before_direct", "bias_after_direct")}.items():
            per_site_b, per_site_a = [], []
            for _, s in g.groupby("site"):
                w = s["n_oco"].to_numpy(float)
                if not np.isfinite(w).any() or w.sum() == 0:
                    continue
                per_site_b.append(abs(np.average(s[b_col], weights=w)))
                per_site_a.append(abs(np.average(s[a_col], weights=w)))
            out[(ref, qf)] = (float(np.mean(per_site_b)),
                              float(np.mean(per_site_a)))
    return out


def heldout_table() -> tuple[list[str], list[str]]:
    """(median table over HEALTHY folds, per-fold health table)."""
    rows, health = [], []
    for surf in ("ocean", "land"):
        for model in MODELS:
            stem = (FOLD_STEM_FULL[surf] if model == "full"
                    else FOLD_STEM[surf].format(v=model))
            per_fold = []
            for d in sorted(glob(stem)):
                p = Path(d) / "run_summary.json"
                if not p.exists():
                    continue
                j = json.loads(p.read_text())
                per_fold.append((Path(d).name.rsplit("_f", 1)[-1],
                                 j.get("primary_metric_value"),
                                 j.get("secondary_metrics", {}).get("de_held_r2")))
            if not per_fold:
                continue
            ok = [(f, rm, r2) for f, rm, r2 in per_fold if rm < 2.0]
            bad = [f for f, rm, r2 in per_fold if rm >= 2.0]
            rows.append((surf, model,
                         float(np.median([rm for _, rm, _ in ok])),
                         float(np.median([r2 for _, _, r2 in ok])),
                         len(ok), bad))
            health.append(
                f"| {surf} | {model} | " +
                " | ".join(f"{rm:.3g} / {r2:.3g}" for _, rm, r2 in per_fold) +
                " |")
    lines = ["| surface | variant | held-out RMSE (ppm) | held-out R² | "
             "healthy folds | diverged folds |",
             "|---|---|---|---|---|---|"]
    for surf, model, rm, r2, n, bad in rows:
        lines.append(f"| {surf} | {model} | {rm:.4f} | {r2:.4f} | {n} | "
                     f"{', '.join('f' + b for b in bad) or '—'} |")
    hlines = ["| surface | variant | f0 RMSE/R² | f1 | f2 | f3 | f4 |",
              "|---|---|---|---|---|---|---|"] + health
    return lines, hlines


def fmt_table(slices, ref_frames, best_bold=True):
    lines = [
        "| slice | n_fp | RMSE before | " +
        " | ".join(f"RMSE {m}" for m in MODELS) + " |",
        "|---|---|---|" + "---|" * len(MODELS),
    ]
    delta_rows = []
    for label, qf, surface, cld in slices:
        rows = {m: slice_row(ref_frames[m], qf, surface, cld) for m in MODELS}
        ns = {int(r.n_footprints) for r in rows.values()}
        if len(ns) != 1:
            raise SystemExit(f"footprint mismatch on '{label}': {ns}")
        n = ns.pop()
        before = rows["full"].rmse_before
        afters = {m: rows[m].rmse_after for m in MODELS}
        best = min(afters, key=afters.get)
        cells = []
        for m in MODELS:
            v = f"{afters[m]:.3f}"
            cells.append(f"**{v}**" if (best_bold and m == best) else v)
        lines.append(f"| {label} | {n:,} | {before:.3f} | " +
                     " | ".join(cells) + " |")
        delta_rows.append((label, {m: afters[m] - afters["full"]
                                   for m in VARIANTS}))
    dl = ["| slice | " + " | ".join(VARIANTS) + " |",
          "|---|" + "---|" * len(VARIANTS)]
    for label, dd in delta_rows:
        dl.append(f"| {label} | " +
                  " | ".join(f"{dd[m]:+.3f}" for m in VARIANTS) + " |")
    return lines, dl


def main():
    ap = argparse.ArgumentParser()
    today = dt.date.today().isoformat()
    ap.add_argument("--out", default=str(
        BASE / f"FEATURESET_ABLATION_QF_{today}.md"))
    args = ap.parse_args()

    ak = {m: load_metrics(m, "ak") for m in MODELS}
    direct = {m: load_metrics(m, "direct") for m in MODELS}

    pooled, pooled_d = fmt_table(POOLED_SLICES, ak)
    near, near_d = fmt_table(NEARCLOUD_SLICES, ak)

    st = {m: station_equal_absbias(m) for m in MODELS}
    st_lines = ["| ref | QF | before | " +
                " | ".join(MODELS) + " |",
                "|---|---|---|" + "---|" * len(MODELS)]
    for ref in ("direct", "ak"):
        for qf in ("all", "qf0", "qf1"):
            before = st["full"][(ref, qf)][0]
            afters = {m: st[m][(ref, qf)][1] for m in MODELS}
            best = min(afters, key=afters.get)
            cells = [f"**{afters[m]:.3f}**" if m == best else f"{afters[m]:.3f}"
                     for m in MODELS]
            st_lines.append(f"| {ref} | {qf} | {before:.3f} | " +
                            " | ".join(cells) + " |")

    md = [
        f"# Mix-DE feature-set ablation — QF-grouped (land + ocean TCCON) — {today}",
        "",
        f"Production **fold-PCA mix deep ensemble** (`{TAG}`; ocean r05 / land r15, "
        "profile) with feature GROUPS removed, scored on TCCON (100 km / 60 min, "
        "AK-harmonized). `full` is the PRODUCTION model (lndo01 reg); the variants "
        "are the `no_*_prof_foldpca` models **retrained 2026-07-17 with the same "
        "lndo01 regularization** (`--norm layer --dropout 0.1`) — variants now "
        "differ from `full` ONLY in feature set. Every tree shares the same "
        "footprints (asserted) and the same `xco2_bc` before-column, so the "
        "tables isolate each feature group. All 5 land + 5 ocean folds pooled "
        "per variant (25 members). Supersedes the 2026-07-15 PRELIMINARY "
        "edition (land-f2 divergence, unreg variants) and "
        "`FEATURESET_ABLATION_QF_2026-07-08.md` (global PCA).",
        "",
        "> **Known leftover (minor):** land f4 of `no_contam` and "
        "`no_contam_and_xco2` still carries the OLD un-regularized checkpoint "
        "(the retrain array's tail was preempted) — a HEALTHY fold in both "
        "trainings (held-out R² 0.43 / 0.24), so it is pooled; config purity "
        "for those two arms would need a `sbatch --array=4` top-up on CURC.",
        "",
        "## Pooled and surface splits (AK reference)", "",
        *pooled, "",
        "_ΔRMSE vs full (ppm; + = worse than full):_", "",
        *pooled_d, "",
        "## Near-cloud land tail (AK reference)", "",
        *near, "",
        "_ΔRMSE vs full (ppm; + = worse than full):_", "",
        *near_d, "",
        "## Mean per-station absolute bias to TCCON (station-equal-weighted, ppm)",
        "",
        "Each station's footprint-weighted mean bias → |·| → averaged over the "
        "stations (every station equal). `direct` = raw window mean; `ak` = "
        "AK-harmonized. `before` shared; best after per row in **bold**.", "",
        *st_lines, "",
    ]
    med_lines, health_lines = heldout_table()
    md += [
        "## Held-out date-kfold validation (median across HEALTHY folds)", "",
        *med_lines, "",
        "### Per-fold health (RMSE ppm / R²; diverged folds excluded above "
        "and from the pooled variant trees)", "",
        *health_lines, "",
    ]
    Path(args.out).write_text("\n".join(md))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

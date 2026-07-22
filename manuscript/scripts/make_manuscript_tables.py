"""make_manuscript_tables.py — LaTeX tables for the AMT manuscript, generated
from the same report CSVs that feed the comparison deck and the markdown docs
(so the manuscript numbers can never drift from the analysis trees).

Style matches log/tccon_station_table_used_avail_yes.tex (booktabs
\\specialrule top/bottom + \\midrule, \\small).

Tables written to manuscript/tables/:
  tab_model_comparison.tex   DE / XGB / Ridge fp-RMSE by slice (AK)
  tab_station_equal_bias.tex station-equal mean |bias| × QF (AK)
  tab_featureset_ablation.tex mix-DE feature-group ablation (AK)
  tab_raw_bc_ml.tex          raw / bc / ML-on-bc / ML-on-raw four series (AK)
  tab_nassar_attribution.tex clear-sky contrast smoothing channel attribution

Usage:  PYTHONPATH=src python manuscript/scripts/make_manuscript_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results" / "model_comparison"
DE_TAG = "de_beta_nll_prof_reg_foldpca_o05l15_m5"
TREES = {
    "DE": BASE / "deep_ensemble" / DE_TAG / "atrain",
    "XGB": BASE / "xgb/xgb_prof_foldpca_o05l15",
    "Ridge": BASE / "linreg/linreg_prof_foldpca_o05l15",
}
RAW_TREE = BASE / "deep_ensemble/de_prof_reg_mix_raw"
ABL_VARIANTS = ["no_spec", "no_contam", "no_xco2", "no_xco2_and_spec",
                "no_contam_and_xco2"]
ABL_HEADS = ["full", "$-$spec", "$-$contam", "$-$xco2", "$-$xco2$-$spec",
             "$-$xco2$-$contam"]
OUT = REPO / "manuscript" / "tables"

SLICES = [
    ("Pooled, QF 0+1", "all", "all", "all"),
    ("Pooled, QF$\\,$=$\\,$0", "qf0", "all", "all"),
    ("Pooled, QF$\\,$=$\\,$1", "qf1", "all", "all"),
    ("Ocean", "all", "ocean", "all"),
    ("Land", "all", "land", "all"),
    ("Land, QF$\\,$=$\\,$1", "qf1", "land", "all"),
    ("Land $\\leq$10\\,km", "all", "land", "0–10 km"),
    ("Land $\\leq$10\\,km, QF$\\,$=$\\,$1", "qf1", "land", "0–10 km"),
    ("Land $\\geq$10\\,km", "all", "land", "≥10 km"),
]

TOP = "\\specialrule{1.2pt}{0pt}{2pt}"
BOT = "\\specialrule{1.2pt}{2pt}{0pt}"


def srow(d, qf, surface, cld):
    m = d[(d.qf_group == qf) & (d.surface == surface) & (d.cld_group == cld)]
    assert len(m) == 1, (qf, surface, cld, len(m))
    return m.iloc[0]


def wrap(colspec, header, rows, caption, label, notes=None):
    lines = [
        "\\begin{table}[htbp]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
        f"\\begin{{tabular}}{{{colspec}}}", TOP,
        header + " \\\\", "\\midrule",
        *[r if r == "\\midrule" else r + " \\\\" for r in rows],
        BOT, "\\end{tabular}",
    ]
    if notes:
        lines.append(f"\\\\[2pt]{{\\footnotesize {notes}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def bold_min(vals, fmts):
    """Format vals with fmts, bolding the minimum."""
    i = int(np.argmin(vals))
    return [f"\\textbf{{{f % v}}}" if j == i else f % v
            for j, (v, f) in enumerate(zip(vals, [fmts] * len(vals)))]


def table_model_comparison():
    mets = {m: pd.read_csv(p / "tccon_metrics_ak_r100km.csv")
            for m, p in TREES.items()}
    rows = []
    for label, qf, surface, cld in SLICES:
        rr = {m: srow(mets[m], qf, surface, cld) for m in TREES}
        ns = {int(r.n_footprints) for r in rr.values()}
        assert len(ns) == 1
        cells = [label,
                 f"{int(rr['DE'].n_footprints):,}".replace(",", "\\,"),
                 f"{rr['DE'].rmse_before:.2f}"]
        cells += bold_min([rr[m].rmse_after for m in TREES], "%.2f")
        rows.append(" & ".join(cells))
    header = "Slice & $n$ & Before & DE & XGB & Ridge"
    cap = ("Footprint RMSE against AK-harmonized TCCON (ppm; 100\\,km / "
           "$\\pm$60\\,min, 75 station-days) before (\\xcobc) and after "
           "correction by the deep ensemble (DE), XGBoost (XGB), and ridge "
           "regression (Ridge). All models share the same footprints and "
           "before-column; the best corrected value per slice is in bold. "
           "QF is the operational \\texttt{xco2\\_quality\\_flag}; "
           "cloud-distance slices use the pre-drift cases where the "
           "Aqua-MODIS collocation is reliable.")
    return wrap("lrrrrr", header, rows, cap, "tab:model-comparison")


def table_station_equal():
    def per_model(tree):
        d = pd.read_csv(tree / "tccon_comparison_r100km.csv")
        out = {}
        b_col, a_col = "bias_before", "bias_after"          # AK-harmonized
        for qf in ("all", "qf0", "qf1"):
            g = d[(d.surface == "all") & (d.qf_group == qf)]
            pb, pa = [], []
            for _, s in g.groupby("site"):
                w = s["n_oco"].to_numpy(float)
                m = np.isfinite(w) & np.isfinite(s[a_col].to_numpy(float))
                if not m.any() or w[m].sum() == 0:
                    continue
                pb.append(abs(np.average(s[b_col][m], weights=w[m])))
                pa.append(abs(np.average(s[a_col][m], weights=w[m])))
            out[qf] = (float(np.mean(pb)), float(np.mean(pa)))
        return out
    st = {m: per_model(p) for m, p in TREES.items()}
    rows = []
    for qf, qlbl in (("all", "QF 0+1"), ("qf0", "QF$\\,$=$\\,$0"),
                     ("qf1", "QF$\\,$=$\\,$1")):
        before = st["DE"][qf][0]
        afters = [st[m][qf][1] for m in TREES]
        rows.append(" & ".join([qlbl, f"{before:.2f}"]
                               + bold_min(afters, "%.2f")))
    header = "QF & Before & DE & XGB & Ridge"
    cap = ("Station-equal mean absolute bias to AK-harmonized TCCON (ppm): "
           "each station's footprint-weighted mean bias is taken in absolute "
           "value and averaged with equal station weights (18 stations). "
           "Best corrected value per row in bold.")
    return wrap("lrrrr", header, rows, cap, "tab:station-equal-bias")


def table_ablation():
    trees = {"full": TREES["DE"]}
    trees.update({v: BASE / "deep_ensemble" / f"de_prof_mix_{v}"
                  for v in ABL_VARIANTS})
    mets = {m: pd.read_csv(p / "tccon_metrics_ak_r100km.csv")
            for m, p in trees.items()}
    rows = []
    for label, qf, surface, cld in SLICES:
        rr = {m: srow(mets[m], qf, surface, cld) for m in trees}
        ns = {int(r.n_footprints) for r in rr.values()}
        assert len(ns) == 1
        vals = [rr[m].rmse_after for m in trees]
        rows.append(" & ".join(
            [label, f"{rr['full'].rmse_before:.2f}"] + bold_min(vals, "%.2f")))
    header = "Slice & Before & " + " & ".join(ABL_HEADS)
    cap = ("Feature-group ablation of the mix deep ensemble on TCCON "
           "(fp-RMSE, ppm; AK-harmonized reference, 100\\,km / "
           "$\\pm$60\\,min). Each variant is retrained from scratch with one "
           "feature group removed, using the identical architecture, "
           "regularization, and date-blocked folds as the production model "
           "(full); all columns share the same footprints and before-column. "
           "Dropping the spectral-cumulant or aerosol-contamination groups is "
           "TCCON-neutral; the \\xcoraw$-$prior departure block carries the "
           "correction.")
    return wrap("lr" + "r" * len(ABL_HEADS), header, rows, cap,
                "tab:featureset-ablation")


def table_raw_bc_ml():
    a = pd.read_csv(TREES["DE"] / "tccon_metrics_ak_r100km.csv")
    b = pd.read_csv(RAW_TREE / "tccon_metrics_ak_r100km.csv")
    rows = []
    for label, qf, surface, cld in SLICES:
        ra, rb = srow(a, qf, surface, cld), srow(b, qf, surface, cld)
        assert int(ra.n_footprints) == int(rb.n_footprints)
        vals = [ra.rmse_raw, ra.rmse_before, ra.rmse_after, rb.rmse_after]
        fmt = [f"{v:.2f}" for v in vals]
        best = 1 + int(np.argmin(vals[1:]))          # raw excluded from bold
        fmt[best] = f"\\textbf{{{fmt[best]}}}"
        rows.append(" & ".join(
            [label] + fmt
            + [f"{v:+.2f}" for v in (ra.bias_raw, ra.bias_before,
                                     ra.bias_after, rb.bias_after)]))
    header = ("Slice & \\multicolumn{4}{c}{fp-RMSE} & "
              "\\multicolumn{4}{c}{Bias} \\\\\n"
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
              " & raw & bc & ML(bc) & ML(raw) & raw & bc & ML(bc) & ML(raw)")
    cap = ("Four XCO$_2$ series against TCCON (ppm; AK-harmonized, "
           "100\\,km / $\\pm$60\\,min): pre-bias-correction \\xcoraw{} (raw), "
           "operationally bias-corrected \\xcobc{} (bc), the production ML "
           "correction applied to \\xcobc{} (ML(bc)), and an ML correction "
           "trained on and applied to \\xcoraw{} directly (ML(raw)). Best of "
           "the three corrected/operational series per slice in bold. "
           "ML(raw) reaching within $\\sim$0.1\\,ppm of the production chain "
           "shows the per-footprint ML layer can absorb most of the "
           "operational bias correction over these scenes.")
    return wrap("lrrrrrrrr", header, rows, cap, "tab:raw-bc-ml")


def table_nassar():
    d = pd.read_csv(BASE / "deep_ensemble" / DE_TAG /
                    "nassar_plumes_variants/nassar_channel_attribution.csv")
    d = d.sort_values(["plant_id", "date"])

    def pct(ratio):
        return f"{100 * (1 - ratio):.0f}" if np.isfinite(ratio) else "--"
    rows = []
    for r in d.itertuples():
        rows.append(" & ".join([
            r.plant_id.capitalize(), r.date,
            pct(r.ctrl_ratio_full), pct(r.ctrl_ratio_no_spec),
            pct(r.ctrl_ratio_no_xco2),
            pct(r.plant_ratio_full), pct(r.plant_ratio_no_xco2)]))
    med = {c: float(np.nanmedian(d[c])) for c in
           ("ctrl_ratio_full", "ctrl_ratio_no_spec", "ctrl_ratio_no_xco2")}
    rows.append("\\midrule")
    rows.append(" & ".join(
        ["Median", "", pct(med["ctrl_ratio_full"]),
         pct(med["ctrl_ratio_no_spec"]), pct(med["ctrl_ratio_no_xco2"]),
         "", ""]))
    header = ("Case & Date & \\multicolumn{3}{c}{Control removal (\\%)} & "
              "\\multicolumn{2}{c}{Plant removal (\\%)} \\\\\n"
              "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
              " & & full & $-$spec & $-$xco2 & full & $-$xco2")
    cap = ("Clear-sky contrast smoothing and its channel attribution at the "
           "Nassar power-plant cases: removal $=1-$ (corrected / original "
           "5--95\\,\\% spread), median over plume-free cloud-matched control "
           "windows (control) and on the plant disk (plant), for the "
           "production model (full) and the retrained variants without the "
           "spectral cumulants ($-$spec) or without the \\xcoraw$-$prior "
           "departure block ($-$xco2). The smoothing is carried by the xco2 "
           "channel; removing the cumulants changes it by $\\sim$1 "
           "percentage point.")
    return wrap("llrrrrr", header, rows, cap, "tab:nassar-attribution")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preamble = (
        "% Generated by manuscript/scripts/make_manuscript_tables.py — do not hand-edit.\n"
        "% Requires booktabs. Define in the preamble:\n"
        "%   \\newcommand{\\xcobc}{$X_{\\mathrm{CO2}}^{\\mathrm{bc}}$}\n"
        "%   \\newcommand{\\xcoraw}{$X_{\\mathrm{CO2}}^{\\mathrm{raw}}$}\n")
    for name, fn in [("tab_model_comparison", table_model_comparison),
                     ("tab_station_equal_bias", table_station_equal),
                     ("tab_featureset_ablation", table_ablation),
                     ("tab_raw_bc_ml", table_raw_bc_ml),
                     ("tab_nassar_attribution", table_nassar)]:
        p = OUT / f"{name}.tex"
        p.write_text(preamble + fn())
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

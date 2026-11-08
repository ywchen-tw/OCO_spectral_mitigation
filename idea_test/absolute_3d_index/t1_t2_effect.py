"""T1 (absolute vs delta effect sizes) and T2 (absolute distance curves).

Population (both tests): the 139 evaluation dates, snow-free, xco2_bc > 0,
finite cloud distance (common.build_date).  Two sub-populations:
  ref : the Fig. 5 reference population -- a finite production-reference
        z_gamma and anomaly (ocean r05 / land r15, |anomaly| <= 100 ppm) and a
        reference window without the other surface type (mixed-window flag
        from workspace/mixed_surface_reference_count.py, evaluated out to 50 km).
  all : every footprint with a cloud distance and a finite value of v.
E_z = (mean near 0-5 km - mean far 20-50 km) / SD far (analysis.land_class.
build_effect_sizes; cross-checked below), n_min = 500 per window.

Illumination classes (land, ref only): z_gamma > 0.5 brightened, |z| <= 0.5
neutral, < -0.5 shadowed.  Two far windows: the same class (selection-matched)
and the whole far window of the ref population.

Outputs: out/t1_effect_sizes.csv, out/t1_illum.csv, out/t1_landclass.csv,
         out/t1_coverage.csv, out/t2_distance_curves.png, out/t2_curves.csv
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
from common import effect_z, illum_class

sys.path.insert(0, str(common.REPO / "workspace"))
from plot_style import MEAN_L_LABEL, apply_manuscript_style, panel_label  # noqa: E402

# (key, absolute column, delta column, label)
QUANTITIES = [
    ("k1_o2a", "o2a_k1_am", "dk1_o2a", f"O$_2$A {MEAN_L_LABEL}"),
    ("k2k1sq_sco2", "sco2_k2k1sq", "dk2k1sq_sco2", "SCO$_2$ $k_2/k_1^2$"),
    ("k1ratio_sw", "sco2_wco2_k1", "dk1ratio_sw", "$k_1$ SCO$_2$ / WCO$_2$"),
    ("fg_over_X", "fg_over_X", "dfg_over_X", r"$f\gamma/X$"),
    ("fA_over_X", "fA_over_X", "dfA_over_X", r"$fA/X$"),
    ("f_o2a", "o2a_f_am", "df_o2a", "O$_2$A floor $f$"),
    ("gamma_o2a", "o2a_exp_int_am", "dexp_o2a", r"O$_2$A $\gamma$"),
    ("fg", "fg", None, r"$f\gamma$ (no X)"),
]
SURFACES = (("ocean", 0), ("land", 1))
QFS = (("all", None), ("0", 0), ("1", 1))
CLASSES = ("shadowed", "neutral", "brightened")
CLASS_COLORS = {"brightened": "#d95f02", "neutral": "#7570b3", "shadowed": "#1b9e77"}
MIN_CLASS_N, MIN_NEAR_N = 10_000, 5_000     # make_landclass_heatmap_figure rules


def load() -> pd.DataFrame:
    cols = ["date", "cld_dist_km", "sfc_type", "xco2_qf", "zexp_o2a", "xco2_bc_anomaly",
            "mixed", "igbp_group"]
    for _, a, d, _ in QUANTITIES:
        cols += [a] + ([d] if d else [])
    df = common.load_table(sorted(set(cols)))
    df["ref_ok"] = (np.isfinite(df.zexp_o2a) & np.isfinite(df.xco2_bc_anomaly)
                    & ~df.mixed)
    df["illum"] = illum_class(df.zexp_o2a.to_numpy())
    return df


def ez_rows(s: pd.DataFrame, tags: dict) -> list[dict]:
    d_all = s.cld_dist_km.to_numpy()
    ref = s.ref_ok.to_numpy()
    out = []
    for key, a, dcol, _ in QUANTITIES:
        va = s[a].to_numpy(dtype=float)
        rec = dict(tags, quantity=key)
        r_abs_all = effect_z(d_all, va)
        r_abs_ref = effect_z(d_all[ref], va[ref])
        rec.update({f"abs_all_{k}": v for k, v in r_abs_all.items()})
        rec.update({f"abs_ref_{k}": v for k, v in r_abs_ref.items()})
        if dcol:
            r_d = effect_z(d_all[ref], s[dcol].to_numpy(dtype=float)[ref])
            rec.update({f"delta_ref_{k}": v for k, v in r_d.items()})
        out.append(rec)
    return out


def t1(df: pd.DataFrame) -> None:
    rows, cov, illum, lc = [], [], [], []
    for sname, code in SURFACES:
        for qname, qv in QFS:
            s = df[(df.sfc_type == code) & ((df.xco2_qf == qv) if qv is not None else True)]
            tags = dict(surface=sname, qf=qname)
            rows += ez_rows(s, tags)
            d = s.cld_dist_km
            for wname, w in (("near 0-5", d < 5), ("0-50", d < 50), ("all", d >= 0)):
                n_all = int((w & np.isfinite(s.o2a_k1_am)).sum())
                n_ref = int((w & s.ref_ok & np.isfinite(s.dk1_o2a)).sum())
                cov.append(dict(tags, window=wname, n_all=n_all, n_ref=n_ref,
                                gain=n_all - n_ref, gain_pct_of_ref=100 * (n_all - n_ref) / n_ref,
                                ref_share_pct=100 * n_ref / n_all))
            if sname == "land":
                r = s[s.ref_ok]
                dd = r.cld_dist_km.to_numpy()
                for c in CLASSES:
                    m = (r.illum == c).to_numpy()
                    far_all = (dd >= 20) & (dd < 50)
                    for key, a, dcol, _ in QUANTITIES:
                        for kind, col in (("abs", a), ("delta", dcol)):
                            if col is None:
                                continue
                            v = r[col].to_numpy(dtype=float)
                            same = effect_z(dd[m], v[m])
                            # near of this class vs the whole ref far window
                            mix = np.where(m | far_all, v, np.nan)
                            dmix = np.where(m & (dd < 5), dd, np.where(far_all, dd, np.inf))
                            allfar = effect_z(dmix, mix)
                            illum.append(dict(tags, illum=c, quantity=key, kind=kind,
                                              E_z_samefar=same["E_z"], ci_samefar=same["ci95"],
                                              E_z_allfar=allfar["E_z"], ci_allfar=allfar["ci95"],
                                              n_near=same["n_near"], n_far_same=same["n_far"]))
                # MCD12C1 groups
                counts = s.igbp_group.value_counts()
                for g in counts.index:
                    if g in ("water", "fill", "ocean") or counts[g] < MIN_CLASS_N:
                        continue
                    sg = s[s.igbp_group == g]
                    for rr in ez_rows(sg, dict(tags, group=g)):
                        lc.append(rr)
    eff = pd.DataFrame(rows)
    eff.to_csv(common.OUT / "t1_effect_sizes.csv", index=False)
    pd.DataFrame(cov).to_csv(common.OUT / "t1_coverage.csv", index=False)
    pd.DataFrame(illum).to_csv(common.OUT / "t1_illum.csv", index=False)
    lcd = pd.DataFrame(lc)
    lcd = lcd[lcd["delta_ref_n_near"].fillna(0) >= MIN_NEAR_N]
    lcd.to_csv(common.OUT / "t1_landclass.csv", index=False)

    show = eff[["surface", "qf", "quantity", "abs_all_E_z", "abs_ref_E_z", "delta_ref_E_z",
                "abs_all_n_near", "delta_ref_n_near", "abs_ref_far_sd", "delta_ref_far_sd"]]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(show.round(3).to_string(index=False))
        print(pd.DataFrame(cov).round(1).to_string(index=False))
        il = pd.DataFrame(illum)
        il = il[il.qf == "all"]
        print(il.pivot_table(index=["quantity", "illum"], columns="kind",
                             values=["E_z_samefar", "E_z_allfar"]).round(2).to_string())
        print(lcd[lcd.qf == "all"][["group", "quantity", "abs_all_E_z", "abs_ref_E_z",
                                    "delta_ref_E_z", "delta_ref_n_near"]].round(2).to_string(index=False))

    # cross-check against the production effect-size routine on one cell
    from analysis.land_class import build_effect_sizes
    s = df[(df.sfc_type == 1) & df.ref_ok].assign(grp="land")
    chk = build_effect_sizes(s, [("dk1_o2a", "dk1")], str(common.OUT), ["land"],
                             n_min=500, group_col="grp", prefix="t1_crosscheck")
    mine = eff.query("surface=='land' and qf=='all' and quantity=='k1_o2a'")["delta_ref_E_z"].iloc[0]
    print(f"cross-check land dk1_o2a: build_effect_sizes {chk.effect_z.iloc[0]:.5f} vs here {mine:.5f}")


def t2(df: pd.DataFrame) -> None:
    edges = np.arange(0, 51, 1.0)
    mid = 0.5 * (edges[:-1] + edges[1:])
    quants = [q for q in QUANTITIES if q[0] != "fg"]
    recs = []

    def curve(s, col):
        b = pd.cut(s.cld_dist_km, edges, right=False, labels=False)
        g = s[col].groupby(b)
        q = g.quantile([0.25, 0.5, 0.75]).unstack()
        n = g.count()
        return q.reindex(range(len(mid))), n.reindex(range(len(mid)))

    apply_manuscript_style(base_size=8.0)
    fig, axes = plt.subplots(len(quants), 3, figsize=(7.2, 1.45 * len(quants) + 0.4),
                             sharex=True, constrained_layout=True)
    for i, (key, col, _, lab) in enumerate(quants):
        for j, (title, sub) in enumerate((
                ("ocean", [("all", df[df.sfc_type == 0], "#1f78b4")]),
                ("land", [("all", df[df.sfc_type == 1], "#6a3d9a")]),
                ("land by illumination class",
                 [(c, df[(df.sfc_type == 1) & df.ref_ok & (df.illum == c)], CLASS_COLORS[c])
                  for c in CLASSES]))):
            ax = axes[i, j]
            for name, s, color in sub:
                q, n = curve(s, col)
                ok = (n >= 500).to_numpy()
                ax.fill_between(mid[ok], q[0.25].to_numpy()[ok], q[0.75].to_numpy()[ok],
                                color=color, alpha=0.18, lw=0)
                ax.plot(mid[ok], q[0.5].to_numpy()[ok], color=color, lw=1.2, label=name)
                for k in range(len(mid)):
                    recs.append(dict(panel=title, group=name, quantity=key, bin_mid=mid[k],
                                     n=int(n.iloc[k]) if np.isfinite(n.iloc[k]) else 0,
                                     q25=q[0.25].iloc[k], median=q[0.5].iloc[k],
                                     q75=q[0.75].iloc[k]))
            ax.axvspan(0, 5, color="0.9", lw=0, zorder=0)
            ax.axvspan(20, 50, color="0.96", lw=0, zorder=0)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(title, fontsize=8.5)
            panel_label(ax, f"({chr(97 + 3 * i + j)})" if 3 * i + j < 26
                        else f"(a{chr(97 + 3 * i + j - 26)})", size=8)
            # share y within a row between the two land panels
        lo = min(axes[i, 1].get_ylim()[0], axes[i, 2].get_ylim()[0])
        hi = max(axes[i, 1].get_ylim()[1], axes[i, 2].get_ylim()[1])
        axes[i, 1].set_ylim(lo, hi)
        axes[i, 2].set_ylim(lo, hi)
    for ax in axes[-1]:
        ax.set_xlabel("Nearest-cloud distance (km)")
    axes[0, 2].legend(frameon=False, fontsize=7, loc="best")
    fig.savefig(common.OUT / "t2_distance_curves.png", dpi=200)
    pd.DataFrame(recs).to_csv(common.OUT / "t2_curves.csv", index=False)
    print("wrote", common.OUT / "t2_distance_curves.png")


if __name__ == "__main__":
    data = load()
    print(f"{len(data):,} footprints; ref_ok {data.ref_ok.mean():.3f}")
    t1(data)
    t2(data)

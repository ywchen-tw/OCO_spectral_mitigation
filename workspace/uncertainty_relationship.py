"""uncertainty_relationship.py — §4.1 relationship study for the uncertainty-aware
TCCON comparison (see src/analysis/UNCERTAINTY_AWARE_TCCON_COMPARISON.md).

Question: how does the OCO-2 L2 retrieval posterior (`xco2_uncertainty`) relate to
the deep-ensemble predictive uncertainty on the corrected anomaly — total
(`de_sigma`), aleatoric (`de_aleatoric_sigma`), epistemic (`de_epistemic_sigma`)?
Are retrieval-σ and DE-σ redundant (→ don't sum), orthogonal (→ quadrature), or
DE-dominated near clouds?  This DOES NOT decide the Side-A budget — the calibration
study (§4.2) does — but it tells us how the candidate components relate, scene by
scene.

Input: one or more Phase-0 uncertainty parquets written by build_deepens_plot_data.py
(carrying de_sigma / de_epistemic_sigma / de_aleatoric_sigma / xco2_uncertainty and
scene columns cld_dist_km / aod_total / sza / snr_*).

Output: <outdir>/uncertainty_relationship.{md,png}.

Example:
    PYTHONPATH=src python workspace/uncertainty_relationship.py \
        --input .../plot_data_unc_land_multi.parquet \
        --tag de_land_beta_nll --outdir results/model_comparison/deep_ensemble/<tag>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DE_COLS = ('de_sigma', 'de_aleatoric_sigma', 'de_epistemic_sigma')
# (column, label, physical bin edges).  None edges → quantile bins at run time.
SCENE_BINS = (
    ('cld_dist_km', 'nearest-cloud distance (km)', [0, 5, 20, 50, 100, np.inf]),
    ('aod_total', 'total AOD', None),
    ('sza', 'solar zenith angle (deg)', None),
    ('snr_wco2', 'WCO2 SNR', None),
)


def _corr(a, b):
    """Pearson and Spearman between two series, NaN-safe."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan, np.nan
    a, b = a[m], b[m]
    pear = float(np.corrcoef(a, b)[0, 1])
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    spear = float(np.corrcoef(ra, rb)[0, 1])
    return pear, spear


def _fmt(x, n=3):
    return 'nan' if not np.isfinite(x) else f'{x:.{n}f}'


def load(inputs, drop_guards=True, sfc_type=None):
    df = pd.concat([pd.read_parquet(p) for p in inputs], ignore_index=True)
    need = ('xco2_uncertainty',) + DE_COLS
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"input missing required columns: {miss}")
    n0 = len(df)
    if sfc_type is not None and 'sfc_type' in df.columns:
        df = df.loc[df['sfc_type'] == sfc_type].reset_index(drop=True)
        print(f"  surface filter sfc_type=={sfc_type}: {n0:,} → {len(df):,} rows")
    if drop_guards:
        g = np.zeros(len(df), bool)
        for c in ('clim_guard', 'anomaly_guard'):
            if c in df.columns:
                g |= df[c].to_numpy(bool)
        df = df.loc[~g].reset_index(drop=True)
    # require finite retrieval + DE sigma
    ok = np.isfinite(df['xco2_uncertainty'])
    for c in DE_COLS:
        ok &= np.isfinite(df[c])
    df = df.loc[ok].reset_index(drop=True)
    print(f"  loaded {n0:,} → {len(df):,} rows after guard/finite filter")
    return df


def overall_stats(df):
    """Correlations + ratio distributions of each DE-σ vs retrieval σ."""
    xr = df['xco2_uncertainty'].to_numpy(float)
    rows = []
    for c in DE_COLS:
        y = df[c].to_numpy(float)
        pear, spear = _corr(xr, y)
        ratio = y / np.where(xr > 0, xr, np.nan)
        rows.append(dict(component=c,
                         median_sigma=float(np.nanmedian(y)),
                         pearson=pear, spearman=spear,
                         ratio_med=float(np.nanmedian(ratio)),
                         ratio_p25=float(np.nanpercentile(ratio, 25)),
                         ratio_p75=float(np.nanpercentile(ratio, 75))))
    return pd.DataFrame(rows)


def scene_table(df, col, edges):
    """Per-scene-bin median σ's and retrieval-vs-aleatoric correlation."""
    if col not in df.columns or not np.isfinite(df[col]).any():
        return None
    x = df[col].to_numpy(float)
    if edges is None:
        qs = np.nanquantile(x, [0, .2, .4, .6, .8, 1.0])
        edges = list(np.unique(qs))
        if len(edges) < 3:
            return None
    lab = pd.cut(df[col], bins=edges, include_lowest=True)
    out = []
    for iv, sub in df.groupby(lab, observed=True):
        xr = sub['xco2_uncertainty'].to_numpy(float)
        al = sub['de_aleatoric_sigma'].to_numpy(float)
        pear, _ = _corr(xr, al)
        out.append(dict(bin=str(iv), n=len(sub),
                        retrieval=float(np.nanmedian(xr)),
                        de_total=float(np.nanmedian(sub['de_sigma'])),
                        de_alea=float(np.nanmedian(al)),
                        de_epi=float(np.nanmedian(sub['de_epistemic_sigma'])),
                        ratio_alea_retr=float(np.nanmedian(
                            al / np.where(xr > 0, xr, np.nan))),
                        pearson=pear))
    return pd.DataFrame(out)


def make_figure(df, path, tag):
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    xr = df['xco2_uncertainty'].to_numpy(float)
    lim = np.nanpercentile(np.concatenate([xr, df['de_sigma']]), 99)
    lim = float(np.ceil(lim * 10) / 10)

    for a, (c, ttl) in zip(ax[0], [('de_aleatoric_sigma', 'DE aleatoric'),
                                   ('de_sigma', 'DE total')]):
        y = df[c].to_numpy(float)
        a.hexbin(xr, y, gridsize=60, extent=(0, lim, 0, lim),
                 mincnt=1, cmap='viridis', bins='log')
        a.plot([0, lim], [0, lim], 'r--', lw=1, label='1:1')
        pear, spear = _corr(xr, y)
        a.set(xlim=(0, lim), ylim=(0, lim),
              xlabel='retrieval σ  xco2_uncertainty (ppm)',
              ylabel=f'{ttl} σ (ppm)',
              title=f'{ttl} vs retrieval   ρ={pear:.2f} (Spearman {spear:.2f})')
        a.legend(loc='upper left', fontsize=8)

    # binned medians vs cld_dist and vs AOD
    for a, (col, xlab, edges) in zip(ax[1], [SCENE_BINS[0], SCENE_BINS[1]]):
        t = scene_table(df, col, edges)
        if t is None:
            a.set_visible(False); continue
        xpos = np.arange(len(t))
        a.plot(xpos, t['retrieval'], 'o-', label='retrieval σ')
        a.plot(xpos, t['de_alea'], 's-', label='DE aleatoric')
        a.plot(xpos, t['de_epi'], '^-', label='DE epistemic')
        a.plot(xpos, t['de_total'], 'd--', label='DE total', color='k', alpha=.6)
        a.set_xticks(xpos)
        a.set_xticklabels([b.split(',')[0].strip('([') for b in t['bin']],
                          rotation=30, fontsize=7)
        a.set(xlabel=xlab, ylabel='median σ (ppm)')
        a.set_title(f'median σ by {col}')
        a.legend(fontsize=8)
    fig.suptitle(f'Retrieval σ vs deep-ensemble σ — {tag}  (n={len(df):,})',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_md(df, overall, path, tag, fig_name):
    L = []
    L.append(f"# Uncertainty relationship (§4.1) — {tag}\n")
    L.append(f"n = {len(df):,} footprints (guards dropped). Retrieval σ = OCO-2 L2 "
             "`xco2_uncertainty`; DE σ = deep-ensemble predictive on the corrected "
             "anomaly.\n")
    med_r = float(np.nanmedian(df['xco2_uncertainty']))
    L.append(f"Median retrieval σ = **{med_r:.3f} ppm**.\n")
    L.append("## Overall\n")
    L.append("| DE component | median σ (ppm) | Pearson ρ | Spearman | "
             "ratio to retrieval (med [p25,p75]) |")
    L.append("|---|--:|--:|--:|--:|")
    for _, r in overall.iterrows():
        L.append(f"| {r['component']} | {_fmt(r['median_sigma'])} | "
                 f"{_fmt(r['pearson'],2)} | {_fmt(r['spearman'],2)} | "
                 f"{_fmt(r['ratio_med'],2)} [{_fmt(r['ratio_p25'],2)}, "
                 f"{_fmt(r['ratio_p75'],2)}] |")
    L.append("")
    # interpretation cue
    al = overall.set_index('component').loc['de_aleatoric_sigma']
    L.append(f"> Read: aleatoric/retrieval median ratio ≈ **{al['ratio_med']:.2f}** "
             f"at Pearson ρ ≈ **{al['pearson']:.2f}**. Similar magnitude but "
             "imperfect correlation ⇒ overlapping-but-not-redundant; the calibration "
             "study (§4.2) decides whether to sum, not this table.\n")
    for col, xlab, edges in SCENE_BINS:
        t = scene_table(df, col, edges)
        if t is None:
            continue
        L.append(f"## By {col}  ({xlab})\n")
        L.append("| bin | n | retrieval | DE total | DE alea | DE epi | "
                 "alea/retr | Pearson |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
        for _, r in t.iterrows():
            L.append(f"| {r['bin']} | {int(r['n']):,} | {_fmt(r['retrieval'])} | "
                     f"{_fmt(r['de_total'])} | {_fmt(r['de_alea'])} | "
                     f"{_fmt(r['de_epi'])} | {_fmt(r['ratio_alea_retr'],2)} | "
                     f"{_fmt(r['pearson'],2)} |")
        L.append("")
    L.append(f"![relationship]({fig_name})\n")
    Path(path).write_text('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', nargs='+', required=True,
                    help="Phase-0 uncertainty parquet(s).")
    ap.add_argument('--tag', default='de', help="label for titles/filenames.")
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--keep-guards', action='store_true',
                    help="do NOT drop clim/anomaly-guarded rows (default drops).")
    ap.add_argument('--sfc-type', type=int, default=None,
                    help="restrict to one surface (0=ocean, 1=land); default both.")
    args = ap.parse_args()

    df = load(args.input, drop_guards=not args.keep_guards, sfc_type=args.sfc_type)
    overall = overall_stats(df)
    print(overall.to_string(index=False))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig_name = 'uncertainty_relationship.png'
    make_figure(df, outdir / fig_name, args.tag)
    write_md(df, overall, outdir / 'uncertainty_relationship.md', args.tag, fig_name)
    print(f"  [saved] {outdir/'uncertainty_relationship.md'}")
    print(f"  [saved] {outdir/fig_name}")


if __name__ == '__main__':
    main()

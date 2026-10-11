"""compare_savgol_fits.py — Savitzky-Golay smoothing-bias A/B on the spectral fits (M9a).

process_orbit(dual_fit=True) writes two parameter sets per sounding into each
fitting_details_*.h5: the production fit on savgol-smoothed ln_T (*_fitting)
and a parallel fit on the raw ln_T (*_fitting_nosg).  This script gathers both
across orbits and quantifies the smoothing effect on k1/k2/intercept per band:
paired stats (median/IQR of the difference, relative difference, Pearson r)
plus a scatter figure.  If the shift is small, one sentence closes review
item M9a; if not, the tables show exactly where it matters.

Usage:
    PYTHONPATH=src python src/spectral/compare_savgol_fits.py \
        --fitting-dir results/fitting_details \
        --output-dir results/model_comparison/savgol_ab
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_WORKSPACE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'workspace')
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)
from plot_style import (apply_manuscript_style, panel_label,  # noqa: E402
                        MEAN_L_LABEL, VAR_L_LABEL)

BANDS = ('o2a', 'wco2', 'sco2')
PARAMS = ('k1', 'k2', 'intercept')
# Rendered-label forms (data keys stay 'o2a'/'k1'/... everywhere).
_BAND_LABELS = {'o2a': 'O$_2$A', 'wco2': 'WCO$_2$', 'sco2': 'SCO$_2$'}
_PARAM_LABELS = {'k1': MEAN_L_LABEL, 'k2': VAR_L_LABEL}


def _dset(tag, param, nosg):
    suffix = '_nosg' if nosg else ''
    return f"{tag}_{param}_fitting{suffix}"


def gather(fitting_dir):
    """Concatenate (sg, nosg) pairs per band/param across all fitting_details files."""
    files = sorted(glob.glob(os.path.join(fitting_dir, 'fitting_details_*.h5')))
    data = {(t, p): ([], []) for t in BANDS for p in PARAMS}
    n_files, n_with_nosg = 0, 0
    for path in files:
        try:
            with h5py.File(path, 'r') as f:
                n_files += 1
                if _dset('o2a', 'k1', True) not in f:
                    continue
                n_with_nosg += 1
                for t in BANDS:
                    for p in PARAMS:
                        sg_key, ns_key = _dset(t, p, False), _dset(t, p, True)
                        if sg_key in f and ns_key in f:
                            data[(t, p)][0].append(f[sg_key][...])
                            data[(t, p)][1].append(f[ns_key][...])
        except OSError:
            continue
    out = {}
    for key, (sg_list, ns_list) in data.items():
        if sg_list:
            out[key] = (np.concatenate(sg_list), np.concatenate(ns_list))
    return out, n_files, n_with_nosg


def paired_stats(sg, ns):
    m = np.isfinite(sg) & np.isfinite(ns)
    sg, ns = sg[m], ns[m]
    if sg.size < 3:
        return None
    d = ns - sg
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(np.abs(sg) > 0, d / np.abs(sg), np.nan)
    rel = rel[np.isfinite(rel)]
    r = float(np.corrcoef(sg, ns)[0, 1]) if sg.size > 2 else np.nan
    return dict(
        n=int(sg.size),
        median_sg=float(np.median(sg)),
        median_diff=float(np.median(d)),
        iqr_diff=float(np.subtract(*np.percentile(d, [75, 25]))),
        median_rel_diff_pct=float(100 * np.median(rel)) if rel.size else np.nan,
        p95_abs_rel_diff_pct=float(100 * np.percentile(np.abs(rel), 95)) if rel.size else np.nan,
        pearson_r=r,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fitting-dir', default='results/fitting_details',
                    help='Directory holding fitting_details_*.h5 (dual-fit outputs).')
    ap.add_argument('--output-dir', default='results/model_comparison/savgol_ab')
    args = ap.parse_args()

    data, n_files, n_with_nosg = gather(args.fitting_dir)
    if not data:
        print(f"No dual-fit (_nosg) datasets found under {args.fitting_dir} "
              f"({n_files} files scanned, {n_with_nosg} with _nosg). "
              "Run fitting.py without --single-fit first.", file=sys.stderr)
        return 1
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for (t, p), (sg, ns) in sorted(data.items()):
        st = paired_stats(sg, ns)
        if st:
            rows.append(dict(band=t, param=p, **st))
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'savgol_ab_stats.csv')
    df.to_csv(csv_path, index=False)
    print(f"[{n_with_nosg}/{n_files} files with _nosg]")
    print(df.to_string(index=False))

    # scatter grid: bands (rows) x k1/k2 (cols)
    apply_manuscript_style()
    fig, axes = plt.subplots(len(BANDS), 2, figsize=(11, 14))
    for i, t in enumerate(BANDS):
        for j, p in enumerate(('k1', 'k2')):
            ax = axes[i, j]
            if (t, p) not in data:
                ax.set_visible(False)
                continue
            sg, ns = data[(t, p)]
            m = np.isfinite(sg) & np.isfinite(ns)
            if m.sum() > 50000:                      # keep the PNG light
                idx = np.random.default_rng(0).choice(np.where(m)[0], 50000, replace=False)
            else:
                idx = np.where(m)[0]
            ax.plot(sg[idx], ns[idx], '.', ms=1, alpha=0.2, color='steelblue')
            lo = float(np.nanpercentile(sg[idx], 0.5)); hi = float(np.nanpercentile(sg[idx], 99.5))
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            st = paired_stats(sg, ns)
            plabel = f'{_BAND_LABELS[t]} {_PARAM_LABELS[p]}'
            if st:
                ax.set_title(f"{plabel}: median $\\Delta$={st['median_diff']:.3g} "
                             f"({st['median_rel_diff_pct']:.2f}%), r={st['pearson_r']:.4f}")
            ax.set_xlabel(f'{plabel} (Savitzky-Golay)')
            ax.set_ylabel(f'{plabel} (no smoothing)')
            ax.grid(alpha=0.3)
            panel_label(ax, f'({chr(ord("a") + i * 2 + j)})')
    fig.suptitle('Spectral-fit parameters: Savitzky-Golay smoothed vs raw ln(T) fits', y=0.995)
    fig.tight_layout()
    png_path = os.path.join(args.output_dir, 'savgol_ab_scatter.png')
    fig.savefig(png_path, bbox_inches='tight')
    print(f"[saved] {csv_path}\n[saved] {png_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

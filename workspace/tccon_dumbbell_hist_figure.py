"""CSV-driven plot-only renderer of the TCCON station-day residual dumbbell,
with a residual-histogram panel underneath.

Panel (a) re-draws the per-station-day residual-to-TCCON dumbbell (raw, B11,
DE; error bar = footprint sigma) from the per-case comparison CSV written by
``tccon_comparison_report.py``, with the same marker styling. Panel (b) shows
the distributions of the three station-day residuals on common bins, each
with a Gaussian of the sample mean and standard deviation overlaid. No
collocation or recomputation happens here, so the figure can be restyled
without re-running the TCCON chain.

Usage (from the repo root):
    python workspace/tccon_dumbbell_hist_figure.py [--ref ak|direct]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (  # noqa: E402
    XCO2_BC_LABEL,
    XCO2_DE_LABEL,
    XCO2_LABEL,
    XCO2_RAW_LABEL,
    apply_manuscript_style,
    panel_label,
)

DEFAULT_CSV = ('results/model_comparison/deep_ensemble/'
               'de_beta_nll_prof_reg_foldpca_o05l15_m5/atrain/'
               'tccon_comparison_r100km.csv')

RAW_COLOR = '#fdbf6f'
BC_COLOR = 'steelblue'
DE_COLOR = 'green'
TCCON_BAND_LABEL = '±TCCON σ (per station-day)'
# Pale yellow so the ±band shading is told apart from the grey per-row TCCON
# sigma segments and from the three product colours.
BAND_COLOR = '#f5e6a3'
BAND_ALPHA = 0.45


def _columns(ref):
    """(label, bias col, sd col, rmse col, color, zorder) per product."""
    s = '' if ref == 'ak' else '_direct'
    return [
        (XCO2_RAW_LABEL, f'bias_raw{s}', f'raw_sd{s}', f'rmse_raw{s}',
         RAW_COLOR, 2),
        (XCO2_BC_LABEL, f'bias_before{s}', f'orig_sd{s}', f'rmse_before{s}',
         BC_COLOR, 3),
        (XCO2_DE_LABEL, f'bias_after{s}', f'corr_sd{s}', f'rmse_after{s}',
         DE_COLOR, 4),
    ]


def _tccon_band(ax, frame):
    """Per-row ±(reported TCCON sigma) segment at residual 0, as in
    tccon_comparison_report._tccon_band (reference independent)."""
    labelled = False
    for i, err in enumerate(frame['tccon_err_mean'].to_numpy(float)):
        if not np.isfinite(err):
            continue
        ax.fill_betweenx([i - 0.4, i + 0.4], -err, err, color='0.35',
                         alpha=0.45, lw=0, zorder=0,
                         label=None if labelled else TCCON_BAND_LABEL)
        labelled = True


def _stat_box(ax, frame, cols, header=None):
    """Top-left stat box: station-day mean |residual| ± sd and mean fp-RMSE,
    with an optional header line naming the station-day subset."""
    lines = [header] if header else []
    for label, bcol, _, rcol, _, _ in cols:
        b = frame[bcol].to_numpy(float)
        b = b[np.isfinite(b)]
        r = frame[rcol].to_numpy(float)
        r = r[np.isfinite(r)]
        if not b.size:
            continue
        rtxt = f"   mean fp-RMSE {np.mean(r):.2f}" if r.size else ""
        # Sample sd (ddof=1) across station-days, as quoted in the text.
        lines.append(f"{label}:  |residual| {np.mean(np.abs(b)):.2f} ± "
                     f"{np.std(np.abs(b), ddof=1):.2f}{rtxt}")
    # Wider pad and line spacing keep the plain-text header clear of the
    # frame (the mathtext superscripts make the product lines taller).
    return ax.text(0.015, 0.985, '\n'.join(lines), transform=ax.transAxes,
                   va='top', ha='left', fontsize=7, zorder=6, linespacing=1.4,
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray',
                             alpha=0.85))


def _draw_dumbbell(ax, frame, cols, band):
    n = len(frame)
    y = np.arange(n)
    biases = frame[[c[1] for c in cols]].to_numpy(float)
    for yi in range(n):
        pts = biases[yi][np.isfinite(biases[yi])]
        if pts.size:
            ax.plot([pts.min(), pts.max()], [yi, yi], '-', color='lightgray',
                    lw=1.2, zorder=1)
    for label, bcol, sdcol, _, color, z in cols:
        ax.errorbar(frame[bcol], y, xerr=frame[sdcol], fmt='o', ms=6,
                    color=color, ecolor=color, elinewidth=0.8, capsize=2.5,
                    capthick=0.8, markeredgecolor='black',
                    markeredgewidth=0.5, label=label, zorder=z)
    ax.axvline(0, color='k', lw=1)
    _tccon_band(ax, frame)
    ax.axvspan(-band, band, color=BAND_COLOR, alpha=BAND_ALPHA, lw=0,
               zorder=0, label=f'±{band:g} ppm residual')
    ax.set_yticks([])
    ax.set_ylabel(f'station-day (sorted by {XCO2_BC_LABEL} residual)')
    ax.set_ylim(-1, n)
    ax.grid(alpha=0.3, axis='x')


def _draw_hist(ax, frame, cols, band, bin_ppm):
    arrays = []
    for _, bcol, _, _, color, _ in cols:
        x = frame[bcol].to_numpy(float)
        arrays.append((x[np.isfinite(x)], color))
    allx = np.concatenate([a for a, _ in arrays])
    # Bins span the residuals only; the error bars would stretch them to
    # empty tails. Edges are aligned on zero so the ±band edges fall on bins.
    lo = np.floor(allx.min() / bin_ppm) * bin_ppm
    hi = np.ceil(allx.max() / bin_ppm) * bin_ppm
    bins = np.arange(lo, hi + bin_ppm / 2, bin_ppm)
    counts = []
    for x, color in arrays:
        ax.hist(x, bins, histtype='stepfilled', color=color, alpha=0.25, lw=0)
        c, _, _ = ax.hist(x, bins, histtype='step', color=color, lw=1.3)
        counts.append(c)
    counts = np.max(counts, axis=0)      # tallest bar per bin, any series
    # Gaussian with the sample mean and sd of each series, scaled to counts
    # (N × bin width × pdf) so it overlays the histogram directly.
    xg = np.linspace(bins[0], bins[-1], 400)
    lines = ['Gaussian fit (dashed), ± 1 standard error:']
    for (x, color), (label, *_) in zip(arrays, cols):
        n = x.size
        mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
        # Standard errors of the mean and of the sd for a normal sample.
        se_mu = sd / np.sqrt(n)
        se_sd = sd / np.sqrt(2 * (n - 1))
        pdf = np.exp(-0.5 * ((xg - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        ax.plot(xg, n * bin_ppm * pdf, '--', color=color, lw=1.4, zorder=5)
        lines.append(f'{label}:  μ = {mu:.2f} ± {se_mu:.2f}   '
                     f'σ = {sd:.2f} ± {se_sd:.2f} ppm')
    # Same box style as the panel (a) stat box so the two read as a pair.
    # The wider pad and line spacing keep the plain-text header line clear
    # of the frame: the mathtext superscripts make the other lines taller.
    txt = ax.text(0.015, 0.96, '\n'.join(lines), transform=ax.transAxes,
                  va='top', ha='left', fontsize=7, zorder=6, linespacing=1.4,
                  bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray',
                            alpha=0.85))
    ax.axvline(0, color='k', lw=1)
    ax.axvspan(-band, band, color=BAND_COLOR, alpha=BAND_ALPHA, lw=0,
               zorder=0)
    # Headroom so the tallest bar does not touch the frame.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
    ax.set_ylabel('station-days')
    ax.set_xlabel(f'{XCO2_LABEL} residual to TCCON (ppm)')
    ax.grid(alpha=0.3, axis='x')
    return txt, bins, counts


def _axes_frac_bbox(fig, ax, artists):
    """Union of the artists' frames as (x0, y0, x1, y1) in axes fractions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for a in artists:
        patch = a.get_bbox_patch() if hasattr(a, 'get_bbox_patch') else None
        boxes.append((patch or a).get_window_extent(renderer))
    inv = ax.transAxes.inverted()
    x0, y0 = inv.transform((min(b.x0 for b in boxes), min(b.y0 for b in boxes)))
    x1, y1 = inv.transform((max(b.x1 for b in boxes), max(b.y1 for b in boxes)))
    return x0, y0, x1, y1


def _clear_dumbbell_annotations(fig, ax, frame, cols, artists, margin=0.02):
    """Raise the top of the y-axis when the stat box / legend would sit on
    data rows. The 99-row A-Train figure has a clear top-left corner (the
    most positive residuals plot on the right), but small subsets spread a
    few rows over the full height and the annotations land on them. Only
    rows whose left extent reaches under the annotations count as a clash,
    so the full figure keeps its canvas."""
    x0, y0, x1, y1 = _axes_frac_bbox(fig, ax, artists)
    xlo, xhi = ax.get_xlim()
    x_right = xlo + x1 * (xhi - xlo)          # annotation right edge, data x
    n = len(frame)
    ylo, yhi = ax.get_ylim()
    # Markers and TCCON bands count; error-bar whiskers do not, since a
    # whisker tip under the box corner is tolerable and would otherwise
    # push the whole 75-row figure down for one long bar.
    left = np.full(n, np.inf)
    for _, bcol, *_ in cols:
        left = np.fmin(left, frame[bcol].to_numpy(float))
    if 'tccon_err_mean' in frame.columns:
        left = np.fmin(left, -np.nan_to_num(frame['tccon_err_mean']
                                            .to_numpy(float)))
    row_top_frac = (np.arange(n) + 0.5 - ylo) / (yhi - ylo)
    clash = (left < x_right) & (row_top_frac > y0 - margin)
    if not clash.any():
        return
    # Put row n-1 (top edge n-0.5) just below the annotation block.
    new_top = (n - 0.5 - ylo) / (y0 - margin) + ylo
    ax.set_ylim(ylo, new_top)


def _clear_hist_annotation(fig, ax, txt, bins, counts, margin=0.03):
    """Keep the panel (b) stat box off the bars: move it to the right
    corner when the bars there are lower than under the left corner, then
    raise the y-axis if the tallest bar under it still reaches the box."""
    x0, y0, x1, y1 = _axes_frac_bbox(fig, ax, [txt])
    xlo, xhi = ax.get_xlim()
    width = (x1 - x0) * (xhi - xlo)
    starts = bins[:-1]
    h_left = counts[starts < xlo + x1 * (xhi - xlo)].max(initial=0)
    h_right = counts[bins[1:] > xhi - width - x0 * (xhi - xlo)].max(initial=0)
    if h_right < h_left:
        txt.set_x(1 - x0)
        txt.set_ha('right')
        h = h_right
    else:
        h = h_left
    top = ax.get_ylim()[1]
    if h <= (y0 - margin) * top:
        return
    ax.set_ylim(0, h / (y0 - margin))


def _legend_below_box(fig, ax, txt, band):
    """Place the legend directly under the stat box, left-aligned with it.

    The box height depends on the rendered mathtext, so its edges are
    measured after a draw instead of guessed. The rounded bbox patch (not the
    bare text extent) is measured, and borderaxespad is zeroed, so the legend
    frame lines up with the box frame instead of being inset by the padding.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = txt.get_bbox_patch().get_window_extent(renderer)
    x_left, y_bottom = ax.transAxes.inverted().transform((bbox.x0, bbox.y0))
    handles, labels = ax.get_legend_handles_labels()
    lookup = dict(zip(labels, handles))
    order = [f'±{band:g} ppm residual', TCCON_BAND_LABEL,
             XCO2_RAW_LABEL, XCO2_BC_LABEL, XCO2_DE_LABEL]
    order = [lab for lab in order if lab in lookup]
    return ax.legend([lookup[lab] for lab in order], order, loc='upper left',
                     bbox_to_anchor=(x_left, y_bottom - 0.012),
                     borderaxespad=0.0, fontsize=7, framealpha=0.85)


def _select_subset(df, qf, cld_min, cld_max):
    """Rows of one QF group, optionally windowed on the station-day mean
    nearest-cloud distance (cld_min <= cld_dist_mu < cld_max, km).

    Returns (frame, filename suffix, stat-box header)."""
    df = df[(df['surface'] == 'all') & (df['qf_group'] == qf)]
    sfx, parts = '', []
    if qf != 'all':
        sfx += f'_{qf}'
        parts.append(f'QF = {qf[-1]} footprints')
    else:
        parts.append('all footprints')
    if cld_min is not None or cld_max is not None:
        cld = df['cld_dist_mu'].to_numpy(float)
        keep = np.isfinite(cld)
        if cld_min is not None:
            keep &= cld >= cld_min
        if cld_max is not None:
            keep &= cld < cld_max
        df = df[keep]
        if cld_min is not None and cld_max is not None:
            sfx += f'_cld{cld_min:g}to{cld_max:g}'
            parts.append(f'{cld_min:g} ≤ mean cloud distance < {cld_max:g} km')
        elif cld_max is not None:
            sfx += f'_cldlt{cld_max:g}'
            parts.append(f'mean cloud distance < {cld_max:g} km')
        else:
            sfx += f'_cldge{cld_min:g}'
            parts.append(f'mean cloud distance ≥ {cld_min:g} km')
    header = ', '.join(parts) + f' ({len(df)} station-days)'
    return df, sfx, header


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--csv', default=DEFAULT_CSV)
    parser.add_argument('--ref', choices=('ak', 'direct'), default='ak')
    parser.add_argument('--qf', choices=('all', 'qf0', 'qf1'), default='all',
                        help='OCO-2 quality-flag group of the per-case rows.')
    parser.add_argument('--cld-min', type=float, default=None,
                        help='Keep station-days with mean nearest-cloud '
                             'distance >= this (km).')
    parser.add_argument('--cld-max', type=float, default=None,
                        help='Keep station-days with mean nearest-cloud '
                             'distance < this (km).')
    parser.add_argument('--out', default=None)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--band-ppm', type=float, default=1.0)
    parser.add_argument('--bin-ppm', type=float, default=0.25,
                        help='Histogram bin width in ppm (default 0.25).')
    args = parser.parse_args()

    csv = Path(args.csv)
    df, sub_sfx, header = _select_subset(pd.read_csv(csv), args.qf,
                                         args.cld_min, args.cld_max)
    if not len(df):
        sys.exit('no station-days match the requested subset')
    sfx = csv.stem.split('tccon_comparison', 1)[-1]
    out = (Path(args.out) if args.out else
           csv.parent / f'tccon_{args.ref}_bias_dumbbell_hist{sub_sfx}{sfx}.png')
    cols = _columns(args.ref)
    # Sort by the B11 residual so the most negative rows sit at the bottom.
    df = df.sort_values(cols[1][1]).reset_index(drop=True)

    apply_manuscript_style()
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.3], hspace=0.06)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_a.tick_params(labelbottom=False)

    _draw_dumbbell(ax_a, df, cols, args.band_ppm)
    txt = _stat_box(ax_a, df, cols, header)
    txt_b, bins, counts = _draw_hist(ax_b, df, cols, args.band_ppm,
                                     args.bin_ppm)
    panel_label(ax_a, '(a)')
    panel_label(ax_b, '(b)')
    leg = _legend_below_box(fig, ax_a, txt, args.band_ppm)
    _clear_dumbbell_annotations(fig, ax_a, df, cols, [txt, leg])
    _clear_hist_annotation(fig, ax_b, txt_b, bins, counts)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'{len(df)} station-days -> {out}')


if __name__ == '__main__':
    main()

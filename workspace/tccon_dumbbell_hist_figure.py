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
import matplotlib.colors  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms  # noqa: E402

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# v2 trees by default; RUN_VERSION= (empty) selects the v1 trees.
from constants import DE_PRODUCTION_TAG  # noqa: E402
DEFAULT_CSV = str(Path('results/model_comparison/deep_ensemble', DE_PRODUCTION_TAG,
                       'atrain/tccon_comparison_r100km.csv'))

# Okabe-Ito blue / vermillion differ in hue AND lightness (readable by
# colour-blind readers and in greyscale); raw is pushed back in light grey.
# The corrected product is X_CO2^ML in the text and every figure (one name per
# thing, 2026-10-12); the shared plot_style label carries it.
RAW_COLOR = '0.6'
BC_COLOR = '#0072B2'
DE_COLOR = '#D55E00'
TCCON_BAND_LABEL = '±TCCON σ'   # per station-day; the caption says so
# Light blue-grey (author, 2026-10-11; was pale yellow #f5e6a3, which Massie disliked in
# Fig. S5) so the ±1 ppm band stays distinct from the neutral grey per-row TCCON
# sigma segments and from the three product colours.
BAND_COLOR = '#b8c7d9'
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


def _stat_table(ax, frame, cols, header=None, fs=9.0,
                xs=(0.10, 0.36, 0.60, 0.83)):
    """Station-day statistics as a small three-column table ABOVE panel (a)
    (author, 2026-10-11: the in-frame box pushed the data into the lower
    60 % of the panel). Columns: station-day mean |residual| ± sd, mean
    fp-RMSE, and station-days within ±1 ppm. The default subset header
    ('all footprints (N station-days)') is left to the caption."""
    rows = []
    for label, bcol, _, rcol, color, _ in cols:
        b = frame[bcol].to_numpy(float)
        b = b[np.isfinite(b)]
        r = frame[rcol].to_numpy(float)
        r = r[np.isfinite(r)]
        if not b.size:
            continue
        # Sample sd (ddof=1) across station-days, as quoted in the text.
        rows.append((label, color,
                     f"{np.mean(np.abs(b)):.2f} ± {np.std(np.abs(b), ddof=1):.2f}",
                     f"{np.mean(r):.2f}" if r.size else "–",
                     f"{int(np.sum(np.abs(b) <= 1.0))}/{b.size}"))
    # xs: label column, then three value columns (axes fractions).
    # Row pitch fixed in points (2.3 x font size), converted to axes
    # fractions, so the table keeps its spacing when the panel height changes.
    # The product labels carry both a sub- and a superscript (about 2.2 x font
    # size tall); a smaller pitch lets one row's subscript hit the next row's
    # superscript.
    fig = ax.figure
    ax_h_pt = ax.get_position().height * fig.get_figheight() * 72.0
    dy = 2.3 * fs / ax_h_pt
    y = 1.02 + dy * len(rows)                # header line on top, products below
    heads = ('', '|residual| (ppm)', 'mean fp-RMSE (ppm)', 'within ±1 ppm')
    out = []
    if header and not header.startswith('all footprints ('):
        out.append(ax.text(xs[0], y + dy, header, transform=ax.transAxes,
                           fontsize=fs, ha='left', va='bottom', color='0.3'))
    for x, h in zip(xs[1:], heads[1:]):
        out.append(ax.text(x, y, h, transform=ax.transAxes, fontsize=fs,
                           ha='center', va='bottom', color='0.3'))
    for k, (label, color, *vals) in enumerate(rows):
        yk = y - dy * (k + 1)
        tc = '0.35' if color == RAW_COLOR else color
        out.append(ax.text(xs[0], yk, label, transform=ax.transAxes,
                           fontsize=fs + 0.5, ha='left', va='bottom', color=tc))
        for x, v in zip(xs[1:], vals):
            out.append(ax.text(x, yk, v, transform=ax.transAxes, fontsize=fs,
                               ha='center', va='bottom', color=tc))
    return out


def _draw_dumbbell(ax, frame, cols, band):
    n = len(frame)
    y = np.arange(n)
    biases = frame[[c[1] for c in cols]].to_numpy(float)
    for yi in range(n):
        pts = biases[yi][np.isfinite(biases[yi])]
        if pts.size:
            ax.plot([pts.min(), pts.max()], [yi, yi], '-', color='lightgray',
                    lw=1.2, zorder=1)
    for k, (label, bcol, sdcol, _, color, z) in enumerate(cols):
        if k == 0:
            # Raw: small hollow grey markers, no error bars (background).
            ax.plot(frame[bcol], y, 'o', ms=4, mfc='none', mec=color,
                    mew=0.8, label=label, zorder=z)
            continue
        ecol = matplotlib.colors.to_rgba(color, 0.55)
        ax.errorbar(frame[bcol], y, xerr=frame[sdcol], fmt='o', ms=5,
                    color=color, ecolor=ecol, elinewidth=0.6, capsize=1.8,
                    capthick=0.6, markeredgecolor='black',
                    markeredgewidth=0.4, label=label, zorder=z)
    ax.axvline(0, color='k', lw=1)
    _tccon_band(ax, frame)
    ax.axvspan(-band, band, color=BAND_COLOR, alpha=BAND_ALPHA, lw=0,
               zorder=0, label=f'±{band:g} ppm residual')
    ax.set_yticks([])
    ax.set_ylabel(f'overpass (sorted by {XCO2_BC_LABEL} residual)')
    # Rows fill the panel (stat table above the frame, legend in the empty
    # upper-left corner that the sort leaves free).
    ax.set_ylim(-1.2, n + 0.2)
    ax.grid(alpha=0.3, axis='x')


def _draw_hist(ax, frame, cols, band, bin_ppm, xlim=None, headroom=1.15):
    """Mirrored residual distributions: the uncorrected products (raw, B11)
    above the zero line, the corrected DE below.

    B11 and DE are thin step histograms with their Gaussians (sample mean and
    sd) as thick solid curves, ±1 SD arrows, and direct labels. Raw sits in
    the background on the upper side as a faint step outline and a dotted
    grey Gaussian (its width equals B11's). Returns the x range drawn."""
    series = []
    for k, (label, bcol, _, _, color, _) in enumerate(cols):
        x = frame[bcol].to_numpy(float)
        series.append((label, x[np.isfinite(x)], color, k == 0))
    allx = np.concatenate([x for _, x, _, _ in series])
    lo = np.floor(allx.min() / bin_ppm) * bin_ppm
    hi = np.ceil(allx.max() / bin_ppm) * bin_ppm
    bins = np.arange(lo, hi + bin_ppm / 2, bin_ppm)
    if xlim is None:
        xlim = (np.floor(allx.min()) - 0.5, np.ceil(allx.max()) + 1.5)
    xg = np.linspace(*xlim, 600)
    peak = 0.0
    for sign, (label, x, color, is_raw) in zip((1, 1, -1), series):
        c, _ = np.histogram(x, bins)
        ax.stairs(sign * c, bins, color=color, lw=0.7 if is_raw else 0.8,
                  alpha=0.35 if is_raw else 0.55, baseline=0,
                  zorder=2 if is_raw else 3)
        n = x.size
        mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
        se_mu, se_sd = sd / np.sqrt(n), sd / np.sqrt(2 * (n - 1))
        g = n * bin_ppm * np.exp(-0.5 * ((xg - mu) / sd) ** 2) / (
            sd * np.sqrt(2 * np.pi))
        gmax = n * bin_ppm / (sd * np.sqrt(2 * np.pi))
        peak = max(peak, gmax, c.max())
        if is_raw:
            # Drawn above the B11 curve: same width, means 0.2 ppm apart.
            ax.plot(xg, g, ':', color='0.15', lw=1.8, zorder=6)
            ax.text(0.015, 0.95,
                    f'{label} (dotted)\nSD = {sd:.2f} ± {se_sd:.2f} ppm\n'
                    f'mean = {mu:.2f} ± {se_mu:.2f} ppm',
                    transform=ax.transAxes, ha='left', va='top',
                    color='0.35', fontsize=9, linespacing=1.3, zorder=7)
            continue
        ax.plot(xg, sign * g, '-', color=color, lw=2.5, zorder=5)
        # ±1 SD arrow at the height where the Gaussian falls to exp(-1/2).
        h = sign * gmax * np.exp(-0.5)
        ax.annotate('', xy=(mu - sd, h), xytext=(mu + sd, h),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=1.3,
                                    shrinkA=0, shrinkB=0), zorder=6)
        # Words, not mu/sigma: the manuscript reserves no symbol for these
        # sample statistics (mu was retired from the notation, 2026-10).
        ax.text(0.985, 0.95 if sign > 0 else 0.05,
                f'{label}\nSD = {sd:.2f} ± {se_sd:.2f} ppm\n'
                f'mean = {mu:.2f} ± {se_mu:.2f} ppm',
                transform=ax.transAxes, ha='right',
                va='top' if sign > 0 else 'bottom', color=color, fontsize=9,
                linespacing=1.3, zorder=7)
    ax.axhline(0, color='k', lw=0.8, zorder=4)
    ax.axvline(0, color='k', lw=0.8, zorder=4)
    ax.axvspan(-band, band, color=BAND_COLOR, alpha=BAND_ALPHA, lw=0,
               zorder=0)
    ax.set_xlim(*xlim)
    top = peak * headroom
    ax.set_ylim(-top, top)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{abs(v):g}'))
    # Ticks every 5 (or finer for small subsets) on the shorter print panel.
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(
        nbins=7, steps=[1, 2, 5, 10], integer=True))
    ax.set_ylabel('overpasses')
    ax.set_xlabel(f'{XCO2_LABEL} residual to TCCON (ppm)')
    ax.grid(alpha=0.3, axis='x')
    return xlim


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
    parser.add_argument('--figsize', type=float, nargs=2,
                        default=(7.4, 7.5), metavar=('W', 'H'),
                        help='Canvas size in inches before the tight crop.')
    parser.add_argument('--height-ratios', type=float, nargs=2,
                        default=(3.6, 1.9), metavar=('A', 'B'),
                        help='Height ratio of panel (a) to panel (b).')
    parser.add_argument('--hspace', type=float, default=0.22)
    parser.add_argument('--no-hist', action='store_true',
                        help='Draw the dumbbell panel only (no histogram '
                             'panel, no panel letter).')
    parser.add_argument('--row-labels', action='store_true',
                        help="Label each row with 'site date' on the y axis.")
    parser.add_argument('--hist-headroom', type=float, default=1.15,
                        help='Panel (b) y half-range as a multiple of the '
                             'tallest bar/Gaussian (room for the stat text).')
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
    # Print-size canvas (2026-10-11): drawn near its printed width (6.5 in
    # text width) so the 9-10 pt fonts print at size; the 99-row panel (a)
    # is shorter than the earlier 7.2 x 10 in canvas so figure + caption fit
    # one page. --figsize / --height-ratios resize it for small subsets.
    fig = plt.figure(figsize=tuple(args.figsize))
    if args.no_hist:
        # Dumbbell only (SI drift figure, 2026-10-11): no panel (b), no letter.
        ax_a, ax_b = fig.add_subplot(111), None
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=list(args.height_ratios),
                              hspace=args.hspace)
        ax_a = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1])

    _draw_dumbbell(ax_a, df, cols, args.band_ppm)
    if args.row_labels:
        # Station code + date per row, as the pre-2026-09-24 drift figure had.
        ax_a.set_yticks(np.arange(len(df)))
        ax_a.set_yticklabels([f'{r.site} {r.date}' for r in df.itertuples()],
                             fontsize=9)
        ax_a.tick_params(axis='y', length=0, pad=3)
    # Both panels share one x range (author, 2026-10-11), so the bar that
    # marked (b)'s zoom range on (a) is gone. Left edge at -9.5 ppm (KSS
    # 2026-10-11 clip); the few whiskers beyond are cut at the axis.
    allb = df[[c[1] for c in cols]].to_numpy(float)
    allb = allb[np.isfinite(allb)]
    xlim = (max(np.floor(allb.min()) - 0.5, -10.0), np.ceil(allb.max()) + 2.0)
    ax_a.set_xlim(*xlim)
    ax_a.set_xlabel(f'{XCO2_LABEL} residual to TCCON (ppm)')
    # With row labels the axes is narrower: the product-name column moves
    # over the tick-label strip so the three headers keep their spacing.
    _stat_table(ax_a, df, cols, header,
                xs=((-0.22, 0.24, 0.56, 0.86) if args.row_labels
                    else (0.10, 0.36, 0.60, 0.83)))
    if ax_b is not None:
        _draw_hist(ax_b, df, cols, args.band_ppm, args.bin_ppm, xlim=xlim,
                   headroom=args.hist_headroom)
        panel_label(ax_a, '(a)')
        panel_label(ax_b, '(b)')
    handles, labels = ax_a.get_legend_handles_labels()
    lookup = dict(zip(labels, handles))
    order = [f'±{args.band_ppm:g} ppm residual', TCCON_BAND_LABEL,
             XCO2_RAW_LABEL, XCO2_BC_LABEL, XCO2_DE_LABEL]
    ax_a.legend([lookup[k] for k in order], order, loc='upper left',
                fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.8',
                borderaxespad=0.6, handlelength=1.3, handletextpad=0.5,
                labelspacing=0.35)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'{len(df)} station-days -> {out}')


if __name__ == '__main__':
    main()

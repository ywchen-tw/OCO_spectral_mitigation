"""tccon_comparison_report.py — combined before/after-vs-TCCON summary across cases.

Reads the active `run_case` lines from a deep-ensemble plotting script, and for each
case reproduces the figure's comparison logic:
    drop guarded footprints → lon/lat box → footprints ≤ --radius-km of the TCCON
    station → TCCON within ± --window-min of the OCO-2 pass on that date.
Then computes mean ± std of the ORIGINAL (xco2_bc), CORRECTED, and TCCON XCO2, and
the bias to TCCON before/after correction.

Outputs (to --output-dir):
    tccon_comparison.csv   — one row per case
    tccon_comparison.md    — markdown table + aggregate summary
    tccon_comparison.png   — (a) corrected/original vs TCCON scatter (1:1)
                              (b) bias-to-TCCON before→after dumbbell, per case

Example:
    PYTHONPATH=src python workspace/tccon_comparison_report.py \
        --script curc_shell_blanca_plot_corr_xco2_deepens.sh \
        --output-dir results/model_comparison/deep_ensemble
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_corrected_xco2 import load_tccon, _haversine_km, get_storage_dir

CORR = 'deep_ensemble_corrected_xco2'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--script', default='curc_shell_blanca_plot_corr_xco2_deepens.sh')
    ap.add_argument('--out-base', default='results/model_comparison/deep_ensemble',
                    help='Base dir holding combined_DATE[_SITE] case dirs.')
    ap.add_argument('--output-dir', default='results/model_comparison/deep_ensemble')
    ap.add_argument('--radius-km', type=float, default=100.0)
    ap.add_argument('--window-min', type=float, default=60.0)
    args = ap.parse_args()

    out_base = Path(args.out_base)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    storage = get_storage_dir()

    # ── parse active run_case lines ────────────────────────────────────────────
    cases = []
    for ln in Path(args.script).read_text().splitlines():
        if not re.match(r'^run_case\s', ln):
            continue
        t = ln.split()
        date, tccon = t[1], t[2]
        lonmin, lonmax, latmin, latmax = map(float, t[3:7])
        rest = t[9:]
        site = rest[2] if len(rest) >= 3 else ''
        cases.append(dict(date=date, tccon=tccon, lonmin=lonmin, lonmax=lonmax,
                          latmin=latmin, latmax=latmax, site=site))

    _tccon_cache = {}
    def tccon_df(name):
        if name not in _tccon_cache:
            p = Path('data/TCCON') / name
            p = p if p.exists() else storage / 'data/TCCON' / name
            _tccon_cache[name] = load_tccon(str(p))
        return _tccon_cache[name]

    rows = []
    for c in cases:
        site = c['site'] or c['tccon'][:2]
        outdir = out_base / (f"combined_{c['date']}_{c['site']}" if c['site']
                             else f"combined_{c['date']}_all_orbits")
        pq = outdir / 'plot_data.parquet'
        if not pq.exists():
            continue
        oco = pd.read_parquet(pq)
        # drop guarded
        g = np.zeros(len(oco), bool)
        for gc in ('clim_guard', 'anomaly_guard'):
            if gc in oco: g |= oco[gc].to_numpy(bool)
        oco = oco[~g]
        # lon/lat box
        oco = oco[(oco['lon'] >= c['lonmin']) & (oco['lon'] <= c['lonmax']) &
                  (oco['lat'] >= c['latmin']) & (oco['lat'] <= c['latmax'])]
        if len(oco) == 0:
            continue
        tc = tccon_df(c['tccon'])
        st_lon = float(tc['lon'].median()) if len(tc) else np.nan
        st_lat = float(tc['lat'].median()) if len(tc) else np.nan
        # ≤ radius of station
        if np.isfinite(st_lon):
            d = _haversine_km(oco['lon'].values, oco['lat'].values, st_lon, st_lat)
            near = oco[d <= args.radius_km]
        else:
            near = oco
        if len(near) == 0:
            continue
        orig = near['xco2_bc'].to_numpy(float)
        corr = near[CORR].to_numpy(float)

        # TCCON within ±window of the OCO pass that day
        tmu = tsd = np.nan; n_tc = 0
        if len(tc) and 'time' in near.columns:
            ot = pd.to_datetime(near['time'], unit='s', utc=True, errors='coerce').dropna()
            if len(ot):
                w = pd.Timedelta(minutes=args.window_min)
                sub = tc[(tc['time'] >= ot.min() - w) & (tc['time'] <= ot.max() + w)]
                if len(sub):
                    tmu = float(sub['xco2'].mean()); tsd = float(sub['xco2'].std()); n_tc = len(sub)
        rows.append(dict(
            site=site, date=c['date'], n_oco=len(near), n_tccon=n_tc,
            orig_mu=np.nanmean(orig), orig_sd=np.nanstd(orig),
            corr_mu=np.nanmean(corr), corr_sd=np.nanstd(corr),
            tccon_mu=tmu, tccon_sd=tsd,
            bias_before=(np.nanmean(orig) - tmu) if n_tc else np.nan,
            bias_after=(np.nanmean(corr) - tmu) if n_tc else np.nan,
        ))

    rep = pd.DataFrame(rows).sort_values(['site', 'date']).reset_index(drop=True)
    rep.to_csv(out_dir / 'tccon_comparison.csv', index=False)

    cmp = rep[rep['n_tccon'] > 0].copy()
    # ── markdown ───────────────────────────────────────────────────────────────
    def f(x, n=2): return '' if pd.isna(x) else f'{x:.{n}f}'
    lines = ['# OCO-2 corrected vs TCCON — combined comparison', '',
             f'{len(rep)} cases ({len(cmp)} with TCCON in ±{args.window_min:g} min, '
             f'≤{args.radius_km:g} km).  XCO2 in ppm (mean ± std).', '',
             '| site | date | n_oco | n_tccon | original | corrected | TCCON | bias before | bias after |',
             '|---|---|--:|--:|---|---|---|--:|--:|']
    for _, r in rep.iterrows():
        lines.append(f"| {r['site']} | {r['date']} | {r['n_oco']} | {r['n_tccon']} | "
                     f"{f(r['orig_mu'])}±{f(r['orig_sd'])} | {f(r['corr_mu'])}±{f(r['corr_sd'])} | "
                     f"{f(r['tccon_mu'])}±{f(r['tccon_sd'])} | {f(r['bias_before'],2)} | {f(r['bias_after'],2)} |")
    site_agg = pd.DataFrame()
    if len(cmp):
        bb, ba = cmp['bias_before'].to_numpy(), cmp['bias_after'].to_numpy()
        lines += ['', '## Aggregate (cases with TCCON)', '',
                  f"- mean |bias|:  before **{np.nanmean(np.abs(bb)):.2f}** → after **{np.nanmean(np.abs(ba)):.2f}** ppm",
                  f"- RMS bias:    before **{np.sqrt(np.nanmean(bb**2)):.2f}** → after **{np.sqrt(np.nanmean(ba**2)):.2f}** ppm",
                  f"- mean OCO std: before **{cmp['orig_sd'].mean():.2f}** → after **{cmp['corr_sd'].mean():.2f}** ppm",
                  f"- improved (|bias| down) in **{int((np.abs(ba)<np.abs(bb)).sum())}/{len(cmp)}** cases"]

        # ── per-site aggregate ─────────────────────────────────────────────────
        c2 = cmp.copy()
        c2['abs_before'] = c2['bias_before'].abs(); c2['abs_after'] = c2['bias_after'].abs()
        c2['improved'] = c2['abs_after'] < c2['abs_before']
        site_agg = (c2.groupby('site')
                    .agg(n=('date', 'size'),
                         mean_abs_bias_before=('abs_before', 'mean'),
                         mean_abs_bias_after=('abs_after', 'mean'),
                         mean_sd_before=('orig_sd', 'mean'),
                         mean_sd_after=('corr_sd', 'mean'),
                         n_improved=('improved', 'sum'))
                    .reset_index().sort_values('mean_abs_bias_after'))
        site_agg.to_csv(out_dir / 'tccon_comparison_by_site.csv', index=False)
        lines += ['', '## Per-site aggregate (cases with TCCON, sorted by post-correction |bias|)', '',
                  '| site | n | mean \\|bias\\| before | after | mean σ before | after | improved |',
                  '|---|--:|--:|--:|--:|--:|--:|']
        for _, s in site_agg.iterrows():
            lines.append(f"| {s['site']} | {int(s['n'])} | {s['mean_abs_bias_before']:.2f} | "
                         f"**{s['mean_abs_bias_after']:.2f}** | {s['mean_sd_before']:.2f} | "
                         f"{s['mean_sd_after']:.2f} | {int(s['n_improved'])}/{int(s['n'])} |")
    (out_dir / 'tccon_comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # ── figure ─────────────────────────────────────────────────────────────────
    if len(cmp):
        cmp = cmp.sort_values('bias_before').reset_index(drop=True)
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 7))
        # (a) scatter vs TCCON (1:1)
        axA.errorbar(cmp['tccon_mu'], cmp['orig_mu'], xerr=cmp['tccon_sd'], yerr=cmp['orig_sd'],
                     fmt='o', ms=5, color='steelblue', alpha=0.5, elinewidth=0.8, label='original XCO₂_bc')
        axA.errorbar(cmp['tccon_mu'], cmp['corr_mu'], xerr=cmp['tccon_sd'], yerr=cmp['corr_sd'],
                     fmt='o', ms=5, color='green', alpha=0.7, elinewidth=0.8, label='corrected')
        lo = float(np.nanmin([cmp['tccon_mu'].min(), cmp['corr_mu'].min(), cmp['orig_mu'].min()])) - 1
        hi = float(np.nanmax([cmp['tccon_mu'].max(), cmp['corr_mu'].max(), cmp['orig_mu'].max()])) + 1
        axA.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')
        axA.set_xlabel('TCCON XCO₂ (ppm)'); axA.set_ylabel('OCO-2 XCO₂ (ppm)')
        axA.set_title('OCO-2 vs TCCON (mean ± std)'); axA.legend(); axA.grid(alpha=0.3)
        axA.set_xlim(lo, hi); axA.set_ylim(lo, hi); axA.set_aspect('equal')
        # (b) bias before→after dumbbell, per case
        y = np.arange(len(cmp))
        for yi, (bb, ba) in enumerate(zip(cmp['bias_before'], cmp['bias_after'])):
            axB.plot([bb, ba], [yi, yi], '-', color='lightgray', lw=1.5, zorder=1)
        axB.scatter(cmp['bias_before'], y, color='steelblue', s=28, label='before', zorder=2)
        axB.scatter(cmp['bias_after'], y, color='green', s=28, label='after', zorder=3)
        axB.axvline(0, color='k', lw=1)
        axB.set_yticks(y); axB.set_yticklabels([f"{r.site} {r.date}" for r in cmp.itertuples()], fontsize=6)
        axB.set_xlabel('XCO₂ bias to TCCON (ppm)'); axB.set_title('Bias before → after correction')
        axB.legend(); axB.grid(alpha=0.3, axis='x')
        fig.tight_layout()
        fig.savefig(out_dir / 'tccon_comparison.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

        # ── per-site bar chart: mean |bias| and σ, before vs after ─────────────
        if len(site_agg):
            sa = site_agg.sort_values('mean_abs_bias_before', ascending=False).reset_index(drop=True)
            x = np.arange(len(sa)); w = 0.38
            fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(max(10, 1.0 * len(sa)), 6))
            bx1.bar(x - w/2, sa['mean_abs_bias_before'], w, color='steelblue', label='before')
            bx1.bar(x + w/2, sa['mean_abs_bias_after'], w, color='green', label='after')
            bx1.set_xticks(x); bx1.set_xticklabels(
                [f"{r.site}\n({int(r.n)}, {int(r.n_improved)}↓)" for r in sa.itertuples()], fontsize=8)
            bx1.set_ylabel('mean |bias to TCCON| (ppm)')
            bx1.set_title('Per-site mean |bias| before → after'); bx1.legend(); bx1.grid(alpha=0.3, axis='y')
            bx2.bar(x - w/2, sa['mean_sd_before'], w, color='steelblue', label='before')
            bx2.bar(x + w/2, sa['mean_sd_after'], w, color='green', label='after')
            bx2.set_xticks(x); bx2.set_xticklabels([r.site for r in sa.itertuples()], fontsize=8)
            bx2.set_ylabel('mean OCO-2 σ (ppm)')
            bx2.set_title('Per-site mean scatter (σ) before → after'); bx2.legend(); bx2.grid(alpha=0.3, axis='y')
            fig2.tight_layout()
            fig2.savefig(out_dir / 'tccon_comparison_by_site.png', dpi=200, bbox_inches='tight')
            plt.close(fig2)
        print(f"\n[saved] {out_dir/'tccon_comparison.csv'}\n[saved] {out_dir/'tccon_comparison.md'}"
              f"\n[saved] {out_dir/'tccon_comparison.png'}\n[saved] {out_dir/'tccon_comparison_by_site.csv'}"
              f"\n[saved] {out_dir/'tccon_comparison_by_site.png'}")


if __name__ == '__main__':
    main()

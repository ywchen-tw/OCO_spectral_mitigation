"""uncertainty_calibration.py — §4.2 calibration study (the AUTHORITATIVE gate) for
the uncertainty-aware TCCON comparison (src/analysis/UNCERTAINTY_AWARE_TCCON_COMPARISON.md).

Question: which Side-A per-footprint uncertainty definition is *calibrated* against
the out-of-sample residual?  The realized per-footprint error is the anomaly
residual  resid = y_true - mu  (= corrected XCO2 - clean-scene truth proxy), so a
candidate σ is calibrated iff the standardized residual z = resid/σ has ⟨z²⟩ ≈ 1
and its central-interval coverage matches nominal (68.3% at 1σ, 95.4% at 2σ).

Candidate budgets (auto-detected from available columns):
  de_total       DE predictive σ         (col `sigma` or `de_sigma`)
  de_aleatoric   DE aleatoric σ          (col `de_aleatoric_sigma`)
  retrieval      OCO-2 L2 posterior      (col `xco2_uncertainty`)
  quad_de_retr   sqrt(de_total² + retrieval²)          [retrieval ⟂ DE hypothesis]
  quad_alea_retr sqrt(de_aleatoric² + retrieval²)      [retrieval + model residual]

MUST be run on OUT-OF-SAMPLE rows or the residuals are optimistically small.
The DE fold `held_out_predictions.parquet` (pooled across folds) is the proper
out-of-fold set and carries y_true / mu / sigma.

Output: <outdir>/uncertainty_calibration.{md,png} + a printed SIDE_A_BUDGET verdict.

Example (out-of-fold, per surface):
    PYTHONPATH=src python workspace/uncertainty_calibration.py \
        --held-out-glob 'results/model_deep_ensemble/de_ocean_beta_nll_prof_r05_f*/held_out_predictions.parquet' \
        --tag ocean --outdir results/model_comparison/deep_ensemble/uncertainty_phase2_ocean
"""
from __future__ import annotations

import argparse
import glob as _glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLD_EDGES = [0, 5, 20, 50, np.inf]
NOMINAL = np.array([0.5, 0.68, 0.8, 0.9, 0.95, 0.99])


def resid_and_candidates(df):
    """Return (resid, {name: sigma_array}) from whatever columns are present."""
    ycol = 'y_true' if 'y_true' in df.columns else 'xco2_bc_anomaly'
    mucol = 'mu' if 'mu' in df.columns else 'pred_anomaly'
    resid = df[ycol].to_numpy(float) - df[mucol].to_numpy(float)

    de_total = None
    for c in ('de_sigma', 'sigma'):
        if c in df.columns:
            de_total = df[c].to_numpy(float); break
    alea = df['de_aleatoric_sigma'].to_numpy(float) if 'de_aleatoric_sigma' in df.columns else None
    retr = df['xco2_uncertainty'].to_numpy(float) if 'xco2_uncertainty' in df.columns else None

    cand = {}
    if de_total is not None:
        cand['de_total'] = de_total
    if alea is not None:
        cand['de_aleatoric'] = alea
    if retr is not None:
        cand['retrieval'] = retr
    if de_total is not None and retr is not None:
        cand['quad_de_retr'] = np.sqrt(de_total**2 + retr**2)
    if alea is not None and retr is not None:
        cand['quad_alea_retr'] = np.sqrt(alea**2 + retr**2)
    return resid, cand


def metrics(resid, sigma):
    m = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
    r, s = resid[m], sigma[m]
    z = r / s
    z2 = float(np.mean(z**2))
    out = dict(n=int(m.sum()), rmse=float(np.sqrt(np.mean(r**2))),
               mean_z=float(np.mean(z)), z2=z2, inflate=float(np.sqrt(z2)),
               cov68=float(np.mean(np.abs(z) < 1.0)),
               cov95=float(np.mean(np.abs(z) < 1.959964)))
    # reliability: empirical central coverage at each nominal level
    emp = [float(np.mean(np.abs(z) < norm.ppf((1 + p) / 2))) for p in NOMINAL]
    out['reliability'] = emp
    return out


def by_cld(resid, sigma, cld):
    lab = pd.cut(pd.Series(cld), bins=CLD_EDGES, include_lowest=True)
    rows = []
    for iv, idx in pd.Series(range(len(cld))).groupby(lab, observed=True):
        i = idx.to_numpy()
        mm = metrics(resid[i], sigma[i])
        rows.append((str(iv), mm['n'], mm['z2'], mm['cov68'], mm['cov95']))
    return rows


def make_figure(resid, cand, cld, path, tag):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # (1) reliability
    for name, s in cand.items():
        mm = metrics(resid, s)
        ax[0].plot(NOMINAL, mm['reliability'], 'o-', label=f"{name} (⟨z²⟩={mm['z2']:.2f})")
    ax[0].plot([0, 1], [0, 1], 'k--', lw=1)
    ax[0].set(xlabel='nominal central coverage', ylabel='empirical coverage',
              title='Reliability (closer to diagonal = better)')
    ax[0].legend(fontsize=8)
    # (2) ⟨z²⟩ by cloud distance
    xs = None
    for name, s in cand.items():
        rows = by_cld(resid, s, cld)
        labels = [r[0].split(',')[0].strip('([') for r in rows]
        z2 = [r[2] for r in rows]
        xs = np.arange(len(rows))
        ax[1].plot(xs, z2, 'o-', label=name)
    ax[1].axhline(1.0, color='k', ls='--', lw=1, label='calibrated (⟨z²⟩=1)')
    if xs is not None:
        ax[1].set_xticks(xs); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set(xlabel='nearest-cloud distance (km)', ylabel='⟨z²⟩',
              title='Calibration vs cloud distance')
    ax[1].legend(fontsize=8)
    # (3) z-hist of best candidate vs N(0,1)
    best = min(cand.items(), key=lambda kv: abs(metrics(resid, kv[1])['z2'] - 1.0))
    z = resid / best[1]
    z = z[np.isfinite(z)]
    ax[2].hist(z, bins=np.linspace(-5, 5, 80), density=True, alpha=.6,
               label=f'z ({best[0]})')
    xx = np.linspace(-5, 5, 200)
    ax[2].plot(xx, norm.pdf(xx), 'r-', lw=1.5, label='N(0,1)')
    ax[2].set(xlim=(-5, 5), xlabel='z = resid/σ', ylabel='density',
              title=f'Standardized residual — best: {best[0]}')
    ax[2].legend(fontsize=8)
    fig.suptitle(f'Side-A calibration (out-of-sample) — {tag}', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_md(resid, cand, cld, path, tag, fig_name):
    L = [f"# Side-A calibration (§4.2, out-of-sample) — {tag}\n",
         f"n = {np.isfinite(resid).sum():,} footprints. Residual = y_true − mu "
         "(corrected − clean-scene truth). Calibrated ⇔ ⟨z²⟩≈1, cov68≈0.683, "
         "cov95≈0.954.\n",
         "## Overall\n",
         "| candidate | median σ | RMSE resid | mean z | **⟨z²⟩** | inflate√ | "
         "cov68 | cov95 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    summ = {}
    for name, s in cand.items():
        mm = metrics(resid, s); summ[name] = mm
        L.append(f"| {name} | {np.nanmedian(s):.3f} | {mm['rmse']:.3f} | "
                 f"{mm['mean_z']:+.3f} | **{mm['z2']:.3f}** | {mm['inflate']:.2f} | "
                 f"{mm['cov68']:.3f} | {mm['cov95']:.3f} |")
    # verdict: candidate with |⟨z²⟩−1| smallest (and not systematically over/under)
    best = min(summ.items(), key=lambda kv: abs(kv[1]['z2'] - 1.0))
    L.append("")
    L.append(f"**SIDE_A_BUDGET verdict ({tag}): `{best[0]}`** — ⟨z²⟩ = "
             f"{best[1]['z2']:.3f} (closest to 1); cov68 {best[1]['cov68']:.3f}, "
             f"cov95 {best[1]['cov95']:.3f}.")
    L.append("> ⟨z²⟩>1 → σ too small (under-confident); <1 → too large "
             "(over-confident). If `de_total` already ≈1, adding `retrieval` in "
             "quadrature (`quad_de_retr`) should push ⟨z²⟩ below 1 → do NOT add.\n")
    # per-cloud-distance for de_total and the best candidate
    for name in dict.fromkeys(['de_total', best[0]]):
        if name not in cand:
            continue
        L.append(f"## ⟨z²⟩ / coverage by cloud distance — {name}\n")
        L.append("| cld_dist (km) | n | ⟨z²⟩ | cov68 | cov95 |")
        L.append("|---|--:|--:|--:|--:|")
        for lab, n, z2, c68, c95 in by_cld(resid, cand[name], cld):
            L.append(f"| {lab} | {n:,} | {z2:.3f} | {c68:.3f} | {c95:.3f} |")
        L.append("")
    L.append(f"![calibration]({fig_name})\n")
    Path(path).write_text('\n'.join(L))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--held-out-glob', default=None,
                    help="glob for fold held_out_predictions.parquet (pooled).")
    ap.add_argument('--input', nargs='+', default=None,
                    help="alternative: explicit parquet(s) with y/mu/σ columns.")
    ap.add_argument('--sfc-type', type=int, default=None,
                    help="restrict to surface (0=ocean,1=land) if sfc_type present.")
    ap.add_argument('--tag', default='de')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--max-abs-ppm', type=float, default=25.0,
                    help="drop |y_true| or |mu| beyond this (deployment anomaly "
                         "guard; removes fill/blow-up leakage). Default 25.")
    args = ap.parse_args()

    paths = []
    if args.held_out_glob:
        paths += sorted(_glob.glob(args.held_out_glob))
    if args.input:
        paths += list(args.input)
    if not paths:
        raise SystemExit("pass --held-out-glob and/or --input")
    print(f"  pooling {len(paths)} parquet(s)")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if args.sfc_type is not None and 'sfc_type' in df.columns:
        df = df[df['sfc_type'] == args.sfc_type].reset_index(drop=True)
    # drop guarded rows if the flags are present (correction skipped there)
    for c in ('clim_guard', 'anomaly_guard'):
        if c in df.columns:
            df = df[~df[c].astype(bool)].reset_index(drop=True)
    # Physical guard matching deployment: the anomaly_guard skips |predicted
    # anomaly| > max_abs_ppm (model blow-ups) and fill-value targets.  The raw
    # fold held_out set has no such flag, so apply it here or a handful of
    # blow-ups leak into RMSE / ⟨z²⟩.
    n_pre = len(df)
    ycol = 'y_true' if 'y_true' in df.columns else 'xco2_bc_anomaly'
    mucol = 'mu' if 'mu' in df.columns else 'pred_anomaly'
    keep = (df[ycol].abs() <= args.max_abs_ppm) & (df[mucol].abs() <= args.max_abs_ppm)
    df = df[keep].reset_index(drop=True)
    if n_pre - len(df):
        print(f"  dropped {n_pre-len(df):,} rows with |{ycol}| or |{mucol}| "
              f"> {args.max_abs_ppm} ppm (blow-ups / fill)")
    print(f"  {len(df):,} rows")

    resid, cand = resid_and_candidates(df)
    cld = df['cld_dist_km'].to_numpy(float) if 'cld_dist_km' in df.columns \
        else np.full(len(df), np.nan)
    print("  candidates:", list(cand.keys()))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig_name = 'uncertainty_calibration.png'
    make_figure(resid, cand, cld, outdir / fig_name, args.tag)
    best = write_md(resid, cand, cld, outdir / 'uncertainty_calibration.md',
                    args.tag, fig_name)
    print(f"  SIDE_A_BUDGET verdict [{args.tag}]: {best[0]}  (⟨z²⟩={best[1]['z2']:.3f})")
    print(f"  [saved] {outdir/'uncertainty_calibration.md'}")


if __name__ == '__main__':
    main()

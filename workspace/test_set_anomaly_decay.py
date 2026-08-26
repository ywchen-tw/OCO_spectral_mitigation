#!/usr/bin/env python3
"""Held-out test-set anomaly decay — before vs after the DE correction.

Counterpart of manuscript Fig. 3b computed on TEST dates: per-date parquets in
results/csv_collection whose date is 2014–2021 and absent from the production
fold manifests (training_dates.json of de_*_beta_nll_prof_reg_foldpca_*_f0..4).
These are the held-out TCCON/ATom/ship/Nassar evaluation dates plus extras —
disjointness is re-asserted at runtime via models.leakage_guard.

Two panels, fig03 style (IQR band + dashed median + solid mean, 1-km bins,
production targets ocean r05 / land r15, thresholds marked):
  (a) xco2_bc_anomaly      — before correction.  RECOMPUTED with the same
      anomaly routine as panel (b) so the two panels differ only by the
      correction: the stored parquet columns were built with a shared
      reference pool (extra_vars validity constraint) and differ in validity
      on ~5 % of rows (values agree to ≤1e-4 ppm; both stored and recomputed
      columns are kept in the cache, and the per-date VERIFY line reports the
      comparison).
  (b) xco2_de_anomaly      — after correction: the production cross-fold pooled
      deep ensemble (25 members/surface, per-fold scalers+PCA, production
      guards clim 50 ppm / |mu| 25 ppm) gives xco2_de = xco2_bc − mu, and the
      within-orbit clear-sky anomaly (src/constants.py params, r05/r15 radii)
      is RECOMPUTED on xco2_de — the reference mean is itself built from
      corrected values, so far-field mu ≈ 0 acts as a built-in negative
      control.

Both panels carry the |anomaly| ≤ MAX_ABS_ANOMALY_PPM training screen
(models.pipeline.filter_target_outliers convention, review item 10.6).

Stages (per-date cache under OUT_DIR/cache makes compute restartable):
  compute — per date: pooled DE inference → xco2_de → recompute the anomaly of
            BOTH xco2_bc and xco2_de at both radii → cache parquet.
  figure  — render the two-panel figure + per-bin stats CSV from the caches.

Usage:
    PYTHONPATH=src python workspace/test_set_anomaly_decay.py --stage all
    PYTHONPATH=src python workspace/test_set_anomaly_decay.py --stage compute --limit 2
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "workspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from models.pipeline import (_ensure_derived_features,                 # noqa: E402
                             compute_xco2_anomaly_date_id,
                             MAX_ABS_ANOMALY_PPM)
from models.leakage_guard import check_training_overlap, load_training_dates  # noqa: E402
from models.deep_ensemble import _member_predict                       # noqa: E402
from apply.apply_deep_ensemble import (_check_required_columns,        # noqa: E402
                                       _domain_report)
from build_deepens_plot_data import _load_fold                         # noqa: E402

MODEL_ROOT = REPO / "results" / "model_deep_ensemble"
OCEAN_FOLDS = [MODEL_ROOT / f"de_ocean_beta_nll_prof_reg_foldpca_r05_f{i}"
               for i in range(5)]
LAND_FOLDS = [MODEL_ROOT / f"de_land_beta_nll_prof_reg_foldpca_r15_f{i}"
              for i in range(5)]

CSV_DIR = REPO / "results" / "csv_collection"
OUT_DIR = REPO / "results" / "figures" / "cld_dist_analysis" / "test_set_anomaly_decay"
CACHE_DIR = OUT_DIR / "cache"

TEST_DATE_MIN, TEST_DATE_MAX = "2014-01-01", "2021-12-31"

# Production correction guards — mirror build_deepens_plot_data._build_surface.
CLIM_MAX_PPM = 50.0     # input guard: xco2_bc − xco2_apriori > 50 ppm
MAX_ABS_MU_PPM = 25.0   # output guard: |predicted anomaly| > 25 ppm

# Production target radii (panel layout matches manuscript Fig. 3b).
RADIUS_BY_SFC = {0: 5.0, 1: 15.0}    # ocean r05, land r15

# Non-feature columns the compute stage itself needs.
AUX_COLS = ("fp_id", "date", "orbit_id", "lat", "cld_dist_km", "sfc_type",
            "xco2_bc", "xco2_apriori", "xco2_bc_anomaly_r05",
            "xco2_bc_anomaly_r15")


# ── stage: compute ──────────────────────────────────────────────────────────

def discover_test_dates() -> dict:
    """{date: parquet path} for 2014–2021 dates outside the training manifests."""
    manifest_dates, missing = load_training_dates([*OCEAN_FOLDS, *LAND_FOLDS])
    if missing:
        raise SystemExit(f"missing training_dates.json in: {missing}")
    print(f"  [manifests] {len(manifest_dates)} training/calibration dates "
          f"across {len(OCEAN_FOLDS) + len(LAND_FOLDS)} fold dirs")
    out = {}
    for p in sorted(CSV_DIR.glob("combined_*_all_orbits.parquet")):
        m = re.search(r"combined_(\d{4}-\d{2}-\d{2})_all", p.name)
        if not m:
            continue
        d = m.group(1)
        if d in manifest_dates or not (TEST_DATE_MIN <= d <= TEST_DATE_MAX):
            continue
        out[d] = p
    return out


def predict_pooled(df, folds, sfc_type, *, tag, ood_thresh=8.0,
                   max_ood_frac=0.02):
    """Cross-fold pooled DE prediction with PRELOADED folds.

    Mirrors build_deepens_plot_data._predict_pooled (per-fold scaler/PCA
    transform, mean over all pooled members) but reuses the loaded folds
    across dates instead of re-reading checkpoints on every call.
    Returns (mu, sigma, kept_df) or (None, None, empty_df).
    """
    pipe0, _, meta0 = folds[0]
    loss = meta0.get("loss", "gaussian_nll")
    nu = meta0.get("nu", 4.0)

    _check_required_columns(df, pipe0)
    df = df[df["sfc_type"] == sfc_type].copy()
    if len(df) == 0:                     # e.g. ocean-only early-mission dates
        return None, None, df
    df = _ensure_derived_features(df)
    X0 = pipe0.transform(df)
    valid = np.all(np.isfinite(X0), axis=1)
    df = df.loc[valid].reset_index(drop=True)
    if len(df) == 0:
        return None, None, df

    overall, worst = _domain_report(X0[valid], pipe0, thresh=ood_thresh)
    if overall > max_ood_frac:
        flagged = [f"{n}({f:.0%})" for n, f in worst[:5] if f > max_ood_frac]
        print(f"  ⚠ DOMAIN WARNING [{tag}]: {overall:.1%} of feature values "
              f"OOD (|z|>{ood_thresh:g}); worst: {flagged}")

    dev = torch.device("cpu")
    mu_stack, var_stack = [], []
    for pipe, members, _ in folds:
        Xi = pipe.transform(df)          # this fold's own scaler
        for m in members:
            mu_i, var_i = _member_predict(m, Xi, dev, loss=loss, nu=nu)
            mu_stack.append(mu_i)
            var_stack.append(var_i)
    mu_stack = np.stack(mu_stack)
    var_stack = np.stack(var_stack)
    mu = mu_stack.mean(0)
    var = mu_stack.var(0) + var_stack.mean(0)      # mixture total variance
    sigma = np.sqrt(np.maximum(var, 1e-12))
    return mu.astype(np.float64), sigma.astype(np.float32), df


def process_date(date, path, folds_by_sfc, *, n_jobs=-1):
    """One test date: pooled inference → xco2_de → recomputed anomalies → cache."""
    df = pd.read_parquet(path)
    n = len(df)
    sid = pd.Index(df["fp_id"])
    if not sid.is_unique:
        raise SystemExit(f"{date}: fp_id not unique — cannot map predictions")

    mu_full = np.zeros(n)
    sigma_full = np.full(n, np.nan, dtype=np.float32)
    has_pred = np.zeros(n, dtype=bool)
    clim_guard = np.zeros(n, dtype=bool)
    anomaly_guard = np.zeros(n, dtype=bool)

    for sfc, folds in folds_by_sfc.items():
        mu, sigma, kept = predict_pooled(df, folds, sfc, tag=f"{date} sfc{sfc}")
        if mu is None:
            print(f"  {date} sfc={sfc}: no valid rows")
            continue
        # production guards (mirror _build_surface): guarded rows get mu = 0
        base = kept["xco2_bc"].to_numpy(float)
        diff = base - kept["xco2_apriori"].to_numpy(float)
        cg = np.isfinite(diff) & (diff > CLIM_MAX_PPM)
        ag = np.isfinite(mu) & (np.abs(mu) > MAX_ABS_MU_PPM)
        g = cg | ag
        if g.any():
            mu = mu.copy()
            mu[g] = 0.0
        pos = sid.get_indexer(kept["fp_id"].to_numpy())
        if (pos < 0).any():
            raise SystemExit(f"{date} sfc={sfc}: fp_id mapping failed")
        mu_full[pos] = mu
        sigma_full[pos] = sigma
        has_pred[pos] = True
        clim_guard[pos] = cg
        anomaly_guard[pos] = ag
        print(f"  {date} sfc={sfc}: {len(kept):,} predicted, "
              f"guards zeroed {int(g.sum())} (clim {int(cg.sum())}, "
              f"|mu| {int(ag.sum())})")

    xb = df["xco2_bc"].to_numpy(float)
    xco2_de = xb - mu_full                       # mu = 0 where no prediction

    lat = df["lat"].to_numpy(float)
    cld = df["cld_dist_km"].to_numpy(float)
    dcol = df["date"].to_numpy()
    ocol = df["orbit_id"].to_numpy()

    # recompute the anomaly of both XCO2 versions with the SAME routine so the
    # two figure panels differ only by the correction
    anoms = {}
    for kind, x in (("bc", xb), ("de", xco2_de)):
        for r, radius in (("r05", 5.0), ("r15", 15.0)):
            anoms[f"xco2_{kind}_anomaly_{r}"] = compute_xco2_anomaly_date_id(
                dcol, ocol, lat, cld, x, min_cld_dist=radius, n_jobs=n_jobs)

    # convention check: recomputed bc anomaly vs the stored parquet columns
    for r in ("r05", "r15"):
        rec = anoms[f"xco2_bc_anomaly_{r}"]
        sto = df[f"xco2_bc_anomaly_{r}"].to_numpy(float)
        both = np.isfinite(rec) & np.isfinite(sto)
        dmax = float(np.abs(rec[both] - sto[both]).max()) if both.any() else np.nan
        mism = int((np.isfinite(rec) != np.isfinite(sto)).sum())
        print(f"  {date} VERIFY {r}: n_both={int(both.sum()):,}  "
              f"max|recomputed−stored|={dmax:.2e} ppm  "
              f"validity-mask mismatches={mism}")

    out = pd.DataFrame({
        "date": np.full(n, date),
        "fp_id": df["fp_id"].to_numpy(),
        "orbit_id": ocol,
        "lat": lat.astype(np.float32),
        "cld_dist_km": cld.astype(np.float32),
        "sfc_type": df["sfc_type"].to_numpy(),
        "xco2_qf": df["xco2_qf"].to_numpy(),
        "xco2_bc": xb.astype(np.float32),
        "pred_mu": mu_full.astype(np.float32),
        "de_sigma": sigma_full,
        "has_pred": has_pred,
        "clim_guard": clim_guard,
        "anomaly_guard": anomaly_guard,
        "xco2_de": xco2_de.astype(np.float32),
        "xco2_bc_anomaly_r05": anoms["xco2_bc_anomaly_r05"].astype(np.float32),
        "xco2_bc_anomaly_r15": anoms["xco2_bc_anomaly_r15"].astype(np.float32),
        "xco2_de_anomaly_r05": anoms["xco2_de_anomaly_r05"].astype(np.float32),
        "xco2_de_anomaly_r15": anoms["xco2_de_anomaly_r15"].astype(np.float32),
        "xco2_bc_anomaly_r05_stored": df["xco2_bc_anomaly_r05"].to_numpy(np.float32),
        "xco2_bc_anomaly_r15_stored": df["xco2_bc_anomaly_r15"].to_numpy(np.float32),
    })
    if "snow_flag" in df.columns:
        out["snow_flag"] = df["snow_flag"].to_numpy()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_DIR / f"{date}.parquet.tmp"
    out.to_parquet(tmp, index=False)
    tmp.rename(CACHE_DIR / f"{date}.parquet")
    return out


def run_compute(args) -> None:
    dates = discover_test_dates()
    if args.dates:
        keep = set(args.dates)
        dates = {d: p for d, p in dates.items() if d in keep}
        missing = keep - set(dates)
        if missing:
            raise SystemExit(f"requested date(s) not in the test pool: {sorted(missing)}")
    print(f"  [test pool] {len(dates)} held-out dates "
          f"({min(dates)} … {max(dates)})")

    # belt-and-braces: the guard re-derives eval dates from the file names and
    # refuses on any overlap with the fold manifests
    check_training_overlap([*OCEAN_FOLDS, *LAND_FOLDS],
                           input_paths=list(dates.values()),
                           tag="test_set_anomaly_decay")

    folds_by_sfc = {0: [_load_fold(d) for d in OCEAN_FOLDS],
                    1: [_load_fold(d) for d in LAND_FOLDS]}
    for sfc, folds in folds_by_sfc.items():
        n_mem = sum(len(m) for _, m, _ in folds)
        print(f"  [model] sfc={sfc}: {len(folds)} folds × members = {n_mem}, "
              f"{folds[0][0].n_features} features")

    todo = sorted(dates)
    if args.limit:
        todo = todo[:args.limit]
    done = skipped = failed = 0
    pipe0 = folds_by_sfc[0][0][0]
    for i, d in enumerate(todo, 1):
        cache = CACHE_DIR / f"{d}.parquet"
        if cache.exists() and not args.force:
            done += 1
            continue
        # cheap schema pre-check so old-format parquets skip loudly, not crash
        import pyarrow.parquet as pq
        cols = set(pq.ParquetFile(dates[d]).schema_arrow.names)
        need = [c for c in AUX_COLS if c not in cols]
        need += [c for c in pipe0.qt_features
                 if c not in cols and c not in set(getattr(pipe0, "fp_cols", []))]
        if need:
            print(f"  ✗ SKIP {d}: parquet lacks {need[:6]}"
                  f"{' …' if len(need) > 6 else ''} (old-format build)")
            skipped += 1
            continue
        print(f"[{i}/{len(todo)}] {d}", flush=True)
        try:
            process_date(d, dates[d], folds_by_sfc, n_jobs=args.n_jobs)
            done += 1
        except SystemExit:
            raise
        except Exception as e:  # keep the sweep alive; report at the end
            print(f"  ✗ FAILED {d}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n  [compute] {done} cached, {skipped} skipped, {failed} failed "
          f"(cache: {CACHE_DIR})")
    if failed:
        raise SystemExit(f"{failed} date(s) failed — rerun --stage compute after fixing")


# ── stage: figure ───────────────────────────────────────────────────────────

# fig03 conventions (make_anomaly_decay_figure.py)
C_OCEAN, C_LAND = "#0072B2", "#D55E00"
BIN_EDGES = np.arange(0.0, 30.5, 1.0)
CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
DXCO2_DE_LABEL = r"$\Delta X_{\mathrm{CO2}}^{\mathrm{DE}}$"


def binned_stats(d, y):
    """Per-bin mean/median/q25/q75 over BIN_EDGES (bins with n<50 → NaN)."""
    idx = np.digitize(d, BIN_EDGES) - 1
    nb = len(BIN_EDGES) - 1
    out = {k: np.full(nb, np.nan) for k in ("mean", "med", "q25", "q75")}
    out["n"] = np.zeros(nb)
    for b in range(nb):
        v = y[idx == b]
        out["n"][b] = len(v)
        if len(v) < 50:
            continue
        out["mean"][b] = v.mean()
        out["q25"][b], out["med"][b], out["q75"][b] = np.percentile(v, [25, 50, 75])
    return out


def _fmt_n(n: int) -> str:
    return f"{n / 1e6:.1f} M" if n >= 1e6 else f"{n / 1e3:.0f} k"


def draw_series(ax, d, y, color, label):
    m = np.isfinite(d) & (d >= 0) & np.isfinite(y)
    st = binned_stats(d[m], y[m])
    ax.fill_between(CENTERS, st["q25"], st["q75"], color=color, alpha=0.22,
                    linewidth=0)
    ax.plot(CENTERS, st["med"], color=color, lw=1.2, ls="--")
    ax.plot(CENTERS, st["mean"], color=color, lw=1.7,
            label=f"{label}, n = {_fmt_n(m.sum())}")
    return st


def run_figure(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import DXCO2_BC_LABEL, apply_manuscript_style

    caches = sorted(CACHE_DIR.glob("*.parquet"))
    if not caches:
        raise SystemExit(f"no cache files in {CACHE_DIR} — run --stage compute")
    cols = ["date", "cld_dist_km", "sfc_type",
            "xco2_bc_anomaly_r05", "xco2_bc_anomaly_r15",
            "xco2_de_anomaly_r05", "xco2_de_anomaly_r15"]
    t = pd.concat([pd.read_parquet(p, columns=cols) for p in caches],
                  ignore_index=True)
    n_dates = t["date"].nunique()
    print(f"  [figure] {len(t):,} soundings from {n_dates} test dates")

    # training-population screen, both bc and de columns (fig03 convention)
    for col in cols[3:]:
        bad = t[col].abs() > MAX_ABS_ANOMALY_PPM
        if bad.any():
            print(f"  screened {int(bad.sum())} rows with |{col}| > "
                  f"{MAX_ABS_ANOMALY_PPM:g} ppm")
            t.loc[bad, col] = np.nan

    d = t["cld_dist_km"].to_numpy(float)
    s = t["sfc_type"].to_numpy()
    ocean, land = (s == 0), (s == 1)

    apply_manuscript_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)

    stats_rows = []
    panels = [
        (axa, "bc", f"(a) before correction ({DXCO2_BC_LABEL})"),
        (axb, "de", f"(b) after DE correction ({DXCO2_DE_LABEL})"),
    ]
    for ax, kind, title in panels:
        for mask, surf, col, color, label in [
                (ocean, "ocean", f"xco2_{kind}_anomaly_r05", C_OCEAN, "Ocean (r05)"),
                (land, "land", f"xco2_{kind}_anomaly_r15", C_LAND, "Land (r15)")]:
            st = draw_series(ax, d[mask], t[col].to_numpy(float)[mask], color,
                             label)
            for b in range(len(CENTERS)):
                stats_rows.append(dict(panel=kind, surface=surf,
                                       bin_center_km=CENTERS[b],
                                       n=int(st["n"][b]), mean=st["mean"][b],
                                       median=st["med"][b], q25=st["q25"][b],
                                       q75=st["q75"][b]))
        ax.axvline(5.0, color=C_OCEAN, lw=0.8, ls=":", alpha=0.7)
        ax.axvline(15.0, color=C_LAND, lw=0.8, ls=":", alpha=0.7)
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_xlim(0, 30)
        ax.set_xlabel("Nearest-cloud distance (km)")
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    axa.set_ylabel(f"{DXCO2_BC_LABEL} (ppm)")
    axb.set_ylabel(f"{DXCO2_DE_LABEL} (ppm)")
    axa.text(0.02, 0.03, f"{n_dates} held-out dates, 2014–2021",
             transform=axa.transAxes, fontsize=7, color="0.35")
    axb.text(0.985, 0.30,
             "solid: bin mean   dashed: median\nshading: interquartile range\n"
             "dotted verticals: reference thresholds",
             transform=axb.transAxes, fontsize=7, ha="right", va="top",
             color="0.35", linespacing=1.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"test_anomaly_decay.{ext}"
        fig.savefig(out)
        print(f"  wrote {out}")
    plt.close(fig)

    stats = pd.DataFrame(stats_rows)
    csv = OUT_DIR / "test_anomaly_decay_binstats.csv"
    stats.to_csv(csv, index=False)
    print(f"  wrote {csv}")

    # headline: near-cloud means inside each surface's response zone
    for surf, mask, rad in (("ocean", ocean, 5.0), ("land", land, 15.0)):
        r = "r05" if surf == "ocean" else "r15"
        near = mask & np.isfinite(d) & (d >= 0) & (d <= rad)
        bc = t[f"xco2_bc_{'anomaly'}_{r}"][near]
        de = t[f"xco2_de_{'anomaly'}_{r}"][near]
        print(f"  [headline] {surf} ≤{rad:g} km: mean {bc.mean():+.3f} → "
              f"{de.mean():+.3f} ppm, median {bc.median():+.3f} → "
              f"{de.median():+.3f} ppm  (n={int(near.sum()):,})")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("compute", "figure", "all"), default="all")
    ap.add_argument("--dates", nargs="+", default=None,
                    help="restrict to these test dates (YYYY-MM-DD)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N test dates (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="recompute dates whose cache already exists")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="joblib workers for the anomaly recomputation")
    args = ap.parse_args()

    if args.stage in ("compute", "all"):
        run_compute(args)
    if args.stage in ("figure", "all"):
        run_figure(args)


if __name__ == "__main__":
    main()

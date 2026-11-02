"""PPDF direct-XCO2 estimators, round 2 (2026-11-01).

Builds on estimators.py (the round-1 port; unchanged) and adds:

1a. surface-binned path models  E2*_am_ab / E2*_am_sc
    per date x surface, the calibration set is split into calib-quantile tertiles of
      ab : the SCO2 continuum reflectance exp(c_am)          (albedo bin)
      sc : the spectral class exp(c_sco2_am) / exp(c_o2a_am)  (soil vs vegetation proxy)
    and one ridge path model is fitted per bin.
1b. pooled path models  E2P* (fitted in run_v2.py's pooled stage, not here): one model
    per surface over many dates, scored on held-out date blocks (fm1620k5 folds).
2.  combined SCO2+WCO2 estimate  E2c*: f_c = w_S f_S + w_W f_W with minimum-variance weights
    from the covariance of the two (out-of-fold) f's over the calibration far-cloud rows,
    where f = 1 is assumed -- no xco2_bc involved.
    (weights clipped to [0, 1]).
    prior blends  BL_<id> / BL2*: est = N0_prior + w [(est_id - prior) - its calib mean],
    i.e. the prior-only null plus a slope term; w is a trimmed-OLS slope of (xco2_bc - prior)
    on (est_id - prior) over the far-cloud QF0 rows of the OTHER orbit fold (their est_id
    values are themselves out-of-fold).  BLP_* (run_v2.py pooled stage) take w pooled over
    the training dates of the fm1620k5 date fold instead.  BL2* regress on the SCO2 and WCO2
    estimates jointly.  Blends are the only place where xco2_bc enters an estimator; it is
    a supervised calibration scored on held-out orbits / dates.

No clear-sky reference statistics (ref_* / r15_* / r05_*) enter any estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from estimators import (F_MAX_DEV, SPEC_BY_ID, assign_folds, e2_path_model,  # noqa: F401
                        estimate as estimate_v1, far_clear_mask, path_features,
                        prepare as prepare_v1)

V1_KEEP = ["E2_o2", "E2w_o2", "E2_t2", "E2w_t2", "E0_am", "E2_am", "E2w_am",
           "E2_am_scr", "E2w_am_scr", "B_IDP", "N0_prior"]
N_BINS = 3


def _bin_spec(eid, band, by):
    return dict(id=eid, kind="e2bin", method="am", band=band, weight=None,
                screen=False, water=False, by=by)


PERDATE_SPECS = [SPEC_BY_ID[e] for e in V1_KEEP] + [
    _bin_spec("E2_am_ab", "sco2", "exp_int_sco2_am"),
    _bin_spec("E2w_am_ab", "wco2", "exp_int_sco2_am"),
    _bin_spec("E2_am_sc", "sco2", "sc_ratio"),
    _bin_spec("E2w_am_sc", "wco2", "sc_ratio"),
]

# (combined id, SCO2 id, WCO2 id)
PERDATE_COMBOS = [("E2c_am", "E2_am", "E2w_am"), ("E2c_am_ab", "E2_am_ab", "E2w_am_ab")]
# (blend id, [input ids]) -- one input: BL_, two inputs: BL2
PERDATE_BLENDS = [
    ("BL_E0_am", ["E0_am"]), ("BL_E2_am", ["E2_am"]), ("BL_E2w_am", ["E2w_am"]),
    ("BL_E2c_am", ["E2c_am"]), ("BL_E2c_am_ab", ["E2c_am_ab"]),
    ("BL2_am", ["E2_am", "E2w_am"]),
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_v1(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["sc_ratio"] = out["exp_int_sco2_am"] / out["exp_int_o2a_am"]
    return out


def e2_binned(calib, apply, band, by):
    f = pd.Series(np.nan, index=apply.index)
    cb = calib[by].dropna()
    if len(cb) < 50 * N_BINS:
        return f
    edges = np.quantile(cb, np.linspace(0, 1, N_BINS + 1)[1:-1])
    bc = np.digitize(calib[by].to_numpy(float), edges)
    ba = np.digitize(apply[by].to_numpy(float), edges)
    ba[~np.isfinite(apply[by].to_numpy(float))] = -1
    for b in range(N_BINS):
        c, a = calib[bc == b], apply[ba == b]
        if len(c) >= 50 and len(a):
            f.loc[a.index] = e2_path_model(c, a, "am", band)
    return f


def estimate(spec, calib, apply):
    if spec["kind"] == "e2bin":
        return e2_binned(calib, apply, spec["band"], spec["by"])
    return estimate_v1(spec, calib, apply)


def run_perdate(df: pd.DataFrame, far_km: float = 20.0, n_folds: int = 2) -> pd.DataFrame:
    """est_<id> (ppm) out-of-fold per surface for PERDATE_SPECS, then combos and blends."""
    est = {s["id"]: np.full(len(df), np.nan) for s in PERDATE_SPECS}
    fold_col = np.full(len(df), -1)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for sfc, g in df.groupby("sfc_type"):
        folds = assign_folds(g, n_folds)
        fold_col[pos[g.index]] = folds
        far = far_clear_mask(g, far_km).to_numpy()
        for k in range(n_folds):
            calib, apply = g[far & (folds != k)], g[folds == k]
            if len(calib) < 50 or len(apply) == 0:
                continue
            ii = pos[apply.index].to_numpy()
            for spec in PERDATE_SPECS:
                f = estimate(spec, calib, apply)
                f = f.where((f - 1.0).abs() <= F_MAX_DEV)
                est[spec["id"]][ii] = (apply["xco2_apriori"] * f).to_numpy()
    out = pd.concat([df, pd.DataFrame({f"est_{e}": v for e, v in est.items()}, index=df.index)], axis=1)
    out["fold"] = fold_col
    weights = []
    for cid, s, w in PERDATE_COMBOS:
        weights += add_combo(out, cid, s, w, far_km)
    for bid, ins in PERDATE_BLENDS:
        weights += add_blend(out, bid, ins, far_km)
    out.attrs["weights"] = weights
    return out


# ── combos and blends (work on any frame with est_* and a 'fold' column) ────────

def _trim(*arrs, nsig=4.0):
    """Mask of rows finite in all arrays; drops |x - median| > nsig*robust SD for each."""
    m = np.ones(len(arrs[0]), bool)
    for a in arrs:
        m &= np.isfinite(a)
    for a in arrs:
        v = a[m]
        if v.size < 10:
            return m
        med = np.median(v)
        rs = 1.4826 * np.median(np.abs(v - med)) or np.std(v)
        m &= np.abs(np.where(np.isfinite(a), a, med) - med) <= nsig * rs
    return m


def add_combo(df, cid, sid, wid, far_km=20.0):
    """Minimum-variance SCO2+WCO2 combination of f (no xco2_bc used)."""
    res = np.full(len(df), np.nan)
    info = []
    far_all = far_clear_mask(df, far_km).to_numpy()
    pri = df["xco2_apriori"].to_numpy(float)
    fs = df[f"est_{sid}"].to_numpy(float) / pri
    fw = df[f"est_{wid}"].to_numpy(float) / pri
    sfc = df["sfc_type"].to_numpy()
    fold = df["fold"].to_numpy()
    for s in np.unique(sfc[np.isfinite(sfc)]):
        for k in np.unique(fold[(sfc == s) & (fold >= 0)]):
            cal = (sfc == s) & (fold != k) & (fold >= 0) & far_all
            app = (sfc == s) & (fold == k)
            m = _trim(fs[cal], fw[cal])
            if m.sum() < 50:
                continue
            cov = np.cov(np.vstack([fs[cal][m], fw[cal][m]]))
            try:
                w = np.linalg.solve(cov, np.ones(2))
            except np.linalg.LinAlgError:
                continue
            w = np.clip(w / w.sum(), 0.0, 1.0)   # convex: the two f's are strongly correlated
            w = w / w.sum()
            res[app] = pri[app] * (w[0] * fs[app] + w[1] * fw[app])
            info.append(dict(est=cid, sfc_type=s, fold=k, w_sco2=w[0], w_wco2=w[1]))
    df[f"est_{cid}"] = res
    return info


def blend_slopes(frames, ins, far_km=20.0):
    """Pooled within-date slope(s) of (xco2_bc - prior) on (est_i - prior) over the far-cloud
    QF0 rows of many dates: x and y are demeaned per date x surface, then trimmed OLS
    without intercept.  Returns {sfc_type: slope vector}."""
    acc = {}
    for df in frames:
        far = far_clear_mask(df, far_km).to_numpy()
        pri = df["xco2_apriori"].to_numpy(float)
        y = df["xco2_bc"].to_numpy(float) - pri
        X = np.column_stack([df[f"est_{i}"].to_numpy(float) - pri for i in ins])
        sfc = df["sfc_type"].to_numpy()
        for s in np.unique(sfc[np.isfinite(sfc)]):
            sel = far & (sfc == s)
            m = _trim(y[sel], *X[sel].T)
            if m.sum() < 50:
                continue
            xs, ys = X[sel][m], y[sel][m]
            acc.setdefault(s, []).append((xs - xs.mean(0), ys - ys.mean()))
    out = {}
    for s, parts in acc.items():
        xs = np.vstack([p[0] for p in parts])
        ys = np.concatenate([p[1] for p in parts])
        out[s] = np.linalg.lstsq(xs, ys, rcond=None)[0]
    return out


def add_blend(df, bid, ins, far_km=20.0, slopes=None):
    """Prior blend sharing the N0_prior offset:
        est = prior + mean_cal(bc - prior) + sum_i w_i [(est_i - prior) - mean_cal(est_i - prior)]
    with 'cal' = the far-cloud QF0 rows of the OTHER orbit fold (exactly N0_prior's offset),
    so the blend differs from the prior-only null only by the w-term.  w is fitted by
    trimmed OLS on the same 'cal' rows (slopes=None; per-date blend BL*) or taken from
    ``slopes`` = {sfc_type: w} (pooled over training dates; BLP*)."""
    res = np.full(len(df), np.nan)
    info = []
    far_all = far_clear_mask(df, far_km).to_numpy()
    pri = df["xco2_apriori"].to_numpy(float)
    y_all = df["xco2_bc"].to_numpy(float) - pri
    X_all = np.column_stack([df[f"est_{i}"].to_numpy(float) - pri for i in ins])
    sfc = df["sfc_type"].to_numpy()
    fold = df["fold"].to_numpy()
    for s in np.unique(sfc[np.isfinite(sfc)]):
        for k in np.unique(fold[(sfc == s) & (fold >= 0)]):
            cal = (sfc == s) & (fold != k) & (fold >= 0) & far_all
            app = (sfc == s) & (fold == k)
            if cal.sum() < 50:
                continue
            X, y = X_all[cal], y_all[cal]
            xok = np.isfinite(X).all(axis=1)
            if xok.sum() < 50:
                continue
            off_y, off_x = np.nanmean(y), X[xok].mean(axis=0)
            if slopes is None:
                m = _trim(y, *X.T)
                if m.sum() < 50:
                    continue
                xs, ys = X[m] - X[m].mean(0), y[m] - y[m].mean()
                w = np.linalg.lstsq(xs, ys, rcond=None)[0]
                r = ys - xs @ w           # second pass: drop 4-sigma residual outliers
                keep = np.abs(r - np.median(r)) <= 4 * 1.4826 * np.median(np.abs(r - np.median(r)))
                if keep.sum() >= 50:
                    w = np.linalg.lstsq(xs[keep], ys[keep], rcond=None)[0]
            elif s in slopes:
                w = np.asarray(slopes[s])
            else:
                continue
            res[app] = pri[app] + off_y + (X_all[app] - off_x) @ w
            info.append(dict(est=bid, sfc_type=s, fold=k, a=off_y,
                             **{f"w_{i}": c for i, c in zip(ins, w)}))
    df[f"est_{bid}"] = res
    return info

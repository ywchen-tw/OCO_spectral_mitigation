"""Helpers for workspace/evaluate_agreed_model.py (agreed-model evaluation, CURC handoff items 4 and 6).

Statistics follow the local study idea_test/fit_order_study/:
  robust_sd, spread_pct            run_study.py
  lat_local_contrast, bin_stats    h2o_analysis.py (same definition; references located by a sorted
                                   latitude search instead of a full scan; result identical)
  ridge_heldout                    coef_anomaly_corr.py (standardised, lambda 0.01, 10 contiguous blocks)
  window_stats                     the reference window of src/spectral/anomaly.compute_xco2_anomaly
                                   (+-lat_thres deg latitude, cld_dist > r, all footprints, >= 5 references,
                                   population std of the reference XCO2 <= std_thres), evaluated with
                                   cumulative sums over the latitude-sorted reference pool.  The pool is the
                                   target's own pool; extra variables do NOT narrow it (each extra mean is over
                                   the pool members where that variable is finite).  The caller checks the
                                   recomputed anomaly against the stored target.
"""
import numpy as np
import pandas as pd

BANDS = ("o2a", "wco2", "sco2")
BAND_UP = {"o2a": "O2A", "wco2": "WCO2", "sco2": "SCO2"}
BINS = [(0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 20.0), (20.0, np.inf)]
BIN_LABELS = ["0-1", "1-3", "3-10", "10-20", ">20"]
KERNEL = {"sco2": 1.60, "wco2": 1.01, "o2a": 1.80}   # fixed kernels; never regressed on xco2
FLAG_PCT = 1.0                                         # O2A local-anomaly flag threshold (%)
FAR_KM = 20.0
KOZIENICE = (51.665, 21.467)                           # lat, lon of the Kozienice power plant
EARTH_R_KM = 6371.0


# --------------------------------------------------------------------------- basic statistics

def robust_sd(x, min_n=10):
    """1.4826 * MAD of the finite values (NaN if fewer than min_n)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < min_n:
        return np.nan
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def nanmedian(x, min_n=1):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) >= min_n else np.nan


def nanpct(x, q, min_n=10):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) >= min_n else np.nan


def fp_relative(x, fp, group=None):
    """x / mean(x over the same footprint [and group]) - 1, for the rows passed (run_study.spread_pct)."""
    s = pd.Series(np.asarray(x, float))
    keys = [pd.Series(fp)] if group is None else [pd.Series(group), pd.Series(fp)]
    return (s / s.groupby(keys).transform("mean") - 1.0).values


def spread_pct(x, fp):
    """Robust SD (%) of x after removing the per-footprint-index mean (run_study.spread_pct)."""
    return 100.0 * robust_sd(fp_relative(x, fp))


def fp_demean_median(x, fp):
    """x minus its per-footprint median."""
    s = pd.Series(np.asarray(x, float))
    return (s - s.groupby(pd.Series(fp)).transform("median")).values


def pearson(x, y, min_n=30):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def spearman(x, y, min_n=30):
    from scipy.stats import spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return float(spearmanr(x[m], y[m])[0])


# --------------------------------------------------------------------------- latitude-local contrast

def lat_local_contrast(v, cld, qf, lat, fp, relative, ref_km=10.0, dlat=1.0, nmin=5):
    """value - median of (cld > ref_km, QF0, |dlat| <= 1 deg, same footprint, >= 5) references.

    Identical to h2o_analysis.lat_local_contrast (self included in its own reference when it qualifies);
    relative=True returns v/ref - 1.
    """
    v = np.asarray(v, float)
    lat = np.asarray(lat, float)
    n = len(v)
    ref = np.full(n, np.nan)
    ref_ok = (cld > ref_km) & (qf == 0) & np.isfinite(v) & np.isfinite(lat)
    for f in np.unique(fp):
        m_f = fp == f
        rs = ref_ok & m_f
        if rs.sum() < nmin:
            continue
        rl, rv = lat[rs], v[rs]
        o = np.argsort(rl, kind="stable")
        rl, rv = rl[o], rv[o]
        js = np.flatnonzero(m_f & np.isfinite(lat))
        lo = np.searchsorted(rl, lat[js] - dlat - 1e-6, "left")
        hi = np.searchsorted(rl, lat[js] + dlat + 1e-6, "right")
        for j, a, b in zip(js, lo, hi):
            if b - a < nmin:
                continue
            sel = np.abs(rl[a:b] - lat[j]) <= dlat
            if sel.sum() >= nmin:
                ref[j] = np.median(rv[a:b][sel])
    con = v - ref
    with np.errstate(divide="ignore", invalid="ignore"):
        return con / ref if relative else con


def bin_stats(con, cld, far, sel=None, min_n=21):
    """h2o_analysis.bin_stats: far = far-clear mask (> 20 km, QF0) for the SNR denominator;
    sel = soundings admitted to the cloud-distance bins (QF0 by default in the caller)."""
    con = np.asarray(con, float)
    far_sd = robust_sd(con[far])
    rec = {"far_median": nanmedian(con[far]), "far_rsd": far_sd}
    for (lo, hi), lab in zip(BINS, BIN_LABELS):
        m = (cld >= lo) & (cld < hi) & np.isfinite(con)
        if sel is not None:
            m &= sel
        k = int(m.sum())
        med = float(np.median(con[m])) if k >= min_n else np.nan
        rec[f"n_{lab}"] = k
        rec[f"con_{lab}"] = med
        rec[f"snr_{lab}"] = med / far_sd if np.isfinite(far_sd) and far_sd > 0 else np.nan
    return rec


# --------------------------------------------------------------------------- pipeline reference window

def window_stats(lat, cld, xco2, min_cld_dist, pool_vars=(), extra=None,
                 lat_thres=0.25, std_thres=1.0, nmin=5):
    """Reference window of compute_xco2_anomaly, vectorised.

    Pool = finite lat & cld > min_cld_dist & finite xco2 & finite(every array in pool_vars).
    For each sounding: refs = pool members with lat in [lat - thres, lat + thres]; valid when n >= nmin and
    population std(xco2_refs) <= std_thres.  Returns (anomaly, means{k}, n_ref, ref_std) with means[k] the
    mean of extra[k] over the refs where extra[k] is finite (NaN where not valid).
    """
    lat = np.asarray(lat, float)
    xco2 = np.asarray(xco2, float)
    N = len(lat)
    valid_lat = np.isfinite(lat)
    pool = valid_lat & (np.asarray(cld, float) > min_cld_dist) & np.isfinite(xco2)
    for p in pool_vars:
        pool &= np.isfinite(np.asarray(p, float))
    idx = np.flatnonzero(pool)
    extra = extra or {}
    if idx.size == 0:
        nanv = np.full(N, np.nan)
        return nanv.copy(), {k: nanv.copy() for k in extra}, np.zeros(N, int), nanv.copy()
    idx = idx[np.argsort(lat[idx], kind="stable")]
    pl = lat[idx]
    q = np.where(valid_lat, lat, np.inf)
    lo = np.searchsorted(pl, q - lat_thres, "left")
    hi = np.searchsorted(pl, q + lat_thres, "right")
    lo[~valid_lat] = 0
    hi[~valid_lat] = 0
    n = hi - lo
    x = xco2[idx]
    x0 = float(np.median(x))
    cs = np.r_[0.0, np.cumsum(x - x0)]
    cs2 = np.r_[0.0, np.cumsum((x - x0) ** 2)]
    with np.errstate(divide="ignore", invalid="ignore"):
        mc = (cs[hi] - cs[lo]) / n
        var = (cs2[hi] - cs2[lo]) / n - mc ** 2
    std = np.sqrt(np.clip(var, 0.0, None))
    ok = valid_lat & (n >= nmin) & (std <= std_thres)
    anomaly = np.where(ok, xco2 - (mc + x0), np.nan)
    means = {}
    for k, v in extra.items():
        vv = np.asarray(v, float)[idx]
        fin = np.isfinite(vv)
        if not fin.any():
            means[k] = np.full(N, np.nan)
            continue
        v0 = float(np.median(vv[fin]))
        cnt = np.r_[0, np.cumsum(fin)]
        csv = np.r_[0.0, np.cumsum(np.where(fin, vv - v0, 0.0))]
        c = cnt[hi] - cnt[lo]
        with np.errstate(divide="ignore", invalid="ignore"):
            m = (csv[hi] - csv[lo]) / c + v0
        means[k] = np.where(ok & (c > 0), m, np.nan)
    return anomaly, means, np.where(ok, n, 0), np.where(ok, std, np.nan)


# --------------------------------------------------------------------------- regression

def ridge_heldout(X, y, nblock=10, lam=1e-2):
    """coef_anomaly_corr.ridge_heldout: standardised predictors, intercept, ridge lambda, contiguous
    blocks in row (sounding) order, leave-one-block-out; R^2 = 1 - var(y - pred)/var(y) over all rows."""
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    blocks = np.arange(len(y)) * nblock // len(y)
    pred = np.full(len(y), np.nan)
    for b in range(nblock):
        tr, te = blocks != b, blocks == b
        A = np.column_stack([np.ones(tr.sum()), Xs[tr]])
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.column_stack([np.ones(te.sum()), Xs[te]]) @ beta
    return float(1 - np.var(y - pred) / np.var(y))


def ols_r2(y, xs):
    a = np.column_stack([np.ones(len(y)), *xs]) if xs else np.ones((len(y), 1))
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    r = y - a @ coef
    tss = (y - y.mean()) @ (y - y.mean())
    return float(1 - (r @ r) / tss), float(r @ r), coef


def partial_r2(y, x_keep, x_test):
    """(RSS_keep - RSS_full) / RSS_keep (h2o_analysis.partial_r2)."""
    _, rss_k, _ = ols_r2(y, x_keep)
    _, rss_f, _ = ols_r2(y, x_keep + x_test)
    return (rss_k - rss_f) / rss_k


# --------------------------------------------------------------------------- geometry / prior

def column_mean_prior(co2_prof, p_prof, q_prof=None):
    """Dry-air pressure-weighted column mean (ppm) of the 72-level CO2Prior profile used to build tau.
    Weights (1 - q) dp on the trapezoid of the level pressures; profile in mol/mol is converted to ppm."""
    c = np.asarray(co2_prof, float)
    p = np.asarray(p_prof, float)
    if np.nanmedian(c) < 1.0:
        c = c * 1e6
    dry = 1.0 - (np.asarray(q_prof, float) if q_prof is not None else 0.0)
    cw = c * dry
    num = np.trapezoid(cw, p, axis=1) if hasattr(np, "trapezoid") else np.trapz(cw, p, axis=1)
    den = np.trapezoid(dry * np.ones_like(c), p, axis=1) if hasattr(np, "trapezoid") else np.trapz(dry * np.ones_like(c), p, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out) | (out <= 0)] = np.nan
    return out


def local_xy_km(lat, lon, lat0, lon0):
    """Equirectangular x (east), y (north) in km relative to (lat0, lon0)."""
    x = np.deg2rad(lon - lon0) * EARTH_R_KM * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * EARTH_R_KM
    return x, y


def along_track_km(lat, lon, lat0, lon0, radius_km=100.0):
    """Signed along-track coordinate (km, positive northward) of each sounding's projection relative to
    the point (lat0, lon0); track direction = principal axis of the soundings within radius_km."""
    x, y = local_xy_km(lat, lon, lat0, lon0)
    d = np.hypot(x, y)
    m = np.isfinite(d) & (d < radius_km)
    if m.sum() < 3:
        return np.full(len(lat), np.nan), d
    xy = np.c_[x[m] - x[m].mean(), y[m] - y[m].mean()]
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    u = vt[0]
    if u[1] < 0:
        u = -u
    return x * u[0] + y * u[1], d

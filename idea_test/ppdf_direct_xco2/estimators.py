"""Direct XCO2 estimators from per-band path cumulants (fit schema doppler_am_t2_v2).

Every estimator returns the CO2 scale factor ``f`` relative to the prior, so the
estimate is ``xco2_apriori * f``.  Calibration ("f = 1 far from clouds") is done
on a calibration frame and applied to a separate frame; ``run_all`` wires the
2-fold-by-orbit hold-out so far-cloud scores are out-of-sample.

Ported 2026-11-01 from the first pass (2026-10-29, retired ``<band>_{k1,k2,intercept}_nosg``
columns of the production-order fit; see git history of this file).  The estimator
definitions, calibration logic and validity cuts are unchanged; each one is now
evaluated on four fit methods of the current schema:

    o2 : order-2 total-tau fit   k1_o2, k2_o2, c_o2           (nearest port of *_nosg)
    o3 : order-3 total-tau fit   k1_o3, k2_o3, c_o3
    t2 : two-tau fit             k1_dry_t2 (dry gas), k1w_t2 (water), k2_t2, c_t2
    am : agreed per-band model   k1_am (dry gas for WCO2/SCO2), k1w_am, k2diag_am,
                                 c_am, a1_am (continuum slope), o2a_f_am, sco2_a2_am

and new variants use the fit 1-sigma errors ``<band>_sig_<field>``:

    *_ivw : path-model ridge with inverse-variance weights 1/sig(k1_target)^2
    *_scr : sigma-screened -- rows with sig(k1)/k1 above the calib-far 75th
            percentile are dropped from calibration and set invalid in application
    E2h*  : path model plus the water k1 of WCO2 and SCO2 (H2O lever; t2, am)

No clear-sky reference statistics (ref_* / r15_* / r05_*) are used by any estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BANDS = ("o2a", "wco2", "sco2")
METHODS = ("o2", "o3", "t2", "am")

# per-method field names (k1 is the dry-gas k1 for t2 / am)
K1 = {"o2": "k1_o2", "o3": "k1_o3", "t2": "k1_dry_t2", "am": "k1_am"}
K2 = {"o2": "k2_o2", "o3": "k2_o3", "t2": "k2_t2", "am": "k2diag_am"}
C0 = {"o2": "c_o2", "o3": "c_o3", "t2": "c_t2", "am": "c_am"}
K1W = {"t2": "k1w_t2", "am": "k1w_am"}

GEOMETRY = ["1_over_cos_sza", "1_over_cos_vza", "glint_angle"]
META_COLUMNS = [
    "xco2_apriori", "xco2_bc", "xco2_raw", "xco2_bc_anomaly", "xco2_bc_anomaly_r15",
    "xco2_strong_idp", "xco2_weak_idp", "cld_dist_km", "sfc_type", "xco2_qf",
    *GEOMETRY, "orbit_id", "time", "fp", "lat", "lon",
]


def load_columns(available: set[str] | None = None) -> list[str]:
    """Columns to read; ``available`` filters out fields a band does not have."""
    cols = []
    for b in BANDS:
        for m in METHODS:
            cols += [f"{b}_{K1[m]}", f"{b}_{K2[m]}", f"{b}_{C0[m]}", f"{b}_sig_{K1[m]}"]
            if m in K1W and b != "o2a":
                cols += [f"{b}_{K1W[m]}", f"{b}_sig_{K1W[m]}"]
        cols += [f"{b}_a1_am", f"{b}_ok_am"]
    cols += ["o2a_f_am", "sco2_a2_am"]
    cols += META_COLUMNS
    if available is not None:
        cols = [c for c in cols if c in available]
    return cols


def path_features(method: str, water: bool = False) -> list[str]:
    """Gas-free observables: O2A k1 (air path), the three k2/k1^2 shape ratios
    (invariant to the band's gas scale), the three continuum reflectances exp(c),
    geometry; for ``am`` also the continuum slopes, O2A fill fraction and SCO2 x^2."""
    f = [f"o2a_{K1[method]}"] + [f"shape_{b}_{method}" for b in BANDS] \
        + [f"exp_int_{b}_{method}" for b in BANDS] + GEOMETRY
    if method == "am":
        f += ["o2a_f_am", "o2a_a1_am", "wco2_a1_am", "sco2_a1_am", "sco2_a2_am"]
    if water:
        f += [f"wco2_{K1W[method]}", f"sco2_{K1W[method]}"]
    return f


IDP_FILL_PPM = 50.0   # |idp - bc| beyond this is a fill / gross failure
SANITY_PPM = 50.0     # |xco2_bc - prior| beyond this is a gross retrieval failure
F_MAX_DEV = 0.2       # |f - 1| beyond this (±80 ppm) is reported as invalid, not as a value
PRED_PATH_RANGE = (0.5, 1.5)   # predicted <l'> outside this x calib median -> invalid
SCREEN_Q = 0.75       # *_scr keeps rows with sig(k1)/k1 <= this calib-far quantile
IVW_CLIP_Q = (0.05, 0.95)      # inverse-variance weights clipped to these calib quantiles


# ── estimator registry ──────────────────────────────────────────────────────────
# kind: e0 / e1 / e2 / idp; band: CO2 band whose k1 is the target

def _spec(eid, kind, method=None, band="sco2", weight=None, screen=False, water=False):
    return dict(id=eid, kind=kind, method=method, band=band, weight=weight,
                screen=screen, water=water)


ESTIMATOR_SPECS = []
for _m in METHODS:
    ESTIMATOR_SPECS += [
        _spec(f"E0_{_m}", "e0", _m), _spec(f"E1_{_m}", "e1", _m),
        _spec(f"E2_{_m}", "e2", _m), _spec(f"E2w_{_m}", "e2", _m, band="wco2"),
    ]
for _m in ("o2", "t2", "am"):
    ESTIMATOR_SPECS += [
        _spec(f"E0_{_m}_scr", "e0", _m, screen=True),
        _spec(f"E2_{_m}_ivw", "e2", _m, weight="ivw"),
        _spec(f"E2w_{_m}_ivw", "e2", _m, band="wco2", weight="ivw"),
        _spec(f"E2_{_m}_scr", "e2", _m, screen=True),
        _spec(f"E2w_{_m}_scr", "e2", _m, band="wco2", screen=True),
    ]
for _m in ("t2", "am"):
    ESTIMATOR_SPECS += [
        _spec(f"E2h_{_m}", "e2", _m, water=True),
        _spec(f"E2wh_{_m}", "e2", _m, band="wco2", water=True),
    ]
ESTIMATOR_SPECS.append(_spec("B_IDP", "idp"))
# Null baseline (added 2026-11-01): the prior plus the calib-far mean (bc - prior).
# Any estimator must beat this to carry XCO2 information beyond the prior.
ESTIMATOR_SPECS.append(_spec("N0_prior", "null"))
ESTIMATORS = tuple(s["id"] for s in ESTIMATOR_SPECS)
SPEC_BY_ID = {s["id"]: s for s in ESTIMATOR_SPECS}


# ── data preparation ────────────────────────────────────────────────────────────

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add shape ratios, intercept reflectances, relative sigmas; drop unusable rows.

    The row population is defined by the order-2 fit exactly as the first pass
    defined it by the *_nosg fit (finite, positive k1 in all three bands); other
    methods' failures make their estimates NaN (counted in frac_valid).
    """
    base = [f"{b}_{K1['o2']}" for b in BANDS]
    out = df.dropna(subset=base + ["xco2_apriori", "xco2_bc"]).copy()
    out = out[(out[base] > 0).all(axis=1)]
    out = out[(out["xco2_bc"] - out["xco2_apriori"]).abs() < SANITY_PPM]
    new = {}
    for m in METHODS:
        for b in BANDS:
            k1 = out[f"{b}_{K1[m]}"].to_numpy(float)
            k2 = out[f"{b}_{K2[m]}"].to_numpy(float)
            c = out[f"{b}_{C0[m]}"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                good = k1 > 0
                if m == "am":
                    good &= out[f"{b}_ok_am"].to_numpy(float) == 1
                new[f"k1pos_{b}_{m}"] = np.where(good, k1, np.nan)
                new[f"shape_{b}_{m}"] = np.where(good, k2 / k1**2, np.nan)
                new[f"exp_int_{b}_{m}"] = np.where(good, np.exp(c), np.nan)
                new[f"sigrel_{b}_{m}"] = np.where(good, np.abs(out[f"{b}_sig_{K1[m]}"].to_numpy(float)) / k1, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            new[f"ratio_s_o2a_{m}"] = new[f"k1pos_sco2_{m}"] / new[f"k1pos_o2a_{m}"]
    out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
    if "sco2_a2_am" in out:   # NaN by design over water: a constant there, not missing
        out["sco2_a2_am"] = out["sco2_a2_am"].where(out["sfc_type"] == 1, 0.0)
    idp_dev = out["xco2_strong_idp"] - out["xco2_bc"]
    out["idp_ok"] = idp_dev.abs() < IDP_FILL_PPM
    return out.sort_values(["orbit_id", "time", "fp"]).reset_index(drop=True)


def far_clear_mask(df: pd.DataFrame, far_km: float = 20.0) -> pd.Series:
    return (df["cld_dist_km"] > far_km) & (df["xco2_qf"] == 0)


def assign_folds(df: pd.DataFrame, n_folds: int = 2) -> np.ndarray:
    """Fold by orbit; a single-orbit frame falls back to along-track halves."""
    orbits = np.sort(df["orbit_id"].unique())
    if len(orbits) >= n_folds:
        fold_of = {o: i % n_folds for i, o in enumerate(orbits)}
        return df["orbit_id"].map(fold_of).to_numpy()
    rank = df["time"].rank(method="first").to_numpy()
    return ((rank - 1) * n_folds // len(df)).astype(int)


# ── estimators: (calib, apply) -> f aligned to apply.index ──────────────────────

def e0_naive(calib, apply, method="o2", band="sco2"):
    col = f"k1pos_{band}_{method}"
    return apply[col] / calib[col].mean()


def e1_o2a_normalized(calib, apply, method="o2"):
    col = f"ratio_s_o2a_{method}"
    return apply[col] / calib[col].mean()


def e2_path_model(calib, apply, method="o2", band="sco2", weight=None, water=False,
                  alpha: float = 1.0):
    """f = k1_target / <l'>_pred, with <l'> modelled from gas-free observables."""
    target = f"k1pos_{band}_{method}"
    feats = [f for f in path_features(method, water) if f in calib]
    feats = [f for f in feats if calib[f].notna().mean() > 0.5]   # all-NaN on this frame
    c = calib.dropna(subset=feats + [target])
    if len(c) < 50:
        return pd.Series(np.nan, index=apply.index)
    sw = None
    if weight == "ivw":
        sig = (c[f"sigrel_{band}_{method}"] * c[target]).to_numpy(float)
        w = 1.0 / np.where(sig > 0, sig, np.nan) ** 2
        lo, hi = np.nanquantile(w, IVW_CLIP_Q)
        w = np.clip(np.nan_to_num(w, nan=lo), lo, hi)
        sw = w / w.mean()
    scaler = StandardScaler().fit(c[feats])
    model = Ridge(alpha=alpha).fit(scaler.transform(c[feats]), c[target], sample_weight=sw)
    x = apply[feats]
    ok = x.notna().all(axis=1)
    pred = pd.Series(np.nan, index=apply.index)
    if ok.any():
        pred[ok] = model.predict(scaler.transform(x[ok]))
    med = float(c[target].median())
    pred = pred.where((pred > PRED_PATH_RANGE[0] * med) & (pred < PRED_PATH_RANGE[1] * med))
    return apply[target] / pred


def b_idp_benchmark(calib, apply):
    """Offset-calibrated strong-band IMAP-DOAS XCO2, expressed as f for uniformity."""
    c = calib[calib["idp_ok"]]
    offset = (c["xco2_strong_idp"] - c["xco2_bc"]).mean()
    est = (apply["xco2_strong_idp"] - offset).where(apply["idp_ok"])
    return est / apply["xco2_apriori"]


def n0_prior(calib, apply):
    """f such that the estimate is the prior + calib-far mean(bc - prior)."""
    offset = (calib["xco2_bc"] - calib["xco2_apriori"]).mean()
    return (apply["xco2_apriori"] + offset) / apply["xco2_apriori"]


def estimate(spec: dict, calib: pd.DataFrame, apply: pd.DataFrame) -> pd.Series:
    kind, m, band = spec["kind"], spec["method"], spec["band"]
    if spec["screen"]:
        sr = f"sigrel_{band}_{m}"
        thr = calib[sr].quantile(SCREEN_Q)
        calib = calib[calib[sr] <= thr]
        keep = apply[sr] <= thr
    else:
        keep = pd.Series(True, index=apply.index)
    if len(calib) < 50:
        return pd.Series(np.nan, index=apply.index)
    if kind == "e0":
        f = e0_naive(calib, apply, m, band)
    elif kind == "e1":
        f = e1_o2a_normalized(calib, apply, m)
    elif kind == "e2":
        f = e2_path_model(calib, apply, m, band, weight=spec["weight"], water=spec["water"])
    elif kind == "idp":
        f = b_idp_benchmark(calib, apply)
    else:
        f = n0_prior(calib, apply)
    return f.where(keep)


def sigma_ppm(spec: dict, df: pd.DataFrame) -> pd.Series:
    """Fit-noise-only 1-sigma of the estimate (ppm): sig(k1)/k1 of the target band
    (in quadrature with O2A for E1) times the prior.  Ignores path-model error, and
    the formal sig_* understate scene-correlated misfit (see cumulant_fit.py)."""
    if spec["kind"] in ("idp", "null"):
        return pd.Series(np.nan, index=df.index)
    m, band = spec["method"], spec["band"]
    s = df[f"sigrel_{band}_{m}"]
    if spec["kind"] == "e1":
        s = np.sqrt(s**2 + df[f"sigrel_o2a_{m}"] ** 2)
    return s * df["xco2_apriori"]


def run_all(df: pd.DataFrame, far_km: float = 20.0, n_folds: int = 2) -> pd.DataFrame:
    """Return df with est_<id> columns (ppm), calibrated out-of-fold per surface,
    and sig_<id> columns (fit-noise ppm)."""
    est = {e: np.full(len(df), np.nan) for e in ESTIMATORS}
    fold_col = np.full(len(df), -1)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for sfc, g in df.groupby("sfc_type"):
        folds = assign_folds(g, n_folds)
        fold_col[pos[g.index]] = folds
        far = far_clear_mask(g, far_km).to_numpy()
        for k in range(n_folds):
            calib = g[far & (folds != k)]
            apply = g[folds == k]
            if len(calib) < 50 or len(apply) == 0:
                continue
            ii = pos[apply.index].to_numpy()
            for spec in ESTIMATOR_SPECS:
                f = estimate(spec, calib, apply)
                f = f.where((f - 1.0).abs() <= F_MAX_DEV)
                est[spec["id"]][ii] = (apply["xco2_apriori"] * f).to_numpy()
    extra = {f"est_{e}": v for e, v in est.items()}
    extra.update({f"sig_{s['id']}": sigma_ppm(s, df).to_numpy() for s in ESTIMATOR_SPECS})
    extra["fold"] = fold_col
    return pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)

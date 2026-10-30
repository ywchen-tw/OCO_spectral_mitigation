"""Direct XCO2 estimators from per-band path cumulants.

Every estimator returns the CO2 scale factor ``f`` relative to the prior, so the
estimate is ``xco2_apriori * f``.  Calibration ("f = 1 far from clouds") is done
on a calibration frame and applied to a separate frame; ``run_all`` wires the
2-fold-by-orbit hold-out so far-cloud scores are out-of-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BANDS = ("o2a", "wco2", "sco2")

LOAD_COLUMNS = (
    [f"{b}_{k}_nosg" for b in BANDS for k in ("k1", "k2", "intercept")]
    + ["xco2_apriori", "xco2_bc", "xco2_raw", "xco2_bc_anomaly",
       "xco2_strong_idp", "xco2_weak_idp", "cld_dist_km", "sfc_type", "xco2_qf",
       "1_over_cos_sza", "1_over_cos_vza", "glint_angle",
       "orbit_id", "time", "fp", "lat", "lon"]
)

# Gas-free observables: O2A k1 (air path), the three k2/k1^2 shape ratios
# (invariant to the band's gas scale), the three continuum intercepts, geometry.
PATH_MODEL_FEATURES = [
    "o2a_k1_nosg", "shape_o2a", "shape_wco2", "shape_sco2",
    "exp_int_o2a", "exp_int_wco2", "exp_int_sco2",
    "1_over_cos_sza", "1_over_cos_vza", "glint_angle",
]

ESTIMATORS = ("E0", "E1", "E2", "E2w", "B_IDP")
IDP_FILL_PPM = 50.0   # |idp - bc| beyond this is a fill / gross failure
SANITY_PPM = 50.0     # |xco2_bc - prior| beyond this is a gross retrieval failure
F_MAX_DEV = 0.2       # |f - 1| beyond this (±80 ppm) is reported as invalid, not as a value
PRED_PATH_RANGE = (0.5, 1.5)   # predicted <l'> outside this x calib median -> invalid


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add shape ratios, intercept reflectances, bound flags; drop unusable rows."""
    out = df.dropna(subset=[f"{b}_k1_nosg" for b in BANDS] + ["xco2_apriori", "xco2_bc"]).copy()
    for b in BANDS:
        k1 = out[f"{b}_k1_nosg"].to_numpy(float)
        k2 = out[f"{b}_k2_nosg"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"shape_{b}"] = np.where(k1 > 0, k2 / k1**2, np.nan)
        out[f"{b}_k2_at_bound"] = k2 <= 1e-6
        out[f"exp_int_{b}"] = np.exp(out[f"{b}_intercept_nosg"].to_numpy(float))
    out = out[(out["o2a_k1_nosg"] > 0) & (out["sco2_k1_nosg"] > 0) & (out["wco2_k1_nosg"] > 0)]
    out = out[(out["xco2_bc"] - out["xco2_apriori"]).abs() < SANITY_PPM]
    out["ratio_s_o2a"] = out["sco2_k1_nosg"] / out["o2a_k1_nosg"]
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

def e0_naive(calib: pd.DataFrame, apply: pd.DataFrame) -> pd.Series:
    return apply["sco2_k1_nosg"] / calib["sco2_k1_nosg"].mean()


def e1_o2a_normalized(calib: pd.DataFrame, apply: pd.DataFrame) -> pd.Series:
    return apply["ratio_s_o2a"] / calib["ratio_s_o2a"].mean()


def e2_path_model(calib: pd.DataFrame, apply: pd.DataFrame, target: str = "sco2_k1_nosg",
                  alpha: float = 1.0) -> pd.Series:
    """f = k1_target / <l'>_pred, with <l'> modelled from gas-free observables."""
    feats = PATH_MODEL_FEATURES
    c = calib.dropna(subset=feats + [target])
    scaler = StandardScaler().fit(c[feats])
    model = Ridge(alpha=alpha).fit(scaler.transform(c[feats]), c[target])
    x = apply[feats]
    ok = x.notna().all(axis=1)
    pred = pd.Series(np.nan, index=apply.index)
    pred[ok] = model.predict(scaler.transform(x[ok]))
    med = float(c[target].median())
    pred = pred.where((pred > PRED_PATH_RANGE[0] * med) & (pred < PRED_PATH_RANGE[1] * med))
    return apply[target] / pred


def b_idp_benchmark(calib: pd.DataFrame, apply: pd.DataFrame) -> pd.Series:
    """Offset-calibrated strong-band IMAP-DOAS XCO2, expressed as f for uniformity."""
    c = calib[calib["idp_ok"]]
    offset = (c["xco2_strong_idp"] - c["xco2_bc"]).mean()
    est = (apply["xco2_strong_idp"] - offset).where(apply["idp_ok"])
    return est / apply["xco2_apriori"]


def run_all(df: pd.DataFrame, far_km: float = 20.0, n_folds: int = 2) -> pd.DataFrame:
    """Return df with est_<id> columns (ppm), calibrated out-of-fold per surface."""
    out = df.copy()
    for e in ESTIMATORS:
        out[f"est_{e}"] = np.nan
    out["fold"] = -1
    for sfc, g in out.groupby("sfc_type"):
        folds = assign_folds(g, n_folds)
        out.loc[g.index, "fold"] = folds
        far = far_clear_mask(g, far_km).to_numpy()
        for k in range(n_folds):
            calib = g[far & (folds != k)]
            apply = g[folds == k]
            if len(calib) < 50 or len(apply) == 0:
                continue
            fs = {
                "E0": e0_naive(calib, apply),
                "E1": e1_o2a_normalized(calib, apply),
                "E2": e2_path_model(calib, apply, "sco2_k1_nosg"),
                "E2w": e2_path_model(calib, apply, "wco2_k1_nosg"),
                "B_IDP": b_idp_benchmark(calib, apply),
            }
            for e, f in fs.items():
                f = f.where((f - 1.0).abs() <= F_MAX_DEV)
                out.loc[apply.index, f"est_{e}"] = (apply["xco2_apriori"] * f).to_numpy()
    return out

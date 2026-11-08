"""Shared data loading for the absolute-3D-index test (idea_test/absolute_3d_index).

Builds one footprint table for the 139 evaluation dates
(results_doppler/model_comparison/xgb/fm1620k5v2xt41_eval_dates/per_date_nok2diag)
from the source parquets results_doppler/csv_collection/combined_<date>_all_orbits.parquet.

Reused, not reimplemented:
  - population screen     analysis.utils.apply_quality_filter(require_qf0=False)
                          (xco2_bc > 0, snow-free, both QF)
  - per-surface reference analysis.spec_sensitivity._apply_production_reference(df, 'v2')
                          (ocean r05 / land r15; d<term>_<band>, z<term>_<band>,
                          xco2_bc_anomaly with the |.| <= 100 ppm screen)
  - mixed-surface windows workspace/mixed_surface_reference_count.load / analyse
                          (same reference windows; evaluated out to 50 km so the
                          far window gets the same exclusion as the near one)
  - MCD12C1 land class    analysis.land_cover.assign_land_cover

Added here (exploratory only):
  - scattering angle and X = tau_R (p_prior/1013.25) P(Theta) / (4 mu mu0)
  - absolute composite quantities and their approximate deltas
    (reference = combination of the stored reference means, as in
    make_landclass_heatmap_figure.SHAPE_ROWS).

Geometry convention (checked in t0_xcheck.py): the Lite solar / sensor
azimuths are both "as seen from the footprint"; glint soundings have
raa = |saa - vaa| ~ 180 deg and cos(glint) = mu0 mu - s0 s cos(raa).  The
single-scattering angle is then cos(Theta) = -mu0 mu - s0 s cos(raa).  The
stored column cos_theta (analysis/build_feature_dataset.py) has the opposite
sign of the azimuth term and is NOT used.
"""
from __future__ import annotations

import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CACHE = OUT / "cache"
for p in (REPO / "src", REPO / "workspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analysis.spec_sensitivity import _apply_production_reference  # noqa: E402
from analysis.utils import apply_quality_filter  # noqa: E402
import mixed_surface_reference_count as mslc  # noqa: E402

EVAL_DIR = (REPO / "results_doppler" / "model_comparison" / "xgb"
            / "fm1620k5v2xt41_eval_dates" / "per_date_nok2diag")
SRC_DIR = REPO / "results_doppler" / "csv_collection"
MCD12C1_DIR = REPO / "data" / "MODIS" / "MCD12C1"
TABLE = CACHE / "eval_footprints.parquet"

TAU_R = 0.0255          # Rayleigh optical depth at 765 nm, 1013.25 hPa
FILL = -999990.0        # Lite fill values are -999999

STEMS = ["o2a_k1_am", "o2a_f_am", "o2a_exp_int_am", "o2a_a1_am", "o2a_k2diag_am",
         "wco2_k1_am", "wco2_k1w_am", "wco2_k2diag_am", "wco2_exp_int_am",
         "wco2_a1_am", "sco2_k1_am", "sco2_k1w_am", "sco2_k2_am",
         "sco2_exp_int_am", "sco2_a1_am", "sco2_a2_am"]
REF_COLS = [f"{p}_{s}_{m}" for p in ("r05", "r15") for s in STEMS
            for m in ("mean", "std")] + [f"{p}_alb_o2a_mean" for p in ("r05", "r15")]
BASE_COLS = ["date", "orbit_id", "fp_id", "lat", "lon", "sza", "vza", "saa", "vaa",
             "glint_angle", "operation_mode", "fp", "alt", "alt_std", "psfc",
             "psfc_prior", "t700", "ws_apriori", "airmass", "cld_dist_km",
             "sfc_type", "xco2_qf", "xco2_bc", "snow_flag", "alb_o2a", "aod_total",
             "xco2_bc_anomaly_r05", "xco2_bc_anomaly_r15"]

# columns kept in the cached table
KEEP = ["date", "orbit_id", "fp_id", "lat", "lon", "sza", "vza", "raa",
        "glint_angle", "operation_mode", "fp", "alt", "alt_std", "psfc_prior",
        "t700", "ws_apriori", "airmass", "cld_dist_km", "sfc_type", "xco2_qf",
        "alb_o2a", "aod_total", "cos_scat", "X", "X_wrongsign",
        # absolute quantities
        "o2a_k1_am", "o2a_f_am", "o2a_exp_int_am", "sco2_k2k1sq", "sco2_wco2_k1",
        "fg_over_X", "fA_over_X", "fg",
        # deltas against the per-surface reference
        "dk1_o2a", "df_o2a", "dexp_o2a", "dk2k1sq_sco2", "dk1ratio_sw",
        "dfg_over_X", "dfA_over_X", "zexp_o2a", "xco2_bc_anomaly", "mixed"]


def eval_dates() -> list[str]:
    return sorted(p.stem for p in EVAL_DIR.glob("*.parquet")
                  if not p.name.startswith("._"))


def scattering_cos(sza, vza, saa, vaa):
    """cos(Theta) for single scattering of the direct beam into the view.

    Returns (cos_theta, raa).  raa = |saa - vaa| folded to [0, 180]."""
    raa = 180.0 - np.abs((np.asarray(saa) - np.asarray(vaa)) % 360.0 - 180.0)
    t0, t1, ph = np.radians(sza), np.radians(vza), np.radians(raa)
    cos_scat = -np.cos(t0) * np.cos(t1) - np.sin(t0) * np.sin(t1) * np.cos(ph)
    return cos_scat, raa


def rayleigh_x(cos_scat, sza, vza, p_hpa):
    """X = tau_R (p/1013.25) P(Theta) / (4 mu mu0), P = 0.75 (1 + cos^2)."""
    mu0, mu = np.cos(np.radians(sza)), np.cos(np.radians(vza))
    tau = TAU_R * np.asarray(p_hpa) / 1013.25
    return tau * 0.75 * (1.0 + cos_scat ** 2) / (4.0 * mu * mu0)


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-12, a / b, np.nan)


def _mixed_flags(path: Path) -> pd.DataFrame:
    """Mixed-surface reference-window flag per footprint (fp_id, mixed),
    via the production replica in workspace/mixed_surface_reference_count.py.
    Targets out to 50 km so the E_z far window (20-50 km) is screened too."""
    data = mslc.load(path, None, "v2")
    land = mslc.analyse(data, surface=1, min_cld_dist=15.0, near_max=50.0,
                        anom_col="xco2_bc_anomaly_r15", ref_mean_col="r15_exp_mean",
                        ref_std_col="r15_exp_std", require_qf0=False)
    ocean = mslc.analyse(data, surface=0, min_cld_dist=5.0, near_max=50.0,
                         anom_col="xco2_bc_anomaly_r05", ref_mean_col="r05_exp_mean",
                         ref_std_col="r05_exp_std", require_qf0=False)
    fl = pd.concat([land, ocean], ignore_index=True)
    return pd.DataFrame({"fp_id": fl["fp_id"].astype("int64"),
                         "mixed": (fl["n_ref_other"] > 0).to_numpy()})


def build_date(date: str) -> pd.DataFrame:
    path = SRC_DIR / f"combined_{date}_all_orbits.parquet"
    names = set(pq.read_schema(path).names)
    cols = [c for c in BASE_COLS + STEMS + REF_COLS if c in names]
    df = pq.read_table(path, columns=cols).to_pandas()
    df = apply_quality_filter(df, require_qf0=False)
    df = df[np.isfinite(df["cld_dist_km"]) & (df["cld_dist_km"] >= 0)].copy()
    for c in ("alt", "t700", "ws_apriori", "alt_std", "aod_total"):
        df.loc[df[c] < FILL, c] = np.nan

    df = _apply_production_reference(df, "v2")
    ocean = (df["sfc_type"] == 0).to_numpy()

    def ref(stem):
        return np.where(ocean, df[f"r05_{stem}_mean"], df[f"r15_{stem}_mean"])

    cos_scat, raa = scattering_cos(df["sza"], df["vza"], df["saa"], df["vaa"])
    df["raa"] = raa
    df["cos_scat"] = cos_scat
    df["X"] = rayleigh_x(cos_scat, df["sza"], df["vza"], df["psfc_prior"])
    # the opposite azimuth sign (stored cos_theta convention), for the check only
    df["X_wrongsign"] = rayleigh_x(
        -np.cos(np.radians(df["sza"])) * np.cos(np.radians(df["vza"]))
        + np.sin(np.radians(df["sza"])) * np.sin(np.radians(df["vza"]))
        * np.cos(np.radians(raa)), df["sza"], df["vza"], df["psfc_prior"])

    f, g, a = df["o2a_f_am"].to_numpy(), df["o2a_exp_int_am"].to_numpy(), df["alb_o2a"].to_numpy()
    X = df["X"].to_numpy()
    df["fg"] = f * g
    df["fg_over_X"] = _safe_div(f * g, X)
    df["fA_over_X"] = _safe_div(f * a, X)
    df["sco2_k2k1sq"] = _safe_div(df["sco2_k2_am"], df["sco2_k1_am"] ** 2)
    df["sco2_wco2_k1"] = _safe_div(df["sco2_k1_am"], df["wco2_k1_am"])
    # composite deltas: reference = composite of the reference means (first order,
    # as make_landclass_heatmap_figure.SHAPE_ROWS); X is the footprint's own geometry
    rf, rg, ra = ref("o2a_f_am"), ref("o2a_exp_int_am"), np.where(
        ocean, df["r05_alb_o2a_mean"], df["r15_alb_o2a_mean"])
    df["dfg_over_X"] = df["fg_over_X"] - _safe_div(rf * rg, X)
    df["dfA_over_X"] = df["fA_over_X"] - _safe_div(rf * ra, X)
    df["dk2k1sq_sco2"] = df["sco2_k2k1sq"] - _safe_div(ref("sco2_k2_am"), ref("sco2_k1_am") ** 2)
    df["dk1ratio_sw"] = df["sco2_wco2_k1"] - _safe_div(ref("sco2_k1_am"), ref("wco2_k1_am"))

    flags = _mixed_flags(path)
    df = df.merge(flags, on="fp_id", how="left")
    df["mixed"] = df["mixed"].fillna(False).astype(bool)
    out = df[KEEP].copy()
    for c in out.select_dtypes("float64").columns:
        out[c] = out[c].astype("float32")
    return out


def _build_one(date: str) -> str:
    p = CACHE / f"{date}.parquet"
    if not p.exists():
        build_date(date).to_parquet(p, index=False)
    return date


def build_table(nproc: int = 6) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    dates = eval_dates()
    with Pool(nproc) as pool:
        for i, d in enumerate(pool.imap_unordered(_build_one, dates), 1):
            print(f"  [{i}/{len(dates)}] {d}", flush=True)
    df = pd.concat([pd.read_parquet(CACHE / f"{d}.parquet") for d in dates],
                   ignore_index=True)
    from analysis.land_cover import assign_land_cover
    df = assign_land_cover(df, MCD12C1_DIR)
    df["igbp_group"] = df["igbp_group"].astype(str)
    df.loc[df["sfc_type"] == 0, "igbp_group"] = "ocean"
    df["month"] = pd.to_datetime(df["date"].astype(str)).dt.month.astype("int8")
    df.to_parquet(TABLE, index=False)
    print(f"wrote {TABLE}: {len(df):,} rows")


def load_table(columns=None) -> pd.DataFrame:
    return pd.read_parquet(TABLE, columns=columns)


# ---------------------------------------------------------------------------
# effect size, same definition as analysis.land_class.build_effect_sizes
# (near 0-5 km minus far 20-50 km, over the far-window SD; analytic 95 % CI)
NEAR = (0.0, 5.0)
FAR = (20.0, 50.0)


def effect_z(d: np.ndarray, v: np.ndarray, n_min: int = 500) -> dict:
    ok = np.isfinite(v)
    yn = v[ok & (d >= NEAR[0]) & (d < NEAR[1])]
    yf = v[ok & (d >= FAR[0]) & (d < FAR[1])]
    rec = {"n_near": len(yn), "n_far": len(yf), "E_z": np.nan, "ci95": np.nan,
           "near_mean": np.nan, "far_mean": np.nan, "far_sd": np.nan}
    if len(yn) >= n_min and len(yf) >= n_min and yf.std(ddof=1) > 0:
        sd = yf.std(ddof=1)
        se = np.hypot(yn.std(ddof=1) / np.sqrt(len(yn)), sd / np.sqrt(len(yf)))
        rec.update(E_z=(yn.mean() - yf.mean()) / sd, ci95=1.96 * se / sd,
                   near_mean=yn.mean(), far_mean=yf.mean(), far_sd=sd)
    return rec


def illum_class(z: np.ndarray) -> np.ndarray:
    """Fig. 5 classes by z_gamma (O2A continuum-reflectance z-score)."""
    return np.where(z > 0.5, "brightened",
                    np.where(z < -0.5, "shadowed",
                             np.where(np.abs(z) <= 0.5, "neutral", "none")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    build_table()

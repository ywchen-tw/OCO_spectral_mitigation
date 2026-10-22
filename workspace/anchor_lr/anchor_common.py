"""anchor_common.py — shared definitions for the TCCON-anchored ML(raw) test.

Chain under test (land only):

    xco2_raw  →  ML(raw)  →  raw_corr = xco2_raw − mu_raw
              →  LR anchor →  raw_corr_anchor = raw_corr − LR(f)

The LR mimics the B11 land parametric bias correction (Lite DUG v3.0 rev B,
Table 5-4; the predictor list is recorded in
manuscript/JQSRT_draft/review/sec1_bigpicture_proposed.tex):

    LR(f) = c0 + Σ_k fp_k·1[fp = k]
          + c·(dpfrac, co2_grad_del, log(dws), aod_fine, aod_ice,
               alb_wco2, alb_wco2²)            (centered at pool medians)

Two assumptions to confirm against the DUG (flagged in the plan):
  * aod_fine = aod_sulfate + aod_oc + aod_bc (the fine-mode MERRA-2 types);
  * "albedo_quad_wco2" is implemented generally as two free terms
    (alb_wco2, alb_wco2²) rather than the DUG's fixed quadratic form.
The B11 scale factor is fixed at 1 (additive form only): over the 2016–2020
pool, slope and intercept are nearly degenerate.

The regression is fit ONCE on the training-date TCCON pool (station-days on
the 116 training dates, never used for evaluation) with out-of-fold mu_raw,
and applied to the ML(raw) evaluation tree de_prof_reg_mix_raw{RUN_SUFFIX}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / 'src', ROOT / 'workspace'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from constants import RUN_SUFFIX                      # noqa: E402
from utils import get_storage_dir                     # noqa: E402

# ── paths ────────────────────────────────────────────────────────────────────
OUT_ROOT = ROOT / 'results' / 'anchor_lr'
POOL_DIR = OUT_ROOT / 'pool'
MANIFEST = OUT_ROOT / 'pool_manifest.csv'
COEF_JSON = OUT_ROOT / 'anchor_lr_land.json'
TRAIN_PARQUET = ROOT / 'results' / 'csv_collection' / 'combined_2016_2020_dates.parquet'
MODEL_ROOT = ROOT / 'results' / 'model_deep_ensemble'
RAW_LAND_FOLD_FMT = 'de_land_beta_nll_prof_reg_raw_foldpca_r15' + RUN_SUFFIX + '_f{k}'
RAW_TREE = ROOT / 'results' / 'model_comparison' / 'deep_ensemble' / f'de_prof_reg_mix_raw{RUN_SUFFIX}'
LAUNCHER = ROOT / 'curc_shell_blanca_plot_corr_xco2_deepens.sh'
TCCON_DIR = ROOT / 'data' / 'TCCON'
N_FOLDS = 5

# ── columns ──────────────────────────────────────────────────────────────────
ML_RAW_CORR_COL = 'deep_ensemble_corrected_xco2'   # = xco2_raw − mu_raw in the raw tree
INC_COL = 'anchor_lr_increment'
CORR_COL = 'anchor_lr_corrected_xco2'
GUARD_COL = 'anchor_lr_guard'


def variant_tag(variant: str) -> str:
    """'' for the default (b11) variant, else '_<variant>'."""
    return '' if variant in ('', 'b11') else f'_{variant}'


def inc_col(variant: str = '') -> str:
    return f'anchor_lr{variant_tag(variant)}_increment'


def corr_col(variant: str = '') -> str:
    return f'anchor_lr{variant_tag(variant)}_corrected_xco2'


def guard_col(variant: str = '') -> str:
    return f'anchor_lr{variant_tag(variant)}_guard'


def coef_json(variant: str = '') -> Path:
    return OUT_ROOT / f'anchor_lr_land{variant_tag(variant)}.json'


def report_dir(variant: str = '') -> Path:
    return RAW_TREE / f'anchor_lr{variant_tag(variant)}'


def report_sfx(variant: str = '') -> str:
    return f'_anchor_lr{variant_tag(variant)}_r100km'
MERGE_KEYS = ['time', 'lon', 'lat']                # plot_data ↔ source parquet (as the report)
# Raw columns the term library needs (merged from the source parquet when a
# plot_data lacks them).  Any missing column just makes its terms NaN.
SOURCE_COLS = ('dpfrac', 'co2_grad_del', 'dws', 'aod_sulfate', 'aod_oc', 'aod_bc',
               'aod_ice', 'alb_wco2', 'alb_o2a', 'alb_sco2', 'fp',
               '1_over_cos_sza', '1_over_cos_vza', 'cos_glint_angle', 'operation_mode',
               'snow_flag', 'psfc_prior', 't700', 'tcwv', 'delT', 'h2o_scale',
               'xco2_uncertainty', 'fp_area_km2', 's31', 'tropopause_pressure',
               'o2a_k1_nosg', 'o2a_k2_nosg', 'wco2_k1_nosg', 'wco2_k2_nosg', 'wco2_k3_nosg',
               'sco2_k1_nosg', 'sco2_k2_nosg')
FILL_LT = -9e5       # Lite fill value −999999 (t700, alt_std, ws, …)
# The B11 land family (Lite DUG v3.0 rev B Table 5-4 + footprint offsets).
CONT_TERMS = ('dpfrac', 'co2_grad_del', 'log_dws', 'aod_fine', 'aod_ice',
              'alb_wco2', 'alb_wco2_sq')
# Candidate terms OUTSIDE the operational correction, chosen to be insensitive
# to nearby clouds (geometry, prior meteorology, scene/time descriptors).
GEOM_TERMS = ('inv_cos_sza', 'inv_cos_vza', 'mode_glint', 'lat', 'lat_sq',
              'doy_sin', 'doy_cos', 'snow_flag')
MET_TERMS = ('log_psfc_prior', 't700', 'tcwv', 'log_trop_p')
# Photon path-length statistics (production no-SG cumulant fit, src/spectral):
# k1 = mean path-length enhancement, k2 = its variance, per band.
SPEC_TERMS = ('o2a_k1', 'o2a_k2', 'wco2_k1', 'wco2_k2', 'wco2_k3', 'sco2_k1', 'sco2_k2')
# Term sets for the sweep (all get intercept + 8 footprint offsets):
#   b11           full B11 land family
#   nocontam      B11 minus the contamination-group terms (log_dws, aod_fine, aod_ice)
#   dp_only       retrieval-state only (dpfrac, co2_grad_del)
#   alb_only      WCO2 albedo quadratic only
#   fp_only       intercept + footprint offsets (station-day-scale anchor)
#   geom          cloud-insensitive geometry / scene / season terms
#   met           prior meteorology (surface pressure, T700, TCWV, tropopause)
#   geom_met      geom + met
#   geom_met_alb  geom + met + albedo quadratic
#   b11_plus      b11 + geom + met
#   spec          path-length cumulants only
#   spec_geom_met cumulants + geom + met
#   spec_b11      b11 + cumulants
TERM_SETS = {
    'b11': CONT_TERMS,
    'nocontam': ('dpfrac', 'co2_grad_del', 'alb_wco2', 'alb_wco2_sq'),
    'dp_only': ('dpfrac', 'co2_grad_del'),
    'alb_only': ('alb_wco2', 'alb_wco2_sq'),
    'fp_only': (),
    'geom': GEOM_TERMS,
    'met': MET_TERMS,
    'geom_met': GEOM_TERMS + MET_TERMS,
    'geom_met_alb': GEOM_TERMS + MET_TERMS + ('alb_wco2', 'alb_wco2_sq'),
    'b11_plus': CONT_TERMS + GEOM_TERMS + MET_TERMS,
    'spec': SPEC_TERMS,
    'spec_geom_met': SPEC_TERMS + GEOM_TERMS + MET_TERMS,
    'spec_b11': CONT_TERMS + SPEC_TERMS,
}
N_FP = 8
DWS_FLOOR = 1e-4     # dws > 0 everywhere in the training set (min 8.5e-5); floor for log
_DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def raw_fold_dir(k: int) -> Path:
    return MODEL_ROOT / RAW_LAND_FOLD_FMT.format(k=k)


def case_name(date: str, site: str) -> str:
    return f'combined_{date}_{site}'


def date_from_name(name: str) -> str | None:
    m = _DATE_RE.search(name)
    return m.group(1) if m else None


def tccon_file(code: str) -> Path | None:
    """Unique <code>*.public.qc.nc under data/TCCON (local, then storage root)."""
    for root in (TCCON_DIR, get_storage_dir() / 'data' / 'TCCON'):
        hits = sorted(Path(root).glob(f'{code}*.public.qc.nc')) if Path(root).is_dir() else []
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise RuntimeError(f'{len(hits)} TCCON files match {code}* in {root}')
    return None


def source_parquet_for_date(date: str) -> Path | None:
    """Per-date source parquet (all 403 columns), as tccon_comparison_report."""
    name = f'combined_{date}_all_orbits.parquet'
    for base in (ROOT / 'results' / 'csv_collection',
                 get_storage_dir() / 'results' / 'csv_collection'):
        p = Path(base) / name
        if p.exists():
            return p
    return None


def active_run_case_dates(script: Path = LAUNCHER) -> set[str]:
    """Dates of the ACTIVE run_case lines (commented lines are not active)."""
    dates = set()
    for ln in Path(script).read_text().splitlines():
        if re.match(r'^run_case\s', ln):
            dates.add(ln.split()[1])
    return dates


def held_fold_for_date(date: str) -> int:
    """Index of the raw land fold whose held set contains *date* (exactly one)."""
    hits = []
    for k in range(N_FOLDS):
        man = json.loads((raw_fold_dir(k) / 'training_dates.json').read_text())
        if date in set(man.get('held_dates') or ()):
            hits.append(k)
    if len(hits) != 1:
        raise RuntimeError(f'{date}: held in {len(hits)} raw land fold(s) {hits}; expected 1')
    return hits[0]


# ── regression design ────────────────────────────────────────────────────────
def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """Column as float with Lite fill values → NaN; all-NaN when absent."""
    if name not in df.columns:
        return np.full(len(df), np.nan)
    v = df[name].to_numpy(float)
    return np.where(v < FILL_LT, np.nan, v)


def continuous_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Every candidate continuous term (uncentered); NaN where inputs are missing."""
    out = pd.DataFrame(index=df.index)
    # B11 land family
    out['dpfrac'] = _col(df, 'dpfrac')
    out['co2_grad_del'] = _col(df, 'co2_grad_del')
    out['log_dws'] = np.log(np.maximum(_col(df, 'dws'), DWS_FLOOR))
    out['aod_fine'] = _col(df, 'aod_sulfate') + _col(df, 'aod_oc') + _col(df, 'aod_bc')
    out['aod_ice'] = _col(df, 'aod_ice')
    alb = _col(df, 'alb_wco2')
    out['alb_wco2'] = alb
    out['alb_wco2_sq'] = alb ** 2
    # geometry / scene / season (cloud-insensitive)
    out['inv_cos_sza'] = _col(df, '1_over_cos_sza')
    out['inv_cos_vza'] = _col(df, '1_over_cos_vza')
    out['mode_glint'] = (_col(df, 'operation_mode') == 1).astype(float)
    lat = _col(df, 'lat')
    out['lat'] = lat
    out['lat_sq'] = lat ** 2
    t = _col(df, 'time')
    doy = (pd.to_datetime(t, unit='s', errors='coerce').dayofyear.to_numpy(float)
           if np.isfinite(t).any() else np.full(len(df), np.nan))
    out['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    out['snow_flag'] = _col(df, 'snow_flag')
    # prior meteorology
    out['log_psfc_prior'] = np.log(_col(df, 'psfc_prior'))
    out['t700'] = _col(df, 't700')
    out['tcwv'] = _col(df, 'tcwv')
    out['log_trop_p'] = np.log(_col(df, 'tropopause_pressure'))
    # path-length cumulants (no-SG production fit)
    for k in SPEC_TERMS:
        out[k] = _col(df, f'{k}_nosg')
    return out


def term_names(terms=CONT_TERMS) -> list[str]:
    return ['intercept'] + [f'fp{k}' for k in range(N_FP - 1)] + list(terms)


def design_matrix(df: pd.DataFrame, centers: dict | None = None, terms=CONT_TERMS,
                  footprint_contrasts: bool = True):
    """Design matrix for the anchor LR.

    Columns: intercept; footprint sum-to-zero contrasts (1[fp=k] − 1[fp=7],
    k = 0..6, so the intercept is the footprint-mean offset and the fp 7 offset
    is −Σ of the others) unless footprint_contrasts=False (per-footprint
    regressions: intercept + terms only); the centered continuous terms.
    Returns (X, names, centers, ok) where ok marks rows with all-finite inputs.
    """
    terms = tuple(terms)
    tv = continuous_terms(df)
    if centers is None:
        centers = {c: float(np.nanmedian(tv[c].to_numpy(float))) for c in terms}
    fp = df['fp'].to_numpy(float)
    n = len(df)
    n_fp_cols = (N_FP - 1) if footprint_contrasts else 0
    X = np.zeros((n, 1 + n_fp_cols + len(terms)))
    X[:, 0] = 1.0
    if footprint_contrasts:
        last = (fp == N_FP - 1).astype(float)
        for k in range(N_FP - 1):
            X[:, 1 + k] = (fp == k).astype(float) - last
    for j, c in enumerate(terms):
        X[:, 1 + n_fp_cols + j] = tv[c].to_numpy(float) - centers[c]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(fp) & (fp >= 0) & (fp < N_FP)
    names = term_names(terms) if footprint_contrasts else ['intercept'] + list(terms)
    return X, names, centers, ok


def footprint_offsets(coef: dict) -> list[float]:
    offs = [float(coef[f'fp{k}']) for k in range(N_FP - 1)]
    return offs + [-float(sum(offs))]


def predict_increment(df: pd.DataFrame, model: dict) -> np.ndarray:
    """LR(f) per row (NaN where any input is missing)."""
    terms = tuple(model.get('terms', {}).get('continuous', CONT_TERMS))
    inc = np.full(len(df), np.nan)
    if model.get('per_footprint'):
        X, names, _, ok = design_matrix(df, centers=model['centers'], terms=terms,
                                        footprint_contrasts=False)
        fp = df['fp'].to_numpy(float)
        for k in range(N_FP):
            coef = model['coef_by_fp'].get(str(k))
            m = ok & (fp == k)
            if coef is None or not m.any():
                continue
            beta = np.array([coef[n] for n in names], float)
            inc[m] = X[m] @ beta
        return inc
    X, names, _, ok = design_matrix(df, centers=model['centers'], terms=terms)
    beta = np.array([model['coef'][n] for n in names], float)
    inc[ok] = X[ok] @ beta
    return inc


def load_model(path: Path = COEF_JSON) -> dict:
    return json.loads(Path(path).read_text())

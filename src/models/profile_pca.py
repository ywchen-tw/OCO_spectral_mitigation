"""ProfilePCA — EOF compression of the vertical atmospheric-profile variables.

build_feature_dataset.compute_sigma_profile_columns() resamples the raw GEOS
temperature / specific-humidity / CO2-prior profiles onto a fixed sigma = P/Psurf
grid (17 levels), producing the flat columns

    t_sigma_00 … t_sigma_16          (temperature,  K)
    q_sigma_00 … q_sigma_16          (specific humidity, kg/kg)
    co2prior_sigma_00 … co2prior_sigma_16   (CO2 prior, mol/mol)

Adjacent sigma levels are strongly correlated, so each profile lives on a very
low-dimensional manifold: a handful of empirical orthogonal functions (EOFs =
principal components) reconstruct it to ~99.9 % variance.  This module fits one
PCA **per variable** (T, q, CO2-prior) — separate, physically interpretable EOF
sets rather than one PCA over the mixed 51-column block — and emits a compact
score matrix (``t_pc01``, ``q_pc01``, ``co2prior_pc01``, …) that downstream models
can consume in place of the 51 raw profile columns.

Design mirrors models/pipeline.py::FeaturePipeline:
  * per-group  log1p (q only, by default) → StandardScaler(per level) → PCA
  * fit on the TRAIN split only, then transform any later data (2016-2020)
  * fully picklable + save()/load(); __module__ pinned for pickle portability
  * deterministic component signs so refits reproduce bit-for-bit

EOFs are fitted SEPARATELY per surface type (land vs ocean atmospheres differ), so
one transformer is saved per surface (profile_pca_land.pkl / profile_pca_ocean.pkl).

Typical usage
-------------
    # fit each surface on its own rows (leakage-safe on the TRAIN split only).
    # n_components=4 keeps only the leading bulk EOFs — these are reanalysis priors,
    # not observations, so higher modes are model noise, not signal.
    land = ProfilePCA.fit(df[df.sfc_type == 1], n_components=4)
    land.save("results/model_mlp_lr/profile_pca_land.pkl")
    print(land.explained_variance_report())

    # later, on the full 2016-2020 frame — transform each surface with its own model:
    land  = ProfilePCA.load("results/model_mlp_lr/profile_pca_land.pkl")
    scores = land.transform_df(df_full[df_full.sfc_type == 1])   # aligned to that index
    df_full = df_full.join(scores)                               # append PC columns

The CLI fits both surfaces in one pass:  python src/models/profile_pca.py --n-components 0.99
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Put src/ on the path so `from utils import get_storage_dir` resolves when this
# file is run directly (python src/models/profile_pca.py).  Same idiom as
# analysis/build_feature_dataset.py; harmless when a launcher already added it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Register under the canonical name 'profile_pca' so pickles are portable
# regardless of how this file was imported (as __main__ via
# ``python src/models/profile_pca.py`` or as models.profile_pca).  Same trick as
# pipeline.py — the picklable helper classes pin __module__ = 'profile_pca'.
if __name__ == '__main__':
    sys.modules['profile_pca'] = sys.modules['__main__']
else:
    sys.modules.setdefault('profile_pca', sys.modules[__name__])


# Sigma grid used by build_feature_dataset.compute_sigma_profile_columns().  Kept
# here (rather than imported) so this transformer stays free of the heavy
# fitting-pipeline import chain; only used to LABEL EOF plots by sigma level.  The
# CLI asserts it matches build_feature_dataset.SIGMA_LEVELS to catch drift.
SIGMA_LEVELS = np.array([
    0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00,
], dtype=np.float64)

# Profile variable groups: prefix → (short PC-column stem, apply log1p?, per-row norm).
# q spans ~4 orders of magnitude between the tropopause and the surface, so it is
# log1p-compressed before scaling (negatives — a few tiny sub-zero humidities —
# are clipped to 0, i.e. log1p(0)=0), exactly as pipeline._apply_log1p does for
# the skewed AOD/area features.  T and CO2-prior are near-Gaussian per level → no
# log.  ``stem`` names the emitted score columns: t_pc01, q_pc01, co2prior_pc01…
#
# ``norm`` divides each sounding's profile by its own column mean/max BEFORE
# scaling, collapsing the profile to unit-level SHAPE.  'mean' is the σ (mass)-
# weighted column mean (trapezoidal Δσ weights; see _sigma_mass_weights) — the
# physically correct bulk-column value, not a simple average that would over-weight
# the finely-sampled near-surface/TOA levels.  CO2-prior needs it: these
# are reanalysis priors that ride the atmospheric CO2 secular trend (~2.5 ppm/yr),
# so a model fit on 2016-2020 sees a raw-level EOF1 that a 2025 profile blows past
# by ~+3 sigma (pure out-of-distribution extrapolation).  Per-row normalisation is
# STATELESS (uses the sounding's own levels), so it is year-invariant by
# construction and the absolute level — already carried by xco2_apriori — is
# dropped.  T and q are stationary (weather/seasonal variance ≫ any multi-year
# drift) → norm=None, keep the real signal.
_PROFILE_GROUPS = {
    't_sigma_':        {'stem': 't',        'log1p': False, 'norm': None},
    'q_sigma_':        {'stem': 'q',        'log1p': True,  'norm': None},
    'co2prior_sigma_': {'stem': 'co2prior', 'log1p': False, 'norm': 'mean'},
}

# Scalar vertical-landmark features appended verbatim after the profile PC scores.
# tropopause_sigma = P_trop / Psurf (build_feature_dataset.compute_sigma_profile_columns)
# is the tropopause height already in the same sigma coordinate as the resampled
# profiles — a single number per sounding, so it cannot enter the per-level EOF PCA
# but rides alongside the PC scores as a companion feature.  tropopause_temp is the
# tropopause temperature (K).  Missing columns are silently skipped at fit time.
_SCALAR_PASSTHROUGH = ['tropopause_sigma', 'tropopause_temp']

# Surface types are fitted with SEPARATE EOFs — land and ocean atmospheres differ
# enough (boundary-layer moisture, orography) that shared modes blur both.  Maps
# the sfc_type flag to the tag used in artifact filenames.
SURFACE_NAMES = {0: 'ocean', 1: 'land'}


def _discover_columns(df: pd.DataFrame, prefix: str) -> list:
    """Return df columns beginning with *prefix*, ordered by their numeric suffix.

    Robust to however many sigma levels the parquet was built with (17 today) —
    the level count is read from the data, never hard-coded.
    """
    cols = [c for c in df.columns if c.startswith(prefix)]

    def _lvl(c):
        tail = c[len(prefix):]
        return int(tail) if tail.isdigit() else 10**9

    return sorted(cols, key=_lvl)


def _apply_log1p(X: np.ndarray) -> np.ndarray:
    """log1p with negatives clamped to 0 (matches pipeline._apply_log1p)."""
    return np.log1p(np.clip(X, 0.0, None))


def _sigma_mass_weights(n_levels: int,
                        sigma_levels: np.ndarray = SIGMA_LEVELS) -> np.ndarray:
    """Trapezoidal σ-layer-thickness weights for a mass-weighted column mean.

    On a sigma = P/Psurf grid the pressure thickness of each layer is dp = Psurf·dσ,
    so equal σ-thickness carries equal dry-air mass and Psurf cancels in any
    normalised column average.  Returns the trapezoidal integration weights over
    ``sigma_levels[:n_levels]`` (Σ = σ_bottom − σ_top), i.e. each level's σ (mass)
    share — coarsely-sampled mid-troposphere levels correctly get MORE weight than
    the finely-sampled near-surface/TOA levels a simple average over-counts.
    """
    s = np.asarray(sigma_levels[:n_levels], dtype=np.float64)
    if n_levels == 1:
        return np.ones(1, dtype=np.float64)
    w = np.empty(n_levels, dtype=np.float64)
    w[0]    = (s[1] - s[0]) / 2.0
    w[-1]   = (s[-1] - s[-2]) / 2.0
    w[1:-1] = (s[2:] - s[:-2]) / 2.0
    return w


class _GroupPCA:
    """Per-variable EOF model: (optional log1p) → StandardScaler → PCA.

    Holds everything needed to transform new data and to interpret / plot the
    EOFs, plus the fitted explained-variance vector.  Picklable; __module__ pinned
    so it unpickles whether profile_pca.py was imported as __main__ or as a module.
    """

    __module__ = 'profile_pca'

    def __init__(self, prefix: str, stem: str, columns: list, log1p: bool,
                 norm: Optional[str] = None):
        self.prefix  = prefix
        self.stem    = stem
        self.columns = list(columns)
        self.log1p   = bool(log1p)
        self.norm    = norm            # None | 'mean' | 'max' — per-row shape normalisation
        self.scaler  = StandardScaler()
        self.pca: Optional[PCA] = None
        self.n_components_: int = 0
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    # ── internal ──────────────────────────────────────────────────────────────
    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        """Extract this group's columns as float32: per-row norm → log1p (as set).

        The per-row normalisation (divide by the sounding's own column mean/max)
        runs on the LINEAR profile first, before any log1p, and is stateless — it
        uses only the sounding's own levels, so a future-year profile with a
        shifted absolute level maps onto the SAME shape the EOFs were fit on.
        """
        X = df[self.columns].to_numpy(dtype=np.float32)
        norm = getattr(self, 'norm', None)   # getattr: old pickles predate this attr
        if norm:
            if norm == 'mean':
                # σ (mass)-weighted column mean — trapezoidal Δσ weights, NaN-aware
                # (finite levels renormalise; fully-finite rows — the ones that
                # survive _valid_mask — use all 17 weights).
                w = _sigma_mass_weights(X.shape[1])[np.newaxis, :]     # [1, L]
                finite = np.isfinite(X)
                wf = np.where(finite, w, 0.0)
                wsum = wf.sum(axis=1, keepdims=True)
                num = np.where(finite, X, 0.0) * wf
                with np.errstate(divide='ignore', invalid='ignore'):
                    denom = np.where(wsum > 0, num.sum(axis=1, keepdims=True) / wsum, np.nan)
            elif norm == 'max':
                denom = np.nanmax(X, axis=1, keepdims=True)
            else:
                raise ValueError(f"norm must be None/'mean'/'max', got {norm!r}")
            with np.errstate(divide='ignore', invalid='ignore'):
                X = np.where(np.isfinite(denom) & (denom != 0), X / denom, np.nan)
        if self.log1p:
            X = _apply_log1p(X)
        return X.astype(np.float32)

    @staticmethod
    def _valid_mask(X: np.ndarray) -> np.ndarray:
        return np.isfinite(X).all(axis=1)

    # ── fit / transform ─────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, n_components: Union[int, float]) -> '_GroupPCA':
        X = self._raw(df)
        valid = self._valid_mask(X)
        n_valid = int(valid.sum())
        if n_valid < 2:
            raise ValueError(
                f"[{self.stem}] only {n_valid} finite rows across {len(self.columns)} "
                f"levels — cannot fit PCA.")
        Xs = self.scaler.fit_transform(X[valid])

        # A float n_components (variance fraction) can never exceed n_features;
        # an int is capped at the level count.  Fit, then freeze the resolved
        # integer count so transform() output width is deterministic.
        max_k = min(len(self.columns), n_valid)
        nc = n_components if isinstance(n_components, float) else min(n_components, max_k)
        pca = PCA(n_components=nc, random_state=42, svd_solver='full')
        pca.fit(Xs)

        # Deterministic sign: make the largest-|loading| entry of each EOF positive
        # so independent refits (and land vs ocean) agree on orientation.
        comps = pca.components_
        signs = np.sign(comps[np.arange(comps.shape[0]),
                               np.argmax(np.abs(comps), axis=1)])
        signs[signs == 0] = 1.0
        pca.components_ = comps * signs[:, None]

        self.pca = pca
        self.n_components_ = pca.n_components_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_.copy()
        logger.info(
            "[%s] fit: %d levels → %d PCs, cum. var %.4f (%d/%d finite rows)",
            self.stem, len(self.columns), self.n_components_,
            float(self.explained_variance_ratio_.sum()), n_valid, len(X),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return [N, n_components_] scores; rows with any non-finite level → NaN."""
        X = self._raw(df)
        valid = self._valid_mask(X)
        out = np.full((X.shape[0], self.n_components_), np.nan, dtype=np.float32)
        if valid.any():
            Xs = self.scaler.transform(X[valid])
            out[valid] = self.pca.transform(Xs).astype(np.float32)
        return out

    @property
    def pc_names(self) -> list:
        return [f'{self.stem}_pc{i + 1:02d}' for i in range(self.n_components_)]


class ProfilePCA:
    """EOF compression of the sigma-grid T / q / CO2-prior profile columns.

    One :class:`_GroupPCA` per variable.  ``transform`` concatenates the groups'
    score matrices in registry order; ``feature_names`` gives the matching column
    names (t_pc01…, q_pc01…, co2prior_pc01…).
    """

    __module__ = 'profile_pca'

    def __init__(self, groups: dict, n_components: Union[int, float],
                 scalars: Optional[list] = None):
        self.groups = groups                 # stem → _GroupPCA
        self.n_components = n_components      # requested (int count or variance frac)
        self.scalars = list(scalars or [])   # scalar landmark cols appended verbatim

    # ── construction ────────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, df: pd.DataFrame,
            n_components: Union[int, float] = 4,
            groups: Optional[dict] = None,
            scalars: Optional[list] = None) -> 'ProfilePCA':
        """Fit one PCA per profile variable present in *df*.

        Parameters
        ----------
        df : DataFrame containing the sigma-grid profile columns.
        n_components : int → fixed PCs per group; float in (0,1) → smallest number
            of PCs reaching that cumulative variance fraction (per group).
            Default 4 — the leading bulk vertical modes only.  These profiles are
            GEOS *reanalysis* priors, not observations: their high-order EOFs are
            model-driven fine structure, not independent signal, so a handful of
            dominant modes (EOF1 alone is ~57-73 % per variable) is the point.
            Do NOT chase 99 % here — that just imports reanalysis noise as columns.
        groups : optional override of :data:`_PROFILE_GROUPS`
            (prefix → {'stem', 'log1p'}); defaults to T / q / CO2-prior.
        scalars : scalar vertical-landmark columns appended verbatim after the PC
            scores (default :data:`_SCALAR_PASSTHROUGH` = tropopause_sigma +
            tropopause_temp; those absent from *df* are dropped).  Pass ``[]`` to
            emit PC scores only.

        Leakage discipline: for blocked-split validation fit on the TRAIN split
        only, exactly like FeaturePipeline.fit — never on the full frame.
        """
        spec = groups or _PROFILE_GROUPS
        fitted = {}
        for prefix, g in spec.items():
            cols = _discover_columns(df, prefix)
            if not cols:
                logger.warning("ProfilePCA.fit: no columns for prefix %r — skipped", prefix)
                continue
            fitted[g['stem']] = _GroupPCA(prefix, g['stem'], cols, g['log1p'],
                                          g.get('norm')).fit(df, n_components)
        if not fitted:
            raise ValueError(
                "ProfilePCA.fit: none of the profile column groups "
                f"{list(spec)} were found in the DataFrame.")

        want_scalars = _SCALAR_PASSTHROUGH if scalars is None else scalars
        kept_scalars = [c for c in want_scalars if c in df.columns]
        missing = [c for c in want_scalars if c not in df.columns]
        if missing:
            logger.warning("ProfilePCA.fit: scalar passthrough columns absent, "
                           "dropped: %s", missing)
        logger.info("ProfilePCA fitted: %d groups → %d PC columns + %d scalar(s) %s",
                    len(fitted), sum(g.n_components_ for g in fitted.values()),
                    len(kept_scalars), kept_scalars or '[]')
        return cls(groups=fitted, n_components=n_components, scalars=kept_scalars)

    # ── transform ───────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return [N, n_features] float32 matrix: PC scores then scalar landmarks.

        Profile-PC entries are NaN for rows with any non-finite level; scalar
        columns pass through verbatim (their own NaNs preserved).
        """
        blocks = [g.transform(df) for g in self.groups.values()]
        if self.scalars:
            blocks.append(df[self.scalars].to_numpy(dtype=np.float32))
        return np.concatenate(blocks, axis=1)

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same as :meth:`transform` but as a DataFrame aligned to ``df.index``."""
        return pd.DataFrame(self.transform(df), columns=self.feature_names, index=df.index)

    @property
    def feature_names(self) -> list:
        names: list = []
        for g in self.groups.values():
            names += g.pc_names
        return names + list(getattr(self, 'scalars', []))

    @property
    def n_features(self) -> int:
        return sum(g.n_components_ for g in self.groups.values()) + len(getattr(self, 'scalars', []))

    # ── diagnostics ─────────────────────────────────────────────────────────────
    def explained_variance_report(self) -> pd.DataFrame:
        """Per-group, per-PC explained-variance table (individual + cumulative)."""
        rows = []
        for g in self.groups.values():
            evr = g.explained_variance_ratio_
            cum = np.cumsum(evr)
            for i, (e, c) in enumerate(zip(evr, cum)):
                rows.append({'group': g.stem, 'pc': f'{g.stem}_pc{i + 1:02d}',
                             'explained_var': float(e), 'cumulative_var': float(c)})
        return pd.DataFrame(rows)

    def plot_scree(self, outdir, tag: str = '') -> None:
        """Cumulative-variance curve, one line per profile variable."""
        import os
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for g in self.groups.values():
            cum = np.cumsum(g.explained_variance_ratio_) * 100
            ax.plot(range(1, len(cum) + 1), cum, marker='o', label=g.stem)
        ax.axhline(99, color='gray', lw=1, ls='--', alpha=0.6)
        ax.set_xlabel('Number of EOFs')
        ax.set_ylabel('Cumulative explained variance (%)')
        ax.set_title(f'Profile PCA scree{" — " + tag if tag else ""}')
        ax.set_ylim(0, 101)
        ax.legend(title='variable')
        fig.tight_layout()
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f'profile_pca_scree{"_" + tag if tag else ""}.png')
        fig.savefig(path, dpi=140)
        plt.close(fig)
        logger.info("  saved → %s", path)

    def plot_eofs(self, outdir, tag: str = '', n_show: int = 4,
                  sigma_levels: np.ndarray = SIGMA_LEVELS) -> None:
        """Plot the leading EOF loadings vs sigma level, one panel per variable.

        sigma increases downward (surface = 1 at the bottom) so the panels read
        like an atmospheric profile.
        """
        import os
        import matplotlib.pyplot as plt
        groups = list(self.groups.values())
        fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 5),
                                 squeeze=False)
        axes = axes[0]
        for ax, g in zip(axes, groups):
            comps = g.pca.components_                     # [n_pc, n_levels]
            sig = sigma_levels[:comps.shape[1]]
            for i in range(min(n_show, comps.shape[0])):
                evr = g.explained_variance_ratio_[i] * 100
                ax.plot(comps[i], sig, marker='.', label=f'EOF{i + 1} ({evr:.1f}%)')
            ax.invert_yaxis()                             # surface at bottom
            ax.axvline(0, color='gray', lw=0.6, ls='--')
            ax.set_xlabel('loading')
            ax.set_ylabel('sigma = P / Psurf')
            ax.set_title(f'{g.stem} EOFs')
            ax.legend(fontsize=8)
        fig.suptitle(f'Profile PCA EOFs{" — " + tag if tag else ""}')
        fig.tight_layout()
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f'profile_pca_eofs{"_" + tag if tag else ""}.png')
        fig.savefig(path, dpi=140)
        plt.close(fig)
        logger.info("  saved → %s", path)

    # ── persistence ─────────────────────────────────────────────────────────────
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info("ProfilePCA saved → %s", path)

    @classmethod
    def load(cls, path) -> 'ProfilePCA':
        from utils import get_storage_dir
        path = Path(path)
        if not path.is_absolute():
            path = get_storage_dir() / path
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ProfilePCA, got {type(obj)}")
        logger.info("ProfilePCA loaded ← %s  (%d groups, %d PC columns)",
                    path, len(obj.groups), obj.n_features)
        return obj

    def __repr__(self) -> str:
        detail = ', '.join(f'{s}:{g.n_components_}' for s, g in self.groups.items())
        scal = getattr(self, 'scalars', [])
        return (f"ProfilePCA(n_components={self.n_components}, groups={{{detail}}}, "
                f"scalars={scal or '[]'})")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _fit_one_surface(df_all: pd.DataFrame, sfc_type: int,
                     n_components: Union[int, float],
                     out_dir: Path, scores_out_base: Optional[Path],
                     groups: Optional[dict] = None) -> None:
    """Fit + save one surface's ProfilePCA, with surface-tagged artifacts.

    Writes profile_pca_<surface>.pkl, ..._explained_variance.csv, and tagged
    scree / EOF plots under out_dir/profile_pca_plots/.
    """
    surf = SURFACE_NAMES.get(sfc_type, str(sfc_type))
    df = df_all[df_all['sfc_type'] == sfc_type]
    print(f"\n=== sfc_type={sfc_type} ({surf}): {len(df):,} rows ===", flush=True)
    if len(df) < 2:
        print(f"  too few rows for {surf} — skipped", flush=True)
        return

    ppca = ProfilePCA.fit(df, n_components=n_components, groups=groups)
    print(f"  {ppca}", flush=True)

    report = ppca.explained_variance_report()
    with pd.option_context('display.max_rows', None):
        print(report.to_string(index=False), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir  = out_dir / 'profile_pca_plots'
    out_path  = out_dir / f'profile_pca_{surf}.pkl'
    report_csv = out_dir / f'profile_pca_{surf}_explained_variance.csv'
    report.to_csv(report_csv, index=False, float_format='%.6f')
    print(f"  report → {report_csv}", flush=True)

    ppca.plot_scree(plot_dir, tag=surf)
    ppca.plot_eofs(plot_dir, tag=surf)
    ppca.save(out_path)
    print(f"  saved ProfilePCA → {out_path}", flush=True)

    if scores_out_base is not None:
        scores = ppca.transform_df(df)
        for idc in ('sounding_id', 'date', 'orbit_id'):
            if idc in df.columns:
                scores.insert(0, idc, df[idc].values)
        scores_path = scores_out_base.with_name(
            f'{scores_out_base.stem}_{surf}{scores_out_base.suffix or ".parquet"}')
        scores.to_parquet(scores_path, index=False, compression='zstd')
        print(f"  scores → {scores_path}  ({scores.shape})", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Fit ProfilePCA (EOF compression of sigma-grid T/q/CO2-prior "
                    "profiles) PER SURFACE TYPE on a combined parquet, save each "
                    "transformer, and write explained-variance reports + plots.")
    parser.add_argument('--data', default=None,
                        help='Input parquet/csv.  Defaults to '
                             '<storage_dir>/results/csv_collection/combined_2020_dates.parquet')
    parser.add_argument('--out-dir', default=None,
                        help='Directory for the per-surface artifacts '
                             '(profile_pca_<surface>.pkl, reports, plots).  '
                             'Defaults to <storage_dir>/results/model_mlp_lr')
    parser.add_argument('--n-components', default='4',
                        help="int → fixed PCs per group (default 4, the leading bulk "
                             "modes — these are reanalysis priors, not observations, so "
                             "high-order EOFs are model noise); float in (0,1) → "
                             "variance fraction per group.")
    parser.add_argument('--sfc-type', type=int, default=None, choices=[0, 1],
                        help='Fit only this surface (0=ocean, 1=land).  '
                             'Default: fit BOTH surfaces separately.')
    parser.add_argument('--co2-norm', default='mean', choices=['none', 'mean', 'max'],
                        help="Per-sounding normalisation of the CO2-prior profile "
                             "before PCA, making the EOFs invariant to the CO2 "
                             "secular trend so 2016-2020 EOFs generalise to later "
                             "years.  'mean' (default) = sigma/mass-weighted column "
                             "mean; 'max'; 'none' (raw — extrapolates badly on "
                             "out-of-range years).")
    parser.add_argument('--scores-out', default=None,
                        help='Optional parquet base path for the PC scores; the '
                             'surface tag is inserted before the extension '
                             '(e.g. scores.parquet → scores_land.parquet).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from utils import get_storage_dir
    storage_dir = get_storage_dir()

    def _resolve(arg, default):
        if arg is None:
            return Path(default) if default is not None else None
        p = Path(arg)
        return p if p.is_absolute() else storage_dir / p

    import platform
    _default_name = ('combined_2016_2020_dates.parquet' if platform.system() == 'Linux'
                     else 'combined_2020_dates.parquet')
    data_path = _resolve(args.data,
                         storage_dir / 'results/csv_collection' / _default_name)
    out_dir   = _resolve(args.out_dir, storage_dir / 'results/model_mlp_lr')
    scores_base = _resolve(args.scores_out, None)

    # n_components: '5' → int 5; '0.999' → float 0.999
    nc_raw = args.n_components
    n_components: Union[int, float] = float(nc_raw) if '.' in nc_raw else int(nc_raw)

    # Fail loud if the local SIGMA_LEVELS copy drifts from the source of truth.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from analysis.build_feature_dataset import SIGMA_LEVELS as SRC_SIGMA
        if not np.array_equal(SIGMA_LEVELS, SRC_SIGMA):
            logger.warning("SIGMA_LEVELS drift vs build_feature_dataset — EOF y-axis "
                           "labels may be wrong; sync the constant.")
    except Exception as e:                       # pragma: no cover — plotting labels only
        logger.info("Could not verify SIGMA_LEVELS against source (%s)", e)

    # Project to only the columns ProfilePCA needs — the combined parquet can be
    # tens of GB (2016-2020 ≈ 18 GB across 273 columns); loading all of it would
    # exhaust memory when we only touch ~54 profile/scalar columns.
    print(f"Loading {data_path}", flush=True)
    is_parquet = str(data_path).endswith('.parquet')
    if is_parquet:
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(data_path).names)
        needed = [c for c in avail
                  if any(c.startswith(p) for p in _PROFILE_GROUPS)]        # profile levels
        needed += [c for c in _SCALAR_PASSTHROUGH if c in avail]           # tropopause scalars
        needed += [c for c in ('sfc_type', 'sounding_id', 'date', 'orbit_id') if c in avail]
        print(f"  reading {len(needed)}/{len(avail)} columns", flush=True)
        df = pd.read_parquet(data_path, columns=needed)
    else:
        df = pd.read_csv(data_path)
    print(f"  rows: {len(df):,}", flush=True)

    # Apply the --co2-norm override to the CO2-prior group's per-row normalisation.
    groups = {p: dict(g) for p, g in _PROFILE_GROUPS.items()}
    groups['co2prior_sigma_']['norm'] = None if args.co2_norm == 'none' else args.co2_norm
    print(f"  CO2-prior profile norm: {groups['co2prior_sigma_']['norm']}", flush=True)

    surfaces = [args.sfc_type] if args.sfc_type is not None else [0, 1]
    for sfc in surfaces:
        _fit_one_surface(df, sfc, n_components, out_dir, scores_base, groups=groups)


if __name__ == '__main__':
    main()

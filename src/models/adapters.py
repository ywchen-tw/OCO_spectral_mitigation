"""Model adapters — uniform predict/save/load interface for the active XCO2 models.

Provides:
    ModelAdapter      — abstract base class
    TabMAdapter       — wraps TabM (BatchEnsemble quantile predictor)
    XGBoostAdapter    — wraps xgboost.XGBRegressor (point predictor, native .ubj format)

Usage (training script, after training):
    TabMAdapter(model, n_features=N, K=K, ...).save(output_dir)
    XGBoostAdapter(xgb_model, feature_names=features).save(output_dir)

Usage (inference):
    tabm_adapter = TabMAdapter.load(output_dir)
    xgb_adapter  = XGBoostAdapter.load(output_dir)
    y_hat        = xgb_adapter.predict(X)              # [N]
    q05, q50, q95 = tabm_adapter.predict_quantiles(X)
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Uniform interface for all bias-correction model types.

    Concrete subclasses wrap a trained model and expose:
    - predict(X)            — point prediction / q50  [N]
    - predict_quantiles(X)  — (q05, q50, q95) for probabilistic models
    - save(output_dir)      — persist weights + metadata
    - load(output_dir)      — reconstruct from disk (classmethod)
    - can_load(output_dir)  — True if all required files exist (classmethod)
    """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction or q50.  X is pipeline-transformed [N, n_features]."""

    def predict_quantiles(self,
                          X: np.ndarray) -> tuple:
        """Return (q05, q50, q95) each [N].  Raises for point-predictor models."""
        raise NotImplementedError(
            f"{self.__class__.__name__} is a point predictor — "
            "use predict() instead of predict_quantiles()."
        )

    @abstractmethod
    def save(self, output_dir) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, output_dir) -> 'ModelAdapter': ...

    @classmethod
    def can_load(cls, output_dir) -> bool:
        """Return True if all required files exist in output_dir."""
        return False   # subclasses override


# ─── Ridge adapter ─────────────────────────────────────────────────────────────


class TabMAdapter(ModelAdapter):
    """Wraps a TabM model (BatchEnsemble MLP with monotonic quantile head).

    model_tabm_best.pt is written by the training
    loop; this adapter saves tabm_meta.pkl with the architecture hyperparameters
    so the model can be reconstructed at load time.  ``can_load()`` checks
    ``arch_version`` so stale checkpoints fall through to a fresh training run.
    """

    CHECKPOINT_FILE = 'model_tabm_best.pt'
    META_FILE       = 'tabm_meta.pkl'
    _ARCH_VERSION   = 1

    def __init__(self, model, n_features: int, K: int = 16, d_model: int = 256,
                 n_layers: int = 4, dropout: float = 0.2,
                 feature_names: list | None = None,
                 feature_set: str = 'full', val_split: str = 'random',
                 aux_cloud: bool = False, n_cloud_classes: int = 1):
        self.model         = model
        self.n_features    = n_features
        self.K             = K
        self.d_model       = d_model
        self.n_layers      = n_layers
        self.dropout       = dropout
        self.feature_names = feature_names
        self.feature_set   = feature_set
        self.val_split     = val_split
        self.aux_cloud     = aux_cloud
        self.n_cloud_classes = n_cloud_classes

    def predict(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        return self.predict_quantiles(X, batch_size=batch_size)[1]

    def predict_quantiles(self, X: np.ndarray, batch_size: int = 8192) -> tuple:
        """Return (q05, q50, q95) each [N] (mean member quantiles)."""
        self.model.eval()
        device = next(self.model.parameters()).device
        q05, q50, q95 = [], [], []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                Xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
                out = self.model(Xb)
                if isinstance(out, dict):
                    out = out["quantiles"]
                out = out.cpu().numpy()
                q05.append(out[:, 0]); q50.append(out[:, 1]); q95.append(out[:, 2])
                del Xb
        return np.concatenate(q05), np.concatenate(q50), np.concatenate(q95)

    def save(self, output_dir) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            'n_features':      self.n_features,
            'K':               self.K,
            'd_model':         self.d_model,
            'n_layers':        self.n_layers,
            'dropout':         self.dropout,
            'feature_names':   self.feature_names,
            'feature_set':     self.feature_set,
            'val_split':       self.val_split,
            'aux_cloud':       self.aux_cloud,
            'n_cloud_classes': self.n_cloud_classes,
            'arch_version':    self._ARCH_VERSION,
        }
        with open(out / self.META_FILE, 'wb') as f:
            pickle.dump(meta, f)
        logger.info("TabMAdapter meta saved → %s", out / self.META_FILE)

    @classmethod
    def load(cls, output_dir, device: torch.device | None = None) -> 'TabMAdapter':
        out = Path(output_dir)
        meta_path = out / cls.META_FILE
        ckpt_path = out / cls.CHECKPOINT_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"TabMAdapter meta not found: {meta_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"TabMAdapter checkpoint not found: {ckpt_path}")

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        saved_ver = meta.get('arch_version', 1)
        if saved_ver != cls._ARCH_VERSION:
            raise RuntimeError(
                f"TabMAdapter checkpoint arch_version={saved_ver} does not match "
                f"current version={cls._ARCH_VERSION}.  Delete {out} and retrain."
            )

        device = device or torch.device('cpu')
        # Import here to avoid a circular import at module load time.
        import sys as _sys
        _main = _sys.modules.get('__main__')
        if _main is not None and hasattr(_main, 'TabM'):
            TabM = _main.TabM
        else:
            from .tabm import TabM
        model = TabM(
            n_features=meta['n_features'], K=meta['K'], d_model=meta['d_model'],
            n_layers=meta['n_layers'], dropout=meta['dropout'],
            aux_cloud=meta.get('aux_cloud', False),
            n_cloud_classes=meta.get('n_cloud_classes', 1),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        logger.info("TabMAdapter loaded ← %s (epoch=%s, val_loss=%s)",
                    out, ckpt.get('epoch', '?'), ckpt.get('val_loss', '?'))
        meta.pop('arch_version', None)
        return cls(model=model, **meta)

    @classmethod
    def can_load(cls, output_dir) -> bool:
        out = Path(output_dir)
        if not ((out / cls.META_FILE).exists() and (out / cls.CHECKPOINT_FILE).exists()):
            return False
        try:
            with open(out / cls.META_FILE, 'rb') as f:
                meta = pickle.load(f)
            if meta.get('arch_version', 1) != cls._ARCH_VERSION:
                logger.warning(
                    "TabMAdapter checkpoint at %s has arch_version=%s (current=%s) — "
                    "will retrain from scratch.",
                    out, meta.get('arch_version', 1), cls._ARCH_VERSION,
                )
                return False
        except Exception:
            return False
        return True


# ─── XGBoost adapter ───────────────────────────────────────────────────────────

class XGBoostAdapter(ModelAdapter):
    """Wraps a fitted xgboost.XGBRegressor saved in native binary format (.ubj)."""

    WEIGHTS_FILE = 'xgb_model.ubj'
    META_FILE    = 'xgb_meta.pkl'

    def __init__(self, model, feature_names=None):
        self.model         = model
        self.feature_names = list(feature_names) if feature_names is not None else None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, output_dir) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out / self.WEIGHTS_FILE))
        meta = {'feature_names': self.feature_names}
        with open(out / self.META_FILE, 'wb') as f:
            pickle.dump(meta, f)
        logger.info("XGBoostAdapter saved → %s", out)

    @classmethod
    def load(cls, output_dir) -> 'XGBoostAdapter':
        import xgboost as xgb
        out          = Path(output_dir)
        weights_path = out / cls.WEIGHTS_FILE
        meta_path    = out / cls.META_FILE
        if not weights_path.exists():
            raise FileNotFoundError(f"XGBoostAdapter weights not found: {weights_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"XGBoostAdapter meta not found: {meta_path}")
        model = xgb.XGBRegressor()
        model.load_model(str(weights_path))
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        logger.info("XGBoostAdapter loaded ← %s", out)
        return cls(model=model, feature_names=meta.get('feature_names'))

    @classmethod
    def can_load(cls, output_dir) -> bool:
        out = Path(output_dir)
        return (out / cls.WEIGHTS_FILE).exists() and (out / cls.META_FILE).exists()


# ─── Hybrid Dual-Tower adapter ─────────────────────────────────────────────────

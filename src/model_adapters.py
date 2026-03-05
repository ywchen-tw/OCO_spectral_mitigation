"""Model adapters — uniform predict/save/load interface for all XCO2 bias-correction models.

Provides:
    ModelAdapter      — abstract base class
    RidgeAdapter      — wraps sklearn Ridge (point predictor)
    MLPAdapter        — wraps residual MLP with target normalisation (point predictor)
    FTAdapter         — wraps UncertainFTTransformerRefined (quantile predictor)
    XGBoostAdapter    — wraps xgboost.XGBRegressor (point predictor, native .ubj format)
    HybridAdapter     — wraps HybridDualTower (quantile predictor)

Also defines _ResBlock and _MLP at module level (extracted from the nested
definitions that previously lived inside mitigation_test() in mlp_lr_models.py).

Usage (training script, after training):
    RidgeAdapter(ridge_model).save(output_dir)
    MLPAdapter(mlp_model, y_mean, y_std).save(output_dir)
    FTAdapter(ft_model, n_features=N, d_token=256, n_heads=8,
              n_layers=4, d_ff=256).save(output_dir)
    XGBoostAdapter(xgb_model, feature_names=features).save(output_dir)
    HybridAdapter(hybrid_model, n_features=N, d_token=128, n_heads=8,
                  n_layers=4, d_ff=256, mlp_hidden=256, fusion_dim=128).save(output_dir)

Usage (inference):
    ridge_adapter = RidgeAdapter.load(output_dir)
    mlp_adapter   = MLPAdapter.load(output_dir)
    ft_adapter    = FTAdapter.load(output_dir)
    xgb_adapter   = XGBoostAdapter.load(output_dir)
    hybrid_adapter = HybridAdapter.load(output_dir)

    y_hat    = ridge_adapter.predict(X)         # [N]
    q05, q50, q95 = ft_adapter.predict_quantiles(X)
    q05, q50, q95 = hybrid_adapter.predict_quantiles(X)
"""

import copy
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─── MLP architecture (module-level — previously nested in mitigation_test()) ──

class _ResBlock(nn.Module):
    """Pre-activation residual block: LN→GELU→Linear→LN→GELU→Dropout→Linear + skip."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _MLP(nn.Module):
    """Residual MLP for XCO2 anomaly regression.

    Architecture: Linear projection → N × ResBlock → Linear head.
    Target is assumed to be normalised (median-centred, IQR-scaled) during
    training; MLPAdapter handles denormalisation at inference time.
    """

    def __init__(self, n: int, d_model: int = 256,
                 n_blocks: int = 3, dropout: float = 0.25):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.blocks = nn.ModuleList([_ResBlock(d_model, dropout) for _ in range(n_blocks)])
        self.head   = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# ─── Abstract base ─────────────────────────────────────────────────────────────

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

class RidgeAdapter(ModelAdapter):
    """Wraps a fitted sklearn Ridge (or any estimator with .predict())."""

    FILENAME = 'ridge_model.pkl'

    def __init__(self, model):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, output_dir) -> None:
        path = Path(output_dir) / self.FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info("RidgeAdapter saved → %s", path)

    @classmethod
    def load(cls, output_dir) -> 'RidgeAdapter':
        path = Path(output_dir) / cls.FILENAME
        if not path.exists():
            raise FileNotFoundError(f"RidgeAdapter checkpoint not found: {path}")
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info("RidgeAdapter loaded ← %s", path)
        return obj

    @classmethod
    def can_load(cls, output_dir) -> bool:
        return (Path(output_dir) / cls.FILENAME).exists()


# ─── MLP adapter ───────────────────────────────────────────────────────────────

class MLPAdapter(ModelAdapter):
    """Wraps a trained _MLP with target normalisation stats."""

    WEIGHTS_FILE = 'mlp_weights.pt'
    META_FILE    = 'mlp_meta.pkl'

    def __init__(self, model: _MLP, y_mean: float, y_std: float,
                 device: torch.device | None = None):
        self.model  = model
        self.y_mean = float(y_mean)
        self.y_std  = float(y_std)
        self.device = device or torch.device('cpu')

    def predict(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Batched inference; returns predictions in original (ppm) scale."""
        self.model.eval()
        out = []
        for start in range(0, len(X), batch_size):
            Xb = torch.tensor(X[start:start + batch_size],
                              dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out.append(self.model(Xb).cpu().numpy())
            del Xb
        return np.concatenate(out) * self.y_std + self.y_mean

    def save(self, output_dir) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Weights (CPU copy for device-agnostic loading)
        torch.save(self.model.cpu().state_dict(), out / self.WEIGHTS_FILE)
        self.model.to(self.device)   # restore

        # Metadata needed to reconstruct _MLP
        meta = {
            'y_mean':      self.y_mean,
            'y_std':       self.y_std,
            'n_in':        next(iter(self.model.input_proj.children())).in_features,
            'd_model':     next(iter(self.model.input_proj.children())).out_features,
            'n_blocks':    len(self.model.blocks),
            'dropout':     self.model.blocks[0].block[5].p if len(self.model.blocks) > 0 else 0.2,
        }
        with open(out / self.META_FILE, 'wb') as f:
            pickle.dump(meta, f)
        logger.info("MLPAdapter saved → %s", out)

    @classmethod
    def load(cls, output_dir, device: torch.device | None = None) -> 'MLPAdapter':
        out = Path(output_dir)
        meta_path    = out / cls.META_FILE
        weights_path = out / cls.WEIGHTS_FILE

        if not meta_path.exists():
            raise FileNotFoundError(f"MLPAdapter meta not found: {meta_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"MLPAdapter weights not found: {weights_path}")

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        device = device or torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else 'cpu'
        )
        model = _MLP(n=meta['n_in'], d_model=meta['d_model'],
                     n_blocks=meta['n_blocks'], dropout=meta['dropout']).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        logger.info("MLPAdapter loaded ← %s", out)
        return cls(model=model, y_mean=meta['y_mean'], y_std=meta['y_std'], device=device)

    @classmethod
    def can_load(cls, output_dir) -> bool:
        out = Path(output_dir)
        return (out / cls.META_FILE).exists() and (out / cls.WEIGHTS_FILE).exists()


# ─── FT-Transformer adapter ────────────────────────────────────────────────────

class FTAdapter(ModelAdapter):
    """Wraps UncertainFTTransformerRefined — supports predict() and predict_quantiles().

    The model_best.pt checkpoint format is **unchanged** from the training
    script.  FTAdapter additionally saves ft_meta.pkl with the architecture
    hyperparameters so the model can be reconstructed without re-specifying
    them at load time.

    ``_ARCH_VERSION`` must be incremented whenever the model's state_dict key
    structure changes (e.g. after refactoring AdvancedTransformerBlock).
    Stale checkpoints are detected by ``can_load()`` via the saved version tag,
    causing the caller to fall through to a fresh training run automatically.
    """

    CHECKPOINT_FILE = 'model_best.pt'
    META_FILE       = 'ft_meta.pkl'
    _ARCH_VERSION   = 3   # bumped: added segment embeddings (group_emb + feature_to_group buffer)

    def __init__(self, model, n_features: int, d_token: int = 128,
                 n_heads: int = 8, n_layers: int = 4, d_ff: int = 256,
                 tokenizer_type: str = 'mlp',
                 feature_names: list | None = None):
        self.model          = model
        self.n_features     = n_features
        self.d_token        = d_token
        self.n_heads        = n_heads
        self.n_layers       = n_layers
        self.d_ff           = d_ff
        self.tokenizer_type = tokenizer_type
        self.feature_names  = feature_names

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Return q50 predictions [N]."""
        return self.predict_quantiles(X, batch_size=batch_size)[1]

    def predict_quantiles(self, X: np.ndarray,
                          batch_size: int = 512) -> tuple:
        """Return (q05, q50, q95) each [N]."""
        self.model.eval()
        q05_list, q50_list, q95_list = [], [], []
        for start in range(0, len(X), batch_size):
            Xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32)
            with torch.no_grad():
                preds = self.model(Xb)   # [batch, 3]
            q05_list.append(preds[:, 0].numpy())
            q50_list.append(preds[:, 1].numpy())
            q95_list.append(preds[:, 2].numpy())
            del Xb, preds
        return (np.concatenate(q05_list),
                np.concatenate(q50_list),
                np.concatenate(q95_list))

    def save(self, output_dir) -> None:
        """Save ft_meta.pkl.  model_best.pt is written by the training loop."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            'n_features':     self.n_features,
            'd_token':        self.d_token,
            'n_heads':        self.n_heads,
            'n_layers':       self.n_layers,
            'd_ff':           self.d_ff,
            'tokenizer_type': self.tokenizer_type,
            'feature_names':  self.feature_names,
            'arch_version':   self._ARCH_VERSION,
        }
        with open(out / self.META_FILE, 'wb') as f:
            pickle.dump(meta, f)
        logger.info("FTAdapter meta saved → %s", out / self.META_FILE)

    @classmethod
    def load(cls, output_dir, device: torch.device | None = None) -> 'FTAdapter':
        out = Path(output_dir)
        meta_path = out / cls.META_FILE
        ckpt_path = out / cls.CHECKPOINT_FILE

        if not meta_path.exists():
            raise FileNotFoundError(f"FTAdapter meta not found: {meta_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"FTAdapter checkpoint not found: {ckpt_path}")

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        saved_ver = meta.get('arch_version', 1)
        if saved_ver != cls._ARCH_VERSION:
            raise RuntimeError(
                f"FTAdapter checkpoint arch_version={saved_ver} does not match "
                f"current version={cls._ARCH_VERSION}.  "
                f"Delete {out} and retrain to generate a compatible checkpoint."
            )

        device = device or torch.device('cpu')

        # Import here to avoid circular import at module load time.
        # Prefer __main__ when models_transformer.py is run directly so that
        # AdvancedTransformerBlock loaded here is the same class object used
        # in plot_attention_map's isinstance() check (avoids module duplication).
        import sys as _sys
        _main = _sys.modules.get('__main__')
        if _main is not None and hasattr(_main, 'UncertainFTTransformerRefined'):
            UncertainFTTransformerRefined = _main.UncertainFTTransformerRefined
        else:
            from models_transformer import UncertainFTTransformerRefined
        model = UncertainFTTransformerRefined(
            n_features=meta['n_features'],
            d_token=meta['d_token'],
            n_heads=meta['n_heads'],
            n_layers=meta['n_layers'],
            d_ff=meta['d_ff'],
            tokenizer_type=meta.get('tokenizer_type', 'linear'),  # 'linear' for old checkpoints
            feature_names=meta.get('feature_names'),
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        logger.info(
            "FTAdapter loaded ← %s  (epoch=%s, val_loss=%s)",
            out, ckpt.get('epoch', '?'), ckpt.get('val_loss', '?')
        )
        meta.pop('arch_version', None)   # not a constructor arg
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
                    "FTAdapter checkpoint at %s has arch_version=%s (current=%s) — "
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

class HybridAdapter(ModelAdapter):
    """Wraps HybridDualTower — supports predict() and predict_quantiles().

    Files written by save():
        model_hybrid_best.pt  — model weights (written by training loop)
        hybrid_meta.pkl       — architecture hyperparameters for reconstruction

    ``_ARCH_VERSION`` must be incremented whenever the HybridDualTower
    state_dict key structure changes (e.g. after refactoring _MLPBranch).
    Stale checkpoints are detected by ``can_load()`` causing the caller to
    fall through to a fresh training run automatically.
    """

    CHECKPOINT_FILE = 'model_hybrid_best.pt'
    META_FILE       = 'hybrid_meta.pkl'
    _ARCH_VERSION   = 1

    def __init__(self, model, n_features: int,
                 d_token: int = 128, n_heads: int = 8, n_layers: int = 4,
                 d_ff: int = 256, mlp_hidden: int = 256, fusion_dim: int = 128,
                 tokenizer_type: str = 'mlp',
                 feature_names: list | None = None):
        self.model          = model
        self.n_features     = n_features
        self.d_token        = d_token
        self.n_heads        = n_heads
        self.n_layers       = n_layers
        self.d_ff           = d_ff
        self.mlp_hidden     = mlp_hidden
        self.fusion_dim     = fusion_dim
        self.tokenizer_type = tokenizer_type
        self.feature_names  = feature_names

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Return q50 predictions [N]."""
        return self.predict_quantiles(X, batch_size=batch_size)[1]

    def predict_quantiles(self, X: np.ndarray,
                          batch_size: int = 512) -> tuple:
        """Return (q05, q50, q95) each [N]."""
        self.model.eval()
        q05_list, q50_list, q95_list = [], [], []
        for start in range(0, len(X), batch_size):
            Xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32)
            with torch.no_grad():
                preds = self.model(Xb)   # [batch, 3]
            q05_list.append(preds[:, 0].numpy())
            q50_list.append(preds[:, 1].numpy())
            q95_list.append(preds[:, 2].numpy())
            del Xb, preds
        return (np.concatenate(q05_list),
                np.concatenate(q50_list),
                np.concatenate(q95_list))

    def save(self, output_dir) -> None:
        """Save hybrid_meta.pkl.  model_hybrid_best.pt is written by the training loop."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            'n_features':     self.n_features,
            'd_token':        self.d_token,
            'n_heads':        self.n_heads,
            'n_layers':       self.n_layers,
            'd_ff':           self.d_ff,
            'mlp_hidden':     self.mlp_hidden,
            'fusion_dim':     self.fusion_dim,
            'tokenizer_type': self.tokenizer_type,
            'feature_names':  self.feature_names,
            'arch_version':   self._ARCH_VERSION,
        }
        with open(out / self.META_FILE, 'wb') as f:
            pickle.dump(meta, f)
        logger.info("HybridAdapter meta saved → %s", out / self.META_FILE)

    @classmethod
    def load(cls, output_dir, device: torch.device | None = None) -> 'HybridAdapter':
        out       = Path(output_dir)
        meta_path = out / cls.META_FILE
        ckpt_path = out / cls.CHECKPOINT_FILE

        if not meta_path.exists():
            raise FileNotFoundError(f"HybridAdapter meta not found: {meta_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"HybridAdapter checkpoint not found: {ckpt_path}")

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        saved_ver = meta.get('arch_version', 1)
        if saved_ver != cls._ARCH_VERSION:
            raise RuntimeError(
                f"HybridAdapter checkpoint arch_version={saved_ver} does not match "
                f"current version={cls._ARCH_VERSION}.  "
                f"Delete {out} and retrain to generate a compatible checkpoint."
            )

        device = device or torch.device('cpu')

        from models_hybrid import HybridDualTower
        model = HybridDualTower(
            n_features=meta['n_features'],
            d_token=meta['d_token'],
            n_heads=meta['n_heads'],
            n_layers=meta['n_layers'],
            d_ff=meta['d_ff'],
            mlp_hidden=meta['mlp_hidden'],
            fusion_dim=meta['fusion_dim'],
            tokenizer_type=meta.get('tokenizer_type', 'mlp'),
            feature_names=meta.get('feature_names'),
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        logger.info(
            "HybridAdapter loaded ← %s  (epoch=%s, val_loss=%s)",
            out, ckpt.get('epoch', '?'), ckpt.get('val_loss', '?')
        )
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
                    "HybridAdapter checkpoint at %s has arch_version=%s (current=%s) — "
                    "will retrain from scratch.",
                    out, meta.get('arch_version', 1), cls._ARCH_VERSION,
                )
                return False
        except Exception:
            return False
        return True

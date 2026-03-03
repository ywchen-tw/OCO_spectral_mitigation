import argparse
import h5py
import numpy as np
import pandas as pd
import os
import sys
import platform
import matplotlib.pyplot as plt
from matplotlib import colors
from pipeline import FeaturePipeline, _ensure_derived_features
from model_adapters import FTAdapter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import seaborn as sns
import copy
import gc
from tqdm import tqdm
import glob
import logging
from pathlib import Path
from datetime import datetime
from utils import get_storage_dir

logger = logging.getLogger(__name__)

# ── Feature group definitions ─────────────────────────────────────────────────
# Single source of truth for both segment embeddings (UncertainFTTransformerRefined)
# and the attention heatmap (plot_attention_map).  Keys are display-formatted strings
# (with \n for the heatmap axis labels).  Features absent from the active feature set
# are silently ignored wherever this dict is consumed.
_FEATURE_GROUPS: dict = {
    'Spectral\nk-coeff':  ['o2a_k1', 'o2a_k2', 'wco2_k1', 'wco2_k2', 'sco2_k1', 'sco2_k2',
                           'o2a_intercept', 'wco2_intercept', 'sco2_intercept'],
    'Geometry':           ['mu_sza', 'mu_vza', 'cos_glint_angle', 'pol_ang_rad',
                           'sin_raa', 'cos_raa'],   # sin_raa/cos_raa: sfc_type=1 only
    'Pressure\n& Meteo':  ['log_P', 'dp', 'h2o_scale', 'delT', 'co2_grad_del',
                           'ws', 's31', 's32', 'airmass_sq'],
    'Surface\nAlbedo':    ['alb_o2a', 'alb_wco2', 'alb_sco2',
                           'alb_o2a_over_cos_sza', 'alb_wco2_over_cos_sza', 'alb_sco2_over_cos_sza'],
    'CO\u2082\nRetrieval': ['xco2_raw_minus_apriori', 
                            'xco2_bc_minus_raw', 
                            'co2_ratio_bc', 'h2o_ratio_bc',
                           'xco2_strong_idp', 'xco2_weak_idp'],
    'SNR &\nDetector':    ['csnr_o2a', 'csnr_wco2', 'csnr_sco2',
                           'snr_o2a', 'snr_wco2', 'snr_sco2',
                           'h_cont_o2a', 'h_cont_wco2', 'h_cont_sco2',
                           'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2'],
    'AOD &\nAerosol':     ['aod_total'],
    'Footprint':          [f'fp_{i}' for i in range(8)],
}


def _build_feature_to_group(feature_names: list,
                             groups: dict | None = None) -> torch.Tensor:
    """Map each feature name to an integer group index (0-based, insertion order).

    Features not found in any group are assigned index 0 with a warning — this
    should not occur for any feature set defined in pipeline.py.

    Parameters
    ----------
    feature_names : list[str]  ordered list of all model input features
    groups        : dict  feature-group mapping (defaults to _FEATURE_GROUPS)

    Returns
    -------
    torch.Tensor  shape [n_features]  dtype long
    """
    if groups is None:
        groups = _FEATURE_GROUPS
    feat_to_group: dict = {}
    for g_idx, feat_list in enumerate(groups.values()):
        for f in feat_list:
            feat_to_group[f] = g_idx
    ids = []
    for f in feature_names:
        if f in feat_to_group:
            ids.append(feat_to_group[f])
        else:
            logger.warning("_build_feature_to_group: feature '%s' not in any group; assigned to group 0.", f)
            ids.append(0)
    return torch.tensor(ids, dtype=torch.long)


def apply_trig_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trigonometric and log-space features from raw OCO-2 angle columns.

    NOTE: When loading from CSV (produced by fitting_data_correction.py) these
    columns are already present — this function is only needed when building
    features directly from raw sounding data (e.g. MultiDateAtmosphericDataset).

    Expected raw columns: sza, vza, saa, vaa, glint_angle, psurf
    Produces: mu_sza, mu_vza, sin_raa, cos_raa, cos_glint_angle,
              glint_prox, cos_theta, Phi_cos_theta, R_rs_factor,
              log_P, dp_psfc_ratio, pol_ang_rad
    """
    df = df.copy()

    sza = df['sza'].to_numpy(dtype=float)
    vza = df['vza'].to_numpy(dtype=float)
    saa = df['saa'].to_numpy(dtype=float)
    vaa = df['vaa'].to_numpy(dtype=float)
    glt = df['glint_angle'].to_numpy(dtype=float)
    psurf = df['psurf'].to_numpy(dtype=float)

    # Relative azimuth angle (0–180°)
    raa = 180.0 - np.abs((saa - vaa) % 360.0 - 180.0)

    mu_sza  = np.cos(np.radians(sza))
    mu_vza  = np.cos(np.radians(vza))
    sin_sza = np.sin(np.radians(sza))
    sin_vza = np.sin(np.radians(vza))
    cos_raa = np.cos(np.radians(raa))
    sin_raa = np.sin(np.radians(raa))

    # Scattering angle — backscattering convention (matches fitting_data_correction.py)
    cos_theta = -mu_sza * mu_vza + sin_sza * sin_vza * cos_raa

    # Rayleigh phase function
    Phi_cos_theta = (3.0 / 4.0) * (1.0 + cos_theta ** 2)

    # Normalised reflectance factor
    R_rs_factor = Phi_cos_theta / (4.0 * np.clip(mu_sza * mu_vza, 1e-9, None))

    # Glint proximity: exponential decay with 10° length scale
    cos_glint  = np.cos(np.radians(glt))
    glint_prox = np.exp(-glt / 10.0)

    # Log base-10 of surface pressure (matches fitting_data_correction.py)
    log_P = np.log10(psurf)

    df['mu_sza']          = mu_sza
    df['mu_vza']          = mu_vza
    df['sin_raa']         = sin_raa
    df['cos_raa']         = cos_raa
    df['cos_glint_angle'] = cos_glint
    df['glint_prox']      = glint_prox
    df['cos_theta']       = cos_theta
    df['Phi_cos_theta']   = Phi_cos_theta
    df['R_rs_factor']     = R_rs_factor
    df['log_P']           = log_P
    if 'dp' in df.columns:
        df['dp_psfc_ratio'] = df['dp'].to_numpy(dtype=float) / np.clip(psurf, 1e-9, None)
    if 'pol_angle' in df.columns:
        df['pol_ang_rad'] = np.radians(df['pol_angle'].to_numpy(dtype=float))

    return df


class MultiDateAtmosphericDataset(Dataset):
    def __init__(self, file_paths, qt_transformer, features):
        self.file_paths = sorted(file_paths)
        self.qt = qt_transformer  # Pre-fitted QuantileTransformer
        self.features = features
        
        # We pre-calculate the cumulative size to map index -> file.
        # Parquet files store row count in metadata (O(1)); CSV needs a column read.
        self.file_sizes = []
        self._is_parquet = [str(f).endswith('.parquet') for f in self.file_paths]
        self._cache: dict = {}   # {file_idx: pd.DataFrame} for parquet files
        for f, is_pq in zip(self.file_paths, self._is_parquet):
            if is_pq:
                import pyarrow.parquet as pq
                self.file_sizes.append(pq.read_metadata(f).num_rows)
            else:
                self.file_sizes.append(len(pd.read_csv(f, usecols=[features[0]])))

        self.cumulative_sizes = np.cumsum(self.file_sizes)
        self.total_size = self.cumulative_sizes[-1]

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # 1. Map flat index → file + row-within-file
        file_idx  = int(np.searchsorted(self.cumulative_sizes, idx, side='right'))
        start_idx = int(self.cumulative_sizes[file_idx - 1]) if file_idx > 0 else 0
        row_in_file = idx - start_idx

        # 2. Read row: parquet uses cached DataFrame; CSV uses skiprows (lazy I/O)
        if self._is_parquet[file_idx]:
            if file_idx not in self._cache:
                self._cache[file_idx] = pd.read_parquet(self.file_paths[file_idx])
            df = self._cache[file_idx].iloc[[row_in_file]].reset_index(drop=True)
        else:
            df = pd.read_csv(
                self.file_paths[file_idx],
                skiprows=range(1, row_in_file + 1),
                nrows=1,
            )

        # 3. Trigonometric transforms (for raw CSV that hasn't been pre-processed)
        df = apply_trig_transforms(df)

        # 4. Quantile-normal transform using the pre-fitted transformer
        x_scaled = self.qt.transform(df[self.features].to_numpy(dtype=float))

        x = torch.tensor(x_scaled[0], dtype=torch.float32)
        y = torch.tensor(float(df['bias'].iloc[0]), dtype=torch.float32)
        return x, y

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        # Each feature gets its own learnable weight and bias
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x shape: [batch, n_features]
        # x.unsqueeze(-1) -> [batch, n_features, 1]
        # Multiplication uses broadcasting: [batch, n_features, d_token]
        return x.unsqueeze(-1) * self.weight + self.bias


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.skip_connection = nn.Linear(input_size, output_size)
        self.dense = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        residual = self.skip_connection(x)
        gated_output = self.gate(x) * self.dense(x)
        return self.norm(residual + gated_output)


class MLPTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.tokenizers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_token // 2),
                nn.GELU(),
                nn.Linear(d_token // 2, d_token)
            ) for _ in range(n_features)
        ])

    def forward(self, x):
        # x: [batch, n_features]
        tokens = [self.tokenizers[i](x[:, i:i+1]) for i in range(len(self.tokenizers))]
        return torch.stack(tokens, dim=1)   # [batch, n_features, d_token]


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(self.norm(x), **kwargs)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic Depth / DropPath per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class AdvancedTransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, d_ff, dropout=0.2, drop_path_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.drop_path_rate = drop_path_rate

        # GRN replaces the plain FFN: per-sample gating suppresses uninformative
        # tokens (e.g. saturated O2-A band under heavy aerosol) while the skip
        # connection preserves gradient flow through the residual stream.
        self.ff = GatedResidualNetwork(
            input_size=d_token, hidden_size=d_ff, output_size=d_token, dropout=dropout,
        )

    def forward(self, x, training=False):
        # Pre-Norm + Attention + DropPath
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + drop_path(attn_out, self.drop_path_rate, training)

        # Pre-Norm + Feed-Forward + DropPath
        x = x + drop_path(self.ff(self.norm2(x)), self.drop_path_rate, training)
        return x


class UncertainFTTransformerRefined(nn.Module):
    """FT-Transformer with [CLS] token aggregation and a GRN regression head.

    A learnable [CLS] token is prepended to the feature token sequence and
    interacts with all feature tokens via self-attention.  After the last
    Transformer block, only the [CLS] token representation is passed to the
    head — it has learned to collect the bias-relevant global atmospheric state.

    Outputs 3 quantiles [q05, q50, q95] compatible with ``quantile_loss``.
    Supports ``return_attn=True`` for compatibility with ``plot_attention_map``.
    """

    def __init__(self, n_features, d_token=128, n_heads=8, n_layers=4, d_ff=512,
                 tokenizer_type: str = 'mlp', drop_path_rate: float = 0.1,
                 feature_names: list | None = None):
        super().__init__()
        self.n_features = n_features
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == 'mlp':
            self.tokenizer = MLPTokenizer(n_features, d_token)
        else:
            self.tokenizer = FeatureTokenizer(n_features, d_token)

        # Segment embeddings — one learnable d_token vector per domain group,
        # summed into each feature token after tokenisation (BERT segment-embedding
        # analogue).  Provides a hierarchical inductive bias: features sharing a
        # physical context (e.g. o2a_k1/o2a_k2/wco2_k1 all in Spectral k-coeff)
        # receive a common additive shift, encouraging the model to treat them as
        # a coherent group rather than independent scalar tokens.
        if feature_names is not None:
            n_groups = len(_FEATURE_GROUPS)
            self.group_emb = nn.Embedding(n_groups, d_token)
            nn.init.trunc_normal_(self.group_emb.weight, std=0.02)
            self.register_buffer(
                'feature_to_group',
                _build_feature_to_group(list(feature_names)),
            )
        else:
            self.group_emb = None

        # Learnable [CLS] token — aggregates global atmospheric state
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        # Stochastic depth: scale drop rate linearly across layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.layers = nn.ModuleList([
            AdvancedTransformerBlock(d_token, n_heads, d_ff, drop_path_rate=dpr[i])
            for i in range(n_layers)
        ])

        # GRN on the [CLS] representation, then a plain linear to 3 quantile outputs.
        # The final nn.Linear has no normalisation so the regression scale is unbounded.
        self.head = nn.Sequential(
            GatedResidualNetwork(
                input_size=d_token, hidden_size=d_token,
                output_size=d_token // 2, dropout=0.1,
            ),
            nn.Linear(d_token // 2, 3),
        )

    def forward(self, x, return_attn: bool = False):
        # x: [batch, n_features]
        x = self.tokenizer(x)                          # [batch, n_features, d_token]

        # Add group embedding: shifts each feature token by a learnable group-context
        # vector.  group_emb(feature_to_group) is [n_features, d_token]; the unsqueeze
        # makes it [1, n_features, d_token] so it broadcasts across the batch dimension.
        if self.group_emb is not None:
            x = x + self.group_emb(self.feature_to_group).unsqueeze(0)

        # Prepend [CLS] token
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)          # [batch, n_features+1, d_token]

        last_attn = None
        for layer in self.layers:
            if return_attn:
                x_norm = layer.norm1(x)
                attn_out, last_attn = layer.attn(x_norm, x_norm, x_norm)
                x = x + attn_out
                x = x + layer.ff(layer.norm2(x))
            else:
                x = layer(x, training=self.training)

        # Extract the [CLS] token (index 0) as the global representation
        x_global = x[:, 0]                             # [batch, d_token]
        out = self.head(x_global)
        if return_attn:
            return out, last_attn
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_token, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_token)
        )

    def forward(self, x):
        # Self-Attention with Residual Connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with Residual Connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class TransformerBlockWithExtraction(nn.Module):
    def __init__(self, d_token, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Set need_weights=True to extract the attention matrix
        self.attention = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_token, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_token)
        )

    def forward(self, x, return_attn=False):
        # attn_weights shape: [batch, n_features, n_features]
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        
        if return_attn:
            return x, attn_weights
        return x    
    
class BiasPredictorFT(nn.Module):
    def __init__(self, n_features, d_token=64, n_heads=8, n_layers=3, d_ff=128):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(d_token * n_features, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1) # Single output: the predicted bias
        )

    def forward(self, x):
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x)
        
        # Flatten [batch, n_features, d_token] to [batch, n_features * d_token]
        x = x.flatten(1)
        return self.head(x).squeeze(-1)
    
class UncertainBiasPredictorFT(nn.Module):
    """FT-Transformer with a 3-way quantile head [q05, q50, q95].

    Uses TransformerBlockWithExtraction for every layer so that
    attention maps can be retrieved at inference time via return_attn=True.
    """
    def __init__(self, n_features, d_token=64, n_heads=8, n_layers=3, d_ff=128):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.layers = nn.ModuleList([
            TransformerBlockWithExtraction(d_token, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_token * n_features, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 3)  # [q05, q50, q95]
        )

    def forward(self, x, return_attn: bool = False):
        x = self.tokenizer(x)
        last_attn = None
        for layer in self.layers:
            if return_attn:
                x, last_attn = layer(x, return_attn=True)
            else:
                x = layer(x)
        out = self.head(x.flatten(1))  # [batch, 3]
        if return_attn:
            return out, last_attn
        return out
    
def quantile_loss(preds, targets, quantiles):
    """Pinball / quantile loss for all three outputs.

    preds: [batch, 3]
    targets: [batch]
    quantiles: [0.05, 0.5, 0.95]
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i]
        # Pinball loss formula
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss.unsqueeze(-1))

    return torch.mean(torch.cat(losses, dim=-1))


def huber_pinball_loss(preds, targets, quantiles, delta: float = 1.0):
    """Combined loss: Huber for q50 (index 1), pinball for q05/q95 (indices 0, 2).

    Using Huber for the median makes the point-prediction robust to the heavy
    left tail of cloud-affected XCO2 anomalies (which can reach −3 ppm) while
    still behaving like MSE for typical clear-sky anomalies near 0.  The q05/q95
    outputs retain their pinball loss so the uncertainty intervals stay calibrated.

    preds      : [batch, 3]
    targets    : [batch]
    quantiles  : [0.05, 0.5, 0.95]
    delta      : Huber transition point (ppm).  Errors with |e| ≤ delta are
                 penalised quadratically; larger errors linearly.
                 Recommended range for xco2_bc_anomaly: 0.5–1.0 ppm.
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i]
        if q == 0.5:
            # Huber loss: quadratic for |e| ≤ delta, linear beyond
            loss = torch.where(
                errors.abs() <= delta,
                0.5 * errors ** 2,
                delta * (errors.abs() - 0.5 * delta),
            )
        else:
            # Pinball loss for q05 / q95 — unchanged
            loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss.unsqueeze(-1))

    return torch.mean(torch.cat(losses, dim=-1))


# Training setup
quantiles = [0.05, 0.5, 0.95]


def _batched_predict(model, X_np, batch_size=2048):
    """Run model inference in batches to avoid OOM from the [N, n_feat*d_token] flatten.

    Parameters
    ----------
    model : nn.Module  (on CPU after training)
    X_np  : np.ndarray  [N, n_features]  float32
    batch_size : int  rows per forward pass

    Returns
    -------
    preds : np.ndarray  [N, 3]  (q05, q50, q95) as float32
    """
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            Xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32)
            parts.append(model(Xb).numpy())
            del Xb
    return np.concatenate(parts, axis=0)


def plot_uncertainty_by_feature(model, X_val, feature_values, feature_name, output_dir):
    """Scatter uncertainty (q95-q05 width) against any feature.

    Parameters
    ----------
    model : UncertainBiasPredictorFT
    X_val : np.ndarray  [N, n_features]  QT-transformed test set
    feature_values : np.ndarray  [N]  values to plot on the x-axis
        Typically extracted from X_val via ``X_val[:, features.index(name)]``.
        Values are QT-transformed — used for ordering/pattern detection only.
    feature_name : str
        Human-readable name used for axis label and output filename.
    output_dir : str
    """
    preds = _batched_predict(model, np.asarray(X_val, dtype=np.float32))
    q05, q50, q95 = preds[:, 0], preds[:, 1], preds[:, 2]
    uncertainty = q95 - q05

    q50_np = q50
    lower_err = np.clip(q50_np - q05, 0, None)
    upper_err = np.clip(q95 - q50_np, 0, None)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(feature_values, q50_np, yerr=[lower_err, upper_err],
                fmt='o', alpha=0.3, markersize=4, ecolor='gray', capsize=2)
    ax.set_xlabel(f"{feature_name} (QT-transformed)")
    ax.set_ylabel("Predicted bias (q50) with uncertainty (q05/q95)")
    ax.set_title(f"Model Confidence vs. {feature_name}")
    safe_name = feature_name.replace(' ', '_').replace('/', '_')
    fig.savefig(os.path.join(output_dir, f"uncertainty_vs_{safe_name}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved uncertainty_vs_%s.png", safe_name)

def plot_attention_map(model, sample_batch, feature_names, output_dir, top_k=20):
    """Generate three attention visualisations suited to high-dimensional feature sets.

    Parameters
    ----------
    sample_batch : torch.Tensor  shape [N, n_features]
        Pass ≥50 samples so that averaging suppresses sample-specific noise.
    feature_names : list[str]
    output_dir : str
    top_k : int
        Number of top features to show in the bar-chart and per-head panels.

    Outputs
    -------
    attention_top_features.png   – horizontal bar chart of top-k most attended features
    attention_group_heatmap.png  – 8×8 domain-group block heatmap
    attention_per_head.png       – one panel per attention head (top-10 features)
    """
    model.eval()
    feature_names = list(feature_names)
    n_samples = sample_batch.shape[0]

    # ── Collect averaged attention over sample_batch (last layer, heads averaged) ─
    with torch.no_grad():
        _, attn_all = model(sample_batch, return_attn=True)
        # attn_all: [N, n_feat+1, n_feat+1] (index 0 = [CLS] token)
        # Slice off CLS row and column to get feature-feature attention.
        attn_all = attn_all[:, 1:, 1:]
    attn_mean = attn_all.cpu().numpy().mean(axis=0)   # [n_feat, n_feat]

    # Column sum → "how much total attention does each feature receive?"
    importance = attn_mean.sum(axis=0)

    # ── Plot 1: Top-K importance bar chart ──────────────────────────────────────
    top_idx   = np.argsort(importance)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = importance[top_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_colors = plt.cm.viridis(np.linspace(0.2, 0.85, top_k))
    ax.barh(range(top_k), top_vals[::-1], color=bar_colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Summed Attention Weight (column sum across all queries)")
    ax.set_title(
        f"Top-{top_k} Features by Received Attention\n"
        f"(avg over {n_samples} samples, last layer, heads averaged)"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "attention_top_features.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved attention_top_features.png")

    # ── Plot 2: Domain-group block heatmap ──────────────────────────────────────
    # Use the module-level _FEATURE_GROUPS — single source of truth shared with
    # the segment embeddings in UncertainFTTransformerRefined.  Features absent
    # from feature_names are silently ignored via the `if f in feat_idx_map` guard.
    groups = _FEATURE_GROUPS

    feat_idx_map = {name: i for i, name in enumerate(feature_names)}
    group_names  = list(groups.keys())
    n_groups     = len(group_names)
    group_matrix = np.zeros((n_groups, n_groups))

    for i, g_q in enumerate(group_names):
        q_idxs = [feat_idx_map[f] for f in groups[g_q] if f in feat_idx_map]
        for j, g_k in enumerate(group_names):
            k_idxs = [feat_idx_map[f] for f in groups[g_k] if f in feat_idx_map]
            if q_idxs and k_idxs:
                group_matrix[i, j] = attn_mean[np.ix_(q_idxs, k_idxs)].mean()

    fig, ax = plt.subplots(figsize=(9, 7))
    im   = ax.imshow(group_matrix, cmap='viridis', aspect='auto')
    vmax = group_matrix.max()
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(group_names, fontsize=9)
    for i in range(n_groups):
        for j in range(n_groups):
            txt_color = 'white' if group_matrix[i, j] < 0.6 * vmax else 'black'
            ax.text(j, i, f"{group_matrix[i, j]:.4f}",
                    ha='center', va='center', fontsize=7, color=txt_color)
    plt.colorbar(im, ax=ax, label='Mean Attention Weight')
    ax.set_title(
        f"Feature Group Attention Interactions\n"
        f"(avg over {n_samples} samples, last layer, heads averaged)"
    )
    ax.set_xlabel("Key (Attended-to) Groups")
    ax.set_ylabel("Query Groups")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "attention_group_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved attention_group_heatmap.png")

    # ── Plot 3: Per-head attention panels (top-10 features by global importance) ─
    # Call the last layer's MHA module directly with average_attn_weights=False
    # to bypass head-averaging and get [N, n_heads, n_feat+1, n_feat+1].
    with torch.no_grad():
        x_tok = model.tokenizer(sample_batch)
        # Prepend [CLS] token to match the training forward pass
        b_viz = sample_batch.shape[0]
        cls_viz = model.cls_token.expand(b_viz, -1, -1)
        x_tok = torch.cat((cls_viz, x_tok), dim=1)
        for layer in model.layers[:-1]:
            x_tok = layer(x_tok)
        last_layer = model.layers[-1]
        if isinstance(last_layer, AdvancedTransformerBlock):
            # Pre-norm must be applied before calling MHA directly
            x_tok_in = last_layer.norm1(x_tok)
            mha      = last_layer.attn
        else:
            x_tok_in = x_tok
            mha      = last_layer.attention
        _, per_head_attn = mha(
            x_tok_in, x_tok_in, x_tok_in,
            need_weights=True, average_attn_weights=False,
        )   # [N, n_heads, n_feat+1, n_feat+1]
        # Slice off the CLS token (index 0) to keep only feature-feature attention
        per_head_attn = per_head_attn[:, :, 1:, 1:]
    per_head_mean = per_head_attn.cpu().numpy().mean(axis=0)   # [n_heads, n_feat, n_feat]

    n_heads    = per_head_mean.shape[0]
    top10_idx  = np.argsort(importance)[::-1][:10]
    top10_names = [feature_names[i] for i in top10_idx]

    ncols = 4
    nrows = (n_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes = axes.flatten()

    for h in range(n_heads):
        sub = per_head_mean[h][np.ix_(top10_idx, top10_idx)]
        im  = axes[h].imshow(sub, cmap='viridis', vmin=0)
        axes[h].set_xticks(range(len(top10_names)))
        axes[h].set_xticklabels(top10_names, rotation=55, ha='right', fontsize=6)
        axes[h].set_yticks(range(len(top10_names)))
        axes[h].set_yticklabels(top10_names, fontsize=6)
        axes[h].set_title(f"Head {h + 1}", fontsize=9, fontweight='bold')
        fig.colorbar(im, ax=axes[h], fraction=0.046, pad=0.04)

    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(
        f"Per-Head Attention — top-10 features by global importance\n"
        f"(last layer, avg over {n_samples} samples)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "attention_per_head.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved attention_per_head.png")

    # ── Plot 4: All-features importance bar chart ────────────────────────────
    all_idx   = np.argsort(importance)[::-1]
    all_names = [feature_names[i] for i in all_idx]
    all_vals  = importance[all_idx]
    n_feat    = len(feature_names)

    fig_h = max(6, n_feat * 0.28)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    bar_colors = plt.cm.viridis(np.linspace(0.2, 0.85, n_feat))
    ax.barh(range(n_feat), all_vals[::-1], color=bar_colors)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(all_names[::-1], fontsize=7)
    ax.set_xlabel("Summed Attention Weight (column sum across all queries)")
    ax.set_title(
        f"All {n_feat} Features by Received Attention\n"
        f"(avg over {n_samples} samples, last layer, heads averaged)"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "attention_all_features_bar.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved attention_all_features_bar.png")

    # ── Plot 5: Full n×n attention heatmap for all features ──────────────────
    sorted_names = all_names  # already sorted by descending importance
    sorted_idx   = all_idx

    attn_sorted = attn_mean[np.ix_(sorted_idx, sorted_idx)]

    fig_size = max(8, n_feat * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(attn_sorted, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean Attention Weight', fraction=0.03, pad=0.03)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(sorted_names, rotation=90, ha='right', fontsize=6)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(sorted_names, fontsize=6)
    ax.set_xlabel("Key (Attended-to) Features")
    ax.set_ylabel("Query Features")
    ax.set_title(
        f"Full Feature Attention Map ({n_feat}×{n_feat})\n"
        f"(features sorted by importance, avg over {n_samples} samples, last layer, heads averaged)"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "attention_all_features_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved attention_all_features_heatmap.png")



def train_uncertainty_transformer(X_train, y_train, X_test, y_test,
                                  features,
                                  output_dir: str = ".",
                                  d_token=128, n_heads=8, n_layers=4, d_ff=256,
                                  batch_size=64, n_epochs=100,
                                  log_every=10, patience=None,
                                  loss_fn: str = 'quantile',
                                  huber_delta: float = 1.0):
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    # Device selection: CUDA > Apple MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              pin_memory=pin, num_workers=0)

    model     = UncertainFTTransformerRefined(n_features=len(features),
                                              d_token=d_token, n_heads=n_heads,
                                              n_layers=n_layers, d_ff=d_ff,
                                              feature_names=features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )
    q_levels  = [0.05, 0.5, 0.95]

    if loss_fn == 'huber':
        def _loss(preds, targets):
            return huber_pinball_loss(preds, targets, q_levels, delta=huber_delta)
    else:
        def _loss(preds, targets):
            return quantile_loss(preds, targets, q_levels)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write("=" * 60)
    tqdm.write("  UncertainFTTransformerRefined — training config")
    tqdm.write("=" * 60)
    tqdm.write(f"  Device      : {device}")
    tqdm.write(f"  Features    : {len(features)}")
    tqdm.write(f"  d_token     : {d_token}  n_heads: {n_heads}  n_layers: {n_layers}  d_ff: {d_ff}")
    tqdm.write(f"  Params      : {n_params:,}")
    tqdm.write(f"  Train size  : {len(train_ds):,}  |  Val size: {len(val_ds):,}")
    tqdm.write(f"  Batch size  : {batch_size}  |  Epochs: {n_epochs}  |  Log every: {log_every}")
    tqdm.write(f"  Early stop  : {'disabled (patience=None)' if patience is None else f'patience={patience} epochs'}")
    tqdm.write(f"  LR schedule : CosineAnnealingLR(T_max={n_epochs}, eta_min=1e-6)")
    tqdm.write(f"  Loss        : {loss_fn}" + (f"  (huber_delta={huber_delta})" if loss_fn == 'huber' else ""))
    ckpt_path = os.path.join(output_dir, 'model_best.pt')
    tqdm.write(f"  Checkpoint  : {ckpt_path}")
    tqdm.write("=" * 60)
    logger.info("Starting training: n_params=%d, device=%s, patience=%s",
                n_params, device, "disabled" if patience is None else patience)

    best_val_loss    = float("inf")
    epochs_no_improve = 0
    train_history    = []   # (epoch, avg_train, avg_val)

    epoch_bar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        grad_norm_sum = 0.0
        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch:3d} [train]",
                         leave=False, unit="batch")
        for batch_x, batch_y in batch_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)          # [batch, 3]
            loss  = _loss(preds, batch_y)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss    += loss.item()
            grad_norm_sum += grad_norm.item()
            batch_bar.set_postfix(loss=f"{loss.item():.5f}", gnorm=f"{grad_norm.item():.3f}")

        model.eval()
        val_loss  = 0.0
        val_preds_q50, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"  Epoch {epoch:3d} [val]  ",
                                          leave=False, unit="batch"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds     = model(batch_x)
                val_loss += _loss(preds, batch_y).item()
                val_preds_q50.append(preds[:, 1].cpu())
                val_targets.append(batch_y.cpu())

        avg_train  = train_loss    / len(train_loader)
        avg_val    = val_loss      / len(val_loader)
        avg_gnorm  = grad_norm_sum / len(train_loader)

        q50_np  = torch.cat(val_preds_q50).numpy()
        tgt_np  = torch.cat(val_targets).numpy()
        val_mae = float(np.abs(q50_np - tgt_np).mean())
        val_r2  = float(1.0 - np.sum((tgt_np - q50_np) ** 2) /
                        np.sum((tgt_np - tgt_np.mean()) ** 2))

        improved = avg_val < best_val_loss
        if improved:
            best_val_loss    = avg_val
            epochs_no_improve = 0
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": best_val_loss,
                        "val_mae": val_mae,
                        "val_r2": val_r2}, ckpt_path)
            tqdm.write(f"  [epoch {epoch:4d}] ✓ checkpoint saved  "
                       f"val_loss={avg_val:.5f}  MAE={val_mae:.4f}  R²={val_r2:.4f}")
            logger.info("Epoch %d: new best val_loss=%.5f MAE=%.4f R²=%.4f",
                        epoch, avg_val, val_mae, val_r2)
        else:
            epochs_no_improve += 1

        scheduler.step()
        train_history.append((epoch, avg_train, avg_val))

        epoch_bar.set_postfix(
            train=f"{avg_train:.5f}",
            val=f"{avg_val:.5f}",
            best=f"{best_val_loss:.5f}",
            MAE=f"{val_mae:.4f}",
            R2=f"{val_r2:.4f}",
            gnorm=f"{avg_gnorm:.3f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
            saved="✓" if improved else "",
        )

        # Early stopping check
        if patience is not None and epochs_no_improve >= patience:
            tqdm.write(f"  [epoch {epoch:4d}] Early stopping triggered — "
                       f"no improvement for {patience} consecutive epochs.")
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

        # Periodic verbose line — survives log-file redirection on CURC
        if (epoch + 1) % log_every == 0 or epoch == 0:
            ts = datetime.now().strftime("%H:%M:%S")
            tqdm.write(
                f"  [{ts}] epoch {epoch+1:4d}/{n_epochs}  "
                f"train={avg_train:.5f}  val={avg_val:.5f}  "
                f"best={best_val_loss:.5f}  MAE={val_mae:.4f}  R²={val_r2:.4f}  "
                f"gnorm={avg_gnorm:.3f}"
            )
            logger.info(
                "Epoch %d/%d  train=%.5f  val=%.5f  best=%.5f  MAE=%.4f  R2=%.4f  gnorm=%.3f",
                epoch + 1, n_epochs, avg_train, avg_val, best_val_loss, val_mae, val_r2, avg_gnorm,
            )

    # ── Restore best checkpoint ────────────────────────────────────────────────
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    tqdm.write("=" * 60)
    tqdm.write(f"  Training complete.")
    tqdm.write(f"  Best epoch    : {best_ckpt['epoch']}")
    tqdm.write(f"  Best val_loss : {best_ckpt['val_loss']:.5f}")
    tqdm.write(f"  Best val MAE  : {best_ckpt.get('val_mae', float('nan')):.4f}")
    tqdm.write(f"  Best val R²   : {best_ckpt.get('val_r2', float('nan')):.4f}")
    tqdm.write("=" * 60)
    logger.info("Training complete. Best epoch=%d val_loss=%.5f MAE=%.4f R2=%.4f",
                best_ckpt['epoch'], best_ckpt['val_loss'],
                best_ckpt.get('val_mae', float('nan')),
                best_ckpt.get('val_r2', float('nan')))

    return model.cpu()  # return on CPU so eval/plot functions work without device awareness


def evaluate_model_X_text(model, X_test, y_test, fig_dir):
    preds = _batched_predict(model, np.asarray(X_test, dtype=np.float32))
    q05, q50, q95 = preds[:, 0], preds[:, 1], preds[:, 2]
    uncertainty = q95 - q05
    residuals = y_test - q50

    q50_np = q50
    lower_err = np.clip(q50_np - q05, 0, None)
    upper_err = np.clip(q95 - q50_np, 0, None)
    
    # import metrics here to avoid circular import issues
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, q50_np)
    r2 = r2_score(y_test, q50_np)
    slope, intercept = np.polyfit(y_test, q50_np, 1)
    

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(y_test, q50_np,
                yerr=np.array([lower_err, upper_err]),
                fmt='o', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("True XCO2 bc anomaly")
    ax.set_ylabel("Predicted XCO2 bc anomaly")
    ax.set_aspect('equal', adjustable='box')
    ax.text(0.05, 0.95, f"R²: {r2:.3f}\nSlope: {slope:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.savefig(os.path.join(fig_dir, "pred_vs_true.png"), dpi=150, bbox_inches='tight')
    

def evaluate_model_X_all(model, 
                         fdir, data_fname, 
                         qt, features,
                         fig_dir, sfc_type=0,):
    
    # Load and preprocess the entire data set (same as in training_data_load)
    data_path = os.path.join(fdir, data_fname)
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    df = df[df['sfc_type'] == sfc_type]
    df = df[df['snow_flag'] == 0]
    # df = df[df['xco2_bc_anomaly'].notna()]  # Drop rows with missing target variable

    # One-hot encode footprint number into 8 binary columns (fp_0 … fp_7)
    fp_dummies = pd.concat(
        {f'fp_{i}': (df['fp'] == i).astype(int) for i in range(8)}, axis=1
    )
    df = pd.concat([df, fp_dummies], axis=1)
    
    # quantity transform the features using the pre-fitted transformer
    X_all = qt.transform(df[features])
    for i in range(8):
        fp_col = f'fp_{i}'
        features.append(fp_col)
        fp_values = df[fp_col].values.reshape(-1, 1)  # reshape to 2D for concatenation
        X_all = np.hstack([X_all, fp_values])  # add one-hot columns to the end of X
        
    xco2_bc = df['xco2_bc'].values
    xco2_bc_anomaly = df['xco2_bc_anomaly'].values
    
    
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_all).float())
        q05, q50, q95 = preds[:, 0], preds[:, 1], preds[:, 2]
        uncertainty = (q95 - q05).numpy()
        residuals = (xco2_bc_anomaly - q50.numpy())
    
    upper_err = np.clip(q95.numpy() - q50.numpy(), 0, None)
    lower_err = np.clip(q50.numpy() - q05.numpy(), 0, None)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(xco2_bc_anomaly, q50.numpy(), 
                yerr=np.array([lower_err, upper_err]),
                fmt='o', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("True XCO2 bc anomaly")
    ax.set_ylabel("Predicted XCO2 bc anomaly")
    fig.savefig(os.path.join(fig_dir, "pred_vs_true.png"), dpi=150, bbox_inches='tight')


def plot_evaluation_by_regime(model, df, qt, features, output_dir):
    """3×4 evaluation plot matching mlp_lr_models.py lines 734-822.

    Row structure (mirrors mlp_lr_models.py lines 706-721):
        0 – Clear-sky FPs (cld_dist > 10 km), valid anomaly   [anomaly_label=True]
        1 – Cloud-affected FPs (cld_dist ≤ 10 km), valid anomaly [anomaly_label=True]
        2 – Cloud FPs with NaN anomaly (cld_dist ≤ 10 km)       [anomaly_label=False]

    Column structure (FT-Transformer adaptation):
        0 – Scatter: xco2_orig vs FT q50 direct prediction  (like LR scatter)
        1 – Scatter: same, colour-coded by uncertainty q95−q05  (FT adaptation of MLP scatter)
        2 – Distribution of anomaly values: [Original, FT q50]
        3 – Distribution of XCO2_bc values: [raw xco2_bc, xco2_bc−q50 corrected]
            (mirrors mlp col-3: raw vs bias-corrected xco2_bc)
    """
    # ── Feature matrix — avoid df.copy() + hstack loop to keep peak RAM low ─────
    qt_features = list(features[:-8])    # non-fp continuous features
    fp_cols     = list(features[-8:])    # fp_0 … fp_7

    # Ensure fp one-hot columns exist in df — build all missing cols at once to
    # avoid DataFrame fragmentation from repeated single-column assignments.
    missing_fp = [i for i in range(8) if f'fp_{i}' not in df.columns]
    if missing_fp:
        df = pd.concat(
            [df, pd.DataFrame(
                {f'fp_{i}': (df['fp'] == i).astype(np.float32) for i in missing_fp},
                index=df.index,
            )],
            axis=1,
        )

    X_qt  = qt.transform(df[qt_features].to_numpy(dtype=float)).astype(np.float32)
    X_fp  = df[fp_cols].to_numpy(dtype=np.float32)
    X_all = np.concatenate([X_qt, X_fp], axis=1)
    del X_qt, X_fp
    gc.collect()

    # ── Batched inference — avoids [N, n_feat*d_token] OOM tensor ─────────────
    preds = _batched_predict(model, X_all)
    del X_all
    gc.collect()
    q05 = preds[:, 0]
    q50 = preds[:, 1]
    q95 = preds[:, 2]
    uncertainty  = np.clip(q95 - q05, 0, None)
    true_anomaly = df['xco2_bc_anomaly'].values.astype(float)
    xco2_bc      = df['xco2_bc'].values.astype(float)

    # ── Recompute anomaly from FT-corrected XCO2 (mirrors mlp_lr_models.py L641-643) ──
    from mlp_lr_models import compute_xco2_anomaly_date_id
    _anomaly_args = {'lat_thres': 0.25, 'std_thres': 1.0, 'min_cld_dist': 10.0}
    xco2_bc_corrected_ft = xco2_bc - q50
    _req_cols = {'date', 'orbit_id', 'lat', 'cld_dist_km'}
    if _req_cols.issubset(df.columns):
        anomaly_ft = compute_xco2_anomaly_date_id(
            df['date'], df['orbit_id'], df['lat'].values,
            df['cld_dist_km'].values, xco2_bc_corrected_ft,
            **_anomaly_args,
        )
    else:
        anomaly_ft = None

    # ── Row masks (mirrors mlp_lr_models.py lines 706-711) ─────────────────────
    valid_anom = np.isfinite(true_anomaly)
    if 'cld_dist_km' in df.columns:
        cd = df['cld_dist_km'].values.astype(float)
        mask_r0 = valid_anom & (cd > 10)
        mask_r1 = valid_anom & (cd <= 10)
        mask_r2 = ~valid_anom & (cd <= 10)
    else:
        mask_r0 = valid_anom
        mask_r1 = np.zeros(len(df), dtype=bool)
        mask_r2 = ~valid_anom

    # row_configs tuple: (xco2_orig, xco2_orig_bc, xco2_ft, mask, row_label, anomaly_label)
    #   xco2_orig    – x-axis of scatter / anomaly distribution (anomaly or xco2_bc)
    #   xco2_orig_bc – raw xco2_bc reference used to build the col-3 corrected distribution
    #   xco2_ft      – FT direct prediction (q50)
    #   anomaly_label – True: use anomaly axis labels; False: use xco2_bc labels
    row_configs = [
        (true_anomaly, xco2_bc, q50, mask_r0,
         'Clear-sky FPs (cld_dist > 10 km)', True),
        (true_anomaly, xco2_bc, q50, mask_r1,
         'Cloud-affected FPs (cld_dist ≤ 10 km)', True),
        (xco2_bc,      xco2_bc, q50, mask_r2,
         'Cloud FPs with NaN anomaly (cld_dist ≤ 10 km)', False),
    ]

    # ── Figure ─────────────────────────────────────────────────────────────────
    plt.close('all')
    fig, axes = plt.subplots(3, 5, figsize=(27, 17))

    for row_i, (xco2_orig, xco2_orig_bc, xco2_ft, mask, row_label, anomaly_label) in enumerate(row_configs):
        ax_sc1, ax_sc2, ax_h, ax_h2, ax_h3 = axes[row_i]

        x_orig = xco2_orig[mask]
        x_bc   = xco2_orig_bc[mask]
        x_ft   = xco2_ft[mask]
        unc    = uncertainty[mask]

        _lo = np.nanpercentile(x_orig[np.isfinite(x_orig)], 1)  if np.isfinite(x_orig).any() else -3.0
        _hi = np.nanpercentile(x_orig[np.isfinite(x_orig)], 99) if np.isfinite(x_orig).any() else  3.0

        # ── Cols 0 & 1: Scatter (hidden for row 2 where x_orig == x_bc) ─────────
        if not (x_orig == x_bc).all():
            v = np.isfinite(x_orig) & np.isfinite(x_ft)

            # Col 0: plain scatter — like LR scatter in mlp
            ax_sc1.scatter(x_orig[v], x_ft[v], c='orange', edgecolor=None, s=5, alpha=0.6)
            ax_sc1.set_xlim(_lo, _hi); ax_sc1.set_ylim(_lo, _hi)
            ax_sc1.set_aspect('equal', adjustable='box')
            ax_sc1.axline((_lo, _lo), slope=1, color='r', linestyle='--')
            if anomaly_label:
                ax_sc1.set_xlabel('Original XCO2_bc anomaly (ppm)')
                ax_sc1.set_ylabel('FT-corrected XCO2_bc anomaly (ppm)')
            else:
                ax_sc1.set_xlabel('Original XCO2_bc (ppm)')
                ax_sc1.set_ylabel('FT-corrected XCO2_bc (ppm)')
            ax_sc1.set_title(f'{row_label}\n[FT scatter]')
            if v.sum() > 1:
                r2 = 1 - np.nansum((x_orig[v] - x_ft[v])**2) / \
                         np.nansum((x_orig[v] - np.nanmean(x_orig[v]))**2)
                ax_sc1.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax_sc1.transAxes, va='top')

            # Col 1: scatter coloured by uncertainty — FT adaptation of MLP scatter
            unc_v    = unc[v]
            vmin_unc = max(np.nanpercentile(unc_v, 5), 1e-4)
            vmax_unc = np.nanpercentile(unc_v, 95)
            sc = ax_sc2.scatter(x_orig[v], x_ft[v], c=unc_v, cmap='plasma',
                                vmin=vmin_unc, vmax=vmax_unc,
                                edgecolor=None, s=5, alpha=0.6)
            ax_sc2.set_xlim(_lo, _hi); ax_sc2.set_ylim(_lo, _hi)
            ax_sc2.set_aspect('equal', adjustable='box')
            ax_sc2.axline((_lo, _lo), slope=1, color='r', linestyle='--')
            if anomaly_label:
                ax_sc2.set_xlabel('Original XCO2_bc anomaly (ppm)')
                ax_sc2.set_ylabel('FT-corrected XCO2_bc anomaly (ppm)')
            else:
                ax_sc2.set_xlabel('Original XCO2_bc (ppm)')
                ax_sc2.set_ylabel('FT-corrected XCO2_bc (ppm)')
            ax_sc2.set_title(f'{row_label}\n[FT scatter, colour = q95−q05]')
            fig.colorbar(sc, ax=ax_sc2, label='q95−q05 (ppm)', pad=0.02)
            if v.sum() > 1:
                ax_sc2.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax_sc2.transAxes, va='top')
        else:
            ax_sc1.set_visible(False)
            ax_sc2.set_visible(False)

        # ── Cols 2 & 3: Distribution histograms (mirrors mlp_lr_models.py L776-813) ──
        # Items: (prediction_array, xco2_bc_ref, colour, label)
        #   xco2_bc_ref is used to: (a) detect if col-2 hist should show, and
        #                           (b) build col-3 corrected xco2_bc = xco2_bc_ref − pred
        items = [
            (x_orig, x_bc, 'blue',   'Original'),
            (x_ft,   x_bc, 'orange', 'FT-corrected'),
        ]

        for _xco2, _xco2_bc, _color, _label in items:
            _v  = _xco2[np.isfinite(_xco2)]
            _v2 = _xco2_bc[np.isfinite(_xco2)]
            if len(_v) == 0:
                continue

            # Col 2: anomaly distribution — skip if _v == _v2 (e.g. row-3 Original)
            if len(_v2) > 0 and not (_v == _v2).all():
                _mu, _sigma = _v.mean(), _v.std()
                _bins = np.linspace(np.nanpercentile(_v, 1), np.nanpercentile(_v, 99), 100)
                ax_h.hist(_v, bins=_bins, color=_color, alpha=0.6, density=True,
                          label=f'{_label}\nμ={_mu:.3f}, σ={_sigma:.3f}')
                ax_h.axvline(_mu,          color=_color, linestyle='-',  linewidth=1.2)
                ax_h.axvline(_mu - _sigma, color=_color, linestyle=':',  linewidth=0.9)
                ax_h.axvline(_mu + _sigma, color=_color, linestyle=':',  linewidth=0.9)

            # Col 3: xco2_bc distribution — raw for Original, corrected (xco2_bc − pred) for others
            if _label != 'Original':
                _v2 = _v2 - _v   # corrected xco2_bc = raw − predicted_anomaly
            _mu2, _sigma2 = _v2.mean(), _v2.std()
            _bins2 = np.linspace(np.nanpercentile(_v2, 1), np.nanpercentile(_v2, 99), 100)
            ax_h2.hist(_v2, bins=_bins2, color=_color, alpha=0.3, density=True,
                       label=f'{_label}\nμ={_mu2:.3f}, σ={_sigma2:.3f}')
            ax_h2.axvline(_mu2,           color=_color, linestyle='-',  linewidth=1.0)
            ax_h2.axvline(_mu2 - _sigma2, color=_color, linestyle=':',  linewidth=0.8)
            ax_h2.axvline(_mu2 + _sigma2, color=_color, linestyle=':',  linewidth=0.8)
            
            # Col 4: ideal distribution of xco2_bc values — only for Original, and only if _v != _v2 (e.g. row-3 Original)
            if _label == 'Original' and not (_v == _v2).all():
                _v2 = _v2 - _v   # corrected xco2_bc = raw − predicted_anomaly
                _mu2, _sigma2 = _v2.mean(), _v2.std()
                bins3 = np.linspace(np.nanpercentile(_v2, 1), np.nanpercentile(_v2, 99), 100)
                ax_h3.hist(_v2, bins=bins3, color=_color, alpha=0.2, density=True,
                        label=f'Idael {_label}-corrected \nμ={_mu2:.3f}, σ={_sigma2:.3f}')
            elif (_v == _v2).all():
                ax_h3.set_visible(False)
            else:
                _mu2, _sigma2 = _v2.mean(), _v2.std()
                bins3 = np.linspace(np.nanpercentile(_v2, 1), np.nanpercentile(_v2, 99), 100)
                ax_h3.hist(_v2, bins=bins3, color=_color, alpha=0.2, density=True,
                        label=f' {_label}\nμ={_mu2:.3f}, σ={_sigma2:.3f}')
            ax_h3.axvline(_mu2,           color=_color, linestyle='-',  linewidth=0.8)
            ax_h3.axvline(_mu2 - _sigma2, color=_color, linestyle=':',  linewidth=0.6)
            ax_h3.axvline(_mu2 + _sigma2, color=_color, linestyle=':',  linewidth=0.6)

        ax_h.set_title(f'{row_label}\n[Distribution]')
        ax_h.set_xlabel('XCO2_bc anomaly (ppm)')
        ax_h.legend(fontsize=10)
        ax_h2.set_title(f'{row_label}\n[Distribution]')
        ax_h2.legend(fontsize=10)
        if not (_v == _v2).all():
            ax_h2.set_xlabel('Corrected XCO2 (ppm)')
        else:
            ax_h2.set_xlabel('XCO2 (ppm)')
        ax_h3.set_title(f'{row_label}\n[Ideal corrected XCO2 distribution]')
        ax_h3.legend(fontsize=10)
        
        

    fig.suptitle('FT-Transformer XCO2_bc by cloud-distance regime', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ft_evaluation_by_regime.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved ft_evaluation_by_regime.png")

    # ── 1×3 recomputed-anomaly comparison (mirrors mlp_lr_models.py L646-701) ──
    if anomaly_ft is None:
        return

    anomaly_orig = true_anomaly   # full-dataset original anomaly

    plt.close('all')
    fig2, (ax_sc1, ax_sc2, ax_hist) = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 0: scatter original vs FT-recomputed anomaly (like LR scatter)
    valid = np.isfinite(anomaly_orig) & np.isfinite(anomaly_ft)
    ax_sc1.scatter(anomaly_orig[valid], anomaly_ft[valid],
                   c='orange', edgecolor=None, s=5, alpha=0.6)
    _lim = np.nanpercentile(np.abs(anomaly_orig[valid]), 99)
    ax_sc1.set_xlim(-_lim, _lim); ax_sc1.set_ylim(-_lim, _lim)
    ax_sc1.set_aspect('equal', adjustable='box')
    ax_sc1.axline((0, 0), slope=1, color='r', linestyle='--')
    ax_sc1.set_xlabel('Original XCO2_bc anomaly (ppm)')
    ax_sc1.set_ylabel('FT-recomputed XCO2_bc anomaly (ppm)')
    ax_sc1.set_title('Original vs FT-recomputed anomaly')
    r2_ft = 1 - np.nansum((anomaly_orig[valid] - anomaly_ft[valid])**2) / \
                np.nansum((anomaly_orig[valid] - np.nanmean(anomaly_orig[valid]))**2)
    ax_sc1.text(0.05, 0.95, f'R²={r2_ft:.3f}', transform=ax_sc1.transAxes, va='top')

    # Panel 1: scatter coloured by uncertainty (FT adaptation of MLP scatter)
    unc_valid = uncertainty[valid]
    vmin_u = max(np.nanpercentile(unc_valid, 5), 1e-4)
    vmax_u = np.nanpercentile(unc_valid, 95)
    sc = ax_sc2.scatter(anomaly_orig[valid], anomaly_ft[valid],
                        c=unc_valid, cmap='plasma', vmin=vmin_u, vmax=vmax_u,
                        edgecolor=None, s=5, alpha=0.6)
    ax_sc2.set_xlim(-_lim, _lim); ax_sc2.set_ylim(-_lim, _lim)
    ax_sc2.set_aspect('equal', adjustable='box')
    ax_sc2.axline((0, 0), slope=1, color='r', linestyle='--')
    ax_sc2.set_xlabel('Original XCO2_bc anomaly (ppm)')
    ax_sc2.set_ylabel('FT-recomputed XCO2_bc anomaly (ppm)')
    ax_sc2.set_title('Original vs FT-recomputed anomaly\n(colour = q95−q05 uncertainty)')
    fig2.colorbar(sc, ax=ax_sc2, label='q95−q05 (ppm)', pad=0.02)
    ax_sc2.text(0.05, 0.95, f'R²={r2_ft:.3f}', transform=ax_sc2.transAxes, va='top')

    # Panel 2: distribution histogram (mirrors mlp L678-695)
    _bins = np.linspace(-3, 3, 211)
    for _anom, _color, _label in [
            (anomaly_orig, 'blue',   'Original'),
            (anomaly_ft,   'orange', 'FT-recomputed'),
    ]:
        _v = _anom[np.isfinite(_anom)]
        _mu, _sigma = np.nanmean(_v), np.nanstd(_v)
        ax_hist.hist(_v, bins=_bins, color=_color, alpha=0.6, density=True,
                     label=f'{_label}\nμ={_mu:.3f}, σ={_sigma:.3f}')
        ax_hist.axvline(_mu,          color=_color, linestyle='-',  linewidth=1.2)
        ax_hist.axvline(_mu - _sigma, color=_color, linestyle=':',  linewidth=0.9)
        ax_hist.axvline(_mu + _sigma, color=_color, linestyle=':',  linewidth=0.9)

    ax_hist.set_xlabel('XCO2_bc anomaly (ppm)')
    ax_hist.set_title('Anomaly distribution comparison')
    ax_hist.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax_hist.legend(fontsize=10)

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'ft_recomputed_anomaly_comparison.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    logger.info("Saved ft_recomputed_anomaly_comparison.png")


# ─── Permutation importance ────────────────────────────────────────────────────
def plot_permutation_importance(model, X_test, y_test, features, output_dir,
                                n_repeats=5, subsample=3000, batch_size=1024):
    """Permutation importance for the FT-Transformer (q50 head).

    For each feature in turn, shuffles that column of X_sub **in-place**
    (restoring it afterward) so no full-array copy is needed per iteration —
    only a single column (shape [n]) is duplicated.  `batch_size` is kept
    small to bound the peak attention-tensor memory [batch, n_feat, n_feat].

    Outputs
    -------
    ft_permutation_importance.csv   – feature, mean_importance, std_importance
    ft_permutation_importance.png   – horizontal bar chart (all features)
    """
    model.eval()
    features = list(features)
    rng = np.random.default_rng(42)

    # ── Sub-sample ──────────────────────────────────────────────────────────────
    n = min(subsample, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    # C-contiguous float32 so torch.tensor() gets a zero-copy view later
    X_sub = np.ascontiguousarray(X_test[idx], dtype=np.float32)
    y_sub = y_test[idx].astype(np.float32)

    def _predict_q50(X_np):
        """Batched inference → q50 predictions (numpy).  Small batch_size
        keeps intermediate attention tensors [batch, n_feat, n_feat] small."""
        out = []
        for start in range(0, len(X_np), batch_size):
            Xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32)
            with torch.no_grad():
                preds = model(Xb)          # [batch, 3]
            out.append(preds[:, 1].numpy())  # q50
            del Xb, preds
        return np.concatenate(out)

    # ── Baseline R² ─────────────────────────────────────────────────────────────
    y_base      = _predict_q50(X_sub)
    ss_tot      = float(((y_sub - y_sub.mean()) ** 2).sum())
    baseline_r2 = 1.0 - float(((y_sub - y_base) ** 2).sum()) / ss_tot
    logger.info("Permutation importance baseline R²: %.4f", baseline_r2)

    # ── Per-feature importance — in-place column swap ───────────────────────────
    # Exclude footprint one-hot columns (fp_0 … fp_7) from both calculation and plot.
    non_fp = [(col, fname) for col, fname in enumerate(features)
              if not (fname.startswith('fp_') and fname[3:].isdigit())]

    importances = np.zeros((len(non_fp), n_repeats))
    rng_inner   = np.random.default_rng(0)

    for i, (col, fname) in enumerate(tqdm(non_fp, desc="Permutation importance", unit="feat")):
        orig_col = X_sub[:, col].copy()      # save one column only (n floats)
        for r in range(n_repeats):
            X_sub[:, col] = rng_inner.permutation(orig_col)   # shuffle in-place
            y_shuf        = _predict_q50(X_sub)
            r2_shuf       = 1.0 - float(((y_sub - y_shuf) ** 2).sum()) / ss_tot
            importances[i, r] = baseline_r2 - r2_shuf
            del y_shuf
        X_sub[:, col] = orig_col             # restore column
        del orig_col
        gc.collect()

    non_fp_names = [fname for _, fname in non_fp]
    mean_imp = importances.mean(axis=1)
    std_imp  = importances.std(axis=1)

    # ── Save CSV ─────────────────────────────────────────────────────────────────
    imp_df = pd.DataFrame({
        'feature':          non_fp_names,
        'mean_importance':  mean_imp,
        'std_importance':   std_imp,
    }).sort_values('mean_importance', ascending=False)
    csv_path = os.path.join(output_dir, 'ft_permutation_importance.csv')
    imp_df.to_csv(csv_path, index=False)
    logger.info("Saved ft_permutation_importance.csv")

    # ── Bar chart ────────────────────────────────────────────────────────────────
    n_feat   = len(non_fp_names)
    fig_h    = max(6, n_feat * 0.28)
    fig, ax  = plt.subplots(figsize=(8, fig_h))

    sorted_names = imp_df['feature'].tolist()
    sorted_mean  = imp_df['mean_importance'].tolist()
    sorted_std   = imp_df['std_importance'].tolist()

    bar_colors = plt.cm.viridis(np.linspace(0.2, 0.85, n_feat))
    ax.barh(range(n_feat), sorted_mean[::-1],
            xerr=sorted_std[::-1], error_kw=dict(ecolor='gray', capsize=3),
            color=bar_colors)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(sorted_names[::-1], fontsize=7)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Mean R² drop when feature is permuted')
    ax.set_title(
        f'FT-Transformer Permutation Importance ({n_feat} features)\n'
        f'(baseline R²={baseline_r2:.3f}, {n_repeats} repeats, n={n} samples)'
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ft_permutation_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved ft_permutation_importance.png")


# ─── Main analysis entry point ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FT-Transformer XCO2 bias correction")
    parser.add_argument('--sfc_type', type=int, default=0,
                        help="Surface type filter for training and evaluation (default: 0 = ocean only).")
    parser.add_argument('--suffix', type=str, default='',
                        help='Subfolder name appended to the base output directory '
                             '(e.g. --suffix v2_reduced).  '
                             'Creates results/model_ft_transformer/<suffix>/.')
    parser.add_argument('--pipeline', type=str, default=None,
                        help='Path to a pre-fitted FeaturePipeline (.pkl).  '
                             'If not supplied, a new pipeline is fitted from the training data '
                             'and saved to <output_dir>/pipeline.pkl.')
    parser.add_argument('--loss', type=str, default='huber',
                        choices=['quantile', 'huber'],
                        help='Loss function for q50: "huber" = Huber for q50 + pinball for q05/q95 (default), '
                             '"quantile" = pinball for all outputs.  '
                             'Huber is robust to the heavy left tail of cloud-affected anomalies.')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                        help='Huber transition point δ (ppm).  Errors with |e| ≤ δ are '
                             'penalised quadratically, larger errors linearly.  '
                             'Only used when --loss huber.  Default: 1.0.')
    args = parser.parse_args()

    storage_dir = get_storage_dir()
    fdir      = storage_dir / 'results/csv_collection'
    if platform.system() == "Linux":
        data_name = 'combined_2016_2020_dates.parquet'  # for full 2-year dataset
    elif platform.system() == "Darwin":
        data_name = 'combined_2020-01-01_all_orbits.parquet'  # for quick testing with one date's data
    base_dir   = storage_dir / 'results/model_ft_transformer'
    output_dir = base_dir / args.suffix if args.suffix else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_type = args.sfc_type
    logger.info(f"Surface type filter: {surface_type} (0=ocean only, 1=land only, 2=sea-ice only)")

    # ── Load data ──────────────────────────────────────────────────────────────
    _dp = os.path.join(fdir, data_name)
    df = pd.read_parquet(_dp) if _dp.endswith('.parquet') else pd.read_csv(_dp)
    df = df[df['sfc_type'] == surface_type]
    df = df[df['snow_flag'] == 0]

    # ── Pipeline: load or fit ──────────────────────────────────────────────────
    pipeline_path = output_dir / 'pipeline.pkl'
    if args.pipeline:
        pipeline = FeaturePipeline.load(args.pipeline)
    elif pipeline_path.exists():
        pipeline = FeaturePipeline.load(pipeline_path)
    else:
        pipeline = FeaturePipeline.fit(df, sfc_type=surface_type)
        pipeline.save(pipeline_path)

    features = pipeline.features

    # ── Transform + split ──────────────────────────────────────────────────────
    # Strategy: extract needed columns from df as float32 first, then free df
    # BEFORE calling QT.transform() so df is never alive at the same time as
    # the QT float64 intermediates (which would otherwise push peak to ~25 GB).
    # Derive any engineered features missing from older CSVs (e.g. airmass_sq).
    df = _ensure_derived_features(df)
    # Ensure fp one-hot cols exist — build all missing at once to avoid fragmentation.
    missing_fp = [i for i in range(8) if f'fp_{i}' not in df.columns]
    if missing_fp:
        df = pd.concat(
            [df, pd.DataFrame(
                {f'fp_{i}': (df['fp'] == i).astype(np.float32) for i in missing_fp},
                index=df.index,
            )],
            axis=1,
        )

    valid_rows = ~df['xco2_bc_anomaly'].isna()
    y_all    = df['xco2_bc_anomaly'].values.astype(np.float32)
    X_qt_raw = df[pipeline.qt_features].to_numpy(dtype=np.float32)   # (N, n_qt)
    X_fp_raw = df[pipeline.fp_cols].to_numpy(dtype=np.float32)       # (N, 8)
    del df
    gc.collect()

    # QT.transform() uses float64 internally; cast result back to float32
    X_qt_out = pipeline.qt.transform(X_qt_raw).astype(np.float32)
    del X_qt_raw
    gc.collect()

    X_all = np.concatenate([X_qt_out, X_fp_raw], axis=1)
    del X_qt_out, X_fp_raw
    gc.collect()

    X = X_all[valid_rows]
    y = y_all[valid_rows]
    del X_all, y_all
    gc.collect()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    del X, y
    gc.collect()
    print("X_train shape", X_train.shape)

    # ── Load or train model ────────────────────────────────────────────────────
    if FTAdapter.can_load(output_dir):
        adapter = FTAdapter.load(output_dir)
        model   = adapter.model
    else:
        if platform.system() == "Darwin":
            epochs = 100  # Fewer epochs for local testing
        else:
            epochs = 500  # More epochs for CURC training

        model = train_uncertainty_transformer(
            X_train, y_train, X_test, y_test,
            features=features,
            output_dir=str(output_dir),
            d_token=256, n_heads=8, n_layers=4, d_ff=512,
            batch_size=1024, n_epochs=epochs,
            patience=50,   # None = run all epochs; int = early-stop after N epochs with no improvement
            loss_fn=args.loss,
            huber_delta=args.huber_delta,
        )

        # ── Persist adapter metadata (model_best.pt already written by training loop) ──
        FTAdapter(model, n_features=pipeline.n_features,
                  d_token=256, n_heads=8, n_layers=4, d_ff=512,
                  tokenizer_type='mlp', feature_names=features).save(output_dir)

    # ── Evaluate: quantile width vs selected features ─────────────────────────
    uncertainty_features = ['glint_prox', 'aod_total', 'mu_sza', 'tcwv', 'csnr_o2a']
    for feat_name in uncertainty_features:
        if feat_name in features:
            feat_vals = X_test[:, features.index(feat_name)]
            plot_uncertainty_by_feature(model, X_test, feat_vals, feat_name, output_dir)

    # ── Evaluate: attention maps (batch-averaged for stability) ────────────────
    n_viz        = min(200, len(X_test))
    sample_batch = torch.tensor(X_test[:n_viz], dtype=torch.float32)
    plot_attention_map(model, sample_batch, features, output_dir)

    # ── Evaluate: permutation importance ───────────────────────────────────────
    plot_permutation_importance(model, X_test, y_test, features, str(output_dir))

    evaluate_model_X_text(model, X_test, y_test, fig_dir=output_dir)

    # ── Evaluate: 3×4 regime comparison plot (mirrors mlp_lr_models.py) ────────
    _dp2 = os.path.join(fdir, data_name)
    df_eval = pd.read_parquet(_dp2) if _dp2.endswith('.parquet') else pd.read_csv(_dp2)
    df_eval = df_eval[df_eval['sfc_type'] == surface_type]
    df_eval = df_eval[df_eval['snow_flag'] == 0]
    plot_evaluation_by_regime(model, df_eval, pipeline.qt, pipeline.features, str(output_dir))
    del df_eval
    gc.collect()



if __name__ == "__main__":
    main()

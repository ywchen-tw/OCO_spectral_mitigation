import h5py
import numpy as np
import pandas as pd
import os
import sys
import platform
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.preprocessing import QuantileTransformer
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
from config import Config
import pickle

logger = logging.getLogger(__name__)

def get_storage_dir():
    if platform.system() == "Darwin":
        logger.info("Detected macOS - using local data directory")
        return Path(Config.get_data_path('local'))
    elif platform.system() == "Linux":
        logger.info("Detected Linux - using CURC storage directory")
        return Path(Config.get_data_path('curc'))
    else:
        logger.warning(f"Unknown platform: {platform.system()}. Using default.")
        return Path(Config.get_data_path('default'))

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
        
        # We pre-calculate the cumulative size to map index -> file
        self.file_sizes = []
        for f in self.file_paths:
            # We assume a quick count of rows (e.g., using a metadata header or fast read)
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

        # 2. Read only that row (skiprows skips data rows, not the header)
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
    """
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

# Training setup
quantiles = [0.05, 0.5, 0.95]

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
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val).float())
        q05, q50, q95 = preds[:, 0], preds[:, 1], preds[:, 2]
        uncertainty = (q95 - q05).numpy()

    q50_np = q50.numpy()
    # Clip to non-negative: independent quantile heads can cross (q05 > q50 or q50 > q95)
    lower_err = np.clip(q50_np - q05.numpy(), 0, None)
    upper_err = np.clip(q95.numpy() - q50_np, 0, None)

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
        # attn_all: [N, n_feat, n_feat]  (PyTorch averages across heads by default)
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
    groups = {
        'Spectral\nk-coeff':  ['o2a_k1', 'o2a_k2', 'wco2_k1', 'wco2_k2', 'sco2_k1', 'sco2_k2'],
        'Geometry':           ['mu_sza', 'mu_vza', 'sin_raa', 'cos_raa', 'cos_theta',
                               'Phi_cos_theta', 'R_rs_factor', 'cos_glint_angle',
                               'glint_prox', 'pol_ang_rad'],
        'Pressure\n& Meteo':  ['log_P', 'airmass', 'dp', 'dp_psfc_ratio', 'dpfrac',
                               'h2o_scale', 'delT', 'co2_grad_del',
                               'ws', 'dws', 's31', 's32', 't700', 'tcwv'],
        'Surface\nAlbedo':    ['alb_o2a', 'alb_wco2', 'alb_sco2'],
        'CO\u2082\nRetrieval': ['fs_rel_0', 'co2_ratio_bc', 'h2o_ratio_bc',
                               'xco2_strong_idp', 'xco2_weak_idp'],
        'SNR &\nDetector':    ['csnr_o2a', 'csnr_wco2', 'csnr_sco2',
                               'snr_o2a', 'snr_wco2', 'snr_sco2',
                               'dp_abp', 'h_cont_o2a', 'h_cont_wco2', 'h_cont_sco2',
                               'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2'],
        'AOD &\nAerosol':     ['aod_total', 'aod_bc', 'aod_dust', 'aod_ice', 'aod_water',
                               'aod_oc', 'aod_seasalt', 'aod_strataer', 'aod_sulfate',
                               'dust_height', 'ice_height', 'water_height'],
        'Footprint':          [f'fp_{i}' for i in range(8)],
    }

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
    # to bypass head-averaging and get [N, n_heads, n_feat, n_feat].
    with torch.no_grad():
        x_tok = model.tokenizer(sample_batch)
        for layer in model.layers[:-1]:
            x_tok = layer(x_tok)
        _, per_head_attn = model.layers[-1].attention(
            x_tok, x_tok, x_tok,
            need_weights=True, average_attn_weights=False,
        )   # [N, n_heads, n_feat, n_feat]
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



def training_data_load(fdir, data_fname, sfc_type=0):
    # 1. Load and filter
    data_path = os.path.join(fdir, data_fname)
    df = pd.read_csv(data_path)
    df = df[df['sfc_type'] == sfc_type]
    df = df[df['snow_flag'] == 0]
    # df = df[df['xco2_bc_anomaly'].notna()]  # Drop rows with missing target variable

    # One-hot encode footprint number into 8 binary columns (fp_0 … fp_7)
    fp_dummies = pd.concat(
        {f'fp_{i}': (df['fp'] == i).astype(int) for i in range(8)}, axis=1
    )
    df = pd.concat([df, fp_dummies], axis=1)

    # (Trig transforms already applied in fitting_data_correction.py when the CSV was built)

    # 2. Statistical Quantile Transform
    # We transform everything to a Normal distribution to help the Transformer converge
    features = ['o2a_k1', 'o2a_k2', 'wco2_k1', 'wco2_k2', 'sco2_k1', 'sco2_k2',
                'mu_sza', 'mu_vza', 
                'sin_raa', 'cos_raa', 'cos_theta', 'Phi_cos_theta', 'R_rs_factor', 
                'cos_glint_angle', 'glint_prox',
                # 'alt', 'alt_std', 
                'ws',
                'log_P', 'airmass', 'dp', 'dp_psfc_ratio', 'dpfrac', 
                'h2o_scale', 'delT', 
                'co2_grad_del', 'alb_o2a', 'alb_wco2',  'alb_sco2', 
                'fs_rel_0', 
                'co2_ratio_bc', 'h2o_ratio_bc', 
                'csnr_o2a', 'csnr_wco2', 'csnr_sco2', 'dp_abp', 
                'h_cont_o2a', 'h_cont_wco2',  'h_cont_sco2', 
                'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2', 
                'xco2_strong_idp', 'xco2_weak_idp', 
                'aod_total', 
                # 'aod_bc', 'aod_dust', 'aod_ice', 'aod_water', 
                # 'aod_oc', 'aod_seasalt', 'aod_strataer', 'aod_sulfate', 'dws', 
                # 'dust_height', 'ice_height', 'water_height',
                'snr_o2a', 'snr_wco2', 'snr_sco2', 'pol_ang_rad', 
                's31', 's32', 't700', 'tcwv',
                ]
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    
    X_all = qt.fit_transform(df[features])
    # X = pd.concat([pd.DataFrame(X1, columns=features), df[[f'fp_{i}' for i in range(8)]]], axis=1)
    for i in range(8):
        # add new one-hot columns to the end of the feature list and to the transformed X
        fp_col = f'fp_{i}'
        features.append(fp_col)
        fp_values = df[fp_col].values.reshape(-1, 1)  # reshape to 2D for concatenation
        X_all = np.hstack([X_all, fp_values])  # add one-hot columns to the end of X
        
    # remove rows if 'xco2_bc_anomaly' is NaN (missing target variable)
    valid_rows = ~df['xco2_bc_anomaly'].isna()
    X = X_all[valid_rows]
    y = df['xco2_bc_anomaly'][valid_rows].values
    
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("X_train shape", X_train.shape)
    
    return X_train, X_test, y_train, y_test, features, qt

def train_transformer(X_train, y_train, X_test, y_test,
                      features,
                      output_dir: str = ".",
                      d_token=64, n_heads=8, n_layers=3, d_ff=128,
                      batch_size=64, n_epochs=100):
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
    print(f"Training on: {device}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              pin_memory=pin, num_workers=0)

    model     = UncertainBiasPredictorFT(n_features=len(features),
                                         d_token=d_token, n_heads=n_heads,
                                         n_layers=n_layers, d_ff=d_ff).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    q_levels  = [0.05, 0.5, 0.95]

    ckpt_path     = os.path.join(output_dir, "model_best.pt")
    best_val_loss = float("inf")

    epoch_bar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch:3d} [train]",
                         leave=False, unit="batch")
        for batch_x, batch_y in batch_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)          # [batch, 3]
            loss  = quantile_loss(preds, batch_y, q_levels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.5f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"  Epoch {epoch:3d} [val]  ",
                                          leave=False, unit="batch"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds    = model(batch_x)
                val_loss += quantile_loss(preds, batch_y, q_levels).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        improved = avg_val < best_val_loss
        if improved:
            best_val_loss = avg_val
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": best_val_loss}, ckpt_path)

        epoch_bar.set_postfix(
            train=f"{avg_train:.5f}",
            val=f"{avg_val:.5f}",
            best=f"{best_val_loss:.5f}",
            saved="✓" if improved else "",
        )

    # Restore the best weights before returning
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    tqdm.write(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} "
               f"(val={best_ckpt['val_loss']:.5f})")

    return model.cpu()  # return on CPU so eval/plot functions work without device awareness


def evaluate_model_X_text(model, X_test, y_test, fig_dir):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test).float())
        q05, q50, q95 = preds[:, 0], preds[:, 1], preds[:, 2]
        uncertainty = (q95 - q05).numpy()
        residuals = (y_test - q50.numpy())
    
    q50_np = q50.numpy()
    lower_err = np.clip(q50_np - q05.numpy(), 0, None)
    upper_err = np.clip(q95.numpy() - q50_np, 0, None)
    
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
    df = pd.read_csv(data_path)
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


# ─── Main analysis entry point ─────────────────────────────────────────────────
def main():
    storage_dir = get_storage_dir()
    fdir      = storage_dir / 'results/csv_collection'
    data_name = 'combined_2020_dates.csv'
    data_name = 'combined_2020-01-01_all_orbits.csv'  # for quick testing with one date's data
    output_dir = storage_dir / 'results/model_ft_transformer'
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, features, qt = training_data_load(
        fdir, data_name, sfc_type=0
    )
    # sys.exit()
    
    if os.path.exists(os.path.join(output_dir, "model_best.pt")):
        print("Found existing checkpoint. Loading model...")
        device = torch.device("cpu")
        model = UncertainBiasPredictorFT(n_features=len(features)).to(device)
        ckpt = torch.load(os.path.join(output_dir, "model_best.pt"), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} with val_loss={ckpt['val_loss']:.5f}")
        
        with open(os.path.join(output_dir, "qt_transformer.pkl"), "rb") as fh:
            qt = pickle.load(fh)
        
    else:
        if platform.system() == "Darwin":
            epochs = 50  # Fewer epochs for local testing
        else:
            epochs = 200  # More epochs for CURC training
        
        model = train_transformer(
            X_train, y_train, X_test, y_test,
            features=features,
            output_dir=str(output_dir),
            d_token=64, n_heads=8, n_layers=3, d_ff=128,
            batch_size=256, n_epochs=epochs,
        )

        # ── Persist model + fitted transformer for inference ──────────────────────
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_fianl_epochs_{epochs}.pt"))
        with open(os.path.join(output_dir, "qt_transformer.pkl"), "wb") as fh:
            pickle.dump(qt, fh)
 
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

    evaluate_model_X_text(model, X_test, y_test, fig_dir=output_dir)
    
    

if __name__ == "__main__":
    main()

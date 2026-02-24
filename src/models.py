import h5py
import numpy as np
import pandas as pd
import os
import sys
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

def plot_uncertainty_by_glint(model, X_val, glint_angles_raw, output_dir):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val).float())
        q05, q50, q95 = preds[:, 0], preds[:, 1], preds[:, 2]
        uncertainty = (q95 - q05).numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(glint_angles_raw, uncertainty, alpha=0.1)
    ax.set_xlabel("Glint Angle (Degrees)")
    ax.set_ylabel("Prediction Uncertainty (Bias Range)")
    ax.set_title("Model Confidence vs. Glint Geometry")
    plt.show()
    fig.savefig(os.path.join(output_dir, "uncertainty_vs_glint.png"), dpi=150, bbox_inches='tight')

def plot_attention_map(model, sample_input, feature_names, output_dir):
    model.eval()
    with torch.no_grad():
        # Get tokens
        x = model.tokenizer(sample_input.unsqueeze(0))
        # Get attention from the first layer
        _, attn_weights = model.layers[0](x, return_attn=True)
    
    # Average across heads if necessary (MultiheadAttention does this by default if average_attn_weights=True)
    attn_matrix = attn_weights[0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attn_matrix, xticklabels=feature_names, yticklabels=feature_names, 
                annot=True, cmap='viridis', cbar_kws={'label': 'Attention Weight'}, ax=ax)
    ax.set_title("Feature Interaction Strengths (Self-Attention)")
    ax.set_xlabel("Key Features")
    ax.set_ylabel("Query Features")
    plt.show()
    fig.savefig(os.path.join(output_dir, "attention_map.png"), dpi=150, bbox_inches='tight')



def training_data_load(fdir, data_fname, sfc_type=0):
    # 1. Load and filter
    data_path = os.path.join(fdir, data_fname)
    df = pd.read_csv(data_path)
    df = df[df['sfc_type_lt'] == sfc_type]
    df = df[df['snow_flag'] == 0]

    # One-hot encode footprint number into 8 binary columns (fp_0 … fp_7)
    for i in range(8):
        df[f'fp_{i}'] = (df['fp'] == i).astype(int)

    # (Trig transforms already applied in fitting_data_correction.py when the CSV was built)

    # 2. Statistical Quantile Transform
    # We transform everything to a Normal distribution to help the Transformer converge
    features = ['o2a_k1', 'o2a_k2', 'wco2_k1', 'wco2_k2', 'sco2_k1', 'sco2_k2',
                'mu_sza', 'mu_vza', 
                'sin_raa', 'cos_raa', 'cos_theta', 'Phi_cos_theta', 'R_rs_factor', 
                'cos_glint_angle', 'glint_prox',
                'alt', 'alt_std', 'ws',
                'log_P', 'airmass', 'dp', 'dp_psfc_ratio', 'dpfrac', 
                'h2o_scale', 'delT', 
                'co2_grad_del', 'alb_o2a', 'alb_wco2',  'alb_sco2', 
                'fs_rel_0', 
                'co2_ratio_bc', 'h2o_ratio_bc', 
                'csnr_o2a', 'csnr_wco2', 'csnr_sco2', 'dp_abp', 
                'h_cont_o2a', 'h_cont_wco2',  'h_cont_sco2', 
                'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2', 
                'xco2_strong_idp', 'xco2_weak_idp', 
                'aod_total', 'aod_bc', 'aod_dust', 'aod_ice', 'aod_water', 
                'aod_oc', 'aod_seasalt', 'aod_strataer', 'aod_sulfate', 'dws', 
                'dust_height', 'ice_height', 'water_height',
                'snr_o2a', 'snr_wco2', 'snr_sco2', 'pol_ang_rad', 
                's31', 's32', 't700', 'tcwv',
                ]
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    
    X = qt.fit_transform(df[features])
    y = df['bias'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, features, qt

def train_transformer(X_train, y_train, X_test, y_test,
                      features,
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = UncertainBiasPredictorFT(n_features=len(features),
                                         d_token=d_token, n_heads=n_heads,
                                         n_layers=n_layers, d_ff=d_ff)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    q_levels  = [0.05, 0.5, 0.95]

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)          # [batch, 3]
            loss  = quantile_loss(preds, batch_y, q_levels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds    = model(batch_x)
                val_loss += quantile_loss(preds, batch_y, q_levels).item()

        print(f"Epoch {epoch:3d} | "
              f"train={train_loss/len(train_loader):.5f}  "
              f"val={val_loss/len(val_loader):.5f}")

    return model


# ─── Main analysis entry point ─────────────────────────────────────────────────
def main():
    fdir      = 'results/csv_collection'
    data_name = 'combined_data_2020_monthly_1st_day.csv'
    output_dir = fdir

    X_train, X_test, y_train, y_test, features, qt = training_data_load(
        fdir, data_name, sfc_type=0
    )

    model = train_transformer(
        X_train, y_train, X_test, y_test,
        features=features,
        d_token=64, n_heads=8, n_layers=3, d_ff=128,
        batch_size=256, n_epochs=100,
    )

    # ── Evaluate: quantile width vs glint angle ────────────────────────────────
    # glint_prox is the last-processed column; recover raw glint_angle from test set
    glint_col_idx = features.index('glint_prox')
    glint_prox_raw = X_test[:, glint_col_idx]       # already QT-transformed — use for sorting only
    plot_uncertainty_by_glint(model, X_test, glint_prox_raw, output_dir)

    # ── Evaluate: attention map for one sample ─────────────────────────────────
    sample = torch.tensor(X_test[0], dtype=torch.float32)
    plot_attention_map(model, sample, features, output_dir)

    # ── Persist model + fitted transformer for inference ──────────────────────
    import pickle
    torch.save(model.state_dict(), os.path.join(output_dir, "model_ft.pt"))
    with open(os.path.join(output_dir, "qt_transformer.pkl"), "wb") as fh:
        pickle.dump(qt, fh)
 

if __name__ == "__main__":
    main()

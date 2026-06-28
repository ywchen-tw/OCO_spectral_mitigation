"""Deep-ensemble MLP with Gaussian-NLL heads + conformal calibration.

Reference: Lakshminarayanan et al. 2017, "Simple and Scalable Predictive Uncertainty
Estimation using Deep Ensembles" (NeurIPS).  Conformal layer: see models/conformal.py.

Motivation (results/model_comparison/ocean_robustness_comparison.md): under k-fold
unseen-date holdout the plain MLP *ties* TabM on point accuracy and is best in the
left tail — but it has no intervals.  This module gives the MLP intervals two ways
and tests whether the accuracy leader can also be well-calibrated:

  M independent MLP members, each a Gaussian head (mu, log_var) trained by NLL.
  Ensemble mixture:  mu* = mean(mu_m)
                     var* = mean(var_m + mu_m^2) - mu*^2   (epistemic + aleatoric)
  Raw 90% interval:  mu* ± 1.645 * sqrt(var*)              (Gaussian approx)

Then a held-out calibration date-block (carved from the train split) recalibrates the
intervals via split and Mondrian conformal (bins = predicted-mu deciles → targets the
low-prediction tail).  Three metric sets are written per run, sharing the same mu (so
RMSE/MAE/R² are identical; only the intervals differ):
  de_raw_<split>       — raw Gaussian-mixture interval
  de_split_<split>     — global split conformal
  de_mondrian_<split>  — regime-conditional (mu-decile) conformal   ← the headline

All intervals are monotone by construction (crossing_rate = 0).
"""

import argparse
import gc
import json
import logging
import os
import pickle
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .pipeline import FeaturePipeline, _ensure_derived_features, filter_target_outliers
from .splits import split_dataframe
from . import conformal as cf
from . import diagnostics as diag
from search.tracking import RunSummary, get_git_commit_hash
from utils import get_storage_dir

logger = logging.getLogger(__name__)

Z90 = 1.6448536269514722  # Gaussian 90% two-sided


class GaussianMLP(nn.Module):
    """n_features → 64 → ReLU → 32 → ReLU → (mu, raw2)[, cloud_logit].

    raw2 is interpreted as log_var (gaussian / beta_nll) or log_scale (student_t)
    by the selected loss; the architecture is identical so checkpoints are
    interchangeable across losses.

    aux_cloud adds a second linear head off the shared body that predicts the
    near-cloud (cld_dist_km <= near_cloud_km) binary logit — a multi-task
    auxiliary task that injects cloud-contamination structure into the shared
    representation (validated to lift near-cloud XCO2 accuracy in the tabm
    ablation).  When off, the architecture/state_dict is identical to the
    single-task model, so existing checkpoints load unchanged.
    """

    def __init__(self, n_features: int, hidden_dims=(64, 32),
                 aux_cloud: bool = False):
        super().__init__()
        dims = [n_features] + list(hidden_dims)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.body = nn.Sequential(*layers)          # default (64,32) == prior arch:
        self.head = nn.Linear(dims[-1], 2)          # body.0/body.2 Linear, head — keys match
        self.aux_cloud = aux_cloud
        if aux_cloud:
            self.cloud_head = nn.Linear(dims[-1], 1)

    def forward(self, x):
        h = self.body(x)
        out = self.head(h)
        mu = out[:, 0]
        raw2 = torch.clamp(out[:, 1], min=-10.0, max=10.0)  # in [~4.5e-5, ~2.2e4]
        if self.aux_cloud:
            return mu, raw2, self.cloud_head(h).squeeze(-1)
        return mu, raw2


def gaussian_nll(mu, log_var, y):
    return 0.5 * (log_var + (y - mu) ** 2 / torch.exp(log_var))


def beta_nll(mu, log_var, y, beta=0.5):
    """Seitzer 2022: scale each point's Gaussian-NLL by stop-grad(var**beta).

    beta=0 recovers plain NLL; beta=1 recovers MSE-on-mean weighting.  Restores
    the mean-fitting gradient in high-variance (near-cloud) regions that plain
    NLL down-weights.
    """
    var = torch.exp(log_var)
    nll = 0.5 * (log_var + (y - mu) ** 2 / var)
    return var.detach() ** beta * nll


def student_t_nll(mu, log_scale, y, nu):
    """Negative log-likelihood of a Student-t(location=mu, scale=exp(log_scale), df=nu).

    Heavy tails let the model represent the heavy-tailed near-cloud residuals
    honestly and make mu robust to tail outliers (vs Gaussian's squared pull).
    nu is a fixed hyperparameter (>2 so the variance exists).
    """
    import math
    z = (y - mu) / torch.exp(log_scale)
    const = (math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)
             - 0.5 * math.log(nu * math.pi))
    log_lik = const - log_scale - 0.5 * (nu + 1.0) * torch.log1p(z ** 2 / nu)
    return -log_lik


def _make_criterion(loss: str, nu: float, beta: float):
    """Return crit(mu, raw2, y, w=None) → scalar.

    The per-element losses above are reduced here so an optional per-sample
    weight `w` can tilt the (weighted) mean toward a regime of interest (e.g.
    near-cloud rows; see --near_cloud_weight).  w=None recovers the plain mean.
    """
    if loss == 'gaussian_nll':
        per = lambda mu, r, y: gaussian_nll(mu, r, y)
    elif loss == 'beta_nll':
        per = lambda mu, r, y: beta_nll(mu, r, y, beta)
    elif loss == 'student_t':
        per = lambda mu, r, y: student_t_nll(mu, r, y, nu)
    else:
        raise ValueError(f"unknown loss {loss!r}")

    def crit(mu, r, y, w=None):
        l = per(mu, r, y)
        if w is None:
            return l.mean()
        return (w * l).sum() / w.sum()   # weighted mean keeps the loss scale stable
    return crit


def _raw_to_var(raw2: np.ndarray, loss: str, nu: float) -> np.ndarray:
    """Convert the head's second output to predictive variance for intervals."""
    if loss == 'student_t':                       # raw2 = log_scale; Var = b^2 * nu/(nu-2)
        return np.exp(2.0 * raw2) * (nu / (nu - 2.0))
    return np.exp(raw2)                            # raw2 = log_var


def _train_member(X_tr, y_tr, X_val, y_val, n_features, *, seed, device,
                  batch_size, n_epochs, patience, ckpt,
                  loss='gaussian_nll', nu=4.0, beta=0.5, w_tr=None, w_val=None,
                  c_tr=None, c_val=None, cloud_aux_weight=0.0, cloud_pos_weight=None,
                  hidden_dims=(64, 32)):
    crit = _make_criterion(loss, nu, beta)
    aux_cloud = cloud_aux_weight > 0.0
    torch.manual_seed(seed); np.random.seed(seed)
    if w_tr is None:
        w_tr = np.ones(len(y_tr), dtype=np.float32)
    if w_val is None:
        w_val = np.ones(len(y_val), dtype=np.float32)
    if c_tr is None:
        c_tr = np.zeros(len(y_tr), dtype=np.float32)
    if c_val is None:
        c_val = np.zeros(len(y_val), dtype=np.float32)
    tr = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr),
                       torch.tensor(w_tr), torch.tensor(c_tr))
    va = TensorDataset(torch.tensor(X_val), torch.tensor(y_val),
                       torch.tensor(w_val), torch.tensor(c_val))
    pin = device.type in ("cuda", "mps"); nw = min(8, os.cpu_count() or 1)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, pin_memory=pin,
                    num_workers=nw, persistent_workers=nw > 0)
    vl = DataLoader(va, batch_size=batch_size, shuffle=False, pin_memory=pin,
                    num_workers=nw, persistent_workers=nw > 0)
    model = GaussianMLP(n_features, hidden_dims=hidden_dims, aux_cloud=aux_cloud).to(device)
    bce = None
    if aux_cloud:
        pw = (torch.tensor(float(cloud_pos_weight), device=device)
              if cloud_pos_weight is not None else None)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                total_steps=n_epochs * len(tl), pct_start=0.05, div_factor=25,
                final_div_factor=1000)
    best, no_imp = float("inf"), 0
    for epoch in range(n_epochs):
        model.train()
        for xb, yb, wb, cb in tl:
            xb, yb, wb, cb = xb.to(device), yb.to(device), wb.to(device), cb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss_val = crit(out[0], out[1], yb, wb)
            if bce is not None:
                loss_val = loss_val + cloud_aux_weight * bce(out[2], cb)
            loss_val.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for xb, yb, wb, cb in vl:
                xb, yb, wb, cb = xb.to(device), yb.to(device), wb.to(device), cb.to(device)
                out = model(xb)
                v = crit(out[0], out[1], yb, wb)
                if bce is not None:
                    v = v + cloud_aux_weight * bce(out[2], cb)
                vloss += v.item()
        vloss /= len(vl)
        if vloss < best:
            best, no_imp = vloss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1
        if patience is not None and no_imp >= patience:
            break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.cpu()


def _member_predict(model, X, device, batch_size=8192, loss='gaussian_nll', nu=4.0):
    model = model.to(device); model.eval()
    mus, raws = [], []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.tensor(X[s:s + batch_size], dtype=torch.float32).to(device)
            out = model(xb)                       # (mu, raw2) or (mu, raw2, cloud_logit)
            mu, raw2 = out[0], out[1]
            mus.append(mu.cpu().numpy()); raws.append(raw2.cpu().numpy())
    return np.concatenate(mus), _raw_to_var(np.concatenate(raws), loss, nu)


def ensemble_predict(members, X, device, loss='gaussian_nll', nu=4.0):
    """Mixture ensemble → (mu*, sigma*).  Per-member variance respects the loss."""
    mu_stack, var_stack = [], []
    for m in members:
        mu, var = _member_predict(m, X, device, loss=loss, nu=nu)
        mu_stack.append(mu); var_stack.append(var)
    mu_stack = np.stack(mu_stack)            # [M, N]
    var_stack = np.stack(var_stack)
    mu_star = mu_stack.mean(0)
    var_star = (var_stack + mu_stack ** 2).mean(0) - mu_star ** 2
    sigma_star = np.sqrt(np.maximum(var_star, 1e-12))
    return mu_star.astype(np.float32), sigma_star.astype(np.float32)


def _member_cloud_prob(model, X, device, batch_size=8192):
    """Per-member near-cloud probability = sigmoid(cloud_logit).  Requires an
    aux_cloud member; returns None for single-task members."""
    if not getattr(model, 'aux_cloud', False):
        return None
    model = model.to(device); model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.tensor(X[s:s + batch_size], dtype=torch.float32).to(device)
            cloud_logit = model(xb)[2]
            probs.append(torch.sigmoid(cloud_logit).cpu().numpy())
    return np.concatenate(probs)


def ensemble_cloud_prob(members, X, device):
    """Ensemble near-cloud probability = mean over members of sigmoid(cloud_logit).
    Returns None if the members have no cloud head."""
    member_probs = [p for p in (_member_cloud_prob(m, X, device) for m in members)
                    if p is not None]
    if not member_probs:
        return None
    return np.stack(member_probs).mean(0).astype(np.float32)


# Cloud-distance bin edges (km) → 5 one-hot groups: [0,2)(2,5)(5,10)(10,15)[15,inf).
# Used as an INPUT feature (oracle = true cld_dist_km bins; the deployable variant
# would feed the classifier's PREDICTED bin instead).  Non-finite cld_dist (no
# cloud found) maps to the far [15,inf) bin.
_CLOUD_BIN_EDGES = np.array([2.0, 5.0, 10.0, 15.0], dtype=float)
N_CLOUD_BINS = len(_CLOUD_BIN_EDGES) + 1


def cloud_bin_onehot(cld_dist_km: np.ndarray) -> np.ndarray:
    """[N] cld_dist_km → [N, 5] one-hot of the distance group."""
    cd = np.where(np.isfinite(cld_dist_km), cld_dist_km, np.inf)
    idx = np.digitize(cd, _CLOUD_BIN_EDGES)            # 0..4
    oh = np.zeros((len(cd), N_CLOUD_BINS), dtype=np.float32)
    oh[np.arange(len(cd)), idx] = 1.0
    return oh


def main():
    p = argparse.ArgumentParser(description="Deep-ensemble MLP + conformal calibration.")
    p.add_argument('--sfc_type', type=int, default=0)
    p.add_argument('--val_split', type=str, default='random',
                   choices=['random', 'date', 'date_kfold'])
    p.add_argument('--n_folds', type=int, default=None)
    p.add_argument('--fold', type=int, default=None)
    p.add_argument('--feature_set', type=str, default='full',
                   choices=['full', 'no_xco2', 'no_spec', 'no_xco2_and_spec',
                            'full_fitqual', 'full_contam', 'full_contam_snow'])
    p.add_argument('--include_snow', action='store_true',
                   help="Keep snow/ice footprints (snow_flag==1) in train/cal/holdout "
                        "instead of the default filter to snow_flag==0.  Required for the "
                        "full_contam_snow feature set to be meaningful (else the flag is "
                        "constant).  Snow is land-only, so this only affects sfc_type=1.")
    p.add_argument('--n_members', type=int, default=5)
    p.add_argument('--hidden_dims', type=str, default='64,32',
                   help="Comma-separated GaussianMLP hidden layer widths. Default "
                        "'64,32' (the current 2-layer arch). e.g. '128,64,32' adds a "
                        "layer; '256,128,64' goes wider+deeper. Capacity sweep knob.")
    p.add_argument('--loss', type=str, default='gaussian_nll',
                   choices=['gaussian_nll', 'beta_nll', 'student_t'],
                   help="Member loss: gaussian_nll (default), beta_nll (Seitzer "
                        "mean-fit fix), or student_t (heavy-tailed, for the "
                        "near-cloud tail).")
    p.add_argument('--nu', type=float, default=4.0,
                   help='Student-t degrees of freedom (fixed; >2). Lower = heavier tails.')
    p.add_argument('--beta', type=float, default=0.5,
                   help='beta_nll weighting exponent (0=NLL, 1=MSE-on-mean).')
    p.add_argument('--batch_size', type=int, default=None,
                   help='Override platform default (Darwin 2048 / Linux 4096).')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override platform default (Darwin 100 / Linux 500).')
    p.add_argument('--calib_frac', type=float, default=0.15,
                   help='Fraction of TRAIN dates carved out as the conformal '
                        'calibration block (date split when possible).')
    p.add_argument('--near_cloud_target', type=float, default=None,
                   help="If set (e.g. 0.975), raise the conformal target in the "
                        "near-cloud Mondrian bins (<= --near_cloud_km) to over-cover "
                        "the outcome-defined near-cloud tail; far bins stay at 0.90. "
                        "Requires --mondrian_col cld_dist_km.")
    p.add_argument('--near_cloud_km', type=float, default=10.0,
                   help="Cloud-distance threshold (km) defining 'near' rows, for "
                        "both --near_cloud_target and --near_cloud_weight.")
    p.add_argument('--near_cloud_weight', type=float, default=1.0,
                   help="Per-sample training-loss weight for near-cloud rows "
                        "(cld_dist_km <= --near_cloud_km).  1.0 = off (uniform). "
                        ">1 upweights the near-cloud regime so the global loss is "
                        "not dominated by the far-cloud majority (~81%% of rows), "
                        "aligning training/early-stopping with near-cloud point "
                        "accuracy.  Applied to BOTH train and the calibration block "
                        "used for early stopping.")
    p.add_argument('--cloud_aux_weight', type=float, default=0.0,
                   help="Multi-task auxiliary loss weight lambda for a near-cloud "
                        "(cld_dist_km <= --near_cloud_km) binary classification head "
                        "on the shared backbone.  0.0 = off (single-task; architecture "
                        "and checkpoints identical to before).  >0 adds lambda*BCE to "
                        "each member's loss; the cloud task injects cloud-contamination "
                        "structure that lifts near-cloud XCO2 accuracy, and the head "
                        "also yields an ensemble near-cloud probability (AUC/AP reported "
                        "when truth is present).")
    p.add_argument('--cloud_bin_feature', choices=['none', 'oracle', 'predicted'],
                   default='none',
                   help="Append a 5-way one-hot of the cld_dist_km group "
                        "([0,2)(2,5)(5,10)(10,15)[15,inf) km) to the INPUT features. "
                        "'oracle' = TRUE cld_dist_km (independent MODIS info; upper "
                        "bound, only deployable where MODIS collocation exists). "
                        "'predicted' = bin from an internal GBDT classifier trained on "
                        "the SAME features (deployable without MODIS; recovers only the "
                        "cloud signal present in the spectra).  'none' (default) = off.")
    p.add_argument('--mondrian_bins', type=int, default=10)
    p.add_argument('--mondrian_col', type=str, default='mu',
                   help="Observable variable for Mondrian bins: 'mu' (predicted-mean "
                        "deciles) or a column name, e.g. 'cld_dist_km' / 'aod_total' "
                        "(physical proxy for the cloud-contaminated tail).")
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--suffix', type=str, default='')
    p.add_argument('--seed', type=int, default=42, help='Base seed; member m uses seed+m.')
    p.add_argument('--data', type=str, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    storage_dir = get_storage_dir()
    fdir = storage_dir / 'results/csv_collection'
    data_name = ('combined_2016_2020_dates.parquet' if platform.system() == "Linux"
                 else 'combined_2020-02-01_all_orbits.parquet')
    base_dir = storage_dir / 'results/model_deep_ensemble'
    output_dir = base_dir / args.suffix if args.suffix else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.monotonic()
    run_id = args.suffix or datetime.now().strftime('%Y%m%d-%H%M%S')
    commit = get_git_commit_hash(storage_dir)

    _dp = args.data if args.data else os.path.join(fdir, data_name)
    df = pd.read_parquet(_dp) if _dp.endswith('.parquet') else pd.read_csv(_dp)
    df = df[df['sfc_type'] == args.sfc_type]
    if args.include_snow:
        logger.info("--include_snow: keeping snow_flag==1 footprints (%d of %d rows are snow)",
                    int((df['snow_flag'] == 1).sum()), len(df))
    else:
        df = df[df['snow_flag'] == 0]
    df = _ensure_derived_features(df)
    df = filter_target_outliers(df)

    train_df, held_df = split_dataframe(df, mode=args.val_split, test_size=args.test_size,
                                        random_state=args.seed,
                                        n_folds=args.n_folds, fold=args.fold)
    del df; gc.collect()

    # Carve a calibration block out of TRAIN (date split if dates allow, else random).
    try:
        if 'date' in train_df.columns and pd.to_datetime(
                train_df['date'].astype(str).str.replace("b'", "").str.replace("'", "")
                if train_df['date'].dtype == object else train_df['date']).nunique() >= 2:
            proper_df, calib_df = split_dataframe(train_df, mode='date', test_size=args.calib_frac)
        else:
            proper_df, calib_df = split_dataframe(train_df, mode='random',
                                                  test_size=args.calib_frac, random_state=args.seed)
    except Exception:
        proper_df, calib_df = split_dataframe(train_df, mode='random',
                                              test_size=args.calib_frac, random_state=args.seed)
    del train_df; gc.collect()

    pipeline = FeaturePipeline.fit(proper_df, sfc_type=args.sfc_type, feature_set=args.feature_set)
    pipeline.save(output_dir / 'deep_ensemble_pipeline.pkl')

    def _prep(frame):
        X = pipeline.transform(frame)
        y = frame['xco2_bc_anomaly'].to_numpy(dtype=np.float32)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        return X[valid], y[valid], frame.loc[valid]

    X_tr, y_tr, train_valid = _prep(proper_df)
    X_cal, y_cal, calib_valid = _prep(calib_df)
    X_te, y_te, held_valid = _prep(held_df)

    # Optional cloud-distance-bin INPUT feature (oracle = true bins; predicted = an
    # internal GBDT classifier on the same features, deployable without MODIS).
    if args.cloud_bin_feature != 'none':
        def _true_bins(frame):
            return np.digitize(np.where(np.isfinite(frame['cld_dist_km'].to_numpy(float)),
                                        frame['cld_dist_km'].to_numpy(float), np.inf),
                               _CLOUD_BIN_EDGES)
        if args.cloud_bin_feature == 'oracle':
            oh_tr, oh_cal, oh_te = (cloud_bin_onehot(f['cld_dist_km'].to_numpy(float))
                                    for f in (train_valid, calib_valid, held_valid))
        else:                                            # 'predicted'
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.metrics import accuracy_score
            b_tr = _true_bins(train_valid)
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                 max_depth=8, random_state=args.seed)
            clf.fit(X_tr, b_tr)
            p_tr, p_cal, p_te = clf.predict(X_tr), clf.predict(X_cal), clf.predict(X_te)
            acc = accuracy_score(_true_bins(held_valid), p_te)
            print(f"[deep_ensemble] predicted cloud-bin classifier held 5-class "
                  f"acc={acc:.4f} (chance≈0.2); train acc={accuracy_score(b_tr, p_tr):.4f}")
            def _oh(idx):
                oh = np.zeros((len(idx), N_CLOUD_BINS), dtype=np.float32)
                oh[np.arange(len(idx)), idx] = 1.0
                return oh
            oh_tr, oh_cal, oh_te = _oh(p_tr), _oh(p_cal), _oh(p_te)
        X_tr = np.concatenate([X_tr, oh_tr], axis=1).astype(np.float32)
        X_cal = np.concatenate([X_cal, oh_cal], axis=1).astype(np.float32)
        X_te = np.concatenate([X_te, oh_te], axis=1).astype(np.float32)
        print(f"[deep_ensemble] cloud_bin_feature={args.cloud_bin_feature}: "
              f"+{N_CLOUD_BINS} one-hot dims (pipeline {pipeline.n_features} → "
              f"model {X_tr.shape[1]})")
    n_features = X_tr.shape[1]
    print(f"[deep_ensemble] proper-train {X_tr.shape}  calib {X_cal.shape}  held {X_te.shape}")

    # Near-cloud loss weighting: tilt the (weighted) objective toward the
    # cld_dist_km <= near_cloud_km regime so the far-cloud majority does not
    # dominate the gradient.  w=1 everywhere when --near_cloud_weight == 1.0.
    def _near_cloud_weights(frame):
        w = np.ones(len(frame), dtype=np.float32)
        if args.near_cloud_weight == 1.0:
            return w
        if 'cld_dist_km' not in frame.columns:
            raise ValueError("--near_cloud_weight requires a 'cld_dist_km' column")
        cd = frame['cld_dist_km'].to_numpy(dtype=float)
        w[np.isfinite(cd) & (cd <= args.near_cloud_km)] = args.near_cloud_weight
        return w
    w_tr = _near_cloud_weights(train_valid)
    w_cal = _near_cloud_weights(calib_valid)
    if args.near_cloud_weight != 1.0:
        print(f"[deep_ensemble] near_cloud_weight={args.near_cloud_weight} on "
              f"cld_dist_km<={args.near_cloud_km}km: "
              f"{int((w_tr > 1).sum())}/{len(w_tr)} train rows "
              f"({100 * (w_tr > 1).mean():.1f}%) upweighted")

    # Multi-task auxiliary cloud labels: binary near-cloud (cld_dist_km <= km).
    # Off (c=None, pos_weight=None) when --cloud_aux_weight == 0.
    def _cloud_labels(frame):
        if 'cld_dist_km' not in frame.columns:
            raise ValueError("--cloud_aux_weight requires a 'cld_dist_km' column")
        cd = frame['cld_dist_km'].to_numpy(dtype=float)
        return (np.isfinite(cd) & (cd <= args.near_cloud_km)).astype(np.float32)
    c_tr = c_cal = None
    cloud_pos_weight = None
    if args.cloud_aux_weight > 0.0:
        c_tr = _cloud_labels(train_valid)
        c_cal = _cloud_labels(calib_valid)
        n_pos = float(c_tr.sum()); n_neg = float(len(c_tr) - n_pos)
        cloud_pos_weight = n_neg / max(n_pos, 1.0)   # balance the BCE
        print(f"[deep_ensemble] cloud_aux_weight={args.cloud_aux_weight} "
              f"(near<={args.near_cloud_km}km): {int(n_pos)}/{len(c_tr)} train rows "
              f"positive ({100 * n_pos / len(c_tr):.1f}%), pos_weight={cloud_pos_weight:.3f}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    epochs, batch_size = (100, 2048) if platform.system() == "Darwin" else (500, 4096)
    if args.epochs is not None:
        epochs = args.epochs
    if args.batch_size is not None:
        batch_size = args.batch_size
    # internal val for early stopping = the calibration block (it is held out of training)
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(',') if x.strip())
    members = []
    for m in range(args.n_members):
        ck = str(output_dir / f'member_{m}.pt')
        print(f"  training member {m+1}/{args.n_members} (seed={args.seed + m})")
        members.append(_train_member(X_tr, y_tr, X_cal, y_cal, n_features,
                                      seed=args.seed + m, device=device, batch_size=batch_size,
                                      n_epochs=epochs, patience=50, ckpt=ck,
                                      loss=args.loss, nu=args.nu, beta=args.beta,
                                      w_tr=w_tr, w_val=w_cal,
                                      c_tr=c_tr, c_val=c_cal,
                                      cloud_aux_weight=args.cloud_aux_weight,
                                      cloud_pos_weight=cloud_pos_weight,
                                      hidden_dims=hidden_dims))

    mu_cal, sig_cal = ensemble_predict(members, X_cal, device, loss=args.loss, nu=args.nu)
    mu_te, sig_te = ensemble_predict(members, X_te, device, loss=args.loss, nu=args.nu)

    # ── 1) raw Gaussian-mixture interval ───────────────────────────────────────
    preds_raw = np.column_stack([mu_te - Z90 * sig_te, mu_te, mu_te + Z90 * sig_te])
    # ── 2) global split conformal ──────────────────────────────────────────────
    preds_split, q_split = cf.split_conformal(y_cal, mu_cal, sig_cal, mu_te, sig_te, alpha=0.10)
    # ── 3) Mondrian conformal, binned by an OBSERVABLE variable ────────────────
    # 'mu' = predicted-mean deciles (does not isolate the y-defined tail, since the
    # model under-predicts it).  A physical proxy ('cld_dist_km' / 'aod_total') bins
    # by the cause of the cloud-contaminated tail and gives that regime its own q.
    def _bin_values(which, frame, mu):
        if which == 'mu':
            return np.asarray(mu, dtype=float)
        if which not in frame.columns:
            raise ValueError(f"--mondrian_col {which!r} not in dataframe columns.")
        v = frame[which].to_numpy(dtype=float)
        if not np.all(np.isfinite(v)):              # e.g. cld_dist_km can be NaN/inf
            v = np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)]))
        return v
    bin_cal = _bin_values(args.mondrian_col, calib_valid, mu_cal)
    bin_te = _bin_values(args.mondrian_col, held_valid, mu_te)
    cal_bin, edges = cf.make_quantile_bins(bin_cal, args.mondrian_bins)
    te_bin, _ = cf.make_quantile_bins(bin_te, args.mondrian_bins, edges=edges)
    # Optionally over-cover the near-cloud bins (the only lever for the
    # outcome-defined near-cloud tail, which a flat target cannot guarantee).
    bin_alpha = None
    if args.near_cloud_target is not None:
        if args.mondrian_col != 'cld_dist_km':
            raise ValueError("--near_cloud_target requires --mondrian_col cld_dist_km")
        is_near_cal = calib_valid['cld_dist_km'].to_numpy(dtype=float) <= args.near_cloud_km
        bin_alpha = cf.regime_alphas(cal_bin, is_near_cal,
                                     near_alpha=1.0 - args.near_cloud_target, far_alpha=0.10)
        print(f"  near-cloud target {args.near_cloud_target} (<= {args.near_cloud_km}km): "
              f"{sum(a < 0.10 for a in bin_alpha.values())}/{len(bin_alpha)} bins elevated")
    preds_mond, q_by_bin = cf.mondrian_conformal(y_cal, mu_cal, sig_cal, cal_bin,
                                                 mu_te, sig_te, te_bin, alpha=0.10,
                                                 bin_alpha=bin_alpha)

    results = {}
    for tag, preds in [('raw', preds_raw), ('split', preds_split), ('mondrian', preds_mond)]:
        g = diag.compute_metrics(y_te, preds)
        strat = diag.stratified_metrics(held_valid, y_te, preds)
        prefix = f"de_{tag}_{args.val_split}"
        diag.save_diagnostics(output_dir, prefix, g, strat)
        results[tag] = g
        print(f"[{prefix}] RMSE={g['rmse']:.4f} R²={g['r2']:.4f} "
              f"cov90={g['coverage_90']:.4f} width={g['mean_interval_width']:.4f} "
              f"cross={g['crossing_rate']}")

    # ── correction effectiveness vs cloud distance (point pred is shared) ──────
    # 'pre' = |y| (uncorrected anomaly), 'post' = |y - mu| (after applying the
    # predicted correction).  This is the deployment metric for the near-cloud
    # bins where corrections are actually applied.
    corr = diag.correction_by_cloud_distance(held_valid, y_te, mu_te)
    if not corr.empty:
        corr_path = output_dir / f'de_correction_clddist_{args.val_split}.csv'
        corr.to_csv(corr_path, index=False)
        print(f"Saved correction-vs-cloud-distance → {corr_path}")
        print(corr[['bin', 'n', 'pre_rms', 'post_rms', 'rms_reduction_pct', 'r2']]
              .to_string(index=False))

    # ── auxiliary near-cloud classification head (multi-task) ──────────────────
    # Ensemble near-cloud probability + its quality vs the true cld_dist_km<=km
    # label.  Only present when --cloud_aux_weight > 0 (members have a cloud head).
    cloud_metrics = {}
    cloud_prob_te = ensemble_cloud_prob(members, X_te, device)
    if cloud_prob_te is not None and 'cld_dist_km' in held_valid.columns:
        from sklearn.metrics import roc_auc_score, average_precision_score
        cd_te = held_valid['cld_dist_km'].to_numpy(dtype=float)
        c_true = (np.isfinite(cd_te) & (cd_te <= args.near_cloud_km)).astype(int)
        if 0 < int(c_true.sum()) < len(c_true):          # need both classes
            auc = float(roc_auc_score(c_true, cloud_prob_te))
            ap = float(average_precision_score(c_true, cloud_prob_te))
            acc = float(((cloud_prob_te >= 0.5).astype(int) == c_true).mean())
            cloud_metrics = {'cloud_auc': auc, 'cloud_ap': ap,
                             'cloud_acc@0.5': acc, 'cloud_pos_rate': float(c_true.mean())}
            print(f"[de_cloud] near<={args.near_cloud_km}km classifier: AUC={auc:.4f} "
                  f"AP={ap:.4f} acc@0.5={acc:.4f} (pos_rate={c_true.mean():.3f})")

    # ── dump per-sounding held-out predictions for post-hoc analysis (no rerun) ─
    held_out_df = pd.DataFrame({'y_true': y_te, 'mu': mu_te, 'sigma': sig_te,
                                'lo_mondrian': preds_mond[:, 0], 'hi_mondrian': preds_mond[:, 2]})
    if cloud_prob_te is not None:
        held_out_df['cloud_prob'] = cloud_prob_te
    for c in ('cld_dist_km', 'sfc_type', 'aod_total', 'fp', 'lat', 'snow_flag'):
        if c in held_valid.columns:
            held_out_df[c] = held_valid[c].to_numpy()
    held_out_df.to_parquet(output_dir / 'held_out_predictions.parquet', index=False)

    with open(output_dir / 'deep_ensemble_meta.pkl', 'wb') as f:
        pickle.dump({'n_features': pipeline.n_features, 'n_members': args.n_members,
                     'q_split': q_split, 'q_by_bin': q_by_bin, 'mondrian_edges': edges.tolist(),
                     'mondrian_col': args.mondrian_col,
                     'feature_set': args.feature_set, 'val_split': args.val_split,
                     'loss': args.loss, 'nu': args.nu, 'beta': args.beta,
                     'near_cloud_target': args.near_cloud_target,
                     'near_cloud_km': args.near_cloud_km,
                     'near_cloud_weight': args.near_cloud_weight,
                     'aux_cloud': args.cloud_aux_weight > 0.0,
                     'cloud_aux_weight': args.cloud_aux_weight,
                     'cloud_bin_feature': args.cloud_bin_feature,
                     'hidden_dims': list(hidden_dims)}, f)

    g = results['mondrian']
    summary = RunSummary(
        run_id=run_id, script_name=os.path.basename(__file__), model_family='deep_ensemble',
        commit=commit, status='success',
        primary_metric_name='de_held_rmse', primary_metric_value=g['rmse'],
        secondary_metrics={'de_held_r2': g['r2'], 'de_mondrian_cov90': g['coverage_90'],
                           'de_split_cov90': results['split']['coverage_90'],
                           'de_raw_cov90': results['raw']['coverage_90'],
                           **cloud_metrics},
        runtime_seconds=float(time.monotonic() - run_start),
        description=f'Deep ensemble M={args.n_members} + conformal, {args.val_split}-split, '
                    f'feature_set={args.feature_set}',
        artifacts={'output_dir': str(output_dir),
                   'metrics_json': str(output_dir / f'de_mondrian_{args.val_split}_metrics.json')},
        config={'sfc_type': args.sfc_type, 'val_split': args.val_split,
                'n_members': args.n_members, 'calib_frac': args.calib_frac,
                'feature_set': args.feature_set, 'seed': args.seed,
                'loss': args.loss, 'nu': args.nu, 'beta': args.beta,
                'cloud_aux_weight': args.cloud_aux_weight},
    )
    with open(output_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, indent=2, sort_keys=True)
    print(f"Saved run summary → {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()

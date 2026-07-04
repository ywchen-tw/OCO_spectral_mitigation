"""Shared training scaffold for the torch models (deep_ensemble, mlp_baseline, tabm).

Single home for the pieces that used to be copy-pasted (with drifting details)
across the three training scripts:

  - TrainConfig      — every optimizer / schedule / early-stop literal in one place
  - select_device()  — device policy incl. the CURC unsupported-CUDA-card guard
  - set_seeds()      — complete seeding (python / numpy / torch + a torch.Generator
                       for DataLoader shuffling), optional deterministic mode
  - make_batches()   — device-resident tensor batches when the data fits on the
                       device (no worker processes, no per-batch host→device
                       copies), transparent DataLoader fallback otherwise
  - train_model()    — the generic AdamW + OneCycle + grad-clip + best-val-
                       checkpoint + early-stop loop

`tabm.py` keeps its specialized inner loop (AMP + per-member quantile losses)
but uses the shared device/seed/optimizer helpers, so the literals cannot
drift again.
"""

import dataclasses
import logging
import math
import os
import platform
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Platform-default (epochs, batch_size): quick local iteration on macOS,
# full-length runs on CURC/Linux.  Previously duplicated as inline literals in
# deep_ensemble.py and mlp_baseline.py.
_PLATFORM_EPOCHS_BATCH = {'Darwin': (100, 2048)}
_DEFAULT_EPOCHS_BATCH = (500, 4096)


def platform_epochs_batch():
    """(epochs, batch_size) default for this platform."""
    return _PLATFORM_EPOCHS_BATCH.get(platform.system(), _DEFAULT_EPOCHS_BATCH)


@dataclasses.dataclass
class TrainConfig:
    """Training hyperparameters shared by all torch models.

    epochs / batch_size default to the platform values (see
    platform_epochs_batch) when left as None; call resolved() to fill them.
    """
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # OneCycleLR shape
    pct_start: float = 0.05
    div_factor: float = 25.0
    final_div_factor: float = 1000.0
    grad_clip: float = 1.0
    patience: Optional[int] = 50
    seed: int = 42
    deterministic: bool = False
    # None = auto: resident when the tensors fit on the device
    gpu_resident: Optional[bool] = None
    num_workers: Optional[int] = None     # DataLoader fallback path only
    log_every: int = 10

    def resolved(self) -> 'TrainConfig':
        ep, bs = platform_epochs_batch()
        return dataclasses.replace(
            self,
            epochs=self.epochs if self.epochs is not None else ep,
            batch_size=self.batch_size if self.batch_size is not None else bs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def cuda_device_supported():
    """True only if the current CUDA GPU's compute capability is in this PyTorch
    build's kernel arch list.  On CURC preemptable QOS a job can land on an older
    card (e.g. Tesla P100, sm_60) whose kernels this PyTorch wasn't compiled for;
    selecting it would crash mid-train with `no kernel image is available`.  When
    unsupported we log and fall back to CPU instead of dying."""
    try:
        major, minor = torch.cuda.get_device_capability()
        cap = major * 10 + minor
        # get_arch_list() → e.g. ['sm_70', 'sm_75', ..., 'sm_120']
        supported = [int(a.split('_')[1]) for a in torch.cuda.get_arch_list()
                     if a.startswith('sm_')]
        if supported and cap < min(supported):
            logger.warning(
                "CUDA GPU %s has compute capability sm_%d, below this PyTorch "
                "build's minimum (%s) — falling back to CPU.",
                torch.cuda.get_device_name(0), cap,
                ", ".join(f"sm_{s}" for s in sorted(supported)))
            return False
        return True
    except Exception as e:  # any probing failure → don't risk a hard crash
        logger.warning("Could not verify CUDA device compatibility (%s) — "
                       "falling back to CPU.", e)
        return False


def select_device() -> torch.device:
    """cuda (if the card is supported by this build) → mps → cpu."""
    if torch.cuda.is_available() and cuda_device_supported():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seeds(seed: int, deterministic: bool = False) -> torch.Generator:
    """Seed python / numpy / torch and return a torch.Generator for shuffling.

    deterministic=True additionally pins cuDNN and torch algorithms (slower;
    use for bit-reproducibility checks, not production runs).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):  # DataLoader worker_init_fn — derive from torch seed
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_optimizer_scheduler(model: nn.Module, cfg: TrainConfig, steps_per_epoch: int):
    """AdamW + OneCycleLR with the shared literals."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch,
        pct_start=cfg.pct_start, div_factor=cfg.div_factor,
        final_div_factor=cfg.final_div_factor)
    return opt, sched


class _ResidentBatches:
    """Mini-batches sliced from device-resident tensors.

    Equivalent to a DataLoader over TensorDataset but with no worker processes
    and no per-batch host→device copies — the tensors live on the device for
    the whole training run.  Shuffling uses the provided (CPU) generator so
    runs are reproducible.
    """

    def __init__(self, tensors, batch_size, shuffle, generator):
        self.tensors = tensors
        self.n = len(tensors[0])
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.generator = generator
        self.device = tensors[0].device

    def __len__(self):
        return max(1, math.ceil(self.n / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n, generator=self.generator).to(self.device)
            for s in range(0, self.n, self.batch_size):
                sel = order[s:s + self.batch_size]
                yield tuple(t[sel] for t in self.tensors)
        else:
            for s in range(0, self.n, self.batch_size):
                yield tuple(t[s:s + self.batch_size] for t in self.tensors)


def _fits_on_device(nbytes: int, device: torch.device) -> bool:
    if device.type == 'cuda':
        try:
            free, _total = torch.cuda.mem_get_info()
            return nbytes < 0.5 * free
        except Exception:
            return False
    if device.type == 'mps':
        return nbytes < 8e9
    return True   # cpu: the arrays are already in RAM


def make_batches(arrays, batch_size, *, shuffle, device, generator,
                 gpu_resident: Optional[bool] = None,
                 num_workers: Optional[int] = None):
    """Batch iterator over a tuple of numpy arrays (first axis = samples).

    gpu_resident None → auto: keep everything on the device when it fits
    (fastest for these tabular models); otherwise fall back to a DataLoader
    with seeded shuffling and worker seeding.
    """
    arrays = tuple(np.ascontiguousarray(a) for a in arrays)
    nbytes = sum(a.nbytes for a in arrays)
    if gpu_resident is None:
        gpu_resident = _fits_on_device(nbytes, device)
    if gpu_resident:
        tensors = tuple(torch.as_tensor(a).to(device) for a in arrays)
        return _ResidentBatches(tensors, batch_size, shuffle, generator)

    ds = TensorDataset(*[torch.as_tensor(a) for a in arrays])
    nw = num_workers if num_workers is not None else min(8, os.cpu_count() or 1)
    pin = device.type in ("cuda", "mps")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=pin,
                      num_workers=nw, persistent_workers=nw > 0,
                      generator=generator if shuffle else None,
                      worker_init_fn=seed_worker if nw > 0 else None)


def train_model(model: nn.Module, criterion, train_batches, val_batches,
                cfg: TrainConfig, ckpt_path: str, device: torch.device,
                log_prefix: str = ''):
    """Generic train loop shared by the tabular torch models.

    criterion(model, batch) -> scalar loss; batch is a tuple of device tensors
    in the order the caller passed to make_batches.  Saves the best-validation
    state_dict (raw — the historical member checkpoint format) to ckpt_path and
    reloads it before returning.
    """
    opt, sched = make_optimizer_scheduler(model, cfg, len(train_batches))
    best, no_imp, best_epoch = float("inf"), 0, -1
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_batches:
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            opt.zero_grad()
            loss = criterion(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in val_batches:
                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                vloss += criterion(model, batch).item()
        vloss /= len(val_batches)
        if vloss < best:
            best, no_imp, best_epoch = vloss, 0, epoch
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_imp += 1
        if cfg.log_every and ((epoch + 1) % cfg.log_every == 0 or epoch == 0):
            logger.info("%s epoch %d/%d val=%.5f best=%.5f (epoch %d)",
                        log_prefix, epoch + 1, cfg.epochs, vloss, best, best_epoch)
        if cfg.patience is not None and no_imp >= cfg.patience:
            logger.info("%s early stop at epoch %d (best %.5f @ %d)",
                        log_prefix, epoch, best, best_epoch)
            break
    if best_epoch < 0 or not os.path.exists(ckpt_path):
        raise RuntimeError(f"{log_prefix} no checkpoint saved to {ckpt_path} — "
                           "validation loss never improved (NaN losses?)")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model, {'best_val': best, 'best_epoch': best_epoch}

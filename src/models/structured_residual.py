"""Physics-block residual network for per-sounding XCO2 anomaly regression.

The model consumes only the scalar output of ``FeaturePipeline``.  It never
uses radiance spectra, neighboring soundings, orbit context, or retrieved
training examples.

Features are partitioned by physical role, encoded independently, and fused by
a small residual MLP.  ``xco2_raw_minus_apriori`` additionally receives a
direct linear path to the predicted mean:

    mu = anchor(xco2_raw_minus_apriori) + residual(all encoded blocks)

This preserves a stable route for the strongest physically meaningful feature
while still allowing nonlinear interactions through the residual branch.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn as nn

from ._backbone_utils import dense_stack


_ANCHOR_FEATURE = "xco2_raw_minus_apriori"
_PROFILE_PREFIXES = ("t_pc", "q_pc", "co2prior_pc", "pca_pc")
_PROFILE_SCALARS = {"tropopause_sigma", "tropopause_temp"}
_SPECTRAL_PREFIXES = ("o2a_k", "wco2_k", "sco2_k")
_SPECTRAL_FEATURES = {
    "exp_o2a_intercept",
    "o2a_exp_intercept-alb",
    "wco2_exp_intercept-alb",
}
_GEOMETRY_FEATURES = {
    "cos_glint_angle",
    "1_over_cos_sza",
    "1_over_cos_vza",
    "sin_raa",
    "pol_ang_rad",
    "fp_area_km2",
}
_CONTAMINATION_FEATURES = {
    "max_declock_wco2",
    "dp_abp",
    "h_cont_o2a",
    "h_cont_wco2",
    "h_cont_sco2",
    "dpfrac",
    "fs_rel_0",
    "dust_height",
    "ice_height",
    "water_height",
    "alt_std",
    "alb_sco2_over_wco2",
}


def feature_block(name: str) -> str:
    """Return the physical block for one transformed pipeline feature."""
    if name == _ANCHOR_FEATURE:
        return "xco2"
    if name.startswith(_PROFILE_PREFIXES) or name in _PROFILE_SCALARS:
        return "profile"
    if name.startswith(_SPECTRAL_PREFIXES) or name in _SPECTRAL_FEATURES:
        return "spectroscopy"
    if name.startswith("fp_") or name.startswith("cloud_bin_"):
        return "geometry"
    if name in _GEOMETRY_FEATURES:
        return "geometry"
    if name.startswith("aod_") or name in _CONTAMINATION_FEATURES:
        return "contamination"
    return "state"


def group_feature_indices(feature_names: Iterable[str]) -> dict[str, list[int]]:
    """Partition feature positions into stable, non-overlapping physical blocks."""
    groups: dict[str, list[int]] = OrderedDict(
        (name, [])
        for name in (
            "xco2",
            "spectroscopy",
            "contamination",
            "state",
            "profile",
            "geometry",
        )
    )
    for idx, name in enumerate(feature_names):
        groups[feature_block(name)].append(idx)
    return {name: idx for name, idx in groups.items() if idx}


class StructuredResidualNet(nn.Module):
    """Per-block encoders plus an explicit XCO2-to-mean residual path."""

    def __init__(
        self,
        feature_names: list[str],
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        block_dim: int = 16,
        aux_cloud: bool = False,
        dropout: float = 0.0,
        norm: str = "none",
    ):
        super().__init__()
        if block_dim < 1:
            raise ValueError(f"block_dim must be positive, got {block_dim}")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("feature_names must be unique")

        self.feature_names = list(feature_names)
        self.feature_groups = group_feature_indices(feature_names)
        self.aux_cloud = bool(aux_cloud)
        self.block_dim = int(block_dim)

        self.encoders = nn.ModuleDict()
        self._index_buffers: dict[str, str] = {}
        for block, indices in self.feature_groups.items():
            buffer_name = f"_idx_{block}"
            self.register_buffer(
                buffer_name,
                torch.tensor(indices, dtype=torch.long),
                persistent=False,
            )
            self._index_buffers[block] = buffer_name
            self.encoders[block] = dense_stack(
                len(indices),
                (self.block_dim,),
                dropout=dropout,
                norm=norm,
            )

        fused_dim = self.block_dim * len(self.encoders)
        self.body = dense_stack(
            fused_dim,
            tuple(hidden_dims),
            dropout=dropout,
            norm=norm,
        )
        self.head = nn.Linear(hidden_dims[-1], 2)

        anchor_indices = self.feature_groups.get("xco2", [])
        self.anchor_head = (
            nn.Linear(1, 1, bias=False) if len(anchor_indices) == 1 else None
        )
        if self.aux_cloud:
            self.cloud_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor):
        encoded = []
        for block, encoder in self.encoders.items():
            indices = getattr(self, self._index_buffers[block])
            encoded.append(encoder(torch.index_select(x, 1, indices)))
        h = self.body(torch.cat(encoded, dim=1))
        out = self.head(h)

        mu = out[:, 0]
        if self.anchor_head is not None:
            anchor_idx = getattr(self, self._index_buffers["xco2"])
            anchor = torch.index_select(x, 1, anchor_idx)
            mu = mu + self.anchor_head(anchor).squeeze(-1)
        raw2 = torch.clamp(out[:, 1], min=-10.0, max=10.0)

        if self.aux_cloud:
            return mu, raw2, self.cloud_head(h).squeeze(-1)
        return mu, raw2

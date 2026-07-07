"""Regime-aware structured residual network.

This backbone keeps the same per-sounding, scalar-feature constraint as
``StructuredResidualNet``: no spectra arrays, no neighboring soundings, and no
retrieval examples at inference time.  The difference is that the residual
branch is a small learned mixture of experts:

    mu = anchor(xco2_raw_minus_apriori) + Σ_k gate_k(x) * expert_k(x)

The gate is learned from the physical feature blocks already present in
``FeaturePipeline``.  In practice this lets each ocean/land model learn soft
regimes such as geometry-dominated, aerosol/cloud-contaminated, profile-driven,
or spectroscopy-driven cases without hard-coding a regime variable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._backbone_utils import dense_stack
from .structured_residual import group_feature_indices


class RegimeStructuredResidualNet(nn.Module):
    """Physical block encoders plus a soft mixture of residual experts."""

    def __init__(
        self,
        feature_names: list[str],
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        block_dim: int = 16,
        n_experts: int = 4,
        dropout: float = 0.0,
        norm: str = "none",
    ):
        super().__init__()
        if block_dim < 1:
            raise ValueError(f"block_dim must be positive, got {block_dim}")
        if n_experts < 2:
            raise ValueError(f"n_experts must be at least 2, got {n_experts}")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("feature_names must be unique")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one width")

        self.feature_names = list(feature_names)
        self.feature_groups = group_feature_indices(feature_names)
        self.block_dim = int(block_dim)
        self.n_experts = int(n_experts)

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
        body_dim = hidden_dims[-1]
        self.gate_head = nn.Linear(body_dim, self.n_experts)
        self.expert_mu_head = nn.Linear(body_dim, self.n_experts)
        self.raw2_head = nn.Linear(body_dim, 1)

        anchor_indices = self.feature_groups.get("xco2", [])
        self.anchor_head = (
            nn.Linear(1, 1, bias=False) if len(anchor_indices) == 1 else None
        )

    def forward(self, x: torch.Tensor):
        encoded = []
        for block, encoder in self.encoders.items():
            indices = getattr(self, self._index_buffers[block])
            encoded.append(encoder(torch.index_select(x, 1, indices)))
        h = self.body(torch.cat(encoded, dim=1))

        gate = torch.softmax(self.gate_head(h), dim=1)
        expert_mu = self.expert_mu_head(h)
        mu = torch.sum(gate * expert_mu, dim=1)
        if self.anchor_head is not None:
            anchor_idx = getattr(self, self._index_buffers["xco2"])
            anchor = torch.index_select(x, 1, anchor_idx)
            mu = mu + self.anchor_head(anchor).squeeze(-1)

        raw2 = torch.clamp(self.raw2_head(h).squeeze(-1), min=-10.0, max=10.0)
        return mu, raw2

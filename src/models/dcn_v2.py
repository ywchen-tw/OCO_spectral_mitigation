"""Compact DCN-V2 backbone for per-sounding XCO2 anomaly regression.

The cross network explicitly learns bounded-degree interactions among the
scalar ``FeaturePipeline`` outputs.  A parallel MLP learns complementary
implicit interactions.  No radiance spectra, neighboring soundings, or orbit
context are consumed.

Reference:
    Wang et al. (2021), "DCN V2: Improved Deep & Cross Network and Practical
    Lessons for Web-scale Learning to Rank Systems."
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._backbone_utils import dense_stack


class LowRankCrossLayer(nn.Module):
    """DCN-V2 low-rank cross layer: x0 ⊙ (U(V(x)) + b) + x."""

    def __init__(self, n_features: int, rank: int):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be positive, got {rank}")
        rank = min(int(rank), int(n_features))
        self.v = nn.Linear(n_features, rank, bias=False)
        self.u = nn.Linear(rank, n_features, bias=True)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.u(self.v(x)) + x


class CrossNetV2(nn.Module):
    def __init__(self, n_features: int, n_layers: int, rank: int):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        self.layers = nn.ModuleList(
            LowRankCrossLayer(n_features, rank) for _ in range(n_layers)
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for layer in self.layers:
            x = layer(x0, x)
        return x


class DCNV2Regressor(nn.Module):
    """Parallel low-rank cross network and deep tower with Gaussian-style head."""

    def __init__(
        self,
        n_features: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        cross_layers: int = 2,
        cross_rank: int = 16,
        aux_cloud: bool = False,
        dropout: float = 0.0,
        norm: str = "none",
    ):
        super().__init__()
        self.aux_cloud = bool(aux_cloud)
        self.cross = CrossNetV2(n_features, cross_layers, cross_rank)
        self.deep = dense_stack(
            n_features,
            tuple(hidden_dims),
            dropout=dropout,
            norm=norm,
        )
        fused_dim = n_features + hidden_dims[-1]
        self.head = nn.Linear(fused_dim, 2)
        if self.aux_cloud:
            self.cloud_head = nn.Linear(fused_dim, 1)

    def forward(self, x: torch.Tensor):
        fused = torch.cat([self.cross(x), self.deep(x)], dim=1)
        out = self.head(fused)
        mu = out[:, 0]
        raw2 = torch.clamp(out[:, 1], min=-10.0, max=10.0)
        if self.aux_cloud:
            return mu, raw2, self.cloud_head(fused).squeeze(-1)
        return mu, raw2

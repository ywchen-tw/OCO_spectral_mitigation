"""Small construction helpers shared by experimental ensemble backbones."""

from __future__ import annotations

import torch.nn as nn


def normalization(kind: str, width: int) -> nn.Module | None:
    if kind == "none":
        return None
    if kind == "layer":
        return nn.LayerNorm(width)
    if kind == "batch":
        return nn.BatchNorm1d(width)
    raise ValueError(f"norm must be 'none'|'layer'|'batch', got {kind!r}")


def dense_stack(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    *,
    dropout: float,
    norm: str,
) -> nn.Sequential:
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one width")
    layers: list[nn.Module] = []
    width = input_dim
    for next_width in hidden_dims:
        layers.append(nn.Linear(width, next_width))
        normalizer = normalization(norm, next_width)
        if normalizer is not None:
            layers.append(normalizer)
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        width = next_width
    return nn.Sequential(*layers)


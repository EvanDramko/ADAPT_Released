"""MLP embedding for Cartesian coordinates."""

from __future__ import annotations

import torch
from torch import nn

from MLIP_arch.activations import build_activation


class CoordMLPEmbedding(nn.Module):
    """Embed per-atom coordinates with a small MLP."""

    def __init__(
        self,
        *,
        output_length: int,
        hidden_dim: int,
        activation_name: str,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            build_activation(activation_name),
            nn.Linear(hidden_dim, hidden_dim),
            build_activation(activation_name),
            nn.Linear(hidden_dim, output_length),
            nn.LayerNorm(output_length),
            nn.Dropout(dropout_rate),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"coords must have shape (batch, n_atoms, 3), got {tuple(coords.shape)}."
            )
        return self.net(coords)


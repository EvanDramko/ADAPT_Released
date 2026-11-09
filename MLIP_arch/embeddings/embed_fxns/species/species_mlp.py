"""MLP embedding for atomic species values."""

from __future__ import annotations

import torch
from torch import nn

from MLIP_arch.activations import build_activation


class SpeciesMLPEmbedding(nn.Module):
    """Embed atomic species values with a small MLP."""

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
            nn.Linear(1, hidden_dim),
            build_activation(activation_name),
            nn.Linear(hidden_dim, hidden_dim),
            build_activation(activation_name),
            nn.Linear(hidden_dim, output_length),
            nn.LayerNorm(output_length),
            nn.Dropout(dropout_rate),
        )

    def forward(self, Z_species: torch.Tensor) -> torch.Tensor:
        if Z_species.ndim != 3 or Z_species.shape[-1] != 1:
            raise ValueError(
                f"Z_species must have shape (batch, n_atoms, 1), got {tuple(Z_species.shape)}."
            )
        return self.net(Z_species.to(dtype=torch.get_default_dtype()))


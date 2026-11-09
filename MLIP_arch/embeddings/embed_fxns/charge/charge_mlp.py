"""MLP embedding for scalar structure charge."""

from __future__ import annotations

import torch
from torch import nn

from MLIP_arch.activations import build_activation


class ChargeMLPEmbedding(nn.Module):
    """Embed one scalar structure charge into one token."""

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

    def forward(
        self,
        charge: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if charge.ndim == 0:
            charge = charge.reshape(1)
            raise ValueError(f"Technically fine, but may be indicative of underlying issues! Shape received for charge is: {charge.shape}")
        if charge.ndim == 1:
            charge = charge.unsqueeze(-1)
        if charge.ndim != 2 or charge.shape[-1] != 1:
            raise ValueError(
                f"charge must have shape (batch,) or (batch, 1), got {tuple(charge.shape)}."
            )
        if batch_size is not None and charge.shape[0] != batch_size:
            raise ValueError(f"charge batch size must be {batch_size}, got {charge.shape[0]}.")
        return self.net(charge).unsqueeze(1)

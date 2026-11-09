"""Direct per-atom force prediction head."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from MLIP_arch.activations import build_activation
from MLIP_arch.core_arch.encoder_backbone import EncoderBackbone


AttentionBuilder = Callable[[int, int, float], nn.Module]


class DirectForceHead(nn.Module):
    """Project each atom token directly to one Cartesian force vector."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation_name: str,
        *,
        num_head_layers: int,
        num_heads: int,
        dropout_rate: float,
        attention_builder: AttentionBuilder,
    ) -> None:
        super().__init__()
        if num_head_layers < 0:
            raise ValueError(f"num_head_layers must be non-negative, got {num_head_layers}.")
        self.head_encoder = None
        if num_head_layers > 0:
            self.head_encoder = EncoderBackbone(
                d_model=d_model,
                d_ff=d_ff,
                num_layers=num_head_layers,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_builder=attention_builder,
                activation_name=activation_name,
            )
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            build_activation(activation_name),
            nn.Linear(d_ff, 3),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        atom_count: int | None = None,
    ) -> torch.Tensor:
        """Return force predictions with shape ``(batch, n_atoms, 3)``."""
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must have shape (batch, seq_len, d_model), got {tuple(tokens.shape)}."
            )
        if valid_mask is not None and valid_mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"valid_mask must have shape {tuple(tokens.shape[:2])}, got {tuple(valid_mask.shape)}."
            )
        if self.head_encoder is not None:
            tokens = self.head_encoder(tokens, valid_mask=valid_mask)

        if atom_count is None:
            atom_count = tokens.shape[1]
        atom_tokens = tokens[:, :atom_count, :]
        atom_mask = None if valid_mask is None else valid_mask[:, :atom_count]

        force = self.net(atom_tokens)
        if atom_mask is None:
            return force
        return torch.where(atom_mask.unsqueeze(-1), force, torch.zeros_like(force))

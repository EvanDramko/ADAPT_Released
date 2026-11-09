"""Embedding-table lookup for integer structure charge."""

from __future__ import annotations

import torch
from torch import nn


class ChargeTableEmbedding(nn.Module):
    """
    Embed integer structure charge with ``nn.Embedding``.

    Charges are shifted by ``-min_charge`` before lookup. Range errors are left
    to ``nn.Embedding`` so the forward path avoids GPU-to-CPU synchronization.
    """

    def __init__(
        self,
        *,
        output_length: int,
        min_charge: int = -10,
        max_charge: int = 10,
    ) -> None:
        super().__init__()
        if max_charge < min_charge:
            raise ValueError("max_charge must be greater than or equal to min_charge.")
        self.min_charge = min_charge
        self.embedding = nn.Embedding(max_charge - min_charge + 1, output_length)

    def forward(
        self,
        charge: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if charge.ndim == 0:
            charge = charge.reshape(1)
            raise ValueError(f"Technically okay, but check for hidden/silent issues. Recieved shape {charge.shape}")
        if charge.ndim == 2 and charge.shape[-1] == 1:
            charge = charge.squeeze(-1)
        if charge.ndim != 1:
            raise ValueError(
                f"charge must have shape (batch,) or (batch, 1), got {tuple(charge.shape)}."
            )
        if batch_size is not None and charge.shape[0] != batch_size:
            raise ValueError(f"charge batch size must be {batch_size}, got {charge.shape[0]}.")
        if charge.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"charge table embedding requires integer charges, got {charge.dtype}.")

        indices = charge.long() - self.min_charge
        token = self.embedding(indices).unsqueeze(1)
        return token

"""Embedding-table lookup for integer structure spin multiplicity."""

from __future__ import annotations

import torch
from torch import nn


class SpinMultiplicityTableEmbedding(nn.Module):
    """
    Embed integer spin multiplicity with ``nn.Embedding``.

    Valid values are 0 through 11 inclusive. Range errors are left to
    ``nn.Embedding`` so the forward path avoids GPU-to-CPU synchronization.
    """

    def __init__(
        self,
        *,
        output_length: int,
        max_spin_multiplicity: int = 11,
    ) -> None:
        super().__init__()
        min_spin_multiplicity: int = 0
        if max_spin_multiplicity < min_spin_multiplicity:
            raise ValueError(
                "max_spin_multiplicity must be greater than or equal to "
                "min_spin_multiplicity."
            )
        
        self.embedding = nn.Embedding(
            max_spin_multiplicity + 1,
            output_length,
        )

    def forward(
        self,
        spin_multiplicity: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if spin_multiplicity.ndim == 0:
            spin_multiplicity = spin_multiplicity.reshape(1)
            raise ValueError(
                "Technically okay, but check for hidden/silent issues. "
                f"Received shape {spin_multiplicity.shape}"
            )
        if spin_multiplicity.ndim == 2 and spin_multiplicity.shape[-1] == 1:
            spin_multiplicity = spin_multiplicity.squeeze(-1)
        if spin_multiplicity.ndim != 1:
            raise ValueError(
                "spin_multiplicity must have shape (batch,) or (batch, 1), "
                f"got {tuple(spin_multiplicity.shape)}."
            )
        if batch_size is not None and spin_multiplicity.shape[0] != batch_size:
            raise ValueError(
                "spin_multiplicity batch size must be "
                f"{batch_size}, got {spin_multiplicity.shape[0]}."
            )
        if spin_multiplicity.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "spin_multiplicity table embedding requires integer values, "
                f"got {spin_multiplicity.dtype}."
            )

        indices = spin_multiplicity.long()
        return self.embedding(indices).unsqueeze(1)

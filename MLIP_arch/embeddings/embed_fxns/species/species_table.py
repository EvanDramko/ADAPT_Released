"""Embedding-table lookup for atomic species values."""

from __future__ import annotations

import torch
from torch import nn


class SpeciesTableEmbedding(nn.Module):
    """
    Embed integer atomic numbers with ``nn.Embedding``.

    Index 0 is reserved for padding. Real atomic numbers are expected to be
    positive integer IDs. Range errors are left to ``nn.Embedding`` so the
    forward path avoids GPU-to-CPU synchronization.
    """

    def __init__(
        self,
        *,
        output_length: int,
        max_atomic_number: int = 118,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if max_atomic_number < 1:
            raise ValueError("max_atomic_number must be positive.")
        self.embedding = nn.Embedding(
            max_atomic_number + 1,
            output_length,
            padding_idx=padding_idx,
        )

    def forward(self, Z_species: torch.Tensor) -> torch.Tensor:
        if Z_species.ndim != 3 or Z_species.shape[-1] != 1:
            raise ValueError(
                f"Z_species must have shape (batch, n_atoms, 1), got {tuple(Z_species.shape)}."
            )
        if Z_species.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"Z_species must contain integer atomic numbers, got {Z_species.dtype}."
            )
        return self.embedding(Z_species.squeeze(-1).long())


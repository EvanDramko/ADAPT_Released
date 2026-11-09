"""Composed MLIP model wrapper."""

from __future__ import annotations

import torch
from torch import nn


class MLIPModel(nn.Module):
    """Compose embedding, encoder backbone, and output heads.

    MLIP model designed to handle data processed through: .xyz -> .adapt.pt -> objectives/ -> Here!
    Charge and spin multiplicity are embedded as appended structure-level tokens
    during encoding.
    """

    def __init__(
        self,
        *,
        structure_embedding: nn.Module,
        encoder_backbone: nn.Module,
        output_heads: nn.ModuleDict,
    ) -> None:
        super().__init__()
        self.structure_embedding = structure_embedding
        self.encoder_backbone = encoder_backbone
        self.output_heads = output_heads

    def forward(
        self,
        coord: torch.Tensor,
        Z_species: torch.Tensor,
        *,
        charge: torch.Tensor | None = None,
        spin_multiplicity: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return predictions from each configured output head.

        Output keys are physical quantities such as ``"energy"`` and ``"force"``.
        Each quantity has exactly one configured prediction head.
        """
        if coord.ndim != 3 or coord.shape[-1] != 3:
            raise ValueError(
                f"positions must have shape (batch, n_atoms, 3), got {tuple(coord.shape)}."
            )
        
        if Z_species.ndim != 3 or Z_species.shape[-1] != 1:
            raise ValueError(
                f"positions must have shape (batch, n_atoms, 1), got {tuple(Z_species.shape)}."
            )

        embedded = self.structure_embedding(
            coord,
            Z_species,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            atom_mask=atom_mask,
        )
        encoded_full = self.encoder_backbone(
            embedded.full_tokens,
            attn_mask=attn_mask,
            valid_mask=embedded.full_mask,
        )
        return {
            output: head(
                encoded_full,
                embedded.full_mask,
                atom_count=coord.shape[1],
            )
            for output, head in self.output_heads.items()
        }

"""Compose field-specific embeddings into structure tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from MLIP_arch.activations import build_activation
from MLIP_arch.embeddings.embed_fxns.charge.charge_mlp import ChargeMLPEmbedding
from MLIP_arch.embeddings.embed_fxns.charge.charge_table import ChargeTableEmbedding
from MLIP_arch.embeddings.embed_fxns.coordinates.coord_mlp import CoordMLPEmbedding
from MLIP_arch.embeddings.embed_fxns.species.species_mlp import SpeciesMLPEmbedding
from MLIP_arch.embeddings.embed_fxns.species.species_table import SpeciesTableEmbedding
from MLIP_arch.embeddings.embed_fxns.spin_multiplicity.spin_multiplicity_table import SpinMultiplicityTableEmbedding


COMBINATION_TYPE: Final[str] = "concatenate"
VALID_COMBINATION_TYPES: Final[set[str]] = {"add_together", "concatenate"}

COORDINATE_EMBEDDINGS: Final[dict[str, type[nn.Module]]] = {
    "coord_mlp": CoordMLPEmbedding,
}
SPECIES_EMBEDDINGS: Final[dict[str, type[nn.Module]]] = {
    "Z_mlp": SpeciesMLPEmbedding,
    "Z_table": SpeciesTableEmbedding,
}
CHARGE_EMBEDDINGS: Final[dict[str, type[nn.Module]]] = {
    "charge_mlp": ChargeMLPEmbedding,
    "charge_table": ChargeTableEmbedding,
}
SPIN_MULTIPLICITY_EMBEDDINGS: Final[dict[str, type[nn.Module]]] = {
    "spin_multiplicity_table": SpinMultiplicityTableEmbedding,
}


@dataclass(frozen=True)
class EmbeddedStructure:
    """Embedded atom tokens plus the full sequence including structure tokens."""

    atom_tokens: torch.Tensor
    full_tokens: torch.Tensor
    atom_mask: torch.Tensor | None
    full_mask: torch.Tensor | None


class StructureEmbedding(nn.Module):
    """Embed coordinate, species, charge, and spin fields into structure tokens."""

    def __init__(
        self,
        *,
        coordinate_embedding: str,
        species_embedding: str,
        charge_embedding: str,
        spin_multiplicity_embedding: str,
        d_model: int,
        hidden_dim: int,
        activation_name: str,
        dropout_rate: float = 0.0,
        max_atomic_number: int = 118,
    ) -> None:
        super().__init__()
        if COMBINATION_TYPE not in VALID_COMBINATION_TYPES:
            raise ValueError(
                f"Unknown COMBINATION_TYPE {COMBINATION_TYPE!r}. "
                f"Valid choices: {sorted(VALID_COMBINATION_TYPES)}."
            )

        coordinate_cls = _get_embedding_cls(
            coordinate_embedding,
            COORDINATE_EMBEDDINGS,
            field_name="coordinate_embedding",
        )
        species_cls = _get_embedding_cls(
            species_embedding,
            SPECIES_EMBEDDINGS,
            field_name="species_embedding",
        )
        charge_cls = _get_embedding_cls(
            charge_embedding,
            CHARGE_EMBEDDINGS,
            field_name="charge_embedding",
        )
        spin_multiplicity_cls = _get_embedding_cls(
            spin_multiplicity_embedding,
            SPIN_MULTIPLICITY_EMBEDDINGS,
            field_name="spin_multiplicity_embedding",
        )

        if COMBINATION_TYPE == "concatenate":
            if d_model % 2:
                raise ValueError(
                    "d_model must be divisible by 2 when COMBINATION_TYPE='concatenate'."
                )
            self.output_length_coordinates = d_model // 2
            self.output_length_species = d_model // 2
        else:
            self.output_length_coordinates = d_model
            self.output_length_species = d_model

        self.d_model = d_model
        self.atom_fusion = None
        if COMBINATION_TYPE == "concatenate":
            self.atom_fusion = nn.Sequential(
                nn.Linear(d_model, d_model),
                build_activation(activation_name),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout_rate),
            )
        self.coordinate_embedding = coordinate_cls(
            output_length=self.output_length_coordinates,
            hidden_dim=hidden_dim,
            activation_name=activation_name,
            dropout_rate=dropout_rate,
        )
        self.species_embedding = _build_species_embedding(
            species_cls,
            output_length=self.output_length_species,
            hidden_dim=hidden_dim,
            activation_name=activation_name,
            dropout_rate=dropout_rate,
            max_atomic_number=max_atomic_number,
        )
        self.charge_embedding = _build_charge_embedding(
            charge_cls,
            output_length=d_model,
            hidden_dim=hidden_dim,
            activation_name=activation_name,
            dropout_rate=dropout_rate,
        )
        self.spin_multiplicity_embedding = spin_multiplicity_cls(
            output_length=d_model,
        )

    def forward(
        self,
        coord: torch.Tensor,
        Z_species: torch.Tensor,
        charge: torch.Tensor | None = None,
        spin_multiplicity: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
    ) -> EmbeddedStructure:
        """Return atom-only and full token sequences.

        If ``charge`` or ``spin_multiplicity`` is missing, a zero-valued token
        is appended. Structure tokens are always marked valid in ``full_mask``.
        """
        coordinate_tokens = self.coordinate_embedding(coord)
        species_tokens = self.species_embedding(Z_species)
        atom_tokens = self._combine_atom_tokens(coordinate_tokens, species_tokens)
        batch_size = atom_tokens.shape[0]

        if atom_mask is not None:
            if atom_mask.shape != atom_tokens.shape[:2]:
                raise ValueError(
                    f"atom_mask must have shape {tuple(atom_tokens.shape[:2])}, "
                    f"got {tuple(atom_mask.shape)}."
                )
            if atom_mask.dtype != torch.bool:
                raise ValueError("atom_mask must be a boolean tensor.")
            atom_tokens = torch.where(
                atom_mask.unsqueeze(-1),
                atom_tokens,
                torch.zeros_like(atom_tokens),
            )

        if charge is None:
            charge_dtype = (
                torch.int64
                if isinstance(self.charge_embedding, ChargeTableEmbedding)
                else atom_tokens.dtype
            )
            charge = torch.zeros((batch_size,), dtype=charge_dtype, device=atom_tokens.device)
        else:
            charge = charge.to(device=atom_tokens.device)
        if spin_multiplicity is None:
            spin_multiplicity = torch.zeros(
                (batch_size,),
                dtype=torch.int64,
                device=atom_tokens.device,
            )
        else:
            spin_multiplicity = spin_multiplicity.to(device=atom_tokens.device)

        charge_token = self.charge_embedding(
            charge,
            batch_size=batch_size,
        )
        spin_multiplicity_token = self.spin_multiplicity_embedding(
            spin_multiplicity,
            batch_size=batch_size,
        )
        structure_tokens = torch.cat([charge_token, spin_multiplicity_token], dim=1)
        full_tokens = torch.cat([atom_tokens, structure_tokens], dim=1)

        full_mask = None
        if atom_mask is not None:
            structure_mask = torch.ones(
                (batch_size, structure_tokens.shape[1]),
                dtype=torch.bool,
                device=atom_mask.device,
            )
            full_mask = torch.cat([atom_mask, structure_mask], dim=1)

        return EmbeddedStructure(
            atom_tokens=atom_tokens,
            full_tokens=full_tokens,
            atom_mask=atom_mask,
            full_mask=full_mask,
        )

    def _combine_atom_tokens(
        self,
        coordinate_tokens: torch.Tensor,
        species_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if coordinate_tokens.shape[:2] != species_tokens.shape[:2]:
            raise ValueError(
                "Coordinate and species embeddings must have matching batch/atom dimensions. "
                f"coords={tuple(coordinate_tokens.shape)}, species={tuple(species_tokens.shape)}."
            )

        if COMBINATION_TYPE == "add_together":
            return coordinate_tokens + species_tokens
        if COMBINATION_TYPE == "concatenate":
            atom_tokens = torch.cat([coordinate_tokens, species_tokens], dim=-1)
            if self.atom_fusion is None:
                raise RuntimeError("atom_fusion must be defined for concatenate mode.")
            return self.atom_fusion(atom_tokens)

        raise RuntimeError(f"Unsupported COMBINATION_TYPE {COMBINATION_TYPE!r}.")


def _get_embedding_cls(
    name: str,
    registry: dict[str, type[nn.Module]],
    *,
    field_name: str,
) -> type[nn.Module]:
    cls = registry.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown {field_name} {name!r}. Valid choices: {sorted(registry)}."
        )
    return cls


def _build_species_embedding(
    cls: type[nn.Module],
    *,
    output_length: int,
    hidden_dim: int,
    activation_name: str,
    dropout_rate: float,
    max_atomic_number: int,
) -> nn.Module:
    if cls is SpeciesTableEmbedding:
        return cls(
            output_length=output_length,
            max_atomic_number=max_atomic_number,
        )
    return cls(
        output_length=output_length,
        hidden_dim=hidden_dim,
        activation_name=activation_name,
        dropout_rate=dropout_rate,
    )


def _build_charge_embedding(
    cls: type[nn.Module],
    *,
    output_length: int,
    hidden_dim: int,
    activation_name: str,
    dropout_rate: float,
) -> nn.Module:
    if cls is ChargeTableEmbedding:
        return cls(output_length=output_length)
    return cls(
        output_length=output_length,
        hidden_dim=hidden_dim,
        activation_name=activation_name,
        dropout_rate=dropout_rate,
    )

"""Architecture configuration dataclasses for MLIP model construction."""

from __future__ import annotations

from dataclasses import dataclass, field
VALID_HEAD_OUTPUTS = frozenset(("energy", "force"))
VALID_HEAD_TYPES = {
    "atomwise_energy": "energy",
    "scalar_decoder_energy": "energy",
    "direct_force": "force",
}


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for atom and structure-level embedding modules.

    Charge and spin multiplicity are appended as structure-level tokens.
    """

    coordinate_embedding: str = "coord_mlp"
    species_embedding: str = "Z_table"
    charge_embedding: str = "charge_table"
    spin_multiplicity_embedding: str = "spin_multiplicity_table"
    max_atomic_number: int = 118


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the Transformer encoder backbone."""

    attention: str = "standard_mha"
    activation: str = "relu"
    d_model: int = 256
    d_ff: int = 512
    num_layers: int = 6
    layer_repetitions: int = 1
    head_num_layers: int = 2
    num_heads: int = 16
    dropout_rate: float = 0.05

    def __post_init__(self) -> None:
        if isinstance(self.layer_repetitions, bool) or not isinstance(
            self.layer_repetitions, int
        ):
            raise TypeError("layer_repetitions must be an integer.")
        if self.layer_repetitions < 1:
            raise ValueError("layer_repetitions must be at least 1.")


@dataclass(frozen=True)
class ArchitectureConfig:
    """Top-level architecture config consumed by MLIP_arch/mlip_factory.py."""

    heads: tuple[str, ...]
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    def __post_init__(self) -> None:
        if not self.heads:
            raise ValueError("ArchitectureConfig.heads must contain at least one output head.")

        unknown_heads = sorted(set(self.heads) - VALID_HEAD_TYPES.keys())
        if unknown_heads:
            raise ValueError(
                f"Unknown output heads {unknown_heads}. "
                f"Valid heads: {sorted(VALID_HEAD_TYPES)}."
            )

        quantities = [VALID_HEAD_TYPES[head] for head in self.heads]
        duplicate_quantities = sorted(
            quantity for quantity in set(quantities) if quantities.count(quantity) > 1
        )
        if duplicate_quantities:
            raise ValueError(
                "Only one output head may be selected for each physical output type/quantity. "
                f"Duplicate outputs from: {duplicate_quantities}."
            )

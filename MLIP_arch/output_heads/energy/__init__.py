"""Energy-output head registry."""

from __future__ import annotations

from torch import nn

from .atomwise_energy import AtomwiseEnergyHead
from .scalar_decoder_energy import ScalarDecoderEnergyHead


ENERGY_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "atomwise_energy": AtomwiseEnergyHead,
    "scalar_decoder_energy": ScalarDecoderEnergyHead,
}


def build_energy_head(head_type: str, **kwargs) -> nn.Module:
    """Build one registered energy head."""
    cls = ENERGY_HEAD_REGISTRY.get(head_type)
    if cls is None:
        raise ValueError(
            f"Unknown energy head '{head_type}'. Available energy heads: {sorted(ENERGY_HEAD_REGISTRY)}."
        )
    return cls(**kwargs)

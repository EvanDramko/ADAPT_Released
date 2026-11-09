"""Output-head registries grouped by predicted physical quantity."""

from __future__ import annotations

from torch import nn

from .energy import ENERGY_HEAD_REGISTRY, build_energy_head
from .forces import FORCE_HEAD_REGISTRY, build_force_head


HEAD_REGISTRIES: dict[str, dict[str, type[nn.Module]]] = {
    "energy": ENERGY_HEAD_REGISTRY,
    "force": FORCE_HEAD_REGISTRY,
}


def available_head_types(output: str) -> tuple[str, ...]:
    """Return registered head types for one physical output quantity."""
    registry = HEAD_REGISTRIES.get(output)
    if registry is None:
        raise ValueError(f"Unknown output '{output}'. Available outputs: {sorted(HEAD_REGISTRIES)}.")
    return tuple(sorted(registry))


def build_output_head(output: str, head_type: str, **kwargs) -> nn.Module:
    """Build one output head from its output category and registered type."""
    if output == "force":
        return build_force_head(head_type, **kwargs)
    if output == "energy":
        return build_energy_head(head_type, **kwargs)
    raise ValueError(f"Unknown output '{output}'. Available outputs: {sorted(HEAD_REGISTRIES)}.")

"""Force-output head registry."""

from __future__ import annotations

from torch import nn

from .direct_force import DirectForceHead


FORCE_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "direct_force": DirectForceHead,
}


def build_force_head(head_type: str, **kwargs) -> nn.Module:
    """Build one registered force head."""
    cls = FORCE_HEAD_REGISTRY.get(head_type)
    if cls is None:
        raise ValueError(
            f"Unknown force head '{head_type}'. Available force heads: {sorted(FORCE_HEAD_REGISTRY)}."
        )
    return cls(**kwargs)

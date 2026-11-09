"""Component-based MLIP architecture package."""

from .mlip_factory import build_mlip_model
from .model import MLIPModel


__all__ = [
    "MLIPModel",
    "build_mlip_model",
]

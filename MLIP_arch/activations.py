"""Activation-function registry for MLFF architecture modules."""

from __future__ import annotations
from collections.abc import Callable
from torch import nn


ActivationBuilder = Callable[[], nn.Module]


ACTIVATION_REGISTRY: dict[str, ActivationBuilder] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def available_activations() -> tuple[str, ...]:
    """Return the registered activation names."""
    return tuple(sorted(ACTIVATION_REGISTRY))


def build_activation(name: str) -> nn.Module:
    """Construct one activation module by registry name."""
    key = name.lower()
    builder = ACTIVATION_REGISTRY.get(key)
    if builder is None:
        raise ValueError(
            f"Unknown activation '{name}'. Available options: {', '.join(available_activations())}."
        )
    return builder()

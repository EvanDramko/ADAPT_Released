"""Shared interface between MLIP models and formulation-agnostic training code."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypedDict
from typing_extensions import NotRequired

import torch
from torch import nn


class TensorBatch(TypedDict):
    """Collated structure tensors with optional supervised targets."""
    pos: torch.Tensor
    Z_species: torch.Tensor
    charge: torch.Tensor
    spin_multiplicity: torch.Tensor
    mask: torch.Tensor
    energy: NotRequired[torch.Tensor]
    force: NotRequired[torch.Tensor]


@dataclass
class ObjectiveResult:
    """Return loss data and predictions from one objective evaluation.

    ``loss`` is the weighted total loss and must retain the computation graph
    used by ``backward()``. Values in ``losses`` are detached, unweighted loss
    components intended for logging. Values in ``predictions`` are detached,
    normalized space model outputs intended for an external metric registry.
    """

    loss: torch.Tensor
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    predictions: dict[str, torch.Tensor] = field(default_factory=dict)


class Objective(ABC):
    """Define formulation-specific forward, prediction, and loss behavior."""

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: TensorBatch,
        *,
        training: bool,
        return_predictions: bool,
    ) -> ObjectiveResult:
        """Build the required graph and return losses and normalized predictions."""

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        batch: TensorBatch,
    ) -> dict[str, torch.Tensor]:
        """Return detached normalized predictions without computing a loss."""

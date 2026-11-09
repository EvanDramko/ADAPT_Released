"""Direct energy and force prediction objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from configs.arch_config import ArchitectureConfig, VALID_HEAD_TYPES
from configs.training_config import ObjectiveConfig
from .base import Objective, ObjectiveResult, TensorBatch
from objectives.utils import _masked_force_mse, _validate_batch


class DirectObjective(Objective):
    """Coordinate direct-output forward passes and normalized-space losses."""

    def __init__(
        self,
        objective_config: ObjectiveConfig,
        architecture_config: ArchitectureConfig,
    ) -> None:
        architecture_quantities = {
            VALID_HEAD_TYPES[head]
            for head in architecture_config.heads
        }
        requested_quantities = set(objective_config.requested_quantities)
        if architecture_quantities != requested_quantities:
            raise ValueError(
                "A direct objective requires architecture head quantities to exactly match "
                "the requested quantities. "
                f"head_quantities={sorted(architecture_quantities)}, "
                f"requested_quantities={sorted(requested_quantities)}."
            )

        self.config = objective_config
        self.outputs = objective_config.requested_quantities
        self.initial_energy_loss: torch.Tensor | None = None
        self.initial_force_loss: torch.Tensor | None = None


    def compute_loss(
        self,
        model: nn.Module,
        batch: TensorBatch,
        *,
        training: bool,
        return_predictions: bool,
    ) -> ObjectiveResult:
        """Run a direct forward pass and calculate normalized-space losses."""
        _validate_batch(batch)

        model.train(training)
        with torch.set_grad_enabled(training):
            predictions = self._forward(model, batch)
            loss_components = self._compute_loss_components(predictions, batch)
            total_loss = self._weighted_total(loss_components)

        reported_predictions = {}
        if return_predictions:
            reported_predictions = {
                output: prediction.detach()
                for output, prediction in predictions.items()
            }

        return ObjectiveResult(
            loss=total_loss,
            losses={
                output: component.detach()
                for output, component in loss_components.items()
            },
            predictions=reported_predictions,
        )


    def predict(
        self,
        model: nn.Module,
        batch: TensorBatch,
    ) -> dict[str, torch.Tensor]:
        """Return detached direct predictions in normalized space."""
        _validate_batch(batch)

        model.train(False)
        with torch.no_grad():
            predictions = self._forward(model, batch)
        return {
            output: prediction.detach()
            for output, prediction in predictions.items()
        }


    def _forward(
        self,
        model: nn.Module,
        batch: TensorBatch,
    ) -> dict[str, torch.Tensor]:
        predictions = model(
            batch["pos"],
            batch["Z_species"].unsqueeze(-1),
            charge=batch["charge"],
            spin_multiplicity=batch["spin_multiplicity"],
            atom_mask=batch["mask"],
        )


        return predictions


    def _compute_loss_components(
        self,
        predictions: dict[str, torch.Tensor],
        batch: TensorBatch,
    ) -> dict[str, torch.Tensor]:
        components: dict[str, torch.Tensor] = {}

        if "energy" in self.outputs:
            energy_target = batch["energy"]
            energy_prediction = predictions["energy"]

            components["energy"] = F.mse_loss(energy_prediction, energy_target)

        if "force" in self.outputs:
            components["force"] = _masked_force_mse(
                predictions["force"],
                batch["force"],
                batch["mask"],
            )

        return components

    def _weighted_total(
        self,
        components: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = 0.0

        if "energy" in components:
            if self.initial_energy_loss is None:
                self.initial_energy_loss = components["energy"].detach().clamp_min(1e-12)
            total = total + self.config.energy_weight * (components["energy"]/self.initial_energy_loss)

        if "force" in components:
            if self.initial_force_loss is None:
                self.initial_force_loss = components["force"].detach().clamp_min(1e-12)
            total = total + self.config.force_weight * (components["force"]/self.initial_force_loss)

        return total

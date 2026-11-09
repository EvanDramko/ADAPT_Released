"""Energy conserving prediction objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from configs.arch_config import ArchitectureConfig, VALID_HEAD_TYPES
from configs.training_config import ObjectiveConfig
from .base import Objective, ObjectiveResult, TensorBatch
from objectives.utils import _masked_force_mse, _validate_batch


class ConservingObjective(Objective):
    """Coordinate conservative-output forward passes and normalized-space losses."""

    def __init__(
        self,
        objective_config: ObjectiveConfig,
        architecture_config: ArchitectureConfig,
        *,
        sigma_y, 
        sigma_x,
        mu_f,
        sigma_f,
    ) -> None:
        
        # force and energy must already be requested quantities as checked in configs. Here, we just check that the architecture matches the training objective. 
        
        # check that the only head present is an energy head. 
        architecture_quantities = {
            VALID_HEAD_TYPES[head]
            for head in architecture_config.heads
        }
        if architecture_quantities != {"energy"}:
            raise ValueError(
                "An energy conserving objective requires exactly one architecture "
                "output quantity: energy. "
                f"head_quantities={sorted(architecture_quantities)}."
            )

        
        self.sigmas_y_over_x = sigma_y /sigma_x
        self.mu_f = mu_f
        self.sigma_f = sigma_f
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
        """Run a forward pass and calculate normalized-space losses."""
        _validate_batch(batch)

        model.train(training)
        with torch.set_grad_enabled(True):
            predictions = self._forward(model, batch, training)
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
        """Return predictions in normalized space."""
        _validate_batch(batch)

        model.train(False)
        with torch.set_grad_enabled(True): # True because we need the gradients for the force predictions
            predictions = self._forward(model, batch, False)
        return {
            output: prediction.detach()
            for output, prediction in predictions.items()
        }


    def _forward(
        self,
        model: nn.Module,
        batch: TensorBatch,
        training: bool,
    ) -> dict[str, torch.Tensor]:
        """We define the following: model f, inputs x, energy prediction y. Then, we calculate
        x^hat <- normalized x
        f(x^hat) = y^hat
        energy prediction (normalized) given by y^hat, energy prediction unnormalized given by y
        calculate gradients, A^hat, by (partial y^hat)/(partial x^hat)
        Denote A = A^hat * -sigma_y/sigma_x where sigma_y is the standard deviation for energy, and x is the standard deviation for forces
        to compare against normalized targets for force, calculate: (A - mu_F)/sigma_F """

        predictions = model(
            batch["pos"].requires_grad_(True),
            batch["Z_species"].unsqueeze(-1),
            charge=batch["charge"],
            spin_multiplicity=batch["spin_multiplicity"],
            atom_mask=batch["mask"],
        )
        norm_energy = predictions["energy"]
        energy_unit_force_grads = torch.autograd.grad(norm_energy.sum(),
                                               batch["pos"],
                                               create_graph=training,
                                               )[0]
        phys_force = -1*energy_unit_force_grads * self.sigmas_y_over_x
        norm_force = (phys_force - self.mu_f)/self.sigma_f

        return {
            "energy": norm_energy,
            "force": norm_force,
        }


    def _compute_loss_components(
        self,
        predictions: dict[str, torch.Tensor],
        batch: TensorBatch,
    ) -> dict[str, torch.Tensor]:
        components: dict[str, torch.Tensor] = {}

        energy_target = batch["energy"]
        energy_prediction = predictions["energy"]
        components["energy"] = F.mse_loss(energy_prediction, energy_target)

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
        if self.initial_energy_loss is None:
            self.initial_energy_loss = components["energy"].detach().clamp_min(1e-12)

        if self.initial_force_loss is None:
            self.initial_force_loss = components["force"].detach().clamp_min(1e-12)

        total = 0.0
        total = total + self.config.energy_weight * (components["energy"]/self.initial_energy_loss)
        total = total + self.config.force_weight * (components["force"]/self.initial_force_loss)
        return total



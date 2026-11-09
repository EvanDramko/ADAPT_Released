"""Registry for evaluating enabled force and energy metrics."""

from __future__ import annotations

import torch


TensorDict = dict[str, torch.Tensor]


def force_metric(
    force_preds: torch.Tensor,
    force_truth: torch.Tensor,
    atom_mask: torch.Tensor,
) -> float:
    """Return mean absolute force error per valid Cartesian component."""
    mask = atom_mask.unsqueeze(-1).to(dtype=force_preds.dtype)
    absolute_error = (force_preds - force_truth).abs()
    denom = mask.sum() * force_preds.shape[-1]
    mae = (absolute_error * mask).sum() / denom.clamp_min(1.0)
    return float(mae.detach().cpu())


def energy_metric(
    energy_preds: torch.Tensor,
    energy_truth: torch.Tensor,
) -> float:
    """Return mean absolute energy error per structure."""
    mae = (energy_preds - energy_truth).abs().mean()
    return float(mae.detach().cpu())


def run_metrics(
    prediction: TensorDict,
    ground_truth: TensorDict,
    *, 
    has_force: bool,
    has_energy: bool,
    atom_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Run metrics supported by the available predictions and targets."""
    results: dict[str, float] = {}

    if has_force:
        if atom_mask is None:
            raise ValueError("atom_mask is required when running force metrics.")
        results["force_metric"] = force_metric(
            prediction["force"],
            ground_truth["force"],
            atom_mask,
        )

    if has_energy:
        results["energy_metric"] = energy_metric(
            prediction["energy"],
            ground_truth["energy"],
        )

    return results

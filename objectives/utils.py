import torch
from .base import TensorBatch


def _masked_force_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    mask = atom_mask.unsqueeze(-1).to(dtype=prediction.dtype)
    squared_error = (prediction - target).square()

    denom = mask.sum() * prediction.shape[-1]
    return (squared_error * mask).sum() / denom


def _validate_batch(
    batch: TensorBatch,
) -> None:
    required_inputs = {"pos", "Z_species", "charge", "spin_multiplicity", "mask"}
    missing_inputs = sorted(required_inputs - set(batch))
    if missing_inputs:
        raise KeyError(f"Batch is missing required model inputs: {missing_inputs}.")

    pos = batch["pos"]
    species = batch["Z_species"]
    mask = batch["mask"]
    batch_size, n_atoms = pos.shape[:2]

    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(
            f"batch['pos'] must have shape (B, N, 3), got {tuple(pos.shape)}."
        )
    if species.shape != (batch_size, n_atoms):
        raise ValueError(
            "batch['Z_species'] must have shape "
            f"{(batch_size, n_atoms)}, got {tuple(species.shape)}."
        )
    if mask.dtype != torch.bool or mask.shape != (batch_size, n_atoms):
        raise ValueError(
            "batch['mask'] must be boolean with shape "
            f"{(batch_size, n_atoms)}, got dtype={mask.dtype} and "
            f"shape={tuple(mask.shape)}."
        )
    if batch["charge"].shape != (batch_size,):
        raise ValueError(
            f"batch['charge'] must have shape ({batch_size},), "
            f"got {tuple(batch['charge'].shape)}."
        )
    if batch["spin_multiplicity"].shape != (batch_size,):
        raise ValueError(
            f"batch['spin_multiplicity'] must have shape ({batch_size},), "
            f"got {tuple(batch['spin_multiplicity'].shape)}."
        )

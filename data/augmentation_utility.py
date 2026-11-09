"""Geometry augmentation utilities for ADAPT tensor batches and datasets."""

from __future__ import annotations

import torch


def center_batch_on_centroid(
    batch: dict[str, torch.Tensor],
    *,
    position_key: str = "pos",
    mask_key: str = "mask",
) -> dict[str, torch.Tensor]:
    """Return a batch copy with each structure centered at its geometric centroid."""
    centered = batch.copy()
    centered[position_key] = center_padded_positions_on_centroid(
        batch[position_key],
        batch.get(mask_key),
    )
    return centered


def center_padded_positions_on_centroid(
    positions: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Center padded ``(B, N, 3)`` positions using equal atom weights."""
    _validate_vector_tensor(positions, "positions", ndim=3)
    if mask is None:
        mask = torch.ones(positions.shape[:2], dtype=torch.bool, device=positions.device)
    if mask.shape != positions.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be bool with shape matching positions[:2].")

    weights = mask.to(dtype=positions.dtype)
    atom_count = weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(positions.dtype).eps)
    center = (positions * weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / atom_count.unsqueeze(-1)
    centered = positions - center
    return torch.where(mask.unsqueeze(-1), centered, torch.zeros_like(centered))


def center_dataset_on_centroid(
    dataset: dict[str, torch.Tensor],
    *,
    position_key: str = "pos",
    offsets_key: str = "offsets",
) -> dict[str, torch.Tensor]:
    """Return a flat .adapt.pt-style dataset copy with each structure centroid-centered."""
    centered = dataset.copy()
    positions = dataset[position_key]
    offsets = dataset[offsets_key]
    _validate_vector_tensor(positions, "positions", ndim=2)
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("offsets must be one-dimensional with at least two entries.")

    centered_positions = positions.clone()
    for index in range(offsets.numel() - 1):
        start = int(offsets[index].item())
        end = int(offsets[index + 1].item())
        structure_pos = positions[start:end]
        if structure_pos.numel() == 0:
            raise ValueError(f"Structure {index} contains no atoms.")
        center = structure_pos.mean(dim=0)
        centered_positions[start:end] = structure_pos - center

    centered[position_key] = centered_positions
    return centered


def rotate_batch(
    batch: dict[str, torch.Tensor],
    *,
    position_key: str = "pos",
    force_key: str = "force",
    mask_key: str = "mask",
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Return a batch copy with random rotations applied to positions and forces."""
    rotated = batch.copy()
    positions = batch[position_key]
    rotations = random_rotation_matrices(
        positions.shape[0],
        device=positions.device,
        dtype=positions.dtype,
        generator=generator,
    )
    mask = batch.get(mask_key)
    rotated[position_key] = apply_rotations(positions, rotations, mask)
    if force_key in batch:
        rotated[force_key] = apply_rotations(batch[force_key], rotations, mask)
    return rotated


def apply_rotations(
    vectors: torch.Tensor,
    rotations: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one ``(3, 3)`` rotation per batch item to ``(B, N, 3)`` vectors."""
    _validate_vector_tensor(vectors, "vectors", ndim=3)
    if rotations.shape != (vectors.shape[0], 3, 3):
        raise ValueError(
            f"rotations must have shape ({vectors.shape[0]}, 3, 3), "
            f"got {tuple(rotations.shape)}."
        )
    rotated = torch.einsum("bij,bnj->bni", rotations.to(vectors.device), vectors)
    if mask is not None:
        if mask.shape != vectors.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool with shape matching vectors[:2].")
        rotated = torch.where(mask.unsqueeze(-1), rotated, torch.zeros_like(rotated))
    return rotated


def random_rotation_matrices(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample uniformly random 3D rotation matrices."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    u1 = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    u2 = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    u3 = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)

    sqrt_u1 = torch.sqrt(u1)
    sqrt_one_minus_u1 = torch.sqrt(1.0 - u1)
    theta1 = 2.0 * torch.pi * u2
    theta2 = 2.0 * torch.pi * u3

    qx = sqrt_one_minus_u1 * torch.sin(theta1)
    qy = sqrt_one_minus_u1 * torch.cos(theta1)
    qz = sqrt_u1 * torch.sin(theta2)
    qw = sqrt_u1 * torch.cos(theta2)
    return quaternion_to_rotation_matrix(qw, qx, qy, qz)


def quaternion_to_rotation_matrix(
    qw: torch.Tensor,
    qx: torch.Tensor,
    qy: torch.Tensor,
    qz: torch.Tensor,
) -> torch.Tensor:
    """Convert unit quaternions to rotation matrices."""
    two = torch.as_tensor(2.0, device=qw.device, dtype=qw.dtype)
    rotations = torch.empty((qw.shape[0], 3, 3), device=qw.device, dtype=qw.dtype)

    rotations[:, 0, 0] = 1 - two * (qy * qy + qz * qz)
    rotations[:, 0, 1] = two * (qx * qy - qz * qw)
    rotations[:, 0, 2] = two * (qx * qz + qy * qw)
    rotations[:, 1, 0] = two * (qx * qy + qz * qw)
    rotations[:, 1, 1] = 1 - two * (qx * qx + qz * qz)
    rotations[:, 1, 2] = two * (qy * qz - qx * qw)
    rotations[:, 2, 0] = two * (qx * qz - qy * qw)
    rotations[:, 2, 1] = two * (qy * qz + qx * qw)
    rotations[:, 2, 2] = 1 - two * (qx * qx + qy * qy)
    return rotations


def _validate_vector_tensor(tensor: torch.Tensor, name: str, *, ndim: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != ndim or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3) with ndim={ndim}.")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point.")

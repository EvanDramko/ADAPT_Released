"""Measure force-prediction agreement under random rotations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.adapt_dataset import StructureDataset, collate_batch
from data.augmentation_utility import apply_rotations, random_rotation_matrices
from data.normalizer import Normalizer
from data.resolve_data_path import resolve_data_path
from evaluation import (
    _build_configs,
    _build_objective,
    _load_dataset_dict,
    _load_model_and_objective_state,
    _move_batch,
    _resolve_device,
)
from MLIP_arch import build_mlip_model
from objectives.base import Objective


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    architecture_config, training_config = _build_configs(args)

    if "force" not in training_config.objective.requested_quantities:
        raise ValueError("Force agreement requires 'force' in --requested-quantities.")

    summary = evaluate_force_agreement(
        architecture_config=architecture_config,
        training_config=training_config,
        model_path=args.model_path,
        data_path=args.data_path,
        random_seed=args.random_seed,
    )
    _print_results(summary)


def evaluate_force_agreement(
    *,
    architecture_config,
    training_config,
    model_path: str | Path,
    data_path: str | Path,
    random_seed: int | None,
) -> dict[str, float]:
    """Compare original force predictions to unrotated predictions on rotated inputs."""
    device = _resolve_device(training_config.device)
    dataset_path = resolve_data_path(data_path)

    normalize_Z = architecture_config.embedding.species_embedding != "Z_table"
    normalize_charge = architecture_config.embedding.charge_embedding != "charge_table"
    normalizer = Normalizer(
        training_config.norm_stats_path,
        normalize_charge=normalize_charge,
        normalize_Z=normalize_Z,
    )

    raw_dataset = _load_dataset_dict(dataset_path)
    for field in ("energy", "force"):
        if field not in raw_dataset:
            raise ValueError(f"Force agreement expects the dataset to contain {field!r}.")

    dataset = StructureDataset(normalizer.normalize_dataset(raw_dataset))
    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = build_mlip_model(architecture_config).to(device)
    objective = _build_objective(
        training_config,
        architecture_config,
        normalizer,
    )
    _load_model_and_objective_state(
        model,
        objective,
        Path(model_path).expanduser(),
        runtime_name=training_config.runtime_name,
        device=device,
    )

    generator = None
    if random_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(random_seed)

    model.eval()
    return _run_force_agreement(
        model,
        objective,
        loader,
        normalizer,
        device,
        generator=generator,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare force predictions on original structures to predictions "
            "from randomly rotated structures after rotating those predicted "
            "forces back to the original frame."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Saved checkpoint file, weights file, or saved-run directory.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Evaluation .xyz or .adapt.pt dataset containing energy and force.",
    )
    parser.add_argument(
        "--formulation",
        required=True,
        help="Objective formulation string, for example 'direct' or 'conservative'.",
    )
    parser.add_argument(
        "--requested-quantities",
        nargs="+",
        choices=("energy", "force"),
        default=("energy", "force"),
        help="Physical quantities to evaluate. Must include force.",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        choices=("atomwise_energy", "direct_force", "scalar_decoder_energy"),
        help="Concrete output heads. If omitted, defaults are inferred.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of structures in each evaluation batch.",
    )
    parser.add_argument(
        "--device",
        help="Evaluation device, for example 'cpu', 'cuda', 'cuda:0', or 'auto'.",
    )
    parser.add_argument(
        "--runtime-name",
        help="Runtime base name used when --model-path is a saved-run directory.",
    )
    parser.add_argument(
        "--attention",
        help="Attention implementation used by the saved model.",
    )
    parser.add_argument(
        "--activation",
        help="Activation implementation used by the saved model.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random rotations.",
    )
    return parser.parse_args(argv)


def _run_force_agreement(
    model: nn.Module,
    objective: Objective,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    *,
    generator: torch.Generator | None,
) -> dict[str, float]:
    component_abs_sum = 0.0
    component_square_sum = 0.0
    max_abs_component = 0.0
    vector_norm_sum = 0.0
    valid_component_count = 0.0
    valid_atom_count = 0.0
    batch_count = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        mask = batch["mask"]

        base_predictions = objective.predict(model, batch)
        base_force = normalizer.unnormalize(
            base_predictions["force"],
            "force",
        )

        rotations = random_rotation_matrices(
            batch["pos"].shape[0],
            device=device,
            dtype=batch["pos"].dtype,
            generator=generator,
        )
        rotated_batch = batch.copy()
        rotated_batch["pos"] = apply_rotations(batch["pos"], rotations, mask)

        rotated_predictions = objective.predict(model, rotated_batch)
        rotated_force = normalizer.unnormalize(
            rotated_predictions["force"],
            "force",
        )
        unrotated_force = apply_rotations(
            rotated_force,
            rotations.transpose(1, 2),
            mask,
        )

        diff = unrotated_force - base_force
        component_mask = mask.unsqueeze(-1).to(dtype=diff.dtype)
        masked_abs = diff.abs() * component_mask
        masked_square = diff.square() * component_mask

        component_abs_sum += float(masked_abs.sum().detach().cpu())
        component_square_sum += float(masked_square.sum().detach().cpu())
        max_abs_component = max(
            max_abs_component,
            float(masked_abs.max().detach().cpu()),
        )
        vector_norm_sum += float(
            (diff.norm(dim=-1) * mask.to(dtype=diff.dtype)).sum().detach().cpu()
        )
        valid_atom_count += float(mask.sum().detach().cpu())
        valid_component_count += float(mask.sum().detach().cpu()) * diff.shape[-1]
        batch_count += 1

    if batch_count == 0:
        raise ValueError("Cannot evaluate force agreement over zero batches.")
    if valid_component_count == 0.0:
        raise ValueError("Cannot evaluate force agreement with zero valid atoms.")

    return {
        "batches": float(batch_count),
        "valid_atoms": valid_atom_count,
        "mae_per_component": component_abs_sum / valid_component_count,
        "rmse_per_component": (component_square_sum / valid_component_count) ** 0.5,
        "mean_vector_norm": vector_norm_sum / valid_atom_count,
        "max_abs_component": max_abs_component,
    }


def _print_results(summary: dict[str, float]) -> None:
    print()
    print("Force Agreement Under Rotation")
    for name, value in summary.items():
        print(f"  {name}: {value:.8g}")


if __name__ == "__main__":
    main()

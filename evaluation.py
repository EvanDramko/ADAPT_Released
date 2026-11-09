"""Evaluate a saved ADAPT model on one labeled dataset."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from configs.arch_config import ArchitectureConfig, EncoderConfig, VALID_HEAD_TYPES
from configs.training_config import ObjectiveConfig, TrainingConfig, VALID_QUANTITIES
from data.adapt_dataset import StructureDataset, collate_batch
from data.normalizer import Normalizer
from data.resolve_data_path import resolve_data_path
from MLIP_arch import build_mlip_model
from objectives.base import Objective
from objectives.conserving import ConservingObjective
from objectives.direct import DirectObjective
from objectives.metric_registry import run_metrics


DEFAULT_HEAD_BY_QUANTITY = {
    "energy": "atomwise_energy",
    "force": "direct_force",
}


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    architecture_config, training_config = _build_configs(args)
    summary, metrics = evaluate_saved_model(
        architecture_config=architecture_config,
        training_config=training_config,
        model_path=args.model_path,
        data_path=args.data_path,
    )
    _print_results(summary, metrics)


def evaluate_saved_model(
    *,
    architecture_config: ArchitectureConfig,
    training_config: TrainingConfig,
    model_path: str | Path,
    data_path: str | Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Load a saved model and evaluate losses/metrics on one labeled dataset."""
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
    _validate_dataset_targets(
        raw_dataset,
        training_config.objective.requested_quantities,
    )
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

    model.eval()
    return _run_evaluation_epoch(
        model,
        objective,
        loader,
        normalizer,
        device,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ADAPT checkpoint on one labeled dataset.",
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
        help="Evaluation .xyz or .adapt.pt dataset with requested targets.",
    )
    parser.add_argument(
        "--formulation",
        required=True,
        help="Objective formulation string, for example 'direct' or 'conservative'.",
    )
    parser.add_argument(
        "--requested-quantities",
        nargs="+",
        choices=sorted(VALID_QUANTITIES),
        default=("energy", "force"),
        help="Physical quantities to evaluate.",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        choices=sorted(VALID_HEAD_TYPES),
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
    return parser.parse_args(argv)


def _build_configs(args: argparse.Namespace) -> tuple[ArchitectureConfig, TrainingConfig]:
    requested_quantities = tuple(args.requested_quantities)
    heads = (
        tuple(args.heads)
        if args.heads is not None
        else _default_heads(requested_quantities, args.formulation)
    )

    objective_config = ObjectiveConfig(
        formulation=args.formulation,
        requested_quantities=requested_quantities,
    )
    training_config = TrainingConfig(objective=objective_config)
    training_replacements = {}
    for name in ("batch_size", "device", "runtime_name"):
        value = getattr(args, name)
        if value is not None:
            training_replacements[name] = value
    training_config = replace(training_config, **training_replacements)

    encoder_config = EncoderConfig()
    encoder_replacements = {}
    for name in ("attention", "activation"):
        value = getattr(args, name)
        if value is not None:
            encoder_replacements[name] = value
    encoder_config = replace(encoder_config, **encoder_replacements)

    return ArchitectureConfig(heads=heads, encoder=encoder_config), training_config


def _default_heads(
    requested_quantities: tuple[str, ...],
    formulation: str,
) -> tuple[str, ...]:
    heads = []
    if "energy" in requested_quantities:
        heads.append(DEFAULT_HEAD_BY_QUANTITY["energy"])
    if "force" in requested_quantities and formulation == "direct":
        heads.append(DEFAULT_HEAD_BY_QUANTITY["force"])
    return tuple(heads)


def _build_objective(
    training_config: TrainingConfig,
    architecture_config: ArchitectureConfig,
    normalizer: Normalizer,
) -> Objective:
    if training_config.objective.formulation == "direct":
        return DirectObjective(training_config.objective, architecture_config)
    if training_config.objective.formulation == "conservative":
        return ConservingObjective(
            training_config.objective,
            architecture_config,
            sigma_y=normalizer.stats["energy"]["std"],
            sigma_x=normalizer.stats["pos"]["std"],
            mu_f=normalizer.stats["force"]["mean"],
            sigma_f=normalizer.stats["force"]["std"],
        )
    raise ValueError(
        f"Unknown formulation: {training_config.objective.formulation!r}"
    )


def _load_model_and_objective_state(
    model: nn.Module,
    objective: Objective,
    model_path: Path,
    *,
    runtime_name: str,
    device: torch.device,
) -> None:
    checkpoint_path = _resolve_checkpoint_path(model_path, runtime_name)
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        objective_state = checkpoint.get("objective_state_dict")
        if objective_state is not None:
            _load_objective_state_dict(objective, objective_state, device)
            print("Loaded objective loss-scale state.")
        else:
            print(
                "No objective loss-scale state found. Evaluation loss scales "
                "will initialize from the first evaluation batch."
            )
    else:
        model.load_state_dict(checkpoint)
        print(
            "Loaded raw model weights. Evaluation loss scales will initialize "
            "from the first evaluation batch."
        )


def _resolve_checkpoint_path(model_path: Path, runtime_name: str) -> Path:
    if model_path.is_dir():
        checkpoint_path = model_path / f"{runtime_name}.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Saved-run directory does not contain checkpoint: {checkpoint_path}"
            )
        return checkpoint_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    return model_path


def _load_objective_state_dict(
    objective: Objective,
    state: object,
    device: torch.device,
) -> None:
    if not isinstance(state, dict):
        raise ValueError(
            "Checkpoint objective_state_dict must be a dictionary, "
            f"got {type(state).__name__}."
        )

    for name in ("initial_energy_loss", "initial_force_loss"):
        value = state.get(name)
        if value is not None:
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"objective_state_dict['{name}'] must be a tensor or None, "
                    f"got {type(value).__name__}."
                )
            value = value.detach().to(device)
        setattr(objective, name, value)


def _run_evaluation_epoch(
    model: nn.Module,
    objective: Objective,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    totals: dict[str, float] = {}
    metric_totals: dict[str, float] = {}
    batch_count = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        result = objective.compute_loss(
            model,
            batch,
            training=False,
            return_predictions=True,
        )

        _accumulate(totals, "loss", result.loss.detach())
        for name, value in result.losses.items():
            _accumulate(totals, name, value)

        physical_predictions = _unnormalize_fields(result.predictions, normalizer)
        physical_truth = _unnormalize_fields(
            {
                field: batch[field]
                for field in ("energy", "force")
                if field in batch
            },
            normalizer,
        )
        metric_values = run_metrics(
            physical_predictions,
            physical_truth,
            has_force="force" in physical_predictions and "force" in physical_truth,
            has_energy="energy" in physical_predictions and "energy" in physical_truth,
            atom_mask=batch["mask"],
        )
        for name, value in metric_values.items():
            metric_totals[name] = metric_totals.get(name, 0.0) + float(value)

        batch_count += 1

    return _average(totals, batch_count), _average(metric_totals, batch_count)


def _load_dataset_dict(path: Path) -> dict[str, torch.Tensor]:
    dataset = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(dataset, dict):
        raise ValueError(f"The dataset file must contain a dictionary: {path}")
    print(
        f"Loaded evaluation dataset from {path}: "
        f"{_dataset_structure_count(dataset)} structures."
    )
    return dataset


def _dataset_structure_count(dataset: dict[str, torch.Tensor]) -> int:
    offsets = dataset.get("offsets")
    if not isinstance(offsets, torch.Tensor):
        raise ValueError("Dataset must contain an 'offsets' tensor.")
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError(
            "Dataset offsets must be a one-dimensional tensor with at least two values."
        )
    return int(offsets.numel() - 1)


def _validate_dataset_targets(
    dataset: dict[str, torch.Tensor],
    requested_quantities: tuple[str, ...],
) -> None:
    missing = [
        quantity
        for quantity in requested_quantities
        if quantity not in dataset
    ]
    if missing:
        raise ValueError(
            f"The evaluation dataset is missing requested target fields: {missing}."
        )


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps":
        raise RuntimeError("MPS is not available for this project.")
    return device


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _unnormalize_fields(
    tensors: dict[str, torch.Tensor],
    normalizer: Normalizer,
) -> dict[str, torch.Tensor]:
    return {
        field: normalizer.unnormalize(tensor, field).detach()
        for field, tensor in tensors.items()
    }


def _accumulate(
    totals: dict[str, float],
    name: str,
    value: torch.Tensor,
) -> None:
    totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())


def _average(totals: dict[str, float], count: int) -> dict[str, float]:
    if count == 0:
        raise ValueError("Cannot average over zero batches.")
    return {
        name: total / count
        for name, total in totals.items()
    }


def _print_results(
    summary: dict[str, float],
    metrics: dict[str, float],
) -> None:
    print()
    print("Evaluation Losses")
    for name, value in sorted(summary.items()):
        print(f"  {name}: {value:.8g}")

    print()
    print("Evaluation Metrics")
    if not metrics:
        print("  No metrics were available for the requested outputs.")
        return
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.8g}")


if __name__ == "__main__":
    main()

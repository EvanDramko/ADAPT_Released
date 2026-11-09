"""Command line entry point for easy-start ADAPT training runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

from configs.arch_config import (
    ArchitectureConfig,
    EncoderConfig,
    VALID_HEAD_TYPES,
)
from configs.training_config import (
    ObjectiveConfig,
    TrainingConfig,
    VALID_QUANTITIES,
)
from training_coordination import run_training_loop


DEFAULT_HEAD_BY_QUANTITY = {
    "energy": "atomwise_energy",
    "force": "direct_force",
}


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    architecture_config, training_config = _build_configs(args)
    _validate_norm_stats(training_config)
    run_training_loop(architecture_config, training_config)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an easy-start ADAPT training job. Optimization hyperparameters "
            "such as learning rate, deterministic mode, and random seed are "
            "intentionally config-file-only."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--train-data-path",
        help="Training .xyz or .adapt.pt file. XYZ files are converted if needed.",
    )
    parser.add_argument(
        "--validation-data-path",
        type=_optional_path,
        help="Validation .xyz or .adapt.pt file, or 'none' to reserve validation from training data.",
    )
    parser.add_argument(
        "--test-data-path",
        type=_optional_path,
        help="Optional test .xyz or .adapt.pt file, or 'none'.",
    )
    parser.add_argument(
        "--saved-model-path",
        type=_optional_path,
        help="Optional checkpoint file or saved-run directory to resume from.",
    )
    parser.add_argument(
        "--results-output-path",
        help="Directory where checkpoint and history files will be written.",
    )
    parser.add_argument(
        "--runtime-name",
        help="Base name for checkpoint and history files.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Total epoch count to train until, including any loaded history.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of structures in each training and evaluation batch.",
    )
    parser.add_argument(
        "--device",
        help="Training device, for example 'cpu', 'cuda', 'cuda:0', or 'auto'.",
    )
    parser.add_argument(
        "--val-eval-freq",
        type=int,
        help="Evaluate validation set every N epochs.",
    )
    parser.add_argument(
        "--rot-augmentation",
        dest="rotation_augmentation",
        action=argparse.BooleanOptionalAction,
        help="Enable or disable random rotation augmentation during training.",
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
        required=True,
        help="Physical quantities to train directly.",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        choices=sorted(VALID_HEAD_TYPES),
        help=(
            "Concrete model output head implementations. If omitted, defaults "
            "are selected from --requested-quantities."
        ),
    )
    parser.add_argument(
        "--attention",
        help="Encoder attention implementation name.",
    )
    parser.add_argument(
        "--activation",
        help="Encoder activation implementation name.",
    )

    return parser.parse_args(argv)


def _build_configs(args: argparse.Namespace) -> tuple[ArchitectureConfig, TrainingConfig]:
    requested_quantities = tuple(args.requested_quantities)
    heads = (
        tuple(args.heads)
        if hasattr(args, "heads")
        else _default_heads(requested_quantities, args.formulation)
    )

    objective_kwargs = {
        "formulation": args.formulation,
        "requested_quantities": requested_quantities,
    }
    objective_config = ObjectiveConfig(**objective_kwargs)

    training_config = TrainingConfig(objective=objective_config)
    training_replacements = {}
    for name in (
        "train_data_path",
        "validation_data_path",
        "test_data_path",
        "saved_model_path",
        "results_output_path",
        "runtime_name",
        "epochs",
        "batch_size",
        "device",
        "val_eval_freq",
        "rotation_augmentation",
    ):
        if hasattr(args, name):
            training_replacements[name] = getattr(args, name)
    training_config = replace(
        training_config,
        **training_replacements,
    )

    encoder_config = EncoderConfig()
    encoder_replacements = {}
    for name in ("attention", "activation"):
        if hasattr(args, name):
            encoder_replacements[name] = getattr(args, name)
    encoder_config = replace(
        encoder_config,
        **encoder_replacements,
    )
    architecture_config = ArchitectureConfig(heads=heads, encoder=encoder_config)

    return architecture_config, training_config


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


def _optional_path(value: str) -> str | None:
    if value.lower() in {"none", "null"}:
        return None
    return value


def _validate_norm_stats(training_config: TrainingConfig) -> None:
    if training_config.norm_stats_path is None:
        raise ValueError(
            "TrainingConfig.norm_stats_path is None. Set a valid stats path "
            "in configs/training_config.py."
        )

    stats_path = Path(training_config.norm_stats_path).expanduser()
    if not stats_path.is_file():
        raise FileNotFoundError(
            "Normalization statistics file does not exist. Set "
            f"TrainingConfig.norm_stats_path to a valid file. Got: {stats_path}"
        )

    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    if not isinstance(stats, dict):
        raise ValueError(
            f"Normalization statistics file must contain a dictionary: {stats_path}"
        )

    required_fields = {"pos", "z", "charge", *training_config.objective.requested_quantities}
    missing_fields = sorted(required_fields - set(stats))
    if missing_fields:
        raise ValueError(
            "Normalization statistics are missing fields required by this run: "
            f"{missing_fields}. Stats path: {stats_path}"
        )

    for field in sorted(required_fields):
        field_stats = stats[field]
        if not isinstance(field_stats, dict):
            raise ValueError(f"Statistics for {field!r} must be a dictionary.")
        if "mean" not in field_stats or "std" not in field_stats:
            raise ValueError(f"Statistics for {field!r} must contain mean and std.")

        mean = torch.as_tensor(field_stats["mean"])
        std = torch.as_tensor(field_stats["std"])
        if mean.numel() != 1 or std.numel() != 1:
            raise ValueError(f"Mean and standard deviation for {field!r} must be scalars.")
        if not torch.isfinite(mean).item() or not torch.isfinite(std).item():
            raise ValueError(f"Statistics for {field!r} must be finite.")
        if std.item() < 0:
            raise ValueError(f"Standard deviation for {field!r} cannot be negative.")

    print(f"Using normalization statistics from: {stats_path}")


if __name__ == "__main__":
    main()

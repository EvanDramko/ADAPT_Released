"""Top-level training orchestration for MLIP models."""

from __future__ import annotations

import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from configs.arch_config import ArchitectureConfig
from configs.training_config import TrainingConfig
from data.adapt_dataset import StructureDataset, collate_batch
from data.augmentation_utility import rotate_batch
from data.normalizer import Normalizer
from data.resolve_data_path import resolve_data_path
from MLIP_arch import build_mlip_model
from MLIP_arch.core_arch.attentions.mask_utility import sanity_check_mask_convention
from objectives.direct import DirectObjective
from objectives.metric_registry import run_metrics
from objectives.base import Objective


VALIDATION_FALLBACK_SIZE = 500


def run_training_loop(
    architecture_config: ArchitectureConfig,
    training_config: TrainingConfig,
) -> dict[str, list[dict[str, float]]]:
    """Run direct-objective training from validated configuration objects."""
    runtime_start = time.perf_counter()

    # check if formulation is set to direct or conservative, and then set an isconservative flag. 
    if training_config.objective.formulation == "direct":
        isconservative = False
    elif training_config.objective.formulation == "conservative":
        isconservative = True
    else: 
        raise NotImplementedError(
            "Only the direct objective is currently implemented in training_coordination.py."
        )

    device = _resolve_device(training_config.device)
    run_paths = _prepare_run_paths(training_config)
    print(f"Saving run outputs to: {run_paths['output_dir'].resolve()}")
    data_generator = _configure_determinism(training_config)

    train_path = resolve_data_path(training_config.train_data_path)
    validation_path = (
        resolve_data_path(training_config.validation_data_path)
        if training_config.validation_data_path is not None
        else None
    )
    test_path = (
        resolve_data_path(training_config.test_data_path)
        if training_config.test_data_path is not None
        else None
    )

    normalize_Z = architecture_config.embedding.species_embedding != "Z_table"
    normalize_charge = architecture_config.embedding.charge_embedding != "charge_table"
    normalizer = Normalizer(training_config.norm_stats_path, normalize_charge=normalize_charge, normalize_Z=normalize_Z)

    if isconservative:
        sigma_y = normalizer.stats["energy"]["std"]
        sigma_x = normalizer.stats["pos"]["std"]
        mu_f = normalizer.stats["force"]["mean"]
        sigma_f = normalizer.stats["force"]["std"]

    train_raw = _load_dataset_dict(train_path, dataset_name="training")
    _validate_dataset_targets(
        train_raw,
        training_config.objective.requested_quantities,
        dataset_name="training",
    )
    train_target_fields = _target_fields(train_raw)
    startnorm = time.time()
    normed_data = normalizer.normalize_dataset(train_raw)
    endnorm = time.time()
    print("Time taken to do the normalization is: ", endnorm-startnorm)
    train_dataset = StructureDataset(normed_data)

    validation_dataset = None
    if validation_path is not None:
        validation_raw = _load_dataset_dict(validation_path, dataset_name="validation")
        _validate_dataset_targets(
            validation_raw,
            training_config.objective.requested_quantities,
            dataset_name="validation",
        )
        _validate_matching_targets(
            train_target_fields,
            _target_fields(validation_raw),
            dataset_name="validation",
        )
        validation_dataset = StructureDataset(normalizer.normalize_dataset(validation_raw))

    test_dataset = None
    if test_path is not None:
        test_raw = _load_dataset_dict(test_path, dataset_name="test")
        _validate_dataset_targets(
            test_raw,
            training_config.objective.requested_quantities,
            dataset_name="test",
        )
        _validate_matching_targets(
            train_target_fields,
            _target_fields(test_raw),
            dataset_name="test",
        )
        test_dataset = StructureDataset(normalizer.normalize_dataset(test_raw))

    if validation_dataset is None:
        if len(train_dataset) <= VALIDATION_FALLBACK_SIZE:
            raise ValueError(
                "No validation_data_path was provided, and the training dataset "
                f"has {len(train_dataset)} structures. At least "
                f"{VALIDATION_FALLBACK_SIZE + 1} structures are required to "
                f"reserve the last {VALIDATION_FALLBACK_SIZE} for validation."
            )
        split_index = len(train_dataset) - VALIDATION_FALLBACK_SIZE
        validation_dataset = Subset(
            train_dataset,
            range(split_index, len(train_dataset)),
        )
        train_dataset = Subset(train_dataset, range(split_index))

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator=data_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
    sanity_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    _run_valid_mask_sanity_check(sanity_loader)

    model = build_mlip_model(architecture_config).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params:,}")
    print()

    if isconservative:
        from objectives.conserving import ConservingObjective

        objective = ConservingObjective(
            training_config.objective,
            architecture_config,
            sigma_y=sigma_y,
            sigma_x=sigma_x,
            mu_f=mu_f,
            sigma_f=sigma_f,
        )
    else:
        objective = DirectObjective(training_config.objective, architecture_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    history = _empty_history()
    _load_saved_run_if_requested(
        model,
        optimizer,
        objective,
        training_config,
        run_paths,
        device,
        history,
    )

    completed_epochs = len(history["train"])
    target_epochs = training_config.epochs
    if completed_epochs >= target_epochs:
        raise ValueError(
            f"Saved history already contains {completed_epochs} completed epoch(s), "
            f"which leaves no remaining epochs to be trained to hit the requested total of {target_epochs}."
        )


    remaining_epochs = target_epochs - completed_epochs
    if completed_epochs:
        print(
            f"Resuming training from epoch {completed_epochs + 1} through "
            f"{target_epochs} ({remaining_epochs} additional epoch(s))."
        )
    else:
        print(f"Training will run for {target_epochs} total epoch(s).")

    for epoch in range(completed_epochs + 1, target_epochs + 1):
        epoch_start = time.perf_counter()
        train_summary = _run_training_epoch(
            model,
            objective,
            train_loader,
            optimizer,
            device,
            rotation_augmentation=training_config.rotation_augmentation,
        )
        train_summary["epoch"] = float(epoch)
        history["train"].append(train_summary)

        if epoch % training_config.val_eval_freq == 0:
            validation_summary, metric_summary = _run_validation_epoch(
                model,
                objective,
                validation_loader,
                normalizer,
                device,
            )
            validation_summary["epoch"] = float(epoch)
            metric_summary["epoch"] = float(epoch)
            history["validation"].append(validation_summary)
            history["validation_metrics"].append(metric_summary)
            if test_loader is not None:
                test_summary, test_metric_summary = _run_validation_epoch(
                    model,
                    objective,
                    test_loader,
                    normalizer,
                    device,
                )
                test_summary["epoch"] = float(epoch)
                test_metric_summary["epoch"] = float(epoch)
                history["test"].append(test_summary)
                history["test_metrics"].append(test_metric_summary)

        elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch}/{target_epochs} "
            f"train_loss={train_summary['loss']:.6g} "
            f"time={elapsed:.2f}s"
        )
        _save_run_artifacts(model, optimizer, objective, history, run_paths, epoch)

    print("Done with training.")
    elapsed = time.perf_counter() - runtime_start
    print(f"Total runtime: {elapsed:.2f}s")
    return history


def _empty_history() -> dict[str, list[dict[str, float]]]:
    return {
        "train": [],
        "validation": [],
        "validation_metrics": [],
        "test": [],
        "test_metrics": [],
    }


def _prepare_run_paths(training_config: TrainingConfig) -> dict[str, Path]:
    output_dir = Path(training_config.results_output_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_name = training_config.runtime_name
    return {
        "output_dir": output_dir,
        "checkpoint": output_dir / f"{runtime_name}.pth",
        "history": output_dir / f"{runtime_name}_histories.npz",
    }


def _configure_determinism(training_config: TrainingConfig) -> torch.Generator | None:
    if not training_config.deterministic:
        return None

    warnings.warn(
        "Deterministic training may hurt model runtime performance.",
        stacklevel=2,
    )
    seed = training_config.random_seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    print(f"Deterministic training enabled with random_seed={seed}.")
    return generator


def _load_saved_run_if_requested(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    objective: Objective,
    training_config: TrainingConfig,
    run_paths: dict[str, Path],
    device: torch.device,
    history: dict[str, list[dict[str, float]]],
) -> None:
    if training_config.saved_model_path is None:
        print("No saved model path provided. Starting from a randomly initialized model.")
        return

    checkpoint_path, history_path = _resolve_saved_run_paths(
        Path(training_config.saved_model_path).expanduser(),
        training_config.runtime_name,
    )
    print(f"Loading saved model from: {checkpoint_path}")
    checkpoint = _load_checkpoint_payload(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded model weights.")

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("Loaded optimizer state.")
    else:
        warnings.warn(
            f"Checkpoint at {checkpoint_path} does not contain optimizer state. "
            "Continuing with a freshly initialized optimizer.",
            stacklevel=2,
        )

    objective_state = checkpoint.get("objective_state_dict")
    if objective_state is not None:
        _load_objective_state_dict(objective, objective_state, device)
        print("Loaded objective loss-scale state.")
    else:
        print("No objective loss-scale state found. Loss scales will be initialized from the next loss calculation.")

    if history_path is not None:
        loaded_history = _load_history(history_path)
        history.clear()
        history.update(loaded_history)
        print(
            f"Loaded history from {history_path} with "
            f"{len(history['train'])} completed training epoch(s)."
        )
        _validate_checkpoint_history_alignment(
            checkpoint,
            checkpoint_path,
            history_path,
            completed_epochs=len(history["train"]),
        )
    else:
        print("No history file found for saved model. Resuming from weights only.")

    if checkpoint_path != run_paths["checkpoint"]:
        print(f"New checkpoints will be written to: {run_paths['checkpoint']}")


def _resolve_saved_run_paths(
    saved_model_path: Path,
    runtime_name: str,
) -> tuple[Path, Path | None]:
    if saved_model_path.is_dir():
        checkpoint_path = saved_model_path / f"{runtime_name}.pth"
        history_path = saved_model_path / f"{runtime_name}_histories.npz"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Saved run directory does not contain checkpoint: {checkpoint_path}"
            )
        return checkpoint_path, history_path if history_path.is_file() else None

    if saved_model_path.is_file():
        history_path = saved_model_path.with_name(
            saved_model_path.stem + "_histories.npz"
        )
        return saved_model_path, history_path if history_path.is_file() else None

    raise FileNotFoundError(f"Saved model path does not exist: {saved_model_path}")


def _load_checkpoint_payload(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    return {"model_state_dict": checkpoint}


def _validate_checkpoint_history_alignment(
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    history_path: Path,
    *,
    completed_epochs: int,
) -> None:
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_history_length = checkpoint.get("history_length")
    if checkpoint_epoch is None and checkpoint_history_length is None:
        warnings.warn(
            f"Checkpoint at {checkpoint_path} does not contain epoch/history metadata. "
            f"Cannot verify alignment with {history_path}; continuing.",
            stacklevel=2,
        )
        return

    mismatches = []
    if checkpoint_epoch is not None and int(checkpoint_epoch) != completed_epochs:
        mismatches.append(
            f"checkpoint epoch={checkpoint_epoch}, history length={completed_epochs}"
        )
    if checkpoint_history_length is not None and int(checkpoint_history_length) != completed_epochs:
        mismatches.append(
            "checkpoint history_length="
            f"{checkpoint_history_length}, history length={completed_epochs}"
        )
    if mismatches:
        raise ValueError(
            "Checkpoint/history mismatch while loading saved run. "
            f"Checkpoint: {checkpoint_path}. History: {history_path}. "
            + "; ".join(mismatches)
        )


def _save_run_artifacts(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    objective: Objective,
    history: dict[str, list[dict[str, float]]],
    run_paths: dict[str, Path],
    epoch: int,
) -> None:
    _save_history(history, run_paths["history"])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "objective_state_dict": _objective_state_dict(objective),
            "epoch": epoch,
            "history_length": len(history["train"]),
        },
        run_paths["checkpoint"],
    )
    print(f"Saved checkpoint to {run_paths['checkpoint']}")
    print(f"Saved histories to {run_paths['history']}")


def _objective_state_dict(objective: Objective) -> dict[str, torch.Tensor | None]:
    return {
        "initial_energy_loss": _detach_optional_scalar(
            getattr(objective, "initial_energy_loss", None)
        ),
        "initial_force_loss": _detach_optional_scalar(
            getattr(objective, "initial_force_loss", None)
        ),
    }


def _detach_optional_scalar(value: object) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "Objective loss-scale values must be tensors or None, "
            f"got {type(value).__name__}."
        )
    return value.detach().cpu()


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


def _save_history(
    history: dict[str, list[dict[str, float]]],
    history_path: Path,
) -> None:
    payload = {
        name: np.array(json.dumps(entries), dtype=np.str_)
        for name, entries in history.items()
    }
    np.savez(history_path, **payload)


def _load_history(history_path: Path) -> dict[str, list[dict[str, float]]]:
    payload = np.load(history_path, allow_pickle=False)
    history = _empty_history()
    for name in history:
        if name in payload.files:
            history[name] = json.loads(str(payload[name].item()))
    return history


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS is disabled due to backend alignment issues in conservative training.

        return torch.device("cpu")

    device = torch.device(requested_device)
    print("Device is: ", device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps":
        raise RuntimeError("MPS is not available for this project.")
    return device


def _load_dataset_dict(path, *, dataset_name: str) -> dict[str, torch.Tensor]:
    dataset = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(dataset, dict):
        raise ValueError(f"The dataset file must contain a dictionary: {path}")
    print(
        f"Loaded {dataset_name} dataset from {path}: "
        f"{_dataset_structure_count(dataset)} structures."
    )
    return dataset


def _dataset_structure_count(dataset: dict[str, torch.Tensor]) -> int:
    offsets = dataset.get("offsets")
    if not isinstance(offsets, torch.Tensor):
        raise ValueError("Dataset must contain an 'offsets' tensor.")
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("Dataset offsets must be a one-dimensional tensor with at least two values.")
    return int(offsets.numel() - 1)


def _validate_dataset_targets(
    dataset: dict[str, torch.Tensor],
    requested_quantities: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    missing = [
        quantity
        for quantity in requested_quantities
        if quantity not in dataset
    ]
    if missing:
        raise ValueError(
            f"The {dataset_name} dataset is missing requested target fields: {missing}."
        )


def _target_fields(dataset: dict[str, torch.Tensor]) -> set[str]:
    return {
        field
        for field in ("energy", "force")
        if field in dataset
    }


def _validate_matching_targets(
    train_fields: set[str],
    other_fields: set[str],
    *,
    dataset_name: str,
) -> None:
    if other_fields != train_fields:
        raise ValueError(
            f"The {dataset_name} dataset target fields must match the training dataset. "
            f"training={sorted(train_fields)}, {dataset_name}={sorted(other_fields)}."
        )


def _run_valid_mask_sanity_check(loader: DataLoader) -> None:
    """Check the collated valid-token mask once without touching train shuffling."""
    batch = next(iter(loader))
    valid_mask = batch.get("mask")
    if valid_mask is None:
        raise KeyError("Collated batches must contain a 'mask' tensor.")
    sanity_check_mask_convention(valid_mask, expected_convention="sdpa")


def _run_training_epoch(
    model: nn.Module,
    objective: DirectObjective,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    rotation_augmentation: bool,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    batch_count = 0

    total_batches = len(loader)
    for batch in loader:
        batch = _move_batch(batch, device)
        if rotation_augmentation:
            batch = rotate_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        result = objective.compute_loss(
            model,
            batch,
            training=True,
            return_predictions=False,
        )
        result.loss.backward()
        optimizer.step()

        _accumulate(totals, "loss", result.loss.detach())
        for name, value in result.losses.items():
            _accumulate(totals, name, value)
        if batch_count % 500 == 0:
            print(
                f"  batch {batch_count + 1}/{total_batches} "
                f"train_loss={result.loss.detach().item():.6g}"
            )
        batch_count += 1

    return _average(totals, batch_count)


def _run_validation_epoch(
    model: nn.Module,
    objective: DirectObjective,
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

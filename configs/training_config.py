"""Configuration dataclasses for training MLIP models."""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
import torch


VALID_OBJECTIVES = frozenset(("direct", "conservative"))
VALID_QUANTITIES = frozenset(("energy", "force"))


@dataclass(frozen=True)
class ObjectiveConfig:
    """Define how model outputs are used to compute the training loss."""

    formulation: str
    requested_quantities: tuple[str, ...]
    energy_weight: float = 0.5
    force_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.formulation not in VALID_OBJECTIVES:
            raise ValueError(
                f"Unknown objective '{self.formulation}'. "
                f"Valid objectives: {sorted(VALID_OBJECTIVES)}."
            )

        if not self.requested_quantities:
            raise ValueError("requested_quantities must contain at least one quantity.")
        if len(set(self.requested_quantities)) != len(self.requested_quantities):
            raise ValueError(
                f"Duplicate quantity predictions are not allowed: {self.requested_quantities}."
            )

        unknown_outputs = sorted(set(self.requested_quantities) - VALID_QUANTITIES)
        if unknown_outputs:
            raise ValueError(
                f"Unknown requested quantities {unknown_outputs}. "
                f"Valid quantities: {sorted(VALID_QUANTITIES)}."
            )

        if self.formulation == "conservative" and not {
            "energy",
            "force",
        }.issubset(self.requested_quantities):
            raise ValueError(
                "A conservative objective requires both energy and forces as targets."
            )
        
        if "energy" in self.requested_quantities:
            if not math.isfinite(self.energy_weight) or self.energy_weight < 0.0:
                raise ValueError("Energy weight is invalid, must be finite and non-negative")
            elif self.energy_weight == 0.0:
                warnings.warn("Running training with zero weight on the energy loss component.")


        if "force" in self.requested_quantities:
            if not math.isfinite(self.force_weight) or self.force_weight < 0.0:
                raise ValueError("Force weight is invalid, must be finite and non-negative")
            elif self.force_weight == 0.0:
                warnings.warn("Running training with zero weight on the force loss component!")


@dataclass(frozen=True)
class TrainingConfig:
    """Define the data and optimization settings for one training run."""

    objective: ObjectiveConfig
    train_data_path: str = "./data/tiny_omol_data/train_sample.xyz"
    norm_stats_path: str = "./data/norm_stats.pt"
    validation_data_path: str | None = "./data/tiny_omol_data/test_sample.xyz"
    test_data_path: str | None = None
    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 1e-4
    device: str = "auto"
    val_eval_freq: int = 1
    saved_model_path: str | None = None
    results_output_path: str = "./outputs/"
    runtime_name: str = "adapt_runtime"
    deterministic: bool = True
    random_seed: int = 42
    rotation_augmentation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ObjectiveConfig):
            raise TypeError(
                "objective must be an ObjectiveConfig, "
                f"got {type(self.objective).__name__}."
            )
        if not self.train_data_path:
            raise ValueError("train_data_path must be a non-empty path.")
        if not self.norm_stats_path:
            raise ValueError("norm_stats_path must be a non-empty path.")
        for name, path in (
            ("validation_data_path", self.validation_data_path),
            ("test_data_path", self.test_data_path),
            ("saved_model_path", self.saved_model_path),
        ):
            if path == "":
                raise ValueError(f"{name} must be None or a non-empty path.")
        
        
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError(f"batch size must be an integer, got {type(self.batch_size)}")
        elif self.batch_size <= 0:
            raise ValueError(f"batch size must be positive, got {self.batch_size}")
        
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise TypeError(f"epochs must be an integer, got {type(self.epochs)}")
        elif self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be finite and positive, "
                f"got {self.learning_rate}."
            )
        if not self.device:
            raise ValueError("device must be a non-empty string.")
        if not self.results_output_path:
            raise ValueError("results_output_path must be a non-empty string.")
        if not self.runtime_name:
            raise ValueError("runtime_name must be a non-empty string.")
        
        if self.val_eval_freq < 1:
            raise ValueError("Validation set evaluation frequency must be a positive integer!")
        if isinstance(self.deterministic, bool) is False:
            raise TypeError(
                f"deterministic must be a boolean, got {type(self.deterministic).__name__}."
            )
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError(
                f"random_seed must be an integer, got {type(self.random_seed).__name__}."
            )
        if isinstance(self.rotation_augmentation, bool) is False:
            raise TypeError(
                "rotation_augmentation must be a boolean, "
                f"got {type(self.rotation_augmentation).__name__}."
            )

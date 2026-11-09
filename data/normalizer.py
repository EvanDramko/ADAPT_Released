import warnings

import torch


NORMALIZABLE_FIELDS = {"pos", "z", "force", "energy", "charge"}
FIELD_ALIASES = {
    "positions": "pos",
    "position": "pos",
    "Z_species": "z",
    "forces": "force",
}


class Normalizer:
    def __init__(self, stats_path, *,  normalize_Z, normalize_charge):
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        if not isinstance(stats, dict):
            raise ValueError("The statistics file must contain a dictionary")

        self.stats = {}
        self.normalize_Z = normalize_Z
        self.normalize_charge = normalize_charge

        for field, field_stats in stats.items():
            if not isinstance(field_stats, dict):
                raise ValueError(f"Statistics for {field!r} must be a dictionary")
            if "mean" not in field_stats or "std" not in field_stats:
                raise ValueError(f"Statistics for {field!r} must contain mean and std")

            mean = torch.as_tensor(field_stats["mean"])
            std = torch.as_tensor(field_stats["std"])
            if mean.numel() != 1 or std.numel() != 1:
                raise ValueError(f"Mean and standard deviation for {field!r} must be scalars")
            if not torch.isfinite(mean).item() or not torch.isfinite(std).item():
                raise ValueError(f"Statistics for {field!r} must be finite")
            if std.item() < 0:
                raise ValueError(f"Standard deviation for {field!r} cannot be negative")
            if std.item() == 0:
                warnings.warn(
                    f"Standard deviation for {field!r} is zero; using 1 instead",
                    stacklevel=2,
                )
                std = torch.ones_like(std)

            self.stats[field] = {
                "mean": mean.squeeze(),
                "std": std.squeeze(),
            }

    def _canonical_field(self, field):
        return FIELD_ALIASES.get(field, field)

    def _get_stats(self, tensor, field):
        field = self._canonical_field(field)
        if field not in self.stats:
            raise KeyError(f"No normalization statistics found for {field!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("The value to normalize must be a torch.Tensor")

        dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
        mean = self.stats[field]["mean"].to(device=tensor.device, dtype=dtype)
        std = self.stats[field]["std"].to(device=tensor.device, dtype=dtype)
        return tensor.to(dtype=dtype), mean, std

    def normalize(self, tensor, field):
        tensor, mean, std = self._get_stats(tensor, field)
        return (tensor - mean) / std

    def unnormalize(self, tensor, field):
        tensor, mean, std = self._get_stats(tensor, field)
        return tensor * std + mean

    def normalize_dataset(self, dataset):
        if not isinstance(dataset, dict):
            raise TypeError("dataset must be a dictionary")

        normalized = dataset.copy()
        for field in NORMALIZABLE_FIELDS:
            if field not in dataset:
                continue
            if field == "z" and not self.normalize_Z:
                continue
            if field == "charge" and not self.normalize_charge:
                continue
            normalized[field] = self.normalize(dataset[field], field)

        return normalized

"""Resolve a dataset path to an ``.adapt.pt`` file."""

from __future__ import annotations

from pathlib import Path

from data.xyz_2_pt import convert


def resolve_data_path(
    data_path: str | Path,
    *,
    compute_stats: bool = False,
) -> Path:
    """Return an ``.adapt.pt`` path, converting from XYZ when necessary.

    Existing ``.adapt.pt`` files are returned unchanged. When XYZ conversion is
    required, all consistently available energy and force fields are included
    automatically. Normalization statistics are not created unless
    ``compute_stats=True``.
    """
    source_path = Path(data_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source_path}")

    if source_path.name.endswith(".adapt.pt"):
        return source_path

    if source_path.suffix.lower() != ".xyz":
        raise ValueError(f"Expected an .xyz or .adapt.pt input path, got: {source_path}")

    resolved_path = source_path.with_suffix(".adapt.pt")
    if resolved_path.is_file():
        return resolved_path

    converted_path, _ = convert(
        source_path,
        resolved_path,
        make_stats=compute_stats,
    )
    return converted_path

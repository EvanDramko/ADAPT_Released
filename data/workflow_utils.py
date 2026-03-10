from __future__ import annotations

from pathlib import Path

from data.xyz_extxyz_converter import convert_and_save_xyz_extxyz_to_ragged_pt


def prepare_ragged_dataset_path(path: str, is_crystal: bool) -> str:
    """
    If `path` is xyz/extxyz, convert to cached `*.ragged.pt` and return that path.
    Otherwise return the original path unchanged.
    """
    src = Path(path).expanduser()
    suffix = src.suffix.lower()
    if suffix not in {".xyz", ".extxyz"}:
        return str(src)

    if not src.exists():
        raise FileNotFoundError(f"Could not find XYZ/EXTXYZ file: {src}")

    out = src.with_suffix(src.suffix + ".ragged.pt")
    needs_refresh = (not out.exists()) or (src.stat().st_mtime > out.stat().st_mtime)
    if needs_refresh:
        print(f"Converting {src} -> {out}")
        convert_and_save_xyz_extxyz_to_ragged_pt(
            input_path=src,
            output_path=out,
            is_crystal=is_crystal,
        )
    return str(out)

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


_HEADER_KV_RE = re.compile(r"(\w+)=(\".*?\"|\S+)")
_ELEMENT_SYMBOLS = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
_SYMBOL_TO_Z = {sym: z for z, sym in enumerate(_ELEMENT_SYMBOLS) if sym}
_FORCE_PROP_NAMES = {"ref_forces", "forces", "force"}


@dataclass(frozen=True)
class PropertySpec:
    name: str
    ptype: str
    count: int


@dataclass(frozen=True)
class ConversionMeta:
    x_feature_names: List[str]
    x_dim: int
    n_frames: int


@dataclass(frozen=True)
class FrameLayout:
    row_width: int
    x_dim: int
    x_feature_names: List[str]
    species_col: int
    pos_start: int
    force_start: int
    numeric_slices: List[Tuple[int, int, int, int]]


def _is_force_property(p: PropertySpec) -> bool:
    return (p.name.lower() in _FORCE_PROP_NAMES) and p.ptype == "R" and p.count == 3


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def _parse_header(header_line: str) -> Dict[str, str]:
    fields = {k: _strip_quotes(v) for k, v in _HEADER_KV_RE.findall(header_line.strip())}
    return fields


def _parse_properties(properties_value: str, frame_idx: int) -> List[PropertySpec]:
    toks = properties_value.split(":")
    # note: toks is not the "tokens" that is given to the transformer, but rather the discrete sections of the header. It is just a naming collision. 
    if len(toks) % 3 != 0: # check if all values come in (name, dtype, count) tuples.
        raise ValueError(
            f"Frame {frame_idx}: malformed Properties field; expected triples name:type:count, got: {properties_value}"
        )

    specs: List[PropertySpec] = []
    for i in range(0, len(toks), 3):
        name, ptype, count_raw = toks[i], toks[i + 1], toks[i + 2]
        try:
            count = int(count_raw)
        except ValueError as exc:
            raise ValueError(
                f"Frame {frame_idx}: invalid count '{count_raw}' for property '{name}'"
            ) from exc
        specs.append(PropertySpec(name=name, ptype=ptype, count=count))
    return specs


def _validate_header_and_properties(
    header_fields: Dict[str, str],
    props: Sequence[PropertySpec],
    frame_idx: int,
    is_crystal: bool,
) -> None:
    if is_crystal:
        if "Lattice" not in header_fields:
            raise ValueError(
                f"Frame {frame_idx}: missing 'Lattice' in header while isCrystal=True"
            )
        if "pbc" not in header_fields:
            raise ValueError(
                f"Frame {frame_idx}: missing 'pbc' in header while isCrystal=True"
            )

    # check if we have position and species listed in the header
    if not any(p.name == "pos" and p.ptype == "R" and p.count == 3 for p in props):
        raise ValueError(
            f"Frame {frame_idx}: Properties must include pos:R:3 (required input coordinates)"
        )
    if not any(p.name == "species" and p.ptype == "S" and p.count == 1 for p in props):
        raise ValueError(
            f"Frame {frame_idx}: Properties must include species:S:1 so atomic number Z can be constructed"
        )

    # "first elements in atomic vector description" => first numeric property included in X must be pos:R:3.
    first_numeric: Optional[PropertySpec] = None
    for p in props:
        if p.ptype in {"R", "I"} and (not _is_force_property(p)):
            first_numeric = p
            break
    if first_numeric is None or not (
        first_numeric.name == "pos" and first_numeric.ptype == "R" and first_numeric.count == 3
    ):
        raise ValueError(
            f"Frame {frame_idx}: first numeric input property must be pos:R:3"
        )

    if not any(_is_force_property(p) for p in props):
        raise ValueError(
            f"Frame {frame_idx}: Properties must include one of REF_forces:R:3, Ref_forces:R:3, or forces:R:3 for Force (aka: Y) labels"
        )


def _atomic_number_from_species(symbol: str, frame_idx: int, atom_i: int) -> float:
    z = _SYMBOL_TO_Z.get(symbol)
    if z is None:
        raise ValueError(
            f"Frame {frame_idx}, atom {atom_i}: unknown chemical symbol '{symbol}' in species:S:1"
        )
    return float(z)


def _build_frame_layout(props: Sequence[PropertySpec], frame_idx: int) -> FrameLayout:
    row_width = sum(p.count for p in props)
    species_col: Optional[int] = None
    pos_start: Optional[int] = None
    force_start: Optional[int] = None
    numeric_slices: List[Tuple[int, int, int, int]] = []
    x_feature_names: List[str] = []

    token_cursor = 0
    x_cursor = 0
    for p in props:
        tok_start = token_cursor
        tok_end = token_cursor + p.count
        token_cursor = tok_end

        if _is_force_property(p):
            force_start = tok_start
            continue
        if p.name == "species":
            species_col = tok_start
            continue
        if p.name == "pos" and p.ptype == "R" and p.count == 3:
            pos_start = tok_start
            x_feature_names.extend(["pos_0", "pos_1", "pos_2", "Z"])
            x_cursor += 4
            continue
        if p.ptype in {"R", "I"}:
            x_start = x_cursor
            x_end = x_cursor + p.count
            numeric_slices.append((tok_start, tok_end, x_start, x_end))
            if p.count == 1:
                x_feature_names.append(p.name)
            else:
                for i in range(p.count):
                    x_feature_names.append(f"{p.name}_{i}")
            x_cursor = x_end

    if species_col is None:
        raise ValueError(f"Frame {frame_idx}: species:S:1 not found while building frame layout")
    if pos_start is None:
        raise ValueError(f"Frame {frame_idx}: pos:R:3 not found while building frame layout")
    if force_start is None:
        raise ValueError(f"Frame {frame_idx}: force property not found while building frame layout")

    return FrameLayout(
        row_width=row_width,
        x_dim=x_cursor,
        x_feature_names=x_feature_names,
        species_col=species_col,
        pos_start=pos_start,
        force_start=force_start,
        numeric_slices=numeric_slices,
    )


def _skip_atom_rows(handle, n_atoms: int, frame_idx: int) -> None:
    for atom_i in range(n_atoms):
        skipped = handle.readline()
        if not skipped:
            raise ValueError(
                f"Frame {frame_idx}: unexpected EOF while skipping atom rows ({atom_i}/{n_atoms})"
            )


def _read_frame_tensors(
    handle,
    n_atoms: int,
    frame_idx: int,
    layout: FrameLayout,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.empty((n_atoms, layout.x_dim), dtype=dtype)
    Y = torch.empty((n_atoms, 3), dtype=dtype)

    for atom_i in range(n_atoms):
        atom_line = handle.readline()
        if not atom_line:
            raise ValueError(
                f"Frame {frame_idx}: unexpected EOF while reading atom rows ({atom_i}/{n_atoms})"
            )
        cols = atom_line.split()
        if len(cols) != layout.row_width:
            raise ValueError(
                f"Frame {frame_idx}, atom {atom_i}: expected {layout.row_width} columns from Properties, got {len(cols)}"
            )

        # pos -> x[0:3]
        try:
            X[atom_i, 0] = float(cols[layout.pos_start])
            X[atom_i, 1] = float(cols[layout.pos_start + 1])
            X[atom_i, 2] = float(cols[layout.pos_start + 2])
        except ValueError as exc:
            raise ValueError(
                f"Frame {frame_idx}, atom {atom_i}: property 'pos' contains non-numeric value"
            ) from exc

        # species -> x[3] = atomic number
        species_symbol = cols[layout.species_col]
        X[atom_i, 3] = _atomic_number_from_species(species_symbol, frame_idx, atom_i)

        # all other numeric properties
        for tok_start, tok_end, x_start, x_end in layout.numeric_slices:
            for j, tok_i in enumerate(range(tok_start, tok_end)):
                try:
                    X[atom_i, x_start + j] = float(cols[tok_i])
                except ValueError as exc:
                    raise ValueError(
                        f"Frame {frame_idx}, atom {atom_i}: non-numeric value in numeric property columns"
                    ) from exc

        # force -> Y
        try:
            Y[atom_i, 0] = float(cols[layout.force_start])
            Y[atom_i, 1] = float(cols[layout.force_start + 1])
            Y[atom_i, 2] = float(cols[layout.force_start + 2])
        except ValueError as exc:
            raise ValueError(
                f"Frame {frame_idx}, atom {atom_i}: force property contains non-numeric value"
            ) from exc

    return X, Y


def load_ragged_from_xyz_extxyz(
    file_path: str | Path,
    is_crystal: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], ConversionMeta]:
    """
    Convert an XYZ/EXTXYZ file into ragged tensors:
      X_list[i]: (n_i, d_x)
      Y_list[i]: (n_i, 3) from REF_forces:R:3

    Required per frame:
    - Properties must include pos:R:3
    - First numeric input property must be pos:R:3
    - Properties must include REF_forces:R:3
    - If is_crystal=True, header must include Lattice and pbc
    """
    in_path = Path(file_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find XYZ/EXTXYZ file: {in_path}")

    X_list: List[torch.Tensor] = []
    Y_list: List[torch.Tensor] = []
    expected_x_dim: Optional[int] = None
    expected_feature_names: Optional[List[str]] = None
    frame_idx = 0

    with in_path.open("r", encoding="utf-8") as f:
        while True:
            natoms_line = f.readline()
            if not natoms_line:
                break

            natoms_line = natoms_line.strip()
            if not natoms_line:
                continue

            try:
                n_atoms = int(natoms_line)
            except ValueError as exc:
                raise ValueError(
                    f"Frame {frame_idx}: expected first line to be integer atom count, got '{natoms_line}'"
                ) from exc
            if n_atoms <= 0:
                raise ValueError(f"Frame {frame_idx}: atom count must be > 0, got {n_atoms}")

            header_line = f.readline()
            if not header_line:
                raise ValueError(f"Frame {frame_idx}: missing header/comment line after atom count")

            header_fields = _parse_header(header_line)
            if "Properties" not in header_fields:
                raise ValueError(
                    f"Frame {frame_idx}: missing Properties in header. This converter requires extxyz-style headers."
                )

            props = _parse_properties(header_fields["Properties"], frame_idx=frame_idx)
            _validate_header_and_properties(
                header_fields=header_fields,
                props=props,
                frame_idx=frame_idx,
                is_crystal=is_crystal,
            )

            layout = _build_frame_layout(props, frame_idx=frame_idx)
            feature_names = layout.x_feature_names
            x_dim = layout.x_dim
            if expected_x_dim is None:
                expected_x_dim = x_dim
                expected_feature_names = feature_names
            elif x_dim != expected_x_dim:
                raise ValueError(
                    f"Frame {frame_idx}: inconsistent X dimension ({x_dim}) vs first frame ({expected_x_dim})"
                )

            X, Y = _read_frame_tensors(f, n_atoms=n_atoms, frame_idx=frame_idx, layout=layout, dtype=dtype)
            X_list.append(X)
            Y_list.append(Y)
            frame_idx += 1

    if frame_idx == 0:
        raise ValueError(f"No structures found in file: {in_path}")

    assert expected_feature_names is not None
    assert expected_x_dim is not None
    meta = ConversionMeta(
        x_feature_names=expected_feature_names,
        x_dim=expected_x_dim,
        n_frames=frame_idx,
    )
    return X_list, Y_list, meta


def load_one_frame_from_xyz_extxyz(
    file_path: str | Path,
    frame_idx: int = 0,
    is_crystal: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, ConversionMeta]:
    """
    Load a single frame from XYZ/EXTXYZ without materializing the whole file.
    Returns:
      X: (n, d_x)
      Y: (n, 3)
      meta: ConversionMeta for that frame
    """
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be >= 0, got {frame_idx}")

    in_path = Path(file_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find XYZ/EXTXYZ file: {in_path}")

    current_idx = 0
    with in_path.open("r", encoding="utf-8") as f:
        while True:
            natoms_line = f.readline()
            if not natoms_line:
                break

            natoms_line = natoms_line.strip()
            if not natoms_line:
                continue

            try:
                n_atoms = int(natoms_line)
            except ValueError as exc:
                raise ValueError(
                    f"Frame {current_idx}: expected first line to be integer atom count, got '{natoms_line}'"
                ) from exc
            if n_atoms <= 0:
                raise ValueError(f"Frame {current_idx}: atom count must be > 0, got {n_atoms}")

            header_line = f.readline()
            if not header_line:
                raise ValueError(f"Frame {current_idx}: missing header/comment line after atom count")

            header_fields = _parse_header(header_line)
            if "Properties" not in header_fields:
                raise ValueError(
                    f"Frame {current_idx}: missing Properties in header. This converter requires extxyz-style headers."
                )

            props = _parse_properties(header_fields["Properties"], frame_idx=current_idx)
            _validate_header_and_properties(
                header_fields=header_fields,
                props=props,
                frame_idx=current_idx,
                is_crystal=is_crystal,
            )

            layout = _build_frame_layout(props, frame_idx=current_idx)

            if current_idx != frame_idx:
                _skip_atom_rows(f, n_atoms=n_atoms, frame_idx=current_idx)
                current_idx += 1
                continue

            X, Y = _read_frame_tensors(f, n_atoms=n_atoms, frame_idx=current_idx, layout=layout, dtype=dtype)
            feature_names = layout.x_feature_names
            meta = ConversionMeta(
                x_feature_names=feature_names,
                x_dim=layout.x_dim,
                n_frames=1,
            )
            return X, Y, meta

    raise ValueError(f"Requested frame_idx={frame_idx}, but file has only {current_idx} frame(s): {in_path}")


def convert_and_save_xyz_extxyz_to_ragged_pt(
    input_path: str | Path,
    output_path: str | Path,
    is_crystal: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    
    X_list, Y_list, meta = load_ragged_from_xyz_extxyz(
        file_path=input_path,
        is_crystal=is_crystal,
        dtype=dtype,
    )

    payload: Dict[str, Any] = {
        "X": X_list,
        "Y": Y_list,
        "meta": {
            "source_path": str(Path(input_path)),
            "x_feature_names": meta.x_feature_names,
            "x_dim": meta.x_dim,
            "n_frames": meta.n_frames,
            "is_crystal": is_crystal,
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return payload


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert xyz/extxyz file into ragged torch tensors saved as a .pt file"
    )
    parser.add_argument("--input", required=True, help="Path to input xyz/extxyz file")
    parser.add_argument("--output", required=True, help="Path to output .pt file")
    parser.add_argument(
        "--is-crystal",
        action="store_true",
        help="Require Lattice and pbc in each frame header",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    payload = convert_and_save_xyz_extxyz_to_ragged_pt(
        input_path=args.input,
        output_path=args.output,
        is_crystal=args.is_crystal,
    )
    print(
        f"Saved {len(payload['X'])} frames to {args.output} | "
        f"X dim = {payload['X'][0].shape[-1]} | Y dim = {payload['Y'][0].shape[-1]}"
    )


if __name__ == "__main__":
    main()

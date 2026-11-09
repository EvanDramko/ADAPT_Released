import argparse
import math
import warnings

import torch

from xyz_2_pt import ATOMIC_NUMBERS, canonical_name, parse_header


SUPPORTED_PROPERTY_TYPES = {"R", "I", "S", "L"}
EXPECTED_PROPERTY_TYPES = {
    "species": "S",
    "positions": "R",
    "forces": "R",
}
EXPECTED_PROPERTY_WIDTHS = {
    "species": 1,
    "positions": 3,
    "forces": 3,
}
SUPPORTED_HEADER_FIELDS = {"energy", "charge", "spin_multiplicity"}


def parse_properties(value, structure_index):
    parts = value.split(":")
    if len(parts) % 3:
        raise ValueError(
            f"Structure {structure_index}: Properties must contain name:type:width triplets"
        )

    properties = []
    for index in range(0, len(parts), 3):
        name, data_type, width_text = parts[index:index + 3]
        if not name:
            raise ValueError(f"Structure {structure_index}: property name cannot be empty")
        if data_type not in SUPPORTED_PROPERTY_TYPES:
            raise ValueError(
                f"Structure {structure_index}: property {name!r} has unsupported type "
                f"{data_type!r}; expected one of {sorted(SUPPORTED_PROPERTY_TYPES)}"
            )
        try:
            width = int(width_text)
        except ValueError as error:
            raise ValueError(
                f"Structure {structure_index}: property {name!r} has invalid width {width_text!r}"
            ) from error
        if width <= 0:
            raise ValueError(
                f"Structure {structure_index}: property {name!r} must have a positive width"
            )

        category = canonical_name(name)
        if category in EXPECTED_PROPERTY_WIDTHS:
            expected_width = EXPECTED_PROPERTY_WIDTHS[category]
            if width != expected_width:
                raise ValueError(
                    f"Structure {structure_index}: property {name!r} must have width "
                    f"{expected_width}, found {width}"
                )
            expected_type = EXPECTED_PROPERTY_TYPES[category]
            if data_type != expected_type:
                raise ValueError(
                    f"Structure {structure_index}: property {name!r} must have type "
                    f"{expected_type!r}, found {data_type!r}"
                )

        properties.append((name, data_type, width, category))
    return properties


def parse_finite_float(value, description):
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{description} must be numeric, found {value!r}") from error

    if not math.isfinite(number):
        raise ValueError(f"{description} must be finite, found {value!r}")
    float32_number = torch.tensor(number, dtype=torch.float32)
    if not torch.isfinite(float32_number).item():
        raise ValueError(f"{description} is not finite after conversion to float32: {value!r}")
    return number


def parse_integer(value, description):
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{description} must be an integer, found {value!r}") from error


def header_values(header, category):
    return [value for name, value in header.items() if canonical_name(name) == category]


def check_xyz(input_path, max_atoms=None):
    if max_atoms is not None and max_atoms < 1:
        raise ValueError("max_atoms must be at least 1")

    structure_count = 0
    reference_layout = None
    reference_has_forces = None
    reference_has_energy = None
    reference_has_charge = None
    reference_has_spin_multiplicity = None
    extra_fields = set()
    blank_lines_found = False

    with open(input_path, encoding="utf-8") as xyz_file:
        while True:
            count_line = xyz_file.readline()
            if not count_line:
                break
            if not count_line.strip():
                blank_lines_found = True
                continue

            structure_count += 1
            try:
                atom_count = int(count_line.strip())
            except ValueError as error:
                raise ValueError(
                    f"Structure {structure_count}: atom count must be an integer, "
                    f"found {count_line.strip()!r}"
                ) from error
            if atom_count < 1:
                raise ValueError(
                    f"Structure {structure_count}: atom count must be at least 1, "
                    f"found {atom_count}"
                )
            if max_atoms is not None and atom_count > max_atoms:
                raise ValueError(
                    f"Structure {structure_count}: atom count {atom_count} exceeds "
                    f"maximum {max_atoms}"
                )

            header_line = xyz_file.readline()
            if not header_line:
                raise ValueError(f"Structure {structure_count}: missing header line")
            if not header_line.strip():
                raise ValueError(f"Structure {structure_count}: header line is empty")
            header = parse_header(header_line)

            properties_keys = [name for name in header if name.lower() == "properties"]
            if len(properties_keys) > 1:
                raise ValueError(
                    f"Structure {structure_count}: Properties appears more than once"
                )
            if properties_keys:
                properties = parse_properties(header[properties_keys[0]], structure_count)
            else:
                properties = [
                    ("species", "S", 1, "species"),
                    ("pos", "R", 3, "positions"),
                ]

            categories = [category for _, _, _, category in properties if category is not None]
            for required in ("species", "positions"):
                if categories.count(required) != 1:
                    raise ValueError(
                        f"Structure {structure_count}: exactly one {required} property is required"
                    )
            for category in ("forces",):
                if categories.count(category) > 1:
                    raise ValueError(
                        f"Structure {structure_count}: {category} property appears more than once"
                    )

            for name, _data_type, _width, category in properties:
                if category is None:
                    extra_fields.add(name)

            energy_values = header_values(header, "energy")
            charge_values = header_values(header, "charge")
            spin_multiplicity_values = header_values(header, "spin_multiplicity")
            if len(energy_values) > 1:
                raise ValueError(f"Structure {structure_count}: energy appears more than once")
            if len(charge_values) > 1:
                raise ValueError(f"Structure {structure_count}: charge appears more than once")
            if len(spin_multiplicity_values) > 1:
                raise ValueError(
                    f"Structure {structure_count}: spin_multiplicity appears more than once"
                )
            if energy_values:
                parse_finite_float(energy_values[0], f"Structure {structure_count} energy")
            if charge_values:
                parse_integer(charge_values[0], f"Structure {structure_count} charge")
            if spin_multiplicity_values:
                parse_integer(
                    spin_multiplicity_values[0],
                    f"Structure {structure_count} spin_multiplicity",
                )

            for name in header:
                if name.lower() == "properties":
                    continue
                category = canonical_name(name)
                if category not in SUPPORTED_HEADER_FIELDS:
                    extra_fields.add(name)

            layout = tuple((name.lower(), data_type, width) for name, data_type, width, _ in properties)
            has_forces = "forces" in categories
            has_energy = bool(energy_values)
            has_charge = bool(charge_values)
            has_spin_multiplicity = bool(spin_multiplicity_values)
            if reference_layout is None:
                reference_layout = layout
                reference_has_forces = has_forces
                reference_has_energy = has_energy
                reference_has_charge = has_charge
                reference_has_spin_multiplicity = has_spin_multiplicity
            else:
                if layout != reference_layout:
                    raise ValueError(
                        f"Structure {structure_count}: property names or layout differ from structure 1"
                    )
                if has_forces != reference_has_forces:
                    raise ValueError(
                        f"Structure {structure_count}: force presence differs from structure 1"
                    )
                if has_energy != reference_has_energy:
                    raise ValueError(
                        f"Structure {structure_count}: energy presence differs from structure 1"
                    )
                if has_charge != reference_has_charge:
                    raise ValueError(
                        f"Structure {structure_count}: charge presence differs from structure 1"
                    )
                if has_spin_multiplicity != reference_has_spin_multiplicity:
                    raise ValueError(
                        f"Structure {structure_count}: spin_multiplicity presence differs "
                        "from structure 1"
                    )

            expected_columns = sum(width for _, _, width, _ in properties)
            for atom_index in range(1, atom_count + 1):
                atom_line = xyz_file.readline()
                if not atom_line:
                    raise ValueError(
                        f"Structure {structure_count}: unexpected end of file after "
                        f"{atom_index - 1} of {atom_count} atoms"
                    )
                if not atom_line.strip():
                    raise ValueError(
                        f"Structure {structure_count}, atom {atom_index}: atom line is empty"
                    )
                values = atom_line.split()
                if len(values) != expected_columns:
                    raise ValueError(
                        f"Structure {structure_count}, atom {atom_index}: expected "
                        f"{expected_columns} columns, found {len(values)}"
                    )

                column = 0
                for name, data_type, width, category in properties:
                    field_values = values[column:column + width]
                    column += width
                    if category == "species":
                        symbol = field_values[0]
                        if symbol not in ATOMIC_NUMBERS:
                            raise ValueError(
                                f"Structure {structure_count}, atom {atom_index}: "
                                f"unknown species {symbol!r}"
                            )
                    elif data_type == "R":
                        for component_index, value in enumerate(field_values, start=1):
                            parse_finite_float(
                                value,
                                f"Structure {structure_count}, atom {atom_index}, "
                                f"{name} component {component_index}",
                            )
                    elif data_type == "I":
                        for component_index, value in enumerate(field_values, start=1):
                            try:
                                int(value)
                            except ValueError as error:
                                raise ValueError(
                                    f"Structure {structure_count}, atom {atom_index}, "
                                    f"{name} component {component_index} must be an integer, "
                                    f"found {value!r}"
                                ) from error
                    elif data_type == "L":
                        for component_index, value in enumerate(field_values, start=1):
                            if value.lower() not in {"t", "f", "true", "false"}:
                                raise ValueError(
                                    f"Structure {structure_count}, atom {atom_index}, "
                                    f"{name} component {component_index} must be logical, "
                                    f"found {value!r}"
                                )

    if structure_count == 0:
        raise ValueError("The XYZ file contains no structures")
    if extra_fields:
        warnings.warn(
            f"Ignored extra fields: {', '.join(sorted(extra_fields))}",
            stacklevel=2,
        )
    if blank_lines_found:
        warnings.warn("Blank lines were found between structures", stacklevel=2)

    return structure_count


def main():
    parser = argparse.ArgumentParser(
        description="Check whether an XYZ file is valid for xyz_2_pt.py."
    )
    parser.add_argument("input", help="XYZ file to check")
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Maximum allowed atoms per structure",
    )
    args = parser.parse_args()

    try:
        structure_count = check_xyz(args.input, args.max_atoms)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Valid XYZ file: {structure_count} structures checked")


if __name__ == "__main__":
    main()

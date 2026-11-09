import argparse
import shlex
import warnings
from pathlib import Path

import torch


COMMON_NAMES = {
    "species": {"species", "element", "elements", "symbol", "symbols"},
    "positions": {"pos", "position", "positions", "coords", "coordinates"},
    "forces": {"force", "forces", "ref_force", "ref_forces"},
    "energy": {"energy", "energies", "ref_energy", "ref_energies"},
    "charge": {"charge"},
    "spin_multiplicity": {"spin_multiplicity"},
}

ELEMENTS = (
    "- H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn "
    "Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
    "Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm "
    "Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ATOMIC_NUMBERS = {symbol: number for number, symbol in enumerate(ELEMENTS) if number}


def canonical_name(name):
    name = name.lower()
    for canonical, aliases in COMMON_NAMES.items():
        if name in aliases:
            return canonical
    return None


def parse_properties(value):
    parts = value.split(":")
    if len(parts) % 3:
        raise ValueError(f"Invalid Properties declaration: {value}")

    properties = []
    for index in range(0, len(parts), 3):
        name, data_type, count_text = parts[index:index + 3]
        try:
            count = int(count_text)
        except ValueError as error:
            raise ValueError(f"Invalid property width for {name}: {count_text}") from error
        properties.append((name, data_type, count))
    return properties


def parse_header(line):
    lexer = shlex.shlex(line, posix=True, punctuation_chars="=")
    lexer.whitespace_split = True
    tokens = list(lexer)
    header = {}
    index = 0
    while index < len(tokens):
        if index + 2 < len(tokens) and tokens[index + 1] == "=":
            key = tokens[index]
            value = tokens[index + 2]
            if not key or key == "=" or value == "=":
                raise ValueError("Invalid header field around '='")
            if key in header:
                raise ValueError(f"Header field {key!r} appears more than once")
            header[key] = value
            index += 3
        else:
            index += 1
    return header


def tensor_stats(tensor):
    values = tensor.to(torch.float64)
    return {
        "mean": values.mean().to(torch.float32),
        "std": values.std(correction=0).to(torch.float32),
    }


def read_xyz(input_path, make_stats=False):
    selected_outputs = None
    reference_property_layout = None
    reference_header_fields = None
    reference_has_charge = None
    reference_has_spin_multiplicity = None
    positions = []
    atomic_numbers = []
    forces = []
    energies = []
    charges = []
    spin_multiplicities = []
    atom_counts = []
    missing_charge = False
    missing_spin_multiplicity = False
    extra_features = set()

    with open(input_path, encoding="utf-8") as xyz_file:
        structure_index = 0
        while True:
            count_line = xyz_file.readline()
            if not count_line:
                break
            if not count_line.strip():
                continue

            structure_index += 1
            try:
                atom_count = int(count_line)
            except ValueError as error:
                raise ValueError(
                    f"Structure {structure_index}: expected an atom count, got {count_line.strip()!r}"
                ) from error

            header_line = xyz_file.readline()
            if not header_line:
                raise ValueError(f"Structure {structure_index}: missing XYZ header")
            header = parse_header(header_line)

            properties_key = next((key for key in header if key.lower() == "properties"), None)
            if properties_key is None:
                properties = [("species", "S", 1), ("pos", "R", 3)]
            else:
                properties = parse_properties(header[properties_key])

            property_layout = tuple(
                (name.lower(), data_type, width)
                for name, data_type, width in properties
            )
            header_fields = frozenset(
                name.lower() for name in header if name.lower() != "properties"
            )
            if reference_property_layout is None:
                reference_property_layout = property_layout
                reference_header_fields = header_fields
            else:
                if property_layout != reference_property_layout:
                    raise ValueError(
                        f"Structure {structure_index}: property names or layout differ "
                        "from structure 1"
                    )
                if header_fields != reference_header_fields:
                    raise ValueError(
                        f"Structure {structure_index}: header fields differ from structure 1. "
                        f"Expected {sorted(reference_header_fields)}, "
                        f"got {sorted(header_fields)}."
                    )

            columns = []
            found_categories = []
            column_start = 0
            for name, _data_type, width in properties:
                category = canonical_name(name)
                columns.append((name, category, column_start, column_start + width))
                column_start += width
                if category is None:
                    extra_features.add(name)
                else:
                    found_categories.append(category)

            if found_categories.count("species") != 1 or found_categories.count("positions") != 1:
                raise ValueError(
                    f"Structure {structure_index}: exactly one species field and one coordinate field "
                    "are required"
                )
            if found_categories.count("forces") > 1:
                raise ValueError(
                    f"Structure {structure_index}: force field appears more than once"
                )

            has_forces = found_categories.count("forces") == 1

            energy_values = [
                value for name, value in header.items() if canonical_name(name) == "energy"
            ]
            if len(energy_values) > 1:
                raise ValueError(f"Structure {structure_index}: energy appears more than once")
            energy_value = energy_values[0] if energy_values else None
            has_energy = energy_value is not None

            available_outputs = set()
            if has_forces:
                available_outputs.add("forces")
            if has_energy:
                available_outputs.add("energy")
            if selected_outputs is None:
                selected_outputs = available_outputs
            elif available_outputs != selected_outputs:
                raise ValueError(
                    f"Structure {structure_index}: available outputs differ from structure 1. "
                    f"Expected {sorted(selected_outputs)}, got {sorted(available_outputs)}."
                )

            charge_values = [
                value for name, value in header.items() if canonical_name(name) == "charge"
            ]
            if len(charge_values) > 1:
                raise ValueError(f"Structure {structure_index}: charge appears more than once")
            has_charge = bool(charge_values)
            if reference_has_charge is None:
                reference_has_charge = has_charge
            elif has_charge != reference_has_charge:
                raise ValueError(
                    f"Structure {structure_index}: charge presence differs from structure 1"
                )
            if not charge_values:
                missing_charge = True
                charge_value = 0
            else:
                charge_value = charge_values[0]

            spin_multiplicity_values = [
                value for name, value in header.items()
                if canonical_name(name) == "spin_multiplicity"
            ]
            if len(spin_multiplicity_values) > 1:
                raise ValueError(
                    f"Structure {structure_index}: spin_multiplicity appears more than once"
                )
            has_spin_multiplicity = bool(spin_multiplicity_values)
            if reference_has_spin_multiplicity is None:
                reference_has_spin_multiplicity = has_spin_multiplicity
            elif has_spin_multiplicity != reference_has_spin_multiplicity:
                raise ValueError(
                    f"Structure {structure_index}: spin_multiplicity presence differs from structure 1"
                )
            if not spin_multiplicity_values:
                missing_spin_multiplicity = True
                spin_multiplicity_value = 0
            else:
                spin_multiplicity_value = spin_multiplicity_values[0]

            allowed_header_names = {"properties", "charge", "energy", "spin_multiplicity"}
            for name in header:
                category = canonical_name(name)
                if name.lower() != "properties" and category not in allowed_header_names:
                    extra_features.add(name)

            structure_positions = []
            structure_numbers = []
            structure_forces = []
            for atom_index in range(atom_count):
                atom_line = xyz_file.readline()
                if not atom_line:
                    raise ValueError(
                        f"Structure {structure_index}: expected {atom_count} atoms, found {atom_index}"
                    )
                values = atom_line.split()
                if len(values) != column_start:
                    raise ValueError(
                        f"Structure {structure_index}, atom {atom_index + 1}: "
                        f"expected {column_start} values, found {len(values)}"
                    )

                for name, category, start, end in columns:
                    field = values[start:end]
                    if category == "species":
                        if len(field) != 1 or field[0] not in ATOMIC_NUMBERS:
                            raise ValueError(
                                f"Structure {structure_index}, atom {atom_index + 1}: "
                                f"unknown species {field[0]!r}"
                            )
                        structure_numbers.append(ATOMIC_NUMBERS[field[0]])
                    elif category == "positions":
                        if len(field) != 3:
                            raise ValueError(f"Coordinate property {name!r} must have width 3")
                        structure_positions.append([float(value) for value in field])
                    elif category == "forces" and "forces" in selected_outputs:
                        if len(field) != 3:
                            raise ValueError(f"Force property {name!r} must have width 3")
                        structure_forces.append([float(value) for value in field])

            positions.extend(structure_positions)
            atomic_numbers.extend(structure_numbers)
            if "forces" in selected_outputs:
                forces.extend(structure_forces)
            if "energy" in selected_outputs:
                energies.append(float(energy_value))
            charges.append(int(charge_value))
            spin_multiplicities.append(int(spin_multiplicity_value))
            atom_counts.append(atom_count)

    if not atom_counts:
        raise ValueError("The XYZ file contains no structures")
    if missing_charge:
        warnings.warn(
            f"{input_path}: one or more structures have no charge; implicitly assuming charge=0",
            stacklevel=2,
        )
    if missing_spin_multiplicity:
        warnings.warn(
            f"{input_path}: one or more structures have no spin_multiplicity; "
            "implicitly assuming spin_multiplicity=0",
            stacklevel=2,
        )
    if extra_features:
        warnings.warn(
            f"{input_path}: ignored unrecognized features: {', '.join(sorted(extra_features))}",
            stacklevel=2,
        )

    found_fields = ["species", "positions"]
    if "forces" in selected_outputs:
        found_fields.append("force")
    if "energy" in selected_outputs:
        found_fields.append("energy")
    if reference_has_charge:
        found_fields.append("charge")
    if reference_has_spin_multiplicity:
        found_fields.append("spin_multiplicity")
    print(f"Found fields: {', '.join(found_fields)}")
    if not reference_has_charge:
        print("Added field: charge (defaulted to 0)")
    if not reference_has_spin_multiplicity:
        print("Added field: spin_multiplicity (defaulted to 0)")

    natoms = torch.tensor(atom_counts, dtype=torch.int64)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), natoms.cumsum(dim=0)))
    dataset = {
        "pos": torch.tensor(positions, dtype=torch.float32),
        "z": torch.tensor(atomic_numbers, dtype=torch.int64),
        "charge": torch.tensor(charges, dtype=torch.int64),
        "spin_multiplicity": torch.tensor(spin_multiplicities, dtype=torch.int64),
        "natoms": natoms,
        "offsets": offsets,
    }
    if "forces" in selected_outputs:
        dataset["force"] = torch.tensor(forces, dtype=torch.float32)
    if "energy" in selected_outputs:
        dataset["energy"] = torch.tensor(energies, dtype=torch.float32)

    norm_stats = None
    if make_stats:
        norm_stats = {
            "z": tensor_stats(dataset["z"]),
            "pos": tensor_stats(dataset["pos"]),
            "charge": tensor_stats(dataset["charge"]),
            "spin_multiplicity": tensor_stats(dataset["spin_multiplicity"]),
        }
        if "force" in dataset:
            norm_stats["force"] = tensor_stats(dataset["force"])
        if "energy" in dataset:
            norm_stats["energy"] = tensor_stats(dataset["energy"])
    return dataset, norm_stats


def convert(input_path, output_path, make_stats=False):
    dataset, norm_stats = read_xyz(input_path, make_stats)
    output_path = Path(output_path)
    if output_path.name.endswith(".adapt.pt"):
        base_name = output_path.name[:-9]
    elif output_path.suffix == ".pt":
        base_name = output_path.stem
    else:
        base_name = output_path.name

    output_path = output_path.with_name(f"{base_name}.adapt.pt")
    torch.save(dataset, output_path)

    stats_path = None
    if make_stats:
        stats_path = output_path.with_name(f"{base_name}.adapt.norm_stats.pt")
        torch.save(norm_stats, stats_path)
    return output_path, stats_path


def main():
    parser = argparse.ArgumentParser(description="Convert an extended XYZ file to PyTorch tensors.")
    parser.add_argument("input", help="Input XYZ file")
    parser.add_argument("output", help="Output name or .pt file")
    parser.add_argument(
        "--make-stats",
        action="store_true",
        help="Save normalization statistics in a separate .pt file",
    )
    args = parser.parse_args()
    output_path, stats_path = convert(
        args.input,
        args.output,
        args.make_stats,
    )
    print(f"Saved dataset to {output_path}")
    if stats_path is not None:
        print(f"Saved normalization statistics to {stats_path}")


if __name__ == "__main__":
    main()

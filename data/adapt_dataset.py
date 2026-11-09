import torch
from torch.utils.data import Dataset


class StructureDataset(Dataset):
    def __init__(self, data):
        if not isinstance(data, dict):
            raise TypeError("StructureDataset expects a dataset dictionary")

        required_fields = {"pos", "z", "charge", "spin_multiplicity", "offsets"}
        missing_fields = required_fields - data.keys()
        if missing_fields:
            raise ValueError(
                f"Missing required fields: {', '.join(sorted(missing_fields))}"
            )

        self.pos = data["pos"]
        self.Z_species = data["z"]
        self.charge = data["charge"]
        self.spin_multiplicity = data["spin_multiplicity"]
        self.offsets = data["offsets"]
        self.force = data.get("force")
        self.energy = data.get("energy")

        self._validate()

    def _validate(self):
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("pos must have shape (number_of_atoms, 3)")
        if self.Z_species.ndim != 1 or len(self.Z_species) != len(self.pos):
            raise ValueError("z must have one value per atom")
        if self.offsets.ndim != 1 or len(self.offsets) < 2:
            raise ValueError("offsets must be a one-dimensional tensor with at least two values")
        if self.offsets[0].item() != 0 or self.offsets[-1].item() != len(self.pos):
            raise ValueError("offsets must start at 0 and end at the total number of atoms")
        if torch.any(self.offsets[1:] < self.offsets[:-1]):
            raise ValueError("offsets must be in ascending order")

        number_of_structures = len(self.offsets) - 1
        if self.charge.ndim != 1 or len(self.charge) != number_of_structures:
            raise ValueError("charge must have one value per structure")
        if (
            self.spin_multiplicity.ndim != 1
            or len(self.spin_multiplicity) != number_of_structures
        ):
            raise ValueError("spin_multiplicity must have one value per structure")
        if self.force is not None and self.force.shape != self.pos.shape:
            raise ValueError("force must have the same shape as pos")
        if self.energy is not None and (
            self.energy.ndim != 1 or len(self.energy) != number_of_structures
        ):
            raise ValueError("energy must have one value per structure")

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

        start = self.offsets[index].item()
        end = self.offsets[index + 1].item()
        sample = {
            "pos": self.pos[start:end],
            "Z_species": self.Z_species[start:end],
            "charge": self.charge[index],
            "spin_multiplicity": self.spin_multiplicity[index],
        }
        if self.force is not None:
            sample["force"] = self.force[start:end]
        if self.energy is not None:
            sample["energy"] = self.energy[index]
        return sample


def collate_batch(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    optional_fields = {"force", "energy"}
    first_fields = optional_fields.intersection(samples[0])

    # xyz_2_pt.py guarantees that every sample in one dataset has the same
    # output fields.

    batch_size = len(samples)
    max_atoms = max(len(sample["Z_species"]) for sample in samples)

    pos = samples[0]["pos"].new_zeros((batch_size, max_atoms, 3))
    Z_species = samples[0]["Z_species"].new_zeros((batch_size, max_atoms))
    mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)

    batch = {
        "pos": pos,
        "Z_species": Z_species,
        "charge": torch.stack([sample["charge"] for sample in samples]),
        "spin_multiplicity": torch.stack(
            [sample["spin_multiplicity"] for sample in samples]
        ),
        "mask": mask,
    }

    if "force" in first_fields:
        batch["force"] = samples[0]["force"].new_zeros((batch_size, max_atoms, 3))
    if "energy" in first_fields:
        batch["energy"] = torch.stack([sample["energy"] for sample in samples])

    for batch_index, sample in enumerate(samples):
        number_of_atoms = len(sample["Z_species"])
        batch["pos"][batch_index, :number_of_atoms] = sample["pos"]
        batch["Z_species"][batch_index, :number_of_atoms] = sample["Z_species"]
        batch["mask"][batch_index, :number_of_atoms] = True
        if "force" in first_fields:
            batch["force"][batch_index, :number_of_atoms] = sample["force"]

    return batch

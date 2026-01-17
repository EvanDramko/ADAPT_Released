# -------- Make Sample cif File --------
# from monty.serialization import loadfn
from pymatgen.core import Element, Structure
import pickle
import numpy as np

# ISSUE WITH VECTORIZATION IS THAT THE MASKING MAKES DIFFERENT LENGTH TENSORS WHICH THE PYMATGEN LIBRARY CALLS CANNOT HANDLE. 

def getStruct(struct):
    """ Args: struct: torch tensor of shape [n=atoms, 12]
    name: A name for the file without the .cif extension
    """
    fea_elem = struct[:, 3:].numpy() # struct should be torch tensor
    cart_coords = struct[:, :3].numpy()
    def get_element_from_row_col(row, col):
        # A minimal example mapping based on a row and columns.
        element_mapping = {}
        # Iterate over atomic numbers 1 to 104.
        for Z in range(1, 104):
            el = Element.from_Z(Z)

            if el.is_rare_earth_metal:
                continue

            if el.is_noble_gas:
                continue
            element_mapping[(el.row, el.group)] = el.symbol

        element = element_mapping.get((row, col))
        if element is None:
            raise ValueError(f"No element mapping found for row {row}, col {col}")
        return element

    sp_rows = [i[1] for i in fea_elem]
    sp_columns = [i[0] for i in fea_elem]

    species = []
    for c,r in zip(sp_columns, sp_rows):
        species.append(get_element_from_row_col(r,c))

    cell = np.eye(3)*16.406184 # define cell 
    struct = Structure(lattice=cell, species=species, coords=cart_coords, coords_are_cartesian=True)
    return struct

def makeCif(struct, name):
    """ Args: struct: torch tensor of shape [n=atoms, 12]
    name: A name for the file without the .cif extension
    """
    fea_elem = struct[:, 3:].numpy() # struct should be torch tensor
    cart_coords = struct[:, :3].numpy()
    def get_element_from_row_col(row, col):
        # A minimal example mapping based on a row and columns.
        element_mapping = {}
        # Iterate over atomic numbers 1 to 104.
        for Z in range(1, 104):
            el = Element.from_Z(Z)

            if el.is_rare_earth_metal:
                continue

            if el.is_noble_gas:
                continue
            element_mapping[(el.row, el.group)] = el.symbol

        element = element_mapping.get((row, col))
        if element is None:
            raise ValueError(f"No element mapping found for row {row}, col {col}")
        return element

    sp_rows = [i[1] for i in fea_elem]
    sp_columns = [i[0] for i in fea_elem]

    species = []
    for c,r in zip(sp_columns, sp_rows):
        species.append(get_element_from_row_col(r,c))

    cell = np.eye(3)*16.406184 # define cell 
    struct = Structure(lattice=cell, species=species, coords=cart_coords, coords_are_cartesian=True)
    fname = name+".cif"
    struct.to(filename=fname)

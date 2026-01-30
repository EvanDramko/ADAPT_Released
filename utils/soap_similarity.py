# primary author: Yihuang Xiong
# imports
from pymatgen.core import Structure
from numpy.typing import NDArray
import sys
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from dscribe.descriptors import SOAP


def get_soap_vec(struct: Structure) -> NDArray:
    """Get the SOAP vector for each site in the structure.

    Args:
        struct: Structure object to compute the SOAP vector for

    Returns:
        NDArray: SOAP vector for each site in the structure,
            shape (n_sites, n_soap_features)
    """
    adaptor = AseAtomsAdaptor()
    species_ = [str(el) for el in struct.composition.elements]

    soap_desc = SOAP(species=species_, r_cut=5, n_max=8, l_max=6, periodic=True,average="inner")
    vecs = soap_desc.create(adaptor.get_atoms(struct))
    return vecs

def cosine_similarity(vec1, vec2) -> float:
    """Cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def getSoapScore(struct1, struct2):
    firstVec = get_soap_vec(struct1)
    secondVec = get_soap_vec(struct2)
    return cosine_similarity(firstVec, secondVec)

if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    struct1 = Structure.from_file(file1)
    struct2 = Structure.from_file(file2)

    print(cosine_similarity(struct1, struct2))

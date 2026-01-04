# What Is This?
This is a released version of the research code for the *ADAPT* paper found at: https://arxiv.org/abs/2509.24115 . The website associated with this project can be found at: https://evandramko.github.io/ADAPT_webpage/ . 

We present a machine learning force field (MLFF) architecture that does not use graph representations, and instead puts the focus on global interactions and precise distance/geometry measures. The targeted use case of this work is crystal defects, where it shows superior performance compared to other state-of-the-art models like MACE (https://github.com/ACEsuit/mace) or MatterSim (https://github.com/microsoft/mattersim). 

The core focus of this project is force predictions, and the development of the code reflects this. We believe that the model is best used a pre-processing step prior to a DFT relaxation. While we acheive low error in force predictions, compounding errors across steps can still create inaccurate final positions when using directly for a full relaxation. 

Please note: this is research code only. A production-level release is intended with an upcoming derivitive work. Please use accordingly. 

# Task Description
Our study focuses on silicon crystals with defects. Defects may be simple (substitutional, vacancy) or more complex combinations thereof. Starting from unstable structures, our MLFF emulates the force and energy predictions of density functional theory (DFT). While less precise than first-principles methods, our model requires only fractions of a second per step—compared to ~15 minutes per step for DFT. We also provide code for generating complete relaxation trajectories using the MLFF.
The workflow has two main components:

(1) MLFF predictions: Given an atomic structure, the model estimates the forces acting on each atom.

(2) Structural updates: These forces are used to update the crystal structure, advancing it one step closer to equilibrium. This functionality has been temporarily removed pending a release of a new version of the code. 

The cycle repeats until the structure converges and changes become negligible. 

# Imports and Dependancies
Navigate to the environment in which you will be running this code. Ensure that you have the following installed:
1) torch (https://pytorch.org/get-started/locally/)
-- Not strictly necessary but good for side functionality --
1) pymatgen (conda install conda-forge::pymatgen)
2) dscribe (conda install conda-forge::dscribe)
3) ASE (conda install conda-forge::ase)
4) mendeleev (conda install conda-forge::mendeleev) 
5) Matplotlib (conda install conda-forge::matplotlib)

# Code Orginization
The model definitions are provided in the folder ```MLFF_arch``` (short for MLFF architectures). Seperate files called ```trainForce.py``` and ```trainEnergy.py```in the project's home directory is used to train the force prediction model. 

The ```getEnergyPred.py``` and ```getForcePred.py``` provide a way to run the model on an end-to-end relaxation for each of the trajectories reserved for testing. By changing relevant file name parameters, any applicable initial point for a trajectory can be used instead. 

## Data Representation
We propose an alternataive architecture strategy to the graph-based representations common in literature. We represent the atomic structures in their "native" space... as atoms floating in Cartesian space. The model was trained on structures existing within a cube of side length 16.406184 exisiting in the first octant. Experiments have shown that many different representations of atoms can be used effectively. Most important is that the coordinates of the atoms are included. However, in the Si Defects dataset which is the primary focus of this project, each atom is represented by a feature vector of length 12; the values it contains in order are: x, y, z coordinates, periodic table column, periodic table row, electronegativity, covalent radius, number of valence electrons, first ionization energy, electron affinity, atomic radius, and molar volume. Structures are represented as a vector of atomic vectors, creating a 2d vector with shape (nx12) for a structure with n atoms. Notably we do not include atomic number or atomic mass in the atom's feature vector, however we provide utilities to add these into the representation if desired. 

We include all steps from all training trajectories as one ragged list of size (L, n, 12). A weighting factor is included for each atom making the total shape of the data (L, n, 13). For more information about this factor, please reference Section 2.1.2 of the associated paper. During runtime, this factor is removed from the data representation, and instead given to the loss function. *While convenient to append this factor to the data representation, please ensure that you account for this expected extra feature channel when using the code.* 

## Neural Network Architecture
We use a Transformer encoder to create the MLFF. Details about the mathematics of the model can be found at: https://evandramko.github.io/files/transformer.pdf and https://evandramko.github.io/files/attention.pdf. We shrink the embedding size to 3 for the final layer, and read the corresponding vector as the force vector of the atom.
Each atom is embedded into the d_model (aka: d_k) dimension through a multi-layer perceptron (MLP).

In the provided saved model, the transformer has dimensions: d_model = 512, d_ff = 1024, num_heads = 8, layers = 8.The embedding MLP has 3 layers and transforms the data sequentially through dimensions [12, 128, d_model, d_model]. 

# Note on Final Deployment
If intending to use in a high-throughput setup, consider using torch.compile and torch.amp to increase performance. Compile gives a compiled version of the model which is known in literature to save ~10% on runtime, and amp performs auto-swtiching between 16 and 32 bit computations to increase efficiency. 


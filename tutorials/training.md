# Training ADAPT Architectures
Author: Evan Dramko
Last Updated: Jan 30th, 2026

## What Does This Tutorial Cover?
This is a tutorial showing how to train or finetune an ADAPT Force or Energy Prediction model. This tutorial covers model training only. It does not describe how to use the trained model to relax or simulate new materials. For a tutorial on inference time usage, see the "ADAPT Inference Time Guide".

This tutorial assumes that ADAPT has been properly downloaded and an appropriate environment has been created. Instructions for this can be found in the "ADAPT Download Guide".


First, make sure to activate your environment containing [pyTorch](https://pytorch.org/get-started/locally/), and run the check on the package to ensure it has been downloaded properly. 
```python
# activate your environment
conda activate adapt_env

# (optional): run the checks on the model
python setup_tester.py # follow in-terminal prompts
```

# Force Prediction Model

## Data Format

Training on force and energy predictions is a per-frame task. That means that we train on (structure, force) pairs with no concern for their relative order in trajectories.

We expect data to be a python dictionary containing two keys "X" and "Y". We expect that each of "X" and "Y" should be a list of length M. A list is used to allow for different numbers of atoms in each example.

In "X" we expect a torch tensor of shape (n, d) where we have n atoms and d atomic descriptors. Atomic decriptors should include the coordinates and other information about the atom. A common choice in literature is (x, y, z, Z) giving the coordinates and atomic number. Native to the Si Defects dataset which motivated the architecture is a set of 3 coordinates and 9 descriptors giving d=12 total features per atom.

In "Y" we expect a torch tensor of shape (n, 3) where we have n atoms and a set of vector coordinates (x,y,z) denoting the force vector acting on that atom.

```python
# A small sample dataset is included in GitHub. The full Si Defect dataset is available on Zenodo
import torch

D_mini = torch.load("./data/miniDataset.pt")
X = D_mini["X"]
Y = D_mini["Y"]
print("Type of D_mini is: ", type(D_mini))
print("Type of X is: ", type(X))
print("Type of Y is: ", type(Y))
print("Type of data point is: ", type(X[0]))
print("Type of force is: ", type(Y[0]))

print("Number of X elements", len(X))
print("Number of Y elements", len(Y))
print("Size of X elements", X[0].shape)
print("Size of Y elements", Y[0].shape)

# Produces Output:
# Type of D_mini is:  <class 'dict'>
# Type of X is:  <class 'list'>
# Type of Y is:  <class 'list'>
# Type of data point is:  <class 'torch.Tensor'>
# Type of force is:  <class 'torch.Tensor'>
# Number of X elements 20
# Number of Y elements 20
# Size of X elements torch.Size([217, 13]) # since we have a weighting pattern, it has 13 elements: 3 coordinates, 9 descriptors, 1 weighting value
# Size of Y elements torch.Size([217, 3])
```

## How to Setup the Training
ADAPT has been created such that all supported modifications to the runtime can be created by modifying the configuration files in ```ADAPT_released/configs```.

Readability and extensibility were taken into consideration when creating the source code, so modifications to the architecture or other usage is possible for users who want exceptionally fine-grained control over the usage.

### Model Configuration
This class sets the hyperparameters defining the model.


1.   **```d_model```**: dimension of the model embedding
2.   **```d_ff```**: dimension of the intermediate layer for the FFN in the encoder block
3.   **```num_layers```**: number of encoder blocks sequenced together
4.   **```d_out```**: the size of the per-atom output. This should always be 3 if predicting forces: (x, y, z) components of the force vector.
5.   **```dropout_rate```**: occurance rate of dropout normalization in training.
6.   **```num_heads```**: how many heads we split the attention computations into. Note that the torch implementation requires ```num_heads``` be a divisor of ```d_model```.

We also note that DataConfig.atom_vec_length is used implicitly to define the embedding layer.

### Data Configuration
This class stores the values needed by the model to adjust to a given dataset.


1.   **```train_path```**: path to the datafile containing the training data
2.   **```test_path```**: path to the datafile containing the testing data
3.   **```has_weighting```**: True if an importance weighting factor has been added as the last feature to the model (see paper), False otherwise.
4.   **```has_Z```**: True if the fourth element (index 3) contains the atomic number, False otherwise
5.   **```batch_size```**: how many data points are used in parallel in each forward pass. A good starting point is often 32 or 64, but can be decreased to reduce memory usage.
6.   **```num_workers```**: how many workers the training dataloader uses to fetch and prepare the data. In general, should not be modified unless you are familiar with torch functions.
7.   **```atom_vec_length```**: How many features describe each atom. Ex: |(x, y, z, d1, d2)| = 5. We do not include the weighting pattern in the count if ```has_weighting = True```.

As before, we also note that DataConfig.atom_vec_length is used implicitly to define the embedding layer.

### Model Paths
This class stores the values needed by the model to adjust to a given dataset.


1.   **```pretrainPath```**: a path to a base model to finetune, else ```None```.
2.   **```stats_path```**: the path to the saved normalization statistics. Also the path where new stats will be saved if necessary.
3.   **```saveBase```**: a base path to add to all saved files (ex: ```./results/trial_run/```)
4.   **```savedModelName```**: the name to use when saving the pretrained model. If left blank, it will save to: ```MLFF_dev{device}_saved.pth```.


### Training Configuration
This class stores the values needed by the model to adjust to a given dataset.


1.   **```epochs```**: How many epochs to train the model
2.   **```alt_loss_fxn```**: a custom loss function to use during training. If left empty, then it will default to either: ```weightedMSE``` (see paper) if ```DataConfig.has_weighting=True```, else ```MSE```.
3.   **```optimizer```**: optimizer to use when training the model.
4.   **```lr```**: constant step size to use in the optimizer. We do not provide built-in support via the config files for schedulers, as this intended for quick usage by those unfamiliar with neural network training details.
5.   **```exact_reproducibility```**: creates deterministic training setups (controls dropout, etc). Not recommended in general, because it incurs a runtime cost.
6.   **```augmentation```**: if set to True then training and testing will occur with random translations (0-360 degrees) and augmentations (0-10 Angstroms) of the data points.

## Running the Training
Once the config files have been set to your desired setup, all you need to do is run the ```trainForce.py``` file with ```python trainForce.py```. A small dataset has been included for convenience. 

# Training the Energy Predictor
## Data Curation

Unlike the force predictor, the energy predictor takes a fixed size input. So, we must specify a maximum number of atoms prior to initializing the model. For structures with fewer than the maximum number of atoms, we recommend adding "padding" atoms (vectors of all 0) to the structure until it reaches the maximum size.

Since all structures must have the same size, we can save the data as a single pyTorch tensor. For M examples with a maximum of n atoms, and atomic descriptor vector length of d, we create a (M, n, d) tensor.

```python
# Sample dataset construction
# Assume M = 4, n = 10, d = 5
import torch
X_1 = torch.randn(2, 10, 5)
X_2 = torch.randn(2, 8, 5)
padding = torch.zeros(2, 2, 5)
X_2p = torch.cat((X_2, padding), dim=-2)
X = torch.cat((X_1, X_2p), dim=0)
print(X.shape)

# >> output: torch.Size([4,10,5])
```
We output a single value for the structure's formation energy. This means that we need an output tensor of shape (M).

```python
Y = torch.randn(4)
print(Y.shape) #-> One energy value per structure
```

## Configuration Files Setup
Much like the force predictor, the setup for the energy predictor is made entirely from config file: ```ADAPT_released/configs/energy_model_hyperparams.py```.

### Data Configuration
This class stores the needed information about the dataset


1.   **```train_path_X```**: path to (M,n,d) training structure tensor
2.   **```train_path_NRG```**: path to (M) training structure formation energy
3.   **```test_path_X```**: path to (P,n,d) testing structure tensor
4.   **```test_path_NRG```**: path to (P) testing structure formation energy
5.   **```has_Z```**: same as in force model
6.   **```batch_size```**: same as in force model
7.   **```num_workers```**: same as in force model
8.   **```max_atoms```**: maximum number of atoms allowed in the input structures
9.   **```atom_vec_length```**: same as in force model
10.   **```stats_has_weighting```**: True if stats file has normalization values for the weighting pattern (i.e: has d+1 values)


### Model Configuration
This class creates the model architecture


1.   **```data```**: do not change! Importing the hyperparameters about the data
2.   **```dimensions_list```**: size and number of layers for the model are specified here


### Model Paths
This class stores the path information for loading and saving results files. The energy predictor checks against the validation set and records its best performing model during runtime. It also always saves the last model weights after it finishes running.


1.   **```pretrainPath```**: same as in force model
2.   **```stats_path```**: same as in force model
3.   **```savedModelName```**: where to save the final model weights. Defaults to ```adapt_nrg_dev{device}_saved.pth```.
4.   **```savedBestName```**: where to save the best performing model against the valaidation set. Defaults to: ```adapt_nrg_best.pth```.


### Train Configuration
This class sets up the training hyperparameters


1.   **```epochs```**: same as in force model
2.   **```alt_loss_fxn```**: loss function used to determine error
3.   **```optimizer```**: same as in force model
4.   **```lr```**: same as in force model
5.   **```exact_reproducibility```**: same as in force model


## Running the Training
Once the config files have been set to your desired setup, all you need to do is run the ```trainEnergy.py``` file. A small dataset has been included for convenience. 

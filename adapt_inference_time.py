"""
# ADAPT Inference Time Usage
Tutorial author: [Evan Dramko](https://evandramko.github.io/)
Last Updated: Jan 29, 2026
Associated Documentation: [website/tutorials](https://evandramko.github.io/ADAPT_webpage/), [paper](https://arxiv.org/abs/2509.24115), [github](https://github.com/EvanDramko/ADAPT_Released), and [data](https://zenodo.org/records/17411327)


## What Does This Tutorial Cover?
This is a tutorial showing how to use a pretrained ADAPT model in deployment or for inference-level tasks. Examples would include running relaxations on a given structure, or determining the formation energy for a proposed structure.
This tutorial assumes that:
1.   ADAPT has been properly downloaded and an appropriate environment has been created. Instructions for this can be found in the "ADAPT Download Guide".
2.   A saved version of the model and normalization statistics has been created. This can be done with the pretrained versions provided in the download, or with the ```trainForce.py``` and ```trainEnergy.py``` code.

## How to Setup the Runtime
The inference time functioning of ADAPT is controlled by the same underlying setup as the training time functioning. To set up the runtime, follow the instructions in the training tutorial on setting up the files in ```ADAPT_released/configs```. Note that some values, like the data paths or target paths for model saving, are not used during inference.

## Using Inference Time
The inference time usage creates an instance of the model by loading in the saved weights and normalization values. It then provides easy wrappers for usage like predicting energies/forces for a structure, or performing a relaxation on a structure.

### Initialization
When creating the initialization, you can specify whether the model is going to be used for either or both of force and energy predictions. This is set with the flags: ```use_energy``` and ```use_force```.

All other initialization values will be filled in automatically from the configuration files.
"""
import warnings
import torch
import inference_time_calcs
import configs

# Reduce noisy warnings during model/stat loading.
warnings.filterwarnings("ignore")

print(">>> example_model = inference_time_calcs.Runner(use_force=True, use_energy=True)")
example_model = inference_time_calcs.Runner(use_force=True, use_energy=True)
print("Initialization complete.\n")

"""
While it is perfectly acceptable to set both force and energy to True even if you only want one, it will increase the memory and runtime burden, as well as perform extra checks on the existence of normalization values, etc for both models.

### Getting Force Predictions
Force predictions (in Angstroms) are found with the ```getOneStepForces``` function. It is setup to take a batch of structures all at once. However, due to pyTorch mechancis, all must be of the same size. Otherwise, you can add padding to the smaller structures in the batch, or just run the prediction multiple times.
Pro Tip: Use ```.unsqueeze(0)``` to add an extra dimension with size 1 to the front of a tensor to make it 3-dimensional.
"""

# ---- Batch of Structures -----
# create a dummy structure batch of 4 samples with 22 atoms
print(">>> x = torch.rand((4, 22, configs.force_model_hyperparam.DataConfig.atom_vec_length))")
x = torch.rand((4, 22, configs.force_model_hyperparam.DataConfig.atom_vec_length))

# predict forces
print(">>> y = example_model.getOneStepForces(x)")
y = example_model.getOneStepForces(x)
print(f"Predicted forces shape is: {y.shape}, should be: (4, 22, 3)")


# ---- Individual Structure -----
# create an individal dummy structure with 22 atoms
print("\n>>> x = torch.rand((22, configs.force_model_hyperparam.DataConfig.atom_vec_length))")
x = torch.rand((22, configs.force_model_hyperparam.DataConfig.atom_vec_length))
print(">>> x = x.unsqueeze(0)")
x = x.unsqueeze(0)
print("Shape of x is: ", x.shape)

# predict forces
print(">>> y = example_model.getOneStepForces(x)")
y = example_model.getOneStepForces(x)
print(f"Predicted forces shape is: {y.shape}, should be: (1, 22, 3)")

"""

---


### Getting Energy Predictions
Energy predictions (in mEv) are found with the ```getEnergy``` function. It is setup to take a batch of structures all at once. Recall that for the energy predictor, all structures should be padded to the maximum number of atoms before being passed into the model."""

print("\n>>> x = torch.rand((2, configs.energy_model_hyperparams.DataConfig.max_atoms, configs.energy_model_hyperparams.DataConfig.atom_vec_length))")
x = torch.rand((2, configs.energy_model_hyperparams.DataConfig.max_atoms, configs.energy_model_hyperparams.DataConfig.atom_vec_length))
print(">>> y = example_model.getStructEnergy(x)")
y = example_model.getStructEnergy(x)
print(f"Predicted energy shape is: {y.shape}, should be: (2, 1)")

"""
### Common Usage Pitfalls
After instantiating the model, try to use it as much as possible before letting it disappear. Instantiation takes time to do, so we want to hold the instance in memory between successive uses of it in the same workflow."""

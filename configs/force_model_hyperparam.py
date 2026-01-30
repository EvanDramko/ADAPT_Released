from dataclasses import dataclass
from pathlib import Path   
from typing import Callable, Optional, Type
import torch


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    d_ff: int = 512
    num_layers: int = 8
    d_out: int = 3
    dropout_rate: float = 0.05
    num_heads: int = 8


@dataclass(frozen=True)
class DataConfig:
    train_path: str = "./data/mini_force_dataset/trainset.pt"
    test_path: str = "./data/mini_force_dataset/trainset.pt" # change to a different set in practice!
    has_weighting: bool = True # true if data[..., -1] is a weighting/importance factor for each atom. False otherwise
    has_Z: bool = False # check as True if element 4 (index 3) is the atomic number. Otherwise, it is assumed indices 3, 4 are column and row of periodic table
    batch_size: int = 64
    num_workers: int = 4 # used in the dataloader... only change if necessary. 
    atom_vec_length: int = 12 # how many features (including three coordinates) are used to describe each atom. (Ex: |(x, y, z, Z)| = 4)


@dataclass(frozen=True)
class ModelPaths:
    pretrainPath: Optional[Path] = Path("./saved_models/force_baseline.pth") # load is saved pretrained model
    stats_path: Optional[Path] = Path("./utils/norm_stats.pt")
    saveBase: str = "./"
    savedModelName: Optional[str] = None # where to save the weights of the new model (should end in .pt or pth)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 80
    alt_loss_fxn: Optional[Callable[[float, float], float]] = None # defaults to weighted mse or regular mse based on has_weighting
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    lr: float = 5e-5 # set learning rate
    exact_reproducibility: bool = False # makes training deterministic at performance cost
    augmentation: bool = False # set to True if you want training to be done with data augmentation (rotation and translation)
    
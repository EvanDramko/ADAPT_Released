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
    train_path: str = "./data/training_data.xyz"
    test_path: str = None
    has_Z: bool = True # check as True if element 4 (index 3) is the atomic number. Otherwise, it is assumed indices 3, 4 are column and row of periodic table
    batch_size: int = 32
    num_workers: int = 4 # used in the dataloader... only change if necessary. 
    atom_vec_length: int = 4 # how many features (including three coordinates) are used to describe each atom. (Ex: |(x, y, z, Z)| = 4)
    isCrystal: bool = False


@dataclass(frozen=True)
class ModelPaths:
    pretrainPath: Optional[Path] = None # Path("./saved_models/force_baseline.pth") # load in saved pretrained model
    stats_path: Path = Path("./utils/norm_stats.pt")
    saveBase: str = "./"
    savedModelName: Optional[str] = None # where to save the weights of the new model (should end in .pt or pth)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 80
    recompute_stats: bool = False
    alt_loss_fxn: Optional[Callable[[float, float], float]] = None # defaults to regular mse
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    lr: float = 5e-5 # set learning rate
    exact_reproducibility: bool = False # makes training deterministic at performance cost
    augmentation: bool = False # set to True if you want training to be done with data augmentation (rotation and translation)
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    

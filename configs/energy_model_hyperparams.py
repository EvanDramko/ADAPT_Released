from pathlib import Path   
from typing import Callable, Optional, Type, List
import torch
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    train_path_X: Path = Path("./data/mini_energy_dataset/x_values.pt")
    train_path_NRG: Path = Path("./data/mini_energy_dataset/nrg_values.pt")
    test_path_X: Path = Path("./data/mini_energy_dataset/x_values.pt")
    test_path_NRG: Path = Path("./data/mini_energy_dataset/nrg_values.pt")
    has_Z: bool = False
    batch_size: int = 64
    num_workers: int = 4
    max_atoms: int = 220
    atom_vec_length: int = 12
    stats_has_weighting: bool = True


@dataclass(frozen=True)
class ModelConfig:
    data: DataConfig

    @property
    def dimensions_list(self) -> List[int]: 
        return [self.data.max_atoms * self.data.atom_vec_length, 4096, 4096, 4096, 4096, 128, 1]


@dataclass(frozen=True)
class ModelPaths:
    pretrainPath: Optional[Path] = Path("./saved_models/energy_baseline.pth") # load a saved pretrained model. Use None keyword if training from scratch
    stats_path: Path = Path("./utils/norm_stats.pt") # needed for inference time, but training time can make the file if it is not available
    savedModelName: Optional[str] = None # alternative model name or path to save the weights of the new model (should end in .pt or pth)
    savedBestName: Optional[str] = None # alternative name/path to save the version that performed best on the validation/test set


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    alt_loss_fxn: Callable[[float, float], float] = torch.nn.MSELoss()
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    lr: float = 5e-5 # set learning rate
    exact_reproducibility: bool = False # makes training deterministic at performance cost. NOT recommended unless strictly necessary
    
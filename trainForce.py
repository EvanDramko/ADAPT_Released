# imports
import os
import warnings
from typing import Optional
import torch
from torch.utils.data import DataLoader
from MLFF_arch.encoderBase import TransformerEncoder as forceMLFF
from MLFF_arch.encoderBase import train, test
from utils import normalizer, dataRotater, dataTranslater
from data.ragged_dataset import RaggedAtomDataset, collate_pad
from data.workflow_utils import prepare_ragged_dataset_path
from configs import force_model_hyperparam as model_hyperparam
import numpy as np

def run_training(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    is_crystal: Optional[bool] = None,
    epochs: Optional[int] = None,
    recompute_stats: bool = False,
    device: Optional[str] = None,
    augmentation: Optional[bool] = None,
) -> None:
    # Set device
    device = device if device is not None else model_hyperparam.TrainConfig.device
    print(f"Using {device} device \n")

    # ========== DETERMINISTIC TRAINING SETTING ==========
    if model_hyperparam.TrainConfig.exact_reproducibility:
        warnings.warn("Exact reproducability will hurt model runtime performance")
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    def seed_worker(worker_id):
        import numpy as np
        import random
        worker_seed = torch.initial_seed() % 2**32 # 2**32 controls integration between torch and numpy
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    if model_hyperparam.TrainConfig.exact_reproducibility:
        g.manual_seed(42)
    # ====================================================

    # Instantiate the model with the values in the config file
    config = model_hyperparam.ModelConfig() 
    MLFF = forceMLFF(
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        d_out=config.d_out,
        dropout_rate=config.dropout_rate,
        num_heads=config.num_heads,
        vecRepLength=model_hyperparam.DataConfig.atom_vec_length
    ).to(device)

    print("Network structure is: ")
    print(MLFF)
    print()
    # Check if the pretrainPath exists
    if (model_hyperparam.ModelPaths.pretrainPath != None):
        MLFF.load_state_dict(torch.load(model_hyperparam.ModelPaths.pretrainPath, map_location=torch.device(device), weights_only=True))
        print(f"Successfully loaded pretrained model from: {model_hyperparam.ModelPaths.pretrainPath}")
    else:
        print("No pretrained path provided. Starting with a randomly initialized model.")


    # set hyperparameters:
    batch_size = model_hyperparam.DataConfig.batch_size
    train_path_in = train_path if train_path is not None else model_hyperparam.DataConfig.train_path
    test_path_in = test_path
    is_crystal_in = is_crystal if is_crystal is not None else model_hyperparam.DataConfig.isCrystal
    #=========================================================================================================================================
    #============================================================= Data Handling =============================================================
    #=========================================================================================================================================
    train_path = prepare_ragged_dataset_path(train_path_in, is_crystal_in)
    test_path = prepare_ragged_dataset_path(test_path_in, is_crystal_in) if test_path_in is not None else None

    # construct datasets
    train_ds = RaggedAtomDataset(from_path=train_path, is_crystal=is_crystal_in)
    train_x_dim = int(train_ds.X_list[0].shape[-1])

    if train_x_dim != model_hyperparam.DataConfig.atom_vec_length:
        raise ValueError(
            f"Train dataset feature mismatch: x_dim={train_x_dim}, "
            f"expected atom_vec_length={model_hyperparam.DataConfig.atom_vec_length}."
        )

    test_ds = None
    test_x_dim = None
    if test_path is not None:
        test_ds = RaggedAtomDataset(from_path=test_path, is_crystal=is_crystal_in)
        test_x_dim = int(test_ds.X_list[0].shape[-1])
        if test_x_dim != model_hyperparam.DataConfig.atom_vec_length:
            raise ValueError(
                f"Test dataset feature mismatch: x_dim={test_x_dim}, "
                f"expected atom_vec_length={model_hyperparam.DataConfig.atom_vec_length}."
            )
        if train_x_dim != test_x_dim:
            raise ValueError(
                f"Train/test feature mismatch: train has x_dim={train_x_dim}, test has x_dim={test_x_dim}. "
                f"Use datasets with the same Properties layout (same atom vector width). "
                f"Example: data/training_data.xyz has 12 features, while data/test_100_set.xyz has 3."
            )
        print(f"Data layout: train_x_dim={train_x_dim}, test_x_dim={test_x_dim}")
    else:
        print(f"Data layout: train_x_dim={train_x_dim}, no test dataset")

    loss_fn = torch.nn.MSELoss()
    if model_hyperparam.TrainConfig.alt_loss_fxn:
        loss_fn = model_hyperparam.TrainConfig.alt_loss_fxn

    # load or extend normalization stats
    stats_path = model_hyperparam.ModelPaths.stats_path
    required_keys = {"x_mean", "x_std", "y_mean_force", "y_std_force"}

    if recompute_stats:
        warnings.warn(
            f"Recomputing normalization stats from scratch on the training split and overwriting {stats_path}."
        )
        stats = normalizer.fit_stats_from_ragged(train_ds.X_list, train_ds.Y_list)
        torch.save(stats, stats_path)
    elif os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location="cpu")
        if not isinstance(stats, dict):
            raise TypeError(f"Stats file at {stats_path} is not a python dict")

        missing_keys = required_keys - stats.keys()
        shape_mismatch = (
            ("x_mean" in stats and int(stats["x_mean"].shape[-1]) != train_x_dim) or
            ("x_std" in stats and int(stats["x_std"].shape[-1]) != train_x_dim) or
            ("y_mean_force" in stats and int(stats["y_mean_force"].shape[-1]) != 3) or
            ("y_std_force" in stats and int(stats["y_std_force"].shape[-1]) != 3)
        )
        if missing_keys:
            warnings.warn(
                f"Stats file at {stats_path} is missing keys {missing_keys}. "
                f"Computing and adding them."
            )

            # compute full stats ONCE
            new_stats = normalizer.fit_stats_from_ragged(train_ds.X_list, train_ds.Y_list)
            # only add the missing keys
            for k in missing_keys:
                stats[k] = new_stats[k]
            torch.save(stats, stats_path)
        elif shape_mismatch:
            warnings.warn(
                f"Stats file at {stats_path} has incompatible shapes for current data "
                f"(expected x_dim={train_x_dim}, y_dim=3). Recomputing stats from training split."
            )
            stats = normalizer.fit_stats_from_ragged(train_ds.X_list, train_ds.Y_list)
            torch.save(stats, stats_path)
        else:
            print(f"Loaded normalization stats from {stats_path}")

    else:
        stats = normalizer.fit_stats_from_ragged(train_ds.X_list, train_ds.Y_list)
        torch.save(stats, stats_path)
        warnings.warn(f"Stats file not found at {stats_path}. Created new stats file from training data.")



    # normalize both splits using the same stats
    train_X_norm, train_Y_norm = normalizer.normalize_ragged_with_stats(
        train_ds.X_list, train_ds.Y_list,
        stats["x_mean"], stats["x_std"], stats["y_mean_force"], stats["y_std_force"],
    )
    # wrap normalized lists in fresh Dataset objects
    train_ds = RaggedAtomDataset(train_X_norm, train_Y_norm)
    if test_ds is not None:
        test_X_norm, test_Y_norm = normalizer.normalize_ragged_with_stats(
            test_ds.X_list, test_ds.Y_list,
            stats["x_mean"], stats["x_std"], stats["y_mean_force"], stats["y_std_force"],
        )
        test_ds = RaggedAtomDataset(test_X_norm, test_Y_norm)

    # create dataloaders
    test_dataloader = None
    if test_ds is not None:
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    train_loader_kwargs = dict(
    dataset=train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=model_hyperparam.DataConfig.num_workers,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_pad,
    worker_init_fn=seed_worker if model_hyperparam.TrainConfig.exact_reproducibility else None,
    generator=g if model_hyperparam.TrainConfig.exact_reproducibility else None
)
    if model_hyperparam.DataConfig.num_workers > 0:
        train_loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    train_dataloader = DataLoader(**train_loader_kwargs)
    #=========================================================================================================================================
    #=========================================================== End Data Handling ===========================================================
    #=========================================================================================================================================
    # Set training and optimization parameters
    epochs_count = epochs if epochs is not None else model_hyperparam.TrainConfig.epochs
    optimizer = model_hyperparam.TrainConfig.optimizer(MLFF.parameters(), lr=model_hyperparam.TrainConfig.lr)
    useAug = model_hyperparam.TrainConfig.augmentation
    if augmentation is not None:
        useAug = augmentation
    
    train_scores = []
    test_scores = []

    # ------ run the model training ------
    for t in range(epochs_count):
        print(f"------------------------ Epoch {t+1}/{epochs_count} ------------------------")
        if useAug == True:
            train_score = train(train_dataloader, MLFF, loss_fn, optimizer, rotation=dataRotater.randomRotate, translation=dataTranslater.randomTranslate)
            if test_dataloader is not None:
                test_score = test(test_dataloader, MLFF, loss_fn, rotation=dataRotater.randomRotate, translation=dataTranslater.randomTranslate)
        else:
            train_score = train(train_dataloader, MLFF, loss_fn, optimizer, rotation=None, translation=None)
            if test_dataloader is not None:
                test_score = test(test_dataloader, MLFF, loss_fn, rotation=None, translation=None)

        # store scores
        train_scores.append(train_score)

        if test_dataloader is not None:
            test_scores.append(test_score)
        
        if ((t+1) % 20 == 0):
            print("Saving the model...")
            saveString = model_hyperparam.ModelPaths.savedModelName or f"MLFF_dev{device}_saved.pth"
            torch.save(MLFF.state_dict(), saveString)
            print("Model saved!")

    
    np.savez(
        "training_history.npz",
        train_scores=train_scores,
        test_scores=test_scores
    )
    print("Done With Training!")

    print("Saving the model...")
    saveString = model_hyperparam.ModelPaths.savedModelName or f"MLFF_dev{device}_saved.pth"
    torch.save(MLFF.state_dict(), saveString)
    print("Model saved -- All Done!")

if __name__ == "__main__":
    run_training()

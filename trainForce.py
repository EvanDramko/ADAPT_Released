# imports
import os
import warnings
import torch
from torch.utils.data import DataLoader
from MLFF_arch.encoderBase import TransformerEncoder as forceMLFF
from MLFF_arch.encoderBase import train, test
from utils import evals, normalizer, dataRotater, dataTranslater
from data.ragged_dataset import RaggedAtomDataset, collate_pad
from configs import force_model_hyperparam as model_hyperparam

if __name__ == "__main__":
    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
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
    loss_fn = evals.weightedMSE if model_hyperparam.DataConfig.has_weighting else torch.nn.MSELoss()
    if model_hyperparam.TrainConfig.alt_loss_fxn:
        loss_fn = model_hyperparam.TrainConfig.alt_loss_fxn

    #=========================================================================================================================================
    #============================================================= Data Handling =============================================================
    #=========================================================================================================================================
    # construct datasets
    train_ds = RaggedAtomDataset(from_path=model_hyperparam.DataConfig.train_path)
    test_ds  = RaggedAtomDataset(from_path=model_hyperparam.DataConfig.test_path)

    # load or extend normalization stats
    stats_path = model_hyperparam.ModelPaths.stats_path
    required_keys = {"x_mean", "x_std", "y_mean_force", "y_std_force"}

    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location="cpu")
        if not isinstance(stats, dict):
            raise TypeError(f"Stats file at {stats_path} is not a python dict")

        missing_keys = required_keys - stats.keys()
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
    test_X_norm, test_Y_norm = normalizer.normalize_ragged_with_stats(
        test_ds.X_list, test_ds.Y_list,
        stats["x_mean"], stats["x_std"], stats["y_mean_force"], stats["y_std_force"],
    )

    # wrap normalized lists in fresh Dataset objects
    train_ds = RaggedAtomDataset(train_X_norm, train_Y_norm)
    test_ds  = RaggedAtomDataset(test_X_norm, test_Y_norm)

    # create dataloaders
    test_dataloader  = DataLoader(test_ds,  batch_size=64, shuffle=False, collate_fn=collate_pad)
    train_dataloader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=model_hyperparam.DataConfig.num_workers,  # tune 4–16 depending on CPU & transforms
    pin_memory=True,  # enables fast, async H2D
    persistent_workers=True, # avoid per-epoch respawn cost
    prefetch_factor=2,  # tune 2–4; watch RAM
    drop_last=False,  # worth it to use a non-full batch in limited data setups (i.e <1M data points)
    collate_fn=collate_pad,
    worker_init_fn=seed_worker if model_hyperparam.TrainConfig.exact_reproducibility else None,
    generator=g if model_hyperparam.TrainConfig.exact_reproducibility else None
)
    #=========================================================================================================================================
    #=========================================================== End Data Handling ===========================================================
    #=========================================================================================================================================
    # Set training and optimization parameters
    epochs = model_hyperparam.TrainConfig.epochs
    optimizer = model_hyperparam.TrainConfig.optimizer(MLFF.parameters(), lr=model_hyperparam.TrainConfig.lr)

    # ------ run the model training ------
    # run training for given epochs
    for t in range(epochs):
        print(f"------------------------ Epoch {t+1}/{epochs} ------------------------")
        if model_hyperparam.TrainConfig.augmentation == True:
            train(train_dataloader, MLFF, loss_fn, optimizer, rotation=dataRotater.randomRotate, translation=dataTranslater.randomTranslate, include_weighting_pattern=model_hyperparam.DataConfig.has_weighting)
            test(test_dataloader, MLFF, loss_fn, rotation=dataRotater.randomRotate, translation=dataTranslater.randomTranslate, include_weighting_pattern=model_hyperparam.DataConfig.has_weighting)
        else:
            train(train_dataloader, MLFF, loss_fn, optimizer, rotation=None, translation=None, include_weighting_pattern=model_hyperparam.DataConfig.has_weighting)
            test(test_dataloader, MLFF, loss_fn, rotation=None, translation=None, include_weighting_pattern=model_hyperparam.DataConfig.has_weighting)
        
        if ((t+1) % 20 == 0):
            print("Saving the model...")
            saveString = model_hyperparam.ModelPaths.savedModelName or f"MLFF_dev{device}_saved.pth"
            torch.save(MLFF.state_dict(), saveString)
            print("Model saved!")
    print("Done With Training!")

    print("Saving the model...")
    saveString = model_hyperparam.ModelPaths.savedModelName or f"MLFF_dev{device}_saved.pth"
    torch.save(MLFF.state_dict(), saveString)
    print("Model saved -- All Done!")

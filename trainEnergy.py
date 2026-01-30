# imports
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.padder import padTensor
import utils.normalizer
from configs import energy_model_hyperparams
from MLFF_arch import energyMLFF
import warnings


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
    if energy_model_hyperparams.TrainConfig.exact_reproducibility:
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
    if energy_model_hyperparams.TrainConfig.exact_reproducibility:
        g.manual_seed(42)
    # ====================================================

    # Create an instance of the energy residual mlp
    MLFF = energyMLFF.nrgMLP(energy_model_hyperparams.ModelConfig(energy_model_hyperparams.DataConfig()).dimensions_list).to(device)
    print("Network structure is: ")
    print(MLFF)
    print()
    if(energy_model_hyperparams.ModelPaths.pretrainPath != None):
        MLFF.load_state_dict(torch.load(energy_model_hyperparams.ModelPaths.pretrainPath, map_location=device))

    # set hyperparameters:
    batch_size = energy_model_hyperparams.DataConfig.batch_size
    loss_fn = energy_model_hyperparams.TrainConfig.alt_loss_fxn

    #=========================================================================================================================================
    #============================================================= Data Handling =============================================================
    #=========================================================================================================================================
    # create Dataset objects
    class TrainingDataset(Dataset):
        def __init__(self, inputData, labels):
            self.X = inputData
            self.Y = labels

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    class TestDataset(Dataset):
        def __init__(self, inputData, labels):
            self.X = inputData
            self.Y = labels

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    

    X_train = torch.load(energy_model_hyperparams.DataConfig.train_path_X, map_location=device).clone().detach().cpu()
    energyTrain = torch.load(energy_model_hyperparams.DataConfig.train_path_NRG, map_location=device).clone().detach()
    X_test = torch.load(energy_model_hyperparams.DataConfig.test_path_X, map_location=device).clone().detach().cpu()
    energyTest = torch.load(energy_model_hyperparams.DataConfig.test_path_NRG, map_location=device).clone().detach()
    print("Structure training set shape is: ", X_train.shape)
    print("Energy output data shape is:", energyTrain.shape)

    print("Structure testing set shape is: ", X_test.shape)
    print("Energy output data shape is:", energyTest.shape)

    if(X_train.shape[-2] > energy_model_hyperparams.DataConfig.max_atoms) or (X_train.shape[-1] != energy_model_hyperparams.DataConfig.atom_vec_length):
        raise Exception("Input dimensions of the training set do not match with data configuration parameters.")
    elif(X_test.shape[-2] > energy_model_hyperparams.DataConfig.max_atoms) or (X_test.shape[-1] != energy_model_hyperparams.DataConfig.atom_vec_length):
        raise Exception("Input dimensions of the testing set do not match with data configuration parameters.")
    elif(energyTrain.ndim > 1) or (energyTest.ndim > 1):
        raise Exception("Energy values should be a single value for the whole structure.")

    def _makeStats(X_train, energyTrain):
        meanX = X_train.mean(dim=(0, 1), keepdim=True)
        stdX  = X_train.std(dim=(0, 1), keepdim=True)
        meanY = energyTrain.mean(dim=0, keepdim=True)
        stdY  = energyTrain.std(dim=0, keepdim=True)
        return {
            "x_mean": meanX.detach().cpu(),
            "x_std": stdX.detach().cpu(),
            "y_mean_energy": meanY.detach().cpu(),
            "y_std_energy": stdY.detach().cpu(),
        }

    # load or extend normalization stats (without dropping existing keys)
    stats_path = energy_model_hyperparams.ModelPaths.stats_path
    required_keys = {"x_mean", "x_std", "y_mean_energy", "y_std_energy"}

    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location="cpu")
        if not isinstance(stats, dict):
            raise TypeError(f"Stats file at {stats_path} is not a dict (got {type(stats)})")

        # treat missing OR invalid entries as needing refresh
        missing = required_keys - stats.keys()

        if missing:
            warnings.warn(
                f"Stats file at {stats_path} is missing/invalid keys {missing}. "
                f"Computing and adding them (preserving existing keys)."
            )
            new_stats = _makeStats(X_train, energyTrain)
            for k in missing:
                stats[k] = new_stats[k]
            torch.save(stats, stats_path)
        else:
            print(f"Loaded normalization stats from {stats_path}")
    else:
        stats = _makeStats(X_train, energyTrain)
        torch.save(stats, stats_path)
        warnings.warn(f"Stats file not found at {stats_path}. Created new stats file from training data.")

    if energy_model_hyperparams.DataConfig.stats_has_weighting:
        X_train = utils.normalizer.normalize_all(0, X_train, mean=stats["x_mean"][..., :-1], std=stats["x_std"][..., :-1])
        X_test  = utils.normalizer.normalize_all(0, X_test, mean=stats["x_mean"][..., :-1], std=stats["x_std"][..., :-1])
    else:
        X_train = utils.normalizer.normalize_all(0, X_train, mean=stats["x_mean"], std=stats["x_std"])
        X_test  = utils.normalizer.normalize_all(0, X_test, mean=stats["x_mean"], std=stats["x_std"])       
    y_train  = utils.normalizer.normalize_all(1, energyTrain, mean=stats["y_mean_energy"], std=stats["y_std_energy"])
    y_test  = utils.normalizer.normalize_all(1, energyTest, mean=stats["y_mean_energy"], std=stats["y_std_energy"])


    # add padding to get to size (max_atoms, vec_rep_length) for each of the four data component and create Dataset objects
    dataInfo = energy_model_hyperparams.DataConfig
    training_data = TrainingDataset(padTensor(X_train, d=dataInfo.max_atoms, w=dataInfo.atom_vec_length), y_train)
    test_data = TestDataset(padTensor(X_test, d=dataInfo.max_atoms, w=dataInfo.atom_vec_length), y_test)

    # create data loader objects
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    #=========================================================================================================================================
    #=========================================================== End Data Handling ===========================================================
    #=========================================================================================================================================

    # Set training and optimization parameters
    epochs = energy_model_hyperparams.TrainConfig.epochs
    steps_per_epoch = len(train_dataloader)
    optimizer = energy_model_hyperparams.TrainConfig.optimizer(MLFF.parameters(), lr=energy_model_hyperparams.TrainConfig.lr)

    # ------ run the model training ------
    # create storage mediums for training and test loss
    testLossHistory = []
    trainLossHistory = []
    best_val = float("inf")
    best_path = energy_model_hyperparams.ModelPaths.savedBestName or "adapt_nrg_best.pth"
    # run training for given epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        trainLossHistory.append(energyMLFF.train(train_dataloader, MLFF, loss_fn, optimizer))
        evalRes = energyMLFF.test(test_dataloader, MLFF, loss_fn)
        testLossHistory.append(evalRes)
        if evalRes < best_val:
            best_val = evalRes
            torch.save(MLFF.state_dict(), best_path)
            print(f"New best val: {best_val:.4f} — saved to {best_path}")
    print("Done!")

    print("Saving the model...")
    saveString = energy_model_hyperparams.ModelPaths.savedModelName or f"adapt_nrg_dev{device}_saved.pth"
    torch.save(MLFF.state_dict(), saveString)
    print("Model saved -- All Done!")

    print("train loss history: ", trainLossHistory)
    print()
    print("test loss history: ", testLossHistory)
    print()

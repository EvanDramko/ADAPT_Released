# imports
from typing import Optional
import torch
import utils.normalizer
from configs import force_model_hyperparam
from MLFF_arch import encoderBase
import warnings
from data.xyz_extxyz_converter import load_ragged_from_xyz_extxyz, load_one_frame_from_xyz_extxyz

def _require(cond: bool, msg: str, exc=ValueError):
    if not cond:
        raise exc(msg)

class Runner:
    def __init__(self, use_force = True, device: Optional[str] = None):
        resolved_device = device if device is not None else force_model_hyperparam.TrainConfig.device
        self.device = torch.device(resolved_device)

        # load models
        if use_force:
            config = force_model_hyperparam.ModelConfig() 
            _require(force_model_hyperparam.ModelPaths.pretrainPath != None, "Must have a saved model to use at inference time for the force predictor")
            self.force_model = encoderBase.TransformerEncoder(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_layers=config.num_layers,
                d_out=config.d_out,
                dropout_rate=config.dropout_rate,
                num_heads=config.num_heads,
                vecRepLength=force_model_hyperparam.DataConfig.atom_vec_length,
            ).to(self.device)
            self.force_model.load_state_dict(torch.load(force_model_hyperparam.ModelPaths.pretrainPath, map_location=torch.device(self.device)))
            self.force_model.eval()
        else:
            raise NotImplementedError("Code Release is only setup for force predictions at this time! Energy coming soon!")

        # load data shape information
        self.max_atoms = None
        self.atom_feature_length = force_model_hyperparam.DataConfig.atom_vec_length
        
        # load normalizaton stats
        stats = torch.load(force_model_hyperparam.ModelPaths.stats_path, map_location=self.device)
        self.meanX = stats["x_mean"]
        self.stdX  = stats["x_std"]
        self.meanY_force = stats["y_mean_force"]
        self.stdY_force = stats["y_std_force"]


    @torch.inference_mode
    def getOneStepForces(self, rawStructure):
        """
        Predicts the per atom forces of the given structure(s)

        Args:
            atomStruct (torch.Tensor): atomic structure (crystal/molecule) as torch tensor: (B,n,v)

        Return:
            torch.Tensor: forces as (B,n,3) tensor
        
        """
        _require(isinstance(rawStructure, torch.Tensor), "rawStructure should be a torch tensor")
        _require(rawStructure.ndim == 3, f"You must have three dimensional input (B batch, n atoms, v feature descriptors). You have {rawStructure.ndim} input dimensions.")
        _require(self.device == next(self.force_model.parameters()).device, f"model is not on the same device ({next(self.force_model.parameters()).device}) as the Runner class ({self.device})", exc=RuntimeError)
        if(self.device == "cpu"): # checks if Runner object is on cpu or gpu
            warnings.warn("CPU runtimes may be lengthy. Typically, gpu accelerated runtimes are recommended.")

        # Move to device and ensure model-input feature width.
        structure = rawStructure.to(torch.float32).to(self.device)
        input_dim = int(structure.shape[-1])

        _require(
            input_dim == self.atom_feature_length,
            f"The atomic description vector must have atomFeatureLength={self.atom_feature_length} components "
            f"Got {rawStructure.shape[-1]}."
        )

        # Normalize with stats of matching dimensionality.
        stats_dim = int(self.meanX.shape[-1])
        if input_dim != stats_dim:
            raise ValueError(
                f"Incompatible feature dims: input={input_dim}, stats={stats_dim}"
            )
        normedStruct = utils.normalizer.normalize_all(0, structure, self.meanX, self.stdX)

        # get force prediction and un-normalize
        forcesPred = self.force_model(normedStruct)
        unnormedForces = utils.normalizer.unnormalize_all(1, forcesPred, self.meanY_force, self.stdY_force)

        return unnormedForces


    @torch.inference_mode
    def getOneStepForcesFromXYZ(self, xyz_path: str, frame_idx: int = 0, is_crystal = None):
        """
        Predict force tensor (n, 3) for one structure frame read from xyz/extxyz.
        """
        if is_crystal is None:
            is_crystal = force_model_hyperparam.DataConfig.isCrystal

        X, _, _ = load_one_frame_from_xyz_extxyz(
            file_path=xyz_path,
            frame_idx=frame_idx,
            is_crystal=bool(is_crystal),
            dtype=torch.float32,
        )
        frame = X.unsqueeze(0)  # (1, n, v)
        return self.getOneStepForces(frame).squeeze(0)


@torch.inference_mode
def run_evaluation(
    xyz_path: str,
    frame_idx: int = 0,
    all_frames: bool = False,
    is_crystal: bool | None = None,
    device: str | None = None,
):
    """
    Evaluate saved force model on xyz/extxyz data that includes REF_forces:R:3 labels.
    Prints MSE/MAE summary and returns metrics dict.

    Args:
        xyz_path (str): path to the xyz file
        frame_idx (int): specific frame you want evaluated (irrelevant if all_frames=True)
        all_frames (bool): whether to evaluate all frames or not
        is_crystal (bool): are the structures crystals or molecules
        device (str): device on which to run the evaluation (defaults to cuda -> mps -> cpu)
    """
    if is_crystal is None:
        is_crystal = force_model_hyperparam.DataConfig.isCrystal

    if all_frames:
        X_list, Y_list, _ = load_ragged_from_xyz_extxyz(
            file_path=xyz_path,
            is_crystal=bool(is_crystal),
            dtype=torch.float32,
        )
        _require(len(X_list) > 0, f"No frames found in {xyz_path}")
        runner = Runner(use_force=True, device=device)
        frame_ids = range(len(X_list))
    else:
        X, Y, _ = load_one_frame_from_xyz_extxyz(
            file_path=xyz_path,
            frame_idx=frame_idx,
            is_crystal=bool(is_crystal),
            dtype=torch.float32,
        )
        runner = Runner(use_force=True, device=device)
        X_list, Y_list = [X], [Y]
        frame_ids = [0]

    total_sq = 0.0
    total_abs = 0.0
    total_count = 0

    for i in frame_ids:
        pred = runner.getOneStepForces(X_list[i].unsqueeze(0)).squeeze(0).cpu()
        target = Y_list[i].cpu()
        diff = pred - target
        total_sq += float((diff ** 2).sum().item())
        total_abs += float(diff.abs().sum().item())
        total_count += int(diff.numel())

    mse = total_sq / max(total_count, 1)
    mae = total_abs / max(total_count, 1)
    n_frames = len(frame_ids) if all_frames else 1

    print(f"Evaluation complete on {n_frames} frame(s) from {xyz_path}")
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")

    return {
        "mse": mse,
        "mae": mae,
        "n_frames": n_frames,
        "path": xyz_path,
    }

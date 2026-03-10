from pathlib import Path

import torch
import inference_time_calcs
from configs import force_model_hyperparam


REPO_ROOT = Path(__file__).resolve().parent
TINY_TEST_PATH = REPO_ROOT / "data" / "tiny_omol_data" / "test_sample.xyz"
FALLBACK_MODEL_PATH = REPO_ROOT / "saved_models" / "iro.pth"


def test_torch_backend():
    assert torch.__version__ is not None

    x = torch.randn(4, 4)
    y = x @ x
    assert torch.isfinite(y).all()

    if torch.cuda.is_available():
        z = x.cuda() @ x.cuda()
        assert torch.isfinite(z).all()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        z = x.to("mps") @ x.to("mps")
        assert torch.isfinite(z).all()


def test_torch_seeding():
    torch.manual_seed(0)

    m1 = torch.nn.Linear(8, 4)
    torch.manual_seed(0)
    m2 = torch.nn.Linear(8, 4)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)

    x = torch.ones(2, 8)
    y1 = m1(x)
    y2 = m2(x)

    assert torch.allclose(y1, y2)


def test_parameter_init(model):
    for name, p in model.named_parameters():
        if p.numel() == 0:
            raise Exception(f"There are 0 trainable parameters in {name}")

        assert torch.isfinite(p).all(), f"{name} has NaNs/Infs"

        if p.numel() > 1:
            std = p.float().std().item()
            mean = p.float().mean().item()

        assert std > 0, f"{name} has {std} variance, parameters are: \n {p}"
        assert abs(mean) < 5.0, f"{name} mean suspiciously large"


def resolve_model_path() -> str:
    configured_model = force_model_hyperparam.ModelPaths.pretrainPath
    if configured_model is not None:
        return str(configured_model)
    if FALLBACK_MODEL_PATH.exists():
        return str(FALLBACK_MODEL_PATH)
    raise FileNotFoundError(
        "Could not find a model checkpoint. Set ModelPaths.pretrainPath or add saved_models/iro.pth."
    )


def resolve_tiny_dataset() -> str:
    if not TINY_TEST_PATH.exists():
        raise FileNotFoundError(f"Could not find tiny test dataset at {TINY_TEST_PATH}")
    return str(TINY_TEST_PATH)


if __name__ == "__main__":
    model_path = resolve_model_path()
    xyz_path = resolve_tiny_dataset()

    # initialize the model
    test_run = inference_time_calcs.Runner(
        device=force_model_hyperparam.TrainConfig.device,
        model_path=model_path,
    )

    # run tensor and xyz/extxyz inference paths once
    x = torch.rand((2, 20, force_model_hyperparam.DataConfig.atom_vec_length))
    z = test_run.getOneStepForces(x)
    print(f"Predicted forces shape (tensor API): {z.shape}, should be: [2, 20, 3]")

    z_xyz = test_run.getOneStepForcesFromXYZ(
        xyz_path=xyz_path,
        frame_idx=0,
        is_crystal=force_model_hyperparam.DataConfig.isCrystal,
    )
    print(f"Predicted forces shape (xyz API, frame 0): {z_xyz.shape}, should be: [n, 3]")

    eval_metrics = inference_time_calcs.run_evaluation(
        xyz_path=xyz_path,
        frame_idx=None,
        all_frames=None,
        is_crystal=force_model_hyperparam.DataConfig.isCrystal,
        device=force_model_hyperparam.TrainConfig.device,
        model_path=model_path,
    )
    print(
        f"Eval smoke test on tiny dataset: frames={eval_metrics['n_frames']} "
        f"MAE={eval_metrics['mae']:.8f} MSE={eval_metrics['mse']:.8f}"
    )

    # run pre-made checks
    test_torch_backend()
    test_torch_seeding()
    print("Testing force model parameter intitialization...")
    test_parameter_init(test_run.force_model)

    print("Finished all the checks! You are good to go!")

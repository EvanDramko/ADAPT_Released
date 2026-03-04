import torch
import inference_time_calcs
from configs import force_model_hyperparam

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


if __name__ == "__main__":
    # initialize the model
    test_run = inference_time_calcs.Runner(device=force_model_hyperparam.TrainConfig.device)

    # run tensor and xyz/extxyz inference paths once
    x = torch.rand((2, 20, force_model_hyperparam.DataConfig.atom_vec_length))
    z = test_run.getOneStepForces(x)
    print(f"Predicted forces shape (tensor API): {z.shape}, should be: [2, 20, 3]")

    xyz_path = force_model_hyperparam.DataConfig.test_path
    if xyz_path.lower().endswith((".xyz", ".extxyz")):
        try:
            z_xyz = test_run.getOneStepForcesFromXYZ(
                xyz_path=xyz_path,
                frame_idx=0,
                is_crystal=force_model_hyperparam.DataConfig.isCrystal,
            )
            print(f"Predicted forces shape (xyz API, frame 0): {z_xyz.shape}, should be: [n, 3]")
        except ValueError as exc:
            print(f"Skipping xyz inference check for {xyz_path}: {exc}")
    else:
        print("The test path did not have an xyz or extxyz file in it!")

    # run pre-made checks
    test_torch_backend()
    test_torch_seeding()
    print("Testing force model parameter intitialization...")
    test_parameter_init(test_run.force_model)

    print("Finished all the checks! You are good to go!")

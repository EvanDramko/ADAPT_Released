import torch

ckpt = torch.load("./energy_baseline.pth", map_location="cpu")

# Common patterns: ckpt might be a dict with keys like 'state_dict', 'model', etc.
for key in ["state_dict", "model_state_dict", "model"]:
    if isinstance(ckpt, dict) and key in ckpt:
        sd = ckpt[key]
        break
else:
    # might already just be a state_dict
    sd = ckpt if isinstance(ckpt, dict) else None

torch.save(sd, "./energy_baseline_min.pth")

# codex generated with minimal oversight!

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple
import torch
from configs import force_model_hyperparam
from data.xyz_extxyz_converter import load_one_frame_from_xyz_extxyz, load_ragged_from_xyz_extxyz
from MLFF_arch.encoderBase import TransformerEncoder
import utils.normalizer


def _build_model(model_path: str, device: torch.device) -> TransformerEncoder:
    cfg = force_model_hyperparam.ModelConfig()
    model = TransformerEncoder(
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        d_out=cfg.d_out,
        dropout_rate=cfg.dropout_rate,
        num_heads=cfg.num_heads,
        vecRepLength=force_model_hyperparam.DataConfig.atom_vec_length,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_stats(stats_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stats = torch.load(stats_path, map_location=device)
    required = ["x_mean", "x_std", "y_mean_force", "y_std_force"]
    missing = [k for k in required if k not in stats]
    if missing:
        raise KeyError(f"Stats file missing keys: {missing}")
    return stats["x_mean"], stats["x_std"], stats["y_mean_force"], stats["y_std_force"]


@torch.inference_mode
def _predict_forces(
    model: TransformerEncoder,
    X: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if X.shape[-1] != model.native_token_dim:
        raise ValueError(
            f"Input feature width mismatch: X has {X.shape[-1]}, model expects {model.native_token_dim}."
        )

    Xb = X.unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, n, v)
    Xn = utils.normalizer.normalize_all(0, Xb, x_mean, x_std)
    pred_n = model(Xn)
    pred = utils.normalizer.unnormalize_all(1, pred_n, y_mean, y_std)
    return pred.squeeze(0).cpu()


def _flatten_for_parity(pred_list: list[torch.Tensor], targ_list: list[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = torch.cat([p.reshape(-1) for p in pred_list], dim=0)
    targ = torch.cat([t.reshape(-1) for t in targ_list], dim=0)
    return pred, targ


def _flatten_force_norms(
    pred_list: list[torch.Tensor],
    targ_list: list[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = torch.cat([torch.norm(p, dim=-1).reshape(-1) for p in pred_list], dim=0)
    targ = torch.cat([torch.norm(t, dim=-1).reshape(-1) for t in targ_list], dim=0)
    return pred, targ


def _metrics(pred: torch.Tensor, targ: torch.Tensor) -> Tuple[float, float, float]:
    diff = pred - targ
    mae = float(diff.abs().mean().item())
    rmse = float(torch.sqrt((diff ** 2).mean()).item())
    denom = torch.sum((targ - torch.mean(targ)) ** 2)
    if float(denom.item()) <= 0.0:
        r2 = float("nan")
    else:
        r2 = float((1.0 - torch.sum(diff ** 2) / denom).item())
    return mae, rmse, r2


def _make_plot(
    pred: torch.Tensor,
    targ: torch.Tensor,
    out_path: str,
    title: str,
    max_points: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to generate parity plots. Install it with `pip install matplotlib`."
        ) from exc

    n = pred.numel()
    if max_points > 0 and n > max_points:
        idx = torch.randperm(n)[:max_points]
        pred_plot = pred[idx]
        targ_plot = targ[idx]
    else:
        pred_plot = pred
        targ_plot = targ

    lo = float(torch.min(torch.min(pred_plot), torch.min(targ_plot)).item())
    hi = float(torch.max(torch.max(pred_plot), torch.max(targ_plot)).item())
    pad = 0.03 * (hi - lo + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.scatter(targ_plot.numpy(), pred_plot.numpy(), s=6, alpha=0.25, color="#FF1F1F", edgecolors="none")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Reference Forces")
    ax.set_ylabel("Predicted Forces")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate force parity plot from a saved model and xyz/extxyz file")
    p.add_argument("--model-path", required=True, help="Path to saved model weights (.pt/.pth)")
    p.add_argument("--xyz-path", required=True, help="Path to xyz/extxyz with force labels")
    p.add_argument(
        "--stats-path",
        default=str(force_model_hyperparam.ModelPaths.stats_path),
        help="Path to normalization stats file",
    )
    p.add_argument("--output", default="parity_plot.png", help="Output PNG path")
    p.add_argument(
        "--device",
        default=force_model_hyperparam.TrainConfig.device,
        help="Torch device string (e.g. cpu, cuda, cuda:1, mps)",
    )
    p.add_argument(
        "--is-crystal",
        action=argparse.BooleanOptionalAction,
        default=force_model_hyperparam.DataConfig.isCrystal,
        help="Require Lattice and pbc in frame headers",
    )
    p.add_argument("--frame-idx", type=int, default=0, help="Frame index (used unless --all-frames)")
    p.add_argument("--all-frames", action="store_true", help="Run on all frames in file")
    p.add_argument("--max-points", type=int, default=250000, help="Max points plotted (metrics use all points)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    device = torch.device(args.device)
    model = _build_model(args.model_path, device)
    x_mean, x_std, y_mean, y_std = _load_stats(args.stats_path, device)

    pred_list: list[torch.Tensor] = []
    targ_list: list[torch.Tensor] = []

    if args.all_frames:
        X_list, Y_list, _ = load_ragged_from_xyz_extxyz(args.xyz_path, is_crystal=bool(args.is_crystal), dtype=torch.float32)
        for X, Y in zip(X_list, Y_list):
            pred_list.append(_predict_forces(model, X, x_mean, x_std, y_mean, y_std, device))
            targ_list.append(Y.cpu())
    else:
        X, Y, _ = load_one_frame_from_xyz_extxyz(
            args.xyz_path,
            frame_idx=args.frame_idx,
            is_crystal=bool(args.is_crystal),
            dtype=torch.float32,
        )
        pred_list.append(_predict_forces(model, X, x_mean, x_std, y_mean, y_std, device))
        targ_list.append(Y.cpu())

    pred, targ = _flatten_for_parity(pred_list, targ_list)
    mae, rmse, r2 = _metrics(pred, targ)
    pred_norm, targ_norm = _flatten_force_norms(pred_list, targ_list)
    mae_norm, rmse_norm, r2_norm = _metrics(pred_norm, targ_norm)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    norm_out = out.with_name(f"{out.stem}_norms{out.suffix}")

    title = f"Force Parity\nMAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.6f}"
    _make_plot(pred, targ, str(out), title=title, max_points=args.max_points)
    norm_title = f"Force Norm Parity\nMAE={mae_norm:.6f}  RMSE={rmse_norm:.6f}  R²={r2_norm:.6f}"
    _make_plot(pred_norm, targ_norm, str(norm_out), title=norm_title, max_points=args.max_points)

    print(f"Saved parity plot to: {out}")
    print(f"Saved force-norm parity plot to: {norm_out}")
    print(f"Points (all dims flattened): {pred.numel()}")
    print(f"MAE={mae:.8f} RMSE={rmse:.8f} R2={r2:.8f}")
    print(f"Force-norm points: {pred_norm.numel()}")
    print(f"Norm MAE={mae_norm:.8f} Norm RMSE={rmse_norm:.8f} Norm R2={r2_norm:.8f}")


if __name__ == "__main__":
    main()

import argparse

from configs import force_model_hyperparam
from inference_time_calcs import run_evaluation
from trainForce import run_training

# python adapt_cli.py train --epochs 100 --train-path ./data/my_train.xyz --test-path None --recompute-stats

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ADAPT command line interface (train or evaluate force model)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_p = subparsers.add_parser("train", help="Train force model")
    train_p.add_argument(
        "--train-path",
        default=force_model_hyperparam.DataConfig.train_path,
        help="Path to training xyz/extxyz (or pre-converted .pt)",
    )
    train_p.add_argument(
        "--test-path",
        default=None,
        help="Optional path to test xyz/extxyz (or pre-converted .pt). If omitted, training runs without evaluation.",
    )
    train_p.add_argument(
        "--is-crystal",
        action=argparse.BooleanOptionalAction,
        default=force_model_hyperparam.DataConfig.isCrystal,
        help="True for crystal, false for molecules. Crystals require Lattice and pbc in headers",
    )
    train_p.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Override number of training epochs",
    )
    train_p.add_argument(
        "--recompute-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute normalization stats from the current training dataset and overwrite stats file",
    )
    train_p.add_argument(
        "--device",
        default=None,
        help="Override compute device for training (e.g. cpu, cuda, mps)",
    )
    train_p.add_argument(
        "--augmentation",
        default=False,
        help="Override data augmentation usage for training",
    )

    eval_p = subparsers.add_parser("eval", help="Evaluate saved model on xyz/extxyz")
    eval_p.add_argument(
        "--path",
        default=force_model_hyperparam.DataConfig.test_path,
        help="Path to xyz/extxyz file with REF_forces:R:3 labels",
    )
    eval_p.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index to evaluate when --all-frames is not set",
    )
    eval_p.add_argument(
        "--all-frames",
        action="store_true",
        help="Evaluate all frames and report aggregate MSE/MAE",
    )
    eval_p.add_argument(
        "--is-crystal",
        action=argparse.BooleanOptionalAction,
        default=force_model_hyperparam.DataConfig.isCrystal,
        help="Require Lattice and pbc in headers",
    )
    eval_p.add_argument(
        "--device",
        default=None,
        help="Override compute device for evaluation (e.g. cpu, cuda, mps)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_training(
            train_path=args.train_path,
            test_path=args.test_path,
            is_crystal=args.is_crystal,
            epochs=args.epochs,
            recompute_stats=args.recompute_stats,
            device=args.device,
            augmentation=args.augmentation,
        )
        return

    if args.mode == "eval":
        run_evaluation(
            xyz_path=args.path,
            frame_idx=args.frame_idx,
            all_frames=args.all_frames,
            is_crystal=args.is_crystal,
            device=args.device,
        )
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

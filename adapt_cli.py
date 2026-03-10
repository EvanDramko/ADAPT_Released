import argparse

from inference_time_calcs import run_evaluation
from trainForce import run_training

# python adapt_cli.py train --epochs 100 --train-path ./data/my_train.xyz --test-path None


def _normalize_optional_str(value):
    if isinstance(value, str) and value.strip().lower() in {"none", ""}:
        return None
    return value

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ADAPT command line interface (train or evaluate force model)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_p = subparsers.add_parser("train", help="Train force model")
    train_p.add_argument(
        "--train-path",
        default=None,
        help="Path to training xyz/extxyz (or pre-converted .pt). If omitted, uses DataConfig.train_path.",
    )
    train_p.add_argument(
        "--test-path",
        default=None,
        help="Optional path to test xyz/extxyz (or pre-converted .pt). If omitted, uses DataConfig.test_path; if that is None, training runs without evaluation.",
    )
    train_p.add_argument(
        "--is-crystal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="True for crystal, false for molecules. Crystals require Lattice and pbc in headers. Defaults to DataConfig.isCrystal (set to False at release).",
    )
    train_p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs. Defaults to TrainConfig.epochs",
    )
    train_p.add_argument(
        "--device",
        default=None,
        help="Override compute device for training (e.g. cpu, cuda, mps). Single gpu support only for now! If omitted, uses TrainConfig.device.",
    )
    train_p.add_argument(
        "--baseline-model",
        default=None,
        help="Optional input path/name for pretrained model weights; overrides ModelPaths.pretrainPath when provided.",
    )
    train_p.add_argument(
        "--augmentation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override data augmentation usage for training. Defaults to TrainConfig.augmentation (set to False at release).",
    )

    eval_p = subparsers.add_parser("eval", help="Evaluate saved model on xyz/extxyz")
    eval_p.add_argument(
        "--path",
        default=None,
        help="Path to xyz/extxyz file with REF_forces:R:3 labels. If omitted, uses DataConfig.test_path.",
    )
    eval_p.add_argument(
        "--frame-idx",
        type=int,
        default=None,
        help="Frame index to evaluate. If omitted, evaluate all frames.",
    )
    eval_p.add_argument(
        "--is-crystal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require Lattice and pbc in headers. Defaults to DataConfig.isCrystal (set to False at release).",
    )
    eval_p.add_argument(
        "--device",
        default=None,
        help="Override compute device for evaluation (e.g. cpu, cuda, mps). If omitted, uses TrainConfig.device.",
    )
    eval_p.add_argument(
        "--model-path",
        default=None,
        help="Optional input path/name for model weights; overrides ModelPaths.pretrainPath when provided.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        normalized_train_path = _normalize_optional_str(args.train_path)
        normalized_test_path = _normalize_optional_str(args.test_path)
        normalized_baseline_model = _normalize_optional_str(args.baseline_model)
        normalized_device = _normalize_optional_str(args.device)
        run_training(
            train_path=normalized_train_path,
            test_path=normalized_test_path,
            is_crystal=args.is_crystal,
            epochs=args.epochs,
            device=normalized_device,
            baseline_model=normalized_baseline_model,
            augmentation=args.augmentation,
        )
        return

    if args.mode == "eval":
        normalized_path = _normalize_optional_str(args.path)
        normalized_model_path = _normalize_optional_str(args.model_path)
        normalized_device = _normalize_optional_str(args.device)

        run_evaluation(
            xyz_path=normalized_path,
            frame_idx=args.frame_idx,
            all_frames=None,
            is_crystal=args.is_crystal,
            device=normalized_device,
            model_path=normalized_model_path,
        )
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

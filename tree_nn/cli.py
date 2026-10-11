from __future__ import annotations

import argparse

from .experiments import ExperimentConfig, run_double_descent, run_experiment


def _experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=("cifar10", "catsdogs"))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument(
        "--cifar-source",
        choices=("torchvision", "huggingface"),
        default="torchvision",
        help="download mirror used for CIFAR-10",
    )
    parser.add_argument(
        "--pet-source",
        choices=("torchvision", "huggingface"),
        default="torchvision",
        help="download mirror used for Oxford-IIIT Pet",
    )
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--cache-images",
        action="store_true",
        help="decode a fixed subset once; useful for repeated CPU evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--trunk-width", type=int, default=24)
    parser.add_argument("--root-hidden", type=int, default=128)
    parser.add_argument("--root-epochs", type=int, default=100)
    parser.add_argument("--root-lr", type=float, default=1e-3)
    parser.add_argument("--root-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--root-target-accuracy",
        type=float,
        default=0.90,
        help="stop root training here so corrective nodes must explain residual errors",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="evaluate complete train/test splits every N root epochs",
    )
    parser.add_argument("--max-nodes", type=int, default=20)
    parser.add_argument("--node-hidden", type=int, default=64)
    parser.add_argument("--node-epochs", type=int, default=30)
    parser.add_argument("--node-lr", type=float, default=3e-3)
    parser.add_argument("--target-accuracy", type=float, default=0.999)
    parser.add_argument("--gate-weight", type=float, default=0.25)
    parser.add_argument("--sparsity-weight", type=float, default=1e-4)
    parser.add_argument("--prune-accuracy", type=float, default=0.995)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="auto")


def _config(args: argparse.Namespace, run_name: str | None = None) -> ExperimentConfig:
    names = ExperimentConfig.__dataclass_fields__
    values = {name: getattr(args, name) for name in names if hasattr(args, name)}
    values["download"] = not args.no_download
    if run_name is not None:
        values["run_name"] = run_name
    return ExperimentConfig(**values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tree-nn", description="Corrective neural-tree experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train one tree and create artifacts")
    _experiment_arguments(train)
    train.add_argument("--run-name", default="main")

    sweep = commands.add_parser("sweep", help="run independent capacity models")
    _experiment_arguments(sweep)
    sweep.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 24, 32, 48, 64],
    )
    sweep.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="independent seeds required for a robust double-descent claim",
    )
    sweep.add_argument("--sweep-name", default="double_descent")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_experiment(_config(args, args.run_name))
    elif args.command == "sweep":
        run_double_descent(_config(args), args.widths, args.seeds, args.sweep_name)


if __name__ == "__main__":
    main()

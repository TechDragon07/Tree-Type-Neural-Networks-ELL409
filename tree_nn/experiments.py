from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from .data import DatasetBundle, get_datasets, materialize_dataset
from .models import CorrectionTree
from .training import (
    TrainingRecord,
    cache_features,
    evaluate_features,
    greedy_prune,
    grow_tree,
    make_loader,
    save_checkpoint,
    save_records,
    seed_everything,
    train_root,
)


@dataclass
class ExperimentConfig:
    dataset: str
    output_dir: str = "outputs"
    data_dir: str = "data"
    seed: int = 42
    image_size: int = 32
    cifar_source: str = "torchvision"
    pet_source: str = "torchvision"
    train_size: int | None = None
    test_size: int | None = None
    label_noise: float = 0.0
    augment: bool = False
    cache_images: bool = False
    batch_size: int = 128
    workers: int = 0
    trunk_width: int = 24
    root_hidden: int = 128
    root_epochs: int = 100
    root_lr: float = 1e-3
    root_weight_decay: float = 0.0
    root_target_accuracy: float = 0.90
    eval_every: int = 1
    max_nodes: int = 20
    node_hidden: int = 64
    node_epochs: int = 30
    node_lr: float = 3e-3
    target_accuracy: float = 0.999
    gate_weight: float = 0.25
    sparsity_weight: float = 1e-4
    prune_accuracy: float = 0.995
    download: bool = True
    device: str = "auto"
    run_name: str = "main"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _write_csv(records: list[TrainingRecord], path: Path) -> None:
    if not records:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def run_experiment(config: ExperimentConfig) -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    run_dir = Path(config.output_dir) / config.dataset / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    bundle = get_datasets(
        name=config.dataset,
        data_dir=config.data_dir,
        image_size=config.image_size,
        train_size=config.train_size,
        test_size=config.test_size,
        label_noise=config.label_noise,
        augment=config.augment,
        cifar_source=config.cifar_source,
        pet_source=config.pet_source,
        seed=config.seed,
        download=config.download,
    )
    if config.cache_images:
        if config.augment:
            raise ValueError("--cache-images cannot be combined with --augment")
        bundle = DatasetBundle(
            materialize_dataset(bundle.train),
            materialize_dataset(bundle.test),
            bundle.num_classes,
            bundle.class_names,
        )
    train_loader = make_loader(
        bundle.train, config.batch_size, shuffle=True, workers=config.workers
    )
    train_eval_loader = make_loader(
        bundle.train, config.batch_size, shuffle=False, workers=config.workers
    )
    test_loader = make_loader(bundle.test, config.batch_size, shuffle=False, workers=config.workers)

    model_config = {
        "num_classes": bundle.num_classes,
        "trunk_width": config.trunk_width,
        "root_hidden": config.root_hidden,
    }
    model = CorrectionTree(**model_config).to(device)
    records = train_root(
        model,
        train_loader,
        test_loader,
        device,
        config.root_epochs,
        config.root_lr,
        config.root_weight_decay,
        config.root_target_accuracy,
        config.eval_every,
    )

    model.freeze_base()
    train_features = cache_features(model, train_eval_loader, device)
    test_features = cache_features(model, test_loader, device)
    records.extend(
        grow_tree(
            model,
            train_features,
            test_features,
            device,
            config.max_nodes,
            config.node_hidden,
            config.node_epochs,
            config.batch_size,
            config.node_lr,
            config.gate_weight,
            config.sparsity_weight,
            config.target_accuracy,
        )
    )

    active_children = greedy_prune(model, train_features, device, config.prune_accuracy)
    pruned_train_loss, pruned_train_accuracy, _, _ = evaluate_features(
        model, train_features, device, active_children
    )
    pruned_test_loss, pruned_test_accuracy, _, _ = evaluate_features(
        model, test_features, device, active_children
    )
    metadata = {
        "dataset": config.dataset,
        "class_names": bundle.class_names,
        "active_children_after_pruning": active_children,
        "unpruned_parameters": model.parameter_count(),
        "pruned_parameters": model.parameter_count(active_children),
        "pruned_train_loss": pruned_train_loss,
        "pruned_train_accuracy": pruned_train_accuracy,
        "pruned_test_loss": pruned_test_loss,
        "pruned_test_accuracy": pruned_test_accuracy,
        "device": str(device),
    }
    save_records(records, run_dir / "metrics.json")
    _write_csv(records, run_dir / "metrics.csv")
    (run_dir / "summary.json").write_text(json.dumps(metadata, indent=2))
    save_checkpoint(model, run_dir / "model.pt", model_config, metadata)

    from .visualize import plot_learning_curves, plot_receptive_fields, plot_tree

    plot_learning_curves(records, run_dir / "learning_curves.png")
    plot_tree(model, bundle.class_names, active_children, run_dir / "tree.png")
    plot_receptive_fields(model, bundle.test, device, run_dir / "receptive_fields.png")
    print(f"Artifacts written to {run_dir}")
    return run_dir


def run_double_descent(
    base_config: ExperimentConfig,
    widths: list[int],
    seeds: list[int],
    sweep_name: str = "double_descent",
) -> Path:
    """Independent capacity runs; plot against measured parameters, not width."""
    sweep_dir = Path(base_config.output_dir) / base_config.dataset / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in seeds:
        for width in widths:
            config = replace(
                base_config,
                seed=seed,
                trunk_width=width,
                root_hidden=width * 4,
                node_hidden=width * 4,
                run_name=f"{sweep_name}_capacity_{width}_seed_{seed}",
            )
            run_dir = Path(config.output_dir) / config.dataset / config.run_name
            metrics_path = run_dir / "metrics.json"
            try:
                metrics = json.loads(metrics_path.read_text())
                completed = bool(metrics)
            except (OSError, json.JSONDecodeError):
                completed = False
            if completed:
                print(f"Reusing completed run {run_dir}")
            else:
                run_dir = run_experiment(config)
                metrics = json.loads((run_dir / "metrics.json").read_text())
            final = metrics[-1]
            rows.append(
                {
                    "seed": seed,
                    "width": width,
                    "parameters": final["parameters"],
                    "train_accuracy": final["train_accuracy"],
                    "test_accuracy": final["test_accuracy"],
                    "train_error": 1.0 - final["train_accuracy"],
                    "test_error": 1.0 - final["test_accuracy"],
                    "nodes": final["nodes"],
                    "interpolates": final["train_accuracy"] >= base_config.target_accuracy,
                }
            )
    with (sweep_dir / "capacity_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (sweep_dir / "capacity_sweep.json").write_text(json.dumps(rows, indent=2))

    from .visualize import plot_double_descent

    plot_double_descent(rows, sweep_dir / "double_descent.png")
    return sweep_dir

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .models import CorrectionTree


@dataclass
class TrainingRecord:
    stage: str
    epoch: int
    nodes: int
    parameters: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    true_class: int | None = None
    predicted_class: int | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: CorrectionTree,
    loader: DataLoader,
    device: torch.device,
    active_children: Iterable[int] | None = None,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images, active_children)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.numel()
    return loss_sum / max(count, 1), correct / max(count, 1)


@torch.no_grad()
def cache_features(
    model: CorrectionTree, loader: DataLoader, device: torch.device
) -> TensorDataset:
    """Cache a frozen trunk once, making stage-wise node growth inexpensive."""
    model.eval()
    all_features: list[Tensor] = []
    all_labels: list[Tensor] = []
    for images, labels in loader:
        all_features.append(model.trunk(images.to(device)).cpu())
        all_labels.append(labels.cpu())
    return TensorDataset(torch.cat(all_features), torch.cat(all_labels))


@torch.no_grad()
def evaluate_features(
    model: CorrectionTree,
    dataset: TensorDataset,
    device: torch.device,
    active_children: Iterable[int] | None = None,
) -> tuple[float, float, Tensor, Tensor]:
    features, labels = dataset.tensors
    logits_parts = []
    for start in range(0, len(labels), 2048):
        logits_parts.append(
            model.logits_from_features(
                features[start : start + 2048].to(device), active_children
            ).cpu()
        )
    logits = torch.cat(logits_parts)
    loss = F.cross_entropy(logits, labels).item()
    predictions = logits.argmax(1)
    accuracy = (predictions == labels).float().mean().item()
    return loss, accuracy, predictions, labels


def dominant_confusion(labels: Tensor, predictions: Tensor) -> tuple[int, int] | None:
    mistakes = labels != predictions
    if not mistakes.any():
        return None
    pairs = torch.stack([labels[mistakes], predictions[mistakes]], dim=1)
    unique_pairs, counts = torch.unique(pairs, dim=0, return_counts=True)
    pair = unique_pairs[counts.argmax()]
    return int(pair[0]), int(pair[1])


def train_root(
    model: CorrectionTree,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    target_accuracy: float = 1.0,
    log_every: int = 1,
) -> list[TrainingRecord]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        list(model.trunk.parameters()) + list(model.root.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    records: list[TrainingRecord] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        if epoch % log_every == 0 or epoch == epochs:
            train_loss, train_accuracy = evaluate(model, train_loader, device)
            test_loss, test_accuracy = evaluate(model, test_loader, device)
            records.append(
                TrainingRecord(
                    "root",
                    epoch,
                    0,
                    model.parameter_count(),
                    train_loss,
                    train_accuracy,
                    test_loss,
                    test_accuracy,
                )
            )
            print(f"root epoch={epoch:03d} train={train_accuracy:.4f} test={test_accuracy:.4f}")
            if train_accuracy >= target_accuracy:
                break
    return records


def grow_tree(
    model: CorrectionTree,
    train_features: TensorDataset,
    test_features: TensorDataset,
    device: torch.device,
    max_nodes: int,
    node_hidden: int,
    node_epochs: int,
    batch_size: int,
    learning_rate: float,
    gate_weight: float = 0.25,
    sparsity_weight: float = 1e-4,
    target_accuracy: float = 1.0,
) -> list[TrainingRecord]:
    """Greedily add a node for the largest remaining confusion pair."""
    model.freeze_base()
    records: list[TrainingRecord] = []
    train_loader = make_loader(train_features, batch_size, shuffle=True)

    for _ in range(max_nodes):
        _, train_accuracy, predictions, labels = evaluate_features(model, train_features, device)
        if train_accuracy >= target_accuracy:
            break
        pair = dominant_confusion(labels, predictions)
        if pair is None:
            break
        true_class, predicted_class = pair
        node = model.add_child(true_class, predicted_class, node_hidden)
        model.freeze_children(except_last=True)
        optimizer = torch.optim.AdamW(node.parameters(), lr=learning_rate, weight_decay=0.0)
        baseline_accuracy = train_accuracy
        best_accuracy = baseline_accuracy
        best_state = deepcopy(node.state_dict())

        route_all = (labels == true_class) & (predictions == predicted_class)
        positive_count = max(int(route_all.sum()), 1)
        negative_count = max(len(labels) - positive_count, 1)
        positive_weight = min(negative_count / positive_count, 100.0)

        for epoch in range(1, node_epochs + 1):
            node.train()
            for features, batch_labels in train_loader:
                features, batch_labels = features.to(device), batch_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    base_logits = model.logits_from_features(
                        features, range(len(model.correction_nodes) - 1)
                    )
                    base_predictions = base_logits.argmax(1)
                    route_target = (
                        (batch_labels == true_class) & (base_predictions == predicted_class)
                    ).float()
                correction, _, gate = node(features)
                per_example = F.cross_entropy(
                    base_logits + correction, batch_labels, reduction="none"
                )
                example_weight = 1.0 + route_target * (positive_weight - 1.0)
                classification_loss = (per_example * example_weight).mean()
                gate_loss = F.binary_cross_entropy(
                    gate,
                    route_target,
                    weight=torch.where(
                        route_target.bool(),
                        torch.full_like(gate, positive_weight),
                        torch.ones_like(gate),
                    ),
                )
                loss = classification_loss + gate_weight * gate_loss + sparsity_weight * gate.mean()
                loss.backward()
                optimizer.step()

            _, epoch_accuracy, _, _ = evaluate_features(model, train_features, device)
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_state = deepcopy(node.state_dict())

        node.load_state_dict(best_state)
        if best_accuracy <= baseline_accuracy:
            del model.correction_nodes[-1]
            print(
                f"rejected pair={true_class}<-{predicted_class}: no training-accuracy improvement"
            )
            break
        train_loss, train_accuracy, _, _ = evaluate_features(model, train_features, device)
        test_loss, test_accuracy, _, _ = evaluate_features(model, test_features, device)
        records.append(
            TrainingRecord(
                "child",
                node_epochs,
                len(model.correction_nodes),
                model.parameter_count(),
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
                true_class,
                predicted_class,
            )
        )
        print(
            f"node={len(model.correction_nodes):02d} pair={true_class}<-{predicted_class} "
            f"train={train_accuracy:.4f} test={test_accuracy:.4f}"
        )
    model.freeze_children()
    return records


def greedy_prune(
    model: CorrectionTree,
    train_features: TensorDataset,
    device: torch.device,
    minimum_accuracy: float,
) -> list[int]:
    """Remove least useful children while preserving a requested train accuracy."""
    active = list(range(len(model.correction_nodes)))
    changed = True
    while changed:
        changed = False
        candidates: list[tuple[float, int]] = []
        for index in active:
            trial = [value for value in active if value != index]
            _, accuracy, _, _ = evaluate_features(model, train_features, device, trial)
            candidates.append((accuracy, index))
        for accuracy, index in sorted(candidates, reverse=True):
            if accuracy >= minimum_accuracy:
                active.remove(index)
                changed = True
                break
    return active


def save_records(records: list[TrainingRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(record) for record in records], indent=2))


def save_checkpoint(
    model: CorrectionTree,
    path: str | Path,
    model_config: dict,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    child_specs = [
        {
            "true_class": child.true_class,
            "predicted_class": child.predicted_class,
            "hidden_dim": child.network.net[1].out_features,
        }
        for child in model.correction_nodes
    ]
    torch.save(
        {
            "model_config": model_config,
            "child_specs": child_specs,
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[CorrectionTree, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CorrectionTree(**checkpoint["model_config"])
    for spec in checkpoint["child_specs"]:
        model.add_child(**spec)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint.get("metadata", {})

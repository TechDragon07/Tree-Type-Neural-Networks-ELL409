from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from torch.utils.data import Dataset

from .data import CIFAR10_MEAN, CIFAR10_STD
from .models import CorrectionTree
from .training import TrainingRecord


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(records: Sequence[TrainingRecord], path: str | Path) -> None:
    if not records:
        return
    x = np.arange(len(records))
    train = [record.train_accuracy for record in records]
    test = [record.test_accuracy for record in records]
    labels = [
        f"root e{record.epoch}" if record.stage == "root" else f"node {record.nodes}"
        for record in records
    ]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(x, train, marker="o", label="train")
    axis.plot(x, test, marker="s", label="test")
    axis.axhline(1.0, color="black", lw=0.8, ls=":")
    axis.set_xticks(x, labels, rotation=45, ha="right")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("accuracy")
    axis.set_title("Stage-wise interpolation")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, path)


def plot_double_descent(rows: list[dict], path: str | Path) -> None:
    widths = sorted({int(row["width"]) for row in rows})
    groups = [[row for row in rows if int(row["width"]) == width] for width in widths]
    parameters = [float(np.mean([row["parameters"] for row in group])) for group in groups]
    train_mean = [float(np.mean([row["train_error"] for row in group])) for group in groups]
    train_std = [float(np.std([row["train_error"] for row in group])) for group in groups]
    test_mean = [float(np.mean([row["test_error"] for row in group])) for group in groups]
    test_std = [float(np.std([row["test_error"] for row in group])) for group in groups]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.errorbar(parameters, train_mean, yerr=train_std, fmt="o-", capsize=3, label="train error")
    axis.errorbar(parameters, test_mean, yerr=test_std, fmt="s-", capsize=3, label="test error")
    interpolating = [
        index
        for index, group in enumerate(groups)
        if np.mean([row["interpolates"] for row in group]) >= 0.5
    ]
    if interpolating:
        threshold = parameters[interpolating[0]]
        axis.axvline(threshold, color="crimson", ls="--", alpha=0.7, label="first interpolation")
    axis.set_xscale("log")
    axis.set_xlabel("mean measured total parameters (log scale)")
    axis.set_ylabel("error")
    axis.set_title("Repeated-seed capacity sweep (mean ± SD)")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, path)


def _image_for_plot(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(CIFAR10_MEAN)[:, None, None]
    std = torch.tensor(CIFAR10_STD)[:, None, None]
    return (image.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _gradient_map(model: CorrectionTree, image: torch.Tensor, node_index: int | None) -> np.ndarray:
    x = image.unsqueeze(0).clone().requires_grad_(True)
    if node_index is None:
        features = model.trunk(x)
        score = model.root(features).max(1).values.mean()
    else:
        output = model.node_output(x, node_index)
        score = (output.strength * output.gate).mean()
    score.backward()
    gradient = x.grad.detach().abs().mean(1).squeeze(0)
    gradient -= gradient.min()
    gradient /= gradient.max().clamp_min(1e-12)
    return gradient.cpu().numpy()


@torch.no_grad()
def _highest_gate_image(
    model: CorrectionTree,
    dataset: Dataset,
    device: torch.device,
    node_index: int,
    candidate_count: int,
) -> tuple[torch.Tensor, float, int]:
    """Select by a fixed max-gate rule rather than manually cherry-picking."""
    best_gate = -1.0
    best_image: torch.Tensor | None = None
    best_label = -1
    count = min(candidate_count, len(dataset))
    for start in range(0, count, 64):
        examples = [dataset[index] for index in range(start, min(start + 64, count))]
        images = torch.stack([example[0] for example in examples]).to(device)
        gates = model.node_output(images, node_index).gate
        value, offset = gates.max(0)
        if float(value) > best_gate:
            best_gate = float(value)
            best_image = images[int(offset)].detach()
            best_label = int(examples[int(offset)][1])
    if best_image is None:
        raise ValueError("Cannot visualize an empty dataset")
    return best_image, best_gate, best_label


def plot_receptive_fields(
    model: CorrectionTree,
    dataset: Dataset,
    device: torch.device,
    path: str | Path,
    sample_index: int = 0,
    candidate_count: int = 256,
) -> None:
    """ERFs for the root and high-gate examples of shallow/deep children."""
    model.eval()
    root_image, _ = dataset[sample_index]
    root_image = root_image.to(device)
    node_indices: list[int | None] = [None]
    if model.correction_nodes:
        node_indices.extend(
            sorted(
                {
                    0,
                    len(model.correction_nodes) // 2,
                    len(model.correction_nodes) - 1,
                }
            )
        )
    fig, axes = plt.subplots(2, len(node_indices), figsize=(3.2 * len(node_indices), 6))
    if len(node_indices) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, node_index in enumerate(node_indices):
        if node_index is None:
            image = root_image
            title = f"root (fixed sample {sample_index})"
        else:
            image, gate, label = _highest_gate_image(
                model, dataset, device, node_index, candidate_count
            )
            node = model.correction_nodes[node_index]
            title = (
                f"child {node_index + 1}: {node.predicted_class}→{node.true_class}\n"
                f"max gate={gate:.3f}, y={label}"
            )
        rgb = _image_for_plot(image)
        heat = _gradient_map(model, image, node_index)
        axes[0, column].imshow(rgb)
        axes[0, column].set_title(title)
        axes[1, column].imshow(rgb)
        axes[1, column].imshow(heat, cmap="magma", alpha=0.62)
        axes[1, column].set_title("input-gradient ERF")
        for row in range(2):
            axes[row, column].axis("off")
    fig.suptitle("Effective receptive field across the corrective hierarchy")
    _save(fig, path)


def plot_tree(
    model: CorrectionTree,
    class_names: Sequence[str],
    active_children: Sequence[int],
    path: str | Path,
) -> None:
    count = len(model.correction_nodes)
    height = max(4.0, 1.0 + 0.62 * count)
    fig, axis = plt.subplots(figsize=(8, height))
    axis.set_xlim(0, 10)
    axis.set_ylim(-count - 0.8, 1.2)
    axis.axis("off")
    root = FancyBboxPatch(
        (3.2, 0.25), 3.6, 0.65, boxstyle="round,pad=0.08", fc="#4472c4", ec="none"
    )
    axis.add_patch(root)
    axis.text(5, 0.58, "shared trunk → root neural node", color="white", ha="center", va="center")
    active = set(active_children)
    for index, child in enumerate(model.correction_nodes):
        y = -index - 0.8
        color = "#70ad47" if index in active else "#aaaaaa"
        axis.plot(
            [5, 5, 2.9], [0.25 if index == 0 else y + 0.7, y + 0.32, y + 0.32], color="#777777"
        )
        box = FancyBboxPatch((2.9, y), 4.2, 0.64, boxstyle="round,pad=0.06", fc=color, ec="none")
        axis.add_patch(box)
        true_name = class_names[child.true_class]
        predicted_name = class_names[child.predicted_class]
        status = "kept" if index in active else "pruned"
        axis.text(
            5,
            y + 0.32,
            f"node {index + 1}: {predicted_name} → {true_name} ({status})",
            color="white",
            ha="center",
            va="center",
        )
    axis.set_title("Corrective neural tree and greedy pruning")
    _save(fig, path)

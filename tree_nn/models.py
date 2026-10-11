from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    """Largest sensible GroupNorm divisor, including for very narrow models."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )


class TinyConvTrunk(nn.Module):
    """Small shared feature extractor; GroupNorm remains stable on tiny error sets."""

    def __init__(self, width: int = 24, dropout: float = 0.0) -> None:
        super().__init__()
        self.width = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                ConvBlock(width, width, stride=1),
                ConvBlock(width, width * 2, stride=2),
                ConvBlock(width * 2, width * 4, stride=2),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = width * 4

    def forward_map(self, x: Tensor, tap: int = -1) -> Tensor:
        x = self.stem(x)
        maps = []
        for block in self.blocks:
            x = block(x)
            maps.append(x)
        return maps[tap]

    def forward(self, x: Tensor) -> Tensor:
        feature_map = self.forward_map(x)
        return self.dropout(F.adaptive_avg_pool2d(feature_map, 1).flatten(1))


class NodeMLP(nn.Module):
    """The small neural network used inside every root/correction node."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class PairCorrectionNode(nn.Module):
    """A gated, non-negative correction from one predicted class to one true class.

    The scalar strength is non-negative. Multiplication by
    ``e_true - e_pred`` therefore enforces the multiclass analogue of positive
    false-negative and negative false-positive child weights.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        true_class: int,
        predicted_class: int,
    ) -> None:
        super().__init__()
        if true_class == predicted_class:
            raise ValueError("A correction pair must contain two different classes")
        self.network = NodeMLP(in_dim, 2, hidden_dim)
        self.num_classes = num_classes
        self.register_buffer("pair", torch.tensor([true_class, predicted_class]))
        output_layer = self.network.net[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, -4.0)

    @property
    def true_class(self) -> int:
        return int(self.pair[0].item())

    @property
    def predicted_class(self) -> int:
        return int(self.pair[1].item())

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw_strength, raw_gate = self.network(features).unbind(dim=1)
        strength = F.softplus(raw_strength)
        gate = torch.sigmoid(raw_gate)
        signed_strength = strength * gate
        correction = features.new_zeros((features.shape[0], self.num_classes))
        correction[:, self.true_class] = signed_strength
        correction[:, self.predicted_class] = -signed_strength
        return correction, strength, gate


@dataclass(frozen=True)
class NodeOutput:
    correction: Tensor
    strength: Tensor
    gate: Tensor


class CorrectionTree(nn.Module):
    """Shared trunk, neural root, and an additive sequence of corrective nodes."""

    def __init__(
        self,
        num_classes: int,
        trunk_width: int = 24,
        root_hidden: int = 128,
        trunk_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.trunk = TinyConvTrunk(trunk_width, dropout=trunk_dropout)
        self.root = NodeMLP(self.trunk.output_dim, num_classes, root_hidden)
        self.correction_nodes = nn.ModuleList()

    @property
    def feature_dim(self) -> int:
        return self.trunk.output_dim

    def add_child(
        self, true_class: int, predicted_class: int, hidden_dim: int
    ) -> PairCorrectionNode:
        node = PairCorrectionNode(
            self.feature_dim,
            hidden_dim,
            self.num_classes,
            true_class,
            predicted_class,
        )
        device = next(self.parameters()).device
        self.correction_nodes.append(node.to(device))
        return node

    def logits_from_features(
        self, features: Tensor, active_children: Iterable[int] | None = None
    ) -> Tensor:
        logits = self.root(features)
        active = set(active_children) if active_children is not None else None
        for index, child in enumerate(self.correction_nodes):
            if active is None or index in active:
                logits = logits + child(features)[0]
        return logits

    def forward(self, x: Tensor, active_children: Iterable[int] | None = None) -> Tensor:
        return self.logits_from_features(self.trunk(x), active_children)

    def node_output(self, x: Tensor, node_index: int) -> NodeOutput:
        features = self.trunk(x)
        correction, strength, gate = self.correction_nodes[node_index](features)
        return NodeOutput(correction, strength, gate)

    def freeze_base(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in self.root.parameters():
            parameter.requires_grad_(False)

    def freeze_children(self, except_last: bool = False) -> None:
        last_index = len(self.correction_nodes) - 1
        for index, child in enumerate(self.correction_nodes):
            trainable = except_last and index == last_index
            for parameter in child.parameters():
                parameter.requires_grad_(trainable)

    def trainable_parameters(self) -> Sequence[nn.Parameter]:
        return tuple(parameter for parameter in self.parameters() if parameter.requires_grad)

    def parameter_count(self, active_children: Iterable[int] | None = None) -> int:
        active = set(active_children) if active_children is not None else None
        modules: list[nn.Module] = [self.trunk, self.root]
        modules.extend(
            child
            for index, child in enumerate(self.correction_nodes)
            if active is None or index in active
        )
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

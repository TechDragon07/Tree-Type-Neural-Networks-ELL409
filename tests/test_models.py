from pathlib import Path

import torch

from tree_nn.analysis import repeated_seed_double_descent
from tree_nn.models import CorrectionTree, PairCorrectionNode
from tree_nn.training import dominant_confusion, load_checkpoint, save_checkpoint


def test_pair_correction_has_required_signs() -> None:
    node = PairCorrectionNode(8, 4, num_classes=3, true_class=2, predicted_class=0)
    correction, strength, gate = node(torch.randn(5, 8))
    assert correction.shape == (5, 3)
    assert torch.all(correction[:, 2] >= 0)
    assert torch.all(correction[:, 0] <= 0)
    assert torch.allclose(correction[:, 1], torch.zeros(5))
    assert torch.all(strength > 0)
    assert torch.all((gate > 0) & (gate < 1))
    assert torch.allclose(correction.sum(1), torch.zeros(5), atol=1e-6)


def test_tree_forward_and_checkpoint_round_trip(tmp_path: Path) -> None:
    config = {"num_classes": 3, "trunk_width": 4, "root_hidden": 8}
    model = CorrectionTree(**config)
    model.add_child(2, 0, hidden_dim=6)
    images = torch.randn(2, 3, 32, 32)
    expected = model(images)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(model, checkpoint, config, {"purpose": "test"})
    restored, metadata = load_checkpoint(checkpoint, torch.device("cpu"))
    assert torch.allclose(restored(images), expected)
    assert metadata["purpose"] == "test"
    assert restored.correction_nodes[0].true_class == 2


def test_dominant_confusion() -> None:
    labels = torch.tensor([0, 0, 0, 1, 2])
    predictions = torch.tensor([1, 1, 2, 1, 0])
    assert dominant_confusion(labels, predictions) == (0, 1)
    assert dominant_confusion(labels, labels) is None


def test_feature_logits_can_be_pruned() -> None:
    model = CorrectionTree(num_classes=2, trunk_width=4, root_hidden=4)
    model.add_child(1, 0, hidden_dim=4)
    features = torch.randn(3, model.feature_dim)
    full = model.logits_from_features(features)
    root_only = model.logits_from_features(features, active_children=[])
    assert full.shape == root_only.shape == (3, 2)
    assert not torch.allclose(full, root_only)


def test_double_descent_claim_requires_repeated_seeds() -> None:
    def rows_for_seed(seed: int, supports: bool) -> list[dict]:
        errors = [0.20, 0.40, 0.30] if supports else [0.40, 0.30, 0.20]
        return [
            {
                "seed": seed,
                "width": width,
                "parameters": width * 100,
                "test_error": error,
                "interpolates": index >= 1,
            }
            for index, (width, error) in enumerate(zip((4, 8, 16), errors))
        ]

    established, supporting, seeds = repeated_seed_double_descent(rows_for_seed(1, True))
    assert not established
    assert (supporting, seeds) == (1, 1)

    rows = rows_for_seed(1, True) + rows_for_seed(2, True) + rows_for_seed(3, False)
    established, supporting, seeds = repeated_seed_double_descent(rows)
    assert established
    assert (supporting, seeds) == (2, 3)

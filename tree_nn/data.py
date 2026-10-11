from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    test: Dataset
    num_classes: int
    class_names: tuple[str, ...]


class HuggingFaceImageDataset(Dataset):
    """Minimal adapter from a Hugging Face image split to torchvision transforms."""

    def __init__(
        self,
        dataset: Any,
        transform: transforms.Compose,
        image_key: str = "img",
        label_key: str = "label",
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        example = self.dataset[index]
        return self.transform(example[self.image_key].convert("RGB")), int(example[self.label_key])


class NoisyLabels(Dataset):
    """Deterministic symmetric label noise without changing image sampling."""

    def __init__(self, dataset: Dataset, num_classes: int, probability: float, seed: int) -> None:
        self.dataset = dataset
        rng = random.Random(seed)
        self.replacements: dict[int, int] = {}
        for index in range(len(dataset)):
            if rng.random() < probability:
                label = _label_at(dataset, index)
                choices = [value for value in range(num_classes) if value != int(label)]
                self.replacements[index] = rng.choice(choices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        return image, self.replacements.get(index, int(label))


def _label_at(dataset: Dataset, index: int) -> int:
    """Read labels without decoding/augmenting images when the dataset exposes them."""
    if isinstance(dataset, Subset):
        return _label_at(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, HuggingFaceImageDataset):
        return int(dataset.dataset[index][dataset.label_key])
    if hasattr(dataset, "targets"):
        return int(dataset.targets[index])
    if hasattr(dataset, "_bin_labels"):
        return int(dataset._bin_labels[index])
    _, label = dataset[index]
    return int(label)


def _limit(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def materialize_dataset(dataset: Dataset) -> TensorDataset:
    """Decode and transform a fixed subset once for repeated CPU evaluation."""
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in range(len(dataset)):
        image, label = dataset[index]
        images.append(image)
        labels.append(int(label))
    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))


def _transforms(image_size: int, augment: bool) -> transforms.Compose:
    operations: list[Any] = [transforms.Resize((image_size, image_size))]
    if augment:
        operations.extend(
            [
                transforms.RandomCrop(image_size, padding=max(2, image_size // 8)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return transforms.Compose(operations)


def get_datasets(
    name: str,
    data_dir: str | Path = "data",
    image_size: int = 32,
    train_size: int | None = None,
    test_size: int | None = None,
    label_noise: float = 0.0,
    augment: bool = False,
    cifar_source: str = "torchvision",
    pet_source: str = "torchvision",
    seed: int = 42,
    download: bool = True,
) -> DatasetBundle:
    """Create leakage-free train/test datasets.

    ``catsdogs`` uses Oxford-IIIT Pet and predicts species rather than breed,
    providing a legally redistributable, directly downloadable cat/dog dataset.
    """
    name = name.lower().replace("-", "")
    root = Path(data_dir)
    if not 0.0 <= label_noise < 1.0:
        raise ValueError("label_noise must be in [0, 1)")

    if name == "cifar10":
        if cifar_source == "huggingface":
            from datasets import load_dataset

            raw = load_dataset("uoft-cs/cifar10")
            train = HuggingFaceImageDataset(raw["train"], _transforms(image_size, augment))
            test = HuggingFaceImageDataset(raw["test"], _transforms(image_size, False))
            class_names = (
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            )
        elif cifar_source == "torchvision":
            train = datasets.CIFAR10(
                root,
                train=True,
                transform=_transforms(image_size, augment),
                download=download,
            )
            test = datasets.CIFAR10(
                root,
                train=False,
                transform=_transforms(image_size, False),
                download=download,
            )
            class_names = tuple(train.classes)
        else:
            raise ValueError("cifar_source must be 'torchvision' or 'huggingface'")
    elif name in {"catsdogs", "catvsdog", "catsvsdogs"}:
        if pet_source == "huggingface":
            from datasets import load_dataset

            raw = load_dataset("timm/oxford-iiit-pet")
            train = HuggingFaceImageDataset(
                raw["train"],
                _transforms(image_size, augment),
                image_key="image",
                label_key="label_cat_dog",
            )
            test = HuggingFaceImageDataset(
                raw["test"],
                _transforms(image_size, False),
                image_key="image",
                label_key="label_cat_dog",
            )
        elif pet_source == "torchvision":
            train = datasets.OxfordIIITPet(
                root,
                split="trainval",
                target_types="binary-category",
                transform=_transforms(image_size, augment),
                download=download,
            )
            test = datasets.OxfordIIITPet(
                root,
                split="test",
                target_types="binary-category",
                transform=_transforms(image_size, False),
                download=download,
            )
        else:
            raise ValueError("pet_source must be 'torchvision' or 'huggingface'")
        class_names = ("cat", "dog")
    else:
        raise ValueError(f"Unknown dataset {name!r}; choose 'cifar10' or 'catsdogs'")

    train = _limit(train, train_size, seed)
    test = _limit(test, test_size, seed + 1)
    if label_noise:
        train = NoisyLabels(train, len(class_names), label_noise, seed + 2)
    return DatasetBundle(train, test, len(class_names), class_names)

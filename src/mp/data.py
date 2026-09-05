"""Calibration and evaluation datasets.

Two roles, deliberately kept apart:

* a **calibration** set supplies the unlabelled images over which the alignment
  gradient is accumulated. Nothing is learned from it and no labels are read.
* an **evaluation** set supplies labelled train/test splits for the frozen
  feature probes that measure what pruning cost.

Compute nodes have no outbound internet, so every dataset must already be on
disk when a job starts; ``scripts/prefetch.py`` materialises them on the login
node. Constructors here never download unless ``allow_download`` is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from torchvision.datasets import (
    DTD,
    CIFAR100,
    EuroSAT,
    Flowers102,
    ImageFolder,
    OxfordIIITPet,
    StanfordCars,
)

#: Where images live on the cluster. Overridable so the same code runs on a laptop.
DEFAULT_DATA_ROOT = os.environ.get("MP_DATA_ROOT", os.path.expanduser("~/datasets"))

#: ImageNet, in the usual ImageFolder layout: ``<root>/train`` and ``<root>/val``.
#: The training split is the generic calibration corpus every training-free
#: method uses, and the two splits give the ImageNet probe a proper protocol.
IMAGENET_ROOT = os.environ.get(
    "MP_IMAGENET_ROOT", os.path.join(DEFAULT_DATA_ROOT, "ImageNet")
)
IMAGENET_TRAIN = os.path.join(IMAGENET_ROOT, "train")
IMAGENET_VAL = os.path.join(IMAGENET_ROOT, "val")


def build_transform(data_config: Dict, train: bool = False) -> Callable:
    """Deterministic resize/crop/normalise matching the backbone's pretraining."""
    size = data_config["input_size"][-1]
    interpolation = T.InterpolationMode.BICUBIC
    crop_pct = data_config.get("crop_pct", 1.0) or 1.0
    resize = int(round(size / crop_pct))

    return T.Compose(
        [
            T.Resize(resize, interpolation=interpolation),
            T.CenterCrop(size),
            T.Lambda(lambda im: im.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=data_config["mean"], std=data_config["std"]),
        ]
    )


class UnlabelledDataset(Dataset):
    """Strips labels, so a calibration loader cannot accidentally use them."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        item = self.dataset[index]
        return item[0] if isinstance(item, (tuple, list)) else item


# --------------------------------------------------------------------------- #
# Evaluation datasets
# --------------------------------------------------------------------------- #


@dataclass
class EvalSplits:
    """A labelled dataset reduced to the two splits the probes need."""

    name: str
    train: Dataset
    test: Dataset
    num_classes: int


class _BalancedSubset(Dataset):
    """A fixed number of images per class, with labels remapped to 0..C-1."""

    def __init__(self, base: ImageFolder, per_class: int, keep: Sequence[int]):
        remap = {label: i for i, label in enumerate(keep)}
        wanted = set(keep)
        counts: Dict[int, int] = {}
        self.items: List[Tuple[int, int]] = []

        for index, (_, label) in enumerate(base.samples):
            if label not in wanted or counts.get(label, 0) >= per_class:
                continue
            counts[label] = counts.get(label, 0) + 1
            self.items.append((index, remap[label]))

        self.base = base

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        index, label = self.items[i]
        return self.base[index][0], label


def _imagenet_splits(
    transform: Callable, num_classes: int, per_class_train: int, per_class_test: int
) -> EvalSplits:
    """A balanced ImageNet subset: train images to fit, val images to score.

    Restricted to a subset of classes so that feature extraction stays
    affordable at every sparsity level of every run, but otherwise the standard
    protocol: the probe never sees a validation image during fitting.
    """
    train_folder = ImageFolder(IMAGENET_TRAIN, transform=transform)
    val_folder = ImageFolder(IMAGENET_VAL, transform=transform)

    keep = sorted({label for _, label in train_folder.samples})[:num_classes]

    return EvalSplits(
        name=f"imagenet{num_classes}",
        train=_BalancedSubset(train_folder, per_class_train, keep),
        test=_BalancedSubset(val_folder, per_class_test, keep),
        num_classes=len(keep),
    )


def load_eval_dataset(
    name: str,
    transform: Callable,
    root: str = DEFAULT_DATA_ROOT,
    allow_download: bool = False,
) -> EvalSplits:
    """Build one evaluation dataset by name."""
    kwargs = {"root": root, "transform": transform, "download": allow_download}

    if name == "cifar100":
        return EvalSplits(
            name,
            CIFAR100(train=True, **kwargs),
            CIFAR100(train=False, **kwargs),
            num_classes=100,
        )
    if name == "pets":
        return EvalSplits(
            name,
            OxfordIIITPet(split="trainval", **kwargs),
            OxfordIIITPet(split="test", **kwargs),
            num_classes=37,
        )
    if name == "dtd":
        return EvalSplits(
            name,
            DTD(split="train", **kwargs),
            DTD(split="test", **kwargs),
            num_classes=47,
        )
    if name == "flowers":
        return EvalSplits(
            name,
            Flowers102(split="train", **kwargs),
            Flowers102(split="test", **kwargs),
            num_classes=102,
        )
    if name == "eurosat":
        # EuroSAT ships a single split; halve it deterministically.
        full = EuroSAT(**kwargs)
        generator = torch.Generator().manual_seed(0)
        order = torch.randperm(len(full), generator=generator).tolist()
        cut = int(0.7 * len(full))
        return EvalSplits(
            name, Subset(full, order[:cut]), Subset(full, order[cut:]), num_classes=10
        )
    if name == "cars":
        return EvalSplits(
            name,
            StanfordCars(split="train", **kwargs),
            StanfordCars(split="test", **kwargs),
            num_classes=196,
        )
    if name.startswith("imagenet"):
        num_classes = int(name.removeprefix("imagenet") or 100)
        return _imagenet_splits(
            transform,
            num_classes=num_classes,
            per_class_train=60,
            per_class_test=50,
        )

    raise KeyError(f"unknown evaluation dataset {name!r}")


EVAL_DATASETS = ["cifar100", "pets", "dtd", "flowers", "eurosat", "cars", "imagenet100"]


def cap_splits(splits: EvalSplits, max_train: int, max_test: int, seed: int = 0) -> EvalSplits:
    """Subsample large splits to a fixed budget, deterministically.

    Feature extraction dominates the cost of the campaign and scales with split
    size, while a linear probe over a few hundred examples per class is already
    saturated. Capping keeps every dataset affordable at the same protocol; the
    caps are held fixed across all runs, so no comparison is affected.
    """

    def _cap(dataset: Dataset, limit: int) -> Dataset:
        if len(dataset) <= limit:
            return dataset
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
        return Subset(dataset, sorted(order))

    return EvalSplits(
        name=splits.name,
        train=_cap(splits.train, max_train),
        test=_cap(splits.test, max_test),
        num_classes=splits.num_classes,
    )


# --------------------------------------------------------------------------- #
# Calibration sets
# --------------------------------------------------------------------------- #


def load_calibration_images(
    source: str,
    transform: Optional[Callable],
    num_samples: int,
    seed: int,
    root: str = DEFAULT_DATA_ROOT,
    allow_download: bool = False,
) -> Dataset:
    """A fixed-size unlabelled image set drawn from ``source``.

    ``source`` is either ``imagenet`` (the generic corpus every published
    training-free method calibrates on) or the name of an evaluation dataset,
    whose *training* split is then used. The second form is what "task-specific"
    pruning means: the calibration images come from the target domain, and no
    labels are involved either way.

    Passing ``transform=None`` yields PIL images, which is what the paired-view
    protocols below need.
    """
    if source == "imagenet":
        # The training split, as every published training-free method uses.
        base = ImageFolder(IMAGENET_TRAIN, transform=transform)
    else:
        base = load_eval_dataset(
            source, transform=transform, root=root, allow_download=allow_download
        ).train

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(base), generator=generator)[:num_samples].tolist()
    return UnlabelledDataset(Subset(base, order))


# --------------------------------------------------------------------------- #
# View protocols
# --------------------------------------------------------------------------- #
#
# The alignment loss is identically zero when the teacher and the student are
# the same network reading the same pixels, and so is its gradient: the
# unpruned point is an exact global minimum of every objective considered here.
# A saliency pass therefore only carries signal if the two views differ. How
# they differ is a design choice, and it is treated here as a component to be
# ablated rather than an implementation detail.


class PairedViewDataset(Dataset):
    """Yields (teacher view, student view) for each calibration image."""

    def __init__(self, base: Dataset, views: Callable):
        self.base = base
        self.views = views

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.base[index]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return self.views(image)


class TwoCropViews:
    """Teacher sees a global crop, student a local crop of the same image.

    The self-distillation protocol of DINO, which SnapViT adopts for pruning.
    Both crops are resized to the model's resolution, so the two token grids
    have the same shape and their Gram matrices are comparable.
    """

    def __init__(self, data_config: Dict, global_scale=(0.25, 1.0), local_scale=(0.05, 0.25)):
        size = data_config["input_size"][-1]
        normalise = [
            T.Lambda(lambda im: im.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=data_config["mean"], std=data_config["std"]),
        ]
        self.global_view = T.Compose(
            [T.RandomResizedCrop(size, scale=global_scale, antialias=True), *normalise]
        )
        self.local_view = T.Compose(
            [T.RandomResizedCrop(size, scale=local_scale, antialias=True), *normalise]
        )

    def __call__(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.global_view(image), self.local_view(image)


class NoisyViews:
    """Teacher sees the clean image, student the same image plus Gaussian noise.

    The perturbation Cut-ViT invokes when it argues that noise injection makes
    the recovered bases noise-invariant. Unlike two crops, this leaves the
    spatial layout intact, so the token-token Gram matrices of the two views
    describe the same scene.
    """

    def __init__(self, data_config: Dict, sigma: float = 0.25):
        self.base = build_transform(data_config)
        self.sigma = sigma

    def __call__(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.base(image)
        return clean, clean + self.sigma * torch.randn_like(clean)


class IdenticalViews:
    """Both networks see the same pixels.

    Retained as a control: it is the configuration in which every objective and
    its gradient are exactly zero, so any resulting ranking is arithmetic noise.
    """

    def __init__(self, data_config: Dict):
        self.base = build_transform(data_config)

    def __call__(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        view = self.base(image)
        return view, view.clone()


VIEW_PROTOCOLS = {
    "two-crop": TwoCropViews,
    "noise": NoisyViews,
    "identical": IdenticalViews,
}


def build_views(protocol: str, data_config: Dict) -> Callable:
    if protocol not in VIEW_PROTOCOLS:
        raise KeyError(f"unknown view protocol {protocol!r}; known: {sorted(VIEW_PROTOCOLS)}")
    return VIEW_PROTOCOLS[protocol](data_config)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

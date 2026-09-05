"""Frozen-feature probes.

Pruning is scored by what the resulting backbone's features can still support,
with the backbone frozen: a weighted k-nearest-neighbour classifier, which
involves no fitting at all, and a linear probe, which fits only a single matrix.
Neither touches the backbone, so differences between rows come from the pruning
and nothing else.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import EvalSplits, make_loader
from .models import token_features


@dataclass
class FeatureBank:
    features: torch.Tensor  # [N, D]
    targets: torch.Tensor  # [N]


def autocast_context(device: torch.device, enabled: bool):
    """bfloat16 autocast on CUDA, a no-op elsewhere.

    Feature extraction dominates the cost of a campaign, and bfloat16 roughly
    halves it. Every row is extracted the same way, so comparisons between rows
    are unaffected; the probes themselves are fitted in float32.
    """
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    representation: str = "cls_avg",
    progress: bool = False,
    amp: bool = True,
) -> FeatureBank:
    """Embed a labelled split.

    ``representation``:
        ``cls``      the class token alone,
        ``avg``      the mean of the patch tokens,
        ``cls_avg``  the two concatenated, the usual choice for DINO-family
                     linear evaluation.
    """
    model.eval()
    all_features, all_targets = [], []

    for images, targets in tqdm(loader, disable=not progress, desc="features"):
        images = images.to(device, non_blocking=True)
        with autocast_context(device, amp):
            tokens = token_features(model, images, tokens="all")
        num_prefix = getattr(model, "num_prefix_tokens", 1)
        cls = tokens[:, 0]
        avg = tokens[:, num_prefix:].mean(dim=1)

        if representation == "cls":
            feats = cls
        elif representation == "avg":
            feats = avg
        elif representation == "cls_avg":
            feats = torch.cat([cls, avg], dim=1)
        else:
            raise ValueError(f"unknown representation {representation!r}")

        all_features.append(feats.float().cpu())
        all_targets.append(torch.as_tensor(targets).cpu())

    return FeatureBank(torch.cat(all_features), torch.cat(all_targets))


@torch.no_grad()
def knn_accuracy(
    train: FeatureBank,
    test: FeatureBank,
    device: torch.device,
    k: int = 20,
    temperature: float = 0.07,
    num_classes: Optional[int] = None,
    chunk: int = 512,
) -> float:
    """Similarity-weighted k-NN on L2-normalised features, as used by DINO."""
    num_classes = num_classes or int(train.targets.max()) + 1
    bank = F.normalize(train.features.to(device), dim=1)
    bank_targets = train.targets.to(device)

    correct = 0
    for start in range(0, len(test.features), chunk):
        query = F.normalize(test.features[start : start + chunk].to(device), dim=1)
        similarity = query @ bank.T

        top_sim, top_idx = similarity.topk(min(k, bank.shape[0]), dim=1)
        weights = (top_sim / temperature).exp()
        neighbour_labels = bank_targets[top_idx]

        votes = torch.zeros(query.shape[0], num_classes, device=device)
        votes.scatter_add_(1, neighbour_labels, weights)

        predicted = votes.argmax(dim=1)
        correct += int((predicted == test.targets[start : start + chunk].to(device)).sum())

    return correct / len(test.targets)


def linear_probe_accuracy(
    train: FeatureBank,
    test: FeatureBank,
    device: torch.device,
    num_classes: int,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> float:
    """Fit one linear layer on standardised frozen features.

    Features are standardised using the training split's statistics, which makes
    a single learning rate work across backbones whose feature scales differ by
    an order of magnitude.
    """
    torch.manual_seed(seed)

    mean = train.features.mean(dim=0, keepdim=True)
    std = train.features.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_train = ((train.features - mean) / std).to(device)
    y_train = train.targets.to(device).long()
    x_test = ((test.features - mean) / std).to(device)
    y_test = test.targets.to(device).long()

    classifier = nn.Linear(x_train.shape[1], num_classes).to(device)
    optimiser = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    num_samples = x_train.shape[0]
    for _ in range(epochs):
        order = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, batch_size):
            index = order[start : start + batch_size]
            loss = F.cross_entropy(classifier(x_train[index]), y_train[index])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        schedule.step()

    with torch.no_grad():
        predicted = classifier(x_test).argmax(dim=1)
        return float((predicted == y_test).float().mean())


def evaluate_backbone(
    model: nn.Module,
    splits: EvalSplits,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 8,
    representation: str = "cls_avg",
    linear_epochs: int = 100,
    seed: int = 0,
    amp: bool = True,
) -> Dict[str, float]:
    """k-NN and linear-probe accuracy for one backbone on one dataset."""
    started = time.perf_counter()

    train_bank = extract_features(
        model,
        make_loader(splits.train, batch_size=batch_size, num_workers=num_workers),
        device,
        representation=representation,
        amp=amp,
    )
    test_bank = extract_features(
        model,
        make_loader(splits.test, batch_size=batch_size, num_workers=num_workers),
        device,
        representation=representation,
        amp=amp,
    )

    return {
        "knn": knn_accuracy(
            train_bank, test_bank, device=device, num_classes=splits.num_classes
        ),
        "linear": linear_probe_accuracy(
            train_bank,
            test_bank,
            device=device,
            num_classes=splits.num_classes,
            epochs=linear_epochs,
            seed=seed,
        ),
        "num_train": len(train_bank.targets),
        "num_test": len(test_bank.targets),
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def dense_retrieval_score(
    model: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 16,
) -> Dict[str, float]:
    """How well pruned patch tokens preserve the teacher's dense correspondences.

    For each image the teacher's patch-token similarity matrix defines, for every
    patch, which other patches are its nearest neighbours. The same is computed
    for the pruned model and the two rankings are compared. This is a
    label-free stand-in for the dense tasks (matching, video propagation) that
    pruning methods are usually scored on, and it needs no segmentation masks.

    Reports recall@1 and recall@5 of the teacher's top neighbour under the
    pruned model's ranking, and the mean Spearman correlation of the two
    similarity rows.
    """
    model.eval()
    teacher.eval()

    r1_total, r5_total, rho_total, count = 0.0, 0.0, 0.0, 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)

        f_t = F.normalize(token_features(teacher, images, tokens="patch").float(), dim=-1)
        f_s = F.normalize(token_features(model, images, tokens="patch").float(), dim=-1)

        sim_t = f_t @ f_t.transpose(1, 2)
        sim_s = f_s @ f_s.transpose(1, 2)

        eye = torch.eye(sim_t.shape[-1], device=device, dtype=torch.bool)
        sim_t = sim_t.masked_fill(eye, -2.0)
        sim_s = sim_s.masked_fill(eye, -2.0)

        target = sim_t.argmax(dim=-1)
        top5 = sim_s.topk(5, dim=-1).indices

        r1_total += float((top5[..., 0] == target).float().mean())
        r5_total += float((top5 == target.unsqueeze(-1)).any(dim=-1).float().mean())

        rank_t = sim_t.argsort(dim=-1).argsort(dim=-1).float()
        rank_s = sim_s.argsort(dim=-1).argsort(dim=-1).float()
        rank_t = rank_t - rank_t.mean(dim=-1, keepdim=True)
        rank_s = rank_s - rank_s.mean(dim=-1, keepdim=True)
        rho = (rank_t * rank_s).sum(-1) / (
            rank_t.norm(dim=-1) * rank_s.norm(dim=-1)
        ).clamp_min(1e-12)
        rho_total += float(rho.mean())
        count += 1

    count = max(count, 1)
    return {
        "dense_r1": r1_total / count,
        "dense_r5": r5_total / count,
        "dense_spearman": rho_total / count,
    }


@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int, int],
    batch_size: int = 64,
    steps: int = 50,
    warmup: int = 10,
) -> Dict[str, float]:
    """Images per second at inference, measured rather than inferred from FLOPs."""
    model.eval()
    images = torch.randn((batch_size, *input_size), device=device)

    for _ in range(warmup):
        model.forward_features(images)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        model.forward_features(images)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    return {"images_per_second": steps * batch_size / elapsed, "batch_size": batch_size}

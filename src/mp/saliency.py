"""The one-shot saliency pass.

A frozen copy of the backbone acts as teacher; the copy being pruned acts as
student. Each reads its own view of the same unlabelled calibration image, an
alignment loss is evaluated between their features, and its gradient is
accumulated into the two weight matrices whose structure maps onto prunable
units. Nothing is updated: the backward pass exists only to produce importance
scores. The two views must differ, or the loss and its gradient are identically
zero; ``mp.data`` defines the protocols that make them differ.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import PrunableViT, token_features
from .objectives import AlignmentObjective, cumulative_explained_variance, gram, spectral_entropy


@dataclass
class SaliencyReport:
    """What one calibration pass measured, beyond the scores themselves."""

    seconds: float
    peak_memory_bytes: int
    num_batches: int
    skipped_batches: int = 0
    loss_terms: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "seconds": self.seconds,
            "peak_memory_gb": self.peak_memory_bytes / 1024**3,
            "num_batches": self.num_batches,
            "skipped_batches": self.skipped_batches,
            "loss_terms": self.loss_terms,
            "diagnostics": self.diagnostics,
        }


def make_teacher(model: nn.Module) -> nn.Module:
    """A frozen, detached copy of the backbone."""
    teacher = copy.deepcopy(model).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def spectrum_diagnostics(
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tokens: str = "patch",
    rank: int = 192,
    max_batches: int = 4,
) -> Dict[str, float]:
    """Spectral entropy and explained variance of the teacher's Gram matrices.

    These describe how much room the spectral exponent has to act: a flat
    spectrum makes every member of the family behave alike, a concentrated one
    makes the choice of exponent decisive.
    """
    stats: Dict[str, List[float]] = {}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(
            device, non_blocking=True
        )
        features = token_features(teacher, images, tokens=tokens)
        for axis in ("spatial", "channel"):
            g = gram(features, axis=axis, pooling="per_image").float()
            stats.setdefault(f"entropy_{axis}", []).append(float(spectral_entropy(g)))
            stats.setdefault(f"cevr{rank}_{axis}", []).append(
                float(cumulative_explained_variance(g, k=rank))
            )
            stats.setdefault(f"rank_{axis}", []).append(float(g.shape[-1]))
    return {key: sum(values) / len(values) for key, values in stats.items() if values}


def estimate_importance(
    prunable: PrunableViT,
    teacher: nn.Module,
    objective: AlignmentObjective,
    loader: DataLoader,
    device: torch.device,
    progress: bool = True,
) -> SaliencyReport:
    """Accumulate the alignment gradient over the calibration set.

    Returns the timing and memory the pass cost, which is the efficiency claim
    every method in this literature makes, measured under identical conditions.
    """
    tokens = objective.config.tokens

    prunable.enable_importance_gradients()
    prunable.model.eval()
    teacher.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    accumulated: Dict[str, float] = {}
    num_batches = 0
    skipped = 0

    # Gradients are accumulated by hand rather than by repeated backward() so
    # that each batch can be checked before it is added. An eigendecomposition
    # backward returns NaN on the occasional batch whose leading spectrum is
    # degenerate, and a single such batch poisons the running sum for good.
    parameters = prunable.gradient_parameters
    buffers = [torch.zeros_like(p) for p in parameters]

    iterator = tqdm(loader, desc=objective.config.name, disable=not progress)
    for teacher_view, student_view in iterator:
        teacher_view = teacher_view.to(device, non_blocking=True)
        student_view = student_view.to(device, non_blocking=True)

        with torch.no_grad():
            features_teacher = token_features(teacher, teacher_view, tokens=tokens).float()

        features_student = token_features(prunable.model, student_view, tokens=tokens).float()

        loss, terms = objective(features_teacher, features_student)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)

        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            skipped += 1
            continue

        for buffer, gradient in zip(buffers, gradients):
            buffer += gradient

        for key, value in terms.items():
            accumulated[key] = accumulated.get(key, 0.0) + value
        accumulated["total"] = accumulated.get("total", 0.0) + float(loss.detach())
        num_batches += 1

    if num_batches == 0:
        raise FloatingPointError(
            f"{objective.config.name}: every one of {skipped} calibration batches "
            "produced a non-finite gradient"
        )

    for parameter, buffer in zip(parameters, buffers):
        parameter.grad = buffer

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = 0

    # A non-finite score is not a degraded ranking, it is no ranking at all:
    # argsort puts NaN wherever it likes and the pruning that follows looks
    # ordinary. Fail here rather than let a run report numbers for it.
    scores = prunable.importance_state()
    for key, tensor in scores.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"{objective.config.name}: {key} importance contains "
                f"{int((~torch.isfinite(tensor)).sum())} non-finite entries"
            )

    return SaliencyReport(
        seconds=time.perf_counter() - started,
        peak_memory_bytes=peak,
        num_batches=num_batches,
        skipped_batches=skipped,
        loss_terms={key: value / max(num_batches, 1) for key, value in accumulated.items()},
    )


# --------------------------------------------------------------------------- #
# Importance baselines that need no alignment objective
# --------------------------------------------------------------------------- #


def magnitude_importance(prunable: PrunableViT) -> Dict[str, torch.Tensor]:
    """Weight magnitude, the standard data-free reference."""
    mlp, head = [], []
    for block in prunable.model.blocks:
        mlp.append((block.mlp.fc1.weight**2).mean(dim=1).detach().cpu())
        num_heads = block.attn.num_heads
        value = block.attn.qkv.weight.reshape(3, num_heads, prunable.head_dim, prunable.embed_dim)[2]
        head.append((value**2).mean(dim=(1, 2)).detach().cpu())
    return {"mlp": torch.stack(mlp), "head": torch.stack(head)}


def random_importance(prunable: PrunableViT, seed: int) -> Dict[str, torch.Tensor]:
    """Uniform random scores: the floor any method must clear."""
    generator = torch.Generator().manual_seed(seed)
    mlp = torch.rand(
        (prunable.num_blocks, prunable.default_hidden_dim), generator=generator
    )
    head = torch.rand((prunable.num_blocks, prunable.default_num_heads), generator=generator)
    return {"mlp": mlp, "head": head}


def lamp_importance(prunable: PrunableViT) -> Dict[str, torch.Tensor]:
    """Layer-adaptive magnitude, normalised by the block's squared weight mass.

    The structured analogue of LAMP: within a block, scores are magnitudes
    normalised by the tail sum of the sorted magnitudes, which makes them
    comparable across blocks without a per-layer hyperparameter.
    """
    base = magnitude_importance(prunable)
    out = {}
    for key, scores in base.items():
        adapted = torch.zeros_like(scores)
        for i, row in enumerate(scores):
            order = row.argsort(descending=True)
            sorted_row = row[order]
            tail = torch.flip(torch.cumsum(torch.flip(sorted_row, [0]), 0), [0])
            adapted[i, order] = sorted_row / tail.clamp_min(1e-12)
        out[key] = adapted
    return out

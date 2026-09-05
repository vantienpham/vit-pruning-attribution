"""Vision transformer backbones and the structured pruning surgery.

Pruning here is structural and one-shot: whole MLP hidden units and whole
attention heads are removed and the weight matrices are physically resized, so a
pruned model is smaller and faster rather than masked. Importance is the
empirical Fisher of an alignment objective, accumulated in a single backward
pass over unlabelled calibration images.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention

#: Backbones used in the paper. All are public checkpoints resolvable through
#: timm, so the pipeline needs no gated download.
BACKBONES: Dict[str, str] = {
    "dinov2-vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2-vits14": "vit_small_patch14_dinov2.lvd142m",
    "dino-vitb16": "vit_base_patch16_224.dino",
    "dino-vits16": "vit_small_patch16_224.dino",
    "deit-vitb16": "deit_base_patch16_224.fb_in1k",
    "clip-vitb16": "vit_base_patch16_clip_224.openai",
    "augreg-vitb16": "vit_base_patch16_224.augreg_in21k_ft_in1k",
}


class DynamicHeadsAttention(Attention):
    """timm attention that tolerates a head count below the embedding width.

    After heads are removed, the concatenated head output is narrower than the
    residual stream, and only the output projection restores the width. timm's
    own forward reshapes to the original dimension and would fail.
    """

    @classmethod
    def from_timm(cls, attn: Attention) -> "DynamicHeadsAttention":
        wrapped = cls(dim=attn.head_dim * attn.num_heads, num_heads=attn.num_heads)
        wrapped.qkv = attn.qkv
        wrapped.proj = attn.proj
        wrapped.q_norm = getattr(attn, "q_norm", nn.Identity())
        wrapped.k_norm = getattr(attn, "k_norm", nn.Identity())
        wrapped.attn_drop = attn.attn_drop
        wrapped.proj_drop = attn.proj_drop
        wrapped.scale = attn.scale
        wrapped.head_dim = attn.head_dim
        wrapped.num_heads = attn.num_heads
        wrapped.fused_attn = attn.fused_attn
        return wrapped

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v

        x = x.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim)
        return self.proj_drop(self.proj(x))


def _replace_attention(model: nn.Module) -> nn.Module:
    for block in model.blocks:
        block.attn = DynamicHeadsAttention.from_timm(block.attn)
    return model


def load_backbone(
    key: str,
    device: torch.device,
    img_size: Optional[int] = None,
) -> Tuple[nn.Module, Dict]:
    """Build a backbone by short name and return it with its data config.

    The classifier head is discarded: every objective and every evaluation in
    this work reads the backbone's token features.
    """
    if key not in BACKBONES:
        raise KeyError(f"unknown backbone {key!r}; known: {sorted(BACKBONES)}")

    kwargs = {"pretrained": True, "num_classes": 0}
    if img_size is not None:
        kwargs["img_size"] = img_size

    model = timm.create_model(BACKBONES[key], **kwargs)
    data_config = timm.data.resolve_model_data_config(model)

    if img_size is not None:
        # resolve_model_data_config reports the checkpoint's native resolution,
        # not the one the model was rebuilt at, so images would arrive at the
        # wrong size and the patch embedding would reject them. DINOv2 in
        # particular is published at 518 and used here at 224.
        data_config["input_size"] = (data_config["input_size"][0], img_size, img_size)

    model = _replace_attention(model).to(device).eval()
    return model, data_config


def token_features(
    model: nn.Module,
    images: torch.Tensor,
    tokens: str = "patch",
) -> torch.Tensor:
    """Final-block token embeddings, [B, L, D].

    Args:
        tokens: ``patch`` drops the class and register tokens, ``cls`` keeps only
            the class token (as a length-1 sequence), ``all`` keeps everything.
    """
    features = model.forward_features(images)
    num_prefix = getattr(model, "num_prefix_tokens", 1)

    if tokens == "patch":
        return features[:, num_prefix:]
    if tokens == "cls":
        return features[:, :1]
    if tokens == "all":
        return features
    raise ValueError(f"unknown token selection {tokens!r}")


# --------------------------------------------------------------------------- #
# Structured pruning
# --------------------------------------------------------------------------- #


@dataclass
class PruningBudget:
    """How much to remove, as a fraction of the original width."""

    mlp_ratio: float
    head_ratio: float

    @property
    def label(self) -> str:
        return f"mlp{self.mlp_ratio:g}-head{self.head_ratio:g}"


#: The (MLP, head) budget pairs used by SnapViT and Cut-ViT, indexed by the
#: overall parameter sparsity they induce on a ViT-B.
BUDGETS: Dict[str, PruningBudget] = {
    "s0": PruningBudget(mlp_ratio=0.0, head_ratio=0.0),
    "s05": PruningBudget(mlp_ratio=0.075, head_ratio=0.0),
    "s10": PruningBudget(mlp_ratio=0.15, head_ratio=0.0),
    "s15": PruningBudget(mlp_ratio=0.20, head_ratio=0.05),
    "s20": PruningBudget(mlp_ratio=0.25, head_ratio=0.1),
    "s30": PruningBudget(mlp_ratio=0.35, head_ratio=0.2),
    "s40": PruningBudget(mlp_ratio=0.45, head_ratio=0.3),
    "s50": PruningBudget(mlp_ratio=0.55, head_ratio=0.4),
}


#: How a global unit budget is spread across blocks.
#:
#:   ``global``           rank all units together on their raw scores
#:   ``linear-decay``     scale block b by linspace(1.2, 0.8), the fixed
#:                        per-block multiplier the published pipeline starts from
#:   ``block-normalised`` divide each block by its own mean score, removing the
#:                        cross-block scale with no constant to choose
#:   ``uniform``          give every block the same ratio, ranking only within
ALLOCATIONS = ("global", "linear-decay", "block-normalised", "uniform")


class PrunableViT:
    """A ViT together with the importance scores used to prune it."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        min_hidden_ratio: float = 0.05,
        min_head_ratio: float = 0.2,
    ):
        self.model = model
        self.device = device
        self.min_hidden_ratio = min_hidden_ratio
        self.min_head_ratio = min_head_ratio

        block = model.blocks[0]
        self.embed_dim = block.attn.qkv.in_features
        self.default_num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim
        self.default_hidden_dim = block.mlp.fc1.out_features
        self.num_blocks = len(model.blocks)

        self.min_hidden_dim = max(1, int(self.default_hidden_dim * min_hidden_ratio))
        self.min_num_heads = max(1, int(self.default_num_heads * min_head_ratio))

    # -- importance --------------------------------------------------------- #

    @property
    def gradient_parameters(self) -> List[nn.Parameter]:
        """The two weight matrices whose gradients carry the importance signal.

        ``mlp.fc1`` rows correspond one-to-one with hidden units, and the value
        slice of ``attn.qkv`` splits cleanly into heads, so a squared-gradient
        reduction over each gives a per-structure score.
        """
        params = []
        for block in self.model.blocks:
            params.append(block.mlp.fc1.weight)
            params.append(block.attn.qkv.weight)
        return params

    def enable_importance_gradients(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.gradient_parameters:
            param.requires_grad_(True)
            param.grad = None

    def mlp_importance(self) -> List[torch.Tensor]:
        """Per-block [hidden_dim] importance: mean squared gradient of fc1 rows."""
        scores = []
        for block in self.model.blocks:
            grad = block.mlp.fc1.weight.grad
            if grad is None:
                raise RuntimeError("no gradient on mlp.fc1; run the saliency pass first")
            scores.append((grad**2).mean(dim=1))
        return scores

    def head_importance(self) -> List[torch.Tensor]:
        """Per-block [num_heads] importance from the value projection gradients."""
        scores = []
        for block in self.model.blocks:
            grad = block.attn.qkv.weight.grad
            if grad is None:
                raise RuntimeError("no gradient on attn.qkv; run the saliency pass first")
            num_heads = block.attn.num_heads
            value = grad.reshape(3, num_heads, self.head_dim, self.embed_dim)[2]
            scores.append((value**2).mean(dim=(1, 2)))
        return scores

    def importance_state(self) -> Dict[str, torch.Tensor]:
        """Importance scores as dense tensors, for saving and for re-use."""
        return {
            "mlp": torch.stack(self.mlp_importance()).detach().cpu(),
            "head": torch.stack(self.head_importance()).detach().cpu(),
        }

    # -- pruning ------------------------------------------------------------ #

    def prune(
        self,
        budget: PruningBudget,
        importance: Optional[Dict[str, torch.Tensor]] = None,
        allocation: str = "global",
    ) -> "PrunableViT":
        """Remove the least important units, in place.

        The budget is global: a fixed total number of hidden units and heads is
        removed across the whole network. How that total spreads over blocks is
        decided by ``allocation`` (see :data:`ALLOCATIONS`), and per-block floors
        keep every block alive.
        """
        state = importance if importance is not None else self.importance_state()

        hidden_dims = [block.mlp.fc1.out_features for block in self.model.blocks]
        head_counts = [block.attn.num_heads for block in self.model.blocks]

        mlp_keep = self._select(
            scores=[row[:width].to(self.device) for row, width in zip(state["mlp"], hidden_dims)],
            num_pruned=int(self.default_hidden_dim * budget.mlp_ratio * self.num_blocks),
            floor=self.min_hidden_dim,
            allocation=allocation,
        )
        head_keep = self._select(
            scores=[row[:width].to(self.device) for row, width in zip(state["head"], head_counts)],
            num_pruned=int(self.default_num_heads * budget.head_ratio * self.num_blocks),
            floor=self.min_num_heads,
            allocation=allocation,
        )

        for i in range(self.num_blocks):
            self._prune_mlp(i, mlp_keep[i])
            self._prune_attention(i, head_keep[i])

        return self

    def _rescale(self, scores: List[torch.Tensor], allocation: str) -> List[torch.Tensor]:
        """Put each block's scores on a comparable scale before ranking globally.

        Squared-gradient magnitudes differ across depth by orders of magnitude,
        so a raw global ranking is settled largely by which block carries the
        larger gradients rather than by which units matter within a block. The
        published pipeline handles this with a per-block multiplier decaying
        linearly with depth; ``block-normalised`` instead divides each block by
        its own mean, which removes the cross-block scale outright and leaves no
        constant to choose.
        """
        if allocation in ("global", "uniform"):
            return scores
        if allocation == "linear-decay":
            weights = torch.linspace(1.2, 0.8, self.num_blocks, device=scores[0].device)
            return [row * weights[i] for i, row in enumerate(scores)]
        if allocation == "block-normalised":
            return [row / row.mean().clamp_min(torch.finfo(row.dtype).tiny) for row in scores]
        raise ValueError(f"unknown allocation {allocation!r}; known: {sorted(ALLOCATIONS)}")

    def _select(
        self,
        scores: List[torch.Tensor],
        num_pruned: int,
        floor: int,
        allocation: str = "global",
    ) -> List[torch.Tensor]:
        """Rank units under the chosen allocation, subject to a per-block floor.

        The floor is applied as a protection mask rather than by overwriting
        scores with a sentinel: a sentinel makes the protected units tie with
        each other, and the tie is then broken arbitrarily by the sort, which
        silently permutes surviving units against the ranking that chose them.
        """
        if allocation == "uniform":
            # Every block loses the same fraction, so the ranking only ever
            # operates within a block. This is the reference that isolates how
            # much of the outcome is due to the depth profile at all.
            return self._select_uniform(scores, num_pruned, floor)

        ranked = self._rescale(scores, allocation)

        protected = []
        for row in ranked:
            mask = torch.zeros_like(row, dtype=torch.bool)
            mask[row.argsort(descending=True)[: min(floor, row.numel())]] = True
            protected.append(mask)

        flat_scores = torch.cat(ranked)
        flat_protected = torch.cat(protected)

        keep = torch.ones_like(flat_scores, dtype=torch.bool)
        if num_pruned > 0:
            candidates = (~flat_protected).nonzero(as_tuple=True)[0]
            order = candidates[flat_scores[candidates].argsort()]
            keep[order[:num_pruned]] = False

        out, offset = [], 0
        for row in ranked:
            out.append(keep[offset : offset + row.numel()])
            offset += row.numel()
        return out

    def _select_uniform(
        self, scores: List[torch.Tensor], num_pruned: int, floor: int
    ) -> List[torch.Tensor]:
        """Spread the budget equally over blocks, ranking only within each."""
        out = []
        remaining = num_pruned
        for i, row in enumerate(scores):
            share = min(
                remaining if i == len(scores) - 1 else round(num_pruned / len(scores)),
                max(0, row.numel() - floor),
                remaining,
            )
            keep = torch.ones_like(row, dtype=torch.bool)
            if share > 0:
                keep[row.argsort()[:share]] = False
            remaining -= share
            out.append(keep)
        return out

    def _prune_mlp(self, index: int, keep: torch.Tensor) -> None:
        """Drop the unselected hidden units, preserving the order of the rest.

        The MLP is equivariant to a joint permutation of fc1 rows and fc2
        columns, so keeping the original order costs nothing and makes the
        surviving units directly comparable to the unpruned model.
        """
        mlp = self.model.blocks[index].mlp
        mlp.fc1.weight.data = mlp.fc1.weight.data[keep]
        mlp.fc1.bias.data = mlp.fc1.bias.data[keep]
        mlp.fc2.weight.data = mlp.fc2.weight.data[:, keep]
        mlp.fc1.out_features = int(keep.sum())
        mlp.fc2.in_features = int(keep.sum())

    def _prune_attention(self, index: int, keep: torch.Tensor) -> None:
        attn = self.model.blocks[index].attn
        num_heads = attn.num_heads
        remaining = int(keep.sum())

        qkv_w = attn.qkv.weight.data.view(3, num_heads, self.head_dim, self.embed_dim)
        attn.qkv.weight.data = qkv_w[:, keep].reshape(-1, self.embed_dim)

        if attn.qkv.bias is not None:
            qkv_b = attn.qkv.bias.data.view(3, num_heads, self.head_dim)
            attn.qkv.bias.data = qkv_b[:, keep].reshape(-1)

        proj_w = attn.proj.weight.data.view(self.embed_dim, num_heads, self.head_dim)
        attn.proj.weight.data = proj_w[:, keep].reshape(self.embed_dim, -1)

        attn.num_heads = remaining
        attn.qkv.out_features = 3 * remaining * self.head_dim
        attn.proj.in_features = remaining * self.head_dim

    # -- accounting --------------------------------------------------------- #

    def widths(self) -> Dict[str, List[int]]:
        return {
            "hidden_dims": [block.mlp.fc1.out_features for block in self.model.blocks],
            "num_heads": [block.attn.num_heads for block in self.model.blocks],
        }

    def clone(self) -> "PrunableViT":
        return PrunableViT(
            model=copy.deepcopy(self.model),
            device=self.device,
            min_hidden_ratio=self.min_hidden_ratio,
            min_head_ratio=self.min_head_ratio,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def encoder_flops(model: nn.Module, num_tokens: int) -> int:
    """Multiply-accumulates in the transformer encoder, per image.

    Counts the four projections and the two attention matmuls per block. Patch
    embedding, normalisation and the head are excluded: they are untouched by
    pruning and identical across every model compared here.
    """
    total = 0
    for block in model.blocks:
        dim = block.attn.qkv.in_features
        heads = block.attn.num_heads
        head_dim = block.attn.head_dim
        inner = heads * head_dim
        hidden = block.mlp.fc1.out_features

        total += num_tokens * dim * 3 * inner  # qkv
        total += 2 * num_tokens * num_tokens * inner  # scores and weighted sum
        total += num_tokens * inner * dim  # output projection
        total += 2 * num_tokens * dim * hidden  # mlp
    return total

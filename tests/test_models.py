"""Checks on the pruning surgery.

A structurally pruned model must still run, must be smaller by the amount the
budget asks for, and must keep the exact units the importance ranking selected.
"""

import sys
from pathlib import Path

import pytest
import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp.models import (  # noqa: E402
    BUDGETS,
    DynamicHeadsAttention,
    PruningBudget,
    PrunableViT,
    count_parameters,
    encoder_flops,
    token_features,
)


@pytest.fixture
def tiny_vit():
    """A randomly initialised ViT small enough to build without a download."""
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=64,
        embed_dim=48,
        depth=3,
        num_heads=4,
    )
    for block in model.blocks:
        block.attn = DynamicHeadsAttention.from_timm(block.attn)
    return model.eval()


@pytest.fixture
def prunable(tiny_vit):
    return PrunableViT(tiny_vit, device=torch.device("cpu"), min_head_ratio=0.25)


def _images():
    torch.manual_seed(0)
    return torch.randn(2, 3, 64, 64)


def test_attention_wrapper_preserves_the_forward_pass(tiny_vit):
    reference = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=64,
        embed_dim=48,
        depth=3,
        num_heads=4,
    ).eval()
    reference.load_state_dict(tiny_vit.state_dict())

    with torch.no_grad():
        images = _images()
        assert torch.allclose(
            tiny_vit.forward_features(images), reference.forward_features(images), atol=1e-5
        )


def test_token_selection(tiny_vit):
    images = _images()
    with torch.no_grad():
        num_prefix = tiny_vit.num_prefix_tokens
        assert token_features(tiny_vit, images, "all").shape[1] == 16 + num_prefix
        assert token_features(tiny_vit, images, "patch").shape[1] == 16
        assert token_features(tiny_vit, images, "cls").shape[1] == 1


def test_pruning_removes_the_requested_units(prunable):
    hidden = prunable.default_hidden_dim
    heads = prunable.default_num_heads
    blocks = prunable.num_blocks

    importance = {
        "mlp": torch.rand(blocks, hidden),
        "head": torch.rand(blocks, heads),
    }
    budget = PruningBudget(mlp_ratio=0.25, head_ratio=0.25)
    prunable.prune(budget, importance=importance)

    widths = prunable.widths()
    assert sum(widths["hidden_dims"]) == blocks * hidden - int(hidden * 0.25 * blocks)
    assert sum(widths["num_heads"]) == blocks * heads - int(heads * 0.25 * blocks)


def test_pruned_model_still_runs(prunable):
    importance = {
        "mlp": torch.rand(prunable.num_blocks, prunable.default_hidden_dim),
        "head": torch.rand(prunable.num_blocks, prunable.default_num_heads),
    }
    prunable.prune(PruningBudget(0.35, 0.25), importance=importance)
    with torch.no_grad():
        out = prunable.model.forward_features(_images())
    assert torch.isfinite(out).all()
    assert out.shape[-1] == prunable.embed_dim


def test_pruning_keeps_the_highest_scoring_units(prunable):
    """Surviving fc1 rows are exactly the units above the global threshold.

    With a floor of 5% of 4x48 = 9 units per block and a 50% budget, the floor
    does not bind here, so the selection is purely the global ranking.
    """
    blocks, hidden = prunable.num_blocks, prunable.default_hidden_dim
    torch.manual_seed(3)
    scores = torch.rand(blocks, hidden)
    reference = [block.mlp.fc1.weight.data.clone() for block in prunable.model.blocks]

    num_pruned = int(hidden * 0.5 * blocks)
    threshold = scores.flatten().sort().values[num_pruned - 1]

    prunable.prune(PruningBudget(0.5, 0.0), importance={
        "mlp": scores, "head": torch.rand(blocks, prunable.default_num_heads)
    })

    for i, block in enumerate(prunable.model.blocks):
        keep = scores[i] > threshold
        assert block.mlp.fc1.out_features == int(keep.sum())
        assert torch.allclose(block.mlp.fc1.weight.data, reference[i][keep])


def test_per_block_floors_are_respected(prunable):
    """A block whose units all score low must still keep its floor."""
    blocks, hidden = prunable.num_blocks, prunable.default_hidden_dim
    scores = torch.ones(blocks, hidden)
    scores[0] = 1e-9  # first block looks worthless

    prunable.prune(PruningBudget(0.5, 0.0), importance={
        "mlp": scores, "head": torch.rand(blocks, prunable.default_num_heads)
    })

    assert prunable.model.blocks[0].mlp.fc1.out_features >= prunable.min_hidden_dim


def test_zero_budget_is_a_no_op(prunable):
    before = count_parameters(prunable.model)
    prunable.prune(BUDGETS["s0"], importance={
        "mlp": torch.rand(prunable.num_blocks, prunable.default_hidden_dim),
        "head": torch.rand(prunable.num_blocks, prunable.default_num_heads),
    })
    assert count_parameters(prunable.model) == before


def test_flops_fall_with_pruning(prunable):
    before = encoder_flops(prunable.model, num_tokens=17)
    prunable.prune(PruningBudget(0.5, 0.5), importance={
        "mlp": torch.rand(prunable.num_blocks, prunable.default_hidden_dim),
        "head": torch.rand(prunable.num_blocks, prunable.default_num_heads),
    })
    assert encoder_flops(prunable.model, num_tokens=17) < before


def test_importance_needs_a_backward_pass(prunable):
    prunable.enable_importance_gradients()
    with pytest.raises(RuntimeError):
        prunable.mlp_importance()


def test_importance_has_the_expected_shape(prunable):
    prunable.enable_importance_gradients()
    features = prunable.model.forward_features(_images())
    features.square().mean().backward()

    state = prunable.importance_state()
    assert state["mlp"].shape == (prunable.num_blocks, prunable.default_hidden_dim)
    assert state["head"].shape == (prunable.num_blocks, prunable.default_num_heads)
    assert torch.isfinite(state["mlp"]).all() and (state["mlp"] >= 0).all()


def test_every_allocation_meets_the_budget(prunable):
    """Whatever the allocation, the total removed must match the budget."""
    from mp.models import ALLOCATIONS

    blocks, hidden = prunable.num_blocks, prunable.default_hidden_dim
    torch.manual_seed(11)
    # Gradient scales differ across depth by orders of magnitude, as they do in
    # a real network; this is exactly what the allocation has to cope with.
    scores = torch.rand(blocks, hidden) * torch.tensor([1e-8, 1.0, 1e3]).unsqueeze(1)

    for allocation in ALLOCATIONS:
        model = prunable.clone()
        model.prune(
            PruningBudget(0.25, 0.0),
            importance={"mlp": scores.clone(),
                        "head": torch.rand(blocks, model.default_num_heads)},
            allocation=allocation,
        )
        widths = model.widths()["hidden_dims"]
        assert sum(widths) == blocks * hidden - int(hidden * 0.25 * blocks), allocation
        assert min(widths) >= model.min_hidden_dim, allocation


def test_block_normalisation_removes_the_cross_block_scale(prunable):
    """A block whose gradients are uniformly tiny must not be gutted for it.

    Under a raw global ranking every unit of the small-scale block sorts below
    every unit of the others, so that block is pruned to its floor whatever its
    units are worth relative to each other.
    """
    blocks, hidden = prunable.num_blocks, prunable.default_hidden_dim
    torch.manual_seed(5)
    scores = torch.rand(blocks, hidden)
    scores[0] *= 1e-9  # same information, far smaller gradients

    # Large enough that the raw ranking exhausts the small-scale block first.
    budget = PruningBudget(0.34, 0.0)

    raw = prunable.clone()
    raw.prune(budget, importance={
        "mlp": scores.clone(), "head": torch.rand(blocks, raw.default_num_heads)
    }, allocation="global")

    normalised = prunable.clone()
    normalised.prune(budget, importance={
        "mlp": scores.clone(), "head": torch.rand(blocks, normalised.default_num_heads)
    }, allocation="block-normalised")

    assert raw.widths()["hidden_dims"][0] == raw.min_hidden_dim
    assert normalised.widths()["hidden_dims"][0] > raw.widths()["hidden_dims"][0]


def test_uniform_allocation_is_flat(prunable):
    blocks, hidden = prunable.num_blocks, prunable.default_hidden_dim
    prunable.prune(PruningBudget(0.25, 0.0), importance={
        "mlp": torch.rand(blocks, hidden),
        "head": torch.rand(blocks, prunable.default_num_heads),
    }, allocation="uniform")
    widths = prunable.widths()["hidden_dims"]
    assert max(widths) - min(widths) <= 1


def test_unknown_allocation_is_rejected(prunable):
    with pytest.raises(ValueError, match="unknown allocation"):
        prunable.prune(PruningBudget(0.25, 0.0), importance={
            "mlp": torch.rand(prunable.num_blocks, prunable.default_hidden_dim),
            "head": torch.rand(prunable.num_blocks, prunable.default_num_heads),
        }, allocation="nonsense")

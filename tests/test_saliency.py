"""Checks on the saliency pass and the view protocols.

The central fact these tests pin down is that the unpruned network is an exact
global minimum of every alignment objective, so a calibration pass carries
signal only when the teacher and student inputs differ.
"""

import sys
from pathlib import Path

import pytest
import timm
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp.data import (  # noqa: E402
    VIEW_PROTOCOLS,
    IdenticalViews,
    NoisyViews,
    PairedViewDataset,
    TwoCropViews,
    build_views,
    make_loader,
)
from mp.models import DynamicHeadsAttention, PrunableViT  # noqa: E402
from mp.objectives import build_objective  # noqa: E402
from mp.saliency import estimate_importance, make_teacher  # noqa: E402

DATA_CONFIG = {"input_size": (3, 64, 64), "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
               "crop_pct": 1.0}


class _Images(Dataset):
    """A handful of deterministic PIL images."""

    def __init__(self, n: int = 4):
        from PIL import Image

        torch.manual_seed(0)
        self.images = [
            Image.fromarray(
                (torch.rand(96, 96, 3) * 255).to(torch.uint8).numpy(), mode="RGB"
            )
            for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i]


@pytest.fixture
def backbone():
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=64,
        embed_dim=48,
        depth=2,
        num_heads=4,
    )
    for block in model.blocks:
        block.attn = DynamicHeadsAttention.from_timm(block.attn)
    return model.eval()


def _loader(protocol: str):
    return make_loader(
        PairedViewDataset(_Images(), build_views(protocol, DATA_CONFIG)),
        batch_size=2,
        num_workers=0,
    )


def test_every_protocol_is_registered():
    assert set(VIEW_PROTOCOLS) == {"two-crop", "noise", "identical"}
    assert isinstance(build_views("two-crop", DATA_CONFIG), TwoCropViews)
    assert isinstance(build_views("noise", DATA_CONFIG), NoisyViews)
    assert isinstance(build_views("identical", DATA_CONFIG), IdenticalViews)


def test_views_have_the_model_resolution():
    for protocol in VIEW_PROTOCOLS:
        teacher_view, student_view = build_views(protocol, DATA_CONFIG)(_Images()[0])
        assert teacher_view.shape == (3, 64, 64), protocol
        assert student_view.shape == (3, 64, 64), protocol


def test_identical_views_agree_and_the_others_do_not():
    image = _Images()[0]
    same_t, same_s = IdenticalViews(DATA_CONFIG)(image)
    assert torch.equal(same_t, same_s)

    for protocol in ("two-crop", "noise"):
        teacher_view, student_view = build_views(protocol, DATA_CONFIG)(image)
        assert not torch.allclose(teacher_view, student_view), protocol


def test_identical_views_give_a_vanishing_gradient(backbone):
    """The unpruned model is an exact minimum, so this pass yields no signal.

    This is the configuration in which a saliency ranking degenerates into
    whatever floating-point discrepancy separates the graph-free teacher pass
    from the graph-building student pass.
    """
    prunable = PrunableViT(backbone, device=torch.device("cpu"))
    teacher = make_teacher(backbone)

    report = estimate_importance(
        prunable,
        teacher=teacher,
        objective=build_objective("cosine"),
        loader=_loader("identical"),
        device=torch.device("cpu"),
        progress=False,
    )

    assert report.loss_terms["total"] < 1e-6
    assert float(prunable.importance_state()["mlp"].max()) < 1e-16


@pytest.mark.parametrize("protocol", ["two-crop", "noise"])
def test_differing_views_give_a_usable_gradient(backbone, protocol):
    prunable = PrunableViT(backbone, device=torch.device("cpu"))
    teacher = make_teacher(backbone)

    report = estimate_importance(
        prunable,
        teacher=teacher,
        objective=build_objective("cosine"),
        loader=_loader(protocol),
        device=torch.device("cpu"),
        progress=False,
    )

    scores = prunable.importance_state()
    assert report.loss_terms["total"] > 1e-4
    assert torch.isfinite(scores["mlp"]).all()
    assert float(scores["mlp"].max()) > 0
    # A usable ranking needs the scores to actually differ from one another.
    assert float(scores["mlp"].std()) > 0


@pytest.mark.parametrize(
    "objective", ["cosine", "mse", "gram-a1", "gram-a2", "gram-a1-k192", "cutvit"]
)
def test_objectives_produce_finite_importance(backbone, objective):
    prunable = PrunableViT(backbone, device=torch.device("cpu"))
    teacher = make_teacher(backbone)

    estimate_importance(
        prunable,
        teacher=teacher,
        objective=build_objective(objective, rank=8),
        loader=_loader("two-crop"),
        device=torch.device("cpu"),
        progress=False,
    )

    scores = prunable.importance_state()
    assert torch.isfinite(scores["mlp"]).all(), objective
    assert torch.isfinite(scores["head"]).all(), objective
    assert float(scores["mlp"].std()) > 0, objective


def test_report_records_cost(backbone):
    prunable = PrunableViT(backbone, device=torch.device("cpu"))
    report = estimate_importance(
        prunable,
        teacher=make_teacher(backbone),
        objective=build_objective("gram-a1"),
        loader=_loader("two-crop"),
        device=torch.device("cpu"),
        progress=False,
    )
    assert report.seconds > 0
    assert report.num_batches == 2
    assert "spectral_spatial" in report.loss_terms


def _ill_conditioned(n: int = 3, length: int = 48, dim: int = 32, rank: int = 4):
    """Features with the spectrum shape real transformer features have.

    A few dozen directions carry the energy and the rest of the spectrum is a
    cloud of near-equal near-zero eigenvalues, which is exactly the input that
    makes an eigendecomposition backward return NaN.
    """
    torch.manual_seed(0)
    basis = torch.randn(n, length, rank)
    scale = torch.exp(-torch.arange(rank).float())
    loads = torch.randn(n, rank, dim) * scale.unsqueeze(0).unsqueeze(-1)
    return basis @ loads + 1e-4 * torch.randn(n, length, dim)


@pytest.mark.parametrize(
    "objective",
    ["cosine", "mse", "cutvit", "cutvit-scaled", "gram-a0.25", "gram-a0.5",
     "gram-a1", "gram-a2", "gram-a1-only", "gram-a1-pooled"],
)
def test_objective_gradients_survive_a_collapsed_spectrum(objective):
    """Every configured objective must give finite gradients on such features."""
    from mp.objectives import build_objective as build

    teacher = _ill_conditioned()
    student = (teacher + 0.05 * torch.randn_like(teacher)).requires_grad_(True)

    config = build(objective).config
    overrides = {"rank": 8} if config.rank is not None else {}
    loss, _ = build(objective, **overrides)(teacher, student)
    loss.backward()

    assert torch.isfinite(loss), objective
    assert torch.isfinite(student.grad).all(), objective
    assert float(student.grad.abs().sum()) > 0, objective


def test_matrix_sqrt_matches_the_eigendecomposition():
    from mp.objectives import gram, matrix_sqrt, top_eigenbasis

    g = gram(_ill_conditioned().double(), "channel")
    approx = matrix_sqrt(g, iterations=20)
    values, vectors = top_eigenbasis(g, g.shape[-1])
    exact = (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    assert torch.allclose(approx, exact, atol=1e-4), (approx - exact).abs().max()


def test_unsupported_exponents_are_refused():
    from mp.objectives import gram, spectral_power

    g = gram(_ill_conditioned(), "channel")
    with pytest.raises(ValueError, match="matmul route"):
        spectral_power(g, alpha=0.3, k=None)
    with pytest.raises(ValueError, match="unbounded"):
        spectral_power(g, alpha=0.5, k=8)

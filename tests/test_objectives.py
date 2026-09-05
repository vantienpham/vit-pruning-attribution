"""Checks on the claims the paper makes about the objective family.

The equivalences asserted here are stated as propositions in the manuscript, so
they are verified rather than assumed.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp.objectives import (  # noqa: E402
    OBJECTIVES,
    AlignmentObjective,
    ObjectiveConfig,
    chordal_subspace_loss,
    cumulative_explained_variance,
    gram,
    pointwise_cosine_loss,
    residual_loss,
    spectral_alignment_loss,
    spectral_entropy,
    spectral_power,
    top_eigenbasis,
)


@pytest.fixture
def features():
    torch.manual_seed(0)
    # Low-rank plus noise, so the spectra are concentrated the way real
    # transformer features are rather than flat.
    basis = torch.randn(4, 32, 8)
    loadings = torch.randn(4, 8, 24)
    signal = basis @ loadings
    return signal + 0.05 * torch.randn(4, 32, 24)


def test_gram_shapes(features):
    assert gram(features, "spatial").shape == (4, 32, 32)
    assert gram(features, "channel").shape == (4, 24, 24)
    assert gram(features, "spatial", pooling="batch_pooled").shape == (32, 32)
    assert gram(features, "channel", pooling="batch_pooled").shape == (24, 24)


def test_gram_is_psd(features):
    values, _ = top_eigenbasis(gram(features, "spatial"), 32)
    assert values.min() >= -1e-5


def test_alpha_zero_is_the_chordal_subspace_loss(features):
    """Proposition: L(alpha=0, K) is Cut-ViT's basis-agnostic constraint.

    phi(G; 0, K) = U_K U_K^T is a projector with ||P||_F^2 = K, so the
    normalised inner product reduces to ||U_p^T U_t||_F^2 / K.
    """
    other = features + 0.3 * torch.randn_like(features)
    g_t = gram(features, "spatial").double()
    g_s = gram(other, "spatial").double()

    for k in (4, 8, 16):
        family = spectral_alignment_loss(g_t, g_s, alpha=0.0, k=k)
        chordal = chordal_subspace_loss(g_t, g_s, k=k)
        assert torch.allclose(family, chordal, atol=1e-8), (k, family, chordal)


def test_chordal_loss_equals_projector_distance(features):
    """L_basis = ||P_p - P_t||_F^2 / (2K), the squared chordal distance."""
    other = features + 0.3 * torch.randn_like(features)
    g_t = gram(features, "spatial").double()
    g_s = gram(other, "spatial").double()
    k = 8

    _, u_t = top_eigenbasis(g_t, k)
    _, u_s = top_eigenbasis(g_s, k)
    p_t = u_t @ u_t.transpose(-1, -2)
    p_s = u_s @ u_s.transpose(-1, -2)
    projector_distance = ((p_s - p_t) ** 2).sum(dim=(-2, -1)).mean() / (2 * k)

    assert torch.allclose(chordal_subspace_loss(g_t, g_s, k=k), projector_distance, atol=1e-8)


def test_alpha_one_is_gram_cosine(features):
    """L(alpha=1, full rank) is the cosine between the Gram matrices."""
    other = features + 0.3 * torch.randn_like(features)
    g_t = gram(features, "channel").double()
    g_s = gram(other, "channel").double()

    inner = (g_t * g_s).sum(dim=(-2, -1))
    norms = g_t.norm(dim=(-2, -1)) * g_s.norm(dim=(-2, -1))
    expected = 1.0 - (inner / norms).mean()

    assert torch.allclose(spectral_alignment_loss(g_t, g_s, alpha=1.0), expected, atol=1e-10)


def test_integer_alpha_matches_the_eigendecomposition(features):
    """The matmul fast path for integer alpha agrees with the spectral form."""
    g = gram(features, "channel").double()
    fast = spectral_power(g, alpha=2.0, k=None)
    values, vectors = top_eigenbasis(g, g.shape[-1])
    slow = (vectors * (values**2).unsqueeze(-2)) @ vectors.transpose(-1, -2)
    assert torch.allclose(fast, slow, atol=1e-6)


def test_every_member_is_invariant_to_basis_rotation(features):
    """Basis invariance holds for the family, not only for the alpha=0 member.

    Rotating the student features by an orthogonal matrix acting on the
    contracted axis leaves the Gram matrix, hence every member, unchanged.
    """
    torch.manual_seed(1)
    q, _ = torch.linalg.qr(torch.randn(24, 24, dtype=torch.float64))
    x = features.double()
    rotated = x @ q

    g_t = gram(x, "spatial")
    for alpha, k in [(0.0, 8), (0.5, None), (1.0, None), (2.0, None)]:
        plain = spectral_alignment_loss(g_t, gram(x, "spatial"), alpha=alpha, k=k)
        turned = spectral_alignment_loss(g_t, gram(rotated, "spatial"), alpha=alpha, k=k)
        assert torch.allclose(plain, turned, atol=1e-8), alpha


def test_losses_vanish_on_identical_features(features):
    g = gram(features, "spatial").double()
    for alpha, k in [(0.0, 8), (0.5, None), (1.0, None), (2.0, None)]:
        assert abs(float(spectral_alignment_loss(g, g, alpha=alpha, k=k))) < 1e-8
    assert abs(float(pointwise_cosine_loss(features, features))) < 1e-6
    assert float(residual_loss(features.double(), g, k=32, axis="spatial")) < 1e-8


def test_losses_increase_with_perturbation(features):
    g_t = gram(features, "spatial").double()
    previous = {}
    for scale in (0.1, 0.4, 1.2):
        torch.manual_seed(7)
        noisy = features + scale * torch.randn_like(features)
        g_s = gram(noisy, "spatial").double()
        for alpha, k in [(0.0, 8), (1.0, None)]:
            value = float(spectral_alignment_loss(g_t, g_s, alpha=alpha, k=k))
            key = (alpha, k)
            if key in previous:
                assert value > previous[key], (key, scale)
            previous[key] = value


def test_residual_is_a_bounded_fraction(features):
    g_t = gram(features, "spatial").double()
    value = float(residual_loss(features.double(), g_t, k=4, axis="spatial"))
    assert 0.0 <= value <= 1.0


def test_spectral_entropy_bounds(features):
    g = gram(features, "spatial")
    entropy = float(spectral_entropy(g))
    assert 0.0 <= entropy <= torch.log(torch.tensor(32.0)) + 1e-5

    flat = torch.eye(16).expand(2, 16, 16)
    assert abs(float(spectral_entropy(flat)) - float(torch.log(torch.tensor(16.0)))) < 1e-4


def test_cevr_is_monotone_and_reaches_one(features):
    g = gram(features, "spatial").double()
    values = [float(cumulative_explained_variance(g, k)) for k in (1, 4, 16, 32)]
    assert values == sorted(values)
    assert abs(values[-1] - 1.0) < 1e-6


def test_objectives_are_differentiable(features):
    """Each configured objective must produce a finite gradient."""
    for name, config in OBJECTIVES.items():
        if config.rank is not None and config.rank > 24:
            config = ObjectiveConfig(**{**config.to_dict(), "rank": 8})
            config.axes = tuple(config.axes)
        student = features.clone().requires_grad_(True)
        loss, terms = AlignmentObjective(config)(features, student)
        loss.backward()
        assert torch.isfinite(loss), name
        assert student.grad is not None and torch.isfinite(student.grad).all(), name
        assert terms, name


def test_entropy_weighting_sums_to_one(features):
    config = ObjectiveConfig(
        name="t", axes=("spatial", "channel"), alpha=1.0, entropy_weighting=True
    )
    objective = AlignmentObjective(config)
    grams = {axis: gram(features, axis) for axis in config.axes}
    weights = objective._axis_weights(grams)
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_spatial_and_channel_entropy_coincide(features):
    """Proposition: the two Gram matrices share a non-zero spectrum.

    F F^T and F^T F are U S^2 U^T and V S^2 V^T for the same S, so any
    statistic of the normalised spectrum is identical for the two, and an axis
    weighting built from their entropies is the constant 1/2.
    """
    x = features.double()
    for i in range(x.shape[0]):
        single = x[i : i + 1]
        h_spatial = float(spectral_entropy(gram(single, "spatial")))
        h_channel = float(spectral_entropy(gram(single, "channel")))
        assert abs(h_spatial - h_channel) < 1e-9, (i, h_spatial, h_channel)

    weight = h_spatial / (h_spatial + h_channel)
    assert abs(weight - 0.5) < 1e-9


def test_batch_pooling_breaks_the_entropy_identity(features):
    """Pooling over a batch sums the per-image Grams, and sums do not agree."""
    x = features.double()
    h_spatial = float(spectral_entropy(gram(x, "spatial", pooling="batch_pooled")))
    h_channel = float(spectral_entropy(gram(x, "channel", pooling="batch_pooled")))
    assert abs(h_spatial - h_channel) > 1e-6

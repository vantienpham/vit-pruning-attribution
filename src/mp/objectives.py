"""Teacher-student alignment objectives for training-free structured pruning.

The saliency signal used by one-shot structured pruning is the empirical Fisher
of an alignment loss L(F_t, F_p) between the frozen teacher features F_t and the
features F_p of the model being pruned. Everything in this module is a candidate
for that L.

The organising idea is that the second-order objectives in this literature are
all functions of a pair of Gram matrices, and differ only in how they reweight
the teacher's spectrum. Writing

    phi(G; alpha, K) = U_K diag(sigma_K ** alpha) U_K^T

for the rank-K spectral power of a PSD Gram matrix G = U diag(sigma) U^T, the
family

    L(alpha, K) = 1 - <phi(G_t), phi(G_p)>_F / (||phi(G_t)||_F ||phi(G_p)||_F)

contains, as special cases:

  * (alpha=0, K)      the normalised squared chordal distance on the Grassmannian
                      Gr(K, .), which is exactly the "basis-agnostic" subspace
                      loss of Cut-ViT: phi(G; 0, K) = U_K U_K^T is a projector,
                      ||P||_F^2 = K, so L = 1 - ||U_p^T U_t||_F^2 / K.
  * (alpha=1, K=full) the cosine between the Gram matrices themselves, i.e.
                      linear CKA on uncentred features. Needs no eigendecomposition.
  * (alpha=2, K=full) the same with squared spectral weighting, also
                      eigendecomposition-free (G^2 = G @ G).

Every member is invariant to orthogonal reparameterisation of the eigenbasis,
because phi(G) is built from G and not from a choice of basis, so basis
invariance is a property of the whole family rather than of one member.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

GramAxis = Literal["spatial", "channel"]
GramPooling = Literal["per_image", "batch_pooled"]

# Eigenvalues below this fraction of the largest are treated as numerically zero.
# Fractional spectral powers are unstable near zero (d/dx x^alpha -> infinity),
# and the tail of a Gram spectrum is dominated by round-off.
EIG_FLOOR = 1e-6


# --------------------------------------------------------------------------- #
# Gram matrices
# --------------------------------------------------------------------------- #


def gram(
    features: torch.Tensor,
    axis: GramAxis,
    pooling: GramPooling = "per_image",
    normalise: bool = True,
) -> torch.Tensor:
    """Second-order feature correlations along one axis.

    Args:
        features: [B, L, D] token embeddings.
        axis: ``spatial`` gives token-token correlations (L x L, "where"),
            ``channel`` gives channel-channel correlations (D x D, "what").
        pooling: ``per_image`` forms one Gram matrix per image and returns a
            batch of them; ``batch_pooled`` folds the batch into the contracted
            axis and returns a single matrix. The released Cut-ViT code pools,
            which makes the spatial Gram a correlation between token *positions*
            across images rather than the layout of one image; the paper's
            Eq. (3) describes the per-image form. Both are provided so the
            difference can be measured.
        normalise: divide by the number of contracted elements, which keeps the
            scale comparable across axes and batch sizes.

    Returns:
        [B, N, N] if ``pooling='per_image'`` else [N, N].
    """
    if features.dim() != 3:
        raise ValueError(f"expected [B, L, D] features, got {tuple(features.shape)}")

    b, length, dim = features.shape

    if pooling == "per_image":
        x = features if axis == "spatial" else features.transpose(1, 2)
        g = x @ x.transpose(1, 2)
        contracted = dim if axis == "spatial" else length
    elif pooling == "batch_pooled":
        if axis == "spatial":
            # [L, B * D]: token position i against token position j, pooled over images.
            x = features.permute(1, 0, 2).reshape(length, b * dim)
            contracted = b * dim
        else:
            # [D, B * L]
            x = features.permute(2, 0, 1).reshape(dim, b * length)
            contracted = b * length
        g = x @ x.transpose(0, 1)
    else:
        raise ValueError(f"unknown pooling {pooling!r}")

    return g / contracted if normalise else g


# --------------------------------------------------------------------------- #
# Spectral machinery
# --------------------------------------------------------------------------- #


def _safe_eigh(g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigendecomposition with a CPU/float64 fallback.

    cuSOLVER fails to converge on ill-conditioned input on the older cards in
    this cluster, and torch's own internal fallback does not always catch it.
    LAPACK in float64 uses a different algorithm and succeeds. Ascending order,
    as returned by ``eigh``.
    """
    g = 0.5 * (g + g.transpose(-1, -2))
    try:
        values, vectors = torch.linalg.eigh(g)
        if torch.isfinite(values).all() and torch.isfinite(vectors).all():
            return values, vectors
    except (torch.linalg.LinAlgError, RuntimeError):
        pass

    values, vectors = torch.linalg.eigh(g.detach().double().cpu())
    return values.to(g.dtype).to(g.device), vectors.to(g.dtype).to(g.device)


def top_eigenbasis(g: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k eigenvalues (descending) and eigenvectors of a PSD Gram matrix.

    Returns:
        values [..., k], vectors [..., N, k].
    """
    values, vectors = _safe_eigh(g)
    k = min(k, values.shape[-1])
    values = values[..., -k:].flip(-1).clamp_min(0.0)
    vectors = vectors[..., -k:].flip(-1)
    return values, vectors


def matrix_sqrt(g: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """PSD matrix square root by Newton--Schulz iteration.

    Built entirely from matrix products, so the backward pass is a product of
    matrix products. Going through an eigendecomposition instead is what makes
    fractional exponents fail: the derivative of the eigenvectors carries
    :math:`1/(\\lambda_i - \\lambda_j)` terms, and a transformer Gram matrix has
    an effective rank of a few dozen out of several hundred, so most of its
    spectrum is a cloud of near-equal near-zero eigenvalues. The forward pass
    looks fine and the gradient comes back NaN.
    """
    scale = g.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    scale = scale.reshape(*scale.shape, 1, 1)
    y = g / scale
    identity = torch.eye(g.shape[-1], device=g.device, dtype=g.dtype).expand_as(g)
    z = identity.clone()

    for _ in range(iterations):
        t = 0.5 * (3.0 * identity - z @ y)
        y = y @ t
        z = t @ z

    return y * scale.sqrt()


def spectral_power(
    g: torch.Tensor,
    alpha: float,
    k: Optional[int] = None,
) -> torch.Tensor:
    """phi(G; alpha, K) = U_K diag(sigma_K ** alpha) U_K^T.

    Three routes, chosen so that nothing ill-conditioned enters the graph:
    integer exponents at full rank are repeated multiplication; halves and
    quarters at full rank are Newton--Schulz square roots; and the truncated
    cases go through an eigendecomposition, which is safe for the projector
    (alpha=0 depends only on the subspace, so degeneracies inside it cancel)
    and for exponents at least one (the weight derivative is bounded).
    """
    full_rank = k is None or k >= g.shape[-1]

    if full_rank:
        if float(alpha).is_integer() and alpha >= 1:
            out = g
            for _ in range(int(alpha) - 1):
                out = out @ g
            return out
        if alpha == 0.5:
            return matrix_sqrt(g)
        if alpha == 0.25:
            return matrix_sqrt(matrix_sqrt(g))
        if 0.0 < alpha < 1.0:
            raise ValueError(
                f"alpha={alpha} at full rank has no matmul route; the "
                "eigendecomposition path returns NaN gradients on these spectra"
            )

    rank = g.shape[-1] if k is None else min(k, g.shape[-1])
    values, vectors = top_eigenbasis(g, rank)
    floor = EIG_FLOOR * values[..., :1].clamp_min(torch.finfo(values.dtype).tiny)

    if alpha == 0.0:
        # Projector onto the leading rank-K eigenspace. Directions whose
        # eigenvalue is numerically zero carry no subspace and are dropped.
        weights = (values > floor).to(g.dtype)
    elif alpha >= 1.0:
        weights = values.clamp_min(floor) ** alpha
    else:
        raise ValueError(
            f"alpha={alpha} with rank {k} would differentiate x**alpha at the "
            "spectrum floor, where the derivative is unbounded"
        )

    return (vectors * weights.unsqueeze(-2)) @ vectors.transpose(-1, -2)


def _matrix_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """<A, B>_F / (||A||_F ||B||_F), averaged over any leading batch dimensions."""
    dims = (-2, -1)
    inner = (a * b).sum(dim=dims)
    norms = a.norm(dim=dims) * b.norm(dim=dims)
    return (inner / norms.clamp_min(1e-12)).mean()


# --------------------------------------------------------------------------- #
# Alignment losses
# --------------------------------------------------------------------------- #


def spectral_alignment_loss(
    g_teacher: torch.Tensor,
    g_student: torch.Tensor,
    alpha: float,
    k: Optional[int] = None,
) -> torch.Tensor:
    """The (alpha, K) member of the family. Zero when the two agree."""
    if alpha == 1.0 and (k is None or k >= g_teacher.shape[-1]):
        # Fast path: no eigendecomposition, no matrix power.
        return 1.0 - _matrix_cosine(g_teacher, g_student)

    phi_t = spectral_power(g_teacher, alpha=alpha, k=k)
    phi_s = spectral_power(g_student, alpha=alpha, k=k)
    return 1.0 - _matrix_cosine(phi_t, phi_s)


def chordal_subspace_loss(
    g_teacher: torch.Tensor,
    g_student: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Cut-ViT's basis-agnostic constraint, written in its original form.

        L = 1 - ||U_p^T U_t||_F^2 / K = ||P_p - P_t||_F^2 / (2K)

    Identical to ``spectral_alignment_loss(..., alpha=0, k=K)`` whenever both
    Gram matrices have rank at least K; kept separate so the equivalence can be
    asserted in the tests rather than assumed.
    """
    _, u_t = top_eigenbasis(g_teacher, k)
    _, u_s = top_eigenbasis(g_student, k)
    m = u_s.transpose(-1, -2) @ u_t
    overlap = (m**2).sum(dim=(-2, -1))
    return (1.0 - overlap / k).mean()


def residual_loss(
    features_student: torch.Tensor,
    g_teacher: torch.Tensor,
    k: int,
    axis: GramAxis,
) -> torch.Tensor:
    """Energy of the student features outside the teacher's leading subspace.

        L = ||(I - U_t U_t^T) F_p||_F^2 / ||F_p||_F^2

    Cut-ViT's Eq. (9) without the normaliser; the ratio form is used here so the
    term is scale-free and comparable across axes and models. The released
    implementation divides an already-averaged square by the element count a
    second time, which shrinks the term by the number of elements and leaves it
    numerically inert; that deviation is reported rather than reproduced.
    """
    _, u_t = top_eigenbasis(g_teacher, k)

    # Put the axis being projected first: spatial projects tokens, channel
    # projects channels.
    x = features_student if axis == "spatial" else features_student.transpose(1, 2)

    if u_t.dim() == 2:  # batch-pooled teacher basis
        u_t = u_t.unsqueeze(0).expand(x.shape[0], -1, -1)

    projected = u_t @ (u_t.transpose(-1, -2) @ x)
    residual = x - projected
    return (residual.pow(2).sum(dim=(-2, -1)) / x.pow(2).sum(dim=(-2, -1)).clamp_min(1e-12)).mean()


def pointwise_cosine_loss(
    features_teacher: torch.Tensor, features_student: torch.Tensor
) -> torch.Tensor:
    """1 - mean token-wise cosine similarity. The rigid point-to-point baseline."""
    return 1.0 - F.cosine_similarity(features_student, features_teacher, dim=-1).mean()


def pointwise_mse_loss(
    features_teacher: torch.Tensor, features_student: torch.Tensor
) -> torch.Tensor:
    """Scale-free mean squared error between token embeddings."""
    scale = features_teacher.pow(2).mean().clamp_min(1e-12)
    return F.mse_loss(features_student, features_teacher) / scale


def spectral_entropy(g: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """Shannon entropy of the normalised Gram spectrum, in nats.

    Cut-ViT uses this to trade the spatial term against the channel term. It is
    reported here as a diagnostic of how concentrated a Gram spectrum is: an
    entropy near log(rank) means a flat spectrum, on which alpha has little
    leverage, and a low entropy means the leading directions dominate.
    """
    rank = g.shape[-1] if k is None else min(k, g.shape[-1])
    values, _ = top_eigenbasis(g, rank)
    p = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1).mean()


def cumulative_explained_variance(g: torch.Tensor, k: int) -> torch.Tensor:
    """Fraction of Gram spectral energy carried by the leading K directions."""
    values, _ = top_eigenbasis(g, g.shape[-1])
    total = values.sum(dim=-1).clamp_min(1e-12)
    return (values[..., : min(k, values.shape[-1])].sum(dim=-1) / total).mean()


# --------------------------------------------------------------------------- #
# Configurable composite objective
# --------------------------------------------------------------------------- #


@dataclass
class ObjectiveConfig:
    """One alignment objective, as a weighted sum of named terms.

    Attributes:
        name: label used in run directories and tables.
        pointwise: ``none``, ``cosine`` or ``mse``.
        pointwise_weight: coefficient on the pointwise term.
        axes: Gram axes carrying a spectral term.
        alpha: spectral exponent; 0 recovers the subspace (chordal) objective.
        rank: K, or ``None`` for no truncation.
        spectral_weight: coefficient on the spectral terms, before axis weighting.
        residual_weight: coefficient on the residual term (0 disables it).
        pooling: how Gram matrices are formed over the batch.
        entropy_weighting: split the spectral weight between axes in proportion
            to the teacher's spectral entropy along each, as Cut-ViT does.
        use_chordal_form: compute alpha=0 through the explicit basis-overlap
            expression rather than through the projector; mathematically
            identical, retained for verification.
        tokens: which tokens enter the features (``patch``, ``cls`` or ``all``).
    """

    name: str
    pointwise: Literal["none", "cosine", "mse"] = "cosine"
    pointwise_weight: float = 1.0
    axes: Tuple[GramAxis, ...] = ()
    alpha: float = 1.0
    rank: Optional[int] = None
    spectral_weight: float = 1.0
    residual_weight: float = 0.0
    pooling: GramPooling = "per_image"
    entropy_weighting: bool = False
    use_chordal_form: bool = False
    tokens: Literal["patch", "cls", "all"] = "patch"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "pointwise": self.pointwise,
            "pointwise_weight": self.pointwise_weight,
            "axes": list(self.axes),
            "alpha": self.alpha,
            "rank": self.rank,
            "spectral_weight": self.spectral_weight,
            "residual_weight": self.residual_weight,
            "pooling": self.pooling,
            "entropy_weighting": self.entropy_weighting,
            "use_chordal_form": self.use_chordal_form,
            "tokens": self.tokens,
        }


class AlignmentObjective:
    """Evaluates an :class:`ObjectiveConfig` on a teacher/student feature pair."""

    def __init__(self, config: ObjectiveConfig):
        self.config = config

    def __call__(
        self,
        features_teacher: torch.Tensor,
        features_student: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.config
        terms: Dict[str, float] = {}
        total = features_student.new_zeros(())

        if cfg.pointwise != "none" and cfg.pointwise_weight != 0.0:
            if cfg.pointwise == "cosine":
                term = pointwise_cosine_loss(features_teacher, features_student)
            else:
                term = pointwise_mse_loss(features_teacher, features_student)
            total = total + cfg.pointwise_weight * term
            terms["pointwise"] = float(term.detach())

        if not cfg.axes:
            return total, terms

        grams_t = {
            axis: gram(features_teacher, axis=axis, pooling=cfg.pooling) for axis in cfg.axes
        }
        grams_s = {
            axis: gram(features_student, axis=axis, pooling=cfg.pooling) for axis in cfg.axes
        }

        axis_weights = self._axis_weights(grams_t)

        for axis in cfg.axes:
            g_t, g_s = grams_t[axis], grams_s[axis]

            if cfg.use_chordal_form and cfg.alpha == 0.0:
                if cfg.rank is None:
                    raise ValueError("the chordal form requires a rank")
                term = chordal_subspace_loss(g_t, g_s, k=cfg.rank)
            else:
                term = spectral_alignment_loss(g_t, g_s, alpha=cfg.alpha, k=cfg.rank)

            weight = cfg.spectral_weight * axis_weights[axis]
            total = total + weight * term
            terms[f"spectral_{axis}"] = float(term.detach())

            if cfg.residual_weight != 0.0:
                if cfg.rank is None:
                    raise ValueError("the residual term requires a rank")
                res = residual_loss(features_student, g_t, k=cfg.rank, axis=axis)
                total = total + cfg.residual_weight * axis_weights[axis] * res
                terms[f"residual_{axis}"] = float(res.detach())

        return total, terms

    def _axis_weights(self, grams_teacher: Dict[GramAxis, torch.Tensor]) -> Dict[GramAxis, float]:
        cfg = self.config
        if not cfg.entropy_weighting or len(cfg.axes) < 2:
            return {axis: 1.0 for axis in cfg.axes}

        with torch.no_grad():
            entropies = {
                axis: float(spectral_entropy(g, k=cfg.rank)) for axis, g in grams_teacher.items()
            }
        total = sum(entropies.values())
        if total <= 0:
            return {axis: 1.0 for axis in cfg.axes}
        return {axis: value / total for axis, value in entropies.items()}


# --------------------------------------------------------------------------- #
# The objectives compared in the paper
# --------------------------------------------------------------------------- #

#: Rank used by Cut-ViT for the low-rank SVD of the Gram matrices.
CUTVIT_RANK = 192


def build_objective(name: str, **overrides) -> AlignmentObjective:
    """Look up one of the named objectives, optionally overriding fields."""
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}; known: {sorted(OBJECTIVES)}")
    config = OBJECTIVES[name]
    if overrides:
        config = ObjectiveConfig(**{**config.to_dict(), **overrides})
        config.axes = tuple(config.axes)
    return AlignmentObjective(config)


def _cfg(name: str, **kwargs) -> ObjectiveConfig:
    return ObjectiveConfig(name=name, **kwargs)


OBJECTIVES: Dict[str, ObjectiveConfig] = {
    # --- pointwise baselines ------------------------------------------------
    "cosine": _cfg("cosine", pointwise="cosine"),
    "mse": _cfg("mse", pointwise="mse"),
    # --- the published subspace objective ------------------------------------
    # Cut-ViT: pointwise cosine plus the rank-K chordal term on both axes, plus
    # the residual term. The 0.005 scaling of the released code is folded into
    # spectral_weight / residual_weight.
    "cutvit": _cfg(
        "cutvit",
        pointwise="cosine",
        axes=("spatial", "channel"),
        alpha=0.0,
        rank=CUTVIT_RANK,
        spectral_weight=1.0,
        residual_weight=1.0,
        pooling="batch_pooled",
        use_chordal_form=True,
    ),
    # The released implementation multiplies both Gram terms by 0.005 and then
    # halves their sum, so they enter the total at 1/400 of the pointwise term.
    # Keeping that scaling separate from the objective itself distinguishes a
    # claim about subspace alignment from a claim about the weight it is given.
    "cutvit-scaled": _cfg(
        "cutvit-scaled",
        pointwise="cosine",
        axes=("spatial", "channel"),
        alpha=0.0,
        rank=CUTVIT_RANK,
        spectral_weight=0.0025,
        residual_weight=0.0025,
        pooling="batch_pooled",
        use_chordal_form=True,
    ),
    "cutvit-entropy": _cfg(
        "cutvit-entropy",
        pointwise="cosine",
        axes=("spatial", "channel"),
        alpha=0.0,
        rank=CUTVIT_RANK,
        spectral_weight=1.0,
        residual_weight=1.0,
        pooling="batch_pooled",
        entropy_weighting=True,
        use_chordal_form=True,
    ),
    # --- the spectral family, alpha sweep at full rank -----------------------
    **{
        f"gram-a{alpha:g}": _cfg(
            f"gram-a{alpha:g}",
            pointwise="cosine",
            axes=("spatial", "channel"),
            alpha=alpha,
            rank=None,
        )
        for alpha in (0.25, 0.5, 1.0, 2.0)
    },
    # --- the same family at Cut-ViT's rank, to separate alpha from K ---------
    # Only the exponents whose truncated form is well conditioned: the
    # projector at alpha=0, whose gradient depends on the subspace and not on
    # the individual eigenvectors, and exponents of at least one, whose weight
    # derivative is bounded. See spectral_power.
    **{
        f"gram-a{alpha:g}-k{CUTVIT_RANK}": _cfg(
            f"gram-a{alpha:g}-k{CUTVIT_RANK}",
            pointwise="cosine",
            axes=("spatial", "channel"),
            alpha=alpha,
            rank=CUTVIT_RANK,
        )
        for alpha in (0.0, 1.0, 2.0)
    },
    # --- single-axis variants of the recommended member ----------------------
    "gram-a1-spatial": _cfg(
        "gram-a1-spatial", pointwise="cosine", axes=("spatial",), alpha=1.0, rank=None
    ),
    "gram-a1-channel": _cfg(
        "gram-a1-channel", pointwise="cosine", axes=("channel",), alpha=1.0, rank=None
    ),
    # --- the spectral term without any pointwise anchor ----------------------
    "gram-a1-only": _cfg(
        "gram-a1-only",
        pointwise="none",
        axes=("spatial", "channel"),
        alpha=1.0,
        rank=None,
    ),
    # --- pooling ablation ----------------------------------------------------
    "gram-a1-pooled": _cfg(
        "gram-a1-pooled",
        pointwise="cosine",
        axes=("spatial", "channel"),
        alpha=1.0,
        rank=None,
        pooling="batch_pooled",
    ),
}


def objective_names() -> List[str]:
    return sorted(OBJECTIVES)

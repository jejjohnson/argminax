"""The Bayesian learning rule with a low-rank-plus-diagonal precision.

The transform carries q(θ) = 𝒩(μ_θ, Λ_θ⁻¹) with Λ_θ = diag(d) + U Uᵀ, a
diagonal plus a rank-R factor, updated with SLANG's rule. It stores
O(D_θ R) numbers and a step costs O(D_θ R²) with a built-in curvature
estimate, between the diagonal rule and the full-rank one. A callable
curvature returns a dense matrix, and its step costs O(D_θ³).
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from argminax._src.blr._curvature import Curvature, resolve_curvature_factor
from argminax._src.blr._optax import (
    StepSize,
    require_non_negative,
    require_optax,
    require_positive,
    require_prior_mean,
    require_step_size,
    step_size,
)
from argminax.exceptions import ValidationError


if TYPE_CHECKING:
    import optax


__all__ = [
    "BLRLowRankState",
    "blr_low_rank",
]


class BLRLowRankState(NamedTuple):
    """The state of `blr_low_rank`: a Gaussian with Λ_θ = diag(d) + U Uᵀ.

    The mean is μ_θ = Λ_θ⁻¹ (Λ_θ μ_θ) and the covariance Λ_θ⁻¹. A
    ``NamedTuple``, as optax states are.

    Attributes:
        diag_precision: The diagonal d of the precision, shape (D_θ,).
        low_rank_factor: The factor U of the precision, shape (D_θ, R).
        nat_mean: The natural mean Λ_θ μ_θ, shape (D_θ,).
        count: The number of steps taken.
    """

    diag_precision: Float[Array, " state"]
    low_rank_factor: Float[Array, "state rank"]
    nat_mean: Float[Array, " state"]
    count: Int[Array, ""]


def _scaled_left_vectors(
    factor: Float[Array, "state factor"],
) -> tuple[
    Float[Array, "state kept"], Float[Array, " kept"], Float[Array, "kept factor"]
]:
    """The thin SVD Ũ = P S Qᵀ of a factor, as P S, the diagonal of S and Qᵀ."""
    left, singular_values, right = jnp.linalg.svd(factor, full_matrices=False)
    # P S, the left singular vectors scaled by the singular values
    scaled = einx.multiply("state kept, kept -> state kept", left, singular_values)
    return scaled, singular_values, right


def _split(
    scaled: Float[Array, "state kept"], rank: int
) -> tuple[Float[Array, "state rank"], Float[Array, " state"]]:
    """The first ``rank`` columns, and the squared row norms of the others."""
    kept = scaled[:, :rank]
    discarded = scaled[:, rank:]
    # e = diag(Ũ Ũᵀ − U Uᵀ), the squared row norms of the discarded columns
    # (jnp, not einx: nothing is discarded when K ≤ R, and einx cannot reduce
    # over an axis of size zero)
    residual = jnp.sum(discarded * discarded, axis=1)
    return kept, residual


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _truncate(
    factor: Float[Array, "state factor"], rank: int
) -> tuple[Float[Array, "state rank"], Float[Array, " state"]]:
    """Keep the top ``rank`` directions of a factor, and the diagonal of the rest.

        Ũ = P S Qᵀ,   U = (P S)[:, :R],   e = diag(Ũ Ũᵀ − U Uᵀ)

    U Uᵀ is the best rank-R approximation of Ũ Ũᵀ, and e ≥ 0 is the diagonal
    of what was discarded. The thin SVD costs O(D_θ K²) for K columns.

    The derivative is that of U Uᵀ and e, which is what the precision is made
    of; U itself is only defined up to a rotation of its columns. JAX's own
    SVD derivative divides by the gaps between all pairs of singular values,
    and so is NaN whenever two are equal, as the zero ones are at the first
    step. Only the gaps across the cut, between a kept and a discarded
    direction, move U Uᵀ and e, so `_truncate_jvp` uses those alone.
    """
    scaled, _, _ = _scaled_left_vectors(factor)
    return _split(scaled, rank)


@_truncate.defjvp
def _truncate_jvp(
    rank: int,
    primals: tuple[Float[Array, "state factor"]],
    tangents: tuple[Float[Array, "state factor"]],
) -> tuple[
    tuple[Float[Array, "state rank"], Float[Array, " state"]],
    tuple[Float[Array, "state rank"], Float[Array, " state"]],
]:
    """The derivative of `_truncate`, from the gaps across the cut alone.

        W = dŨ Q,   M = Wᵀ (P S) + (P S)ᵀ W
        d(P S)ᵢ = Wᵢ + Σⱼ (P S)ⱼ Mⱼᵢ / (sᵢ² − sⱼ²)

    P S (D_θ, K) holds the scaled left singular vectors of Ũ, Q (K, K) the
    right ones and s the singular values; i and j index the columns. This is
    first-order perturbation theory for the eigenvectors Q of ŨᵀŨ, carried to
    P S = Ũ Q. The sum runs over the j on the other side of the cut from i:
    a pair on the same side only rotates the columns among themselves, which
    leaves U Uᵀ and e unchanged, so it is left out, whatever its gap. A pair
    across the cut with no gap between them has no derivative (the top R
    directions are not unique there) and is left out as well, which gives
    the derivative that keeps the current directions.
    """
    (factor,) = primals
    (factor_dot,) = tangents
    scaled, singular_values, right = _scaled_left_vectors(factor)
    kept, residual = _split(scaled, rank)

    # W = dŨ Q
    rotated = einx.dot("state factor, kept factor -> state kept", factor_dot, right)
    # M = Wᵀ (P S) + (P S)ᵀ W, the change of QᵀŨᵀŨ Q
    cross = einx.dot("state row, state column -> row column", scaled, rotated)
    gram_dot = cross + cross.T
    # sᵢ² − sⱼ² at [j, i]
    squares = singular_values * singular_values
    gaps = einx.subtract("column, row -> row column", squares, squares)
    is_kept = jnp.arange(squares.shape[0]) < rank
    across_cut = einx.not_equal("row, column -> row column", is_kept, is_kept)
    # a gap at the rounding level of the largest sᵢ² counts as no gap
    floor = jnp.finfo(squares.dtype).eps * squares.shape[0] * jnp.max(squares)
    resolved = across_cut & (jnp.abs(gaps) > floor)
    mixing = jnp.where(resolved, gram_dot / jnp.where(resolved, gaps, 1.0), 0.0)
    # d(P S) = W + (P S) (M ⊘ gaps)
    scaled_dot = rotated + einx.dot(
        "state row, row column -> state column", scaled, mixing
    )

    kept_dot = scaled_dot[:, :rank]
    # de = 2 Σ (P S) ⊙ d(P S) over the discarded columns (jnp: see `_split`)
    residual_dot = 2.0 * jnp.sum(scaled[:, rank:] * scaled_dot[:, rank:], axis=1)
    return (kept, residual), (kept_dot, residual_dot)


def blr_low_rank(
    learning_rate: StepSize = 1e-2,
    *,
    rank: int = 10,
    prior_precision: float = 1e-4,
    prior_mean: Float[Array, " state"] | None = None,
    curvature: Literal["ggn", "identity"] | Curvature = "ggn",
    damping: float = 1e-6,
    solver: lx.AbstractLinearSolver | gx.AbstractSolveStrategy | None = None,
) -> optax.GradientTransformation:
    """The Bayesian learning rule with a low-rank-plus-diagonal precision.

        Ũ = [√(1 − α) U, √α G]
        U ← top_R(Ũ)
        d ← (1 − α) d + α δ + diag(Ũ Ũᵀ − U Uᵀ) + β
        μ_θ ← μ_θ − α Λ_θ⁻¹ (g + δ (μ_θ − μ₀)),   Λ_θ = diag(d) + U Uᵀ

    The transform learns q(θ) = 𝒩(μ_θ, Λ_θ⁻¹) over a flat parameter vector
    θ ∈ ℝ^D_θ, with d ∈ ℝ^D_θ positive and U (D_θ, R) a rank-R factor.
    g = ∇_θ 𝒥 ∈ ℝ^D_θ is the gradient of the minibatch cost 𝒥, as any optax
    optimiser takes it; G (D_θ, K) is a factor of the curvature estimate,
    C = G Gᵀ ≈ ∇²_θ 𝒥; 𝒩(μ₀, δ⁻¹ I) is the prior; α is the step size and β
    the damping. (1 − α) U Uᵀ + α C = Ũ Ũᵀ exactly, so the only approximation
    is the truncation to R directions, and the diagonal of what it discards is
    kept in d: the diagonal of the precision is always exact, and the whole
    precision is exact when R ≥ D_θ. The update it returns is the change of
    the mean, which starts at the parameters given to ``init``.

    The mean takes a step preconditioned by the new precision. It equals the
    natural-mean recursion of `blr_full_rank` when the precision is exact and
    β = 0 (with damping, that recursion also shrinks the mean by β Λ_θ⁻¹ μ_θ
    a step, which this one does not), and stays well defined when the
    precision is truncated, where the recursion would amplify the truncation
    error by Λ_θ⁻¹ along weakly curved directions. As in `blr_full_rank`, the
    damping is added at every step, so d settles β/α above its undamped value.

    Algorithm:
        1. Solve for the mean μ_θ = Λ_θ⁻¹ (Λ_θ μ_θ) by the Woodbury identity.
        2. Factor the curvature at μ_θ: G = g (``"ggn"``), no columns
           (``"identity"``) or the positive part of the callable's C(μ_θ, g).
        3. Stack Ũ = [√(1 − α) U, √α G] and keep its top R directions by a
           thin SVD.
        4. Update d with the decayed diagonal, the prior, the diagonal of
           the discarded part and the damping.
        5. Step the mean with the new precision and store Λ_θ μ_θ.

    Args:
        learning_rate: The step size α in (0, 1], or an optax schedule of the
            step count.
        rank: The rank R ≥ 1 of the factor U; a rank above D_θ is used as
            D_θ.
        prior_precision: The prior precision δ > 0.
        prior_mean: The prior mean μ₀, shape (D_θ,); ``None`` is zero.
        curvature: The curvature estimate C: ``"ggn"``, ``"identity"`` or a
            callable ``curvature(mean, grad)`` returning the (D_θ, D_θ)
            curvature of the cost. A callable's C is factored by an
            eigendecomposition, O(D_θ³) a step, and its negative part is
            dropped.
        damping: The damping β ≥ 0 added to d at every step.
        solver: A lineax solver or a gaussx solve strategy for the solves
            with Λ_θ; ``None`` lets gaussx use the Woodbury identity.

    Returns:
        An ``optax.GradientTransformation`` whose state is a
        `BLRLowRankState`.

    Raises:
        ValidationError: If ``rank`` is not a positive integer, ``curvature``
            is neither a known name nor a callable, a constant
            ``learning_rate`` is outside (0, 1], ``prior_precision`` is not
            positive or ``damping`` is negative; or, from ``init`` and
            ``update``, if ``prior_mean`` is not shaped like the parameters.
        MissingExtraError: If optax is not installed.

    Examples:
        One step from the prior: the gradient becomes the single direction of
        the factor, and the precision is diag(d) + U Uᵀ = δ I + α g gᵀ.

        >>> import jax.numpy as jnp
        >>> from argminax.blr import blr_low_rank
        >>> optimiser = blr_low_rank(0.5, rank=1, prior_precision=1.0, damping=0.0)
        >>> state = optimiser.init(jnp.zeros(3))
        >>> grads = jnp.array([2.0, 0.0, 0.0])
        >>> updates, state = optimiser.update(grads, state)
        >>> factor = state.low_rank_factor
        >>> factor.shape
        (3, 1)
        >>> precision = jnp.diag(state.diag_precision) + factor @ factor.T
        >>> bool(
        ...     jnp.allclose(precision, jnp.eye(3) + 0.5 * jnp.outer(grads, grads))
        ... )
        True

        A rank above the number of parameters is used as that number:

        >>> optimiser = blr_low_rank(0.5, rank=10)
        >>> optimiser.init(jnp.zeros(3)).low_rank_factor.shape
        (3, 3)

        The rank must be a positive integer:

        >>> blr_low_rank(rank=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'rank' must be a positive integer, got 0

    References:
        Mishkin et al. (2018), "SLANG: Fast Structured Covariance
        Approximations for Bayesian Deep Learning with Natural Gradient",
        arXiv:1811.04504.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        msg = f"'rank' must be a positive integer, got {rank!r}"
        raise ValidationError(msg)
    require_step_size(learning_rate)
    require_positive(prior_precision, "prior_precision")
    require_non_negative(damping, "damping")
    factor_of = resolve_curvature_factor(curvature)
    optax = require_optax()

    def solve(
        diag_precision: Float[Array, " state"],
        low_rank_factor: Float[Array, "state rank"],
        vector: Float[Array, " state"],
    ) -> Float[Array, " state"]:
        # (diag(d) + U Uᵀ)⁻¹ v
        operator = gx.low_rank_plus_diag(diag_precision, low_rank_factor, psd=True)
        return gx.solve(operator, vector, solver=solver)

    def init_fn(params: Float[Array, " state"]) -> BLRLowRankState:
        require_prior_mean(prior_mean, params.shape)
        size = params.shape[0]
        # d = δ, U = 0: the prior
        diag_precision = jnp.full(size, prior_precision, params.dtype)
        low_rank_factor = jnp.zeros((size, min(rank, size)), params.dtype)
        # Λ_θ μ_θ = δ θ, so the mean starts at the caller's parameters
        nat_mean = prior_precision * params
        return BLRLowRankState(
            diag_precision, low_rank_factor, nat_mean, jnp.zeros([], jnp.int32)
        )

    def update_fn(
        grads: Float[Array, " state"],
        state: BLRLowRankState,
        params: Any = None,
    ) -> tuple[Float[Array, " state"], BLRLowRankState]:
        alpha = step_size(learning_rate, state.count)
        # the state's dtype, set from the parameters at init, is the working one
        dtype = state.nat_mean.dtype
        grads = jnp.asarray(grads, dtype)
        # checked here as well as in init: a state can come from another transform
        require_prior_mean(prior_mean, grads.shape)
        size = grads.shape[0]
        centre = jnp.zeros(size, dtype) if prior_mean is None else prior_mean
        centre = jnp.asarray(centre, dtype)

        # --- the mean and the curvature factor there ---
        mean = solve(state.diag_precision, state.low_rank_factor, state.nat_mean)
        curvature_factor = jnp.asarray(factor_of(mean, grads), dtype)

        # --- the precision: stack, truncate, keep the discarded diagonal ---
        # √(1 − α) U
        decayed = jnp.sqrt(jnp.maximum(1 - alpha, 0.0)) * state.low_rank_factor
        # √α G
        incoming = jnp.sqrt(alpha) * curvature_factor
        # Ũ = [√(1 − α) U, √α G]
        # (jnp, not einx: G has no columns for the "identity" curvature, and
        # einx cannot concatenate an axis of size zero)
        stacked = jnp.concatenate([decayed, incoming], axis=1)
        low_rank_factor, residual = _truncate(stacked, min(rank, size))
        # d ← (1 − α) d + α δ + diag(Ũ Ũᵀ − U Uᵀ) + β
        diag_precision = (1 - alpha) * state.diag_precision + alpha * prior_precision
        diag_precision = diag_precision + residual + damping

        # --- the mean, preconditioned by the new precision ---
        # g + δ (μ_θ − μ₀)
        joint_grad = grads + prior_precision * (mean - centre)
        # Δθ = −α Λ_θ⁻¹ (g + δ (μ_θ − μ₀))
        updates = solve(diag_precision, low_rank_factor, -alpha * joint_grad)
        new_mean = mean + updates
        # Λ_θ μ_θ = d ⊙ μ_θ + U (Uᵀ μ_θ)
        operator = gx.low_rank_plus_diag(diag_precision, low_rank_factor, psd=True)
        nat_mean = operator.mv(new_mean)
        return updates, BLRLowRankState(
            diag_precision, low_rank_factor, nat_mean, state.count + 1
        )

    return optax.GradientTransformation(init_fn, update_fn)

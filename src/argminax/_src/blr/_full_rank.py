"""The Bayesian learning rule with a full-rank Gaussian, and Newton as its limit.

The transform carries q(θ) = 𝒩(μ_θ, Λ_θ⁻¹) over a flat parameter vector, with a
dense D_θ × D_θ precision, so a step costs two solves, O(D_θ³). With the exact
curvature its fixed point is the Laplace approximation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import einx
import gaussx as gx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from argminax._src.blr._curvature import Curvature, resolve_curvature
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
    "BLRFullRankState",
    "blr_full_rank",
    "newton",
]


#: A solver for the precision system: a lineax solver or a gaussx strategy.
type Solver = lx.AbstractLinearSolver | gx.AbstractSolveStrategy


class BLRFullRankState(NamedTuple):
    """The state of `blr_full_rank` and `newton`: a Gaussian in natural parameters.

    The mean is μ_θ = Λ_θ⁻¹ (Λ_θ μ_θ) and the covariance Λ_θ⁻¹. A
    ``NamedTuple``, as optax states are.

    Attributes:
        precision: The precision Λ_θ, shape (D_θ, D_θ).
        nat_mean: The natural mean Λ_θ μ_θ, shape (D_θ,).
        count: The number of steps taken.
    """

    precision: Float[Array, "state state"]
    nat_mean: Float[Array, " state"]
    count: Int[Array, ""]


def _transform(
    *,
    learning_rate: StepSize,
    prior_precision: float,
    prior_mean: Float[Array, " state"] | None,
    curvature: Curvature,
    damping: float,
    solver: Solver | None,
    psd: bool,
) -> optax.GradientTransformation:
    """Build the full-rank transform; `blr_full_rank` documents the update.

    The step is written here and not delegated to `gaussx.blr_full_update`
    for two reasons: the curvature needs the mean, which that function solves
    for and does not return, and it always tags the precision positive
    semidefinite, which `newton` cannot promise (``psd=False``). The tests pin
    the step to `gaussx.blr_full_update` on the log-joint.
    """
    optax = require_optax()
    tags = lx.positive_semidefinite_tag if psd else ()

    def mean_of(
        precision: Float[Array, "state state"], nat_mean: Float[Array, " state"]
    ) -> Float[Array, " state"]:
        # μ_θ = Λ_θ⁻¹ (Λ_θ μ_θ)
        operator = lx.MatrixLinearOperator(precision, tags)
        return gx.solve(operator, nat_mean, solver=solver)

    def init_fn(params: Float[Array, " state"]) -> BLRFullRankState:
        require_prior_mean(prior_mean, params.shape)
        size = params.shape[0]
        # Λ_θ = δ I
        precision = prior_precision * jnp.eye(size, dtype=params.dtype)
        # Λ_θ μ_θ = δ θ, so the mean starts at the caller's parameters
        nat_mean = prior_precision * params
        return BLRFullRankState(precision, nat_mean, jnp.zeros([], jnp.int32))

    def update_fn(
        grads: Float[Array, " state"],
        state: BLRFullRankState,
        params: Any = None,
    ) -> tuple[Float[Array, " state"], BLRFullRankState]:
        alpha = step_size(learning_rate, state.count)
        # the state's dtype, set from the parameters at init, is the working one
        dtype = state.nat_mean.dtype
        grads = jnp.asarray(grads, dtype)
        # checked here as well as in init: a state can come from another transform
        require_prior_mean(prior_mean, grads.shape)
        size = grads.shape[0]
        identity = jnp.eye(size, dtype=dtype)
        centre = jnp.zeros(size, dtype) if prior_mean is None else prior_mean
        centre = jnp.asarray(centre, dtype)

        # --- the mean and the curvature there ---
        mean = mean_of(state.precision, state.nat_mean)
        estimate = jnp.asarray(curvature(mean, grads), dtype)

        # --- the damped step on the natural parameters ---
        # Λ_θ ← (1 − α) Λ_θ + α (δ I + C) + β I
        target_precision = prior_precision * identity + estimate
        precision = (1 - alpha) * state.precision + alpha * target_precision
        precision = precision + damping * identity
        # C μ_θ
        curved_mean = einx.dot("row column, column -> row", estimate, mean)
        # Λ_θ μ_θ ← (1 − α) Λ_θ μ_θ + α (δ μ₀ − g + C μ_θ)
        target_nat_mean = prior_precision * centre - grads + curved_mean
        nat_mean = (1 - alpha) * state.nat_mean + alpha * target_nat_mean

        # --- the change of the mean ---
        # Δθ = Λ_θ'⁻¹ (Λ_θ μ_θ)' − μ_θ
        updates = mean_of(precision, nat_mean) - mean
        return updates, BLRFullRankState(precision, nat_mean, state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


def blr_full_rank(
    learning_rate: StepSize = 1e-2,
    *,
    prior_precision: float = 1e-4,
    prior_mean: Float[Array, " state"] | None = None,
    curvature: Literal["ggn", "identity"] | Curvature = "ggn",
    damping: float = 1e-6,
    solver: Solver | None = None,
) -> optax.GradientTransformation:
    """The Bayesian learning rule with a full-rank Gaussian, as an optax transform.

        Λ_θ ← (1 − α) Λ_θ + α (δ I + C) + β I
        Λ_θ μ_θ ← (1 − α) Λ_θ μ_θ + α (δ μ₀ − g + C μ_θ)

    The transform learns q(θ) = 𝒩(μ_θ, Λ_θ⁻¹) over a flat parameter vector
    θ ∈ ℝ^D_θ, with Λ_θ a dense (D_θ, D_θ) precision. g = ∇_θ 𝒥 ∈ ℝ^D_θ is the
    gradient of the minibatch cost 𝒥 (a negative log-likelihood), as any optax
    optimiser takes it; C (D_θ, D_θ) is an estimate of its curvature ∇²_θ 𝒥,
    positive semidefinite; 𝒩(μ₀, δ⁻¹ I) is the prior; α is the step size and
    β the damping. The update it returns is the change of the mean, so
    ``optax.apply_updates`` keeps the parameters equal to μ_θ, and the mean
    starts at the parameters given to ``init``. Flatten a parameter pytree
    with ``jax.flatten_util.ravel_pytree`` first.

    For a quadratic cost with constant curvature C, α = 1 and β = 0 give the
    exact Gaussian posterior, Λ_θ = δ I + C, in one step from any start;
    otherwise the fixed point with the exact curvature is the Laplace
    approximation. The damping is added at every step, so the precision
    settles at δ I + C + (β/α) I, a shift of β/α above the undamped value.

    Algorithm:
        1. Solve for the mean μ_θ = Λ_θ⁻¹ (Λ_θ μ_θ).
        2. Estimate the curvature at μ_θ: C = g gᵀ (``"ggn"``), C = 0
           (``"identity"``) or the callable's C(μ_θ, g).
        3. Move the precision and the natural mean towards their targets
           with step α, and add β I to the precision.
        4. Solve for the new mean and return its change.

    Args:
        learning_rate: The step size α in (0, 1], or an optax schedule of the
            step count.
        prior_precision: The prior precision δ > 0.
        prior_mean: The prior mean μ₀, shape (D_θ,); ``None`` is zero.
        curvature: The curvature estimate C: ``"ggn"``, ``"identity"`` or a
            callable ``curvature(mean, grad)`` returning the (D_θ, D_θ)
            curvature of the cost.
        damping: The damping β ≥ 0 added to the diagonal of the precision at
            every step.
        solver: A lineax solver or a gaussx solve strategy for the two solves
            with Λ_θ; ``None`` lets gaussx pick from the operator's structure
            (Cholesky, since Λ_θ is tagged positive semidefinite).

    Returns:
        An ``optax.GradientTransformation`` whose state is a
        `BLRFullRankState`.

    Raises:
        ValidationError: If ``curvature`` is neither a known name nor a
            callable, a constant ``learning_rate`` is outside (0, 1],
            ``prior_precision`` is not positive or ``damping`` is negative;
            or, from ``init`` and ``update``, if ``prior_mean`` is not shaped
            like the parameters.
        MissingExtraError: If optax is not installed.

    Examples:
        One step with the exact curvature, α = 1 and no damping reaches the
        posterior of a quadratic cost 𝒥(θ) = ½ (θ − 3)ᵀ P (θ − 3):

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from argminax.blr import blr_full_rank
        >>> lik_precision = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        >>> cost = lambda theta: 0.5 * (theta - 3.0) @ lik_precision @ (theta - 3.0)
        >>> optimiser = blr_full_rank(
        ...     1.0,
        ...     prior_precision=1.0,
        ...     curvature=lambda mean, grad: lik_precision,
        ...     damping=0.0,
        ... )
        >>> theta = jnp.zeros(2)
        >>> state = optimiser.init(theta)
        >>> updates, state = optimiser.update(jax.grad(cost)(theta), state)
        >>> bool(jnp.allclose(state.precision, jnp.eye(2) + lik_precision))
        True
        >>> posterior_mean = jnp.linalg.solve(
        ...     jnp.eye(2) + lik_precision, lik_precision @ jnp.full(2, 3.0)
        ... )
        >>> bool(jnp.allclose(theta + updates, posterior_mean))
        True

        The default ``"ggn"`` curvature adds g gᵀ, so the precision stays
        symmetric and grows along the gradient:

        >>> optimiser = blr_full_rank(0.5, prior_precision=1.0, damping=0.0)
        >>> state = optimiser.init(jnp.zeros(2))
        >>> _, state = optimiser.update(jnp.array([2.0, 0.0]), state)
        >>> bool(jnp.allclose(state.precision, jnp.array([[3.0, 0.0], [0.0, 1.0]])))
        True

        A curvature that is neither a name nor a callable is rejected:

        >>> blr_full_rank(curvature="exact")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'curvature' must be one of ('ggn', 'identity') or ...

    References:
        Khan and Rue (2023), "The Bayesian Learning Rule", arXiv:2107.04562.
    """
    require_step_size(learning_rate)
    require_positive(prior_precision, "prior_precision")
    require_non_negative(damping, "damping")
    return _transform(
        learning_rate=learning_rate,
        prior_precision=prior_precision,
        prior_mean=prior_mean,
        curvature=resolve_curvature(curvature),
        damping=damping,
        solver=solver,
        psd=True,
    )


def newton(
    curvature: Curvature,
    *,
    damping: float = 1e-6,
    solver: Solver | None = None,
) -> optax.GradientTransformation:
    """Damped Newton's method, as an optax transform.

        μ_θ ← μ_θ − (β I + C)⁻¹ (g + β μ_θ)

    This is `blr_full_rank` with step size α = 1, prior precision δ = β, a
    zero prior mean and no further damping: a Newton step on the cost plus
    (β/2) ‖θ‖², which tends to the classical step as β → 0. g = ∇_θ 𝒥 ∈ ℝ^D_θ
    is the gradient of the cost and C (D_θ, D_θ) its Hessian at the mean. The
    Hessian may be indefinite away from a minimum, so the system is solved
    without assuming it is positive semidefinite. For a solve that runs to a
    tolerance use `argminax.solve.GaussNewtonCG` or an optimistix solver; for
    one step's algebra, `gaussx.newton_update`.

    Args:
        curvature: A callable ``curvature(mean, grad)`` returning the
            (D_θ, D_θ) Hessian of the cost at the mean.
        damping: The damping β > 0 added to the diagonal of the Hessian.
        solver: A lineax solver or a gaussx solve strategy for β I + C;
            ``None`` lets gaussx pick from the operator's structure (LU).

    Returns:
        An ``optax.GradientTransformation`` whose state is a
        `BLRFullRankState`.

    Raises:
        ValidationError: If ``curvature`` is not a callable or ``damping`` is
            not positive.
        MissingExtraError: If optax is not installed.

    Examples:
        One step minimises a quadratic cost, up to the damping:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from argminax.blr import newton
        >>> hessian = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        >>> cost = lambda theta: 0.5 * (theta - 3.0) @ hessian @ (theta - 3.0)
        >>> optimiser = newton(lambda mean, grad: jax.hessian(cost)(mean))
        >>> theta = jnp.zeros(2)
        >>> state = optimiser.init(theta)
        >>> updates, state = optimiser.update(jax.grad(cost)(theta), state)
        >>> bool(jnp.allclose(theta + updates, 3.0, atol=1e-5))
        True

        The curvature must be a callable; a name has no exact Hessian:

        >>> newton("ggn")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'curvature' must be a callable, got 'ggn'
    """
    if not callable(curvature):
        msg = f"'curvature' must be a callable, got {curvature!r}"
        raise ValidationError(msg)
    require_positive(damping, "damping")
    return _transform(
        learning_rate=1.0,
        prior_precision=damping,
        prior_mean=None,
        curvature=curvature,
        damping=0.0,
        solver=solver,
        psd=False,
    )

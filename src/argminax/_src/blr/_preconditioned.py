"""Gradient descent preconditioned by a fixed operator, as optax transforms.

The mean form of the Bayesian learning rule is μ_θ ← μ_θ − α Λ_θ⁻¹ g. The
other transforms of this package learn the precision Λ_θ as they step; these
two take it from the caller, as an operator P, and never change it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gaussx as gx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from argminax._src.blr._optax import (
    StepSize,
    require_optax,
    require_positive,
)
from argminax.exceptions import ValidationError


if TYPE_CHECKING:
    import optax


__all__ = [
    "preconditioned_sgd",
    "scale_by_preconditioner",
]


def _require_preconditioner(operator: lx.AbstractLinearOperator) -> None:
    """Reject an operator that is not square or not tagged positive semidefinite."""
    if operator.in_size() != operator.out_size():
        msg = (
            "'operator' must be square, got "
            f"{operator.out_size()} by {operator.in_size()}"
        )
        raise ValidationError(msg)
    if not lx.is_positive_semidefinite(operator):
        msg = (
            "'operator' must carry lineax's positive_semidefinite_tag: a "
            "preconditioner that is not positive definite need not give a "
            "descent direction"
        )
        raise ValidationError(msg)


def scale_by_preconditioner(
    operator: lx.AbstractLinearOperator,
    *,
    solver: lx.AbstractLinearSolver | gx.AbstractSolveStrategy | None = None,
) -> optax.GradientTransformation:
    """Multiply the gradient by the inverse of a fixed operator.

        u = P⁻¹ g

    g ∈ ℝ^D_θ is the incoming update (the gradient of the cost), P the
    (D_θ, D_θ) symmetric positive definite preconditioner and u the outgoing
    update. Like optax's own ``scale_by_*`` transforms it applies neither a
    step size nor a sign: chain ``optax.scale_by_learning_rate`` after it, or
    use `preconditioned_sgd`.

    This is gradient descent in the coordinates z = Lᵀ θ, with P = L Lᵀ: the
    Hessian H of the cost becomes L⁻¹ H L⁻ᵀ there, with the eigenvalues of
    P⁻¹ H. A constant step size α then converges on a quadratic cost when
    α < 2 / λ_max(P⁻¹ H), fastest at α = 2 / (λ_min + λ_max), where the error
    shrinks by (κ − 1) / (κ + 1) a step with κ = λ_max / λ_min. So P is worth
    having when P⁻¹ H is better conditioned than H: a P that carries a
    correlated prior precision is, an identity is not.

    Algorithm:
        1. Factor P once, when the transform is built: the Cholesky factor of
           a dense matrix, or whatever the lineax ``solver`` prepares.
        2. At each step solve P u = g with that factor, or with gaussx's
           structural rules (a diagonal divides, a low-rank update uses the
           Woodbury identity).

    Args:
        operator: The preconditioner P, square and tagged
            ``lineax.positive_semidefinite_tag``.
        solver: How to solve with P. ``None`` factors a dense
            ``lineax.MatrixLinearOperator`` once by Cholesky and solves any
            other operator by its structure with ``gaussx.solve``. A lineax
            solver is initialised once on P as it is given, and reused. A
            gaussx solve strategy owns the solve at every step; an iterative
            one should run to a tight tolerance, since a loose solve makes u
            differ from P⁻¹ g from step to step.

    Returns:
        An ``optax.GradientTransformation`` with an empty state, for a flat
        parameter vector of length D_θ. The update keeps the gradient's dtype.

    Raises:
        ValidationError: If ``operator`` is not square or not tagged positive
            semidefinite; or, from ``init``, if the parameters are not one
            vector of the operator's size.
        MissingExtraError: If optax is not installed.

    Examples:
        A diagonal preconditioner divides the gradient elementwise. lineax
        does not tag a diagonal operator positive semidefinite on its own
        (its entries could be negative), so the tag is added:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from argminax.blr import scale_by_preconditioner
        >>> diagonal = lx.DiagonalLinearOperator(jnp.array([1.0, 4.0]))
        >>> operator = lx.TaggedLinearOperator(
        ...     diagonal, lx.positive_semidefinite_tag
        ... )
        >>> scaling = scale_by_preconditioner(operator)
        >>> state = scaling.init(jnp.zeros(2))
        >>> updates, state = scaling.update(jnp.array([2.0, 2.0]), state)
        >>> bool(jnp.allclose(updates, jnp.array([2.0, 0.5])))
        True

        It chains like any optax transform, here with a decaying step size:

        >>> import optax
        >>> schedule = optax.exponential_decay(0.5, 1, 0.5)
        >>> optimiser = optax.chain(
        ...     scale_by_preconditioner(operator),
        ...     optax.scale_by_learning_rate(schedule),
        ... )
        >>> state = optimiser.init(jnp.zeros(2))
        >>> updates, state = optimiser.update(jnp.array([2.0, 2.0]), state)
        >>> bool(jnp.allclose(updates, -0.5 * jnp.array([2.0, 0.5])))
        True

        The operator has to be tagged positive semidefinite:

        >>> untagged = lx.MatrixLinearOperator(jnp.eye(2))
        >>> scale_by_preconditioner(untagged)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'operator' must carry lineax's positive_semidefinite_tag
    """
    _require_preconditioner(operator)
    optax = require_optax()
    size = operator.in_size()
    working_dtype = operator.in_structure().dtype

    lineax_solver = solver
    if solver is None and isinstance(operator, lx.MatrixLinearOperator):
        lineax_solver = lx.Cholesky()
    if isinstance(lineax_solver, lx.AbstractLinearSolver):
        # factor P once; every step reuses the factor
        factor = lineax_solver.init(operator, {})

        def solve(vector: Float[Array, " state"]) -> Float[Array, " state"]:
            # u = P⁻¹ g
            solution = lx.linear_solve(operator, vector, lineax_solver, state=factor)
            return solution.value

    else:

        def solve(vector: Float[Array, " state"]) -> Float[Array, " state"]:
            # u = P⁻¹ g, by the structure of P or by the gaussx strategy
            return gx.solve(operator, vector, solver=solver)

    def init_fn(params: Float[Array, " state"]) -> optax.EmptyState:
        shape = jnp.shape(params)
        if shape != (size,):
            msg = (
                "the parameters must be one vector of the operator's size "
                f"({size},), got shape {shape}"
            )
            raise ValidationError(msg)
        return optax.EmptyState()

    def update_fn(
        updates: Float[Array, " state"],
        state: optax.EmptyState,
        params: Any = None,
    ) -> tuple[Float[Array, " state"], optax.EmptyState]:
        incoming = jnp.asarray(updates)
        # solve in the operator's dtype, and hand back the gradient's
        scaled = solve(incoming.astype(working_dtype))
        return scaled.astype(incoming.dtype), state

    return optax.GradientTransformation(init_fn, update_fn)


def preconditioned_sgd(
    learning_rate: StepSize,
    operator: lx.AbstractLinearOperator,
    *,
    solver: lx.AbstractLinearSolver | gx.AbstractSolveStrategy | None = None,
) -> optax.GradientTransformation:
    """Gradient descent preconditioned by a fixed operator.

        θ ← θ − α P⁻¹ g

    θ ∈ ℝ^D_θ is the parameter vector, g = ∇_θ 𝒥 the gradient of the cost 𝒥,
    P the (D_θ, D_θ) symmetric positive definite preconditioner and α the
    step size. With P the Hessian of a quadratic cost and α = 1 it is
    Newton's method and one step reaches the minimiser; with P the identity
    it is ``optax.sgd``. See `scale_by_preconditioner` for when a
    preconditioner helps and how the solve is done.

    A few steps of it on a 4D-Var cost, with P the Hessian of the 3D-Var
    cost, is the analysis of Solvik et al. (2025).

    Args:
        learning_rate: The step size α > 0, or an optax schedule of the step
            count.
        operator: The preconditioner P, square and tagged
            ``lineax.positive_semidefinite_tag``.
        solver: How to solve with P; see `scale_by_preconditioner`.

    Returns:
        An ``optax.GradientTransformation`` for a flat parameter vector of
        length D_θ.

    Raises:
        ValidationError: If a constant ``learning_rate`` is not positive, or
            ``operator`` is not square or not tagged positive semidefinite.
        MissingExtraError: If optax is not installed.

    Examples:
        One step of size 1, preconditioned by the Hessian A of the quadratic
        cost 𝒥(θ) = ½ θᵀ A θ − bᵀ θ, lands on its minimiser A⁻¹ b:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from argminax.blr import preconditioned_sgd
        >>> A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> b = jnp.array([1.0, -2.0])
        >>> hessian = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
        >>> optimiser = preconditioned_sgd(1.0, hessian)
        >>> theta = jnp.zeros(2)
        >>> state = optimiser.init(theta)
        >>> updates, state = optimiser.update(A @ theta - b, state)
        >>> bool(jnp.allclose(A @ (theta + updates), b, atol=1e-5))
        True

        A structured operator keeps its structure: a diagonal plus a rank-R
        factor is solved by the Woodbury identity, O(D_θ R²) a step.

        >>> import gaussx as gx
        >>> precision = gx.low_rank_plus_diag(
        ...     jnp.full(50, 2.0), jnp.ones((50, 1)), psd=True
        ... )
        >>> optimiser = preconditioned_sgd(0.1, precision)
        >>> state = optimiser.init(jnp.zeros(50))
        >>> updates, state = optimiser.update(jnp.ones(50), state)
        >>> updates.shape
        (50,)

    References:
        Solvik, Penny and Hoyer (2025), "4D-Var using Hessian approximation
        and backpropagation applied to automatically differentiable numerical
        and machine learning models", Journal of Advances in Modeling Earth
        Systems 17, e2024MS004608, doi:10.1029/2024MS004608.
    """
    if isinstance(learning_rate, (int, float)):
        require_positive(learning_rate, "learning_rate")
    optax = require_optax()
    return optax.chain(
        scale_by_preconditioner(operator, solver=solver),
        optax.scale_by_learning_rate(learning_rate),
    )

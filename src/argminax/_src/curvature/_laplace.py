"""The Gaussian at a minimum, 𝒩(x̂, Σ_x∣y), with its precision as a lineax operator.

`gauss_newton` builds it from whitened residuals, densely for a small state and
matrix-free above that; `push_forward` maps a posterior in whitened
coordinates back to physical ones.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from argminax.exceptions import ValidationError


__all__ = [
    "Laplace",
    "gauss_newton",
    "push_forward",
]


# Up to this many unknowns `gauss_newton` forms Λ_x∣y densely; the same threshold
# as gaussx.AutoSolver's.
_DENSE_THRESHOLD = 1000
_PSD_TAGS = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


def _tagged_diagonal(
    operator: lx.AbstractLinearOperator,
) -> lx.DiagonalLinearOperator | None:
    """The diagonal operator under any tags, or ``None`` if it is not one."""
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    return operator if isinstance(operator, lx.DiagonalLinearOperator) else None


class Laplace(eqx.Module):
    """The Gaussian 𝒩(x̂, Σ_x∣y) at a minimum, with Σ_x∣y = Λ_x∣y⁻¹ never formed.

        p(x ∣ y) ≈ 𝒩(x; x̂, Λ_x∣y⁻¹)

    x̂ is the mode and Λ_x∣y the posterior precision, a lineax operator tagged
    positive semidefinite whose input and output have the shape of x̂ (a
    vector of length D_x, or a gridded state such as (T, N_Ω) for T time steps
    and N_Ω grid cells). The covariance, its log-determinant and
    samples all go through ``solver``, a gaussx solve strategy, so the same
    posterior is computed the same way by every library that builds one.

    Attributes:
        mode: The mode x̂, of any array shape.
        precision: The posterior precision Λ_x∣y, a lineax operator on arrays
            of the shape of x̂, tagged positive semidefinite.
        solver: The gaussx strategy for solves with Λ_x∣y: `gaussx.DenseSolver`
            (the default), `gaussx.CGSolver`, or a lineax solver wrapped in
            `gaussx.LineaxSolver`. A mode that is not a vector needs a lineax
            solver or an iterative strategy, since gaussx's dense fallback
            assumes a vector.

    Raises:
        ValidationError: If the shapes of ``precision`` do not match
            ``mode``, or if it is not tagged positive semidefinite.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from argminax.curvature import Laplace
        >>> precision = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> posterior = Laplace(
        ...     jnp.zeros(2),
        ...     lx.TaggedLinearOperator(precision, lx.positive_semidefinite_tag),
        ... )
        >>> covariance = posterior.covariance().mv(jnp.ones(2))
        >>> bool(jnp.allclose(covariance, jnp.array([0.5, 0.25])))
        True

        A gridded mode keeps its shape, here with a lineax CG solver.

        >>> import gaussx as gx
        >>> import jax
        >>> grid = jax.ShapeDtypeStruct((2, 3), jnp.float32)
        >>> identity = lx.IdentityLinearOperator(grid)
        >>> gridded = Laplace(
        ...     jnp.zeros((2, 3), jnp.float32),
        ...     lx.TaggedLinearOperator(
        ...         2.0 * identity, lx.positive_semidefinite_tag
        ...     ),
        ...     gx.LineaxSolver(lx.CG(rtol=1e-6, atol=1e-6)),
        ... )
        >>> gridded.covariance().mv(jnp.ones((2, 3), jnp.float32)).shape
        (2, 3)

        A precision without the tag is rejected, since sampling and the
        default solves rely on it.

        >>> Laplace(jnp.zeros(2), precision)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'precision' must be tagged positive semidefinite, ...
    """

    mode: Float[Array, "*state"]
    precision: lx.AbstractLinearOperator
    solver: gx.AbstractSolveStrategy = gx.DenseSolver()

    def __check_init__(self) -> None:
        shape = jnp.shape(self.mode)
        in_shape = self.precision.in_structure().shape
        out_shape = self.precision.out_structure().shape
        if in_shape != shape or out_shape != shape:
            msg = (
                f"'precision' must map arrays of shape {shape}, the shape of "
                f"'mode', got {in_shape} to {out_shape}"
            )
            raise ValidationError(msg)
        if not lx.is_positive_semidefinite(self.precision):
            msg = (
                "'precision' must be tagged positive semidefinite, got tags "
                "without lx.positive_semidefinite_tag"
            )
            raise ValidationError(msg)

    def covariance(self) -> lx.AbstractLinearOperator:
        """Return the posterior covariance Σ_x∣y = Λ_x∣y⁻¹ as a lazy operator.

            Σ_x∣y v = Λ_x∣y⁻¹ v

        A diagonal Λ_x∣y (tagged PSD, as `Laplace` requires) is inverted in
        closed form with any strategy. Otherwise, with `gaussx.DenseSolver`
        on a vector mode, or a lineax solver in `gaussx.LineaxSolver`, this
        is `gaussx.inv`, which keeps the structure of Λ_x∣y (a low-rank
        update, a Kronecker product is inverted in closed form) and falls back
        to that solver. Any other strategy applies Σ_x∣y by one solve with
        Λ_x∣y per product.

        Returns:
            Σ_x∣y, a lazy operator tagged symmetric and positive semidefinite.
        """
        diagonal = _tagged_diagonal(self.precision)
        if diagonal is not None:
            # a diagonal Λ_x∣y, tagged PSD, is inverted in closed form: gaussx.inv
            # does not look through the tag, so it is peeled here
            return lx.TaggedLinearOperator(gx.inv(diagonal), _PSD_TAGS)
        if isinstance(self.solver, gx.LineaxSolver):
            return gx.inv(self.precision, solver=self.solver.solver)
        if isinstance(self.solver, gx.DenseSolver) and jnp.ndim(self.mode) == 1:
            return gx.inv(self.precision)
        return lx.FunctionLinearOperator(
            lambda vector: self.solver.solve(self.precision, vector),
            self.precision.in_structure(),
            _PSD_TAGS,
        )

    def logdet_precision(self) -> Float[Array, ""]:
        """Return the log-determinant of the precision, log |Λ_x∣y|.

            log |Λ_x∣y| = −log |Σ_x∣y|

        Exact with `gaussx.DenseSolver` or a solve-only strategy
        (`gaussx.logdet`); a stochastic estimate with `gaussx.CGSolver`, the
        default of `gauss_newton`'s matrix-free path.

        Returns:
            The scalar log |Λ_x∣y|.
        """
        if isinstance(self.solver, gx.AbstractSolverStrategy):
            return self.solver.logdet(self.precision)
        return gx.logdet(self.precision)

    def sample(self, n: int, *, key: PRNGKeyArray) -> Float[Array, " member *state"]:
        """Draw samples from 𝒩(x̂, Σ_x∣y) by a Cholesky factor of the precision.

            x⁽ᵉ⁾ = x̂ + L⁻ᵀ v⁽ᵉ⁾,   Λ_x∣y = L Lᵀ,   v⁽ᵉ⁾ ~ 𝒩(0, I)

        so Cov(x⁽ᵉ⁾) = L⁻ᵀ L⁻¹ = Λ_x∣y⁻¹, for e = 1, …, N_ℰ draws. The factor
        is dense, over the flattened
        state: for a state too large for that, use
        `sample_by_perturbation`.

        Args:
            n: Number of draws N_ℰ.
            key: Random key for the standard normal draws v⁽ᵉ⁾.

        Returns:
            The draws x⁽ᵉ⁾, shape (N_ℰ, *x̂.shape), in the dtype of ``mode``.
        """
        # Λ_x∣y = L Lᵀ over the flattened state, densifying a matrix-free one
        factor = gx.cholesky(self.precision)
        # Lᵀ, an operator, so the solve below stays triangular
        upper = factor.T
        size = jnp.size(self.mode)
        # v⁽ᵉ⁾ ~ 𝒩(0, I), one row per draw
        noise = jax.random.normal(key, (n, size), self.mode.dtype)

        def whitened_draw(
            standard: Float[Array, " state"],
        ) -> Float[Array, "*state"]:
            # reshape, not einx: the factor's structure and the mode's shape are
            # only known at trace time and may have any number of axes
            standard = standard.reshape(upper.in_structure().shape)
            # L⁻ᵀ v⁽ᵉ⁾, by a triangular solve with Lᵀ
            solution = lx.linear_solve(upper, standard, solver=lx.Triangular())
            solved = solution.value
            return solved.reshape(jnp.shape(self.mode))

        draws = jax.vmap(whitened_draw)(noise)
        # x⁽ᵉ⁾ = x̂ + L⁻ᵀ v⁽ᵉ⁾
        return self.mode + draws


def gauss_newton(
    residuals: Callable[[Float[Array, " state"], Any], PyTree[Array]],
    mode: Float[Array, " state"],
    args: Any = None,
    *,
    dense: bool | None = None,
    jac: Literal["fwd", "bwd"] = "fwd",
    solver: gx.AbstractSolveStrategy | None = None,
) -> Laplace:
    """Build the Laplace posterior at a minimum from whitened residuals.

        Λ_x∣y = J_rᵀ J_r,   J_r = ∂r/∂x at x̂,   𝒥(x) = ½ ‖r(x)‖²

    r(x) ∈ ℝ^D_r stacks the measurement and prior residuals, each whitened by
    its covariance, and J_r is its D_r × D_x Jacobian, so Λ_x∣y is the
    Gauss–Newton approximation of the Hessian of the cost 𝒥 and
    Σ_x∣y = Λ_x∣y⁻¹. Gauss–Newton drops the second-order term ∑ᵢ rᵢ ∇²rᵢ: exact
    for a linear forward model, fine at small residuals, optimistic (too
    small a covariance) at large ones.

    Algorithm:
        1. Dense (D_x up to 1000 by default): form J_r by `jax.jacfwd` (or
           `jax.jacrev` with ``jac="bwd"``), then Λ_x∣y = J_rᵀ J_r, symmetrised
           before it is tagged positive semidefinite; solves use
           `gaussx.DenseSolver`.
        2. Matrix-free: Λ_x∣y v = J_rᵀ (J_r v) by `gaussx.gauss_newton_operator`,
           never formed; solves use `gaussx.CGSolver`.

    A forward model behind ``jax.custom_vjp`` (a diffrax solve with its
    default adjoint) refuses forward mode: pass ``jac="bwd"``, which forms
    J_r from reverse passes on either path.

    Args:
        residuals: The whitened residuals r(x, args), returning a pytree.
        mode: The minimum x̂, shape (D_x,).
        args: Passed to ``residuals`` as its second argument.
        dense: Form Λ_x∣y densely (``True``), matrix-free (``False``), or by
            the size of x̂ (``None``).
        jac: How J_r is reached: ``"fwd"`` (Jacobian-vector products) or
            ``"bwd"`` (vector-Jacobian products).
        solver: The gaussx strategy for the posterior; default
            `gaussx.DenseSolver` when dense, `gaussx.CGSolver` when
            matrix-free, with rtol = atol = max(1e-10, 100 × machine
            epsilon) for the dtype of x̂ (1e-10 in float64, about 1e-5 in
            float32).

    Returns:
        The posterior 𝒩(x̂, Λ_x∣y⁻¹).

    Raises:
        ValidationError: If ``mode`` is not one-dimensional or ``jac`` is
            neither ``"fwd"`` nor ``"bwd"``.

    Examples:
        One unknown observed once with unit noise and a unit prior:
        r(x) = (y − x, x), so Λ_x∣y = 2 and Σ_x∣y = ½.

        >>> import jax.numpy as jnp
        >>> from argminax.curvature import gauss_newton
        >>> def residuals(x, y):
        ...     return y - x, x
        >>> posterior = gauss_newton(residuals, jnp.array([0.5]), jnp.array([1.0]))
        >>> bool(jnp.allclose(posterior.covariance().as_matrix(), 0.5))
        True

        The matrix-free path gives the same covariance.

        >>> matrix_free = gauss_newton(
        ...     residuals, jnp.array([0.5]), jnp.array([1.0]), dense=False
        ... )
        >>> bool(jnp.allclose(matrix_free.covariance().as_matrix(), 0.5))
        True

    References:
        Rodgers, C. D. (2000). Inverse Methods for Atmospheric Sounding,
        chapter 5. Nocedal, J. & Wright, S. J. (2006). Numerical
        Optimization, chapter 10.
    """
    if jnp.ndim(mode) != 1:
        msg = f"'mode' must be one-dimensional, got shape {jnp.shape(mode)}"
        raise ValidationError(msg)
    if jac not in ("fwd", "bwd"):
        msg = f"'jac' must be 'fwd' or 'bwd', got {jac!r}"
        raise ValidationError(msg)
    state_size = jnp.shape(mode)[0]
    use_dense = state_size <= _DENSE_THRESHOLD if dense is None else dense

    def flat_residuals(state: Float[Array, " state"]) -> Float[Array, " residual"]:
        residual_tree = residuals(state, args)
        flat, _ = ravel_pytree(residual_tree)
        return flat

    if not use_dense:
        # Λ_x∣y v = J_rᵀ (J_r v), PSD-tagged by gaussx
        precision = gx.gauss_newton_operator(flat_residuals, mode, jac=jac)
        machine_epsilon = float(jnp.finfo(mode.dtype).eps)
        # tol = max(10⁻¹⁰, 100 × machine epsilon): 1e-10 in float64, ~1e-5 in f32
        tolerance = max(1e-10, 100.0 * machine_epsilon)
        strategy = (
            gx.CGSolver(rtol=tolerance, atol=tolerance) if solver is None else solver
        )
        return Laplace(mode, precision, strategy)

    differentiate = jax.jacfwd if jac == "fwd" else jax.jacrev
    # J_r = ∂r/∂x at x̂
    jacobian = differentiate(flat_residuals)(mode)
    # J_rᵀ J_r
    gram = einx.dot("residual row, residual col -> row col", jacobian, jacobian)
    # Λ_x∣y = ½ (J_rᵀ J_r + (J_rᵀ J_r)ᵀ), symmetrised so the PSD tag is exact
    matrix = gx.symmetrize(gram)
    precision = lx.MatrixLinearOperator(matrix, _PSD_TAGS)
    strategy = gx.DenseSolver() if solver is None else solver
    return Laplace(mode, precision, strategy)


def push_forward(
    posterior: Laplace,
    prior_sqrt: lx.AbstractLinearOperator,
    prior_mean: Float[Array, " state"] | None = None,
) -> Laplace:
    """Map a posterior in whitened coordinates back to physical ones.

        x = μ_x + L_x ξ,   x̂ = μ_x + L_x ξ̂,
        Σ_x∣y = L_x Σ_ξ∣y L_xᵀ,   Λ_x∣y = L_x⁻ᵀ Λ_ξ∣y L_x⁻¹

    A retrieval in whitened coordinates ξ ∈ ℝ^D_x has the prior 𝒩(0, I), and
    its posterior 𝒩(ξ̂, Σ_ξ∣y) has precision Λ_ξ∣y. With the prior covariance
    Σ_x = L_x L_xᵀ (D_x × D_x) and prior mean μ_x, the physical state is
    x = μ_x + L_x ξ, so the posterior over x has mode μ_x + L_x ξ̂ and
    precision L_x⁻ᵀ Λ_ξ∣y L_x⁻¹, applied by two solves with L_x around a
    product with Λ_ξ∣y.

    Algorithm:
        1. Map the mode: x̂ = μ_x + L_x ξ̂.
        2. Apply the precision matrix-free: Λ_x∣y v = L_x⁻ᵀ Λ_ξ∣y L_x⁻¹ v, by
           a solve with L_x, a product with Λ_ξ∣y and a solve with L_xᵀ.

    Args:
        posterior: The posterior over ξ.
        prior_sqrt: A square root L_x of the prior covariance Σ_x, (D_x, D_x).
        prior_mean: The prior mean μ_x; default ``None``, zero.

    Returns:
        The posterior over x, with the same solver strategy.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from argminax.curvature import gauss_newton, push_forward
        >>> def residuals(xi, y):
        ...     return y - 2.0 * xi, xi
        >>> whitened = gauss_newton(residuals, jnp.array([0.4]), jnp.array([1.0]))
        >>> physical = push_forward(
        ...     whitened,
        ...     lx.DiagonalLinearOperator(jnp.array([2.0])),
        ...     jnp.array([1.0]),
        ... )
        >>> bool(jnp.allclose(physical.mode, 1.8))
        True
        >>> bool(jnp.allclose(physical.covariance().as_matrix(), 4.0 / 5.0))
        True
    """
    # L_x ξ̂
    mode = prior_sqrt.mv(posterior.mode)
    if prior_mean is not None:
        # x̂ = μ_x + L_x ξ̂
        mode = prior_mean + mode
    # L_xᵀ, an operator, so each solve below stays structured
    transposed_sqrt = prior_sqrt.T

    def physical_precision(vector: Float[Array, " state"]) -> Float[Array, " state"]:
        # L_x⁻¹ v
        whitened = lx.linear_solve(prior_sqrt, vector).value
        # Λ_ξ∣y L_x⁻¹ v
        curved = posterior.precision.mv(whitened)
        # L_x⁻ᵀ Λ_ξ∣y L_x⁻¹ v
        solution = lx.linear_solve(transposed_sqrt, curved)
        return solution.value

    precision = lx.FunctionLinearOperator(
        physical_precision, posterior.precision.in_structure(), _PSD_TAGS
    )
    return Laplace(mode, precision, posterior.solver)

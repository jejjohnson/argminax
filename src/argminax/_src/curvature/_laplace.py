"""The Gaussian at a minimum, N(x̂, Ŝ), with its precision as a lineax operator.

`gauss_newton` builds it from whitened residuals, densely for a small state and
matrix-free above that; `push_forward` maps a posterior in whitened
coordinates back to physical ones.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

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


# Up to this many unknowns `gauss_newton` forms Ĥ densely; the same threshold
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
    """The Gaussian N(x̂, Ŝ) at a minimum, with Ŝ = Ĥ⁻¹ never formed.

        p(x) ≈ N(x; x̂, Ĥ⁻¹)

    x̂ is the mode and Ĥ the precision, a lineax operator tagged positive
    semidefinite whose input and output have the shape of x̂ (a vector, or a
    gridded state such as (T, N)). The covariance, its log-determinant and
    samples all go through ``solver``, a gaussx solve strategy, so the same
    posterior is computed the same way by every library that builds one.

    Attributes:
        mode: The mode x̂, of any array shape.
        precision: The precision Ĥ, a lineax operator on arrays of the shape
            of x̂, tagged positive semidefinite.
        solver: The gaussx strategy for solves with Ĥ: `gaussx.DenseSolver`
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
        """Return the covariance Ŝ = Ĥ⁻¹ as a lazy operator.

            Ŝ v = Ĥ⁻¹ v

        A diagonal Ĥ (tagged PSD, as `Laplace` requires) is inverted in
        closed form with any strategy. Otherwise, with `gaussx.DenseSolver`
        on a vector mode, or a lineax solver in `gaussx.LineaxSolver`, this
        is `gaussx.inv`, which keeps the structure of Ĥ (a low-rank update, a
        Kronecker product is inverted in closed form) and falls back to that
        solver. Any other strategy applies Ŝ by one solve with Ĥ per product.

        Returns:
            Ŝ, a lazy operator tagged symmetric and positive semidefinite.
        """
        diagonal = _tagged_diagonal(self.precision)
        if diagonal is not None:
            # a diagonal Ĥ, tagged PSD, is inverted in closed form: gaussx.inv
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
        """Return the log-determinant of the precision, log |Ĥ|.

            log |Ĥ| = −log |Ŝ|

        Exact with `gaussx.DenseSolver` or a solve-only strategy
        (`gaussx.logdet`); a stochastic estimate with `gaussx.CGSolver`, the
        default of `gauss_newton`'s matrix-free path.

        Returns:
            The scalar log |Ĥ|.
        """
        if isinstance(self.solver, gx.AbstractSolverStrategy):
            return self.solver.logdet(self.precision)
        return gx.logdet(self.precision)

    def sample(self, n: int, *, key: PRNGKeyArray) -> Float[Array, "n ..."]:
        """Draw samples from N(x̂, Ŝ) by a Cholesky factor of the precision.

            x = x̂ + L⁻ᵀ z,   Ĥ = L Lᵀ,   z ~ N(0, I)

        so Cov(x) = L⁻ᵀ L⁻¹ = Ĥ⁻¹. The factor is dense, over the flattened
        state: for a state too large for that, use
        `sample_by_perturbation`.

        Args:
            n: Number of draws.
            key: Random key for z.

        Returns:
            The draws, shape (n, *x̂.shape), in the dtype of ``mode``.
        """
        # Ĥ = L Lᵀ over the flattened state, densifying a matrix-free precision
        factor = gx.cholesky(self.precision)
        upper = factor.T
        size = jnp.size(self.mode)
        noise = jax.random.normal(key, (n, size), self.mode.dtype)

        def whitened_draw(standard: Float[Array, " size"]) -> Float[Array, "*state"]:
            # L⁻ᵀ z, by a triangular solve with Lᵀ
            standard = standard.reshape(upper.in_structure().shape)
            solved = lx.linear_solve(upper, standard, solver=lx.Triangular()).value
            return solved.reshape(jnp.shape(self.mode))

        return self.mode + jax.vmap(whitened_draw)(noise)


def gauss_newton(
    residuals: Callable[[Float[Array, " N"], Any], PyTree[Array]],
    mode: Float[Array, " N"],
    args: Any = None,
    *,
    dense: bool | None = None,
    jac: Literal["fwd", "bwd"] = "fwd",
    solver: gx.AbstractSolveStrategy | None = None,
) -> Laplace:
    """Build the Laplace posterior at a minimum from whitened residuals.

        Ĥ = J_rᵀ J_r,   J_r = ∂r/∂x at x̂,   J(x) = ½ ‖r(x)‖²

    r stacks the measurement and prior residuals, each whitened by its
    covariance, so Ĥ is the Gauss–Newton approximation of the Hessian of J
    and Ŝ = Ĥ⁻¹. Gauss–Newton drops the second-order term ∑ᵢ rᵢ ∇²rᵢ: exact
    for a linear forward model, fine at small residuals, optimistic (too
    small a covariance) at large ones.

    Algorithm:
        1. Dense (N up to 1000 by default): form J_r by `jax.jacfwd` (or
           `jax.jacrev` with ``jac="bwd"``), then Ĥ = J_rᵀ J_r, symmetrised
           before it is tagged positive semidefinite; solves use
           `gaussx.DenseSolver`.
        2. Matrix-free: Ĥ v = J_rᵀ (J_r v) by `gaussx.gauss_newton_operator`,
           never formed; solves use `gaussx.CGSolver`.

    A forward model behind ``jax.custom_vjp`` (a diffrax solve with its
    default adjoint) refuses forward mode: pass ``jac="bwd"``, which forms
    J_r from reverse passes on either path.

    Args:
        residuals: The whitened residuals r(x, args), returning a pytree.
        mode: The minimum x̂, shape (N,).
        args: Passed to ``residuals`` as its second argument.
        dense: Form Ĥ densely (``True``), matrix-free (``False``), or by
            the size of x̂ (``None``).
        jac: How J_r is reached: ``"fwd"`` (Jacobian-vector products) or
            ``"bwd"`` (vector-Jacobian products).
        solver: The gaussx strategy for the posterior; default
            `gaussx.DenseSolver` when dense, `gaussx.CGSolver` when
            matrix-free, with rtol = atol = max(1e-10, 100 ε_mach) for the
            dtype of x̂ (1e-10 in float64, about 1e-5 in float32).

    Returns:
        The posterior N(x̂, Ĥ⁻¹).

    Raises:
        ValidationError: If ``mode`` is not one-dimensional or ``jac`` is
            neither ``"fwd"`` nor ``"bwd"``.

    Examples:
        One unknown observed once with unit noise and a unit prior:
        r(x) = (y − x, x), so Ĥ = 2 and Ŝ = ½.

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
    use_dense = jnp.shape(mode)[0] <= _DENSE_THRESHOLD if dense is None else dense

    def flat_residuals(state: Float[Array, " N"]) -> Float[Array, " M"]:
        return ravel_pytree(residuals(state, args))[0]

    if not use_dense:
        # Ĥ v = J_rᵀ (J_r v), PSD-tagged by gaussx
        precision = gx.gauss_newton_operator(flat_residuals, mode, jac=jac)
        # 1e-10 in float64, 100 ε_mach (about 1e-5) in float32
        tolerance = max(1e-10, 100.0 * float(jnp.finfo(mode.dtype).eps))
        strategy = (
            gx.CGSolver(rtol=tolerance, atol=tolerance) if solver is None else solver
        )
        return Laplace(mode, precision, strategy)

    differentiate = jax.jacfwd if jac == "fwd" else jax.jacrev
    jacobian = differentiate(flat_residuals)(mode)
    # Ĥ = J_rᵀ J_r, symmetrised so the PSD tag is exact
    matrix = gx.symmetrize(jacobian.T @ jacobian)
    precision = lx.MatrixLinearOperator(matrix, _PSD_TAGS)
    return Laplace(mode, precision, gx.DenseSolver() if solver is None else solver)


def push_forward(
    posterior: Laplace,
    prior_sqrt: lx.AbstractLinearOperator,
    prior_mean: Float[Array, " N"] | None = None,
) -> Laplace:
    """Map a posterior in whitened coordinates back to physical ones.

        x = xₐ + Lₐ ξ,   x̂ = xₐ + Lₐ ξ̂,   Pₓ = Lₐ Ŝ Lₐᵀ,   Hₓ = Lₐ⁻ᵀ Ĥ Lₐ⁻¹

    A retrieval in whitened coordinates ξ has the prior N(0, I); with
    Sₐ = Lₐ Lₐᵀ the physical state is x = xₐ + Lₐ ξ, so the posterior over x
    has mode xₐ + Lₐ ξ̂ and precision Lₐ⁻ᵀ Ĥ Lₐ⁻¹, applied by two solves
    with Lₐ around a product with Ĥ.

    Args:
        posterior: The posterior over ξ.
        prior_sqrt: A square root Lₐ of the prior covariance, (N, N).
        prior_mean: The prior mean xₐ; default ``None``, zero.

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
    # x̂ = xₐ + Lₐ ξ̂
    mode = prior_sqrt.mv(posterior.mode)
    if prior_mean is not None:
        mode = prior_mean + mode
    transposed_sqrt = prior_sqrt.T

    def physical_precision(vector: Float[Array, " N"]) -> Float[Array, " N"]:
        # Lₐ⁻¹ v
        whitened = lx.linear_solve(prior_sqrt, vector).value
        # Lₐ⁻ᵀ Ĥ Lₐ⁻¹ v
        return lx.linear_solve(transposed_sqrt, posterior.precision.mv(whitened)).value

    precision = lx.FunctionLinearOperator(
        physical_precision, posterior.precision.in_structure(), _PSD_TAGS
    )
    return Laplace(mode, precision, posterior.solver)

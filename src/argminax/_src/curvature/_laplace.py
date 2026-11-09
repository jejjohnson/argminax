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


class Laplace(eqx.Module):
    """The Gaussian N(x̂, Ŝ) at a minimum, with Ŝ = Ĥ⁻¹ never formed.

        p(x) ≈ N(x; x̂, Ĥ⁻¹)

    x̂ is the mode and Ĥ the precision, a lineax operator tagged positive
    semidefinite. The covariance, its log-determinant and samples all go
    through ``solver``, a gaussx solver strategy, so the same posterior is
    computed the same way by every library that builds one.

    Attributes:
        mode: The mode x̂, shape (N,).
        precision: The precision Ĥ, an (N, N) lineax operator tagged
            positive semidefinite.
        solver: The gaussx strategy for solves with Ĥ and its
            log-determinant.

    Raises:
        ValidationError: If ``mode`` is not one-dimensional, if ``precision``
            is not N × N, or if it is not tagged positive semidefinite.

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

        A precision without the tag is rejected, since sampling and the
        default solves rely on it.

        >>> Laplace(jnp.zeros(2), precision)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'precision' must be tagged positive semidefinite, ...
    """

    mode: Float[Array, " N"]
    precision: lx.AbstractLinearOperator
    solver: gx.AbstractSolverStrategy = gx.DenseSolver()

    def __check_init__(self) -> None:
        if jnp.ndim(self.mode) != 1:
            msg = f"'mode' must be one-dimensional, got shape {jnp.shape(self.mode)}"
            raise ValidationError(msg)
        size = jnp.shape(self.mode)[0]
        if self.precision.in_size() != size or self.precision.out_size() != size:
            msg = (
                f"'precision' must be {size} × {size} to match 'mode', got "
                f"{self.precision.out_size()} × {self.precision.in_size()}"
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

        Each matrix-vector product is one solve with Ĥ through ``solver``.

        Returns:
            Ŝ, an (N, N) operator tagged symmetric and positive semidefinite.
        """
        return lx.FunctionLinearOperator(
            lambda vector: self.solver.solve(self.precision, vector),
            self.precision.in_structure(),
            _PSD_TAGS,
        )

    def logdet_precision(self) -> Float[Array, ""]:
        """Return the log-determinant of the precision, log |Ĥ|.

            log |Ĥ| = −log |Ŝ|

        Exact with the default `gaussx.DenseSolver`; a stochastic estimate
        with `gaussx.CGSolver`, the default of `gauss_newton`'s matrix-free
        path.

        Returns:
            The scalar log |Ĥ|.
        """
        return self.solver.logdet(self.precision)

    def sample(self, n: int, *, key: PRNGKeyArray) -> Float[Array, "n N"]:
        """Draw samples from N(x̂, Ŝ) by a Cholesky factor of the precision.

            x = x̂ + L⁻ᵀ z,   Ĥ = L Lᵀ,   z ~ N(0, I)

        so Cov(x) = L⁻ᵀ L⁻¹ = Ĥ⁻¹. The factor is dense: for a state too large
        for that, use `sample_by_perturbation`.

        Args:
            n: Number of draws.
            key: Random key for z.

        Returns:
            The draws, shape (n, N), in the dtype of ``mode``.
        """
        # Ĥ = L Lᵀ, densifying a matrix-free precision
        factor = gx.cholesky(self.precision)
        noise = jax.random.normal(key, (n, *jnp.shape(self.mode)), self.mode.dtype)

        def whitened_draw(standard: Float[Array, " N"]) -> Float[Array, " N"]:
            # L⁻ᵀ z, by a triangular solve with Lᵀ
            return lx.linear_solve(factor.T, standard, solver=lx.Triangular()).value

        return self.mode + jax.vmap(whitened_draw)(noise)


def gauss_newton(
    residuals: Callable[[Float[Array, " N"], Any], PyTree[Array]],
    mode: Float[Array, " N"],
    args: Any = None,
    *,
    dense: bool | None = None,
    jac: Literal["fwd", "bwd"] = "fwd",
    solver: gx.AbstractSolverStrategy | None = None,
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
            `gaussx.DenseSolver` when dense, `gaussx.CGSolver` with
            rtol = atol = 1e-10 when matrix-free.

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
        strategy = gx.CGSolver(rtol=1e-10, atol=1e-10) if solver is None else solver
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

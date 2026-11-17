"""Variances, a low-rank form and samples of a Laplace posterior, without forming Ŝ.

Each needs only products and solves with the precision Ĥ, so they work at
state sizes where Ŝ cannot be stored; randomness comes only through keys.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from argminax._src.curvature._laplace import Laplace
from argminax.exceptions import ValidationError


__all__ = [
    "VarianceEstimate",
    "low_rank",
    "marginal_variance",
    "sample_by_perturbation",
]


def _require_vector_mode(posterior: Laplace) -> None:
    if jnp.ndim(posterior.mode) != 1:
        msg = (
            "'posterior' must have a one-dimensional mode, got shape "
            f"{jnp.shape(posterior.mode)}; flatten the state first"
        )
        raise ValidationError(msg)


def _require_count(value: int, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"'{name}' must be an integer of at least {minimum}, got {value!r}"
        raise ValidationError(msg)


class VarianceEstimate(eqx.Module):
    """A probed estimate of the marginal variances, with its standard error.

    Attributes:
        estimate: The estimated diag(Ŝ), shape (N,).
        stderr: The standard error of each entry, shape (N,).
    """

    estimate: Float[Array, " state"]
    stderr: Float[Array, " state"]


def marginal_variance(
    posterior: Laplace, *, num_probes: int = 64, key: PRNGKeyArray | None = None
) -> VarianceEstimate | Float[Array, " state"]:
    """Return the marginal variances diag(Ŝ) of a Laplace posterior.

        diag(Ŝ) ≈ (1/K) ∑ₖ zₖ ⊙ Ŝ zₖ,   zₖ Rademacher

    Without a key the diagonal is exact, from one solve with Ĥ per unknown
    (`gaussx.diag_inv` with ``method="solve"``). With a key it is probed:
    each Rademacher probe zₖ costs one solve, and the result carries the
    standard error of the K probes. A solve stopped at a loose tolerance
    biases every probe the same way, so set the solver's tolerance below the
    standard error you need.

    Algorithm:
        1. Without a key: diag(Ŝ) exactly, by one solve with Ĥ per unknown.
        2. With a key: draw K Rademacher probes zₖ, form zₖ ⊙ Ĥ⁻¹ zₖ by one
           solve each, and return their mean and its standard error.

    Args:
        posterior: The posterior N(x̂, Ĥ⁻¹).
        num_probes: Number of probes K, used only with a key.
        key: Random key for the probes; ``None`` for the exact diagonal.

    Returns:
        diag(Ŝ), shape (N,), without a key; a `VarianceEstimate` with one.

    Raises:
        ValidationError: If the mode is not one-dimensional, or
            ``num_probes`` is below 2.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from argminax.curvature import gauss_newton, marginal_variance
        >>> def residuals(x, y):
        ...     return y - x, x
        >>> posterior = gauss_newton(residuals, jnp.zeros(3), jnp.ones(3))
        >>> bool(jnp.allclose(marginal_variance(posterior), 0.5))
        True

        A diagonal Ŝ is probed exactly by Rademacher probes, so the standard
        error vanishes.

        >>> probed = marginal_variance(posterior, key=jax.random.key(0))
        >>> bool(jnp.allclose(probed.estimate, 0.5))
        True
        >>> bool(jnp.allclose(probed.stderr, 0.0))
        True
    """
    _require_vector_mode(posterior)
    if key is None:
        return gx.diag_inv(posterior.precision, method="solve", solver=posterior.solver)
    _require_count(num_probes, "num_probes", minimum=2)
    size = jnp.shape(posterior.mode)[0]
    # zₖ Rademacher, one row per probe
    probes = jax.random.rademacher(key, (num_probes, size), posterior.mode.dtype)

    def probe_term(probe: Float[Array, " state"]) -> Float[Array, " state"]:
        # Ŝ z = Ĥ⁻¹ z, by one solve
        covariance_probe = posterior.solver.solve(posterior.precision, probe)
        # z ⊙ Ŝ z
        return probe * covariance_probe

    terms = jax.vmap(probe_term)(probes)

    # --- mean and standard error over the K probes ---
    # diag(Ŝ) ≈ (1/K) ∑ₖ zₖ ⊙ Ŝ zₖ
    estimate = einx.mean("member state -> state", terms)
    # tₖ − t̄
    deviations = einx.subtract("member state, state -> member state", terms, estimate)
    # ∑ₖ (tₖ − t̄)²
    squared_deviations = einx.sum("member state -> state", deviations**2)
    # s² = ∑ₖ (tₖ − t̄)² / (K − 1), the unbiased sample variance
    sample_variance = squared_deviations / (num_probes - 1)
    # SE = s / √K
    stderr = jnp.sqrt(sample_variance) / jnp.sqrt(num_probes)
    return VarianceEstimate(estimate=estimate, stderr=stderr)


def low_rank(posterior: Laplace, rank: int, *, key: PRNGKeyArray) -> gx.LowRankUpdate:
    """Return a low-rank form of the covariance of a whitened posterior.

        Ĥ = I + V diag(μ − 1) Vᵀ + …,   Ŝ ≈ I − V diag((μ − 1)/μ) Vᵀ

    In whitened coordinates (prior N(0, I), argminax's convention) the
    precision is the identity plus the data's curvature, so the R largest
    eigenpairs (μᵢ, vᵢ) of Ĥ carry everything the measurements taught, and
    Ŝ is the prior I less the variance they removed. The form is exact when
    the data's curvature has rank R or less (at most the number of
    measurements), and leaves the prior variance in every direction it
    misses.

    Algorithm:
        1. The R largest eigenpairs of Ĥ by `gaussx.randomized_eigh`
           (Rayleigh–Ritz with oversampling and power iterations), from
           products with Ĥ only.
        2. Ŝ ≈ I + V diag(dᵢ) Vᵀ with dᵢ = −(μᵢ − 1)/μᵢ, as a
           `gaussx.LowRankUpdate` of the identity with orthonormal V.

    Args:
        posterior: A posterior in whitened coordinates, whose precision has
            every eigenvalue at least 1.
        rank: Number of eigenpairs R, from 1 to N.
        key: Random key for the test matrix.

    Returns:
        Ŝ as an (N, N) `gaussx.LowRankUpdate`, tagged symmetric and positive
        semidefinite.

    Raises:
        ValidationError: If the mode is not one-dimensional, or ``rank`` is
            not an integer from 1 to N.

    Examples:
        Three unknowns, one measurement of their sum with unit noise:
        Ĥ = I + 1 1ᵀ has one eigenvalue above 1, so rank 1 is exact.

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from argminax.curvature import gauss_newton, low_rank
        >>> def residuals(x, y):
        ...     return y - jnp.sum(x, keepdims=True), x
        >>> posterior = gauss_newton(residuals, jnp.zeros(3), jnp.ones(1))
        >>> covariance = low_rank(posterior, 1, key=jax.random.key(0))
        >>> exact = posterior.covariance().as_matrix()
        >>> bool(jnp.allclose(covariance.as_matrix(), exact))
        True

    References:
        Bui-Thanh, T., Ghattas, O., Martin, J. & Stadler, G. (2013). A
        computational framework for infinite-dimensional Bayesian inverse
        problems. Part I: the linearized case. SIAM J. Sci. Comput. 35(6),
        A2494–A2523. arXiv:1308.1313.
    """
    _require_vector_mode(posterior)
    size = jnp.shape(posterior.mode)[0]
    _require_count(rank, "rank", minimum=1)
    if rank > size:
        msg = f"'rank' must be at most the state size {size}, got {rank}"
        raise ValidationError(msg)
    # the R largest eigenpairs (μᵢ, vᵢ) of Ĥ, from products with Ĥ only
    eigenvalues, eigenvectors = gx.randomized_eigh(posterior.precision, rank, key=key)
    # dᵢ = −(μᵢ − 1)/μᵢ, the variance the data removed along vᵢ
    removed = -(eigenvalues - 1.0) / eigenvalues
    identity = lx.IdentityLinearOperator(posterior.precision.in_structure())
    # Ŝ ≈ I + V diag(dᵢ) Vᵀ
    return gx.LowRankUpdate(
        identity,
        eigenvectors,
        removed,
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
        orthonormal=True,
    )


def sample_by_perturbation(
    residuals: Callable[[Float[Array, " state"], Any], PyTree[Array]],
    posterior: Laplace,
    n: int,
    args: Any = None,
    *,
    key: PRNGKeyArray,
) -> Float[Array, " member state"]:
    """Draw from a Gauss–Newton posterior by perturbing its residuals.

        Ĥ δ = J_rᵀ z,   z ~ N(0, I_M),   Cov(δ) = Ĥ⁻¹ J_rᵀ J_r Ĥ⁻¹ = Ĥ⁻¹

    Each draw is x̂ + δ, one product with J_rᵀ and one solve with Ĥ, so Ŝ
    and a square root of it are never formed. The covariance is exact
    because the prior rows are part of J_r, so Ĥ = J_rᵀ J_r: ``posterior``
    must come from `gauss_newton` on the same ``residuals`` and ``args``. An
    inexact solve biases every draw; set the solver's tolerance accordingly.

    Algorithm:
        1. Linearise the residuals at x̂ once, keeping the pullback v ↦ J_rᵀ v.
        2. For each draw, z ~ N(0, I_M), then δ = Ĥ⁻¹ J_rᵀ z by one pullback
           and one solve, and return x̂ + δ.

    Args:
        residuals: The whitened residuals r(x, args) the posterior was built
            from.
        posterior: The posterior N(x̂, Ĥ⁻¹).
        n: Number of draws.
        args: Passed to ``residuals`` as its second argument.
        key: Random key for z.

    Returns:
        The draws, shape (n, N).

    Raises:
        ValidationError: If ``n`` is not a positive integer.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from argminax.curvature import gauss_newton, sample_by_perturbation
        >>> def residuals(x, y):
        ...     return y - x, x
        >>> posterior = gauss_newton(residuals, jnp.zeros(2), jnp.ones(2))
        >>> draws = sample_by_perturbation(
        ...     residuals, posterior, 5, jnp.ones(2), key=jax.random.key(0)
        ... )
        >>> draws.shape
        (5, 2)

    References:
        Bardsley, J. M. et al. (2014). Randomize-then-optimize: a method
        for sampling from posterior distributions in nonlinear inverse
        problems. SIAM J. Sci. Comput. 36(4), A1895–A1910.
        doi:10.1137/140964023.
    """
    _require_count(n, "n", minimum=1)

    def flat_residuals(state: Float[Array, " state"]) -> Float[Array, " residual"]:
        residual_tree = residuals(state, args)
        flat, _ = ravel_pytree(residual_tree)
        return flat

    # r(x̂), and the pullback v ↦ J_rᵀ v at x̂
    residual_at_mode, pullback = jax.vjp(flat_residuals, posterior.mode)
    residual_size = residual_at_mode.shape[0]
    # z ~ N(0, I_M), one row per draw
    noise = jax.random.normal(key, (n, residual_size), residual_at_mode.dtype)

    def perturbation(standard: Float[Array, " residual"]) -> Float[Array, " state"]:
        # J_rᵀ z
        (pulled,) = pullback(standard)
        # δ = Ĥ⁻¹ J_rᵀ z
        return posterior.solver.solve(posterior.precision, pulled)

    perturbations = jax.vmap(perturbation)(noise)
    # x = x̂ + δ
    return posterior.mode + perturbations

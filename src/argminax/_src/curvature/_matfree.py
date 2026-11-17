"""Variances, a low-rank form and samples of a Laplace posterior, without Σ_x∣y.

Each needs only products and solves with the posterior precision Λ_x∣y, so
they work at state sizes D_x where the posterior covariance Σ_x∣y cannot be
stored; randomness comes only through keys.
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
        estimate: The estimated diag(Σ_x∣y), shape (D_x,).
        stderr: The standard error of each entry, shape (D_x,).
    """

    estimate: Float[Array, " state"]
    stderr: Float[Array, " state"]


def marginal_variance(
    posterior: Laplace, *, num_probes: int = 64, key: PRNGKeyArray | None = None
) -> VarianceEstimate | Float[Array, " state"]:
    """Return the marginal variances diag(Σ_x∣y) of a Laplace posterior.

        diag(Σ_x∣y) ≈ (1/N_ℰ) ∑ₑ v⁽ᵉ⁾ ⊙ Σ_x∣y v⁽ᵉ⁾,   v⁽ᵉ⁾ Rademacher

    Without a key the diagonal is exact, from one solve with Λ_x∣y per unknown
    (`gaussx.diag_inv` with ``method="solve"``). With a key it is probed:
    each Rademacher probe v⁽ᵉ⁾ ∈ ℝ^D_x costs one solve, and the result carries
    the standard error of the N_ℰ probes. A solve stopped at a loose tolerance
    biases every probe the same way, so set the solver's tolerance below the
    standard error you need.

    Algorithm:
        1. Without a key: diag(Σ_x∣y) exactly, by one solve with Λ_x∣y per
           unknown.
        2. With a key: draw N_ℰ Rademacher probes v⁽ᵉ⁾, form
           v⁽ᵉ⁾ ⊙ Λ_x∣y⁻¹ v⁽ᵉ⁾ by one solve each, and return their mean and
           its standard error.

    Args:
        posterior: The posterior 𝒩(x̂, Λ_x∣y⁻¹).
        num_probes: Number of probes N_ℰ, used only with a key.
        key: Random key for the probes; ``None`` for the exact diagonal.

    Returns:
        diag(Σ_x∣y), shape (D_x,), without a key; a `VarianceEstimate` with
        one.

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

        A diagonal Σ_x∣y is probed exactly by Rademacher probes, so the standard
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
    # v⁽ᵉ⁾ Rademacher, one row per probe
    probes = jax.random.rademacher(key, (num_probes, size), posterior.mode.dtype)

    def probe_term(probe: Float[Array, " state"]) -> Float[Array, " state"]:
        # Σ_x∣y v⁽ᵉ⁾ = Λ_x∣y⁻¹ v⁽ᵉ⁾, by one solve
        covariance_probe = posterior.solver.solve(posterior.precision, probe)
        # w⁽ᵉ⁾ = v⁽ᵉ⁾ ⊙ Σ_x∣y v⁽ᵉ⁾
        return probe * covariance_probe

    terms = jax.vmap(probe_term)(probes)

    # --- mean and standard error over the N_ℰ probes ---
    # diag(Σ_x∣y) ≈ w̄ = (1/N_ℰ) ∑ₑ w⁽ᵉ⁾
    estimate = einx.mean("member state -> state", terms)
    # w⁽ᵉ⁾ − w̄
    deviations = einx.subtract("member state, state -> member state", terms, estimate)
    # ∑ₑ (w⁽ᵉ⁾ − w̄)²
    squared_deviations = einx.sum("member state -> state", deviations**2)
    # σ̂² = ∑ₑ (w⁽ᵉ⁾ − w̄)² / (N_ℰ − 1), the unbiased sample variance
    sample_variance = squared_deviations / (num_probes - 1)
    # SE = σ̂ / √N_ℰ
    stderr = jnp.sqrt(sample_variance) / jnp.sqrt(num_probes)
    return VarianceEstimate(estimate=estimate, stderr=stderr)


def low_rank(posterior: Laplace, rank: int, *, key: PRNGKeyArray) -> gx.LowRankUpdate:
    """Return a low-rank form of the covariance of a whitened posterior.

        Λ_x∣y = I + Φ diag(γ − 1) Φᵀ + …,   Σ_x∣y ≈ I − Φ diag((γ − 1)/γ) Φᵀ

    In whitened coordinates (prior 𝒩(0, I), argminax's convention) the
    precision is the identity plus the data's curvature, so the R largest
    eigenpairs (γ_d, φ_d) of Λ_x∣y, stacked as Φ ∈ ℝ^(D_x × R), carry
    everything the measurements taught, and Σ_x∣y is the prior I less the
    variance they removed. The form is exact when
    the data's curvature has rank R or less (at most the number of
    measurements), and leaves the prior variance in every direction it
    misses.

    Algorithm:
        1. The R largest eigenpairs of Λ_x∣y by `gaussx.randomized_eigh`
           (Rayleigh–Ritz with oversampling and power iterations), from
           products with Λ_x∣y only.
        2. Σ_x∣y ≈ I + Φ diag(c) Φᵀ with c_d = −(γ_d − 1)/γ_d, as a
           `gaussx.LowRankUpdate` of the identity with orthonormal Φ.

    Args:
        posterior: A posterior in whitened coordinates, whose precision has
            every eigenvalue at least 1.
        rank: Number of eigenpairs R, from 1 to D_x.
        key: Random key for the test matrix.

    Returns:
        Σ_x∣y as a (D_x, D_x) `gaussx.LowRankUpdate`, tagged symmetric and positive
        semidefinite.

    Raises:
        ValidationError: If the mode is not one-dimensional, or ``rank`` is
            not an integer from 1 to D_x.

    Examples:
        Three unknowns, one measurement of their sum with unit noise:
        Λ_x∣y = I + 1 1ᵀ has one eigenvalue above 1, so rank 1 is exact.

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
    # the R largest eigenpairs (γ_d, φ_d) of Λ_x∣y, from products with it only
    eigenvalues, eigenvectors = gx.randomized_eigh(posterior.precision, rank, key=key)
    # c_d = −(γ_d − 1)/γ_d, the variance the data removed along φ_d
    removed = -(eigenvalues - 1.0) / eigenvalues
    identity = lx.IdentityLinearOperator(posterior.precision.in_structure())
    # Σ_x∣y ≈ I + Φ diag(c) Φᵀ
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

        Λ_x∣y δ⁽ᵉ⁾ = J_rᵀ v⁽ᵉ⁾,   v⁽ᵉ⁾ ~ 𝒩(0, I_D_r),
        Cov(δ⁽ᵉ⁾) = Λ_x∣y⁻¹ J_rᵀ J_r Λ_x∣y⁻¹ = Λ_x∣y⁻¹

    Each draw is x̂ + δ⁽ᵉ⁾, one product with J_rᵀ and one solve with Λ_x∣y,
    so Σ_x∣y and a square root of it are never formed. Here J_r is the
    D_r × D_x Jacobian of the whitened residuals r at x̂. The covariance is
    exact because the prior rows are part of J_r, so Λ_x∣y = J_rᵀ J_r: ``posterior``
    must come from `gauss_newton` on the same ``residuals`` and ``args``. An
    inexact solve biases every draw; set the solver's tolerance accordingly.

    Algorithm:
        1. Linearise the residuals at x̂ once, keeping the pullback v ↦ J_rᵀ v.
        2. For each draw, v⁽ᵉ⁾ ~ 𝒩(0, I_D_r), then δ⁽ᵉ⁾ = Λ_x∣y⁻¹ J_rᵀ v⁽ᵉ⁾ by
           one pullback and one solve, and return x̂ + δ⁽ᵉ⁾.

    Args:
        residuals: The whitened residuals r(x, args) the posterior was built
            from.
        posterior: The posterior 𝒩(x̂, Λ_x∣y⁻¹).
        n: Number of draws N_ℰ.
        args: Passed to ``residuals`` as its second argument.
        key: Random key for the standard normal draws v⁽ᵉ⁾.

    Returns:
        The draws, shape (N_ℰ, D_x).

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
    # v⁽ᵉ⁾ ~ 𝒩(0, I_D_r), one row per draw
    noise = jax.random.normal(key, (n, residual_size), residual_at_mode.dtype)

    def perturbation(standard: Float[Array, " residual"]) -> Float[Array, " state"]:
        # J_rᵀ v⁽ᵉ⁾
        (pulled,) = pullback(standard)
        # δ⁽ᵉ⁾ = Λ_x∣y⁻¹ J_rᵀ v⁽ᵉ⁾
        return posterior.solver.solve(posterior.precision, pulled)

    perturbations = jax.vmap(perturbation)(noise)
    # x⁽ᵉ⁾ = x̂ + δ⁽ᵉ⁾
    return posterior.mode + perturbations

"""The Laplace evidence at a minimum, and the prior precision that maximises it.

Both are model comparison from the curvature at the minimum: one number for a
fixed posterior, and the prior strength that the data support best.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from argminax._src.curvature import Laplace
from argminax._src.diagnose._retrieval import (
    TraceEstimate,
    _logdet_estimate,
    _require_vector_mode,
)
from argminax.exceptions import ValidationError


__all__ = [
    "evidence",
    "optimize_prior_precision",
]


def evidence(
    posterior: Laplace,
    cost_at_mode: Float[ArrayLike, ""],
    obs_cov_logdet: Float[ArrayLike, ""],
    *,
    prior_cov_logdet: Float[ArrayLike, ""] = 0.0,
    num_probes: int = 32,
    num_matvecs: int = 30,
    key: PRNGKeyArray | None = None,
) -> TraceEstimate:
    """Return the Laplace evidence log p(y) at a minimum.

        log p(y) ≈ −𝒥(x̂) − ½ log det Λ_x∣y − ½ log det Σ_x
                   − ½ log det(2π Σ_ε)

    𝒥(x̂) = ½ ‖r(x̂)‖² is the cost at the minimum, with r the whitened
    residuals; Λ_x∣y is the posterior precision (D_x × D_x), Σ_x the prior
    covariance (D_x × D_x) and Σ_ε the measurement-error covariance
    (D_y × D_y). In whitened coordinates (prior 𝒩(0, I), the default)
    log det Σ_x = 0; in physical ones pass it as ``prior_cov_logdet``. The
    approximation is exact for a linear forward model and inherits the
    Gauss–Newton curvature's optimism at large residuals. Without a key
    log det Λ_x∣y is exact (`gaussx.logdet`); with one it is stochastic
    Lanczos quadrature with a standard error, as in `information_content`.

    Args:
        posterior: The posterior 𝒩(x̂, Λ_x∣y⁻¹), with a vector mode.
        cost_at_mode: 𝒥(x̂), the cost at the minimum.
        obs_cov_logdet: log det(2π Σ_ε) = D_y log 2π + log det Σ_ε, for D_y
            measurements.
        prior_cov_logdet: log det Σ_x, for a posterior in physical
            coordinates.
        num_probes: Number of probes N_ℰ, used only with a key.
        num_matvecs: Lanczos steps per probe, capped at D_x.
        key: Random key for the probes; ``None`` for the exact value.

    Returns:
        log p(y) and its standard error (0 when exact).

    Raises:
        ValidationError: If the mode is not one-dimensional or
            ``num_probes`` is below 2.

    Examples:
        One unknown with a unit prior, measured once with unit noise:
        y ~ 𝒩(0, 2), so at y = 1 the evidence is log 𝒩(1; 0, 2).

        >>> import jax.numpy as jnp
        >>> from argminax.curvature import gauss_newton
        >>> from argminax.diagnose import evidence
        >>> def residuals(x, y):
        ...     return y - x, x
        >>> measurement = jnp.ones(1)
        >>> posterior = gauss_newton(residuals, jnp.array([0.5]), measurement)
        >>> # 𝒥(x̂) = ½ (0.5² + 0.5²) and log det(2π Σ_ε) = log 2π
        >>> log_evidence = evidence(posterior, 0.25, jnp.log(2.0 * jnp.pi))
        >>> exact = -0.25 - 0.5 * jnp.log(4.0 * jnp.pi)
        >>> bool(jnp.allclose(log_evidence.value, exact))
        True

        The stochastic path needs at least two probes:

        >>> import jax
        >>> evidence(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     posterior, 0.25, 0.0, num_probes=1, key=jax.random.key(0)
        ... )
        Traceback (most recent call last):
        ...
        ValidationError: 'num_probes' must be an integer of at least 2, got 1

    References:
        Rodgers, C. D. (2000). Inverse Methods for Atmospheric Sounding,
        chapter 2. MacKay, D. J. C. (2003). Information Theory, Inference,
        and Learning Algorithms, chapter 27.
    """
    _require_vector_mode(posterior)
    log_det = _logdet_estimate(
        posterior, num_probes=num_probes, num_matvecs=num_matvecs, key=key
    )
    cost = jnp.asarray(cost_at_mode)
    prior_log_det = jnp.asarray(prior_cov_logdet)
    obs_log_det = jnp.asarray(obs_cov_logdet)
    # log det Λ_x∣y + log det Σ_x + log det(2π Σ_ε)
    total_log_det = log_det.value + prior_log_det + obs_log_det
    # log p(y) ≈ −𝒥(x̂) − ½ (log det Λ_x∣y + log det Σ_x + log det(2π Σ_ε))
    value = -cost - 0.5 * total_log_det
    # only log det Λ_x∣y is estimated, so it carries the whole error
    stderr = 0.5 * log_det.stderr
    return TraceEstimate(value=value, stderr=stderr)


def optimize_prior_precision(
    ggn_eigenvalues: Float[Array, " state"],
    fit_term: Callable[[Float[Array, ""]], Float[Array, ""]],
    *,
    num_grid: int = 4097,
    num_refine: int = 100,
    bounds: tuple[float, float] | None = None,
) -> Float[Array, ""]:
    """Return the isotropic prior precision δ that maximises the Laplace evidence.

        δ̂ = argmax_δ  f(δ) + (D_θ/2) log δ − ½ ∑ᵢ log(gᵢ + δ)

    for a prior 𝒩(0, δ⁻¹ I) on D_θ parameters θ, as a last-layer Laplace
    tunes it (Daxberger et al. 2021).
    gᵢ are the eigenvalues of Λ_lik, the Gauss–Newton curvature of the
    log-likelihood (D_θ × D_θ), so log det Λ_θ∣y(δ) = ∑ᵢ log(gᵢ + δ) for the
    posterior precision Λ_θ∣y(δ) = Λ_lik + δ I, and f(δ) is the rest of the
    evidence as a function of δ: the fit of the mode re-solved at each δ.
    For a quadratic model about θ⋆ (pyrox-nn's last-layer Laplace), in the
    eigenbasis Q of Λ_lik with d = Qᵀ θ⋆, b̃ = Qᵀ b and
    ω = (g ⊙ d + b̃)/(g + δ),
    f(δ) = b̃ᵀ(ω − d) − ½ ∑ᵢ gᵢ (ωᵢ − dᵢ)² − (δ/2) ‖ω‖².

    Algorithm:
        1. Evaluate the evidence on ``num_grid`` points of log δ, by
           default across half the dtype's exponent range each way, so δ
           and every term stay finite. A grid finds the global maximum of an
           evidence with more than one local maximum, where a local 1-D
           search could stop at the wrong one.
        2. Refine by ``num_refine`` golden-section steps in the grid cell
           either side of the best point.

    Args:
        ggn_eigenvalues: The eigenvalues g of the Gauss–Newton curvature
            Λ_lik, non-negative (clip round-off below zero first).
        fit_term: f(δ), a scalar function of the prior precision.
        num_grid: Grid points on log δ.
        num_refine: Golden-section steps, each shrinking the bracket by
            0.618.
        bounds: The range of log δ to search; default ±½ log of the largest
            float of the eigenvalues' dtype.

    Returns:
        δ̂, a scalar in the dtype of the eigenvalues.

    Raises:
        ValidationError: If ``num_grid`` is below 3 or ``num_refine`` is
            negative.

    Examples:
        No fit term and unit curvature, g = (1, 1): the evidence
        log δ − log(1 + δ) only grows, so δ̂ sits at the top of the range.

        >>> import jax.numpy as jnp
        >>> from argminax.diagnose import optimize_prior_precision
        >>> best = optimize_prior_precision(
        ...     jnp.ones(2), lambda lam: 0.0 * lam, bounds=(-4.0, 4.0)
        ... )
        >>> bool(jnp.allclose(jnp.log(best), 4.0))
        True

        A fit term f(δ) = −δ and g = (1,): the evidence peaks where
        2δ² + 2δ − 1 = 0, at δ̂ = (√3 − 1)/2.

        >>> best = optimize_prior_precision(jnp.ones(1), lambda lam: -lam)
        >>> bool(jnp.allclose(best, (jnp.sqrt(3.0) - 1.0) / 2.0, rtol=1e-4))
        True

        A grid needs three points to bracket a maximum:

        >>> optimize_prior_precision(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     jnp.ones(2), lambda lam: 0.0 * lam, num_grid=2
        ... )
        Traceback (most recent call last):
        ...
        ValidationError: 'num_grid' must be an integer of at least 3, got 2

    References:
        MacKay, D. J. C. (1992). Bayesian interpolation. Neural Computation
        4(3), 415–447. Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen,
        R., Bauer, M. & Hennig, P. (2021). Laplace redux: effortless Bayesian
        deep learning. NeurIPS 34.
    """
    if isinstance(num_grid, bool) or not isinstance(num_grid, int) or num_grid < 3:
        msg = f"'num_grid' must be an integer of at least 3, got {num_grid!r}"
        raise ValidationError(msg)
    if (
        isinstance(num_refine, bool)
        or not isinstance(num_refine, int)
        or num_refine < 0
    ):
        msg = f"'num_refine' must be a non-negative integer, got {num_refine!r}"
        raise ValidationError(msg)
    eigenvalues = jnp.asarray(ggn_eigenvalues)
    dtype = eigenvalues.dtype
    size = eigenvalues.shape[0]

    def log_evidence(log_lam: Float[Array, ""]) -> Float[Array, ""]:
        # δ = exp(log δ)
        lam = jnp.exp(log_lam)
        # log(gᵢ + δ)
        log_shifted = jnp.log(eigenvalues + lam)
        # ∑ᵢ log(gᵢ + δ) = log det Λ_θ∣y(δ)
        log_det = einx.sum("state ->", log_shifted)
        # f(δ) + (D_θ/2) log δ − ½ ∑ᵢ log(gᵢ + δ)
        return fit_term(lam) + 0.5 * size * log_lam - 0.5 * log_det

    # --- grid over log δ ---
    if bounds is None:
        largest = float(jnp.finfo(dtype).max)
        # ±½ log of the largest float keeps δ and every term finite
        reach = 0.5 * math.log(largest)
        bounds = (-reach, reach)
    low, high = bounds
    grid = jnp.linspace(low, high, num_grid, dtype=dtype)
    grid_evidence = jax.vmap(log_evidence)(grid)
    best = jnp.argmax(grid_evidence)
    step = grid[1] - grid[0]
    # the grid cell either side of the best point, clipped to the range
    lower = jnp.maximum(grid[best] - step, low)
    upper = jnp.minimum(grid[best] + step, high)

    # --- golden-section refinement in the best cell ---
    # φ − 1 = (√5 − 1)/2
    ratio = (math.sqrt(5.0) - 1.0) / 2.0

    def golden(
        _: int, bracket: tuple[Float[Array, ""], Float[Array, ""]]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        left, right = bracket
        width = right - left
        # the two interior points, a fraction φ − 1 in from each end
        inner_left = right - ratio * width
        inner_right = left + ratio * width
        keep_left = log_evidence(inner_left) >= log_evidence(inner_right)
        new_left = jnp.where(keep_left, left, inner_left)
        new_right = jnp.where(keep_left, inner_right, right)
        return new_left, new_right

    lower, upper = jax.lax.fori_loop(0, num_refine, golden, (lower, upper))
    # log δ̂ at the centre of the final bracket
    midpoint = 0.5 * (lower + upper)
    # exp in the eigenvalues' dtype, which the docstring promises
    return jnp.exp(midpoint).astype(dtype)

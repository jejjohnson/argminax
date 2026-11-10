"""The Laplace evidence at a minimum, and the prior precision that maximises it.

Both are model comparison from the curvature at the minimum: one number for a
fixed posterior, and the prior strength that the data support best.
"""

from __future__ import annotations

import math
from collections.abc import Callable

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

        log p(y) ≈ −J(x̂) − ½ log |Ĥ| − ½ log |Sₐ| − ½ log |2π Sε|

    J(x̂) = ½ ‖r(x̂)‖² is the cost at the minimum. In whitened coordinates
    (prior N(0, I), the default) log |Sₐ| = 0; in physical ones pass it as
    ``prior_cov_logdet``. The approximation is exact for a linear forward
    model and inherits the Gauss–Newton curvature's optimism at large
    residuals. Without a key log |Ĥ| is exact (`gaussx.logdet`); with one it
    is stochastic Lanczos quadrature with a standard error, as in
    `information_content`.

    Args:
        posterior: The posterior N(x̂, Ĥ⁻¹), with a vector mode.
        cost_at_mode: J(x̂), the cost at the minimum.
        obs_cov_logdet: log |2π Sε| = M log 2π + log |Sε|, for M measurements.
        prior_cov_logdet: log |Sₐ|, for a posterior in physical coordinates.
        num_probes: Number of probes K, used only with a key.
        num_matvecs: Lanczos steps per probe, capped at N.
        key: Random key for the probes; ``None`` for the exact value.

    Returns:
        log p(y) and its standard error (0 when exact).

    Raises:
        ValidationError: If the mode is not one-dimensional or
            ``num_probes`` is below 2.

    Examples:
        One unknown with a unit prior, measured once with unit noise:
        y ~ N(0, 2), so at y = 1 the evidence is log N(1; 0, 2).

        >>> import jax.numpy as jnp
        >>> from argminax.curvature import gauss_newton
        >>> from argminax.diagnose import evidence
        >>> def residuals(x, y):
        ...     return y - x, x
        >>> measurement = jnp.ones(1)
        >>> posterior = gauss_newton(residuals, jnp.array([0.5]), measurement)
        >>> # J(x̂) = ½ (0.5² + 0.5²) and log |2π Sε| = log 2π
        >>> log_evidence = evidence(posterior, 0.25, jnp.log(2.0 * jnp.pi))
        >>> exact = -0.25 - 0.5 * jnp.log(4.0 * jnp.pi)
        >>> bool(jnp.allclose(log_evidence.value, exact))
        True

    References:
        Rodgers, C. D. (2000). Inverse Methods for Atmospheric Sounding,
        chapter 2. MacKay, D. J. C. (2003). Information Theory, Inference,
        and Learning Algorithms, chapter 27.
    """
    _require_vector_mode(posterior)
    log_det = _logdet_estimate(
        posterior, num_probes=num_probes, num_matvecs=num_matvecs, key=key
    )
    # −J(x̂) − ½ (log |Ĥ| + log |Sₐ| + log |2π Sε|)
    value = -jnp.asarray(cost_at_mode) - 0.5 * (
        log_det.value + jnp.asarray(prior_cov_logdet) + jnp.asarray(obs_cov_logdet)
    )
    return TraceEstimate(value=value, stderr=0.5 * log_det.stderr)


def optimize_prior_precision(
    ggn_eigenvalues: Float[Array, " N"],
    fit_term: Callable[[Float[Array, ""]], Float[Array, ""]],
    *,
    num_grid: int = 4097,
    num_refine: int = 100,
    bounds: tuple[float, float] | None = None,
) -> Float[Array, ""]:
    """Return the isotropic prior precision λ that maximises the Laplace evidence.

        λ̂ = argmax_λ  f(λ) + (N/2) log λ − ½ ∑ᵢ log(gᵢ + λ)

    gᵢ are the eigenvalues of the Gauss–Newton curvature of the
    log-likelihood, so log |Ĥ_λ| = ∑ᵢ log(gᵢ + λ) for the posterior precision
    Ĥ_λ = G + λ I, and f(λ) is the rest of the evidence as a function of λ:
    the fit of the mode re-solved at each λ. For a quadratic model about
    w⋆ (pyrox-nn's last-layer Laplace), in the eigenbasis of G with
    d = Qᵀ w⋆, β = Qᵀ b and ω = (g ⊙ d + β)/(g + λ),
    f(λ) = βᵀ(ω − d) − ½ ∑ᵢ gᵢ (ωᵢ − dᵢ)² − (λ/2) ‖ω‖².

    Algorithm:
        1. Evaluate the evidence on ``num_grid`` points of log λ, by
           default across half the dtype's exponent range each way, so λ
           and every term stay finite. A grid finds the global maximum of an
           evidence with more than one local maximum, where a local 1-D
           search could stop at the wrong one.
        2. Refine by ``num_refine`` golden-section steps in the grid cell
           either side of the best point.

    Args:
        ggn_eigenvalues: The eigenvalues g of the Gauss–Newton curvature,
            non-negative (clip round-off below zero first).
        fit_term: f(λ), a scalar function of the prior precision.
        num_grid: Grid points on log λ.
        num_refine: Golden-section steps, each shrinking the bracket by
            0.618.
        bounds: The range of log λ to search; default ±½ log of the largest
            float of the eigenvalues' dtype.

    Returns:
        λ̂, a scalar in the dtype of the eigenvalues.

    Raises:
        ValidationError: If ``num_grid`` is below 3 or ``num_refine`` is
            negative.

    Examples:
        No fit term and unit curvature, g = (1, 1): the evidence
        log λ − log(1 + λ) only grows, so λ̂ sits at the top of the range.

        >>> import jax.numpy as jnp
        >>> from argminax.diagnose import optimize_prior_precision
        >>> best = optimize_prior_precision(
        ...     jnp.ones(2), lambda lam: 0.0 * lam, bounds=(-4.0, 4.0)
        ... )
        >>> bool(jnp.allclose(jnp.log(best), 4.0))
        True

    References:
        MacKay, D. J. C. (1992). Bayesian interpolation. Neural Computation
        4(3), 415–447.
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
        lam = jnp.exp(log_lam)
        # f(λ) + (N/2) log λ − ½ ∑ᵢ log(gᵢ + λ)
        return (
            fit_term(lam)
            + 0.5 * size * log_lam
            - 0.5 * jnp.sum(jnp.log(eigenvalues + lam))
        )

    # --- grid over log λ ---
    if bounds is None:
        reach = 0.5 * math.log(float(jnp.finfo(dtype).max))
        bounds = (-reach, reach)
    low, high = bounds
    grid = jnp.linspace(low, high, num_grid, dtype=dtype)
    best = jnp.argmax(jax.vmap(log_evidence)(grid))
    step = grid[1] - grid[0]
    lower = jnp.maximum(grid[best] - step, low)
    upper = jnp.minimum(grid[best] + step, high)

    # --- golden-section refinement in the best cell ---
    ratio = (math.sqrt(5.0) - 1.0) / 2.0

    def golden(
        _: int, bracket: tuple[Float[Array, ""], Float[Array, ""]]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        left, right = bracket
        inner_left = right - ratio * (right - left)
        inner_right = left + ratio * (right - left)
        keep_left = log_evidence(inner_left) >= log_evidence(inner_right)
        return (
            jnp.where(keep_left, left, inner_left),
            jnp.where(keep_left, inner_right, right),
        )

    lower, upper = jax.lax.fori_loop(0, num_refine, golden, (lower, upper))
    return jnp.exp(0.5 * (lower + upper)).astype(dtype)

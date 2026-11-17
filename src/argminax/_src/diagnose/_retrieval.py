"""What the curvature at a minimum says: averaging kernel, DOFS, information content.

Each holds for any linear-Gaussian posterior, not only for retrievals. Exact
for a small state, stochastic with a reported standard error above, and
always under ``jit`` and ``vmap``, with randomness only from keys.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from argminax._src.curvature import Laplace
from argminax.exceptions import MissingExtraError, ValidationError


__all__ = [
    "TraceEstimate",
    "averaging_kernel",
    "degrees_of_freedom",
    "information_content",
]


class TraceEstimate(eqx.Module):
    """A scalar diagnostic and its standard error.

    Attributes:
        value: The estimate.
        stderr: Its standard error: exactly 0 when the value was computed
            exactly, otherwise the sample standard deviation of the probe
            terms over √K. It does not see the bias of an inexact solve.
    """

    value: Float[Array, ""]
    stderr: Float[Array, ""]


def _require_vector_mode(posterior: Laplace) -> None:
    if jnp.ndim(posterior.mode) != 1:
        msg = (
            "'posterior' must have a one-dimensional mode, got shape "
            f"{jnp.shape(posterior.mode)}; flatten the state first"
        )
        raise ValidationError(msg)


def _require_probes(num_probes: int) -> None:
    if (
        isinstance(num_probes, bool)
        or not isinstance(num_probes, int)
        or num_probes < 2
    ):
        msg = f"'num_probes' must be an integer of at least 2, got {num_probes!r}"
        raise ValidationError(msg)


def _mean_and_stderr(terms: Float[Array, " member"]) -> TraceEstimate:
    """The mean of K probe terms and its standard error."""
    num_terms = terms.shape[0]
    # t̄ = (1/K) ∑ₖ tₖ
    mean = einx.mean("member ->", terms)
    # s = √(∑ₖ (tₖ − t̄)² / (K − 1)), the sample standard deviation
    spread = jnp.std(terms, ddof=1)
    # s / √K
    stderr = spread / jnp.sqrt(num_terms)
    return TraceEstimate(value=mean, stderr=stderr)


def _exact(value: Float[Array, ""]) -> TraceEstimate:
    """An exactly computed value, whose standard error is 0."""
    stderr = jnp.zeros_like(value)
    return TraceEstimate(value=value, stderr=stderr)


def averaging_kernel(
    posterior: Laplace, prior_cov: lx.AbstractLinearOperator | None = None
) -> lx.AbstractLinearOperator:
    """Return the averaging kernel A = ∂x̂/∂x of a Laplace posterior, lazily.

        A = I − Ŝ Sₐ⁻¹,   in whitened coordinates A = I − Ĥ⁻¹

    A says how much a change in the true state moves the retrieved one:
    A = I means the measurement decides everything, A = 0 means the prior
    does (Rodgers 2000, ch. 2). Without ``prior_cov`` the posterior is taken
    to be in whitened coordinates (prior N(0, I)), where A is symmetric
    with eigenvalues σᵢ²/(1 + σᵢ²) in [0, 1). Each product is one solve
    with Ĥ (and, with ``prior_cov``, one with Sₐ); rows of A are never
    formed unless asked for.

    Args:
        posterior: The posterior N(x̂, Ĥ⁻¹), with a vector mode.
        prior_cov: The prior covariance Sₐ, for a posterior in physical
            coordinates; ``None`` for whitened ones.

    Returns:
        A, an (N, N) lazy operator; tagged symmetric and positive
        semidefinite in whitened coordinates, untagged otherwise.

    Raises:
        ValidationError: If the mode is not one-dimensional.

    Examples:
        One unknown measured with the prior's own variance: Ĥ = 2, so half
        the retrieved state comes from the measurement.

        >>> import jax.numpy as jnp
        >>> from argminax.curvature import gauss_newton
        >>> from argminax.diagnose import averaging_kernel
        >>> def residuals(x, y):
        ...     return y - x, x
        >>> posterior = gauss_newton(residuals, jnp.zeros(1), jnp.ones(1))
        >>> kernel = averaging_kernel(posterior)
        >>> bool(jnp.allclose(kernel.as_matrix(), 0.5))
        True

        In physical coordinates with Sₐ = 2: Ĥ = 1 + ½, Ŝ = ⅔, so
        A = 1 − Ŝ Sₐ⁻¹ = ⅔.

        >>> import lineax as lx
        >>> prior_cov = lx.DiagonalLinearOperator(jnp.full(1, 2.0))
        >>> def physical_residuals(x, y):
        ...     return y - x, x / jnp.sqrt(2.0)
        >>> physical = gauss_newton(physical_residuals, jnp.zeros(1), jnp.ones(1))
        >>> kernel = averaging_kernel(physical, prior_cov)
        >>> bool(jnp.allclose(kernel.as_matrix(), 2.0 / 3.0))
        True
    """
    _require_vector_mode(posterior)
    covariance = posterior.covariance()
    structure = posterior.precision.in_structure()
    if prior_cov is None:

        def whitened(vector: Float[Array, " state"]) -> Float[Array, " state"]:
            # Ĥ⁻¹ v
            covariance_vector = covariance.mv(vector)
            # A v = v − Ĥ⁻¹ v
            return vector - covariance_vector

        return lx.FunctionLinearOperator(
            whitened,
            structure,
            (lx.symmetric_tag, lx.positive_semidefinite_tag),
        )

    def physical(vector: Float[Array, " state"]) -> Float[Array, " state"]:
        # Sₐ⁻¹ v
        prior_precision_vector = gx.solve(prior_cov, vector)
        # Ŝ Sₐ⁻¹ v
        covariance_vector = covariance.mv(prior_precision_vector)
        # A v = v − Ŝ Sₐ⁻¹ v
        return vector - covariance_vector

    return lx.FunctionLinearOperator(physical, structure)


def degrees_of_freedom(
    kernel: lx.AbstractLinearOperator,
    *,
    num_probes: int = 32,
    key: PRNGKeyArray | None = None,
) -> TraceEstimate:
    """Return the degrees of freedom for signal, the trace of the averaging kernel.

        dₛ = tr A = ∑ᵢ σᵢ²/(1 + σᵢ²),   dₛ ≈ (1/K) ∑ₖ zₖᵀ A zₖ

    σᵢ are the singular values of the whitened Jacobian. Without a key the
    trace is exact (`gaussx.trace`, which forms the diagonal of A by N
    products); with one it is Hutchinson's estimate over K Rademacher probes
    from the key, each one product with A (so one solve), with the standard
    error of the K terms. Because dₛ = N − tr(Ĥ⁻¹) in whitened coordinates
    and zᵀz = N for a Rademacher probe, this is the same estimator as
    probing tr(Ĥ⁻¹).

    Algorithm:
        1. Without a key, return the exact trace of A.
        2. With one, draw K Rademacher probes zₖ from the key.
        3. Average zₖᵀ A zₖ over the probes, with the standard error of the
           K terms.

    Args:
        kernel: The averaging kernel A, from `averaging_kernel`.
        num_probes: Number of probes K, used only with a key.
        key: Random key for the probes; ``None`` for the exact trace.

    Returns:
        dₛ and its standard error (0 when exact).

    Raises:
        ValidationError: If ``num_probes`` is below 2.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from argminax.curvature import gauss_newton
        >>> from argminax.diagnose import averaging_kernel, degrees_of_freedom
        >>> def residuals(x, y):
        ...     return y - x, x
        >>> posterior = gauss_newton(residuals, jnp.zeros(3), jnp.ones(3))
        >>> exact = degrees_of_freedom(averaging_kernel(posterior))
        >>> bool(jnp.allclose(exact.value, 1.5)), bool(exact.stderr == 0.0)
        (True, True)

        A diagonal A is probed exactly by Rademacher probes.

        >>> probed = degrees_of_freedom(
        ...     averaging_kernel(posterior), key=jax.random.key(0)
        ... )
        >>> bool(jnp.allclose(probed.value, 1.5))
        True

        One probe has no spread to estimate, so it is rejected:

        >>> degrees_of_freedom(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     averaging_kernel(posterior), num_probes=1, key=jax.random.key(0)
        ... )
        Traceback (most recent call last):
        ...
        ValidationError: 'num_probes' must be an integer of at least 2, got 1

    References:
        Hutchinson, M. F. (1990). A stochastic estimator of the trace of the
        influence matrix for Laplacian smoothing splines. Communications in
        Statistics, Simulation and Computation 19(2), 433–450.
        doi:10.1080/03610919008812866.
    """
    if key is None:
        return _exact(gx.trace(kernel))
    _require_probes(num_probes)
    structure = kernel.in_structure()
    probe_shape = (num_probes, *structure.shape)
    # zₖ ∈ {−1, +1}ᴺ
    probes = jax.random.rademacher(key, probe_shape, structure.dtype)

    def probe_term(probe: Float[Array, " state"]) -> Float[Array, ""]:
        # A z
        kernel_probe = kernel.mv(probe)
        # zᵀ A z
        return jnp.vdot(probe, kernel_probe)

    terms = jax.vmap(probe_term)(probes)
    # dₛ ≈ (1/K) ∑ₖ zₖᵀ A zₖ
    return _mean_and_stderr(terms)


def _slq_logdet(
    operator: lx.AbstractLinearOperator,
    *,
    num_probes: int,
    num_matvecs: int,
    key: PRNGKeyArray,
) -> TraceEstimate:
    """log |M| by stochastic Lanczos quadrature, with its standard error."""
    try:
        from matfree import decomp, funm
    except ImportError as error:  # pragma: no cover - matfree comes with gaussx
        msg = (
            "a stochastic log-determinant needs the 'matfree' extra: "
            "pip install 'argminax[matfree]'"
        )
        raise MissingExtraError(msg) from error
    structure = operator.in_structure()
    size = structure.shape[0]
    # Lanczos cannot take more steps than the operator has dimensions
    num_steps = min(num_matvecs, size)
    tridiagonalise = decomp.tridiag_sym(num_steps)
    integrand = funm.monte_carlo_funm_sym_logdet(tridiagonalise)
    # zₖ ∈ {−1, +1}ᴺ
    probes = jax.random.rademacher(key, (num_probes, size), structure.dtype)
    matvec: Callable[[Array], Array] = operator.mv

    def probe_term(probe: Float[Array, " state"]) -> Float[Array, ""]:
        # zᵀ log(M) z, by m Lanczos steps
        return integrand(matvec, probe)

    terms = jax.vmap(probe_term)(probes)
    # log |M| ≈ (1/K) ∑ₖ zₖᵀ log(M) zₖ
    return _mean_and_stderr(terms)


def _logdet_estimate(
    posterior: Laplace,
    *,
    num_probes: int,
    num_matvecs: int,
    key: PRNGKeyArray | None,
) -> TraceEstimate:
    """log |Ĥ|, exact (`gaussx.logdet`) without a key, by SLQ with one."""
    if key is None:
        return _exact(gx.logdet(posterior.precision))
    _require_probes(num_probes)
    return _slq_logdet(
        posterior.precision, num_probes=num_probes, num_matvecs=num_matvecs, key=key
    )


def information_content(
    posterior: Laplace,
    prior_cov: lx.AbstractLinearOperator | None = None,
    *,
    num_probes: int = 32,
    num_matvecs: int = 30,
    key: PRNGKeyArray | None = None,
) -> TraceEstimate:
    """Return the information content of the measurement, in nats.

        H = ½ log |Sₐ Ŝ⁻¹| = ½ (log |Ĥ| + log |Sₐ|) = ½ ∑ᵢ log(1 + σᵢ²)

    H is the entropy the measurement removed from the prior (Rodgers 2000,
    ch. 2); in whitened coordinates (``prior_cov=None``) log |Sₐ| = 0.
    Without a key log |Ĥ| is exact (`gaussx.logdet`); with one it is
    stochastic Lanczos quadrature over K Rademacher probes of
    ``num_matvecs`` Lanczos steps each, from the ``matfree`` extra, with
    the standard error of the K terms. log |Sₐ| is always exact.

    Algorithm:
        1. log |Ĥ|: exact without a key; with one, stochastic Lanczos
           quadrature over K Rademacher probes, with its standard error.
        2. log |Sₐ|: exact, or 0 in whitened coordinates.
        3. H = ½ (log |Ĥ| + log |Sₐ|), with half the standard error of
           log |Ĥ|.

    Args:
        posterior: The posterior N(x̂, Ĥ⁻¹), with a vector mode.
        prior_cov: The prior covariance Sₐ, for a posterior in physical
            coordinates; ``None`` for whitened ones.
        num_probes: Number of probes K, used only with a key.
        num_matvecs: Lanczos steps per probe, capped at N.
        key: Random key for the probes; ``None`` for the exact value.

    Returns:
        H and its standard error (0 when exact).

    Raises:
        ValidationError: If the mode is not one-dimensional or
            ``num_probes`` is below 2.
        MissingExtraError: With a key, if ``matfree`` is not installed.

    Examples:
        One unknown measured with the prior's own variance: Ĥ = 2, so
        H = ½ log 2.

        >>> import jax.numpy as jnp
        >>> from argminax.curvature import gauss_newton
        >>> from argminax.diagnose import information_content
        >>> def residuals(x, y):
        ...     return y - x, x
        >>> posterior = gauss_newton(residuals, jnp.zeros(1), jnp.ones(1))
        >>> content = information_content(posterior)
        >>> bool(jnp.allclose(content.value, 0.5 * jnp.log(2.0)))
        True

        In physical coordinates with Sₐ = 2: Ĥ = 1 + ½, so
        H = ½ log(Sₐ Ĥ) = ½ log 3.

        >>> import lineax as lx
        >>> prior_cov = lx.DiagonalLinearOperator(jnp.full(1, 2.0))
        >>> def physical_residuals(x, y):
        ...     return y - x, x / jnp.sqrt(2.0)
        >>> physical = gauss_newton(physical_residuals, jnp.zeros(1), jnp.ones(1))
        >>> content = information_content(physical, prior_cov)
        >>> bool(jnp.allclose(content.value, 0.5 * jnp.log(3.0)))
        True

    References:
        Ubaru, S., Chen, J. & Saad, Y. (2017). Fast estimation of tr(f(A))
        via stochastic Lanczos quadrature. SIAM J. Matrix Anal. Appl. 38(4),
        1075–1099. doi:10.1137/16M1104974.
    """
    _require_vector_mode(posterior)
    log_det = _logdet_estimate(
        posterior, num_probes=num_probes, num_matvecs=num_matvecs, key=key
    )
    # log |Sₐ|, which is 0 in whitened coordinates
    prior_log_det = 0.0 if prior_cov is None else gx.logdet(prior_cov)
    # log |Ĥ| + log |Sₐ|
    total_log_det = log_det.value + prior_log_det
    # H = ½ (log |Ĥ| + log |Sₐ|)
    value = 0.5 * total_log_det
    # log |Sₐ| is exact, so only log |Ĥ| carries an error
    stderr = 0.5 * log_det.stderr
    return TraceEstimate(value=value, stderr=stderr)

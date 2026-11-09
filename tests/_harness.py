"""Reference checks shared by every argminax test.

Every solver, curvature operator and diagnostic is tested the same few ways,
and this module holds one helper per way:

| Helper | Checks against |
|---|---|
| `linear_gaussian_problem` | closed forms of a linear model (Rodgers 2000, ch. 2) |
| `assert_matches_scipy_least_squares` | `scipy.optimize.least_squares` |
| `assert_grad_matches_fd` | central finite differences |
| `assert_within_stderr` | the standard error of a Monte Carlo mean |
| `assert_sample_covariance` | the sampling spread of a sample covariance |
| `assert_traces` | eager, `jit` and `vmap` evaluations of the same function |

For y = Kx + ε with ε ~ N(0, Sε) and prior x ~ N(xₐ, Sₐ), the cost is
J(x) = ½ (y − Kx)ᵀ Sε⁻¹ (y − Kx) + ½ (x − xₐ)ᵀ Sₐ⁻¹ (x − xₐ), and every
quantity at its minimum has a closed form, so the linear-Gaussian problem is
the primary oracle. Everything here is dense `jnp.linalg`, which is fine in
tests and never allowed in library code.

Shapes: N state; M measurements; B batch; S samples.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.stats
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, Float, PRNGKeyArray, PyTree


__all__ = [
    "LinearGaussian",
    "assert_grad_matches_fd",
    "assert_matches_scipy_least_squares",
    "assert_sample_covariance",
    "assert_traces",
    "assert_within_stderr",
    "linear_gaussian_problem",
    "whitened_residuals",
]


class LinearGaussian(eqx.Module):
    """A linear-Gaussian retrieval and the exact answers at its minimum.

    The first five fields define the problem; the rest are its closed forms
    (Rodgers 2000, ch. 2), computed densely by `linear_gaussian_problem`.

    Attributes:
        K: Forward model (Jacobian), shape (M, N).
        S_eps: Measurement-error covariance Sε, shape (M, M).
        S_a: Prior covariance Sₐ, shape (N, N).
        x_a: Prior state xₐ, shape (N,).
        y: Measurement, shape (M,).
        x_hat: The minimum x̂ = xₐ + Ŝ Kᵀ Sε⁻¹ (y − K xₐ), shape (N,).
        S_hat: Posterior covariance Ŝ = (Kᵀ Sε⁻¹ K + Sₐ⁻¹)⁻¹, shape (N, N).
        A: Averaging kernel A = I − Ŝ Sₐ⁻¹, shape (N, N).
        d_s: Degrees of freedom for signal dₛ = ∑ᵢ σᵢ² / (1 + σᵢ²), a scalar.
        H_info: Information content H = ½ ∑ᵢ log(1 + σᵢ²), a scalar.
        log_evidence: log N(y; K xₐ, K Sₐ Kᵀ + Sε), a scalar.
    """

    K: Float[Array, "M N"]
    S_eps: Float[Array, "M M"]
    S_a: Float[Array, "N N"]
    x_a: Float[Array, " N"]
    y: Float[Array, " M"]
    x_hat: Float[Array, " N"]
    S_hat: Float[Array, "N N"]
    A: Float[Array, "N N"]
    d_s: Float[Array, ""]
    H_info: Float[Array, ""]
    log_evidence: Float[Array, ""]


def _random_covariance(
    key: PRNGKeyArray, size: int, dtype: Any
) -> Float[Array, "size size"]:
    """A well-conditioned covariance Q diag(λ) Qᵀ with λ drawn from [0.5, 2]."""
    rotation_key, spectrum_key = jax.random.split(key)
    # Q from the QR factor of a Gaussian matrix, so it is orthogonal
    rotation, _ = jnp.linalg.qr(jax.random.normal(rotation_key, (size, size), dtype))
    # λ in [0.5, 2] bounds the condition number by 4
    spectrum = jax.random.uniform(spectrum_key, (size,), dtype, 0.5, 2.0)
    # Σ = Q diag(λ) Qᵀ, symmetrised so it is exactly symmetric in floating point
    covariance = (rotation * spectrum) @ rotation.T
    return 0.5 * (covariance + covariance.T)


def linear_gaussian_problem(
    key: PRNGKeyArray, n_state: int, n_obs: int, *, dtype: Any = jnp.float64
) -> LinearGaussian:
    """Draw a linear-Gaussian retrieval and compute its exact answers.

    y = K x + ε,  ε ~ N(0, Sε),  x ~ N(xₐ, Sₐ)

    Algorithm:
        1. Draw K with entries N(0, 1/N), well-conditioned covariances Sε and
           Sₐ (eigenvalues in [0.5, 2]), xₐ, a true state from the prior and
           a measurement y from the forward model.
        2. Ŝ = (Kᵀ Sε⁻¹ K + Sₐ⁻¹)⁻¹, x̂ = xₐ + Ŝ Kᵀ Sε⁻¹ (y − K xₐ) and
           A = I − Ŝ Sₐ⁻¹.
        3. The singular values σᵢ of K̃ = Lε⁻¹ K Lₐ, with Lε and Lₐ the
           Cholesky factors of Sε and Sₐ, give dₛ and H. They are the
           singular values of Sε^(-1/2) K Sₐ^(1/2) too, since the two
           matrices differ by orthogonal factors on each side.
        4. The log evidence is the log density of y under
           N(K xₐ, K Sₐ Kᵀ + Sε).

    Args:
        key: Random key for the draws.
        n_state: State size N.
        n_obs: Number of measurements M.
        dtype: Floating dtype of every array.

    Returns:
        The problem and its closed forms.

    Examples:
        >>> import jax
        >>> problem = linear_gaussian_problem(jax.random.key(0), 3, 5)
        >>> problem.K.shape, problem.S_hat.shape
        ((5, 3), (3, 3))
    """
    keys = jax.random.split(key, 6)

    # --- the problem ---
    # K with entries N(0, 1/N), so Kx stays O(1); math, not NumPy, scalars keep
    # the dtype
    forward = jax.random.normal(keys[0], (n_obs, n_state), dtype) / math.sqrt(n_state)
    noise_covariance = _random_covariance(keys[1], n_obs, dtype)
    prior_covariance = _random_covariance(keys[2], n_state, dtype)
    prior_state = jax.random.normal(keys[3], (n_state,), dtype)
    noise_factor = jnp.linalg.cholesky(noise_covariance)
    prior_factor = jnp.linalg.cholesky(prior_covariance)
    # x = xₐ + Lₐ z, a draw from the prior
    true_state = prior_state + prior_factor @ jax.random.normal(
        keys[4], (n_state,), dtype
    )
    # y = K x + Lε z, a draw from the forward model
    measurement = forward @ true_state + noise_factor @ jax.random.normal(
        keys[5], (n_obs,), dtype
    )

    # --- the minimum and its covariance ---
    identity = jnp.eye(n_state, dtype=dtype)
    # Sε⁻¹ K and Sₐ⁻¹ by solves against the covariances
    noise_precision_forward = jnp.linalg.solve(noise_covariance, forward)
    prior_precision = jnp.linalg.solve(prior_covariance, identity)
    # Ŝ⁻¹ = Kᵀ Sε⁻¹ K + Sₐ⁻¹
    posterior_precision = forward.T @ noise_precision_forward + prior_precision
    posterior_covariance = jnp.linalg.inv(posterior_precision)
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    # x̂ = xₐ + Ŝ Kᵀ Sε⁻¹ (y − K xₐ)
    innovation = measurement - forward @ prior_state
    retrieved = prior_state + posterior_covariance @ (
        noise_precision_forward.T @ innovation
    )
    # A = I − Ŝ Sₐ⁻¹
    averaging_kernel = identity - posterior_covariance @ prior_precision

    # --- dₛ and H from the singular values of the whitened Jacobian ---
    # K̃ = Lε⁻¹ K Lₐ
    whitened_forward = solve_triangular(noise_factor, forward, lower=True) @ (
        prior_factor
    )
    singular_values = jnp.linalg.svd(whitened_forward, compute_uv=False)
    signal_to_noise = singular_values**2
    # dₛ = ∑ᵢ σᵢ² / (1 + σᵢ²)
    degrees_of_freedom = jnp.sum(signal_to_noise / (1.0 + signal_to_noise))
    # H = ½ ∑ᵢ log(1 + σᵢ²)
    information_content = 0.5 * jnp.sum(jnp.log1p(signal_to_noise))

    # --- log N(y; K xₐ, K Sₐ Kᵀ + Sε) ---
    marginal_covariance = forward @ prior_covariance @ forward.T + noise_covariance
    marginal_factor = jnp.linalg.cholesky(marginal_covariance)
    whitened_innovation = solve_triangular(marginal_factor, innovation, lower=True)
    # log det C = 2 ∑ log diag(L_C)
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(marginal_factor)))
    # −½ (y − K xₐ)ᵀ C⁻¹ (y − K xₐ) − ½ log det C − (M/2) log 2π
    log_evidence = (
        -0.5 * jnp.sum(whitened_innovation**2)
        - 0.5 * log_determinant
        - 0.5 * n_obs * math.log(2.0 * math.pi)
    )

    return LinearGaussian(
        K=forward,
        S_eps=noise_covariance,
        S_a=prior_covariance,
        x_a=prior_state,
        y=measurement,
        x_hat=retrieved,
        S_hat=posterior_covariance,
        A=averaging_kernel,
        d_s=degrees_of_freedom,
        H_info=information_content,
        log_evidence=log_evidence,
    )


def whitened_residuals(
    problem: LinearGaussian,
) -> Callable[[Float[Array, " N"], Any], tuple[Float[Array, " M"], Float[Array, " N"]]]:
    """Return the whitened residuals of a linear-Gaussian problem.

    r(x) = (Lε⁻¹ (y − K x), Lₐ⁻¹ (x − xₐ)),  so  J(x) = ½ ‖r(x)‖²

    Lε and Lₐ are the Cholesky factors of Sε and Sₐ, one choice of
    Sε^(1/2) and Sₐ^(1/2). The function takes optimistix's ``fn(y, args)``
    form and ignores ``args``, so it goes straight into
    ``optx.least_squares``.

    Args:
        problem: The problem whose residuals to build.

    Returns:
        The residual function x ↦ r(x), a pair of shapes (M,) and (N,).

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> problem = linear_gaussian_problem(jax.random.key(0), 3, 5)
        >>> measurement, prior = whitened_residuals(problem)(problem.x_a, None)
        >>> measurement.shape, prior.shape
        ((5,), (3,))
        >>> bool(jnp.allclose(prior, 0.0))
        True
    """
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    prior_factor = jnp.linalg.cholesky(problem.S_a)

    def residuals(
        state: Float[Array, " N"], args: Any
    ) -> tuple[Float[Array, " M"], Float[Array, " N"]]:
        del args
        # Lε⁻¹ (y − K x)
        measurement = solve_triangular(
            noise_factor, problem.y - problem.K @ state, lower=True
        )
        # Lₐ⁻¹ (x − xₐ)
        prior = solve_triangular(prior_factor, state - problem.x_a, lower=True)
        return measurement, prior

    return residuals


def _relative_error(actual: Array, expected: Array) -> float:
    """‖a − b‖ / ‖b‖, or ‖a − b‖ when b is zero."""
    difference = np.linalg.norm(np.ravel(actual) - np.ravel(expected))
    scale = np.linalg.norm(np.ravel(expected))
    return float(difference / scale) if scale > 0 else float(difference)


def _assert_close(actual: Any, expected: Any, *, rtol: float, what: str) -> None:
    """Assert ‖a − b‖ ≤ rtol ‖b‖ (norm-wise, so entries near zero do not fail)."""
    error = _relative_error(np.asarray(actual), np.asarray(expected))
    assert error <= rtol, f"{what}: relative error {error:.3e} exceeds {rtol:.1e}"


def assert_matches_scipy_least_squares(
    residuals: Callable[[PyTree[Array], Any], PyTree[Array]],
    x0: PyTree[Array],
    solution: PyTree[Array],
    *,
    rtol: float,
    args: Any = None,
) -> None:
    """Assert that a least-squares solution matches SciPy's.

    x̂ = argminₓ ½ ‖r(x)‖²

    SciPy runs ``scipy.optimize.least_squares`` (trust-region reflective,
    with the Jacobian from `jax.jacfwd`) from the same start, with its
    ``xtol``, ``ftol`` and ``gtol`` at 1e-15, so its own error sits well
    below any ``rtol`` a test asks for.

    Args:
        residuals: The residual function r(x, args), returning a pytree.
        x0: Starting point, an array or a pytree of arrays.
        solution: The solution under test, with the structure of ``x0``.
        rtol: Allowed norm-wise relative difference from SciPy's solution.
        args: Passed to ``residuals`` as its second argument.

    Raises:
        AssertionError: If SciPy fails to converge or the solutions differ.
    """
    flat_start, unravel = ravel_pytree(x0)

    def flat_residuals(flat_state: Float[Array, " N"]) -> Float[Array, " R"]:
        return ravel_pytree(residuals(unravel(flat_state), args))[0]

    jacobian = jax.jacfwd(flat_residuals)
    reference = scipy.optimize.least_squares(
        lambda x: np.asarray(flat_residuals(jnp.asarray(x))),
        np.asarray(flat_start),
        jac=lambda x: np.asarray(jacobian(jnp.asarray(x))),
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
    )
    assert reference.success, f"SciPy did not converge: {reference.message}"
    _assert_close(
        ravel_pytree(solution)[0],
        reference.x,
        rtol=rtol,
        what="solution against scipy.optimize.least_squares",
    )


def assert_grad_matches_fd(
    fn: Callable[[PyTree[Array]], Float[Array, ""]],
    x: PyTree[Array],
    *,
    eps: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Assert that `jax.grad` of a scalar function matches central differences.

    ∂f/∂xᵢ ≈ (f(x + h eᵢ) − f(x − h eᵢ)) / 2h

    The truncation error is O(h²) and the rounding error O(ε_mach |f| / h),
    about 1e-10 relative at the default h = 1e-6. The defaults are for
    float64, which every argminax test runs in; float32 needs h near 5e-3
    and a much looser ``rtol``.

    Args:
        fn: The scalar function f.
        x: The point, an array or a pytree of arrays.
        eps: Step h.
        rtol: Allowed norm-wise relative difference.

    Raises:
        AssertionError: If the two gradients differ.
    """
    flat_point, unravel = ravel_pytree(x)
    # one compile for the 2N evaluations below
    flat_fn = jax.jit(lambda flat: fn(unravel(flat)))
    gradient = ravel_pytree(jax.grad(fn)(x))[0]
    directions = jnp.eye(flat_point.size, dtype=flat_point.dtype)
    finite_differences = jnp.stack(
        [
            (
                flat_fn(flat_point + eps * direction)
                - flat_fn(flat_point - eps * direction)
            )
            / (2.0 * eps)
            for direction in directions
        ]
    )
    _assert_close(
        gradient,
        finite_differences,
        rtol=rtol,
        what="gradient against finite differences",
    )


def assert_within_stderr(
    samples: Float[Array, "S ..."],
    exact: Array,
    *,
    n_sigma: float = 4.0,
) -> None:
    """Assert that a Monte Carlo mean lies within n_sigma standard errors.

    |x̄ − μ| ≤ t s / √S,  x̄ = (1/S) ∑ₛ xₛ,  s² = (1/(S − 1)) ∑ₛ (xₛ − x̄)²

    Pass the per-probe estimates (one Hutchinson term zᵀAz per probe, say),
    not their mean. Because s is itself estimated, the bound t is the
    Student-t quantile with S − 1 degrees of freedom at the two-sided level
    that n_sigma has under a Gaussian, so a correct estimator fails one check
    in about 16,000 per entry at 4σ for any S; t → n_sigma as S grows. A
    rounding floor of 100 ε_mach |μ| keeps a zero-variance estimator (a
    Rademacher probe on a diagonal operator) from failing on the last bit.

    Args:
        samples: Per-sample estimates, leading axis S ≥ 2.
        exact: The exact value μ, with the trailing shape of ``samples``.
        n_sigma: Allowed number of standard errors.

    Raises:
        AssertionError: If any entry lies outside its bound.
    """
    samples = np.asarray(samples)
    exact = np.asarray(exact)
    count = samples.shape[0]
    mean = samples.mean(axis=0)
    # s / √S, the standard error of the mean
    standard_error = samples.std(axis=0, ddof=1) / np.sqrt(count)
    # P(|Z| > n_sigma) for a standard normal Z, matched by the t quantile
    two_sided_level = 2.0 * scipy.stats.norm.sf(n_sigma)
    t_bound = scipy.stats.t.isf(0.5 * two_sided_level, count - 1)
    rounding = 100.0 * np.finfo(samples.dtype).eps * np.abs(exact)
    deviation = np.abs(mean - exact)
    bound = t_bound * standard_error + rounding
    worst = np.max(deviation / np.maximum(standard_error, np.finfo(float).tiny))
    assert np.all(deviation <= bound), (
        f"Monte Carlo mean is {worst:.2f} standard errors from the exact value "
        f"(allowed {t_bound:.2f}, the t quantile for {n_sigma}σ)"
    )


def assert_sample_covariance(
    samples: Float[Array, "S N"],
    covariance: Float[Array, "N N"],
    *,
    n_sigma: float = 4.0,
) -> None:
    """Assert that zero-mean draws have the given covariance, entry by entry.

    Σ̂ = (1/S) ∑ₛ xₛ xₛᵀ,  sd(Σ̂ᵢⱼ) = √((Σᵢⱼ² + Σᵢᵢ Σⱼⱼ) / S)

    The standard deviation is exact for S draws from N(0, Σ) with the mean
    known to be zero (Wishart moments), so pass draws with their mean
    removed by construction (perturbations about x̂), not by the sample
    mean. Each entry of Σ̂ is then close to Gaussian for S in the hundreds.

    Args:
        samples: Draws xₛ, shape (S, N).
        covariance: The covariance Σ they should have, shape (N, N).
        n_sigma: Allowed number of standard deviations per entry.

    Raises:
        AssertionError: If any entry of Σ̂ lies outside its bound.
    """
    samples = np.asarray(samples)
    covariance = np.asarray(covariance)
    count = samples.shape[0]
    # Σ̂ = (1/S) Xᵀ X
    sample_covariance = samples.T @ samples / count
    variance_diagonal = np.diag(covariance)
    # sd(Σ̂ᵢⱼ) = √((Σᵢⱼ² + Σᵢᵢ Σⱼⱼ) / S)
    entry_sd = np.sqrt(
        (covariance**2 + np.outer(variance_diagonal, variance_diagonal)) / count
    )
    deviation = np.abs(sample_covariance - covariance)
    worst = np.max(deviation / entry_sd)
    assert worst <= n_sigma, (
        f"sample covariance entry is {worst:.2f} standard deviations from the "
        f"target (allowed {n_sigma})"
    )


def assert_traces(
    fn: Callable[..., PyTree[Array]], *args: Any, rtol: float = 1e-12
) -> None:
    """Assert that eager, jit and vmap evaluations of a function agree.

    Every argument carries a leading batch axis B. The reference is an eager
    Python loop over the batch; `jax.jit` of each call, `jax.vmap` over the
    batch and `jax.jit` of the `jax.vmap` must each match it, leaf by leaf,
    in shape, dtype and value.

    Args:
        fn: The function, taking one batch element of each argument.
        *args: Pytrees with a common leading batch axis.
        rtol: Allowed norm-wise relative difference, per output leaf.

    Raises:
        AssertionError: If any evaluation differs from the eager loop.
    """
    batch = jax.tree_util.tree_leaves(args)[0].shape[0]

    def element(index: int) -> tuple[Any, ...]:
        return jax.tree_util.tree_map(lambda leaf: leaf[index], args)

    def stack(outputs: list[PyTree[Array]]) -> PyTree[Array]:
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *outputs)

    eager = stack([fn(*element(index)) for index in range(batch)])
    jitted_fn = jax.jit(fn)
    candidates = {
        "jit": stack([jitted_fn(*element(index)) for index in range(batch)]),
        "vmap": jax.vmap(fn)(*args),
        "jit(vmap)": jax.jit(jax.vmap(fn))(*args),
    }
    for name, outputs in candidates.items():
        expected_leaves, expected_structure = jax.tree_util.tree_flatten(eager)
        actual_leaves, actual_structure = jax.tree_util.tree_flatten(outputs)
        assert actual_structure == expected_structure, (
            f"{name}: output structure {actual_structure} differs from eager "
            f"{expected_structure}"
        )
        for actual, expected in zip(actual_leaves, expected_leaves, strict=True):
            assert actual.shape == expected.shape, (
                f"{name}: shape {actual.shape} differs from eager {expected.shape}"
            )
            assert actual.dtype == expected.dtype, (
                f"{name}: dtype {actual.dtype} differs from eager {expected.dtype}"
            )
            _assert_close(actual, expected, rtol=rtol, what=f"{name} against eager")

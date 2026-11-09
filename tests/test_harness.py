"""Tests for the reference harness in ``tests/_harness.py``.

Each helper must pass on its exact case and fail on a perturbed one, and the
closed forms of the linear-Gaussian problem must agree with each other: a
harness that is wrong would pass every solver it checks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from _harness import (
    LinearGaussian,
    assert_grad_matches_fd,
    assert_matches_scipy_least_squares,
    assert_sample_covariance,
    assert_traces,
    assert_within_stderr,
    linear_gaussian_problem,
    whitened_residuals,
)
from hypothesis import given, settings, strategies as st


def _relative(actual: jax.Array, expected: jax.Array) -> float:
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


def _cost(problem: LinearGaussian):
    residuals = whitened_residuals(problem)

    def cost(state: jax.Array) -> jax.Array:
        # J(x) = ½ ‖r(x)‖²
        measurement, prior = residuals(state, None)
        return 0.5 * (jnp.sum(measurement**2) + jnp.sum(prior**2))

    return cost


@pytest.fixture
def problem() -> LinearGaussian:
    return linear_gaussian_problem(jax.random.key(0), 4, 6)


# --- the closed forms agree with each other ---


def _assert_closed_forms_agree(problem: LinearGaussian) -> None:
    n_state = problem.x_a.shape[0]
    cost = _cost(problem)
    identity = jnp.eye(n_state)
    # Ŝ = (∇²J)⁻¹, from the dense Hessian of the whitened cost
    hessian = jax.hessian(cost)(problem.x_hat)
    assert _relative(problem.S_hat, jnp.linalg.inv(hessian)) <= 1e-12
    # ∇J(x̂) = 0, relative to the gradient at the prior
    gradient_at_minimum = jax.grad(cost)(problem.x_hat)
    gradient_at_prior = jax.grad(cost)(problem.x_a + 1.0)
    assert jnp.linalg.norm(gradient_at_minimum) <= 1e-12 * jnp.linalg.norm(
        gradient_at_prior
    )
    # A = Ŝ Kᵀ Sε⁻¹ K, the second form of the averaging kernel
    kernel = problem.S_hat @ problem.K.T @ jnp.linalg.solve(problem.S_eps, problem.K)
    assert _relative(problem.A, kernel) <= 1e-12
    # dₛ = tr A
    assert _relative(problem.d_s, jnp.trace(problem.A)) <= 1e-12
    # H = −½ log det(I − A) = −½ log det(Ŝ Sₐ⁻¹)
    _, log_det = jnp.linalg.slogdet(identity - problem.A)
    assert _relative(problem.H_info, -0.5 * log_det) <= 1e-12
    # Laplace is exact here: log p(y) = −J(x̂) + ½ log det Ŝ − ½ log det Sₐ
    #                                   − ½ log det(2π Sε)
    _, log_det_posterior = jnp.linalg.slogdet(problem.S_hat)
    _, log_det_prior = jnp.linalg.slogdet(problem.S_a)
    _, log_det_noise = jnp.linalg.slogdet(2.0 * jnp.pi * problem.S_eps)
    laplace = (
        -cost(problem.x_hat)
        + 0.5 * log_det_posterior
        - 0.5 * log_det_prior
        - 0.5 * log_det_noise
    )
    assert _relative(problem.log_evidence, laplace) <= 1e-12


def test_closed_forms_agree(problem: LinearGaussian) -> None:
    _assert_closed_forms_agree(problem)


@pytest.mark.slow
@settings(max_examples=10, deadline=None)
@given(
    n_state=st.integers(1, 6),
    n_obs=st.integers(1, 6),
    seed=st.integers(0, 2**31 - 1),
)
def test_closed_forms_agree_for_any_size(n_state: int, n_obs: int, seed: int) -> None:
    _assert_closed_forms_agree(
        linear_gaussian_problem(jax.random.key(seed), n_state, n_obs)
    )


def test_problem_keeps_the_requested_dtype() -> None:
    problem = linear_gaussian_problem(jax.random.key(0), 3, 2, dtype=jnp.float32)
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(problem)} == {
        jnp.dtype(jnp.float32)
    }


def test_residuals_vanish_on_the_prior_term_at_the_prior(
    problem: LinearGaussian,
) -> None:
    _, prior = whitened_residuals(problem)(problem.x_a, None)
    assert jnp.allclose(prior, 0.0)


def test_gauss_newton_reaches_the_closed_form(problem: LinearGaussian) -> None:
    solver = optx.GaussNewton(rtol=1e-12, atol=1e-12)
    solution = optx.least_squares(
        whitened_residuals(problem), solver, problem.x_a, throw=False
    )
    assert solution.result == optx.RESULTS.successful
    assert _relative(solution.value, problem.x_hat) <= 1e-12


# --- assert_matches_scipy_least_squares ---


def test_scipy_least_squares_passes_on_the_minimum(problem: LinearGaussian) -> None:
    assert_matches_scipy_least_squares(
        whitened_residuals(problem), problem.x_a, problem.x_hat, rtol=1e-10
    )


def test_scipy_least_squares_fails_off_the_minimum(problem: LinearGaussian) -> None:
    with pytest.raises(AssertionError, match="least_squares"):
        assert_matches_scipy_least_squares(
            whitened_residuals(problem),
            problem.x_a,
            problem.x_hat + 1e-6,
            rtol=1e-10,
        )


def test_scipy_least_squares_on_a_nonlinear_pytree_problem() -> None:
    # r(a, b) = (a exp(b tⱼ) − yⱼ), fitted from noiseless data
    times = jnp.linspace(0.0, 1.0, 8)
    data = 2.0 * jnp.exp(-0.5 * times)

    def residuals(params, args):
        return params["amplitude"] * jnp.exp(params["rate"] * times) - data

    start = {"amplitude": jnp.array(1.0), "rate": jnp.array(0.0)}
    truth = {"amplitude": jnp.array(2.0), "rate": jnp.array(-0.5)}
    assert_matches_scipy_least_squares(residuals, start, truth, rtol=1e-10)


# --- assert_grad_matches_fd ---


def test_grad_matches_fd_on_the_cost(problem: LinearGaussian) -> None:
    assert_grad_matches_fd(_cost(problem), problem.x_a + 0.3)


def test_grad_matches_fd_fails_on_a_wrong_gradient() -> None:
    @jax.custom_jvp
    def scaled_wrong(x):
        return jnp.sum(jnp.sin(x))

    @scaled_wrong.defjvp
    def _(primals, tangents):
        (x,), (dx,) = primals, tangents
        # 1 % too large: the bug the check must catch
        return scaled_wrong(x), 1.01 * jnp.sum(jnp.cos(x) * dx)

    with pytest.raises(AssertionError, match="finite differences"):
        assert_grad_matches_fd(scaled_wrong, jnp.array([0.1, 0.7, -0.4]))


# --- assert_within_stderr ---


def _hutchinson_terms(matrix: jax.Array, n_probes: int, key) -> jax.Array:
    # zᵀ A z for Gaussian probes z, each an unbiased estimate of tr A
    probes = jax.random.normal(key, (n_probes, matrix.shape[0]))
    return jnp.einsum("si,ij,sj->s", probes, matrix, probes)


def test_within_stderr_passes_on_a_hutchinson_trace(problem: LinearGaussian) -> None:
    terms = _hutchinson_terms(problem.A, 2000, jax.random.key(1))
    assert_within_stderr(terms, problem.d_s)


def test_within_stderr_fails_on_a_shifted_target(problem: LinearGaussian) -> None:
    terms = _hutchinson_terms(problem.A, 2000, jax.random.key(1))
    standard_error = np.std(np.asarray(terms), ddof=1) / np.sqrt(terms.shape[0])
    with pytest.raises(AssertionError, match="standard errors"):
        assert_within_stderr(terms, problem.d_s + 10.0 * standard_error)


def test_within_stderr_accepts_a_zero_variance_estimator() -> None:
    # A Rademacher probe gives tr D exactly for a diagonal D
    diagonal = jnp.arange(1.0, 5.0)
    probes = jax.random.rademacher(jax.random.key(2), (16, 4), dtype=jnp.float64)
    terms = jnp.sum(probes * diagonal * probes, axis=1)
    assert_within_stderr(terms, jnp.sum(diagonal))


# --- assert_sample_covariance ---


def _posterior_draws(problem: LinearGaussian, count: int, key) -> jax.Array:
    # Lŝ z with z ~ N(0, I): zero-mean draws with covariance Ŝ
    factor = jnp.linalg.cholesky(problem.S_hat)
    return jax.random.normal(key, (count, factor.shape[0])) @ factor.T


def test_sample_covariance_passes_on_the_posterior(problem: LinearGaussian) -> None:
    draws = _posterior_draws(problem, 4000, jax.random.key(3))
    assert_sample_covariance(draws, problem.S_hat)


def test_sample_covariance_fails_on_a_scaled_target(problem: LinearGaussian) -> None:
    draws = _posterior_draws(problem, 4000, jax.random.key(3))
    with pytest.raises(AssertionError, match="standard deviations"):
        assert_sample_covariance(draws, 1.2 * problem.S_hat)


# --- assert_traces ---


@pytest.mark.slow
def test_traces_agree_on_a_batch_of_problems() -> None:
    keys = jax.random.split(jax.random.key(4), 3)
    problems = jax.vmap(lambda key: linear_gaussian_problem(key, 3, 4))(keys)

    def retrieve(problem: LinearGaussian) -> tuple[jax.Array, jax.Array]:
        # x̂ = xₐ − Ŝ ∇J(xₐ), one Newton step, exact for a quadratic J
        gradient = jax.grad(_cost(problem))(problem.x_a)
        return problem.x_a - problem.S_hat @ gradient, problem.d_s

    assert_traces(retrieve, problems)


def test_traces_fail_on_a_function_with_python_state() -> None:
    calls: list[int] = []

    def counting(x: jax.Array) -> jax.Array:
        # Each eager call sees a new count; a trace sees one count for all
        calls.append(1)
        return x * len(calls)

    with pytest.raises(AssertionError, match="against eager"):
        assert_traces(counting, jnp.ones((3, 2)))


def test_traces_fail_on_a_shape_change_under_trace() -> None:
    def reshaping(x: jax.Array) -> jax.Array:
        # (1,) under a tracer, () eagerly: same values, different shape
        if isinstance(x, jax.core.Tracer):
            return x.sum(keepdims=True)
        return x.sum()

    with pytest.raises(AssertionError, match="shape"):
        assert_traces(reshaping, jnp.ones((3, 2)))


def test_within_stderr_holds_its_level_for_few_samples() -> None:
    # 2000 correct estimates from S = 3 draws each: the Gaussian bound would
    # fail about 6 % of them, the t bound about 1 in 16,000
    draws = jax.random.normal(jax.random.key(5), (2000, 3))
    failures = 0
    for trial in draws:
        try:
            assert_within_stderr(trial, jnp.array(0.0))
        except AssertionError:
            failures += 1
    assert failures <= 2

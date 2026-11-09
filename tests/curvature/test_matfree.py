"""Tests for `marginal_variance`, `low_rank` and `sample_by_perturbation`.

Every check is against the dense Ŝ of the harness's linear-Gaussian problem,
with stochastic estimates bounded by their own sampling distribution.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from _harness import (
    assert_sample_covariance,
    linear_gaussian_problem,
    whitened_residuals,
)
from jax.scipy.linalg import solve_triangular

from argminax import ValidationError
from argminax.curvature import (
    Laplace,
    VarianceEstimate,
    gauss_newton,
    low_rank,
    marginal_variance,
    sample_by_perturbation,
)


def _relative(actual, expected) -> float:
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


@pytest.fixture
def problem():
    return linear_gaussian_problem(jax.random.key(0), 5, 7)


def _whitened_control(problem):
    """Residuals in the whitened control ξ (prior N(0, I)), and the exact Ŝ_ξ."""
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    prior_factor = jnp.linalg.cholesky(problem.S_a)

    def residuals(control, args):
        # x = xₐ + Lₐ ξ, so r(ξ) = (Lε⁻¹ (y − K (xₐ + Lₐ ξ)), ξ)
        state = problem.x_a + prior_factor @ control
        return (
            solve_triangular(noise_factor, problem.y - problem.K @ state, lower=True),
            control,
        )

    mode = solve_triangular(prior_factor, problem.x_hat - problem.x_a, lower=True)
    # Ŝ_ξ = Lₐ⁻¹ Ŝ Lₐ⁻ᵀ
    left = solve_triangular(prior_factor, problem.S_hat, lower=True)
    covariance = solve_triangular(prior_factor, left.T, lower=True)
    return residuals, mode, covariance


# --- marginal variances ---


@pytest.mark.parametrize("dense", [True, False])
def test_exact_marginal_variance(problem, dense: bool) -> None:
    posterior = gauss_newton(whitened_residuals(problem), problem.x_hat, dense=dense)
    variance = marginal_variance(posterior)
    assert _relative(variance, jnp.diag(problem.S_hat)) <= 1e-10


def test_probed_diagonal_is_within_four_standard_errors(problem) -> None:
    posterior = gauss_newton(whitened_residuals(problem), problem.x_hat, dense=False)
    probed = marginal_variance(posterior, num_probes=400, key=jax.random.key(1))
    assert isinstance(probed, VarianceEstimate)
    assert jnp.all(probed.stderr > 0.0)
    deviation = jnp.abs(probed.estimate - jnp.diag(problem.S_hat))
    assert jnp.all(deviation <= 4.0 * probed.stderr)


@pytest.mark.slow
def test_vmap_over_pixels_gives_independent_probes() -> None:
    keys = jax.random.split(jax.random.key(2), 4)
    problems = jax.vmap(lambda key: linear_gaussian_problem(key, 3, 4))(keys)
    probe_keys = jax.random.split(jax.random.key(3), 4)

    def probe(problem, key):
        posterior = gauss_newton(whitened_residuals(problem), problem.x_hat)
        return marginal_variance(posterior, num_probes=200, key=key)

    probed = jax.vmap(probe)(problems, probe_keys)
    exact = jax.vmap(jnp.diag)(problems.S_hat)
    assert jnp.all(jnp.abs(probed.estimate - exact) <= 4.0 * probed.stderr)
    # the same problem in every pixel: a shared key would repeat the estimate
    same = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf[0], leaf.shape), problems
    )
    repeated = jax.vmap(probe)(same, probe_keys)
    assert not jnp.allclose(repeated.estimate[0], repeated.estimate[1])
    shared = jax.vmap(probe)(same, jnp.stack([probe_keys[0]] * 4))
    assert jnp.allclose(shared.estimate[0], shared.estimate[1])


# --- low rank ---


def test_low_rank_is_exact_at_the_number_of_measurements() -> None:
    # 8 unknowns, 3 measurements: the data's curvature has rank 3
    problem = linear_gaussian_problem(jax.random.key(4), 8, 3)
    residuals, mode, covariance = _whitened_control(problem)
    posterior = gauss_newton(residuals, mode)
    approximation = low_rank(posterior, 3, key=jax.random.key(5))
    assert _relative(approximation.as_matrix(), covariance) <= 1e-10
    assert lx.is_positive_semidefinite(approximation)


@pytest.mark.slow
def test_low_rank_error_shrinks_with_rank() -> None:
    problem = linear_gaussian_problem(jax.random.key(6), 12, 9)
    residuals, mode, covariance = _whitened_control(problem)
    posterior = gauss_newton(residuals, mode)
    errors = [
        _relative(
            low_rank(posterior, rank, key=jax.random.key(7)).as_matrix(), covariance
        )
        for rank in (2, 5, 9)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] <= 1e-10
    # a truncated form keeps a valid covariance
    truncated = low_rank(posterior, 2, key=jax.random.key(7)).as_matrix()
    assert jnp.all(jnp.linalg.eigvalsh(truncated) > 0.0)


@pytest.mark.parametrize("rank", [0, 6, 1.5])
def test_rank_must_be_from_one_to_n(problem, rank) -> None:
    posterior = gauss_newton(whitened_residuals(problem), problem.x_hat)
    with pytest.raises(ValidationError, match="'rank' must be"):
        low_rank(posterior, rank, key=jax.random.key(0))


# --- perturbation sampling ---


@pytest.mark.parametrize("dense", [True, False])
def test_perturbation_draws_have_the_posterior_covariance(problem, dense) -> None:
    residuals = whitened_residuals(problem)
    posterior = gauss_newton(residuals, problem.x_hat, dense=dense)
    draws = sample_by_perturbation(residuals, posterior, 10_000, key=jax.random.key(8))
    assert draws.shape == (10_000, 5)
    assert_sample_covariance(draws - problem.x_hat, problem.S_hat)


def test_perturbation_draws_pass_args_through() -> None:
    problem = linear_gaussian_problem(jax.random.key(9), 3, 4)
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    prior_factor = jnp.linalg.cholesky(problem.S_a)

    def residuals(state, measurement):
        return (
            solve_triangular(noise_factor, measurement - problem.K @ state, lower=True),
            solve_triangular(prior_factor, state - problem.x_a, lower=True),
        )

    posterior = gauss_newton(residuals, problem.x_hat, problem.y)
    draws = sample_by_perturbation(
        residuals, posterior, 4000, problem.y, key=jax.random.key(10)
    )
    assert_sample_covariance(draws - problem.x_hat, problem.S_hat)


@pytest.mark.parametrize("bad", [0, -2, 2.0])
def test_draw_and_probe_counts_are_validated(problem, bad) -> None:
    residuals = whitened_residuals(problem)
    posterior = gauss_newton(residuals, problem.x_hat)
    with pytest.raises(ValidationError, match="'n' must be"):
        sample_by_perturbation(residuals, posterior, bad, key=jax.random.key(0))
    with pytest.raises(ValidationError, match="'num_probes' must be"):
        marginal_variance(posterior, num_probes=bad, key=jax.random.key(0))


@pytest.mark.parametrize("dense", [True, False])
def test_float32_stays_float32(dense: bool) -> None:
    # 20 unknowns, so the matrix-free CG must converge in float32 too
    problem = linear_gaussian_problem(jax.random.key(11), 20, 30, dtype=jnp.float32)
    residuals = whitened_residuals(problem)
    posterior = gauss_newton(residuals, problem.x_hat, dense=dense)
    assert marginal_variance(posterior).dtype == jnp.float32
    probed = marginal_variance(posterior, key=jax.random.key(0))
    assert probed.estimate.dtype == jnp.float32
    draws = sample_by_perturbation(residuals, posterior, 3, key=jax.random.key(0))
    assert draws.dtype == jnp.float32


def test_marginal_variance_and_low_rank_need_a_vector_mode() -> None:
    identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2, 3), jnp.float64))
    posterior = Laplace(
        jnp.zeros((2, 3)),
        lx.TaggedLinearOperator(identity, lx.positive_semidefinite_tag),
        gx.LineaxSolver(lx.CG(rtol=1e-8, atol=1e-8)),
    )
    with pytest.raises(ValidationError, match="one-dimensional mode"):
        marginal_variance(posterior)
    with pytest.raises(ValidationError, match="one-dimensional mode"):
        low_rank(posterior, 1, key=jax.random.key(0))

"""Tests for `Laplace`, `gauss_newton` and `push_forward`.

The gate (G1-closed-form, #9): for a linear forward model, `gauss_newton`
gives Ŝ equal to the dense (Kᵀ Sε⁻¹ K + Sₐ⁻¹)⁻¹ to 1e-10 in float64. The port
gate (G1-port) compares with vardax's covariances, stored in
``tests/curvature/data/`` (see its README).
"""

from __future__ import annotations

from pathlib import Path

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import (
    assert_grad_matches_fd,
    assert_sample_covariance,
    assert_traces,
    linear_gaussian_problem,
    whitened_residuals,
)
from jax.scipy.linalg import solve_triangular

from argminax import ValidationError
from argminax.curvature import Laplace, gauss_newton, push_forward


FIXTURES = np.load(Path(__file__).parent / "data" / "vardax_posterior.npz")


def _relative(actual, expected) -> float:
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


@pytest.fixture
def problem():
    return linear_gaussian_problem(jax.random.key(0), 5, 7)


# --- gate G1-closed-form ---


@pytest.mark.parametrize("dense", [True, False])
def test_gate_covariance_is_the_closed_form(problem, dense: bool) -> None:
    posterior = gauss_newton(whitened_residuals(problem), problem.x_hat, dense=dense)
    assert _relative(posterior.covariance().as_matrix(), problem.S_hat) <= 1e-10


def test_dense_and_matrix_free_paths_agree(problem) -> None:
    residuals = whitened_residuals(problem)
    dense = gauss_newton(residuals, problem.x_hat, dense=True)
    matrix_free = gauss_newton(residuals, problem.x_hat, dense=False)
    assert (
        _relative(matrix_free.covariance().as_matrix(), dense.covariance().as_matrix())
        <= 1e-8
    )
    assert _relative(
        matrix_free.precision.as_matrix(), dense.precision.as_matrix()
    ) <= (1e-12)


@pytest.mark.parametrize("jac", ["fwd", "bwd"])
@pytest.mark.parametrize("dense", [True, False])
def test_both_jacobian_modes_give_the_same_precision(problem, jac, dense) -> None:
    posterior = gauss_newton(
        whitened_residuals(problem), problem.x_hat, dense=dense, jac=jac
    )
    expected = jnp.linalg.inv(problem.S_hat)
    assert _relative(posterior.precision.as_matrix(), expected) <= 1e-10


def test_the_dense_threshold_picks_the_path(problem) -> None:
    posterior = gauss_newton(whitened_residuals(problem), problem.x_hat)
    assert isinstance(posterior.precision, lx.MatrixLinearOperator)
    assert isinstance(posterior.solver, gx.DenseSolver)


def test_logdet_precision_is_exact_when_dense(problem) -> None:
    posterior = gauss_newton(whitened_residuals(problem), problem.x_hat)
    _, log_det_covariance = jnp.linalg.slogdet(problem.S_hat)
    assert _relative(posterior.logdet_precision(), -log_det_covariance) <= 1e-10


def test_precision_is_tagged_psd_and_its_matrix_is_psd(problem) -> None:
    for dense in (True, False):
        posterior = gauss_newton(
            whitened_residuals(problem), problem.x_hat, dense=dense
        )
        assert lx.is_positive_semidefinite(posterior.precision)
        assert jnp.all(jnp.linalg.eigvalsh(posterior.precision.as_matrix()) > 0.0)
    assert lx.is_positive_semidefinite(posterior.covariance())


# --- sampling ---


def test_samples_have_the_posterior_covariance(problem) -> None:
    posterior = gauss_newton(whitened_residuals(problem), problem.x_hat)
    draws = posterior.sample(10_000, key=jax.random.key(1))
    assert draws.shape == (10_000, 5)
    assert_sample_covariance(draws - problem.x_hat, problem.S_hat)


def test_samples_depend_on_the_key(problem) -> None:
    posterior = gauss_newton(whitened_residuals(problem), problem.x_hat)
    first = posterior.sample(3, key=jax.random.key(1))
    again = posterior.sample(3, key=jax.random.key(1))
    other = posterior.sample(3, key=jax.random.key(2))
    assert jnp.array_equal(first, again)
    assert not jnp.allclose(first, other)


def test_float32_stays_float32() -> None:
    problem = linear_gaussian_problem(jax.random.key(3), 3, 4, dtype=jnp.float32)
    posterior = gauss_newton(whitened_residuals(problem), problem.x_hat)
    assert posterior.precision.as_matrix().dtype == jnp.float32
    assert posterior.sample(4, key=jax.random.key(0)).dtype == jnp.float32


# --- push_forward ---


def test_push_forward_recovers_the_physical_posterior(problem) -> None:
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    prior_factor = jnp.linalg.cholesky(problem.S_a)

    def whitened_control_residuals(control, args):
        # x = xₐ + Lₐ ξ, so r(ξ) = (Lε⁻¹ (y − K (xₐ + Lₐ ξ)), ξ)
        state = problem.x_a + prior_factor @ control
        measurement = solve_triangular(
            noise_factor, problem.y - problem.K @ state, lower=True
        )
        return measurement, control

    # ξ̂ = Lₐ⁻¹ (x̂ − xₐ)
    control_mode = solve_triangular(
        prior_factor, problem.x_hat - problem.x_a, lower=True
    )
    whitened = gauss_newton(whitened_control_residuals, control_mode)
    sqrt = lx.MatrixLinearOperator(prior_factor, lx.lower_triangular_tag)
    physical = push_forward(whitened, sqrt, problem.x_a)
    assert _relative(physical.mode, problem.x_hat) <= 1e-12
    assert _relative(physical.covariance().as_matrix(), problem.S_hat) <= 1e-10


# --- gate G1-port: vardax's covariances ---


def _whiten(covariance: jax.Array):
    factor = jnp.linalg.cholesky(covariance)
    return lambda vector: solve_triangular(factor, vector, lower=True)


@pytest.mark.parametrize("adapter", ["laplace", "gauss_newton"])
def test_port_reproduces_vardax_three_dvar(adapter: str) -> None:
    observation = jnp.asarray(FIXTURES["threedvar_H"])
    whiten_noise = _whiten(jnp.asarray(FIXTURES["threedvar_R"]))
    whiten_prior = _whiten(jnp.asarray(FIXTURES["threedvar_B"]))
    background = jnp.asarray(FIXTURES["threedvar_xb"])
    measurement = jnp.asarray(FIXTURES["threedvar_y"])

    def residuals(state, args):
        return (
            whiten_noise(measurement - observation @ state),
            whiten_prior(state - background),
        )

    posterior = gauss_newton(residuals, jnp.asarray(FIXTURES["threedvar_analysis"]))
    expected = jnp.asarray(FIXTURES[f"threedvar_{adapter}"])
    assert _relative(posterior.covariance().as_matrix(), expected) <= 1e-10


@pytest.mark.parametrize("adapter", ["laplace", "gauss_newton"])
def test_port_reproduces_vardax_strong_four_dvar(adapter: str) -> None:
    dynamics = jnp.asarray(FIXTURES["fourdvar_A"])
    whiten_noise = _whiten(jnp.asarray(FIXTURES["fourdvar_R"]))
    whiten_prior = _whiten(jnp.asarray(FIXTURES["fourdvar_B"]))
    background = jnp.asarray(FIXTURES["fourdvar_xb"])
    window = jnp.asarray(FIXTURES["fourdvar_y"])

    def residuals(initial_state, args):
        # xₜ = Aᵗ x₀, observed at t = 0, 1, 2
        states = [initial_state]
        for _ in range(window.shape[0] - 1):
            states.append(dynamics @ states[-1])
        observed = [
            whiten_noise(window[time] - state) for time, state in enumerate(states)
        ]
        return (*observed, whiten_prior(initial_state - background))

    posterior = gauss_newton(residuals, jnp.asarray(FIXTURES["fourdvar_analysis"]))
    expected = jnp.asarray(FIXTURES[f"fourdvar_{adapter}"])
    assert _relative(posterior.covariance().as_matrix(), expected) <= 1e-10


# --- jit, vmap and grad ---


@pytest.mark.slow
def test_traces_under_jit_and_vmap() -> None:
    keys = jax.random.split(jax.random.key(4), 3)
    problems = jax.vmap(lambda key: linear_gaussian_problem(key, 3, 4))(keys)

    def summary(problem):
        posterior = gauss_newton(whitened_residuals(problem), problem.x_hat)
        return posterior.covariance().as_matrix(), posterior.logdet_precision()

    assert_traces(summary, problems)


def test_logdet_gradient_matches_finite_differences() -> None:
    measurement = jnp.array([0.3, -0.8, 1.1])

    def log_det(scale):
        def residuals(state, args):
            # r(x) = (y − s sin x, x): nonlinear in the scale s
            return measurement - scale[0] * jnp.sin(state), state

        return gauss_newton(residuals, jnp.array([0.2, -0.4, 0.9])).logdet_precision()

    assert_grad_matches_fd(log_det, jnp.array([1.3]))


# --- validation ---


def test_mode_must_be_one_dimensional(problem) -> None:
    with pytest.raises(ValidationError, match="'mode' must be one-dimensional"):
        gauss_newton(whitened_residuals(problem), problem.x_hat[None])


def test_jac_must_be_fwd_or_bwd(problem) -> None:
    with pytest.raises(ValidationError, match="'jac' must be 'fwd' or 'bwd'"):
        gauss_newton(whitened_residuals(problem), problem.x_hat, jac="central")


def test_precision_must_match_the_mode() -> None:
    precision = lx.MatrixLinearOperator(jnp.eye(3), lx.positive_semidefinite_tag)
    with pytest.raises(ValidationError, match="'precision' must be 2 × 2"):
        Laplace(jnp.zeros(2), precision)


def test_precision_must_be_tagged_psd() -> None:
    with pytest.raises(ValidationError, match="tagged positive semidefinite"):
        Laplace(jnp.zeros(2), lx.MatrixLinearOperator(jnp.eye(2)))

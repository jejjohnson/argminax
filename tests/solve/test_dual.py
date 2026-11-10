"""Tests for `dual_solve` and `incremental_dual_solve`.

Gate G3-dual (#11): on a linear H, `dual_solve` and `GaussNewtonCG` give the
same increment to the CG tolerance. The port reproduces radtrax's
``run_dual_psas`` with R = σ² I (fixture in ``tests/solve/data/``).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest
from _harness import assert_grad_matches_fd, linear_gaussian_problem
from jax.scipy.linalg import solve_triangular

from argminax import ValidationError
from argminax.solve import GaussNewtonCG, dual_solve, incremental_dual_solve


RADTRAX = np.load(Path(__file__).parent / "data" / "radtrax_solve.npz")
_PSD = (lx.symmetric_tag, lx.positive_semidefinite_tag)


def _relative(actual, expected) -> float:
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


def _psd(matrix):
    return lx.MatrixLinearOperator(matrix, _PSD)


@pytest.fixture
def problem():
    # more state than observations, the case the dual is for
    return linear_gaussian_problem(jax.random.key(0), 12, 5)


def _linear_dual(problem):
    innovation = problem.y - problem.K @ problem.x_a
    return dual_solve(
        lambda dx: problem.K @ dx,
        lambda dy: problem.K.T @ dy,
        _psd(problem.S_a),
        _psd(problem.S_eps),
        innovation,
    )


# --- gate G3-dual ---


def test_gate_dual_and_primal_increments_agree(problem) -> None:
    dual = _linear_dual(problem)
    assert dual.result == lx.RESULTS.successful
    # the closed form: x̂ − xₐ
    assert _relative(dual.value, problem.x_hat - problem.x_a) <= 1e-9
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    prior_factor = jnp.linalg.cholesky(problem.S_a)

    def residuals(increment, args):
        # r(δx) = (Lε⁻¹ (d − K δx), Lₐ⁻¹ δx)
        innovation = problem.y - problem.K @ problem.x_a
        return (
            solve_triangular(
                noise_factor, innovation - problem.K @ increment, lower=True
            ),
            solve_triangular(prior_factor, increment, lower=True),
        )

    primal = optx.least_squares(
        residuals,
        GaussNewtonCG(rtol=1e-12, atol=1e-12, max_cg_steps=100),
        jnp.zeros(12),
    )
    assert _relative(dual.value, primal.value) <= 1e-9


def test_the_dual_system_is_m_by_m(problem) -> None:
    dual = _linear_dual(problem)
    assert dual.multiplier.shape == (5,)
    assert dual.value.shape == (12,)
    # CG on 5 unknowns needs at most 5 iterations in exact arithmetic
    assert dual.num_steps <= 8


# --- port: radtrax's run_dual_psas ---


def test_port_reproduces_radtrax_dual_psas() -> None:
    observation = jnp.asarray(RADTRAX["jacobian"])
    background = float(RADTRAX["background_variance"])
    noise = 1.0 / float(RADTRAX["obs_inv_variance"])
    size, count = observation.shape[1], observation.shape[0]
    background_cov = lx.TaggedLinearOperator(
        background
        * lx.IdentityLinearOperator(jax.ShapeDtypeStruct((size,), jnp.float64)),
        _PSD,
    )
    # R = σ² I, radtrax's scalar R⁻¹
    obs_cov = lx.TaggedLinearOperator(
        noise * lx.IdentityLinearOperator(jax.ShapeDtypeStruct((count,), jnp.float64)),
        _PSD,
    )
    solution = dual_solve(
        lambda dx: observation @ dx,
        lambda dy: observation.T @ dy,
        background_cov,
        obs_cov,
        jnp.asarray(RADTRAX["innovation"]),
    )
    assert solution.result == lx.RESULTS.successful
    assert _relative(solution.value, RADTRAX["dual_delta"]) <= 1e-8


# --- the incremental outer loop ---


def test_incremental_loop_reaches_the_nonlinear_minimum(problem) -> None:
    def forward(state):
        # H(x) = K x + 0.1 sin(K x), mildly nonlinear
        linear = problem.K @ state
        return linear + 0.1 * jnp.sin(linear)

    analysis = incremental_dual_solve(
        forward,
        problem.x_a,
        _psd(problem.S_a),
        _psd(problem.S_eps),
        problem.y,
        num_outer=8,
    )
    assert analysis.result == lx.RESULTS.successful
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    prior_factor = jnp.linalg.cholesky(problem.S_a)

    def residuals(state, args):
        return (
            solve_triangular(noise_factor, problem.y - forward(state), lower=True),
            solve_triangular(prior_factor, state - problem.x_a, lower=True),
        )

    primal = optx.least_squares(
        residuals, GaussNewtonCG(rtol=1e-12, atol=1e-12, max_cg_steps=100), problem.x_a
    )
    assert _relative(analysis.value, primal.value) <= 1e-8


def test_one_outer_iteration_is_exact_for_a_linear_forward(problem) -> None:
    analysis = incremental_dual_solve(
        lambda state: problem.K @ state,
        problem.x_a,
        _psd(problem.S_a),
        _psd(problem.S_eps),
        problem.y,
        num_outer=1,
    )
    assert _relative(analysis.value, problem.x_hat) <= 1e-9


# --- differentiability, jit and vmap ---


def test_gradient_in_the_innovation_matches_finite_differences(problem) -> None:
    def increment_norm(innovation):
        solution = dual_solve(
            lambda dx: problem.K @ dx,
            lambda dy: problem.K.T @ dy,
            _psd(problem.S_a),
            _psd(problem.S_eps),
            innovation,
        )
        return jnp.sum(jnp.sin(solution.value))

    assert_grad_matches_fd(increment_norm, problem.y - problem.K @ problem.x_a)


def test_gradient_in_the_background_covariance(problem) -> None:
    innovation = problem.y - problem.K @ problem.x_a

    def increment_sum(scale):
        solution = dual_solve(
            lambda dx: problem.K @ dx,
            lambda dy: problem.K.T @ dy,
            _psd(scale[0] * problem.S_a),
            _psd(problem.S_eps),
            innovation,
        )
        return jnp.sum(solution.value)

    assert_grad_matches_fd(increment_sum, jnp.array([1.3]))


@pytest.mark.slow
def test_traces_under_jit_and_vmap() -> None:
    keys = jax.random.split(jax.random.key(1), 4)
    problems = jax.vmap(lambda key: linear_gaussian_problem(key, 6, 3))(keys)

    def pixel(problem):
        return _linear_dual(problem).value

    batched = jax.jit(jax.vmap(pixel))(problems)
    expected = problems.x_hat - problems.x_a
    assert jnp.allclose(batched, expected, rtol=1e-8, atol=1e-10)


# --- validation ---


def test_covariances_must_be_tagged_psd(problem) -> None:
    with pytest.raises(ValidationError, match="'obs_cov' must be tagged"):
        dual_solve(
            lambda dx: problem.K @ dx,
            lambda dy: problem.K.T @ dy,
            _psd(problem.S_a),
            lx.MatrixLinearOperator(problem.S_eps),
            problem.y,
        )


def test_num_outer_must_be_positive(problem) -> None:
    with pytest.raises(ValidationError, match="'num_outer' must be a positive integer"):
        incremental_dual_solve(
            lambda state: problem.K @ state,
            problem.x_a,
            _psd(problem.S_a),
            _psd(problem.S_eps),
            problem.y,
            num_outer=0,
        )

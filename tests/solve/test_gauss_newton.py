"""Tests for `GaussNewtonCG`.

Gates (#11): G3-linear, on a linear forward model one Gauss–Newton step reaches
x̂ to the CG tolerance; G3-scipy, small nonlinear problems match
``scipy.optimize.least_squares`` at the solver tolerance; G3-port, radtrax's
``run_gauss_newton`` is reproduced (fixture in ``tests/solve/data/``).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from _harness import (
    assert_matches_scipy_least_squares,
    linear_gaussian_problem,
    whitened_residuals,
)

from argminax import ValidationError
from argminax.solve import GaussNewtonCG


RADTRAX = np.load(Path(__file__).parent / "data" / "radtrax_solve.npz")


def _relative(actual, expected) -> float:
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


@pytest.fixture
def problem():
    return linear_gaussian_problem(jax.random.key(0), 8, 12)


def _solve(residuals, solver, start, args=None, **kwargs):
    return optx.least_squares(residuals, solver, start, args, throw=False, **kwargs)


# --- gate G3-linear ---


def test_gate_one_step_reaches_the_minimum(problem) -> None:
    solver = GaussNewtonCG(rtol=1e-12, atol=1e-12, max_cg_steps=100)
    # optimistix returns an update one step after computing it, so one
    # Gauss–Newton update is two optimistix steps
    one_update = _solve(whitened_residuals(problem), solver, problem.x_a, max_steps=2)
    assert _relative(one_update.value, problem.x_hat) <= 1e-10
    solution = _solve(whitened_residuals(problem), solver, problem.x_a)
    assert solution.result == optx.RESULTS.successful
    assert _relative(solution.value, problem.x_hat) <= 1e-10


@pytest.mark.parametrize(
    "solver",
    [
        GaussNewtonCG(rtol=1e-12, atol=1e-12, damping=0.5),
        GaussNewtonCG(rtol=1e-12, atol=1e-12, trust_region=True),
    ],
    ids=["fixed-damping", "trust-region"],
)
def test_levenberg_marquardt_reaches_the_same_minimum(problem, solver) -> None:
    solution = _solve(whitened_residuals(problem), solver, problem.x_a)
    assert solution.result == optx.RESULTS.successful
    assert _relative(solution.value, problem.x_hat) <= 1e-9


def test_a_truncated_cg_still_converges(problem) -> None:
    # two CG iterations per step: inexact steps, more of them, same minimum
    truncated = GaussNewtonCG(rtol=1e-12, atol=1e-12, max_cg_steps=2)
    exact = GaussNewtonCG(rtol=1e-12, atol=1e-12, max_cg_steps=100)
    residuals = whitened_residuals(problem)
    slow = _solve(residuals, truncated, problem.x_a, max_steps=500)
    fast = _solve(residuals, exact, problem.x_a)
    assert slow.result == optx.RESULTS.successful
    assert _relative(slow.value, problem.x_hat) <= 1e-9
    assert slow.stats["num_steps"] > fast.stats["num_steps"]


# --- gate G3-scipy ---


def _exponential_decay(params, args):
    # r(a, k) = a exp(−k t) − y, a nonlinear two-parameter fit
    times, data = args
    return params[0] * jnp.exp(-params[1] * times) - data


def _rosenbrock(params, args):
    # r(x) = (10 (x₂ − x₁²), 1 − x₁), minimum at (1, 1)
    return jnp.stack([10.0 * (params[1] - params[0] ** 2), 1.0 - params[0]])


@pytest.mark.parametrize("trust_region", [False, True], ids=["gauss-newton", "lm"])
def test_gate_matches_scipy_on_nonlinear_problems(trust_region: bool) -> None:
    solver = GaussNewtonCG(rtol=1e-12, atol=1e-12, trust_region=trust_region)
    times = jnp.linspace(0.0, 2.0, 9)
    data = 1.7 * jnp.exp(-0.8 * times) + 0.02 * jnp.sin(7.0 * times)
    start = jnp.array([1.0, 0.3])
    solution = _solve(_exponential_decay, solver, start, (times, data))
    assert solution.result == optx.RESULTS.successful
    assert_matches_scipy_least_squares(
        _exponential_decay, start, solution.value, rtol=1e-9, args=(times, data)
    )
    start = jnp.array([-1.2, 1.0])
    solution = _solve(_rosenbrock, solver, start, max_steps=500)
    assert solution.result == optx.RESULTS.successful
    assert_matches_scipy_least_squares(_rosenbrock, start, solution.value, rtol=1e-9)


# --- gate G3-port: radtrax's run_gauss_newton ---


def test_gate_port_reproduces_radtrax_gauss_newton() -> None:
    observation = jnp.asarray(RADTRAX["jacobian"])
    innovation = jnp.asarray(RADTRAX["innovation"])
    sqrt_background = jnp.asarray(RADTRAX["sqrt_background"])
    noise_scale = jnp.sqrt(jnp.asarray(RADTRAX["obs_inv_variance"]))

    def residuals(control, args):
        # r(ξ) = (ξ, √R⁻¹ (d − H L ξ))
        return control, noise_scale * (
            innovation - observation @ (sqrt_background @ control)
        )

    start = jnp.zeros(observation.shape[1])
    # radtrax asked for atol = 1e-14, which optimistix's Cauchy test cannot meet
    # here (plain optx.GaussNewton runs out of steps too, and radtrax never
    # checks); 1e-10 converges in radtrax's recorded 3 steps
    solution = _solve(residuals, GaussNewtonCG(rtol=1e-10, atol=1e-10), start)
    assert solution.result == optx.RESULTS.successful
    assert solution.stats["num_steps"] == RADTRAX["gauss_newton_steps"]
    assert _relative(solution.value, RADTRAX["gauss_newton_xi"]) <= 1e-8
    increment = sqrt_background @ solution.value
    assert _relative(increment, RADTRAX["gauss_newton_delta"]) <= 1e-8


# --- vmap over pixels, gradients ---


@pytest.mark.slow
def test_vmap_over_pixels_reports_per_pixel_steps() -> None:
    keys = jax.random.split(jax.random.key(1), 5)
    problems = jax.vmap(lambda key: linear_gaussian_problem(key, 4, 6))(keys)
    solver = GaussNewtonCG(rtol=1e-12, atol=1e-12)

    def pixel(problem):
        solution = _solve(whitened_residuals(problem), solver, problem.x_a)
        return solution.value, solution.stats["num_steps"], solution.result._value

    values, steps, results = jax.jit(jax.vmap(pixel))(problems)
    assert steps.shape == (5,)
    assert jnp.all(results == optx.RESULTS.successful._value)
    assert jnp.allclose(values, problems.x_hat, rtol=1e-9)


def test_implicit_gradient_through_the_solve_is_the_closed_form(problem) -> None:
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    prior_factor = jnp.linalg.cholesky(problem.S_a)

    def residuals(state, measurement):
        return (
            jax.scipy.linalg.solve_triangular(
                noise_factor, measurement - problem.K @ state, lower=True
            ),
            jax.scipy.linalg.solve_triangular(
                prior_factor, state - problem.x_a, lower=True
            ),
        )

    solver = GaussNewtonCG(rtol=1e-12, atol=1e-12)

    def retrieve(measurement):
        solution = optx.least_squares(
            residuals, solver, problem.x_a, measurement, adjoint=optx.ImplicitAdjoint()
        )
        return solution.value

    # ∂x̂/∂y = Ŝ Kᵀ Sε⁻¹
    exact = problem.S_hat @ problem.K.T @ jnp.linalg.inv(problem.S_eps)
    assert _relative(jax.jacrev(retrieve)(problem.y), exact) <= 1e-9


# --- validation ---


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"damping": -1.0}, "'damping' must be non-negative"),
        ({"damping": 0.5, "trust_region": True}, "'damping' must be 0"),
        ({"max_cg_steps": 0}, "'max_cg_steps' must be a positive integer"),
        ({"max_cg_steps": True}, "'max_cg_steps' must be a positive integer"),
    ],
)
def test_settings_are_validated(kwargs, message) -> None:
    with pytest.raises(ValidationError, match=message):
        GaussNewtonCG(rtol=1e-6, atol=1e-6, **kwargs)

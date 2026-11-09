"""Tests for the k-step and one-step adjoints.

The gate (G1-kstep, #9): on gradient descent for f(x, θ) = θx²/2 with unit
step, x₀ = 1 and θ = 0.5, two updates and k = 1 give ∂ξ₂/∂θ = −ξ₁ = −0.5,
and the forward value is bit-identical to the plain solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest
from _harness import (
    LinearGaussian,
    assert_grad_matches_fd,
    assert_traces,
    linear_gaussian_problem,
)
from jax.scipy.linalg import solve_triangular

from argminax import ArgminaxError, ValidationError
from argminax._src.adjoint import _kstep
from argminax.adjoint import KStep, OneStep


def _quadratic(x: jax.Array, theta: jax.Array) -> jax.Array:
    # f(x, θ) = θ x² / 2
    return 0.5 * theta * jnp.sum(x**2)


def _descend(theta: jax.Array, adjoint: optx.AbstractAdjoint, max_steps: int):
    # optimistix's first step only evaluates f at x₀, so n updates take n + 1 steps
    solver = optx.GradientDescent(learning_rate=1.0, rtol=1e-14, atol=1e-14)
    solution = optx.minimise(
        _quadratic,
        solver,
        jnp.ones(1),
        theta,
        adjoint=adjoint,
        max_steps=max_steps,
        throw=False,
    )
    return solution.value[0]


# --- gate G1-kstep ---


def test_gate_kstep_gradient_is_the_last_update() -> None:
    gradient = jax.grad(_descend)(jnp.array(0.5), KStep(k=1), 3)
    assert gradient == pytest.approx(-0.5, abs=1e-15)


def test_gate_kstep_forward_value_is_bit_identical() -> None:
    solve = jax.jit(_descend, static_argnums=(1, 2))
    plain = solve(jnp.array(0.5), optx.RecursiveCheckpointAdjoint(), 3)
    truncated = solve(jnp.array(0.5), KStep(k=1), 3)
    assert truncated == pytest.approx(0.25, abs=1e-15)
    assert truncated == plain


def test_k_covering_every_update_is_the_full_unrolled_gradient() -> None:
    theta = jnp.array(0.5)
    # ξ₂ = (1 − θ)², so ∂ξ₂/∂θ = −2 (1 − θ) = −1
    full = jax.grad(_descend)(theta, optx.RecursiveCheckpointAdjoint(), 3)
    for k in (2, 5):
        assert jax.grad(_descend)(theta, KStep(k=k), 3) == pytest.approx(
            float(full), abs=1e-15
        )
    assert full == pytest.approx(-1.0, abs=1e-15)


@pytest.mark.parametrize("max_steps", [2, 3, 4])
def test_a_solve_shorter_than_k_differentiates_the_start(max_steps: int) -> None:
    def from_start(start, adjoint):
        solver = optx.GradientDescent(learning_rate=1.0, rtol=1e-14, atol=1e-14)
        solution = optx.minimise(
            _quadratic,
            solver,
            start,
            jnp.array(0.5),
            adjoint=adjoint,
            max_steps=max_steps,
            throw=False,
        )
        return solution.value[0]

    start = jnp.ones(1)
    full = jax.grad(from_start)(start, optx.RecursiveCheckpointAdjoint())
    truncated = jax.grad(from_start)(start, KStep(k=5))
    # ξ_T = (1 − θ)ᵀ x₀ with T = max_steps − 1 updates
    assert full[0] == pytest.approx(0.5 ** (max_steps - 1), abs=1e-15)
    assert truncated[0] == pytest.approx(float(full[0]), abs=1e-15)
    _, tangent = jax.jvp(lambda x: from_start(x, KStep(k=5)), (start,), (start,))
    assert tangent == pytest.approx(float(full[0]), abs=1e-15)


def test_forward_mode_gives_the_same_gradient() -> None:
    def solve(theta):
        return _descend(theta, KStep(k=1), 3)

    _, tangent = jax.jvp(solve, (jnp.array(0.5),), (jnp.array(1.0),))
    assert tangent == pytest.approx(-0.5, abs=1e-15)


# --- one step at an exact minimum equals the implicit gradient ---


def _least_squares_retrieval(problem: LinearGaussian):
    """x̂(y) by Gauss–Newton on the whitened residuals, as a function of y."""
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    prior_factor = jnp.linalg.cholesky(problem.S_a)

    def residuals(state, measurement):
        return (
            solve_triangular(noise_factor, measurement - problem.K @ state, lower=True),
            solve_triangular(prior_factor, state - problem.x_a, lower=True),
        )

    def retrieve(measurement, adjoint):
        solution = optx.least_squares(
            residuals,
            optx.GaussNewton(rtol=1e-12, atol=1e-12),
            problem.x_a,
            measurement,
            adjoint=adjoint,
            throw=False,
        )
        return solution.value, solution.result

    return retrieve


def test_one_step_equals_the_implicit_gradient_for_an_exact_newton_step() -> None:
    problem = linear_gaussian_problem(jax.random.key(0), 4, 6)
    retrieve = _least_squares_retrieval(problem)
    _, result = retrieve(problem.y, OneStep())
    assert result == optx.RESULTS.successful
    one_step = jax.jacrev(lambda y: retrieve(y, OneStep())[0])(problem.y)
    implicit = jax.jacrev(lambda y: retrieve(y, optx.ImplicitAdjoint())[0])(problem.y)
    # ∂x̂/∂y = Ŝ Kᵀ Sε⁻¹, the closed form
    exact = problem.S_hat @ problem.K.T @ jnp.linalg.inv(problem.S_eps)
    assert jnp.linalg.norm(one_step - implicit) <= 1e-10 * jnp.linalg.norm(implicit)
    assert jnp.linalg.norm(one_step - exact) <= 1e-10 * jnp.linalg.norm(exact)


def test_one_step_gradient_matches_finite_differences() -> None:
    problem = linear_gaussian_problem(jax.random.key(1), 3, 5)
    retrieve = _least_squares_retrieval(problem)
    assert_grad_matches_fd(
        lambda y: jnp.sum(jnp.sin(retrieve(y, OneStep())[0])), problem.y
    )


# --- minimise and least_squares, under jit and vmap ---


def test_bfgs_forward_value_is_bit_identical_and_converged() -> None:
    def cost(x, theta):
        return jnp.sum(jnp.cosh(x - theta)) + 0.1 * jnp.sum(x**4)

    def solve(theta, adjoint):
        solution = optx.minimise(
            cost, optx.BFGS(rtol=1e-6, atol=1e-6), jnp.zeros(3), theta, adjoint=adjoint
        )
        return solution.value

    theta = jnp.array([0.3, -0.2, 0.5])
    assert jnp.array_equal(
        solve(theta, KStep(k=2)), solve(theta, optx.RecursiveCheckpointAdjoint())
    )
    # four updates through BFGS's line search reach the implicit gradient, to
    # the accuracy of a minimum found at rtol = 1e-6
    implicit = jax.jacrev(solve)(theta, optx.ImplicitAdjoint())
    four_step = jax.jacrev(solve)(theta, KStep(k=4))
    assert jnp.linalg.norm(four_step - implicit) <= 1e-4 * jnp.linalg.norm(implicit)


@pytest.mark.slow
def test_kstep_traces_under_jit_and_vmap() -> None:
    def gradient(theta):
        return jax.grad(_descend)(theta, KStep(k=2), 6)

    assert_traces(gradient, jnp.array([0.2, 0.5, 0.7]))


@pytest.mark.slow
def test_one_step_least_squares_traces_under_jit_and_vmap() -> None:
    problem = linear_gaussian_problem(jax.random.key(2), 3, 4)
    retrieve = _least_squares_retrieval(problem)
    measurements = problem.y + jax.random.normal(jax.random.key(3), (3, 4))
    assert_traces(
        lambda y: jax.jacrev(lambda m: retrieve(m, OneStep())[0])(y), measurements
    )


# --- the configuration contract ---


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_k_must_be_a_positive_integer(k) -> None:
    with pytest.raises(ValidationError, match="'k' must be a positive integer"):
        KStep(k=k)


def test_one_step_rejects_another_k() -> None:
    with pytest.raises(ValidationError, match=r"use KStep\(k=2\)"):
        OneStep(k=2)


@pytest.mark.parametrize("adjoint", [KStep(k=3), OneStep()])
def test_config_round_trips(adjoint: KStep) -> None:
    rebuilt = type(adjoint).from_config(adjoint.get_config())
    assert rebuilt == adjoint
    assert type(rebuilt) is type(adjoint)


# --- optimistix's private layout, which KStep reads ---


_SEEN: list[dict[str, bool]] = []


class _Spy(optx.AbstractAdjoint):
    """Check what optimistix passes to `apply` (inside the trace), then solve."""

    def apply(self, primal_fn, rewrite_fn, inputs, tags):
        fn, solver, start, args, options, max_steps, f_struct, aux_struct, _ = inputs
        structs = jax.tree_util.tree_leaves(
            (f_struct, aux_struct), is_leaf=_kstep._is_static
        )
        _SEEN.append(
            {
                "length": len(inputs) == 9,
                "fn": callable(fn),
                "solver": isinstance(solver, optx.AbstractIterativeSolver),
                "y0": jnp.shape(start) == (2,),
                "args": jnp.shape(args) == (),
                "options": isinstance(options, dict),
                "max_steps": isinstance(max_steps, int),
                "tags": isinstance(tags, frozenset),
                "structs": all(_kstep._is_static(leaf) for leaf in structs),
                "y_eval": _kstep._proposes_next_iterate(inputs),
            }
        )
        return optx.RecursiveCheckpointAdjoint().apply(
            primal_fn, rewrite_fn, inputs, tags
        )


@pytest.mark.parametrize("entry", ["minimise", "least_squares"])
def test_optimistix_inputs_layout(entry: str) -> None:
    """(fn, solver, y0, args, options, max_steps, f_struct, aux_struct, tags)."""
    _SEEN.clear()
    y0 = jnp.ones(2)
    if entry == "minimise":
        optx.minimise(_quadratic, optx.BFGS(1e-6, 1e-6), y0, 0.5, adjoint=_Spy())
    else:
        optx.least_squares(
            lambda x, a: a * x, optx.GaussNewton(1e-6, 1e-6), y0, 0.5, adjoint=_Spy()
        )
    (checks,) = _SEEN
    assert all(checks.values()), checks


def test_a_changed_inputs_layout_is_reported() -> None:
    with pytest.raises(ArgminaxError, match=r"pinned 0\.1\.x"):
        _kstep._proposes_next_iterate((None,) * 8)

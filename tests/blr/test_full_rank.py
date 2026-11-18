"""Tests for `blr_full_rank` and `newton`.

The port tests compare against optax_bayes's ``blr_full_rank_for_loss`` and
``newton_for_loss`` on the regression of ``_REGRESSION.py``; the rest pin the
step to its closed forms: exact recovery of a linear-Gaussian posterior in one
step (gate G1 of #58), `gaussx.blr_full_update` on the log-joint, and the
shift the damping gives the fixed point.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import linear_gaussian_problem
from _regression import REFERENCE, REGRESSION, run
from hypothesis import given, settings, strategies as st

from argminax import ValidationError
from argminax.blr import BLRFullRankState, blr_full_rank, newton


def _retrieval(n_state: int = 4, n_obs: int = 6):
    """A linear-Gaussian cost with an isotropic prior, and its exact posterior.

    𝒥(θ) = ½ (y − Kθ)ᵀ Σ_ε⁻¹ (y − Kθ), prior 𝒩(μ₀, δ⁻¹ I).
    """
    problem = linear_gaussian_problem(jax.random.key(3), n_state, n_obs)
    delta = 0.7
    noise_precision = jnp.linalg.inv(problem.S_eps)
    # C = Kᵀ Σ_ε⁻¹ K
    curvature = problem.K.T @ noise_precision @ problem.K
    # Λ = δ I + C,  μ = Λ⁻¹ (Kᵀ Σ_ε⁻¹ y + δ μ₀)
    precision = delta * jnp.eye(n_state) + curvature
    mean = jnp.linalg.solve(
        precision, problem.K.T @ noise_precision @ problem.y + delta * problem.x_a
    )

    def cost(theta):
        misfit = problem.y - problem.K @ theta
        return 0.5 * misfit @ noise_precision @ misfit

    return cost, curvature, delta, problem.x_a, precision, mean


def test_ggn_reproduces_optax_bayes() -> None:
    optimiser = blr_full_rank(
        0.05, prior_precision=0.5, prior_mean=REGRESSION.prior_mean, damping=1e-6
    )
    iterates, _ = run(optimiser, REGRESSION.start, REGRESSION.cost)
    np.testing.assert_allclose(iterates, REFERENCE["full_ggn"], rtol=1e-10)


def test_exact_curvature_reproduces_optax_bayes() -> None:
    optimiser = blr_full_rank(
        0.3,
        prior_precision=0.5,
        prior_mean=REGRESSION.prior_mean,
        curvature=REGRESSION.curvature,
        damping=1e-6,
    )
    iterates, _ = run(optimiser, REGRESSION.start, REGRESSION.cost)
    np.testing.assert_allclose(iterates, REFERENCE["full_exact"], rtol=1e-10)


def test_newton_reproduces_optax_bayes() -> None:
    optimiser = newton(REGRESSION.curvature, damping=1e-6)
    iterates, _ = run(optimiser, REGRESSION.start, REGRESSION.cost)
    np.testing.assert_allclose(iterates, REFERENCE["newton"], rtol=1e-10)


def test_one_exact_step_recovers_the_linear_gaussian_posterior() -> None:
    cost, curvature, delta, prior_mean, precision, mean = _retrieval()
    optimiser = blr_full_rank(
        1.0,
        prior_precision=delta,
        prior_mean=prior_mean,
        curvature=lambda mean, grad: curvature,
        damping=0.0,
    )
    theta = jnp.full(4, 5.0)
    state = optimiser.init(theta)
    updates, state = optimiser.update(jax.grad(cost)(theta), state)
    np.testing.assert_allclose(state.precision, precision, rtol=1e-10)
    np.testing.assert_allclose(theta + updates, mean, rtol=1e-10)


def test_step_equals_the_gaussx_step_on_the_log_joint() -> None:
    cost, curvature, delta, prior_mean, _, _ = _retrieval()
    alpha, damping = 0.3, 0.05
    optimiser = blr_full_rank(
        alpha,
        prior_precision=delta,
        prior_mean=prior_mean,
        curvature=lambda mean, grad: curvature,
        damping=damping,
    )
    theta = jnp.linspace(-1.0, 2.0, 4)
    state = optimiser.init(theta)
    grads = jax.grad(cost)(theta)
    for _ in range(3):
        mean = jnp.linalg.solve(state.precision, state.nat_mean)
        # ∇ log p(𝒟, θ) = −g − δ (μ − μ₀),  ∇² log p(𝒟, θ) = −(C + δ I)
        nat_mean, precision = gx.blr_full_update(
            state.nat_mean,
            state.precision,
            -grads - delta * (mean - prior_mean),
            -(curvature + delta * jnp.eye(4)),
            alpha,
            convention="precision",
        )
        _, state = optimiser.update(grads, state)
        np.testing.assert_allclose(state.nat_mean, nat_mean, rtol=1e-12)
        np.testing.assert_allclose(
            state.precision, precision + damping * jnp.eye(4), rtol=1e-12
        )


def test_damping_shifts_the_fixed_point_by_damping_over_step_size() -> None:
    cost, curvature, delta, prior_mean, precision, _ = _retrieval()
    alpha, damping = 0.5, 0.1
    optimiser = blr_full_rank(
        alpha,
        prior_precision=delta,
        prior_mean=prior_mean,
        curvature=lambda mean, grad: curvature,
        damping=damping,
    )
    theta = jnp.zeros(4)
    state = optimiser.init(theta)

    def step(carry, _):
        theta, state = carry
        updates, state = optimiser.update(jax.grad(cost)(theta), state)
        return (theta + updates, state), None

    (_, state), _ = jax.lax.scan(step, (theta, state), length=200)
    # Λ_∞ = δ I + C + (β/α) I
    shifted = precision + (damping / alpha) * jnp.eye(4)
    np.testing.assert_allclose(state.precision, shifted, rtol=1e-8)


def test_newton_minimises_a_quadratic_cost() -> None:
    cost, _, _, _, _, _ = _retrieval()
    optimiser = newton(lambda mean, grad: jax.hessian(cost)(mean), damping=1e-10)
    theta = jnp.full(4, 2.0)
    state = optimiser.init(theta)
    for _ in range(3):
        updates, state = optimiser.update(jax.grad(cost)(theta), state)
        theta = theta + updates
    assert float(jnp.linalg.norm(jax.grad(cost)(theta))) < 1e-8


def test_jit_and_scan_agree_with_an_eager_loop() -> None:
    optimiser = blr_full_rank(0.05, prior_precision=0.5)
    eager, _ = run(optimiser, REGRESSION.start, REGRESSION.cost)

    def step(carry, _):
        theta, state = carry
        updates, state = optimiser.update(jax.grad(REGRESSION.cost)(theta), state)
        return (theta + updates, state), theta + updates

    _, scanned = jax.jit(
        lambda carry: jax.lax.scan(step, carry, length=eager.shape[0])
    )((REGRESSION.start, optimiser.init(REGRESSION.start)))
    np.testing.assert_allclose(scanned, eager, rtol=1e-10)


@pytest.mark.parametrize("prior_mean", [jnp.zeros(1), jnp.zeros(2), 0.0])
def test_prior_mean_of_another_shape_is_rejected_not_broadcast(prior_mean) -> None:
    optimiser = blr_full_rank(0.1, prior_mean=prior_mean)
    with pytest.raises(ValidationError, match="prior_mean"):
        optimiser.init(jnp.zeros(3))
    # a state built by another transform reaches update without this init
    state = blr_full_rank(0.1).init(jnp.zeros(3))
    with pytest.raises(ValidationError, match="prior_mean"):
        optimiser.update(jnp.ones(3), state)


def test_float32_parameters_stay_float32() -> None:
    theta = jnp.ones(3, jnp.float32)
    optimiser = blr_full_rank(0.1, prior_mean=jnp.zeros(3))
    state = optimiser.init(theta)
    # a float64 gradient must not promote the state
    updates, state = optimiser.update(jnp.full(3, 0.5, jnp.float64), state)
    assert isinstance(state, BLRFullRankState)
    assert updates.dtype == jnp.float32
    assert state.precision.dtype == jnp.float32
    assert state.nat_mean.dtype == jnp.float32


@pytest.mark.parametrize(
    "solver",
    [lx.Cholesky(), gx.CGSolver(rtol=1e-12, atol=1e-12)],
    ids=["lineax-cholesky", "gaussx-cg"],
)
def test_a_lineax_solver_and_a_gaussx_strategy_agree_with_the_default(solver) -> None:
    grads = jax.grad(REGRESSION.cost)(REGRESSION.start)
    updates = {}
    for name, choice in (("default", None), ("chosen", solver)):
        optimiser = blr_full_rank(0.2, prior_precision=0.5, solver=choice)
        state = optimiser.init(REGRESSION.start)
        updates[name], _ = optimiser.update(grads, state)
    np.testing.assert_allclose(updates["chosen"], updates["default"], rtol=1e-8)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), alpha=st.floats(0.01, 1.0))
def test_precision_stays_symmetric_positive_definite(seed, alpha) -> None:
    optimiser = blr_full_rank(alpha, prior_precision=1e-2)
    state = optimiser.init(jnp.zeros(3))
    gradients = jax.random.normal(jax.random.key(seed), (20, 3))
    for grads in gradients:
        _, state = optimiser.update(grads, state)
    np.testing.assert_allclose(state.precision, state.precision.T, rtol=1e-12)
    assert float(jnp.linalg.eigvalsh(state.precision).min()) > 0.0


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"curvature": "exact"}, "curvature"),
        ({"curvature": 3}, "curvature"),
        ({"learning_rate": 2.0}, "learning_rate"),
        ({"prior_precision": -1.0}, "prior_precision"),
        ({"damping": -1e-3}, "damping"),
    ],
)
def test_bad_options_are_rejected_by_name(kwargs, name) -> None:
    with pytest.raises(ValidationError, match=name):
        blr_full_rank(**kwargs)


@pytest.mark.parametrize(
    ("args", "kwargs", "name"),
    [(("ggn",), {}, "curvature"), ((lambda m, g: m,), {"damping": 0.0}, "damping")],
)
def test_newton_rejects_a_name_and_zero_damping(args, kwargs, name) -> None:
    with pytest.raises(ValidationError, match=name):
        newton(*args, **kwargs)

"""Tests for `blr_diagonal`, the Bayesian learning rule with a diagonal Gaussian.

The port tests compare against optax_bayes's ``blr_diagonal_for_loss`` on the
regression of ``conftest.py``; the rest pin the update to its closed forms and
to the optax protocol.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from _regression import REFERENCE, REGRESSION, run
from hypothesis import given, settings, strategies as st

from argminax import MissingExtraError, ValidationError
from argminax.blr import BLRDiagonalState, blr_diagonal


def test_ggn_reproduces_optax_bayes() -> None:
    optimiser = blr_diagonal(0.05, prior_precision=0.5, prior_mean=0.1, damping=1e-6)
    start = REGRESSION.tree(REGRESSION.start)
    iterates, _ = run(optimiser, start, REGRESSION.tree_cost, REGRESSION.flat)
    np.testing.assert_allclose(iterates, REFERENCE["diag_ggn"], rtol=1e-10)


def test_identity_reproduces_optax_bayes() -> None:
    optimiser = blr_diagonal(0.01, prior_precision=2.0, curvature="identity")
    start = REGRESSION.tree(REGRESSION.start)
    iterates, _ = run(optimiser, start, REGRESSION.tree_cost, REGRESSION.flat)
    np.testing.assert_allclose(iterates, REFERENCE["diag_identity"], rtol=1e-10)


def test_schedule_reproduces_optax_bayes() -> None:
    schedule = optax.exponential_decay(0.05, transition_steps=10, decay_rate=0.5)
    optimiser = blr_diagonal(schedule, prior_precision=0.5, prior_mean=0.1)
    start = REGRESSION.tree(REGRESSION.start)
    iterates, _ = run(optimiser, start, REGRESSION.tree_cost, REGRESSION.flat)
    np.testing.assert_allclose(iterates, REFERENCE["diag_schedule"], rtol=1e-10)


def test_one_step_matches_the_update_equations() -> None:
    alpha, delta, prior_mean = 0.3, 2.0, 0.5
    theta = jnp.array([1.0, -2.0])
    grads = jnp.array([0.7, -1.5])
    optimiser = blr_diagonal(alpha, prior_precision=delta, prior_mean=prior_mean)
    state = optimiser.init(theta)
    updates, state = optimiser.update(grads, state)
    # Λ_θ ← (1 − α) δ + α (δ + g²)
    precision = (1 - alpha) * delta + alpha * (delta + grads**2)
    # Λ_θ μ_θ ← (1 − α) δ θ + α (δ μ₀ − g + g² θ)
    nat_mean = (1 - alpha) * delta * theta + alpha * (
        delta * prior_mean - grads + grads**2 * theta
    )
    np.testing.assert_allclose(state.precision, precision, rtol=1e-12)
    np.testing.assert_allclose(state.nat_mean, nat_mean, rtol=1e-12)
    np.testing.assert_allclose(updates, nat_mean / precision - theta, rtol=1e-12)
    assert int(state.count) == 1


def test_identity_converges_to_the_stationary_point_of_cost_plus_prior() -> None:
    delta, prior_mean = 2.0, 0.1
    optimiser = blr_diagonal(
        0.02, prior_precision=delta, prior_mean=prior_mean, curvature="identity"
    )
    theta = REGRESSION.start
    state = optimiser.init(theta)

    def step(carry, _):
        theta, state = carry
        updates, state = optimiser.update(jax.grad(REGRESSION.cost)(theta), state)
        return (theta + updates, state), None

    (theta, _), _ = jax.lax.scan(step, (theta, state), length=4000)
    # (XᵀX + δ I) θ* = Xᵀy + δ μ₀
    gram = REGRESSION.inputs.T @ REGRESSION.inputs + delta * jnp.eye(4)
    expected = jnp.linalg.solve(
        gram, REGRESSION.inputs.T @ REGRESSION.targets + delta * prior_mean
    )
    np.testing.assert_allclose(theta, expected, rtol=1e-8)


def test_chains_with_clipping_on_a_nested_pytree() -> None:
    params = {
        "layer": {"w": jnp.ones((2, 3)), "b": jnp.zeros(2)},
        "scale": jnp.ones(()),
    }
    cost = lambda p: sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(p))
    optimiser = optax.chain(optax.clip_by_global_norm(1.0), blr_diagonal(0.1))
    state = optimiser.init(params)
    updates, state = optimiser.update(jax.grad(cost)(params), state)
    assert jax.tree.structure(updates) == jax.tree.structure(params)
    assert isinstance(state[1], BLRDiagonalState)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(updates))


def test_jit_and_scan_agree_with_an_eager_loop() -> None:
    optimiser = blr_diagonal(0.05, prior_precision=0.5)
    start = REGRESSION.tree(REGRESSION.start)
    eager, _ = run(optimiser, start, REGRESSION.tree_cost, REGRESSION.flat)

    def step(carry, _):
        params, state = carry
        updates, state = optimiser.update(jax.grad(REGRESSION.tree_cost)(params), state)
        params = optax.apply_updates(params, updates)
        return (params, state), REGRESSION.flat(params)

    _, scanned = jax.jit(
        lambda carry: jax.lax.scan(step, carry, length=eager.shape[0])
    )((start, optimiser.init(start)))
    np.testing.assert_allclose(scanned, eager, rtol=1e-12)


def test_float32_parameters_stay_float32() -> None:
    theta = jnp.ones(3, jnp.float32)
    optimiser = blr_diagonal(0.1)
    state = optimiser.init(theta)
    updates, state = optimiser.update(jnp.full(3, 0.5, jnp.float32), state)
    assert updates.dtype == jnp.float32
    assert state.precision.dtype == jnp.float32
    assert state.nat_mean.dtype == jnp.float32


@settings(max_examples=50, deadline=None)
@given(
    grad=st.floats(-1e6, 1e6, allow_nan=False),
    alpha=st.floats(1e-3, 1.0),
    damping=st.floats(0.0, 10.0),
)
def test_precision_never_falls_below_the_damping(grad, alpha, damping) -> None:
    optimiser = blr_diagonal(alpha, prior_precision=1e-4, damping=damping)
    state = optimiser.init(jnp.array([1.0]))
    for _ in range(3):
        _, state = optimiser.update(jnp.array([grad]), state)
    assert float(state.precision[0]) >= damping


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"curvature": "hutchinson"}, "curvature"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": 1.5}, "learning_rate"),
        ({"prior_precision": 0.0}, "prior_precision"),
        ({"damping": -1.0}, "damping"),
    ],
)
def test_bad_options_are_rejected_by_name(kwargs, name) -> None:
    with pytest.raises(ValidationError, match=name):
        blr_diagonal(**kwargs)


def test_missing_optax_names_the_extra(monkeypatch) -> None:
    # a None entry makes `import optax` raise ImportError, as an absent package does
    monkeypatch.setitem(__import__("sys").modules, "optax", None)
    with pytest.raises(MissingExtraError, match=r"argminax\[optax\]"):
        blr_diagonal()

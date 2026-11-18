"""Tests for `ivon`, against optax_bayes's ``ivon`` and its closed forms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from _regression import REFERENCE, REGRESSION, run

from argminax import ValidationError
from argminax.blr import IVONState, ivon


_OPTIONS = dict(beta1=0.9, beta2=0.99, weight_decay=0.1, clip_radius=1.0, ess=30.0)


def test_reproduces_optax_bayes() -> None:
    start = REGRESSION.tree(REGRESSION.start)
    iterates, _ = run(
        ivon(0.05, **_OPTIONS),
        start,
        REGRESSION.tree_cost,
        REGRESSION.flat,
        pass_params=True,
    )
    np.testing.assert_allclose(iterates, REFERENCE["ivon"], rtol=1e-10)


def test_reproduces_optax_bayes_without_params() -> None:
    start = REGRESSION.tree(REGRESSION.start)
    iterates, _ = run(
        ivon(0.05, **_OPTIONS), start, REGRESSION.tree_cost, REGRESSION.flat
    )
    np.testing.assert_allclose(iterates, REFERENCE["ivon_no_params"], rtol=1e-10)


def test_converges_to_the_weight_decay_minimiser() -> None:
    # 𝒥(θ) + ½ λ θ² with 𝒥(θ) = ½ (θ − 3)² is smallest at θ* = 3 / (1 + λ)
    weight_decay = 0.5
    optimiser = ivon(0.05, beta2=0.99, weight_decay=weight_decay)
    theta = jnp.array([0.0])
    state = optimiser.init(theta)

    def step(carry, _):
        theta, state = carry
        updates, state = optimiser.update(theta - 3.0, state, theta)
        return (theta + updates, state), None

    (theta, _), _ = jax.lax.scan(step, (theta, state), length=3000)
    np.testing.assert_allclose(theta, 3.0 / (1.0 + weight_decay), rtol=1e-6)


def test_schedule_is_read_at_the_step_count() -> None:
    schedule = optax.piecewise_constant_schedule(0.1, {1: 0.5})
    optimiser = ivon(schedule, beta1=0.0, weight_decay=1.0, hess_init=0.0, beta2=0.0)
    theta = jnp.array([0.0])
    state = optimiser.init(theta)
    grads = jnp.array([1.0])
    # with β₁ = β₂ = 0 and N = 1: Δθ = −α g / (g² + λ) = −α / 2
    first, state = optimiser.update(grads, state, theta)
    second, state = optimiser.update(grads, state, theta)
    np.testing.assert_allclose(first, -0.1 / 2.0, rtol=1e-12)
    np.testing.assert_allclose(second, -0.05 / 2.0, rtol=1e-12)


def test_jit_agrees_and_float32_stays_float32() -> None:
    theta = {"w": jnp.ones((2, 2), jnp.float32)}
    grads = {"w": jnp.full((2, 2), 0.3, jnp.float32)}
    optimiser = ivon(0.1)
    state = optimiser.init(theta)
    eager, eager_state = optimiser.update(grads, state, theta)
    jitted, jitted_state = jax.jit(optimiser.update)(grads, state, theta)
    assert isinstance(eager_state, IVONState)
    assert eager["w"].dtype == jnp.float32
    assert eager_state.hess["w"].dtype == jnp.float32
    np.testing.assert_allclose(jitted["w"], eager["w"], rtol=1e-6)
    np.testing.assert_allclose(jitted_state.hess["w"], eager_state.hess["w"], rtol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"weight_decay": 0.0}, "weight_decay"),
        ({"beta1": 1.0}, "beta1"),
        ({"beta2": -0.1}, "beta2"),
        ({"hess_init": -2.0}, "hess_init"),
        ({"ess": -1.0}, "ess"),
        ({"ess": 0.0}, "ess"),
    ],
)
def test_bad_options_are_rejected_by_name(kwargs, name) -> None:
    with pytest.raises(ValidationError, match=name):
        ivon(**kwargs)

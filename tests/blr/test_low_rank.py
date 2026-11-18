"""Tests for `blr_low_rank`, the rule with a low-rank-plus-diagonal precision.

The port tests compare against optax_bayes's ``blr_low_rank_for_loss`` on the
regression of ``_regression.py``; the rest pin SLANG's update: the precision
is exact at full rank, and its diagonal is exact at any rank.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd
from _regression import REFERENCE, REGRESSION, run
from hypothesis import given, settings, strategies as st

from argminax import ValidationError
from argminax.blr import BLRLowRankState, blr_full_rank, blr_low_rank


_PRIOR = dict(prior_precision=0.5, prior_mean=REGRESSION.prior_mean, damping=1e-6)


def _precision(state: BLRLowRankState):
    # Λ_θ = diag(d) + U Uᵀ
    factor = state.low_rank_factor
    return jnp.diag(state.diag_precision) + factor @ factor.T


def test_ggn_reproduces_optax_bayes() -> None:
    optimiser = blr_low_rank(0.05, rank=2, **_PRIOR)
    iterates, _ = run(optimiser, REGRESSION.start, REGRESSION.cost)
    np.testing.assert_allclose(iterates, REFERENCE["low_ggn"], rtol=1e-10)


def test_exact_curvature_reproduces_optax_bayes() -> None:
    optimiser = blr_low_rank(0.3, rank=2, curvature=REGRESSION.curvature, **_PRIOR)
    iterates, _ = run(optimiser, REGRESSION.start, REGRESSION.cost)
    np.testing.assert_allclose(iterates, REFERENCE["low_exact"], rtol=1e-10)


@pytest.mark.parametrize(
    "curvature", ["ggn", REGRESSION.curvature], ids=["ggn", "exact"]
)
def test_full_rank_factor_matches_blr_full_rank(curvature) -> None:
    # without damping: with it, the natural-mean recursion of blr_full_rank
    # also shrinks the mean by β Λ⁻¹ μ a step, which the mean-form step does not
    options = {**_PRIOR, "damping": 0.0, "curvature": curvature}
    low = blr_low_rank(0.3, rank=4, **options)
    full = blr_full_rank(0.3, **options)
    low_iterates, low_state = run(low, REGRESSION.start, REGRESSION.cost)
    full_iterates, full_state = run(full, REGRESSION.start, REGRESSION.cost)
    # the two take different routes to the same numbers (an SVD and a Woodbury
    # solve against a dense Cholesky), and the rounding compounds over 50
    # steps: the largest difference seen is 1.1e-8 relative, with "ggn"
    np.testing.assert_allclose(low_iterates, full_iterates, rtol=1e-6)
    np.testing.assert_allclose(_precision(low_state), full_state.precision, rtol=1e-6)


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    size=st.integers(2, 6),
    rank=st.integers(1, 3),
    alpha=st.floats(0.05, 1.0),
)
def test_diagonal_of_the_precision_is_exact_at_any_rank(
    seed, size, rank, alpha
) -> None:
    delta, damping = 0.3, 0.01
    optimiser = blr_low_rank(alpha, rank=rank, prior_precision=delta, damping=damping)
    state = optimiser.init(jnp.zeros(size))
    gradients = jax.random.normal(jax.random.key(seed), (5, size))
    for grads in gradients:
        before = jnp.diagonal(_precision(state))
        _, state = optimiser.update(grads, state)
        # diag Λ_θ ← (1 − α) diag Λ_θ + α (δ + g²) + β
        expected = (1 - alpha) * before + alpha * (delta + grads**2) + damping
        np.testing.assert_allclose(
            jnp.diagonal(_precision(state)), expected, rtol=1e-10
        )


def test_identity_converges_to_the_stationary_point_of_cost_plus_prior() -> None:
    delta = 2.0
    optimiser = blr_low_rank(
        0.02,
        rank=2,
        prior_precision=delta,
        prior_mean=REGRESSION.prior_mean,
        curvature="identity",
        damping=0.0,
    )
    theta = REGRESSION.start
    state = optimiser.init(theta)

    def step(carry, _):
        theta, state = carry
        updates, state = optimiser.update(jax.grad(REGRESSION.cost)(theta), state)
        return (theta + updates, state), None

    (theta, state), _ = jax.lax.scan(step, (theta, state), length=4000)
    # (XᵀX + δ I) θ* = Xᵀy + δ μ₀
    gram = REGRESSION.inputs.T @ REGRESSION.inputs + delta * jnp.eye(4)
    expected = jnp.linalg.solve(
        gram, REGRESSION.inputs.T @ REGRESSION.targets + delta * REGRESSION.prior_mean
    )
    np.testing.assert_allclose(theta, expected, rtol=1e-8)
    # with no curvature the factor stays empty of signal and d stays at δ
    np.testing.assert_allclose(state.diag_precision, delta, rtol=1e-12)


def test_rank_above_the_number_of_parameters_is_clamped_and_runs() -> None:
    optimiser = blr_low_rank(0.1, rank=50)
    state = optimiser.init(jnp.zeros(3))
    updates, state = optimiser.update(jnp.array([1.0, -2.0, 0.5]), state)
    assert state.low_rank_factor.shape == (3, 3)
    assert bool(jnp.all(jnp.isfinite(updates)))


def test_jit_and_scan_agree_with_an_eager_loop() -> None:
    optimiser = blr_low_rank(0.05, rank=2, prior_precision=0.5)
    eager, _ = run(optimiser, REGRESSION.start, REGRESSION.cost)

    def step(carry, _):
        theta, state = carry
        updates, state = optimiser.update(jax.grad(REGRESSION.cost)(theta), state)
        return (theta + updates, state), theta + updates

    _, scanned = jax.jit(
        lambda carry: jax.lax.scan(step, carry, length=eager.shape[0])
    )((REGRESSION.start, optimiser.init(REGRESSION.start)))
    np.testing.assert_allclose(scanned, eager, rtol=1e-10)


def _cost_after(scale, optimiser, num_steps):
    # the cost after ``num_steps`` steps from the start ``scale`` · 1
    theta = scale * jnp.ones_like(REGRESSION.start)
    state = optimiser.init(theta)
    for _ in range(num_steps):
        updates, state = optimiser.update(jax.grad(REGRESSION.cost)(theta), state)
        theta = theta + updates
    return REGRESSION.cost(theta)


def _assert_differentiable(optimiser, num_steps) -> None:
    def cost(scale):
        return _cost_after(scale, optimiser, num_steps)

    scale = jnp.asarray(0.3)
    reverse = jax.grad(cost)(scale)
    assert jnp.isfinite(reverse)
    np.testing.assert_allclose(reverse, jax.jacfwd(cost)(scale), rtol=1e-10)
    assert_grad_matches_fd(cost, scale)


@pytest.mark.parametrize("rank", [1, 2, REGRESSION.start.shape[0]])
@pytest.mark.parametrize("num_steps", [1, 3])
def test_gradient_through_training_from_init_is_finite_and_right(
    rank, num_steps
) -> None:
    # The first factor is [0, …, 0, √α g], whose zero singular values repeat:
    # the stock SVD derivative is NaN there. Rank 1 truncates from step two,
    # the full rank never does.
    optimiser = blr_low_rank(0.1, rank=rank, prior_precision=0.5)
    _assert_differentiable(optimiser, num_steps)


def _repeated(mean, grad):
    # C = (1 + μ₁²) I: every eigenvalue is the same, and moves with the mean
    return (1.0 + mean[0] ** 2) * jnp.eye(mean.shape[0], dtype=mean.dtype)


def _indefinite(mean, grad):
    # C = (1 + μ₂²) diag(1, −½, 2, 0, …): one negative and several zero eigenvalues
    signs = jnp.zeros_like(mean).at[:3].set(jnp.array([1.0, -0.5, 2.0]))
    return (1.0 + mean[1] ** 2) * jnp.diag(signs)


def _rank_one(mean, grad):
    # C = μ μᵀ: all but one eigenvalue is zero
    return jnp.outer(mean, mean)


@pytest.mark.parametrize("curvature", [_repeated, _indefinite, _rank_one])
@pytest.mark.parametrize("num_steps", [1, 3])
def test_gradient_through_a_callable_curvature_is_finite_and_right(
    curvature, num_steps
) -> None:
    # The stock eigendecomposition derivative is NaN at a repeated eigenvalue,
    # and the square root's at a zero or a dropped negative one.
    size = REGRESSION.start.shape[0]
    optimiser = blr_low_rank(0.1, rank=size, prior_precision=0.5, curvature=curvature)
    _assert_differentiable(optimiser, num_steps)


@pytest.mark.parametrize("prior_mean", [jnp.zeros(1), jnp.zeros(2), 0.0])
def test_prior_mean_of_another_shape_is_rejected_not_broadcast(prior_mean) -> None:
    optimiser = blr_low_rank(0.1, prior_mean=prior_mean)
    with pytest.raises(ValidationError, match="prior_mean"):
        optimiser.init(jnp.zeros(3))
    # a state built by another transform reaches update without this init
    state = blr_low_rank(0.1).init(jnp.zeros(3))
    with pytest.raises(ValidationError, match="prior_mean"):
        optimiser.update(jnp.ones(3), state)


def test_float32_parameters_stay_float32() -> None:
    theta = jnp.ones(3, jnp.float32)
    optimiser = blr_low_rank(0.1, rank=2, prior_mean=jnp.zeros(3))
    state = optimiser.init(theta)
    # a float64 gradient must not promote the state
    updates, state = optimiser.update(jnp.full(3, 0.5, jnp.float64), state)
    assert updates.dtype == jnp.float32
    assert state.diag_precision.dtype == jnp.float32
    assert state.low_rank_factor.dtype == jnp.float32
    assert state.nat_mean.dtype == jnp.float32


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"rank": 0}, "rank"),
        ({"rank": 2.0}, "rank"),
        ({"curvature": "exact"}, "curvature"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"prior_precision": 0.0}, "prior_precision"),
        ({"damping": -1.0}, "damping"),
    ],
)
def test_bad_options_are_rejected_by_name(kwargs, name) -> None:
    with pytest.raises(ValidationError, match=name):
        blr_low_rank(**kwargs)

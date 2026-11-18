"""Tests for `blr_posterior`, the `Laplace` read from an optimiser state.

The port tests compare the mode and the covariance against what optax_bayes's
``get_posterior_*`` functions return for the same final state; gates G2 and G3
of #58 tie the result to `argminax.curvature.gauss_newton` and to
`argminax.diagnose` on a linear-Gaussian problem.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import assert_sample_covariance, linear_gaussian_problem
from _regression import REFERENCE, REGRESSION, run
from jax.flatten_util import ravel_pytree

from argminax import ValidationError
from argminax.blr import (
    blr_diagonal,
    blr_full_rank,
    blr_low_rank,
    blr_posterior,
    ivon,
)
from argminax.curvature import Laplace, gauss_newton, marginal_variance
from argminax.diagnose import averaging_kernel, degrees_of_freedom


_PRIOR = dict(prior_precision=0.5, prior_mean=REGRESSION.prior_mean, damping=1e-6)
_PSD = lx.positive_semidefinite_tag


def _dense(operator: lx.AbstractLinearOperator):
    return operator.as_matrix()


def _diagonal_state():
    optimiser = blr_diagonal(0.05, prior_precision=0.5, prior_mean=0.1, damping=1e-6)
    start = REGRESSION.tree(REGRESSION.start)
    _, state = run(optimiser, start, REGRESSION.tree_cost, REGRESSION.flat)
    return state


def _full_rank_state():
    optimiser = blr_full_rank(0.3, curvature=REGRESSION.curvature, **_PRIOR)
    _, state = run(optimiser, REGRESSION.start, REGRESSION.cost)
    return state


def _low_rank_state():
    optimiser = blr_low_rank(0.3, rank=2, curvature=REGRESSION.curvature, **_PRIOR)
    _, state = run(optimiser, REGRESSION.start, REGRESSION.cost)
    return state


def test_diagonal_state_reproduces_optax_bayes() -> None:
    posterior = blr_posterior(_diagonal_state())
    # optax_bayes flattens the weights as (w, b); ravel_pytree sorts the keys,
    # so the reference is reordered to (b, w)
    order = np.array([2, 3, 0, 1])
    np.testing.assert_allclose(
        posterior.mode, REFERENCE["diag_ggn_mean"][order], rtol=1e-10
    )
    np.testing.assert_allclose(
        marginal_variance(posterior), REFERENCE["diag_ggn_variance"][order], rtol=1e-10
    )


def test_ivon_state_reproduces_optax_bayes() -> None:
    optimiser = ivon(
        0.05, beta1=0.9, beta2=0.99, weight_decay=0.1, clip_radius=1.0, ess=30.0
    )
    start = REGRESSION.tree(REGRESSION.start)
    iterates, state = run(
        optimiser, start, REGRESSION.tree_cost, REGRESSION.flat, pass_params=True
    )
    params = REGRESSION.tree(jnp.asarray(iterates[-1]))
    posterior = blr_posterior(state, params, weight_decay=0.1)
    order = np.array([2, 3, 0, 1])
    np.testing.assert_allclose(
        posterior.mode, REFERENCE["ivon_mean"][order], rtol=1e-10
    )
    np.testing.assert_allclose(
        marginal_variance(posterior), REFERENCE["ivon_variance"][order], rtol=1e-10
    )


@pytest.mark.parametrize(
    ("make_state", "name"),
    [(_full_rank_state, "full_exact"), (_low_rank_state, "low_exact")],
    ids=["full-rank", "low-rank"],
)
def test_dense_states_reproduce_optax_bayes(make_state, name) -> None:
    posterior = blr_posterior(make_state())
    np.testing.assert_allclose(posterior.mode, REFERENCE[f"{name}_mean"], rtol=1e-9)
    np.testing.assert_allclose(
        _dense(posterior.covariance()), REFERENCE[f"{name}_covariance"], rtol=1e-8
    )


def test_low_rank_precision_keeps_its_structure() -> None:
    posterior = blr_posterior(_low_rank_state())
    assert isinstance(posterior.precision, gx.LowRankUpdate)
    assert isinstance(posterior.covariance(), gx.LowRankUpdate)


def _exact_fit():
    """One exact full-rank step on a linear-Gaussian cost with an isotropic prior."""
    problem = linear_gaussian_problem(jax.random.key(3), 4, 6)
    delta = 0.7
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    noise_precision = jnp.linalg.inv(problem.S_eps)
    # C = Kᵀ Σ_ε⁻¹ K
    curvature = problem.K.T @ noise_precision @ problem.K

    def cost(theta):
        misfit = problem.y - problem.K @ theta
        return 0.5 * misfit @ noise_precision @ misfit

    def residuals(theta, args):
        # r(θ) = (L_ε⁻¹ (y − Kθ), √δ (θ − μ₀)), so 𝒥 + prior = ½ ‖r‖²
        measurement = jnp.linalg.solve(noise_factor, problem.y - problem.K @ theta)
        return measurement, jnp.sqrt(delta) * (theta - problem.x_a)

    optimiser = blr_full_rank(
        1.0,
        prior_precision=delta,
        prior_mean=problem.x_a,
        curvature=lambda mean, grad: curvature,
        damping=0.0,
    )
    theta = jnp.zeros(4)
    state = optimiser.init(theta)
    _, state = optimiser.update(jax.grad(cost)(theta), state)
    return state, residuals, delta, curvature


def test_exact_fit_has_the_gauss_newton_precision() -> None:
    state, residuals, _, _ = _exact_fit()
    posterior = blr_posterior(state)
    reference = gauss_newton(residuals, posterior.mode)
    np.testing.assert_allclose(
        _dense(posterior.precision), _dense(reference.precision), rtol=1e-10
    )


def test_degrees_of_freedom_of_an_exact_fit_match_the_closed_form() -> None:
    state, _, delta, curvature = _exact_fit()
    posterior = blr_posterior(state)
    prior_cov = lx.MatrixLinearOperator(jnp.eye(4) / delta, _PSD)
    kernel = averaging_kernel(posterior, prior_cov)
    # dₛ = tr(I − δ Λ_θ⁻¹),  Λ_θ = δ I + C
    precision = delta * jnp.eye(4) + curvature
    expected = jnp.trace(jnp.eye(4) - delta * jnp.linalg.inv(precision))
    np.testing.assert_allclose(degrees_of_freedom(kernel).value, expected, rtol=1e-8)


@pytest.mark.parametrize(
    "make_state",
    [_diagonal_state, _full_rank_state, _low_rank_state],
    ids=["diagonal", "full-rank", "low-rank"],
)
def test_samples_have_the_posterior_covariance(make_state) -> None:
    posterior = blr_posterior(make_state())
    draws = posterior.sample(20_000, key=jax.random.key(7))
    # perturbations about the mode: zero mean by construction, as
    # assert_sample_covariance requires for its Wishart bound
    assert_sample_covariance(draws - posterior.mode, _dense(posterior.covariance()))


def test_pytree_state_is_flattened_in_ravel_pytree_order() -> None:
    params = {"layer": {"w": jnp.arange(6.0).reshape(2, 3), "b": jnp.ones(2)}}
    optimiser = blr_diagonal(0.1, prior_precision=1.0)
    posterior = blr_posterior(optimiser.init(params))
    flat, unravel = ravel_pytree(params)
    np.testing.assert_allclose(posterior.mode, flat, rtol=1e-12)
    restored = unravel(posterior.mode)
    assert jax.tree.structure(restored) == jax.tree.structure(params)
    np.testing.assert_allclose(restored["layer"]["w"], params["layer"]["w"])


def test_runs_under_jit_and_keeps_float32() -> None:
    theta = jnp.ones(3, jnp.float32)
    optimiser = blr_full_rank(0.1)
    state = optimiser.init(theta)
    posterior = jax.jit(blr_posterior)(state)
    assert isinstance(posterior, Laplace)
    assert posterior.mode.dtype == jnp.float32
    np.testing.assert_allclose(posterior.mode, theta, rtol=1e-6)


def test_a_solve_strategy_is_used_and_kept() -> None:
    solver = gx.CGSolver(rtol=1e-12, atol=1e-12)
    state = _full_rank_state()
    posterior = blr_posterior(state, solver=solver)
    assert posterior.solver is solver
    np.testing.assert_allclose(posterior.mode, blr_posterior(state).mode, rtol=1e-8)


def test_ivon_state_needs_params_and_weight_decay() -> None:
    theta = jnp.zeros(2)
    state = ivon(0.1).init(theta)
    with pytest.raises(ValidationError, match="IVONState needs"):
        blr_posterior(state)
    with pytest.raises(ValidationError, match="IVONState needs"):
        blr_posterior(state, theta)


def test_other_states_reject_params_and_unknown_states_are_rejected() -> None:
    theta = jnp.zeros(2)
    state = blr_diagonal(0.1).init(theta)
    with pytest.raises(ValidationError, match="only for an IVONState"):
        blr_posterior(state, theta)
    with pytest.raises(ValidationError, match="'state' must be"):
        blr_posterior((theta, theta))

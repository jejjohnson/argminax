"""Tests for `Reparametrised`, positive and bounded states by a change of variable.

The round trip is exact away from saturation; on a positive 1-D toy, mapped
samples stay positive and their mean matches Gauss–Hermite quadrature of the
posterior mapped through g, within the Monte Carlo error of the draws.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import assert_within_stderr

from argminax import ValidationError
from argminax.curvature import gauss_newton
from argminax.solve import Reparametrised


TRANSFORMS = [
    Reparametrised("exp"),
    Reparametrised("softplus", lower=1.0),
    Reparametrised("sigmoid", lower=-2.0, upper=3.0),
]


@pytest.mark.parametrize("reparam", TRANSFORMS, ids=lambda r: r.transform)
def test_inverse_undoes_forward_away_from_saturation(reparam) -> None:
    u = jnp.linspace(-4.0, 4.0, 41)
    assert jnp.allclose(reparam.inverse(reparam.forward(u)), u, rtol=0, atol=1e-12)


@pytest.mark.parametrize("reparam", TRANSFORMS, ids=lambda r: r.transform)
def test_forward_respects_the_bounds(reparam) -> None:
    x = reparam.forward(jnp.linspace(-30.0, 30.0, 61))
    lower = 0.0 if reparam.lower is None else reparam.lower
    assert jnp.all(x >= lower)
    if reparam.upper is not None:
        assert jnp.all(x <= reparam.upper)


@pytest.mark.parametrize("reparam", TRANSFORMS, ids=lambda r: r.transform)
def test_derivative_is_the_gradient_of_forward(reparam) -> None:
    u = jnp.linspace(-3.0, 3.0, 13)
    autodiff = jax.vmap(jax.grad(lambda v: reparam.forward(v)))(u)
    assert jnp.allclose(reparam.derivative(u), autodiff, rtol=1e-12)


def _positive_toy():
    """One emission rate x = exp(u) > 0, measured once near zero."""
    reparam = Reparametrised("exp")
    measurement, noise, prior_mean, prior_sd = 0.3, 0.2, jnp.log(0.5), 1.0

    def residuals(u, args):
        # r(u) = ((y − exp(u))/σ, (u − μ)/τ)
        return (measurement - reparam.forward(u)) / noise, (u - prior_mean) / prior_sd

    # û from a few Newton steps on the scalar cost
    u = jnp.array([prior_mean])
    cost_gradient = jax.grad(
        lambda v: 0.5 * sum(jnp.sum(r**2) for r in residuals(v, None))
    )
    for _ in range(50):
        hessian = jax.hessian(
            lambda v: 0.5 * sum(jnp.sum(r**2) for r in residuals(v, None))
        )(u)
        u = u - jnp.linalg.solve(hessian, cost_gradient(u))
    return reparam, gauss_newton(residuals, u)


def test_mapped_samples_are_positive_and_match_quadrature() -> None:
    reparam, posterior = _positive_toy()
    draws = reparam.sample(posterior, 20_000, key=jax.random.key(0))
    assert draws.shape == (20_000, 1)
    assert jnp.all(draws > 0.0)
    # E[g(u)] under N(û, Ŝᵤ), by 40-point Gauss–Hermite quadrature
    mode = float(posterior.mode[0])
    spread = float(jnp.sqrt(posterior.covariance().as_matrix()[0, 0]))
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    exact = np.sum(weights * np.exp(mode + spread * nodes)) / np.sqrt(2.0 * np.pi)
    # within the Monte Carlo error of the draws (4 standard errors)
    assert_within_stderr(draws[:, 0], jnp.asarray(exact))


def test_push_forward_is_the_linearisation() -> None:
    reparam, posterior = _positive_toy()
    physical = reparam.push_forward(posterior)
    slope = reparam.derivative(posterior.mode)
    # x̂ = g(û), Ŝₓ = g′(û)² Ŝᵤ
    assert jnp.allclose(physical.mode, reparam.forward(posterior.mode))
    expected = slope[0] ** 2 * posterior.covariance().as_matrix()
    assert jnp.allclose(physical.covariance().as_matrix(), expected, rtol=1e-10)
    assert lx.is_positive_semidefinite(physical.precision)


def test_push_forward_is_poor_near_the_bound() -> None:
    """The reason the docstring prefers mapped samples near the bound."""
    reparam, posterior = _positive_toy()
    physical = reparam.push_forward(posterior)
    draws = reparam.sample(posterior, 20_000, key=jax.random.key(1))
    linear_draws = physical.sample(20_000, key=jax.random.key(1))
    # the linearised Gaussian puts mass below zero; the mapped draws never do
    assert jnp.mean(linear_draws < 0.0) > 0.0
    assert jnp.all(draws > 0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"transform": "tanh"}, "'transform' must be one of"),
        ({"transform": "exp", "upper": 1.0}, "'upper' must be None"),
        ({"transform": "sigmoid", "lower": 0.0}, "'lower' and 'upper' must both"),
        ({"transform": "sigmoid", "lower": 1.0, "upper": 0.0}, "lower < upper"),
    ],
)
def test_settings_are_validated(kwargs, message) -> None:
    with pytest.raises(ValidationError, match=message):
        Reparametrised(**kwargs)

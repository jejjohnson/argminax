"""Tests for `site_fixed_point` (#100).

The loop is compared with the code it replaces, frozen from pyrox (whose
packages sit above argminax, so the fixture carries what pyrox computes; see
``tests/solve/data/README_pyrox_sites.md``): pyrox-ssm's ``_solve_sites`` on a
Markov GP with posterior linearisation, run on pyrox's own step exported with
``jax.export``, and the fixed points of pyrox-gp's eager
``ExpectationPropagation`` and ``PosteriorLinearization`` loops on Bernoulli
and Poisson data. The dense EP and PL updates below are written with gaussx's
released site primitives, as a caller of `site_fixed_point` writes them.
"""

from __future__ import annotations

from pathlib import Path

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest
from _harness import assert_grad_matches_fd

from argminax import ValidationError
from argminax.solve import SiteFixedPoint, site_fixed_point


DATA = np.load(Path(__file__).parent / "data" / "pyrox_sites.npz")
_PSD = (lx.symmetric_tag, lx.positive_semidefinite_tag)
_FLOOR = 1e-6
_ORDER = 20


def _relative(actual, expected) -> float:
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


# --- the frozen Markov case: pyrox-ssm's own step, exported ---


def test_matches_markov_fixture() -> None:
    exported = jax.export.deserialize(bytearray(DATA["markov_step"].tobytes()))

    def step(sites, _):
        return tuple(exported.call(sites))

    start = (
        jnp.asarray(DATA["markov_start_nat1"]),
        jnp.asarray(DATA["markov_start_nat2"]),
    )
    fit = site_fixed_point(
        step,
        start,
        max_iter=int(DATA["markov_max_steps"]),
        tol=float(DATA["markov_tol"]),
    )
    assert isinstance(fit, SiteFixedPoint)
    # bit for bit: the same loop on the same step
    assert np.array_equal(fit.sites[0], DATA["markov_nat1"])
    assert np.array_equal(fit.sites[1], DATA["markov_nat2"])
    assert int(fit.n_iter) == int(DATA["markov_n_steps"])
    assert bool(fit.converged) == bool(DATA["markov_converged"])
    # the exported step is pyrox's step: its native run gave the same sites
    assert np.array_equal(DATA["markov_native_nat1"], DATA["markov_nat1"])
    # the masked site holds (0, 0) throughout and the loop still converged
    masked = ~DATA["markov_observed"]
    assert masked.any()
    assert np.all(np.asarray(fit.sites[0])[masked] == 0.0)
    assert np.all(np.asarray(fit.sites[1])[masked] == 0.0)


# --- dense EP and PL updates on gaussx's released primitives ---


def _log_density(name: str):
    if name == "bernoulli":

        def log_density(latent, observation):
            # log p(y ∣ f) = y f − log(1 + exp f), pyrox-gp's logit link
            return observation * latent - jax.nn.softplus(latent)

    else:

        def log_density(latent, observation):
            # log p(y ∣ f) = y f − exp f − log y!, pyrox-gp's log link
            normaliser = jax.scipy.special.gammaln(observation + 1.0)
            return observation * latent - jnp.exp(latent) - normaliser

    return log_density


def _posterior(kernel, prior_mean, information, precision):
    """Marginals of q(f) = 𝒩(m, V) under diagonal sites, dense (test oracle)."""
    # K + Λ_site⁻¹
    regularised = kernel + jnp.diag(1.0 / precision)
    # m = μ + K (K + Λ_site⁻¹)⁻¹ (Λμ_site / Λ_site − μ)
    weights = jnp.linalg.solve(regularised, information / precision - prior_mean)
    mean = prior_mean + kernel @ weights
    # diag(V) = diag(K) − diag(K (K + Λ_site⁻¹)⁻¹ K)
    solved = jnp.linalg.solve(regularised, kernel)
    reduction = einx.dot("row col, row col -> col", kernel, solved)
    return mean, jnp.diag(kernel) - reduction


def _expectation(fn, mean, variance):
    """The mean of fn(f) under f ~ 𝒩(mean, variance), Gauss–Hermite of order 20."""
    covariance = lx.MatrixLinearOperator(
        variance[None, None], lx.positive_semidefinite_tag
    )
    state = gx.GaussianState(mean=mean[None], cov=covariance)
    integrator = gx.GaussHermiteIntegrator(order=_ORDER)
    expectation = gx.mean_expectation(
        lambda latent: jnp.atleast_1d(fn(latent[0])), state, integrator
    )
    return expectation[0]


def _make_update(method: str, likelihood: str, damping: float = 0.5):
    log_density = _log_density(likelihood)

    def update(sites, args):
        kernel, prior_mean, observations = args
        information, precision = sites
        mean, variance = _posterior(kernel, prior_mean, information, precision)
        # q₋ₙ(fₙ) = q(fₙ) / siteₙ(fₙ)
        cavity_mean, cavity_variance = gx.cavity_distribution(
            mean, variance, information, precision, precision_floor=_FLOOR
        )
        if method == "ep":

            def tilted(m, v, y):
                return gx.ep_tilted_moments(
                    lambda latent: log_density(latent, y), m, v, order=_ORDER
                )

            tilted_mean, tilted_variance = jax.vmap(tilted)(
                cavity_mean, cavity_variance, observations
            )
            # Λ_site = 1/v_tilted − 1/v_cavity, floored
            target_precision = jnp.maximum(
                1.0 / tilted_variance - 1.0 / cavity_variance, _FLOOR
            )
            # Λμ_site = m_tilted / v_tilted − m_cavity / v_cavity
            target_information = (
                tilted_mean / tilted_variance - cavity_mean / cavity_variance
            )
        else:

            def moments(m, v, y):
                gradient = jax.grad(log_density)
                curvature = jax.grad(gradient)
                # ḡ = E_cav[∂f log p], h̄ = E_cav[∂²f log p]
                mean_gradient = _expectation(lambda f: gradient(f, y), m, v)
                mean_curvature = _expectation(lambda f: curvature(f, y), m, v)
                return mean_gradient, mean_curvature

            mean_gradient, mean_curvature = jax.vmap(moments)(
                cavity_mean, cavity_variance, observations
            )
            target_information, target_precision = gx.newton_update(
                cavity_mean, mean_gradient, mean_curvature, precision_floor=_FLOOR
            )
        information, precision = gx.damped_natural_update(
            information, precision, target_information, target_precision, lr=damping
        )
        return information, jnp.maximum(precision, _FLOOR)

    return update


def _start(size: int, dtype=jnp.float64):
    # pyrox-gp's start: zero information, the floor as precision
    return jnp.zeros(size, dtype), jnp.full((size,), _FLOOR, dtype)


@pytest.mark.parametrize("likelihood", ["bernoulli", "poisson"])
def test_log_densities_match_pyrox(likelihood: str) -> None:
    log_density = _log_density(likelihood)
    ours = log_density(
        jnp.asarray(DATA[f"{likelihood}_probe_f"]), jnp.asarray(DATA[f"{likelihood}_y"])
    )
    assert jnp.allclose(ours, DATA[f"{likelihood}_probe_log_prob"], rtol=1e-14)


@pytest.mark.parametrize("likelihood", ["bernoulli", "poisson"])
@pytest.mark.parametrize("method", ["ep", "pl"])
def test_ep_and_pl_fixed_points(method: str, likelihood: str) -> None:
    kernel = jnp.asarray(DATA["dense_K"])
    args = (
        kernel,
        jnp.asarray(DATA["dense_mean"]),
        jnp.asarray(DATA[f"{likelihood}_y"]),
    )
    fit = site_fixed_point(
        _make_update(method, likelihood),
        _start(kernel.shape[0]),
        args,
        max_iter=500,
        tol=1e-10,
    )
    assert bool(fit.converged)
    assert bool(DATA[f"{method}_{likelihood}_converged"])
    tag = f"{method}_{likelihood}"
    # pyrox stops when the mean moves by under 1e-12, so both are at the fixed
    # point to well below the issue's 1e-6
    assert _relative(fit.sites[0], DATA[f"{tag}_nat1"]) <= 1e-6
    assert _relative(fit.sites[1], DATA[f"{tag}_nat2"]) <= 1e-6
    mean, _ = _posterior(kernel, args[1], *fit.sites)
    assert _relative(mean, DATA[f"{tag}_q_mean"]) <= 1e-6


def test_ep_fixed_point_in_float32() -> None:
    kernel = jnp.asarray(DATA["dense_K"], jnp.float32)
    args = (
        kernel,
        jnp.asarray(DATA["dense_mean"], jnp.float32),
        jnp.asarray(DATA["bernoulli_y"], jnp.float32),
    )
    fit = site_fixed_point(
        _make_update("ep", "bernoulli"), _start(12, jnp.float32), args, max_iter=200
    )
    assert fit.sites[0].dtype == jnp.float32
    assert bool(fit.converged)
    # the default tol of 1e-6, in float32 rounding
    assert _relative(fit.sites[0], DATA["ep_bernoulli_nat1"]) <= 1e-4
    assert _relative(fit.sites[1], DATA["ep_bernoulli_nat2"]) <= 1e-4


# --- transforms and gradients ---


def _evidence_like(scale, damping: float = 0.5, method: str = "ep"):
    """A scalar of the fixed point that depends on the kernel scale."""
    kernel = scale[0] * jnp.asarray(DATA["dense_K"])
    prior_mean = jnp.asarray(DATA["dense_mean"])
    args = (kernel, prior_mean, jnp.asarray(DATA["bernoulli_y"]))
    fit = site_fixed_point(
        _make_update(method, "bernoulli", damping),
        _start(kernel.shape[0]),
        args,
        max_iter=500,
        tol=1e-12,
    )
    mean, variance = _posterior(kernel, prior_mean, *fit.sites)
    return jnp.sum(jnp.sin(mean)) + jnp.sum(jnp.log(variance))


@pytest.mark.parametrize("method", ["ep", "pl"])
def test_gradient_matches_finite_differences(method: str) -> None:
    assert_grad_matches_fd(
        lambda scale: _evidence_like(scale, method=method), jnp.array([1.3]), rtol=1e-5
    )


def test_gradient_independent_of_damping() -> None:
    scale = jnp.array([1.3])
    slow = jax.grad(lambda scale: _evidence_like(scale, damping=0.3))(scale)
    fast = jax.grad(lambda scale: _evidence_like(scale, damping=0.8))(scale)
    assert _relative(slow, fast) <= 1e-6


def test_runs_under_jit_and_vmap_over_datasets() -> None:
    kernel = jnp.asarray(DATA["dense_K"])
    prior_mean = jnp.asarray(DATA["dense_mean"])
    labels = jnp.asarray(DATA["bernoulli_y"])
    datasets = jnp.stack([labels, 1.0 - labels])
    update = _make_update("ep", "bernoulli")

    def fit(observations):
        result = site_fixed_point(
            update, _start(12), (kernel, prior_mean, observations)
        )
        return result.sites, result.converged

    batched_sites, batched_converged = jax.jit(jax.vmap(fit))(datasets)
    assert bool(jnp.all(batched_converged))
    for index, observations in enumerate(datasets):
        sites, _ = fit(observations)
        # a batched loop runs until its last member stops, and the extra
        # iterations of the others move their sites by under tol / 10
        assert _relative(batched_sites[0][index], sites[0]) <= 1e-6
        assert _relative(batched_sites[1][index], sites[1]) <= 1e-6


@pytest.mark.parametrize(
    "update",
    [
        # s ← s² + θ + 1 overflows to inf within a few iterations
        lambda sites, theta: sites**2 + theta + 1.0,
        # s ← log(s − 10) + θ turns NaN at the first step
        lambda sites, theta: jnp.log(sites - 10.0) + theta,
    ],
    ids=["overflow", "nan"],
)
def test_divergence_is_reported_not_raised(update) -> None:
    def total(theta):
        fit = site_fixed_point(update, jnp.zeros(3), theta, max_iter=20)
        return jnp.sum(fit.sites), fit

    value, fit = total(jnp.ones(3))
    assert not bool(fit.converged)
    # the value a sampler sees is non-finite, so it rejects the point
    assert not bool(jnp.isfinite(value))
    # and the backward pass raises nothing: the adjoint solves for v = 0
    gradient = jax.jit(jax.grad(lambda theta: total(theta)[0]))(jnp.ones(3))
    assert gradient.shape == (3,)


def test_recursive_checkpoint_adjoint_gives_the_same_gradient() -> None:
    def update(sites, theta):
        return 0.5 * sites + 0.5 * (0.3 * sites + theta)

    def total(theta, adjoint):
        fit = site_fixed_point(update, jnp.zeros(2), theta, tol=1e-12, adjoint=adjoint)
        return jnp.sum(fit.sites**2)

    theta = jnp.array([1.0, -2.0])
    implicit = jax.grad(total)(theta, None)
    unrolled = jax.grad(total)(theta, optx.RecursiveCheckpointAdjoint())
    assert jnp.allclose(implicit, unrolled, rtol=1e-8)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "'max_iter' must be a positive integer"),
        ({"max_iter": True}, "'max_iter' must be a positive integer"),
        ({"tol": 0.0}, "'tol' must be positive"),
        (
            {"adjoint": lx.GMRES(rtol=1e-8, atol=1e-8)},
            "'adjoint' must be an optimistix",
        ),
    ],
)
def test_settings_are_validated(kwargs, message) -> None:
    with pytest.raises(ValidationError, match=message):
        site_fixed_point(lambda sites, _: sites, jnp.zeros(2), **kwargs)

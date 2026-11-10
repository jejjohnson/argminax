"""Tests for `laplace_mode` and its covariance and site forms (#25).

The precision form is compared with gaussx's ``laplace_mode``, which it replaces
(gaussx#814), on gaussx's own problem types; the covariance and site forms with
pyrox-gp's ``LaplaceInference``, ``GaussNewtonInference`` and
``LaplaceMarkovInference`` on pyrox-gp's own test problems (fixtures in
``tests/solve/data/``, see ``README_pyrox_gp.md``), in float64 and float32.
"""

from __future__ import annotations

from pathlib import Path

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import assert_grad_matches_fd

from argminax import MissingExtraError, ValidationError
from argminax.curvature import Laplace
from argminax.solve import (
    laplace_mode,
    laplace_mode_covariance,
    laplace_mode_sites,
)


DATA = Path(__file__).parent / "data"
PYROX = {64: np.load(DATA / "pyrox_gp_64.npz"), 32: np.load(DATA / "pyrox_gp_32.npz")}
_PSD = (lx.symmetric_tag, lx.positive_semidefinite_tag)


def _relative(actual, expected) -> float:
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


# --- gaussx's problem types, for the precision form ---


def _counts(n: int, dtype=jnp.float64) -> gx.PoissonLikelihood:
    rates = np.exp(np.sin(np.arange(n) / 3.0))
    return gx.PoissonLikelihood(
        jnp.asarray(np.random.default_rng(0).poisson(rates), dtype)
    )


def _gaussian_prior(n: int, scale: float = 1.0, dtype=jnp.float64) -> gx.GaussianMRF:
    precision = 2.0 * np.eye(n) - 0.9 * (np.eye(n, k=1) + np.eye(n, k=-1))
    operator = lx.MatrixLinearOperator(
        jnp.asarray(scale * precision, dtype), lx.positive_semidefinite_tag
    )
    return gx.GaussianMRF(jnp.zeros(n, dtype), operator)


def _intrinsic_prior(n: int, tau=10.0) -> gx.IntrinsicGMRF:
    times = jnp.arange(n, dtype=jnp.float64)
    null_space = jnp.column_stack([jnp.ones(n), times - times.mean()])
    return gx.IntrinsicGMRF(jnp.zeros(n), tau, gx.rw2_structure(n), null_space)


PRECISION_CASES = {
    "gaussian": (lambda: _gaussian_prior(20), lambda: _counts(20), {}),
    "intrinsic-hard": (lambda: _intrinsic_prior(20), lambda: _counts(20), {}),
    "odd-n-selection": (
        lambda: gx.IntrinsicGMRF(
            jnp.zeros(22),
            10.0,
            gx.rw2_structure(21),
            jnp.zeros((22, 2))
            .at[:21, 0]
            .set(1.0)
            .at[:21, 1]
            .set(jnp.arange(21.0) - 10.0),
        ),
        lambda: _counts(21),
        {
            "projector": gx.SparseOperator.from_coo(
                np.arange(21), np.arange(21), jnp.ones(21), (21, 22)
            )
        },
    ),
}


@pytest.mark.parametrize("case", sorted(PRECISION_CASES))
def test_port_reproduces_gaussx_laplace_mode(case: str) -> None:
    make_prior, make_likelihood, kwargs = PRECISION_CASES[case]
    prior, likelihood = make_prior(), make_likelihood()
    ours = laplace_mode(prior, likelihood, **kwargs)
    theirs = gx.laplace_mode(prior, likelihood, **kwargs)
    assert bool(ours.converged)
    assert int(ours.n_iter) == int(theirs.n_iter)
    assert _relative(ours.mode, theirs.mode) <= 1e-10
    assert abs(float(ours.log_marginal - theirs.log_marginal)) <= 1e-10 * abs(
        float(theirs.log_marginal)
    )
    assert type(ours.hessian) is type(theirs.hessian)


def test_port_reproduces_gaussx_in_float32() -> None:
    prior = _gaussian_prior(12, dtype=jnp.float32)
    likelihood = _counts(12, dtype=jnp.float32)
    ours = laplace_mode(prior, likelihood)
    theirs = gx.laplace_mode(prior, likelihood)
    assert ours.mode.dtype == jnp.float32
    assert _relative(ours.mode, theirs.mode) <= 1e-5
    assert abs(float(ours.log_marginal - theirs.log_marginal)) <= 1e-4 * abs(
        float(theirs.log_marginal)
    )


def test_mode_gradient_matches_finite_differences() -> None:
    likelihood = _counts(10)

    def summary(scale):
        fit = laplace_mode(_gaussian_prior(10, scale=scale[0]), likelihood)
        return jnp.sum(jnp.sin(fit.mode))

    assert_grad_matches_fd(summary, jnp.array([1.3]))


def test_log_marginal_gradient_matches_gaussx() -> None:
    likelihood = _counts(20)

    def ours(log_tau):
        return laplace_mode(
            _intrinsic_prior(20, jnp.exp(log_tau)), likelihood
        ).log_marginal

    def theirs(log_tau):
        return gx.laplace_mode(
            _intrinsic_prior(20, jnp.exp(log_tau)), likelihood
        ).log_marginal

    assert jax.grad(ours)(2.0) == pytest.approx(float(jax.grad(theirs)(2.0)), rel=1e-9)


def test_a_vjp_does_one_solve() -> None:
    likelihood = _counts(8)

    def summary(scale):
        return jnp.sum(laplace_mode(_gaussian_prior(8, scale=scale), likelihood).mode)

    jaxpr = str(jax.make_jaxpr(jax.grad(summary))(1.0))
    # the implicit tangent is one custom_linear_solve with the factor at x̂
    assert jaxpr.count("custom_linear_solve") == 1


@pytest.mark.slow
def test_runs_under_jit_and_vmap_over_datasets() -> None:
    prior = _gaussian_prior(6)
    datasets = jnp.asarray(np.random.default_rng(1).poisson(2.0, (4, 6)), jnp.float64)

    def fit_one(counts):
        return laplace_mode(prior, gx.PoissonLikelihood(counts)).mode

    batched = jax.jit(jax.vmap(fit_one))(datasets)
    looped = jnp.stack([fit_one(counts) for counts in datasets])
    assert jnp.allclose(batched, looped, rtol=1e-10)


def test_result_works_with_gaussx_vb_mean_correction() -> None:
    prior, likelihood = _gaussian_prior(10), _counts(10)
    ours = laplace_mode(prior, likelihood)
    theirs = gx.laplace_mode(prior, likelihood)
    subspace = jnp.arange(3)
    corrected = gx.vb_mean_correction(ours, prior, likelihood, subspace=subspace)
    expected = gx.vb_mean_correction(theirs, prior, likelihood, subspace=subspace)
    assert jnp.allclose(corrected, expected, rtol=1e-10)


def test_posterior_is_the_laplace_at_the_mode() -> None:
    prior, likelihood = _gaussian_prior(6), _counts(6)
    fit = laplace_mode(prior, likelihood)
    posterior = fit.posterior
    assert isinstance(posterior, Laplace)
    covariance = posterior.covariance().as_matrix()
    assert jnp.allclose(covariance, jnp.linalg.inv(fit.hessian.as_matrix()), rtol=1e-10)


def test_precision_form_needs_numpyro(monkeypatch) -> None:
    monkeypatch.delattr(gx, "laplace_model")
    with pytest.raises(MissingExtraError, match=r"argminax\[numpyro\]"):
        laplace_mode(_gaussian_prior(3), _counts(3))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "'max_iter' must be a positive integer"),
        ({"tol": 0.0}, "'tol' must be positive"),
        ({"damping": 1.5}, r"'damping' must lie in \(0, 1\]"),
    ],
)
def test_settings_are_validated(kwargs, message) -> None:
    with pytest.raises(ValidationError, match=message):
        laplace_mode(_gaussian_prior(3), _counts(3), **kwargs)


# --- pyrox-gp's problems, for the covariance and site forms ---


def _bernoulli(labels):
    def log_likelihood(latent):
        # log p(y ∣ f) = y f − log(1 + exp f), pyrox-gp's logit link
        return labels * latent - jax.nn.softplus(latent)

    return log_likelihood


def test_bernoulli_matches_pyrox() -> None:
    data = PYROX[64]
    log_prob = _bernoulli(jnp.asarray(data["dense_y"]))(
        jnp.asarray(data["bernoulli_probe_f"])
    )
    assert jnp.allclose(log_prob, data["bernoulli_probe_log_prob"], rtol=1e-14)


@pytest.mark.parametrize(
    ("strategy", "floor"), [("laplace", 1e-6), ("gauss_newton", 1e-3)]
)
def test_port_reproduces_pyrox_dense_inference(strategy: str, floor: float) -> None:
    data = PYROX[64]
    prior = lx.MatrixLinearOperator(jnp.asarray(data["dense_K"]), _PSD)
    fit = laplace_mode_covariance(
        prior,
        jnp.asarray(data["dense_mean"]),
        _bernoulli(jnp.asarray(data["dense_y"])),
        max_iter=50,
        tol=1e-8,
        precision_floor=floor,
    )
    assert bool(fit.converged)
    # pyrox starts at f = μ = 0, where numpyro's stable Bernoulli log-density
    # has a kink and JAX returns ∂/∂f = 0 or 1 instead of y − ½; its first
    # step is off, so it needs more steps to the same fixed point
    assert int(fit.n_iter) <= int(data[f"{strategy}_n_iter"])
    for ours, field in [
        (fit.mean, "q_mean"),
        (fit.variance, "q_var"),
        (fit.site_information, "nat1"),
        (fit.site_precision, "nat2"),
    ]:
        assert _relative(ours, data[f"{strategy}_{field}"]) <= 1e-10, field
    assert fit.log_marginal == pytest.approx(
        float(data[f"{strategy}_log_marginal"]), rel=1e-10
    )


def test_port_reproduces_pyrox_dense_inference_in_float32() -> None:
    data = PYROX[32]
    prior = lx.MatrixLinearOperator(jnp.asarray(data["dense_K"]), _PSD)
    fit = laplace_mode_covariance(
        prior,
        jnp.asarray(data["dense_mean"]),
        _bernoulli(jnp.asarray(data["dense_y"])),
        max_iter=50,
        tol=1e-8,
    )
    assert fit.mean.dtype == jnp.float32
    # pyrox runs all 50 steps in float32 (its tol is below float32's reach);
    # argminax stops at 64 ε, so the two agree to float32 accuracy
    assert bool(fit.converged)
    assert _relative(fit.mean, data["laplace_q_mean"]) <= 1e-5
    assert _relative(fit.variance, data["laplace_q_var"]) <= 1e-5


@pytest.mark.parametrize("bits", [64, 32])
def test_port_reproduces_pyrox_markov_inference(bits: int) -> None:
    data = PYROX[bits]
    kernel = jnp.asarray(data["markov_K"])
    mean = jnp.asarray(data["markov_mean"])
    prior = lx.MatrixLinearOperator(kernel, _PSD)

    def posterior_mean(information, precision):
        # the dense form of pyrox's Kalman smoother on the Matern-1/2 SDE
        regularised = lx.TaggedLinearOperator(
            prior + lx.DiagonalLinearOperator(1.0 / precision), _PSD
        )
        return mean + kernel @ gx.solve(regularised, information / precision - mean)

    fit = laplace_mode_sites(
        posterior_mean,
        _bernoulli(jnp.asarray(data["markov_y"])),
        mean,
        max_iter=30,
        tol=1e-7,
    )
    assert bool(fit.converged)
    # Kalman and dense solves of the same model differ by rounding, and the
    # loop stops at max |Δf| < 1e-7
    tolerance = 1e-6 if bits == 64 else 1e-4
    assert _relative(fit.mode, data["markov_q_mean"]) <= tolerance
    assert _relative(fit.site_information, data["markov_nat1"]) <= tolerance
    assert _relative(fit.site_precision, data["markov_nat2"]) <= tolerance


def test_covariance_mode_gradient_matches_finite_differences() -> None:
    data = PYROX[64]
    labels = _bernoulli(jnp.asarray(data["dense_y"]))
    kernel = jnp.asarray(data["dense_K"])

    def summary(scale):
        prior = lx.MatrixLinearOperator(scale[0] * kernel, _PSD)
        fit = laplace_mode_covariance(
            prior, jnp.zeros(kernel.shape[0]), labels, max_iter=50, tol=1e-12
        )
        return jnp.sum(jnp.sin(fit.mode)) + fit.log_marginal

    assert_grad_matches_fd(summary, jnp.array([1.4]))


def test_site_mode_gradient_matches_finite_differences() -> None:
    data = PYROX[64]
    labels = _bernoulli(jnp.asarray(data["markov_y"]))
    kernel = jnp.asarray(data["markov_K"])
    size = kernel.shape[0]

    def summary(scale):
        prior = lx.MatrixLinearOperator(scale[0] * kernel, _PSD)

        def posterior_mean(information, precision):
            regularised = lx.TaggedLinearOperator(
                prior + lx.DiagonalLinearOperator(1.0 / precision), _PSD
            )
            return scale[0] * kernel @ gx.solve(regularised, information / precision)

        fit = laplace_mode_sites(
            posterior_mean, labels, jnp.zeros(size), max_iter=60, tol=1e-12
        )
        return jnp.sum(jnp.sin(fit.mode))

    assert_grad_matches_fd(summary, jnp.array([0.8]))


def test_covariance_posterior_is_the_laplace_at_the_mode() -> None:
    data = PYROX[64]
    kernel = jnp.asarray(data["dense_K"])
    fit = laplace_mode_covariance(
        lx.MatrixLinearOperator(kernel, _PSD),
        jnp.asarray(data["dense_mean"]),
        _bernoulli(jnp.asarray(data["dense_y"])),
        max_iter=50,
        tol=1e-8,
    )
    covariance = fit.posterior.covariance().as_matrix()
    assert jnp.allclose(jnp.diag(covariance), fit.variance, rtol=1e-8)


def test_covariance_form_needs_a_psd_prior() -> None:
    with pytest.raises(ValidationError, match="'prior_cov' must be tagged"):
        laplace_mode_covariance(
            lx.MatrixLinearOperator(jnp.eye(2)), jnp.zeros(2), _bernoulli(jnp.ones(2))
        )

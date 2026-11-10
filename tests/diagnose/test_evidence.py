"""Tests for the Laplace evidence and the prior-precision search.

The evidence equals the dense linear-Gaussian log evidence of the harness to
1e-10. The search reproduces pyrox-nn's optimize_prior_precision on its own
test problem, a Bayesian linear regression on frozen random features fitted at
a stale λ = 50; the λ̂ values are pyrox-nn's at 3d16c53 (pyrox origin/main,
9 November 2026), and the problem below is copied from its test.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest
from _harness import linear_gaussian_problem, whitened_residuals
from jax.scipy.linalg import solve_triangular

from argminax import ValidationError
from argminax.curvature import Laplace, gauss_newton
from argminax.diagnose import evidence, optimize_prior_precision


def _relative(actual, expected) -> float:
    return float(jnp.abs(actual - expected) / jnp.abs(expected))


@pytest.fixture
def problem():
    return linear_gaussian_problem(jax.random.key(0), 5, 7)


def _cost(residuals, state) -> jax.Array:
    # J(x) = ½ ‖r(x)‖²
    return 0.5 * sum(
        jnp.sum(part**2) for part in jax.tree_util.tree_leaves(residuals(state, None))
    )


def _obs_logdet(problem) -> jax.Array:
    # log |2π Sε|
    return jnp.linalg.slogdet(2.0 * math.pi * problem.S_eps)[1]


def test_evidence_is_the_dense_log_evidence_in_physical_coordinates(problem) -> None:
    residuals = whitened_residuals(problem)
    posterior = gauss_newton(residuals, problem.x_hat)
    log_evidence = evidence(
        posterior,
        _cost(residuals, problem.x_hat),
        _obs_logdet(problem),
        prior_cov_logdet=jnp.linalg.slogdet(problem.S_a)[1],
    )
    assert _relative(log_evidence.value, problem.log_evidence) <= 1e-10
    assert log_evidence.stderr == 0.0


def test_evidence_is_the_dense_log_evidence_in_whitened_coordinates(problem) -> None:
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    prior_factor = jnp.linalg.cholesky(problem.S_a)

    def residuals(control, args):
        state = problem.x_a + prior_factor @ control
        return (
            solve_triangular(noise_factor, problem.y - problem.K @ state, lower=True),
            control,
        )

    mode = solve_triangular(prior_factor, problem.x_hat - problem.x_a, lower=True)
    posterior = gauss_newton(residuals, mode)
    log_evidence = evidence(posterior, _cost(residuals, mode), _obs_logdet(problem))
    assert _relative(log_evidence.value, problem.log_evidence) <= 1e-10
    # the stochastic log-determinant lands within four standard errors
    probed = evidence(
        posterior,
        _cost(residuals, mode),
        _obs_logdet(problem),
        num_probes=200,
        key=jax.random.key(1),
    )
    assert probed.stderr > 0.0
    assert abs(probed.value - problem.log_evidence) <= 4.0 * probed.stderr


@pytest.mark.slow
def test_evidence_traces_under_jit_and_vmap() -> None:
    keys = jax.random.split(jax.random.key(2), 3)
    problems = jax.vmap(lambda key: linear_gaussian_problem(key, 3, 4))(keys)

    def log_evidence(problem):
        residuals = whitened_residuals(problem)
        posterior = gauss_newton(residuals, problem.x_hat)
        return evidence(
            posterior,
            _cost(residuals, problem.x_hat),
            _obs_logdet(problem),
            prior_cov_logdet=jnp.linalg.slogdet(problem.S_a)[1],
        ).value

    values = jax.jit(jax.vmap(log_evidence))(problems)
    assert jnp.allclose(values, problems.log_evidence, rtol=1e-10)


# --- pyrox-nn's stale Bayesian linear regression ---

SIGMA, N, D, H = 0.3, 40, 2, 8
PYROX_LAMBDA = {"full": 0.9381113264678532, "diag": 24.135627596775784}


def _stale_blr():
    """Design Φ, targets y, the stale fit w⋆ at λ = 50, and the GGN pieces."""
    k1, k2, k3, _ = jr.split(jr.key(0), 4)
    x = jr.normal(k1, (N, D))
    features = jr.normal(k2, (D, H))
    design = jnp.concatenate([jnp.tanh(x @ features), jnp.ones((N, 1))], axis=1)
    targets = jnp.sin(x[:, 0]) + SIGMA * jr.normal(k3, (N,))
    precision = design.T @ design / SIGMA**2 + 50.0 * jnp.eye(H + 1)
    stale = jnp.linalg.solve(precision, design.T @ targets / SIGMA**2)
    # G = σ⁻² Φᵀ Φ and b = σ⁻² Φᵀ (y − Φ w⋆)
    ggn = design.T @ design / SIGMA**2
    gradient = design.T @ (targets - design @ stale) / SIGMA**2
    return design, targets, stale, ggn, gradient


def _blr_log_evidence(design, targets, lam):
    covariance = SIGMA**2 * jnp.eye(N) + design @ design.T / lam
    return jax.scipy.stats.multivariate_normal.logpdf(targets, jnp.zeros(N), covariance)


def _fit_term(eigenvalues, rotated_mode, rotated_gradient):
    # f(λ) = βᵀ(ω − d) − ½ ∑ᵢ gᵢ (ωᵢ − dᵢ)² − (λ/2) ‖ω‖²,  ω = (g ⊙ d + β)/(g + λ)
    def fit(lam):
        omega = (eigenvalues * rotated_mode + rotated_gradient) / (eigenvalues + lam)
        delta = omega - rotated_mode
        return (
            jnp.dot(rotated_gradient, delta)
            - 0.5 * jnp.sum(eigenvalues * delta**2)
            - 0.5 * lam * jnp.sum(omega**2)
        )

    return fit


def _assert_same_optimum(eigenvalues, fit, best, reference) -> None:
    """λ̂ is pyrox's maximiser: same evidence, and λ to the precision a maximum has.

    Near a smooth maximum E(log λ) ≈ E* − ½ |E''| δ², so evidence values that
    agree to ε_mach |E*| locate log λ only to √(2 ε_mach |E*| / |E''|), about
    4e-8 here; the eigendecomposition rounds differently across platforms, so
    the maximiser moves by that much (2.8e-8 relative on CI). The evidence is
    what the search maximises, so it must agree to round-off.
    """
    size = eigenvalues.shape[0]

    def log_evidence(lam):
        # f(λ) + (N/2) log λ − ½ ∑ᵢ log(gᵢ + λ)
        return (
            fit(lam)
            + 0.5 * size * jnp.log(lam)
            - 0.5 * jnp.sum(jnp.log(eigenvalues + lam))
        )

    ours, theirs = log_evidence(best), log_evidence(jnp.asarray(reference))
    assert ours >= theirs - 1e-12 * abs(theirs)
    assert abs(jnp.log(best) - math.log(reference)) <= 1e-6


def test_port_reproduces_pyrox_with_the_full_curvature() -> None:
    design, targets, stale, ggn, gradient = _stale_blr()
    eigenvalues, rotation = jnp.linalg.eigh(ggn)
    eigenvalues = jnp.clip(eigenvalues, 0.0)
    fit = _fit_term(eigenvalues, rotation.T @ stale, rotation.T @ gradient)
    best = optimize_prior_precision(eigenvalues, fit)
    _assert_same_optimum(eigenvalues, fit, best, PYROX_LAMBDA["full"])
    # and it maximises the exact BLR evidence: no worse than a fine grid
    grid = jnp.linspace(-6.0, 6.0, 4001)
    values = jax.vmap(lambda g: _blr_log_evidence(design, targets, jnp.exp(g)))(grid)
    assert _blr_log_evidence(design, targets, best) >= jnp.max(values) - 1e-10


def test_port_reproduces_pyrox_with_the_diagonal_curvature() -> None:
    _, _, stale, ggn, gradient = _stale_blr()
    fit = _fit_term(jnp.diag(ggn), stale, gradient)
    best = optimize_prior_precision(jnp.diag(ggn), fit)
    _assert_same_optimum(jnp.diag(ggn), fit, best, PYROX_LAMBDA["diag"])


def test_search_traces_under_jit_and_vmap() -> None:
    def search(scale):
        eigenvalues = scale * jnp.array([1.0, 2.0, 4.0])
        # f(λ) = −λ ‖w‖² / 2 with ‖w‖² = 3: the optimum is interior
        return optimize_prior_precision(eigenvalues, lambda lam: -1.5 * lam)

    scales = jnp.array([0.5, 1.0, 2.0])
    batched = jax.jit(jax.vmap(search))(scales)
    looped = jnp.stack([search(scale) for scale in scales])
    assert jnp.allclose(batched, looped, rtol=1e-12)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [("num_grid", 2, "'num_grid' must be"), ("num_refine", -1, "'num_refine' must be")],
)
def test_search_settings_are_validated(keyword, value, message) -> None:
    with pytest.raises(ValidationError, match=message):
        optimize_prior_precision(jnp.ones(2), lambda lam: 0.0 * lam, **{keyword: value})


def test_evidence_needs_a_vector_mode() -> None:
    identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2, 3), jnp.float64))
    posterior = Laplace(
        jnp.zeros((2, 3)),
        lx.TaggedLinearOperator(identity, lx.positive_semidefinite_tag),
    )
    with pytest.raises(ValidationError, match="one-dimensional mode"):
        evidence(posterior, 0.0, 0.0)

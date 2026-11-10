"""Tests for the averaging kernel, the degrees of freedom and the information content.

Gates (#10): G2-closed-form, A, dₛ and H match the dense formulas of the
linear-Gaussian problem to 1e-10; G2-hutchinson, the probed dₛ lies within four
reported standard errors of the exact one, and the reported standard error
matches the spread over 200 keys to 10 %.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import linear_gaussian_problem, whitened_residuals
from jax.scipy.linalg import solve_triangular

from argminax import ValidationError
from argminax.curvature import Laplace, gauss_newton
from argminax.diagnose import (
    TraceEstimate,
    averaging_kernel,
    degrees_of_freedom,
    information_content,
)


RADTRAX = np.load(Path(__file__).parent / "data" / "radtrax_dfs.npz")


def _relative(actual, expected) -> float:
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


def _psd(matrix):
    return lx.MatrixLinearOperator(
        matrix, (lx.symmetric_tag, lx.positive_semidefinite_tag)
    )


@pytest.fixture
def problem():
    return linear_gaussian_problem(jax.random.key(0), 6, 4)


def _whitened_posterior(problem) -> Laplace:
    """The posterior over ξ, with x = xₐ + Lₐ ξ and prior N(0, I)."""
    noise_factor = jnp.linalg.cholesky(problem.S_eps)
    prior_factor = jnp.linalg.cholesky(problem.S_a)

    def residuals(control, args):
        state = problem.x_a + prior_factor @ control
        return (
            solve_triangular(noise_factor, problem.y - problem.K @ state, lower=True),
            control,
        )

    mode = solve_triangular(prior_factor, problem.x_hat - problem.x_a, lower=True)
    return gauss_newton(residuals, mode)


# --- gate G2-closed-form ---


def test_gate_physical_kernel_dofs_and_information(problem) -> None:
    posterior = gauss_newton(whitened_residuals(problem), problem.x_hat)
    prior_cov = _psd(problem.S_a)
    kernel = averaging_kernel(posterior, prior_cov)
    assert _relative(kernel.as_matrix(), problem.A) <= 1e-10
    dofs = degrees_of_freedom(kernel)
    assert _relative(dofs.value, problem.d_s) <= 1e-10
    assert dofs.stderr == 0.0
    content = information_content(posterior, prior_cov)
    assert _relative(content.value, problem.H_info) <= 1e-10
    assert content.stderr == 0.0


def test_gate_whitened_kernel_dofs_and_information(problem) -> None:
    posterior = _whitened_posterior(problem)
    kernel = averaging_kernel(posterior)
    matrix = kernel.as_matrix()
    # A_ξ = I − Ĥ_ξ⁻¹: symmetric, eigenvalues σᵢ²/(1 + σᵢ²) in [0, 1)
    assert lx.is_symmetric(kernel)
    assert jnp.allclose(matrix, matrix.T, atol=1e-14)
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    assert jnp.all(eigenvalues > -1e-12)
    assert jnp.all(eigenvalues < 1.0)
    assert _relative(degrees_of_freedom(kernel).value, problem.d_s) <= 1e-10
    assert _relative(information_content(posterior).value, problem.H_info) <= 1e-10


# --- gate G2-hutchinson ---


def test_gate_hutchinson_is_within_four_standard_errors(problem) -> None:
    kernel = averaging_kernel(_whitened_posterior(problem))
    keys = jax.random.split(jax.random.key(1), 200)
    estimates = jax.vmap(lambda key: degrees_of_freedom(kernel, key=key))(keys)
    assert jnp.all(estimates.stderr > 0.0)
    assert jnp.all(jnp.abs(estimates.value - problem.d_s) <= 4.0 * estimates.stderr)
    # the reported standard error is the spread of the estimates over keys
    spread = jnp.std(estimates.value, ddof=1)
    reported = jnp.mean(estimates.stderr)
    assert abs(reported / spread - 1.0) <= 0.10


def test_probed_information_content_is_within_four_standard_errors(problem) -> None:
    posterior = _whitened_posterior(problem)
    probed = information_content(posterior, num_probes=200, key=jax.random.key(2))
    assert probed.stderr > 0.0
    assert abs(probed.value - problem.H_info) <= 4.0 * probed.stderr


# --- port: radtrax's degrees_of_freedom_for_signal ---


def test_port_reproduces_radtrax_dofs() -> None:
    size = RADTRAX["hess_xi"].shape[0]
    whitened = Laplace(jnp.zeros(size), _psd(jnp.asarray(RADTRAX["hess_xi"])))
    model_space = Laplace(jnp.zeros(size), _psd(jnp.asarray(RADTRAX["hess_x"])))
    background = _psd(jnp.diag(jnp.asarray(RADTRAX["variances"])))
    kernels = {
        "xi": averaging_kernel(whitened),
        "x": averaging_kernel(model_space, background),
    }
    for space, kernel in kernels.items():
        exact = degrees_of_freedom(kernel)
        assert _relative(exact.value, RADTRAX["exact"]) <= 1e-10
        for num_probes in (16, 64):
            probed = degrees_of_freedom(
                kernel, num_probes=num_probes, key=jax.random.key(0)
            )
            radtrax = float(RADTRAX[f"dfs_{space}_{num_probes}"])
            # within the estimate's own standard error, which vanishes here
            # because both Hessians are diagonal; 1e-9 covers radtrax's CG
            assert abs(probed.value - radtrax) <= 4.0 * probed.stderr + 1e-9


# --- jit and vmap over pixels ---


@pytest.mark.slow
def test_vmap_over_64_pixels_with_split_keys() -> None:
    keys = jax.random.split(jax.random.key(3), 64)
    problems = jax.vmap(lambda key: linear_gaussian_problem(key, 3, 4))(keys)
    probe_keys = jax.random.split(jax.random.key(4), 64)

    def pixel(problem, key):
        posterior = gauss_newton(whitened_residuals(problem), problem.x_hat)
        kernel = averaging_kernel(posterior, _psd(problem.S_a))
        return degrees_of_freedom(kernel, key=key)

    estimates = jax.jit(jax.vmap(pixel))(problems, probe_keys)
    assert estimates.value.shape == (64,)
    assert jnp.all(jnp.abs(estimates.value - problems.d_s) <= 4.0 * estimates.stderr)


def test_jit_traces_without_host_callbacks(problem) -> None:
    def diagnostics(key):
        posterior = _whitened_posterior(problem)
        kernel = averaging_kernel(posterior)
        return (
            degrees_of_freedom(kernel, key=key),
            information_content(posterior, key=key),
        )

    jaxpr = str(jax.make_jaxpr(diagnostics)(jax.random.key(5)))
    assert "callback" not in jaxpr
    result = jax.jit(diagnostics)(jax.random.key(5))
    assert isinstance(result[0], TraceEstimate)


# --- validation ---


def test_num_probes_must_be_at_least_two(problem) -> None:
    posterior = _whitened_posterior(problem)
    kernel = averaging_kernel(posterior)
    with pytest.raises(ValidationError, match="'num_probes' must be"):
        degrees_of_freedom(kernel, num_probes=1, key=jax.random.key(0))
    with pytest.raises(ValidationError, match="'num_probes' must be"):
        information_content(posterior, num_probes=True, key=jax.random.key(0))


def test_a_gridded_mode_is_rejected() -> None:
    identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2, 3), jnp.float64))
    posterior = Laplace(
        jnp.zeros((2, 3)),
        lx.TaggedLinearOperator(identity, lx.positive_semidefinite_tag),
    )
    with pytest.raises(ValidationError, match="one-dimensional mode"):
        averaging_kernel(posterior)
    with pytest.raises(ValidationError, match="one-dimensional mode"):
        information_content(posterior)

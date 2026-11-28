"""Tests for `laplace_mode` and its covariance and site forms (#25).

The precision form is compared with the results of gaussx's ``laplace_mode``, which
it replaces (gaussx#814) and which gaussx 0.8.0 removed, frozen from v0.6.13 on
gaussx's own problem types (``tests/_gaussx_problems.py``, ``README_gaussx_laplace.md``
in ``tests/solve/data/``); the covariance and site forms with
pyrox-gp's ``LaplaceInference``, ``GaussNewtonInference`` and
``LaplaceMarkovInference`` on pyrox-gp's own test problems (fixtures in
``tests/solve/data/``, see ``README_pyrox_gp.md``), in float64 and float32.
"""

from __future__ import annotations

from pathlib import Path

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest
from _gaussx_problems import (
    FLOAT32_SIZE,
    LOG_TAU,
    PRECISION_CASES,
    VB_SIZE,
    VB_SUBSPACE,
    counts,
    gaussian_prior,
    intrinsic_prior,
)
from _harness import assert_grad_matches_fd

from argminax import ValidationError
from argminax._src.solve._laplace_mode import _default_adjoint
from argminax.adjoint import safe_implicit_adjoint
from argminax.curvature import Laplace
from argminax.solve import (
    laplace_mode,
    laplace_mode_covariance,
    laplace_mode_sites,
)


DATA = Path(__file__).parent / "data"
GAUSSX = np.load(DATA / "gaussx_laplace_mode.npz")
PYROX = {64: np.load(DATA / "pyrox_gp_64.npz"), 32: np.load(DATA / "pyrox_gp_32.npz")}
_PSD = (lx.symmetric_tag, lx.positive_semidefinite_tag)


def _relative(actual, expected) -> float:
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


# --- gaussx's problem types, for the precision form ---


@pytest.mark.parametrize("case", sorted(PRECISION_CASES))
def test_port_reproduces_gaussx_laplace_mode(case: str) -> None:
    make_prior, make_likelihood, kwargs = PRECISION_CASES[case]
    ours = laplace_mode(make_prior(), make_likelihood(), **kwargs)
    assert bool(ours.converged)
    assert int(ours.n_iter) == int(GAUSSX[f"{case}_n_iter"])
    assert _relative(ours.mode, GAUSSX[f"{case}_mode"]) <= 1e-10
    log_marginal = float(GAUSSX[f"{case}_log_marginal"])
    assert abs(float(ours.log_marginal) - log_marginal) <= 1e-10 * abs(log_marginal)
    assert type(ours.hessian).__name__ == str(GAUSSX[f"{case}_hessian_type"])


def test_port_reproduces_gaussx_in_float32() -> None:
    prior = gaussian_prior(FLOAT32_SIZE, dtype=jnp.float32)
    ours = laplace_mode(prior, counts(FLOAT32_SIZE, dtype=jnp.float32))
    assert ours.mode.dtype == jnp.float32
    assert _relative(ours.mode, GAUSSX["float32_mode"]) <= 1e-5
    log_marginal = float(GAUSSX["float32_log_marginal"])
    assert abs(float(ours.log_marginal) - log_marginal) <= 1e-4 * abs(log_marginal)


def test_mode_gradient_matches_finite_differences() -> None:
    likelihood = counts(10)

    def summary(scale):
        fit = laplace_mode(gaussian_prior(10, scale=scale[0]), likelihood)
        return jnp.sum(jnp.sin(fit.mode))

    assert_grad_matches_fd(summary, jnp.array([1.3]))


def test_log_marginal_gradient_matches_gaussx() -> None:
    likelihood = counts(20)

    def ours(log_tau):
        return laplace_mode(
            intrinsic_prior(20, jnp.exp(log_tau)), likelihood
        ).log_marginal

    assert jax.grad(ours)(LOG_TAU) == pytest.approx(
        float(GAUSSX["log_marginal_grad"]), rel=1e-9
    )


def test_a_vjp_does_one_solve() -> None:
    likelihood = counts(8)

    def summary(scale):
        return jnp.sum(laplace_mode(gaussian_prior(8, scale=scale), likelihood).mode)

    jaxpr = str(jax.make_jaxpr(jax.grad(summary))(1.0))
    # the implicit tangent is one custom_linear_solve with the factor at x̂
    assert jaxpr.count("custom_linear_solve") == 1


@pytest.mark.slow
def test_runs_under_jit_and_vmap_over_datasets() -> None:
    prior = gaussian_prior(6)
    datasets = jnp.asarray(np.random.default_rng(1).poisson(2.0, (4, 6)), jnp.float64)

    def fit_one(counts):
        return laplace_mode(prior, gx.PoissonLikelihood(counts)).mode

    batched = jax.jit(jax.vmap(fit_one))(datasets)
    looped = jnp.stack([fit_one(counts) for counts in datasets])
    assert jnp.allclose(batched, looped, rtol=1e-10)


def test_result_works_with_gaussx_vb_mean_correction() -> None:
    prior, likelihood = gaussian_prior(VB_SIZE), counts(VB_SIZE)
    ours = laplace_mode(prior, likelihood)
    corrected = gx.vb_mean_correction(
        ours, prior, likelihood, subspace=jnp.array(VB_SUBSPACE)
    )
    assert _relative(corrected, GAUSSX["vb_mean"]) <= 1e-10


def test_posterior_is_the_laplace_at_the_mode() -> None:
    prior, likelihood = gaussian_prior(6), counts(6)
    fit = laplace_mode(prior, likelihood)
    posterior = fit.posterior
    assert isinstance(posterior, Laplace)
    covariance = posterior.covariance().as_matrix()
    assert jnp.allclose(covariance, jnp.linalg.inv(fit.hessian.as_matrix()), rtol=1e-10)


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
        laplace_mode(gaussian_prior(3), counts(3), **kwargs)


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


def test_covariance_log_marginal_is_finite_on_a_badly_conditioned_float32_prior() -> (
    None
):
    """#91: an RBF Gram matrix on 200 dense points, solved against in float32,
    gave a NaN log-marginal while the mode and the variances were finite."""

    def fit(dtype):
        inputs = jnp.linspace(0.0, 10.0, 200, dtype=dtype)
        differences = inputs[:, None] - inputs[None, :]
        jitter = 1e-6 * jnp.eye(200, dtype=dtype)
        kernel = jnp.exp(-0.5 * differences**2) + jitter
        labels = (jnp.sin(inputs) > 0).astype(dtype)
        prior = lx.MatrixLinearOperator(kernel, _PSD)
        return laplace_mode_covariance(prior, jnp.zeros(200, dtype), _bernoulli(labels))

    single = fit(jnp.float32)
    double = fit(jnp.float64)
    assert single.log_marginal.dtype == jnp.float32
    assert bool(single.converged)
    assert bool(jnp.isfinite(single.log_marginal))
    assert abs(float(single.log_marginal) - float(double.log_marginal)) <= 1e-3


def _student_t_problem(dtype):
    """40 noisy points of sin(x) with two outliers, under a Student-t likelihood."""
    generator = np.random.default_rng(0)
    inputs_np = np.linspace(-3.0, 3.0, 40)
    targets_np = np.sin(inputs_np) + 0.1 * generator.normal(size=40)
    targets_np[12], targets_np[28] = 4.0, -4.0
    inputs = jnp.asarray(inputs_np, dtype)
    targets = jnp.asarray(targets_np, dtype)
    differences = inputs[:, None] - inputs[None, :]
    jitter = 1e-6 * jnp.eye(40, dtype=dtype)
    kernel = jnp.exp(-0.5 * differences**2 / 0.7**2) + jitter
    dof, scale = 3.0, 0.3

    def log_likelihood(latent):
        # log Student-t(yₙ ∣ fₙ, ν = 3, s = 0.3)
        standardised = (targets - latent) / scale
        normaliser = (
            jax.scipy.special.gammaln((dof + 1) / 2)
            - jax.scipy.special.gammaln(dof / 2)
            - 0.5 * jnp.log(dof * jnp.pi)
            - jnp.log(scale)
        )
        return normaliser - (dof + 1) / 2 * jnp.log1p(standardised**2 / dof)

    prior = lx.MatrixLinearOperator(kernel, _PSD)
    return prior, jnp.zeros(40, dtype), log_likelihood


@pytest.mark.parametrize(("dtype", "tolerance"), [("float64", 1e-5), ("float32", 5e-5)])
def test_covariance_log_marginal_is_accurate_at_a_mode_converged_to_tolerance(
    dtype: str, tolerance: float
) -> None:
    """#94: the fixed-point identity K_XX⁻¹ (f̂ − μ_f) = g(f̂) gave an evidence
    error first order in the mode error, 3e-4 in float32 at tol 1e-6."""
    reference = laplace_mode_covariance(
        *_student_t_problem(jnp.float64), max_iter=5000, tol=1e-12, damping=0.3
    )
    assert bool(reference.converged)
    fit = laplace_mode_covariance(
        *_student_t_problem(jnp.dtype(dtype)), max_iter=200, tol=1e-6, damping=0.3
    )
    assert bool(fit.converged)
    error = abs(float(fit.log_marginal) - float(reference.log_marginal))
    assert error <= tolerance


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


# --- the matrix-free implicit gradient (#99) ---
#
# The case of the issue's table: Bernoulli labels at 150 points, a Matérn-3/2
# prior, and the gradient of a random linear functional of the mode with
# respect to the kernel variance and lengthscale. The dense reference is the
# tangent solve argminax used before #99, a dense LU of the Jacobian.

_GRADIENT_SIZE = 150
_DENSE_LU = optx.ImplicitAdjoint(linear_solver=lx.AutoLinearSolver(well_posed=True))


def _matern32_problem(size: int = _GRADIENT_SIZE, dtype=jnp.float64):
    generator = np.random.default_rng(1)
    inputs = jnp.asarray(np.sort(generator.uniform(-3.0, 3.0, size)), dtype)
    labels = jnp.asarray(np.sin(2.0 * np.asarray(inputs)) > 0, dtype)
    weights = jnp.asarray(generator.normal(size=size), dtype)
    # |xᵢ − xⱼ|, one row per training point
    distances = jnp.abs(einx.subtract("row, col -> row col", inputs, inputs))

    def prior(params):
        variance, lengthscale = params[0], params[1]
        # k(r) = σ² (1 + √3 r / l) exp(−√3 r / l), plus jitter
        scaled = jnp.sqrt(3.0) * distances / lengthscale
        kernel = variance * (1.0 + scaled) * jnp.exp(-scaled)
        jitter = 1e-6 * jnp.eye(size, dtype=dtype)
        return lx.MatrixLinearOperator(kernel + jitter, _PSD)

    return prior, labels, weights


def _mode_functional(floor: float, adjoint=None, dtype=jnp.float64):
    prior, labels, weights = _matern32_problem(dtype=dtype)

    def functional(params):
        fit = laplace_mode_covariance(
            prior(params),
            jnp.zeros(_GRADIENT_SIZE, dtype),
            _bernoulli(labels),
            max_iter=100,
            tol=1e-12,
            precision_floor=floor,
            adjoint=adjoint,
        )
        # wᵀ f̂
        return einx.dot("state, state ->", weights, fit.mode), fit

    return functional


_PARAMS = jnp.array([1.3, 0.8])


def test_matrix_free_gradient_matches_dense() -> None:
    functional = _mode_functional(1e-6)
    _, fit = functional(_PARAMS)
    assert bool(fit.converged)
    # no site is floored: the Newton map is exact
    assert bool(jnp.all(fit.site_precision > 1e-6))
    gradient = jax.grad(lambda params: functional(params)[0])(_PARAMS)
    dense = jax.grad(lambda params: _mode_functional(1e-6, _DENSE_LU)(params)[0])(
        _PARAMS
    )
    assert _relative(gradient, dense) <= 1e-8
    assert_grad_matches_fd(lambda params: functional(params)[0], _PARAMS)


def test_gradient_with_floored_sites() -> None:
    """A floor of 0.2 floors a large share of the sites; ∂μ_f∣y/∂f ≠ 0."""
    functional = _mode_functional(0.2)
    _, fit = functional(_PARAMS)
    assert bool(fit.converged)
    floored = float(jnp.mean(fit.site_precision == 0.2))
    assert floored >= 0.2
    gradient = jax.grad(lambda params: functional(params)[0])(_PARAMS)
    dense = jax.grad(lambda params: _mode_functional(0.2, _DENSE_LU)(params)[0])(
        _PARAMS
    )
    assert _relative(gradient, dense) <= 1e-8
    assert_grad_matches_fd(lambda params: functional(params)[0], _PARAMS)


class _CountingSolver(lx.AbstractLinearSolver):
    """A lineax solver that counts, at run time, the products ``inner`` takes.

    The count comes from a debug callback, which JAX drops when it transposes
    a linear map, so the products are counted through ``jax.jvp``: the same
    tangent solve, untransposed. ``rtol`` and ``atol`` are the inner solver's,
    so the solver is still seen as iterative.
    """

    inner: lx.AbstractLinearSolver
    counter: list = eqx.field(static=True)

    @property
    def rtol(self) -> float:
        return self.inner.rtol

    @property
    def atol(self) -> float:
        return self.inner.atol

    def init(self, operator, options):
        def counted(vector):
            jax.debug.callback(lambda _: self.counter.append(1), vector)
            return operator.mv(vector)

        wrapped = lx.FunctionLinearOperator(counted, operator.in_structure())
        return self.inner.init(wrapped, options)

    def compute(self, state, vector, options):
        return self.inner.compute(state, vector, options)

    def transpose(self, state, options):
        return self.inner.transpose(state, options)

    def conj(self, state, options):
        return self.inner.conj(state, options)

    def assume_full_rank(self):
        return self.inner.assume_full_rank()


def _gmres_products(floor: float) -> int:
    counter: list = []
    gmres = lx.GMRES(rtol=1e-8, atol=1e-8, restart=20, max_steps=50)
    adjoint = safe_implicit_adjoint(_CountingSolver(gmres, counter))
    functional = _mode_functional(floor, adjoint)
    tangent = jnp.array([1.0, -0.5])
    _, derivative = jax.jvp(
        lambda params: functional(params)[0], (_PARAMS,), (tangent,)
    )
    jax.effects_barrier()
    assert bool(jnp.isfinite(derivative))
    return len(counter)


def test_gradient_products_at_the_mode() -> None:
    # J_T = 0 at the unfloored mode: the one product that checks the guess
    # v = −b finds it within tolerance, and GMRES is never called
    assert _gmres_products(1e-6) == 0
    # with floored sites J_T ≠ 0, the guess is rejected and GMRES runs from it
    assert _gmres_products(0.2) > 0


def test_default_adjoint_is_matrix_free_gmres() -> None:
    adjoint = _default_adjoint(jnp.float64)
    inner = adjoint.linear_solver.inner
    assert isinstance(inner, lx.GMRES)
    assert (inner.rtol, inner.atol, inner.restart, inner.max_steps) == (
        1e-8,
        1e-8,
        20,
        50,
    )
    # float32 cannot reach 1e-8: floored at 64 ε, as the loop's tolerance is
    single = _default_adjoint(jnp.float32).linear_solver.inner
    assert single.rtol == pytest.approx(64 * float(jnp.finfo(jnp.float32).eps))


def test_nonconverged_gradient_does_not_raise() -> None:
    prior, labels, weights = _matern32_problem()

    def functional(params):
        fit = laplace_mode_covariance(
            prior(params),
            jnp.zeros(_GRADIENT_SIZE),
            _bernoulli(labels),
            max_iter=1,
            precision_floor=0.2,
        )
        return einx.dot("state, state ->", weights, fit.mode), fit

    _, fit = functional(_PARAMS)
    assert not bool(fit.converged)
    gradient = jax.grad(lambda params: functional(params)[0])(_PARAMS)
    # finite, or NaN for a sampler to reject; never an exception
    assert bool(jnp.all(jnp.isfinite(gradient) | jnp.isnan(gradient)))


def test_gradient_in_float32() -> None:
    functional = _mode_functional(1e-6, dtype=jnp.float32)
    params = _PARAMS.astype(jnp.float32)
    gradient = jax.grad(lambda params: functional(params)[0])(params)
    assert gradient.dtype == jnp.float32
    double = jax.grad(lambda params: _mode_functional(1e-6)(params)[0])(_PARAMS)
    # float32 rounding through 150 solves against a Matérn Gram matrix
    assert _relative(gradient, double) <= 1e-3


@pytest.mark.slow
def test_jvp_hessian_and_vmap_over_datasets() -> None:
    size = 30
    prior, labels, weights = _matern32_problem(size)

    def functional(params, labels):
        fit = laplace_mode_covariance(
            prior(params), jnp.zeros(size), _bernoulli(labels), max_iter=100, tol=1e-12
        )
        return einx.dot("state, state ->", weights, fit.mode)

    tangent = jnp.array([1.0, -0.5])
    _, forward = jax.jvp(
        lambda params: functional(params, labels), (_PARAMS,), (tangent,)
    )
    reverse = jax.grad(functional)(_PARAMS, labels)
    assert forward == pytest.approx(
        float(einx.dot("param, param ->", reverse, tangent)), rel=1e-8
    )

    hessian = jax.hessian(functional)(_PARAMS, labels)
    # the Hessian against central differences of the gradient
    step = 1e-5
    columns = [
        (
            jax.grad(functional)(_PARAMS + step * direction, labels)
            - jax.grad(functional)(_PARAMS - step * direction, labels)
        )
        / (2.0 * step)
        for direction in jnp.eye(2)
    ]
    # row j is ∂ⱼ∇, the Hessian's j-th column; the Hessian is symmetric
    finite_differences = jnp.stack(columns)
    assert _relative(hessian, finite_differences) <= 1e-5

    datasets = jnp.stack([labels, 1.0 - labels])
    batched = jax.jit(jax.vmap(jax.grad(functional), in_axes=(None, 0)))(
        _PARAMS, datasets
    )
    looped = jnp.stack([jax.grad(functional)(_PARAMS, data) for data in datasets])
    assert jnp.allclose(batched, looped, rtol=1e-10)


def _shapes_in(jaxpr) -> set[tuple[int, ...]]:
    """Every array shape in a jaxpr, its sub-jaxprs included."""
    shapes = set()
    for equation in jaxpr.eqns:
        for variable in [*equation.invars, *equation.outvars]:
            shape = getattr(variable.aval, "shape", None)
            if shape is not None:
                shapes.add(tuple(shape))
        for value in equation.params.values():
            for sub in value if isinstance(value, tuple | list) else [value]:
                # a ClosedJaxpr holds its Jaxpr; a Jaxpr has equations
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    shapes |= _shapes_in(inner)
    return shapes


@pytest.mark.parametrize(("adjoint", "dense"), [(None, False), (_DENSE_LU, True)])
def test_site_gradient_forms_no_n_by_n_array(adjoint, dense: bool) -> None:
    """A matrix-free posterior-mean map keeps the gradient matrix-free.

    The independent prior f ~ 𝒩(0, diag(k)) gives μ_f∣y = Λμ_site / (1/k + Λ_site)
    elementwise; the dense LU materialises the N × N Jacobian of the fixed point,
    GMRES does not.
    """
    size = 37
    labels = jnp.asarray(np.arange(size) % 3 == 0, jnp.float64)

    def functional(variance):
        def posterior_mean(information, precision):
            # μ_f∣y = Λμ_site / (1/k + Λ_site)
            return information / (1.0 / variance + precision)

        fit = laplace_mode_sites(
            posterior_mean,
            _bernoulli(labels),
            jnp.zeros(size),
            max_iter=50,
            tol=1e-10,
            adjoint=adjoint,
        )
        return jnp.sum(jnp.sin(fit.mode))

    variance = jnp.full(size, 1.5)
    jaxpr = jax.make_jaxpr(jax.grad(functional))(variance).jaxpr
    assert ((size, size) in _shapes_in(jaxpr)) is dense
    assert_grad_matches_fd(functional, variance)


def test_site_gradient_matches_dense() -> None:
    data = PYROX[64]
    labels = _bernoulli(jnp.asarray(data["markov_y"]))
    kernel = jnp.asarray(data["markov_K"])
    size = kernel.shape[0]

    def summary(scale, adjoint):
        prior = lx.MatrixLinearOperator(scale[0] * kernel, _PSD)

        def posterior_mean(information, precision):
            regularised = lx.TaggedLinearOperator(
                prior + lx.DiagonalLinearOperator(1.0 / precision), _PSD
            )
            return scale[0] * kernel @ gx.solve(regularised, information / precision)

        fit = laplace_mode_sites(
            posterior_mean,
            labels,
            jnp.zeros(size),
            max_iter=60,
            tol=1e-12,
            precision_floor=0.2,
            adjoint=adjoint,
        )
        return jnp.sum(jnp.sin(fit.mode))

    scale = jnp.array([0.8])
    gradient = jax.grad(summary)(scale, None)
    dense = jax.grad(summary)(scale, _DENSE_LU)
    assert _relative(gradient, dense) <= 1e-8


def test_adjoint_is_validated() -> None:
    with pytest.raises(ValidationError, match="'adjoint' must be an optimistix"):
        laplace_mode_covariance(
            lx.MatrixLinearOperator(jnp.eye(2), _PSD),
            jnp.zeros(2),
            _bernoulli(jnp.ones(2)),
            adjoint=optx.RecursiveCheckpointAdjoint(),
        )

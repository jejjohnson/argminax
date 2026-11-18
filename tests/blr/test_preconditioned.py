"""Tests for `scale_by_preconditioner` and `preconditioned_sgd`.

The step is θ ← θ − α P⁻¹ g with a fixed P. The tests pin its limits (Newton
for P the Hessian, ``optax.sgd`` for P the identity, a division for a diagonal
P), its covariance under a linear change of variables, and the contraction
rate a quadratic cost gives it.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optax
import pytest
from _harness import assert_grad_matches_fd
from hypothesis import given, settings, strategies as st

from argminax import ValidationError
from argminax.blr import preconditioned_sgd, scale_by_preconditioner


def _spd(key, size: int, shift: float = 1.0):
    # A Aᵀ + shift · I, symmetric positive definite
    factor = jax.random.normal(key, (size, size))
    return factor @ factor.T + shift * jnp.eye(size)


def _psd_operator(matrix):
    return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)


def _psd_diagonal(diagonal):
    # lineax does not tag a diagonal operator: its entries could be negative
    operator = lx.DiagonalLinearOperator(diagonal)
    return lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)


def _run(optimiser, grad, start, num_steps: int):
    theta = start
    state = optimiser.init(theta)
    iterates = []
    for _ in range(num_steps):
        updates, state = optimiser.update(grad(theta), state)
        theta = theta + updates
        iterates.append(theta)
    return jnp.stack(iterates)


def test_hessian_preconditioner_is_newton_in_one_step() -> None:
    # 𝒥(θ) = ½ θᵀ A θ − bᵀ θ, minimiser A⁻¹ b
    hessian = _spd(jax.random.key(0), 5)
    rhs = jax.random.normal(jax.random.key(1), (5,))
    optimiser = preconditioned_sgd(1.0, _psd_operator(hessian))
    theta = jax.random.normal(jax.random.key(2), (5,))
    state = optimiser.init(theta)
    updates, _ = optimiser.update(hessian @ theta - rhs, state)
    np.testing.assert_allclose(
        theta + updates, jnp.linalg.solve(hessian, rhs), rtol=1e-12
    )


def test_identity_preconditioner_is_sgd() -> None:
    grads = jax.random.normal(jax.random.key(0), (4,))
    identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((4,), grads.dtype))
    identity = lx.TaggedLinearOperator(identity, lx.positive_semidefinite_tag)
    ours = preconditioned_sgd(0.3, identity)
    reference = optax.sgd(0.3)
    theta = jnp.zeros(4)
    updates, _ = ours.update(grads, ours.init(theta))
    expected, _ = reference.update(grads, reference.init(theta))
    np.testing.assert_allclose(updates, expected, rtol=1e-14)


def test_diagonal_preconditioner_divides() -> None:
    diagonal = jnp.array([0.5, 2.0, 8.0])
    grads = jnp.array([1.0, -3.0, 4.0])
    scaling = scale_by_preconditioner(_psd_diagonal(diagonal))
    updates, _ = scaling.update(grads, scaling.init(jnp.zeros(3)))
    np.testing.assert_allclose(updates, grads / diagonal, rtol=1e-14)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**16), size=st.integers(2, 6))
def test_iterates_are_covariant_under_a_linear_change_of_variables(seed, size) -> None:
    # 𝒥̃(z) = 𝒥(C z) with P̃ = Cᵀ P C gives z⁽ⁱ⁾ = C⁻¹ θ⁽ⁱ⁾
    keys = jax.random.split(jax.random.key(seed), 5)
    hessian = _spd(keys[0], size)
    preconditioner = _spd(keys[1], size)
    rhs = jax.random.normal(keys[2], (size,))
    change = _spd(keys[3], size)
    start = jax.random.normal(keys[4], (size,))

    theta = _run(
        preconditioned_sgd(0.1, _psd_operator(preconditioner)),
        lambda theta: hessian @ theta - rhs,
        start,
        4,
    )
    changed = _psd_operator(change.T @ preconditioner @ change)
    z = _run(
        preconditioned_sgd(0.1, changed),
        lambda z: change.T @ (hessian @ (change @ z) - rhs),
        jnp.linalg.solve(change, start),
        4,
    )
    np.testing.assert_allclose(z @ change.T, theta, rtol=1e-8, atol=1e-10)


def test_error_contracts_at_the_rate_of_the_preconditioned_spectrum() -> None:
    # In the P-norm the error shrinks by at most ρ = (κ − 1) / (κ + 1) a step at
    # α = 2 / (λ_min + λ_max), with λ the eigenvalues of P⁻¹ H.
    keys = jax.random.split(jax.random.key(7), 4)
    hessian = _spd(keys[0], 6)
    preconditioner = _spd(keys[1], 6)
    rhs = jax.random.normal(keys[2], (6,))
    minimiser = jnp.linalg.solve(hessian, rhs)
    chol = jnp.linalg.cholesky(preconditioner)
    whitened = jnp.linalg.solve(chol, jnp.linalg.solve(chol, hessian).T)
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (whitened + whitened.T))
    step = 2.0 / (eigenvalues[0] + eigenvalues[-1])
    rate = (eigenvalues[-1] - eigenvalues[0]) / (eigenvalues[-1] + eigenvalues[0])

    start = jax.random.normal(keys[3], (6,))
    iterates = _run(
        preconditioned_sgd(float(step), _psd_operator(preconditioner)),
        lambda theta: hessian @ theta - rhs,
        start,
        10,
    )
    errors = jnp.concatenate([start[None], iterates]) - minimiser
    # ‖e‖_P = ‖Lᵀ e‖₂
    norms = jnp.linalg.norm(errors @ chol, axis=1)
    assert jnp.all(norms[1:] <= rate * norms[:-1] * (1 + 1e-9))


def test_structured_operator_matches_its_dense_matrix() -> None:
    diagonal = 1.0 + jax.random.uniform(jax.random.key(0), (30,))
    factor = jax.random.normal(jax.random.key(1), (30, 3))
    grads = jax.random.normal(jax.random.key(2), (30,))
    structured = scale_by_preconditioner(
        gx.low_rank_plus_diag(diagonal, factor, psd=True)
    )
    dense = scale_by_preconditioner(
        _psd_operator(jnp.diag(diagonal) + factor @ factor.T)
    )
    start = jnp.zeros(30)
    from_structure, _ = structured.update(grads, structured.init(start))
    from_dense, _ = dense.update(grads, dense.init(start))
    np.testing.assert_allclose(from_structure, from_dense, rtol=1e-10)


@pytest.mark.parametrize(
    "solver",
    [None, lx.Cholesky(), lx.CG(rtol=1e-12, atol=1e-12), gx.CGSolver(rtol=1e-12)],
    ids=["default", "cholesky", "lineax-cg", "gaussx-cg"],
)
def test_every_solver_gives_the_same_update(solver) -> None:
    matrix = _spd(jax.random.key(0), 5)
    grads = jax.random.normal(jax.random.key(1), (5,))
    scaling = scale_by_preconditioner(_psd_operator(matrix), solver=solver)
    updates, _ = scaling.update(grads, scaling.init(jnp.zeros(5)))
    np.testing.assert_allclose(updates, jnp.linalg.solve(matrix, grads), rtol=1e-8)


def test_jit_and_scan_agree_with_an_eager_loop() -> None:
    hessian = _spd(jax.random.key(0), 4)
    rhs = jax.random.normal(jax.random.key(1), (4,))
    optimiser = preconditioned_sgd(0.2, _psd_operator(_spd(jax.random.key(2), 4)))
    start = jnp.ones(4)
    eager = _run(optimiser, lambda theta: hessian @ theta - rhs, start, 6)

    def step(carry, _):
        theta, state = carry
        updates, state = optimiser.update(hessian @ theta - rhs, state)
        return (theta + updates, state), theta + updates

    _, scanned = jax.jit(lambda carry: jax.lax.scan(step, carry, length=6))(
        (start, optimiser.init(start))
    )
    np.testing.assert_allclose(scanned, eager, rtol=1e-12)


def test_float32_gradients_stay_float32_under_a_float64_operator() -> None:
    scaling = scale_by_preconditioner(_psd_operator(2.0 * jnp.eye(3)))
    grads = jnp.ones(3, jnp.float32)
    updates, _ = scaling.update(grads, scaling.init(jnp.zeros(3, jnp.float32)))
    assert updates.dtype == jnp.float32
    np.testing.assert_allclose(updates, 0.5, rtol=1e-6)


def test_schedule_is_read_at_the_step_count() -> None:
    schedule = optax.piecewise_constant_schedule(0.1, {1: 0.5})
    optimiser = preconditioned_sgd(schedule, _psd_diagonal(jnp.ones(1)))
    state = optimiser.init(jnp.zeros(1))
    first, state = optimiser.update(jnp.ones(1), state)
    second, state = optimiser.update(jnp.ones(1), state)
    np.testing.assert_allclose(first, -0.1, rtol=1e-12)
    np.testing.assert_allclose(second, -0.05, rtol=1e-12)


def test_gradient_through_the_steps_matches_finite_differences() -> None:
    # with respect to the start and to the diagonal of the preconditioner
    hessian = _spd(jax.random.key(0), 3)
    rhs = jax.random.normal(jax.random.key(1), (3,))

    def cost_after(inputs):
        start, diagonal = inputs
        optimiser = preconditioned_sgd(0.3, _psd_diagonal(diagonal))
        theta = _run(optimiser, lambda theta: hessian @ theta - rhs, start, 3)[-1]
        return 0.5 * theta @ hessian @ theta - rhs @ theta

    inputs = (jnp.array([0.4, -0.2, 0.1]), jnp.array([2.0, 3.0, 5.0]))
    assert_grad_matches_fd(cost_after, inputs)


@pytest.mark.parametrize(
    ("build", "name"),
    [
        (lambda: scale_by_preconditioner(lx.MatrixLinearOperator(jnp.eye(2))), "tag"),
        (
            lambda: scale_by_preconditioner(lx.MatrixLinearOperator(jnp.ones((2, 3)))),
            "square",
        ),
        (
            lambda: preconditioned_sgd(0.0, _psd_diagonal(jnp.ones(2))),
            "learning_rate",
        ),
    ],
)
def test_bad_options_are_rejected_by_name(build, name) -> None:
    with pytest.raises(ValidationError, match=name):
        build()


@pytest.mark.parametrize("shape", [(3,), (2, 1), ()])
def test_parameters_of_another_shape_are_rejected_at_init(shape) -> None:
    scaling = scale_by_preconditioner(_psd_diagonal(jnp.ones(2)))
    with pytest.raises(ValidationError, match="parameters"):
        scaling.init(jnp.zeros(shape))

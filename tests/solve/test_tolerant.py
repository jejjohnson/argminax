"""Tests for `TolerantLinearSolver`: a stopped or non-finite solve returns a value.

Gates (#87): well-posed solves are the inner solver's, bitwise; a capped solve
keeps the inner iterate and reports success; a non-finite system gives zeros and
success under ``jax.jit``; ``transpose``, ``conj`` and ``assume_full_rank``
delegate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from argminax.solve import TolerantLinearSolver


def _spd_operator(size: int = 6) -> lx.MatrixLinearOperator:
    factor = jax.random.normal(jax.random.key(0), (size, size))
    # A = F Fᵀ + D I, symmetric positive definite
    matrix = factor @ factor.T + size * jnp.eye(size)
    return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)


def _nonsymmetric_operator(size: int = 6) -> lx.MatrixLinearOperator:
    perturbation = jax.random.normal(jax.random.key(1), (size, size))
    # A = 3 I + P / D, well conditioned and not symmetric
    return lx.MatrixLinearOperator(3 * jnp.eye(size) + perturbation / size)


INNER = [
    pytest.param(lx.CG(rtol=1e-12, atol=1e-12), _spd_operator, id="cg"),
    pytest.param(
        lx.GMRES(rtol=1e-12, atol=1e-12, restart=6), _nonsymmetric_operator, id="gmres"
    ),
    pytest.param(lx.LU(), _nonsymmetric_operator, id="lu"),
]


@pytest.mark.parametrize(("inner", "make_operator"), INNER)
def test_well_posed_solve_is_the_inner_solve(inner, make_operator):
    operator = make_operator()
    vector = jnp.arange(1.0, 7.0)
    expected = lx.linear_solve(operator, vector, inner)
    actual = lx.linear_solve(operator, vector, TolerantLinearSolver(inner))
    assert actual.result == lx.RESULTS.successful
    assert jnp.array_equal(actual.value, expected.value)


def test_capped_gmres_keeps_the_inner_iterate():
    operator = _nonsymmetric_operator()
    vector = jnp.arange(1.0, 7.0)
    inner = lx.GMRES(rtol=1e-14, atol=1e-14, restart=1, max_steps=2)
    capped = lx.linear_solve(operator, vector, inner, throw=False)
    assert capped.result == lx.RESULTS.max_steps_reached
    tolerant = lx.linear_solve(operator, vector, TolerantLinearSolver(inner))
    assert tolerant.result == lx.RESULTS.successful
    assert jnp.array_equal(tolerant.value, capped.value)


def test_capped_solve_raises_without_accept_stopped():
    operator = _nonsymmetric_operator()
    inner = lx.GMRES(rtol=1e-14, atol=1e-14, restart=1, max_steps=2)
    solver = TolerantLinearSolver(inner, accept_stopped=False)
    solution = lx.linear_solve(operator, jnp.ones(6), solver, throw=False)
    assert solution.result == lx.RESULTS.max_steps_reached


@pytest.mark.parametrize("where", ["operator", "vector"])
def test_nonfinite_system_gives_zeros_under_jit(where):
    inner = lx.GMRES(rtol=1e-10, atol=1e-10, restart=6, max_steps=20)

    @jax.jit
    def solve(matrix, vector):
        operator = lx.MatrixLinearOperator(matrix)
        return lx.linear_solve(operator, vector, TolerantLinearSolver(inner))

    matrix = _nonsymmetric_operator().matrix
    vector = jnp.ones(6)
    if where == "operator":
        matrix = matrix.at[0, 0].set(jnp.nan)
    else:
        vector = vector.at[2].set(jnp.inf)
    solution = solve(matrix, vector)
    assert solution.result == lx.RESULTS.successful
    assert jnp.array_equal(solution.value, jnp.zeros(6))


def test_nonfinite_system_passes_through_without_zeroing():
    solver = TolerantLinearSolver(lx.LU(), zero_if_nonfinite=False)
    vector = jnp.ones(6).at[0].set(jnp.nan)
    solution = lx.linear_solve(_nonsymmetric_operator(), vector, solver, throw=False)
    assert not jnp.all(jnp.isfinite(solution.value))


def test_singular_direct_solve_is_reported_as_the_inner_solver_reports_it():
    operator = lx.MatrixLinearOperator(jnp.zeros((3, 3)))
    expected = lx.linear_solve(operator, jnp.ones(3), lx.LU(), throw=False)
    actual = lx.linear_solve(
        operator, jnp.ones(3), TolerantLinearSolver(lx.LU()), throw=False
    )
    assert expected.result != lx.RESULTS.successful
    assert actual.result == expected.result


def test_transpose_conj_and_rank_delegate():
    inner = lx.LU()
    solver = TolerantLinearSolver(inner)
    operator = _nonsymmetric_operator()
    state = solver.init(operator, {})
    assert solver.assume_full_rank() == inner.assume_full_rank()
    transposed, _ = solver.transpose(state, {})
    expected, _ = inner.transpose(inner.init(operator, {}), {})
    assert jax.tree.all(jax.tree.map(jnp.array_equal, transposed, expected))
    conjugated, _ = solver.conj(state, {})
    expected_conj, _ = inner.conj(inner.init(operator, {}), {})
    assert jax.tree.all(jax.tree.map(jnp.array_equal, conjugated, expected_conj))

"""Tests for `safe_implicit_adjoint`: an implicit adjoint that never raises.

Gates (#87): on a contraction its gradient is ``optx.ImplicitAdjoint``'s and the
finite-difference gradient; where the adjoint solve is capped, ``ImplicitAdjoint``
raises and this one returns a finite gradient; a non-finite fit gives a
non-finite value without an exception.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import pytest

from argminax import ValidationError
from argminax.adjoint import safe_implicit_adjoint
from argminax.solve import TolerantLinearSolver


COUPLING = jnp.array([[0.0, 0.3, 0.1], [0.2, 0.0, 0.3], [0.1, 0.2, 0.0]])


def _step(state, theta):
    # T(s, θ) = tanh(θ ⊙ (C s) + 1), a contraction for |θ| < 1
    return jnp.tanh(theta * (COUPLING @ state) + 1.0)


def _loss(adjoint, *, throw=True):
    def loss(theta):
        solver = optx.FixedPointIteration(rtol=1e-12, atol=1e-12)
        solution = optx.fixed_point(
            _step,
            solver,
            jnp.zeros(3),
            theta,
            adjoint=adjoint,
            max_steps=1000,
            throw=throw,
        )
        return jnp.sum(solution.value**2)

    return loss


THETA = jnp.array([0.5, -0.4, 0.7])


def test_gradient_matches_implicit_adjoint_on_a_contraction():
    solver = optx.FixedPointIteration(rtol=1e-12, atol=1e-12)
    solution = optx.fixed_point(_step, solver, jnp.zeros(3), THETA, max_steps=1000)
    assert solution.result == optx.RESULTS.successful
    gmres = lx.GMRES(rtol=1e-12, atol=1e-12, restart=3)
    expected = jax.grad(_loss(optx.ImplicitAdjoint(linear_solver=gmres)))(THETA)
    actual = jax.grad(_loss(safe_implicit_adjoint(gmres)))(THETA)
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_gradient_matches_finite_differences():
    gradient = jax.grad(_loss(safe_implicit_adjoint()))(THETA)
    loss = _loss(safe_implicit_adjoint())
    eps = 1e-6
    basis = jnp.eye(3)
    # central differences, error O(ε²)
    finite = jnp.stack(
        [(loss(THETA + eps * e) - loss(THETA - eps * e)) / (2 * eps) for e in basis]
    )
    assert jnp.allclose(gradient, finite, rtol=1e-6, atol=1e-9)


def test_capped_adjoint_solve_gives_a_finite_gradient_where_implicit_raises():
    capped = lx.GMRES(rtol=1e-14, atol=1e-14, restart=1, max_steps=1)
    # eager, equinox's runtime error surfaces wrapped in JAX's
    with pytest.raises(
        (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
        match="maximum number of solver steps",
    ):
        jax.grad(_loss(optx.ImplicitAdjoint(linear_solver=capped)))(THETA)
    gradient = jax.grad(_loss(safe_implicit_adjoint(capped)))(THETA)
    assert jnp.all(jnp.isfinite(gradient))


def test_nonfinite_fit_gives_a_nonfinite_value_without_raising():
    adjoint = safe_implicit_adjoint(lx.GMRES(rtol=1e-10, atol=1e-10, max_steps=20))

    @jax.jit
    def value_and_grad(theta):
        # the forward loop keeps its last iterate, as a sampler's model does
        return jax.value_and_grad(_loss(adjoint, throw=False))(theta)

    value, gradient = value_and_grad(THETA.at[0].set(jnp.nan))
    assert not jnp.isfinite(value)
    assert gradient.shape == (3,)


def test_default_solver_is_optimistix_default():
    adjoint = safe_implicit_adjoint()
    assert isinstance(adjoint, optx.ImplicitAdjoint)
    assert isinstance(adjoint.linear_solver, TolerantLinearSolver)
    assert adjoint.linear_solver.inner == optx.ImplicitAdjoint().linear_solver


def test_rejects_a_non_lineax_solver():
    with pytest.raises(ValidationError, match="linear_solver"):
        safe_implicit_adjoint("gmres")

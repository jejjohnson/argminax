"""A lineax solver wrapper whose stopped or non-finite solves return values.

lineax raises when a linear solve fails, which ends an outer solve or a sampler
at the first bad point. Inexact Gauss–Newton takes a truncated Krylov solve as
its step, and an implicit adjoint inside NUTS must return a value at any
proposal, so both wrap their solver in `TolerantLinearSolver`.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from jaxtyping import Array, Bool, PyTree


__all__ = [
    "TolerantLinearSolver",
]


def _all_finite(tree: PyTree) -> Bool[Array, ""]:
    """Whether every inexact array leaf of a pytree is finite."""
    leaves = jtu.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return jnp.all(jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


class TolerantLinearSolver(lx.AbstractLinearSolver):
    """Wrap a lineax solver so a capped, stagnant or non-finite solve returns a value.

        A v = b,   v = inner(A, b), except
        v = v⁽ᵏ⁾ (the last iterate)   when the solve stops early,
        v = 0                         when A or b is non-finite

    The wrapped solver solves A v = b for v ∈ ℝ^D as ``inner`` does. Three
    outcomes change: a solve that stops at ``max_steps`` (``max_steps_reached``)
    or stagnates (``stagnation``) keeps its last iterate v⁽ᵏ⁾ and is reported
    as successful, a truncated Krylov solution, as inexact Gauss–Newton takes
    for its step; and where the operator A or the right-hand side b holds a
    non-finite value, the solution is zero and reported as successful, so a
    value computed downstream stays non-finite and a sampler rejects the point
    instead of crashing. Every other result (a singular direct solve, a
    breakdown) is passed through as ``inner`` reports it, and on a well-posed
    system the wrapper returns exactly what ``inner`` returns.

    Attributes:
        inner: The lineax solver that does the solve.
        accept_stopped: Whether a solve stopped at ``max_steps`` or by
            stagnation is reported as successful with its last iterate.
        zero_if_nonfinite: Whether a non-finite operator or right-hand side
            gives v = 0, reported as successful.

    Examples:
        A CG solve capped at one step keeps its iterate instead of raising:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import argminax
        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> operator = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> solver = argminax.solve.TolerantLinearSolver(
        ...     lx.CG(rtol=1e-10, atol=1e-10, max_steps=1)
        ... )
        >>> solution = lx.linear_solve(operator, jnp.array([1.0, 2.0]), solver)
        >>> bool(solution.result == lx.RESULTS.successful)
        True

        A non-finite right-hand side gives zeros:

        >>> bad = lx.linear_solve(operator, jnp.array([jnp.nan, 2.0]), solver)
        >>> bad.value.tolist()
        [0.0, 0.0]
    """

    inner: lx.AbstractLinearSolver
    accept_stopped: bool = eqx.field(static=True, default=True)
    zero_if_nonfinite: bool = eqx.field(static=True, default=True)

    def init(self, operator: lx.AbstractLinearOperator, options: dict[str, Any]) -> Any:
        return self.inner.init(operator, options)

    def compute(
        self, state: Any, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], lx.RESULTS, dict[str, Any]]:
        solution, result, stats = self.inner.compute(state, vector, options)
        if self.accept_stopped:
            stopped = (result == lx.RESULTS.max_steps_reached) | (
                result == lx.RESULTS.stagnation
            )
            result = lx.RESULTS.where(stopped, lx.RESULTS.successful, result)
        if self.zero_if_nonfinite:
            # the operator's arrays sit in the solver state
            broken = jnp.logical_not(_all_finite((state, vector)))
            solution = jtu.tree_map(
                lambda leaf: jnp.where(broken, jnp.zeros_like(leaf), leaf), solution
            )
            result = lx.RESULTS.where(broken, lx.RESULTS.successful, result)
        return solution, result, stats

    def transpose(self, state: Any, options: dict[str, Any]) -> Any:
        return self.inner.transpose(state, options)

    def conj(self, state: Any, options: dict[str, Any]) -> Any:
        return self.inner.conj(state, options)

    def assume_full_rank(self) -> bool:
        return self.inner.assume_full_rank()

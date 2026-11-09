"""K-step and one-step adjoints: differentiate through the last k iterate updates.

The solve runs to the end under ``stop_gradient`` while a ring buffer keeps the
last few loop carries; the final steps are then replayed from the buffer, live
in the solve's arguments, so the forward value is the plain solve's and the
gradient flows through those steps only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
from jaxtyping import Array, PyTree

from argminax.exceptions import ArgminaxError, ValidationError


__all__ = [
    "KStep",
    "OneStep",
]


# optimistix 0.1.x passes `adjoint.apply` the tuple
#   (fn, solver, y0, args, options, max_steps, f_struct, aux_struct, tags)
# and `primal_fn` takes it with one more entry, the while loop to run. This
# layout is private, so `pyproject.toml` pins optimistix to 0.1.x and
# `tests/adjoint/test_kstep.py::test_optimistix_inputs_layout` checks it.
_INPUTS_LENGTH = 9


def _is_static(leaf: Any) -> bool:
    return isinstance(leaf, eqxi.Static)


def _proposes_next_iterate(inputs: tuple[Any, ...]) -> bool:
    """Whether the solver's state holds the next iterate as ``y_eval``.

    optimistix's search-based solvers (gradient descent, nonlinear CG, BFGS,
    Gauss–Newton, Levenberg–Marquardt, dogleg) compute the next iterate in one
    step and return it from the next, so each iterate update spans two steps.
    """
    if len(inputs) != _INPUTS_LENGTH:
        msg = (
            f"optimistix passed {len(inputs)} solve inputs, not {_INPUTS_LENGTH}: "
            f"this optimistix {optx.__version__} is outside the pinned 0.1.x"
        )
        raise ArgminaxError(msg)
    fn, solver, y0, args, options, _, f_struct, aux_struct, tags = inputs

    def unwrap(tree: PyTree) -> PyTree:
        return jtu.tree_map(lambda leaf: leaf.value, tree, is_leaf=_is_static)

    state = eqx.filter_eval_shape(
        solver.init, fn, y0, args, options, unwrap(f_struct), unwrap(aux_struct), tags
    )
    return hasattr(state, "y_eval")


class KStep(optx.AbstractAdjoint):
    """Differentiate a minimiser through its last k iterate updates only.

        ξ̂ = Φᵏ(ξ̄, θ),   ξ̄ = sg(ξ_{T−k})

    Φ is one iterate update of the solver, θ the solve's ``args``, T the
    number of updates the solve took and sg the stop-gradient, so ∂ξ̂/∂θ
    flows through the last k updates and the backward pass costs O(k) memory
    whatever T is. The value of ξ̂ is the plain solve's ξ_T, bit for bit. At a
    converged fixed point of a Newton-like Φ, ∂Φ/∂ξ vanishes and the estimate
    is exact for any k; for a solver stopped early (learned 4D-Var), a larger k
    trades memory for less bias.

    Algorithm:
        1. Run the solve under ``stop_gradient`` in a ``lax.while_loop``
           whose carry also holds a ring buffer of the last k + 1 loop carries,
           since T is known only when the loop stops.
        2. From the buffered carry at update T − k, replay the remaining
           solver steps, live in ``args``, and return the replayed iterate.
           optimistix's search-based solvers (those whose state holds the
           next iterate as ``y_eval``: gradient descent, nonlinear CG, BFGS,
           Gauss–Newton, Levenberg–Marquardt) compute an update in one step
           and return it from the next, so for them the replay is k + 1
           steps; for any other solver it is k.

    A step whose line search rejects the proposal leaves the iterate
    unchanged and carries no gradient. A line-search solver run to a
    tolerance near machine precision can end on a run of rejections, and
    then the k-step gradient is zero: at a converged minimum use
    ``optx.ImplicitAdjoint`` instead. The gradient with respect to ``y0`` is
    zero unless the solve takes fewer than k updates.

    Attributes:
        k: Number of trailing iterate updates that carry gradients, at
            least 1.

    Raises:
        ValidationError: If ``k`` is not a positive integer.

    Examples:
        Gradient descent with unit step on f(x, θ) = θx²/2 from x₀ = 1:
        ξₜ₊₁ = (1 − θ) ξₜ, so after two updates at θ = 0.5 the last one
        gives ∂ξ₂/∂θ = −ξ₁ = −0.5 (optimistix counts one extra step, hence
        ``max_steps=3``).

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import optimistix as optx
        >>> from argminax.adjoint import KStep
        >>> def cost(x, theta):
        ...     return 0.5 * theta * jnp.sum(x**2)
        >>> def solve(theta):
        ...     solver = optx.GradientDescent(1.0, rtol=1e-12, atol=1e-12)
        ...     solution = optx.minimise(
        ...         cost,
        ...         solver,
        ...         jnp.ones(1),
        ...         theta,
        ...         adjoint=KStep(k=1),
        ...         max_steps=3,
        ...         throw=False,
        ...     )
        ...     return solution.value[0]
        >>> bool(jnp.allclose(solve(0.5), 0.25))
        True
        >>> bool(jnp.allclose(jax.grad(solve)(0.5), -0.5))
        True

        A non-positive ``k`` is rejected.

        >>> KStep(k=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'k' must be a positive integer, got 0

    References:
        Bolte, J., Pauwels, E. & Vaiter, S. (2023). One-step differentiation
        of iterative algorithms. NeurIPS 36. arXiv:2305.13768.
    """

    k: int = eqx.field(static=True, default=1)

    def __check_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            msg = f"'k' must be a positive integer, got {self.k!r}"
            raise ValidationError(msg)

    def get_config(self) -> dict[str, Any]:
        """Return the fields needed to rebuild the adjoint with `from_config`."""
        return {"k": self.k}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Rebuild an adjoint from the output of `get_config`."""
        return cls(**config)

    def apply(
        self,
        primal_fn: Callable,
        rewrite_fn: Callable,
        inputs: PyTree,
        tags: frozenset[object],
    ) -> PyTree[Array]:
        """Run the solve, differentiable through its last k updates only.

        Args:
            primal_fn: optimistix's solver loop, called with ``inputs`` and a
                while loop.
            rewrite_fn: The optimality condition (unused).
            inputs: optimistix's private solve inputs.
            tags: The operator tags of the solve (unused).

        Returns:
            What ``primal_fn`` returns: the final iterate and the solver's
            bookkeeping, from the replayed steps.
        """
        del rewrite_fn, tags
        # --- how many solver steps hold the last k updates ---
        lag = int(_proposes_next_iterate(inputs))
        window = self.k + lag

        # --- the solve, under stop_gradient, buffering the last carries ---
        dynamic, static = eqx.partition(inputs, eqx.is_inexact_array)
        frozen_inputs = eqx.combine(jax.lax.stop_gradient(dynamic), static)
        recorded: dict[str, Any] = {}

        def recording_loop(
            cond_fun: Callable, body_fun: Callable, init_carry: Any, *, max_steps: Any
        ) -> Any:
            # every slot starts as the initial carry, so a solve with fewer
            # than `window` steps replays from the start
            buffer = jtu.tree_map(
                lambda leaf: jnp.broadcast_to(leaf, (window, *jnp.shape(leaf))),
                init_carry,
            )

            def keep_going(loop_carry: tuple[Any, Any, Array]) -> Array:
                carry, _, steps = loop_carry
                running = cond_fun(carry)
                if max_steps is None:
                    return running
                return running & (steps < max_steps)

            def record_and_step(
                loop_carry: tuple[Any, Any, Array],
            ) -> tuple[Any, Any, Array]:
                carry, buffer, steps = loop_carry
                # the carry entering step s goes to slot s mod window
                slot = steps % window
                buffer = jtu.tree_map(
                    lambda stored, leaf: stored.at[slot].set(leaf), buffer, carry
                )
                return body_fun(carry), buffer, steps + 1

            final_carry, buffer, steps = jax.lax.while_loop(
                keep_going, record_and_step, (init_carry, buffer, jnp.array(0))
            )
            recorded["buffer"] = buffer
            recorded["steps"] = steps
            return final_carry

        primal_fn((*frozen_inputs, recording_loop))
        buffer = recorded["buffer"]
        steps = recorded["steps"]

        # --- replay the last steps, live in args ---
        def replay_loop(
            cond_fun: Callable, body_fun: Callable, init_carry: Any, *, max_steps: Any
        ) -> Any:
            del cond_fun, init_carry, max_steps
            # the carry entering step T − window sits in slot T mod window
            carry = jtu.tree_map(lambda stored: stored[steps % window], buffer)
            replayed = jnp.minimum(steps, window)
            for index in range(window):
                carry = jax.lax.cond(
                    index < replayed, body_fun, lambda unchanged: unchanged, carry
                )
            return carry

        return primal_fn((*inputs, replay_loop))


class OneStep(KStep):
    """Differentiate a minimiser through its last iterate update only.

        ξ̂ = Φ(ξ̄, θ),   ξ̄ = sg(ξ_{T−1})

    `KStep` with k = 1 (Bolte, Pauwels and Vaiter, 2023): O(1) memory, as
    ``optx.ImplicitAdjoint``, without needing strict convergence, and exact
    at the fixed point of a Newton-like update. ``k`` other than 1 is
    rejected: ask for `KStep` with that ``k`` instead.

    Raises:
        ValidationError: If ``k`` is not 1.

    Examples:
        >>> from argminax.adjoint import KStep, OneStep
        >>> OneStep()
        OneStep()
        >>> isinstance(OneStep(), KStep)
        True
        >>> OneStep(k=2)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'k' of OneStep must be 1, got 2; use KStep(k=2)

    References:
        Bolte, J., Pauwels, E. & Vaiter, S. (2023). One-step differentiation
        of iterative algorithms. NeurIPS 36. arXiv:2305.13768.
    """

    k: int = eqx.field(static=True, default=1)

    def __check_init__(self) -> None:
        if self.k != 1:
            msg = f"'k' of OneStep must be 1, got {self.k!r}; use KStep(k={self.k!r})"
            raise ValidationError(msg)

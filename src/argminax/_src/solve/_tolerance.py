"""The convergence floor the solve loops share."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def floored_tolerance(tol: float, dtype: Any) -> float:
    """tol, floored at 64 machine epsilons: below that float32 never converges.

    Args:
        tol: The requested tolerance.
        dtype: The floating dtype the loop runs in.

    Returns:
        max(tol, 64 ε) as a host float, so the floor also works under jit.
    """
    # a host constant, so the floor also works under jit
    epsilon = float(jnp.finfo(dtype).eps)
    return max(tol, 64.0 * epsilon)

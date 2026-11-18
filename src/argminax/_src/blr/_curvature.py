"""The curvature estimates of the dense Bayesian learning rule transforms.

A curvature estimate is C ≈ ∇²_θ 𝒥, the curvature of the minibatch cost 𝒥 at
the mean, and is positive semidefinite: a name (``"ggn"`` or ``"identity"``)
or a callable of the mean and the gradient.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from argminax.exceptions import ValidationError


#: C = curvature(μ_θ, g): the (D_θ, D_θ) curvature of the cost, from the mean
#: and the gradient, both of shape (D_θ,).
type Curvature = Callable[
    [Float[Array, " state"], Float[Array, " state"]], Float[Array, "state state"]
]

_NAMES = ("ggn", "identity")


def _ggn(
    mean: Float[Array, " state"], grad: Float[Array, " state"]
) -> Float[Array, "state state"]:
    # C = g gᵀ
    return einx.dot("row, column -> row column", grad, grad)


def _identity(
    mean: Float[Array, " state"], grad: Float[Array, " state"]
) -> Float[Array, "state state"]:
    # C = 0
    size = grad.shape[0]
    return jnp.zeros((size, size), grad.dtype)


def resolve_curvature(curvature: Literal["ggn", "identity"] | Curvature) -> Curvature:
    """Turn a curvature name into its function, and pass a callable through.

    Args:
        curvature: ``"ggn"`` for the rank-one C = g gᵀ, ``"identity"`` for
            C = 0, or a callable ``curvature(mean, grad)`` returning C.

    Returns:
        A callable of the mean μ_θ and the gradient g, both (D_θ,), returning
        the (D_θ, D_θ) curvature C.

    Raises:
        ValidationError: If ``curvature`` is neither a known name nor a
            callable.
    """
    if callable(curvature):
        return curvature
    if curvature == "ggn":
        return _ggn
    if curvature == "identity":
        return _identity
    msg = f"'curvature' must be one of {_NAMES} or a callable, got {curvature!r}"
    raise ValidationError(msg)

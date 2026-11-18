"""The curvature estimates of the dense Bayesian learning rule transforms.

A curvature estimate is C ≈ ∇²_θ 𝒥, the curvature of the minibatch cost 𝒥 at
the mean, and is positive semidefinite: a name (``"ggn"`` or ``"identity"``)
or a callable of the mean and the gradient. The full-rank rule takes it as a
matrix, the low-rank rule as a factor G with C = G Gᵀ.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from argminax.exceptions import ValidationError


#: C = curvature(μ_θ, g): the (D_θ, D_θ) curvature of the cost, from the mean
#: and the gradient, both of shape (D_θ,).
type Curvature = Callable[
    [Float[Array, " state"], Float[Array, " state"]], Float[Array, "state state"]
]

#: G = factor(μ_θ, g): a (D_θ, K) factor of the curvature, C = G Gᵀ.
type CurvatureFactor = Callable[
    [Float[Array, " state"], Float[Array, " state"]], Float[Array, "state factor"]
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


def _ggn_factor(
    mean: Float[Array, " state"], grad: Float[Array, " state"]
) -> Float[Array, "state 1"]:
    # G = g, one column, so G Gᵀ = g gᵀ
    return einx.id("state -> state 1", grad)


def _identity_factor(
    mean: Float[Array, " state"], grad: Float[Array, " state"]
) -> Float[Array, "state 0"]:
    # G has no columns, so G Gᵀ = 0
    return jnp.zeros((grad.shape[0], 0), grad.dtype)


@jax.custom_jvp
def psd_factor(matrix: Float[Array, "state state"]) -> Float[Array, "state state"]:
    """Factor the positive semidefinite part of a symmetric matrix.

        C = V S Vᵀ,   G = V √max(S, 0),   G Gᵀ = V max(S, 0) Vᵀ

    V holds the eigenvectors of C and S its eigenvalues. G Gᵀ is the closest
    positive semidefinite matrix to C in the Frobenius norm: negative
    curvature is dropped, since it would make a precision indefinite.

    The derivative is that of G Gᵀ, which is what the precision is made of;
    G itself is only defined up to a rotation of its columns. JAX's own
    derivative of an eigendecomposition divides by the gaps between the
    eigenvalues, and that of the square root by √S, so it is NaN at a
    repeated or a non-positive eigenvalue. `_psd_factor_jvp` divides by
    neither.

    Args:
        matrix: The symmetric matrix C, shape (D_θ, D_θ).

    Returns:
        The factor G, shape (D_θ, D_θ).

    Examples:
        >>> import jax.numpy as jnp
        >>> from argminax._src.blr._curvature import psd_factor
        >>> indefinite = jnp.array([[2.0, 0.0], [0.0, -1.0]])
        >>> factor = psd_factor(indefinite)
        >>> bool(
        ...     jnp.allclose(factor @ factor.T, jnp.array([[2.0, 0.0], [0.0, 0.0]]))
        ... )
        True

        The gradient through it is finite at a repeated eigenvalue:

        >>> import jax
        >>> def trace_of_positive_part(scale):
        ...     factor = psd_factor(scale * jnp.eye(2))
        ...     return jnp.sum(factor * factor)
        >>> bool(jnp.allclose(jax.grad(trace_of_positive_part)(3.0), 2.0))
        True
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    # √max(S, 0)
    scales = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    # G = V √max(S, 0)
    return einx.multiply("state factor, factor -> state factor", eigenvectors, scales)


@psd_factor.defjvp
def _psd_factor_jvp(
    primals: tuple[Float[Array, "state state"]],
    tangents: tuple[Float[Array, "state state"]],
) -> tuple[Float[Array, "state state"], Float[Array, "state state"]]:
    """The derivative of `psd_factor`, as a dG with d(G Gᵀ) = dG Gᵀ + G dGᵀ.

        M = Vᵀ dC V
        d(G Gᵀ) = V (Φ ⊙ M) Vᵀ,   Φᵢⱼ = (sᵢ⁺ − sⱼ⁺) / (sᵢ − sⱼ)
        dG = V (Ψ ⊙ M),   Ψᵢⱼ = Φᵢⱼ √sⱼ⁺ / (sᵢ⁺ + sⱼ⁺)

    s⁺ = max(s, 0) is the positive part of the eigenvalues s. The first line
    is the derivative of a matrix function of a symmetric matrix (the
    Daleckii–Krein formula) for the positive part. Φ needs no division by a
    gap between two eigenvalues of one sign: it is 1 when both are positive
    and 0 when neither is, and only a positive and a non-positive eigenvalue,
    which cannot be equal, are divided. Ψ is the split of Φ between dG Gᵀ and
    its transpose, Ψᵢⱼ √sⱼ⁺ + √sᵢ⁺ Ψⱼᵢ = Φᵢⱼ, and is zero where sᵢ⁺ + sⱼ⁺ = 0.
    """
    (matrix,) = primals
    (matrix_dot,) = tangents
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    positive = jnp.maximum(eigenvalues, 0.0)
    roots = jnp.sqrt(positive)
    # G = V √max(S, 0)
    factor = einx.multiply("state factor, factor -> state factor", eigenvectors, roots)

    # M = Vᵀ dC V, from the symmetric part of dC, as eigh reads C
    half = einx.dot(
        "row left, row column, column right -> left right",
        eigenvectors,
        matrix_dot,
        eigenvectors,
    )
    rotated = 0.5 * (half + einx.id("left right -> right left", half))

    is_positive = eigenvalues > 0.0
    both = einx.logical_and("row, column -> row column", is_positive, is_positive)
    mixed = einx.not_equal("row, column -> row column", is_positive, is_positive)
    # sᵢ − sⱼ and sᵢ⁺ − sⱼ⁺ at [i, j]
    gaps = einx.subtract("row, column -> row column", eigenvalues, eigenvalues)
    positive_gaps = einx.subtract("row, column -> row column", positive, positive)
    # Φ: 1 for two positive eigenvalues, the divided difference across the
    # sign change (where the gap cannot be zero), 0 otherwise
    divided = positive_gaps / jnp.where(mixed, gaps, 1.0)
    weights = jnp.where(both, 1.0, jnp.where(mixed, divided, 0.0))
    # Ψᵢⱼ = Φᵢⱼ √sⱼ⁺ / (sᵢ⁺ + sⱼ⁺)
    sums = einx.add("row, column -> row column", positive, positive)
    has_weight = sums > 0.0
    shares = einx.multiply("row column, column -> row column", weights, roots)
    shares = jnp.where(has_weight, shares / jnp.where(has_weight, sums, 1.0), 0.0)
    # dG = V (Ψ ⊙ M)
    factor_dot = einx.dot(
        "state row, row column -> state column", eigenvectors, shares * rotated
    )
    return factor, factor_dot


def resolve_curvature_factor(
    curvature: Literal["ggn", "identity"] | Curvature,
) -> CurvatureFactor:
    """Turn a curvature name or callable into a function returning a factor.

    Args:
        curvature: ``"ggn"`` for G = g (one column), ``"identity"`` for a
            factor with no columns, or a callable ``curvature(mean, grad)``
            returning the dense C, which is factored by `psd_factor` at
            O(D_θ³) a call.

    Returns:
        A callable of the mean μ_θ and the gradient g, both (D_θ,), returning
        a (D_θ, K) factor G with C = G Gᵀ.

    Raises:
        ValidationError: If ``curvature`` is neither a known name nor a
            callable.
    """
    if callable(curvature):
        dense = curvature
        return lambda mean, grad: psd_factor(dense(mean, grad))
    if curvature == "ggn":
        return _ggn_factor
    if curvature == "identity":
        return _identity_factor
    msg = f"'curvature' must be one of {_NAMES} or a callable, got {curvature!r}"
    raise ValidationError(msg)

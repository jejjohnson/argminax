"""The Bayesian learning rule with a diagonal Gaussian, as an optax transform.

The transform carries q(θ) = 𝒩(μ_θ, diag(Λ_θ)⁻¹) in its state, in natural
parameters, and moves it one minibatch at a time. It works on a parameter
pytree of any size and needs no linear solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import gaussx as gx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree

from argminax._src.blr._optax import (
    StepSize,
    require_non_negative,
    require_optax,
    require_positive,
    require_step_size,
    step_size,
)
from argminax.exceptions import ValidationError


if TYPE_CHECKING:
    import optax


__all__ = [
    "BLRDiagonalState",
    "blr_diagonal",
]


_CURVATURES = ("ggn", "identity")


class BLRDiagonalState(NamedTuple):
    """The state of `blr_diagonal`: a diagonal Gaussian in natural parameters.

    The mean is μ_θ = (Λ_θ μ_θ) / Λ_θ, leaf by leaf, and the variance is
    1 / Λ_θ. A ``NamedTuple``, as optax states are.

    Attributes:
        precision: The precision Λ_θ of each parameter, a pytree shaped like
            the parameters.
        nat_mean: The natural mean Λ_θ ⊙ μ_θ, a pytree shaped like the
            parameters.
        count: The number of steps taken.
    """

    precision: PyTree[Float[Array, "*state"]]
    nat_mean: PyTree[Float[Array, "*state"]]
    count: Int[Array, ""]


def _curvature(
    grad: Float[Array, "*state"], kind: Literal["ggn", "identity"]
) -> Float[Array, "*state"]:
    """The diagonal curvature estimate C of the cost at one leaf."""
    if kind == "ggn":
        # C = g ⊙ g
        return grad * grad
    # C = 0
    return jnp.zeros_like(grad)


def blr_diagonal(
    learning_rate: StepSize = 1e-2,
    *,
    prior_precision: float = 1e-4,
    prior_mean: float = 0.0,
    curvature: Literal["ggn", "identity"] = "ggn",
    damping: float = 1e-6,
) -> optax.GradientTransformation:
    """The Bayesian learning rule with a diagonal Gaussian, as an optax transform.

        Λ_θ ← max((1 − α) Λ_θ + α (δ + C), β)
        Λ_θ μ_θ ← (1 − α) Λ_θ μ_θ + α (δ μ₀ − g + C ⊙ μ_θ)

    The transform learns q(θ) = 𝒩(μ_θ, diag(Λ_θ)⁻¹) over the parameters θ, a
    pytree of arrays. g = ∇_θ 𝒥 is the gradient of the minibatch cost 𝒥 (a
    negative log-likelihood), as any optax optimiser takes it; C is a diagonal
    estimate of its curvature ∇²_θ 𝒥; 𝒩(μ₀, δ⁻¹) is the prior on every
    parameter; α is the step size and β a floor on the precision. Every
    quantity has the shape of its parameter leaf. The update it returns is the
    change of the mean, so ``optax.apply_updates`` keeps the parameters equal
    to μ_θ, and the mean starts at the parameters given to ``init``.

    Where the precision has settled at Λ_θ = δ + C, the mean satisfies
    g + δ (μ_θ − μ₀) = 0: it is a stationary point of the cost plus the prior.

    Algorithm:
        1. Recover the mean μ_θ = (Λ_θ μ_θ) / Λ_θ from the state.
        2. Estimate the curvature: C = g ⊙ g (``"ggn"``) or C = 0
           (``"identity"``, which pins Λ_θ to δ).
        3. Take one damped step of `gaussx.blr_diag_update` on the log-joint,
           whose gradient is −g − δ (μ_θ − μ₀) and whose Hessian diagonal is
           −(C + δ).
        4. Floor the precision at β and return the change of the mean.

    Args:
        learning_rate: The step size α in (0, 1], or an optax schedule of the
            step count.
        prior_precision: The prior precision δ > 0, the same for every
            parameter.
        prior_mean: The prior mean μ₀, the same for every parameter.
        curvature: The curvature estimate C: ``"ggn"`` or ``"identity"``.
        damping: The floor β ≥ 0 on the precision.

    Returns:
        An ``optax.GradientTransformation`` whose state is a
        `BLRDiagonalState`.

    Raises:
        ValidationError: If ``curvature`` is not a known name, a constant
            ``learning_rate`` is outside (0, 1], ``prior_precision`` is not
            positive or ``damping`` is negative.
        MissingExtraError: If optax is not installed.

    Examples:
        A drop-in for ``optax.adam``, on a pytree of parameters:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import optax
        >>> from argminax.blr import blr_diagonal
        >>> params = {"w": jnp.ones((2, 3)), "b": jnp.zeros(2)}
        >>> cost = lambda p: jnp.sum(p["w"] ** 2) + jnp.sum((p["b"] - 1.0) ** 2)
        >>> optimiser = blr_diagonal(0.1, prior_precision=1.0)
        >>> state = optimiser.init(params)
        >>> updates, state = optimiser.update(jax.grad(cost)(params), state)
        >>> params = optax.apply_updates(params, updates)
        >>> state.precision["w"].shape
        (2, 3)

        With ``"identity"`` the precision stays at the prior and each step is
        a gradient step on the cost plus the prior, of size α / δ:

        >>> optimiser = blr_diagonal(0.5, prior_precision=2.0, curvature="identity")
        >>> theta = jnp.array([1.0])
        >>> state = optimiser.init(theta)
        >>> updates, state = optimiser.update(jnp.array([4.0]), state)
        >>> bool(jnp.allclose(updates, -0.5 * (4.0 + 2.0 * 1.0) / 2.0))
        True
        >>> bool(jnp.allclose(state.precision, 2.0))
        True

        An unknown curvature name is rejected when the transform is built:

        >>> blr_diagonal(
        ...     curvature="hutchinson"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'curvature' must be one of ('ggn', 'identity'), ...

    References:
        Khan and Rue (2023), "The Bayesian Learning Rule", arXiv:2107.04562.
    """
    if curvature not in _CURVATURES:
        msg = f"'curvature' must be one of {_CURVATURES}, got {curvature!r}"
        raise ValidationError(msg)
    require_step_size(learning_rate)
    require_positive(prior_precision, "prior_precision")
    require_non_negative(damping, "damping")
    optax = require_optax()

    def init_fn(params: PyTree[Float[Array, "*state"]]) -> BLRDiagonalState:
        # Λ_θ = δ
        precision = jax.tree.map(lambda p: jnp.full_like(p, prior_precision), params)
        # Λ_θ μ_θ = δ θ, so the mean starts at the caller's parameters
        nat_mean = jax.tree.map(lambda p: prior_precision * p, params)
        return BLRDiagonalState(precision, nat_mean, jnp.zeros([], jnp.int32))

    def update_leaf(
        nat_mean: Float[Array, "*state"],
        precision: Float[Array, "*state"],
        grad: Float[Array, "*state"],
        alpha: Float[Array, ""] | float,
    ) -> tuple[Float[Array, "*state"], Float[Array, "*state"], Float[Array, "*state"]]:
        # μ_θ = (Λ_θ μ_θ) / Λ_θ
        mean = nat_mean / precision
        estimate = _curvature(grad, curvature)
        # ∇ log p(𝒟, θ) = −g − δ (μ_θ − μ₀)
        joint_grad = -grad - prior_precision * (mean - prior_mean)
        # diag ∇² log p(𝒟, θ) = −(C + δ)
        joint_hessian = -(estimate + prior_precision)
        # Λ_θ μ_θ ← (1 − α) Λ_θ μ_θ + α (δ μ₀ − g + C ⊙ μ_θ)
        # Λ_θ ← (1 − α) Λ_θ + α (δ + C)
        new_nat_mean, new_precision = gx.blr_diag_update(
            nat_mean,
            precision,
            joint_grad,
            joint_hessian,
            alpha,
            convention="precision",
        )
        # Λ_θ ← max(Λ_θ, β)
        new_precision = jnp.maximum(new_precision, damping)
        # Δθ = μ_θ' − μ_θ
        update = new_nat_mean / new_precision - mean
        return update, new_precision, new_nat_mean

    def update_fn(
        grads: PyTree[Float[Array, "*state"]],
        state: BLRDiagonalState,
        params: Any = None,
    ) -> tuple[PyTree[Float[Array, "*state"]], BLRDiagonalState]:
        alpha = step_size(learning_rate, state.count)
        stepped = jax.tree.map(
            lambda eta, lam, g: update_leaf(eta, lam, g, alpha),
            state.nat_mean,
            state.precision,
            grads,
        )
        # one (update, precision, natural mean) triple per leaf, split by field
        updates, precision, nat_mean = jax.tree.transpose(
            jax.tree.structure(grads), jax.tree.structure((0, 0, 0)), stepped
        )
        return updates, BLRDiagonalState(precision, nat_mean, state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)

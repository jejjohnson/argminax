"""IVON, the improved variational online Newton optimiser, as an optax transform.

A diagonal Bayesian learning rule variant made for deep networks: momentum on
the gradient, a running curvature estimate, weight decay as the prior and a
clipped step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree

from argminax._src.blr._optax import (
    StepSize,
    require_optax,
    require_positive,
    require_rate,
    step_size,
)


if TYPE_CHECKING:
    import optax


__all__ = [
    "IVONState",
    "ivon",
]


class IVONState(NamedTuple):
    """The state of `ivon`.

    The Gaussian it describes has the current parameters as its mean and
    1 / (h + λ) as its variance, with λ the weight decay. A ``NamedTuple``, as
    optax states are.

    Attributes:
        momentum: The running mean m of the gradients, a pytree shaped like
            the parameters.
        hess: The running curvature estimate h, a pytree shaped like the
            parameters.
        count: The number of steps taken.
    """

    momentum: PyTree[Float[Array, "*state"]]
    hess: PyTree[Float[Array, "*state"]]
    count: Int[Array, ""]


def ivon(
    learning_rate: StepSize = 1e-3,
    *,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    weight_decay: float = 1e-4,
    hess_init: float = 1.0,
    clip_radius: float = float("inf"),
    ess: float = 1.0,
) -> optax.GradientTransformation:
    """IVON, a diagonal Bayesian learning rule for deep networks.

        m ← β₁ m + (1 − β₁) g
        h ← β₂ h + (1 − β₂) N g ⊙ g
        Δθ = −α clip((m / (1 − β₁ᵗ) + λ θ) / (h + λ), ±r)

    g = ∇_θ 𝒥 is the gradient of the minibatch cost 𝒥, m its running mean and
    h a running estimate of the curvature, each shaped like its parameter
    leaf; β₁ and β₂ are the two averaging rates, t the step count, λ the
    weight decay (the prior precision), N the effective sample size, α the
    step size and r the clipping radius. The Gaussian it carries is
    𝒩(θ, diag(h + λ)⁻¹), around the current parameters θ.

    Algorithm:
        1. Average the gradient into the momentum m.
        2. Average N g ⊙ g into the curvature estimate h.
        3. Correct the momentum for its zero start, add the weight decay
           λ θ, divide by h + λ and clip to ±r.
        4. Scale by −α.

    Args:
        learning_rate: The step size α, or an optax schedule of the step
            count.
        beta1: The averaging rate β₁ of the momentum, in [0, 1).
        beta2: The averaging rate β₂ of the curvature estimate, in [0, 1).
        weight_decay: The weight decay λ > 0, which is also the prior
            precision.
        hess_init: The starting value of h.
        clip_radius: The radius r of the clip on each parameter's step.
        ess: The effective sample size N; the size of the training set for a
            calibrated variance, 1 to use IVON as an optimiser only.

    Returns:
        An ``optax.GradientTransformation`` whose state is an `IVONState`.
        Pass ``params`` to its ``update``: without them the weight decay term
        λ θ is left out of the step.

    Raises:
        ValidationError: If ``weight_decay`` is not positive, or ``beta1`` or
            ``beta2`` is outside [0, 1).
        MissingExtraError: If optax is not installed.

    Examples:
        One step from a zero start: the momentum correction makes the first
        step the gradient itself, here divided by h + λ.

        >>> import jax.numpy as jnp
        >>> from argminax.blr import ivon
        >>> optimiser = ivon(0.1, beta2=0.5, weight_decay=1.0, hess_init=1.0)
        >>> theta = jnp.array([0.0])
        >>> state = optimiser.init(theta)
        >>> grads = jnp.array([2.0])
        >>> updates, state = optimiser.update(grads, state, theta)
        >>> hess = 0.5 * 1.0 + 0.5 * 2.0**2
        >>> bool(jnp.allclose(state.hess, hess))
        True
        >>> bool(jnp.allclose(updates, -0.1 * 2.0 / (hess + 1.0)))
        True

        The clip bounds every parameter's step by α r:

        >>> optimiser = ivon(0.1, weight_decay=1.0, clip_radius=0.5)
        >>> state = optimiser.init(theta)
        >>> updates, state = optimiser.update(jnp.array([100.0]), state, theta)
        >>> bool(jnp.allclose(updates, -0.05))
        True

    References:
        Shen et al. (2024), "Variational Learning is Effective for Large Deep
        Networks", arXiv:2402.17641.
    """
    require_positive(weight_decay, "weight_decay")
    require_rate(beta1, "beta1")
    require_rate(beta2, "beta2")
    optax = require_optax()

    def init_fn(params: PyTree[Float[Array, "*state"]]) -> IVONState:
        momentum = jax.tree.map(jnp.zeros_like, params)
        hess = jax.tree.map(lambda p: jnp.full_like(p, hess_init), params)
        return IVONState(momentum, hess, jnp.zeros([], jnp.int32))

    def update_fn(
        grads: PyTree[Float[Array, "*state"]],
        state: IVONState,
        params: Any = None,
    ) -> tuple[PyTree[Float[Array, "*state"]], IVONState]:
        alpha = step_size(learning_rate, state.count)
        count = state.count + 1
        # m ← β₁ m + (1 − β₁) g
        momentum = jax.tree.map(
            lambda m, g: beta1 * m + (1 - beta1) * g, state.momentum, grads
        )
        # h ← β₂ h + (1 − β₂) N g ⊙ g
        hess = jax.tree.map(
            lambda h, g: beta2 * h + (1 - beta2) * ess * g * g, state.hess, grads
        )
        # 1 − β₁ᵗ, the correction for the momentum's zero start
        debias = 1.0 - beta1**count
        # the weight decay term λ θ, left out when the parameters are not given
        decay = (
            jax.tree.map(jnp.zeros_like, grads)
            if params is None
            else jax.tree.map(lambda p: weight_decay * p, params)
        )

        def step(
            m: Float[Array, "*state"],
            h: Float[Array, "*state"],
            d: Float[Array, "*state"],
        ) -> Float[Array, "*state"]:
            # (m / (1 − β₁ᵗ) + λ θ) / (h + λ)
            direction = (m / debias + d) / (h + weight_decay)
            # Δθ = −α clip(·, ±r)
            return -alpha * jnp.clip(direction, -clip_radius, clip_radius)

        updates = jax.tree.map(step, momentum, hess, decay)
        return updates, IVONState(momentum, hess, count)

    return optax.GradientTransformation(init_fn, update_fn)

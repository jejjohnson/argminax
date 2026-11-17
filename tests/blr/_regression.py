"""The regression problem behind the optax_bayes reference numbers, and its loop.

``data/optax_bayes.npz`` holds optax_bayes's iterates on this problem (see
``data/README.md``); `Regression` rebuilds it from the stored inputs, so a port
test runs argminax on exactly what optax_bayes saw.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax


NUM_STEPS = 50


class Regression:
    """𝒥(θ) = ½ ‖Xθ − y‖² with 30 observations and 4 weights."""

    def __init__(self, reference) -> None:
        self.inputs = jnp.asarray(reference["inputs"])
        self.targets = jnp.asarray(reference["targets"])
        self.start = jnp.asarray(reference["start"])
        self.prior_mean = jnp.asarray(reference["prior_mean"])

    def cost(self, theta):
        # 𝒥(θ) = ½ ‖Xθ − y‖²
        return 0.5 * jnp.sum((self.inputs @ theta - self.targets) ** 2)

    def curvature(self, mean, grads):
        # C = ∇²𝒥 = XᵀX
        return self.inputs.T @ self.inputs

    @staticmethod
    def tree(theta):
        return {"w": theta[:2].reshape(2, 1), "b": theta[2:]}

    @staticmethod
    def flat(params):
        return jnp.concatenate([params["w"].reshape(-1), params["b"]])

    def tree_cost(self, params):
        return self.cost(self.flat(params))


def run(optimiser, params, cost, to_flat=lambda p: p, *, pass_params=False):
    """Step ``optimiser`` NUM_STEPS times; return the flat iterates and the state."""
    state = optimiser.init(params)
    grad = jax.grad(cost)
    iterates = []
    for _ in range(NUM_STEPS):
        given = params if pass_params else None
        updates, state = optimiser.update(grad(params), state, given)
        params = optax.apply_updates(params, updates)
        iterates.append(np.asarray(to_flat(params)))
    return np.stack(iterates), state


#: optax_bayes's iterates and posteriors, and the problem they were recorded on
REFERENCE = np.load(Path(__file__).parent / "data" / "optax_bayes.npz")
REGRESSION = Regression(REFERENCE)

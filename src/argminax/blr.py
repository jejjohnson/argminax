"""The Bayesian learning rule, as optax transforms that carry a Gaussian.

A transform here is an ``optax.GradientTransformation``: it takes the gradient
g = ∇_θ 𝒥 of a minibatch cost, as ``optax.adam`` does, and keeps a Gaussian
q(θ) = 𝒩(μ_θ, Λ_θ⁻¹) over the parameters θ in its state, so the uncertainty is
there during training and not only at a converged minimum. The caller steps
it; unlike the solvers of `argminax.solve` it never runs to a tolerance.
optax comes with the ``argminax[optax]`` extra.

| Name | What it gives you |
|---|---|
| `blr_diagonal` | a diagonal Gaussian over a parameter pytree, a drop-in for Adam |
| `ivon` | IVON: the diagonal rule with momentum, weight decay and a clipped step |
| `BLRDiagonalState`, `IVONState` | the optimiser states that hold the Gaussian |
"""

from __future__ import annotations

from argminax._src.blr import BLRDiagonalState, IVONState, blr_diagonal, ivon


__all__ = [
    "BLRDiagonalState",
    "IVONState",
    "blr_diagonal",
    "ivon",
]

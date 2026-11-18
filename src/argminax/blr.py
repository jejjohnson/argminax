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
| `blr_full_rank` | a full-rank Gaussian over a flat vector, ending at Laplace |
| `blr_low_rank` | a diagonal-plus-rank-R precision, O(D_θ R²) a step (SLANG) |
| `newton` | damped Newton's method, the full-rank rule with step size 1 |
| `ivon` | IVON: the diagonal rule with momentum, weight decay and a clipped step |
| `BLRDiagonalState`, `BLRFullRankState` | the states that hold the Gaussian |
| `BLRLowRankState` | the low-rank state: the diagonal d and the factor U |
| `IVONState` | IVON's state: its momentum and curvature estimate |
"""

from __future__ import annotations

from argminax._src.blr import (
    BLRDiagonalState,
    BLRFullRankState,
    BLRLowRankState,
    IVONState,
    blr_diagonal,
    blr_full_rank,
    blr_low_rank,
    ivon,
    newton,
)


__all__ = [
    "BLRDiagonalState",
    "BLRFullRankState",
    "BLRLowRankState",
    "IVONState",
    "blr_diagonal",
    "blr_full_rank",
    "blr_low_rank",
    "ivon",
    "newton",
]

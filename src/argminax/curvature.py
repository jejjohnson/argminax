"""Curvature at a minimum, as gaussx and lineax operators.

The Gaussian 𝒩(x̂, Σ_x∣y) at a minimum x̂ ∈ ℝ^D_x is the same object whether it
is called the inverse Gauss–Newton Hessian of 4D-Var, a Laplace approximation
or a retrieval's posterior covariance. `Laplace` holds it with its posterior
precision Λ_x∣y as a PSD-tagged operator, and Σ_x∣y = Λ_x∣y⁻¹ is never formed.

| Name | What it gives you |
|---|---|
| `Laplace` | 𝒩(x̂, Λ_x∣y⁻¹): covariance, log-determinant, samples |
| `gauss_newton` | the Laplace posterior from whitened residuals, Λ_x∣y = J_rᵀ J_r |
| `push_forward` | a whitened posterior mapped back to physical coordinates |
| `marginal_variance` | diag(Σ_x∣y), exact or probed with a standard error |
| `low_rank` | Σ_x∣y as the identity less a low-rank update, for whitened posteriors |
| `sample_by_perturbation` | draws by perturbing the residuals, with no square root |
"""

from __future__ import annotations

from argminax._src.curvature import (
    Laplace,
    VarianceEstimate,
    gauss_newton,
    low_rank,
    marginal_variance,
    push_forward,
    sample_by_perturbation,
)


__all__ = [
    "Laplace",
    "VarianceEstimate",
    "gauss_newton",
    "low_rank",
    "marginal_variance",
    "push_forward",
    "sample_by_perturbation",
]

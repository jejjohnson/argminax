"""Curvature at a minimum, as gaussx and lineax operators.

The Gaussian N(x̂, Ŝ) at a minimum is the same object whether it is called the
inverse Gauss–Newton Hessian of 4D-Var, a Laplace approximation or a
retrieval's posterior covariance. `Laplace` holds it with its precision Ĥ as a
PSD-tagged operator, and Ŝ = Ĥ⁻¹ is never formed.

| Name | What it gives you |
|---|---|
| `Laplace` | N(x̂, Ĥ⁻¹): covariance, log-determinant, samples |
| `gauss_newton` | the Laplace posterior from whitened residuals, Ĥ = J_rᵀ J_r |
| `push_forward` | a whitened posterior mapped back to physical coordinates |
"""

from __future__ import annotations

from argminax._src.curvature import Laplace, gauss_newton, push_forward


__all__ = [
    "Laplace",
    "gauss_newton",
    "push_forward",
]

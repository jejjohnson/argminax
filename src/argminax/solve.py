"""Solvers optimistix lacks, for retrievals and data assimilation.

Every solver here is an optimistix solver, so ``optx.minimise`` and
``optx.least_squares`` keep working, and its curvature at the end is
`argminax.curvature`'s.

| Name | What it gives you |
|---|---|
| `GaussNewtonCG` | Gauss–Newton and Levenberg–Marquardt with a matrix-free inner CG |
| `dual_solve` | the observation-space (PSAS) increment, CG on (H B Hᵀ + R) λ = d |
| `incremental_dual_solve` | its outer loop for a nonlinear forward model |
"""

from __future__ import annotations

from argminax._src.solve import (
    DualSolution,
    GaussNewtonCG,
    dual_solve,
    incremental_dual_solve,
)


__all__ = [
    "DualSolution",
    "GaussNewtonCG",
    "dual_solve",
    "incremental_dual_solve",
]

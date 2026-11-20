"""Solvers optimistix lacks, for retrievals and data assimilation.

Every solver here is an optimistix solver, so ``optx.minimise`` and
``optx.least_squares`` keep working, and its curvature at the end is
`argminax.curvature`'s.

| Name | What it gives you |
|---|---|
| `GaussNewtonCG` | Gauss–Newton and Levenberg–Marquardt with a matrix-free inner CG |
| `dual_solve` | the observation-space (PSAS) increment, CG on (H Σ_x Hᵀ + Σ_ε) w = r |
| `incremental_dual_solve` | its outer loop for a nonlinear forward model |
| `Reparametrised` | bounded states x = g(u), solved in an unconstrained control u |
| `laplace_mode` | the damped Newton loop to the mode of a GMRF model (precision form) |
| `laplace_mode_covariance` | the same for a GP with prior covariance K_XX |
| `laplace_mode_sites` | the same given a posterior-mean map of sites (Markov GPs) |
| `TolerantLinearSolver` | a lineax solver whose failed solves return values |
"""

from __future__ import annotations

from argminax._src.solve import (
    CovarianceLaplaceResult,
    DualSolution,
    GaussNewtonCG,
    LaplaceModeResult,
    Reparametrised,
    SiteLaplaceResult,
    TolerantLinearSolver,
    dual_solve,
    incremental_dual_solve,
    laplace_mode,
    laplace_mode_covariance,
    laplace_mode_sites,
)


__all__ = [
    "CovarianceLaplaceResult",
    "DualSolution",
    "GaussNewtonCG",
    "LaplaceModeResult",
    "Reparametrised",
    "SiteLaplaceResult",
    "TolerantLinearSolver",
    "dual_solve",
    "incremental_dual_solve",
    "laplace_mode",
    "laplace_mode_covariance",
    "laplace_mode_sites",
]

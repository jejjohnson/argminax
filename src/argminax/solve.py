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
| `site_fixed_point` | the differentiable loop over Gaussian sites (EP, PL) |
| `TolerantLinearSolver` | a lineax solver whose failed solves return values |

`laplace_mode_covariance` and `laplace_mode_sites` differentiate their mode
implicitly through ``adjoint=``, an ``optx.ImplicitAdjoint``. The default is
``argminax.adjoint.safe_implicit_adjoint(lx.GMRES(rtol=1e-8, atol=1e-8,
restart=20, max_steps=50))``, which forms no N × N matrix and never raises;
pass another ``ImplicitAdjoint`` to change the solver or its tolerances.
`laplace_mode_covariance` also takes a gaussx solver strategy as ``solver=``
for every solve against K_XX + Λ_site⁻¹; with an iterative one
(``gaussx.CGSolver``, ``gaussx.BBMMSolver``) and a matrix-free K_XX, no N × N
matrix is formed.
"""

from __future__ import annotations

from argminax._src.solve import (
    CovarianceLaplaceResult,
    DualSolution,
    GaussNewtonCG,
    LaplaceModeResult,
    Reparametrised,
    SiteFixedPoint,
    SiteLaplaceResult,
    TolerantLinearSolver,
    dual_solve,
    incremental_dual_solve,
    laplace_mode,
    laplace_mode_covariance,
    laplace_mode_sites,
    site_fixed_point,
)


__all__ = [
    "CovarianceLaplaceResult",
    "DualSolution",
    "GaussNewtonCG",
    "LaplaceModeResult",
    "Reparametrised",
    "SiteFixedPoint",
    "SiteLaplaceResult",
    "TolerantLinearSolver",
    "dual_solve",
    "incremental_dual_solve",
    "laplace_mode",
    "laplace_mode_covariance",
    "laplace_mode_sites",
    "site_fixed_point",
]

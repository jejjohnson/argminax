"""Solvers optimistix lacks, for retrievals and data assimilation.

Every solver here is an optimistix solver, so ``optx.minimise`` and
``optx.least_squares`` keep working, and its curvature at the end is
`argminax.curvature`'s.

| Name | What it gives you |
|---|---|
| `GaussNewtonCG` | Gauss–Newton and Levenberg–Marquardt with a matrix-free inner CG |
"""

from __future__ import annotations

from argminax._src.solve import GaussNewtonCG


__all__ = [
    "GaussNewtonCG",
]

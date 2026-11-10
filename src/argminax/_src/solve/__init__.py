"""Solvers optimistix lacks, each an optimistix solver or a function built on one."""

from __future__ import annotations

from argminax._src.solve._dual import DualSolution, dual_solve, incremental_dual_solve
from argminax._src.solve._gauss_newton import GaussNewtonCG
from argminax._src.solve._laplace_mode import (
    CovarianceLaplaceResult,
    LaplaceModeResult,
    SiteLaplaceResult,
    laplace_mode,
    laplace_mode_covariance,
    laplace_mode_sites,
)
from argminax._src.solve._reparam import Reparametrised


__all__ = [
    "CovarianceLaplaceResult",
    "DualSolution",
    "GaussNewtonCG",
    "LaplaceModeResult",
    "Reparametrised",
    "SiteLaplaceResult",
    "dual_solve",
    "incremental_dual_solve",
    "laplace_mode",
    "laplace_mode_covariance",
    "laplace_mode_sites",
]

"""Solvers optimistix lacks, each an optimistix solver or a function built on one."""

from __future__ import annotations

from argminax._src.solve._dual import DualSolution, dual_solve, incremental_dual_solve
from argminax._src.solve._gauss_newton import GaussNewtonCG


__all__ = [
    "DualSolution",
    "GaussNewtonCG",
    "dual_solve",
    "incremental_dual_solve",
]

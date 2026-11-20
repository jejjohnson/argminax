"""Adjoints of a minimiser: how its solution moves with its arguments."""

from __future__ import annotations

from argminax._src.adjoint._kstep import KStep, OneStep
from argminax._src.adjoint._mapping import to_optimistix_adjoint
from argminax._src.adjoint._safe_implicit import safe_implicit_adjoint


__all__ = [
    "KStep",
    "OneStep",
    "safe_implicit_adjoint",
    "to_optimistix_adjoint",
]

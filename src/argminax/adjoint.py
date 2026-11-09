"""Adjoints of a minimiser stopped early.

At a converged minimum, ``optx.ImplicitAdjoint`` already differentiates the
solution by the implicit function theorem; argminax adds the adjoints for a
solver that stops before it converges, as learned 4D-Var does. Each is an
``optx.AbstractAdjoint``, passed as ``adjoint=`` to ``optx.minimise`` or
``optx.least_squares``.

| Name | What it differentiates |
|---|---|
| `KStep` | the last k iterate updates |
| `OneStep` | the last iterate update (k = 1) |

`to_optimistix_adjoint` turns a pipekit-cycle adjoint spec into the optimistix
adjoint that implements it.
"""

from __future__ import annotations

from argminax._src.adjoint import KStep, OneStep, to_optimistix_adjoint


__all__ = [
    "KStep",
    "OneStep",
    "to_optimistix_adjoint",
]

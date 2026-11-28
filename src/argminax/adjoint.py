"""Adjoints of a minimiser stopped early.

At a converged minimum, ``optx.ImplicitAdjoint`` already differentiates the
solution by the implicit function theorem; argminax adds the adjoints for a
solver that stops before it converges, as learned 4D-Var does. Each is an
``optx.AbstractAdjoint``, passed as ``adjoint=`` to ``optx.minimise`` or
``optx.least_squares``. `safe_implicit_adjoint` is ``optx.ImplicitAdjoint``
for a fixed point or minimum differentiated inside NUTS or SVI, where a failed
linear solve must become a value the sampler rejects rather than an exception.
With GMRES it is the default adjoint of `argminax.solve.laplace_mode_covariance`
and `argminax.solve.laplace_mode_sites`.

| Name | What it differentiates |
|---|---|
| `KStep` | the last k iterate updates |
| `OneStep` | the last iterate update (k = 1) |
| `safe_implicit_adjoint` | the converged solution, with a solve that never raises |

`to_optimistix_adjoint` turns a pipekit-cycle adjoint spec into the optimistix
adjoint that implements it.
"""

from __future__ import annotations

from argminax._src.adjoint import (
    KStep,
    OneStep,
    safe_implicit_adjoint,
    to_optimistix_adjoint,
)


__all__ = [
    "KStep",
    "OneStep",
    "safe_implicit_adjoint",
    "to_optimistix_adjoint",
]

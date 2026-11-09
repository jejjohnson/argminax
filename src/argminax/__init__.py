"""Optimisation with uncertainty at the minimum, for JAX.

argminax finds x̂ = argmin J(x) and then says how sharp that minimum is and how
x̂ depends on what went in. It extends optimistix (its solvers and adjoints are
optimistix solvers and adjoints, so ``optx.minimise`` and
``optx.least_squares`` keep working) and builds its curvature on gaussx
operators. Four public modules arrive with the first feature that needs each:

| Module | What it gives you |
|---|---|
| `argminax.adjoint` | k-step and one-step adjoints for early stopping |
| `argminax.curvature` | Gauss–Newton curvature and Ŝ as gaussx operators |
| `argminax.diagnose` | averaging kernel, degrees of freedom, evidence |
| `argminax.solve` | Gauss–Newton with inner CG, dual solve, Laplace mode |

Examples:
    >>> import argminax
    >>> argminax.__version__
    '0.0.1'
"""

from __future__ import annotations

from argminax.exceptions import ArgminaxError, MissingExtraError, ValidationError


__version__ = "0.0.2"

__all__ = [
    "ArgminaxError",
    "MissingExtraError",
    "ValidationError",
    "__version__",
]

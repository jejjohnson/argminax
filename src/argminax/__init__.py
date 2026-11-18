"""Optimisation with uncertainty at the minimum, for JAX.

argminax finds x̂ = argmin 𝒥(x) and then says how sharp that minimum is and how
x̂ depends on what went in. It extends optimistix (its solvers and adjoints are
optimistix solvers and adjoints, so ``optx.minimise`` and
``optx.least_squares`` keep working) and builds its curvature on gaussx
operators. `argminax.blr` extends optax the same way, with transforms that
carry the Gaussian during training. Each public module covers one area:

| Module | What it gives you |
|---|---|
| `argminax.adjoint` | k-step and one-step adjoints for early stopping |
| `argminax.blr` | the Bayesian learning rule as optax transforms |
| `argminax.curvature` | Gauss–Newton curvature and the posterior covariance |
| `argminax.diagnose` | averaging kernel, degrees of freedom, evidence |
| `argminax.solve` | Gauss–Newton with inner CG, dual solve, Laplace mode |

Examples:
    >>> import argminax
    >>> isinstance(argminax.__version__, str)
    True
"""

from __future__ import annotations

from argminax.exceptions import ArgminaxError, MissingExtraError, ValidationError


__version__ = "0.0.7"

__all__ = [
    "ArgminaxError",
    "MissingExtraError",
    "ValidationError",
    "__version__",
]

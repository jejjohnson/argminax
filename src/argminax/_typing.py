"""Shared type aliases.

The aliases are written with [PEP 695](https://peps.python.org/pep-0695/)
``type`` statements (Python 3.12+). They follow optimistix's calling
convention, ``fn(y, args)``, so a function written for ``optx.minimise`` or
``optx.least_squares`` is already one of these.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jaxtyping import Array, Float, PyTree


__all__ = [
    "Objective",
    "Residuals",
]


type Objective = Callable[[PyTree[Array], Any], Float[Array, ""]]
"""A scalar cost J(x, args), the ``fn`` of ``optx.minimise``."""

type Residuals = Callable[[PyTree[Array], Any], PyTree[Array]]
"""Residuals r(x, args) with J = ½ ‖r‖², the ``fn`` of ``optx.least_squares``."""

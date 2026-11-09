"""Map pipekit-cycle adjoint specs onto the optimistix adjoints that implement them.

pipekit-cycle sits above argminax, so its specs are matched by class name and
read by attribute, never imported.
"""

from __future__ import annotations

from typing import Any

import optimistix as optx

from argminax._src.adjoint._kstep import KStep
from argminax.exceptions import ValidationError


__all__ = [
    "to_optimistix_adjoint",
]


def to_optimistix_adjoint(spec: Any) -> optx.AbstractAdjoint:
    """Return the optimistix adjoint that implements an adjoint spec.

    Specs are matched by class name and their fields read with ``getattr``,
    so pipekit-cycle's ``pipekit_cycle.adjoints`` and any identically named
    classes are accepted, and argminax never imports pipekit-cycle.

    | Spec | Adjoint |
    |---|---|
    | an ``optx.AbstractAdjoint`` | itself |
    | ``ImplicitAdjoint()`` | ``optx.ImplicitAdjoint()`` |
    | ``TruncatedAdjoint(k)`` | `KStep` with that ``k`` |
    | ``RecursiveCheckpointAdjoint(n)`` | ``optx.RecursiveCheckpointAdjoint(n)`` |
    | ``DirectAdjoint``, ``BacksolveAdjoint`` | ``ValidationError`` (integrix's) |

    The implicit adjoint solves with the exact Hessian of the cost, which is
    positive definite only at a strict local minimum, so its default linear
    solver is kept (LU unless the solve's tags say more) rather than CG, and
    argminax adds no implicit adjoint of its own. Check ``solution.result``
    before trusting its gradient.

    Args:
        spec: An adjoint spec, or an optimistix adjoint to pass through.

    Returns:
        The optimistix adjoint for the spec.

    Raises:
        ValidationError: For ``DirectAdjoint`` and ``BacksolveAdjoint``, which
            have no meaning for a minimiser, or for an unrecognised spec.

    Examples:
        >>> import dataclasses
        >>> import optimistix as optx
        >>> from argminax.adjoint import to_optimistix_adjoint
        >>> @dataclasses.dataclass
        ... class TruncatedAdjoint:
        ...     k: int = 1
        >>> to_optimistix_adjoint(TruncatedAdjoint(k=3))
        KStep(k=3)

        An optimistix adjoint passes through unchanged.

        >>> adjoint = optx.ImplicitAdjoint()
        >>> to_optimistix_adjoint(adjoint) is adjoint
        True

        An adjoint through time is rejected.

        >>> class BacksolveAdjoint:
        ...     pass
        >>> to_optimistix_adjoint(
        ...     BacksolveAdjoint()
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: BacksolveAdjoint is an adjoint through time, ...
    """
    if isinstance(spec, optx.AbstractAdjoint):
        return spec
    # by name, so pipekit-cycle's specs match without importing it
    name = type(spec).__name__
    if name == "ImplicitAdjoint":
        return optx.ImplicitAdjoint()
    if name == "TruncatedAdjoint":
        return KStep(k=getattr(spec, "k", 1))
    if name == "RecursiveCheckpointAdjoint":
        return optx.RecursiveCheckpointAdjoint(
            checkpoints=getattr(spec, "checkpoints", None)
        )
    if name in ("DirectAdjoint", "BacksolveAdjoint"):
        msg = (
            f"{name} is an adjoint through time, with no meaning for a "
            "minimiser: use integrix, which owns adjoints of a march "
            "(to_diffrax_adjoint). For a minimiser, the unrolled default is "
            "optx.RecursiveCheckpointAdjoint()."
        )
        raise ValidationError(msg)
    msg = f"'spec' is not a recognised adjoint spec, got {spec!r}"
    raise ValidationError(msg)

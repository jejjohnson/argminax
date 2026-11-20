"""An implicit adjoint whose linear solve never raises, safe inside a sampler."""

from __future__ import annotations

import lineax as lx
import optimistix as optx

from argminax._src.solve._tolerant import TolerantLinearSolver
from argminax.exceptions import ValidationError


__all__ = [
    "safe_implicit_adjoint",
]


def safe_implicit_adjoint(
    linear_solver: lx.AbstractLinearSolver | None = None,
) -> optx.ImplicitAdjoint:
    """``optx.ImplicitAdjoint`` whose linear solve never raises; safe inside NUTS.

        (I − ∂ₛT)ᵀ v = ∇ₛL,   ∇_θ L = (∂_θ T)ᵀ v

    At a fixed point s* = T(s*, θ), with s* ∈ ℝ^D_s the state and θ the
    parameters, the implicit function theorem gives the gradient of a loss
    L(s*) through one linear solve for v ∈ ℝ^D_s; at a minimum the Hessian of
    the cost takes the place of I − ∂ₛT. lineax raises when that solve fails,
    which ends a NUTS chain at the first such proposal. Here the solve goes
    through `argminax.solve.TolerantLinearSolver`: a solve capped at
    ``max_steps`` or stagnant keeps its last iterate, and a non-finite system
    (a forward fit that overflowed at an extreme θ) gives v = 0, so the log
    density stays non-finite and the sampler rejects the point. Every other
    failure is reported as the inner solver reports it.

    Args:
        linear_solver: The lineax solver for v. ``None`` uses
            ``lx.AutoLinearSolver(well_posed=None)``, as
            ``optx.ImplicitAdjoint`` does; a matrix-free fixed point, whose
            I − ∂ₛT is not symmetric, wants ``lx.GMRES``.

    Returns:
        An ``optx.ImplicitAdjoint``, passed as ``adjoint=`` to
        ``optx.fixed_point``, ``optx.root_find``, ``optx.minimise`` or
        ``optx.least_squares``.

    Raises:
        ValidationError: If ``linear_solver`` is not a lineax solver.

    Examples:
        The gradient of the fixed point of s = cos(θ s) with respect to θ:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import optimistix as optx
        >>> import argminax
        >>> adjoint = argminax.adjoint.safe_implicit_adjoint(
        ...     lx.GMRES(rtol=1e-8, atol=1e-8, max_steps=20)
        ... )
        >>> def fixed_point(theta):
        ...     step = lambda s, theta: jnp.cos(theta * s)
        ...     solver = optx.FixedPointIteration(rtol=1e-8, atol=1e-8)
        ...     solution = optx.fixed_point(
        ...         step, solver, jnp.zeros(2), theta, adjoint=adjoint
        ...     )
        ...     return solution.value.sum()
        >>> jax.grad(fixed_point)(jnp.array(0.5)).shape
        ()

        The default solver is optimistix's:

        >>> default = argminax.adjoint.safe_implicit_adjoint()
        >>> type(default.linear_solver.inner).__name__
        'AutoLinearSolver'
    """
    if linear_solver is None:
        linear_solver = lx.AutoLinearSolver(well_posed=None)
    if not isinstance(linear_solver, lx.AbstractLinearSolver):
        msg = (
            "'linear_solver' must be a lineax AbstractLinearSolver, "
            f"got {type(linear_solver).__name__}"
        )
        raise ValidationError(msg)
    return optx.ImplicitAdjoint(linear_solver=TolerantLinearSolver(linear_solver))

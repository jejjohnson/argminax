"""The differentiable fixed-point loop over Gaussian sites, for EP and PL.

One site update is the caller's (and gaussx's: `gaussx.cavity_distribution`,
`gaussx.ep_tilted_moments`, `gaussx.newton_update`,
`gaussx.damped_natural_update`); the loop that iterates it to a tolerance and
differentiates through the fixed point is here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import optimistix as optx
from jaxtyping import Array, Bool, Int, PyTree

from argminax.exceptions import ValidationError


__all__ = [
    "SiteFixedPoint",
    "site_fixed_point",
]


class SiteFixedPoint(eqx.Module):
    """The fixed point s* = T(s*, args) of a site update, and how the loop ended.

    Attributes:
        sites: The sites after the last iteration, a pytree shaped like the
            starting sites, implicitly differentiable in ``args``.
        n_iter: Iterations run.
        converged: Whether the stopping test was met; ``False`` also when an
            iterate turned non-finite, and when the update cycles. Implicit
            gradients assume it.
    """

    sites: PyTree[Array]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]


def _default_adjoint() -> optx.ImplicitAdjoint:
    """safe_implicit_adjoint(GMRES(rtol=atol=1e-10, restart=20, max_steps=50))."""
    # imported here: argminax._src.adjoint imports argminax._src.solve._tolerant,
    # so a module-level import would be circular
    from argminax._src.adjoint import safe_implicit_adjoint

    solver = lx.GMRES(rtol=1e-10, atol=1e-10, restart=20, max_steps=50)
    return safe_implicit_adjoint(solver)


def _floored_tolerance(tol: float, sites: PyTree[Array]) -> float:
    """tol, floored at 64 machine epsilons of the sites' dtype."""
    leaves = jtu.tree_leaves(eqx.filter(sites, eqx.is_inexact_array))
    dtype = jnp.result_type(*leaves)
    # a host constant, so the floor also works under jit
    epsilon = float(jnp.finfo(dtype).eps)
    return max(tol, 64.0 * epsilon)


def _max_norm(tree: PyTree[Array]) -> Array:
    """‖·‖_∞ over a pytree, in the default float dtype."""
    norm = optx.max_norm(tree)
    return norm.astype(jnp.result_type(float))


def site_fixed_point(
    update: Callable[[PyTree[Array], Any], PyTree[Array]],
    sites: PyTree[Array],
    args: PyTree[Any] = None,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    adjoint: optx.AbstractAdjoint | None = None,
) -> SiteFixedPoint:
    """Iterate s ← T(s, args) over Gaussian sites to a fixed point, differentiably.

        s⁽ⁱ⁺¹⁾ = T(s⁽ⁱ⁾, args),   stop when
        maxⱼ |sⱼ⁽ⁱ⁾ − sⱼ⁽ⁱ⁻¹⁾| / (tol (1 + |sⱼ⁽ⁱ⁾|) / 10) < 1

    s are the site parameters, any pytree of arrays (for N diagonal sites,
    the naturals (Λμ_site, Λ_site) ∈ ℝ^2N), and T is ``update``, one damped
    site update of expectation propagation, posterior linearisation or a
    Newton method, with every array the gradient should reach in ``args``.
    The test runs over every component of every leaf; the tenth keeps the
    posterior marginals within about ``tol`` of the fixed point, since a damped
    map contracts slowly and the sites settle one iteration behind the mean. A
    masked site that holds (0, 0) never delays it. The gradient of a loss L at
    the fixed point s* comes from the implicit function theorem,

        (I − ∂ₛT)ᵀ v = ∇ₛL,   ∇_args L = (∂_args T)ᵀ v,

    one linear solve however many iterations ran. ∂ₛT is not symmetric, so the
    default solves it with GMRES from products alone, and damping, which
    scales I − ∂ₛT by α without moving s*, leaves the gradient unchanged.

    Damping, precision floors, step caps and the first-iteration special case
    of cavity methods stay in ``update``: they differ between strategies.

    Algorithm:
        1. Run ``optx.fixed_point`` with ``optx.FixedPointIteration`` at
           ``rtol = atol = tol / 10`` (tol floored at 64 machine epsilons of
           the dtype), for at most ``max_iter`` iterations, without raising.
        2. Report the iterations run and whether the test was met.

    Args:
        update: The site update T, ``update(sites, args) -> sites``, returning
            a pytree with the structure, shapes and dtypes of ``sites``.
        sites: The starting sites s⁽⁰⁾.
        args: Passed to ``update``; the arrays gradients flow to.
        max_iter: Maximum iterations.
        tol: The tolerance of the stopping test above.
        adjoint: How gradients pass through the loop, any optimistix adjoint.
            ``None`` is ``argminax.adjoint.safe_implicit_adjoint(lx.GMRES(
            rtol=1e-10, atol=1e-10, restart=20, max_steps=50))``, which forms
            no Jacobian and never raises; ``optx.RecursiveCheckpointAdjoint()``
            differentiates through the iterations instead.

    Returns:
        The `SiteFixedPoint`. A non-finite iterate is returned, with
        ``converged`` false, rather than raised, so a log density built on it
        stays non-finite and a sampler rejects the point; the default adjoint
        then solves for v = 0 instead of raising (`TolerantLinearSolver`), so
        the backward pass raises nothing either. An update that cycles (EP on
        a non-log-concave likelihood can) also ends with ``converged`` false,
        which is the only signal of it.

    Raises:
        ValidationError: If ``max_iter`` is not a positive integer, ``tol`` is
            not positive, or ``adjoint`` is not an optimistix adjoint.

    Examples:
        A damped update with a closed-form fixed point s* = θ / (1 − ρ):

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from argminax.solve import site_fixed_point
        >>> def update(sites, theta):
        ...     # T(s) = (1 − α) s + α (ρ s + θ), with α = ½ and ρ = 0.3
        ...     return 0.5 * sites + 0.5 * (0.3 * sites + theta)
        >>> theta = jnp.array([1.0, 2.0])
        >>> fit = site_fixed_point(
        ...     update, jnp.zeros(2), theta, max_iter=100, tol=1e-10
        ... )
        >>> bool(fit.converged), bool(jnp.allclose(fit.sites, theta / 0.7))
        (True, True)

        The fixed point is differentiable in ``args``, here
        ∂(∑ s*)/∂θ = 1 / 0.7 per site:

        >>> def total(theta):
        ...     return site_fixed_point(update, jnp.zeros(2), theta).sites.sum()
        >>> bool(jnp.allclose(jax.grad(total)(theta), 1.0 / 0.7))
        True

        Sites as a pair of naturals, and a budget too small to converge:

        >>> def pair(sites, theta):
        ...     information, precision = sites
        ...     return (0.5 * information + theta, 0.5 * precision + 1.0)
        >>> short = site_fixed_point(
        ...     pair, (jnp.zeros(2), jnp.ones(2)), theta, max_iter=2
        ... )
        >>> bool(short.converged), int(short.n_iter)
        (False, 2)

    References:
        Minka, T. P. (2001). Expectation Propagation for Approximate Bayesian
        Inference. UAI.
        García-Fernández, Á. F., Tronarp, F. & Särkkä, S. (2019). Gaussian
        Process Classification Using Posterior Linearization. IEEE Signal
        Processing Letters 26(5), 735–739.
    """
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        msg = f"'max_iter' must be a positive integer, got {max_iter!r}"
        raise ValidationError(msg)
    if not tol > 0:
        msg = f"'tol' must be positive, got {tol!r}"
        raise ValidationError(msg)
    if adjoint is None:
        adjoint = _default_adjoint()
    if not isinstance(adjoint, optx.AbstractAdjoint):
        msg = (
            "'adjoint' must be an optimistix AbstractAdjoint, "
            f"got {type(adjoint).__name__}"
        )
        raise ValidationError(msg)

    # --- the loop ---
    floored_tol = _floored_tolerance(tol, sites)
    # |Δsⱼ| < (tol / 10) (1 + |sⱼ|), per component
    # the norm in the default float dtype, the dtype optimistix starts its
    # relative error in: float32 sites under jax_enable_x64 otherwise give a
    # float32 error against a float64 start, which lax.while_loop rejects
    solver = optx.FixedPointIteration(
        rtol=0.1 * floored_tol, atol=0.1 * floored_tol, norm=_max_norm
    )
    solution = optx.fixed_point(
        update,
        solver,
        sites,
        args,
        max_steps=max_iter,
        adjoint=adjoint,
        throw=False,
    )
    converged = solution.result == optx.RESULTS.successful
    return SiteFixedPoint(
        sites=solution.value,
        n_iter=solution.stats["num_steps"],
        converged=converged,
    )

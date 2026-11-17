"""The Gaussian an optimiser state holds, as a `Laplace` posterior.

Every transform of `argminax.blr` carries q(θ) = 𝒩(μ_θ, Λ_θ⁻¹) in its state.
`blr_posterior` reads it out as the same `Laplace` object a Gauss–Newton fit
returns, so sampling, the log-determinant and `argminax.diagnose` work on it.
"""

from __future__ import annotations

import gaussx as gx
import jax
import lineax as lx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree

from argminax._src.blr._diagonal import BLRDiagonalState
from argminax._src.blr._full_rank import BLRFullRankState
from argminax._src.blr._ivon import IVONState
from argminax._src.blr._low_rank import BLRLowRankState
from argminax._src.curvature import Laplace
from argminax.exceptions import ValidationError


__all__ = [
    "blr_posterior",
]


_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})

type _State = BLRDiagonalState | BLRFullRankState | BLRLowRankState | IVONState


def _ravel(tree: PyTree[Float[Array, "*state"]]) -> Float[Array, " state"]:
    """Flatten a pytree to one vector, in the order of ``ravel_pytree``."""
    flat, _ = ravel_pytree(tree)
    return flat


def _diagonal(precision: Float[Array, " state"]) -> lx.AbstractLinearOperator:
    return lx.TaggedLinearOperator(lx.DiagonalLinearOperator(precision), _PSD)


def _solve(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " state"],
    solver: gx.AbstractSolveStrategy | None,
) -> Float[Array, " state"]:
    if solver is None:
        return gx.solve(operator, vector)
    return solver.solve(operator, vector)


def blr_posterior(
    state: _State,
    params: PyTree[Float[Array, "*state"]] | None = None,
    *,
    weight_decay: float | None = None,
    solver: gx.AbstractSolveStrategy | None = None,
) -> Laplace:
    """Read the Gaussian out of a Bayesian learning rule state, as a `Laplace`.

        q(θ) = 𝒩(μ_θ, Λ_θ⁻¹)

    μ_θ ∈ ℝ^D_θ is the mean and Λ_θ (D_θ, D_θ) the precision, returned as the
    ``mode`` and the ``precision`` operator of a `Laplace`; the covariance
    Λ_θ⁻¹ is never formed. The precision keeps the structure of the state:

    | State | Mode μ_θ | Precision Λ_θ |
    |---|---|---|
    | `BLRDiagonalState` | (Λ_θ μ_θ) / Λ_θ | a diagonal operator |
    | `BLRFullRankState` | Λ_θ⁻¹ (Λ_θ μ_θ), one solve | a dense operator |
    | `BLRLowRankState` | Λ_θ⁻¹ (Λ_θ μ_θ), one Woodbury solve | diag(d) + U Uᵀ |
    | `IVONState` | ``params`` | diag(h + λ), h its curvature estimate |

    The diagonal and IVON states are pytrees shaped like the parameters; their
    mode and precision are flattened to one vector of length D_θ in the order
    of ``jax.flatten_util.ravel_pytree``, whose second return value for the
    parameters maps a draw back to their pytree. A `newton` state is a
    `BLRFullRankState` whose precision is the damped Hessian, positive
    definite only near a minimum: read it there.

    Args:
        state: The state of `blr_diagonal`, `blr_full_rank`, `blr_low_rank`,
            `newton` or `ivon`.
        params: The current parameters, IVON's mean. Required for an
            `IVONState`, whose state does not hold them, and rejected
            otherwise.
        weight_decay: IVON's weight decay λ, the value given to `ivon`.
            Required for an `IVONState` and rejected otherwise.
        solver: The gaussx solve strategy of the returned `Laplace`, also
            used for the solve that gives the mode; ``None`` is
            `gaussx.DenseSolver`, which picks the solve from the structure of
            the precision.

    Returns:
        The `Laplace` posterior 𝒩(μ_θ, Λ_θ⁻¹) over the flat parameter vector.
        ``Laplace.sample`` factors a low-rank or a dense precision densely,
        O(D_θ³); a diagonal one is sampled in closed form.

    Raises:
        ValidationError: If ``state`` is not one of the four state types, an
            `IVONState` comes without ``params`` or ``weight_decay``, or
            another state comes with either.

    Examples:
        A diagonal state on a parameter pytree, read as one flat Gaussian:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from jax.flatten_util import ravel_pytree
        >>> from argminax.blr import blr_diagonal, blr_posterior
        >>> params = {"w": jnp.ones((2, 2)), "b": jnp.zeros(1)}
        >>> optimiser = blr_diagonal(0.5, prior_precision=2.0, curvature="identity")
        >>> state = optimiser.init(params)
        >>> posterior = blr_posterior(state)
        >>> posterior.mode.shape
        (5,)
        >>> variance = posterior.covariance().mv(jnp.ones(5))
        >>> bool(jnp.allclose(variance, 0.5))
        True
        >>> _, unravel = ravel_pytree(params)
        >>> draws = posterior.sample(3, key=jax.random.key(0))
        >>> jax.vmap(unravel)(draws)["w"].shape
        (3, 2, 2)

        IVON's state holds neither its mean nor its weight decay, so both are
        passed:

        >>> from argminax.blr import ivon
        >>> optimiser = ivon(0.1, weight_decay=1.0, hess_init=3.0)
        >>> theta = jnp.array([0.5, -0.5])
        >>> posterior = blr_posterior(
        ...     optimiser.init(theta), theta, weight_decay=1.0
        ... )
        >>> bool(jnp.allclose(posterior.covariance().mv(jnp.ones(2)), 0.25))
        True

        >>> blr_posterior(
        ...     optimiser.init(theta)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: an IVONState needs 'params' and 'weight_decay', ...
    """
    strategy = gx.DenseSolver() if solver is None else solver
    if isinstance(state, IVONState):
        if params is None or weight_decay is None:
            msg = (
                "an IVONState needs 'params' and 'weight_decay', the mean and the "
                "prior precision its state does not hold"
            )
            raise ValidationError(msg)
        # Λ_θ = diag(h + λ)
        precision = _ravel(state.hess) + weight_decay
        return Laplace(_ravel(params), _diagonal(precision), strategy)
    if params is not None or weight_decay is not None:
        msg = (
            "'params' and 'weight_decay' are only for an IVONState, got a "
            f"{type(state).__name__}"
        )
        raise ValidationError(msg)
    if isinstance(state, BLRDiagonalState):
        # μ_θ = (Λ_θ μ_θ) / Λ_θ, leaf by leaf
        mean = jax.tree.map(lambda eta, lam: eta / lam, state.nat_mean, state.precision)
        return Laplace(_ravel(mean), _diagonal(_ravel(state.precision)), strategy)
    if isinstance(state, BLRFullRankState):
        operator = lx.MatrixLinearOperator(state.precision, _PSD)
        # μ_θ = Λ_θ⁻¹ (Λ_θ μ_θ)
        mode = _solve(operator, state.nat_mean, solver)
        return Laplace(mode, operator, strategy)
    if isinstance(state, BLRLowRankState):
        # Λ_θ = diag(d) + U Uᵀ
        operator = gx.low_rank_plus_diag(
            state.diag_precision, state.low_rank_factor, psd=True
        )
        # μ_θ = Λ_θ⁻¹ (Λ_θ μ_θ), by the Woodbury identity
        mode = _solve(operator, state.nat_mean, solver)
        return Laplace(mode, operator, strategy)
    msg = (
        "'state' must be a BLRDiagonalState, BLRFullRankState, BLRLowRankState or "
        f"IVONState, got a {type(state).__name__}"
    )
    raise ValidationError(msg)

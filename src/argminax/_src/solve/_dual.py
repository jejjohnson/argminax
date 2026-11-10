"""The observation-space (dual, PSAS) solve and its incremental outer loop.

With M observations and N unknowns, CG runs on the M × M system
(H B Hᵀ + R) λ = d instead of the N × N primal one, which is the cheaper side
when M ≪ N; the increment is δx̂ = B Hᵀ λ.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from argminax.exceptions import ValidationError


__all__ = [
    "DualSolution",
    "dual_solve",
    "incremental_dual_solve",
]


_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


class DualSolution(eqx.Module):
    """The result of an observation-space solve.

    Attributes:
        value: The increment δx̂ = B Hᵀ λ (`dual_solve`), or the analysis
            x̂ = x_b + δx̂ (`incremental_dual_solve`), shape (N,).
        multiplier: The dual variable λ, shape (M,).
        result: lineax's ``RESULTS`` of the (last) CG solve; check it before
            trusting ``value`` or a gradient through it.
        num_steps: CG iterations of the (last) solve.
    """

    value: Float[Array, " N"]
    multiplier: Float[Array, " M"]
    result: lx.RESULTS
    num_steps: Int[Array, ""]


def _require_operator(operator: lx.AbstractLinearOperator, name: str) -> None:
    if not lx.is_positive_semidefinite(operator):
        msg = f"'{name}' must be tagged positive semidefinite, got tags without it"
        raise ValidationError(msg)


def dual_solve(
    jvp: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    vjp: Callable[[Float[Array, " M"]], Float[Array, " N"]],
    background_cov: lx.AbstractLinearOperator,
    obs_cov: lx.AbstractLinearOperator,
    innovation: Float[Array, " M"],
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> DualSolution:
    """Solve the 3D-Var increment in observation space.

        (H B Hᵀ + R) λ = d,   δx̂ = B Hᵀ λ,   d = y − H(x_b)

    H is the observation operator linearised at the background x_b (given as
    ``jvp``: δx ↦ H δx and ``vjp``: δy ↦ Hᵀ δy), B the background covariance
    and R the observation-error covariance. For a linear H, δx̂ is the
    minimiser of ½ δxᵀ B⁻¹ δx + ½ (d − H δx)ᵀ R⁻¹ (d − H δx), the same
    increment as the primal Gauss–Newton step; for a nonlinear H, use
    `incremental_dual_solve`. CG converges in the dual operator's norm, so
    its tolerance does not bound the error on δx̂ directly.

    Algorithm:
        1. Wrap v ↦ H B Hᵀ v + R v as a PSD-tagged operator on the M
           observations.
        2. Solve for λ with ``solver`` (CG preconditioned by R⁻¹ by default).
        3. Map back: δx̂ = B Hᵀ λ.

    Args:
        jvp: The linearised observation operator, δx ↦ H δx, (N,) → (M,).
        vjp: Its transpose, δy ↦ Hᵀ δy, (M,) → (N,).
        background_cov: B, (N, N), tagged positive semidefinite.
        obs_cov: R, (M, M), tagged positive semidefinite.
        innovation: d = y − H(x_b), shape (M,).
        solver: The lineax solver for the dual system; default CG with
            rtol = atol = 1e-10 (1e-5 in float32) and 1000 steps,
            preconditioned by R⁻¹.

    Returns:
        The increment δx̂, the multiplier λ, the CG result and step count.

    Raises:
        ValidationError: If B or R is not tagged positive semidefinite.

    Examples:
        One state observed twice with unit noise and a unit prior:
        δx̂ = Hᵀ (H Hᵀ + I)⁻¹ d with H = (1, 1)ᵀ, so δx̂ = (d₁ + d₂)/3.

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from argminax.solve import dual_solve
        >>> observation = jnp.array([[1.0], [1.0]])
        >>> psd = lx.positive_semidefinite_tag
        >>> eye = lambda n: lx.MatrixLinearOperator(jnp.eye(n), psd)
        >>> solution = dual_solve(
        ...     lambda dx: observation @ dx,
        ...     lambda dy: observation.T @ dy,
        ...     eye(1),
        ...     eye(2),
        ...     jnp.array([1.0, 2.0]),
        ... )
        >>> bool(jnp.allclose(solution.value, 1.0))
        True
        >>> bool(solution.result == lx.RESULTS.successful)
        True

    References:
        Courtier, P. (1997). Dual formulation of four-dimensional variational
        assimilation. Q. J. R. Meteorol. Soc. 123, 2449. doi:10.1002/qj.49712354414.
    """
    _require_operator(background_cov, "background_cov")
    _require_operator(obs_cov, "obs_cov")
    structure = jax.eval_shape(lambda: innovation)

    def dual_matvec(multiplier: Float[Array, " M"]) -> Float[Array, " M"]:
        # B Hᵀ v
        spread = background_cov.mv(vjp(multiplier))
        # H B Hᵀ v + R v
        return jvp(spread) + obs_cov.mv(multiplier)

    dual_operator = lx.FunctionLinearOperator(dual_matvec, structure, _PSD)
    options = {}
    if solver is None:
        # 1e-10 in float64, 100 ε_mach (about 1e-5) in float32
        tolerance = max(1e-10, 100.0 * float(jnp.finfo(innovation.dtype).eps))
        solver = lx.CG(rtol=tolerance, atol=tolerance, max_steps=1000)
        # R⁻¹ as the preconditioner, structured where R is
        preconditioner = lx.TaggedLinearOperator(gx.inv(obs_cov), _PSD)
        options = {"preconditioner": preconditioner}
    # λ = (H B Hᵀ + R)⁻¹ d
    solution = lx.linear_solve(
        dual_operator, innovation, solver=solver, options=options, throw=False
    )
    multiplier = solution.value
    # δx̂ = B Hᵀ λ
    increment = background_cov.mv(vjp(multiplier))
    num_steps = solution.stats.get("num_steps", jnp.array(0))
    return DualSolution(
        value=increment,
        multiplier=multiplier,
        result=solution.result,
        num_steps=jnp.asarray(num_steps),
    )


def incremental_dual_solve(
    forward: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    background_state: Float[Array, " N"],
    background_cov: lx.AbstractLinearOperator,
    obs_cov: lx.AbstractLinearOperator,
    observation: Float[Array, " M"],
    *,
    num_outer: int = 5,
    solver: lx.AbstractLinearSolver | None = None,
) -> DualSolution:
    """Incremental 3D-Var in observation space, for a nonlinear forward model.

        dₖ = y − H(xₖ) + Hₖ (xₖ − x_b),   xₖ₊₁ = x_b + B Hₖᵀ (Hₖ B Hₖᵀ + R)⁻¹ dₖ

    Hₖ is the forward model H linearised at xₖ. Each outer iteration is one
    Gauss–Newton step on the primal cost ½ (x − x_b)ᵀ B⁻¹ (x − x_b) +
    ½ (y − H(x))ᵀ R⁻¹ (y − H(x)), solved in observation space by
    `dual_solve`, so at a fixed point xₖ is that cost's minimum. The number
    of outer iterations is fixed (``num_outer``), as in operational
    incremental 4D-Var.

    Algorithm:
        1. Start from x₀ = x_b.
        2. Linearise H at xₖ (`jax.linearize`, and `jax.linear_transpose`
           for Hₖᵀ), form dₖ, solve in observation space, and set xₖ₊₁.
        3. Stop after ``num_outer`` iterations.

    Args:
        forward: The observation operator H, x ↦ H(x), (N,) → (M,).
        background_state: x_b, shape (N,).
        background_cov: B, (N, N), tagged positive semidefinite.
        obs_cov: R, (M, M), tagged positive semidefinite.
        observation: y, shape (M,).
        num_outer: Number of outer iterations, at least 1.
        solver: The lineax solver for each dual system (see `dual_solve`).

    Returns:
        The analysis x̂ in ``value``, with the last solve's multiplier, CG
        result and step count.

    Raises:
        ValidationError: If ``num_outer`` is not a positive integer.

    Examples:
        A linear H is solved by the first outer iteration.

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from argminax.solve import incremental_dual_solve
        >>> psd = lx.positive_semidefinite_tag
        >>> eye = lambda n: lx.MatrixLinearOperator(jnp.eye(n), psd)
        >>> solution = incremental_dual_solve(
        ...     lambda x: jnp.concatenate([x, x]),
        ...     jnp.zeros(1),
        ...     eye(1),
        ...     eye(2),
        ...     jnp.array([1.0, 2.0]),
        ...     num_outer=1,
        ... )
        >>> bool(jnp.allclose(solution.value, 1.0))
        True
    """
    if isinstance(num_outer, bool) or not isinstance(num_outer, int) or num_outer < 1:
        msg = f"'num_outer' must be a positive integer, got {num_outer!r}"
        raise ValidationError(msg)

    def outer(state: Float[Array, " N"]) -> DualSolution:
        # H(xₖ) and δx ↦ Hₖ δx
        predicted, jvp = jax.linearize(forward, state)
        # δy ↦ Hₖᵀ δy
        transpose = jax.linear_transpose(jvp, state)

        def vjp(cotangent: Float[Array, " M"]) -> Float[Array, " N"]:
            return transpose(cotangent)[0]

        # dₖ = y − H(xₖ) + Hₖ (xₖ − x_b)
        innovation = observation - predicted + jvp(state - background_state)
        solution = dual_solve(
            jvp, vjp, background_cov, obs_cov, innovation, solver=solver
        )
        # xₖ₊₁ = x_b + δx̂
        return eqx.tree_at(
            lambda current: current.value, solution, background_state + solution.value
        )

    solution = outer(background_state)
    for _ in range(num_outer - 1):
        solution = outer(solution.value)
    return solution

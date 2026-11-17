"""The observation-space (dual, PSAS) solve and its incremental outer loop.

With D_y observations and D_x unknowns, CG runs on the D_y × D_y system
(H Σ_x Hᵀ + Σ_ε) w = r instead of the D_x × D_x primal one, which is the
cheaper side when D_y ≪ D_x; the increment is δx̂ = Σ_x Hᵀ w.
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
        value: The increment δx̂ = Σ_x Hᵀ w (`dual_solve`), or the analysis
            x̂ = μ_x + δx̂ (`incremental_dual_solve`), shape (D_x,).
        multiplier: The observation-space weights w, shape (D_y,).
        result: lineax's ``RESULTS`` of the (last) CG solve; check it before
            trusting ``value`` or a gradient through it.
        num_steps: CG iterations of the (last) solve.
    """

    value: Float[Array, " state"]
    multiplier: Float[Array, " obs"]
    result: lx.RESULTS
    num_steps: Int[Array, ""]


def _require_operator(operator: lx.AbstractLinearOperator, name: str) -> None:
    if not lx.is_positive_semidefinite(operator):
        msg = f"'{name}' must be tagged positive semidefinite, got tags without it"
        raise ValidationError(msg)


def dual_solve(
    jvp: Callable[[Float[Array, " state"]], Float[Array, " obs"]],
    vjp: Callable[[Float[Array, " obs"]], Float[Array, " state"]],
    background_cov: lx.AbstractLinearOperator,
    obs_cov: lx.AbstractLinearOperator,
    innovation: Float[Array, " obs"],
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> DualSolution:
    """Solve the 3D-Var increment in observation space.

        (H Σ_x Hᵀ + Σ_ε) w = r,   δx̂ = Σ_x Hᵀ w,   r = y − h(μ_x)

    H = J_h (D_y × D_x) is the observation operator h linearised at the
    background (prior) mean μ_x ∈ ℝ^D_x, given as ``jvp``: δx ↦ H δx and
    ``vjp``: δy ↦ Hᵀ δy; Σ_x (D_x × D_x) is the background covariance,
    Σ_ε (D_y × D_y) the observation-error covariance, r ∈ ℝ^D_y the
    innovation and w ∈ ℝ^D_y the observation-space weights. For a linear h,
    δx̂ is the minimiser of 𝒥(δx) = ½ δxᵀ Σ_x⁻¹ δx + ½ (r − H δx)ᵀ Σ_ε⁻¹
    (r − H δx), the same increment as the primal Gauss–Newton step; for a
    nonlinear h, use `incremental_dual_solve`. CG converges in the dual
    operator's norm, so its tolerance does not bound the error on δx̂
    directly.

    Algorithm:
        1. Wrap v ↦ H Σ_x Hᵀ v + Σ_ε v as a PSD-tagged operator on the D_y
           observations.
        2. Solve for w with ``solver`` (CG preconditioned by Σ_ε⁻¹ by
           default).
        3. Map back: δx̂ = Σ_x Hᵀ w.

    Args:
        jvp: The linearised observation operator, δx ↦ H δx,
            (D_x,) → (D_y,).
        vjp: Its transpose, δy ↦ Hᵀ δy, (D_y,) → (D_x,).
        background_cov: Σ_x, (D_x, D_x), tagged positive semidefinite.
        obs_cov: Σ_ε, (D_y, D_y), tagged positive semidefinite.
        innovation: r = y − h(μ_x), shape (D_y,).
        solver: The lineax solver for the dual system; default CG with
            rtol = atol = 1e-10 (1e-5 in float32) and 1000 steps,
            preconditioned by Σ_ε⁻¹.

    Returns:
        The increment δx̂, the weights w, the CG result and step count.

    Raises:
        ValidationError: If Σ_x or Σ_ε is not tagged positive semidefinite.

    Examples:
        One state observed twice with unit noise and a unit prior:
        δx̂ = Hᵀ (H Hᵀ + I)⁻¹ r with H = (1, 1)ᵀ, so δx̂ = (r₁ + r₂)/3.

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

        The covariances must carry the PSD tag, since CG relies on it:

        >>> untagged = lx.MatrixLinearOperator(jnp.eye(2))
        >>> dual_solve(
        ...     lambda dx: observation @ dx,
        ...     lambda dy: observation.T @ dy,
        ...     eye(1),
        ...     untagged,
        ...     jnp.array([1.0, 2.0]),
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'obs_cov' must be tagged positive semidefinite

    References:
        Courtier, P. (1997). Dual formulation of four-dimensional variational
        assimilation. Q. J. R. Meteorol. Soc. 123, 2449. doi:10.1002/qj.49712354414.
    """
    _require_operator(background_cov, "background_cov")
    _require_operator(obs_cov, "obs_cov")
    structure = jax.eval_shape(lambda: innovation)

    # --- the dual operator ---
    def dual_matvec(multiplier: Float[Array, " obs"]) -> Float[Array, " obs"]:
        # Hᵀ v
        back_projected = vjp(multiplier)
        # Σ_x Hᵀ v
        spread = background_cov.mv(back_projected)
        # H Σ_x Hᵀ v
        observed = jvp(spread)
        # Σ_ε v
        noise = obs_cov.mv(multiplier)
        # (H Σ_x Hᵀ + Σ_ε) v
        return observed + noise

    dual_operator = lx.FunctionLinearOperator(dual_matvec, structure, _PSD)

    # --- the default solver ---
    options = {}
    if solver is None:
        # a host constant from the dtype, so the default also works under jit
        machine_epsilon = float(jnp.finfo(innovation.dtype).eps)
        # tol = max(1e-10, 100 × machine epsilon): 1e-10 in float64, about
        # 1e-5 in float32
        tolerance = max(1e-10, 100.0 * machine_epsilon)
        solver = lx.CG(rtol=tolerance, atol=tolerance, max_steps=1000)
        # Σ_ε⁻¹, structured where Σ_ε is
        obs_precision = gx.inv(obs_cov)
        preconditioner = lx.TaggedLinearOperator(obs_precision, _PSD)
        options = {"preconditioner": preconditioner}

    # --- solve and map back ---
    # w = (H Σ_x Hᵀ + Σ_ε)⁻¹ r
    solution = lx.linear_solve(
        dual_operator, innovation, solver=solver, options=options, throw=False
    )
    multiplier = solution.value
    # Hᵀ w
    back_projected = vjp(multiplier)
    # δx̂ = Σ_x Hᵀ w
    increment = background_cov.mv(back_projected)
    num_steps = solution.stats.get("num_steps", jnp.array(0))
    return DualSolution(
        value=increment,
        multiplier=multiplier,
        result=solution.result,
        num_steps=jnp.asarray(num_steps),
    )


def incremental_dual_solve(
    forward: Callable[[Float[Array, " state"]], Float[Array, " obs"]],
    background_state: Float[Array, " state"],
    background_cov: lx.AbstractLinearOperator,
    obs_cov: lx.AbstractLinearOperator,
    observation: Float[Array, " obs"],
    *,
    num_outer: int = 5,
    solver: lx.AbstractLinearSolver | None = None,
) -> DualSolution:
    """Incremental 3D-Var in observation space, for a nonlinear forward model.

        r⁽ⁱ⁾ = y − h(x⁽ⁱ⁾) + H⁽ⁱ⁾ (x⁽ⁱ⁾ − μ_x)
        x⁽ⁱ⁺¹⁾ = μ_x + Σ_x H⁽ⁱ⁾ᵀ (H⁽ⁱ⁾ Σ_x H⁽ⁱ⁾ᵀ + Σ_ε)⁻¹ r⁽ⁱ⁾

    H⁽ⁱ⁾ = J_h(x⁽ⁱ⁾) is the forward model h linearised at the outer iterate
    x⁽ⁱ⁾ ∈ ℝ^D_x, μ_x the background (prior) mean, Σ_x the background
    covariance and Σ_ε the observation-error covariance. Each outer iteration
    is one Gauss–Newton step on the primal cost 𝒥(x) = ½ (x − μ_x)ᵀ Σ_x⁻¹
    (x − μ_x) + ½ (y − h(x))ᵀ Σ_ε⁻¹ (y − h(x)), solved in observation space
    by `dual_solve`, so at a fixed point x⁽ⁱ⁾ is that cost's minimum. The
    number of outer iterations is fixed (``num_outer``), as in operational
    incremental 4D-Var.

    Algorithm:
        1. Start from x⁽⁰⁾ = μ_x.
        2. Linearise h at x⁽ⁱ⁾ (`jax.linearize`, and `jax.linear_transpose`
           for H⁽ⁱ⁾ᵀ), form r⁽ⁱ⁾, solve in observation space, and set
           x⁽ⁱ⁺¹⁾.
        3. Stop after ``num_outer`` iterations.

    Args:
        forward: The observation operator h, x ↦ h(x), (D_x,) → (D_y,).
        background_state: The background (prior) mean μ_x, shape (D_x,).
        background_cov: Σ_x, (D_x, D_x), tagged positive semidefinite.
        obs_cov: Σ_ε, (D_y, D_y), tagged positive semidefinite.
        observation: y, shape (D_y,).
        num_outer: Number of outer iterations, at least 1.
        solver: The lineax solver for each dual system (see `dual_solve`).

    Returns:
        The analysis x̂ in ``value``, with the last solve's weights w, CG
        result and step count.

    Raises:
        ValidationError: If ``num_outer`` is not a positive integer.

    Examples:
        A linear h is solved by the first outer iteration.

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

        At least one outer iteration:

        >>> incremental_dual_solve(
        ...     lambda x: x, jnp.zeros(1), eye(1), eye(1), jnp.ones(1), num_outer=0
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'num_outer' must be a positive integer, got 0
    """
    if isinstance(num_outer, bool) or not isinstance(num_outer, int) or num_outer < 1:
        msg = f"'num_outer' must be a positive integer, got {num_outer!r}"
        raise ValidationError(msg)

    def outer(state: Float[Array, " state"]) -> DualSolution:
        # h(x⁽ⁱ⁾) and δx ↦ H⁽ⁱ⁾ δx
        predicted, jvp = jax.linearize(forward, state)
        # δy ↦ H⁽ⁱ⁾ᵀ δy
        transpose = jax.linear_transpose(jvp, state)

        def vjp(cotangent: Float[Array, " obs"]) -> Float[Array, " state"]:
            # linear_transpose returns a tuple, one cotangent per primal input
            (back_projected,) = transpose(cotangent)
            return back_projected

        # x⁽ⁱ⁾ − μ_x
        departure = state - background_state
        # H⁽ⁱ⁾ (x⁽ⁱ⁾ − μ_x)
        linear_departure = jvp(departure)
        # r⁽ⁱ⁾ = y − h(x⁽ⁱ⁾) + H⁽ⁱ⁾ (x⁽ⁱ⁾ − μ_x)
        innovation = observation - predicted + linear_departure
        solution = dual_solve(
            jvp, vjp, background_cov, obs_cov, innovation, solver=solver
        )
        # x⁽ⁱ⁺¹⁾ = μ_x + δx̂
        analysis = background_state + solution.value
        return eqx.tree_at(lambda current: current.value, solution, analysis)

    solution = outer(background_state)
    for _ in range(num_outer - 1):
        solution = outer(solution.value)
    return solution

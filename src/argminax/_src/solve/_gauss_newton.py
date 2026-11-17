"""Gauss–Newton and Levenberg–Marquardt with a matrix-free, budgeted inner CG.

Each step solves (J_rᵀ J_r + β I) δ = −J_rᵀ r by conjugate gradients on the
normal equations, from Jacobian-vector and vector-Jacobian products only, so
the state size D_x is limited by memory for vectors, not matrices.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lineax as lx
import optimistix as optx
from jaxtyping import Array, PyTree, Scalar

# optimistix's own flag-to-callback conversion; private, but optimistix is pinned
# to 0.1.x (see "Boundaries" in AGENTS.md)
from optimistix._misc import default_verbose

from argminax.exceptions import ValidationError


__all__ = [
    "GaussNewtonCG",
]


class _TruncatedNormalCG(lx.AbstractLinearSolver):
    """CG on the normal equations, accepting the iterate when the budget runs out.

    Inexact Gauss–Newton takes a truncated CG solve as its step: the iterate
    after ``max_steps`` is still a descent direction. lineax reports a capped
    solve as ``max_steps_reached``, which would stop the outer solve, so this
    solver reports it as successful; every other failure passes through.
    """

    inner: lx.Normal

    def init(self, operator: lx.AbstractLinearOperator, options: dict[str, Any]) -> Any:
        return self.inner.init(operator, options)

    def compute(
        self, state: Any, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], lx.RESULTS, dict[str, Any]]:
        solution, result, stats = self.inner.compute(state, vector, options)
        truncated = result == lx.RESULTS.max_steps_reached
        result = lx.RESULTS.where(truncated, lx.RESULTS.successful, result)
        return solution, result, stats

    def transpose(self, state: Any, options: dict[str, Any]) -> Any:
        return self.inner.transpose(state, options)

    def conj(self, state: Any, options: dict[str, Any]) -> Any:
        return self.inner.conj(state, options)

    def assume_full_rank(self) -> bool:
        return self.inner.assume_full_rank()


class GaussNewtonCG(optx.AbstractGaussNewton):
    """Gauss–Newton or Levenberg–Marquardt, each step by a matrix-free inner CG.

        (J_rᵀ J_r + β⁽ⁱ⁾ I) δ⁽ⁱ⁾ = −J_rᵀ r(x⁽ⁱ⁾),   x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ + α⁽ⁱ⁾ δ⁽ⁱ⁾

    The cost is 𝒥(x) = ½ ‖r(x)‖², with r ∈ ℝ^D_r the stacked residuals and
    x⁽ⁱ⁾ ∈ ℝ^D_x the iterate. The Jacobian J_r = ∂r/∂x at x⁽ⁱ⁾ (D_r × D_x) is
    never formed: CG runs on v ↦ J_rᵀ(J_r v) + β⁽ⁱ⁾ v, one Jacobian-vector
    and one vector-Jacobian product per iteration, capped at
    ``max_cg_steps``. With ``damping=0`` and no trust region this is
    Gauss–Newton (damping weight β = 0, step size α = 1); ``damping > 0`` is
    Levenberg–Marquardt with a fixed β; ``trust_region=True`` is
    Levenberg–Marquardt with β⁽ⁱ⁾ adapted from the ratio of actual to
    predicted decrease (optimistix's ``ClassicalTrustRegion``). It is an
    optimistix least-squares solver, called through ``optx.least_squares``.

    Algorithm:
        1. Linearise the residuals at x⁽ⁱ⁾, giving v ↦ J_r v and its
           transpose.
        2. Solve (J_rᵀ J_r + β⁽ⁱ⁾ I) δ⁽ⁱ⁾ = −J_rᵀ r(x⁽ⁱ⁾) by CG on the normal
           equations, stopping after ``max_cg_steps`` iterations at most.
        3. Accept the step with α = 1 (Gauss–Newton, fixed β) or let the
           trust region adapt β⁽ⁱ⁾ from the actual and predicted decrease.
        4. Stop when the iterate and the cost 𝒥 change by less than
           ``rtol`` and ``atol``.

    A capped CG returns its last iterate as the step (inexact Gauss–Newton);
    the CG tolerances bound how far that step is from the exact one, so set
    them below the accuracy the solve needs. With whitened residuals the prior
    rows keep J_rᵀ J_r ⪰ I, so undamped CG stays well posed. A forward model
    behind diffrax's ``custom_vjp`` refuses the forward-mode products this
    needs: give it ``diffrax.DirectAdjoint`` or another forward-capable
    adjoint first. A solve that stops unconverged is reported through
    ``RESULTS``: with ``throw=False``, check ``solution.result`` before
    trusting the value or its gradient.

    Attributes:
        rtol: Relative tolerance on the iterate and the cost (outer loop).
        atol: Absolute tolerance on the iterate and the cost (outer loop).
        norm: The norm the tolerances are measured in.
        descent: The optimistix descent, built from the arguments below.
        search: The optimistix search, built from the arguments below.
        verbose: optimistix's verbosity callback.

    Raises:
        ValidationError: If ``damping`` is negative, ``max_cg_steps`` is not a
            positive integer, or ``damping`` is combined with
            ``trust_region``.

    Examples:
        A linear least-squares problem is solved by one Gauss–Newton step.

        >>> import jax.numpy as jnp
        >>> import optimistix as optx
        >>> from argminax.solve import GaussNewtonCG
        >>> design = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        >>> data = jnp.array([1.0, 2.0, 2.0])
        >>> def residuals(x, args):
        ...     return design @ x - data
        >>> solution = optx.least_squares(
        ...     residuals, GaussNewtonCG(rtol=1e-5, atol=1e-5), jnp.zeros(2)
        ... )
        >>> exact = jnp.linalg.lstsq(design, data)[0]
        >>> bool(jnp.allclose(solution.value, exact, atol=1e-4))
        True

        Levenberg–Marquardt with a trust region:

        >>> lm = GaussNewtonCG(rtol=1e-5, atol=1e-5, trust_region=True)
        >>> solution = optx.least_squares(residuals, lm, jnp.zeros(2))
        >>> bool(jnp.allclose(solution.value, exact, atol=1e-4))
        True

    References:
        Nocedal, J. & Wright, S. J. (2006). Numerical Optimization, 2nd ed.,
        chapters 7 and 10.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: optx.AbstractDescent
    search: optx.AbstractSearch
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        *,
        damping: float = 0.0,
        trust_region: bool = False,
        max_cg_steps: int = 50,
        cg_rtol: float | None = None,
        cg_atol: float | None = None,
        verbose: bool | Callable[..., None] = False,
    ):
        """Build the solver.

        Args:
            rtol: Relative tolerance of the outer loop.
            atol: Absolute tolerance of the outer loop.
            norm: The norm the tolerances are measured in.
            damping: A fixed Levenberg–Marquardt damping weight β ≥ 0; 0 is
                Gauss–Newton.
            trust_region: Adapt β by optimistix's classical trust region.
            max_cg_steps: The CG budget per outer step.
            cg_rtol: Relative CG tolerance; default ``rtol / 10``.
            cg_atol: Absolute CG tolerance; default ``atol / 10``.
            verbose: optimistix verbosity (``True`` or a callback).
        """
        if damping < 0:
            msg = f"'damping' must be non-negative, got {damping!r}"
            raise ValidationError(msg)
        if damping > 0 and trust_region:
            msg = (
                "'damping' must be 0 with 'trust_region=True': the trust region "
                f"chooses λ itself, got damping={damping!r}"
            )
            raise ValidationError(msg)
        if (
            isinstance(max_cg_steps, bool)
            or not isinstance(max_cg_steps, int)
            or max_cg_steps < 1
        ):
            msg = f"'max_cg_steps' must be a positive integer, got {max_cg_steps!r}"
            raise ValidationError(msg)
        # the inner solve is tighter than the outer one, so CG error does not
        # set the outer accuracy
        # tol_CG = tol / 10, for both rtol and atol
        inner_rtol = rtol / 10 if cg_rtol is None else cg_rtol
        inner_atol = atol / 10 if cg_atol is None else cg_atol
        conjugate_gradient = lx.CG(
            rtol=inner_rtol, atol=inner_atol, max_steps=max_cg_steps
        )
        # CG on J_rᵀ J_r δ = −J_rᵀ r, from J_r v and J_rᵀ v alone
        normal_equations = lx.Normal(conjugate_gradient)
        linear_solver = _TruncatedNormalCG(normal_equations)
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        if trust_region:
            # β⁽ⁱ⁾ from the actual / predicted decrease ratio
            self.descent = optx.DampedNewtonDescent(linear_solver=linear_solver)
            self.search = optx.ClassicalTrustRegion()
        elif damping > 0:
            # optimistix's damped step solves (J_rᵀ J_r + I / γ) δ = −J_rᵀ r,
            # with γ its step size
            self.descent = optx.DampedNewtonDescent(linear_solver=linear_solver)
            # γ = 1 / β
            step_size = 1.0 / damping
            self.search = optx.LearningRate(step_size)
        else:
            self.descent = optx.NewtonDescent(linear_solver=linear_solver)
            self.search = optx.LearningRate(1.0)
        self.verbose = default_verbose(verbose)

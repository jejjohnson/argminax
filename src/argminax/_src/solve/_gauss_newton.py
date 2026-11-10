"""Gauss–Newton and Levenberg–Marquardt with a matrix-free, budgeted inner CG.

Each step solves (JᵀJ + λI) δ = −Jᵀr by conjugate gradients on the normal
equations, from Jacobian-vector and vector-Jacobian products only, so the state
size is limited by memory for vectors, not matrices.
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

        (Jₖᵀ Jₖ + λₖ I) δₖ = −Jₖᵀ r(xₖ),   xₖ₊₁ = xₖ + αₖ δₖ

    Jₖ = ∂r/∂x at xₖ is never formed: CG runs on v ↦ Jₖᵀ(Jₖ v) + λₖ v, one
    Jacobian-vector and one vector-Jacobian product per iteration, capped at
    ``max_cg_steps``. With ``damping=0`` and no trust region this is
    Gauss–Newton (λ = 0, α = 1); ``damping > 0`` is Levenberg–Marquardt with a
    fixed λ; ``trust_region=True`` is Levenberg–Marquardt with λ adapted from
    the ratio of actual to predicted decrease (optimistix's
    ``ClassicalTrustRegion``). It is an optimistix least-squares solver, called
    through ``optx.least_squares``.

    A capped CG returns its last iterate as the step (inexact Gauss–Newton);
    the CG tolerances bound how far that step is from the exact one, so set
    them below the accuracy the solve needs. With whitened residuals the prior
    rows keep JᵀJ ⪰ I, so undamped CG stays well posed. A forward model
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
            damping: A fixed Levenberg–Marquardt λ ≥ 0; 0 is Gauss–Newton.
            trust_region: Adapt λ by optimistix's classical trust region.
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
        inner_rtol = rtol / 10 if cg_rtol is None else cg_rtol
        inner_atol = atol / 10 if cg_atol is None else cg_atol
        linear_solver = _TruncatedNormalCG(
            lx.Normal(lx.CG(rtol=inner_rtol, atol=inner_atol, max_steps=max_cg_steps))
        )
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        if trust_region:
            # λ from the actual / predicted decrease ratio
            self.descent = optx.DampedNewtonDescent(linear_solver=linear_solver)
            self.search = optx.ClassicalTrustRegion()
        elif damping > 0:
            # optimistix's damped step solves (JᵀJ + I / μ) δ = −Jᵀr; μ = 1 / λ
            self.descent = optx.DampedNewtonDescent(linear_solver=linear_solver)
            self.search = optx.LearningRate(1.0 / damping)
        else:
            self.descent = optx.NewtonDescent(linear_solver=linear_solver)
            self.search = optx.LearningRate(1.0)
        self.verbose = default_verbose(verbose)

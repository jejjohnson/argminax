"""Damped Newton loops to the mode of a latent Gaussian model, and the Gaussian there.

One Newton step's algebra is gaussx's (`gaussx.laplace_model` for a GMRF prior,
`gaussx.newton_update` for site naturals); the loop that iterates it to a
tolerance is here, as a ``lax.while_loop``, with the mode differentiated
implicitly. Three entry points cover the three forms the stack uses:

- `laplace_mode`: precision form, a GMRF prior (gaussx's own, and pyrox-lgm's);
- `laplace_mode_covariance`: covariance form, a prior covariance K_XX
  (pyrox-gp's dense GPs);
- `laplace_mode_sites`: site form, where the caller supplies the posterior mean
  given site naturals (pyrox-gp's Markov GPs, through a Kalman smoother).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from argminax._src.curvature import Laplace
from argminax._src.solve._tolerance import floored_tolerance
from argminax._src.solve._tolerant import TolerantLinearSolver
from argminax.exceptions import ValidationError


__all__ = [
    "CovarianceLaplaceResult",
    "LaplaceModeResult",
    "SiteLaplaceResult",
    "laplace_mode",
    "laplace_mode_covariance",
    "laplace_mode_sites",
]


_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def _validate(max_iter: int, tol: float, damping: float) -> None:
    """Check the loop settings every entry point shares."""
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        msg = f"'max_iter' must be a positive integer, got {max_iter!r}"
        raise ValidationError(msg)
    if not tol > 0:
        msg = f"'tol' must be positive, got {tol!r}"
        raise ValidationError(msg)
    if not 0 < damping <= 1:
        msg = f"'damping' must lie in (0, 1], got {damping!r}"
        raise ValidationError(msg)


def _stop_gradient(tree: Any) -> Any:
    """Stop gradients through the inexact arrays of a pytree."""
    dynamic, static = eqx.partition(tree, eqx.is_inexact_array)
    frozen = jax.lax.stop_gradient(dynamic)
    return eqx.combine(frozen, static)


def _sup_norm(vector: Float[Array, " state"]) -> Float[Array, ""]:
    """‖v‖_∞, zero for an empty vector."""
    magnitude = jnp.abs(vector)
    # einx has no initial value, and an empty state must give 0, not an error
    return jnp.max(magnitude, initial=0.0)


def _site_derivatives(
    log_likelihood: Callable[[Float[Array, " state"]], Float[Array, " state"]],
    state: Float[Array, " state"],
) -> tuple[Float[Array, " state"], Float[Array, " state"]]:
    """g = ∇_f log p and h = diag(∇²_f log p) of a site-factorising likelihood.

    Site n depends on fₙ alone, so the Hessian of ∑ₙ log pₙ is diagonal and
    its product with a vector of ones is that diagonal.
    """

    def total(latent: Float[Array, " state"]) -> Float[Array, ""]:
        site_log_densities = log_likelihood(latent)
        # ∑ₙ log pₙ
        return einx.sum("state ->", site_log_densities)

    gradient_fn = jax.grad(total)
    # g = ∇ ∑ₙ log pₙ
    gradient = gradient_fn(state)
    ones = jnp.ones_like(state)
    # h = ∇²(∑ₙ log pₙ) 1, the diagonal for a factorised likelihood
    _, curvature = jax.jvp(gradient_fn, (state,), (ones,))
    return gradient, curvature


def _sites(
    log_likelihood: Callable[[Float[Array, " state"]], Float[Array, " state"]],
    state: Float[Array, " state"],
    precision_floor: float,
) -> tuple[Float[Array, " state"], Float[Array, " state"]]:
    """The Newton site naturals (Λμ_site, Λ_site) at f, by `gaussx.newton_update`."""
    gradient, curvature = _site_derivatives(log_likelihood, state)
    # Λ_site = max(−h, floor), Λμ_site = g + Λ_site ⊙ f
    information, precision = gx.newton_update(
        state, gradient, curvature, precision_floor=precision_floor
    )
    return information, precision


def _start_at_estimate(_: Any, start: Array) -> Array:
    """custom_root's forward solve: the loop has converged, so keep its estimate."""
    return start


def _default_adjoint(dtype: Any) -> optx.ImplicitAdjoint:
    """safe_implicit_adjoint(GMRES(rtol=1e-8, atol=1e-8, restart=20, max_steps=50)).

    The tolerances are floored at 64 machine epsilons of the dtype, as the
    loop's are: below that a float32 solve cannot converge and would spend its
    whole budget on rounding error.
    """
    # imported here: argminax._src.adjoint imports argminax._src.solve._tolerant,
    # so a module-level import would be circular
    from argminax._src.adjoint import safe_implicit_adjoint

    tolerance = floored_tolerance(1e-8, dtype)
    solver = lx.GMRES(rtol=tolerance, atol=tolerance, restart=20, max_steps=50)
    return safe_implicit_adjoint(solver)


def _resolve_adjoint(
    adjoint: optx.ImplicitAdjoint | None, dtype: Any
) -> optx.ImplicitAdjoint:
    """The caller's implicit adjoint, or the matrix-free default."""
    if adjoint is None:
        return _default_adjoint(dtype)
    if not isinstance(adjoint, optx.ImplicitAdjoint):
        msg = (
            "'adjoint' must be an optimistix ImplicitAdjoint (for example "
            "argminax.adjoint.safe_implicit_adjoint(...)), "
            f"got {type(adjoint).__name__}"
        )
        raise ValidationError(msg)
    return adjoint


class _NewtonGuessSolver(lx.AbstractLinearSolver):
    """Solve (J_T − I) v = b by first trying v = −b, exact when J_T = 0.

        r = b − (J_T − I)(−b) = J_T b
        v = −b               if |rᵢ| ≤ atol + rtol |bᵢ| for every i
        v = inner(b)         otherwise, started from −b

    One product checks the guess; ``inner`` runs only when it is not within
    the tolerance of ``inner`` itself. It is a lineax solver, so lineax keeps
    treating the solve as linear in b when it transposes or differentiates it.

    Attributes:
        inner: The iterative solver for the system.
        rtol: Its relative tolerance.
        atol: Its absolute tolerance.
    """

    inner: lx.AbstractLinearSolver
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)

    def init(self, operator: lx.AbstractLinearOperator, options: dict[str, Any]) -> Any:
        return operator, self.inner.init(operator, options)

    def compute(
        self, state: Any, vector: Array, options: dict[str, Any]
    ) -> tuple[Array, lx.RESULTS, dict[str, Any]]:
        operator, inner_state = state
        # v = −b, exact when J_T = 0
        guess = -vector
        # r = b − (J_T − I)(−b) = J_T b
        residual = vector - operator.mv(guess)
        # atol + rtol |bᵢ|, the inner solver's own scale
        scale = self.atol + self.rtol * jnp.abs(vector)
        accepted = _sup_norm(residual / scale) <= 1.0

        def keep_guess(operand: None) -> tuple[Array, lx.RESULTS]:
            return guess, lx.RESULTS.successful

        def solve(operand: None) -> tuple[Array, lx.RESULTS]:
            started = {**options, "y0": guess}
            solution, result, _stats = self.inner.compute(inner_state, vector, started)
            return solution, result

        solution, result = jax.lax.cond(accepted, keep_guess, solve, None)
        return solution, result, {}

    def transpose(self, state: Any, options: dict[str, Any]) -> Any:
        operator, inner_state = state
        inner_transposed, inner_options = self.inner.transpose(inner_state, options)
        return (operator.transpose(), inner_transposed), inner_options

    def conj(self, state: Any, options: dict[str, Any]) -> Any:
        operator, inner_state = state
        inner_conjugated, inner_options = self.inner.conj(inner_state, options)
        return (operator.conj(), inner_conjugated), inner_options

    def assume_full_rank(self) -> bool:
        return self.inner.assume_full_rank()


def _tangent_solver(solver: lx.AbstractLinearSolver) -> lx.AbstractLinearSolver:
    """The adjoint's solver, behind the guess v = −b when it is iterative.

    An iterative lineax solver is recognised by its ``rtol`` and ``atol``
    (through a `TolerantLinearSolver`); a direct solver is exact and is used
    as it is.
    """
    inner = solver.inner if isinstance(solver, TolerantLinearSolver) else solver
    rtol = getattr(inner, "rtol", None)
    atol = getattr(inner, "atol", None)
    if rtol is None or atol is None:
        return solver
    return _NewtonGuessSolver(solver, rtol, atol)


def _implicit_fixed_point(
    update: Callable[[Float[Array, " state"]], Float[Array, " state"]],
    estimate: Float[Array, " state"],
    adjoint: optx.ImplicitAdjoint,
) -> Float[Array, " state"]:
    """Attach implicit tangents to a fixed point f̂ = μ_f∣y(f̂).

        ∂f̂/∂θ = (I − J_T)⁻¹ ∂μ_f∣y/∂θ,   J_T = ∂μ_f∣y/∂f

    μ_f∣y(f) is ``update``: the posterior mean under the sites at f, with θ
    every array it closes over. The root is μ_f∣y(f) − f, whose Jacobian
    J_T − I is not symmetric in general, so each tangent solve goes to the
    adjoint's linear solver, given only products with the Jacobian (GMRES by
    default). For an exact Newton map J_T = 0 at the mode, so an iterative
    solver is first offered v = −b and runs only when that guess misses its
    tolerance (`_NewtonGuessSolver`): floored sites make the map
    quasi-Newton, with J_T small but not zero.
    """
    solver = _tangent_solver(adjoint.linear_solver)

    def root(state: Float[Array, " state"]) -> Float[Array, " state"]:
        # μ_f∣y(f) − f
        return update(state) - state

    def tangent_solve(matvec: Callable, vector: Float[Array, " state"]) -> Array:
        structure = jax.eval_shape(lambda: vector)
        operator = lx.FunctionLinearOperator(matvec, structure)
        solution = lx.linear_solve(operator, vector, solver)
        return solution.value

    frozen_estimate = jax.lax.stop_gradient(estimate)
    return jax.lax.custom_root(root, frozen_estimate, _start_at_estimate, tangent_solve)


# ---------------------------------------------------------------------------
# Precision form: a GMRF prior, gaussx's Newton step
# ---------------------------------------------------------------------------


class LaplaceModeResult(eqx.Module):
    """The Gaussian 𝒩(x̂, Λ_x∣y⁻¹) at the mode, with gaussx's `LaplaceResult` fields.

    The fields are those of ``gaussx.LaplaceResult``, so code that reads it
    (``gaussx.vb_mean_correction``, pyrox-lgm's ``inla()``) works on this
    result unchanged; `posterior` gives the same Gaussian as an
    `argminax.curvature.Laplace`.

    Attributes:
        mode: The mode x̂, implicitly differentiable in every array the prior
            and likelihood hold.
        hessian: Λ_x∣y = Λ_x + Hᵀ Λ_site H at the mode, in gaussx's structure.
        factor: The factorisation of Λ_x∣y, with ``solve(b)`` and ``logdet()``.
        log_marginal: The Laplace approximation of log p(y ∣ θ).
        n_iter: Newton steps taken.
        converged: Whether the last step met the tolerance; implicit gradients
            assume it.
    """

    mode: Float[Array, " state"]
    hessian: lx.AbstractLinearOperator
    factor: Any
    log_marginal: Float[Array, ""]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]

    @property
    def posterior(self) -> Laplace:
        """The Gaussian at the mode as an `argminax.curvature.Laplace`."""
        precision = self.hessian
        if not lx.is_positive_semidefinite(precision):
            precision = lx.TaggedLinearOperator(precision, _PSD)
        return Laplace(self.mode, precision)


def _damped_newton(
    target: Callable[[Float[Array, " state"]], Float[Array, " state"]],
    start: Float[Array, " state"],
    max_iter: int,
    tol: float,
    damping: float,
) -> tuple[Float[Array, " state"], Int[Array, ""], Bool[Array, ""]]:
    """x ← x + α (x_Newton − x) until ‖Δx‖_∞ ≤ tol (1 + ‖x‖_∞), or ``max_iter``."""

    def threshold(state: Float[Array, " state"]) -> Float[Array, ""]:
        size = _sup_norm(state)
        # tol (1 + ‖x‖_∞)
        return tol * (1.0 + size)

    def keep_going(carry: tuple[Array, Array, Array]) -> Array:
        state, iteration, step_norm = carry
        return (iteration < max_iter) & (step_norm > threshold(state))

    def step(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        state, iteration, _ = carry
        newton_state = target(state)
        # x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ + α (x_Newton − x⁽ⁱ⁾)
        next_state = state + damping * (newton_state - state)
        # ‖Δx‖_∞
        step_norm = _sup_norm(next_state - state)
        return next_state, iteration + 1, step_norm

    initial_norm = jnp.asarray(jnp.inf, dtype=start.dtype)
    initial = (start, jnp.asarray(0), initial_norm)
    state, n_iter, step_norm = jax.lax.while_loop(keep_going, step, initial)
    converged = step_norm <= threshold(state)
    return state, n_iter, converged


def _implicit_mode(
    model: Any, estimate: Float[Array, " state"], factor: Any
) -> Float[Array, " state"]:
    """Attach implicit tangents to the mode: ∂x̂/∂θ = Λ_x∣y⁻¹ ∂θ(−∇𝒥), one solve.

    The root is −∇𝒥(x), or with a hard constraint −P ∇𝒥(P x) − Π_V x
    (P = I − Π_V), whose Jacobian −(P Λ_x∣y P + Π_V) is symmetric; tangents are
    solved with the factor of Λ_x∣y at the mode, kriged for the constraint.
    """
    constrained = model.null_space is not None
    frozen = _stop_gradient(model)

    def root(state: Array) -> Array:
        if not constrained:
            # −∇𝒥(x)
            return model.grad(state)
        # P x = x − Π_V x
        inner = state - model.null_component(state)
        # −∇𝒥(P x)
        ascent = model.grad(inner)
        # −P ∇𝒥(P x) − Π_V x
        return ascent - model.null_component(ascent) - model.null_component(state)

    def negative_inverse(_: Any, vector: Array) -> Array:
        # Π_V b
        null = frozen.null_component(vector)
        # Λ_x∣y⁻¹ (b − Π_V b)
        unconstrained = factor.solve(vector - null)
        # krige(Λ_x∣y⁻¹ (b − Π_V b))
        inner = frozen.krige(factor, unconstrained)[0]
        # −(P Λ_x∣y P + Π_V)⁻¹ b
        return -(inner + null)

    def tangent_solve(matvec: Callable, vector: Array) -> Array:
        return jax.lax.custom_linear_solve(
            matvec, vector, negative_inverse, symmetric=True
        )

    frozen_estimate = jax.lax.stop_gradient(estimate)
    return jax.lax.custom_root(root, frozen_estimate, _start_at_estimate, tangent_solve)


def laplace_mode(
    prior: Any,
    likelihood: Any,
    *,
    projector: lx.AbstractLinearOperator | None = None,
    offset: Float[ArrayLike, " obs"] | Float[ArrayLike, ""] | None = None,
    init: Float[ArrayLike, " state"] | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    damping: float = 1.0,
) -> LaplaceModeResult:
    """Mode and Gaussian approximation of a latent Gaussian model, in precision form.

        Λ_x∣y⁽ⁱ⁾ x_Newton = Λ_x μ_x + Hᵀ (g⁽ⁱ⁾ + Λ_site⁽ⁱ⁾ H x⁽ⁱ⁾)
        x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ + α (x_Newton − x⁽ⁱ⁾)

    The model is x ~ ``prior`` (a gaussx ``GaussianMRF`` or ``IntrinsicGMRF``
    over x ∈ ℝ^D_x, precision Λ_x, mean μ_x) and y ∣ H x + o ~ ``likelihood``,
    a site-factorising gaussx likelihood holding y ∈ ℝ^D_y. At iteration i,
    g⁽ⁱ⁾ = ∇ log p(y ∣ H x⁽ⁱ⁾ + o) and Λ_site⁽ⁱ⁾ = −∇² of the same, per site.
    This is gaussx's ``laplace_mode`` moved here (jejjohnson/gaussx#814), with
    the same signature and result: each step is `gaussx.LaplaceModel.newton`,
    Λ_x∣y keeps the structure gaussx gives it (banded, sparse, diagonal or
    dense), and a hard constraint Vᵀ x = 0 is kriged at every step.

    Algorithm:
        1. Build the model (`gaussx.laplace_model`) and project the start off
           the null space.
        2. Run the damped Newton loop, a ``lax.while_loop`` with gradients
           stopped, until ‖Δx‖_∞ ≤ tol (1 + ‖x‖_∞), with tol floored at 64
           machine epsilons of the dtype.
        3. Attach implicit tangents to the mode (``jax.lax.custom_root``),
           solved with the factor of Λ_x∣y at the mode: one solve per VJP.
        4. Refactorise at the mode and assemble the log-marginal
           (`gaussx.laplace_result`).

    Args:
        prior: The latent field, a gaussx ``GaussianMRF`` or ``IntrinsicGMRF``.
        likelihood: A site-factorising gaussx likelihood holding y.
        projector: The projector H, (D_y, D_x); ``None`` is the identity.
        offset: The offset o, a scalar or of length D_y.
        init: Starting point x⁽⁰⁾; zeros by default.
        max_iter: Maximum Newton steps.
        tol: Relative tolerance on the step's sup-norm.
        damping: Step size α in (0, 1].

    Returns:
        The `LaplaceModeResult`, with gaussx's ``LaplaceResult`` fields and a
        `posterior` property.

    Raises:
        ValidationError: If ``max_iter``, ``tol`` or ``damping`` is out of range.

    Examples:
        Poisson counts on an independent Gaussian prior:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> from argminax.solve import laplace_mode
        >>> prior = gaussx.GaussianMRF(
        ...     jnp.zeros(3), lx.DiagonalLinearOperator(jnp.ones(3))
        ... )
        >>> counts = gaussx.PoissonLikelihood(jnp.array([0.0, 1.0, 4.0]))
        >>> fit = laplace_mode(prior, counts)
        >>> bool(fit.converged), fit.mode.shape
        (True, (3,))
        >>> fit.posterior.covariance().mv(jnp.ones(3)).shape
        (3,)

        Damping slows the loop but does not move the mode:

        >>> damped = laplace_mode(prior, counts, damping=0.5)
        >>> bool(jnp.allclose(damped.mode, fit.mode, atol=1e-6))
        True

        A damping outside (0, 1] is rejected before any work is done:

        >>> laplace_mode(prior, counts, damping=0.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'damping' must lie in (0, 1], got 0.0
    """
    _validate(max_iter, tol, damping)

    # --- the model and its start ---
    model = gx.laplace_model(prior, likelihood, projector=projector, offset=offset)
    dtype = prior.loc.dtype
    size = prior.event_shape[0]
    start = jnp.zeros(size, dtype) if init is None else jnp.asarray(init, dtype)
    # Π_V x⁽⁰⁾
    start_null = model.null_component(start)
    # x⁽⁰⁾ ← x⁽⁰⁾ − Π_V x⁽⁰⁾
    start = start - start_null

    # --- the loop, with gradients stopped ---
    frozen_model = _stop_gradient(model)
    frozen_start = jax.lax.stop_gradient(start)
    floored_tol = floored_tolerance(tol, dtype)
    estimate, n_iter, converged = _damped_newton(
        frozen_model.newton, frozen_start, max_iter, floored_tol, damping
    )

    # --- implicit tangents, with the factor at the mode ---
    frozen_prior = _stop_gradient(prior)
    frozen_fit = gx.laplace_result(
        frozen_model, frozen_prior, estimate, n_iter, converged
    )
    mode = _implicit_mode(model, estimate, frozen_fit.factor)

    # --- the Gaussian and the log-marginal at the mode ---
    fit = gx.laplace_result(model, prior, mode, n_iter, converged)
    return LaplaceModeResult(
        mode=fit.mode,
        hessian=fit.hessian,
        factor=fit.factor,
        log_marginal=fit.log_marginal,
        n_iter=fit.n_iter,
        converged=fit.converged,
    )


# ---------------------------------------------------------------------------
# Site naturals: shared by the covariance and site forms
# ---------------------------------------------------------------------------


class SiteLaplaceResult(eqx.Module):
    """The mode of a latent GP and the Newton site naturals there.

    Attributes:
        mode: The mode f̂, implicitly differentiable through the fixed point
            f̂ = μ_f∣y(f̂) of the posterior-mean map.
        site_information: The site information Λμ_site = g + Λ_site ⊙ f̂.
        site_precision: The site precision Λ_site = max(−h, floor).
        n_iter: Iterations taken.
        converged: Whether the last step met the tolerance.
    """

    mode: Float[Array, " state"]
    site_information: Float[Array, " state"]
    site_precision: Float[Array, " state"]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]


class CovarianceLaplaceResult(eqx.Module):
    """The Gaussian at the mode of a GP with prior covariance K_XX.

        q(f) = 𝒩(μ_f∣y, Σ_f∣y)
        Σ_f∣y = (K_XX⁻¹ + Λ_site)⁻¹ = K_XX − K_XX (K_XX + Λ_site⁻¹)⁻¹ K_XX

    with f ∈ ℝ^N the latent values at the N training inputs X, K_XX (N × N)
    the prior covariance and Λ_site the diagonal matrix of site precisions.

    Attributes:
        mode: The mode f̂ (the last iterate), implicitly differentiable.
        site_information: The site information Λμ_site at f̂.
        site_precision: The site precision Λ_site at f̂.
        mean: The posterior mean μ_f∣y given the sites at f̂, equal to f̂ at
            convergence.
        variance: diag(Σ_f∣y), the marginal variances; ``None`` when the fit
            was made with ``with_variance=False``.
        log_marginal: The Laplace log-marginal by `gaussx.site_log_evidence`,
            with the objective evaluated exactly at f₁ = ``mean`` and the
            sites at f̂; ``None`` when the fit was made with
            ``with_log_marginal=False``. For an exact solver it is the
            standard form
            ∑ₙ log p(yₙ ∣ f₁ₙ) − ½ (f₁ − μ_f)ᵀ K_XX⁻¹ (f₁ − μ_f)
            − ½ log |I + Λ_site^½ K_XX Λ_site^½|,
            and for a stochastic log-determinant the equal
            pseudo-observation form, which needs only solves against and the
            log-determinant of A = K_XX + Λ_site⁻¹.
        n_iter: Iterations taken.
        converged: Whether the last step met the tolerance.
        prior_cov: The prior covariance K_XX the fit was made with.
        site_posterior: The Gaussian at f̂ as a `gaussx.SitePosterior`, which
            solves through the fit's solver strategy: its ``variance`` gives
            diag(Σ_f∣y) later, and its ``covariance`` Σ_f∣y as a lazy
            operator.
    """

    mode: Float[Array, " state"]
    site_information: Float[Array, " state"]
    site_precision: Float[Array, " state"]
    mean: Float[Array, " state"]
    variance: Float[Array, " state"] | None
    log_marginal: Float[Array, ""] | None
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]
    prior_cov: lx.AbstractLinearOperator
    site_posterior: gx.SitePosterior

    @property
    def posterior(self) -> Laplace:
        """𝒩(f̂, Σ_f∣y) as a `Laplace`, with precision K_XX⁻¹ + Λ_site.

        Built on access, from lazy operators: nothing is densified until the
        `Laplace` is used.
        """
        # K_XX⁻¹
        inverse = gx.inv(self.prior_cov)
        # Λ_site as a diagonal matrix
        sites = lx.DiagonalLinearOperator(self.site_precision)
        # K_XX⁻¹ + Λ_site
        untagged = inverse + sites
        precision = lx.TaggedLinearOperator(untagged, _PSD)
        return Laplace(self.mode, precision)


def _site_loop(
    update: Callable[[Float[Array, " state"], Any], tuple[Array, Any]],
    start: tuple[Float[Array, " state"], Any],
    max_iter: int,
    tol: float,
) -> tuple[Float[Array, " state"], Any, Int[Array, ""], Bool[Array, ""]]:
    """Iterate (f, carry) ← update(f, carry) until ‖Δf‖_∞ < tol."""

    def keep_going(state: tuple[Array, Any, Array, Array]) -> Array:
        _, _, iteration, step_norm = state
        return (iteration < max_iter) & (step_norm >= tol)

    def step(state: tuple[Array, Any, Array, Array]) -> tuple[Array, Any, Array, Array]:
        latent, carry, iteration, _ = state
        next_latent, next_carry = update(latent, carry)
        # ‖Δf‖_∞
        step_norm = _sup_norm(next_latent - latent)
        return next_latent, next_carry, iteration + 1, step_norm

    latent, carry = start
    initial_norm = jnp.asarray(jnp.inf, latent.dtype)
    initial = (latent, carry, jnp.asarray(0), initial_norm)
    latent, carry, n_iter, step_norm = jax.lax.while_loop(keep_going, step, initial)
    converged = step_norm < tol
    return latent, carry, n_iter, converged


def laplace_mode_covariance(
    prior_cov: lx.AbstractLinearOperator,
    prior_mean: Float[Array, " state"],
    log_likelihood: Callable[[Float[Array, " state"]], Float[Array, " state"]],
    *,
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 1.0,
    precision_floor: float = 1e-6,
    adjoint: optx.ImplicitAdjoint | None = None,
    solver: gx.AbstractSolverStrategy | None = None,
    with_variance: bool = True,
    with_log_marginal: bool = True,
    key: Array | None = None,
) -> CovarianceLaplaceResult:
    """Laplace approximation of a GP with prior covariance K_XX, by damped Newton.

        f_Newton = μ_f + K_XX A⁻¹ (μ_site − μ_f),   A = K_XX + Λ_site⁻¹
        f ← (1 − α) f + α f_Newton

    f ∈ ℝ^N are the latent values at the N training inputs X, with prior mean
    μ_f and covariance K_XX. The sites (Λμ_site, Λ_site) = `gaussx.newton_update`
    at f, with Λ_site = max(−h, floor), g = ∇_f log p and h = diag(∇²_f log p),
    are a Gaussian pseudo-likelihood with mean μ_site = Λμ_site / Λ_site, and
    f_Newton is the posterior mean under it (Rasmussen and Williams 2006,
    algorithm 3.1), by `gaussx.site_posterior`. Every solve against A goes
    through ``solver``, so with an iterative strategy and a matrix-free K_XX
    no N × N matrix is formed. This is the loop of pyrox-gp's
    ``LaplaceInference`` and ``GaussNewtonInference`` (which differ only in
    their floor), as a ``lax.while_loop``.

    ``solver=None`` (or ``gaussx.DenseSolver``) factors A densely and gives
    the standard, exact log-marginal; ``gaussx.CGSolver``, with or without a
    preconditioner, and ``gaussx.BBMMSolver`` solve by CG and give the
    pseudo-observation form with a stochastic log-determinant (the solver
    table is on the ``argminax.solve`` API page).

    **Inexact Newton.** An iterative solver leaves a residual in every step,
    and the test ‖Δf‖_∞ < tol cannot resolve below it: give the solver an
    rtol (and atol) at least ten times tighter than ``tol``, relative to the
    scale of f. A strategy built with ``throw=True`` (gaussx's default)
    raises when it runs out of steps, from inside the loop; inside a sampler
    such as NUTS build it with ``throw=False``, and ``converged`` reports the
    failure.

    Algorithm:
        1. From f = μ_f, iterate the damped update until ‖Δf‖_∞ < tol (tol
           floored at 64 machine epsilons of the dtype) or ``max_iter``; each
           step is `gaussx.site_posterior` with ``solver``.
        2. Make the mode implicit through its fixed point f̂ = μ_f∣y(f̂),
           with the tangent solve of ``adjoint`` (matrix-free by default).
        3. At f̂: the sites and the posterior mean (`gaussx.site_posterior`),
           the marginal variances (``SitePosterior.variance``) and the
           Laplace log-marginal (`gaussx.site_log_evidence`), each when asked
           for.

    Args:
        prior_cov: K_XX, (N, N), tagged positive semidefinite (jitter included).
        prior_mean: μ_f, shape (N,).
        log_likelihood: f ↦ (log p(yₙ ∣ fₙ))ₙ, one log-density per site; site
            n must depend on fₙ alone.
        max_iter: Maximum iterations.
        tol: Absolute tolerance on ‖Δf‖_∞.
        damping: Step size α in (0, 1].
        precision_floor: The floor on Λ_site; raise it (pyrox-gp's
            ``GaussNewtonInference`` uses 1e-3) for non-log-concave
            likelihoods.
        adjoint: How the mode is differentiated, an ``optx.ImplicitAdjoint``
            whose linear solver solves (I − J_T)ᵀ v = ∇_f̂ L, with J_T the
            Jacobian of the fixed-point map, from products with J_T alone.
            ``None`` is ``argminax.adjoint.safe_implicit_adjoint(lx.GMRES(
            rtol=1e-8, atol=1e-8, restart=20, max_steps=50))``, tolerances
            floored at 64 machine epsilons of the dtype, which forms no N × N
            matrix and never raises (a solve that stops early keeps its last
            iterate). J_T = 0 at an unfloored mode, so an iterative solver is
            first offered the exact answer for that case and is called only
            when it misses the solver's tolerance. Pass
            ``optx.ImplicitAdjoint(lx.AutoLinearSolver(well_posed=True))``
            for the dense LU used before, or a tighter GMRES for a hard
            problem; an rtol far below the accuracy of the solves in the loop
            only spends steps.
        solver: The gaussx solver strategy for every solve against
            A = K_XX + Λ_site⁻¹, in the loop, in the tangents and at the
            mode. ``None`` is one dense Cholesky of A per step (a structured
            K_XX is densified, with a ``gaussx.DenseFallbackWarning``).
        with_variance: Whether to compute ``variance``: N solves against A
            with an iterative solver, one at a time. Pass ``False`` for a
            large N, and ask ``site_posterior.variance`` later.
        with_log_marginal: Whether to compute ``log_marginal``.
        key: PRNG key of a stochastic log-determinant's probes in
            ``log_marginal``; ``None`` uses the solver's own seed. Ignored by
            an exact solver.

    Returns:
        The `CovarianceLaplaceResult`.

    Raises:
        ValidationError: If ``prior_cov`` is not tagged positive semidefinite,
            ``max_iter``, ``tol`` or ``damping`` is out of range, ``adjoint``
            is not an ``optx.ImplicitAdjoint`` or ``solver`` is not a gaussx
            solver strategy.

    Examples:
        Bernoulli labels on an independent prior:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from argminax.solve import laplace_mode_covariance
        >>> labels = jnp.array([0.0, 1.0, 1.0])
        >>> def log_likelihood(f):
        ...     return labels * f - jax.nn.softplus(f)
        >>> prior = lx.MatrixLinearOperator(
        ...     jnp.eye(3), lx.positive_semidefinite_tag
        ... )
        >>> fit = laplace_mode_covariance(prior, jnp.zeros(3), log_likelihood)
        >>> bool(fit.converged), bool(jnp.allclose(fit.mode, fit.mean, atol=1e-5))
        (True, True)

        The marginal variances are the diagonal of the Gaussian at the mode:

        >>> covariance = fit.posterior.covariance().as_matrix()
        >>> bool(jnp.allclose(jnp.diag(covariance), fit.variance))
        True

        Conjugate gradients reach the same mode, with tolerances well below
        the loop's:

        >>> import gaussx
        >>> conjugate = gaussx.CGSolver(rtol=1e-10, atol=1e-10)
        >>> iterative = laplace_mode_covariance(
        ...     prior, jnp.zeros(3), log_likelihood, solver=conjugate
        ... )
        >>> bool(jnp.allclose(iterative.mode, fit.mode, atol=1e-6))
        True

        BBMM gives a stochastic log-marginal from ``key``; the variances can
        wait until they are needed:

        >>> import jax.random as jr
        >>> bbmm = gaussx.BBMMSolver(rtol=1e-10, atol=1e-10, num_probes=8)
        >>> stochastic = laplace_mode_covariance(
        ...     prior,
        ...     jnp.zeros(3),
        ...     log_likelihood,
        ...     solver=bbmm,
        ...     with_variance=False,
        ...     key=jr.key(0),
        ... )
        >>> stochastic.variance is None, stochastic.log_marginal.shape
        (True, ())
        >>> later = stochastic.site_posterior.variance(method="exact")
        >>> bool(jnp.allclose(later, fit.variance, atol=1e-6))
        True

        The mode is differentiable, by default through a matrix-free GMRES
        tangent solve; the dense LU gives the same gradient:

        >>> import optimistix as optx
        >>> def middle_mode(scale, adjoint=None):
        ...     prior = lx.MatrixLinearOperator(
        ...         scale * jnp.eye(3), lx.positive_semidefinite_tag
        ...     )
        ...     fit = laplace_mode_covariance(
        ...         prior, jnp.zeros(3), log_likelihood, adjoint=adjoint
        ...     )
        ...     return fit.mode[1]
        >>> dense = optx.ImplicitAdjoint(lx.AutoLinearSolver(well_posed=True))
        >>> gradient = jax.grad(middle_mode)(2.0)
        >>> bool(jnp.allclose(gradient, jax.grad(middle_mode)(2.0, dense)))
        True

        A prior covariance without the positive semidefinite tag is rejected:

        >>> untagged = lx.MatrixLinearOperator(jnp.eye(3))
        >>> laplace_mode_covariance(untagged, jnp.zeros(3), log_likelihood)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'prior_cov' must be tagged positive semidefinite

    References:
        Rasmussen, C. E. & Williams, C. K. I. (2006). Gaussian Processes for
        Machine Learning, section 3.4.

        Gyger, T., Furrer, R. & Sigrist, F. (2026). JMLR 27(115), 1–60,
        arXiv:2507.05064: preconditioned CG and stochastic Lanczos
        quadrature in place of the dense steps.
    """
    _validate(max_iter, tol, damping)
    if not lx.is_positive_semidefinite(prior_cov):
        msg = "'prior_cov' must be tagged positive semidefinite, got tags without it"
        raise ValidationError(msg)
    if solver is not None and not isinstance(solver, gx.AbstractSolverStrategy):
        msg = (
            "'solver' must be a gaussx solver strategy (for example "
            f"gaussx.CGSolver()) or None, got {type(solver).__name__}"
        )
        raise ValidationError(msg)
    prior_mean = jnp.asarray(prior_mean)
    adjoint = _resolve_adjoint(adjoint, prior_mean.dtype)

    def fixed_point(latent: Array) -> Array:
        posterior = _site_posterior(
            latent, prior_cov, prior_mean, log_likelihood, precision_floor, solver
        )
        return posterior.mean

    # --- the loop, with gradients stopped ---
    # the loop runs with every input frozen, closed-over arrays included, so
    # gradients come only from the implicit fixed point below
    closed_likelihood = eqx.filter_closure_convert(log_likelihood, prior_mean)
    frozen_likelihood = _stop_gradient(closed_likelihood)
    frozen_cov, frozen_mean, frozen_solver = _stop_gradient(
        (prior_cov, prior_mean, solver)
    )

    def update(latent: Array, carry: None) -> tuple[Array, None]:
        posterior = _site_posterior(
            latent,
            frozen_cov,
            frozen_mean,
            frozen_likelihood,
            precision_floor,
            frozen_solver,
        )
        # f ← (1 − α) f + α f_Newton
        damped = (1.0 - damping) * latent + damping * posterior.mean
        return damped, carry

    floored_tol = floored_tolerance(tol, prior_mean.dtype)
    estimate, _, n_iter, converged = _site_loop(
        update, (frozen_mean, None), max_iter, floored_tol
    )
    mode = _implicit_fixed_point(fixed_point, estimate, adjoint)

    # --- the Gaussian and the log-marginal at f̂ ---
    posterior = _site_posterior(
        mode, prior_cov, prior_mean, log_likelihood, precision_floor, solver
    )
    variance = None
    if with_variance:
        # "exact" never needs a key: N solves for an iterative solver, where
        # "auto" would turn stochastic above N = 2048
        variance = posterior.variance(method="exact")
    log_marginal = None
    if with_log_marginal:
        # the objective at f₁ = μ_f∣y, where gaussx's identity
        # K_XX⁻¹ (f₁ − μ_f) = w holds exactly; Ψ is stationary at the mode, so
        # this costs a second-order error in the mode's, not a first-order one
        site_log_densities = log_likelihood(posterior.mean)
        # ∑ₙ log p(yₙ ∣ f₁ₙ)
        log_density = einx.sum("state ->", site_log_densities)
        log_marginal = gx.site_log_evidence(posterior, log_density, key=key)
    return CovarianceLaplaceResult(
        mode=mode,
        site_information=posterior.site_information,
        site_precision=posterior.site_precision,
        mean=posterior.mean,
        variance=variance,
        log_marginal=log_marginal,
        n_iter=n_iter,
        converged=converged,
        prior_cov=prior_cov,
        site_posterior=posterior,
    )


def _site_posterior(
    latent: Float[Array, " state"],
    prior_cov: lx.AbstractLinearOperator,
    prior_mean: Float[Array, " state"],
    log_likelihood: Callable[[Float[Array, " state"]], Float[Array, " state"]],
    precision_floor: float,
    solver: gx.AbstractSolverStrategy | None,
) -> gx.SitePosterior:
    """The Gaussian under the Newton sites at f, whose mean is f_Newton.

    f_Newton = μ_f + K_XX (K_XX + Λ_site⁻¹)⁻¹ (μ_site − μ_f)
    """
    information, precision = _sites(log_likelihood, latent, precision_floor)
    return gx.site_posterior(
        prior_mean, prior_cov, information, precision, solver=solver
    )


def laplace_mode_sites(
    posterior_mean: Callable[
        [Float[Array, " state"], Float[Array, " state"]], Float[Array, " state"]
    ],
    log_likelihood: Callable[[Float[Array, " state"]], Float[Array, " state"]],
    prior_mean: Float[Array, " state"],
    *,
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 1.0,
    precision_floor: float = 1e-6,
    adjoint: optx.ImplicitAdjoint | None = None,
) -> SiteLaplaceResult:
    """Laplace approximation by damping the site naturals, given a posterior-mean map.

        (Λμ_site, Λ_site) ← (1 − α) (Λμ_site, Λ_site) + α newton_update(f)
        f ← μ_f∣y(Λμ_site, Λ_site)

    ``posterior_mean`` maps site naturals to the posterior mean under the
    Gaussian pseudo-likelihood they define, for whatever prior the caller has:
    for a Markov GP, a Kalman filter and RTS smoother in O(N D_x³), with D_x
    the size of the SDE state, which stays with the caller. Here f ∈ ℝ^N are
    the latent values at the N inputs, with prior mean μ_f. This is the loop of
    pyrox-gp's ``LaplaceMarkovInference`` and ``GaussNewtonMarkovInference``,
    as a ``lax.while_loop``: it damps the naturals
    (`gaussx.damped_natural_update`) rather than the mean.

    Algorithm:
        1. Start from f = μ_f, Λμ_site = floor · μ_f and Λ_site = floor.
        2. Iterate: the Newton sites at f, a damped natural update, Λ_site
           floored, and f from ``posterior_mean``, until ‖Δf‖_∞ < tol (floored
           at 64 machine epsilons) or ``max_iter``.
        3. Make the mode implicit through its fixed point
           f̂ = μ_f∣y(sites(f̂)), with the tangent solve of ``adjoint``,
           and return the undamped sites at f̂.

    Args:
        posterior_mean: (Λμ_site, Λ_site) ↦ μ_f∣y, the posterior mean given
            the site naturals, each of shape (N,).
        log_likelihood: f ↦ (log p(yₙ ∣ fₙ))ₙ, one log-density per site.
        prior_mean: μ_f, shape (N,).
        max_iter: Maximum iterations.
        tol: Absolute tolerance on ‖Δf‖_∞.
        damping: Step size α in (0, 1] on the naturals.
        precision_floor: The floor on Λ_site.
        adjoint: How the mode is differentiated, an ``optx.ImplicitAdjoint``
            whose linear solver solves (I − J_T)ᵀ v = ∇_f̂ L, with J_T the
            Jacobian of the fixed-point map, from products with J_T alone.
            ``None`` is ``argminax.adjoint.safe_implicit_adjoint(lx.GMRES(
            rtol=1e-8, atol=1e-8, restart=20, max_steps=50))``, tolerances
            floored at 64 machine epsilons of the dtype, which forms no N × N
            matrix and never raises (a solve that stops early keeps its last
            iterate). J_T = 0 at an unfloored mode, so an iterative solver is
            first offered the exact answer for that case and is called only
            when it misses the solver's tolerance. Pass
            ``optx.ImplicitAdjoint(lx.AutoLinearSolver(well_posed=True))``
            for the dense LU used before, or a tighter GMRES for a hard
            problem; an rtol far below the accuracy of ``posterior_mean``
            only spends steps.

    Returns:
        The `SiteLaplaceResult`: the mode and the sites at it, from which the
        caller's smoother gives the marginals and the log-marginal.

    Raises:
        ValidationError: If ``max_iter``, ``tol`` or ``damping`` is out of range,
            or ``adjoint`` is not an ``optx.ImplicitAdjoint``.

    Examples:
        With an independent unit prior the posterior mean has a closed form:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from argminax.solve import laplace_mode_sites
        >>> labels = jnp.array([0.0, 1.0])
        >>> def log_likelihood(f):
        ...     return labels * f - jax.nn.softplus(f)
        >>> def posterior_mean(information, precision):
        ...     # μ_f∣y = Λμ_site / (1 + Λ_site) for μ_f = 0 and unit variance
        ...     return information / (1.0 + precision)
        >>> fit = laplace_mode_sites(posterior_mean, log_likelihood, jnp.zeros(2))
        >>> bool(fit.converged), fit.mode.shape
        (True, (2,))

        At the mode, the posterior-mean map returns the mode itself:

        >>> refit = posterior_mean(fit.site_information, fit.site_precision)
        >>> bool(jnp.allclose(refit, fit.mode, atol=1e-5))
        True

        A non-positive iteration budget is rejected:

        >>> laplace_mode_sites(
        ...     posterior_mean, log_likelihood, jnp.zeros(2), max_iter=0
        ... )
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'max_iter' must be a positive integer, got 0
    """
    _validate(max_iter, tol, damping)
    prior_mean = jnp.asarray(prior_mean)
    adjoint = _resolve_adjoint(adjoint, prior_mean.dtype)

    def fixed_point(latent: Array) -> Array:
        information, precision = _sites(log_likelihood, latent, precision_floor)
        # Λ_site ← max(Λ_site, floor)
        floored_precision = jnp.maximum(precision, precision_floor)
        # μ_f∣y(Λμ_site, Λ_site)
        return posterior_mean(information, floored_precision)

    # --- the loop, with gradients stopped ---
    # the loop runs with every input frozen, closed-over arrays included, so
    # gradients come only from the implicit fixed point below
    closed_mean = eqx.filter_closure_convert(posterior_mean, prior_mean, prior_mean)
    frozen_mean = _stop_gradient(closed_mean)
    closed_likelihood = eqx.filter_closure_convert(log_likelihood, prior_mean)
    frozen_likelihood = _stop_gradient(closed_likelihood)

    def update(latent: Array, naturals: tuple[Array, Array]) -> tuple[Array, Any]:
        current_information, current_precision = naturals
        target_information, target_precision = _sites(
            frozen_likelihood, latent, precision_floor
        )
        # (Λμ_site, Λ_site) ← (1 − α) (Λμ_site, Λ_site) + α newton_update(f)
        information, precision = gx.damped_natural_update(
            current_information,
            current_precision,
            target_information,
            target_precision,
            lr=damping,
        )
        # diagonal sites in, diagonal sites out (gaussx types the operator form too)
        diagonal_precision = cast(Array, precision)
        # Λ_site ← max(Λ_site, floor)
        floored_precision = jnp.maximum(diagonal_precision, precision_floor)
        # f ← μ_f∣y(Λμ_site, Λ_site)
        next_latent = frozen_mean(information, floored_precision)
        return next_latent, (information, floored_precision)

    start_mean = jax.lax.stop_gradient(prior_mean)
    # Λμ_site⁽⁰⁾ = floor · μ_f, Λ_site⁽⁰⁾ = floor
    start_sites = (
        start_mean * precision_floor,
        jnp.full_like(start_mean, precision_floor),
    )
    floored_tol = floored_tolerance(tol, prior_mean.dtype)
    estimate, _, n_iter, converged = _site_loop(
        update, (start_mean, start_sites), max_iter, floored_tol
    )
    mode = _implicit_fixed_point(fixed_point, estimate, adjoint)

    # --- the undamped sites at f̂ ---
    information, precision = _sites(log_likelihood, mode, precision_floor)
    # Λ_site ← max(Λ_site, floor)
    floored_precision = jnp.maximum(precision, precision_floor)
    return SiteLaplaceResult(
        mode=mode,
        site_information=information,
        site_precision=floored_precision,
        n_iter=n_iter,
        converged=converged,
    )

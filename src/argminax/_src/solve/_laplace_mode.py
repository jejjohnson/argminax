"""Damped Newton loops to the mode of a latent Gaussian model, and the Gaussian there.

One Newton step's algebra is gaussx's (`gaussx.laplace_model` for a GMRF prior,
`gaussx.newton_update` for site naturals); the loop that iterates it to a
tolerance is here, as a ``lax.while_loop``, with the mode differentiated
implicitly. Three entry points cover the three forms the stack uses:

- `laplace_mode`: precision form, a GMRF prior (gaussx's own, and pyrox-lgm's);
- `laplace_mode_covariance`: covariance form, a prior covariance K (pyrox-gp's
  dense GPs);
- `laplace_mode_sites`: site form, where the caller supplies the posterior mean
  given site naturals (pyrox-gp's Markov GPs, through a Kalman smoother).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from argminax._src.curvature import Laplace
from argminax.exceptions import MissingExtraError, ValidationError


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


def _floored_tolerance(tol: float, dtype: Any) -> float:
    """tol, floored at 64 machine epsilons: below that float32 never converges."""
    return max(tol, 64.0 * float(jnp.finfo(dtype).eps))


def _validate(max_iter: int, tol: float, damping: float) -> None:
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
    return eqx.combine(jax.lax.stop_gradient(dynamic), static)


def _site_derivatives(
    log_likelihood: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    state: Float[Array, " N"],
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    """g and h = diag(∇² log p) of a likelihood that factorises over sites.

    Site n depends on fₙ alone, so the Hessian of ∑ₙ log pₙ is diagonal and
    its product with a vector of ones is that diagonal.
    """

    def total(latent: Float[Array, " N"]) -> Float[Array, ""]:
        return jnp.sum(log_likelihood(latent))

    gradient_fn = jax.grad(total)
    # g = ∇ ∑ₙ log pₙ
    gradient = gradient_fn(state)
    # h = ∇²(∑ₙ log pₙ) 1, the diagonal for a factorised likelihood
    _, curvature = jax.jvp(gradient_fn, (state,), (jnp.ones_like(state),))
    return gradient, curvature


def _sites(
    log_likelihood: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    state: Float[Array, " N"],
    precision_floor: float,
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    """The Newton site naturals (λ, Λ) at f, by `gaussx.newton_update`."""
    gradient, curvature = _site_derivatives(log_likelihood, state)
    # Λ = max(−h, floor), λ = g + Λ ⊙ f
    information, precision = gx.newton_update(
        state, gradient, curvature, precision_floor=precision_floor
    )
    return information, precision


def _implicit_fixed_point(
    update: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    estimate: Float[Array, " N"],
) -> Float[Array, " N"]:
    """Attach implicit tangents to a fixed point f̂ = Φ(f̂).

        ∂f̂/∂θ = (I − ∂Φ/∂f)⁻¹ ∂Φ/∂θ

    The root is Φ(f) − f; its Jacobian is not symmetric in general, so the
    tangent solve is a dense LU on the N × N Jacobian (N products with it).
    """

    def root(state: Float[Array, " N"]) -> Float[Array, " N"]:
        return update(state) - state

    def tangent_solve(matvec: Callable, vector: Float[Array, " N"]) -> Array:
        structure = jax.eval_shape(lambda: vector)
        operator = lx.FunctionLinearOperator(matvec, structure)
        return lx.linear_solve(
            operator, vector, lx.AutoLinearSolver(well_posed=True)
        ).value

    return jax.lax.custom_root(
        root, jax.lax.stop_gradient(estimate), lambda _, start: start, tangent_solve
    )


# ---------------------------------------------------------------------------
# Precision form: a GMRF prior, gaussx's Newton step
# ---------------------------------------------------------------------------


class LaplaceModeResult(eqx.Module):
    """The Gaussian N(x̂, Λ_x|y⁻¹) at the mode, with gaussx's `LaplaceResult` fields.

    The fields are those of ``gaussx.LaplaceResult``, so code that reads it
    (``gaussx.vb_mean_correction``, pyrox-lgm's ``inla()``) works on this
    result unchanged; `posterior` gives the same Gaussian as an
    `argminax.curvature.Laplace`.

    Attributes:
        mode: The mode x̂, implicitly differentiable in every array the prior
            and likelihood hold.
        hessian: Λ_x|y = Λ_x + Aᵀ Λ_site A at the mode, in gaussx's structure.
        factor: The factorisation of Λ_x|y, with ``solve(b)`` and ``logdet()``.
        log_marginal: The Laplace log-marginal log π̃(y ∣ θ).
        n_iter: Newton steps taken.
        converged: Whether the last step met the tolerance; implicit gradients
            assume it.
    """

    mode: Float[Array, " N"]
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


def _require_gmrf_step() -> None:
    if not hasattr(gx, "laplace_model"):
        msg = (
            "laplace_mode's precision form needs the 'numpyro' extra, for "
            "gaussx's GMRF priors and Laplace step: pip install 'argminax[numpyro]'"
        )
        raise MissingExtraError(msg)


def _damped_newton(
    target: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    start: Float[Array, " N"],
    max_iter: int,
    tol: float,
    damping: float,
) -> tuple[Float[Array, " N"], Int[Array, ""], Bool[Array, ""]]:
    """x ← x + α (x̃ − x) until ‖Δx‖_∞ ≤ tol (1 + ‖x‖_∞), or ``max_iter`` steps."""

    def threshold(state: Float[Array, " N"]) -> Float[Array, ""]:
        # tol (1 + ‖x‖_∞)
        return tol * (1.0 + jnp.max(jnp.abs(state), initial=0.0))

    def keep_going(carry: tuple[Array, Array, Array]) -> Array:
        state, iteration, step_norm = carry
        return (iteration < max_iter) & (step_norm > threshold(state))

    def step(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        state, iteration, _ = carry
        # x⁽ʲ⁺¹⁾ = x⁽ʲ⁾ + α (x̃ − x⁽ʲ⁾)
        next_state = state + damping * (target(state) - state)
        # ‖Δx‖_∞
        step_norm = jnp.max(jnp.abs(next_state - state), initial=0.0)
        return next_state, iteration + 1, step_norm

    initial = (start, jnp.asarray(0), jnp.asarray(jnp.inf, dtype=start.dtype))
    state, n_iter, step_norm = jax.lax.while_loop(keep_going, step, initial)
    return state, n_iter, step_norm <= threshold(state)


def _implicit_mode(model: Any, estimate: Float[Array, " N"], factor: Any) -> Array:
    """Attach implicit tangents to the mode: ∂x̂/∂θ = Λ_x|y⁻¹ ∂θ(−∇𝒥), one solve.

    The root is −∇𝒥(x), or with a hard constraint −P ∇𝒥(P x) − Π_V x
    (P = I − Π_V), whose Jacobian −(P Λ_x|y P + Π_V) is symmetric; tangents are
    solved with the factor of Λ_x|y at the mode, kriged for the constraint.
    """
    constrained = model.null_space is not None
    frozen = _stop_gradient(model)

    def root(state: Array) -> Array:
        if not constrained:
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
        # krige(Λ_x|y⁻¹ (b − Π_V b))
        inner = frozen.krige(factor, factor.solve(vector - null))[0]
        return -(inner + null)

    def tangent_solve(matvec: Callable, vector: Array) -> Array:
        return jax.lax.custom_linear_solve(
            matvec, vector, negative_inverse, symmetric=True
        )

    return jax.lax.custom_root(
        root, jax.lax.stop_gradient(estimate), lambda _, start: start, tangent_solve
    )


def laplace_mode(
    prior: Any,
    likelihood: Any,
    *,
    projector: lx.AbstractLinearOperator | None = None,
    offset: Float[ArrayLike, " M"] | Float[ArrayLike, ""] | None = None,
    init: Float[ArrayLike, " N"] | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    damping: float = 1.0,
) -> LaplaceModeResult:
    """Mode and Gaussian approximation of a latent Gaussian model, in precision form.

        Λ_x|y⁽ʲ⁾ x̃ = Λ_x μ_x + Aᵀ (g⁽ʲ⁾ + Λ_site⁽ʲ⁾ A x⁽ʲ⁾)
        x⁽ʲ⁺¹⁾ = x⁽ʲ⁾ + α (x̃ − x⁽ʲ⁾)

    The model is x ~ ``prior`` (a gaussx ``GaussianMRF`` or ``IntrinsicGMRF``,
    precision Λ_x, mean μ_x), η = A x + o and y ∣ η ~ ``likelihood``, a
    site-factorising gaussx likelihood holding y. This is gaussx's
    ``laplace_mode`` moved here (jejjohnson/gaussx#814), with the same
    signature and result: each step is `gaussx.LaplaceModel.newton`, Λ_x|y
    keeps the structure gaussx gives it (banded, sparse, diagonal or dense),
    and a hard constraint Vᵀ x = 0 is kriged at every step.

    Algorithm:
        1. Build the model (`gaussx.laplace_model`) and project the start off
           the null space.
        2. Run the damped Newton loop, a ``lax.while_loop`` with gradients
           stopped, until ‖Δx‖_∞ ≤ tol (1 + ‖x‖_∞), with tol floored at 64
           machine epsilons of the dtype.
        3. Attach implicit tangents to the mode (``jax.lax.custom_root``),
           solved with the factor of Λ_x|y at the mode: one solve per VJP.
        4. Refactorise at the mode and assemble the log-marginal
           (`gaussx.laplace_result`).

    Args:
        prior: The latent field, a gaussx ``GaussianMRF`` or ``IntrinsicGMRF``.
        likelihood: A site-factorising gaussx likelihood holding y.
        projector: The projector A, (M, N); ``None`` is the identity.
        offset: The offset o, a scalar or of length M.
        init: Starting point x⁽⁰⁾; zeros by default.
        max_iter: Maximum Newton steps.
        tol: Relative tolerance on the step's sup-norm.
        damping: Step size α in (0, 1].

    Returns:
        The `LaplaceModeResult`, with gaussx's ``LaplaceResult`` fields and a
        `posterior` property.

    Raises:
        ValidationError: If ``max_iter``, ``tol`` or ``damping`` is out of range.
        MissingExtraError: If the ``numpyro`` extra is not installed.

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
    """
    _validate(max_iter, tol, damping)
    _require_gmrf_step()
    model = gx.laplace_model(prior, likelihood, projector=projector, offset=offset)
    dtype = prior.loc.dtype
    size = prior.event_shape[0]
    start = jnp.zeros(size, dtype) if init is None else jnp.asarray(init, dtype)
    # x⁽⁰⁾ ← x⁽⁰⁾ − Π_V x⁽⁰⁾
    start = start - model.null_component(start)

    # --- the loop, with gradients stopped ---
    frozen_model = _stop_gradient(model)
    estimate, n_iter, converged = _damped_newton(
        frozen_model.newton,
        jax.lax.stop_gradient(start),
        max_iter,
        _floored_tolerance(tol, dtype),
        damping,
    )

    # --- implicit tangents, with the factor at the mode ---
    frozen_fit = gx.laplace_result(
        frozen_model, _stop_gradient(prior), estimate, n_iter, converged
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
            f̂ = Φ(f̂) of the posterior-mean map.
        site_information: The site information λ = g + Λ ⊙ f̂.
        site_precision: The site precision Λ = max(−h, floor).
        n_iter: Iterations taken.
        converged: Whether the last step met the tolerance.
    """

    mode: Float[Array, " N"]
    site_information: Float[Array, " N"]
    site_precision: Float[Array, " N"]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]


class CovarianceLaplaceResult(eqx.Module):
    """The Gaussian at the mode of a GP with prior covariance K.

        q(f) = N(m, V),   V = (K⁻¹ + Λ)⁻¹ = K − K (K + Λ⁻¹)⁻¹ K

    Attributes:
        mode: The mode f̂ (the last iterate), implicitly differentiable.
        site_information: The site information λ at f̂.
        site_precision: The site precision Λ at f̂.
        mean: The posterior mean m given the sites at f̂, equal to f̂ at
            convergence.
        variance: diag(V), the marginal variances.
        log_marginal: The Laplace log-marginal at f̂,
            ∑ₙ log p(yₙ ∣ f̂ₙ) − ½ (f̂ − μ)ᵀ K⁻¹ (f̂ − μ) − ½ log |I + √Λ K √Λ|.
        n_iter: Iterations taken.
        converged: Whether the last step met the tolerance.
    """

    mode: Float[Array, " N"]
    site_information: Float[Array, " N"]
    site_precision: Float[Array, " N"]
    mean: Float[Array, " N"]
    variance: Float[Array, " N"]
    log_marginal: Float[Array, ""]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]
    prior_cov: lx.AbstractLinearOperator

    @property
    def posterior(self) -> Laplace:
        """N(f̂, V) as an `argminax.curvature.Laplace`, with precision K⁻¹ + Λ."""
        inverse = gx.inv(self.prior_cov)
        sites = lx.DiagonalLinearOperator(self.site_precision)
        precision = lx.TaggedLinearOperator(inverse + sites, _PSD)
        return Laplace(self.mode, precision)


def _site_loop(
    update: Callable[[Float[Array, " N"], Any], tuple[Array, Any]],
    start: tuple[Float[Array, " N"], Any],
    max_iter: int,
    tol: float,
) -> tuple[Float[Array, " N"], Any, Int[Array, ""], Bool[Array, ""]]:
    """Iterate (f, carry) ← update(f, carry) until max |Δf| < tol."""

    def keep_going(state: tuple[Array, Any, Array, Array]) -> Array:
        _, _, iteration, step_norm = state
        return (iteration < max_iter) & (step_norm >= tol)

    def step(state: tuple[Array, Any, Array, Array]) -> tuple[Array, Any, Array, Array]:
        latent, carry, iteration, _ = state
        next_latent, next_carry = update(latent, carry)
        # max |Δf|
        step_norm = jnp.max(jnp.abs(next_latent - latent), initial=0.0)
        return next_latent, next_carry, iteration + 1, step_norm

    latent, carry = start
    initial = (latent, carry, jnp.asarray(0), jnp.asarray(jnp.inf, latent.dtype))
    latent, carry, n_iter, step_norm = jax.lax.while_loop(keep_going, step, initial)
    return latent, carry, n_iter, step_norm < tol


def laplace_mode_covariance(
    prior_cov: lx.AbstractLinearOperator,
    prior_mean: Float[Array, " N"],
    log_likelihood: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    *,
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 1.0,
    precision_floor: float = 1e-6,
) -> CovarianceLaplaceResult:
    """Laplace approximation of a GP with prior covariance K, by damped Newton.

        f̃ = μ + K (K + Λ⁻¹)⁻¹ (λ/Λ − μ),   f ← (1 − α) f + α f̃

    The sites (λ, Λ) = `gaussx.newton_update` at f, with Λ = max(−h, floor),
    are a Gaussian pseudo-likelihood, and f̃ is the posterior mean under it
    (Rasmussen and Williams 2006, algorithm 3.1). This is the loop of
    pyrox-gp's ``LaplaceInference`` and ``GaussNewtonInference`` (which differ
    only in their floor), as a ``lax.while_loop``.

    Algorithm:
        1. From f = μ, iterate the damped update until max |Δf| < tol (tol
           floored at 64 machine epsilons of the dtype) or ``max_iter``.
        2. Make the mode implicit through its fixed point f̂ = Φ(f̂).
        3. At f̂: the sites, the posterior mean and marginal variances, and the
           Laplace log-marginal.

    Args:
        prior_cov: K, (N, N), tagged positive semidefinite (jitter included).
        prior_mean: μ, shape (N,).
        log_likelihood: f ↦ (log p(yₙ ∣ fₙ))ₙ, one log-density per site; site
            n must depend on fₙ alone.
        max_iter: Maximum iterations.
        tol: Absolute tolerance on max |Δf|.
        damping: Step size α in (0, 1].
        precision_floor: The floor on Λ; raise it (pyrox-gp's
            ``GaussNewtonInference`` uses 1e-3) for non-log-concave
            likelihoods.

    Returns:
        The `CovarianceLaplaceResult`.

    Raises:
        ValidationError: If ``prior_cov`` is not tagged positive semidefinite,
            or ``max_iter``, ``tol`` or ``damping`` is out of range.

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

    References:
        Rasmussen, C. E. & Williams, C. K. I. (2006). Gaussian Processes for
        Machine Learning, section 3.4.
    """
    _validate(max_iter, tol, damping)
    if not lx.is_positive_semidefinite(prior_cov):
        msg = "'prior_cov' must be tagged positive semidefinite, got tags without it"
        raise ValidationError(msg)
    prior_mean = jnp.asarray(prior_mean)

    def fixed_point(latent: Array) -> Array:
        return _covariance_update(
            latent, prior_cov, prior_mean, log_likelihood, precision_floor
        )

    # the loop runs with every input frozen, closed-over arrays included, so
    # gradients come only from the implicit fixed point below
    frozen_likelihood = _stop_gradient(
        eqx.filter_closure_convert(log_likelihood, prior_mean)
    )
    frozen_cov, frozen_mean = _stop_gradient((prior_cov, prior_mean))

    def update(latent: Array, carry: None) -> tuple[Array, None]:
        target = _covariance_update(
            latent, frozen_cov, frozen_mean, frozen_likelihood, precision_floor
        )
        # f ← (1 − α) f + α f̃
        return (1.0 - damping) * latent + damping * target, carry

    estimate, _, n_iter, converged = _site_loop(
        update,
        (frozen_mean, None),
        max_iter,
        _floored_tolerance(tol, prior_mean.dtype),
    )
    mode = _implicit_fixed_point(fixed_point, estimate)

    # --- the Gaussian and the log-marginal at f̂ ---
    information, precision = _sites(log_likelihood, mode, precision_floor)
    mean = _covariance_update(
        mode, prior_cov, prior_mean, log_likelihood, precision_floor
    )
    regularised = _regularised(prior_cov, precision)
    variance = _marginal_variance(prior_cov, regularised)
    # f̂ − μ
    residual = mode - prior_mean
    # ½ (f̂ − μ)ᵀ K⁻¹ (f̂ − μ)
    quadratic = 0.5 * jnp.vdot(residual, gx.solve(prior_cov, residual))
    # log |I + √Λ K √Λ| = log |K + Λ⁻¹| + ∑ log Λ
    log_det = gx.logdet(regularised) + jnp.sum(jnp.log(precision))
    log_marginal = jnp.sum(log_likelihood(mode)) - quadratic - 0.5 * log_det
    return CovarianceLaplaceResult(
        mode=mode,
        site_information=information,
        site_precision=precision,
        mean=mean,
        variance=variance,
        log_marginal=log_marginal,
        n_iter=n_iter,
        converged=converged,
        prior_cov=prior_cov,
    )


def _regularised(
    prior_cov: lx.AbstractLinearOperator, precision: Float[Array, " N"]
) -> lx.AbstractLinearOperator:
    """K + Λ⁻¹, tagged positive semidefinite."""
    return lx.TaggedLinearOperator(
        prior_cov + lx.DiagonalLinearOperator(1.0 / precision), _PSD
    )


def _covariance_update(
    latent: Float[Array, " N"],
    prior_cov: lx.AbstractLinearOperator,
    prior_mean: Float[Array, " N"],
    log_likelihood: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    precision_floor: float,
) -> Float[Array, " N"]:
    """One undamped update f ↦ μ + K (K + Λ⁻¹)⁻¹ (λ/Λ − μ), the sites at f."""
    information, precision = _sites(log_likelihood, latent, precision_floor)
    # ỹ = λ / Λ
    pseudo_target = information / precision
    # (K + Λ⁻¹)⁻¹ (ỹ − μ)
    weights = gx.solve(_regularised(prior_cov, precision), pseudo_target - prior_mean)
    # μ + K (K + Λ⁻¹)⁻¹ (ỹ − μ)
    return prior_mean + prior_cov.mv(weights)


def _marginal_variance(
    prior_cov: lx.AbstractLinearOperator, regularised: lx.AbstractLinearOperator
) -> Float[Array, " N"]:
    """diag(V) = diag(K) − diag(K (K + Λ⁻¹)⁻¹ K), one solve per site."""
    structure = prior_cov.in_structure()
    identity = jnp.eye(structure.shape[0], dtype=structure.dtype)

    def column_term(basis: Array) -> Array:
        # kₙ = K eₙ
        column = prior_cov.mv(basis)
        # kₙᵀ (K + Λ⁻¹)⁻¹ kₙ
        return jnp.vdot(column, gx.solve(regularised, column))

    return gx.diag(prior_cov) - jax.vmap(column_term)(identity)


def laplace_mode_sites(
    posterior_mean: Callable[
        [Float[Array, " N"], Float[Array, " N"]], Float[Array, " N"]
    ],
    log_likelihood: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    prior_mean: Float[Array, " N"],
    *,
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 1.0,
    precision_floor: float = 1e-6,
) -> SiteLaplaceResult:
    """Laplace approximation by damping the site naturals, given a posterior-mean map.

        (λ, Λ) ← (1 − α) (λ, Λ) + α newton_update(f),   f ← m(λ, Λ)

    ``posterior_mean`` maps site naturals to the posterior mean under the
    Gaussian pseudo-likelihood they define, for whatever prior the caller has:
    for a Markov GP, a Kalman filter and RTS smoother in O(N d³), which stays
    with the caller. This is the loop of pyrox-gp's ``LaplaceMarkovInference``
    and ``GaussNewtonMarkovInference``, as a ``lax.while_loop``: it damps the
    naturals (`gaussx.damped_natural_update`) rather than the mean.

    Algorithm:
        1. Start from f = μ, λ = floor μ and Λ = floor.
        2. Iterate: the Newton sites at f, a damped natural update, Λ floored,
           and f from ``posterior_mean``, until max |Δf| < tol (floored at 64
           machine epsilons) or ``max_iter``.
        3. Make the mode implicit through its fixed point f̂ = m(sites(f̂)),
           and return the undamped sites at f̂.

    Args:
        posterior_mean: (λ, Λ) ↦ m, the posterior mean given site naturals,
            each of shape (N,).
        log_likelihood: f ↦ (log p(yₙ ∣ fₙ))ₙ, one log-density per site.
        prior_mean: μ, shape (N,).
        max_iter: Maximum iterations.
        tol: Absolute tolerance on max |Δf|.
        damping: Step size α in (0, 1] on the naturals.
        precision_floor: The floor on Λ.

    Returns:
        The `SiteLaplaceResult`: the mode and the sites at it, from which the
        caller's smoother gives the marginals and the log-marginal.

    Raises:
        ValidationError: If ``max_iter``, ``tol`` or ``damping`` is out of range.

    Examples:
        With an independent unit prior the posterior mean has a closed form:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from argminax.solve import laplace_mode_sites
        >>> labels = jnp.array([0.0, 1.0])
        >>> def log_likelihood(f):
        ...     return labels * f - jax.nn.softplus(f)
        >>> def posterior_mean(information, precision):
        ...     # m = (μ + λ) / (1 + Λ) for μ = 0 and unit prior variance
        ...     return information / (1.0 + precision)
        >>> fit = laplace_mode_sites(posterior_mean, log_likelihood, jnp.zeros(2))
        >>> bool(fit.converged), fit.mode.shape
        (True, (2,))
    """
    _validate(max_iter, tol, damping)
    prior_mean = jnp.asarray(prior_mean)

    def fixed_point(latent: Array) -> Array:
        information, precision = _sites(log_likelihood, latent, precision_floor)
        return posterior_mean(information, jnp.maximum(precision, precision_floor))

    # the loop runs with every input frozen, closed-over arrays included, so
    # gradients come only from the implicit fixed point below
    frozen_mean = _stop_gradient(
        eqx.filter_closure_convert(posterior_mean, prior_mean, prior_mean)
    )
    frozen_likelihood = _stop_gradient(
        eqx.filter_closure_convert(log_likelihood, prior_mean)
    )

    def update(latent: Array, naturals: tuple[Array, Array]) -> tuple[Array, Any]:
        target_information, target_precision = _sites(
            frozen_likelihood, latent, precision_floor
        )
        information, precision = gx.damped_natural_update(
            naturals[0], naturals[1], target_information, target_precision, lr=damping
        )
        # diagonal sites in, diagonal sites out (gaussx types the operator form too)
        precision = jnp.maximum(cast(Array, precision), precision_floor)
        return frozen_mean(information, precision), (information, precision)

    start_mean = jax.lax.stop_gradient(prior_mean)
    start_sites = (
        start_mean * precision_floor,
        jnp.full_like(start_mean, precision_floor),
    )
    estimate, _, n_iter, converged = _site_loop(
        update,
        (start_mean, start_sites),
        max_iter,
        _floored_tolerance(tol, prior_mean.dtype),
    )
    mode = _implicit_fixed_point(fixed_point, estimate)
    information, precision = _sites(log_likelihood, mode, precision_floor)
    return SiteLaplaceResult(
        mode=mode,
        site_information=information,
        site_precision=jnp.maximum(precision, precision_floor),
        n_iter=n_iter,
        converged=converged,
    )

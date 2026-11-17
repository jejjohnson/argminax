"""Positive and bounded states by reparametrisation: solve in u, report in x = g(u).

A log, softplus or scaled-sigmoid transform keeps the solve unconstrained and
the curvature well defined, so a positive emission rate never gets a posterior
that includes negative values.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from argminax._src.curvature import Laplace
from argminax.exceptions import ValidationError


__all__ = [
    "Reparametrised",
]


_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
_TRANSFORMS = ("exp", "softplus", "sigmoid")


class Reparametrised(eqx.Module):
    """A bound on the state, imposed by solving in an unconstrained variable u.

        x = g(u):   exp:  x = a + exp(u)
                    softplus:  x = a + log(1 + exp(u))
                    sigmoid:  x = a + (b − a) / (1 + exp(−u))

    u has the shape of the state x, and is the control the solver iterates
    on; a is ``lower`` (default 0 for ``exp`` and ``softplus``) and b is
    ``upper``, needed by ``sigmoid`` only. Write the cost 𝒥 in u (the
    residuals call `forward` on it), solve and build the posterior in u, then
    report it in x with `push_forward` or `sample`. The two differ where it
    matters: `push_forward` is the linearisation Σ_x∣y ≈ J_g Σ_u∣y J_gᵀ with
    the Jacobian J_g = diag(g′(û)), which fails where g′ varies across the
    posterior width, that is near the bound; `sample` maps draws in u through
    g, which stays exact for the Gaussian in u and never crosses the bound, so
    prefer it there. An
    averaging kernel or degrees of freedom computed from the posterior in u
    are those of u, not of x.

    paramax (used by somax and gauss_flows) stores a constrained parameter
    inside a model as ``paramax.Parameterize(fn, u)``, applied by
    ``paramax.unwrap``. To keep one transform across a model and a
    retrieval, pass this class's `forward` as ``fn``:
    ``paramax.Parameterize(Reparametrised("softplus").forward, u)``. argminax
    does not depend on paramax: here u is the solver's state, which optimistix
    iterates on directly, and `inverse`, `derivative`, `push_forward` and
    `sample` have no paramax counterpart.

    Attributes:
        transform: ``"exp"``, ``"softplus"`` or ``"sigmoid"``.
        lower: The lower bound a.
        upper: The upper bound b (``sigmoid`` only).

    Raises:
        ValidationError: For an unknown transform, an ``upper`` bound with
            ``exp`` or ``softplus``, or a ``sigmoid`` without
            ``lower < upper``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from argminax.solve import Reparametrised
        >>> positive = Reparametrised("softplus")
        >>> x = positive.forward(jnp.array([-3.0, 0.0, 3.0]))
        >>> bool(jnp.all(x > 0.0))
        True
        >>> bool(jnp.allclose(positive.inverse(x), jnp.array([-3.0, 0.0, 3.0])))
        True

        A bounded state:

        >>> fraction = Reparametrised("sigmoid", lower=0.0, upper=1.0)
        >>> bool(jnp.allclose(fraction.forward(jnp.zeros(1)), 0.5))
        True
    """

    transform: Literal["exp", "softplus", "sigmoid"] = eqx.field(static=True)
    lower: float | None = eqx.field(static=True, default=None)
    upper: float | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        if self.transform not in _TRANSFORMS:
            msg = f"'transform' must be one of {_TRANSFORMS}, got {self.transform!r}"
            raise ValidationError(msg)
        if self.transform == "sigmoid":
            if self.lower is None or self.upper is None or not self.lower < self.upper:
                msg = (
                    "'lower' and 'upper' must both be given, with lower < upper, "
                    f"for 'sigmoid', got lower={self.lower!r}, upper={self.upper!r}"
                )
                raise ValidationError(msg)
        elif self.upper is not None:
            msg = (
                f"'upper' must be None for {self.transform!r}, which bounds from "
                f"below only, got {self.upper!r}"
            )
            raise ValidationError(msg)

    def _offset(self) -> float:
        return 0.0 if self.lower is None else self.lower

    def forward(self, u: Float[Array, "*state"]) -> Float[Array, "*state"]:
        """Map the unconstrained u to the bounded state x = g(u)."""
        offset = self._offset()
        if self.transform == "exp":
            # exp(u)
            growth = jnp.exp(u)
            # x = a + exp(u)
            return offset + growth
        if self.transform == "softplus":
            # log(1 + exp(u))
            growth = jax.nn.softplus(u)
            # x = a + log(1 + exp(u))
            return offset + growth
        assert self.upper is not None
        # b − a
        width = self.upper - offset
        # σ(u)
        fraction = jax.nn.sigmoid(u)
        # x = a + (b − a) σ(u)
        return offset + width * fraction

    def inverse(self, x: Float[Array, "*state"]) -> Float[Array, "*state"]:
        """Map the bounded state x back to u = g⁻¹(x)."""
        offset = self._offset()
        # x − a
        shifted = x - offset
        if self.transform == "exp":
            # u = log(x − a)
            return jnp.log(shifted)
        if self.transform == "softplus":
            # 1 − exp(−(x − a)), by expm1 so it keeps its digits near the bound
            complement = -jnp.expm1(-shifted)
            # u = log(exp(x − a) − 1) = (x − a) + log(1 − exp(−(x − a))), stable
            # far from the bound
            return shifted + jnp.log(complement)
        assert self.upper is not None
        # b − a
        width = self.upper - offset
        # the fraction (x − a)/(b − a)
        fraction = shifted / width
        # log((x − a)/(b − a))
        log_fraction = jnp.log(fraction)
        # log(1 − (x − a)/(b − a))
        log_complement = jnp.log1p(-fraction)
        # u = logit((x − a)/(b − a))
        return log_fraction - log_complement

    def derivative(self, u: Float[Array, "*state"]) -> Float[Array, "*state"]:
        """Return g′(u), the diagonal of the Jacobian J_g = ∂x/∂u."""
        if self.transform == "exp":
            # g′(u) = exp(u)
            return jnp.exp(u)
        if self.transform == "softplus":
            # g′(u) = σ(u)
            return jax.nn.sigmoid(u)
        assert self.upper is not None
        # b − a
        width = self.upper - self._offset()
        # σ(u)
        probability = jax.nn.sigmoid(u)
        # g′(u) = (b − a) σ(u) (1 − σ(u))
        return width * probability * (1.0 - probability)

    def push_forward(self, posterior: Laplace) -> Laplace:
        """Linearise a posterior in u into one in x.

            x̂ = g(û),   Λ_x∣y = J_g⁻ᵀ Λ_u∣y J_g⁻¹,   J_g = diag(g′(û))

        so that Σ_x∣y ≈ J_g Σ_u∣y J_gᵀ, with Λ_u∣y the posterior precision in
        u and Λ_x∣y the one in x. Accurate only where g′ is nearly constant
        across the posterior width; near the bound, use `sample`.

        Args:
            posterior: The posterior 𝒩(û, Λ_u∣y⁻¹) in u.

        Returns:
            The linearised posterior in x, with the same solver strategy.

        Examples:
            At û = 0 the softplus slope is g′(0) = ½, so Λ_x∣y = 4 Λ_u∣y:

            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> from argminax.curvature import Laplace
            >>> from argminax.solve import Reparametrised
            >>> precision = lx.MatrixLinearOperator(
            ...     jnp.eye(2), lx.positive_semidefinite_tag
            ... )
            >>> posterior = Laplace(jnp.zeros(2), precision)
            >>> physical = Reparametrised("softplus").push_forward(posterior)
            >>> bool(jnp.allclose(physical.mode, jnp.log(2.0)))
            True
            >>> bool(jnp.allclose(physical.precision.as_matrix(), 4.0 * jnp.eye(2)))
            True
        """
        # g′(û), the diagonal of J_g
        slope = self.derivative(posterior.mode)

        def physical_precision(
            vector: Float[Array, "*state"],
        ) -> Float[Array, "*state"]:
            # J_g⁻¹ v
            whitened = vector / slope
            # Λ_u∣y J_g⁻¹ v
            curved = posterior.precision.mv(whitened)
            # J_g⁻ᵀ Λ_u∣y J_g⁻¹ v
            return curved / slope

        structure = posterior.precision.in_structure()
        precision = lx.FunctionLinearOperator(physical_precision, structure, _PSD)
        # x̂ = g(û)
        physical_mode = self.forward(posterior.mode)
        return Laplace(physical_mode, precision, posterior.solver)

    def sample(
        self, posterior: Laplace, n: int, *, key: PRNGKeyArray
    ) -> Float[Array, " member *state"]:
        """Draw from the posterior in u and map the draws to x = g(u).

        Exact for the Gaussian in u, and every draw respects the bound.

        Args:
            posterior: The posterior 𝒩(û, Λ_u∣y⁻¹) in u.
            n: Number of draws.
            key: Random key.

        Returns:
            The draws x⁽ᵉ⁾ in x, shape (n, *x̂.shape).

        Examples:
            Every draw of a positive state stays positive, however wide the
            posterior in u:

            >>> import jax
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> from argminax.curvature import Laplace
            >>> from argminax.solve import Reparametrised
            >>> precision = lx.MatrixLinearOperator(
            ...     0.01 * jnp.eye(2), lx.positive_semidefinite_tag
            ... )
            >>> posterior = Laplace(jnp.zeros(2), precision)
            >>> draws = Reparametrised("exp").sample(
            ...     posterior, 100, key=jax.random.key(0)
            ... )
            >>> draws.shape
            (100, 2)
            >>> bool(jnp.all(draws > 0.0))
            True
        """
        # u⁽ᵉ⁾ ~ 𝒩(û, Λ_u∣y⁻¹)
        unconstrained = posterior.sample(n, key=key)
        # x⁽ᵉ⁾ = g(u⁽ᵉ⁾)
        return self.forward(unconstrained)

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

    a is ``lower`` (default 0 for ``exp`` and ``softplus``) and b is
    ``upper``, needed by ``sigmoid`` only. Write the cost in u (the residuals
    call `forward` on it), solve and build the posterior in u, then report it
    in x with `push_forward` or `sample`. The two differ where it matters:
    `push_forward` is the linearisation Ŝₓ ≈ G Ŝᵤ Gᵀ with G = diag(g′(û)),
    which fails where g′ varies across the posterior width, that is near the
    bound; `sample` maps draws in u through g, which stays exact for the
    Gaussian in u and never crosses the bound, so prefer it there. An
    averaging kernel or degrees of freedom computed from the posterior in u
    are those of u, not of x.

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
        if self.transform == "exp":
            # x = a + exp(u)
            return self._offset() + jnp.exp(u)
        if self.transform == "softplus":
            # x = a + log(1 + exp(u))
            return self._offset() + jax.nn.softplus(u)
        assert self.upper is not None
        # x = a + (b − a) σ(u)
        return self._offset() + (self.upper - self._offset()) * jax.nn.sigmoid(u)

    def inverse(self, x: Float[Array, "*state"]) -> Float[Array, "*state"]:
        """Map the bounded state x back to u = g⁻¹(x)."""
        shifted = x - self._offset()
        if self.transform == "exp":
            # u = log(x − a)
            return jnp.log(shifted)
        if self.transform == "softplus":
            # u = log(exp(x − a) − 1) = s + log(1 − exp(−s)), stable for large s
            return shifted + jnp.log(-jnp.expm1(-shifted))
        assert self.upper is not None
        # u = logit((x − a)/(b − a))
        fraction = shifted / (self.upper - self._offset())
        return jnp.log(fraction) - jnp.log1p(-fraction)

    def derivative(self, u: Float[Array, "*state"]) -> Float[Array, "*state"]:
        """Return g′(u), the diagonal of G = ∂x/∂u."""
        if self.transform == "exp":
            # g′(u) = exp(u)
            return jnp.exp(u)
        if self.transform == "softplus":
            # g′(u) = σ(u)
            return jax.nn.sigmoid(u)
        assert self.upper is not None
        # g′(u) = (b − a) σ(u) (1 − σ(u))
        probability = jax.nn.sigmoid(u)
        return (self.upper - self._offset()) * probability * (1.0 - probability)

    def push_forward(self, posterior: Laplace) -> Laplace:
        """Linearise a posterior in u into one in x.

            x̂ = g(û),   Ŝₓ ≈ G Ŝᵤ Gᵀ,   Ĥₓ = G⁻ᵀ Ĥᵤ G⁻¹,   G = diag(g′(û))

        Accurate only where g′ is nearly constant across the posterior width;
        near the bound, use `sample`.

        Args:
            posterior: The posterior N(û, Ĥᵤ⁻¹) in u.

        Returns:
            The linearised posterior in x, with the same solver strategy.
        """
        slope = self.derivative(posterior.mode)

        def physical_precision(
            vector: Float[Array, "*state"],
        ) -> Float[Array, "*state"]:
            # G⁻¹ v
            whitened = vector / slope
            # G⁻ᵀ Ĥᵤ G⁻¹ v
            return posterior.precision.mv(whitened) / slope

        precision = lx.FunctionLinearOperator(
            physical_precision, posterior.precision.in_structure(), _PSD
        )
        return Laplace(self.forward(posterior.mode), precision, posterior.solver)

    def sample(
        self, posterior: Laplace, n: int, *, key: PRNGKeyArray
    ) -> Float[Array, "n ..."]:
        """Draw from the posterior in u and map the draws to x = g(u).

        Exact for the Gaussian in u, and every draw respects the bound.

        Args:
            posterior: The posterior N(û, Ĥᵤ⁻¹) in u.
            n: Number of draws.
            key: Random key.

        Returns:
            The draws in x, shape (n, *x̂.shape).
        """
        return self.forward(posterior.sample(n, key=key))

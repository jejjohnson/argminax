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
        # s = x − a
        shifted = x - offset
        if self.transform == "exp":
            # u = log(x − a)
            return jnp.log(shifted)
        if self.transform == "softplus":
            # 1 − exp(−s), by expm1 so it keeps its digits for small s
            complement = -jnp.expm1(-shifted)
            # u = log(exp(s) − 1) = s + log(1 − exp(−s)), stable for large s
            return shifted + jnp.log(complement)
        assert self.upper is not None
        # b − a
        width = self.upper - offset
        # p = (x − a)/(b − a)
        fraction = shifted / width
        # log p
        log_fraction = jnp.log(fraction)
        # log(1 − p)
        log_complement = jnp.log1p(-fraction)
        # u = logit(p) = log p − log(1 − p)
        return log_fraction - log_complement

    def derivative(self, u: Float[Array, "*state"]) -> Float[Array, "*state"]:
        """Return g′(u), the diagonal of G = ∂x/∂u."""
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

            x̂ = g(û),   Ŝₓ ≈ G Ŝᵤ Gᵀ,   Ĥₓ = G⁻ᵀ Ĥᵤ G⁻¹,   G = diag(g′(û))

        Accurate only where g′ is nearly constant across the posterior width;
        near the bound, use `sample`.

        Args:
            posterior: The posterior N(û, Ĥᵤ⁻¹) in u.

        Returns:
            The linearised posterior in x, with the same solver strategy.

        Examples:
            At û = 0 the softplus slope is g′(0) = ½, so Ĥₓ = 4 Ĥᵤ:

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
        # g′(û), the diagonal of G
        slope = self.derivative(posterior.mode)

        def physical_precision(
            vector: Float[Array, "*state"],
        ) -> Float[Array, "*state"]:
            # G⁻¹ v
            whitened = vector / slope
            # Ĥᵤ G⁻¹ v
            curved = posterior.precision.mv(whitened)
            # G⁻ᵀ Ĥᵤ G⁻¹ v
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
            posterior: The posterior N(û, Ĥᵤ⁻¹) in u.
            n: Number of draws.
            key: Random key.

        Returns:
            The draws in x, shape (n, *x̂.shape).

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
        # u⁽ᵉ⁾ ~ N(û, Ĥᵤ⁻¹)
        unconstrained = posterior.sample(n, key=key)
        # x⁽ᵉ⁾ = g(u⁽ᵉ⁾)
        return self.forward(unconstrained)

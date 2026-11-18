"""What every transform shares: the optax import, the step size and the checks."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from argminax.exceptions import MissingExtraError, ValidationError


#: A step size α, constant or an optax schedule of the step count.
type StepSize = float | Callable[[Int[Array, ""]], Float[Array, ""] | float]


def require_optax() -> ModuleType:
    """Import optax, or say which extra provides it."""
    try:
        import optax
    except ImportError as error:
        msg = (
            "the Bayesian learning rule transforms need the 'optax' extra: "
            "pip install 'argminax[optax]'"
        )
        raise MissingExtraError(msg) from error
    return optax


def require_step_size(learning_rate: StepSize) -> None:
    """Reject a constant step size outside (0, 1]; a schedule is not checked."""
    if not isinstance(learning_rate, (int, float)):
        return
    if not 0.0 < learning_rate <= 1.0:
        msg = f"'learning_rate' must be in (0, 1], got {learning_rate!r}"
        raise ValidationError(msg)


def require_positive(value: float, name: str) -> None:
    if not value > 0.0:
        msg = f"'{name}' must be positive, got {value!r}"
        raise ValidationError(msg)


def require_non_negative(value: float, name: str) -> None:
    if not value >= 0.0:
        msg = f"'{name}' must be non-negative, got {value!r}"
        raise ValidationError(msg)


def require_rate(value: float, name: str) -> None:
    """Reject an averaging rate outside [0, 1)."""
    if not 0.0 <= value < 1.0:
        msg = f"'{name}' must be in [0, 1), got {value!r}"
        raise ValidationError(msg)


def require_prior_mean(
    prior_mean: Float[Array, " state"] | None, shape: tuple[int, ...]
) -> None:
    """Reject a prior mean that is not shaped like the parameters.

    A prior mean of another shape would broadcast against them, and the
    transform would then learn the posterior of a prior nobody asked for.
    """
    if prior_mean is None:
        return
    given = jnp.shape(prior_mean)
    if given != shape:
        msg = f"'prior_mean' must have the parameters' shape {shape}, got {given}"
        raise ValidationError(msg)


def step_size(
    learning_rate: StepSize, count: Int[Array, ""]
) -> Float[Array, ""] | float:
    """The step size α at this step: the constant, or the schedule at ``count``."""
    if isinstance(learning_rate, (int, float)):
        return learning_rate
    return learning_rate(count)

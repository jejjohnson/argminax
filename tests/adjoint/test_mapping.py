"""Tests for `to_optimistix_adjoint`.

The specs are stand-ins with pipekit-cycle's class names and fields: argminax
never imports pipekit-cycle, not even in its tests.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest
from _harness import assert_grad_matches_fd

from argminax import ValidationError
from argminax.adjoint import KStep, to_optimistix_adjoint


@dataclasses.dataclass(frozen=True)
class ImplicitAdjoint:
    pass


@dataclasses.dataclass(frozen=True)
class TruncatedAdjoint:
    k: int = 1


@dataclasses.dataclass(frozen=True)
class RecursiveCheckpointAdjoint:
    checkpoints: int | None = None


@dataclasses.dataclass(frozen=True)
class DirectAdjoint:
    pass


@dataclasses.dataclass(frozen=True)
class BacksolveAdjoint:
    pass


def test_implicit_maps_to_the_optimistix_default() -> None:
    adjoint = to_optimistix_adjoint(ImplicitAdjoint())
    assert adjoint == optx.ImplicitAdjoint()


@pytest.mark.parametrize("k", [1, 4])
def test_truncated_maps_to_kstep(k: int) -> None:
    assert to_optimistix_adjoint(TruncatedAdjoint(k=k)) == KStep(k=k)


def test_truncated_without_k_is_one_step() -> None:
    class TruncatedAdjoint:
        pass

    assert to_optimistix_adjoint(TruncatedAdjoint()) == KStep(k=1)


@pytest.mark.parametrize("checkpoints", [None, 7])
def test_recursive_checkpoint_keeps_its_checkpoints(checkpoints: int | None) -> None:
    adjoint = to_optimistix_adjoint(RecursiveCheckpointAdjoint(checkpoints))
    assert adjoint == optx.RecursiveCheckpointAdjoint(checkpoints=checkpoints)


@pytest.mark.parametrize(
    "adjoint", [optx.ImplicitAdjoint(), optx.RecursiveCheckpointAdjoint(), KStep(k=2)]
)
def test_optimistix_adjoints_pass_through(adjoint: optx.AbstractAdjoint) -> None:
    assert to_optimistix_adjoint(adjoint) is adjoint


@pytest.mark.parametrize("spec", [DirectAdjoint(), BacksolveAdjoint()])
def test_adjoints_through_time_name_integrix(spec) -> None:
    with pytest.raises(ValidationError, match="integrix"):
        to_optimistix_adjoint(spec)


def test_an_unknown_spec_is_rejected() -> None:
    with pytest.raises(ValidationError, match="'spec' is not a recognised"):
        to_optimistix_adjoint(object())


def test_truncated_with_a_bad_k_is_rejected_by_kstep() -> None:
    with pytest.raises(ValidationError, match="'k' must be a positive integer"):
        to_optimistix_adjoint(TruncatedAdjoint(k=0))


def test_implicit_gradient_matches_finite_differences() -> None:
    def residuals(state, theta):
        # r(x, θ) = (x − θ, 0.3 sin x): nonlinear, with a strict minimum
        return state - theta, 0.3 * jnp.sin(state)

    adjoint = to_optimistix_adjoint(ImplicitAdjoint())

    def retrieve(theta):
        solution = optx.least_squares(
            residuals,
            optx.GaussNewton(rtol=1e-14, atol=1e-14),
            theta,
            theta,
            adjoint=adjoint,
            throw=False,
        )
        return solution.value, solution.result

    theta = jnp.array([0.4, -1.1, 2.0])
    _, result = retrieve(theta)
    assert result == optx.RESULTS.successful
    assert_grad_matches_fd(lambda t: jnp.sum(retrieve(t)[0] ** 2), theta)
    assert jax.jacrev(lambda t: retrieve(t)[0])(theta).shape == (3, 3)

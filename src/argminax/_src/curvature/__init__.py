"""Curvature at a minimum: the Gaussian there and its covariance."""

from __future__ import annotations

from argminax._src.curvature._laplace import Laplace, gauss_newton, push_forward
from argminax._src.curvature._matfree import (
    VarianceEstimate,
    low_rank,
    marginal_variance,
    sample_by_perturbation,
)


__all__ = [
    "Laplace",
    "VarianceEstimate",
    "gauss_newton",
    "low_rank",
    "marginal_variance",
    "push_forward",
    "sample_by_perturbation",
]

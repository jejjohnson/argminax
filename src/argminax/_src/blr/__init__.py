"""The Bayesian learning rule: optax transforms that carry a Gaussian."""

from __future__ import annotations

from argminax._src.blr._diagonal import BLRDiagonalState, blr_diagonal
from argminax._src.blr._full_rank import BLRFullRankState, blr_full_rank, newton
from argminax._src.blr._ivon import IVONState, ivon
from argminax._src.blr._low_rank import BLRLowRankState, blr_low_rank
from argminax._src.blr._posterior import blr_posterior
from argminax._src.blr._preconditioned import (
    preconditioned_sgd,
    scale_by_preconditioner,
)


__all__ = [
    "BLRDiagonalState",
    "BLRFullRankState",
    "BLRLowRankState",
    "IVONState",
    "blr_diagonal",
    "blr_full_rank",
    "blr_low_rank",
    "blr_posterior",
    "ivon",
    "newton",
    "preconditioned_sgd",
    "scale_by_preconditioner",
]

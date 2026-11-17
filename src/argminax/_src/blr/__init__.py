"""The Bayesian learning rule: optax transforms that carry a Gaussian."""

from __future__ import annotations

from argminax._src.blr._diagonal import BLRDiagonalState, blr_diagonal
from argminax._src.blr._full_rank import BLRFullRankState, blr_full_rank, newton
from argminax._src.blr._ivon import IVONState, ivon


__all__ = [
    "BLRDiagonalState",
    "BLRFullRankState",
    "IVONState",
    "blr_diagonal",
    "blr_full_rank",
    "ivon",
    "newton",
]

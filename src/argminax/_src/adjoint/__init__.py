"""Adjoints of a minimiser: how its solution moves with its arguments."""

from __future__ import annotations

from argminax._src.adjoint._kstep import KStep, OneStep


__all__ = [
    "KStep",
    "OneStep",
]

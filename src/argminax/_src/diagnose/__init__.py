"""Retrieval diagnostics: what the curvature at a minimum says."""

from __future__ import annotations

from argminax._src.diagnose._evidence import evidence, optimize_prior_precision
from argminax._src.diagnose._retrieval import (
    TraceEstimate,
    averaging_kernel,
    degrees_of_freedom,
    information_content,
)


__all__ = [
    "TraceEstimate",
    "averaging_kernel",
    "degrees_of_freedom",
    "evidence",
    "information_content",
    "optimize_prior_precision",
]

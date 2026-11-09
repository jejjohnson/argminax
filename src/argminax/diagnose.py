"""Retrieval diagnostics at a minimum, for any linear-Gaussian posterior.

Each takes a `argminax.curvature.Laplace` posterior and is exact for a small
state or stochastic, with a standard error, from a key; all run under ``jit``
and ``vmap``, so a million pixels are one ``vmap`` with split keys.

| Name | What it gives you |
|---|---|
| `averaging_kernel` | A = I − Ŝ Sₐ⁻¹, lazily |
| `degrees_of_freedom` | dₛ = tr A, exact or by Hutchinson probes |
| `information_content` | H = ½ log \\|Sₐ Ŝ⁻¹\\|, exact or by Lanczos quadrature |
| `evidence` | the Laplace evidence log p(y) at the minimum |
| `optimize_prior_precision` | the prior precision λ that maximises the evidence |
| `TraceEstimate` | a diagnostic with its standard error |
"""

from __future__ import annotations

from argminax._src.diagnose import (
    TraceEstimate,
    averaging_kernel,
    degrees_of_freedom,
    evidence,
    information_content,
    optimize_prior_precision,
)


__all__ = [
    "TraceEstimate",
    "averaging_kernel",
    "degrees_of_freedom",
    "evidence",
    "information_content",
    "optimize_prior_precision",
]

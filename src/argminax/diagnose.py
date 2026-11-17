"""Retrieval diagnostics at a minimum, for any linear-Gaussian posterior.

Each takes a `argminax.curvature.Laplace` posterior and is exact for a small
state or stochastic, with a standard error, from a key; all run under ``jit``
and ``vmap``, so a million pixels are one ``vmap`` with split keys. Σ_x is the
prior covariance and Σ_x∣y, Λ_x∣y the posterior covariance and precision.

| Name | What it gives you |
|---|---|
| `averaging_kernel` | the averaging kernel J_x̂ = I − Σ_x∣y Λ_x, lazily |
| `degrees_of_freedom` | dₛ = tr J_x̂, exact or by Hutchinson probes |
| `information_content` | 𝕀[x; y] = ½ log det(Σ_x Λ_x∣y), exact or by Lanczos |
| `evidence` | the Laplace evidence log p(y) at the minimum |
| `optimize_prior_precision` | the prior precision δ that maximises the evidence |
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

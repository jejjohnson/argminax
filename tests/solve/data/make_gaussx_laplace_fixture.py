"""gaussx's `laplace_mode` on the problems of `tests/_gaussx_problems.py` (#86).

gaussx 0.8.0 removed `laplace_mode` (gaussx#848), so this runs only against
gaussx v0.6.13 to v0.7.x. From the repository root, in an environment with that gaussx:

    uv run --with 'gaussx @ git+https://github.com/jejjohnson/gaussx@v0.6.13' \
        python tests/solve/data/make_gaussx_laplace_fixture.py \
        tests/solve/data/gaussx_laplace_mode.npz
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
from _gaussx_problems import (
    FLOAT32_SIZE,
    LOG_TAU,
    PRECISION_CASES,
    VB_SIZE,
    VB_SUBSPACE,
    counts,
    gaussian_prior,
    intrinsic_prior,
)
from loguru import logger


jax.config.update("jax_enable_x64", True)


def main(path: str) -> None:
    out: dict[str, np.ndarray] = {}
    for name, (make_prior, make_likelihood, kwargs) in PRECISION_CASES.items():
        fit = gx.laplace_mode(make_prior(), make_likelihood(), **kwargs)
        out[f"{name}_mode"] = np.asarray(fit.mode)
        out[f"{name}_n_iter"] = np.asarray(fit.n_iter)
        out[f"{name}_log_marginal"] = np.asarray(fit.log_marginal)
        out[f"{name}_hessian_type"] = np.asarray(type(fit.hessian).__name__)

    prior = gaussian_prior(FLOAT32_SIZE, dtype=jnp.float32)
    fit = gx.laplace_mode(prior, counts(FLOAT32_SIZE, dtype=jnp.float32))
    out["float32_mode"] = np.asarray(fit.mode)
    out["float32_log_marginal"] = np.asarray(fit.log_marginal)

    likelihood = counts(20)

    def log_marginal(log_tau):
        return gx.laplace_mode(
            intrinsic_prior(20, jnp.exp(log_tau)), likelihood
        ).log_marginal

    out["log_marginal_grad"] = np.asarray(jax.grad(log_marginal)(LOG_TAU))

    prior, likelihood = gaussian_prior(VB_SIZE), counts(VB_SIZE)
    fit = gx.laplace_mode(prior, likelihood)
    out["vb_mean"] = np.asarray(
        gx.vb_mean_correction(fit, prior, likelihood, subspace=jnp.array(VB_SUBSPACE))
    )
    np.savez(path, **out)
    logger.info(
        "gaussx {} wrote {}: {}",
        gx.__version__ if hasattr(gx, "__version__") else "?",
        path,
        sorted(out),
    )


if __name__ == "__main__":
    main(sys.argv[1])

"""gaussx's problem types for the precision form of `laplace_mode` (#25, #86).

`tests/solve/test_laplace_mode.py` solves these with argminax and compares with
the results frozen in `tests/solve/data/gaussx_laplace_mode.npz`;
`tests/solve/data/make_gaussx_laplace_fixture.py` writes that file from gaussx's
own `laplace_mode`, so both read the problems from here.
"""

from __future__ import annotations

import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpy as np


def counts(n: int, dtype=jnp.float64) -> gx.PoissonLikelihood:
    rates = np.exp(np.sin(np.arange(n) / 3.0))
    return gx.PoissonLikelihood(
        jnp.asarray(np.random.default_rng(0).poisson(rates), dtype)
    )


def gaussian_prior(n: int, scale: float = 1.0, dtype=jnp.float64) -> gx.GaussianMRF:
    precision = 2.0 * np.eye(n) - 0.9 * (np.eye(n, k=1) + np.eye(n, k=-1))
    operator = lx.MatrixLinearOperator(
        jnp.asarray(scale * precision, dtype), lx.positive_semidefinite_tag
    )
    return gx.GaussianMRF(jnp.zeros(n, dtype), operator)


def intrinsic_prior(n: int, tau=10.0) -> gx.IntrinsicGMRF:
    times = jnp.arange(n, dtype=jnp.float64)
    null_space = jnp.column_stack([jnp.ones(n), times - times.mean()])
    return gx.IntrinsicGMRF(jnp.zeros(n), tau, gx.rw2_structure(n), null_space)


PRECISION_CASES = {
    "gaussian": (lambda: gaussian_prior(20), lambda: counts(20), {}),
    "intrinsic-hard": (lambda: intrinsic_prior(20), lambda: counts(20), {}),
    "odd-n-selection": (
        lambda: gx.IntrinsicGMRF(
            jnp.zeros(22),
            10.0,
            gx.rw2_structure(21),
            jnp.zeros((22, 2))
            .at[:21, 0]
            .set(1.0)
            .at[:21, 1]
            .set(jnp.arange(21.0) - 10.0),
        ),
        lambda: counts(21),
        {
            "projector": gx.SparseOperator.from_coo(
                np.arange(21), np.arange(21), jnp.ones(21), (21, 22)
            )
        },
    ),
}

# the log-marginal gradient is taken in log τ of the intrinsic prior, at this point
LOG_TAU = 2.0
# the float32 case, and the VB mean correction of a Gaussian prior
FLOAT32_SIZE = 12
VB_SIZE = 10
VB_SUBSPACE = (0, 1, 2)

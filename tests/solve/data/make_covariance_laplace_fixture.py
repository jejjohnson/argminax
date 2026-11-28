"""The dense covariance-form Laplace loop, frozen before its solver strategy (#98).

`laplace_mode_covariance` before #98 solved every step with a dense
factorisation. This script freezes what that code gives on a Bernoulli and a
Poisson problem at N = 200, so `tests/solve/test_laplace_mode.py` checks that
``solver=None`` still gives it. Run it from the repository root, on a checkout
of the commit to freeze:

    uv run --with cyclopts --with loguru \\
        python tests/solve/data/make_covariance_laplace_fixture.py \\
        tests/solve/data/covariance_laplace.npz
"""

from __future__ import annotations

from pathlib import Path

import cyclopts
import einx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from loguru import logger


jax.config.update("jax_enable_x64", True)


app = cyclopts.App(help=(__doc__ or "").splitlines()[0])

SIZE = 200
_PSD = (lx.symmetric_tag, lx.positive_semidefinite_tag)


def kernel_matrix(inputs: np.ndarray) -> np.ndarray:
    """The Matérn-3/2 Gram matrix, σ² = 1.5, l = 0.9, jitter 1e-6.

    The test builds it with this same NumPy code, so the fixture stores only
    the inputs.
    """
    # |xᵢ − xⱼ|
    distances = np.abs(einx.subtract("row, col -> row col", inputs, inputs))
    # k(r) = σ² (1 + √3 r / l) exp(−√3 r / l), plus jitter
    scaled = np.sqrt(3.0) * distances / 0.9
    return 1.5 * (1.0 + scaled) * np.exp(-scaled) + 1e-6 * np.eye(inputs.size)


def _problem() -> dict[str, np.ndarray]:
    """200 sorted inputs on [−5, 5] and two datasets on them."""
    generator = np.random.default_rng(98)
    inputs = np.sort(generator.uniform(-5.0, 5.0, SIZE))
    # labels: sin(1.5 x) > 0, one in ten flipped
    flips = generator.uniform(size=SIZE) < 0.1
    labels = np.logical_xor(np.sin(1.5 * inputs) > 0.0, flips).astype(np.float64)
    # counts: Poisson with rate exp(0.5 + sin x)
    counts = generator.poisson(np.exp(0.5 + np.sin(inputs))).astype(np.float64)
    return {
        "inputs": inputs,
        "bernoulli_y": labels,
        "bernoulli_mean": np.zeros(SIZE),
        "poisson_y": counts,
        "poisson_mean": np.full(SIZE, 0.5),
    }


def _log_likelihood(name: str, targets: jax.Array):
    if name == "bernoulli":

        def bernoulli(latent):
            # log p(y ∣ f) = y f − log(1 + exp f)
            return targets * latent - jax.nn.softplus(latent)

        return bernoulli

    def poisson(latent):
        # log p(y ∣ f) = y f − exp f − log y!
        return targets * latent - jnp.exp(latent) - jax.scipy.special.gammaln(
            targets + 1.0
        )

    return poisson


@app.default
def main(out: Path) -> None:
    """Write the problems and the dense fits to ``out``.

    Args:
        out: The ``.npz`` file to write.
    """
    from argminax.solve import laplace_mode_covariance

    data = _problem()
    kernel = kernel_matrix(data["inputs"])
    prior = lx.MatrixLinearOperator(jnp.asarray(kernel), _PSD)
    for name in ("bernoulli", "poisson"):
        targets = jnp.asarray(data[f"{name}_y"])
        fit = laplace_mode_covariance(
            prior,
            jnp.asarray(data[f"{name}_mean"]),
            _log_likelihood(name, targets),
            max_iter=50,
            tol=1e-10,
        )
        logger.info(
            "{}: {} steps, converged {}, log-marginal {:.6f}",
            name,
            int(fit.n_iter),
            bool(fit.converged),
            float(fit.log_marginal),
        )
        data.update(
            {
                f"{name}_mode": np.asarray(fit.mode),
                f"{name}_mean_at_mode": np.asarray(fit.mean),
                f"{name}_variance": np.asarray(fit.variance),
                f"{name}_nat1": np.asarray(fit.site_information),
                f"{name}_nat2": np.asarray(fit.site_precision),
                f"{name}_log_marginal": np.asarray(fit.log_marginal),
                f"{name}_n_iter": np.asarray(fit.n_iter),
            }
        )
    np.savez_compressed(out, **data)
    logger.info("wrote {}", out)


if __name__ == "__main__":
    app()

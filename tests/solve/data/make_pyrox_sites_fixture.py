"""pyrox's site fixed points, frozen for `site_fixed_point` (#100).

pyrox sits above argminax, so `tests/solve/test_sites.py` reads what pyrox
computes from this fixture instead of importing it. Run it from the repository
root with pyrox's environment and a pyrox source tree (``packages/*/src``):

    uv run --no-project --python <pyrox>/.venv/bin/python \\
        --with cyclopts --with loguru \\
        python tests/solve/data/make_pyrox_sites_fixture.py \\
        tests/solve/data/pyrox_sites.npz --pyrox <pyrox>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cyclopts
import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger


jax.config.update("jax_enable_x64", True)


app = cyclopts.App(help=(__doc__ or "").splitlines()[0])


def _markov(out: dict[str, np.ndarray]) -> None:
    """pyrox-ssm's PosteriorLinearizationMarkov, its `_solve_sites` call alone."""
    from pyrox_gp import BernoulliLikelihood
    from pyrox_ssm import MarkovGPPrior, MaternSDE, PosteriorLinearizationMarkov
    from pyrox_ssm._inference_nongauss_markov import (
        _mask_sites,
        _prior_marginal_variance,
        _solve_sites,
        markov_implicit_adjoint,
    )
    from pyrox_ssm._markov import _observation_mask
    from pyrox_ssm._markov_sites import as_gaussx_likelihood

    # --- the problem: 16 times, one masked, Bernoulli labels ---
    times = jnp.linspace(0.0, 4.0, 16)
    labels = (jnp.cos(1.5 * times) > 0.0).astype(times.dtype)
    mask = jnp.ones(16, dtype=bool).at[5].set(False)
    prior = MarkovGPPrior(MaternSDE(variance=1.0, lengthscale=0.6, order=1), times)
    strategy = PosteriorLinearizationMarkov()

    # --- the start of _fit_cavity_sites: one update from the prior marginals ---
    observed = _observation_mask(mask, times)
    site_lik = as_gaussx_likelihood(BernoulliLikelihood(), labels, mask=observed)
    prior_mean = prior.mean(times)
    zeros = jnp.zeros(16, dtype=prior_mean.dtype)
    floor = jnp.full((16,), strategy.precision_floor, dtype=prior_mean.dtype)
    sites = _mask_sites(zeros, floor, observed, strategy.precision_floor)
    prior_variance = _prior_marginal_variance(prior).astype(prior_mean.dtype)
    start = strategy._update(sites, prior_mean, prior_variance, site_lik, observed)
    args = (prior, site_lik, observed)

    # --- the step, exported, so argminax runs pyrox's step without pyrox ---
    def step(sites):
        return strategy._step(sites, args)

    spec = jax.ShapeDtypeStruct((16,), prior_mean.dtype)
    exported = jax.export.export(jax.jit(step))((spec, spec))
    blob = exported.serialize()

    def exported_step(sites, _):
        return tuple(exported.call(sites))

    adjoint = markov_implicit_adjoint()
    settings = {"tol": strategy.tol, "max_steps": strategy.max_iter - 1}
    reference, n_steps, converged = _solve_sites(
        exported_step, start, None, adjoint=adjoint, **settings
    )
    native, native_steps, _ = _solve_sites(
        strategy._step, start, args, adjoint=adjoint, **settings
    )
    fit = strategy.fit(prior, BernoulliLikelihood(), labels, mask=mask)
    logger.info(
        "markov: {} steps, converged {}; exported vs native {:.1e} ({} steps); "
        "fit vs native {:.1e}",
        int(n_steps),
        bool(converged),
        float(jnp.max(jnp.abs(reference[0] - native[0]))),
        int(native_steps),
        float(jnp.max(jnp.abs(fit.site_nat1 - native[0]))),
    )
    out.update(
        markov_step=np.frombuffer(blob, dtype=np.uint8),
        markov_start_nat1=np.asarray(start[0]),
        markov_start_nat2=np.asarray(start[1]),
        markov_observed=np.asarray(observed),
        markov_tol=np.asarray(settings["tol"]),
        markov_max_steps=np.asarray(settings["max_steps"]),
        markov_nat1=np.asarray(reference[0]),
        markov_nat2=np.asarray(reference[1]),
        markov_n_steps=np.asarray(n_steps),
        markov_converged=np.asarray(converged),
        markov_native_nat1=np.asarray(native[0]),
        markov_native_nat2=np.asarray(native[1]),
    )


def _dense(out: dict[str, np.ndarray]) -> None:
    """pyrox-gp's eager ExpectationPropagation and PosteriorLinearization."""
    from pyrox_gp import (
        RBF,
        BernoulliLikelihood,
        ExpectationPropagation,
        GPPrior,
        PoissonLikelihood,
        PosteriorLinearization,
    )
    from pyrox_gp._inference_nongauss import (
        _log_prob_per_point_factory,
        _prior_K,
        _site_data,
    )

    inputs = jnp.linspace(-3.0, 3.0, 12)[:, None]
    problems = {
        "bernoulli": (
            BernoulliLikelihood(),
            (jnp.sin(inputs[:, 0]) > 0.0).astype(inputs.dtype),
        ),
        "poisson": (
            PoissonLikelihood(),
            jnp.floor(jnp.exp(1.0 + jnp.sin(inputs[:, 0]))),
        ),
    }
    prior = GPPrior(kernel=RBF(init_lengthscale=1.0, init_variance=1.0), X=inputs)
    out.update(dense_K=np.asarray(_prior_K(prior)), dense_mean=np.asarray(prior.mean(inputs)))
    # tight settings, so each eager loop stops at its fixed point
    strategies = {
        "ep": ExpectationPropagation(max_iter=500, tol=1e-12),
        "pl": PosteriorLinearization(max_iter=500, tol=1e-12),
    }
    for name, (likelihood, observations) in problems.items():
        out[f"{name}_y"] = np.asarray(observations)
        probe = jnp.linspace(-2.0, 2.0, 12)
        site_data = _site_data(likelihood, observations, inputs)
        log_prob = _log_prob_per_point_factory(likelihood)(probe, site_data)
        out[f"{name}_probe_f"] = np.asarray(probe)
        out[f"{name}_probe_log_prob"] = np.asarray(log_prob)
        for method, strategy in strategies.items():
            fit = strategy.fit(prior, likelihood, observations)
            tag = f"{method}_{name}"
            out[f"{tag}_nat1"] = np.asarray(fit.site_nat1)
            out[f"{tag}_nat2"] = np.asarray(fit.site_nat2)
            out[f"{tag}_q_mean"] = np.asarray(fit.q_mean)
            out[f"{tag}_n_iter"] = np.asarray(fit.n_iter)
            out[f"{tag}_converged"] = np.asarray(fit.converged)
            logger.info(
                "{}: {} iterations, converged {}",
                tag,
                int(fit.n_iter),
                bool(fit.converged),
            )


@app.default
def main(path: Path, *, pyrox: Path) -> None:
    """Write pyrox's site fixed points to an ``.npz`` file.

    Args:
        path: Where to write the fixture.
        pyrox: A pyrox source tree, with ``packages/<package>/src``.
    """
    for package in ("pyrox", "pyrox-gp", "pyrox-ssm"):
        sys.path.insert(0, str(pyrox / "packages" / package / "src"))
    out: dict[str, np.ndarray] = {}
    _markov(out)
    _dense(out)
    np.savez_compressed(path, **out)
    logger.info("wrote {} arrays to {}", len(out), path)


if __name__ == "__main__":
    app()

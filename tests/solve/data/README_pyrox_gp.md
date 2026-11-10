# pyrox-gp Laplace fixtures

`pyrox_gp_64.npz` and `pyrox_gp_32.npz` hold pyrox-gp's Newton loops on its own test problems, in float64 and float32, so `tests/solve/test_laplace_mode.py` can check the port (#25) without importing pyrox, which sits above argminax.

- pyrox at `3d16c53` (`origin/main`, 9 November 2026).
- Dense (`laplace_*`, `gauss_newton_*`): `tests/gp/test_inference_nongauss.py::_make_classification_problem`, 12 points on [−3, 3], Bernoulli labels sin(x) > 0, an RBF kernel with unit lengthscale and variance; `LaplaceInference` and `GaussNewtonInference` with `max_iter=50`, `tol=1e-8`. `dense_K` is the kernel matrix with pyrox's jitter.
- Markov (`markov_*`): `tests/gp/test_inference_nongauss_markov.py::test_markov_laplace_matches_dense_laplace_on_matern12`, 14 times on [0, 4], labels cos(1.5 t) > 0, a Matern-1/2 SDE with lengthscale 0.6; `LaplaceMarkovInference` with `max_iter=30`, `tol=1e-7`. `markov_K` is the exact Matern-1/2 covariance, the dense form of the SDE.
- `bernoulli_probe_*`: pyrox's per-point Bernoulli log-density at a few points, so the test can check its own.
- pyrox's dense loops start at f = μ = 0, where numpyro 0.21's stable `Bernoulli(logits=f).log_prob` has a kink and JAX's gradient there is 0 (y = 0) or 1 (y = 1) instead of y − ½. The first step is off, so pyrox takes 6 iterations where an exact gradient takes 4; the fixed point is the same.
- In float32, pyrox's dense loops never reach `tol=1e-8` (50 of 50 iterations); argminax floors the tolerance at 64 machine epsilons, so it stops earlier, at float32 accuracy.

Regenerate from a pyrox checkout with the script below (`python make_pyrox_gp_fixture.py pyrox_gp_64.npz 64`, then `32`), after pointing its `sys.path` line at `packages/pyrox-gp/src` and `packages/pyrox/src`:

```python
"""pyrox-gp's Newton loops on its own test problems, for argminax#25."""
import sys
sys.path[:0] = ["packages/pyrox-gp/src", "packages/pyrox/src"]
import jax
x64 = sys.argv[2] == "64"
jax.config.update("jax_enable_x64", x64)
import jax.numpy as jnp
import numpy as np
from pyrox_gp import (BernoulliLikelihood, GaussNewtonInference, GPPrior, LaplaceInference,
                      LaplaceMarkovInference, MarkovGPPrior, Matern, MaternSDE, RBF)
from pyrox_gp._inference_nongauss import _prior_K, _log_prob_per_point_factory

out = {}
def record(tag, cond, K=None, mean=None, y=None):
    out.update({f"{tag}_nat1": cond.site_nat1, f"{tag}_nat2": cond.site_nat2, f"{tag}_q_mean": cond.q_mean,
                f"{tag}_q_var": cond.q_var, f"{tag}_log_marginal": cond.log_marginal_approx,
                f"{tag}_n_iter": cond.n_iter, f"{tag}_converged": cond.converged})

# --- dense: pyrox-gp's _make_classification_problem ---
X = jnp.linspace(-3.0, 3.0, 12)[:, None]
y = (jnp.sin(X[:, 0]) > 0.0).astype(X.dtype)
prior = GPPrior(kernel=RBF(init_lengthscale=1.0, init_variance=1.0), X=X)
out.update(dense_K=_prior_K(prior), dense_mean=prior.mean(prior.X), dense_y=y)
record("laplace", LaplaceInference(max_iter=50, tol=1e-8).fit(prior, BernoulliLikelihood(), y))
record("gauss_newton", GaussNewtonInference(max_iter=50, tol=1e-8).fit(prior, BernoulliLikelihood(), y))
# the per-point log-density, so the test can check its own Bernoulli
f_probe = jnp.linspace(-3.0, 3.0, 12, dtype=X.dtype)
out["bernoulli_probe_f"] = f_probe
out["bernoulli_probe_log_prob"] = _log_prob_per_point_factory(BernoulliLikelihood())(f_probe, y)

# --- Markov: the Matern-1/2 problem of test_markov_laplace_matches_dense_laplace_on_matern12 ---
times = jnp.linspace(0.0, 4.0, 14)
y_m = (jnp.cos(1.5 * times) > 0.0).astype(times.dtype)
markov_prior = MarkovGPPrior(MaternSDE(variance=1.0, lengthscale=0.6, order=0), times)
record("markov", LaplaceMarkovInference(max_iter=30, tol=1e-7).fit(markov_prior, BernoulliLikelihood(), y_m))
out.update(markov_times=times, markov_y=y_m, markov_mean=markov_prior.mean(times),
           markov_K=jnp.exp(-jnp.abs(times[:, None] - times[None, :]) / 0.6))
np.savez(sys.argv[1], **{k: np.asarray(v) for k, v in out.items()})
print(x64, {k: np.asarray(v).dtype.name for k, v in out.items() if k.endswith("q_mean")}, {k: int(v) for k, v in out.items() if k.endswith("n_iter")})
```

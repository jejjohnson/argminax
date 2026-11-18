# optax_bayes fixtures

`optax_bayes.npz` holds optax_bayes's iterates and posteriors on one small regression, so the tests in `tests/blr/` can check the port of the Bayesian learning rule (gate G4 of #58) without depending on optax_bayes, which is being retired (optax_bayes#68).

- optax_bayes 0.0.3: the source at `a10da90`, whose `src/` differs from the release commit `5f8e8cc` only in the version string. Recorded on 17 November 2026 in float64, with optax 0.2.9 and gaussx 0.6.13.
- The problem is 𝒥(θ) = ½ ‖Xθ − y‖² with 30 observations and 4 weights: `inputs` (30, 4), `targets` (30,), the starting point `start` (4,) and a prior mean `prior_mean` (4,), all drawn from `numpy.random.default_rng(20261117)`. `tests/blr/_regression.py` rebuilds it from these arrays.
- Each run is 50 steps from `start`, stored as the (50, 4) array of iterates. The pytree runs hold the weights as `{"w": θ[:2] as (2, 1), "b": θ[2:]}`.
- optax_bayes's functions take cost gradients here (its `*_for_loss` names), which is the only convention argminax has.

| Key | optax_bayes call |
|---|---|
| `diag_ggn` (+ `_mean`, `_variance`) | `blr_diagonal_for_loss(0.05, prior_precision=0.5, prior_mean=0.1, damping=1e-6)` |
| `diag_identity` | `blr_diagonal_for_loss(0.01, prior_precision=2.0, hessian_estimator="identity")` |
| `diag_schedule` | `blr_with_schedule(optax.exponential_decay(0.05, 10, 0.5), prior_precision=0.5, prior_mean=0.1)` |
| `ivon` (+ `_mean`, `_variance`), `ivon_no_params` | `ivon(0.05, beta1=0.9, beta2=0.99, weight_decay=0.1, clip_radius=1.0, ess=30.0)`, with and without `params` passed to `update` |
| `full_ggn` (+ `_mean`, `_covariance`) | `blr_full_rank_for_loss(0.05, hessian_estimator="ggn", prior_precision=0.5, prior_mean=prior_mean, damping=1e-6)` |
| `full_exact` (+ `_mean`, `_covariance`) | the same with step size 0.3 and the exact Hessian XᵀX |
| `newton` | `newton_for_loss(hessian, damping=1e-6)` |
| `low_ggn` (+ `_mean`, `_covariance`) | `blr_low_rank_for_loss(0.05, rank=2, hessian_estimator="ggn", prior_precision=0.5, prior_mean=prior_mean, damping=1e-6)` |
| `low_exact` (+ `_mean`, `_covariance`) | the same with step size 0.3 and the exact Hessian XᵀX |

The `_mean`, `_variance` and `_covariance` arrays are what `get_posterior_diagonal`, `get_posterior_ivon`, `get_posterior_full_rank` and `get_posterior_low_rank` return for the final state.

Regenerate from an environment with optax_bayes, optax and gaussx installed, with `python make_optax_bayes_fixtures.py optax_bayes.npz`, where the script is:

```python
"""Record optax_bayes's iterates and posteriors on one small regression."""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)
import optax_bayes as ob

NUM_STEPS = 50
rng = np.random.default_rng(20261117)
inputs = jnp.asarray(rng.normal(size=(30, 4)))
targets = jnp.asarray(rng.normal(size=30))
start = jnp.asarray(rng.normal(size=4))
prior_mean = jnp.asarray(rng.normal(size=4))


def cost(theta):
    # 𝒥(θ) = ½ ‖Xθ − y‖²
    return 0.5 * jnp.sum((inputs @ theta - targets) ** 2)


def tree(theta):
    return {"w": theta[:2].reshape(2, 1), "b": theta[2:]}


def flat(params):
    return jnp.concatenate([params["w"].reshape(-1), params["b"]])


def run(opt, params, grad_fn, to_flat, pass_params=False):
    state = opt.init(params)
    iterates = []
    for _ in range(NUM_STEPS):
        grads = grad_fn(params)
        updates, state = opt.update(grads, state, params if pass_params else None)
        params = optax.apply_updates(params, updates)
        iterates.append(np.asarray(to_flat(params)))
    return np.stack(iterates), state


tree_grad = jax.grad(lambda p: cost(flat(p)))
flat_grad = jax.grad(cost)
hessian = jax.hessian(cost)
ident = lambda x: x
out = dict(inputs=inputs, targets=targets, start=start, prior_mean=prior_mean)

# --- diagonal ---
opt = ob.blr_diagonal_for_loss(0.05, prior_precision=0.5, prior_mean=0.1, damping=1e-6)
out["diag_ggn"], state = run(opt, tree(start), tree_grad, flat)
mean, variance = ob.get_posterior_diagonal(state)
out.update(diag_ggn_mean=flat(mean), diag_ggn_variance=flat(variance))
opt = ob.blr_diagonal_for_loss(0.01, prior_precision=2.0, hessian_estimator="identity")
out["diag_identity"], _ = run(opt, tree(start), tree_grad, flat)
schedule = optax.exponential_decay(0.05, transition_steps=10, decay_rate=0.5)
opt = ob.blr_with_schedule(schedule, prior_precision=0.5, prior_mean=0.1)
out["diag_schedule"], _ = run(opt, tree(start), tree_grad, flat)

# --- IVON ---
opt = ob.ivon(0.05, beta1=0.9, beta2=0.99, weight_decay=0.1, clip_radius=1.0, ess=30.0)
out["ivon"], state = run(opt, tree(start), tree_grad, flat, pass_params=True)
mean, variance = ob.get_posterior_ivon(state, tree(jnp.asarray(out["ivon"][-1])), 0.1)
out.update(ivon_mean=flat(mean), ivon_variance=flat(variance))
out["ivon_no_params"], _ = run(opt, tree(start), tree_grad, flat)

# --- full rank ---
kwargs = dict(prior_precision=0.5, prior_mean=prior_mean, damping=1e-6)
opt = ob.blr_full_rank_for_loss(0.05, hessian_estimator="ggn", **kwargs)
out["full_ggn"], state = run(opt, start, flat_grad, ident)
mean, covariance = ob.get_posterior_full_rank(state)
out.update(full_ggn_mean=mean, full_ggn_covariance=covariance)
opt = ob.blr_full_rank_for_loss(0.3, hessian_estimator=lambda m, g: hessian(m), **kwargs)
out["full_exact"], state = run(opt, start, flat_grad, ident)
mean, covariance = ob.get_posterior_full_rank(state)
out.update(full_exact_mean=mean, full_exact_covariance=covariance)
opt = ob.newton_for_loss(hessian, damping=1e-6)
out["newton"], _ = run(opt, start, flat_grad, ident)

# --- low rank ---
opt = ob.blr_low_rank_for_loss(0.05, rank=2, hessian_estimator="ggn", **kwargs)
out["low_ggn"], state = run(opt, start, flat_grad, ident)
mean, covariance = ob.get_posterior_low_rank(state)
out.update(low_ggn_mean=mean, low_ggn_covariance=covariance)
opt = ob.blr_low_rank_for_loss(0.3, rank=2, hessian_estimator=lambda m, g: hessian(m), **kwargs)
out["low_exact"], state = run(opt, start, flat_grad, ident)
mean, covariance = ob.get_posterior_low_rank(state)
out.update(low_exact_mean=mean, low_exact_covariance=covariance)

np.savez(sys.argv[1], **{name: np.asarray(value) for name, value in out.items()})
for name, value in out.items():
    print(name, np.asarray(value).shape, float(np.abs(np.asarray(value)).max()))
```

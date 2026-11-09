# radtrax DFS fixture

`radtrax_dfs.npz` holds radtrax's `degrees_of_freedom_for_signal` on its own non-identity-background test problem (`tests/retrieval/var3d/test_diagnostics.py::test_dfs_nonidentity_background`), so `tests/diagnose/test_retrieval.py` can check the port (#20) without importing radtrax, which sits above argminax.

- radtrax at `0aff245` (`origin/main`, 9 November 2026), float64.
- `hess_x`, `hess_xi`: the dense model-space and whitened Hessians (9 × 9) at the zero state; `variances`: the diagonal background B.
- `exact`: N − tr(Hess_ξ⁻¹); `dfs_x_K`, `dfs_xi_K`: radtrax's model-space and whitened estimates with K = 16 and 64 probes, seed 0.
- Both Hessians are diagonal (a diagonal background and pixel-independent observations), so Rademacher probes recover the exact trace, as the values show.

Regenerate by dropping this test next to radtrax's `tests/retrieval/var3d/conftest.py` and running it with `FIXTURE_OUT=radtrax_dfs.npz`:

```python
"""Write radtrax's DFS on its non-identity-background problem, for argminax#20."""

import os

import jax
import jax.numpy as jnp
import numpy as np

from radtrax.retrieval.var3d.background import build_diagonal_background
from radtrax.retrieval.var3d.control import WhiteningTransform
from radtrax.retrieval.var3d.cost import build_cost_x, build_cost_xi
from radtrax.retrieval.var3d.diagnostics import degrees_of_freedom_for_signal

jax.config.update("jax_enable_x64", True)


def test_write_fixture(obs_model_no_optics):
    model = obs_model_no_optics
    ny, nx = 3, 3
    n = ny * nx
    variances = np.array([0.05, 0.1, 0.3, 0.7, 1.0, 1.5, 3.0, 6.0, 12.0])
    B = build_diagonal_background(variances)
    W = WhiteningTransform.from_background(B)
    y = model.forward(jnp.full((ny, nx), 1e-7), linear=False)
    kw = dict(
        forward_fn=model.make_forward(linear=False),
        obs_inv_variance=1e-11,
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    cost_x = build_cost_x(background_op=B, **kw)
    cost_xi = build_cost_xi(whitening=W, **kw)
    basis = np.eye(n)
    hess_x = np.stack([np.asarray(cost_x.hvp(jnp.zeros(n), jnp.asarray(c))) for c in basis], axis=1)
    hess_xi = np.stack([np.asarray(cost_xi.hvp(jnp.zeros(n), jnp.asarray(c))) for c in basis], axis=1)
    out = {"variances": variances, "hess_x": hess_x, "hess_xi": hess_xi, "exact": n - np.trace(np.linalg.inv(hess_xi))}
    for n_probes in (16, 64):
        out[f"dfs_x_{n_probes}"] = degrees_of_freedom_for_signal(
            hessian_vector_product=lambda v: cost_x.hvp(jnp.zeros(n), v),
            state_size=n, background_op=B, whitened=False, n_probes=n_probes, seed=0)
        out[f"dfs_xi_{n_probes}"] = degrees_of_freedom_for_signal(
            hessian_vector_product=lambda v: cost_xi.hvp(jnp.zeros(n), v),
            state_size=n, whitened=True, n_probes=n_probes, seed=0)
    np.savez(os.environ["FIXTURE_OUT"], **{k: np.asarray(v) for k, v in out.items()})
    print({k: (float(v) if np.ndim(v) == 0 else np.shape(v)) for k, v in out.items()})
```

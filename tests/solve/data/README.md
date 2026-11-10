# radtrax solver fixture

`radtrax_solve.npz` holds radtrax's `run_gauss_newton` and `run_dual_psas` on its own linear dual-equals-primal problem (`tests/retrieval/var3d/test_solve.py::test_dual_psas_matches_primal_for_linear_forward`), so `tests/solve/` can check the ports (#23, #24) without importing radtrax, which sits above argminax.

- radtrax at `0aff245` (`origin/main`, 9 November 2026), float64, on a 4 × 4 pixel grid with the linear forward model.
- `jacobian` (3216 × 16): the linear observation operator H; `innovation`: d = y − H(x_b); `background_variance`: B = 10⁻¹² I; `obs_inv_variance`: R⁻¹ = 10⁸; `sqrt_background`: the whitening factor L with B = L Lᵀ.
- `gauss_newton_xi`, `gauss_newton_delta`: radtrax's `run_gauss_newton` (rtol 1e-10, atol 1e-14) on the whitened residuals r(ξ) = (ξ, √R⁻¹ (y − H(x_b + L ξ))), and δx = L ξ̂; `gauss_newton_steps`: its step count.
- `dual_delta`: radtrax's `run_dual_psas` (CG rtol 1e-10, atol 1e-14). It agrees with `gauss_newton_delta` to 1.7e-17.

Regenerate by dropping this test next to radtrax's `tests/retrieval/var3d/conftest.py` and running it with `FIXTURE_OUT=radtrax_solve.npz` (then `np.savez_compressed` it):

```python
"""Write radtrax's run_gauss_newton and run_dual_psas on its linear dual == primal problem."""

import os

import jax
import jax.numpy as jnp
import numpy as np

from radtrax.retrieval.var3d.background import build_diagonal_background
from radtrax.retrieval.var3d.control import WhiteningTransform
from radtrax.retrieval.var3d.solve import run_dual_psas, run_gauss_newton

jax.config.update("jax_enable_x64", True)


def test_write_fixture(obs_model_no_optics):
    model = obs_model_no_optics
    ny, nx = 4, 4
    n = ny * nx
    y_obs = model.forward(jnp.full((ny, nx), 1e-7), linear=True)
    B = build_diagonal_background(1e-12, n_pixels=n)
    R_inv = 1e8
    forward = model.make_forward(linear=True)
    x_b = jnp.zeros((ny, nx))
    base = forward(x_b)

    def H(delta_flat):
        return (forward(x_b + delta_flat.reshape(ny, nx)) - base).reshape(-1)

    jacobian = jax.jacfwd(H)(jnp.zeros(n))
    W = WhiteningTransform.from_background(B)
    sqrt_B = np.stack([np.asarray(W.apply(jnp.asarray(c))) for c in np.eye(n)], axis=1)

    def residual(xi):
        delta = W.apply(xi)
        misfit = (y_obs - forward(x_b + delta.reshape(ny, nx))).reshape(-1)
        return jnp.concatenate([xi, jnp.sqrt(R_inv) * misfit])

    gn = run_gauss_newton(residual_fn=residual, initial_state=jnp.zeros(n), rtol=1e-10, atol=1e-14)
    dual = run_dual_psas(forward_fn=forward, background_op=B, obs_inv_variance=R_inv,
                         background_state=x_b, observation=y_obs, state_shape=(ny, nx),
                         cg_rtol=1e-10, cg_atol=1e-14)
    out = {
        "jacobian": np.asarray(jacobian),
        "innovation": np.asarray((y_obs - base).reshape(-1)),
        "background_variance": np.asarray(1e-12),
        "obs_inv_variance": np.asarray(R_inv),
        "sqrt_background": sqrt_B,
        "gauss_newton_xi": np.asarray(gn.state),
        "gauss_newton_delta": np.asarray(W.apply(jnp.asarray(gn.state))),
        "gauss_newton_steps": np.asarray(gn.n_steps),
        "dual_delta": np.asarray(dual.state),
    }
    np.savez(os.environ["FIXTURE_OUT"], **out)
    print({k: np.shape(v) for k, v in out.items()}, float(np.max(np.abs(out["gauss_newton_delta"] - out["dual_delta"]))))
```

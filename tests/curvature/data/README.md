# vardax posterior fixtures

`vardax_posterior.npz` holds vardax's `LaplaceCovariance` and `GaussNewtonHessian` covariances on two small problems, so `tests/curvature/test_laplace.py` can check the port (gate G1-port, #9) without importing vardax, which sits above argminax.

- vardax at `bfb00ed` (`origin/main`, 9 November 2026), float64, CG at rtol = atol = 1e-14.
- `threedvar_*`: `ThreeDVar` with a dense linear observation operator H (5 × 4) and dense B and R.
- `fourdvar_*`: `StrongFourDVar` with linear dynamics xₜ₊₁ = A xₜ, identity observations at t = 0, 1, 2 and dense B and R.
- Both vardax covariances equal the closed forms (HᵀR⁻¹H + B⁻¹)⁻¹ and (∑ₜ (Aᵗ)ᵀR⁻¹Aᵗ + B⁻¹)⁻¹ to 3e-16.

Regenerate from a vardax checkout with `PYTHONPATH=src .venv/bin/python make_vardax_fixtures.py vardax_posterior.npz`, where the script is:

```python
"""Compute vardax's Laplace and Gauss–Newton posteriors on two small problems."""
import sys
import jax, jax.numpy as jnp, lineax as lx, numpy as np, equinox as eqx
jax.config.update("jax_enable_x64", True)
import vardax

def spd(key, n):
    a = jax.random.normal(key, (n, n))
    return a @ a.T / n + 0.5 * jnp.eye(n)

psd = lambda m: lx.MatrixLinearOperator(m, lx.positive_semidefinite_tag)
cg = lx.CG(rtol=1e-14, atol=1e-14, max_steps=2000)
out = {}

# --- problem 3dvar: linear H (M, N), dense B and R ---
k = jax.random.split(jax.random.key(20261109), 6)
N, M = 4, 5
H = jax.random.normal(k[0], (M, N)); B = spd(k[1], N); R = spd(k[2], M)
xb = jax.random.normal(k[3], (N,)); y = jax.random.normal(k[4], (M,))
model = vardax.ThreeDVar(obs_op=vardax.LinearObs(lx.MatrixLinearOperator(H)), prior_mean=xb, prior_cov_op=psd(B), obs_cov_op=psd(R))
batch = vardax.Batch1D(input=y[None], mask=jnp.ones((1, M)))
analysis = model(batch)[0]
eye = jnp.eye(N)
for name, adapter in [("laplace", vardax.LaplaceCovariance(prior_cov_op=psd(B), obs_cov_op=psd(R), solver=cg)),
                      ("gauss_newton", vardax.GaussNewtonHessian(prior_cov_op=psd(B), obs_cov_op=psd(R), n_krylov=2000, tol=1e-14))]:
    post = adapter(analysis, model.as_analysis_step(), batch)
    out[f"threedvar_{name}"] = np.asarray(jax.vmap(post.cov.mv)(eye))
out.update(threedvar_H=H, threedvar_B=B, threedvar_R=R, threedvar_xb=xb, threedvar_y=y, threedvar_analysis=analysis)

# --- problem strong4dvar: x_{t+1} = A x_t, identity obs, T + 1 = 3 ---
k = jax.random.split(jax.random.key(7), 6)
N, T1 = 3, 3
A = 0.8 * jnp.eye(N) + 0.2 * jax.random.normal(k[0], (N, N)); B = spd(k[1], N); R = spd(k[2], N)
xb = jax.random.normal(k[3], (N,)); y = jax.random.normal(k[4], (T1, N))

class Linear(eqx.Module):
    matrix: jax.Array
    dt: float = 1.0
    def step(self, x, dt):
        return self.matrix @ x

model = vardax.StrongFourDVar(forward=Linear(A), obs_op=vardax.MaskedIdentity(), prior_mean=xb, prior_cov_op=psd(B), obs_cov_op=psd(R))
batch = vardax.Batch1D(input=y[None], mask=jnp.ones((1, T1, N)))
analysis = model(batch)[0]
eye = jnp.eye(N)
for name, adapter in [("laplace", vardax.LaplaceCovariance(prior_cov_op=psd(B), obs_cov_op=psd(R), solver=cg)),
                      ("gauss_newton", vardax.GaussNewtonHessian(prior_cov_op=psd(B), obs_cov_op=psd(R), n_krylov=2000, tol=1e-14))]:
    post = adapter(analysis, model, batch)
    out[f"fourdvar_{name}"] = np.asarray(jax.vmap(post.cov.mv)(eye))
out.update(fourdvar_A=A, fourdvar_B=B, fourdvar_R=R, fourdvar_xb=xb, fourdvar_y=y, fourdvar_analysis=analysis)

np.savez(sys.argv[1], **{key: np.asarray(value) for key, value in out.items()})
for key in sorted(out):
    print(key, np.asarray(out[key]).shape)
```

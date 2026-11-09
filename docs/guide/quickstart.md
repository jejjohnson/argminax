# Quickstart

## Install

argminax is not on PyPI yet; install it from GitHub:

```bash
pip install "argminax @ git+https://github.com/jejjohnson/argminax.git"
```

That brings in JAX, Equinox, lineax, optimistix and gaussx. The `matfree` extra adds stochastic Lanczos quadrature for log-determinants of large operators; the functions that need it import it when they are called, and raise [`MissingExtraError`](xref:api#argminax.MissingExtraError) naming the extra when it is missing.

```python
import argminax

argminax.__version__  # '0.0.1'
```

## Double precision

Curvature and diagnostics invert or factor a Hessian, which loses digits fast in single precision. Turn on 64-bit floats at the start of your program, before any array is created:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

argminax's own tests run this way and compare against closed forms to $10^{-12}$.

## The calling convention

argminax follows optimistix: a cost is a function `fn(y, args)` of the state and a pytree of fixed arguments, so a function written for `optx.minimise` or `optx.least_squares` already works with argminax. The two shapes are named in `argminax._typing`:

| Alias | Signature | Used by |
|---|---|---|
| `Objective` | `fn(x, args) -> scalar`, the cost $J$ | `optx.minimise` |
| `Residuals` | `fn(x, args) -> pytree`, with $J = \frac{1}{2} \lVert r \rVert^2$ | `optx.least_squares` |

For the retrieval cost, the residuals are whitened by the two covariances, $r(x) = (S_\varepsilon^{-1/2} (y - F(x)), S_a^{-1/2} (x - x_a))$, which is what the Gauss–Newton solvers and the curvature at the minimum read.

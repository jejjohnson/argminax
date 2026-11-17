# Quickstart

## Install

argminax is not on PyPI yet; install it from GitHub:

```bash
pip install "argminax @ git+https://github.com/jejjohnson/argminax.git"
```

That brings in JAX, Equinox, lineax, optimistix and gaussx. The `matfree` extra adds stochastic Lanczos quadrature for log-determinants of large operators; the functions that need it import it when they are called, and raise [`MissingExtraError`](xref:api#argminax.MissingExtraError) naming the extra when it is missing.

```python
import argminax

argminax.__version__  # a dotted version string
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

## A first retrieval

Three unknowns, five noisy measurements and a nonlinear forward model $F(x) = K \tanh(x)$. The steps are the four you repeat for any retrieval: solve for $\hat x$, read the uncertainty off the curvature, count the degrees of freedom for signal, and differentiate $\hat x$ with respect to the data.

```python
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float

from argminax.curvature import gauss_newton
from argminax.diagnose import averaging_kernel, degrees_of_freedom
from argminax.solve import GaussNewtonCG

# Curvature and diagnostics factor a Hessian, so use double precision.
jax.config.update("jax_enable_x64", True)

# Shapes: N = 3 state elements; M = 5 measurements
K: Float[Array, "5 3"] = jnp.array(
    [
        [1.0, 0.5, 0.1],
        [0.2, 1.0, 0.4],
        [0.1, 0.3, 1.0],
        [0.6, 0.2, 0.2],
        [0.3, 0.7, 0.5],
    ]
)
prior_mean: Float[Array, " 3"] = jnp.zeros(3)
noise_std, prior_std = 0.3, 1.0


def forward(x: Float[Array, " 3"]) -> Float[Array, " 5"]:
    return K @ jnp.tanh(x)  # (5,)


def residuals(x, y):
    # r = (Sε^(-1/2) (y − F(x)), Sₐ^(-1/2) (x − xₐ)), so J(x) = ½ ‖r‖²
    return (y - forward(x)) / noise_std, (x - prior_mean) / prior_std


y: Float[Array, " 5"] = forward(jnp.array([0.8, -0.4, 0.3])) + 0.05  # (5,)

# x̂ = argmin J(x): an optimistix least-squares solve, matrix-free inside.
solver = GaussNewtonCG(rtol=1e-8, atol=1e-8)
solution = optx.least_squares(residuals, solver, prior_mean, args=y)
x_hat: Float[Array, " 3"] = solution.value  # (3,)  [0.667, -0.238, 0.271]

# How sharp is the minimum? Ŝ = Ĥ⁻¹ with Ĥ = Jᵣᵀ Jᵣ at x̂.
posterior = gauss_newton(residuals, x_hat, y)
covariance = posterior.covariance().as_matrix()  # (3, 3)
std = jnp.sqrt(jnp.diag(covariance))  # (3,)  [0.427, 0.344, 0.340]

# How much of x̂ came from the measurement? dₛ = tr A with A = I − Ŝ Sₐ⁻¹.
dofs = degrees_of_freedom(averaging_kernel(posterior))  # dₛ ≈ 2.58 of 3


# x̂ is differentiable in the data: one more linear solve, not a pass through every iteration.
def retrieve(y):
    return optx.least_squares(residuals, solver, prior_mean, args=y).value


dx_dy = jax.jacobian(retrieve)(y)  # (3, 5)

# A swath of pixels is one vmap.
batch = jnp.stack([y, y + 0.1, y - 0.1])  # (3, 5)
x_hats = jax.jit(jax.vmap(retrieve))(batch)  # (3, 3)
```

- `GaussNewtonCG` is an optimistix least-squares solver, so it is called through `optx.least_squares`, and `solution.result` reports whether it converged.
- `argminax.curvature.gauss_newton` builds the Laplace posterior $\mathcal{N}(\hat x, \hat S)$ from the same residuals, with $\hat S = \hat H^{-1}$ and $\hat H = J_r^\top J_r$ at $\hat x$. It is a gaussx operator, so `covariance().as_matrix()` is only for small problems like this one.
- `degrees_of_freedom` is exact without a `key` and a Hutchinson estimate with one, which returns its standard error, so a million pixels are one `vmap` over split keys.
- The gradient `dx_dy` uses optimistix's `ImplicitAdjoint`, so it costs one more linear solve, not a pass through every iteration.

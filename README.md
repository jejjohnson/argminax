<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="argminax" src="docs/assets/logo-light.svg" width="340">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/jejjohnson/argminax/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/jejjohnson/argminax/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/jejjohnson/argminax/actions/workflows/lint.yml"><img alt="Lint" src="https://github.com/jejjohnson/argminax/actions/workflows/lint.yml/badge.svg"></a>
  <a href="https://github.com/jejjohnson/argminax/actions/workflows/typecheck.yml"><img alt="Type Check" src="https://github.com/jejjohnson/argminax/actions/workflows/typecheck.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/jejjohnson/argminax"><img alt="codecov" src="https://codecov.io/gh/jejjohnson/argminax/branch/main/graph/badge.svg"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://jejjohnson.github.io/argminax/"><b>Docs</b></a> ·
  <a href="https://jejjohnson.github.io/argminax/reference/"><b>API</b></a> ·
  <a href="https://jejjohnson.github.io/argminax/quickstart/"><b>Quickstart</b></a> ·
  <a href="https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/argminax.md"><b>Design</b></a>
</p>

**Optimisation with uncertainty at the minimum for JAX: solvers, curvature, retrieval diagnostics and minimiser adjoints on optimistix and gaussx.**

A retrieval is not finished when the optimiser stops. argminax finds $`\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} J(\mathbf{x})`$, then says how sharp that minimum is (the posterior covariance $`\hat{\mathbf{S}}`$, the averaging kernel $`\mathbf{A}`$, the degrees of freedom for signal) and how $`\hat{\mathbf{x}}`$ depends on what went in. For a measurement $`\mathbf{y}`$, forward model $`F`$, prior state $`\mathbf{x}_a`$ and covariances $`\mathbf{S}_\varepsilon`$ and $`\mathbf{S}_a`$, the cost it minimises most often is

```math
J(\mathbf{x}) = \frac{1}{2} (\mathbf{y} - F(\mathbf{x}))^\top \mathbf{S}_\varepsilon^{-1} (\mathbf{y} - F(\mathbf{x})) + \frac{1}{2} (\mathbf{x} - \mathbf{x}_a)^\top \mathbf{S}_a^{-1} (\mathbf{x} - \mathbf{x}_a),
```

and the curvature at the minimum is $`\hat{\mathbf{S}} = \hat{\mathbf{H}}^{-1}`$, with $`\hat{\mathbf{H}}`$ the Gauss–Newton Hessian of $`J`$ at $`\hat{\mathbf{x}}`$. The sharper the valley, the narrower the Gaussian under it, which is the picture in the logo.

argminax extends optimistix as gaussx extends lineax: its solvers are optimistix solvers and its adjoints are optimistix adjoints, so `optx.minimise` and `optx.least_squares` keep working, and the curvature it returns is a gaussx operator that is never formed densely.

## What it gives you

| Module | What it gives you |
|---|---|
| `argminax.solve` | `GaussNewtonCG` (Gauss–Newton and Levenberg–Marquardt with a matrix-free inner CG), `dual_solve` (the observation-space increment), `Reparametrised` (positive and bounded states), and `laplace_mode` for GMRF, GP and Markov-GP models |
| `argminax.curvature` | `Laplace`, the Gaussian $`\mathcal{N}(\hat{\mathbf{x}}, \hat{\mathbf{H}}^{-1})`$ with its covariance, log-determinant and samples; `gauss_newton` builds it from whitened residuals |
| `argminax.diagnose` | `averaging_kernel`, `degrees_of_freedom`, `information_content`, `evidence` and the prior-precision search, exact or stochastic with a standard error |
| `argminax.adjoint` | `KStep` and `OneStep`, the adjoints for a minimiser stopped early, as in learned 4D-Var |

## Installation

argminax is not on PyPI; the stack pins it by git tag. It brings in JAX, Equinox, lineax, optimistix and gaussx. The `matfree` extra adds stochastic Lanczos quadrature for log-determinants of large operators.

```bash
pip install "argminax @ git+https://github.com/jejjohnson/argminax.git"
```

## Quick Start

A nonlinear retrieval of three unknowns from five measurements: solve for $`\hat{\mathbf{x}}`$, read the uncertainty off the curvature, count how much the measurement told us, and differentiate the answer with respect to the data.

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

Nothing here is specific to this model. Swap `forward` for a radiative transfer code, a plume model or a PDE solve, and the same four lines give the retrieval, its covariance, its degrees of freedom and its gradient. The [quickstart](https://jejjohnson.github.io/argminax/quickstart/) adds the calling convention and why to run in double precision.

## Where it sits

argminax is Core tier 1 of the [GeoStack](https://github.com/jejjohnson/geostack): it imports gaussx (tier 0) and extends optimistix, and everything that minimises can import it. vardax, radtrax, plumax, pyrox-gp, pyrox-lgm and pyrox-nn take their solvers, curvature and diagnostics from it. The design is the [argminax design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/argminax.md).

## Development

```bash
git clone https://github.com/jejjohnson/argminax.git
cd argminax
uv run poe install     # every dependency group + the pre-commit hooks
uv run poe check       # lint, format check, typecheck, tests with the 90 % coverage gate
uv run poe docs        # the MyST prose and the Zensical API reference, assembled into public/
```

`uv run poe` lists every task. [`AGENTS.md`](AGENTS.md) holds the rules for contributors and coding agents: the module map, the contracts, the tests that enforce them and the house style. [`docs/contributing.md`](docs/contributing.md) has the label taxonomy and the epic model.

## License

MIT; see [`LICENSE`](LICENSE).

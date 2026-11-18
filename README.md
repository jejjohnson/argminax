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

Everything in argminax is about one object, the minimiser of a cost as a function of what the cost was built from:

```math
\hat{\mathbf{x}}(\boldsymbol{\theta}) = \arg\min_{\mathbf{x}} J(\mathbf{x}, \boldsymbol{\theta})
```

$`\mathbf{x}`$ is the state we solve for and $`\boldsymbol{\theta}`$ is everything that went into the cost and stays fixed during the solve: the measurement, the prior, the covariances, the parameters of the forward model. A solver gives one value of $`\hat{\mathbf{x}}`$. argminax treats it as a function, and answers three questions about it.

| Question | Equation | Module |
|---|---|---|
| Where is the minimum? | $`\nabla_{\mathbf{x}} J(\hat{\mathbf{x}}, \boldsymbol{\theta}) = \mathbf{0}`$ | `argminax.solve`, or any optimistix solver |
| How sharp is it? | $`\hat{\mathbf{H}} = \nabla^2_{\mathbf{x}} J(\hat{\mathbf{x}}, \boldsymbol{\theta})`$ and $`\hat{\mathbf{S}} = \hat{\mathbf{H}}^{-1}`$ | `argminax.curvature`, `argminax.diagnose` |
| How does it move when $`\boldsymbol{\theta}`$ moves? | $`\hat{\mathbf{H}} \partial_{\boldsymbol{\theta}} \hat{\mathbf{x}} = -\partial_{\boldsymbol{\theta}} \nabla_{\mathbf{x}} J(\hat{\mathbf{x}}, \boldsymbol{\theta})`$ | `optx.ImplicitAdjoint`, `argminax.adjoint` |

The third row is implicit differentiation: differentiate the first row with respect to $`\boldsymbol{\theta}`$ and the implicit function theorem gives a linear system in the Hessian $`\hat{\mathbf{H}}`$ ([Blondel et al., 2022](https://arxiv.org/abs/2105.15183)). So the derivative of a converged solve costs one more linear solve, whatever the solver did to get there and however many iterations it took, and the same matrix answers the second and the third question. When $`J`$ is a negative log posterior, its inverse $`\hat{\mathbf{S}}`$ is the covariance of the Laplace approximation, and the derivative with respect to the measurement is the gain matrix of retrieval theory. That is what makes a minimiser a layer: an outer cost $`L(\hat{\mathbf{x}}(\boldsymbol{\theta}))`$ can be minimised over $`\boldsymbol{\theta}`$ by gradient descent (bilevel optimisation, hyperparameter learning, learned 4D-Var), with $`\nabla_{\boldsymbol{\theta}} L = (\partial_{\boldsymbol{\theta}} \hat{\mathbf{x}})^\top \nabla_{\mathbf{x}} L`$.

The cost argminax minimises most often is the retrieval cost. For a measurement $`\mathbf{y}`$, forward model $`F`$, prior state $`\mathbf{x}_a`$ and covariances $`\mathbf{S}_\varepsilon`$ and $`\mathbf{S}_a`$,

```math
J(\mathbf{x}, \boldsymbol{\theta}) = \frac{1}{2} (\mathbf{y} - F(\mathbf{x}))^\top \mathbf{S}_\varepsilon^{-1} (\mathbf{y} - F(\mathbf{x})) + \frac{1}{2} (\mathbf{x} - \mathbf{x}_a)^\top \mathbf{S}_a^{-1} (\mathbf{x} - \mathbf{x}_a)
```

with $`\boldsymbol{\theta} = (\mathbf{y}, \mathbf{x}_a, \mathbf{S}_\varepsilon, \mathbf{S}_a)`$ and whatever parameters $`F`$ has. Here $`\hat{\mathbf{H}}`$ is taken in its Gauss–Newton form, $`\hat{\mathbf{S}}`$ is the posterior covariance, and the averaging kernel $`\mathbf{A}`$ and the degrees of freedom for signal follow from it. The logo is that picture: iterates descending to the minimum of a cost, and the covariance ellipse they land in.

Two things go beyond the table. A solver stopped early has no stationarity condition to differentiate, so `argminax.adjoint` differentiates through its last few iterations (`KStep`, `OneStep`). And a model in training never reaches the minimum, so `argminax.blr` carries the Gaussian $`\mathcal{N}(\hat{\mathbf{x}}, \hat{\mathbf{S}})`$ along as an optax transform (the Bayesian learning rule).

argminax extends optimistix as gaussx extends lineax: its solvers are optimistix solvers and its adjoints are optimistix adjoints, so `optx.minimise` and `optx.least_squares` keep working, and the curvature it returns is a gaussx operator that is never formed densely.

## What it gives you

| Module | What it gives you |
|---|---|
| `argminax.solve` | `GaussNewtonCG` (Gauss–Newton and Levenberg–Marquardt with a matrix-free inner CG), `dual_solve` (the observation-space increment), `Reparametrised` (positive and bounded states), and `laplace_mode` for GMRF, GP and Markov-GP models |
| `argminax.blr` | the Bayesian learning rule as optax transforms that carry the Gaussian during training: `blr_diagonal`, `blr_full_rank`, `blr_low_rank`, `ivon`, `newton`, and `blr_posterior`, which reads any of their states as a `Laplace` |
| `argminax.curvature` | `Laplace`, the Gaussian $`\mathcal{N}(\hat{\mathbf{x}}, \hat{\mathbf{H}}^{-1})`$ with its covariance, log-determinant and samples; `gauss_newton` builds it from whitened residuals |
| `argminax.diagnose` | `averaging_kernel`, `degrees_of_freedom`, `information_content`, `evidence` and the prior-precision search, exact or stochastic with a standard error |
| `argminax.adjoint` | `KStep` and `OneStep`, the adjoints for a minimiser stopped early, as in learned 4D-Var |

## Installation

argminax is not on PyPI; the stack pins it by git tag. It brings in JAX, Equinox, lineax, optimistix and gaussx. The `matfree` extra adds stochastic Lanczos quadrature for log-determinants of large operators. The `optax` extra adds optax, for the transforms of `argminax.blr`.

```bash
pip install "argminax @ git+https://github.com/jejjohnson/argminax.git"
```

## Quick Start

A nonlinear retrieval of three unknowns from five measurements, with $`\boldsymbol{\theta}`$ the measurement and the width of the prior. The three questions in order: solve for $`\hat{\mathbf{x}}(\boldsymbol{\theta})`$, read the uncertainty off the curvature, and differentiate $`\hat{\mathbf{x}}`$ with respect to $`\boldsymbol{\theta}`$.

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
noise_std = 0.3


def forward(x: Float[Array, " 3"]) -> Float[Array, " 5"]:
    return K @ jnp.tanh(x)  # (5,)


def residuals(x, theta):
    # θ = (y, log σₐ): the measurement and the log of the prior's width
    # r = ((y − F(x)) / σε, (x − xₐ) / σₐ), so J(x, θ) = ½ ‖r‖²
    y, log_prior_std = theta
    return (y - forward(x)) / noise_std, (x - prior_mean) / jnp.exp(log_prior_std)


x_ref: Float[Array, " 3"] = jnp.array([0.8, -0.4, 0.3])
y: Float[Array, " 5"] = forward(x_ref) + 0.05  # (5,)
theta = (y, jnp.asarray(0.0))

# --- 1. Where is the minimum?  x̂(θ) = argmin J(x, θ) ---
solver = GaussNewtonCG(rtol=1e-8, atol=1e-8)


def retrieve(theta):
    # an optimistix least-squares solve, matrix-free inside
    return optx.least_squares(residuals, solver, prior_mean, args=theta).value


x_hat: Float[Array, " 3"] = retrieve(theta)  # (3,)  [0.667, -0.238, 0.271]

# --- 2. How sharp is it?  Ŝ = Ĥ⁻¹ with Ĥ = Jᵣᵀ Jᵣ at x̂ ---
posterior = gauss_newton(residuals, x_hat, theta)
covariance = posterior.covariance().as_matrix()  # (3, 3)
std = jnp.sqrt(jnp.diag(covariance))  # (3,)  [0.427, 0.344, 0.340]

# How much of x̂ came from the measurement? dₛ = tr A with A = I − Ŝ Sₐ⁻¹.
dofs = degrees_of_freedom(averaging_kernel(posterior))  # dₛ ≈ 2.58 of 3

# --- 3. How does it move with θ?  Ĥ ∂x̂/∂θ = −∂_θ ∇ₓJ ---
# One more linear solve per direction, not a pass through every iteration.
dx_dy, dx_dlog_std = jax.jacobian(retrieve)(theta)
dx_dy.shape  # (3, 5), the gain: how x̂ answers each measurement
dx_dlog_std  # (3,)  [0.247, -0.175, 0.103], how x̂ answers a wider prior


# The minimiser as a layer: an outer cost of x̂(θ), and its gradient in θ.
def outer_cost(theta):
    # L(θ) = ‖x̂(θ) − x_ref‖²
    return jnp.sum((retrieve(theta) - x_ref) ** 2)


_, dcost_dlog_std = jax.grad(outer_cost)(theta)  # −0.128: widen the prior

# A swath of pixels is one vmap over θ.
batch = jnp.stack([y, y + 0.1, y - 0.1])  # (3, 5)
x_hats = jax.jit(jax.vmap(lambda y: retrieve((y, 0.0))))(batch)  # (3, 3)
```

Nothing here is specific to this model. Swap `forward` for a radiative transfer code, a plume model or a PDE solve, and the same lines give the retrieval, its covariance, its degrees of freedom and its derivatives. For a forward model that is linear, the two derivatives above are quantities retrieval theory already names: $`\partial_{\mathbf{y}} \hat{\mathbf{x}} = \hat{\mathbf{S}} \mathbf{K}^\top \mathbf{S}_\varepsilon^{-1}`$ is the gain matrix, and the averaging kernel is $`\mathbf{A} = (\partial_{\mathbf{y}} \hat{\mathbf{x}}) \mathbf{K}`$. The [quickstart](https://jejjohnson.github.io/argminax/quickstart/) derives the third question and adds the calling convention and why to run in double precision.

### During training

The solvers above run to a minimum first. The transforms of `argminax.blr` carry the same Gaussian while they train (the variable they optimise is a model's parameter vector, so it is written $`\boldsymbol{\theta}`$ there, and plays the part of $`\mathbf{x}`$ above): each is an optax `GradientTransformation` (the Bayesian learning rule), so it drops into an optax loop, and its state reads out as the `Laplace` object used above. Here a line is fitted to 200 noisy points; the cost is quadratic, so one full-rank step with the exact curvature is the exact posterior.

```python
import lineax as lx

from argminax.blr import blr_full_rank, blr_posterior
from argminax.curvature import marginal_variance

# Shapes: N = 200 observations; D = 2 parameters (slope, intercept)
key_inputs, key_noise = jax.random.split(jax.random.key(0))
inputs = jax.random.uniform(key_inputs, (200,), minval=-2.0, maxval=2.0)
targets = 2.0 * inputs - 1.0 + 0.5 * jax.random.normal(key_noise, (200,))
features = jnp.stack([inputs, jnp.ones(200)], axis=1)  # (200, 2)


def cost(theta):
    # 𝒥(θ) = ‖y − Φθ‖² / (2σ²)
    return 0.5 * jnp.sum((targets - features @ theta) ** 2) / 0.5**2


curvature = features.T @ features / 0.5**2  # C = ΦᵀΦ / σ², (2, 2)
optimiser = blr_full_rank(
    1.0, prior_precision=1.0, curvature=lambda mean, grad: curvature, damping=0.0
)
theta = jnp.zeros(2)
state = optimiser.init(theta)
updates, state = optimiser.update(jax.grad(cost)(theta), state)

posterior = blr_posterior(state)  # an argminax.curvature.Laplace
posterior.mode  # (2,)  [2.016, -0.989]
jnp.sqrt(marginal_variance(posterior))  # (2,)  [0.031, 0.035]

prior_cov = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
degrees_of_freedom(averaging_kernel(posterior, prior_cov)).value  # 1.998 of 2
```

`blr_diagonal` and `ivon` work on a parameter pytree of any size as a drop-in for `optax.adam`. The [guide](https://jejjohnson.github.io/argminax/blr/) derives the rule and says what the variance of each curvature estimate means. These need the `optax` extra: `pip install "argminax[optax] @ git+https://github.com/jejjohnson/argminax.git"`.

## Where it sits

argminax is Core tier 1 of the [GeoStack](https://github.com/jejjohnson/geostack): it imports gaussx (tier 0) and extends optimistix and, for the transforms of `argminax.blr`, optax, and everything that minimises can import it. vardax, radtrax, plumax, pyrox-gp, pyrox-lgm and pyrox-nn take their solvers, curvature and diagnostics from it. The design is the [argminax design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/argminax.md).

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

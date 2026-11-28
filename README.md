<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="argminax" src="docs/assets/logo-light.svg" width="340">
</picture>

**Optimisation with uncertainty at the minimum for JAX: solvers, curvature, retrieval diagnostics, minimiser adjoints and Bayesian optimisers, on optimistix, optax and gaussx.**

[![Tests](https://github.com/jejjohnson/argminax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/argminax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/argminax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/argminax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/argminax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/argminax/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/argminax/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/argminax/)
[![codecov](https://codecov.io/gh/jejjohnson/argminax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/argminax)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

📖 [Documentation](https://jejjohnson.github.io/argminax/) ·
🚀 [Quickstart](https://jejjohnson.github.io/argminax/quickstart/) ·
🧠 [Bayesian learning rule](https://jejjohnson.github.io/argminax/blr/) ·
📚 [API reference](https://jejjohnson.github.io/argminax/reference/) ·
🧭 [Design](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/argminax.md)

</div>

---

A retrieval is not finished when the optimiser stops. We also want to know how sharp the minimum is (how much the measurement actually constrained the state) and how the answer moves when the measurement, the prior or the forward model change. So argminax treats the minimiser of a cost as a function of everything the cost was built from,

```math
\hat{\mathbf{x}}(\mathbf{y}, \boldsymbol{\theta}) = \mathrm{argmin}_{\mathbf{x}} \ \mathcal{J}(\mathbf{x}; \mathbf{y}, \boldsymbol{\theta})
```

where $`\mathbf{x} \in \mathbb{R}^{D_x}`$ is the state we solve for, $`\mathbf{y} \in \mathbb{R}^{D_y}`$ is the measurement, $`\boldsymbol{\theta} \in \mathbb{R}^{D_\theta}`$ holds the parameters (the prior, the covariances, the parameters of the forward model) and $`\mathcal{J}`$ is a scalar cost; $`\mathbf{y}`$ and $`\boldsymbol{\theta}`$ stay fixed during the solve. A solver gives one value of $`\hat{\mathbf{x}}`$, whereas argminax answers questions about the whole function, and the answers all come from the same matrix, the Hessian of the cost at the minimum.

```python
# Shapes: D_x state elements; D_y measurements; D_θ parameters
# (residuals, solver, prior_mean, retrieve and the inputs as in the quickstart below)

# (y, θ), what stays fixed during the solve
args = (measurement, log_prior_std)
# today (optimistix)
# x̂(y, θ) = argmin_x 𝒥(x; y, θ)
x_hat = optx.least_squares(residuals, solver, prior_mean, args=args).value  # → (D_x,)
# today (argminax)
# 𝒩(x̂, Σ_x∣y), the Laplace posterior at the minimum
posterior = gauss_newton(residuals, x_hat, args)  # (D_x, D_x) PSD operator
# today (jax)
# G = ∂x̂/∂y and ∂x̂/∂θ, by the implicit function theorem
gain, jacobian = jax.jacobian(retrieve, argnums=(0, 1))(*args)  # (D_x, D_y), (D_x, D_θ)
```

None of it is a new framework. The solvers of argminax are [`optimistix`](https://github.com/patrick-kidger/optimistix) solvers and its adjoints are optimistix adjoints, so `optx.minimise` and `optx.least_squares` keep working, and `jax.grad` through a converged solve is one more linear solve. The curvature it returns is a [`gaussx`](https://github.com/jejjohnson/gaussx) operator on [`lineax`](https://github.com/patrick-kidger/lineax), so a Hessian over a large state is never formed densely unless you ask. And for a model that is trained rather than solved, `argminax.blr` gives [`optax`](https://github.com/google-deepmind/optax) gradient transformations that carry the same Gaussian along the way (the Bayesian learning rule), so they drop into an optax training loop like any other optimiser.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_overview/argminax_overview.png" alt="argminax in one picture: you supply a cost, the measurement y and the parameters θ, and a solver or a training loop; they define one minimum and the Hessian there; five questions, one module each, read that Hessian; out come the retrieval, the Laplace posterior, the diagnostics, the derivatives and the trained posterior; optimistix, gaussx with lineax, and optax underneath" width="100%"></p>

## 🌟 Highlights

- 🎯 **Solvers optimistix lacks**: Gauss–Newton and Levenberg–Marquardt with a matrix-free inner CG, the observation-space (dual) solve, bounded states, and the Laplace mode of GMRF, GP and Markov-GP models.
- 📐 **Curvature as operators**: the Laplace posterior $`\mathcal{N}(\hat{\mathbf{x}}, \boldsymbol{\Sigma}_{x \mid y})`$ with its covariance, log-determinant, marginal variances and samples, never a dense matrix you did not ask for.
- 🛰️ **Retrieval diagnostics**: the averaging kernel, the degrees of freedom for signal, the information content and the evidence, exact for a small state and stochastic with a standard error for a large one.
- 🧷 **Adjoints for any stopping point**: implicit differentiation at a converged minimum through optimistix, and `KStep` and `OneStep` (Bolte et al., 2023) for a solver stopped early, as in learned 4D-Var.
- 🧠 **Bayesian optimisers for optax**: the Bayesian learning rule (Khan and Rue, 2023) as optax `GradientTransformation`s, from a diagonal Gaussian for any parameter pytree (`blr_diagonal`, `ivon`) to full-rank and low-rank ones, whose state reads out as the same Laplace posterior.
- ✅ **Tested against closed forms**: linear-Gaussian closed forms, SciPy and finite differences, in double precision.

## 📐 Five questions about one minimiser

We read $`\mathcal{J}`$ as a negative log posterior and write $`\boldsymbol{\Sigma}_x \in \mathbb{R}^{D_x \times D_x}`$ for the covariance of its prior, so the Hessian at the minimum is the inverse of the Laplace covariance $`\boldsymbol{\Sigma}_{x \mid y} \in \mathbb{R}^{D_x \times D_x}`$. During training the state is instead described by a Gaussian $`q = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1})`$ over $`\mathbf{x}`$, with mean $`\boldsymbol{\mu} \in \mathbb{R}^{D_x}`$ and positive-definite precision $`\boldsymbol{\Lambda} \in \mathbb{R}^{D_x \times D_x}`$.

| Question | Equation | Module |
|---|---|---|
| Where is the minimum? | $`\nabla_{\mathbf{x}} \mathcal{J}(\hat{\mathbf{x}}; \mathbf{y}, \boldsymbol{\theta}) = \mathbf{0}`$ | `argminax.solve`, or any optimistix solver |
| How sharp is it? | $`\boldsymbol{\Sigma}_{x \mid y}^{-1} = \nabla^2_{\mathbf{x}} \mathcal{J}(\hat{\mathbf{x}}; \mathbf{y}, \boldsymbol{\theta})`$ | `argminax.curvature` |
| How much came from the measurement? | $`d_s = \mathrm{tr}(\mathbf{I} - \boldsymbol{\Sigma}_{x \mid y} \boldsymbol{\Sigma}_x^{-1})`$ | `argminax.diagnose` |
| How does it move when $`\mathbf{y}`$ or $`\boldsymbol{\theta}`$ moves? | $`\boldsymbol{\Sigma}_{x \mid y}^{-1} [\partial_{\mathbf{y}} \hat{\mathbf{x}}, \partial_{\boldsymbol{\theta}} \hat{\mathbf{x}}] = -[\nabla^2_{\mathbf{x}\mathbf{y}} \mathcal{J}, \nabla^2_{\mathbf{x}\boldsymbol{\theta}} \mathcal{J}]`$ | `optx.ImplicitAdjoint`, `argminax.adjoint` |
| What Gaussian does training leave? | $`\mathbb{E}_q[\nabla_{\mathbf{x}} \mathcal{J}] = \mathbf{0}, \ \boldsymbol{\Lambda} = \mathbb{E}_q[\nabla^2_{\mathbf{x}} \mathcal{J}]`$ | `argminax.blr` |

The third row compares the posterior with the prior. The trace counts the directions the measurement decided, so $`d_s`$ is near 0 when the prior decides everything and approaches $`D_x`$ when the measurement does; `argminax.diagnose` also gives the averaging kernel, the information content and the evidence.

The fourth row is implicit differentiation, with every term evaluated at the minimum. Differentiating the first row with respect to $`\mathbf{y}`$ and $`\boldsymbol{\theta}`$ gives a linear system in the same Hessian with one right-hand side for each (Blondel et al., 2022), so the gain $`\mathbf{G} = \partial_{\mathbf{y}} \hat{\mathbf{x}} \in \mathbb{R}^{D_x \times D_y}`$ and the $`D_x \times D_\theta`$ Jacobian $`\partial_{\boldsymbol{\theta}} \hat{\mathbf{x}}`$ of a converged solve cost one more linear solve per direction, whatever the solver did to get there and however many iterations it took. The gain says how the retrieval answers the measurement. The Jacobian with respect to the parameters is what makes a minimiser a layer: an outer loss $`\mathcal{L}(\hat{\mathbf{x}}(\mathbf{y}, \boldsymbol{\theta}))`$ can be minimised over $`\boldsymbol{\theta}`$ by gradient descent (bilevel optimisation, hyperparameter learning, learned 4D-Var), with

```math
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\theta}} = \left(\partial_{\boldsymbol{\theta}} \hat{\mathbf{x}}\right)^\top \nabla_{\mathbf{x}} \mathcal{L}
```

The fifth row is for a model that is trained rather than solved. The Bayesian learning rule moves a Gaussian $`q`$ over $`\mathbf{x}`$ with every optax step, and the equation in the table is its fixed point (Khan and Rue, 2023). With the gradient and the curvature evaluated at the mean, the usual way to run an optimiser, and the exact curvature, that fixed point is the first two rows again: $`\boldsymbol{\mu} = \hat{\mathbf{x}}`$ and $`\boldsymbol{\Lambda} = \boldsymbol{\Sigma}_{x \mid y}^{-1}`$, which `blr_posterior` reads out as the same Laplace posterior the other modules use.

The cost we minimise most often is the retrieval cost. For a measurement $`\mathbf{y} \in \mathbb{R}^{D_y}`$, a forward model $`h: \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}`$, a prior $`\mathcal{N}(\boldsymbol{\mu}_x, \boldsymbol{\Sigma}_x)`$ and a measurement-error covariance $`\boldsymbol{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}`$,

```math
\mathcal{J}(\mathbf{x}; \mathbf{y}, \boldsymbol{\theta}) = \frac{1}{2} \Vert \mathbf{y} - h(\mathbf{x}) \Vert^2_{\boldsymbol{\Sigma}_\varepsilon^{-1}} + \frac{1}{2} \Vert \mathbf{x} - \boldsymbol{\mu}_x \Vert^2_{\boldsymbol{\Sigma}_x^{-1}}
```

with $`\boldsymbol{\theta} = (\boldsymbol{\mu}_x, \boldsymbol{\Sigma}_\varepsilon, \boldsymbol{\Sigma}_x)`$ plus whatever parameters $`h`$ has. Its Gauss–Newton curvature is $`\boldsymbol{\Sigma}_{x \mid y}^{-1} = \mathbf{H}^\top \boldsymbol{\Sigma}_\varepsilon^{-1} \mathbf{H} + \boldsymbol{\Sigma}_x^{-1}`$, with $`\mathbf{H} = \mathbf{J}_h(\hat{\mathbf{x}}) \in \mathbb{R}^{D_y \times D_x}`$ the Jacobian of $`h`$ at the minimum, and the quantities of retrieval theory (Rodgers, 2000) are readings of it. The gain is $`\mathbf{G} = \boldsymbol{\Sigma}_{x \mid y} \mathbf{H}^\top \boldsymbol{\Sigma}_\varepsilon^{-1}`$, the averaging kernel (Rodgers' $`\mathbf{A}`$) is $`\mathbf{J}_{\hat{x}} = \mathbf{G} \mathbf{H} = \mathbf{I} - \boldsymbol{\Sigma}_{x \mid y} \boldsymbol{\Sigma}_x^{-1} \in \mathbb{R}^{D_x \times D_x}`$, and the degrees of freedom for signal are $`d_s = \mathrm{tr} \ \mathbf{J}_{\hat{x}}`$. For a linear $`h`$ these hold exactly.

## 📥 Install

argminax is not on PyPI; the stack pins it by git tag. It brings in JAX, Equinox, einx, lineax, optimistix and gaussx, and needs Python 3.12 or newer.

```bash
pip install "argminax @ git+https://github.com/jejjohnson/argminax.git"
```

Two extras add the rest: `optax` for the transforms of `argminax.blr` and `matfree` for stochastic Lanczos quadrature of the log-determinant of a large operator. Each is imported only when a function needs it, which raises `MissingExtraError` naming the extra when it is missing.

```bash
pip install "argminax[optax] @ git+https://github.com/jejjohnson/argminax.git"
```

## 🛰️ Quickstart: a nonlinear retrieval

Three unknowns from five noisy measurements, with the forward model $`h(\mathbf{x}) = \mathbf{W} \tanh(\mathbf{x})`$ for a fixed $`\mathbf{W} \in \mathbb{R}^{D_y \times D_x}`$, the prior $`\mathcal{N}(\mathbf{0}, \sigma_x^2 \mathbf{I})`$ and the measurement error $`\boldsymbol{\Sigma}_\varepsilon = \sigma_\varepsilon^2 \mathbf{I}`$. The measurement is $`\mathbf{y}`$, the one parameter is $`\boldsymbol{\theta} = \log \sigma_x`$, the log of the prior's width, and the steps are the first four questions in order.

```python
import einx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float

from argminax.curvature import gauss_newton, marginal_variance
from argminax.diagnose import averaging_kernel, degrees_of_freedom
from argminax.solve import GaussNewtonCG


# The curvature and the diagnostics factor a Hessian, so work in double precision.
jax.config.update("jax_enable_x64", True)

# Shapes: D_x = 3 state elements; D_y = 5 measurements

# W, the weights of the forward model
weights = jnp.array(  # → (D_y, D_x)
    [
        [1.0, 0.5, 0.1],
        [0.2, 1.0, 0.4],
        [0.1, 0.3, 1.0],
        [0.6, 0.2, 0.2],
        [0.3, 0.7, 0.5],
    ]
)
# μ_x = 0 and σ_ε = 0.3
prior_mean = jnp.zeros(3)  # → (D_x,)
noise_std = 0.3


# user
# h(x) = W tanh(x)
def forward(x: Float[Array, " state"]) -> Float[Array, " obs"]:
    return einx.dot("obs state, state -> obs", weights, jnp.tanh(x))


# user
# r(x; y, θ) = ((y − h(x)) / σ_ε, (x − μ_x) / σ_x), so 𝒥(x; y, θ) = ½ ‖r(x; y, θ)‖²
def residuals(x, args):
    measurement, log_prior_std = args
    misfit = (measurement - forward(x)) / noise_std
    departure = (x - prior_mean) / jnp.exp(log_prior_std)
    return misfit, departure


# y = h(x_true) + 0.05
true_state = jnp.array([0.8, -0.4, 0.3])  # → (D_x,)
measurement = forward(true_state) + 0.05  # (D_x,) → (D_y,)
# θ = log σ_x, with σ_x = 1
log_prior_std = jnp.asarray(0.0)

# today (argminax)
solver = GaussNewtonCG(rtol=1e-8, atol=1e-8)


# user
# x̂(y, θ) = argmin_x 𝒥(x; y, θ)
def retrieve(measurement, log_prior_std):
    args = (measurement, log_prior_std)
    # today (optimistix)
    solution = optx.least_squares(residuals, solver, prior_mean, args=args)
    return solution.value


# 1. Where is the minimum?
x_hat = retrieve(measurement, log_prior_std)  # → (D_x,)

# 2. How sharp is it?
# today (argminax)
# 𝒩(x̂, Σ_x∣y), with Σ_x∣y⁻¹ = J_rᵀ J_r at x̂
posterior = gauss_newton(residuals, x_hat, (measurement, log_prior_std))
# today (argminax)
# √diag(Σ_x∣y)
posterior_std = jnp.sqrt(marginal_variance(posterior))  # → (D_x,)

# 3. How much came from the measurement?
# today (argminax)
# dₛ = tr J_x̂, with J_x̂ = I − Σ_x∣y Σ_x⁻¹ and Σ_x = I here
dofs = degrees_of_freedom(averaging_kernel(posterior)).value  # → ()

# 4. How does it move when y or θ moves?
# today (jax)
# Σ_x∣y⁻¹ [∂x̂/∂y, ∂x̂/∂θ] = −[∇²_xy 𝒥, ∇²_xθ 𝒥], one linear solve per direction
jacobians = jax.jacobian(retrieve, argnums=(0, 1))(measurement, log_prior_std)
gain, dx_dlog_prior_std = jacobians  # (D_x, D_y), (D_x,)

print(x_hat.round(3), posterior_std.round(3), dofs.round(2), dx_dlog_prior_std.round(3))
```

This prints `[0.667 -0.238 0.271] [0.427 0.344 0.34] 2.58 [0.247 -0.175 0.103]`: the retrieval, its posterior standard deviations, 2.58 of the 3 degrees of freedom coming from the measurement, and how the retrieval moves when the prior widens. `gain` is the $`D_x \times D_y`$ gain $`\mathbf{G}`$. Nothing here is specific to this model, so swapping `forward` for a radiative transfer code, a plume model or a PDE solve gives the same five quantities.

The [quickstart](https://jejjohnson.github.io/argminax/quickstart/) takes this retrieval further: the minimiser as a layer under an outer loss, a swath of pixels as one `vmap`, `KStep` for a solver stopped early, and the same Gaussian carried through training by an optax transform.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_roles/argminax_roles-dark.svg">
    <img alt="argminax on the GeoStack role ladder: algorithms (laplace_mode, the dual solves, the diagnostics, blr_posterior), the Laplace model, strategies for solving, stepping and differentiating, no primitives of its own (one step is gaussx's), and the Hessian as a gaussx structure, with optimistix as the floor its solvers and adjoints extend" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_roles/argminax_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does; the table lists them by module.

| Module | Symbols | Notes |
|---|---|---|
| `argminax.solve` | `GaussNewtonCG`, `dual_solve`, `incremental_dual_solve`, `Reparametrised`, `laplace_mode`, `laplace_mode_covariance`, `laplace_mode_sites`, `TolerantLinearSolver` | Gauss–Newton and Levenberg–Marquardt with a matrix-free inner CG, the observation-space (PSAS) increment and its outer loop, bounded states $`\mathbf{x} = g(\mathbf{u})`$ solved in an unconstrained control, and the damped Newton loop to the mode of GMRF, GP and Markov-GP models, and a lineax solver whose stopped or non-finite solves return values instead of raising |
| `argminax.curvature` | `Laplace`, `gauss_newton`, `marginal_variance`, `low_rank`, `push_forward`, `sample_by_perturbation` | The Gaussian $`\mathcal{N}(\hat{\mathbf{x}}, \boldsymbol{\Sigma}_{x \mid y})`$ held by its precision as a PSD-tagged operator, so $`\boldsymbol{\Sigma}_{x \mid y}`$ is never formed |
| `argminax.diagnose` | `averaging_kernel`, `degrees_of_freedom`, `information_content`, `evidence`, `optimize_prior_precision` | Exact for a small state, stochastic with a standard error for a large one, and always under `jit` and `vmap` |
| `argminax.adjoint` | `KStep`, `OneStep`, `safe_implicit_adjoint`, `to_optimistix_adjoint` | Optimistix adjoints for a minimiser stopped early, passed as `adjoint=`; `safe_implicit_adjoint` is the implicit adjoint whose linear solve never raises inside a sampler; `to_optimistix_adjoint` maps a pipekit-cycle adjoint spec onto one |
| `argminax.blr` | `blr_diagonal`, `ivon`, `blr_full_rank`, `blr_low_rank`, `newton`, `preconditioned_sgd`, `scale_by_preconditioner`, `blr_posterior` | The Bayesian learning rule as optax transforms; `blr_posterior` reads any of their states as a `Laplace`, so the diagnostics above apply during training too |

## 📖 Documentation

- **[Quickstart](https://jejjohnson.github.io/argminax/quickstart/)**: the five questions derived, the calling convention, double precision, the worked retrieval in full and the same Gaussian during training.
- **[The Bayesian learning rule](https://jejjohnson.github.io/argminax/blr/)**: the update every `argminax.blr` transform shares, what each one stores and costs, and what to watch numerically.
- **[BLR walkthrough](https://jejjohnson.github.io/argminax/blr-walkthrough/)**: an executed notebook that fits a line three ways and compares each with the exact posterior.
- **[API reference](https://jejjohnson.github.io/argminax/reference/)**: every public symbol, generated from the docstrings.
- **[Design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/argminax.md)**: the scope, the boundaries and the decisions, in the GeoStack docs.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_ecosystem/argminax_ecosystem-dark.svg">
    <img alt="The packages that import argminax: vardax, pyrox-lgm and pyrox-nn in GeoML, radtrax in GeoModels, and plumax through its assim extra, each with the argminax modules it uses; argminax builds on gaussx, optimistix, lineax, JAX and Equinox, and on optax through an extra" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_ecosystem/argminax_ecosystem-light.svg" width="100%">
  </picture>
</p>

argminax is Core tier 1 of the [GeoStack](https://github.com/jejjohnson/geostack). It owns what happens at and around a minimum, and composes the rest in:

| Concern | Owner |
|---|---|
| Minimisers, least squares and their adjoints (`argminax.solve` and `argminax.adjoint` extend them) | [`optimistix`](https://github.com/patrick-kidger/optimistix) |
| Gradient transformations for training (`argminax.blr` extends them) | [`optax`](https://github.com/google-deepmind/optax) |
| Linear solves (CG) and operators | [`lineax`](https://github.com/patrick-kidger/lineax) |
| Structured operators, Gauss–Newton operators, traces and log-determinants | [`gaussx`](https://github.com/jejjohnson/gaussx) |
| Stochastic Lanczos quadrature (the `matfree` extra) | [`matfree`](https://github.com/pnkraemer/matfree) |

Dependencies point one way: vardax, radtrax, pyrox-lgm and pyrox-nn import argminax today, and plumax through its `assim` extra, never the reverse. The rest of their switch-over to its solvers, curvature and diagnostics, pyrox-gp's included, is Wave 4 ([#12](https://github.com/jejjohnson/argminax/issues/12)).

## 🌳 Repository layout

```
src/argminax/
├── __init__.py        ← the version and the exceptions
├── solve.py           ← public modules, each re-exporting from _src/
├── curvature.py
├── diagnose.py
├── adjoint.py
├── blr.py
├── exceptions.py
└── _src/              ← solve/, curvature/, diagnose/, adjoint/, blr/
docs/                  ← MyST guides and notebooks, plus the Zensical API reference
tests/                 ← one folder per module, against closed forms, SciPy and finite differences
```

## ⚙️ Development

```bash
git clone https://github.com/jejjohnson/argminax.git
cd argminax
uv run poe install     # every dependency group + the pre-commit hooks
uv run poe check       # lint, format check, typecheck, tests with the 90 % coverage gate
uv run poe docs        # the MyST prose and the Zensical API reference, assembled into public/
```

`uv run poe` lists every task. [`AGENTS.md`](AGENTS.md) holds the rules for contributors and coding agents: the module map, the contracts, the tests that enforce them and the house style. [`docs/contributing.md`](docs/contributing.md) has the label taxonomy and the epic model.

## 🎓 References

- Blondel, Berthet, Cuturi, Frostig, Hoyer, Llinares-López, Pedregosa and Vert (2022). [Efficient and Modular Implicit Differentiation](https://arxiv.org/abs/2105.15183). *NeurIPS*.
- Bolte, Pauwels and Vaiter (2023). [One-step differentiation of iterative algorithms](https://arxiv.org/abs/2305.13768). *NeurIPS*.
- Khan and Rue (2023). [The Bayesian Learning Rule](https://arxiv.org/abs/2107.04562). *JMLR*.
- Mishkin, Kunstner, Nielsen, Schmidt and Khan (2018). [SLANG: Fast Structured Covariance Approximations for Bayesian Deep Learning with Natural Gradient](https://arxiv.org/abs/1811.04504). *NeurIPS*.
- Rodgers (2000). *Inverse Methods for Atmospheric Sounding: Theory and Practice*. World Scientific.
- Shen et al. (2024). [Variational Learning is Effective for Large Deep Networks](https://arxiv.org/abs/2402.17641). *ICML*.

## 📄 License

MIT. See [LICENSE](LICENSE).

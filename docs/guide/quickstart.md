# Quickstart

## Install

argminax is not on PyPI yet; install it from GitHub:

```bash
pip install "argminax @ git+https://github.com/jejjohnson/argminax.git"
```

That brings in JAX, Equinox, lineax, optimistix and gaussx. The `matfree` extra adds stochastic Lanczos quadrature for log-determinants of large operators; the functions that need it import it when they are called, and raise [`MissingExtraError`](xref:api#argminax.MissingExtraError) naming the extra when it is missing. The `optax` extra adds optax, for the transforms of `argminax.blr`, which carry the Gaussian during training and have [their own guide](blr.md).

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

## The problem

Everything in argminax is about one object, the minimiser of a cost as a function of what the cost was built from:

$$\hat{\vect{x}}(\vect{y}, \gvect{\theta}) = \mathrm{argmin}_{\vect{x}} \ \mathcal{J}(\vect{x}; \vect{y}, \gvect{\theta})$$

$\vect{x} \in \mathbb{R}^{D_x}$ is the state we solve for, $\vect{y} \in \mathbb{R}^{D_y}$ is the measurement, and $\gvect{\theta} \in \mathbb{R}^{D_\theta}$ holds the parameters: the prior, the covariances, the parameters of the forward model. Both went into the cost and stay fixed during the solve. $\mathcal{J}$ is a scalar cost, twice differentiable, which we read as a negative log posterior with a prior of covariance $\gmat{\Sigma}_x \in \mathbb{R}^{D_x \times D_x}$, so its Hessian at the minimum is the inverse of the Laplace covariance $\gmat{\Sigma}_{x \mid y} \in \mathbb{R}^{D_x \times D_x}$. A solver gives one value of $\hat{\vect{x}}$; argminax treats it as a function of $\vect{y}$ and $\gvect{\theta}$ and answers five questions about it. During training the state is instead described by a Gaussian $q = \mathcal{N}(\gvect{\mu}, \gmat{\Lambda}^{-1})$ over $\vect{x}$, with mean $\gvect{\mu} \in \mathbb{R}^{D_x}$ and positive-definite precision $\gmat{\Lambda} \in \mathbb{R}^{D_x \times D_x}$.

| Question | Equation | Module |
|---|---|---|
| Where is the minimum? | $\nabla_{\vect{x}} \mathcal{J}(\hat{\vect{x}}; \vect{y}, \gvect{\theta}) = \vect{0}$ | `argminax.solve`, or any optimistix solver |
| How sharp is it? | $\gmat{\Sigma}_{x \mid y}^{-1} = \nabla^2_{\vect{x}} \mathcal{J}(\hat{\vect{x}}; \vect{y}, \gvect{\theta})$ | `argminax.curvature` |
| How much came from the measurement? | $d_s = \mathrm{tr}(\mat{I} - \gmat{\Sigma}_{x \mid y} \gmat{\Sigma}_x^{-1})$ | `argminax.diagnose` |
| How does it move when $\vect{y}$ or $\gvect{\theta}$ moves? | $\gmat{\Sigma}_{x \mid y}^{-1} [\partial_{\vect{y}} \hat{\vect{x}}, \partial_{\gvect{\theta}} \hat{\vect{x}}] = -[\nabla^2_{\vect{x}\vect{y}} \mathcal{J}, \nabla^2_{\vect{x}\gvect{\theta}} \mathcal{J}]$ | `optx.ImplicitAdjoint`, `argminax.adjoint` |
| What Gaussian does training leave? | $\mathbb{E}_q[\nabla_{\vect{x}} \mathcal{J}] = \vect{0}, \ \gmat{\Lambda} = \mathbb{E}_q[\nabla^2_{\vect{x}} \mathcal{J}]$ | `argminax.blr` |

The third row counts the directions the measurement decided: $d_s$ is near 0 when the prior decides everything and approaches $D_x$ when the measurement does. The fifth row is the fixed point of the Bayesian learning rule (Khan and Rue, 2023), and with the gradient and the exact curvature evaluated at the mean it is the first two rows again, $\gvect{\mu} = \hat{\vect{x}}$ and $\gmat{\Lambda} = \gmat{\Sigma}_{x \mid y}^{-1}$ (see [During training](#during-training)).

```{figure} https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_hessian/argminax_hessian.png
:alt: The Hessian at the minimum in the middle; on the left, each Gauss–Newton step solves with it, the Laplace posterior factors it, and the Bayesian learning rule ends on it; on the right, the averaging kernel and degrees of freedom solve with it, the information content and evidence take its log-determinant, and the implicit adjoint solves with it once per direction
:width: 100%
:label: fig-hessian

One Hessian, every answer: what each module does with the curvature at the minimum. The schematic is kept in [geostack](https://github.com/jejjohnson/geostack/tree/main/docs/images/libraries/argminax_hessian).
```

### Implicit differentiation

$\hat{\vect{x}}(\vect{y}, \gvect{\theta})$ has no closed form in general, so it cannot be differentiated as written. There are two ways around that. One is to unroll: differentiate through every iteration of the solver, which costs memory in proportion to the number of iterations and ties the derivative to the solver. The other uses what the solver was looking for. At the minimum the gradient vanishes for every $\vect{y}$ and $\gvect{\theta}$,

$$\nabla_{\vect{x}} \mathcal{J}(\hat{\vect{x}}(\vect{y}, \gvect{\theta}); \vect{y}, \gvect{\theta}) = \vect{0}$$

and differentiating this identity with respect to $\gvect{\theta}$ by the chain rule gives

$$\nabla^2_{\vect{x}} \mathcal{J} \partial_{\gvect{\theta}} \hat{\vect{x}} + \nabla^2_{\vect{x}\gvect{\theta}} \mathcal{J} = \mat{0}$$

with both terms evaluated at $(\hat{\vect{x}}; \vect{y}, \gvect{\theta})$. The implicit function theorem says the Jacobian $\partial_{\gvect{\theta}} \hat{\vect{x}}$, of shape $(D_x, D_\theta)$, exists when the Hessian $\nabla^2_{\vect{x}} \mathcal{J} = \gmat{\Sigma}_{x \mid y}^{-1}$, of shape $(D_x, D_x)$, is invertible, and it is then the solution of a linear system:

$$\gmat{\Sigma}_{x \mid y}^{-1} \partial_{\gvect{\theta}} \hat{\vect{x}} = -\nabla^2_{\vect{x}\gvect{\theta}} \mathcal{J}$$

The right-hand side, of shape $(D_x, D_\theta)$, is the mixed second derivative of the cost. Nothing in this system mentions the solver or its iterations, so the derivative of a converged solve costs one more linear solve in each direction (Blondel et al., 2022). In JAX it is what `jax.grad`, `jax.jvp` and `jax.jacobian` do through an optimistix solve, whose default adjoint is `optx.ImplicitAdjoint`. Differentiating with respect to $\vect{y}$ instead puts $\nabla^2_{\vect{x}\vect{y}} \mathcal{J}$ on the right-hand side, and the solution is the gain $\mat{G} = \partial_{\vect{y}} \hat{\vect{x}} \in \mathbb{R}^{D_x \times D_y}$.

Three things follow from it.

- **The same matrix gives the uncertainty.** $\gmat{\Sigma}_{x \mid y}$ is the covariance of the Laplace approximation $\mathcal{N}(\hat{\vect{x}}, \gmat{\Sigma}_{x \mid y})$, and $\partial_{\gvect{\theta}} \hat{\vect{x}} = -\gmat{\Sigma}_{x \mid y} \nabla^2_{\vect{x}\gvect{\theta}} \mathcal{J}$. How sharp the minimum is and how it moves are two readings of one operator, which is why argminax keeps the solvers, the curvature and the adjoints in one package.
- **A minimiser is a layer.** An outer loss $\mathcal{L}(\hat{\vect{x}}(\vect{y}, \gvect{\theta}))$ can be minimised over $\gvect{\theta}$ by gradient descent, with $\mathrm{d}\mathcal{L} / \mathrm{d}\gvect{\theta} = (\partial_{\gvect{\theta}} \hat{\vect{x}})^\top \nabla_{\vect{x}} \mathcal{L}$, which is one solve with the Hessian in reverse mode. This is bilevel optimisation: hyperparameter learning, a retrieval inside a neural network, learned 4D-Var.
- **The derivative is a result in its own right.** For the retrieval cost below with a linear forward model $h(\vect{x}) = \mat{H} \vect{x}$, the gain is $\mat{G} = \gmat{\Sigma}_{x \mid y} \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1}$, the gain matrix of retrieval theory, and the averaging kernel is $\mat{J}_{\hat{x}} = \mat{G} \mat{H} \in \mathbb{R}^{D_x \times D_x}$ (Rodgers, 2000, where it is $\mat{A}$).

```{figure} https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_adjoints/argminax_adjoints.png
:alt: Three ways to differentiate a solve: unrolling differentiates every iteration, at a cost that grows with their number; the implicit adjoint skips the iterations and solves one linear system with the Hessian at the minimum, which needs a converged solve; KStep and OneStep stop the gradient before the last k updates, for a solver stopped early
:width: 100%
:label: fig-adjoints

Three ways to differentiate a solve: which iterations the gradient flows through, what each costs, and when each is right. The schematic is kept in [geostack](https://github.com/jejjohnson/geostack/tree/main/docs/images/libraries/argminax_adjoints).
```

### Where it stops holding

The identity needs a converged solve. Two cases are outside it, and each has its own module.

- A solver stopped after a fixed number of iterations is not at a stationary point, so there is no identity to differentiate. [`KStep`](xref:api#argminax.adjoint.KStep) and [`OneStep`](xref:api#argminax.adjoint.OneStep) differentiate through its last few iterations.
- A model in training never reaches the minimum. The transforms of `argminax.blr` carry the Gaussian $\mathcal{N}(\hat{\vect{x}}, \gmat{\Sigma}_{x \mid y})$ along as they step (see [During training](#during-training) below, and [their own guide](blr.md)). The variable they optimise is a model's parameter vector, so it is written $\gvect{\theta}$ there: it plays the part of $\vect{x}$ here.

### The retrieval cost

The cost argminax minimises most often is a measurement misfit plus a prior. For a measurement $\vect{y} \in \mathbb{R}^{D_y}$, a forward model $h: \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$, a prior $\mathcal{N}(\gvect{\mu}_x, \gmat{\Sigma}_x)$ and a measurement-error covariance $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$,

$$\mathcal{J}(\vect{x}; \vect{y}, \gvect{\theta}) = \frac{1}{2} \Vert \vect{y} - h(\vect{x}) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}} + \frac{1}{2} \Vert \vect{x} - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}}$$

with $\gvect{\theta} = (\gvect{\mu}_x, \gmat{\Sigma}_\varepsilon, \gmat{\Sigma}_x)$ and whatever parameters $h$ has. The curvature is taken in its Gauss–Newton form, $\gmat{\Sigma}_{x \mid y}^{-1} = \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H} + \gmat{\Sigma}_x^{-1}$ with $\mat{H} = \mat{J}_h(\hat{\vect{x}}) \in \mathbb{R}^{D_y \times D_x}$ the Jacobian of $h$ at the minimum, which is positive definite whatever the residual, and $\gmat{\Sigma}_{x \mid y}$ is the posterior covariance.

## The calling convention

argminax follows optimistix: a cost is a function `fn(x, args)` of the state and a pytree of fixed arguments, so a function written for `optx.minimise` or `optx.least_squares` already works with argminax. In the notation above the first argument is $\vect{x}$ and `args` holds the measurement and the parameters, $(\vect{y}, \gvect{\theta})$, and anything in `args` can be differentiated. The two shapes are named in `argminax._typing`:

| Alias | Signature | Used by |
|---|---|---|
| `Objective` | `fn(x, args) -> scalar`, the cost $\mathcal{J}$ | `optx.minimise` |
| `Residuals` | `fn(x, args) -> pytree`, with $\mathcal{J} = \frac{1}{2} \Vert \vect{r} \Vert^2$ | `optx.least_squares` |

For the retrieval cost, the residuals are whitened by the two covariances, $\vect{r}(\vect{x}; \vect{y}, \gvect{\theta}) = (\gmat{\Sigma}_\varepsilon^{-1/2} (\vect{y} - h(\vect{x})), \gmat{\Sigma}_x^{-1/2} (\vect{x} - \gvect{\mu}_x))$, which is what the Gauss–Newton solvers and the curvature at the minimum read.

## A first retrieval

Three unknowns from five noisy measurements, with the forward model $h(\vect{x}) = \mat{W} \tanh(\vect{x})$ for a fixed $\mat{W} \in \mathbb{R}^{D_y \times D_x}$, the prior $\mathcal{N}(\vect{0}, \sigma_x^2 \mat{I})$ and the measurement error $\gmat{\Sigma}_\varepsilon = \sigma_\varepsilon^2 \mat{I}$. The measurement is $\vect{y}$ and the one parameter is $\gvect{\theta} = \log \sigma_x$, the log of the prior's width. The steps are the first four questions in order (solve for $\hat{\vect{x}}(\vect{y}, \gvect{\theta})$, read the uncertainty off the curvature, count how much came from the measurement, and differentiate $\hat{\vect{x}}$ with respect to $\vect{y}$ and $\gvect{\theta}$), and then two things the derivative is for.

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

# Shapes: D_x = 3 state elements; D_y = 5 measurements; N_pixel = 3 pixels

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


# --- 1. Where is the minimum? ---
x_hat = retrieve(measurement, log_prior_std)  # → (D_x,)

# --- 2. How sharp is it? ---
# today (argminax)
# 𝒩(x̂, Σ_x∣y), with Σ_x∣y⁻¹ = J_rᵀ J_r at x̂
posterior = gauss_newton(residuals, x_hat, (measurement, log_prior_std))
# today (argminax)
# √diag(Σ_x∣y)
posterior_std = jnp.sqrt(marginal_variance(posterior))  # → (D_x,)

# --- 3. How much came from the measurement? ---
# today (argminax)
# dₛ = tr J_x̂, with J_x̂ = I − Σ_x∣y Σ_x⁻¹ and Σ_x = I here
dofs = degrees_of_freedom(averaging_kernel(posterior)).value  # → ()

# --- 4. How does it move when y or θ moves? ---
# today (jax)
# Σ_x∣y⁻¹ [∂x̂/∂y, ∂x̂/∂θ] = −[∇²_xy 𝒥, ∇²_xθ 𝒥], one linear solve per direction
jacobians = jax.jacobian(retrieve, argnums=(0, 1))(measurement, log_prior_std)
gain, dx_dlog_prior_std = jacobians  # (D_x, D_y), (D_x,)


# user
# ℒ(θ) = ‖x̂(y, θ) − x_true‖², an outer loss on the retrieval
def outer_loss(log_prior_std):
    error = retrieve(measurement, log_prior_std) - true_state
    return einx.sum("state ->", error**2)


# --- the minimiser as a layer ---
# today (jax)
# dℒ/dθ = (∂x̂/∂θ)ᵀ ∇_x ℒ, one solve with the Hessian in reverse mode
dloss_dlog_prior_std = jax.grad(outer_loss)(log_prior_std)  # → ()


# --- a swath of pixels ---
# y⁽ᵖ⁾ for p = 1, 2, 3
swath = jnp.stack(  # → (N_pixel, D_y)
    [measurement, measurement + 0.1, measurement - 0.1]
)
# today (jax)
# x̂(y⁽ᵖ⁾, θ), one retrieval per pixel with θ shared: (N_pixel, D_y) → (N_pixel, D_x)
x_hats = jax.jit(jax.vmap(retrieve, in_axes=(0, None)))(swath, log_prior_std)

print(
    x_hat.round(3), posterior_std.round(3), dofs.round(2), dloss_dlog_prior_std.round(3)
)
```

This prints `[0.667 -0.238 0.271] [0.427 0.344 0.34] 2.58 -0.128`: the retrieval, its posterior standard deviations, 2.58 of the 3 degrees of freedom coming from the measurement, and the gradient of the outer loss with respect to the prior's width.

- `GaussNewtonCG` is an optimistix least-squares solver, so it is called through `optx.least_squares`, which returns a solution whose `result` reports whether the solve converged (`retrieve` keeps only its `value`). Check it before trusting a derivative: the identity above holds at a minimum only.
- `argminax.curvature.gauss_newton` builds the Laplace posterior $\mathcal{N}(\hat{\vect{x}}, \gmat{\Sigma}_{x \mid y})$ from the same residuals. It holds the precision as a gaussx operator, so the covariance is never formed. The prior here is $\mathcal{N}(\vect{0}, \mat{I})$, so the posterior is already in whitened coordinates and `averaging_kernel` needs no prior covariance; for any other prior, pass $\gmat{\Sigma}_x$ as `prior_cov`.
- `degrees_of_freedom` is exact without a `key` and a Hutchinson estimate with one, which returns its standard error, so a million pixels are one `vmap` over split keys.
- `jax.jacobian(retrieve, argnums=(0, 1))` and `jax.grad(outer_loss)` use optimistix's `ImplicitAdjoint`: one more linear solve in each direction, not a pass through every iteration. `gain` is the $D_x \times D_y$ gain $\mat{G}$, how $\hat{\vect{x}}$ answers each measurement, and `dx_dlog_prior_std` is `[0.247, -0.175, 0.103]`, how it answers a wider prior. The Jacobian agrees with $-\gmat{\Sigma}_{x \mid y} \nabla^2_{\vect{x}\gvect{\theta}} \mathcal{J}$, computed by hand with the exact Hessian, to $10^{-16}$.
- The outer gradient is negative, so a wider prior would bring this retrieval closer to the reference. Hand it to an optax optimiser and that is hyperparameter learning.
- For a solver stopped early, pass `adjoint=KStep(3)` to `optx.least_squares` to differentiate its last three iterations only. On this solve it gives `[0.248, -0.175, 0.103]` for the derivative with respect to the prior's width, against `[0.247, -0.175, 0.103]` from the implicit adjoint, and `OneStep()` gives `[0.283, -0.192, 0.108]`.

Nothing here is specific to this model. Swap `forward` for a radiative transfer code, a plume model or a PDE solve, and the same lines give the retrieval, its covariance, its degrees of freedom and its derivatives.

(during-training)=
## During training

The solvers above run to a minimum first. A model in training never gets there, so the transforms of `argminax.blr` carry the same Gaussian while they train. Each one is an optax `GradientTransformation` (the Bayesian learning rule of Khan and Rue, 2023), so it drops into an optax loop, and its state reads out as the `Laplace` object used above. The variable they optimise is the model's parameter vector $\gvect{\theta} \in \mathbb{R}^{D_\theta}$, which plays the part of $\vect{x}$ above, and the Gaussian they carry is $\mathcal{N}(\gvect{\mu}_\theta, \gmat{\Sigma}_\theta)$.

Here a line is fitted to $N$ noisy points $(u_n, y_n)$ with features $\gmat{\Phi} = [\vect{u}, \vect{1}] \in \mathbb{R}^{N \times D_\theta}$, noise $\sigma_\varepsilon = 0.5$ and the cost $\mathcal{J}(\gvect{\theta}) = \Vert \vect{y} - \gmat{\Phi} \gvect{\theta} \Vert^2 / (2 \sigma_\varepsilon^2)$. The cost is quadratic, so one full-rank step with the exact curvature $\mat{C} = \gmat{\Phi}^\top \gmat{\Phi} / \sigma_\varepsilon^2$ lands on the exact posterior.

```python
import einx
import jax
import jax.numpy as jnp
import lineax as lx

from argminax.blr import blr_full_rank, blr_posterior
from argminax.curvature import marginal_variance
from argminax.diagnose import averaging_kernel, degrees_of_freedom


jax.config.update("jax_enable_x64", True)

# Shapes: N = 200 samples; D_θ = 2 parameters (slope, intercept)

key_inputs, key_noise = jax.random.split(jax.random.key(0))
# uₙ ~ 𝒰(−2, 2)
inputs = jax.random.uniform(key_inputs, (200,), minval=-2.0, maxval=2.0)  # → (N,)
# yₙ = 2 uₙ − 1 + σ_ε εₙ, with σ_ε = 0.5 and εₙ ~ 𝒩(0, 1)
noise = jax.random.normal(key_noise, (200,))  # → (N,)
targets = 2.0 * inputs - 1.0 + 0.5 * noise  # (N,) → (N,)
# Φ = [u, 1]
features = einx.id(  # (N,) → (N, D_θ)
    "n, n -> n (1 + 1)", inputs, jnp.ones(200)
)


# user
# 𝒥(θ) = ‖y − Φθ‖² / (2σ_ε²)
def cost(theta):
    prediction = einx.dot("n p, p -> n", features, theta)
    return 0.5 * einx.sum("n ->", (targets - prediction) ** 2) / 0.5**2


# C = ΦᵀΦ / σ_ε², the exact curvature of 𝒥
curvature = einx.dot("n p, n q -> p q", features, features) / 0.5**2  # → (D_θ, D_θ)

# today (argminax)
# one step of size α = 1 with the exact curvature and the prior 𝒩(0, I)
optimiser = blr_full_rank(
    1.0, prior_precision=1.0, curvature=lambda mean, grad: curvature, damping=0.0
)
theta = jnp.zeros(2)  # → (D_θ,)
# today (optax)
state = optimiser.init(theta)
# today (optax)
updates, state = optimiser.update(jax.grad(cost)(theta), state)

# today (argminax)
# 𝒩(μ_θ, Σ_θ), read as an argminax.curvature.Laplace
posterior = blr_posterior(state)
# today (argminax)
# √diag(Σ_θ)
posterior_std = jnp.sqrt(marginal_variance(posterior))  # → (D_θ,)

# the prior covariance I
prior_cov = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
# today (argminax)
# dₛ = tr J_θ̂, how much of the fit came from the data
dofs = degrees_of_freedom(averaging_kernel(posterior, prior_cov)).value  # → ()

print(posterior.mode.round(3), posterior_std.round(3), dofs.round(3))
```

This prints `[2.016 -0.989] [0.031 0.035] 1.998`: the slope and intercept, their posterior standard deviations, and nearly both degrees of freedom coming from the data.

The posterior is a `Laplace`, so the diagnostics of the retrieval above apply to a model in training unchanged. `blr_diagonal` and `ivon` work on a parameter pytree of any size as a drop-in for `optax.adam`, and the [guide](blr.md) derives the rule and says what the variance of each curvature estimate means.

## References

- Blondel, Berthet, Cuturi, Frostig, Hoyer, Llinares-López, Pedregosa and Vert (2022), "Efficient and Modular Implicit Differentiation", arXiv:2105.15183.
- Khan and Rue (2023), "The Bayesian Learning Rule", arXiv:2107.04562.
- Rodgers (2000), "Inverse Methods for Atmospheric Sounding: Theory and Practice", World Scientific.

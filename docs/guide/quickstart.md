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

$$\hat{\vect{x}}(\gvect{\theta}) = \arg\min_{\vect{x}} J(\vect{x}, \gvect{\theta})$$

$\vect{x} \in \mathbb{R}^{D_x}$ is the state we solve for, and $\gvect{\theta} \in \mathbb{R}^{D_\theta}$ is everything that went into the cost and stays fixed during the solve: the measurement, the prior, the covariances, the parameters of the forward model. $J$ is a scalar cost, twice differentiable. A solver gives one value of $\hat{\vect{x}}$; argminax treats it as a function of $\gvect{\theta}$ and answers three questions about it.

| Question | Equation | Module |
|---|---|---|
| Where is the minimum? | $\nabla_{\vect{x}} J(\hat{\vect{x}}, \gvect{\theta}) = \vect{0}$ | `argminax.solve`, or any optimistix solver |
| How sharp is it? | $\hat{\mat{H}} = \nabla^2_{\vect{x}} J(\hat{\vect{x}}, \gvect{\theta})$ and $\hat{\mat{S}} = \hat{\mat{H}}^{-1}$ | `argminax.curvature`, `argminax.diagnose` |
| How does it move when $\gvect{\theta}$ moves? | $\hat{\mat{H}} \partial_{\gvect{\theta}} \hat{\vect{x}} = -\partial_{\gvect{\theta}} \nabla_{\vect{x}} J(\hat{\vect{x}}, \gvect{\theta})$ | `optx.ImplicitAdjoint`, `argminax.adjoint` |

### Implicit differentiation

$\hat{\vect{x}}(\gvect{\theta})$ has no closed form in general, so it cannot be differentiated as written. There are two ways around that. One is to unroll: differentiate through every iteration of the solver, which costs memory in proportion to the number of iterations and ties the derivative to the solver. The other uses what the solver was looking for. At the minimum the gradient vanishes for every $\gvect{\theta}$,

$$\nabla_{\vect{x}} J(\hat{\vect{x}}(\gvect{\theta}), \gvect{\theta}) = \vect{0}$$

and differentiating this identity with respect to $\gvect{\theta}$ by the chain rule gives

$$\nabla^2_{\vect{x}} J \partial_{\gvect{\theta}} \hat{\vect{x}} + \partial_{\gvect{\theta}} \nabla_{\vect{x}} J = \mat{0}$$

with both terms evaluated at $(\hat{\vect{x}}, \gvect{\theta})$. The implicit function theorem says the Jacobian $\partial_{\gvect{\theta}} \hat{\vect{x}}$, of shape $(D_x, D_\theta)$, exists when the Hessian $\hat{\mat{H}} = \nabla^2_{\vect{x}} J$, of shape $(D_x, D_x)$, is invertible, and it is then the solution of a linear system:

$$\hat{\mat{H}} \partial_{\gvect{\theta}} \hat{\vect{x}} = -\partial_{\gvect{\theta}} \nabla_{\vect{x}} J$$

The right-hand side, of shape $(D_x, D_\theta)$, is the mixed second derivative of the cost. Nothing in this system mentions the solver or its iterations, so the derivative of a converged solve costs one more linear solve in each direction (Blondel et al., 2022). In JAX it is what `jax.grad`, `jax.jvp` and `jax.jacobian` do through an optimistix solve, whose default adjoint is `optx.ImplicitAdjoint`.

Three things follow from it.

- **The same matrix gives the uncertainty.** When $J$ is a negative log posterior, $\hat{\mat{S}} = \hat{\mat{H}}^{-1}$ is the covariance of the Laplace approximation $\mathcal{N}(\hat{\vect{x}}, \hat{\mat{S}})$. How sharp the minimum is and how it moves are two readings of one operator, which is why argminax keeps the solvers, the curvature and the adjoints in one package.
- **A minimiser is a layer.** An outer cost $L(\hat{\vect{x}}(\gvect{\theta}))$ can be minimised over $\gvect{\theta}$ by gradient descent, with $\nabla_{\gvect{\theta}} L = (\partial_{\gvect{\theta}} \hat{\vect{x}})^\top \nabla_{\vect{x}} L$, which is one solve with $\hat{\mat{H}}$ in reverse mode. This is bilevel optimisation: hyperparameter learning, a retrieval inside a neural network, learned 4D-Var.
- **The derivative is a result in its own right.** For the retrieval cost below with a linear forward model $\mat{K}$, $\partial_{\vect{y}} \hat{\vect{x}} = \hat{\mat{S}} \mat{K}^\top \mat{S}_\varepsilon^{-1}$ is the gain matrix of retrieval theory, and the averaging kernel is $\mat{A} = (\partial_{\vect{y}} \hat{\vect{x}}) \mat{K}$ (Rodgers, 2000).

### Where it stops holding

The identity needs a converged solve. Two cases are outside it, and each has its own module.

- A solver stopped after a fixed number of iterations is not at a stationary point, so there is no identity to differentiate. [`KStep`](xref:api#argminax.adjoint.KStep) and [`OneStep`](xref:api#argminax.adjoint.OneStep) differentiate through its last few iterations.
- A model in training never reaches the minimum. The transforms of `argminax.blr` carry the Gaussian $\mathcal{N}(\hat{\vect{x}}, \hat{\mat{S}})$ along as they step, and have [their own guide](blr.md). The variable they optimise is a model's parameter vector, so that guide writes it $\gvect{\theta}$: it plays the part of $\vect{x}$ here.

### The retrieval cost

The cost argminax minimises most often is a measurement misfit plus a prior. For a measurement $\vect{y} \in \mathbb{R}^{D_y}$, forward model $F$, prior state $\vect{x}_a$ and covariances $\mat{S}_\varepsilon$ and $\mat{S}_a$,

$$J(\vect{x}, \gvect{\theta}) = \frac{1}{2} (\vect{y} - F(\vect{x}))^\top \mat{S}_\varepsilon^{-1} (\vect{y} - F(\vect{x})) + \frac{1}{2} (\vect{x} - \vect{x}_a)^\top \mat{S}_a^{-1} (\vect{x} - \vect{x}_a)$$

with $\gvect{\theta} = (\vect{y}, \vect{x}_a, \mat{S}_\varepsilon, \mat{S}_a)$ and whatever parameters $F$ has. The curvature is taken in its Gauss–Newton form, $\hat{\mat{H}} = \mat{K}^\top \mat{S}_\varepsilon^{-1} \mat{K} + \mat{S}_a^{-1}$ with $\mat{K} = \partial_{\vect{x}} F(\hat{\vect{x}})$, which is positive definite whatever the residual, and $\hat{\mat{S}} = \hat{\mat{H}}^{-1}$ is the posterior covariance.

## The calling convention

argminax follows optimistix: a cost is a function `fn(y, args)` of the state and a pytree of fixed arguments, so a function written for `optx.minimise` or `optx.least_squares` already works with argminax. In the notation above the first argument is $\vect{x}$ and `args` is $\gvect{\theta}$, and anything in `args` can be differentiated. The two shapes are named in `argminax._typing`:

| Alias | Signature | Used by |
|---|---|---|
| `Objective` | `fn(x, args) -> scalar`, the cost $J$ | `optx.minimise` |
| `Residuals` | `fn(x, args) -> pytree`, with $J = \frac{1}{2} \lVert \vect{r} \rVert^2$ | `optx.least_squares` |

For the retrieval cost, the residuals are whitened by the two covariances, $\vect{r}(\vect{x}) = (\mat{S}_\varepsilon^{-1/2} (\vect{y} - F(\vect{x})), \mat{S}_a^{-1/2} (\vect{x} - \vect{x}_a))$, which is what the Gauss–Newton solvers and the curvature at the minimum read.

## A first retrieval

Three unknowns, five noisy measurements and a nonlinear forward model $F(\vect{x}) = \mat{K} \tanh(\vect{x})$, with $\gvect{\theta}$ the measurement and the log of the prior's width. The steps are the three questions in order: solve for $\hat{\vect{x}}(\gvect{\theta})$, read the uncertainty off the curvature, and differentiate $\hat{\vect{x}}$ with respect to $\gvect{\theta}$.

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

- `GaussNewtonCG` is an optimistix least-squares solver, so it is called through `optx.least_squares`, which returns a solution whose `result` reports whether the solve converged (`retrieve` keeps only its `value`). Check it before trusting a derivative: the identity above holds at a minimum only.
- `argminax.curvature.gauss_newton` builds the Laplace posterior $\mathcal{N}(\hat{\vect{x}}, \hat{\mat{S}})$ from the same residuals. It is a gaussx operator, so `covariance().as_matrix()` is only for small problems like this one.
- `degrees_of_freedom` is exact without a `key` and a Hutchinson estimate with one, which returns its standard error, so a million pixels are one `vmap` over split keys.
- `jax.jacobian(retrieve)` and `jax.grad(outer_cost)` use optimistix's `ImplicitAdjoint`: one more linear solve in each direction, not a pass through every iteration. Here the Jacobian agrees with $-\hat{\mat{H}}^{-1} \partial_{\gvect{\theta}} \nabla_{\vect{x}} J$, computed by hand with the exact Hessian, to $10^{-16}$.
- The outer gradient is negative, so a wider prior would bring this retrieval closer to the reference. Hand it to an optax optimiser and that is hyperparameter learning.
- For a solver stopped early, pass `adjoint=KStep(3)` to `optx.least_squares` to differentiate its last three iterations only. On this solve it gives `[0.248, -0.175, 0.103]` for the derivative with respect to the prior's width, against `[0.247, -0.175, 0.103]` from the implicit adjoint, and `OneStep()` gives `[0.283, -0.192, 0.108]`.

## References

- Blondel, Berthet, Cuturi, Frostig, Hoyer, Llinares-López, Pedregosa and Vert (2022), "Efficient and Modular Implicit Differentiation", arXiv:2105.15183.
- Rodgers (2000), "Inverse Methods for Atmospheric Sounding: Theory and Practice", World Scientific.

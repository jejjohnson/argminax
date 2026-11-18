# The Bayesian learning rule

The solvers in argminax run to a minimum and then read the Gaussian there. The transforms in `argminax.blr` carry that Gaussian while they train: each one is an optax `GradientTransformation` whose state holds $q(\gvect{\theta}) = \mathcal{N}(\gvect{\mu}_\theta, \gmat{\Lambda}_\theta^{-1})$ over the parameters, and each step moves it by one minibatch. This page derives the update they share, lists what each transform stores and costs, and says what to watch for numerically. The main reference is Khan and Rue (2023).

They need optax, which comes with the `optax` extra:

```bash
pip install "argminax[optax] @ git+https://github.com/jejjohnson/argminax.git"
```

```{figure} https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_blr/argminax_blr.png
:alt: argminax.blr: a plain optax loop passes the gradient of the cost to a Bayesian-learning-rule transform for the diagonal, low-rank or full-rank family; the optimiser state holds the precision and the natural mean, anchored by the prior; blr_posterior reads the Gaussian out as a Laplace for sampling
:width: 100%
:label: fig-blr-overview

Train with optax, read off a posterior. The schematic is kept in [geostack](https://github.com/jejjohnson/geostack/tree/main/docs/images/libraries/argminax_blr) with the stack's other library schematics.
```

## The rule

### Notation

| Symbol | Shape | Meaning |
|---|---|---|
| $\gvect{\theta}$ | $(D_\theta,)$ | the parameters being trained |
| $\mathcal{J}(\gvect{\theta})$ | scalar | the cost of one minibatch, a negative log-likelihood |
| $\vect{g}$ | $(D_\theta,)$ | its gradient $\nabla_\theta \mathcal{J}$, what any optax optimiser takes |
| $\mat{C}$ | $(D_\theta, D_\theta)$ | an estimate of its curvature $\nabla^2_\theta \mathcal{J}$, positive semidefinite |
| $\gvect{\mu}_\theta$, $\gmat{\Lambda}_\theta$ | $(D_\theta,)$, $(D_\theta, D_\theta)$ | the mean and the precision of $q$ |
| $\gvect{\mu}_0$, $\delta$ | $(D_\theta,)$, scalar | the prior $\mathcal{N}(\gvect{\mu}_0, \delta^{-1} \mat{I})$ |
| $\alpha$ | scalar | the step size, in $(0, 1]$ |
| $\beta$ | scalar | the damping, $\beta \ge 0$ |

:::{note}
Every transform takes the gradient of the **cost**, and a curvature is the curvature of the cost. Khan and Rue, and optax_bayes before it moved here, write the rule for the log-likelihood $\ell = -\mathcal{J}$: their gradient is $-\vect{g}$ and their Hessian is $-\mat{C}$.
:::

### From the variational objective to the update

Keep a distribution $q(\gvect{\theta})$ from an exponential family, with natural parameters $\gvect{\lambda}$ and expectation parameters $\vect{m} = \mathbb{E}_q[\vect{T}(\gvect{\theta})]$ ($\vect{T}$ the sufficient statistics), and minimise the expected cost plus the divergence from the prior $p$:

$$\mathcal{L}(\gvect{\lambda}) = \mathbb{E}_q[\mathcal{J}(\gvect{\theta})] + \mathrm{KL}(q \Vert p)$$

In a minimal exponential family the natural gradient with respect to $\gvect{\lambda}$ is the plain gradient with respect to $\vect{m}$. Natural-gradient descent on $\mathcal{L}$, with a prior in the same family (natural parameters $\gvect{\lambda}_0$), is the Bayesian learning rule:

$$\gvect{\lambda} \leftarrow (1 - \alpha) \gvect{\lambda} + \alpha (\gvect{\lambda}_0 - \nabla_{\vect{m}} \mathbb{E}_q[\mathcal{J}])$$

Each step is a convex combination of the current natural parameters and a target.

For a Gaussian, the natural parameters are $\gmat{\Lambda}_\theta \gvect{\mu}_\theta$ and $-\frac{1}{2} \gmat{\Lambda}_\theta$, and the expectation parameters are $\gvect{\mu}_\theta$ and $\gvect{\mu}_\theta \gvect{\mu}_\theta^\top + \gmat{\Lambda}_\theta^{-1}$. The identities of Bonnet and Price (Opper and Archambeau, 2009) turn the two gradients into an expected gradient and an expected Hessian:

$$\nabla_{\vect{m}_1} \mathbb{E}_q[\mathcal{J}] = \vect{g} - \mat{C} \gvect{\mu}_\theta$$

$$\nabla_{\vect{m}_2} \mathbb{E}_q[\mathcal{J}] = \frac{1}{2} \mat{C}$$

with $\vect{g} = \mathbb{E}_q[\nabla_\theta \mathcal{J}]$ and $\mat{C} = \mathbb{E}_q[\nabla^2_\theta \mathcal{J}]$. The term $\mat{C} \gvect{\mu}_\theta$ comes from the dependence of the covariance on the first expectation parameter; leaving it out gives the wrong fixed point. Substituting, the rule becomes the update that every transform here builds on:

$$\gmat{\Lambda}_\theta \leftarrow (1 - \alpha) \gmat{\Lambda}_\theta + \alpha (\delta \mat{I} + \mat{C})$$

$$\gmat{\Lambda}_\theta \gvect{\mu}_\theta \leftarrow (1 - \alpha) \gmat{\Lambda}_\theta \gvect{\mu}_\theta + \alpha (\delta \gvect{\mu}_0 - \vect{g} + \mat{C} \gvect{\mu}_\theta)$$

The new mean is the new natural mean divided by the new precision. Every `update` returns the change of the mean, so after `optax.apply_updates` the parameters equal $\gvect{\mu}_\theta$, and `init` starts the mean at the parameters it is given.

### The mean form

Substituting the precision update into the natural-mean update gives the same step written on the mean:

$$\gvect{\mu}_\theta \leftarrow \gvect{\mu}_\theta - \alpha \gmat{\Lambda}_\theta^{-1} (\vect{g} + \delta (\gvect{\mu}_\theta - \gvect{\mu}_0))$$

with $\gmat{\Lambda}_\theta$ the new precision: a step down the gradient of the cost plus the prior, preconditioned by the precision. The two forms agree when the stored precision is the one the rule prescribes and there is no damping. The low-rank transform stores an approximation of it, and uses the mean form because it stays a well-defined preconditioned step there.

### What the fixed point is

The transforms use whatever gradient and curvature they are given. Evaluating them at the mean replaces the expectations under $q$ by their value there, which is the usual way to run an optimiser. The fixed point with the exact curvature is then the Laplace approximation: $\vect{g} + \delta (\gvect{\mu}_\theta - \gvect{\mu}_0) = \vect{0}$ at the mean, and $\gmat{\Lambda}_\theta = \delta \mat{I} + \nabla^2_\theta \mathcal{J}$ there. For a cost that is quadratic in $\gvect{\theta}$ this is the exact posterior, and one step with $\alpha = 1$ reaches it from any start.

## The transforms

| Transform | Precision | Storage | Cost of a step | Parameters |
|---|---|---|---|---|
| [`blr_diagonal`](xref:api#argminax.blr.blr_diagonal) | diagonal | $O(D_\theta)$ | $O(D_\theta)$ | any pytree |
| [`blr_full_rank`](xref:api#argminax.blr.blr_full_rank) | dense | $O(D_\theta^2)$ | $O(D_\theta^3)$, two solves | a flat vector |
| [`blr_low_rank`](xref:api#argminax.blr.blr_low_rank) | diagonal plus a rank-$R$ factor | $O(D_\theta R)$ | $O(D_\theta R^2)$ with a built-in curvature, $O(D_\theta^3)$ with a callable | a flat vector |
| [`ivon`](xref:api#argminax.blr.ivon) | diagonal | $O(D_\theta)$ | $O(D_\theta)$ | any pytree |
| [`newton`](xref:api#argminax.blr.newton) | dense | $O(D_\theta^2)$ | $O(D_\theta^3)$ | a flat vector |
| [`preconditioned_sgd`](xref:api#argminax.blr.preconditioned_sgd) | any operator, given and fixed | that of the operator | one solve with it | a flat vector |

The low-rank cost of $O(D_\theta R^2)$ holds for `"ggn"` and `"identity"`, whose factor has one column or none. A callable returns a dense $(D_\theta, D_\theta)$ matrix, which the transform factors by an eigendecomposition: that step costs $O(D_\theta^3)$ and $O(D_\theta^2)$ of temporary storage whatever the rank, so use a callable there only when $D_\theta$ is small.

The full-rank and low-rank transforms need one flat vector; flatten a parameter pytree with `jax.flatten_util.ravel_pytree`. A step size can be a float or an optax schedule of the step count.

**Diagonal.** The update above, elementwise, with $\mat{C}$ a vector: the squared gradient $\vect{g} \odot \vect{g}$ (`curvature="ggn"`) or zero (`"identity"`). The damping is a floor, $\gmat{\Lambda}_\theta \leftarrow \max(\gmat{\Lambda}_\theta, \beta)$.

**Full rank.** The update above with a dense precision, plus $\beta \mat{I}$ added at every step. The curvature is $\vect{g} \vect{g}^\top$ (`"ggn"`), zero (`"identity"`) or a callable `curvature(mean, grad)`.

**Low rank.** The precision is $\gmat{\Lambda}_\theta = \mathrm{diag}(\vect{d}) + \mat{U} \mat{U}^\top$, with $\mat{U}$ of shape $(D_\theta, R)$, updated as in SLANG (Mishkin et al., 2018). The curvature comes as a factor, $\mat{C} = \mat{G} \mat{G}^\top$, so the prescribed low-rank part is exactly $\tilde{\mat{U}} \tilde{\mat{U}}^\top$ with $\tilde{\mat{U}} = [\sqrt{1 - \alpha} \mat{U}, \sqrt{\alpha} \mat{G}]$. A thin SVD keeps its top $R$ directions, and the diagonal of what is discarded moves into $\vect{d}$:

$$\vect{d} \leftarrow (1 - \alpha) \vect{d} + \alpha \delta + \mathrm{diag}(\tilde{\mat{U}} \tilde{\mat{U}}^\top - \mat{U} \mat{U}^\top) + \beta$$

So the diagonal of the precision is always what the rule prescribes, and with $R \ge D_\theta$ the whole precision is.

**IVON.** A diagonal variant for deep networks (Shen et al., 2024), with momentum $\vect{m}$ on the gradient (averaging rate $\beta_1$), a running curvature estimate $\vect{h}$ (rate $\beta_2$, effective sample size $N$), weight decay $\lambda$ as the prior precision, and a step clipped to a radius $r$. Its Gaussian is $\mathcal{N}(\gvect{\theta}, \mathrm{diag}(\vect{h} + \lambda)^{-1})$ around the current parameters. This implementation uses the squared gradient as its curvature estimate; the paper's IVON uses a reparameterisation-trick Hessian estimate and adds a correction term that keeps $\vect{h}$ positive.

**Newton.** The full-rank rule with $\alpha = 1$, prior precision $\beta$, a zero prior mean and the exact Hessian: $\gvect{\mu}_\theta \leftarrow \gvect{\mu}_\theta - (\beta \mat{I} + \mat{C})^{-1} (\vect{g} + \beta \gvect{\mu}_\theta)$. The Hessian can be indefinite away from a minimum, so the solve does not assume it is positive semidefinite. For a solve that runs to a tolerance, use [`GaussNewtonCG`](xref:api#argminax.solve.GaussNewtonCG) or an optimistix solver.

**A fixed precision.** [`preconditioned_sgd`](xref:api#argminax.blr.preconditioned_sgd) is the mean form with $\gmat{\Lambda}_\theta$ replaced by an operator $\mat{P}$ the caller supplies and that never changes: $\gvect{\theta} \leftarrow \gvect{\theta} - \alpha \mat{P}^{-1} \vect{g}$. It carries no Gaussian (its state is empty), and any prior belongs in the gradient it is given. $\mat{P}$ is a lineax or gaussx operator tagged positive semidefinite, factored once when the transform is built, or solved by its structure. [`scale_by_preconditioner`](xref:api#argminax.blr.scale_by_preconditioner) is the same step without the step size, for an optax chain.

## A worked example

Two hundred noisy observations of the line $y = 2x - 1$, with noise standard deviation 0.5 and a unit prior on the slope and the intercept. The cost is quadratic, so one full-rank step with the exact curvature gives the exact posterior, and the state goes straight into the diagnostics.

The [walkthrough notebook](../notebooks/blr_walkthrough.ipynb) runs the same problem three ways, with figures: `blr_diagonal` on minibatches, `blr_full_rank` converging to the exact posterior, and draws and diagnostics from the state.

```python
import jax
import jax.numpy as jnp
import lineax as lx

from argminax.blr import blr_full_rank, blr_posterior
from argminax.curvature import marginal_variance
from argminax.diagnose import averaging_kernel, degrees_of_freedom

jax.config.update("jax_enable_x64", True)

# Shapes: N = 200 observations; D = 2 parameters (slope, intercept)
key_inputs, key_noise = jax.random.split(jax.random.key(0))
inputs = jax.random.uniform(key_inputs, (200,), minval=-2.0, maxval=2.0)
targets = 2.0 * inputs - 1.0 + 0.5 * jax.random.normal(key_noise, (200,))
features = jnp.stack([inputs, jnp.ones(200)], axis=1)  # (200, 2)


def cost(theta):
    # 𝒥(θ) = ‖y − Φθ‖² / (2σ²)
    return 0.5 * jnp.sum((targets - features @ theta) ** 2) / 0.5**2


# C = ΦᵀΦ / σ², the curvature of the cost
curvature = features.T @ features / 0.5**2  # (2, 2)

optimiser = blr_full_rank(
    1.0, prior_precision=1.0, curvature=lambda mean, grad: curvature, damping=0.0
)
theta = jnp.zeros(2)
state = optimiser.init(theta)
updates, state = optimiser.update(jax.grad(cost)(theta), state)

# The Gaussian in the state, as the Laplace object the rest of argminax uses.
posterior = blr_posterior(state)
posterior.mode  # (2,)  [2.016, -0.989]
jnp.sqrt(marginal_variance(posterior))  # (2,)  [0.031, 0.035]

# How many of the two parameters did the data determine?
prior_cov = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
degrees_of_freedom(averaging_kernel(posterior, prior_cov)).value  # 1.998 of 2
```

The mean and the standard deviations equal the closed-form posterior of Bayesian linear regression, $(\delta \mat{I} + \mat{C})^{-1}$, to the digits shown.

In a training loop the transform is used like any optax optimiser, and the diagonal one works on the parameter pytree directly:

```python
import optax

from argminax.blr import blr_diagonal

optimiser = optax.chain(optax.clip_by_global_norm(1.0), blr_diagonal(1e-2))
state = optimiser.init(params)

for batch in batches:
    grads = jax.grad(minibatch_cost)(params, batch)
    updates, state = optimiser.update(grads, state)
    params = optax.apply_updates(params, updates)

posterior = blr_posterior(state[1])  # the BLR state is the second link of the chain
```

[`blr_posterior`](xref:api#argminax.blr.blr_posterior) flattens a pytree state in the order of `jax.flatten_util.ravel_pytree`, so `ravel_pytree(params)[1]` maps the mode or a draw back to the parameter pytree.

## What to watch for

### The curvature estimate decides what the variance means

| Estimate | How | What you get |
|---|---|---|
| The exact curvature $\nabla^2_\theta \mathcal{J}$ | a callable | the Laplace approximation at the fixed point, and the exact posterior of a quadratic cost. It can be indefinite for a cost that is not convex; the low-rank transform drops the negative part, the full-rank one does not. |
| The squared gradient, $\vect{g} \odot \vect{g}$ or $\vect{g} \vect{g}^\top$ | `"ggn"` | always positive semidefinite and cheap, but it is built from the gradient it is given, so it is an empirical-Fisher-type estimate (Kunstner et al., 2019), not the curvature. |
| Zero | `"identity"` | the precision stays at the prior, and the mean converges to the minimiser of the cost plus the prior. |

The squared gradient has two consequences worth knowing before you read a variance off it. It scales with the square of the cost: on the example above, with the cost summed over all 200 observations, `blr_diagonal` with `"ggn"` had moved the slope from 0 to 0.008 after 300 steps, because the precision grew as $\vect{g} \odot \vect{g}$ and the steps shrank with it; on the mean cost per observation it reached 2.07 after 1,000 steps. And it vanishes at a stationary point, so the precision decays towards the prior there: on that mean cost the standard deviations it reported were 1.4 and 2.3, against 0.031 and 0.035 for the exact posterior. Use `"ggn"` as an adaptive optimiser, and pass the exact or a Gauss–Newton curvature when you need the variance.

The rank-one $\vect{g} \vect{g}^\top$ can also be unstable under a nearly flat prior, where the precision is close to singular in every direction the recent gradients do not span. Use an informative prior or a better curvature estimate.

### Damping shifts the fixed point

The precision has to stay positive definite, which the rule guarantees for a positive semidefinite $\mat{C}$. The damping guards the other cases, in a different way in each family:

| Transform | Damping |
|---|---|
| `blr_diagonal` | a floor, $\gmat{\Lambda}_\theta \leftarrow \max(\gmat{\Lambda}_\theta, \beta)$ |
| `blr_full_rank` | $\beta \mat{I}$ added at every step |
| `blr_low_rank` | $\beta$ added to $\vect{d}$ at every step |

Added damping accumulates: the full-rank precision settles at $\delta \mat{I} + \mat{C} + (\beta / \alpha) \mat{I}$, a shift of $\beta / \alpha$. With the defaults ($\beta = 10^{-6}$, $\alpha = 10^{-2}$) the shift is $10^{-4}$, the size of the default prior precision. Set `damping=0.0` when the curvature is positive semidefinite and you want the unshifted fixed point, as the example above does.

### Monte Carlo instead of the mean

To take the expectations under $q$ by Monte Carlo and not at the mean, draw parameters from the current Gaussian before each gradient (`blr_posterior(state).sample(1, key=key)` gives a draw) and evaluate the gradient there. The built-in `"ggn"` curvature is built from the gradient it is given, so it follows the draw. A callable does not: the full-rank and low-rank transforms call `curvature(mean, grad)` with the mean in their state, so a callable that uses its first argument returns the curvature at the mean, next to a gradient taken at the draw. To evaluate both at the draw, build the transform inside the step, with a callable that closes over the draw:

```python
def step(mean, state, key):
    draw = blr_posterior(state).sample(1, key=key)[0]
    # the curvature at the draw, whatever mean the transform passes in
    optimiser = blr_full_rank(1e-2, curvature=lambda _, grad: jax.hessian(cost)(draw))
    updates, state = optimiser.update(jax.grad(cost)(draw), state)
    return mean + updates, state
```

A transform holds no state of its own, so the state of one passes to the next. The update is the change of the mean in either case: add it to the mean, never to the draw.

### A fixed preconditioner helps only where it fixes the conditioning

`preconditioned_sgd` is gradient descent in the coordinates $\vect{z} = \mat{L}^\top \gvect{\theta}$, with $\mat{P} = \mat{L} \mat{L}^\top$, so what decides its speed is the condition number of $\mat{P}^{-1} \mat{H}$, with $\mat{H}$ the Hessian of the cost. A constant step converges on a quadratic cost when $\alpha < 2 / \lambda_{\max}(\mat{P}^{-1} \mat{H})$.

The case it was added for is a few steps on a 4D-Var cost, preconditioned by the Hessian of the 3D-Var cost (Solvik et al., 2025): $\mat{P} = \gmat{\Sigma}_x^{-1} + N_t \mat{H}_h^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H}_h$, with $\gmat{\Sigma}_x$ the background covariance, $\gmat{\Sigma}_\varepsilon$ the observation-error covariance, $\mat{H}_h$ the Jacobian of the observation operator and $N_t$ the number of observation times. We measured it on one window of Lorenz-96 (36 sites, 18 observed at 3 times, a window of 0.1 time units), with a background covariance $\sigma_x^2 \exp(-d / \ell)$ of correlation length $\ell$ over the circular distance $d$ between sites:

| Background covariance | Condition number of $\mat{H}$ | of $\mat{P}^{-1} \mat{H}$ | Cost after 3 plain steps | after 3 preconditioned steps | Minimum |
|---|---|---|---|---|---|
| $\sigma_x^2 \mat{I}$ | 2.36 | 1.94 | 22.644 | 22.651 | 22.641 |
| $\ell = 1$ | 5.4 | 2.03 | 22.580 | 22.403 | 22.399 |
| $\ell = 3$ | 19.5 | 1.85 | 22.715 | 21.489 | 21.488 |
| $\ell = 6$ | 41.4 | 1.67 | 23.102 | 21.130 | 21.130 |

The plain steps use $\alpha = 1 / \lambda_{\max}(\mat{H})$ and the preconditioned ones $\alpha = 1$. With an identity background covariance the cost is already well conditioned and the preconditioner buys nothing. With a correlated one it reaches the minimum in three steps, where plain gradient descent at $\ell = 6$ had not after thirty. The approximation also degrades as the window grows, because $\mat{P}$ leaves out the model's propagator: on the same system with an identity background covariance, the largest eigenvalue of $\mat{P}^{-1} \mat{H}$ was 1.5 at a window of 0.1 time units and 3.4 at 0.2, past the stability limit of a unit step. For longer windows use a solver that re-linearises, such as [`GaussNewtonCG`](xref:api#argminax.solve.GaussNewtonCG).

The transform does not guard against a step that is too large; the loop that calls it should.

### Precision and JAX

- The state, the updates and the posterior keep the dtype of the parameters; a gradient in another dtype is cast to it.
- Every `update` is a pure function with no Python control flow on traced values, so it runs under `jax.jit`, and the states are `NamedTuple` pytrees, so they can be carried through `jax.lax.scan`.
- `Laplace.sample` factors a dense or a low-rank precision densely, $O(D_\theta^3)$; a diagonal one is sampled in closed form.

## The same rule, other names

Each row is the rule with one choice of family, curvature, step size and prior. Khan and Rue (2023, Table 1) give more.

| Algorithm | Family | Curvature | Step size | Here |
|---|---|---|---|---|
| Gradient descent with weight decay | diagonal | zero | $\alpha$ (a step of $\alpha / \delta$) | `blr_diagonal(curvature="identity")` |
| RMSprop-like scaling, without the square root | diagonal | squared gradient | about $1 - \beta_2$ | `blr_diagonal` |
| Newton's method | full rank | exact | 1 | `newton` |
| Preconditioned gradient descent | any, fixed | supplied by the caller | $\alpha$ | `preconditioned_sgd` |
| The Laplace approximation | full rank | exact, at the mean | the fixed point | `blr_full_rank` with an exact callable |
| A conjugate (linear-Gaussian) update | full rank | exact | 1 | `blr_full_rank(1.0)` with an exact callable, as in the example |
| Online and continual learning | any | any | $\alpha$ | not expressible yet, except when the previous posterior is isotropic (see below) |
| SLANG | low rank | a Gauss–Newton factor | $\alpha$ | `blr_low_rank` |
| IVON | diagonal | squared gradient, in this implementation | $1 - \beta_2$ | `ivon` |

The prior of every transform is isotropic, $\mathcal{N}(\gvect{\mu}_0, \delta^{-1} \mat{I})$ (and `blr_diagonal` takes one scalar prior mean for every parameter), so neither a Kalman measurement update with a general predictive covariance nor a posterior handed on as the next prior is expressible yet. Starting a new transform at the old posterior's mode, or carrying the old state on, does not do it: the next updates still pull towards the configured isotropic prior, and the uncertainty learned before decays at the rate $\alpha$.

## References

- Khan and Rue (2023), "The Bayesian Learning Rule", arXiv:2107.04562.
- Opper and Archambeau (2009), "The Variational Gaussian Approximation Revisited", Neural Computation 21(3).
- Mishkin, Kunstner, Nielsen, Schmidt and Khan (2018), "SLANG: Fast Structured Covariance Approximations for Bayesian Deep Learning with Natural Gradient", arXiv:1811.04504.
- Shen et al. (2024), "Variational Learning is Effective for Large Deep Networks", arXiv:2402.17641.
- Kunstner, Balles and Hennig (2019), "Limitations of the Empirical Fisher Approximation for Natural Gradient Descent", arXiv:1905.12558.
- Solvik, Penny and Hoyer (2025), "4D-Var using Hessian approximation and backpropagation applied to automatically differentiable numerical and machine learning models", Journal of Advances in Modeling Earth Systems 17, e2024MS004608, doi:10.1029/2024MS004608.

# The Bayesian learning rule: derivation and algorithms

A solver in argminax runs to a minimum and reads the Gaussian there. The transforms in `argminax.blr` carry that Gaussian while they train, and they all come from one update, the Bayesian learning rule of Khan and Rue (2023). This page derives the rule from the variational objective, writes out the algorithm of every transform next to the code that implements it, states the numerical safeguards the transforms rely on, and lists the classical algorithms the rule reduces to. The [guide](../guide/blr.md) is the place to start if you want to use a transform; this page is the reference behind it.

Every code block on this page was run as written against argminax 0.0.7, and each one ends by checking the hand-written step against the transform that ships.

## Notation

The letter says what kind of object a symbol is and the subscript says which variable it belongs to. We work with the precision throughout (never the covariance), because the rule is linear in it.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $\gvect{\theta}$ | $\mathbb{R}^{D_\theta}$ | the parameters being trained |
| $\mathcal{J}(\gvect{\theta})$ | scalar | the cost of one minibatch, a negative log-likelihood |
| $\vect{g} = \nabla_{\gvect{\theta}} \mathcal{J}$ | $\mathbb{R}^{D_\theta}$ | its gradient, which is what any optax optimiser takes |
| $\mat{C} \approx \nabla^2_{\gvect{\theta}} \mathcal{J}$ | $D_\theta \times D_\theta$ | an estimate of its curvature, positive semidefinite |
| $q(\gvect{\theta}; \gvect{\nu})$ | density | the distribution we keep, with natural parameters $\gvect{\nu} \in \mathbb{R}^{D_\nu}$ |
| $\vect{T}(\gvect{\theta})$, $\gvect{\mu}_T$ | $\mathbb{R}^{D_\nu}$ | the sufficient statistics of $q$ and their mean, the expectation parameters |
| $\gvect{\mu}_\theta$, $\gmat{\Lambda}_\theta$ | $\mathbb{R}^{D_\theta}$, $D_\theta \times D_\theta$ | the mean and the precision of a Gaussian $q$ |
| $\gvect{\mu}_0$, $\delta$ | $\mathbb{R}^{D_\theta}$, scalar | the prior $p(\gvect{\theta}) = \mathcal{N}(\gvect{\mu}_0, \delta^{-1} \mat{I})$, with $\delta > 0$ |
| $\alpha$ | scalar | the step size, in $(0, 1]$ |
| $\beta$ | scalar | the damping, $\beta \ge 0$ |
| $i$ | integer | the step count, written as a superscript $\gvect{\mu}_\theta^{(i)}$ |

:::{note} The sign convention
Every transform takes the gradient of the **cost**, and a curvature is the curvature of the cost. Khan and Rue, and optax_bayes before it moved here, write the rule for the log-likelihood $-\mathcal{J}$, so their gradient is $-\vect{g}$ and their Hessian is $-\mat{C}$. The formulas below are theirs with those two signs flipped.
:::

## From a point estimate to a distribution

An optimiser returns one point, $\gvect{\theta}^\star = \mathrm{argmin}_{\gvect{\theta}} \mathcal{J}(\gvect{\theta})$, and says nothing about how well the data determined it. So instead of a point we keep a distribution $q(\gvect{\theta}; \gvect{\nu})$ and choose its parameters by minimising the expected cost plus the divergence from the prior,

$$\mathcal{L}_\text{ELBO}(\gvect{\nu}) = \mathbb{E}_{q}[\mathcal{J}(\gvect{\theta})] + \mathrm{KL}[q(\gvect{\theta}; \gvect{\nu}) \Vert p(\gvect{\theta})].$$ (eq-blr-objective)

This is the negative of the evidence lower bound, so it is the quantity we minimise. Without any restriction on $q$ its minimiser is the exact posterior. The first term asks $q$ to sit where the cost is low, and the second one stops it from collapsing onto a single point.

## Natural gradients in an exponential family

We take $q$ from an exponential family,

$$q(\gvect{\theta}; \gvect{\nu}) = h(\gvect{\theta}) \exp\left(\gvect{\nu}^\top \vect{T}(\gvect{\theta}) - A(\gvect{\nu})\right),$$ (eq-blr-expfam)

where $h$ is the base measure and $A$ the log-partition function, both scalar. Two facts about such a family do all the work. The gradient of the log-partition function is the mean of the sufficient statistics, $\gvect{\mu}_T = \mathbb{E}_q[\vect{T}(\gvect{\theta})] = \nabla_{\gvect{\nu}} A$, and its Hessian is the Fisher information $\mat{F}_\nu = \nabla^2_{\gvect{\nu}} A = \partial \gvect{\mu}_T / \partial \gvect{\nu}$, of shape $D_\nu \times D_\nu$.

A plain gradient step on $\gvect{\nu}$ depends on how the family happens to be parameterised. The natural gradient removes that dependence by measuring distance with the Fisher information, and for a minimal family (one where $\mat{F}_\nu$ is invertible) it is very cheap. The chain rule gives $\nabla_{\gvect{\nu}} f = \mat{F}_\nu \nabla_{\gvect{\mu}_T} f$ for any scalar $f$, so

$$\mat{F}_\nu^{-1} \nabla_{\gvect{\nu}} f = \nabla_{\gvect{\mu}_T} f.$$ (eq-blr-natural-gradient)

Intuitively, the natural gradient with respect to the natural parameters is the plain gradient with respect to the expectation parameters, and no Fisher matrix is ever inverted.

## The rule

Natural-gradient descent on [](#eq-blr-objective) with step size $\alpha$ is $\gvect{\nu}^{(i+1)} = \gvect{\nu}^{(i)} - \alpha \nabla_{\gvect{\mu}_T} \mathcal{L}_\text{ELBO}$, so we need the gradient of each term. Take a prior from the same family, with natural parameters $\gvect{\nu}_0$. The base measures cancel in the divergence, which leaves $\mathrm{KL}[q \Vert p] = (\gvect{\nu} - \gvect{\nu}_0)^\top \gvect{\mu}_T - A(\gvect{\nu}) + A(\gvect{\nu}_0)$. Differentiating with respect to $\gvect{\mu}_T$, the two terms that come from the dependence of $\gvect{\nu}$ on $\gvect{\mu}_T$ cancel, because $\nabla_{\gvect{\nu}} A = \gvect{\mu}_T$:

$$\nabla_{\gvect{\mu}_T} \mathrm{KL}[q \Vert p] = \gvect{\nu} - \gvect{\nu}_0.$$ (eq-blr-kl-gradient)

Substituting it gives the Bayesian learning rule,

$$\gvect{\nu}^{(i+1)} = (1 - \alpha) \gvect{\nu}^{(i)} + \alpha \left(\gvect{\nu}_0 - \nabla_{\gvect{\mu}_T} \mathbb{E}_{q}[\mathcal{J}(\gvect{\theta})]\right),$$ (eq-blr-rule)

with the expectation taken under the current $q$. Each step is a convex combination of the current natural parameters and a target, which is why the step size lives in $(0, 1]$. With $\alpha = 1$ the old parameters are forgotten and the step jumps straight to the target.

## The Gaussian case

For $q = \mathcal{N}(\gvect{\mu}_\theta, \gmat{\Lambda}_\theta^{-1})$ the sufficient statistics are $\gvect{\theta}$ and $\gvect{\theta} \gvect{\theta}^\top$. The natural parameters are $\gmat{\Lambda}_\theta \gvect{\mu}_\theta$ (which we call the natural mean) and $-\frac{1}{2} \gmat{\Lambda}_\theta$, and the expectation parameters are $\gvect{\mu}_{T,1} = \gvect{\mu}_\theta$ and $\gvect{\mu}_{T,2} = \gvect{\mu}_\theta \gvect{\mu}_\theta^\top + \gmat{\Lambda}_\theta^{-1}$.

The identities of Bonnet and Price (Opper and Archambeau, 2009) turn the derivatives of a Gaussian expectation with respect to its mean and its covariance into an expected gradient and an expected Hessian. We write $\vect{g} = \mathbb{E}_q[\nabla_{\gvect{\theta}} \mathcal{J}]$ and $\mat{C} = \mathbb{E}_q[\nabla^2_{\gvect{\theta}} \mathcal{J}]$ for the two. The covariance is $\gmat{\Lambda}_\theta^{-1} = \gvect{\mu}_{T,2} - \gvect{\mu}_{T,1} \gvect{\mu}_{T,1}^\top$, so the chain rule gives

$$\nabla_{\gvect{\mu}_{T,1}} \mathbb{E}_q[\mathcal{J}] = \vect{g} - \mat{C} \gvect{\mu}_\theta,$$ (eq-blr-gradient-first)

$$\nabla_{\gvect{\mu}_{T,2}} \mathbb{E}_q[\mathcal{J}] = \frac{1}{2} \mat{C}.$$ (eq-blr-gradient-second)

The term $\mat{C} \gvect{\mu}_\theta$ comes from the dependence of the covariance on the first expectation parameter. It is easy to forget, and leaving it out gives the wrong fixed point.

Putting [](#eq-blr-gradient-first) and [](#eq-blr-gradient-second) into [](#eq-blr-rule), with the prior's natural parameters $\delta \gvect{\mu}_0$ and $-\frac{1}{2} \delta \mat{I}$, we get the update that every transform here builds on:

$$\gmat{\Lambda}_\theta^{(i+1)} = (1 - \alpha) \gmat{\Lambda}_\theta^{(i)} + \alpha (\delta \mat{I} + \mat{C}),$$ (eq-blr-precision)

$$\gmat{\Lambda}_\theta^{(i+1)} \gvect{\mu}_\theta^{(i+1)} = (1 - \alpha) \gmat{\Lambda}_\theta^{(i)} \gvect{\mu}_\theta^{(i)} + \alpha (\delta \gvect{\mu}_0 - \vect{g} + \mat{C} \gvect{\mu}_\theta^{(i)}).$$ (eq-blr-natural-mean)

The new mean is the new natural mean divided by the new precision. So the precision is a running average of the prior precision plus the curvature, and it stays positive definite for as long as $\mat{C}$ is positive semidefinite.

### The mean form

The same step can be written on the mean. Multiplying [](#eq-blr-precision) by $\gvect{\mu}_\theta^{(i)}$ and subtracting it from [](#eq-blr-natural-mean) leaves $\gmat{\Lambda}_\theta^{(i+1)} (\gvect{\mu}_\theta^{(i+1)} - \gvect{\mu}_\theta^{(i)}) = -\alpha (\vect{g} + \delta (\gvect{\mu}_\theta^{(i)} - \gvect{\mu}_0))$, which is

$$\gvect{\mu}_\theta^{(i+1)} = \gvect{\mu}_\theta^{(i)} - \alpha \left(\gmat{\Lambda}_\theta^{(i+1)}\right)^{-1} \left(\vect{g} + \delta (\gvect{\mu}_\theta^{(i)} - \gvect{\mu}_0)\right).$$ (eq-blr-mean-form)

This is a step down the gradient of the cost plus the prior, preconditioned by the new precision. The two forms agree exactly when the stored precision is the one [](#eq-blr-precision) prescribes and there is no damping. They differ when the stored precision is only an approximation of it, as in the low-rank transform. There the mean form stays a well-defined preconditioned step, whereas the natural form would multiply the approximation error by the inverse of the precision along weakly curved directions.

### The fixed point

The transforms use whatever gradient and curvature they are given. Evaluating both at the mean replaces the expectations under $q$ by their value there (the delta method), which is the usual way to run an optimiser. A fixed point of [](#eq-blr-precision) and [](#eq-blr-mean-form) with the exact curvature then satisfies the two conditions

$$\vect{g} + \delta (\gvect{\mu}_\theta - \gvect{\mu}_0) = \vect{0},$$ (eq-blr-fixed-point-mean)

$$\gmat{\Lambda}_\theta = \delta \mat{I} + \nabla^2_{\gvect{\theta}} \mathcal{J}(\gvect{\mu}_\theta),$$ (eq-blr-fixed-point-precision)

which is the Laplace approximation: the mean is a stationary point of the cost plus the prior, and the precision is the Hessian there. For a cost that is quadratic in $\gvect{\theta}$ this is the exact posterior, and one step with $\alpha = 1$ reaches it from any start. It is also the Gaussian that [`gauss_newton`](xref:api#argminax.curvature.gauss_newton) builds at a converged minimum, so a transform and a solver end at the same object.

To take the expectations by Monte Carlo and not at the mean, draw parameters from the current $q$ before each gradient and evaluate the gradient and the curvature at the draw. The [guide](../guide/blr.md) shows how, and what to watch for with a callable curvature.

## The families

A family is a choice of structure for the precision. Each one trades storage and cost per step against how much of the curvature it can hold.

| Transform | Precision $\gmat{\Lambda}_\theta$ | Storage | Cost of a step | Parameters |
| --- | --- | --- | --- | --- |
| [`blr_diagonal`](xref:api#argminax.blr.blr_diagonal) | $\op{diag}(\vect{d})$ | $O(D_\theta)$ | $O(D_\theta)$ | any pytree |
| [`blr_full_rank`](xref:api#argminax.blr.blr_full_rank) | dense | $O(D_\theta^2)$ | $O(D_\theta^3)$, two solves | a flat vector |
| [`blr_low_rank`](xref:api#argminax.blr.blr_low_rank) | $\op{diag}(\vect{d}) + \mat{U} \mat{U}^\top$ | $O(D_\theta R)$ | $O(D_\theta R^2)$ with a built-in curvature | a flat vector |
| [`ivon`](xref:api#argminax.blr.ivon) | $\op{diag}(\bar{\vect{c}} + \delta)$ | $O(D_\theta)$ | $O(D_\theta)$ | any pytree |
| [`newton`](xref:api#argminax.blr.newton) | dense | $O(D_\theta^2)$ | $O(D_\theta^3)$ | a flat vector |
| [`preconditioned_sgd`](xref:api#argminax.blr.preconditioned_sgd) | an operator the caller fixes | that of the operator | one solve with it | a flat vector |

Here $\vect{d} \in \mathbb{R}^{D_\theta}$ is a positive diagonal, $\mat{U} \in \mathbb{R}^{D_\theta \times R}$ a factor of rank $R$, and $\bar{\vect{c}} \in \mathbb{R}^{D_\theta}$ a running curvature estimate.

Three things hold for every transform that learns its precision. The state starts at the prior precision with its mean at the parameters given to `init`. The prior enters every step through $\delta$ and $\gvect{\mu}_0$. And `update` returns the change of the mean, $\gvect{\mu}_\theta^{(i+1)} - \gvect{\mu}_\theta^{(i)}$, so after `optax.apply_updates` the parameters equal the mean.

### Diagonal

With $\gmat{\Lambda}_\theta = \op{diag}(\vect{d})$ the natural mean is $\vect{d} \odot \gvect{\mu}_\theta$ and the curvature is a vector $\vect{c} \in \mathbb{R}^{D_\theta}$, the squared gradient $\vect{g} \odot \vect{g}$ (`curvature="ggn"`) or zero (`"identity"`). The updates [](#eq-blr-precision) and [](#eq-blr-natural-mean) become elementwise, and no linear solve is needed:

$$\vect{d}^{(i+1)} = \max\left((1 - \alpha) \vect{d}^{(i)} + \alpha (\delta + \vect{c}), \beta\right),$$ (eq-blr-diagonal-precision)

$$\vect{d}^{(i+1)} \odot \gvect{\mu}_\theta^{(i+1)} = (1 - \alpha) \vect{d}^{(i)} \odot \gvect{\mu}_\theta^{(i)} + \alpha (\delta \mu_0 - \vect{g} + \vect{c} \odot \gvect{\mu}_\theta^{(i)}).$$ (eq-blr-diagonal-natural-mean)

The damping is a floor on the precision here, and the prior mean $\mu_0$ is one scalar for every parameter. The transform works leaf by leaf on a parameter pytree, and takes its step from `gaussx.blr_diag_update` on the log-joint. [](#code-blr-diagonal) writes the same step by hand.

```{code-block} python
:caption: One step of [](#eq-blr-diagonal-precision) and [](#eq-blr-diagonal-natural-mean) by hand, checked against `blr_diagonal`.
:label: code-blr-diagonal
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from argminax.blr import blr_diagonal

jax.config.update("jax_enable_x64", True)

# Shapes: D_θ = 3 parameters


# user
def diagonal_step_by_hand(
    diag_precision: Float[Array, " state"],
    nat_mean: Float[Array, " state"],
    grad: Float[Array, " state"],
    *,
    step_size: float,
    prior_precision: float,
    prior_mean: float,
    damping: float,
) -> tuple[Float[Array, " state"], Float[Array, " state"], Float[Array, " state"]]:
    # μ_θ = (d ⊙ μ_θ) / d
    mean = nat_mean / diag_precision  # (D_θ,) → (D_θ,)
    # c = g ⊙ g
    curvature = grad * grad  # (D_θ,) → (D_θ,)
    # d ← (1 − α) d + α (δ + c)
    target_precision = prior_precision + curvature
    new_precision = (1 - step_size) * diag_precision + step_size * target_precision
    # d ← max(d, β)
    new_precision = jnp.maximum(new_precision, damping)  # (D_θ,) → (D_θ,)
    # d ⊙ μ_θ ← (1 − α) d ⊙ μ_θ + α (δ μ₀ − g + c ⊙ μ_θ)
    target_nat_mean = prior_precision * prior_mean - grad + curvature * mean
    new_nat_mean = (1 - step_size) * nat_mean + step_size * target_nat_mean
    # Δθ = μ_θ' − μ_θ
    update = new_nat_mean / new_precision - mean  # (D_θ,) → (D_θ,)
    return update, new_precision, new_nat_mean


theta = jnp.array([1.0, -2.0, 0.5])
grad = jnp.array([0.3, -1.2, 2.0])

# today (argminax)
optimiser = blr_diagonal(0.1, prior_precision=2.0, prior_mean=0.5, damping=1e-6)
state = optimiser.init(theta)
updates, new_state = optimiser.update(grad, state)

by_hand = diagonal_step_by_hand(
    state.precision,
    state.nat_mean,
    grad,
    step_size=0.1,
    prior_precision=2.0,
    prior_mean=0.5,
    damping=1e-6,
)
bool(jnp.allclose(updates, by_hand[0]))  # True
bool(jnp.allclose(new_state.precision, by_hand[1]))  # True
bool(jnp.allclose(new_state.nat_mean, by_hand[2]))  # True
```

### Full rank

The full-rank transform stores a dense precision and applies [](#eq-blr-precision) and [](#eq-blr-natural-mean) as they stand, with $\beta \mat{I}$ added to the precision at every step:

$$\gmat{\Lambda}_\theta^{(i+1)} = (1 - \alpha) \gmat{\Lambda}_\theta^{(i)} + \alpha (\delta \mat{I} + \mat{C}) + \beta \mat{I}.$$ (eq-blr-full-rank-precision)

The curvature is $\vect{g} \vect{g}^\top$ (`"ggn"`), zero (`"identity"`) or a callable `curvature(mean, grad)` that returns a $D_\theta \times D_\theta$ matrix. A step needs two solves with the precision, one for the mean before the update and one after. The precision is tagged positive semidefinite, so gaussx solves by Cholesky unless a `solver` says otherwise.

```{code-block} python
:caption: One step of [](#eq-blr-full-rank-precision) and [](#eq-blr-natural-mean) by hand, checked against `blr_full_rank`.
:label: code-blr-full-rank
import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from argminax.blr import blr_full_rank

jax.config.update("jax_enable_x64", True)

# Shapes: D_θ = 3 parameters


# user
def full_rank_step_by_hand(
    precision: Float[Array, "state state"],
    nat_mean: Float[Array, " state"],
    grad: Float[Array, " state"],
    *,
    step_size: float,
    prior_precision: float,
    prior_mean: Float[Array, " state"],
    damping: float,
) -> tuple[Float[Array, " state"], Float[Array, "state state"], Float[Array, " state"]]:
    identity = jnp.eye(grad.shape[0])
    # today (gaussx)
    # μ_θ = Λ_θ⁻¹ (Λ_θ μ_θ), by Cholesky since Λ_θ is tagged PSD
    operator = lx.MatrixLinearOperator(precision, lx.positive_semidefinite_tag)
    mean = gx.solve(operator, nat_mean)  # (D_θ,) → (D_θ,)
    # C = g gᵀ
    curvature = einx.dot("row, column -> row column", grad, grad)  # → (D_θ, D_θ)
    # Λ_θ ← (1 − α) Λ_θ + α (δ I + C) + β I
    target_precision = prior_precision * identity + curvature
    new_precision = (1 - step_size) * precision + step_size * target_precision
    new_precision = new_precision + damping * identity  # → (D_θ, D_θ)
    # C μ_θ
    curved_mean = einx.dot("row column, column -> row", curvature, mean)
    # Λ_θ μ_θ ← (1 − α) Λ_θ μ_θ + α (δ μ₀ − g + C μ_θ)
    target_nat_mean = prior_precision * prior_mean - grad + curved_mean
    new_nat_mean = (1 - step_size) * nat_mean + step_size * target_nat_mean
    # today (gaussx)
    # Δθ = Λ_θ'⁻¹ (Λ_θ μ_θ)' − μ_θ
    new_operator = lx.MatrixLinearOperator(new_precision, lx.positive_semidefinite_tag)
    update = gx.solve(new_operator, new_nat_mean) - mean  # (D_θ,) → (D_θ,)
    return update, new_precision, new_nat_mean


theta = jnp.array([1.0, -2.0, 0.5])
grad = jnp.array([0.3, -1.2, 2.0])
prior_mean = jnp.array([0.5, 0.0, -0.5])

# today (argminax)
optimiser = blr_full_rank(0.1, prior_precision=2.0, prior_mean=prior_mean, damping=1e-6)
state = optimiser.init(theta)
updates, new_state = optimiser.update(grad, state)

by_hand = full_rank_step_by_hand(
    state.precision,
    state.nat_mean,
    grad,
    step_size=0.1,
    prior_precision=2.0,
    prior_mean=prior_mean,
    damping=1e-6,
)
bool(jnp.allclose(updates, by_hand[0]))  # True
bool(jnp.allclose(new_state.precision, by_hand[1]))  # True
bool(jnp.allclose(new_state.nat_mean, by_hand[2]))  # True
```

### Low rank

The low-rank transform follows SLANG (Mishkin et al., 2018). It stores $\gmat{\Lambda}_\theta = \op{diag}(\vect{d}) + \mat{U} \mat{U}^\top$, and it takes the curvature as a factor, $\mat{C} = \mat{G} \mat{G}^\top$ with $\mat{G} \in \mathbb{R}^{D_\theta \times K}$: the gradient as one column for `"ggn"`, no columns for `"identity"`, and the positive part of a callable's matrix otherwise. With a factor, the precision that [](#eq-blr-precision) prescribes is again a diagonal plus a low-rank term, with no approximation:

$$(1 - \alpha) \gmat{\Lambda}_\theta^{(i)} + \alpha (\delta \mat{I} + \mat{C}) = \op{diag}\left((1 - \alpha) \vect{d}^{(i)} + \alpha \delta\right) + \tilde{\mat{U}} \tilde{\mat{U}}^\top,$$ (eq-blr-low-rank-prescribed)

where the two factors are stacked side by side,

$$\tilde{\mat{U}} = \left[\sqrt{1 - \alpha} \mat{U}^{(i)}, \sqrt{\alpha} \mat{G}\right].$$ (eq-blr-low-rank-stack)

The stacked factor $\tilde{\mat{U}}$ has $R + K$ columns, so it grows at every step. A thin SVD, $\tilde{\mat{U}} = \mat{V} \mat{S} \mat{W}^\top$, keeps its top $R$ directions, and the diagonal of what is discarded moves into $\vect{d}$:

$$\mat{U}^{(i+1)} = [\mat{V} \mat{S}]_{:, 1:R},$$ (eq-blr-low-rank-factor)

$$\vect{d}^{(i+1)} = (1 - \alpha) \vect{d}^{(i)} + \alpha \delta + \op{diag}\left(\tilde{\mat{U}} \tilde{\mat{U}}^\top - \mat{U}^{(i+1)} \mat{U}^{(i+1)\top}\right) + \beta.$$ (eq-blr-low-rank-diagonal)

So the only approximation is the truncation, and the diagonal of the precision is always what the rule prescribes. With $R \ge D_\theta$ nothing is discarded and the whole precision is exact. The mean takes the mean-form step [](#eq-blr-mean-form) with the new precision, solved by the Woodbury identity.

A step costs $O(D_\theta (R + K)^2)$, which is $O(D_\theta R^2)$ for the two built-in curvatures. A callable returns a dense matrix, which the transform factors by an eigendecomposition at $O(D_\theta^3)$ and $O(D_\theta^2)$ of temporary storage, whatever the rank. So a callable is only worth using here when $D_\theta$ is small.

```{code-block} python
:caption: One step of [](#eq-blr-low-rank-stack), [](#eq-blr-low-rank-factor) and [](#eq-blr-low-rank-diagonal) by hand, on a step that discards a direction, checked against `blr_low_rank`.
:label: code-blr-low-rank
import einx
import gaussx as gx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from argminax.blr import blr_low_rank

jax.config.update("jax_enable_x64", True)

# Shapes: D_θ = 4 parameters; R = 2 kept directions; K = 1 curvature column


# user
def low_rank_step_by_hand(
    diag_precision: Float[Array, " state"],
    low_rank_factor: Float[Array, "state rank"],
    nat_mean: Float[Array, " state"],
    grad: Float[Array, " state"],
    *,
    step_size: float,
    prior_precision: float,
    prior_mean: Float[Array, " state"],
    damping: float,
) -> tuple[
    Float[Array, " state"],
    Float[Array, " state"],
    Float[Array, "state rank"],
    Float[Array, " state"],
]:
    rank = low_rank_factor.shape[1]
    # today (gaussx)
    # μ_θ = (diag(d) + U Uᵀ)⁻¹ (Λ_θ μ_θ), by the Woodbury identity
    operator = gx.low_rank_plus_diag(diag_precision, low_rank_factor, psd=True)
    mean = gx.solve(operator, nat_mean)  # (D_θ,) → (D_θ,)
    # G = g, one column, so C = G Gᵀ = g gᵀ
    curvature_factor = einx.id("state -> state 1", grad)  # (D_θ,) → (D_θ, K)
    # Ũ = [√(1 − α) U, √α G]
    decayed = jnp.sqrt(1 - step_size) * low_rank_factor
    incoming = jnp.sqrt(step_size) * curvature_factor
    stacked = einx.id(  # → (D_θ, R + K)
        "state rank, state factor -> state (rank + factor)", decayed, incoming
    )
    # Ũ = V S Wᵀ, the thin SVD
    left, singular_values, _ = jnp.linalg.svd(stacked, full_matrices=False)
    # V S, the left singular vectors scaled by the singular values
    scaled = einx.multiply("state kept, kept -> state kept", left, singular_values)
    # U ← top_R(Ũ), the first R columns of V S
    new_factor = scaled[:, :rank]  # (D_θ, R + K) → (D_θ, R)
    # e = diag(Ũ Ũᵀ − U Uᵀ), the squared row norms of the discarded columns
    discarded = scaled[:, rank:]
    residual = einx.sum("state [kept]", discarded * discarded)  # → (D_θ,)
    # d ← (1 − α) d + α δ + e + β
    new_diag = (1 - step_size) * diag_precision + step_size * prior_precision
    new_diag = new_diag + residual + damping  # → (D_θ,)
    # g + δ (μ_θ − μ₀)
    joint_grad = grad + prior_precision * (mean - prior_mean)
    # today (gaussx)
    # Δθ = −α (diag(d) + U Uᵀ)⁻¹ (g + δ (μ_θ − μ₀)), with the new d and U
    new_operator = gx.low_rank_plus_diag(new_diag, new_factor, psd=True)
    update = gx.solve(new_operator, -step_size * joint_grad)  # (D_θ,) → (D_θ,)
    # Λ_θ μ_θ = (diag(d) + U Uᵀ) (μ_θ + Δθ)
    new_nat_mean = new_operator.mv(mean + update)  # (D_θ,) → (D_θ,)
    return update, new_diag, new_factor, new_nat_mean


theta = jnp.array([1.0, -2.0, 0.5, 0.0])
prior_mean = jnp.array([0.5, 0.0, -0.5, 1.0])
grads = jnp.array(
    [[0.3, -1.2, 2.0, 0.1], [1.0, 0.4, -0.7, 0.2], [-0.5, 0.9, 0.3, -1.1]]
)

# today (argminax)
optimiser = blr_low_rank(
    0.1, rank=2, prior_precision=2.0, prior_mean=prior_mean, damping=1e-6
)
state = optimiser.init(theta)
# two steps fill both columns of U, so the third one discards a direction
_, state = optimiser.update(grads[0], state)
_, state = optimiser.update(grads[1], state)
updates, new_state = optimiser.update(grads[2], state)

by_hand = low_rank_step_by_hand(
    state.diag_precision,
    state.low_rank_factor,
    state.nat_mean,
    grads[2],
    step_size=0.1,
    prior_precision=2.0,
    prior_mean=prior_mean,
    damping=1e-6,
)
bool(jnp.allclose(updates, by_hand[0]))  # True
bool(jnp.allclose(new_state.diag_precision, by_hand[1]))  # True
# U is defined up to the sign of each column, so compare U Uᵀ
by_hand_outer = einx.dot("row rank, column rank -> row column", by_hand[2], by_hand[2])
shipped = new_state.low_rank_factor
shipped_outer = einx.dot("row rank, column rank -> row column", shipped, shipped)
bool(jnp.allclose(shipped_outer, by_hand_outer))  # True
bool(jnp.allclose(new_state.nat_mean, by_hand[3]))  # True
```

### IVON

IVON (Shen et al., 2024) is a diagonal variant made for deep networks. It adds momentum to the gradient, a running average for the curvature, and a clip on the step. With $\bar{\vect{g}} \in \mathbb{R}^{D_\theta}$ the running mean of the gradients, $\bar{\vect{c}} \in \mathbb{R}^{D_\theta}$ the running curvature estimate, $\rho_1$ and $\rho_2$ their averaging rates in $[0, 1)$, $N_\text{ess}$ the effective sample size and $r_\text{clip}$ the clipping radius, one step is

$$\bar{\vect{g}}^{(i)} = \rho_1 \bar{\vect{g}}^{(i-1)} + (1 - \rho_1) \vect{g},$$ (eq-blr-ivon-momentum)

$$\bar{\vect{c}}^{(i)} = \rho_2 \bar{\vect{c}}^{(i-1)} + (1 - \rho_2) N_\text{ess} \vect{g} \odot \vect{g},$$ (eq-blr-ivon-curvature)

$$\gvect{\theta}^{(i)} = \gvect{\theta}^{(i-1)} - \alpha \op{clip}\left(\frac{\bar{\vect{g}}^{(i)} / (1 - \rho_1^i) + \delta \gvect{\theta}^{(i-1)}}{\bar{\vect{c}}^{(i)} + \delta}, \pm r_\text{clip}\right),$$ (eq-blr-ivon-step)

with the division and the clip taken elementwise. The weight decay $\delta$ plays the part of the prior precision, around a zero prior mean, and the factor $1 - \rho_1^i$ corrects the momentum for its zero start. The Gaussian it carries is $\mathcal{N}(\gvect{\theta}, \op{diag}(\bar{\vect{c}} + \delta)^{-1})$ around the current parameters. The state holds neither the mean nor the weight decay, so [`blr_posterior`](xref:api#argminax.blr.blr_posterior) needs both passed in.

This implementation uses the squared gradient as its curvature estimate. The IVON of the paper estimates the Hessian by the reparameterisation trick and adds a correction term that keeps $\bar{\vect{c}}$ positive, and this transform does neither.

```{code-block} python
:caption: One step of [](#eq-blr-ivon-momentum), [](#eq-blr-ivon-curvature) and [](#eq-blr-ivon-step) by hand, checked against `ivon`.
:label: code-blr-ivon
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from argminax.blr import ivon

jax.config.update("jax_enable_x64", True)

# Shapes: D_θ = 3 parameters


# user
def ivon_step_by_hand(
    momentum: Float[Array, " state"],
    hess: Float[Array, " state"],
    grad: Float[Array, " state"],
    params: Float[Array, " state"],
    count: int,
    *,
    step_size: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    clip_radius: float,
    ess: float,
) -> tuple[Float[Array, " state"], Float[Array, " state"], Float[Array, " state"]]:
    # ḡ ← β₁ ḡ + (1 − β₁) g
    new_momentum = beta1 * momentum + (1 - beta1) * grad  # (D_θ,) → (D_θ,)
    # c̄ ← β₂ c̄ + (1 − β₂) N_ess g ⊙ g
    new_hess = beta2 * hess + (1 - beta2) * ess * grad * grad  # (D_θ,) → (D_θ,)
    # ḡ / (1 − β₁ⁱ), the correction for the zero start, at step i = count + 1
    debiased = new_momentum / (1 - beta1 ** (count + 1))
    # (ḡ / (1 − β₁ⁱ) + δ θ) / (c̄ + δ)
    direction = (debiased + weight_decay * params) / (new_hess + weight_decay)
    # Δθ = −α clip(·, ±r_clip)
    update = -step_size * jnp.clip(direction, -clip_radius, clip_radius)
    return update, new_momentum, new_hess


theta = jnp.array([1.0, -2.0, 0.5])
grad = jnp.array([0.3, -1.2, 2.0])

# today (argminax)
optimiser = ivon(
    0.1, beta1=0.9, beta2=0.99, weight_decay=0.5, clip_radius=1.0, ess=50.0
)
state = optimiser.init(theta)
updates, new_state = optimiser.update(grad, state, theta)

by_hand = ivon_step_by_hand(
    state.momentum,
    state.hess,
    grad,
    theta,
    int(state.count),
    step_size=0.1,
    beta1=0.9,
    beta2=0.99,
    weight_decay=0.5,
    clip_radius=1.0,
    ess=50.0,
)
bool(jnp.allclose(updates, by_hand[0]))  # True
bool(jnp.allclose(new_state.momentum, by_hand[1]))  # True
bool(jnp.allclose(new_state.hess, by_hand[2]))  # True
```

### Newton's method

[`newton`](xref:api#argminax.blr.newton) is the full-rank rule with $\alpha = 1$, the damping $\beta$ as the prior precision, a zero prior mean and the exact Hessian as the curvature. With $\alpha = 1$ the old state drops out of [](#eq-blr-precision), and the mean form [](#eq-blr-mean-form) becomes

$$\gvect{\mu}_\theta^{(i+1)} = \gvect{\mu}_\theta^{(i)} - (\beta \mat{I} + \mat{C})^{-1} (\vect{g} + \beta \gvect{\mu}_\theta^{(i)}),$$ (eq-blr-newton)

a Newton step on the cost plus $\frac{\beta}{2} \Vert \gvect{\theta} \Vert_2^2$, which tends to the classical step as $\beta \to 0$. Away from a minimum the Hessian can be indefinite, so the system is solved without the positive semidefinite tag (by LU, not Cholesky). This is Newton's method inside an optax loop. For a solve that runs to a tolerance we would use [`GaussNewtonCG`](xref:api#argminax.solve.GaussNewtonCG) or an optimistix solver.

### A fixed precision

[`preconditioned_sgd`](xref:api#argminax.blr.preconditioned_sgd) is the mean form [](#eq-blr-mean-form) with the learned precision replaced by an operator $\mat{P} \in \mathbb{R}^{D_\theta \times D_\theta}$ that the caller supplies and that never changes:

$$\gvect{\theta}^{(i+1)} = \gvect{\theta}^{(i)} - \alpha \mat{P}^{-1} \vect{g}.$$ (eq-blr-preconditioned)

It carries no Gaussian (its state is empty), and any prior belongs in the gradient it is given. $\mat{P}$ is a lineax or gaussx operator tagged positive semidefinite, factored once when the transform is built or solved by its structure. [`scale_by_preconditioner`](xref:api#argminax.blr.scale_by_preconditioner) is the same step without the step size, for an optax chain. The guide has the measurements that say when a fixed preconditioner helps.

## Numerical requirements

### Keeping the precision positive

The precision has to stay positive definite, and [](#eq-blr-precision) guarantees that only for a positive semidefinite $\mat{C}$. The damping guards the other cases, in a different way in each family.

| Transform | Safeguard |
| --- | --- |
| `blr_diagonal` | a floor, $\vect{d} \leftarrow \max(\vect{d}, \beta)$ |
| `blr_full_rank` | $\beta \mat{I}$ added at every step |
| `blr_low_rank` | $\beta$ added to $\vect{d}$ at every step, and the negative part of a callable's curvature dropped when it is factored |
| `ivon` | $\bar{\vect{c}} \ge 0$ by construction (a non-negative start and a squared gradient), so $\bar{\vect{c}} + \delta > 0$ |
| `newton` | none, since the solve does not assume a positive definite system |

Added damping accumulates. Setting $\gmat{\Lambda}_\theta^{(i+1)} = \gmat{\Lambda}_\theta^{(i)}$ in [](#eq-blr-full-rank-precision) gives the precision the full-rank transform settles at,

$$\gmat{\Lambda}_\theta = \delta \mat{I} + \mat{C} + \frac{\beta}{\alpha} \mat{I},$$ (eq-blr-damping-shift)

a shift of $\beta / \alpha$ above the undamped value. With the defaults ($\beta = 10^{-6}$, $\alpha = 10^{-2}$) the shift is $10^{-4}$, which is the size of the default prior precision. So we set `damping=0.0` when the curvature is positive semidefinite and we want the unshifted fixed point. Lin et al. (2020) give an update that stays positive definite without damping, which is not implemented here.

### Curvature estimates

| Estimate | How | Properties |
| --- | --- | --- |
| The exact curvature $\nabla^2_{\gvect{\theta}} \mathcal{J}$ | a callable (full rank, low rank, Newton) | $O(D_\theta^2)$ memory. It gives the Laplace approximation at the fixed point and the exact posterior of a quadratic cost. It can be indefinite for a cost that is not convex; the low-rank transform drops the negative part and the full-rank one does not. |
| The squared gradient, $\vect{g} \odot \vect{g}$ or $\vect{g} \vect{g}^\top$ | `"ggn"` | Always positive semidefinite and cheap. It is built from the gradient it is given, so with a minibatch gradient it is an empirical-Fisher-type estimate (Kunstner et al., 2019) and not the curvature. It vanishes at a stationary point, where the precision decays to the prior. |
| A Gauss–Newton matrix | a callable | Positive semidefinite, and equal to the Fisher information for an exponential-family likelihood with its natural parameterisation (Martens, 2020). It needs the Jacobian of the model, which the caller has to build. |
| Zero | `"identity"` | The precision stays at the prior, and the mean converges to the minimiser of the cost plus the prior. |

The rank-one $\vect{g} \vect{g}^\top$ can also be unstable under a nearly flat prior, where the precision is close to singular in every direction the recent gradients do not span. An informative prior or a better curvature estimate fixes it. The guide has measured examples of what the squared gradient does to the variance.

### Floating point and JAX

| Issue | Handling |
| --- | --- |
| Float32 parameters with `jax_enable_x64` on | The state, the updates and the posterior keep the dtype of the parameters, and a gradient in another dtype is cast to it. |
| An ill-conditioned precision | Damping, an informative prior, or a `solver` (a lineax solver or a gaussx strategy such as `gaussx.CGSolver`). |
| Cancellation in the mean | The state holds the natural mean and the precision, and the mean is formed only when it is needed. |
| Tracing | Every `update` is a pure function with no Python control flow on traced values, so it runs under `jax.jit`. The states are `NamedTuple` pytrees, so they can be carried through `jax.lax.scan`. |
| Differentiating through a step | The truncation of the low-rank factor and the factoring of a callable's curvature have their own derivative rules, because JAX's rules for the SVD and the eigendecomposition are NaN at repeated singular values or eigenvalues. |
| Pytrees | The diagonal and IVON states mirror the parameter pytree. The full-rank, low-rank, Newton and preconditioned transforms need one flat vector, from `jax.flatten_util.ravel_pytree`. |

## The same rule, other names

Each row is the rule with one choice of family, curvature, step size and prior. Khan and Rue (2023, Table 1) give more.

| Algorithm | Family | Curvature | Step size | Prior | Here |
| --- | --- | --- | --- | --- | --- |
| Gradient descent with weight decay | diagonal, fixed at $\delta$ | zero | $\alpha$ (a step of $\alpha / \delta$) | $\mathcal{N}(\mu_0, \delta^{-1})$ | `blr_diagonal(curvature="identity")` |
| RMSprop-like scaling, without the square root | diagonal | squared gradient | about $1 - \rho_2$ | weak | `blr_diagonal` |
| Vprop (Khan et al., 2018) | diagonal | squared gradient, at a draw from $q$ | $\alpha$ | $\mathcal{N}(0, \delta^{-1})$ | `blr_diagonal`, with the gradient taken at a draw |
| Newton's method | full rank | exact | 1 | flat ($\beta \to 0$) | `newton` |
| Preconditioned gradient descent | any, fixed | supplied by the caller | $\alpha$ | in the gradient | `preconditioned_sgd` |
| The Laplace approximation | full rank | exact, at the mean | the fixed point | given | `blr_full_rank` with an exact callable |
| A conjugate update, the Kalman measurement step | full rank | exact | 1 | isotropic | `blr_full_rank(1.0)` with an exact callable |
| SLANG | low rank | a factor of the curvature (the gradient, for the built-in estimate) | $\alpha$ | $\mathcal{N}(\vect{0}, \delta^{-1} \mat{I})$ | `blr_low_rank` |
| IVON | diagonal | squared gradient, in this implementation | $1 - \rho_2$ | weight decay | `ivon` |
| Online and continual learning | any | any | $\alpha$ | the previous posterior | not expressible yet, see below |

Two of the rows are worth working through, because they show how little is left of the rule in each case.

**Gradient descent.** With zero curvature the precision stays at $\delta$, and [](#eq-blr-mean-form) gives $\gvect{\mu}_\theta^{(i+1)} = \gvect{\mu}_\theta^{(i)} - (\alpha / \delta) (\vect{g} + \delta (\gvect{\mu}_\theta^{(i)} - \mu_0))$. This is gradient descent on the cost with a step of $\alpha / \delta$ and weight decay towards $\mu_0$.

```{code-block} python
:caption: The diagonal rule with zero curvature is gradient descent with weight decay.
:label: code-blr-gradient-descent
import jax
import jax.numpy as jnp

from argminax.blr import blr_diagonal

jax.config.update("jax_enable_x64", True)

# Shapes: D_θ = 3 parameters
theta = jnp.array([1.0, -2.0, 0.5])
grad = jnp.array([0.3, -1.2, 2.0])

# today (argminax)
optimiser = blr_diagonal(0.1, prior_precision=2.0, prior_mean=0.5, curvature="identity")
state = optimiser.init(theta)
updates, state = optimiser.update(grad, state)

# Δθ = −(α / δ) (g + δ (θ − μ₀))
gradient_step = -(0.1 / 2.0) * (grad + 2.0 * (theta - 0.5))  # (D_θ,) → (D_θ,)
bool(jnp.allclose(updates, gradient_step))  # True
bool(jnp.allclose(state.precision, 2.0))  # True
```

**The Kalman measurement update.** Take a linear-Gaussian observation $\vect{y} = \mat{H} \gvect{\theta} + \gvect{\varepsilon}$, with $\vect{y} \in \mathbb{R}^{D_y}$, $\mat{H} \in \mathbb{R}^{D_y \times D_\theta}$ and noise $\gvect{\varepsilon} \sim \mathcal{N}(\vect{0}, \gmat{\Lambda}_\varepsilon^{-1})$. The cost is $\mathcal{J}(\gvect{\theta}) = \frac{1}{2} \Vert \vect{y} - \mat{H} \gvect{\theta} \Vert^2_{\gmat{\Lambda}_\varepsilon}$, its gradient is $\vect{g} = -\mat{H}^\top \gmat{\Lambda}_\varepsilon (\vect{y} - \mat{H} \gvect{\theta})$ and its curvature is $\mat{C} = \mat{H}^\top \gmat{\Lambda}_\varepsilon \mat{H}$. One step of [](#eq-blr-precision) and [](#eq-blr-natural-mean) with $\alpha = 1$ gives

$$\gmat{\Lambda}_\theta = \delta \mat{I} + \mat{H}^\top \gmat{\Lambda}_\varepsilon \mat{H},$$ (eq-blr-kalman-precision)

$$\gvect{\mu}_\theta = \gmat{\Lambda}_\theta^{-1} (\delta \gvect{\mu}_0 + \mat{H}^\top \gmat{\Lambda}_\varepsilon \vect{y}),$$ (eq-blr-kalman)

which is the Kalman measurement update in information form, from any start. [](#code-blr-kalman) checks it against the same update written with the gain.

```{code-block} python
:caption: One full-rank step with the exact curvature and $\alpha = 1$ is the Kalman measurement update [](#eq-blr-kalman).
:label: code-blr-kalman
import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx

from argminax.blr import blr_full_rank, blr_posterior

jax.config.update("jax_enable_x64", True)

# Shapes: D_θ = 3 parameters; D_y = 5 observations
key_operator, key_noise = jax.random.split(jax.random.key(0))
obs_operator = jax.random.normal(key_operator, (5, 3))  # H, (D_y, D_θ)
noise_std = 0.5
truth = jnp.array([1.0, -2.0, 0.5])
observations = obs_operator @ truth + noise_std * jax.random.normal(key_noise, (5,))
prior_mean = jnp.array([0.5, 0.0, -0.5])
prior_precision = 2.0


# user
def cost(theta):
    # 𝒥(θ) = ½ ‖y − H θ‖² / σ²
    residual = observations - obs_operator @ theta
    return 0.5 * jnp.sum(residual**2) / noise_std**2


# C = Hᵀ Λ_ε H, with Λ_ε = σ⁻² I
curvature = einx.dot(  # → (D_θ, D_θ)
    "obs row, obs column -> row column", obs_operator, obs_operator
)
curvature = curvature / noise_std**2

# today (argminax)
# one step with α = 1 and β = 0, from any start
optimiser = blr_full_rank(
    1.0,
    prior_precision=prior_precision,
    prior_mean=prior_mean,
    curvature=lambda mean, grad: curvature,
    damping=0.0,
)
start = jnp.array([10.0, 10.0, 10.0])
state = optimiser.init(start)
_, state = optimiser.update(jax.grad(cost)(start), state)
posterior = blr_posterior(state)

# the same update in covariance form, through the gain
# H Λ_θ⁻¹ Hᵀ + Λ_ε⁻¹, the covariance of y under the prior Λ_θ = δ I
obs_cov = einx.dot("row state, column state -> row column", obs_operator, obs_operator)
obs_cov = obs_cov / prior_precision + noise_std**2 * jnp.eye(5)  # → (D_y, D_y)
# today (gaussx)
# (H Λ_θ⁻¹ Hᵀ + Λ_ε⁻¹)⁻¹ (y − H μ₀)
obs_cov_operator = lx.MatrixLinearOperator(obs_cov, lx.positive_semidefinite_tag)
weights = gx.solve(obs_cov_operator, observations - obs_operator @ prior_mean)
# Λ_θ⁻¹ Hᵀ (H Λ_θ⁻¹ Hᵀ + Λ_ε⁻¹)⁻¹ (y − H μ₀)
increment = einx.dot("obs state, obs -> state", obs_operator, weights)  # → (D_θ,)
increment = increment / prior_precision
# μ_θ = μ₀ + the increment
kalman_mean = prior_mean + increment

bool(jnp.allclose(posterior.mode, kalman_mean))  # True
# Λ_θ = δ I + Hᵀ Λ_ε H
exact_precision = prior_precision * jnp.eye(3) + curvature  # → (D_θ, D_θ)
bool(jnp.allclose(state.precision, exact_precision))  # True
```

The prior of every transform is isotropic, $\mathcal{N}(\gvect{\mu}_0, \delta^{-1} \mat{I})$, and `blr_diagonal` takes one scalar prior mean for every parameter. So the update above covers a Kalman step only when the predictive covariance is a multiple of the identity, and a posterior cannot be handed on as the next prior. This is why the last row of the table is not expressible yet. Starting a new transform at the old posterior's mode, or carrying the old state on, does not do it: the next updates still pull towards the configured isotropic prior, and the uncertainty learned before decays at the rate $\alpha$.

## Symbols and arguments

| Symbol | Argument or field |
| --- | --- |
| $\alpha$ | `learning_rate`, a float or an optax schedule of the step count |
| $\delta$ | `prior_precision`; `weight_decay` in `ivon`; `damping` in `newton` |
| $\gvect{\mu}_0$ | `prior_mean`: a scalar in `blr_diagonal`, a vector in `blr_full_rank` and `blr_low_rank`, fixed at zero in `ivon` and `newton` |
| $\mat{C}$, or its factor $\mat{G}$ | `curvature`: `"ggn"`, `"identity"` or a callable `curvature(mean, grad)` |
| $\beta$ | `damping` |
| $R$ | `rank` |
| $\vect{g}$ | the gradient of the cost passed to `update` |
| $\rho_1$, $\rho_2$, $N_\text{ess}$, $r_\text{clip}$ | `beta1`, `beta2`, `ess`, `clip_radius` in `ivon` |
| $\mat{P}$ | `operator` in `preconditioned_sgd` and `scale_by_preconditioner` |
| $\vect{d}$ | `BLRDiagonalState.precision`, `BLRLowRankState.diag_precision` |
| $\gmat{\Lambda}_\theta$ | `BLRFullRankState.precision` |
| $\gmat{\Lambda}_\theta \gvect{\mu}_\theta$ | the `nat_mean` field of the three BLR states |
| $\mat{U}$ | `BLRLowRankState.low_rank_factor` |
| $\bar{\vect{g}}$, $\bar{\vect{c}}$ | `IVONState.momentum`, `IVONState.hess` (which starts at `hess_init`) |
| $i$ | the `count` field of every state |
| $\mathcal{N}(\gvect{\mu}_\theta, \gmat{\Lambda}_\theta^{-1})$ | `blr_posterior(state)`, a [`Laplace`](xref:api#argminax.curvature.Laplace) with `mode` and `precision` |

## Where the rule is used

**Regularised estimation.** With `curvature="identity"` the prior becomes L2 regularisation: `prior_precision=1e-2` is a prior of $\mathcal{N}(\vect{0}, 100 \mat{I})$, or a weight decay of 0.01.

**Bayesian neural networks.** `blr_diagonal` or `ivon` on the parameter pytree, with a draw of the weights before each gradient if the expectations are to be taken by Monte Carlo.

**Retrievals and data assimilation.** A full-rank step with $\alpha = 1$ and the exact curvature is the measurement update [](#eq-blr-kalman), and its state reads out as a `Laplace`, so the averaging kernel and the degrees of freedom of [`argminax.diagnose`](xref:api#argminax.diagnose.degrees_of_freedom) work on it directly. Chaining such steps, with each posterior as the next prior, needs a prior with a full precision, which the transforms do not take yet.

**Continual learning.** The posterior after one task is the prior for the next, which is the rule's form of elastic weight consolidation (Kirkpatrick et al., 2017). It needs the same missing piece, a prior that is not isotropic.

## References

- Khan and Rue (2023), "The Bayesian Learning Rule", Journal of Machine Learning Research 24(281), arXiv:2107.04562.
- Khan, Nielsen, Tangkaratt, Lin, Gal and Srivastava (2018), "Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam", ICML, arXiv:1806.04854.
- Mishkin, Kunstner, Nielsen, Schmidt and Khan (2018), "SLANG: Fast Structured Covariance Approximations for Bayesian Deep Learning with Natural Gradient", NeurIPS, arXiv:1811.04504.
- Shen et al. (2024), "Variational Learning is Effective for Large Deep Networks", ICML, arXiv:2402.17641.
- Lin, Schmidt and Khan (2020), "Handling the Positive-Definite Constraint in the Bayesian Learning Rule", ICML, arXiv:2002.10060.
- Kunstner, Balles and Hennig (2019), "Limitations of the Empirical Fisher Approximation for Natural Gradient Descent", NeurIPS, arXiv:1905.12558.
- Martens (2020), "New Insights and Perspectives on the Natural Gradient Method", Journal of Machine Learning Research 21(146), arXiv:1412.1193.
- Opper and Archambeau (2009), "The Variational Gaussian Approximation Revisited", Neural Computation 21(3).
- Kirkpatrick et al. (2017), "Overcoming Catastrophic Forgetting in Neural Networks", Proceedings of the National Academy of Sciences 114(13), arXiv:1612.00796.

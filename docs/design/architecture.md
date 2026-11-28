# Architecture

argminax is one package, `src/argminax/`. The logic lives in `argminax._src.<module>`, and each area has one public module (`argminax.curvature`, `argminax.blr`, and so on) that only re-exports it. This page says how the areas sit on top of each other and on their foundations, what a Bayesian learning rule transform promises, where the code of optax_bayes went when it moved here, and which checks keep all of it honest. It describes argminax 0.0.7; the maths of the transforms is on the [Bayesian learning rule](../math/bayesian_learning_rule.md) page.

## The layers

The figure below lists what each area imports, and its arrows read "imports". Dependencies point one way, so nothing at the bottom knows about anything above it.

```{figure} https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_imports/argminax_imports-light.svg
:alt: argminax's five public modules and what each imports: diagnose (gaussx, lineax, and matfree through an extra), solve (optimistix, gaussx and lineax) and blr (gaussx, lineax, and optax through an extra) each import curvature, which imports gaussx and lineax; adjoint imports only optimistix
:width: 100%
:label: fig-imports

Each public module, what it imports (dashed when through an extra), and the imports between them, read from the source. The schematic is kept in [geostack](https://github.com/jejjohnson/geostack/tree/main/docs/images/libraries/argminax_imports).
```

`curvature` is the bottom of the package: it owns the [`Laplace`](xref:api#argminax.curvature.Laplace) object, a mode and a precision operator, which every other area either builds or reads. `diagnose`, `solve` and `blr` import it, `adjoint` imports nothing else from the package, and no module imports `diagnose` or `blr`. The division of labour with gaussx is that one step's algebra is gaussx's and the loop is ours. A single Newton or Bayesian learning rule step, the structured solves and the log-determinants come from gaussx, and argminax iterates them and reads the curvature at the end.

Each foundation is extended, not wrapped. A solver here is an optimistix solver and an adjoint is an optimistix adjoint, so `optx.minimise` and `optx.least_squares` keep working. A transform in `argminax.blr` is an optax `GradientTransformation`, so it composes in `optax.chain` like any other.

```{figure} https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_roles/argminax_roles-light.svg
:alt: argminax on the GeoStack role ladder: algorithms, the Laplace model, strategies for solving, stepping and differentiating, no primitives of its own (one step is gaussx's), and the Hessian as a gaussx structure, with optimistix as the floor its solvers and adjoints extend
:width: 100%
:label: fig-role-ladder

The same package by role rather than by import: each public name on its rung of the GeoStack role ladder. The schematic is kept in [geostack](https://github.com/jejjohnson/geostack/tree/main/docs/images/libraries/argminax_roles).
```

## The three layers of the Bayesian learning rule

optax_bayes was organised in three layers (primitives, transforms, wrappers). The same three are still visible inside `argminax._src.blr`, although most of the bottom one now belongs to gaussx.

| Layer | What it holds | Where it lives |
| --- | --- | --- |
| 0, primitives | one step's algebra, the structured solves | gaussx: `blr_diag_update`, `solve`, `low_rank_plus_diag` |
| 0, primitives | the curvature estimates and their factors; the optax import, the step size and the option checks | `_curvature.py`, `_optax.py` |
| 1, transforms | `blr_diagonal`, `blr_full_rank`, `blr_low_rank`, `ivon` and their states | `_diagonal.py`, `_full_rank.py`, `_low_rank.py`, `_ivon.py` |
| 2, presets and readers | `newton`, `preconditioned_sgd`, `scale_by_preconditioner`, `blr_posterior` | `_full_rank.py`, `_preconditioned.py`, `_posterior.py` |

Layer 0 is stateless maths on arrays. Layer 1 turns it into `init` and `update` pairs. Layer 2 holds what is a special case of a transform (Newton's method is the full-rank rule with a step size of one, and a fixed preconditioner is the mean form with the precision frozen) and what reads a state back out.

### What a transform promises

- **It is stepped, not solved.** The caller steps a transform on minibatches. It never runs to a tolerance and it reports no optimistix `RESULTS`, which is the line between `argminax.blr` and `argminax.solve`.
- **It takes the gradient of the cost**, as every optax optimiser does, and a curvature estimate is the curvature of the cost (positive semidefinite), never the Hessian of a log-likelihood.
- **Its state is a `NamedTuple` of arrays** in natural parameters (the precision and the natural mean), as optax states are. So it is a pytree: it can be carried through `jax.lax.scan` and it checkpoints like any other optax state. This is the one exception to the rule that results in argminax are `eqx.Module`s.
- **Its update is the change of the mean**, so after `optax.apply_updates` the parameters equal the mean of the Gaussian, and `init` starts the mean at the parameters it is given.
- **Its Gaussian reads out as a `Laplace`.** [`blr_posterior`](xref:api#argminax.blr.blr_posterior) returns the mode and a precision operator that keeps the structure of the state (diagonal, dense, or diagonal plus low rank), and the covariance is never formed. Sampling, the marginal variances and every diagnostic in `argminax.diagnose` then work on it.
- **optax is imported when a transform is built.** `import argminax.blr` works without the `optax` extra, and calling a transform without it raises `MissingExtraError` naming `argminax[optax]`.

### Where optax_bayes went

The move changed more than the import path. The table lists every source file of optax_bayes 0.0.3 and what stands in its place.

| optax_bayes file | Held | In argminax |
| --- | --- | --- |
| `primitives.py` | `blr_diag_update_step` | `gaussx.blr_diag_update`, called on the log-joint |
| `conversions.py` | natural and mean parameters of a diagonal Gaussian | not ported; the states hold natural parameters and the transforms form the mean inline |
| `hessians.py` | `ggn_diag`, `ggn_outer`, low-rank factors | `_curvature.py`: `resolve_curvature`, `resolve_curvature_factor`, `psd_factor` |
| `linalg.py`, `_optional.py` | PSD-tagged operators and solves, the lazy gaussx import | dropped; gaussx is a hard dependency and is called directly |
| `types.py` | the four state `NamedTuple`s | each state sits next to its transform |
| `diagonal.py`, `full_rank.py`, `low_rank.py`, `ivon.py` | the transforms | `_diagonal.py`, `_full_rank.py`, `_low_rank.py`, `_ivon.py` |
| `wrappers.py` | `*_for_loss`, which negated loss gradients | dropped; every transform takes the gradient of the cost |
| `schedules.py` | `blr_with_schedule` | dropped; `learning_rate` is a float or an optax schedule in every transform |
| `presets.py` | `newton`, `newton_for_loss` | `newton` in `_full_rank.py` |
| `posterior.py`, `sampling.py` | `get_posterior_*`, `sample_posterior_*`, `sample_ivon` | `blr_posterior`, then `Laplace.sample` |

`preconditioned_sgd` and `scale_by_preconditioner` are new here and have no ancestor in optax_bayes.

## gaussx integration

The transforms use gaussx for every solve, and never invert a precision.

| Operation | gaussx call | Notes |
| --- | --- | --- |
| The diagonal step | `gaussx.blr_diag_update(..., convention="precision")` | on the log-joint, with the prior folded into the gradient and the Hessian diagonal |
| The dense mean $\gvect{\mu}_\theta = \gmat{\Lambda}_\theta^{-1} (\gmat{\Lambda}_\theta \gvect{\mu}_\theta)$ | `gaussx.solve` on a `lineax.MatrixLinearOperator` tagged positive semidefinite | Cholesky; `newton` leaves the tag off, since its Hessian may be indefinite |
| The low-rank precision $\op{diag}(\vect{d}) + \mat{U} \mat{U}^\top$ | `gaussx.low_rank_plus_diag(d, U, psd=True)` | a low-rank-update operator |
| The low-rank solve | `gaussx.solve` | the Woodbury identity, $O(D_\theta R^2)$ |
| The posterior | a `Laplace` with a `gaussx.DenseSolver` strategy | the precision operator keeps its structure and the covariance is never formed |

Here $\gvect{\mu}_\theta \in \mathbb{R}^{D_\theta}$ is the mean and $\gmat{\Lambda}_\theta$ the $D_\theta \times D_\theta$ precision of the Gaussian over the parameters, $\vect{d} \in \mathbb{R}^{D_\theta}$ a positive diagonal and $\mat{U} \in \mathbb{R}^{D_\theta \times R}$ a factor of rank $R$.

The full-rank step is the one place where the algebra is written in argminax and not delegated. `gaussx.blr_full_update` solves for the mean without returning it, and it always tags the precision positive semidefinite, which `newton` cannot promise. So `_full_rank._transform` writes the two lines itself, and a test pins them to `gaussx.blr_full_update`.

Every `solver=` argument takes a lineax solver or a gaussx solve strategy such as `gaussx.DenseSolver()` or `gaussx.CGSolver()`.

The other areas lean on gaussx in the same way. The calls below are the ones in the source today.

| Area | gaussx calls |
| --- | --- |
| `curvature` | `gauss_newton_operator`, `logdet`, `inv`, `cholesky`, `diag_inv`, `symmetrize`, `randomized_eigh`, `LowRankUpdate`, and the `DenseSolver`, `CGSolver` and `LineaxSolver` strategies |
| `diagnose` | `solve`, `logdet`, `trace` |
| `solve` | `newton_update`, `damped_natural_update`, `laplace_model`, `laplace_result`, `solve`, `logdet`, `inv`, `diag` |
| `blr` | `blr_diag_update`, `solve`, `low_rank_plus_diag`, `DenseSolver` |
| `adjoint` | none; it extends `optx.AbstractAdjoint` only |

## Package layout

```text
src/argminax/
├── __init__.py            # the exceptions and __version__
├── exceptions.py          # ArgminaxError, ValidationError, MissingExtraError
├── _typing.py             # Objective, Residuals
├── adjoint.py             # public: re-exports _src.adjoint
├── blr.py                 # public: re-exports _src.blr
├── curvature.py           # public: re-exports _src.curvature
├── diagnose.py            # public: re-exports _src.diagnose
├── solve.py               # public: re-exports _src.solve
└── _src/
    ├── adjoint/
    │   ├── _kstep.py          # KStep, OneStep
    │   └── _mapping.py        # to_optimistix_adjoint
    ├── blr/
    │   ├── _optax.py          # L0: the optax import, the step size, the option checks
    │   ├── _curvature.py      # L0: curvature names and callables, their factors
    │   ├── _diagonal.py       # L1: blr_diagonal, BLRDiagonalState
    │   ├── _full_rank.py      # L1: blr_full_rank, BLRFullRankState; L2: newton
    │   ├── _low_rank.py       # L1: blr_low_rank, BLRLowRankState
    │   ├── _ivon.py           # L1: ivon, IVONState
    │   ├── _preconditioned.py # L2: preconditioned_sgd, scale_by_preconditioner
    │   └── _posterior.py      # L2: blr_posterior
    ├── curvature/
    │   ├── _laplace.py        # Laplace, gauss_newton, push_forward
    │   └── _matfree.py        # marginal_variance, low_rank, sample_by_perturbation
    ├── diagnose/
    │   ├── _retrieval.py      # averaging_kernel, degrees_of_freedom, information_content
    │   └── _evidence.py       # evidence, optimize_prior_precision
    └── solve/
        ├── _gauss_newton.py   # GaussNewtonCG
        ├── _dual.py           # dual_solve, incremental_dual_solve
        ├── _laplace_mode.py   # laplace_mode and its covariance and site forms
        └── _reparam.py        # Reparametrised
```

## Dependencies

```{figure} https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_ecosystem/argminax_ecosystem-light.svg
:alt: The packages that import argminax, each with the argminax modules it uses, and what argminax builds on
:width: 100%
:label: fig-ecosystem

Who imports argminax today, with the modules each uses, and what argminax builds on. The schematic is kept in [geostack](https://github.com/jejjohnson/geostack/tree/main/docs/images/libraries/argminax_ecosystem).
```

| Package | Required | Role |
| --- | --- | --- |
| `jax`, `jaxlib` | yes | arrays, autodiff, `jit`, `vmap`, `scan` |
| `equinox` | yes | solvers, adjoints, operators and results as modules |
| `lineax` | yes | linear solves, operator types and the structure tags |
| `optimistix` | yes, pinned to `>=0.1.0,<0.2` | `minimise`, `least_squares`, the solver and adjoint protocols, `RESULTS` |
| `gaussx` | yes, pinned to the tag `v0.8.3` | structured operators, solves, log-determinants and one step's algebra |
| `einx` | yes | reductions, contractions and rearranges on dense arrays |
| `jaxtyping` | yes | array annotations with named dimensions |
| `optax` | the `optax` extra | the `GradientTransformation` protocol, schedules and `optax.chain` for `argminax.blr` |
| `matfree` | the `matfree` extra | stochastic Lanczos quadrature for log-determinants of large operators |

optimistix is pinned to one minor version because `KStep.apply` reads the private layout of its solve inputs. gaussx was an optional extra of optax_bayes and is a hard dependency here.

## Quality gates

| Check | Command | Runs on |
| --- | --- | --- |
| Tests and doctests, with the 90 % coverage gate, on Python 3.12 and 3.13 | `uv run pytest -v` | every pull request and push to `main` |
| A bare install imports argminax, and `argminax.blr` without optax | the `bare-install` job of `ci.yml` | every pull request and push to `main` |
| Lint and format, on the whole repository | `ruff check .`, `ruff format --check .` | every pull request and push to `main` |
| The lockfile matches `pyproject.toml` | `uv lock --check` | every pull request and push to `main` |
| Types | `ty check src/argminax scripts` | every pull request and push to `main` |
| Both halves of the docs, with every internal link checked | `uv run poe docs` | every pull request and push to `main` |

`uv run poe check` runs the lint, format, type and test checks locally in one go. Commits follow Conventional Commits, and release-please cuts the releases from them.

The Bayesian learning rule came with six gates of its own, from the epic that moved it ([#58](https://github.com/jejjohnson/argminax/issues/58)). Each one is a test in `tests/blr/`, or the bare-install job.

| Gate | Passes when | Test |
| --- | --- | --- |
| Exact recovery | one full-rank step with the exact curvature, a step size of one and no damping gives the closed-form posterior of a linear-Gaussian problem | `test_one_exact_step_recovers_the_linear_gaussian_posterior` |
| One posterior | the `Laplace` read from that state has the precision of `argminax.curvature.gauss_newton` at the same point | `test_exact_fit_has_the_gauss_newton_precision` |
| Diagnostics | the degrees of freedom of that posterior equal the closed form | `test_degrees_of_freedom_of_an_exact_fit_match_the_closed_form` |
| Port | each transform reproduces the iterates of optax_bayes 0.0.3, stored in `tests/blr/data/optax_bayes.npz` | the `reproduces_optax_bayes` tests of each file |
| optax protocol | a transform composes in `optax.chain`, keeps float32 states in float32, and gives the same iterates under `jax.jit` and in a `jax.lax.scan` as eagerly | `test_chains_with_clipping_on_a_nested_pytree` (the diagonal transform only), `test_float32_parameters_stay_float32` and `test_jit_and_scan_agree_with_an_eager_loop` (`test_jit_agrees_and_float32_stays_float32` for IVON) |
| Bare install | without optax, `import argminax.blr` succeeds and a transform raises `MissingExtraError` | the `bare-install` job |

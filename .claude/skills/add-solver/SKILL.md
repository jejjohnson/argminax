---
name: add-solver
description: Add a solver or an adjoint to argminax as an optimistix extension — logic under argminax/_src/solve/ or _src/adjoint/, a re-export from argminax.solve or argminax.adjoint, an optimistix subclass reporting through RESULTS, an eqx.Module with static configuration and __check_init__, an executable docstring, the closed-form test against the linear-Gaussian problem, the SciPy reference for a nonlinear cost, the finite-difference gradient test, and the jit / vmap / grad smoke test. Use when asked to add, port or implement a solver, a Newton loop or a minimiser adjoint in src/argminax.
---

# Add a solver or an adjoint

Read "Boundaries", "What it is built on" and "The contracts" in `AGENTS.md` first; this is the step-by-step. Then read the live issue (`gh issue view <N>`): it names the source being moved in, the reference and the tolerance, and it is newer than the design page.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` (argminax and the upstream sections for optimistix, lineax and gaussx) and the "Reuse before you write" table in `AGENTS.md`. optimistix already has BFGS, Gauss–Newton, Levenberg–Marquardt, Newton root finding and `ImplicitAdjoint`; gaussx already has one Newton step (`newton_update`) and the Gauss–Newton precision (`gauss_newton_operator`). argminax adds only what they leave out.
- Code moving in from another stack library (vardax's adjoints, radtrax's and plumax's `run_gauss_newton`, gaussx's `laplace_mode`) is moved, not rewritten: keep its behaviour and add a parity test against the source at its pinned commit to 1e-12 relative in float64.

## 2. Pick its home

| It is… | Private home | Public module |
|---|---|---|
| A minimiser, a least-squares solver, a dual solve, a mode finder, a reparametrisation | `src/argminax/_src/solve/` | `argminax.solve` |
| An adjoint of a minimiser stopped early (`KStep`, `OneStep`), an adjoint-spec mapping | `src/argminax/_src/adjoint/` | `argminax.adjoint` |

One concept per file. The first feature in a module creates the public module (`src/argminax/<module>.py`, which only re-exports), its `_src/<module>/__init__.py`, its `SUBMODULES` entry in `tests/test_public_api.py` and its docs page.

## 3. Write it

- **Imports** under the fixed aliases: `import equinox as eqx`, `import jax`, `import jax.numpy as jnp`, `import lineax as lx`, `import optimistix as optx`, `import gaussx`, `from jaxtyping import Array, Float, PyTree`. Nothing from pipekit, pipekit-cycle, integrix, vardax, pyrox or a GeoModels library ("Boundaries").
- **An optimistix subclass.** A solver subclasses the optimistix base for its problem (`optx.AbstractMinimiser`, `optx.AbstractLeastSquaresSolver`, `optx.AbstractGaussNewton`, …) and implements its `init`, `step`, `terminate` and `postprocess`; it is used through `optx.minimise` or `optx.least_squares`, never through an entry point of its own alone. An adjoint subclasses `optx.AbstractAdjoint` and works under any optimistix solve. Anything that reads optimistix's private solve inputs says so in a comment and is covered by a test, because the optimistix pin moves only with that test.
- **State in an `eqx.Module`.** `rtol`, `atol`, `norm`, a damping λ, a number of inner CG steps are `eqx.field(static=True)` where they are configuration. Static checks in `__check_init__`, raising `ValidationError` that names the field.
- **It traces.** The loop is optimistix's (or `lax.while_loop`), never a Python `for` with a `break` on a traced condition; no `float()` or `.item()` on a traced value; non-convergence is reported through `RESULTS`, never raised. No `jax.jit` inside library code.
- **Linear algebra** through lineax and gaussx: an inner solve is `lx.linear_solve` with `lx.CG` on a PSD-tagged operator (`gaussx.gauss_newton_operator`), never `jnp.linalg.solve` or `inv`.
- **Precision is the caller's.** Preserve the state's dtype; constants as Python scalars or in the state's dtype.
- **Docstring** in the `python-style` layout: one-line summary (the capability-index entry), the defining update in Unicode (`x ← x − (KᵀSε⁻¹K + Sₐ⁻¹)⁻¹ ∇J`), `Algorithm:` with numbered steps, `Args:` with meaning and symbol, `Returns:`, `Raises:`, two or three `Examples:` whose outputs are shapes, `allclose` booleans or `RESULTS`, and `References:` (Nocedal and Wright 2006, Rodgers 2000, the paper that defines the method).

## 4. Export and document

- Re-export from the public module's sorted `__all__`. Nothing is added to the top-level `argminax.__all__`.
- A **new public module** joins `SUBMODULES` in `tests/test_public_api.py`, and gets `docs/api/<module>.md` with a `::: argminax.<module>` block, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`.
- `uv run poe capabilities`.

## 5. Tests

In `tests/<module>/test_<name>.py`, with x64 on (from `conftest.py`), using the helpers in `tests/_harness.py`:

- **Closed form.** On `linear_gaussian_problem(key, n_state, n_obs)` the minimum is known exactly: the solver reaches `problem.x_hat` to 1e-12 relative from `problem.x_a`, and `solution.result == optx.RESULTS.successful`. A Gauss–Newton method on a linear model converges in one outer step: assert that too.
- **SciPy reference** for a nonlinear cost: `assert_matches_scipy_least_squares(residuals, x0, solution.value, rtol=…)` on a small problem (an exponential or Rosenbrock-type residual), at a tolerance the comment justifies from SciPy's own `xtol` / `ftol`.
- **Gradient through the solve.** First assert the solve converged. Then `assert_grad_matches_fd` on a scalar function of `solution.value` with respect to an argument (the prior state xₐ, a parameter of F); for an adjoint, also compare with `optx.ImplicitAdjoint` at a converged minimum.
- **Transform invariance** (`assert_traces`): eager, `jit` and `vmap` over a batch of problems agree to 1e-12 relative.
- **Properties** with hypothesis: drawn `n_state`, `n_obs` and seed, the closed-form check holds; more states than observations still converges, because the prior regularises.
- **Edges:** a one-dimensional state, an ill-conditioned prior, float32 input keeps float32, `max_steps` reached reports `RESULTS.max_steps_reached` rather than raising, each invalid argument raises `ValidationError` (`match=` on the field name).

## 6. Verify

```bash
uv run pytest --no-cov tests/<module>/ src/argminax/_src/<module>/ tests/test_public_api.py tests/test_capabilities.py -v
uv run poe test-cov
```

then the `pre-pr-check` skill.

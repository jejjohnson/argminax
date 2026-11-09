---
name: add-diagnostic
description: Add a curvature operator or a retrieval diagnostic to argminax — logic under argminax/_src/curvature/ or _src/diagnose/, a re-export from argminax.curvature or argminax.diagnose, a matrix-free gaussx operator with the right PSD tag, the closed-form test against the linear-Gaussian problem (Ŝ, A, d_s, H_info, evidence), the sampling bound for a stochastic estimate, the finite-difference gradient test and the jit / vmap / grad smoke test. Use when asked to add, port or implement a Laplace covariance, Gauss–Newton curvature, low-rank or diagonal form, averaging kernel, degrees of freedom, information content or evidence in src/argminax.
---

# Add a curvature operator or a diagnostic

Read "Boundaries", "What it is built on" and "The contracts" in `AGENTS.md` first; this is the step-by-step. Then read the live issue (`gh issue view <N>`): it names the source being moved in, the closed form and the tolerance, and it is newer than the design page.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` (argminax and the upstream sections for lineax and gaussx) and the "Reuse before you write" table in `AGENTS.md`. gaussx already has the Gauss–Newton precision (`gauss_newton_operator`, `gauss_newton_precision`), structured solves and log-determinants (`logdet`, `SLQLogdet`, …); argminax builds the covariance at the minimum and what it says on top of them.
- Code moving in from another stack library (vardax's `GaussNewtonHessian` and `LaplaceCovariance`, radtrax's `var3d/diagnostics.py`, plumax's `PosteriorCovariance`) is moved, not rewritten: keep its behaviour and add a parity test against the source at its pinned commit to 1e-12 relative in float64.

## 2. Pick its home

| It is… | Private home | Public module |
|---|---|---|
| Curvature at a point, the Laplace covariance Ŝ, its low-rank or diagonal form, perturbation sampling | `src/argminax/_src/curvature/` | `argminax.curvature` |
| The averaging kernel A, degrees of freedom d_s, information content, evidence, the prior-precision search | `src/argminax/_src/diagnose/` | `argminax.diagnose` |

`diagnose` reads the operators `curvature` builds, never the other way round. One concept per file. The first feature in a module creates the public module (`src/argminax/<module>.py`, which only re-exports), its `_src/<module>/__init__.py`, its `SUBMODULES` entry in `tests/test_public_api.py` and its docs page.

## 3. Write it

- **Imports** under the fixed aliases: `import equinox as eqx`, `import jax`, `import jax.numpy as jnp`, `import lineax as lx`, `import gaussx`, `from jaxtyping import Array, Float, PyTree`. `matfree` only inside the function that needs it, raising `MissingExtraError` naming `argminax[matfree]` when it is missing.
- **Matrix-free operators.** Curvature is built from `jax.linearize` and `jax.linear_transpose` of the residuals (or taken from `gaussx.gauss_newton_operator`) and returned as a gaussx or lineax operator; library code never calls `.as_matrix()` or forms a Jacobian densely. Ŝ is an inverse applied by a solve, never `jnp.linalg.inv`.
- **Structure tags are promises.** Tag `lx.positive_semidefinite_tag` (and `symmetric_tag`) only where the operator is PSD by construction: Gauss–Newton and Fisher curvature, Ŝ built from them. A full Hessian is PSD only at a minimum, so it is not tagged unless the caller says so.
- **It traces.** No Python control flow, `float()` or `.item()` on traced values; a stochastic estimate (Hutchinson probes, perturbation samples) takes a keyword-only `key` last, never a host RNG; no `jax.jit` inside library code.
- **Precision is the caller's.** Preserve the state's dtype; constants as Python scalars or in the state's dtype.
- **Docstring** in the `python-style` layout: one-line summary (the capability-index entry), the defining equation in Unicode (`Ŝ = (KᵀSε⁻¹K + Sₐ⁻¹)⁻¹`, `A = I − ŜSₐ⁻¹`, `d_s = tr A`), `Algorithm:` with numbered steps, `Args:` with meaning and symbol, `Returns:`, `Raises:`, two or three `Examples:` whose outputs are shapes or `allclose` booleans, and `References:` (Rodgers 2000, chapter 2, for the retrieval diagnostics).

## 4. Export and document

- Re-export from the public module's sorted `__all__`. Nothing is added to the top-level `argminax.__all__`.
- A **new public module** joins `SUBMODULES` in `tests/test_public_api.py`, and gets `docs/api/<module>.md` with a `::: argminax.<module>` block, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`.
- `uv run poe capabilities`.

## 5. Tests

In `tests/<module>/test_<name>.py`, with x64 on (from `conftest.py`), using the helpers in `tests/_harness.py`:

- **Closed form.** On `linear_gaussian_problem(key, n_state, n_obs)` every quantity is known: the operator's dense form equals `problem.S_hat`, the averaging kernel `problem.A`, the degrees of freedom `problem.d_s`, the information content `problem.H_info`, the evidence `problem.log_evidence`, all to 1e-12 relative in float64. Densify only in the test (`operator.as_matrix()`).
- **Structure.** The tag is right: `lx.is_positive_semidefinite(operator)` where promised, and the dense form's eigenvalues are non-negative on the test problem.
- **Stochastic estimates.** A Hutchinson trace or a perturbation sample is checked with `assert_within_stderr` or `assert_sample_covariance` (bounds from its own sampling distribution, 4σ by default), never a tolerance tuned on one key.
- **Gradient check** (`assert_grad_matches_fd`): a scalar of the diagnostic (d_s, the log evidence) against central differences in a parameter of the problem (the prior variance, a parameter of F).
- **Transform invariance** (`assert_traces`): eager, `jit` and `vmap` over a batch of problems agree to 1e-12 relative.
- **Properties** with hypothesis: drawn `n_state`, `n_obs` and seed; 0 ≤ d_s ≤ min(N, M); the low-rank form converges to the full one as its rank reaches N.
- **Edges:** a one-dimensional state, more states than observations, a near-singular prior, float32 input keeps float32, each invalid argument raises `ValidationError` (`match=` on the field name), and a call without the `matfree` extra raises `MissingExtraError`.

## 6. Verify

```bash
uv run pytest --no-cov tests/<module>/ src/argminax/_src/<module>/ tests/test_public_api.py tests/test_capabilities.py -v
uv run poe test-cov
```

then the `pre-pr-check` skill.

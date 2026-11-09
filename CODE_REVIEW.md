# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this repository. Read "Reuse before you write", "The contracts" and "What enforces them" in [`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report format. Most defects worth finding are a contract broken in a way the tests cannot see (an operator tagged PSD that is not, a gradient trusted through a solve that never converged, Python control flow that breaks under `jit`, a docstring that promises what the code does not do) or a re-implementation of something the package, gaussx or optimistix already has, not style.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import order, `__all__` ordering, doctest output, the coverage gate); review what they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against [`docs/api/capabilities.md`](docs/api/capabilities.md) and the private helpers in "Reuse before you write" in `AGENTS.md`. A re-implemented solve loop, linear solve, curvature product, log-determinant, adjoint or test reference is a **High** finding, with the existing name to use (in argminax, gaussx, optimistix or lineax).
- No new runtime dependency without discussion; an optional one sits behind an extra and an in-function import that names the extra. `uv.lock` is updated with any dependency change.
- Imports keep pointing one way (the module table in `AGENTS.md`), and the layer rule holds: argminax never imports pipekit, pipekit-cycle, integrix, vardax, pyrox or a GeoModels library. Logic lives in `argminax._src`; a public module only re-exports.
- The optimistix pin (`<0.2`) moves only with a re-run of the adjoint tests.
- A rename or removal of a public name keeps the old name working with a `DeprecationWarning` for at least one release.

### 2. The contracts

- **Public functions** validate static inputs once, at the boundary, and raise `ValidationError` whose message names the argument; never mutate an input; preserve the input dtype; run under `jit`, `vmap` and `grad`; never call `jax.jit` themselves.
- **JAX rules**: no Python `if`, `while` or `break` on a traced value; no `float()`, `int()` or `.item()` on one; no host RNG (randomness comes from a keyword-only `key`); no `jnp.linalg.inv`, and no dense solve or materialised Jacobian in library code.
- **Solvers and adjoints** subclass optimistix's abstract solvers and `optx.AbstractAdjoint`; non-convergence is reported through `RESULTS`, never raised; a gradient through a solve is trusted only after `solution.result` is checked.
- **Curvature** is a gaussx (lineax) operator carrying the right structure tag, built matrix-free; it matches its linear-Gaussian closed form to 1e-12 in float64 through `tests/_harness.py`.
- **Exceptions** derive from `ArgminaxError` and the closest stdlib exception; errors are chained (`raise … from err`).

### 3. Public API and documentation

- New names: in the `_src` module, re-exported from the public module's `__all__` (exceptions also from the top-level `__all__`); a new public module in `SUBMODULES` (`tests/test_public_api.py`), with its `docs/api/<module>.md` page, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; `docs/api/capabilities.md` regenerated (`uv run poe capabilities`).
- Docstrings in the `python-style` layout: summary, the defining equation in Unicode (`eman-style-math`; `$…$` in a docstring is a finding), `Algorithm:` for a method, `Args:` with meaning and symbol, `Returns:` and `Raises:` matching the code, and two or three executable `Examples:` (typical, edge, composition) whose output is what the code prints (shapes, `allclose` booleans or reprs, not raw floats).
- Maths in MyST and MkDocs pages uses the `eman-style-math` macros, defines every symbol with its shape, keeps each expression on one line and has no spacing macros (`\,`, `\quad`, …).
- Prose pages: MyST directives, not MkDocs-Material syntax; API links by the public name (`xref:api#argminax.<Name>`); a new page or notebook in the `docs/myst.yml` toc with a basename unique across `docs/`.
- Notebooks committed executed, with no error outputs, and every number quoted in the prose matching the printed output.

### 4. Tests

- Each behaviour is checked against the linear-Gaussian closed form, a SciPy reference or finite differences through `tests/_harness.py`, plus a `jit` / `vmap` / `grad` smoke test; edge cases (a single state, more states than observations, an ill-conditioned prior, invalid arguments) have their own tests.
- New code is covered (the 90 % gate is global, so a well-covered package can hide an untested module: look at the diff, not the total).
- Randomness comes from a fixed key; a statistical assertion is bounded by the statistic's own sampling distribution (`assert_within_stderr`, `assert_sample_covariance`), with the bound's source in a comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for end-to-end workflows and optional dependencies.

### 5. Modern Python and dependencies

- `from __future__ import annotations`; type hints on every public function; `X | None`, built-in generics, PEP 695 `type` aliases.
- `eqx.Module` for solvers, operators and results, with configuration as `eqx.field(static=True)`; `Enum` / `StrEnum` for fixed sets of options; `pathlib`; f-strings; context managers for resources.
- Specific exceptions, chained with `raise … from …`; guard clauses over deep nesting.
- `python-style` comments and naming: comments above the line, never beside it; a Unicode equation above each mathematical line; descriptive names, with the symbol only in the comment; one operation per line. Its JAX rules too (jaxtyping dims, einx, lineax, `eqx.error_if`, `key` keyword-only and last, the four tests per public forward).
- New scripts use `cyclopts` and `loguru`, not argparse, click, typer, `print` progress or stdlib `logging`; argminax has no CLI.
- A dependency, when one is justified, follows the preferences in `.github/instructions/python-coding.instructions.md`.

### 6. Security

- No secrets or credentials in code, tests, notebooks or docs.
- `subprocess` without `shell=True`; file paths taken from users are not joined into places they could escape.

---

## Package-Specific Checks

Every example here runs, with `jax_enable_x64` on.

### No Python control flow on a traced value

```python
import jax
import jax.numpy as jnp
from jax import lax


def gradient(x):
    return 2.0 * x


# ❌ `break` on a traced norm: fine eagerly, a TracerBoolConversionError under jit
def descend_eagerly(x, tolerance=1e-8, max_steps=100):
    for _ in range(max_steps):
        if jnp.linalg.norm(gradient(x)) < tolerance:
            break
        x = x - 0.25 * gradient(x)
    return x


# ✅ The stopping rule is the loop's condition, so jit, vmap and grad trace it
def descend(x, tolerance=1e-8, max_steps=100):
    def keep_going(carry):
        step, state = carry
        return (step < max_steps) & (jnp.linalg.norm(gradient(state)) >= tolerance)

    def take_step(carry):
        step, state = carry
        return step + 1, state - 0.25 * gradient(state)

    _, x = lax.while_loop(keep_going, take_step, (0, x))
    return x


assert jnp.allclose(jax.jit(descend)(jnp.ones(3)), 0.0, atol=1e-8)
```

### No `float()` on a traced value, and no host RNG

```python
import jax
import jax.numpy as jnp
import numpy as np


# ❌ float() fails under jit, and the NumPy probes are drawn once, at trace
#    time, so every call of the jitted function reuses the same probes
def hutchinson_trace_on_host(matvec, size, n_probes=64):
    probes = np.random.choice([-1.0, 1.0], size=(n_probes, size))
    return float(jnp.mean(jnp.sum(probes * jax.vmap(matvec)(probes), axis=1)))


# ✅ The probes come from a key, and the estimate stays an array
def hutchinson_trace(matvec, size, n_probes=64, *, key):
    probes = jax.random.rademacher(key, (n_probes, size), dtype=jnp.float64)
    return jnp.mean(jnp.sum(probes * jax.vmap(matvec)(probes), axis=1))


diagonal = jnp.arange(1.0, 5.0)
estimate = hutchinson_trace(lambda v: diagonal * v, 4, key=jax.random.key(0))
# A Rademacher probe gives the exact trace of a diagonal operator
assert jnp.allclose(estimate, diagonal.sum())
```

### A PSD tag is a promise

```python
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx


def residuals(x):
    return jnp.sin(x) - 0.5


def cost(x):
    return 0.5 * jnp.sum(residuals(x) ** 2)


x = jnp.array([2.0, 2.5, 3.0])

# ❌ Away from the minimum the full Hessian of J is indefinite, so the tag
#    lies and CG on this operator can diverge or return garbage
hessian = jax.hessian(cost)(x)
wrong = lx.MatrixLinearOperator(hessian, lx.positive_semidefinite_tag)

# ✅ The Gauss–Newton curvature JᵀJ is PSD by construction, and gaussx builds
#    it matrix-free with the tag already on
curvature = gaussx.gauss_newton_operator(residuals, x)
assert lx.is_positive_semidefinite(curvature)
assert jnp.all(jnp.linalg.eigvalsh(curvature.as_matrix()) >= 0.0)
assert jnp.any(jnp.linalg.eigvalsh(hessian) < 0.0)
```

### Check convergence before trusting a gradient through a solve

```python
import jax
import jax.numpy as jnp
import optimistix as optx


def cost(x, target):
    return jnp.sum((x - target) ** 2) + 0.1 * jnp.sum(x**4)


solver = optx.BFGS(rtol=1e-10, atol=1e-10)


# ❌ throw=False and no look at the result: if the solve stopped at
#    max_steps, the implicit gradient is the gradient of a point that is not
#    a minimum, and nothing says so
def retrieve_blindly(target):
    solution = optx.minimise(cost, solver, jnp.zeros(2), target, throw=False)
    return solution.value


# ✅ Return the result with the value, and assert on it in the test
def retrieve(target):
    solution = optx.minimise(cost, solver, jnp.zeros(2), target, throw=False)
    return solution.value, solution.result


target = jnp.array([1.0, -1.0])
_, result = retrieve(target)
assert result == optx.RESULTS.successful
sensitivity = jax.jacfwd(lambda t: retrieve(t)[0])(target)
assert sensitivity.shape == (2, 2)
```

### Precision is the caller's

```python
import jax.numpy as jnp
import numpy as np


# ❌ A float64 NumPy constant promotes a float32 state to float64 under x64
def damp_and_promote(step):
    return step * np.float64(0.5)


# ✅ A Python scalar takes the dtype of the array it meets
def damp(step):
    return step * 0.5


step = jnp.ones(3, dtype=jnp.float32)
assert damp_and_promote(step).dtype == jnp.float64
assert damp(step).dtype == jnp.float32
```

### Typed errors that name the argument, on static information only

```python
from argminax import ValidationError


# ❌ A RuntimeError that names neither the argument nor the value
def check_max_steps_vaguely(max_steps: int) -> int:
    if max_steps <= 0:
        raise RuntimeError("bad input")
    return max_steps


# ✅ The package's error (also a ValueError), the argument's name, the value.
#    A traced value is never checked this way: use eqx.error_if, or report
#    it through optimistix's RESULTS.
def check_max_steps(max_steps: int) -> int:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        msg = f"'max_steps' must be an integer, got {max_steps!r}"
        raise ValidationError(msg)
    if max_steps <= 0:
        msg = f"'max_steps' must be a positive integer, got {max_steps}"
        raise ValidationError(msg)
    return max_steps


assert check_max_steps(10) == 10
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step

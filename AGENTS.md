# AGENTS.md

Standing instructions for **every** coding agent working in this repository (Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth: `CLAUDE.md` and `.github/copilot-instructions.md` point here.

The house style (Python, maths notation, prose, pseudocode, reports) comes from the maintainer's personal skills in [jejagents](https://github.com/jejjohnson/jejagents); "House style" and "Personal skills" below say which skill covers what and carry the essentials, so an agent without those skills installed can still follow them.

## What this repo is

argminax is the GeoStack's optimisation library with uncertainty at the minimum. It finds x̂ = argmin J(x), then says how sharp that minimum is (the posterior covariance Ŝ = (KᵀSε⁻¹K + Sₐ⁻¹)⁻¹, the averaging kernel A = I − ŜSₐ⁻¹, the degrees of freedom for signal) and how x̂ depends on what went in. It extends optimistix as gaussx extends lineax: its solvers are optimistix solvers and its adjoints are optimistix adjoints, so `optx.minimise` and `optx.least_squares` keep working.

The design lives in the geostack repository, in [`docs/design/libraries/argminax.md`](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/argminax.md) and [`argminax_details.md`](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/argminax_details.md). The backlog is five wave epics, [#8](https://github.com/jejjohnson/argminax/issues/8) to [#12](https://github.com/jejjohnson/argminax/issues/12), under [geostack#134](https://github.com/jejjohnson/geostack/issues/134). The issues are newer than the design page where the two disagree, so read the live issue before starting work.

People build on this package, so **the package is a library of primitives**: new code composes what is here, and anything genuinely new lands where the next person will find and reuse it.

It is one package, `src/argminax/`. Logic lives in `argminax._src.<module>`, and each area has one public module (`src/argminax/<module>.py`) that only re-exports it. The top-level `argminax.__all__` holds the exceptions and `__version__`; everything else is reached through its area module (`argminax.curvature`, `argminax.solve`, …). Imports only point down this table, and each area module appears with the first feature issue that needs it:

| Module | Owns | Lands in |
|---|---|---|
| `_typing.py` | PEP 695 aliases for optimistix's `fn(y, args)` convention: `Objective`, `Residuals` | #13 |
| `exceptions.py` | `ArgminaxError` and its subclasses `ValidationError` (also a `ValueError`) and `MissingExtraError` (also an `ImportError`) | #13 |
| `adjoint` | `KStep` and `OneStep` adjoints for a minimiser stopped early; `to_optimistix_adjoint` | Wave 1 (#15, #16) |
| `curvature` | Gauss–Newton curvature and the Laplace covariance Ŝ as gaussx operators; their low-rank and diagonal forms; perturbation sampling | Wave 1 (#17, #18) |
| `diagnose` | the averaging kernel, degrees of freedom for signal, information content, evidence at the minimum, the prior-precision search | Wave 2 (#20, #21) |
| `solve` | `GaussNewtonCG` (Gauss–Newton and Levenberg–Marquardt with an inner CG), `dual_solve`, `laplace_mode`, positivity and bounds by reparametrisation | Wave 3 (#23 to #26) |

`curvature` sits below `diagnose`, which reads the operators it builds; `adjoint` and `solve` sit beside `curvature`, and none of the four imports `diagnose`.

### Boundaries

- **Core tier 1.** argminax imports JAX, Equinox, lineax, optimistix, jaxtyping and gaussx (tier 0), and extends optimistix. It never imports pipekit, pipekit-cycle, integrix, vardax, pyrox or any GeoModels library (radtrax, plumax, somax, …): those import argminax. `to_optimistix_adjoint` reads pipekit-cycle's adjoint specs by class name for exactly this reason (#16).
- **One step is gaussx's; the loop is argminax's.** A single Newton or Gauss–Newton step's algebra (`newton_update`, `gauss_newton_precision`), structured solves and log-determinants come from gaussx; argminax iterates them to a tolerance and reads the curvature at the end.
- **The caller owns the cost.** A consumer passes J or its residuals r in, in optimistix's `fn(y, args)` form; argminax never builds a forward model, an observation operator or a covariance from data.
- **optimistix is pinned to one minor version** (`optimistix>=0.1.0,<0.2`), because `KStep.apply` reads the private layout of optimistix's solve inputs. Raising the pin means re-running the adjoint tests against the new layout.
- **Optional extras are imported lazily.** `matfree` (stochastic Lanczos quadrature) is the `argminax[matfree]` extra; a function that needs it imports it when called and raises `MissingExtraError` naming the extra.

Dependencies point one way. The consumers are vardax, radtrax, plumax, pyrox-gp, pyrox-lgm, pyrox-nn and gaussx (whose `laplace_mode` moves here); their switch-over issues sit under [#12](https://github.com/jejjohnson/argminax/issues/12). Keep their imports working: a rename or removal of a public name goes through a deprecation (the old name keeps working with a `DeprecationWarning` that names the replacement for at least one release).

## What it is built on

Each foundation brings a rule. Breaking one runs fine on your example and fails in someone else's pipeline or in CI.

| Foundation | The package uses it for | The rule it brings |
|---|---|---|
| **JAX** | every solve, curvature product and diagnostic | Traceable: static shapes, no Python control flow on traced values (no `if`, `while` or `break` on a traced condition; use `lax.while_loop`, `lax.cond`, `jnp.where`), no `float()` / `int()` / `.item()` on a traced value, no host RNG (`numpy.random`, `random`); randomness comes from a `key` argument. `jit`, `vmap` and `grad` agree with an eager call. |
| **Equinox** | solvers, adjoints, operators and results as `eqx.Module`s | Arrays are always leaves; configuration is `eqx.field(static=True)`. Field checks in `__check_init__`, checks on traced values with `eqx.error_if`. |
| **lineax** | linear solves, and the PSD tags on every curvature operator | Solve, never invert: no `jnp.linalg.inv`, and no dense `jnp.linalg.solve` in library code. A Gauss–Newton or Laplace operator carries `lx.positive_semidefinite_tag` (or `symmetric_tag`), and nothing claims PSD that is not. |
| **optimistix** | `minimise`, `least_squares`, `root_find`, the solver and adjoint protocols, `RESULTS` | argminax's solvers subclass optimistix's abstract solvers and its adjoints subclass `optx.AbstractAdjoint`; a non-converged solve is reported through `RESULTS`, never an exception. Check `solution.result` before trusting a gradient through the solve. |
| **gaussx** | structured operators, solves, log-determinants and one step's algebra | Curvature is returned as a gaussx operator and never formed densely in library code. |
| **jaxtyping** | array annotations | Named dims from the shared vocabulary ("House style"). |
| **SciPy, hypothesis** (test group) | references for the nonlinear solvers; drawn problem sizes and seeds | Test oracles, not runtime dependencies: library code never imports them, and a bare install imports argminax without the test group (SciPy is installed anyway, as a dependency of JAX). |
| **pytest + `--doctest-modules`** | the suite and every docstring `Examples:` block | An example's expected output is what the code prints; a stale example is a failing test. |

## Reuse before you write

Before writing a helper, a solver, an operator or a diagnostic, find out whether it exists:

1. **Search the capability index.** [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every public name once, at its public home, grouped by the module that defines it, with a one-line summary, and the top-level API of JAX, Equinox, lineax, optimistix and gaussx. It is generated (`uv run poe capabilities`) and checked by `tests/test_capabilities.py`, so it is current.
2. **Search the private helpers** in the table below: several invariants live in exactly one place each.
3. **If it is missing, add it to the module that owns the concept**, at the lowest layer every caller can reach, not inline in the one caller you have today; a helper two modules need goes into `argminax._src`.
4. **One object, one name.** A name means one thing across the package and its submodules (`tests/test_capabilities.py`); a deliberate exception is listed with its reason in `ALLOWED_SHARED_NAMES` in `scripts/capabilities.py`.

| You are about to write… | Use instead |
|---|---|
| A solve loop around `optx.minimise` internals, a hand-rolled BFGS or Newton | `optx.minimise`, `optx.least_squares` with an optimistix solver; `argminax.solve.GaussNewtonCG` for Gauss–Newton or Levenberg–Marquardt with a matrix-free inner CG |
| `jnp.linalg.solve`, `jnp.linalg.inv`, a hand-rolled CG | lineax (`lx.linear_solve` with `lx.CG`, `lx.Cholesky`, …) or the gaussx solve for the operator |
| A dense `K.T @ inv(S_eps) @ K + inv(S_a)` | `gaussx.gauss_newton_operator` (matrix-free and PSD-tagged) or `gaussx.gauss_newton_precision` for a small dense Jacobian; `argminax.curvature` builds Ŝ on them in Wave 1 (#17) |
| A log-determinant | gaussx (`gaussx.logdet` and the structured forms); `matfree` behind the `argminax[matfree]` extra for stochastic Lanczos quadrature |
| One Newton step | `gaussx.newton_update` |
| Differentiating through a converged minimum | `optx.ImplicitAdjoint`; argminax adds `KStep` and `OneStep` only for a solver stopped early (Wave 1, #15) |
| A type for a cost or residual function | `Objective`, `Residuals` (`argminax._typing`) |
| A new exception for bad input or a missing extra | `ValidationError`, `MissingExtraError` |
| A linear-Gaussian test problem, a SciPy or finite-difference check, a sampling bound | the reference harness in `tests/_harness.py` (#14) |

## The contracts

Everything composes because it keeps these contracts; the tests in "What enforces them" check most of them.

### 1. Public functions and modules

- **Validate once, at the boundary.** A public function or module checks its static inputs (shapes, dtypes, option names) and raises `ValidationError` whose message names the argument: `'max_steps' must be a positive integer, got 0`. Static checks go in `__check_init__`; checks on traced arrays use `eqx.error_if`, so they survive `jit`. Internal helpers assume well-formed data.
- **Pure and traceable.** Never mutate an input. Every public function runs under `jit`, `vmap` and `grad` with static shapes, and none of them calls `jax.jit` itself: the caller jits once at the top.
- **Precision is the caller's.** Preserve the input dtype and never cast silently; create constants in the input's dtype (`jnp.asarray(c, dtype=x.dtype)`) or as Python scalars.
- **Documented.** Google-style docstring in the `python-style` layout: a one-line summary, the defining equation in Unicode (`eman-style-math`, no `$…$` in docstrings), `Algorithm:` for a function that implements a method, `Args:` (meaning and symbol, no type), `Returns:`, `Raises:` and an executable `Examples:` block whose outputs are shapes, `allclose` booleans or reprs, never raw floats.

### 2. Solvers and adjoints

- A solver is an optimistix solver: it subclasses `optx.AbstractMinimiser` or `optx.AbstractLeastSquaresSolver` (or another optimistix base) and is called through `optx.minimise` / `optx.least_squares`, never by its own entry point alone.
- An adjoint subclasses `optx.AbstractAdjoint` and works under any optimistix solve.
- Convergence is reported through `RESULTS` in the returned solution. A test that differentiates through a solve first asserts the solve converged.

### 3. Curvature and diagnostics

- Curvature is a gaussx (lineax) operator with the right structure tag, built matrix-free from `jax.linearize` and `jax.linear_transpose` of the residuals; library code never forms it densely.
- Every curvature or diagnostic equals its linear-Gaussian closed form (Rodgers 2000, chapter 2) to 1e-12 relative in float64, through the harness in `tests/_harness.py`.
- A stochastic estimate (Hutchinson traces, perturbation samples) is tested against a bound from its own sampling distribution, never a tolerance picked by eye ("Tests that assert on random draws").

### 4. Exceptions

- Every error derives from `ArgminaxError` **and** the closest standard-library exception, so `except ValueError` keeps working. A new error goes into `exceptions.py`, is exported, and joins the parametrised lists in `tests/test_exceptions.py`.

### The public API

- Logic lives in `argminax._src.<module>`; the public module only re-exports, from its own sorted `__all__`. The top-level `argminax.__all__` holds only the exceptions and `__version__`; both stay sorted (ruff's `RUF022` and `test_all_is_sorted_and_unique` agree on the order).
- A new public module joins `SUBMODULES` (a subpackage joins `SUBPACKAGES`) in `tests/test_public_api.py`, and gets its own page in `docs/api/` (a `::: argminax.<module>` block), a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; a name on an existing module's page is picked up by its `:::` block.
- Run `uv run poe capabilities` after any public API change.
- Link prose to the API by the public name (``[`ArgminaxError`](xref:api#argminax.ArgminaxError)``), see "Documentation".

## What enforces them

Most rules here are tests, so CI tells you when one breaks. Read a test's docstring before changing what it checks.

| Test | Enforces |
|---|---|
| `tests/test_public_api.py` | `__all__` sorted, unique and importable; `SUBMODULES` and `SUBPACKAGES` are exactly the modules and packages; every module declares a sorted `__all__`; the `py.typed` marker ships; `__version__` is dotted |
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no name is bound to two different objects |
| `--doctest-modules` on `src/argminax` (`addopts`) | Every docstring `Examples:` block runs and prints what it says |
| `tests/test_exceptions.py` | The hierarchy: every error is an `ArgminaxError` and its stdlib base |
| `tests/conftest.py` | Every test runs with `jax_enable_x64` on |
| `tests/test_harness.py` | The reference harness in `tests/_harness.py`: the linear-Gaussian closed forms agree with each other (Ŝ with the inverse Hessian of J, the evidence with the Laplace identity) to 1e-12, and each helper passes on its exact case and fails on a perturbed one |
| The `bare-install` job in `ci.yml` | A plain `pip install .` imports argminax without the test group |
| `uv lock --check` in `lint.yml`, and the checks in `release-please.yml` | `uv.lock` matches `pyproject.toml` on every PR; on the release PR, which GitHub runs no workflows for, the lockfile and the tests are checked from the release job and posted as the "release PR checks" check run |
| `tests/test_build_docs.py` | The two-tool docs pipeline: inventory parsing, anchor repair, link checking, the inventory port and the absolute nav URL in `docs/myst.yml`, the files copied to the site root (`ROOT_FILES`) |
| `[tool.coverage.report] fail_under = 90` | Coverage of `src/argminax` in every coverage-enabled run (CI, `uv run poe test-cov`) |
| `strict_markers = true` | Only the registered tier markers (`slow`, `integration`) exist |
| ruff (`select = E, F, I, UP, B, SIM, RUF`), ty | Lint (the whole repo, notebook code cells included), format, types on `src/argminax` and `scripts` |

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in `.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any agent can read and follow them), and two read-only review checklists in `.claude/agents/`:

| Job | Recipe |
|---|---|
| Add a solver or an adjoint (an optimistix extension) | `add-solver` |
| Add a curvature operator or a retrieval diagnostic | `add-diagnostic` |
| Add or update a guide page (MyST prose) | `add-guide-page` |
| Add or update an example notebook | `add-notebook` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `argminax-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `numerics-reviewer.md`) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `uv run poe gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |

There is no downstream Claude Code plugin or `llms.txt` yet. Add them (a `plugins/argminax/` plugin with a `use-argminax` skill, published by `.claude-plugin/marketplace.json`) once Wave 1 gives consumers something to learn.

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`, `ls` and other non-Python commands need no `uv run`.

```bash
uv run poe                  # list every task (they live in [tool.poe.tasks] in pyproject.toml)
uv run poe install          # uv sync --all-groups + pre-commit hooks
uv run poe test             # every test + doctests, no coverage (fast)
uv run poe test-cov         # the same with coverage and the 90 % gate (what CI runs)
uv run poe lint             # ruff check .   (entire repo)
uv run poe format           # ruff format . && ruff check --fix .
uv run poe typecheck        # ty check src/argminax scripts (what CI runs)
uv run poe check            # lint + format-check + typecheck + test-cov (everything CI checks)
uv run poe capabilities     # regenerate docs/api/capabilities.md
uv run poe docs             # build both docs halves into public/ and check every link
uv run poe docs-check       # strict validation of both halves, no themed render
uv run poe docs-api         # Zensical API reference only (zensical build --strict)
uv run poe docs-serve       # build, then serve public/ at :8000
uv run poe precommit        # every pre-commit hook on every file
uv run poe build            # wheel + sdist into dist/
```

Extra arguments after a task name are passed to its command (`uv run poe test tests/test_exceptions.py -k stdlib`).

`addopts` turns coverage on, and a partial run never reaches the 90 % gate, so pass `--no-cov` when running part of the suite:

```bash
uv run pytest --no-cov tests/test_exceptions.py::test_errors_alias_the_closest_stdlib_exception -v
```

### Test tiers

The markers are registered in `pyproject.toml` (`strict_markers = true`):

- **Unmarked (fast):** unit tests and doctests, each well under a second.
- **`@pytest.mark.slow`:** a test over ~1 s (a large sweep, a long fit). For a parametrised test, mark only the expensive cases (`pytest.param(..., marks=pytest.mark.slow)`).
- **`@pytest.mark.integration`:** end-to-end workflows, optional dependencies, external tools.

CI ("Tests", `ci.yml`) runs every tier with coverage on Python 3.12 and 3.13 (`uv run pytest -v`). Select the fast tier with `uv run pytest -m "not slow and not integration"`; when the slow tiers grow, give them a CI job of their own rather than dropping them.

### Tests that assert on random draws

argminax draws random numbers only from a `key` argument (perturbation sampling, Hutchinson probes), never from a host RNG.

- **Incidental randomness** (any draw would do): a fixed key (`jax.random.key(0)`), so a failure reproduces.
- **Sampling behaviour under test** (a Monte Carlo estimate, a statistical property): bound the statistic by its own sampling distribution (a number of standard errors, a test's p-value floor), not by a tolerance tuned on one draw, and say in a comment where the bound comes from. `assert_within_stderr` and `assert_sample_covariance` in `tests/_harness.py` do this for Hutchinson-type estimates and sample covariances.

### Before every commit

All of these must pass, from the repo root:

1. `uv run poe test`: zero failures (doctests included).
2. `uv run --group lint ruff check .`: the **entire** repo, which includes `tests/`, `scripts/` and the code cells of `docs/notebooks/*.ipynb`. Never lint a subdirectory: `ruff check src/argminax/` misses errors CI catches.
3. `uv run --group lint ruff format --check .`
4. `uv run poe typecheck` (`ty check src/argminax scripts`, as CI runs it).
5. After changing a public API: `uv run poe capabilities`, and the `docs/api` page.
6. After changing docs, docstrings, `mkdocs.yml` or `docs/myst.yml`: `uv run poe docs` (or at least `uv run poe docs-api` when only docstrings changed).
7. After changing a dependency: `uv lock`, and commit `uv.lock`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several readings, name them instead of picking one silently; if something is unclear, stop and ask; push back when a simpler approach exists.
2. **Simplicity first.** The minimum code that solves the problem: no speculative features, no single-use abstractions, no configurability nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing style; don't refactor or add docstrings to code you didn't change; remove only what your change made unused, and mention (don't delete) unrelated dead code.
4. **Goal-driven.** Turn the task into a check (a failing test, a reproduced bug, a closed form to match) and loop until it passes; for multi-step work, state the plan with a check per step.

These are the `karpathy-coding-principles` skill in four lines. When you suggest a change, propose the code; when you fix a bug or add a feature, add the test that checks it.

## House style

### Python

`python-style` is the house style. Its general rules:

- Python 3.12+, `from __future__ import annotations` in every module, type hints on every public function, `X | None` over `Optional`, specific exceptions with `raise … from …`, pure functions with side effects isolated and explicit.
- Every module opens with a docstring saying what it holds, in one or two sentences.
- Docstrings follow the layout in "The contracts" above; `Examples:` gives two or three cases (typical, edge, composition), each building its own small data, and `References:` names the paper when one defines the method.
- Comments go above the line, never beside it. A mathematical line has its Unicode equation above it (`# x̄ = (1/N) ∑ᵢ xᵢ`), a long function is split into stages with step headers (`# --- whiten ---`), and every other comment says why, not what.
- Names are descriptive (`posterior_precision`, not `Lam`); the symbol lives only in the equation comment.
- One operation per line, no nested calls that hide an intermediate result, 88 columns.
- One package base error; subclasses are named by what failed.
- Extract only real duplicates (code that must change together); code that merely looks alike stays separate.

Its JAX/Equinox rules apply when the package uses JAX: logic under `pkg/_src/` with public modules that only re-export, jaxtyping with the shared dim vocabulary, einx instead of `axis=` / `.T` / `reshape` / `[:, None]`, lineax for solves, arrays in `eqx.Module`s with `__check_init__` and `eqx.error_if`, `key` keyword-only and last, no silent dtype casts, `get_config()` / `from_config()`, and the four tests per public forward function (closed-form oracle, hypothesis properties, gradient check, transform invariance).

argminax uses JAX, so these rules apply in full. It keeps one departure from `python-style`: every error inherits the closest standard-library exception as well as the package base (an open question in `python-style`, settled here).

**CLIs and scripts** use `cyclopts` for arguments and `loguru` for output; never argparse, click, typer, progress `print`s or stdlib `logging`. For a repo script they are in the `dev` group already; for a package CLI they are runtime dependencies and need the usual discussion. argminax has no CLI. One exception remains: `scripts/capabilities.py` still uses `argparse` and `print`; move it over when you next change it (`scripts/build_docs.py` shows the pattern: a `cyclopts.App` with `@app.default`, and `loguru` for every message).

### Maths notation

Maths follows `eman-style-math` in every format: the letter says what kind of object a symbol is, the subscript says which variable it belongs to, and every symbol is defined, with its shape, before it is used.

- **Code and docstrings:** Unicode only (x̄, σ², Σ⁻¹, ∑, 𝔼, ⊙), never `$…$`; identifiers stay ASCII.
- **MyST prose and MkDocs pages:** `$…$` and `$$…$$` with the kind-named macros (`\vect{x}`, `\gvect{\mu}`, `\mat{K}`, `\gmat{\Sigma}`). The macros are installed in `docs/myst.yml` (`math:`) and `docs/api/javascripts/mathjax.js`; regenerate both with the skill's `scripts/emit_macros.py` when the macros change.
- **GitHub issues and PRs:** expand the macros first (`scripts/expand_macros.py`), display maths in a `math` fence.
- **Everywhere:** each `$…$` expression on one source line; no spacing or sizing macros (`\,`, `\;`, `\quad`, `\qquad`, `\big`, `\tfrac`, `\boxed`, `\cdots`, `\operatorname`). Check a page with `uv run --with cyclopts --with loguru --with pyyaml python ~/.claude/skills/eman-style-math/scripts/check_math.py <file> --target myst` (the skill's scripts need those three packages, which this repo does not install).

### Prose and diagrams

- Markdown prose is never hard-wrapped: one line per paragraph or list item in docs, skills and agent instructions; issue bodies follow `report` (one sentence per line); PR text, one line per paragraph. Don't rewrap untouched text in a file you edit.
- Docs, READMEs and release notes written in the maintainer's voice follow `eman-prose-tech` (the docs register): plain words, "we" for the maintainers, parentheses rather than em dashes.
- Pseudocode in design pages, issues and READMEs follows `code-sketch`: a shapes header, the equation above each line, an aligned shape column and a status marker (`today (pkg)`, `proposed (X)`, `user`) on every call into the stack.
- Never draw ASCII box-and-arrow diagrams; use Mermaid (both docs halves render it). Directory trees stay plain text.

## Personal skills

The maintainer's skills live in [jejagents](https://github.com/jejjohnson/jejagents) and install into `~/.claude/skills/` with its `install.sh`. Use them when they are installed; the sections above carry what an agent without them needs. Where a skill and this file disagree, this file wins: it records this repository's deliberate departures.

| Job | Skill |
|---|---|
| Write, review or refactor Python | `python-style` |
| Maths in docstrings, docs pages, issues or PRs | `eman-style-math` |
| Pseudocode in a design page, issue or README | `code-sketch` |
| Prose in the maintainer's voice (docs, README, release notes, PR descriptions) | `eman-prose-tech` |
| Write up a research, feature, bug, design, epic or chore report | `report`, then file it here with `create-gh-issue` |
| File or link issues in another of the maintainer's repos | `gh-issue` (in this repo, `create-gh-issue` and `link-gh-issues` follow it) |
| A derivation walkthrough for the docs | `derivation`, with `derivation-myst` for a page in `docs/` |
| A learning explainer page | `explainer` |
| General coding discipline | `karpathy-coding-principles` ("Coding principles" above) |

## Git, commits and pull requests

- **Never** push to or merge into `main` unless explicitly told to ("push to main", "merge to main"). Work on a feature branch and commit locally; never run `git push` unless asked. "Merge the branch" means push the feature branch, not merge into `main`. Confirm before any action that affects a shared branch.
- Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) with a lowercase subject (`feat(curvature): add …`); CI validates PR titles (`conventional-commits.yml`). Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`. Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- **No AI attribution.** No `Co-Authored-By:` trailer for an AI on any commit and no "Generated with …" line in a PR description, unless the maintainer asks for one; keep human co-authors' trailers.
- **PR descriptions are for a human who did not watch the work:** why first, a small before/after table, what stays the same, one or two bullets on how it was tested. Label behaviour as "Today (bug): …" and "After this PR: …". Link review comments rather than naming reviewers.
- Write one `Closes #N` line per issue.
- Releases are cut by release-please (`release-please-config.json`): it bumps the version and writes `CHANGELOG.md` from the `feat`, `fix`, `perf` and `revert` commits; don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** An agent called for a small follow-up tends to write a fresh description scoped to its own work and silently discard the rest: read the existing description first and only append checklist items or update their status. Change the title only to fix a Conventional Commits violation.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md); issues follow the label taxonomy and epic model in [`docs/contributing.md`](docs/contributing.md).

### Pull Request Review Comments

Fix a review comment in code, push the fix to the PR branch, then resolve its thread; don't post a reply comment. Don't resolve threads you didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

GraphQL is blocked in Claude sessions; there, and whenever the `gh` CLI is unavailable, use the GitHub MCP tools for the same operations (they can resolve review threads).

## Documentation

The docs are built by **two tools** and deployed as one site (`docs/README.md` has the rationale); `docs.yml` builds them on every PR and push to `main`, and `pages.yml` deploys `public/` to GitHub Pages from `main`.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, guides, example notebooks | mystmd | `docs/*.md`, `docs/guide/`, `docs/notebooks/` (toc in `docs/myst.yml`) | `/` |
| API reference | Zensical + mkdocstrings | `docs/api/` (`docs_dir` in `mkdocs.yml`) | `/reference/` |

`uv run poe docs` runs `scripts/build_docs.py`: it builds the API with `zensical build --strict` (Zensical reads `mkdocs.yml`), serves `site/` on port 8910 so mystmd can read its `objects.inv` (mystmd loads inventories only over http), runs `myst build --html --strict`, rewrites the localhost API URLs to `/reference/`, repairs anchors lowercased by mystmd's `$`-expansion bug, and fails if any internal link in the assembled `public/` site does not resolve. mystmd is a Node CLI (`npm install -g mystmd`), not a uv dependency; if its theme download is blocked, `docs/README.md` says how to build against a local clone of the theme.

- **Prose is MyST Markdown**, not MkDocs-Material syntax: `:::{note}`, `:::{tab-set}`, `:::{dropdown}` directives, not `!!!` / `===` / `???`.
- **Link to the API** with the `xref:` protocol and the **top-level** exported name: ``[`ArgminaxError`](xref:api#argminax.ArgminaxError)``, not `argminax.exceptions.ArgminaxError` (mystmd lowercases `$`-abbreviated anchors; see `docs/README.md`). A missing target fails `myst build --strict`.
- **URLs are flat**: a page's URL is its basename, with `_` turned into `-` (`guide/quickstart.md` → `/quickstart/`, `notebooks/gauss_newton_tour.ipynb` → `/gauss-newton-tour/`), so keep basenames unique across `docs/`, `docs/guide/` and `docs/notebooks/`; frontmatter `slug:` is ignored.
- **Site-nav URLs stay absolute** (`site.nav` in `docs/myst.yml`): the theme prefixes root-relative ones with `BASE_URL` a second time (`tests/test_build_docs.py` checks it).
- **API pages** (`docs/api/*.md`) are MkDocs-format pages, built by Zensical (which runs only the plugins it supports and ignores the rest silently; check its plugin list before adding one): prose plus `:::` blocks, `!!!` admonitions, listed in the `nav` of `mkdocs.yml`; docstrings are Google style (`docstring_style: google`) with Unicode maths; maths in the API pages themselves goes through MathJax (`pymdownx.arithmatex`), with the `eman-style-math` macros loaded by `docs/api/javascripts/mathjax.js`.
- **Notebooks** in `docs/notebooks/` are committed as executed `.ipynb` files, authored as jupytext percent `.py` and deleted after conversion; mystmd renders the stored outputs without re-executing, so each must be listed in the `docs/myst.yml` toc. Figures render inline with `plt.show()`: no `savefig`, no committed PNGs. `notebooks/` at the root holds scratch notebooks, outside the docs. Full standards: `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never committed); track work in GitHub issues. Draft a whole wave of issues from `docs/templates/wave-backlog.md` (see `docs/contributing.md`).

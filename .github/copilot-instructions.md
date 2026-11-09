# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the single source of truth for every coding agent working here (the module map, "reuse before you write", the contracts, the tests that enforce them, commands, the pre-commit checklist, the two-tool docs, git and PR rules).

The essentials, in case you only read this file:

- One package, `src/argminax/`, at Core tier 1 of the GeoStack: it imports JAX, Equinox, lineax, optimistix, jaxtyping and gaussx, and never pipekit, pipekit-cycle, integrix, vardax, pyrox or a GeoModels library. Logic lives in `argminax._src.<module>`; the public modules `solve`, `curvature`, `diagnose` and `adjoint` only re-export. Search [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a helper.
- Keep the contracts in `AGENTS.md`:
  - **public functions** validate static inputs once, at the boundary, raise `ValidationError` naming the argument, never mutate their input, preserve the input dtype and run under `jit`, `vmap` and `grad`;
  - **JAX rules**: no host RNG (randomness comes from a `key`), no `float()` or `.item()` on a traced value, no Python `if`, `while` or `break` on a traced condition, no `jnp.linalg.inv`;
  - **solvers and adjoints** are optimistix subclasses and report non-convergence through `RESULTS`;
  - **curvature** is a gaussx operator with the right PSD tag, never formed densely, and matches its linear-Gaussian closed form to 1e-12 in float64;
  - every exception derives from `ArgminaxError` and its stdlib base.

The rules for every project:

- Every docstring `Examples:` block runs (`--doctest-modules`): the expected output must be what the code prints.
- Before committing, from the repo root: `uv run poe test`, `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `uv run poe typecheck`; `uv run poe capabilities` after a public API change. Lint the whole repo, never a subdirectory.
- When you suggest a change, propose the code; when you fix a bug, add the test that reproduces it. Don't nitpick what ruff and the formatter handle.
- House style ("House style" in `AGENTS.md`): the maintainer's `python-style` for Python (comments above the line, a Unicode equation above each mathematical line, descriptive names), `eman-style-math` for maths (Unicode in code and docstrings, macros in MyST and MkDocs pages), `cyclopts` and `loguru` for new CLIs and scripts, Markdown never hard-wrapped, Mermaid rather than ASCII diagrams.
- Partial test runs need `--no-cov` (the 90 % coverage gate fails a subset).
- Never push or merge to `main` unless told to; never replace an existing PR description, only append to it.
- No AI attribution: no `Co-Authored-By:` trailer for an AI and no "Generated with …" line, unless asked. PR descriptions open with why, then a small before/after table; one `Closes #N` line per issue.
- Step-by-step recipes (add a solver or adjoint, a curvature operator or diagnostic, a guide page or notebook; pre-PR check; review) are plain Markdown in `.claude/skills/<name>/SKILL.md`, and the two review checklists in `.claude/agents/`; follow them as written.
- Path-scoped standards live in `.github/instructions/`; code review follows [`CODE_REVIEW.md`](../CODE_REVIEW.md).

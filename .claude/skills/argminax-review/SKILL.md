---
name: argminax-review
description: Review a change or pull request in this repository against its own rules — CODE_REVIEW.md, the contracts in AGENTS.md, reuse of what the package already provides, and numerics (PSD tags, dtype, convergence before gradients, tracing, closed-form and sampling tolerances, doctest accuracy). Use when asked to review a diff, branch or PR in this repo.
---

# Review a change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note which modules, tests and docs it touches.
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: a re-written solve loop, linear solve, curvature product, log-determinant or test reference is the main way a package drifts from itself and from gaussx and optimistix.
3. **Numerics** — run the `numerics-reviewer` subagent in parallel with step 2: PSD tags, dtype preservation, convergence checked before a gradient is trusted, tracing (no host RNG, no `float()` or Python `break` on traced values), dense inverses, guarded `jnp.where` branches, closed-form and sampling tolerances, doctest accuracy.
4. **Contracts** — for each module touched, the rules under "The contracts" in `AGENTS.md`: static validation at the boundary with `'name' must …` messages, logic in `argminax._src` and re-exports only in the public module, solvers and adjoints as optimistix subclasses reporting through `RESULTS`, curvature as tagged gaussx operators never formed densely, the closed-form test through `tests/_harness.py`, the `jit` / `vmap` / `grad` smoke test, the layer rule in "Boundaries" (no import of pipekit, pipekit-cycle, integrix, vardax, pyrox or a GeoModels library), exceptions deriving from `ArgminaxError` and their stdlib base.
5. **Checklist** — the rest of `CODE_REVIEW.md` (public API, API page and `docs/api/index.md` row, capability index, MyST prose, tests and tiers), skipping what ruff, ty or the tests already enforce.
6. **House style** — the `python-style` review checklist if the skill is installed, otherwise "House style" in `AGENTS.md`: docstring layout with a Unicode equation, comments above the line with the equation above each mathematical line, descriptive names, maths in the `eman-style-math` notation, `cyclopts` and `loguru` in new scripts.
7. **Verify claims** — for anything you flag as a bug, run it (`uv run python -c "..."`) and report what it printed.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with priority, file, lines and a concrete change, summary).

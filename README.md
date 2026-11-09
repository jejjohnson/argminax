# argminax

[![Tests](https://github.com/jejjohnson/argminax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/argminax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/argminax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/argminax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/argminax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/argminax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/argminax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/argminax/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/argminax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/argminax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Author: J. Emmanuel Johnson

Optimisation with uncertainty at the minimum for JAX: solvers, curvature, retrieval diagnostics and minimiser adjoints on optimistix and gaussx.

A retrieval is not finished when the optimiser stops. argminax finds x̂ = argmin J(x), then says how sharp that minimum is (the posterior covariance Ŝ = (KᵀSε⁻¹K + Sₐ⁻¹)⁻¹, the averaging kernel A = I − ŜSₐ⁻¹, the degrees of freedom for signal) and how x̂ depends on what went in. It sits at Core tier 1 of the [GeoStack](https://github.com/jejjohnson/geostack): it imports gaussx and extends optimistix, as gaussx extends lineax, so its solvers and adjoints plug into `optx.minimise` and `optx.least_squares`. It takes over the curvature, diagnostics, Newton loops and minimiser adjoints that vardax, radtrax, plumax, pyrox and gaussx each rebuilt.

## Status

argminax is in Wave 0: the package, its CI and the reference test harness. The design is the GeoStack [argminax design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/argminax.md), and the backlog is five wave epics under [geostack#134](https://github.com/jejjohnson/geostack/issues/134).

| Wave | Delivers | Epic |
|---|---|---|
| 0 | the package and the reference harness (linear-Gaussian closed forms, SciPy, finite differences, sampling bounds) | [#8](https://github.com/jejjohnson/argminax/issues/8) |
| 1 | `KStep` and `OneStep` adjoints and the Laplace covariance, moved in from vardax | [#9](https://github.com/jejjohnson/argminax/issues/9) |
| 2 | the averaging kernel, degrees of freedom, information content and evidence | [#10](https://github.com/jejjohnson/argminax/issues/10) |
| 3 | `GaussNewtonCG`, the dual solve, `laplace_mode` and reparametrised bounds | [#11](https://github.com/jejjohnson/argminax/issues/11) |
| 4 | the stack's libraries switch to argminax | [#12](https://github.com/jejjohnson/argminax/issues/12) |

## Installation

argminax is not on PyPI; the stack pins it by git tag. It brings in JAX, Equinox, lineax, optimistix and gaussx; the `matfree` extra adds stochastic Lanczos quadrature for log-determinants of large operators.

```bash
pip install "argminax @ git+https://github.com/jejjohnson/argminax.git"
```

## Development

```bash
git clone https://github.com/jejjohnson/argminax.git
cd argminax
uv run poe install     # every dependency group + the pre-commit hooks
uv run poe check       # lint, format check, typecheck, tests with the 90 % coverage gate
uv run poe docs        # the MyST prose and the Zensical API reference, assembled into public/
```

`uv run poe` lists every task. [`AGENTS.md`](AGENTS.md) holds the rules for contributors and coding agents: the module map, the contracts, the tests that enforce them and the house style. [`docs/contributing.md`](docs/contributing.md) has the label taxonomy and the epic model.

## License

MIT; see [`LICENSE`](LICENSE).

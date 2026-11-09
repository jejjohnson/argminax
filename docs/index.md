# argminax

> Optimisation with uncertainty at the minimum for JAX: solvers, curvature, retrieval diagnostics and minimiser adjoints on optimistix and gaussx.

A retrieval is not finished when the optimiser stops. argminax finds $\hat x = \arg\min_x J(x)$, then says how sharp that minimum is (the posterior covariance $\hat S$, the averaging kernel $A$, the degrees of freedom for signal) and how $\hat x$ depends on what went in. For a measurement $y$, forward model $F$, prior state $x_a$ and covariances $S_\varepsilon$ and $S_a$, the cost it minimises most often is

$$
J(x) = \frac{1}{2} (y - F(x))^\top S_\varepsilon^{-1} (y - F(x)) + \frac{1}{2} (x - x_a)^\top S_a^{-1} (x - x_a).
$$

argminax sits at Core tier 1 of the GeoStack: it imports gaussx and extends optimistix, as gaussx extends lineax, so its solvers and adjoints plug into `optx.minimise` and `optx.least_squares`. The design is in the GeoStack [argminax design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/argminax.md), and the backlog is tracked in waves under [geostack#134](https://github.com/jejjohnson/geostack/issues/134).

:::{note} Status
argminax is in Wave 0: the package, its CI and the reference test harness. The adjoints and the curvature at the minimum arrive in Wave 1.
:::

## Installation

::::{tab-set}
:::{tab-item} pip

```bash
pip install "argminax @ git+https://github.com/jejjohnson/argminax.git"
```
:::
:::{tab-item} With matfree

```bash
pip install "argminax[matfree] @ git+https://github.com/jejjohnson/argminax.git"
```
:::
:::{tab-item} From source

```bash
git clone https://github.com/jejjohnson/argminax.git
cd argminax
uv run poe install
```
:::
::::

## The four modules

| Module | Question it answers | Wave |
|---|---|---|
| `argminax.adjoint` | how does $\hat x$ move when what went in moves, for a solver stopped early? | 1 |
| `argminax.curvature` | how sharp is the minimum? ($\hat S$ as a gaussx operator) | 1 |
| `argminax.diagnose` | how much came from the measurement? ($A$, $d_s$, information content, evidence) | 2 |
| `argminax.solve` | where is the minimum, for the solvers optimistix lacks? | 3 |

## Next steps

- [Quickstart](guide/quickstart.md): installing, double precision and the calling convention
- [API reference](https://jejjohnson.github.io/argminax/reference/): every module and the capability index

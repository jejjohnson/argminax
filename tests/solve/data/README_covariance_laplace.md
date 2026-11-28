# Dense covariance-form Laplace fixture

`covariance_laplace.npz` holds what `laplace_mode_covariance` gave before it took a gaussx solver strategy (#98), when every step was a dense factorisation, so `tests/solve/test_laplace_mode.py::test_default_matches_frozen_fixtures` checks that `solver=None` still gives it and `test_solvers_agree_with_dense` compares the iterative solvers with it.

- argminax at `bbbdc05` (`origin/main`, v0.0.10, 28 November 2026) with gaussx v0.8.3, float64.
- The problem: 200 inputs drawn uniformly on [−5, 5] and sorted (`inputs`, NumPy seed 98), and a Matérn-3/2 Gram matrix with variance 1.5, lengthscale 0.9 and jitter 1e-6, which the script and the test both build from `inputs` with the same NumPy code.
- Bernoulli (`bernoulli_*`): labels sin(1.5 x) > 0 with one in ten flipped (`bernoulli_y`), prior mean 0 (`bernoulli_mean`), the logit link.
- Poisson (`poisson_*`): counts drawn with rate exp(0.5 + sin x) (`poisson_y`), prior mean 0.5 (`poisson_mean`), the log link.
- For each, `laplace_mode_covariance` with `max_iter=50`, `tol=1e-10`: `_mode`, `_mean_at_mode`, `_variance`, `_nat1`, `_nat2`, `_log_marginal` and `_n_iter` (6 and 7 steps).

Regenerate with `make_covariance_laplace_fixture.py`, on a checkout of the commit to freeze; its docstring gives the command.

# gaussx laplace_mode fixture

`gaussx_laplace_mode.npz` holds the results of gaussx's `laplace_mode`, which gaussx 0.8.0 removed (gaussx#848), so `tests/solve/test_laplace_mode.py` can keep checking the port (#25) to 1e-10 against the code it replaced.

- gaussx v0.6.13, float64 unless the key says `float32`, on the problems of `tests/_gaussx_problems.py`.
- For each of `gaussian`, `intrinsic-hard` and `odd-n-selection`: `<case>_mode`, `<case>_n_iter`, `<case>_log_marginal` and `<case>_hessian_type` (the class name of the Hessian operator).
- `float32_mode`, `float32_log_marginal`: the Gaussian prior of 12 points in float32.
- `log_marginal_grad`: the derivative of the intrinsic case's log-marginal with respect to log τ at log τ = 2.
- `vb_mean`: `gx.vb_mean_correction` of the Gaussian case with the subspace (0, 1, 2).

Regenerate with `make_gaussx_laplace_fixture.py`, which needs a gaussx that still has `laplace_mode` (v0.6.13 up to v0.7.x); its docstring gives the `uv run --with` command.

# Adjoints

Adjoints of a minimiser stopped early, passed as `adjoint=` to `optx.minimise` or `optx.least_squares`. At a converged minimum, use optimistix's own `ImplicitAdjoint`.

`safe_implicit_adjoint` with `lx.GMRES(rtol=1e-8, atol=1e-8, restart=20, max_steps=50)` is the default `adjoint=` of `argminax.solve.laplace_mode_covariance` and `argminax.solve.laplace_mode_sites`; pass `safe_implicit_adjoint` with another lineax solver, or any `optx.ImplicitAdjoint`, to change it.

![Three ways to differentiate a solve: unrolling through every iteration, the implicit adjoint at the minimum, and KStep and OneStep through the last k updates](https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_adjoints/argminax_adjoints.png)

::: argminax.adjoint
    options:
      show_root_heading: true
      show_root_full_path: false

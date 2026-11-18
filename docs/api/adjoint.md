# Adjoints

Adjoints of a minimiser stopped early, passed as `adjoint=` to `optx.minimise` or `optx.least_squares`. At a converged minimum, use optimistix's own `ImplicitAdjoint`.

![Three ways to differentiate a solve: unrolling through every iteration, the implicit adjoint at the minimum, and KStep and OneStep through the last k updates](https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_adjoints/argminax_adjoints.png)

::: argminax.adjoint
    options:
      show_root_heading: true
      show_root_full_path: false

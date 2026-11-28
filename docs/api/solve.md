# Solvers

Solvers optimistix lacks, each an optimistix solver called through `optx.minimise` or `optx.least_squares`.

The covariance and site forms of the Laplace loop (`laplace_mode_covariance`, `laplace_mode_sites`) differentiate their mode implicitly, through `adjoint=`. The default, `safe_implicit_adjoint(lx.GMRES(rtol=1e-8, atol=1e-8, restart=20, max_steps=50))`, solves the tangent system from products with the Jacobian of the fixed-point map, so no N × N matrix is formed, and a failed solve returns a value instead of raising. To change it, pass any `optx.ImplicitAdjoint`: a tighter GMRES for a hard problem, or `optx.ImplicitAdjoint(lx.AutoLinearSolver(well_posed=True))` for the dense LU used before.

::: argminax.solve
    options:
      show_root_heading: true
      show_root_full_path: false

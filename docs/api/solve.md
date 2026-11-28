# Solvers

Solvers optimistix lacks, each an optimistix solver called through `optx.minimise` or `optx.least_squares`.

The covariance and site forms of the Laplace loop (`laplace_mode_covariance`, `laplace_mode_sites`) differentiate their mode implicitly, through `adjoint=`. The default, `safe_implicit_adjoint(lx.GMRES(rtol=1e-8, atol=1e-8, restart=20, max_steps=50))`, solves the tangent system from products with the Jacobian of the fixed-point map, so no N × N matrix is formed, and a failed solve returns a value instead of raising. To change it, pass any `optx.ImplicitAdjoint`: a tighter GMRES for a hard problem, or `optx.ImplicitAdjoint(lx.AutoLinearSolver(well_posed=True))` for the dense LU used before.

`laplace_mode_covariance` solves every step against $\mat{A} = \mat{K}_{XX} + \gmat{\Lambda}_\text{site}^{-1}$ through the gaussx solver strategy passed as `solver=`, by `gaussx.site_posterior`, and takes its log-marginal from `gaussx.site_log_evidence`. With an iterative strategy and a matrix-free prior (a `lx.FunctionLinearOperator`, say), no $N \times N$ array is formed in the loop, in the gradient of the mode or in the log-marginal; pass `with_variance=False` for a large $N$, since the marginal variances cost $N$ solves, and ask `fit.site_posterior.variance(...)` later for the points that need them.

| `solver=` | Each Newton step | `log_marginal` |
|---|---|---|
| `None`, `gaussx.DenseSolver()` | one Cholesky of $\mat{A}$, $O(N^3)$; a structured prior is densified with a `gaussx.DenseFallbackWarning` | standard form, exact |
| `gaussx.CGSolver(rtol=..., atol=...)` | conjugate gradients, one product with $\mat{K}_{XX}$ per CG step | pseudo-observation form, stochastic Lanczos log-determinant |
| `gaussx.CGSolver(preconditioner=gaussx.PartialCholeskyPreconditioner(rank=k, shift=None))` | preconditioned CG, with the per-point shifts $\vect{d} = 1 / \gvect{\lambda}$ read from $\mat{A}$ | as CG |
| `gaussx.BBMMSolver(...)` | conjugate gradients | pseudo-observation form, one shared mBCG pass, probes drawn from `key=` |
| `gaussx.compose_solver(solver, logdet=gaussx.DenseLogdet())` | `solver` | standard form, exact |

An iterative step is inexact, and the loop's test $\Vert \Delta \vect{f} \Vert_\infty < \text{tol}$ cannot resolve below the solver's error, so give the solver an `rtol` at least ten times tighter than `tol`, relative to the scale of $\vect{f}$. A gaussx strategy raises when it runs out of steps (`throw=True`, its default); inside a sampler such as NUTS build it with `throw=False`, and `converged` reports the failure. CG starts every step from zero: gaussx strategies take no initial guess yet, so a warm start from the previous step is a follow-up.

`site_fixed_point` is the loop over Gaussian sites that is not a Newton method: expectation propagation and posterior linearisation iterate a damped site update s ← T(s) to a fixed point, and their gradient comes from the implicit function theorem at that point (by default `safe_implicit_adjoint(lx.GMRES(rtol=1e-10, atol=1e-10, restart=20, max_steps=50))`, from products alone). The split is that the loop and its gradient are argminax's, and one update is the caller's, built from gaussx's site primitives (`gaussx.cavity_distribution`, `gaussx.ep_tilted_moments`, `gaussx.newton_update`, `gaussx.damped_natural_update`). Damping, precision floors, step caps and the first-iteration special case of cavity methods stay in the update, since they differ between strategies.

::: argminax.solve
    options:
      show_root_heading: true
      show_root_full_path: false

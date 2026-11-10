# Changelog

## [0.0.6](https://github.com/jejjohnson/argminax/compare/v0.0.5...v0.0.6) (2026-11-10)


### Bug Fixes

* **deps:** pin gaussx to its v0.6.13 release instead of a pull-request commit ([#52](https://github.com/jejjohnson/argminax/issues/52)) ([5281c39](https://github.com/jejjohnson/argminax/commit/5281c3964fe2fac86acaeee6ce17ad1842ae8a97))

## [0.0.5](https://github.com/jejjohnson/argminax/compare/v0.0.4...v0.0.5) (2026-11-10)


### Features

* **solve:** add laplace_mode, with its covariance and site forms ([#49](https://github.com/jejjohnson/argminax/issues/49)) ([eabc2ef](https://github.com/jejjohnson/argminax/commit/eabc2ef7e19a8f154619c6c4e87f38467aa7c49a)), closes [#25](https://github.com/jejjohnson/argminax/issues/25)

## [0.0.4](https://github.com/jejjohnson/argminax/compare/v0.0.3...v0.0.4) (2026-11-10)


### Features

* **solve:** add dual_solve, the observation-space solve, and its incremental outer loop ([#46](https://github.com/jejjohnson/argminax/issues/46)) ([42b3197](https://github.com/jejjohnson/argminax/commit/42b3197ddb971336c8668b2847d04cddd4371463)), closes [#24](https://github.com/jejjohnson/argminax/issues/24)
* **solve:** add GaussNewtonCG, Gauss–Newton and Levenberg–Marquardt with a matrix-free inner CG ([#43](https://github.com/jejjohnson/argminax/issues/43)) ([d9d07e9](https://github.com/jejjohnson/argminax/commit/d9d07e968c9a644e9195aec811944034e1c33ab0)), closes [#23](https://github.com/jejjohnson/argminax/issues/23)
* **solve:** add Reparametrised, positive and bounded states by a change of variable ([#47](https://github.com/jejjohnson/argminax/issues/47)) ([b3bbbf1](https://github.com/jejjohnson/argminax/commit/b3bbbf15c5a057c867ecf035f2b06be0d28e1a74)), closes [#26](https://github.com/jejjohnson/argminax/issues/26)

## [0.0.3](https://github.com/jejjohnson/argminax/compare/v0.0.2...v0.0.3) (2026-11-10)


### Features

* **curvature:** let Laplace hold a gridded mode and keep the precision's structure in its covariance ([#36](https://github.com/jejjohnson/argminax/issues/36)) ([0c6d312](https://github.com/jejjohnson/argminax/commit/0c6d3128513f51139564d5454ffea07ebfd37ca6))
* **diagnose:** add the averaging kernel, degrees of freedom for signal and information content ([#38](https://github.com/jejjohnson/argminax/issues/38)) ([3f3b0fb](https://github.com/jejjohnson/argminax/commit/3f3b0fb9fa671b47eb9f773a695b86ce8bc4dfa5)), closes [#20](https://github.com/jejjohnson/argminax/issues/20)
* **diagnose:** add the evidence at the minimum and the prior-precision search ([#39](https://github.com/jejjohnson/argminax/issues/39)) ([13fd7e9](https://github.com/jejjohnson/argminax/commit/13fd7e924cd9b55e5f7fe2144ea255fbf33db96a))

## [0.0.2](https://github.com/jejjohnson/argminax/compare/v0.0.1...v0.0.2) (2026-11-09)


### Features

* **adjoint:** add KStep and OneStep with an apply that works under any optimistix solve ([#30](https://github.com/jejjohnson/argminax/issues/30)) ([3995efa](https://github.com/jejjohnson/argminax/commit/3995efa3c3dcb9769f9100dd34a6f6bc253ff05f))
* **adjoint:** add to_optimistix_adjoint, mapping pipekit-cycle adjoint specs by class name ([#31](https://github.com/jejjohnson/argminax/issues/31)) ([9bde880](https://github.com/jejjohnson/argminax/commit/9bde880ccdce2b4ba47e7f0975cacd1282ad2a74)), closes [#16](https://github.com/jejjohnson/argminax/issues/16)
* **curvature:** add Laplace and gauss_newton, the Gaussian at the minimum as a gaussx operator ([b98e3a5](https://github.com/jejjohnson/argminax/commit/b98e3a59bfb694775d0ca46e242bc451d65874ea))
* **curvature:** add the low-rank and diagonal forms of the posterior covariance, and perturbation sampling ([c7b4cc2](https://github.com/jejjohnson/argminax/commit/c7b4cc2defa581afd5dafc25eb0a58aa70ca547a))

## Changelog

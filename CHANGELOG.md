# Changelog

## [0.0.10](https://github.com/jejjohnson/argminax/compare/v0.0.9...v0.0.10) (2026-11-28)


### Features

* **solve:** site_fixed_point, the differentiable loop over Gaussian sites ([#106](https://github.com/jejjohnson/argminax/issues/106)) ([934ff1f](https://github.com/jejjohnson/argminax/commit/934ff1f18aeaef14f71ba7048cfb650f3d65efbc)), closes [#100](https://github.com/jejjohnson/argminax/issues/100)
* **solve:** the implicit gradient of laplace_mode_covariance and laplace_mode_sites is matrix-free ([#105](https://github.com/jejjohnson/argminax/issues/105)) ([cb0a895](https://github.com/jejjohnson/argminax/commit/cb0a895944ef66b03b96c25b187f4974ecfed870)), closes [#99](https://github.com/jejjohnson/argminax/issues/99)


### Bug Fixes

* **solve:** evaluate the laplace_mode_covariance objective exactly at the newton target from the mode ([#95](https://github.com/jejjohnson/argminax/issues/95)) ([3a42cc7](https://github.com/jejjohnson/argminax/commit/3a42cc77ef0898997104e43a0c0f7a515ca5d164)), closes [#94](https://github.com/jejjohnson/argminax/issues/94)

## [0.0.9](https://github.com/jejjohnson/argminax/compare/v0.0.8...v0.0.9) (2026-11-21)


### Bug Fixes

* **solve:** keep the laplace_mode_covariance log marginal finite on a badly conditioned float32 prior ([#92](https://github.com/jejjohnson/argminax/issues/92)) ([da6c765](https://github.com/jejjohnson/argminax/commit/da6c7652b98c093812cd09fc20dbc28e35402dc2)), closes [#91](https://github.com/jejjohnson/argminax/issues/91)

## [0.0.8](https://github.com/jejjohnson/argminax/compare/v0.0.7...v0.0.8) (2026-11-20)


### Features

* **adjoint:** add safe_implicit_adjoint and TolerantLinearSolver, an implicit adjoint whose linear solve never raises inside a sampler ([089d61f](https://github.com/jejjohnson/argminax/commit/089d61f33344a269694f345765f918675d323d1e))
* **adjoint:** add safe_implicit_adjoint and TolerantLinearSolver, an implicit adjoint whose linear solve never raises inside a sampler ([5773b48](https://github.com/jejjohnson/argminax/commit/5773b4864301cd17dedce36986df551f9d2b96cb)), closes [#87](https://github.com/jejjohnson/argminax/issues/87)

## [0.0.7](https://github.com/jejjohnson/argminax/compare/v0.0.6...v0.0.7) (2026-11-18)


### Features

* **blr:** add blr_posterior, the Laplace posterior of any optimiser state ([#68](https://github.com/jejjohnson/argminax/issues/68)) ([7694aa6](https://github.com/jejjohnson/argminax/commit/7694aa6948286560649c904313121e7f238f8eee))
* **blr:** add preconditioned_sgd, gradient descent through the inverse of a fixed operator ([#74](https://github.com/jejjohnson/argminax/issues/74)) ([81dc462](https://github.com/jejjohnson/argminax/commit/81dc462142503d8abbcd7fa7aa83085c706aeb35))
* **blr:** add the diagonal Bayesian learning rule and IVON as optax transforms ([#65](https://github.com/jejjohnson/argminax/issues/65)) ([e41a569](https://github.com/jejjohnson/argminax/commit/e41a569b45a106246e96d4507bbd6aef04134a9d))
* **blr:** add the full-rank Bayesian learning rule and the Newton preset ([#66](https://github.com/jejjohnson/argminax/issues/66)) ([837b3e5](https://github.com/jejjohnson/argminax/commit/837b3e5fb0fdd80b2b7ccff8a011e5fe06ff2b96))
* **blr:** add the low-rank Bayesian learning rule with SLANG's update ([#67](https://github.com/jejjohnson/argminax/issues/67)) ([bd76dd5](https://github.com/jejjohnson/argminax/commit/bd76dd53abaa284a52caf3b5b2369c1cbc82782a))

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

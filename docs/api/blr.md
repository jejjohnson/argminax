# Bayesian learning rule

optax transforms that carry a Gaussian over the parameters in their state, so the uncertainty is there during training. They need the `argminax[optax]` extra. The [guide to the Bayesian learning rule](../blr/) derives the update and says what each transform stores and costs.

::: argminax.blr
    options:
      show_root_heading: true
      show_root_full_path: false

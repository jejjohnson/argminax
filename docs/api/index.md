# API Reference

Everything listed in `argminax.__all__` is importable from the top level, and each area of the library is a public module that re-exports its private implementation in `argminax._src`:

```python
import argminax as amx

amx.ArgminaxError  # the base of every argminax error
```

The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each, together with the public API of JAX, Equinox, lineax, optimistix and gaussx.

## Modules

| Module | Contents |
|---|---|
| [Adjoints](adjoint.md) | `KStep`, `OneStep`, `to_optimistix_adjoint` |
| [Curvature](curvature.md) | `Laplace`, `gauss_newton`, `push_forward` |
| [Type aliases](typing.md) | `Objective`, `Residuals` |
| [Exceptions](exceptions.md) | `ArgminaxError`, `ValidationError`, `MissingExtraError` |

`argminax.diagnose` arrives in Wave 2 and `argminax.solve` in Wave 3, each with its own page.

## Package overview

::: argminax
    options:
      members: false
      show_root_heading: false
      show_source: false

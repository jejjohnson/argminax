# Exceptions

Every error raised by the package derives from [`ArgminaxError`][argminax.ArgminaxError] *and* from the closest standard-library exception.

![argminax's exceptions: ArgminaxError inherits from Exception; ValidationError inherits from ArgminaxError and ValueError and is raised in every module; MissingExtraError inherits from ArgminaxError and ImportError and names the missing optax or matfree extra](https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/argminax_exceptions/argminax_exceptions-light.svg)

::: argminax.exceptions

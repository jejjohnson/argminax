"""Exception hierarchy for `argminax`.

Every error raised by this package derives from
[`ArgminaxError`][argminax.ArgminaxError], so a caller can trap the whole
package with a single ``except`` clause. Each concrete error also inherits from
the closest standard-library exception, so code written against the stdlib
keeps working without importing anything from this package.

Errors are raised only on static information (shapes, dtypes, options) at the
boundary. A failure that depends on traced values, such as a solve that did
not converge, is reported through optimistix's ``RESULTS`` instead, because
Python cannot branch on a traced value.
"""

from __future__ import annotations


__all__ = [
    "ArgminaxError",
    "MissingExtraError",
    "ValidationError",
]


class ArgminaxError(Exception):
    """Base class for every exception raised by ``argminax``.

    Examples:
        >>> from argminax import ArgminaxError, ValidationError
        >>> issubclass(ValidationError, ArgminaxError)
        True
    """


class ValidationError(ArgminaxError, ValueError):
    """Raised when an argument fails validation.

    The checks are static: shapes, dtypes and option names, never traced
    values. Also a `ValueError`, so ``except ValueError`` still catches it.

    Examples:
        >>> from argminax import ValidationError
        >>> issubclass(ValidationError, ValueError)
        True
    """


class MissingExtraError(ArgminaxError, ImportError):
    """Raised when a function needs an optional extra that is not installed.

    The functions behind the ``argminax[matfree]`` extra import it when they
    are called and raise this error, whose message names the extra to install.
    Also an `ImportError`, so ``except ImportError`` still catches it.

    Examples:
        >>> from argminax import MissingExtraError
        >>> issubclass(MissingExtraError, ImportError)
        True
    """

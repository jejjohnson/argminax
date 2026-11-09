"""Shared pytest configuration.

Every test runs in double precision: the reference checks compare against
closed forms to 1e-12, which single precision cannot reach.
"""

from __future__ import annotations

import jax


jax.config.update("jax_enable_x64", True)

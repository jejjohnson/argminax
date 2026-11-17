"""Fixtures for the Bayesian learning rule tests: the optax_bayes reference numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _regression import Regression


@pytest.fixture(scope="session")
def reference():
    return np.load(Path(__file__).parent / "data" / "optax_bayes.npz")


@pytest.fixture(scope="session")
def regression(reference) -> Regression:
    return Regression(reference)

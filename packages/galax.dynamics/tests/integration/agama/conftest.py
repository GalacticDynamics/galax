"""Fixtures for the agama comparisons."""

from types import ModuleType

import jax
import pytest


@pytest.fixture(scope="session")
def agama() -> ModuleType:
    """Import agama and set its units to Msun / kpc / km/s -- *not* G = 1."""
    import agama  # deferred: optional, and built by hand

    # Comparing float32 galax with float64 agama measures the dtype. pytest-env
    # sets this; fail loudly if a run overrode it.
    assert jax.config.jax_enable_x64, "agama comparisons need JAX_ENABLE_X64=True"
    agama.setUnits(mass=1, length=1, velocity=1)
    return agama

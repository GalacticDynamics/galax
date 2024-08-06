"""Test the :mod:`galax.integrate` module."""

from galax.dynamics import integrate
from galax.dynamics._dynamics.integrate import api, base, builtin, funcs, utils


def test_all() -> None:
    """Test the API."""
    assert set(integrate.__all__) == set(
        api.__all__ + base.__all__ + builtin.__all__ + funcs.__all__ + utils.__all__
    )

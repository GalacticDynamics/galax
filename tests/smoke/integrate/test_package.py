"""Test the :mod:`galax.integrate` module."""

from galax import integrate
from galax.integrate import _base, _builtin


def test_all() -> None:
    """Test the API."""
    assert set(integrate.__all__) == set(_base.__all__ + _builtin.__all__)

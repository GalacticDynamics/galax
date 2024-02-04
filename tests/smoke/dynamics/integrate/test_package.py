"""Test the :mod:`galax.integrate` module."""

from galax.dynamics import integrate
from galax.dynamics._dynamics.integrate import _api, _base, _builtin


def test_all() -> None:
    """Test the API."""
    assert set(integrate.__all__) == set(
        _api.__all__ + _base.__all__ + _builtin.__all__
    )

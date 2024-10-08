"""Test the :mod:`galax.integrate` module."""

from galax.dynamics import integrate
from galax.dynamics._src.integrate import core, funcs, type_hints, utils


def test_all() -> None:
    """Test the API."""
    assert set(integrate.__all__) == set(
        core.__all__ + funcs.__all__ + utils.__all__ + type_hints.__all__
    )

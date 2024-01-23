"""Test the :mod:`galax.integrate` module."""

from galax import integrate


def test_all():
    """Test the API."""
    assert set(integrate.__all__) == set(
        integrate._base.__all__
        + integrate._builtin.__all__
        + integrate._timespec.__all__
    )

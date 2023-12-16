"""Testing :mod:`galax.dynamics` module."""

import galax.dynamics as gd


def test_all():
    """Test the `galax.potential` API."""
    assert set(gd.__all__) == set(
        gd._core.__all__ + gd._orbit.__all__ + gd.mockstream.__all__
    )

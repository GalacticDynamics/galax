"""Testing :mod:`galax.nbody` module."""

import galax.nbody as gn


def test_all() -> None:
    """Test the `galax.nbody` API."""
    assert set(gn.__all__) == {"NBodyField"}
    for name in gn.__all__:
        assert hasattr(gn, name)

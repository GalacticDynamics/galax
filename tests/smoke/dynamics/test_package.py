"""Testing :mod:`galax.dynamics` module."""

import galax.dynamics as gd
from galax.dynamics._dynamics import mockstream, orbit


def test_all() -> None:
    """Test the `galax.potential` API."""
    assert set(gd.__all__) == {"mockstream", *orbit.__all__, *mockstream.__all__}

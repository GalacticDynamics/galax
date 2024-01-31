"""Testing :mod:`galax.dynamics` module."""

import galax.dynamics as gd
from galax.dynamics._dynamics import base, core, mockstream, orbit


def test_all() -> None:
    """Test the `galax.potential` API."""
    assert set(gd.__all__) == {
        "mockstream",
        *base.__all__,
        *core.__all__,
        *orbit.__all__,
        *mockstream.__all__,
    }

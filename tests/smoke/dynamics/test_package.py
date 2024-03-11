"""Testing :mod:`galax.dynamics` module."""

import galax.dynamics as gd
from galax.dynamics._dynamics import base, mockstream, orbit


def test_all() -> None:
    """Test the `galax.potential` API."""
    assert set(gd.__all__) == {
        "integrate",
        "mockstream",
        *base.__all__,
        *orbit.__all__,
        *mockstream.__all__,
        "integrate_orbit",
        "evaluate_orbit",
    }

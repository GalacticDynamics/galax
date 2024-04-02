"""Testing :mod:`galax.dynamics` module."""

import galax.coordinates as gc
from galax.coordinates._psp import base, core, utils


def test_all() -> None:
    """Test the `galax.coordinates` API."""
    assert set(gc.__all__) == {
        "operators",
        *base.__all__,
        *core.__all__,
        *utils.__all__,
    }

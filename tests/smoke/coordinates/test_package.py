"""Testing :mod:`galax.dynamics` module."""

import galax.coordinates as gc
from galax.coordinates._psp import base, psp, pspt, utils


def test_all() -> None:
    """Test the `galax.coordinates` API."""
    assert set(gc.__all__) == {
        "operators",
        *base.__all__,
        *psp.__all__,
        *pspt.__all__,
        *utils.__all__,
    }

"""Testing :mod:`galax.dynamics` module."""

import galax.coordinates as gc
from galax.coordinates._psp import base, base_composite, base_psp, core, interp, utils


def test_all() -> None:
    """Test the `galax.coordinates` API."""
    assert set(gc.__all__) == {
        "operators",
        *base.__all__,
        *base_psp.__all__,
        *base_composite.__all__,
        *core.__all__,
        *interp.__all__,
        *utils.__all__,
    }

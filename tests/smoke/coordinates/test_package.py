"""Testing :mod:`galax.dynamics` module."""

import galax.coordinates as gc
from galax.coordinates._src.psps import (
    base,
    base_composite,
    base_psp,
    core,
    core_composite,
    interp,
    utils,
)


def test_all() -> None:
    """Test the `galax.coordinates` API."""
    assert set(gc.__all__) == {
        "ops",
        "frames",
        *base.__all__,
        *base_psp.__all__,
        *base_composite.__all__,
        *core.__all__,
        *core_composite.__all__,
        *interp.__all__,
        *utils.__all__,
    }

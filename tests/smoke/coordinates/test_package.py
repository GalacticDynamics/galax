"""Testing :mod:`galax.dynamics` module."""

import galax.coordinates as gc
from galax.coordinates import _base, _psp, _pspt, _utils


def test_all() -> None:
    """Test the `galax.potential` API."""
    assert set(gc.__all__) == {
        *_base.__all__,
        *_psp.__all__,
        *_pspt.__all__,
        *_utils.__all__,
    }

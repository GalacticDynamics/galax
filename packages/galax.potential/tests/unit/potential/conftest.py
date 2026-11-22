"""Fixtures for the potential tests."""

from collections.abc import Iterator

import mpmath as mp
import pytest


@pytest.fixture(autouse=True)
def _mp_precision(request: pytest.FixtureRequest) -> Iterator[None]:
    """Run each test at the mpmath precision of its module, then restore it.

    A test module sets ``MP_DPS`` (default 30). ``mp.mp.dps`` is global, so a
    test that set it for good would change the references that another module
    builds later in the same process.
    """
    with mp.workdps(getattr(request.module, "MP_DPS", 30)):
        yield

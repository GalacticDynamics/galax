"""Hand freed compile memory back to the OS after each test (glibc).

The Staeckel tests compile large jacfwd/jacrev programs (up to ~1 GB on
top of the worker). XLA frees each one, but glibc can keep the freed pages
in its per-thread arenas, so a worker's RSS stays near its largest compile;
four such workers at the end of a ``-n 4`` run thrashed a 16 GB CI runner.
macOS hands the pages back itself.
"""

import ctypes
import sys

import pytest

# glibc only (musl and macOS have no malloc_trim)
_trim = (
    getattr(ctypes.CDLL(None), "malloc_trim", None) if sys.platform == "linux" else None
)


@pytest.fixture(autouse=True)
def _malloc_trim():
    yield
    if _trim is not None:
        _trim(0)

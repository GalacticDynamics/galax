"""Drop compiled programs and hand freed memory back after each test.

The Staeckel tests compile large jacfwd/jacrev programs (up to ~1 GB on
top of the worker). XLA frees each one, but glibc can keep the freed pages
in its per-thread arenas, so a worker's RSS stays near its largest compile;
four such workers at the end of a ``-n 4`` run thrashed a 16 GB CI runner.
macOS hands the pages back itself. The JAX caches also keep every
executable a worker has compiled, so they are cleared after each test
too (test_staeckel_limits.py peaks at 0.7 GB instead of 2.5 GB).
"""

import ctypes
import sys

import jax
import pytest

# glibc only (musl and macOS have no malloc_trim)
_trim = (
    getattr(ctypes.CDLL(None), "malloc_trim", None) if sys.platform == "linux" else None
)
if _trim is not None:  # int malloc_trim(size_t pad)
    _trim.argtypes, _trim.restype = [ctypes.c_size_t], ctypes.c_int


@pytest.fixture(autouse=True)
def _malloc_trim():
    yield
    jax.clear_caches()  # drop this test's executables before the next compiles
    if _trim is not None:
        _trim(0)

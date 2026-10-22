"""Optional-dependency probes for the `galax.dynamics` tests.

Deliberately independent of `galax.interop.*`: probe the foreign libraries
directly so a missing optional distribution cannot break test collection.
"""

__all__ = ["OptDeps"]

from packaging.version import Version

from optional_dependencies import OptionalDependencyEnum
from optional_dependencies.utils import NOT_INSTALLED, is_installed


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]
    """Libraries the `galax.dynamics` tests compare against."""

    # agama is built by hand and put on `PYTHONPATH` (see docs/testing.rst), so
    # it has no distribution metadata for `auto()` to find. Probe importability
    # instead, without importing it; the version is then unknown.
    AGAMA = Version("0") if is_installed("agama") else NOT_INSTALLED

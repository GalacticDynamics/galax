"""Optional-dependency probes for the `galax.potential` tests.

Deliberately independent of `galax.interop.*`: those are optional
distributions, so importing one to decide whether to skip a test makes the
test suite fail to collect when it is absent. Probe the foreign libraries
directly instead.
"""

__all__ = ["GSL_ENABLED", "OptDeps"]

from packaging.version import Version

from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import (
    NOT_INSTALLED,
    InstalledState,
    chain_checks,
    get_version,
    is_installed,
)


def _agama_version() -> Version | InstalledState:
    """Agama's version, or ``0`` when it is importable but has no metadata.

    agama is built by hand and put on `PYTHONPATH` (see docs/testing.rst), so
    it has no distribution metadata for `auto()`/`get_version` to find. Probe
    importability instead, without importing it; its version is then unknown.
    """
    version = get_version("agama")
    if version is NOT_INSTALLED and is_installed("agama"):
        return Version("0")
    return version


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]
    """Libraries the `galax.potential` tests compare against."""

    GALA = chain_checks(get_version("gala"), is_installed("gala.dynamics"))
    GALPY = auto()
    AGAMA = _agama_version()


# gala-specific, and needs gala itself: degrade to `False` when it is absent,
# as `galax.interop.gala.optional_deps` does.
GSL_ENABLED: bool
if OptDeps.GALA.installed:
    from gala._cconfig import GSL_ENABLED
else:
    GSL_ENABLED = False

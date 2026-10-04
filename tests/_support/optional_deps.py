"""Optional-dependency probes for the test suite.

Deliberately independent of `galax.interop.*`: those are optional
distributions, so importing one to decide whether to skip a test makes the
test suite fail to collect when it is absent. Probe the foreign libraries
directly instead.
"""

__all__ = ["GSL_ENABLED", "OptDeps"]

from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import chain_checks, get_version, is_installed


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]
    """Optional dependencies for the galax test suite."""

    ASTROPY = auto()
    GALA = chain_checks(get_version("gala"), is_installed("gala.dynamics"))
    GALPY = auto()
    MATPLOTLIB = auto()


# gala-specific, and needs gala itself: degrade to `False` when it is absent,
# as `galax.interop.gala.optional_deps` does.
GSL_ENABLED: bool
if OptDeps.GALA.installed:
    from gala._cconfig import GSL_ENABLED
else:
    GSL_ENABLED = False

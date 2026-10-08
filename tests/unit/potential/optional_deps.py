"""Optional-dependency probes for the `galax.potential` tests.

Deliberately independent of `galax.interop.*`: those are optional
distributions, so importing one to decide whether to skip a test makes the
test suite fail to collect when it is absent. Probe the foreign libraries
directly instead.
"""

__all__ = ["GSL_ENABLED", "OptDeps"]

from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import chain_checks, get_version, is_installed


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]
    """Libraries the `galax.potential` tests compare against."""

    GALA = chain_checks(get_version("gala"), is_installed("gala.dynamics"))
    GALPY = auto()


# gala-specific, and needs gala itself: degrade to `False` when it is absent,
# as `galax.interop.gala.optional_deps` does.
GSL_ENABLED: bool
if OptDeps.GALA.installed:
    from gala._cconfig import GSL_ENABLED
else:
    GSL_ENABLED = False

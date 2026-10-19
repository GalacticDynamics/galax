"""Optional-dependency probes for the `galax.dynamics` tests.

Deliberately independent of `galax.interop.*`: probe the foreign libraries
directly so a missing optional distribution cannot break test collection.
"""

__all__ = ["GALA_STAECKEL", "GSL_ENABLED", "OptDeps"]

from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import chain_checks, get_version, is_installed


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]
    """Libraries the `galax.dynamics` tests compare against."""

    GALA = chain_checks(get_version("gala"), is_installed("gala.dynamics"))
    GALPY = auto()


# gala-specific, and needs gala itself: degrade to `False` when it is absent.
GSL_ENABLED: bool
GALA_STAECKEL: bool
if OptDeps.GALA.installed:
    import gala.dynamics.actionangle as _gaa
    from gala._cconfig import GSL_ENABLED

    GALA_STAECKEL = hasattr(_gaa, "find_actions_staeckel_fudge") and GSL_ENABLED
else:
    GSL_ENABLED = False
    GALA_STAECKEL = False

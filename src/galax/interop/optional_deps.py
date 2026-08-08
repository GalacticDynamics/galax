"""Optional dependencies."""

__all__ = ["OptDeps", "GSL_ENABLED"]


from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import chain_checks, get_version, is_installed


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]
    """Optional dependencies for ``galax``."""

    ASTROPY = auto()
    GALA = chain_checks(get_version("gala"), is_installed("gala.dynamics"))
    GALPY = auto()
    MATPLOTLIB = auto()


GSL_ENABLED: bool
if OptDeps.GALA.installed:
    from gala._cconfig import GSL_ENABLED
else:
    GSL_ENABLED = False
